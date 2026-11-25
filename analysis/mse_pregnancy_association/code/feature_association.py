"""Fixed five-feature association study, separate from prediction AUC.

One feature at a time, fixed clinical adjustment, patient-cluster uncertainty,
and one quadratic extension per feature. No feature or threshold selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

from analysis.mse_pregnancy_association.code.frozen_feature_auc import CLINICAL, AMPLITUDE, COORD, GROUP, OUTCOME, SEED, digest, save
from analysis.mse_pregnancy_association.code.batch_wave_counts import RATE

FEATURES = AMPLITUDE+[RATE]+COORD
LABELS = ['Anterior amplitude', 'Posterior amplitude', 'Activity frequency', 'ER CC', 'ER MSE']


def bh_adjust(values):
    """Benjamini-Hochberg adjusted p values, retaining all predeclared tests."""
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1.)
    return result


def fit_cluster_logit(x, y, groups):
    """Unpenalized logistic MLE and CR1 patient-cluster sandwich covariance.

    Bread = inverse observed information. Meat = sum of outer products of
    patient-summed scores. CR1 factor = G/(G-1) * (N-1)/(N-K).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    if (x.ndim != 2 or len(x) != len(y) or len(y) != len(groups) or
        not np.isfinite(x).all() or not np.isfinite(y).all() or
        set(np.unique(y)) != {0., 1.} or np.linalg.matrix_rank(x) != x.shape[1]):
        raise ValueError('Expected finite full-rank design, two outcomes, and patient IDs')
    def objective(beta):
        eta = x@beta
        return float(np.sum(np.logaddexp(0., eta)-y*eta))
    def gradient(beta):
        return x.T@(expit(x@beta)-y)
    def hessian(beta):
        p = expit(x@beta)
        return x.T@((p*(1-p))[:, None]*x)
    result = minimize(objective, np.zeros(x.shape[1]), jac=gradient, hess=hessian,
                      method='Newton-CG', options=dict(xtol=1e-10, maxiter=100))
    score_max = float(abs(gradient(result.x)).max())
    # Some L-BFGS/Newton variants report precision-loss status after reaching
    # a stationary point. The score norm is the numerical convergence check;
    # retain the original tolerance and do not accept a materially nonstationary
    # solution solely because a status flag says success.
    if score_max > 1e-6:
        raise RuntimeError(f'Logit did not converge: {result.message}; score={score_max}')
    bread = np.linalg.inv(hessian(result.x))
    patients, inverse = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(patients), x.shape[1]))
    np.add.at(scores, inverse, x*(y-expit(x@result.x))[:, None])
    n, k = x.shape
    correction = len(patients)/(len(patients)-1)*(n-1)/(n-k)
    covariance = (bread@scores.T@scores@bread)*correction
    return dict(beta=result.x, covariance=covariance, n=n, patients=len(patients),
                log_likelihood=-float(result.fun), score_max=score_max,
                information_condition_number=float(np.linalg.cond(hessian(result.x))),
                converged=bool(score_max <= 1e-6), optimizer_success=bool(result.success),
                optimizer_message=str(result.message), covariance_CR1_factor=float(correction))


def wald(fit, indices):
    beta = fit['beta'][indices]
    covariance = fit['covariance'][np.ix_(indices, indices)]
    statistic = float(beta@np.linalg.solve(covariance, beta))
    return float(chi2.sf(statistic, len(indices)))


def cluster_mean_difference(data, feature, repeats):
    """Mean pregnancy-minus-nonpregnancy difference, resampling patients."""
    use = data[[feature, GROUP, OUTCOME]].dropna()
    patients, inverse = np.unique(use[GROUP], return_inverse=True)
    y = use[OUTCOME].to_numpy(int)
    x = use[feature].to_numpy(float)
    totals = np.zeros((len(patients), 2))
    counts = np.zeros_like(totals)
    np.add.at(totals, (inverse, y), x)
    np.add.at(counts, (inverse, y), 1.)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        den = weights@counts
        if np.all(den > 0):
            means = (weights@totals)/den
            differences.append(float(means[1]-means[0]))
    means = totals.sum(0)/counts.sum(0)
    lo, hi = np.quantile(differences, [.025, .975])
    return dict(mean_difference=float(means[1]-means[0]), CI_low=float(lo), CI_high=float(hi),
                patients=len(patients), bootstrap_valid=len(differences))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.cohort)
    protocol = dict(features=FEATURES, clinical_adjustment=CLINICAL,
        one_feature_per_model=True, outcome=OUTCOME, patient_cluster=GROUP,
        missingness='Descriptives use all available feature values. Regression uses complete cases for the feature plus fixed clinical covariates; no median imputation for inference.',
        linear='Unpenalized logistic MLE, OR per one sample SD; unadjusted and adjusted use identical complete cases.',
        uncertainty='Patient-cluster CR1 sandwich; normal/chi-square Wald inference. Descriptive mean differences use patient bootstrap.',
        nonlinear='One fixed quadratic extension z + z^2 per feature; no knot/cutpoint or model search. Test z^2 and joint z,z^2.',
        multiplicity='Separate BH adjustment across five adjusted linear tests, five curvature tests, and five joint quadratic tests; all displayed.',
        curves='Quadratic adjusted odds ratios relative to the feature median, pointwise 95% intervals, observed range only; not causal or validated optimal values.',
        bootstrap=args.bootstrap, seed=SEED, thresholds_changed=False,
        source_sha256=source_hash,
        reference='https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html',
        exploratory_after_prior_AUC=True)
    save(args.output/'protocol_before_fit.json', protocol)
    data = pd.read_csv(args.cohort, dtype={GROUP: str})
    descriptive = []
    differences = []
    rows = []
    audits = []
    curves = []
    for feature in FEATURES:
        for outcome in (0, 1):
            group = data.loc[data[OUTCOME] == outcome]
            values = group[feature].dropna()
            descriptive.append(dict(feature=feature, outcome=outcome, exams=len(group),
                available=len(values), missing=int(group[feature].isna().sum()),
                mean=float(values.mean()), SD=float(values.std()), median=float(values.median()),
                Q1=float(values.quantile(.25)), Q3=float(values.quantile(.75))))
        differences.append(dict(feature=feature, **cluster_mean_difference(data, feature, args.bootstrap)))
        use = data[[feature]+CLINICAL+[GROUP, OUTCOME, 'case_id']].dropna().reset_index(drop=True)
        scale = float(use[feature].std(ddof=1))
        center = float(use[feature].mean())
        z = (use[feature].to_numpy(float)-center)/scale
        clinical = use[CLINICAL].to_numpy(float)
        clinical = (clinical-clinical.mean(0))/clinical.std(0)
        y = use[OUTCOME].to_numpy(int)
        groups = use[GROUP].to_numpy(str)
        basic = np.column_stack([np.ones(len(use)), z])
        adjusted = np.column_stack([basic, clinical])
        quad = np.column_stack([adjusted, z*z])
        unadjusted_fit = fit_cluster_logit(basic, y, groups)
        adjusted_fit = fit_cluster_logit(adjusted, y, groups)
        quadratic_fit = fit_cluster_logit(quad, y, groups)
        row = dict(feature=feature, n=len(use), patients=int(use[GROUP].nunique()),
            positive=int(y.sum()), negative=int((1-y).sum()),
            missing_feature=int(data[feature].isna().sum()),
            extra_missing_clinical=int(data[feature].notna().sum()-len(use)),
            feature_SD=scale, feature_mean=center)
        for label, fit in [('unadjusted', unadjusted_fit), ('adjusted', adjusted_fit)]:
            beta = float(fit['beta'][1])
            se = float(np.sqrt(fit['covariance'][1, 1]))
            row.update({label+'_OR_per_SD':float(np.exp(beta)),
                label+'_CI_low':float(np.exp(beta-1.959963984540054*se)),
                label+'_CI_high':float(np.exp(beta+1.959963984540054*se)),
                label+'_p':float(2*norm.sf(abs(beta/se)))})
        row.update(quadratic_curvature_beta=float(quadratic_fit['beta'][-1]),
            nonlinearity_p=wald(quadratic_fit, [len(quad[0])-1]),
            quadratic_joint_p=wald(quadratic_fit, [1, len(quad[0])-1]))
        rows.append(row)
        for label, fit in [('unadjusted', unadjusted_fit), ('adjusted', adjusted_fit), ('quadratic', quadratic_fit)]:
            audits.append(dict(feature=feature, model=label, cases=use['case_id'].tolist(), **fit))
        x_values = np.linspace(float(use[feature].min()), float(use[feature].max()), 101)
        grid = (x_values-center)/scale
        reference = (float(use[feature].median())-center)/scale
        contrasts = np.zeros((len(grid), quad.shape[1]))
        contrasts[:, 1] = grid-reference
        contrasts[:, -1] = grid*grid-reference*reference
        estimate = contrasts@quadratic_fit['beta']
        se = np.sqrt(np.maximum(0., np.einsum('ij,jk,ik->i', contrasts, quadratic_fit['covariance'], contrasts)))
        for i, value in enumerate(x_values):
            curves.append(dict(feature=feature, value=float(value), reference_median=float(use[feature].median()),
                adjusted_OR=float(np.exp(estimate[i])),
                CI_low=float(np.exp(estimate[i]-1.959963984540054*se[i])),
                CI_high=float(np.exp(estimate[i]+1.959963984540054*se[i]))))
    results = pd.DataFrame(rows)
    for column in ['adjusted_p', 'nonlinearity_p', 'quadratic_joint_p']:
        results[column.replace('_p', '_BH_q')] = bh_adjust(results[column])
    descriptive_df = pd.DataFrame(descriptive)
    differences_df = pd.DataFrame(differences)
    curves_df = pd.DataFrame(curves)
    for name, table in [('associations', results), ('group_descriptives', descriptive_df),
                        ('group_mean_differences', differences_df), ('quadratic_curves', curves_df)]:
        table.to_csv(args.output/(name+'.csv'), index=False, encoding='utf-8-sig')
    from analysis.mse_pregnancy_association.code.measurement.io import jsonable
    save(args.output/'fit_audits.json', jsonable(audits))
    with pd.ExcelWriter(args.output/'feature_association.xlsx', engine='openpyxl') as writer:
        results.to_excel(writer, sheet_name='OR_and_nonlinearity', index=False)
        descriptive_df.to_excel(writer, sheet_name='groups', index=False)
        differences_df.to_excel(writer, sheet_name='mean_difference', index=False)
        curves_df.to_excel(writer, sheet_name='quadratic_curves', index=False)
    plot_results(args.output, results, curves_df)
    if digest(args.cohort) != source_hash:
        raise RuntimeError('Source analysis cohort changed')
    summary = dict(exams=len(data), patients=int(data[GROUP].nunique()), features=rows,
        adjusted_linear_BH_q=results['adjusted_BH_q'].tolist(),
        nonlinearity_BH_q=results['nonlinearity_BH_q'].tolist(),
        quadratic_joint_BH_q=results['quadratic_joint_BH_q'].tolist(),
        frozen_features_changed=False, source_unchanged=True)
    save(args.output/'summary.json', summary)
    lines = ['# 冻结特征与临床妊娠的关联分析', '',
        f'{len(data)}次检查、{data[GROUP].nunique()}名患者；仅分析预先指定的五项特征，特征和阈值未改。',
        '组间描述使用各特征可用值；调整模型按该特征及七个临床变量完整病例计算，无单次插补。每项的未调整/调整模型使用相同病例。',
        'OR是特征增加一个样本SD对应的妊娠优势比，不是妊娠概率倍数，也不是因果效应。重复检查按患者聚类CR1标准误处理。',
        '非线性只固定增加一个二次项，不寻找最佳分组、截点、结点或曲线；分别报告曲率检验和两项联合检验。五个特征的线性、曲率及联合检验分别作BH校正。',
        'CC/MSE只复用既有单block标量；两者单独可用时各自纳入，未平均多个block。波频率仍含既有计数误差，不因本分析而宣称已验证。',
        '', '|特征|完整病例|调整OR/SD（95%区间）|P|BH q|非线性P|非线性q|',
        '|---|---:|---|---:|---:|---:|---:|']
    for name, row in zip(LABELS, results.to_dict('records')):
        lines.append(f'|{name}|{row["n"]}|{row["adjusted_OR_per_SD"]:.3f}（{row["adjusted_CI_low"]:.3f}–{row["adjusted_CI_high"]:.3f}）|{row["adjusted_p"]:.4f}|{row["adjusted_BH_q"]:.4f}|{row["nonlinearity_p"]:.4f}|{row["nonlinearity_BH_q"]:.4f}|')
    lines += ['', '完整病例数不同，不能把不同子集的特征效果直接当作同一人群比较。未分析新的患者级ER、RMS聚合，也未改变0.6或计数算法。',
        '这是在已看过AUC后预先固定本轮方案的探索性分析，不能算预注册或独立验证；不能用单个P值宣称有效或无效。',
        '图中曲线为相对中位数的调整OR及逐点95%区间，不是经过验证的最优频率。', '',
        '聚类协方差方法依据：'+protocol['reference']]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(results[['feature','n','adjusted_OR_per_SD','adjusted_CI_low','adjusted_CI_high',
                   'adjusted_p','adjusted_BH_q','nonlinearity_p','nonlinearity_BH_q']].to_string(index=False))


def plot_results(output, results, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feature, label in zip(axes.flat, FEATURES, LABELS):
        rows = curves.loc[curves['feature'] == feature]
        factor = 100. if feature in AMPLITUDE else 1.
        x = rows['value'].to_numpy()*factor
        ax.plot(x, rows['adjusted_OR'].to_numpy())
        ax.fill_between(x, rows['CI_low'].to_numpy(), rows['CI_high'].to_numpy(), alpha=.2)
        ax.axhline(1., color='gray', ls='--', lw=.8)
        ax.set_yscale('log')
        row = results.loc[results['feature'] == feature].iloc[0]
        ax.set_title(f'{label}, n={row["n"]}\ncurvature p={row["nonlinearity_p"]:.3f}', fontsize=10)
        ax.set_xlabel('Interval strain SD (%)' if feature in AMPLITUDE else 'Activities/min' if feature == RATE else label)
        ax.set_ylabel('Adjusted OR vs median')
    axes.flat[-1].set_visible(False)
    fig.suptitle('Fixed quadratic association; pointwise 95% patient-cluster intervals')
    fig.tight_layout()
    fig.savefig(output/'quadratic_associations.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    run(parser.parse_args())
