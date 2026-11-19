"""Add existing significant-activity frequency to the prior frozen AUC cohort.

Frequency is primary because exposure differs. Raw count is a separately named
sensitivity analysis. No outcome-driven change to activity identity or salience.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from experiments.batch_wave_counts import COUNT, RATE
from experiments.frozen_feature_auc import (CLINICAL, AMPLITUDE, GROUP, OUTCOME,
    SEED, digest, save, evaluate, bootstrap_oof)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.previous/'analysis_cohort.csv', args.previous/'protocol_before_fit.json',
               args.counts/'wave_counts.csv', args.counts/'protocol_before_counting.json']
    hashes = {str(p.resolve()): digest(p) for p in sources}
    count_protocol = json.loads(sources[-1].read_text(encoding='utf-8'))
    protocol = dict(primary_wave_feature=RATE, raw_count_sensitivity=COUNT,
        clinical_features=CLINICAL, amplitudes=AMPLITUDE,
        model='existing Logistic C=1e6 liblinear, median imputation and scaling within training folds',
        folds=args.folds, repeats=args.repeats, bootstrap=args.bootstrap, first_seed=SEED,
        same_patient_grouping_and_cohort_as_prior_amplitude_analysis=True,
        uncertainty='patient-cluster bootstrap of fixed primary OOF predictions; no bootstrap refit',
        count_protocol=count_protocol, no_threshold_or_algorithm_tuning=True,
        interpretation='exploratory AUC of algorithm-defined significant propagation activity, not validated total wave count',
        sources_sha256=hashes)
    save(args.output/'protocol_before_fit.json', protocol)
    data = pd.read_csv(sources[0], dtype={GROUP: str})
    counts = pd.read_csv(sources[2])
    wave_columns = counts[['case_id', COUNT, RATE, 'processing_status',
        'failure_reason', 'effective_observation_seconds', 'unresolved_activity_rows']].rename(
        columns={'processing_status': 'wave_processing_status', 'failure_reason': 'wave_failure_reason'})
    data = data.merge(wave_columns, on='case_id', how='left', validate='one_to_one',
                      indicator='wave_count_match')
    if (data['wave_count_match'] != 'both').any():
        raise ValueError('A prior examination has no counting result or explicit failure record')
    if ((data['wave_processing_status'] != 'completed') & data[RATE].notna()).any():
        raise ValueError('Failed or unobservable exams must remain missing, not zero')
    data.to_csv(args.output/'analysis_cohort.csv', index=False, encoding='utf-8-sig')
    main_models = dict(clinical=CLINICAL, amplitude=AMPLITUDE,
        clinical_amp=CLINICAL+AMPLITUDE, wave_frequency=[RATE],
        clinical_wave=CLINICAL+[RATE], clinical_amp_wave=CLINICAL+AMPLITUDE+[RATE])
    metrics, differences, predictions = evaluate('wave_frequency_main', data, main_models, args)
    # Explicitly compare adding wave frequency to the prior combined baseline.
    draws = bootstrap_oof(data, predictions, args.bootstrap, SEED+1000)
    delta = draws['clinical_amp_wave']-draws['clinical_amp']
    lo, hi = np.quantile(delta, [.025, .975])
    repeated = pd.read_csv(args.output/'wave_frequency_main'/'repeat_metrics.csv')
    extra = dict(analysis='wave_frequency_main', comparison='clinical_amp_wave minus clinical_amp',
        n=len(data), delta_AUC=float(roc_auc_score(data[OUTCOME], predictions['clinical_amp_wave'])-
                                  roc_auc_score(data[OUTCOME], predictions['clinical_amp'])),
        CI_low=float(lo), CI_high=float(hi),
        repeat_mean_delta_AUC=float(repeated.loc[repeated['model']=='clinical_amp_wave','AUC'].mean()-
                                    repeated.loc[repeated['model']=='clinical_amp','AUC'].mean()))
    differences.append(extra)
    pd.DataFrame(differences).to_csv(args.output/'wave_frequency_main'/'paired_AUC_differences.csv',
                                    index=False, encoding='utf-8-sig')
    m, d, _ = evaluate('raw_count_sensitivity', data,
        dict(clinical=CLINICAL, wave_count=[COUNT], clinical_count=CLINICAL+[COUNT]), args)
    metrics += m
    differences += d
    if data[RATE].isna().any():
        available = data.loc[data[RATE].notna()].reset_index(drop=True)
        m, d, _ = evaluate('observable_count_sensitivity', available, main_models, args)
        metrics += m
        differences += d
    # Same seed, group, cohort and baseline should reproduce the prior AUC.
    prior = pd.read_csv(args.previous/'amplitude_main'/'out_of_fold_predictions.csv')
    current = pd.read_csv(args.output/'wave_frequency_main'/'out_of_fold_predictions.csv')
    if prior['case_id'].tolist() != current['case_id'].tolist():
        raise ValueError('Examination order differs from the prior paired comparison')
    np.testing.assert_array_equal(prior['fold'], current['fold'])
    np.testing.assert_allclose(prior['clinical'], current['clinical'], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(prior['clinical_amp'], current['clinical_amp'], atol=1e-10, rtol=1e-10)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError('Sealed source data changed during AUC analysis')
    metrics_df = pd.DataFrame(metrics)
    delta_df = pd.DataFrame(differences)
    metrics_df.to_csv(args.output/'all_AUC_metrics.csv', index=False, encoding='utf-8-sig')
    delta_df.to_csv(args.output/'all_paired_differences.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(args.output/'wave_count_AUC.xlsx', engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='AUC', index=False)
        delta_df.to_excel(writer, sheet_name='paired_delta', index=False)
        counts.to_excel(writer, sheet_name='counts', index=False)
        data.to_excel(writer, sheet_name='cohort', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in ('clinical', 'wave_frequency', 'clinical_wave', 'clinical_amp_wave'):
        fpr, tpr, _ = roc_curve(data[OUTCOME], predictions[model])
        ax.plot(fpr, tpr, label=f'{model}: {roc_auc_score(data[OUTCOME],predictions[model]):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=.8)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Grouped out-of-fold ROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(args.output/'OOF_ROC.png', dpi=180)
    plt.close(fig)
    summary = dict(exams=len(data), patients=int(data[GROUP].nunique()),
        count_available=int(data[RATE].notna().sum()),
        count_missing=int(data[RATE].isna().sum()),
        primary_metrics=[r for r in metrics if r['analysis']=='wave_frequency_main'],
        primary_differences=[r for r in differences if r['analysis']=='wave_frequency_main'],
        same_baseline_predictions_verified=True, thresholds_changed=False,
        count_validation_status='exploratory; known local false positives, duplicate identities and weak-activity misses')
    save(args.output/'summary.json', summary)
    lines = ['# 显著传播活动次数/频率的AUC对照', '',
        f'{len(data)}次检查、{data[GROUP].nunique()}名患者；次数/频率可用{summary["count_available"]}次，缺失{summary["count_missing"]}次。',
        '复用已有约2秒A(s,t)，periodic+partial、受限拆分、ownership、全片两壁共同归一化>0.6和现有传播判据；方向仅QC。未重跑追踪，未改任何阈值或算法。',
        '使用所有已有native sections，与历史次数流程一致；不是仅幅度ROI0.2–0.8，也不扩展或重新采样原图。每例保存实际空间范围。',
        '频率=去重后显著传播活动数/可观察measurement intervals并集分钟。失败或完全无可分析支持记缺失；可观察但未检测到活动记0。',
        '该计数方法仍有局部误计、相邻活动身份重复和0.6漏弱活动等已知限制；AUC不能使其变为已验证的全部生理蠕动次数。',
        f'与之前幅度分析使用同一队列和分组：{args.folds}折、{args.repeats}次固定种子；训练折内插补/标准化，同一患者不跨折。临床和临床+幅度预测逐例核对一致。',
        f'主报告第一个预设种子折外AUC；另列重复均值，不选最高值。95%区间为{args.bootstrap}次患者聚类bootstrap固定折外预测，未重训。', '',
        '|模型|折外AUC|95%区间|重复均值|', '|---|---:|---|---:|']
    for r in summary['primary_metrics']:
        lines.append(f'|{r["model"]}|{r["primary_OOF_AUC"]:.4f}|{r["CI_low"]:.4f}–{r["CI_high"]:.4f}|{r["repeat_mean_AUC"]:.4f}|')
    lines += ['', '## 配对差值']
    for r in summary['primary_differences']:
        lines.append(f'- {r["comparison"]}: ΔAUC={r["delta_AUC"]:+.4f}，95%区间{r["CI_low"]:+.4f}～{r["CI_high"]:+.4f}。')
    lines += ['', '原始总次数另列敏感性分析，不与次数/分混称；患者ER/多block协调性未自行平均。',
              '仅内部探索测试，不代表外部验证。不使用本轮结局改0.6、传播判定或事件身份。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(metrics_df[['analysis','model','n','primary_OOF_AUC','repeat_mean_AUC']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000)
    run(parser.parse_args())
