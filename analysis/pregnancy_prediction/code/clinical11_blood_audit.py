"""Outcome-blind audit/freeze of the one authorized PI+VFI blood module."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.pregnancy_prediction.code.clinical9_data_audit import (ROOT, BOOK, OLD, DOPPLER, METRICS, MSE, blood_tokens,
    half_rounding, sha, save)

PRIOR = ROOT / 'outputs/clinical9_frozen_mse_20261122'
OUT = ROOT / 'outputs/clinical11_PI_VFI_mse_sensitivity_20261122'
PAPER = Path(os.environ.get('DATA_ROOT', 'data')) / 'Downloads/ARAD-D-26-03354 (1).pdf'
BLOOD = ['doppler_PI_frozen', 'doppler_VFI_frozen']


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def audit():
    OUT.mkdir(exist_ok=False)
    old_protocol = json.loads((PRIOR / 'PROTOCOL_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert sha(BOOK) == old_protocol['source_hashes'][str(BOOK)]
    # Outcomes, pregnancy variables and outcome-defined comparisons are not loaded.
    raw = pd.read_excel(BOOK, 0, usecols=['临床检查编号', '临床_描述'] + list(DOPPLER.values()),
                        dtype={'临床检查编号': str}).rename(columns={'临床检查编号': 'exam_id'})
    prior = pd.read_csv(PRIOR / '06_BLOOD_RECORD_AUDIT_LOCAL.csv', dtype={'exam_id': str}).set_index('exam_id')
    assert len(raw) == 358 and raw.exam_id.is_unique
    values = raw[['exam_id']].copy()
    for metric, column in DOPPLER.items():
        values[metric] = pd.to_numeric(raw[column], errors='coerce')
        np.testing.assert_allclose(values[metric], prior.loc[raw.exam_id, metric], equal_nan=True)
    records = []
    for i, row in raw.iterrows():
        tokens = blood_tokens(row['临床_描述'])
        r = dict(exam_id=row.exam_id, original_description=row['临床_描述'])
        issues = []
        for k in METRICS:
            value = values.loc[i, k]
            parsed = sorted(set(float(t) for t in tokens[k]))
            r[k], r[k + '_raw'], r[k + '_description'] = value, row[DOPPLER[k]], json.dumps(parsed)
            if len(parsed) == 1 and np.isfinite(value):
                status = 'match' if np.isclose(value, parsed[0], rtol=0, atol=1e-10) else 'mismatch'
            else:
                status = 'missing_or_unmeasurable' if not parsed else 'missing_structured_or_ambiguous_description'
            r[k + '_status'] = status
            if status == 'mismatch':
                issues.append(k + ':description_mismatch')
            if np.isfinite(value) and ((k in ['VI','FI','VFI'] and not 0 <= value <= 100)
                or (k in ['SD','PI'] and value <= 0) or (k == 'RI' and not 0 <= value <= 1)):
                issues.append(k + ':scale_review')
        vi, fi, vfi = [values.loc[i,k] for k in ['VI','FI','VFI']]
        if np.isfinite([vi,fi,vfi]).all():
            hv, hf, ho = [half_rounding(v,tokens[k]) for k,v in [('VI',vi),('FI',fi),('VFI',vfi)]]
            lo, hi = max(0,vi-hv)*max(0,fi-hf)/100, (vi+hv)*(fi+hf)/100
            compatible = vfi+ho >= lo-1e-12 and vfi-ho <= hi+1e-12
            r.update(VFI_expected=vi*fi/100, VFI_signed_error=vfi-vi*fi/100,
                VFI_abs_error=abs(vfi-vi*fi/100), VFI_rounding_compatible=bool(compatible),
                VFI_formula_product_interval_low=lo, VFI_formula_product_interval_high=hi,
                VFI_half_recording_unit=ho)
            if not compatible:
                issues.append('VFI:formula_rounding_review')
        sd, ri = values.loc[i,'SD'], values.loc[i,'RI']
        if np.isfinite([sd,ri]).all() and sd > 0:
            r['RI_expected_from_SD'], r['RI_SD_abs_error'] = 1-1/sd, abs(ri-(1-1/sd))
        if np.isfinite(fi) and fi > 100:
            r['FI_high_value_provenance'] = ('original_report_value_not_extraction_error' if r['FI_status']=='match'
                                             else 'unresolved_source_vs_extraction')
            r['FI_decimal_or_unit_conclusion'] = 'cannot_determine_without_original_device_report; no_division_or_decimal_guess'
        r['review_reasons'] = ';'.join(issues)
        records.append(r)
    records = pd.DataFrame(records)
    csv('02_BLOOD_RECORD_AUDIT_LOCAL.csv', records)
    high = records[records.FI.gt(100)]
    assert len(high) == 12 and high.FI_status.eq('match').all()
    csv('03_FI_HIGH_VALUE_SOURCE_REVIEW_LOCAL.csv', high)
    formula = records[records.VFI_rounding_compatible.eq(False)]
    csv('04_VFI_FORMULA_REVIEW_LOCAL.csv', formula)
    # Only frozen exam/patient maps and MSE availability are needed to define sample masks.
    meta = pd.read_csv(PRIOR / '04_MASTER_CLINICAL9_LOCAL_311.csv', usecols=['exam_id','patient_id',MSE],
                       dtype={'exam_id':str,'patient_id':str})
    outer = pd.read_csv(OLD / '02a_GLOBAL_PATIENT_FOLDS.csv', usecols=['repeat','exam_id'], dtype={'exam_id':str})
    eligible_ids = set(outer.loc[outer['repeat'].eq(0),'exam_id'])
    assert len(eligible_ids) == 306
    joined = meta.merge(values[['exam_id','PI','VFI']], on='exam_id', how='left', validate='one_to_one', sort=False)
    assert joined.exam_id.tolist() == meta.exam_id.tolist()
    joined['existing_eligible'] = joined.exam_id.isin(eligible_ids)
    joined['PI_VFI_observed'] = joined[['PI','VFI']].notna().all(axis=1)
    joined['Blood_full_observed'] = joined.existing_eligible & joined.PI_VFI_observed
    joined['Blood_MSE_observed'] = joined.Blood_full_observed & joined[MSE].notna()
    joined['missing_reason'] = [';'.join([k for k,missing in [('PI_missing',pd.isna(r.PI)),
        ('VFI_missing',pd.isna(r.VFI)),('MSE_missing',pd.isna(r[MSE])),
        ('outside_existing306',not r.existing_eligible)] if missing]) for _,r in joined.iterrows()]
    csv('06_MEMBERSHIP_BEFORE_OUTCOMES_LOCAL.csv', joined)
    csv('07_PI_VFI_VALUES_LOCAL_311.csv', joined[['exam_id','PI','VFI']].rename(columns={'PI':BLOOD[0],'VFI':BLOOD[1]}))
    cohorts = {'source358':set(raw.exam_id),'existing306':eligible_ids,
        'original_MSE185':set(joined.loc[joined.existing_eligible & joined[MSE].notna(),'exam_id']),
        'Blood_full_observed':set(joined.loc[joined.Blood_full_observed,'exam_id']),
        'Blood_MSE_observed':set(joined.loc[joined.Blood_MSE_observed,'exam_id'])}
    distributions, correlations, extremes = [], [], []
    for cohort, ids in cohorts.items():
        part = values[values.exam_id.isin(ids)]
        for k in METRICS:
            v = part[k]
            distributions.append(dict(cohort=cohort, variable=k, n_exams=len(part), n_available=int(v.notna().sum()),
                n_missing=int(v.isna().sum()), missing_rate=float(v.isna().mean()),
                minimum=float(v.min()), Q1=float(v.quantile(.25)), median=float(v.median()),
                Q3=float(v.quantile(.75)), IQR=float(v.quantile(.75)-v.quantile(.25)), maximum=float(v.max())))
            if cohort == 'source358':
                for label, extreme in [('minimum',v.min()),('maximum',v.max())]:
                    for exam in part.loc[v.eq(extreme),'exam_id']:
                        extremes.append(dict(variable=k, kind=label, value=float(extreme), exam_id=exam,
                                             meaning='observed extreme; no automatic exclusion'))
        for family in [['SD','PI','RI'],['VI','FI','VFI']]:
            for i,a in enumerate(family):
                for b in family[i+1:]:
                    pair = part[[a,b]].dropna()
                    correlations.append(dict(cohort=cohort, variable1=a, variable2=b, n_pair=len(pair),
                        Pearson=float(pair[a].corr(pair[b])), Spearman=float(pair[a].corr(pair[b],method='spearman'))))
    csv('01_BLOOD_AVAILABILITY_DISTRIBUTIONS.csv',distributions)
    csv('05_BLOOD_CORRELATIONS.csv',correlations)
    csv('08_BLOOD_OBSERVED_EXTREMES_LOCAL.csv',extremes)
    # Directly reported PI/VFI remain usable for this limited sensitivity; formula flags
    # cannot identify whether VI, FI or VFI is responsible, so no component is guessed.
    assert values.PI.dropna().gt(0).all() and values.VFI.dropna().between(0,100).all()
    assert not records.PI_status.eq('mismatch').any() and not records.VFI_status.eq('mismatch').any()
    summary = dict(stage='quality_and_definition_frozen_before_outcome_load',
        outcome_columns_loaded=False, raw358_n=358, all_numeric_values_match_previous_audit=True,
        n_FI_above100=len(high), all_FI_high_values_present_in_original_description=True,
        FI_decimal_and_units_unresolved=True, n_VFI_formula_review=len(formula),
        PI_VFI_quality_decision='usable_as_source_reported_continuous_values_for_limited_exploratory_sensitivity; not measurement-validation certification',
        cleaning='source numeric values unchanged; nonnumeric/missing to NaN; explicit measured zero retained; no winsorization/log/threshold search',
        formula_review_policy='retain and flag; disagreement cannot uniquely assign error to VI/FI/VFI; no derived replacement or patient exclusion',
        model_blood_fields=BLOOD, no_other_blood_fields_allowed=True,
        sample_policy='both PI and VFI observed in both comparisons; MSE additionally observed for core comparison; original Clinical9 missing stays training-fold imputed',
        blood_full_exams=int(joined.Blood_full_observed.sum()), blood_full_patients=int(joined.loc[joined.Blood_full_observed,'patient_id'].nunique()),
        blood_MSE_exams=int(joined.Blood_MSE_observed.sum()), blood_MSE_patients=int(joined.loc[joined.Blood_MSE_observed,'patient_id'].nunique()),
        original_MSE_exams=185,
        matching_control='refit frozen Clinical9 and Clinical9+MSE on the same blood-observed MSE subset to separate sample restriction from blood adjustment; not feature/model search',
        manuscript=dict(path=str(PAPER), sha256=sha(PAPER), status='Manuscript Draft, not independently verified published paper',
            reference='ARAD-D-26-03354; Prospective validation of a Doppler-enhanced multimodal ultrasound–clinical model for predicting clinical pregnancy after IVF/ICSI single embryo transfer',
            PDF_pages={'ultrasound_measurements':11,'feature_interpretation':20,'interpretation_discussion':23,'candidate_clinical_variables':49},
            applicability='full Doppler/B-mode/dynamic gated fusion differs from two scalar PI/VFI sensitivity; no claim of replicating its network or performance'),
        source_hashes={str(p):sha(p) for p in [BOOK,PAPER,PRIOR/'PROTOCOL_BEFORE_FIT.json',
            PRIOR/'06_BLOOD_RECORD_AUDIT_LOCAL.csv',PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',OLD/'02a_GLOBAL_PATIENT_FOLDS.csv']})
    save(OUT/'BLOOD_AUDIT_BEFORE_OUTCOME.json',summary)
    text='# PI+VFI血流扩展：无结局质量审计与定义冻结\n\n'
    text+=f'此脚本未加载结局列。358源值与前次逐值审计一致；FI>100有12条且全部见于原始描述，不能归因于本轮提取，单位或小数点仍无法确认。VFI公式有{len(formula)}条超出按记录小数位计算的舍入区间，完整编号、原文与误差见04表。PI/VFI没有可确认的结构/原文数值冲突；保留报告原值与疑点，限作来源可追溯的探索性敏感性，不视为测量学已验证。\n\n'
    text+=f'冻结仅增加PI与VFI两个连续变量。全临床血流可用{summary["blood_full_exams"]}检查/{summary["blood_full_patients"]}患者；MSE和两项血流同时可用{summary["blood_MSE_exams"]}/{summary["blood_MSE_patients"]}。缺失血流不补为0、不因FI或VFI公式单项疑点删人；按用户要求主比较只限PI/VFI已观察病例，Clinical9缺失保持训练折处理。\n\n'
    text+='在该MSE共同子集中重拟冻结Clinical9±MSE作为必要参照，避免把185例缩小后的样本变化当作血流解释作用。旧185例的全部历史结果另列，不能直接把不同样本Δ之差解释为机制差异。\n\n'
    text+='已核读本地ARAD-D-26-03354 (1).pdf（55页稿件）：PDF第11页测量、20/23页解释及49页候选表支持其覆盖6项血流、强调PI/VFI，但完整模型还包含血流信号时空变化、B-mode与动态信息。当前只做预先限定的两个标量模块，不宣称复现完整模型或其独立验证。\n'
    (OUT/'BLOOD_AUDIT_REPORT_BEFORE_FIT.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
