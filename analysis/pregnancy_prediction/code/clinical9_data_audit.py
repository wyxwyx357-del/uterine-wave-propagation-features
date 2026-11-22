"""Source-only clinical9 definitions and audit; no predictor/outcome association."""
import os
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OUT = ROOT / 'outputs/clinical9_frozen_mse_20261122'
BOOK = Path(os.environ.get('CLINICAL_SOURCE_PATH', 'data/clinical_source.xlsx'))
MASTER = ROOT / 'outputs/extended6_increment_protocol_20261120/MASTER_CLINICAL_SOURCE_ALIGNED_311.csv'
OLD = ROOT / 'outputs/auc_first_pass_extended6_globalfolds_20261120'
LIMITED = ROOT / 'outputs/clinical_baseline_limited_comparison_20261120'
HISTORICAL = ROOT / 'outputs/all_dynamic_foldwise_evaluation_20261120'
CORE = ['age', 'transfer_embryo_count_component', 'blastocyst_transfer', 'bmi',
        'infertility_years', 'endometrial_thickness_mm']
NUM6 = ['age', 'transfer_embryo_count_component', 'bmi', 'infertility_years', 'endometrial_thickness_mm']
EXTRA = ['endometrial_volume_ml', 'endometrial_pattern_pure_C', 'days_before_transfer']
NUM9 = NUM6 + ['endometrial_volume_ml', 'days_before_transfer']
MSE = 'record_MSE_time_equal'
METRICS = ['SD', 'PI', 'RI', 'VI', 'FI', 'VFI']
DOPPLER = {k: '临床_' + ('S/D' if k == 'SD' else k) + '（内膜血流）' for k in METRICS}
SOURCE6 = dict(zip(CORE, ['年龄', '移植胚胎数_解析', '胚胎种类_含囊胚', 'BMI', '不孕年限_年', '内膜厚度_mm']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def csv(name, obj):
    pd.DataFrame(obj).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def normalize(value):
    return '' if pd.isna(value) else unicodedata.normalize('NFKC', str(value)).strip()


def pattern(value):
    text = re.sub(r'\s+', '', normalize(value)).upper()
    text = text.replace('型', '').replace('—', '-').replace('–', '-').replace('－', '-').replace('~', '-')
    if text in {'A', 'A-B', 'B', 'B-C', 'C'}:
        return text
    return np.nan


def timing(value):
    text = normalize(value)
    matches = re.findall(r'移植\s*前\s*第?\s*(\d+|[零〇一二两三四五六七八九十]+)\s*(?:天|日)', text)
    cn = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
          '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    values = []
    for token in matches:
        if token.isdigit():
            values.append(int(token))
        elif token in cn:
            values.append(cn[token])
        elif re.fullmatch('[一二三四五六七八九]?十[一二三四五六七八九]?', token):
            a, b = token.split('十')
            values.append((cn[a] if a else 1) * 10 + (cn[b] if b else 0))
    if re.search(r'移植\s*(?:当|同)\s*(?:天|日)', text):
        values.append(0)
    values = sorted(set(values))
    if len(values) == 1 and not re.search(r'移植\s*后\s*\d+\s*(?:天|日)', text):
        return float(values[0]), 'explicit_description'
    return np.nan, 'conflicting_intervals' if len(values) > 1 else 'not_explicitly_parsed'


def blood_tokens(value):
    """Only endometrial blood-flow section; never substitute uterine artery values."""
    text = normalize(value)
    start = re.search(r'(?:子宫)?内膜(?:血流(?:参数)?|参数)', text)
    if not start:
        return {k: [] for k in METRICS}
    section = text[start.end():]
    end = re.search(r'子宫动脉|经阴道|立体扫查|子宫(?:前|后|水平)', section)
    if end:
        section = section[:end.start()]
    result = {}
    for k in METRICS:
        key = r'S\s*/\s*D' if k == 'SD' else k
        result[k] = re.findall(r'(?<![A-Z])' + key + r'(?:\s*\([^)]*\))?\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)', section.upper())
    return result


def half_rounding(value, tokens):
    # Literal decimal presentation determines rounding interval, not a clinical cutoff.
    matching = [t for t in tokens if np.isclose(float(t), value, atol=1e-10, rtol=0)]
    token = matching[0] if matching else str(value)
    places = len(token.split('.')[1]) if '.' in token else 0
    return .5 * 10 ** -places


def summarize(frame, cohort):
    rows = []
    for field in CORE + EXTRA + ['uterine_position_clean']:
        value = frame[field]
        row = dict(cohort=cohort, feature=field, n_exams=len(frame), n_unique_patients=frame.patient_id.nunique(),
                   repeated_patient_groups=int(frame.patient_id.value_counts().gt(1).sum()),
                   n_available=int(value.notna().sum()), n_missing=int(value.isna().sum()), missing_rate=float(value.isna().mean()))
        if field != 'uterine_position_clean':
            q = pd.to_numeric(value, errors='coerce').describe(percentiles=[.25, .5, .75]).to_dict()
            row.update({k: float(v) for k, v in q.items()})
        rows.append(row)
    return rows


def audit():
    OUT.mkdir(exist_ok=False)
    raw_columns = ['临床检查编号', '临床_容积(ml)', '临床_分型', '临床_描述', '临床_子宫位置'] + list(DOPPLER.values())
    raw = pd.read_excel(BOOK, 0, usecols=raw_columns, dtype={'临床检查编号': str})
    source_model = pd.read_excel(BOOK, 1, usecols=['临床检查编号', '患者聚类标识'] + list(SOURCE6.values()),
                                 dtype={'临床检查编号': str, '患者聚类标识': str})
    assert len(raw) == 358 and raw['临床检查编号'].is_unique and source_model['临床检查编号'].is_unique
    raw = raw.rename(columns={'临床检查编号': 'exam_id'})
    raw['endometrial_volume_ml'] = pd.to_numeric(raw['临床_容积(ml)'], errors='coerce')
    raw['endometrial_pattern_clean'] = raw['临床_分型'].map(pattern)
    raw['endometrial_pattern_pure_C'] = raw.endometrial_pattern_clean.map(
        lambda v: np.nan if pd.isna(v) else float(v == 'C'))
    parsed = raw['临床_描述'].map(timing)
    raw['days_before_transfer'] = parsed.map(lambda x: x[0])
    raw['days_parse_status'] = parsed.map(lambda x: x[1])
    raw['uterine_position_clean'] = raw['临床_子宫位置'].map(
        lambda v: {'前': 'anterior', '后': 'posterior', '水平': 'horizontal'}.get(normalize(v), np.nan))
    blind358 = source_model.rename(columns={'临床检查编号': 'exam_id', '患者聚类标识': 'patient_id',
                                           **{v: k for k, v in SOURCE6.items()}}).merge(raw, on='exam_id', validate='one_to_one')
    # Formal master retains every original registry row and its original six values/labels.
    fields = ['case_id', 'exam_id', 'patient_id', 'cohort', 'outcome', 'model_identity_eligible', MSE] + CORE
    master = pd.read_csv(MASTER, usecols=fields, dtype={'exam_id': str, 'patient_id': str})
    joined = master.merge(raw, on='exam_id', how='left', validate='one_to_one', sort=False)
    assert joined.exam_id.tolist() == master.exam_id.tolist() and len(joined) == 311
    eligible = joined.model_identity_eligible.eq(True) & joined.outcome.isin([0, 1])
    mse = eligible & joined[MSE].notna()
    assert eligible.sum() == 306 and mse.sum() == 185
    cohorts = {'source358': blind358, 'registry311': joined, 'eligible306': joined[eligible],
               'MSE185': joined[mse], 'MSE_unavailable121': joined[eligible & ~mse]}
    csv('01_VARIABLE_AVAILABILITY_DISTRIBUTIONS.csv', [r for c, d in cohorts.items() for r in summarize(d, c)])
    categories = []
    for c, d in cohorts.items():
        for field in ['endometrial_pattern_clean', 'endometrial_pattern_pure_C', 'days_before_transfer', 'uterine_position_clean']:
            for level, count in d[field].fillna('missing_or_unassessable').value_counts().items():
                categories.append(dict(cohort=c, feature=field, level=level, n_exams=int(count),
                                       proportion=float(count / len(d)),
                                       n_unique_patients=d.loc[d[field].fillna('missing_or_unassessable').eq(level), 'patient_id'].nunique()))
    csv('02_CATEGORY_COUNTS_NO_OUTCOME_SPLIT.csv', categories)
    csv('03_TIMING_AND_PATTERN_PROVENANCE_LOCAL.csv', raw[['exam_id', '临床_分型', 'endometrial_pattern_clean',
        'endometrial_pattern_pure_C', 'days_before_transfer', 'days_parse_status', '临床_描述']])
    csv('04_MASTER_CLINICAL9_LOCAL_311.csv', joined)
    # Distribution contrast uses MSE availability, never pregnancy status.
    contrast = []
    a, b = joined[mse], joined[eligible & ~mse]
    for field in CORE + EXTRA:
        av, bv = a[field].dropna(), b[field].dropna()
        denom = np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2)
        contrast.append(dict(feature=field, MSE_available_n=int(av.size), MSE_unavailable_n=int(bv.size),
            available_mean=float(av.mean()), unavailable_mean=float(bv.mean()),
            available_median=float(av.median()), unavailable_median=float(bv.median()),
            available_Q1=float(av.quantile(.25)), available_Q3=float(av.quantile(.75)),
            unavailable_Q1=float(bv.quantile(.25)), unavailable_Q3=float(bv.quantile(.75)),
            descriptive_standardized_mean_difference=float((av.mean() - bv.mean()) / denom) if denom > 0 else np.nan))
    csv('05_MSE_AVAILABILITY_DISTRIBUTION_CONTRAST.csv', contrast)
    blood_audit(raw, cohorts)
    # Labels read only after definitions and source-only distributions are fixed.
    labels = pd.read_excel(BOOK, 0, usecols=['临床检查编号', '临床_临床妊娠'], dtype={'临床检查编号': str})
    labels_model = pd.read_excel(BOOK, 1, usecols=['临床检查编号', '结局_临床妊娠'], dtype={'临床检查编号': str})
    labels = labels.merge(labels_model, on='临床检查编号', validate='one_to_one')
    label_audit = joined[['exam_id', 'patient_id', 'outcome', 'model_identity_eligible', MSE]].merge(
        labels, left_on='exam_id', right_on='临床检查编号', how='left', validate='one_to_one')
    expected = pd.to_numeric(label_audit['结局_临床妊娠'], errors='coerce')
    np.testing.assert_allclose(label_audit.outcome, expected, equal_nan=True)
    label_audit['raw_label_anomaly'] = ~label_audit['临床_临床妊娠'].isin(['是', '否'])
    label_audit['raw_dash_coded_zero_existing_policy'] = label_audit['临床_临床妊娠'].isin(['-', '－', '—']) & label_audit.outcome.eq(0)
    csv('09_OUTCOME_SOURCE_AUDIT_LOCAL.csv', label_audit)
    summary = [dict(cohort=c, n_exams=len(d), n_unique_patients=d.patient_id.nunique(),
        repeated_patient_groups=int(d.patient_id.value_counts().gt(1).sum()),
        pregnancy_n=int(d.outcome.eq(1).sum()), non_pregnancy_n=int(d.outcome.eq(0).sum()),
        clinical9_complete_n=int(d[CORE + EXTRA].notna().all(axis=1).sum()))
        for c, d in cohorts.items() if c in ['eligible306', 'MSE185', 'MSE_unavailable121']]
    csv('10_COHORT_SUMMARY.csv', summary)
    source_counts = raw.endometrial_pattern_clean.fillna('missing_or_unassessable').value_counts().to_dict()
    record = dict(stage='audited_before_prediction', source_hashes={str(p): sha(p) for p in [BOOK, MASTER]},
        source_pattern_counts={k: int(v) for k, v in source_counts.items()},
        pattern_encoding='pure C=1; A/A-B/B/B-C=0; unknown/unassessable=missing;1 df; source-compatible, not a triple-line indicator',
        timing='explicit pre-transfer day/day-of-transfer only; day-of=0; no typo completion, no incomplete unit inference',
        source_timing_missing=int(raw.days_before_transfer.isna().sum()),
        volume='raw clinical volume in mL, continuous, no cutoff/log/nonlinear search',
        all_core6_and_MSE_preserved=True, no_new_exclusions=True,
        source_raw_dash_count=int(labels['临床_临床妊娠'].isin(['-', '－', '—']).sum()),
        eligible_raw_dash_count=int((label_audit.raw_dash_coded_zero_existing_policy & label_audit.model_identity_eligible.eq(True)).sum()),
        MSE_raw_dash_count=int((label_audit.raw_dash_coded_zero_existing_policy & label_audit[MSE].notna()).sum()),
        no_outcome_association_or_feature_performance_examined=True)
    save(OUT / 'DATA_AUDIT_BEFORE_FIT.json', record)
    report = '# Clinical9数据审计（先于预测结果）\n\n'
    report += 'Clinical6原值、原结局、患者键、MSE定义均不变；原311行全保留，沿用306次可分析检查和其中185次MSE病例，不从358源表增加未建立旧分折的病例。\n\n'
    report += f'358源表规范分型：{record["source_pattern_counts"]}。冻结纯C型与其余可评估分型的1df指示，不等同三线征；A/A-B稀少，不单估计其效应。缺失及无法评估不编码为非C，只在训练折众数填补。\n\n'
    report += f'内膜容积按原mL连续值；时间只提取描述中明确的移植前N天/日（含中文数字及第N天），当天0，未明确者缺失。源358时间缺失{record["source_timing_missing"]}条。移植前2缺单位和移值错字不补猜；所记录间隔是描述中的明确数字，尚无实际移植时间戳独立核实。\n\n'
    report += '01/02表报告各队列有效数、缺失率和分布；05表及02表提供MSE有无的分布对照，标准化均数差仅描述，不筛变量、不做结局组间比较。血流只审计，不进入基线。\n\n'
    report += f'原始结局横杠：358源表{record["source_raw_dash_count"]}条，306队列{record["eligible_raw_dash_count"]}条，MSE队列{record["MSE_raw_dash_count"]}条。沿用源建模表既有0编码，单独标记，不视为已核实阴性，亦不因本轮结果更改标签或样本。未链接5条保持缺失。\n\n'
    report += '血流每条原始描述、结构化值、仅内膜血流段解析值、公式残差和异常理由见06–08及BLOOD_AUDIT_SUMMARY.json。位于文字极端的取值不自动当错误；舍入区间不一致只提示核查，不自动修值或删病例。\n'
    (OUT / 'DATA_AUDIT_REPORT_BEFORE_FIT.md').write_text(report, encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


def blood_audit(raw, cohorts):
    rows, flags = [], []
    values = raw[['exam_id']].copy()
    for key, column in DOPPLER.items():
        values[key] = pd.to_numeric(raw[column], errors='coerce')
    for i, row in raw.iterrows():
        tokens = blood_tokens(row['临床_描述'])
        result = dict(exam_id=row.exam_id, original_description=row['临床_描述'])
        reasons = []
        for key in METRICS:
            v, column = values.loc[i, key], DOPPLER[key]
            extracted = sorted(set(float(t) for t in tokens[key]))
            result.update({key: v, key + '_raw': row[column], key + '_description_values': json.dumps(extracted)})
            if len(extracted) == 1 and np.isfinite(v):
                status = 'match' if np.isclose(v, extracted[0], rtol=0, atol=1e-10) else 'mismatch'
            elif len(extracted) == 1:
                status = 'structured_missing_description_present'
            else:
                status = 'description_missing' if not extracted else 'description_ambiguous'
            result[key + '_description_status'] = status
            if status in ['mismatch', 'structured_missing_description_present', 'description_ambiguous']:
                reasons.append(key + ':' + status)
            if pd.notna(row[column]) and not np.isfinite(v):
                reasons.append(key + ':nonnumeric_or_unmeasurable')
            if np.isfinite(v) and ((key in ['VI', 'FI', 'VFI'] and not 0 <= v <= 100)
                                  or (key in ['SD', 'PI'] and v <= 0) or (key == 'RI' and not 0 <= v <= 1)):
                reasons.append(key + ':outside_expected_scale_review_not_automatic_error')
        vi, fi, vfi, sd, ri = [values.loc[i, k] for k in ['VI', 'FI', 'VFI', 'SD', 'RI']]
        if np.isfinite([vi, fi, vfi]).all():
            hvi, hfi, hvfi = [half_rounding(v, tokens[k]) for v, k in [(vi, 'VI'), (fi, 'FI'), (vfi, 'VFI')]]
            low = max(0, vi - hvi) * max(0, fi - hfi) / 100
            high = (vi + hvi) * (fi + hfi) / 100
            compatible = vfi + hvfi >= low - 1e-12 and vfi - hvfi <= high + 1e-12
            result.update(VFI_expected=vi * fi / 100, VFI_signed_residual=vfi - vi * fi / 100,
                VFI_abs_residual=abs(vfi - vi * fi / 100), VFI_rounding_compatible=compatible,
                VFI_half_rounding=hvfi, VI_half_rounding=hvi, FI_half_rounding=hfi)
            if not compatible:
                reasons.append('VFI:incompatible_with_literal_recording_rounding')
        if np.isfinite([sd, ri]).all() and sd > 0:
            hs, hr = half_rounding(sd, tokens['SD']), half_rounding(ri, tokens['RI'])
            expected_ri = 1 - 1 / sd
            result.update(RI_expected_from_SD=expected_ri, RI_abs_residual=abs(ri - expected_ri))
            if sd > hs:
                compatible = ri + hr >= 1 - 1 / (sd - hs) - 1e-12 and ri - hr <= 1 - 1 / (sd + hs) + 1e-12
                result['RI_SD_rounding_compatible_same_waveform_assumption'] = compatible
                if not compatible:
                    reasons.append('RI_SD:rounding_discrepancy_assumes_same_waveform')
        result['review_reasons'] = ';'.join(reasons)
        rows.append(result)
        if reasons:
            flags.append(result)
    audit_frame = pd.DataFrame(rows)
    csv('06_BLOOD_RECORD_AUDIT_LOCAL.csv', audit_frame)
    csv('07_BLOOD_REVIEW_RECORDS_LOCAL.csv', flags)
    distributions, correlations, extrema = [], [], []
    for cohort in ['source358', 'eligible306', 'MSE185', 'MSE_unavailable121']:
        part = values[values.exam_id.isin(cohorts[cohort].exam_id)]
        for key in METRICS:
            v = part[key]
            distributions.append(dict(cohort=cohort, feature=key, n_exams=len(part), n_available=int(v.notna().sum()),
                missing_rate=float(v.isna().mean()), **v.describe(percentiles=[.25, .5, .75]).to_dict()))
            if cohort == 'source358':
                for extreme, value in [('minimum', v.min()), ('maximum', v.max())]:
                    extrema.extend(dict(feature=key, extreme=extreme, exam_id=exam, value=float(value),
                                        meaning='observed extreme, not automatically an error') for exam in part.loc[v.eq(value), 'exam_id'])
        for i, a in enumerate(METRICS):
            for b in METRICS[i + 1:]:
                pair = part[[a, b]].dropna()
                correlations.append(dict(cohort=cohort, feature1=a, feature2=b, n=len(pair),
                    Pearson=float(pair[a].corr(pair[b])), Spearman=float(pair[a].corr(pair[b], method='spearman'))))
    csv('08a_BLOOD_DISTRIBUTIONS.csv', distributions)
    csv('08b_BLOOD_CORRELATIONS.csv', correlations)
    csv('08c_BLOOD_OBSERVED_EXTREMES_LOCAL.csv', extrema)
    summary = dict(n_source=358, review_records=len(flags),
        description_status_counts={k: audit_frame[k + '_description_status'].value_counts().to_dict() for k in METRICS},
        FI_above_100=int(values.FI.gt(100).sum()),
        VFI_rounding_discrepancy_n=int(audit_frame.VFI_rounding_compatible.eq(False).sum()),
        VFI_abs_residual_median=float(audit_frame.VFI_abs_residual.median()),
        VFI_abs_residual_max=float(audit_frame.VFI_abs_residual.max()),
        no_repair_no_patient_deletion_no_blood_model=True,
        interpretation='Rounding intervals derive from literal decimal precision; RI/SD relation assumes same waveform. Flags are review items, not clinical exclusions.',
        references=['https://pmc.ncbi.nlm.nih.gov/articles/PMC5824680/',
                    'https://onlinelibrary.wiley.com/doi/10.1002/uog.6110'])
    save(OUT / 'BLOOD_AUDIT_SUMMARY.json', summary)


if __name__ == '__main__':
    audit()
