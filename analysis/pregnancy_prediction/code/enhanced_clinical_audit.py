"""Outcome-blind PI/VFI missing-value rules for Enhanced Clinical baseline v1."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.pregnancy_prediction.code.clinical9_data_audit import ROOT,BOOK,OLD,DOPPLER,MSE,sha,save,blood_tokens

PRIOR=ROOT/'outputs/clinical9_frozen_mse_20261122'
BLOOD_PRIOR=ROOT/'outputs/clinical11_PI_VFI_mse_sensitivity_20261122'
OUT=ROOT/'outputs/enhanced_clinical_v1_20261122'
BLOOD=['PI_clean','VFI_clean']


def csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')


def audit():
    OUT.mkdir(exist_ok=False)
    prior_protocol=json.loads((BLOOD_PRIOR/'PROTOCOL_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert sha(BOOK)==prior_protocol['source_hashes'][str(BOOK)]
    # No outcome column, prediction file or patient-level label is loaded here.
    raw=pd.read_excel(BOOK,0,usecols=['临床检查编号','临床_描述']+list(DOPPLER.values()),
        dtype={'临床检查编号':str}).rename(columns={'临床检查编号':'exam_id'})
    old=pd.read_csv(BLOOD_PRIOR/'02_BLOOD_RECORD_AUDIT_LOCAL.csv',dtype={'exam_id':str}).set_index('exam_id')
    assert len(raw)==358 and raw.exam_id.is_unique
    rows=[]
    for _,r in raw.iterrows():
        previous=old.loc[r.exam_id]
        tokens=blood_tokens(r['临床_描述'])
        value={key:pd.to_numeric(r[col],errors='coerce') for key,col in DOPPLER.items()}
        for key,v in value.items():
            np.testing.assert_allclose(v,previous[key],equal_nan=True)
        pi_reasons,vfi_reasons=[],[]
        if np.isfinite(value['PI']) and value['PI']<=0:
            pi_reasons.append('nonpositive_PI_unconfirmed')
        if previous.PI_status=='mismatch':
            pi_reasons.append('PI_source_description_conflict')
        if np.isfinite(value['VFI']) and not 0<=value['VFI']<=100:
            vfi_reasons.append('VFI_outside_recorded_scale_unconfirmed')
        if previous.VFI_status=='mismatch':
            vfi_reasons.append('VFI_source_description_conflict')
        if previous.VFI_rounding_compatible is False or previous.VFI_rounding_compatible==False:
            vfi_reasons.append('VFI_VI_FI_formula_conflict_unresolved')
        if np.isfinite(value['FI']) and value['FI']>100:
            vfi_reasons.append('FI_scale_unresolved_for_derived_VFI')
        if np.isfinite(value['VI']) and not 0<=value['VI']<=100:
            vfi_reasons.append('VI_scale_unresolved_for_derived_VFI')
        # Missing descriptive parsing alone is not an abnormal value; no new cutoff
        # labels a finite extreme as erroneous. Existing raw missing stays missing.
        result=dict(exam_id=r.exam_id,PI_raw=value['PI'],VFI_raw=value['VFI'],VI_audit_only=value['VI'],FI_audit_only=value['FI'],
            PI_clean=np.nan if pi_reasons else value['PI'],VFI_clean=np.nan if vfi_reasons else value['VFI'],
            PI_mask_reason=';'.join(pi_reasons),VFI_mask_reason=';'.join(vfi_reasons),
            PI_description_status=previous.PI_status,VFI_description_status=previous.VFI_status,
            VFI_expected_audit_only=previous.get('VFI_expected',np.nan),
            VFI_formula_abs_error=previous.get('VFI_abs_error',np.nan),
            FI_original_report_values=json.dumps([float(t) for t in tokens['FI']]),
            original_description=r['临床_描述'])
        rows.append(result)
    cleaned=pd.DataFrame(rows)
    assert cleaned.PI_mask_reason.eq('').all()
    assert cleaned.VFI_mask_reason.ne('').sum()==25
    csv('01_PI_VFI_CLEANING_PROVENANCE_LOCAL.csv',cleaned)
    meta=pd.read_csv(PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',usecols=['exam_id','patient_id',MSE],
        dtype={'exam_id':str,'patient_id':str})
    outer=pd.read_csv(OLD/'02a_GLOBAL_PATIENT_FOLDS.csv',usecols=['repeat','exam_id'],dtype={'exam_id':str})
    eligible_ids=set(outer.loc[outer['repeat'].eq(0),'exam_id'])
    d=meta.merge(cleaned.drop(columns='original_description'),on='exam_id',how='left',validate='one_to_one',sort=False)
    assert d.exam_id.tolist()==meta.exam_id.tolist()
    csv('02_CLEAN_BLOOD_VALUES_LOCAL_311.csv',d[['exam_id']+BLOOD])
    cohorts={'source358':cleaned,'eligible306':d[d.exam_id.isin(eligible_ids)],
        'MSE185':d[d.exam_id.isin(eligible_ids)&d[MSE].notna()]}
    distributions=[]
    for cohort,frame in cohorts.items():
        for field in ['PI_raw','PI_clean','VFI_raw','VFI_clean']:
            v=frame[field]
            distributions.append(dict(cohort=cohort,field=field,n_exams=len(frame),n_available=int(v.notna().sum()),
                n_missing=int(v.isna().sum()),missing_rate=float(v.isna().mean()),minimum=float(v.min()),Q1=float(v.quantile(.25)),
                median=float(v.median()),Q3=float(v.quantile(.75)),maximum=float(v.max())))
    csv('03_BLOOD_AVAILABILITY_BEFORE_AFTER.csv',distributions)
    summary=dict(stage='audit_and_cleaning_rules_frozen_before_outcome_load',outcome_columns_loaded=False,
        prior_policy='previous Clinical11 retained flags/raw VFI; its source/results remain unchanged',
        new_user_policy='unconfirmed anomalous values become missing in Enhanced-v1 copy, no guessing',
        PI_rule='original finite positive value; source-description conflict to missing; no percentile cutoff',
        VFI_rule='source conflict/out-of-scale, formula conflict beyond recorded rounding, or unconfirmed FI/VI scale makes VFI missing; never replace by VI*FI/100',
        source_formula_flags=13,source_FI_scale_flags=12,source_VFI_newly_masked=25,
        no_outcome_or_AUC_based_cleaning=True,no_patient_deletion_for_clinical_missing=True,
        clinically_missing='training-only median continuous, mode categorical; no missingness predictor indicators',
        expected_exams={'Full306':306,'MSE185':185},
        clean_blood_missing={c:{f:int(frame[f].isna().sum()) for f in BLOOD} for c,frame in cohorts.items()},
        limitations='masking uncertain values does not verify remaining measurements; FI units/decimal unresolved',
        source_hashes={str(p):sha(p) for p in [BOOK,PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',
            BLOOD_PRIOR/'02_BLOOD_RECORD_AUDIT_LOCAL.csv',BLOOD_PRIOR/'STOP_RULE.json',
            OLD/'02a_GLOBAL_PATIENT_FOLDS.csv']})
    save(OUT/'BLOOD_CLEANING_BEFORE_OUTCOME.json',summary)
    text='# Enhanced Clinical v1：无结局血流清洗\n\n'
    text+='按本轮明确新指令，无法确认的异常在新副本记缺失：VFI公式冲突13条与FI>100量表疑点12条无重叠，共25条VFI设缺失；PI未发现需新增置空的可确认异常。12条FI高值均在原描述中，不能据此猜小数点或单位。公式冲突不能定位哪个分量有误，故不推导替代值。仅描述无法解析或数值位于样本极端，不自动算异常。\n\n'
    text+=f'源358、306和MSE185的PI/VFI清洗前后分布/缺失见03表；置空原因逐条见01表。各集合清洗后缺失：{summary["clean_blood_missing"]}。本轮不要求血流完整病例，不删患者；原306/288和MSE185/179全部保留，临床缺失仅训练折填补。\n\n'
    text+='清洗脚本未读取结局列或预测结果；此前分析已公开，不称整体研究从未接触结局。新策略只适用于Enhanced-v1，不覆盖此前保留原报告值的Clinical11敏感性。\n'
    (OUT/'DATA_AUDIT_BEFORE_FIT.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
