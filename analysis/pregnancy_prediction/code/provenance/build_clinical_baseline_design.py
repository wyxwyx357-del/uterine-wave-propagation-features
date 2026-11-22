"""Outcome-blind clinical construct/evidence design; outcomes only for N/prevalence.

No statistical model fitting or outcome association. Source data are never edited.
"""
import os
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('DATA_ROOT', 'data')) / 'outputs'
OUT=ROOT/'clinical_baseline_design_20261120'
OUT.mkdir(exist_ok=True)
BOOK = Path(os.environ.get('CLINICAL_SOURCE_PATH', 'data/clinical_source.xlsx'))
MASTER=ROOT/'expanded_sample_registry_311_20261120/MODEL_INPUT_MASTER_TABLE_311.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(name,frame):pd.DataFrame(frame).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def js(value):return json.dumps(value,ensure_ascii=False)
def table(frame):
    return '\n'.join(['|'+'|'.join(map(str,frame.columns))+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['|'+'|'.join(str(v).replace('|','/').replace('\n',' ') for v in r)+'|' for r in frame.itertuples(index=False,name=None)])

sources={str(p):sha(p) for p in [BOOK,MASTER]}
registry=pd.read_csv(MASTER,dtype={'patient_id':str,'exam_id':str})
raw=pd.read_excel(BOOK,0);model=pd.read_excel(BOOK,1)
assert raw['临床检查编号'].is_unique and model['临床检查编号'].is_unique
raw=raw.rename(columns={c:'raw__'+c for c in raw if c!='临床检查编号'})
model=model.rename(columns={c:'model__'+c for c in model if c!='临床检查编号'})
# Patient outcomes do not enter the clinical data frame used for construct decisions.
base=registry[['case_id','exam_id','patient_id','cohort','model_identity_eligible']].copy()
data=base.merge(raw,left_on='exam_id',right_on='临床检查编号',how='left',validate='many_to_one').drop(columns='临床检查编号')
data=data.merge(model,left_on='exam_id',right_on='临床检查编号',how='left',validate='many_to_one').drop(columns='临床检查编号')
values=base.copy()
field_sources={
 'age':'model__年龄','bmi':'model__BMI','infertility_years':'model__不孕年限_年',
 'infertility_months':'raw__临床_不育年限(月)','endometrial_thickness_mm':'model__内膜厚度_mm',
 'transfer_embryo_count_component':'model__移植胚胎数_解析','transfer_embryo_count_multiple':'model__移植胚胎数_多胚胎',
 'blastocyst_transfer':'model__胚胎种类_含囊胚','transferred_embryo_stage':'model__胚胎阶段_解析',
 'endometrial_type':'raw__临床_分型','endometrial_type_C':'model__内膜分型_C型',
 'uterine_position':'raw__临床_子宫位置','endometrial_volume_ml':'raw__临床_容积(ml)',
 'clinical_peristalsis_positive_cpm':'raw__临床_蠕动（正向）','clinical_peristalsis_negative_cpm':'raw__临床_蠕动（反向）'}
field_sources.update({'doppler_'+k:'raw__临床_'+label+'（内膜血流）' for k,label in [('SD','S/D'),('PI','PI'),('RI','RI'),('VI','VI'),('FI','FI'),('VFI','VFI')]})
categorical={'transferred_embryo_stage','endometrial_type','uterine_position'}
for dest,source in field_sources.items():
    values[dest]=data[source] if dest in categorical else pd.to_numeric(data[source],errors='coerce')
positive=data['raw__临床_蠕动（正向）']
values['clinical_peristalsis_positive_disordered']=positive.astype('string').str.strip().eq('杂乱无序').astype(float).where(positive.notna())
values['stage_known']=values.transferred_embryo_stage.isin(['囊胚','细胞胚'])
# Do not promote an administrative placeholder to a biological third stage.
values['stage_for_candidate']=values.transferred_embryo_stage.where(values.stage_known)
csv('CANDIDATE_VALUES_SOURCE_AUDIT_ONLY.csv',values)

discrepancies=[]
for field in ['age','bmi','infertility_years','endometrial_thickness_mm','transfer_embryo_count_multiple','blastocyst_transfer','endometrial_type_C']:
    old=pd.to_numeric(registry[field],errors='coerce');new=pd.to_numeric(values[field],errors='coerce')
    diff=old.notna().ne(new.notna()) | (old.notna()&new.notna()&~np.isclose(old,new))
    for idx in np.flatnonzero(diff):
        discrepancies.append(dict(case_id=registry.loc[idx,'case_id'],exam_id=registry.loc[idx,'exam_id'],field=field,
            current_master_value=old.iloc[idx],source358_model_value=new.iloc[idx],
            reason='master filled raw integer year and omitted month fraction' if field=='infertility_years' else 'master backfilled raw BMI that source modeling sheet already treated as missing' if field=='bmi' else 'requires source review',
            action='record only; original master and frozen results unchanged'))
csv('CLINICAL_SOURCE_DISCREPANCIES.csv',discrepancies)
years=pd.to_numeric(data['raw__临床_不育年限(年)'],errors='coerce')
months=values.infertility_months
duration_ok=years.notna()&months.notna()&values.infertility_years.notna()
duration_agree=np.isclose(values.loc[duration_ok,'infertility_years'],years[duration_ok]+months[duration_ok]/12)
count_ok=values.transfer_embryo_count_component.notna()&values.transfer_embryo_count_multiple.notna()
count_agree=values.loc[count_ok,'transfer_embryo_count_component'].gt(1).eq(values.loc[count_ok,'transfer_embryo_count_multiple'])
stage_ok=values.stage_known & values.blastocyst_transfer.notna()
stage_agree=values.loc[stage_ok,'transferred_embryo_stage'].eq('囊胚').astype(int).eq(values.loc[stage_ok,'blastocyst_transfer'])
pattern_agreement=pd.crosstab(values.endometrial_type,values.endometrial_type_C,dropna=False)
pattern_agreement.to_csv(OUT/'ENDOMETRIAL_PATTERN_ENCODING_AUDIT.csv',encoding='utf-8-sig')

construct_specs=[
 ('Female age',['age'],'age','',False,'超声检查时女性年龄；并非保证为取卵时年龄','A','core',1,'yes','年龄是重复出现的主要预后构念；不从本队列效应决定是否纳入'),
 ('BMI',['bmi'],'bmi','',False,'女性体质量指数，kg/m²','B','extended',1,'yes','FET临床妊娠证据较混合；外部证据支持考虑，不等同线性保护/危险方向'),
 ('Infertility duration',['infertility_years','infertility_months'],'infertility_years','infertility_months',True,'358建模表既有总年限=原年+月份余数/12','B','extended',1,'yes','保留完整时长，不将月份余数当第二个独立变量'),
 ('Endometrial thickness',['endometrial_thickness_mm'],'endometrial_thickness_mm','',False,'临床超声内膜厚度，mm；需明确周期测量时点','B','extended',1,'yes','关联证据较多但异质性与测量时点问题存在；不是单一容受性诊断'),
 ('Embryo number',['transfer_embryo_count_component','transfer_embryo_count_multiple'],'transfer_embryo_count_component','transfer_embryo_count_multiple',True,'已确定的实际移植胚胎数1/2/3；用户确认预测时点已知','A','core',1,'yes','优先实际计数，multiple为替代表示；不同时纳入，也不推导移植治疗建议'),
 ('Embryo developmental stage',['transferred_embryo_stage','blastocyst_transfer'],'blastocyst_transfer','stage_for_candidate',True,'已确定的实际阶段；有效值囊胚/细胞胚，二者可用1df表示','A','core',1,'yes','当前有效阶段只有两类，与囊胚指示一致；待补录不能作为生物学第三类'),
 ('Endometrial pattern',['endometrial_type','endometrial_type_C'],'endometrial_type_C','endometrial_type',True,'C型与非纯C型既有二值编码；B-C型原表编码为0需保留解释','C','sensitivity',1,'yes','形态证据不稳定，完整分类稀疏且增加df，不放核心'),
 ('Uterine position',['uterine_position'],'uterine_position','',False,'前/后/水平的名义分类','C','sensitivity',2,'yes','不是有充分支持的核心IVF妊娠预后因素；不依据旧模型表现纳入'),
 ('Endometrial volume',['endometrial_volume_ml'],'endometrial_volume_ml','',False,'内膜容积mL；勾画及设备协议需要说明','C','sensitivity',1,'yes','与厚度有相关生理信息但非完全派生，独立预测证据有限'),
 ('Doppler resistance',['doppler_SD','doppler_PI','doppler_RI'],'doppler_PI','doppler_RI;doppler_SD','related','频谱血流阻抗指标；SD与RI在同一波形下存在数学关系','C','sensitivity',1,'yes','不把三项当独立信息；当前血管部位标注不足，子宫动脉证据不能直接转用于内膜微血管'),
 ('Doppler vascularity',['doppler_VI','doppler_FI','doppler_VFI'],'doppler_VI','doppler_FI;doppler_VFI','related','3D能量多普勒血管比例VI、平均强度FI及派生VFI','C','sensitivity',1,'yes','VI/FI不完全相同，VFI通常由VI×FI/100派生；FI异常未解决，FI不进入本轮候选方案'),
 ('Clinically reported peristalsis',['clinical_peristalsis_positive_cpm','clinical_peristalsis_negative_cpm','clinical_peristalsis_positive_disordered'],'clinical_peristalsis_positive_cpm','clinical_peristalsis_negative_cpm;clinical_peristalsis_positive_disordered','related','人工记录的正/反向频率和杂乱类别；方向与C2F/F2C映射未确认','B','sensitivity',3,'yes','与影像运动构念重叠；加人工指标改变增量研究问题，可单独作为更强比较基线，不称信息泄漏'),
 ('Infertility cause unstructured',['raw__临床_诊断'],'','',False,'诊断自由文本，未建立结局盲的结构化病因编码','D','exclude',None,'unclear','不能从包含复杂诊疗信息的文本自动抽取为核心变量'),
 ('Prior reproductive history',['not_structured_in_current_table'],'','',False,'既往妊娠/分娩/失败移植史未有可核实独立结构字段','D','exclude',None,'unclear','外部文献重要但当前不可用；不把本周期结局当既往史'),
 ('Treatment cycle type',['not_structured_in_current_table'],'','',False,'自然/人工周期及新鲜/冻融流程未有统一已核实字段','D','exclude',None,'unclear','不能从囊胚/细胞胚目录猜新鲜或冻融；记录为剩余混杂及外推限制'),
 ('Post-outcome fields',['raw__临床_手术结局','raw__临床_生化妊娠','raw__临床_临床妊娠'],'','',False,'当前周期结局及可能后验描述','D','exclude',None,'no','只作为结局或核对字段，禁止预测输入和变量排序'),
]

quality=[];constructs=[];candidates=[];outliers=[]
for name,fields,preferred,alternative,duplicate,definition,grade,role,df,at_time,reason in construct_specs:
    present=[f for f in fields if f in values]
    v=values[preferred] if preferred in values else pd.Series(np.nan,index=values.index)
    numeric=pd.to_numeric(v,errors='coerce')
    repeated=[]
    for patient,sub in values[values.model_identity_eligible.eq(True)].groupby('patient_id'):
        if len(sub)>1:
            unique=sub[preferred].dropna().nunique() if preferred in sub else 0
            repeated.append(unique)
    issue=''
    if name=='BMI':issue='原始BMI有1条101.9；358建模表已置缺失而旧master补回，需源病历核实，不能猜成10.19或体重。原始16.1–101.9，建模有效16.1–41.9。12–60是源建模表历史处理，非本轮新阈值。'
    if name=='Infertility duration':issue=f'年+月/12在{int(duration_ok.sum())}/{int(duration_ok.sum())}可比较源记录中一致；旧master有44条漏月差异。本轮只标记，不修复正式值。'
    if name=='Embryo number':issue=f'计数>1与multiple在{int(count_ok.sum())}条可比较记录中一致；3胚胎只有2条，计数线性是简约假设，1/2/3名义表示需2df。'
    if name=='Embryo developmental stage':issue='囊胚185、细胞胚119，另2条待补录；待补录不当真实阶段，现有indicator已缺失。无混合阶段的充分结构信息。'
    if name=='Endometrial pattern':issue='C176/B120/B-C5/A2/无法评估1/无型1；原二值编码非纯C含B-C。完整有效A/B/C/B-C需3df且稀疏。'
    if name=='Doppler resistance':issue='血管、采样位置、设备及周期时点未充分标准化；不凭高相关或某个取值自动删患者。'
    if name=='Doppler vascularity':issue='FI>100有9条，另1条86.53仅触发旧>80提醒而非判为错误；最大388.5。文件2/文件3及源行可追溯，但原设备量纲/源检查报告未提供，不能认定小数点错误或自动除10。'
    if name=='Clinically reported peristalsis':issue='正向字段1条杂乱无序、1条原缺失；direction未映射C2F/F2C，检查时段与保存视频未确认一致；指示变量有1条阳性，稀疏。'
    if name in ['Prior reproductive history','Treatment cycle type','Infertility cause unstructured','Post-outcome fields']:issue=reason
    levels=js(v.dropna().astype(str).value_counts().to_dict()) if preferred in categorical else ''
    range_text=f'{numeric.min():g}–{numeric.max():g}' if numeric.notna().any() else ''
    field_avail={f:int(values[f].notna().sum()) for f in present}
    quality.append(dict(construct=name,preferred_field=preferred,n_exams=311,n_available=int(v.notna().sum()),missing_rate=float(v.isna().mean()),
        n_available_patients=int(values.loc[v.notna(),'patient_id'].nunique()),observed_range=range_text,categorical_levels=levels,
        field_specific_n=js(field_avail),suspicious_values=issue,derived_duplicate=str(duplicate),
        repeated_patient_consistency=f'{len(repeated)} repeated groups; {sum(x>1 for x in repeated)} groups with differing observed values; cycle-dependent changes are not automatically errors',
        likely_measurement_error='suspected_unresolved' if name in ['BMI','Doppler vascularity'] else 'not established',
        usable_for_model_yes_no='conditional' if role in ['sensitivity','extended'] else 'yes' if role=='core' else 'no',
        source_policy='preferred values read from existing358 modeling/raw sheet; current master unchanged'))
    constructors=dict(construct_name=name,raw_fields=';'.join(fields),preferred_field=preferred,alternative_field=alternative,
        duplicate_or_derived=str(duplicate),definition=definition,
        measurement_time='user confirms actual number/stage fixed by ultrasound completion' if name.startswith('Embryo') else 'pre-outcome clinical ultrasound/cycle record; exact phase varies' if at_time=='yes' else 'not established or post-outcome',
        available_before_outcome=at_time,available_at_prediction_time=at_time,missing_rate=float(v.isna().mean()),data_quality_issue=issue,
        coding_issue=reason,number_of_model_parameters_if_included=df,df_required=df,linear_df=df,
        nonlinear_df_possible='3 total df per continuous variable with a pre-specified 4-knot RCS; +2 beyond linear; not fitted' if name in ['Female age','BMI','Infertility duration','Endometrial thickness','Endometrial volume','Doppler resistance','Doppler vascularity'] else 'embryo count: linear1df or nominal1/2/3 needs2df; no automatic spline' if name=='Embryo number' else 'categorical K-1; binary1df; no automatic spline',
        notes='year/month, count/indicator, stage/indicator or pattern/indicator are alternatives, not independent additions; no current outcome association examined')
    constructs.append(constructors)
    candidates.append(dict(construct=name,literature_evidence_grade=grade,data_quality='source BMI policy must be harmonized; raw abnormal value unresolved' if name=='BMI' else '44 master/source discrepancies; prefer existing358 total-year representation' if name=='Infertility duration' else 'FI quality unresolved; VI/VFI not automatically cleared' if name=='Doppler vascularity' else 'sparse/heterogeneous coding; see quality table' if name in ['Endometrial pattern','Clinically reported peristalsis'] else 'source definition needs standardization' if role in ['exclude','sensitivity'] else 'acceptable basic coding, retain missing values',missingness=float(v.isna().mean()),
        prediction_time_available=at_time,df_required=df,duplicate_status=str(duplicate),recommended_priority='1' if role=='core' else '2' if role=='extended' else '3' if role=='sensitivity' else 'not eligible',
        baseline_role=role,preferred_field=preferred,reason=reason,
        freeze_status='candidate only; source harmonization and parameter budget review pending'))
csv('CLINICAL_CONSTRUCT_MAP.csv',constructs);csv('CLINICAL_DATA_QUALITY.csv',quality);csv('CLINICAL_BASELINE_CANDIDATES.csv',candidates)

for field in ['bmi','doppler_FI']:
    vv=pd.to_numeric(data['raw__临床_女方BMI'],errors='coerce') if field=='bmi' else values.doppler_FI
    flag=vv.eq(101.9) if field=='bmi' else vv.gt(80)
    for idx in np.flatnonzero(flag):
        outliers.append(dict(exam_id=data.loc[idx,'exam_id'],case_id=data.loc[idx,'case_id'],field=field,raw_value=vv.iloc[idx],
            source_file_label=data.loc[idx,'raw__临床数据来源文件'],source_excel_row=data.loc[idx,'raw__临床数据来源行'],
            classification='source modeling sheet already missing; unresolved raw value' if field=='bmi' else 'outside conventional0to100 FI scale if applicable; confirm device/units' if vv.iloc[idx]>100 else 'old80 alert only; not inherently invalid',
            automatic_correction=False))
csv('SUSPICIOUS_SOURCE_VALUES_LOCAL.csv',outliers)
doppler=['doppler_SD','doppler_PI','doppler_RI','doppler_VI','doppler_FI','doppler_VFI']
corr=[]
for i,a in enumerate(doppler):
    for b in doppler[i+1:]:
        pair=values[[a,b]].dropna()
        corr.append(dict(field1=a,field2=b,n=len(pair),spearman=float(pair[a].corr(pair[b],method='spearman')),use='description only; no outcome screening or automatic deletion'))
csv('DOPPLER_CONSTRUCT_CORRELATION.csv',corr)
residual=(values.doppler_VFI-values.doppler_VI*values.doppler_FI/100).abs()
video_dates=pd.to_datetime(data['raw__视频检查时间（首帧）'],errors='coerce')
clinical_dates=pd.to_datetime(data['raw__临床_容受检查时间'],errors='coerce')
same_day=(clinical_dates.dt.normalize()-video_dates.dt.normalize()).dt.days

# Three candidate baselines; no algorithm is fitted or chosen.
options={'Option A Core':['age','transfer_embryo_count_component','blastocyst_transfer'],
         'Option B Extended':['age','transfer_embryo_count_component','blastocyst_transfer','bmi','infertility_years','endometrial_thickness_mm'],
         'Option C Sensitivity':['age','transfer_embryo_count_multiple','blastocyst_transfer','bmi','infertility_years','endometrial_thickness_mm','endometrial_type_C']}
scopes={'registry311':np.ones(len(values),bool),'Clinical_previous296':registry.A_clinical7_analysis_eligible,
        'B179':registry.B_clinical_and_four_imaging_analysis_eligible,
        'Clinical_activity240':registry.clinical_plus_algorithm_activity_rate_exploratory_analysis_eligible}
option_rows=[]
for option,fields in options.items():
    for scope,mask in scopes.items():
        eligible=mask&values[fields].notna().all(axis=1)&values.model_identity_eligible.eq(True)
        option_rows.append(dict(option=option,analysis_scope=scope,fields=';'.join(fields),clinical_parameters=len(fields),
            scope_n=int(np.asarray(mask).sum()),complete_exams=int(eligible.sum()),patients=values.loc[eligible,'patient_id'].nunique(),
            missing_exams=int(np.asarray(mask).sum()-eligible.sum()),n_with_outcome=int(registry.loc[eligible,'outcome'].isin([0,1]).sum()),
            note='counts from source358 representation, not existing master backfill; no outcomes used to rank options'))
csv('BASELINE_OPTION_AVAILABILITY.csv',option_rows)

# Riley/pmsampsize binary criteria, inverted for p. No local fitted R² or AUC used.
refs={'Clinical_previous296':registry.A_clinical7_analysis_eligible,
      'B179':registry.B_clinical_and_four_imaging_analysis_eligible,
      'Clinical_activity240':registry.clinical_plus_algorithm_activity_rate_exploratory_analysis_eligible}
source7=values[['age','bmi','infertility_years','transfer_embryo_count_multiple','endometrial_thickness_mm','blastocyst_transfer','endometrial_type_C']].notna().all(axis=1)&values.model_identity_eligible.eq(True)
refs['Clinical_source_consistent295']=source7
refs['Core_source302']=values[options['Option A Core']].notna().all(axis=1)&values.model_identity_eligible.eq(True)
refs['Extended_source298']=values[options['Option B Extended']].notna().all(axis=1)&values.model_identity_eligible.eq(True)
budgets=[];required=[];cohorts=[]
scenarios=[('weak_002',.02),('weak_005',.05),('moderate_010',.10),('moderate_015',.15),('optimistic_sensitivity_020',.20)]
for name,mask in refs.items():
    # Only outcome counts/proportions are used in this section.
    y=registry.loc[mask,'outcome'].dropna()
    n_exams=len(y);n_patients=registry.loc[mask&registry.outcome.notna(),'patient_id'].nunique()
    event_rate=float(y.mean())
    cohorts.append(dict(analysis_set=name,n_exams=n_exams,n_patients=n_patients,events=int(y.sum()),non_events=int(y.eq(0).sum()),event_rate=event_rate))
    max_r2=1-math.exp(2*(event_rate*math.log(event_rate)+(1-event_rate)*math.log(1-event_rate)))
    for signal,r2 in scenarios:
        s2=r2/(r2+.05*max_r2)
        denom2=(s2-1)*math.log(1-r2/s2)
        for shrink in [.9,.95]:
            denom1=(shrink-1)*math.log(1-r2/shrink)
            n3=math.ceil((1.96/.05)**2*event_rate*(1-event_rate))
            for unit,n_eff in [('exam_independence_approximation',n_exams),('unique_patient_conservative_reference',n_patients)]:
                cap=min(math.floor(n_eff*denom1+1e-12),math.floor(n_eff*denom2+1e-12))
                pass3=n_eff>=n3
                budgets.append(dict(analysis_set=name,n_exams=n_exams,n_patients=n_patients,event_rate=event_rate,events=int(y.sum()),
                    assumed_R2=r2,R2_scale='anticipated adjusted Cox-Snell; not local fit',signal_scenario=signal,
                    observation_basis=unit,n_for_budget=n_eff,allowed_candidate_parameters=cap if pass3 else 0,
                    allowed_parameters_shrinkage_and_optimism_only=cap,shrinkage_target=shrink,
                    Nagelkerke_optimism_target=.05,overall_risk_precision_target=.05,overall_risk_minimum_n=n3,overall_risk_precision_met=pass3,
                    achieved_risk_halfwidth_approx=1.96*math.sqrt(event_rate*(1-event_rate)/n_eff),max_Cox_Snell_R2=max_r2,
                    comments='full3-criteria budget=0 when overall-risk precision fails; cap shown separately is NOT a full pmsampsize pass. Cluster correlation not modeled; patient-n reference retains exam prevalence because no single patient-level cycle outcome was defined. Intercept excluded from predictor parameter p; count all imaging/spline/interaction/missing-indicator df.'))
                for option,fields in options.items():
                    for extra,extra_df in [('clinical_only',0),('plus_four_imaging',4),('plus_activity',1),('plus_five_imaging',5)]:
                        p=len(fields)+extra_df
                        n1=math.ceil(p/denom1);n2=math.ceil(p/denom2)
                        required.append(dict(analysis_set=name,observation_basis=unit,n_available=n_eff,option=option,
                            imaging_extension=extra,clinical_df=len(fields),imaging_df=extra_df,total_predictor_df=p,
                            assumed_R2=r2,shrinkage_target=shrink,n_required_shrinkage=n1,n_required_optimism=n2,
                            n_required_overall_risk=n3,n_required_all_criteria=max(n1,n2,n3),
                            all_criteria_met=n_eff>=max(n1,n2,n3),note='scenario reference, not a license to tune to an observed AUC'))
csv('MODEL_PARAMETER_BUDGET.csv',budgets);csv('MODEL_TOTAL_PARAMETER_REQUIREMENTS.csv',required);csv('COHORT_BUDGET_COUNTS.csv',cohorts)
checks=dict(n_registry=311,identity_linked_exams=int(values.model_identity_eligible.eq(True).sum()),
    outcome_used_only_for_counts_prevalence=True,no_models_fit=True,no_auc_computed=True,no_outcome_variable_ranking=True,
    source_files_unchanged=all(sha(Path(p))==h for p,h in sources.items()),
    year_month_pairs=int(duration_ok.sum()),year_month_matches=int(duration_agree.sum()),
    embryo_number_pairs=int(count_ok.sum()),embryo_number_matches=int(count_agree.sum()),
    stage_pairs=int(stage_ok.sum()),stage_matches=int(stage_agree.sum()),
    master_duration_discrepancies=sum(r['field']=='infertility_years' for r in discrepancies),
    master_BMI_missing_policy_discrepancies=sum(r['field']=='bmi' for r in discrepancies),
    FI_above100=int(values.doppler_FI.gt(100).sum()),FI_above80_old_alert=int(values.doppler_FI.gt(80).sum()),
    VFI_formula_abs_residual_median=float(residual.median()),VFI_formula_abs_residual_p90=float(residual.quantile(.9)),
    clinical_video_date_pairs=int(same_day.notna().sum()),clinical_video_same_day=int(same_day.eq(0).sum()),
    embryo_timing_user_confirmed='2026-11-20: actual embryo count and stage determined by ultrasound completion',
    clinical_baseline_frozen=False,source_sha256=sources)
assert checks['source_files_unchanged'] and duration_agree.all() and count_agree.all() and stage_agree.all()
(OUT/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k!='source_sha256'},ensure_ascii=True))
print(pd.DataFrame(option_rows).query("analysis_scope=='registry311'")[['option','clinical_parameters','complete_exams','patients']].to_string(index=False))
print(pd.DataFrame(budgets).query("shrinkage_target==.9 and observation_basis=='unique_patient_conservative_reference'")[['analysis_set','assumed_R2','allowed_candidate_parameters','allowed_parameters_shrinkage_and_optimism_only','overall_risk_minimum_n']].to_string(index=False))
