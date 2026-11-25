"""Read existing tables only; desktop whitelist defines the main denominator.

No model imports, feature extraction, video reads, or outcome comparisons.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import os
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('DATA_ROOT', 'data')) / 'outputs'
PROJECT = ROOT / 'uterine-wave-propagation-features'
OLD = PROJECT / 'outputs/feature_inventory_20261120'
OUT = PROJECT / 'outputs/feature_inventory_desktop276_20261120'
LEGACY = ROOT / 'R2_frequency_GTX1080_20261103/R2_frequency_GTX1080_20261103/R2_frequency_frozen_20261102'
WB = Path(os.environ.get('DATA_ROOT', 'data')) / 'Desktop/一分钟以上患者视频.xlsx'
OUT.mkdir(exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
def read(name):
    return pd.read_csv(OLD / name)
def clean(s):
    return s.astype('string').str.strip()
def md(df):
    lines=['|'+'|'.join(map(str,df.columns))+'|','|'+'|'.join(['---']*len(df.columns))+'|']
    lines += ['|'+'|'.join(str(v).replace('|','/').replace('\n',' ') for v in row)+'|' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)

w = pd.read_excel(WB)
assert len(w) == 276 and w['临床检查编号'].notna().all()
assert not w['临床检查编号'].duplicated().any()
previous = read('MODEL_INPUT_MASTER_TABLE.csv')
main = pd.DataFrame({'exam_id': clean(w['临床检查编号']),
                     'case_id_desktop': clean(w['视频目录标识']),
                     'patient_name_key': clean(w['临床患者姓名']),
                     'desktop_video_duration_s': pd.to_numeric(w['视频时长（秒，CSV估计）'], errors='coerce'),
                     'desktop_acquisition_date': w['视频检查时间（首帧）'],
                     'desktop_source': w['临床数据来源文件']})
main = main.merge(previous, on='exam_id', how='left', validate='one_to_one', indicator=True)
main['existing_result_link'] = main['_merge'].eq('both')
main = main.drop(columns='_merge')
main['cohort'] = 'desktop_276'
main['case_id'] = main['case_id'].fillna(main['case_id_desktop'])
main['video_duration_s'] = main['video_duration_s'].fillna(main['desktop_video_duration_s'])
main['acquisition_date'] = main['acquisition_date'].fillna(main['desktop_acquisition_date'])
main['source'] = main['source'].fillna(main['desktop_source'])
dates=pd.to_datetime(main['desktop_acquisition_date'],errors='coerce')
main['video_year']=main['video_year'].fillna(dates.dt.year)
main['video_month']=dates.dt.month
main['source_batch']=main['source_batch'].fillna(main['desktop_source'])
# Name matches identify plausible repeat groups, not verified immutable identities.
main['patient_id'] = 'name:' + main['patient_name_key']
main['patient_id_status'] = 'provisional_exact_clinical_name'
for col in w.columns:
    main['desktop_raw__' + col] = w[col].to_numpy()
numeric = {'age':'临床_年龄','bmi':'临床_女方BMI','infertility_years':'临床_不育年限(年)',
           'infertility_months':'临床_不育年限(月)','endometrial_thickness_mm':'临床_子宫内膜厚度(mm)',
           'endometrial_volume_ml':'临床_容积(ml)','clinical_peristalsis_positive_cpm':'临床_蠕动（正向）',
           'clinical_peristalsis_negative_cpm':'临床_蠕动（反向）'}
numeric.update({f'doppler_{k}':f'临床_{label}（内膜血流）' for k,label in [('SD','S/D'),('PI','PI'),('RI','RI'),('VI','VI'),('FI','FI'),('VFI','VFI')]})
for target, source in numeric.items():
    main[target] = main[target].fillna(pd.to_numeric(w[source], errors='coerce'))
for target, source in [('uterine_position','临床_子宫位置'),('endometrial_type','临床_分型')]:
    main[target] = main[target].replace('<missing>',np.nan).fillna(w[source])
# Existing derived clinical codes are reused, never reparsed with a new rule.
clinical_book = Path(os.environ.get('DATA_ROOT', 'data')) / 'Desktop/最终人工复核队列_358次检查.xlsx'
model_sheet = pd.read_excel(clinical_book, sheet_name=1)
model_sheet['exam_id'] = clean(model_sheet['临床检查编号'])
assert not model_sheet['exam_id'].duplicated().any()
code_map = {'transfer_embryo_count_multiple':'移植胚胎数_多胚胎',
            'blastocyst_transfer':'胚胎种类_含囊胚', 'endometrial_type_C':'内膜分型_C型',
            'transfer_embryo_count_component':'移植胚胎数_解析',
            'transferred_embryo_stage':'胚胎阶段_解析'}
for target, source in code_map.items():
    if source in model_sheet:
        linked = main['exam_id'].map(model_sheet.set_index('exam_id')[source])
        main[target] = main[target].replace('<missing>',np.nan).fillna(linked)
main['patient_id'] = main.exam_id.map(model_sheet.set_index('exam_id')['患者聚类标识']).fillna(main.patient_id)
main['patient_id_status'] = np.where(main.exam_id.isin(model_sheet.exam_id),'existing_clinical_cluster_key','provisional_exact_clinical_name')
# Outcome is retained only for requested availability/class counts, never predictors.
clinical_outcome = pd.to_numeric(w['临床_临床妊娠'],errors='coerce')
main['outcome'] = main['outcome'].fillna(clinical_outcome.where(clinical_outcome.isin([0,1])))
main['outcome'] = main['outcome'].fillna(main.exam_id.map(model_sheet.set_index('exam_id')['结局_临床妊娠']))
main['clinical_link_available'] = main.exam_id.isin(model_sheet.exam_id)
input_path=PROJECT/'outputs/frozen_feature_pipeline_20261119/input_inventory.csv'
inputs=pd.read_csv(input_path).set_index('case_id')
for field in ['processing_status','failure_reason','video_frame_count','video_fps','time_factor','coordinate_scale']:
    main['input_audit_'+field]=main.case_id.map(inputs[field])
save(main[['exam_id','case_id','existing_result_link','input_audit_processing_status','input_audit_failure_reason']], 'COHORT_INPUT_STATUS.csv')
hold = previous.loc[~previous.exam_id.isin(main.exam_id)].copy()
hold['cohort'] = 'short_holdout_separate'
hold['patient_id_status'] = 'unverified_exam_fallback'
hold['patient_id'] = 'exam_unverified:' + hold.exam_id.astype(str)
master = pd.concat([main,hold],ignore_index=True)

# Reuse archived scalar features via exact case_id only; no recomputation.
inventory = read('FEATURE_INVENTORY.csv')
raw_rows = []
archive_columns = {}
legacy_tables = []
for folder in ['R2_usable_dynamic_features_328_20261111','R2_dynamic_features_combined328_outcome_blind_20261110',
               'R2_literature_aligned_features_v2_20261112','R2_spatial_propagation_v2_technical_pilot_5_20261111',
               'R2_bilateral_frequency_amplitude_frozen_v1_20261112','R2_bilateral_frequency_amplitude_frozen_v1_1_20261112',
               'R2_bilateral_frequency_amplitude_frozen_v1_2_20261112']:
    for path in sorted((LEGACY/'outputs'/folder).glob('*.csv')):
        table = pd.read_csv(path)
        if 'case_id' not in table or table.case_id.duplicated().any():
            continue
        legacy_tables.append((path,table))
for path,table in legacy_tables:
    tag = path.parent.name + '__' + path.stem
    for col in table.columns:
        if col == 'case_id' or any(s in col.lower() for s in ['path','name','outcome','pregnan','patient_id','exam_id']):
            continue
        values = master.case_id.map(table.set_index('case_id')[col])
        if path.parent.name == 'R2_usable_dynamic_features_328_20261111' and len(table.columns)==13 and col in master:
            master[col] = master[col].fillna(values)
        if pd.api.types.is_numeric_dtype(table[col]):
            column = 'archive__' + tag + '__' + col
            archive_columns[column] = values
            raw_rows.append(dict(feature_name=column,original_field=col,source_file=str(path),
                                 current_status='exploratory',feature_group='QC/Technical' if any(t in col.lower() for t in ['valid','support','duration','count_regions','quality','status']) else
                                 'Frequency' if 'frequency' in col else 'Propagation/Spacetime' if any(t in col for t in ['phase','lag','propagation']) else
                                 'Motion/Displacement/Velocity' if any(t in col for t in ['velocity','displacement','motion']) else
                                 'Amplitude/Strain' if any(t in col for t in ['strain','amplitude']) else 'ER-MSE/Disparity' if any(t in col.lower() for t in ['mse','cc','hausdorff']) else 'QC/Technical',
                                 unit='definition_unclear',physical_meaning='archived scalar; see original field and source',
                                 patient_aggregation='definition_unclear',tier='B',role='explanatory only',
                                 posthoc_feature='unknown',outcome_informed='unknown',level='patient'))
master = pd.concat([master, pd.DataFrame(archive_columns)],axis=1).copy()

candidates = read('MODEL_CANDIDATE_FEATURES.csv')
candidates['recommended_model_role'] = candidates.recommended_model_role.replace({'exploratory predictor':'candidate predictor'})
candidates['formal_or_exploratory'] = np.where(candidates.tier.eq('A'),'formal candidate','exploratory')
activity_rate='significant_propagation_activity_frequency_cpm'
rate_row=candidates.feature_name.eq(activity_rate)
candidates.loc[rate_row,'tier']='B'
candidates.loc[rate_row,'formal_or_exploratory']='exploratory'
candidates.loc[rate_row,'recommended_model_role']='candidate predictor'
candidates.loc[rate_row,'reason_for_inclusion']='2026-11-20用户接受算法检出的显著传播活动频率作为探索指标；不等同真实波频率，不依据AUC分级'
candidates.loc[rate_row,'reason_for_exclusion']='不作已验证primary；误计、相邻重复、弱活动漏检限制仍保留'
candidates.loc[rate_row,'measurement_validity']='partial for algorithm activity index; true wave-count accuracy unvalidated'
qc_fields=['video_duration_s','acquisition_date','source','video_year','source_batch','processing_status','failure_reason',
           'anterior_valid_fraction','posterior_valid_fraction','raw_valid_fraction','native_section_count',
           'anterior_common_region_duration_s','posterior_common_region_duration_s','mode_window_count','paired_window_count',
           'effective_observation_seconds','unresolved_activity_rows','median_support','p10_support','fraction_below_real_Q1',
           'fraction_below_real_median','anterior_median_support','posterior_median_support','support_difference_ant_minus_post',
           'directional_band_fraction_anterior_median','directional_band_fraction_posterior_median','video_month']
for field in qc_fields:
    if field not in inventory.feature_name.values:
        raw_rows.append(dict(feature_name=field,standardized_name=field,source_file=str(OLD/'MODEL_INPUT_MASTER_TABLE.csv'),
                             feature_group='QC/Technical',current_status='exploratory',unit='technical metadata; see field',
                             physical_meaning='existing technical/support field, not biological predictor',
                             patient_aggregation='existing exam summary; no recomputation',tier='C',role='QC only',
                             posthoc_feature='unknown',outcome_informed='unknown',level='patient'))
baseline = ['age','bmi','infertility_years','transfer_embryo_count_multiple','endometrial_thickness_mm','blastocyst_transfer','endometrial_type_C']
amp = ['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude']
dynamic_a = amp + ['record_MSE_time_equal','record_CC_time_equal']
freq = ['longitudinal_anterior_frequency_cpm','longitudinal_posterior_frequency_cpm','longitudinal_anterior_posterior_abs_difference_cpm']
candidates.loc[candidates.feature_name.isin(freq),'recommended_model_role']='exclude'
candidates.loc[candidates.feature_name.isin(freq),'reason_for_exclusion']='2026-11-20用户明确不纳入后续模型；保留历史值，不作测量学失败判定'
candidates.loc[candidates.feature_name.isin(freq),'formal_or_exploratory']='historical only; user excluded'
clinical = candidates.loc[candidates.feature_group.eq('Clinical'),'feature_name'].tolist()
sets = {'Set 0 Clinical only':baseline,'Set 1 Clinical + frequency':baseline+freq,
        'Set 2 Clinical + ER-MSE':baseline+['record_MSE_time_equal'],
        'Set 3 Clinical + ER/MSE family':baseline+['record_MSE_time_equal','record_CC_time_equal','record_MSE_z','record_mean_bias_sq'],
        'Set 4 Clinical + amplitude':baseline+amp,
        'Set 5 Clinical + strain-rate (archived exploratory)':baseline+['posterior_longitudinal_strain_rate_rms_per_s_exploratory'],
        'Set 6 Clinical + velocity (archived exploratory)':baseline+['posterior_motion_velocity_pct_long_axis_per_s_exploratory'],
        'Set 7 Clinical + all defensible dynamic':baseline+dynamic_a,
        'Set 8 All clinical + all candidate dynamic':clinical+candidates.loc[~candidates.feature_group.eq('Clinical'),'feature_name'].tolist(),
        'Dataset A':baseline,'Dataset B':baseline+dynamic_a,
        'Exploratory B + significant activity rate':baseline+dynamic_a+[activity_rate]}
available_rows=[]
for name in list(dict.fromkeys(candidates.feature_name.tolist()+[r['feature_name'] for r in raw_rows])):
    if name not in master:
        master[name] = np.nan
    for cohort,frame in [('desktop_276',master[master.cohort.eq('desktop_276')]),('short_holdout_separate',master[master.cohort.eq('short_holdout_separate')])]:
        ok = frame[name].notna() & ~frame[name].astype(str).isin(['<missing>',''])
        available_rows.append(dict(feature_name=name,cohort=cohort,n_exams=len(frame),
            n_unique_patients_provisional=frame.patient_id.nunique(),repeated_patient_count_provisional=int(frame.patient_id.value_counts().gt(1).sum()),
            n_available=int(ok.sum()),n_available_patients_provisional=frame.loc[ok,'patient_id'].nunique(),
            n_missing=int((~ok).sum()),availability_percent=100*ok.mean(),
            discovery_cohort_n=int((ok & frame.exam_id.isin(previous.loc[previous.cohort.eq('discovery'),'exam_id'])).sum()),
            holdout_cohort_n=int(ok.sum()) if cohort.startswith('short') else 0,
            anterior_available=int(ok.sum()) if 'anterior' in name else np.nan,
            posterior_available=int(ok.sum()) if 'posterior' in name else np.nan,
            bilateral_available=np.nan,missing_due_to_video_duration_n=np.nan,
            missing_due_to_tracking_n=np.nan,missing_due_to_clinical_link_n=np.nan,
            missing_reason_summary='unattributed unless case-level failure is documented; not inferred from NaN'))
availability=pd.DataFrame(available_rows)
save(availability,'FEATURE_AVAILABILITY.csv')
save(read('FEATURE_AVAILABILITY.csv'),'ARCHIVED_SOURCE_AVAILABILITY.csv')
for i,row in candidates.iterrows():
    a=availability[(availability.feature_name==row.feature_name)&(availability.cohort=='desktop_276')].iloc[0]
    candidates.loc[i,'n_available']=a.n_available
    candidates.loc[i,'missing_rate']=1-a.availability_percent/100
save(candidates,'MODEL_CANDIDATE_FEATURES.csv')
setrows=[]
lossrows=[]
for cohort,frame in master.groupby('cohort'):
    for label,columns in sets.items():
        cc=frame[columns].notna().all(axis=1)
        y=frame.loc[cc,'outcome']
        setrows.append(dict(cohort=cohort,feature_set=label,features=';'.join(columns),n_exams=len(frame),
            n_unique_patients_provisional=frame.patient_id.nunique(),repeated_patient_count_provisional=int(frame.patient_id.value_counts().gt(1).sum()),
            pregnancy_outcome_available_count=int(frame.outcome.notna().sum()),complete_case_n=int(cc.sum()),
            complete_case_patients_provisional=frame.loc[cc,'patient_id'].nunique(),complete_case_with_outcome_n=int(y.notna().sum()),
            missing_rate=1-cc.mean(),feature_count=len(columns),events=int(y.eq(1).sum()),non_events=int(y.eq(0).sum()),
            event_non_event_ratio=f'{int(y.eq(1).sum())}:{int(y.eq(0).sum())}'))
    baseok=frame[baseline].notna().all(axis=1)
    for feature in candidates.loc[~candidates.feature_group.eq('Clinical'),'feature_name']:
        ok=baseok&frame[feature].notna()
        lossrows.append(dict(cohort=cohort,feature_name=feature,clinical_complete_n=int(baseok.sum()),
                            after_adding_feature_n=int(ok.sum()),additional_missing_exams=int((baseok&~ok).sum())))
setsummary=pd.DataFrame(setrows)
save(setsummary,'FEATURE_SET_SAMPLE_SIZE.csv')
save(pd.DataFrame(lossrows),'FEATURE_INCREMENTAL_SAMPLE_LOSS.csv')

# Preserve original assertions, but separate them from source-verified evidence.
validation=read('FEATURE_VALIDATION_STATUS.csv')
validation['previous_evidence_source']=validation.evidence_source
validation['evidence_source_status']='prior family-level assignment; not feature-specific proof'
validation['recommended_use']=validation.recommended_use.map({'fixed clinical':'formal','candidate predictor':'formal','exploratory predictor':'exploratory','explanatory only':'exploratory','QC only':'exclude','exclude':'exclude'})
validation['limitation']=validation.limitation.fillna('')+'; 旧表按family继承的yes不可视为每个版本已验证；见VALIDATION_EVIDENCE.csv。'
for col in ['tracking_validated','synthetic_truth_test','rigid_control','brightness_control','repeatability_test','split_half','duration_stability','support_audit','external_reference']:
    validation['previous_'+col]=validation[col]
    validation[col]='unclear'
evidence=[]
evidence_specs=[('legacy amplitude split-half',LEGACY/'outputs/R2_bilateral_amplitude_splithalf_reliability_20261112_v1/summary.json'),
 ('legacy corrected frequency cache',LEGACY/'outputs/R2_longitudinal_frequency_corrected_328_v3_cache_reverify_20261112/correction_summary.json'),
 ('legacy CUDA known-strain',LEGACY/'outputs/R2_bilateral_amplitude_cuda_controls_20261112_v1/summary.json'),
 ('legacy frozen bilateral v1.2 amplitude',LEGACY/'outputs/R2_bilateral_frequency_amplitude_frozen_v1_2_20261112/summary.json'),
 ('legacy frequency reproducibility',LEGACY/'outputs/R2_longitudinal_reproducibility_v1_1_20261106/重复性验证报告.md'),
 ('legacy motion strain technical audit',LEGACY/'outputs/R2_feature_definition_audit_20261112_v5/技术核查结论_中文.md'),
 ('current A0 amplitude interface controls',PROJECT/'outputs/feature_definition_validation_20261117_v1/REPORT.md'),
 ('current Huang bridge',PROJECT/'outputs/huang_bridge_20261120/REPORT.md'),
 ('current rigid/support',PROJECT/'outputs/er_directional_support_audit_20261120/REPORT.md'),
 ('current MSE duration',PROJECT/'outputs/short_duration_bridge_20261120/REPORT.md'),
 ('current MSE holdout measurement',PROJECT/'outputs/short_duration_holdout_validation_20261120/PHASE_B_IMAGING_QC.md')]
evidence_specs += [('archived frequency support original-vs-consensus protocol',ROOT/'archive_20261116/R2_frequency_support_v2a/PROTOCOL.md'),
                   ('archived frequency optimization protocol',ROOT/'archive_20261116/R2_frequency_optimization_v1/PROTOCOL.md')]
for scope,path in evidence_specs:
    if path.exists():
        content=path.read_text(encoding='utf-8-sig')
        evidence.append(dict(scope=scope,source_file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             evidence_excerpt=content[:18000],applicability='only named algorithm/version; no automatic transfer to other family members'))
save(pd.DataFrame(evidence),'VALIDATION_EVIDENCE.csv')
current=validation.feature_name.eq('record_MSE_time_equal')
validation.loc[current,'duration_stability']='yes'
validation.loc[current,'support_audit']='yes'
validation.loc[current,'evidence_source']='huang_bridge_20261120/REPORT.md; short_duration_bridge_20261120/REPORT.md; er_directional_support_audit_20261120/REPORT.md'
validation.loc[current,'evidence_source_status']='direct version-specific reports inspected'
rate_mask=validation.feature_name.eq(activity_rate)
validation.loc[rate_mask,'tier']='B'
validation.loc[rate_mask,'recommended_use']='exploratory'
validation.loc[rate_mask,'validation_result']='partial'
validation.loc[rate_mask,'limitation']='用户接受算法活动频率定义作探索；真实波计数准确性仍未验证，误计/重复/漏检局限保留；非AUC驱动分级'
inventory.loc[inventory.feature_name.eq(activity_rate),'tier']='B'
inventory.loc[inventory.feature_name.eq(activity_rate),'role']='candidate predictor'
for feature in amp:
    mask=validation.feature_name.eq(feature)
    validation.loc[mask,'synthetic_truth_test']='partial'
    validation.loc[mask,'rigid_control']='partial'
    validation.loc[mask,'support_audit']='partial'
    validation.loc[mask,'evidence_source']='feature_definition_validation_20261117_v1/REPORT.md (12 texture caches, summary agreement; not human repeat acquisition)'
    validation.loc[mask,'evidence_source_status']='direct A0 amplitude report; no legacy split-half transfer'
for feature in freq:
    mask=validation.feature_name.eq(feature)
    validation.loc[mask,'repeatability_test']='partial'
    validation.loc[mask,'duration_stability']='partial'
    validation.loc[mask,'evidence_source']='R2_longitudinal_reproducibility_v1_1_20261106/重复性验证报告.md; R2_longitudinal_frequency_corrected_328_v3_cache_reverify_20261112/correction_summary.json'
    validation.loc[mask,'evidence_source_status']='legacy lineage evidence; exact three60s version unconfirmed'
for feature in ['posterior_motion_amplitude_pct_local_width_exploratory','posterior_motion_velocity_pct_long_axis_per_s_exploratory','posterior_longitudinal_strain_rate_rms_per_s_exploratory']:
    mask=validation.feature_name.eq(feature)
    validation.loc[mask,'duration_stability']='partial'
    validation.loc[mask,'evidence_source']='R2_feature_definition_audit_20261112_v5/技术核查结论_中文.md'
    validation.loc[mask,'evidence_source_status']='direct archived processing-chain audit; overlapping trims not repeat acquisition'
save(validation,'FEATURE_VALIDATION_STATUS.csv')
inventory['notes']=inventory.notes.fillna('')+'; 原版本登记保留；可用性以desktop_276重算，未知定义不修复。'
inventory=pd.concat([inventory,pd.DataFrame(raw_rows)],ignore_index=True)
inventory['feature_key']=inventory.source_file.astype(str)+'::'+inventory.feature_name
inventory['definition_review_status']=np.where(inventory.current_status.eq('unclear')|inventory.unit.eq('definition_unclear'),'definition_unclear','prior_definition_retained')
inventory['value_table_current_audit']=str(OUT/'MODEL_INPUT_MASTER_TABLE.csv')
inventory['measurement_evidence_status']='see FEATURE_VALIDATION_STATUS and VALIDATION_EVIDENCE; no automatic inheritance'
for field in ['source_code','code_location','directionality','notes','standardized_name']:
    inventory[field]=inventory[field].fillna('definition_unclear' if field!='notes' else 'archived exact source; definition not repaired')
save(inventory,'FEATURE_INVENTORY.csv')
extra_validation=[]
for row in raw_rows:
    extra_validation.append(dict(feature_name=row['feature_name'],feature_group=row['feature_group'],tier=row['tier'],
        **{c:'unclear' for c in ['tracking_validated','synthetic_truth_test','rigid_control','brightness_control','repeatability_test','split_half','duration_stability','support_audit','external_reference']},
        validation_result='unclear',limitation='version-specific evidence not established; not inherited from family',
        recommended_use='exclude' if row['role']=='QC only' else 'exploratory',evidence_source=row['source_file']))
validation=pd.concat([validation,pd.DataFrame(extra_validation)],ignore_index=True)
save(validation,'FEATURE_VALIDATION_STATUS.csv')

ci=read('CLINICAL_VARIABLE_INVENTORY.csv')
for i,row in ci.iterrows():
    if row['name'] not in main: continue
    v=master.loc[master.cohort.eq('desktop_276'),row['name']]
    ci.loc[i,'missing']=int(v.isna().sum())
    ci.loc[i,'n_exams']=276
    ci.loc[i,'n_unique_patients_provisional']=main.patient_id.nunique()
    ci.loc[i,'categorical_levels']=json.dumps(v.dropna().astype(str).unique().tolist(),ensure_ascii=False) if row['name'] in ['uterine_position','endometrial_type','transferred_embryo_stage'] else ''
    grouped=main.groupby('patient_id')[row['name']].agg(['count','nunique'])
    ci.loc[i,'repeated_patient_consistency']=f'repeated groups={int(grouped["count"].gt(1).sum())}; groups with differing values={int(((grouped["count"]>1)&(grouped["nunique"]>1)).sum())}; cycle-varying fields may differ'
save(ci,'CLINICAL_VARIABLE_INVENTORY.csv')
for family,cols in [('clinical',baseline),('amplitude',amp),('frequency',freq),('ER_MSE',['record_MSE_time_equal']),('TierA',baseline+dynamic_a)]:
    master[family+'_available']=master[cols].notna().all(axis=1)
save(master,'MODEL_INPUT_MASTER_TABLE.csv')
families=candidates[['feature_name','feature_group']].merge(availability,on='feature_name')
family_summary=families.groupby(['cohort','feature_group']).agg(feature_count=('feature_name','size'),min_available=('n_available','min'),max_available=('n_available','max')).reset_index()
save(family_summary,'FEATURE_FAMILY_AVAILABILITY.csv')

# File catalogue: inspect schemas, JSON/report text, source definitions and NPZ headers only.
catalog=[]
npz_dirs=set()
def source_paths(base):
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'measurement','inputs','frozen_measurement','code_snapshot','reproducibility','__pycache__','data','patients','cases','node_modules','.git'}
                   and not ((Path(folder)/d).lstat().st_file_attributes & 1024)]
        depth=len(Path(folder).relative_to(base).parts)
        if depth>=2: dirs[:]=[]
        for name in files: yield Path(folder)/name
for base in [PROJECT/'uterine_wave',PROJECT/'experiments',PROJECT/'outputs',LEGACY/'outputs',LEGACY/'frozen_longitudinal_posterior_v1',ROOT/'R2_long_video_preprocessing',ROOT/'archive_20261116/R2_frequency_support_v2a',ROOT/'archive_20261116/R2_frequency_optimization_v1']:
    for path in source_paths(base):
        if not path.is_file() or OUT in path.parents or path.suffix not in ['.py','.csv','.json','.md','.npz']: continue
        if any(t in path.name.lower() for t in ['auc','prediction','association','coefficient','odds','roc','shap']): continue
        if any(p in {'measurement','inputs','frozen_measurement','code_snapshot','reproducibility','__pycache__'} for p in path.relative_to(base).parts[:-1]): continue
        if path.suffix=='.npz' and path.parent in npz_dirs: continue
        if path.suffix=='.npz': npz_dirs.add(path.parent)
        entry=dict(path=str(path),kind=path.suffix,bytes=path.stat().st_size,read_status='')
        if path.suffix=='.csv':
            try: entry['schema']=json.dumps(pd.read_csv(path,nrows=0).columns.tolist(),ensure_ascii=False); entry['read_status']='header_read'
            except (pd.errors.EmptyDataError,UnicodeDecodeError,pd.errors.ParserError) as e: entry['read_status']=type(e).__name__
        elif path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:
                entry['schema']=json.dumps(z.namelist());entry['read_status']='zip member metadata only; no arrays loaded'
        else:
            with path.open(encoding='utf-8-sig',errors='replace') as handle: content=handle.read(131072)
            keys=['60','ICC','Spearman','Brown','gain','longest','median','Hausdorff','split','known','C2F','F2C','brightness','25%']
            excerpts=[f'{i+1}: {line[:400]}' for i,line in enumerate(content.splitlines()) if any(k.lower() in line.lower() for k in keys)]
            entry['schema']='\n'.join(excerpts[:24]);entry['read_status']='text_read_first_128k_chars'
        catalog.append(entry)
save(pd.DataFrame(catalog),'SOURCE_CATALOG.csv')
summary={'desktop_exams':len(main),'linked_existing_discovery':int(main.existing_result_link.sum()),
         'no_existing_current_results':int((~main.existing_result_link).sum()),
         'desktop_provisional_name_groups':int(main.patient_id.nunique()),
         'desktop_repeated_name_groups':int(main.patient_id.value_counts().gt(1).sum()),
         'holdout_exams_separate':len(hold),'holdout_identity_verified':False,
         'formal_code_changed':False,'new_auc_or_models':False,'feature_recomputed':False,
         'source_catalog_files':len(catalog),'desktop_sha256':hashlib.sha256(WB.read_bytes()).hexdigest()}
summary['missing_current_input_reasons']=main.loc[~main.existing_result_link,'input_audit_failure_reason'].value_counts().to_dict()
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
mainsets=setsummary[setsummary.cohort.eq('desktop_276')]
loss=pd.DataFrame(lossrows);loss=loss[loss.cohort.eq('desktop_276')].sort_values('additional_missing_exams',ascending=False)
tiertext='\n\n'.join('### Tier '+tier+'\n\n'+'\n'.join('- `'+r.feature_name+'`：'+r.recommended_model_role for r in candidates[candidates.tier.eq(tier)].itertuples()) for tier in ['A','B','C'])
report=f'''# 全项目特征盘点与桌面276例可用性复核

主队列由桌面《一分钟以上患者视频.xlsx》固定，共276次检查；保留尚无当前特征的53次检查。短视频hold-out单列，不并入主队列分母。全程只读取已有结果，没有模型训练、AUC计算、按结局比较或重算视频。

## 队列和身份

{json.dumps(summary,ensure_ascii=False,indent=2)}

patient_id优先复用358例原表的患者聚类标识，无匹配时保留姓名暂定键，逐例状态在master中；原聚类标识的身份基础仍需结合原始记录核实，不能排除同名异人。hold-out仍只有检查编号替代键，跨队列患者独立性未证明。不得据此宣称独立患者验证。后续分组CV需先完成稳定身份核查。

## 原始字段和版本

保留原261条字段来源登记，并补充归档表的逐列来源命名空间；总登记{len(inventory)}行。原57个候选字段分9个family，候选不等于57个均可入模。SOURCE_CATALOG.csv列出实际检查的源码、CSV表头、JSON、报告和NPZ成员metadata。归档版本原始列通过archive__前缀保留，不能与当前A0互换。原登记的泛化source_file和关键行号尚非逐项完整证明，definition_unclear保持不修复。

## 逐类可用性

{md(family_summary[family_summary.cohort.eq('desktop_276')])}

各family数值为单字段最小/最大可用检查数；联合支持按FEATURE_SET_SAMPLE_SIZE.csv计算，不能把最大值称为全family共同人数。完整逐特征人数和暂定患者数见FEATURE_AVAILABILITY.csv。缺失原因不能仅由NaN推断；未知原因保留空值，不伪报tracking或时长导致的精确例数。

## 组合样本量

{md(mainsets[['feature_set','complete_case_n','complete_case_patients_provisional','feature_count','missing_rate','events','non_events']])}

Set 5/6仅列归档探索特征的可用性，不代表批准正式使用。Set 8包含不建议入模字段，仅用于展示完整交集；临床全部变量含重复编码，不是建议的最终设计矩阵。events/non-events只描述可用性，未做组间检验。

## 单项增加造成的损失

{md(loss[['feature_name','clinical_complete_n','after_adding_feature_n','additional_missing_exams']].head(15))}

这是同一276分母中，临床完整者加一个字段后的精确损失；不能把旧328来源可用数直接用于当前队列。53例未完成当前链也计入缺失，不因输入问题删行。

## 验证证据

VALIDATION_EVIDENCE.csv提供实际来源、哈希和原始测量学摘录。旧盘点按family统一填写yes，不能证明每个版本做过对应验证；本次保留previous_*列，将未逐字段核实项标为unclear，避免继承错误验证。

- 旧R2幅度split-half为739个区域：Pearson 0.4679、Spearman 0.4213、Spearman–Brown约0.6375/0.5929，半段相对差中位0.4842；这些不是当前2秒A0幅度的验证数值。
- 旧修正频率检查3891个缓存窗口，758→739个eligible区域，后壁211/328检查可用；0.25 cpm为原算法consensus容差，不是本轮新增判据。三60秒窗与longest-valid版本必须分别引用原文件，不能以同名字段自动替换；具体ICC和量化分辨率见SOURCE_CATALOG原报告/源码定位，未逐版本确认的值保持unclear。
- 当前Huang桥接157个共同可计算记录，MSE Pearson 0.9846、Spearman 0.9668、中位绝对差0.00918；是两测量定义一致性，不是人体真值。
- 当前MSE 45秒first/middle/last与全程ICC 0.792/0.834/0.744，三位置ICC 0.644；时长选择是项目冻结设计，不是普遍有效性门槛，也不能移植到CC或幅度。
- 方向控制和rigid/support审计支持连续误差解释；ER为比例，低绝对运动仍可产生偏置，不新增一票否决阈值。同步同向不保证CC为正，须区分方向与时间波形相关。
- CPU/CUDA、known-strain和材料点稳定性证据保持原版本范围；不以25%自定误差边界直接判定全部strain无效。
- 旧CUDA known-strain共12个点对控制，gain范围0.8007–1.3514、相对误差约1.99%–35.14%；CPU旧v1.2两壁图像控制误差约27.85%/26.32%。这是控制条件下的响应，不等于患者误差；不能用追踪全有效替代精度证据。
- 旧频率PCA轴扰动ICC 0.966（416个高可信共同区域）、裁窗ICC 0.771（401个）；是处理选择敏感性，不是独立重复采集。当前方向控制C2F/F2C各support四分位正确率1.0，仅限该夹具，不声称真实患者准确率100%。
- hold-out影像35输入、34完成、33有MSE；结果已被查看，不能重新作为未触碰数据调试。身份独立性仍待核查。

## 分层与事后解释变量

以下保留原盘点的候选层级，依据定义冻结和已有测量证据，不依据AUC；完整有效性仍为partial。formal表示可列正式候选，并非临床已验证。未逐字段复核的测量项不可据旧yes升级。

{tiertext}

mean_bias_sq、双壁平均ER、SD尺度、时间形状、MSE_z和ER差等保留posthoc标记，仅作解释。旧记录未能确定提出时间的字段保持unknown，不编造yes/no。用户2026-11-20接受significant_propagation_activity_frequency_cpm作为算法活动频率探索候选，改列Tier B；不称真实生理波频率，不修改计数算法或原始值。其余event计数不自动升级。独立探索建模结果见outputs/exploratory_activity_rate_increment_20261120，本盘点不依据AUC分级。

## 十六项结论

1. 原标准候选57项；总来源登记{len(inventory)}行包含版本别名，不能当独立预测变量数。
2. 9个family，技术QC单列。
3. 各类检查数见逐类表，暂定患者数见逐特征表；身份未验证人数不称真实患者数。
4. 缺失最严重项见增量损失表和availability，不能混用328、258与276分母。
5. 无特征具备完整人体gold-standard验证；已有基本证据支持若干正式候选，不要求绝对材料身份证明。
6. 旧频率、motion/strain/velocity和小样本传播属于版本限定探索证据。
7. event真实波身份仍有限制；算法显著传播活动频率获用户授权进入探索组合，不能升级为已验证主要生理指标。定义不明、历史版本和QC不混入当前主要生物特征模型。
8. A/B/C逐项列于上节与MODEL_CANDIDATE_FEATURES.csv；未新增性能驱动筛选。
9. Tier A单字段最大可用检查数为{int(candidates.loc[candidates.tier.eq('A'),'n_available'].max())}，对应暂定患者数须按具体字段查看，不等于全部Tier A交集。
10. 全Tier A complete-case即Dataset B，检查和暂定患者数见下节。
11. Clinical+frequency见Set 1，不与活动计数率混淆。
12. Clinical+ER-MSE见Set 2。
13. Clinical+全部当前正式候选动态特征见Set 7。
14. 单字段损失见FEATURE_INCREMENTAL_SAMPLE_LOSS.csv，直接显示哪项跨过200。
15. 少量临床缺失可考虑后续训练折内处理；影像缺失同时报告支持和缺失指示，不填0；不因event缺失少而推荐入模。临床月份/年限、胚胎派生编码保留重复性标记。
16. 后续仅保留Dataset A/B/C方案，先核实身份和版本，不在本轮选择模型。

## Dataset A B C

{md(mainsets[mainsets.feature_set.str.startswith('Dataset')][['feature_set','n_exams','complete_case_n','complete_case_patients_provisional','feature_count','missing_rate']])}

Dataset B称“临床+已有基本测量证据的影像候选”，不称完全验证。2026-11-20用户明确移除旧R2纵向主频及双壁频率差，原Dataset C取消，尚未指定替代组合。Set 1仅保留历史可用性交集，不代表后续纳入；原始值不删除。表中n_exams为完整主队列，complete_case_n为可联合计算交集；所有276行仍保留在master。hold-out另列于组合表，未合并估算训练样本。

## 尚待核实

稳定患者身份、三60秒窗与最长连续段的逐版本ICC/量化分辨率、CPU/CUDA逐版本增益，以及原盘点全部关键行号仍需继续逐项证据核对。旧输入清单的逐例失败原因已保留在COHORT_INPUT_STATUS.csv；它是历史输入检查状态，不覆盖后续实际运行结果。归档V2a协议明确原739个正式区域由576个完整窗值和163个多窗共识值组成；这不能冒称三个60秒窗版本。SOURCE_CATALOG提供定位而非完成这些验证的声明。正式代码、原特征和旧结果均未改写。

SOURCE_CATALOG的目录检查深度为每个列出根下两层；文本读取前128k字符，NPZ按目录取首份成员metadata样本，未加载图像数组。它不是每份患者缓存的完整内容审查。临床结局只用于所请求的可用性及类别人数统计。直接读取的原报告和关键CSV已单独核对，不以catalog关键词摘录作为全部验证结论。
'''
(OUT/'FEATURE_INVENTORY_REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True))
print(mainsets[mainsets.feature_set.str.startswith('Dataset')][['feature_set','complete_case_n','complete_case_patients_provisional','feature_count']].to_string(index=False))
