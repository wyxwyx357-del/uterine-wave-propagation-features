"""Register desktop276 plus short35, no extraction or model fitting."""
import os
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('DATA_ROOT', 'data')) / 'outputs'
OUT=ROOT/'expanded_sample_registry_311_20261120'
OUT.mkdir(exist_ok=True)
sources=[ROOT/'feature_inventory_desktop276_20261120/MODEL_INPUT_MASTER_TABLE.csv',
         ROOT/'short_activity_auc_20261120/SHORT_MASTER_ALL35.csv']
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
old=pd.read_csv(sources[0],dtype={'patient_id':str,'exam_id':str})
long=old[old.cohort.eq('desktop_276')].copy()
short=pd.read_csv(sources[1],dtype={'patient_id':str,'exam_id':str})
assert len(long)==276 and len(short)==35
assert not set(long.case_id)&set(short.case_id)
assert not set(long.exam_id)&set(short.exam_id)
long['model_identity_eligible']=long.patient_id_status.eq('existing_clinical_cluster_key')
short['model_identity_eligible']=short.patient_identity_status.eq('existing_clinical_cluster_exact_exam_join')
long['cohort']='desktop_276'
short['cohort']='short_35'
long['outcome_source']='existing_final_clinical_workbook'
short['outcome_source']=np.where(short.outcome_clinical.notna(),'final_clinical_workbook_exact_join','unavailable_clinical_label')
# Keep alternative labels separate; never silently mix directory and clinical labels.
short['outcome']=short.outcome_clinical
short['patient_id_status']=short.patient_identity_status
long['prior_test_exposure']='development_and_internal_exploratory'
short['prior_test_exposure']='MSE_and_activity_AUC_already_inspected'
master=pd.concat([long,short],ignore_index=True).copy()
master['research_registry_included']=True
master['outcome_eligible']=master.outcome.isin([0,1])
clinical=['age','bmi','infertility_years','transfer_embryo_count_multiple','endometrial_thickness_mm','blastocyst_transfer','endometrial_type_C']
B=clinical+['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude','record_MSE_time_equal','record_CC_time_equal']
rate='significant_propagation_activity_frequency_cpm'
sets={'A_clinical7':clinical,'B_clinical_and_four_imaging':B,
      'B_plus_algorithm_activity_rate_exploratory':B+[rate],
      'clinical_plus_algorithm_activity_rate_exploratory':clinical+[rate],
      'algorithm_activity_rate_only_exploratory':[rate]}
rows=[]
for set_name,columns in sets.items():
    master[set_name+'_measurement_complete']=master[columns].notna().all(axis=1)
    master[set_name+'_analysis_eligible']=master[set_name+'_measurement_complete']&master.model_identity_eligible&master.outcome_eligible
    for cohort,frame in [('all_311',master),('desktop_276',master[master.cohort=='desktop_276']),('short_35',master[master.cohort=='short_35'])]:
        raw=frame[set_name+'_measurement_complete']
        ok=frame[set_name+'_analysis_eligible']
        rows.append(dict(cohort=cohort,feature_set=set_name,features=';'.join(columns),feature_count=len(columns),registry_exams=len(frame),
                         measurement_complete_exams=int(raw.sum()),identity_and_clinical_outcome_eligible_exams=int(ok.sum()),
                         eligible_patient_groups=int(frame.loc[ok,'patient_id'].nunique()),
                         additional_short_exams=int((ok&frame.cohort.eq('short_35')).sum()),
                         missing_measurement_rate=float(1-raw.mean())))
summary=pd.DataFrame(rows)
master.to_csv(OUT/'MODEL_INPUT_MASTER_TABLE_311.csv',index=False,encoding='utf-8-sig')
summary.to_csv(OUT/'FEATURE_SET_SAMPLE_SIZE.csv',index=False,encoding='utf-8-sig')
known=master[master.model_identity_eligible]
checks=dict(total_registered_exams=len(master),desktop_exams=len(long),short_exams=len(short),
            exam_ids_unique=bool(master.exam_id.is_unique),case_ids_unique=bool(master.case_id.is_unique),
            identity_linked_exams=len(known),identity_linked_patient_groups=int(known.patient_id.nunique()),
            identity_unverified_exams=int((~master.model_identity_eligible).sum()),
            repeated_linked_patient_groups=int(known.patient_id.value_counts().gt(1).sum()),
            known_cross_cohort_patient_overlap=len(set(long.loc[long.model_identity_eligible,'patient_id'])&set(short.loc[short.model_identity_eligible,'patient_id'])),
            no_new_model=True,no_new_feature_extraction=True,original_files_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()),
            source_sha256=hashes)
assert len(master)==311 and checks['exam_ids_unique'] and checks['case_ids_unique'] and checks['original_files_unchanged']
(OUT/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
select=summary[summary.cohort.eq('all_311')]
lines=['# 扩展研究样本登记','',
       '按用户2026-11-20要求，将35次短视频纳入扩展研究样本库，与桌面276次检查共同登记，共311次检查。登记不等于每个特征或模型均有311例可分析；原始冻结文件保持不变。','',
       '## 身份与结局','',
       f'现有临床表连接身份的检查{len(known)}次，对应{known.patient_id.nunique()}个既有患者聚类键；另5次短视频身份尚未核实，不将其检查编号当真实独立患者。已连接的30次短视频与276队列未发现患者键重叠。现有键仍依赖原表身份准确性，不能作为绝对同名身份保证。','',
       '主结局统一采用最终临床复核表；目录标签和旧MSE分析标签分列保留。1例标签冲突的两份原值均保留，未连接临床者不混入主要联合模型分析；全部35行仍保留在登记库。','',
       '## 特征组合可用性','',
       '|组合|特征数|测量完整检查|身份与临床结局均支持的检查|患者聚类数|新增短视频检查|',
       '|---|---:|---:|---:|---:|---:|']
for r in select.itertuples():lines.append(f'|{r.feature_set}|{r.feature_count}|{r.measurement_complete_exams}|{r.identity_and_clinical_outcome_eligible_exams}|{r.eligible_patient_groups}|{r.additional_short_exams}|')
lines+=['','## 后续使用约定','',
        '合理用途是扩展开发/探索队列。后续任何交叉验证按patient_id分组，保留cohort和video_duration_s供分层描述与敏感性检查；不把35次短视频视为35份独立、全特征完整样本，不补0，不以插补掩盖缺乏临床连接。','',
        '短视频已有MSE及活动频率AUC结果被查看，合入开发样本后不再作为未触碰的独立验证集。未来独立验证另留新数据。不同采集时长可能改变活动率误差及ER-MSE重复性，需要按特征评价，不以45秒协议一票否决其他特征。','',
        '旧R2纵向主频仍不进入候选组合。算法显著活动频率维持探索角色，其测量局限不因本次入库而消失。本轮仅合并登记和可用性统计，未新增模型或重提取。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k!='source_sha256'},ensure_ascii=True))
print(select[['feature_set','identity_and_clinical_outcome_eligible_exams','eligible_patient_groups','additional_short_exams']].to_string(index=False))
