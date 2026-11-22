"""Source-align clinical values and freeze the reviewed incremental-analysis plan.

Writes a new version only. No extraction, model fitting, AUC or prior-result edits.
"""
import os
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(os.environ.get('DATA_ROOT', 'data')) / 'outputs'
OUT=ROOT/'extended6_increment_protocol_20261120'
OUT.mkdir(exist_ok=True)
OLD=ROOT/'expanded_sample_registry_311_20261120/MODEL_INPUT_MASTER_TABLE_311.csv'
CLINICAL=ROOT/'clinical_baseline_design_20261120/CANDIDATE_VALUES_SOURCE_AUDIT_ONLY.csv'
BOOK = Path(os.environ.get('CLINICAL_SOURCE_PATH', 'data/clinical_source.xlsx'))
REQUEST = Path(os.environ.get('REQUEST_TEXT_PATH', 'data/request.txt'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def table(d):
    return '\n'.join(['|'+'|'.join(map(str,d.columns))+'|','|'+'|'.join(['---']*len(d.columns))+'|']+
        ['|'+'|'.join(str(v).replace('|','/').replace('\n',' ') for v in row)+'|' for row in d.itertuples(index=False,name=None)])
before={str(p):digest(p) for p in [OLD,CLINICAL,BOOK,REQUEST]}
master=pd.read_csv(OLD,dtype={'patient_id':str,'exam_id':str})
old_master_columns=master.columns.tolist()
source=pd.read_csv(CLINICAL,dtype={'patient_id':str,'exam_id':str})
assert len(master)==311 and master.case_id.tolist()==source.case_id.tolist()
extended=['age','transfer_embryo_count_component','blastocyst_transfer','bmi','infertility_years','endometrial_thickness_mm']
core=extended[:3]
historical=['age','transfer_embryo_count_multiple','blastocyst_transfer','bmi','infertility_years','endometrial_thickness_mm','endometrial_type_C']
clinical_fields=list(dict.fromkeys(extended+historical))
changes=[]
for col in clinical_fields:
    old=pd.to_numeric(master[col],errors='coerce')
    new=pd.to_numeric(source[col],errors='coerce')
    diff=old.notna().ne(new.notna()) | (old.notna()&new.notna()&~np.isclose(old,new))
    for i in np.flatnonzero(diff):
        changes.append(dict(case_id=master.loc[i,'case_id'],exam_id=master.loc[i,'exam_id'],field=col,
            previous_master_value=old.iloc[i],source_aligned_value=new.iloc[i],
            reason='read existing358 normalized years including month fraction' if col=='infertility_years' else 'respect existing358 BMI missing policy; do not backfill raw101.9',
            original_source_changed=False))
    master[col]=new
assert sum(c['field']=='infertility_years' for c in changes)==44
assert sum(c['field']=='bmi' for c in changes)==1
csv('CLINICAL_VALUE_ALIGNMENT_AUDIT.csv',changes)
master['clinical_source_policy']='existing358 modeling input; no raw backfill over explicit modeling missing'
master['clinical_baseline_version']='Extended6_reviewed_v1_20261120'
master['primary6_clinical_complete']=master[extended].notna().all(axis=1)
master['core3_clinical_complete']=master[core].notna().all(axis=1)
master['historical7_clinical_complete']=master[historical].notna().all(axis=1)
master['model_identity_outcome_eligible']=master.model_identity_eligible.eq(True)&master.outcome.isin([0,1])
eligible=master.model_identity_outcome_eligible
# Historical flags remain preserved but clearly retired: they were based on seven fields.
old_flags=[c for c in old_master_columns if c.endswith(('_analysis_eligible','_measurement_complete','_available','_test_eligible')) or c in ['clinical7_complete','B_rate_complete']]
master=master.rename(columns={c:'previous_registry__'+c for c in old_flags})
blocks=[
 ('MSE','primary_imaging',['record_MSE_time_equal'],1,'single predesignated primary increment'),
 ('ActivityRate','key_secondary_exploratory',['significant_propagation_activity_frequency_cpm'],1,'algorithm significant activity rate, not true physiological count'),
 ('Amplitude','secondary',['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude'],2,'one bilateral block, no wall selection'),
 ('CC','secondary_explanatory',['record_CC_time_equal'],1,'separate from MSE; incomplete duration bridge'),
 ('Multimodal','exploratory',['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude','record_MSE_time_equal','record_CC_time_equal'],4,'four-imaging joint model, not primary'),
 ('FullImaging','exploratory',['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude','record_MSE_time_equal','record_CC_time_equal','significant_propagation_activity_frequency_cpm'],5,'five-imaging joint model, not primary')]
rows=[];members=[]
for name,tier,columns,df,meaning in blocks:
    observed=master[columns].notna().all(axis=1)
    main=eligible&observed
    cc=main&master.primary6_clinical_complete
    master['plan_'+name+'_eligible']=main
    master['plan_'+name+'_clinical6_complete_sensitivity']=cc
    y=master.loc[main,'outcome']
    identity=master.loc[main,['exam_id','patient_id','case_id']].to_csv(index=False).encode('utf-8')
    rows.append(dict(analysis=name,tier=tier,baseline_fields=';'.join(extended),imaging_fields=';'.join(columns),
        baseline_df=6,added_imaging_df=df,total_df=6+df,n_main_exams=int(main.sum()),n_main_patients=master.loc[main,'patient_id'].nunique(),
        n_clinical6_complete_exams=int(cc.sum()),n_clinical6_complete_patients=master.loc[cc,'patient_id'].nunique(),
        n_with_clinical_missing=int((main&~master.primary6_clinical_complete).sum()),
        events=int(y.sum()),non_events=int(y.eq(0).sum()),
        short_main_exams=int((main&master.cohort.eq('short_35')).sum()),interpretation=meaning,
        cohort_membership_sha256=hashlib.sha256(identity).hexdigest()))
    for idx in master.index:
        members.append(dict(analysis=name,case_id=master.loc[idx,'case_id'],exam_id=master.loc[idx,'exam_id'],
            main_eligible=bool(main.loc[idx]),clinical_complete_sensitivity=bool(cc.loc[idx]),
            missing_identity_or_outcome=not bool(eligible.loc[idx]),missing_imaging_fields=';'.join(c for c in columns if pd.isna(master.loc[idx,c])),
            missing_clinical_fields=';'.join(c for c in extended if pd.isna(master.loc[idx,c]))))
summary=pd.DataFrame(rows)
csv('ANALYSIS_MODEL_REGISTRY.csv',summary);csv('ANALYSIS_COHORT_MEMBERSHIP_LOCAL.csv',members)
csv('MASTER_CLINICAL_SOURCE_ALIGNED_311.csv',master)

# Split-only rehearsal: stratification uses labels, no prediction or selection performed.
outer_rows=[];inner_rows=[];split_audit=[]
seeds=list(range(20260531,20260541))
for row in rows:
    name=row['analysis']
    for scope,col in [('main_imaging_supported','plan_'+name+'_eligible'),
                      ('clinical6_complete_sensitivity','plan_'+name+'_clinical6_complete_sensitivity')]:
        d=master[master[col]].reset_index(drop=True)
        y=d.outcome.to_numpy(int);groups=d.patient_id.to_numpy(str)
        for repeat,seed in enumerate(seeds):
            assignment=np.full(len(d),-1,int)
            outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
            for fold,(tr,te) in enumerate(outer.split(np.zeros(len(d)),y,groups),1):
                assert not set(groups[tr])&set(groups[te])
                assert len(np.unique(y[tr]))==2 and len(np.unique(y[te]))==2
                assignment[te]=fold
                seed_inner=20260531+1000+100*repeat+fold
                inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed_inner)
                for inner_fold,(itr,iva) in enumerate(inner.split(np.zeros(len(tr)),y[tr],groups[tr]),1):
                    assert not set(groups[tr][itr])&set(groups[tr][iva])
                    assert len(np.unique(y[tr][itr]))==2 and len(np.unique(y[tr][iva]))==2
                    for j in iva:
                        inner_rows.append(dict(analysis=name,scope=scope,repeat=repeat,outer_fold=fold,inner_fold=inner_fold,
                                               exam_id=d.iloc[tr[j]].exam_id,patient_id=d.iloc[tr[j]].patient_id))
                split_audit.append(dict(analysis=name,scope=scope,repeat=repeat,fold=fold,train_n=len(tr),test_n=len(te),patient_overlap=0,inner_patient_overlap=0))
            assert (assignment>0).all()
            for i,r in d.iterrows():
                outer_rows.append(dict(analysis=name,scope=scope,repeat=repeat,seed=seed,fold=int(assignment[i]),exam_id=r.exam_id,patient_id=r.patient_id))
csv('OUTER_FOLD_ASSIGNMENTS_LOCAL.csv',outer_rows)
csv('INNER_VALIDATION_ASSIGNMENTS_LOCAL.csv',inner_rows)
csv('SPLIT_ONLY_AUDIT.csv',split_audit)
protocol=dict(version='Extended6_increment_v1_20261120',freeze_date='2026-11-20',status='specification_frozen_no_new_performance_computed',
    clinical_baseline_primary=extended,clinical_baseline_df=6,
    clinical_sensitivity={'parsimonious_core3':core,'historical7':historical},
    prediction_time='ultrasound completed, actual embryo number/stage known per user, current pregnancy outcome unknown',
    variables_fixed=True,feature_selection=False,data_driven_cutoffs=False,nonlinearity='linear continuous predictors; no new splines or interactions',
    primary_imaging='MSE',activity_role='key secondary exploratory, not co-primary physiological wave test',
    blocks=rows,cohort_policy='observed required imaging + existing clinical identity + clinical outcome; clinical missing imputed only within training',
    primary_pairing='within each cohort all clinical/augmented models and all baseline sensitivities use the same outer splits',
    missing={'clinical':'training-only median; no missing flags in this6df plan',
             'inner':'fit median and scale in each inner-training split, never all outer-training before inner validation',
             'imaging':'not imputed for defining each imaging analysis; no-zero replacement',
             'identity_outcome':'unverified identity or absent clinical label retained in registry, not scored in main analysis'},
    evaluation_unit='examination/cycle, repeated patients remain grouped',
    algorithm='L2-penalized Logistic/Ridge; lbfgs; unpenalized intercept; max_iter10000',
    regularization={'search_parameter':'C','C_grid':[.001,.01,.1,1.,10.,100.],
                    'meaning':'inverse regularization strength; do not claim numerically identical to lambda in other software',
                    'inner_folds':3,'inner_score':'negative_log_loss','tie_rule':'exact score ties choose smaller C',
                    'scope':'each clinical and augmented model chooses C separately using same inner splits; do not use outer AUC',
                    'boundary':'record boundary choice, do not expand grid after observing test performance'},
    outer_folds=5,outer_seeds=seeds,primary_repeat_seed=seeds[0],
    repetition='10 fixed seeds for stability, primary estimates from first seed; no best-seed selection or independent-replicate interpretation',
    bootstrap={'iterations':2000,'unit':'patient cluster; all cycles together with multiplicity','paired':True,
               'scores':'fixed primary-repeat OOF predictions','interval':'percentile95; skip and count single-class replicates',
               'seed':20262531,'limitation':'conditional fixed-prediction interval; not full nested-pipeline refit uncertainty'},
    metrics={'primary':'paired delta_AUC for MSE',
             'secondary':['AUC','Brier','paired delta_Brier augmented-minus-reference negative favours augmented',
                          'OOF calibration intercept/slope and descriptive calibration curve'],
             'calibration':'diagnostic only, not refit/calibrate deployed scores on outer test; no calibration model predictor selection'},
    amplitude_inference='evaluate the whole2df block; no ordinary Wald/LRT p-values on tuned Ridge coefficients; betaA=betaP=0 not implied by delta_AUC interval',
    multiplicity='one primary MSE increment; all other imaging analyses and baseline alternatives labeled secondary/exploratory, no winner promotion',
    evidence_context='previous exploratory AUC already seen; new future protocol freeze is not prospective or external validation',
    prior_35_short='part of expanded development; no longer untouched holdout',
    source_alignment='44 duration values aligned to existing358 total years, raw BMI101.9 remains missing per source; old sources/results preserved',
    source_hashes=before,no_new_auc=True,no_model_fit=True,
    analysis_data_sha256=digest(OUT/'MASTER_CLINICAL_SOURCE_ALIGNED_311.csv'),
    outer_folds_sha256=digest(OUT/'OUTER_FOLD_ASSIGNMENTS_LOCAL.csv'),
    inner_folds_sha256=digest(OUT/'INNER_VALIDATION_ASSIGNMENTS_LOCAL.csv'),
    analysis_registry_sha256=digest(OUT/'ANALYSIS_MODEL_REGISTRY.csv'))
save('PROTOCOL_FROZEN.json',protocol)
data_dictionary=[]
for f,reason in [('age','routine female age'),('transfer_embryo_count_component','actual count, not multiple indicator'),
                 ('blastocyst_transfer','two known stages equivalent to existing binary representation'),
                 ('bmi','routine clinical predictor, preserve source missing policy'),
                 ('infertility_years','existing total years including month fraction'),
                 ('endometrial_thickness_mm','routine endometrial measurement available at prediction')]:
    data_dictionary.append(dict(field=f,role='primary_clinical_baseline',df=1,encoding='binary0/1' if f=='blastocyst_transfer' else 'continuous_or_original_count',
                                available_at_prediction=True,reason=reason))
csv('PRIMARY_CLINICAL_BASELINE_FIELDS.csv',data_dictionary)
old_candidates=pd.read_csv(ROOT/'clinical_baseline_design_20261120/CLINICAL_BASELINE_CANDIDATES.csv')
old_candidates['previous_baseline_role']=old_candidates.baseline_role
primary_constructs=['Female age','Embryo number','Embryo developmental stage','BMI','Infertility duration','Endometrial thickness']
old_candidates.loc[old_candidates.construct.isin(primary_constructs),'baseline_role']='core'
old_candidates.loc[old_candidates.construct.isin(primary_constructs),'reason']='reviewed decision: fixed Extended6 clinical comparator; not selected by local AUC; literature grade not upgraded'
old_candidates['freeze_status']='superseded by reviewed Extended6 protocol; baseline sensitivity roles in PROTOCOL_FROZEN.json'
csv('CLINICAL_BASELINE_CANDIDATES_UPDATED.csv',old_candidates)

# Re-express existing Riley scenarios for the actual image-supported analysis sets.
budget_rows=[]
for row in rows:
    pi=row['events']/row['n_main_exams']
    maxr=1-math.exp(2*(pi*math.log(pi)+(1-pi)*math.log(1-pi)))
    for r2 in [.02,.05,.10,.15,.20]:
        s=.9;s2=r2/(r2+.05*maxr)
        q1=(s-1)*math.log(1-r2/s);q2=(s2-1)*math.log(1-r2/s2)
        n3=math.ceil((1.96/.05)**2*pi*(1-pi))
        for unit,n in [('exam',row['n_main_exams']),('patient_reference',row['n_main_patients'])]:
            budget_rows.append(dict(analysis=row['analysis'],observation_basis=unit,n=n,event_rate=pi,assumed_Cox_Snell_R2=r2,
                total_predictor_df=row['total_df'],allowed_two_criteria_df=min(math.floor(n*q1),math.floor(n*q2)),
                n_required_all_criteria=max(math.ceil(row['total_df']/q1),math.ceil(row['total_df']/q2),n3),
                overall_risk_precision_required_n=n3,interpretation='split constructs reduce extra df but do not guarantee sample adequacy; shrinkage not a cure'))
csv('UPDATED_PARAMETER_BUDGET.csv',budget_rows)
assert all(digest(Path(p))==h for p,h in before.items())
assert not master.case_id.duplicated().any()
save('CHECKS.json',dict(original_sources_unchanged=True,registered_exams=311,alignment_changes=len(changes),
    duration_changes=44,BMI_policy_changes=1,primary6_complete_exams=int(master.primary6_clinical_complete.sum()),
    primary6_complete_patient_groups=master.loc[master.primary6_clinical_complete,'patient_id'].nunique(),
    primary_clinical_fields=6,outer_inner_split_audits=len(split_audit),all_patient_overlaps_zero=True,
    no_model_fit=True,no_new_auc=True,protocol_frozen=True,source_hashes=before))
text=f'''# Extended6临床主基线与分构念影像增量方案

版本：2026-11-20 v1。本方案采用用户提供的审查意见；Extended6设为主临床基线，Core3改为精简敏感性，原7项为历史编码敏感性。本轮冻结方案、统一临床来源并预生成分组折，不拟合模型或计算新AUC。既往探索结果保留。

## 临床与影像层级

正式临床参考为年龄、实际胚胎数、囊胚指示、BMI、含月份的不孕总年限、内膜厚度，6df。用户确认前述胚胎信息在超声完成时已确定。连续变量保持连续，不找35岁、BMI24或内膜8mm等数据驱动切点，不筛变量。Core3和原7项只替代临床编码作敏感性，不挑表现更好的基线。

{table(summary[['analysis','tier','baseline_df','added_imaging_df','total_df','n_main_exams','n_main_patients','n_clinical6_complete_exams','n_with_clinical_missing','events','non_events']])}

MSE为唯一主要影像增量；ActivityRate为关键次要探索增量，仍是算法活动率而非真实波频率。双壁幅度作为2df整体次要块，不挑单壁。CC单独次要/解释，不默认捆绑MSE。四影像联合10df、再加活动率11df均探索。

各块的M0必须在该块同病例上重拟合，不能引用全体临床模型的AUC做差；不同块的样本不同，不能按各自AUC跨块排名。原179检查是旧7项加四影像的交集，不是新MSE问题必须受限的n。

## 新源值版数据与缺失

本轮新建MASTER_CLINICAL_SOURCE_ALIGNED_311.csv，不覆盖旧master或桌面358表。44条年限按358既有总年值带入月份，BMI101.9按358建模表原有缺失政策保留缺失，不再从raw补回，逐例变更在审计表。没有修改影像定义、FI、追踪或原始结局。

Extended6完整为298次/280患者。主分析以该块影像可计算＋临床身份/结局已连接为范围，临床少量缺失在训练折内处理：MSE185次/179患者、活动246次/231患者。临床6项完整敏感性分别182次和242次；不是仍写旧7项的179和240。影像本身的不可观察值不插补成已观察，不填0。身份未核实的5次保留登记但不进主分析。

## 验证与调参

外层patient-grouped5折，内层patient-grouped3折；临床与增强模型、两种敏感性基线在同一个影像分析集中使用完全相同外层折。所有插补/标准化必须在每个inner训练折内重新拟合，不能先用完整outer训练集预处理再做inner验证。outer test绝不参与处理或选参。

拟用Ridge Logistic，lbfgs、截距不惩罚、最多10000次迭代。只在内层选择C=[0.001,0.01,0.1,1,10,100]，C是该实现的逆惩罚强度，非跨软件统一数值lambda。按内层负log-loss选择；同分取更小C。两模型分别调C、共用分折；不按outer AUC选参、扩网格或筛变量。

10个预设seed为20260531至20260540，首seed的合并OOF为主要结果，其余作稳定性，不挑seed、不当10个独立实验。已生成外层和内层验证分配文件，全部患者交叉为0；本轮只分折，没有模型拟合或预测。

## 指标和区间

主要指标为MSE的配对ΔAUC。其余报告AUC、Brier及配对差、校准截距/斜率与描述性曲线；ΔBrier=增强−基线，负值更好。患者聚类bootstrap2000次，同一患者所有周期一起抽入，所有模型用同一次抽样。

冻结为对首轮固定OOF预测做百分位区间；这是条件于已得预测的区间，**不是完整模型训练与调参重拟合区间**。分折重复描述另一部分不稳定性，不能称已覆盖全部训练不确定性。单类别bootstrap重复跳过并记录数目，普通DeLong不作为唯一结果。校准仅诊断，不在外层测试集重新校准预测后再报性能。

双壁幅度作为整体块评价ΔAUC/ΔBrier。不能对调参后Ridge系数直接做普通Wald/LRT，不能把ΔAUC区间解释为βAmpA=βAmpP=0的正式参数检验；若需要该零假设检验，应另行指定适用推断流程。本轮不补做P值。

只有MSE一个主要增量，其他保持次要/探索，不用最好结果升级主结论。Core3及旧7项敏感性原则上复用主影像集与分折，避免换样本制造基线差异。

## 样本量、文献与时间线

保留Extended6能使临床比较器更完整；拆开影像块降低新增复杂度，但6→7df也不保证满足Riley情景。UPDATED_PARAMETER_BUDGET.csv保留这一限制，不宣称Ridge修复了样本不足或可以临床部署。

新增核对：[Shingshetty2024综述](https://pubmed.ncbi.nlm.nih.gov/38492930/)确实列年龄、BMI、时长等反复使用因素，但它是以活产为重点的综述，不应误写为专门临床妊娠系统综述。[Wang2025 FET研究](https://pubmed.ncbi.nlm.nih.gov/40636712/)有7302周期并含厚度，但使用RF及前向选择；只补充变量先例，不采用其筛选方法。[BMJ2024方法指南](https://www.bmj.com/content/386/bmj-2023-078276.long)支持外部知识选候选、审查复杂度与缺失。内部A/B证据级不因本次纳入就升级。

之前临床/影像探索性AUC和短视频结果已经查看。本次仅是下一轮预设增量方案的冻结，不能声称所有分析前从未看过结局；短视频已纳入扩展开发，不是全新外部验证。研究表述为exploratory internally validated incremental prediction analysis，实际跑完相应验证后才能报告结果。本轮没有新结果。

## 文件

PROTOCOL_FROZEN.json为机器可读规则；ANALYSIS_MODEL_REGISTRY.csv和COHORT_MEMBERSHIP保留每块样本与原因；PRIMARY_CLINICAL_BASELINE_FIELDS.csv列6项。数据、折分配、审计表均本地保存，不上传患者信息。旧Core优先报告保留为历史，当前决策以本协议为准。
'''
(OUT/'PROTOCOL_FROZEN.md').write_text(text,encoding='utf-8')
print(summary[['analysis','total_df','n_main_exams','n_main_patients','n_clinical6_complete_exams']].to_string(index=False))
print('Frozen specification only; no model fitting or new AUC.')
