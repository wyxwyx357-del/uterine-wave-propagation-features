"""Train fixed existing models on discovery, evaluate saved short-video features.

Exact clinical joins only. Retain all short rows and both label sources.
No short-cohort fitting, selection, calibration, or detection changes.
"""
import os
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analysis.mse_pregnancy_association.code.frozen_feature_auc import make_model, bootstrap_oof, GROUP, OUTCOME, SEED
from analysis.mse_pregnancy_association.code.exploratory_activity_rate_increment import CLINICAL, B, RATE

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OUT=ROOT/'outputs/short_activity_auc_20261120'
INVENTORY=ROOT/'outputs/feature_inventory_desktop276_20261120/MODEL_INPUT_MASTER_TABLE.csv'
SHORT=ROOT/'outputs/short_duration_holdout_validation_20261120'
COUNTS=ROOT/'outputs/short_activity_rate_20261120/wave_counts.csv'
BOOK = Path(os.environ.get('CLINICAL_SOURCE_PATH', 'data/clinical_source.xlsx'))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,df):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def table(df):
    return '\n'.join(['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']+
                     ['|'+'|'.join(str(v) for v in r)+'|' for r in df.itertuples(index=False,name=None)])

def run():
    OUT.mkdir(exist_ok=False)
    files=[INVENTORY,COUNTS,BOOK,SHORT/'inputs/input_manifest.json',SHORT/'measurement/manifest.csv']
    hashes={str(p):sha(p) for p in files}
    master=pd.read_csv(INVENTORY,dtype={'patient_id':str,'exam_id':str})
    train=master[master.cohort.eq('desktop_276') & master.existing_result_link.eq(True)].copy()
    short=master[master.cohort.eq('short_holdout_separate')].copy()
    assert len(short)==35 and len(train)==223
    short=short.drop(columns=[RATE,'significant_propagation_activity_count','effective_observation_seconds','unresolved_activity_rows'])
    counts=pd.read_csv(COUNTS)
    short=short.merge(counts[['case_id',RATE,'significant_propagation_activity_count','effective_observation_seconds','unresolved_activity_rows','processing_status','failure_reason']],
                       on='case_id',how='left',validate='one_to_one',suffixes=('','_counter'))
    final=pd.read_excel(BOOK,sheet_name=0)
    clinical=pd.read_excel(BOOK,sheet_name=1)
    assert final['视频目录标识'].is_unique and clinical['临床检查编号'].is_unique
    mapping=final[['视频目录标识','临床检查编号']].rename(columns={'视频目录标识':'case_id','临床检查编号':'linked_exam_id'})
    clinical=clinical.rename(columns={'临床检查编号':'linked_exam_id'})
    short=short.merge(mapping,on='case_id',how='left',validate='one_to_one').merge(clinical,on='linked_exam_id',how='left',validate='many_to_one')
    names=['年龄','BMI','不孕年限_年','移植胚胎数_多胚胎','内膜厚度_mm','胚胎种类_含囊胚','内膜分型_C型']
    for dest,source in zip(CLINICAL,names):short[dest]=pd.to_numeric(short[source],errors='coerce')
    short['prior_exam_key']=short.exam_id
    short['exam_id']=short.linked_exam_id.fillna(short.exam_id)
    short['patient_id']=short['患者聚类标识'].fillna(short.patient_id)
    short['patient_identity_status']=np.where(short.linked_exam_id.notna(),'existing_clinical_cluster_exact_exam_join','unverified_exam_fallback')
    short['outcome_prior_MSE_analysis']=short.outcome
    short['outcome_clinical']=pd.to_numeric(short['结局_临床妊娠'],errors='coerce')
    # Directory source retained separately; never overwrite frozen MSE results.
    from analysis.mse_pregnancy_association.code.short_holdout_outcome_validation import _outcome_from_path
    manifest=json.loads((SHORT/'inputs/input_manifest.json').read_text(encoding='utf-8'))
    path_labels={}
    for r in manifest['cases']:
        try:path_labels[r['case_id']]=_outcome_from_path(r['video_path'])[0]
        except ValueError:path_labels[r['case_id']]=np.nan
    short['outcome_directory']=short.case_id.map(path_labels)
    short['outcome_disagreement']=short.outcome_clinical.notna() & short.outcome_directory.notna() & short.outcome_clinical.ne(short.outcome_directory)
    short['outcome']=short.outcome_clinical
    verified=short.linked_exam_id.notna() & short.patient_identity_status.eq('existing_clinical_cluster_exact_exam_join')
    shared=set(train.patient_id)&set(short.loc[verified,'patient_id'])
    # Remove whole shared patient groups from training if found, never split cycles.
    train=train[~train.patient_id.isin(shared)].reset_index(drop=True)
    short['clinical7_complete']=short[CLINICAL].notna().all(axis=1)
    short['B_rate_complete']=short[B+[RATE]].notna().all(axis=1)
    short['primary_test_eligible']=verified & short.B_rate_complete & short.outcome_clinical.isin([0,1])
    short['clinical_rate_test_eligible']=verified & short.clinical7_complete & short[RATE].notna() & short.outcome_clinical.isin([0,1])
    csv('SHORT_MASTER_ALL35.csv',short)
    csv('LABEL_SOURCE_AUDIT.csv',short[['case_id','exam_id','patient_id','patient_identity_status','outcome_clinical','outcome_directory','outcome_prior_MSE_analysis','outcome_disagreement']])
    groups=[('primary_common_B_support',train[train[B+[RATE]].notna().all(axis=1)],short[short.primary_test_eligible],
             {'clinical':CLINICAL,'clinical_plus_activity_rate':CLINICAL+[RATE],'B':B,'B_plus_activity_rate':B+[RATE]},True),
            ('secondary_clinical_rate',train[train[RATE].notna()],short[short.clinical_rate_test_eligible],
             {'clinical':CLINICAL,'clinical_plus_activity_rate':CLINICAL+[RATE]},True),
            ('diagnostic_rate_only_directory_labels',train[train[RATE].notna()],short[short[RATE].notna() & short.outcome_directory.isin([0,1])],
             {'activity_rate_only':[RATE]},False)]
    save('protocol_before_fit.json',dict(feature=RATE,user_authorized_short_extraction_and_auc=True,
        label_primary='final clinically reviewed workbook, same source as discovery labels; directory labels sensitivity only',
        discordant_label_cases=int(short.outcome_disagreement.sum()),label_repair=False,
        model='existing Logistic C=1e6 liblinear, median imputer and standardization fitted discovery only',
        no_tuning=True,no_short_training=True,no_threshold_change=True,bootstrap=1000,
        no_test_driven_calibration=True,short_cohort_previously_inspected=True,
        interpretation='exploratory held-cohort evaluation, not untouched external validation; no measurement validity claim',
        patient_overlap_groups_removed=len(shared),clinical_linked_cases=int(verified.sum()),
        analyses={name:dict(train_n=len(tr),test_n=len(te),features=models,identity_verified=iv) for name,tr,te,models,iv in groups},
        source_hashes=hashes))
    metrics=[];deltas=[];predictions=[];coefs=[];foldchecks=[]
    for name,tr,te,models,identity_verified in groups:
        tr=tr.reset_index(drop=True);te=te.reset_index(drop=True)
        if name.startswith('diagnostic'):te['outcome']=te.outcome_directory
        if len(te)==0 or te.outcome.nunique()!=2:
            metrics.append(dict(analysis=name,status='not_evaluable_both_classes_required',test_n=len(te)))
            continue
        if identity_verified:assert not set(tr.patient_id)&set(te.patient_id)
        assert tr.outcome.isin([0,1]).all()
        te[GROUP]=te.patient_id;te[OUTCOME]=te.outcome.astype(int)
        scores={}
        for model,columns in models.items():
            fit=make_model().fit(tr[columns],tr.outcome.to_numpy(int))
            scores[model]=fit.predict_proba(te[columns])[:,1]
            assert np.isfinite(scores[model]).all()
            coefs.extend(dict(analysis=name,model=model,feature=f,coefficient=float(c)) for f,c in zip(columns,fit.steps[-1][1].coef_[0]))
        # Save predictions before measuring short-cohort discrimination.
        for i,row in te.iterrows():
            predictions.append(dict(analysis=name,case_id=row.case_id,exam_id=row.exam_id,patient_id=row.patient_id,
                                    outcome=int(row.outcome),**{m:float(s[i]) for m,s in scores.items()}))
        csv('fixed_test_predictions.csv',pd.DataFrame(predictions))
        draws=bootstrap_oof(te,scores,1000,SEED+1000)
        for model,p in scores.items():
            lo,hi=np.quantile(draws[model],[.025,.975])
            metrics.append(dict(analysis=name,model=model,train_n=len(tr),train_patients=tr.patient_id.nunique(),
                test_n=len(te),test_patients=te.patient_id.nunique(),events=int(te.outcome.sum()),non_events=int(te.outcome.eq(0).sum()),
                AUC=float(roc_auc_score(te.outcome,p)),CI_low=float(lo),CI_high=float(hi),bootstrap_valid=len(draws[model]),identity_verified=identity_verified))
        for base,extra in [('clinical','clinical_plus_activity_rate'),('B','B_plus_activity_rate')]:
            if base in scores and extra in scores:
                lo,hi=np.quantile(draws[extra]-draws[base],[.025,.975])
                deltas.append(dict(analysis=name,comparison=extra+' minus '+base,test_n=len(te),
                    delta_AUC=float(roc_auc_score(te.outcome,scores[extra])-roc_auc_score(te.outcome,scores[base])),CI_low=float(lo),CI_high=float(hi)))
        if identity_verified:
            same=te.outcome_directory.isin([0,1])
            if te.loc[same,'outcome_directory'].nunique()==2:
                for model,p in scores.items():
                    metrics.append(dict(analysis=name+'_matched_label_subset_clinical',model=model,train_n=len(tr),
                        test_n=int(same.sum()),events=int(te.loc[same,'outcome'].sum()),non_events=int(te.loc[same,'outcome'].eq(0).sum()),
                        AUC=float(roc_auc_score(te.loc[same,'outcome'],p[same])),identity_verified=True))
                    metrics.append(dict(analysis=name+'_directory_label_sensitivity',model=model,train_n=len(tr),
                        test_n=int(same.sum()),events=int(te.loc[same,'outcome_directory'].sum()),non_events=int(te.loc[same,'outcome_directory'].eq(0).sum()),
                        AUC=float(roc_auc_score(te.loc[same,'outcome_directory'],p[same])),identity_verified=True))
        foldchecks.append(dict(analysis=name,train_n=len(tr),test_n=len(te),known_patient_overlap=len(set(tr.patient_id)&set(te.patient_id)),identity_verified=identity_verified))
    csv('AUC_metrics.csv',pd.DataFrame(metrics));csv('paired_AUC_differences.csv',pd.DataFrame(deltas));csv('model_coefficients.csv',pd.DataFrame(coefs))
    csv('patient_overlap_audit.csv',pd.DataFrame(foldchecks))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    checks=dict(all35_retained=len(short)==35,sources_unchanged=True,short_rows_used_for_fit=0,
        all_verified_comparisons_patient_disjoint=all(r['known_patient_overlap']==0 for r in foldchecks if r['identity_verified']),
        clinical_linked=int(verified.sum()),clinical7_complete=int((verified & short.clinical7_complete).sum()),
        label_conflicts=int(short.outcome_disagreement.sum()),short_rate_available=int(short[RATE].notna().sum()),
        primary_short_n=int(short.primary_test_eligible.sum()),secondary_short_n=int(short.clinical_rate_test_eligible.sum()))
    save('CHECKS.json',checks)
    report='''# 短视频显著传播活动频率与固定模型AUC测试

35次短视频全部保留；沿用原223例算法、0.6显著性、噪声参照及全部代码哈希。34份已有A0时空图完成活动频率计算，1次原几何输入失败无图，保持缺失，没有重追踪。实际时长45.37–47.68秒。

## 队列与标签

'''+json.dumps(checks,ensure_ascii=False,indent=2)+'''

按case_id精确连接358例最终人工复核表，再以检查编号连接临床7项及患者聚类键，不按姓名拼音猜配。30次连接成功，5次未连接保留；29次临床7项完整。已连接患者键未发现与训练患者重叠，但这依赖既有聚类键的身份准确性。主测试使用最终复核临床妊娠标签，与训练队列标签来源一致；1次目录标签与临床标签不同，保留两者，未修改旧MSE验证文件。

模型、缩放与插补只在原发现队列拟合；短视频仅预测，未调参、校准、翻转分数或选阈值。B和B+活动频率在同一训练/测试交集配对；临床+频率另有较大样本的次要比较。原训练中的6次活动频率缺失不填0，本次含活动频率模型使用原217次可观察支持队列或其B共同支持子集。

## 固定模型在短视频上的AUC

'''+table(pd.DataFrame(metrics).round(4))+'''

## 同测试集配对增量

'''+table(pd.DataFrame(deltas).round(4))+'''

区间是固定预测下按短视频患者聚类bootstrap，不重拟合训练模型。小样本且未妊娠例数少，区间不稳定；不跨不同测试子集直接比较AUC。目录标签敏感性使用相同已保存预测，不据结果决定哪份标签更好。

## 解释边界

rate-only目录标签分析包括未能建立稳定患者连接者，仅作诊断，不能称独立患者验证。主测试只用精确临床连接且特征可用者，未把临床全部缺失者用训练中位数伪装成已观察患者。

这些短视频此前MSE结果已被查看，本次属于固定方法的追加探索评估，不是未触碰的外部验证。AUC不验证活动误报、漏检或真实波次数；没有人工真值的误差保持未知。完整病例、身份/标签冲突、缺失与逐例支持均在SHORT_MASTER_ALL35.csv、LABEL_SOURCE_AUDIT.csv及short_activity_rate_20261120中保留。只本地输出。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=True));print(pd.DataFrame(metrics).round(4).to_string(index=False));print(pd.DataFrame(deltas).round(4).to_string(index=False))

if __name__=='__main__':run()
