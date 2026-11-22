"""Fixed exploratory comparison of saved significant-activity rate, no extraction.

Reuses existing Logistic and patient-cluster bootstrap; no model selection.
"""
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analysis.mse_pregnancy_association.code.frozen_feature_auc import make_model, bootstrap_oof, GROUP, OUTCOME, SEED
from analysis.mse_pregnancy_association.code.all_features_auc import grouped_splits

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
SOURCE=ROOT/'outputs/feature_inventory_desktop276_20261120/MODEL_INPUT_MASTER_TABLE.csv'
COUNT_ROOT=ROOT/'outputs/wave_counts_223_20261119'
OUT=ROOT/'outputs/exploratory_activity_rate_increment_20261120'
RATE='significant_propagation_activity_frequency_cpm'
CLINICAL=['age','bmi','infertility_years','transfer_embryo_count_multiple',
          'endometrial_thickness_mm','blastocyst_transfer','endometrial_type_C']
B=CLINICAL+['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude',
            'record_MSE_time_equal','record_CC_time_equal']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows): pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def table(df):
    rows=['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']
    rows+=['|'+'|'.join(str(v) for v in row)+'|' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def run():
    OUT.mkdir(exist_ok=False)
    paths=[SOURCE,COUNT_ROOT/'wave_counts.csv',COUNT_ROOT/'protocol_before_counting.json']
    hashes={str(p):digest(p) for p in paths}
    master=pd.read_csv(SOURCE,dtype={'patient_id':str,'exam_id':str})
    discovery=master[master.cohort.eq('desktop_276') & master.existing_result_link.eq(True)].copy()
    assert len(discovery)==223 and not discovery.exam_id.duplicated().any()
    counts=pd.read_csv(paths[1])
    paired=discovery[['case_id',RATE]].merge(counts[['case_id',RATE]],on='case_id',suffixes=('_master','_saved'),validate='one_to_one')
    np.testing.assert_allclose(paired[RATE+'_master'],paired[RATE+'_saved'],equal_nan=True)
    # No missing-rate measurement is reinterpreted as zero.
    assert discovery[RATE].notna().sum()==217
    discovery[GROUP]=discovery.patient_id
    discovery[OUTCOME]=pd.to_numeric(discovery.outcome)
    assert discovery[OUTCOME].isin([0,1]).all()
    primary=discovery[discovery[B+[RATE]].notna().all(axis=1)].copy().reset_index(drop=True)
    observable=discovery[discovery[RATE].notna()].copy().reset_index(drop=True)
    analyses=[('primary_B_complete',primary,{'B':B,'B_plus_activity_rate':B+[RATE]}),
              ('secondary_rate_observable',observable,{'clinical':CLINICAL,'clinical_plus_activity_rate':CLINICAL+[RATE]}),
              ('sensitivity_all_discovery',discovery.reset_index(drop=True),{'clinical':CLINICAL,'clinical_plus_activity_rate':CLINICAL+[RATE]})]
    protocol=dict(user_authorization='2026-11-20: test algorithm-detected significant propagation activity frequency as exploratory feature',
        feature=RATE,endpoint='clinical pregnancy; exploratory internal grouped prediction',
        model='reuse existing Logistic C=1e6 liblinear; training-fold median imputation and standardization',
        folds=5,repeats=5,first_seed=SEED,bootstrap=1000,
        comparisons={name:models for name,_,models in analyses},
        cohorts={name:len(d) for name,d,_ in analyses},
        uncertainty='paired patient-cluster bootstrap of fixed first-repeat OOF predictions, no refitting; repeat range describes split sensitivity',
        no_hyperparameter_search=True,no_feature_selection=True,no_cutoff_search=True,
        missing='primary requires existing B/rate support; 217 observable analysis retains clinical missing for fold imputation; all223 sensitivity retains 6 missing rates for fold imputation',
        holdout_used=False,detector_or_thresholds_changed=False,extraction_rerun=False,
        source_hashes=hashes,count_protocol=json.loads(paths[2].read_text(encoding='utf-8')),
        interpretation='algorithm activity rate, not true physiological wave rate; model AUC does not validate measurement')
    save('protocol_before_fit.json',protocol)
    metrics=[];deltas=[];repeats=[];audits=[];coefficients=[];predictions=[]
    for analysis,data,models in analyses:
        data=data.reset_index(drop=True)
        y=data[OUTCOME].to_numpy(int); groups=data[GROUP].to_numpy(str)
        names=list(models)
        scores_by_repeat=[]
        for repeat in range(5):
            scores={name:np.full(len(data),np.nan) for name in models}
            assignments=np.full(len(data),-1,int)
            for fold,(train,test) in enumerate(grouped_splits(y,groups,5,SEED+repeat),1):
                assignments[test]=fold
                for name,columns in models.items():
                    fitted=make_model().fit(data.iloc[train][columns],y[train])
                    scores[name][test]=fitted.predict_proba(data.iloc[test][columns])[:,1]
                    audits.append(dict(analysis=analysis,repeat=repeat,fold=fold,model=name,train_n=len(train),test_n=len(test),
                                       patient_overlap=len(set(groups[train])&set(groups[test])),train_class_count=len(np.unique(y[train])),test_class_count=len(np.unique(y[test]))))
                    if RATE in columns:
                        # SD-standardized coefficient, not causal or adjusted effect inference.
                        coefficients.append(dict(analysis=analysis,repeat=repeat,fold=fold,model=name,
                            activity_rate_coefficient=float(fitted.steps[-1][1].coef_[0][columns.index(RATE)])))
            assert (assignments>0).all() and all(np.isfinite(p).all() for p in scores.values())
            aucs={name:float(roc_auc_score(y,p)) for name,p in scores.items()}
            scores_by_repeat.append(aucs)
            repeats.extend(dict(analysis=analysis,repeat=repeat,model=name,AUC=auc) for name,auc in aucs.items())
            if repeat==0:
                draws=bootstrap_oof(data,scores,1000,SEED+1000)
                for name in models:
                    lo,hi=np.quantile(draws[name],[.025,.975])
                    metrics.append(dict(analysis=analysis,model=name,n_exams=len(data),n_patients=data.patient_id.nunique(),
                                        events=int(y.sum()),non_events=int((y==0).sum()),AUC=aucs[name],CI_low=float(lo),CI_high=float(hi)))
                delta=draws[names[1]]-draws[names[0]]
                lo,hi=np.quantile(delta,[.025,.975])
                deltas.append(dict(analysis=analysis,comparison=names[1]+' minus '+names[0],n_exams=len(data),n_patients=data.patient_id.nunique(),
                                   delta_AUC=aucs[names[1]]-aucs[names[0]],CI_low=float(lo),CI_high=float(hi)))
                for i,row in data.iterrows():
                    predictions.append(dict(analysis=analysis,exam_id=row.exam_id,case_id=row.case_id,patient_id=row.patient_id,
                                            outcome=int(y[i]),fold=int(assignments[i]),**{name:float(scores[name][i]) for name in models}))
        for row in metrics:
            if row['analysis']==analysis:
                values=[v[row['model']] for v in scores_by_repeat]
                row.update(repeat_mean_AUC=float(np.mean(values)),repeat_min_AUC=min(values),repeat_max_AUC=max(values))
        delta_values=[v[names[1]]-v[names[0]] for v in scores_by_repeat]
        deltas[-1].update(repeat_mean_delta=float(np.mean(delta_values)),repeat_min_delta=min(delta_values),repeat_max_delta=max(delta_values))
    csv('AUC_metrics.csv',metrics);csv('paired_AUC_differences.csv',deltas);csv('repeat_metrics.csv',repeats)
    csv('fold_audit.csv',audits);csv('rate_coefficients.csv',coefficients);csv('first_repeat_OOF.csv',predictions)
    support_cols=['exam_id','patient_id','case_id',RATE,'significant_propagation_activity_count','effective_observation_seconds',
                  'unresolved_activity_rows','anterior_valid_fraction','posterior_valid_fraction']
    discovery[support_cols].to_csv(OUT/'case_measurement_support.csv',index=False,encoding='utf-8-sig')
    assert all(digest(Path(p))==value for p,value in hashes.items())
    assert all(r['patient_overlap']==0 for r in audits)
    save('CHECKS.json',dict(sources_unchanged=True,patient_overlap_zero=True,all_OOF_finite=True,
                           saved_rates_exactly_reproduced=True,holdout_unused=True,primary_exams=len(primary),rate_observable_exams=len(observable)))
    report='''# 显著传播活动频率探索性增量测试

用户接受以“算法检出的显著传播活动频率”测试，不将它称为真实蠕动波频率。复用保存的223例结果，不重提取、不修改0.6或噪声参照。原计数协议标记algorithm_frozen=false；本轮固定其既有输出，不宣称检测器已完成验证。

主比较是同153次检查的B与B+活动频率，模型固定沿用既有Logistic，无模型选择；5折按患者分组，5个既有固定种子重复，插补与标准化仅训练折内完成。仅报告内部探索结果，hold-out未使用。主分析是B完整支持子集，结果不能直接推广至全部276例。

## AUC

'''+table(pd.DataFrame(metrics).round(4))+'''

## 配对增量

'''+table(pd.DataFrame(deltas).round(4))+'''

95%区间为首轮固定折外预测的患者聚类bootstrap（1000次），不是训练全过程重拟合区间。5次重复的范围仅反映分折敏感性；不把重复或窗口当独立患者。每个比较均在相同病例、相同分折上配对，三个分析之间的AUC不能直接相减归因为特征增量。全部223例敏感性中6例无可观察支持的频率保持缺失，仅训练折中位数插补，不填0。

## 测量解释与后续角色

本轮按探索指标纳入，和AUC结果无关。local_once误计、相邻活动重复、弱活动漏检仍是已知局限；无逐例人工真值，无法从本测试估计患者中的误报率、漏检率或计数准确率。case_measurement_support.csv保留逐例频率、次数、有效观察时长和已有支持字段；未证明的误差不填零。

旧R2纵向主频未纳入。B仍为7项临床+双壁幅度+完整记录ER-MSE/CC；B+活动频率为12项探索组合，不升级为已验证主要指标。输出仅本地保存。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))
    print(pd.DataFrame(deltas).round(4).to_string(index=False))

if __name__=='__main__': run()
