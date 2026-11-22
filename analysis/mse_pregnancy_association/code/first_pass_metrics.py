"""Summarize fixed10-repeat mean OOF; paired patient-cluster bootstrap2000.

Runnable independently after the fitting stage; never refits prediction models.
"""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize, brentq
from sklearn.metrics import roc_auc_score

def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(root,name,obj):pd.DataFrame(obj).to_csv(root/name,index=False,encoding='utf-8-sig')
def md(d):
    return '\n'.join(['|'+'|'.join(map(str,d.columns))+'|','|'+'|'.join(['---']*len(d.columns))+'|']+
        ['|'+'|'.join(str(v).replace('|','/').replace('\n',' ') for v in row)+'|' for row in d.itertuples(index=False,name=None)])
def calibration(y,p):
    if len(np.unique(y))!=2:return np.nan,np.nan,'one_class'
    z=logit(np.clip(p,1e-12,1-1e-12))
    intercept=float(brentq(lambda a:float(np.mean(expit(a+z)-y)),-50,50))
    if np.ptp(z)<1e-12:return intercept,np.nan,'constant_predictions_slope_not_identifiable'
    def fun(b):
        eta=b[0]+b[1]*z;error=expit(eta)-y
        return float(np.mean(np.logaddexp(0,eta)-y*eta)),np.array([error.mean(),np.mean(error*z)])
    fit=minimize(fun,[0.,1.],jac=True,method='BFGS',options={'maxiter':1000,'gtol':1e-8})
    return intercept,float(fit.x[1]),'converged' if fit.success else 'diagnostic_optimizer_warning:'+str(fit.message)
def metric(y,p):
    a,s,status=calibration(y,p)
    return dict(AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                Brier=float(np.mean((p-y)**2)),calibration_intercept=a,calibration_slope=s,calibration_status=status)
def auc_plan(y,p):
    unique,inverse=np.unique(p,return_inverse=True)
    return dict(inverse=inverse,size=len(unique),positive=y.astype(float),negative=(1-y).astype(float))
def fast_auc(plan,w):
    pos=np.bincount(plan['inverse'],weights=w*plan['positive'],minlength=plan['size'])
    neg=np.bincount(plan['inverse'],weights=w*plan['negative'],minlength=plan['size'])
    if not pos.sum() or not neg.sum():return np.nan
    return float(np.dot(pos,np.cumsum(neg)-.5*neg)/(pos.sum()*neg.sum()))
def boot_pair(y,groups,p0,p1,seed):
    plans=[auc_plan(y,p0),auc_plan(y,p1)]
    unique,inverse=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed)
    # Numerical check of tie-aware weighted AUC against the standard implementation.
    check_rng=np.random.default_rng(seed+1)
    for plan,p in zip(plans,[p0,p1]):
        np.testing.assert_allclose(fast_auc(plan,np.ones(len(y))),roc_auc_score(y,p),atol=1e-12)
        for _ in range(3):
            w=check_rng.integers(1,4,len(y)).astype(float)
            np.testing.assert_allclose(fast_auc(plan,w),roc_auc_score(y,p,sample_weight=w),atol=1e-12)
    squared=np.array([(p0-y)**2,(p1-y)**2])
    draws=[];invalid=0
    for _ in range(2000):
        weights=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse].astype(float)
        a0=fast_auc(plans[0],weights);a1=fast_auc(plans[1],weights)
        if not np.isfinite(a0) or not np.isfinite(a1):invalid+=1;continue
        b0=float(np.dot(weights,squared[0])/weights.sum());b1=float(np.dot(weights,squared[1])/weights.sum())
        draws.append([a0,a1,a1-a0,b0,b1,b1-b0])
    a=np.asarray(draws)
    ci=np.quantile(a,[.025,.975],axis=0) if len(a) else np.full((2,6),np.nan)
    return ci,len(a),invalid

def summarize(root):
    root=Path(root)
    pred=pd.read_csv(root/'03_OOF_PREDICTIONS.csv',dtype={'exam_id':str,'patient_id':str})
    summary=pd.read_csv(root/'ANALYSIS_SET_SUMMARY.csv')
    protocol=json.loads((root/'PROTOCOL_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert not pred.duplicated(['analysis_set','model','repeat','exam_id']).any()
    counts=pred.groupby(['analysis_set','model','exam_id']).size()
    assert counts.eq(10).all()
    keys=['analysis_set','module','role','model','outcome','patient_id','exam_id','case_id','video_duration_s','long_short_source']
    averaged=pred.groupby(keys,dropna=False,as_index=False).agg(predicted_probability=('predicted_probability','mean'),repeat_count=('repeat','nunique'))
    assert averaged.repeat_count.eq(10).all()
    csv(root,'03b_MEAN_OOF_PREDICTIONS.csv',averaged)
    repeatrows=[];repeatdelta=[]
    for (analysis,repeat),f in pred.groupby(['analysis_set','repeat'],sort=False):
        pivot=f.pivot(index=['exam_id','patient_id','outcome','fold'],columns='role',values='predicted_probability').reset_index()
        assert pivot[['baseline','augmented']].notna().all().all()
        vals={}
        for role in ['baseline','augmented']:
            q=f[f.role==role];y=q.outcome.to_numpy(int);p=q.predicted_probability.to_numpy(float)
            vals[role]=metric(y,p)
            repeatrows.append(dict(analysis_set=analysis,repeat=repeat,model=q.model.iloc[0],role=role,
                                  n_exam=len(q),n_patient=q.patient_id.nunique(),**vals[role]))
        repeatdelta.append(dict(analysis_set=analysis,repeat=repeat,Delta_AUC=vals['augmented']['AUC']-vals['baseline']['AUC'],
                                Delta_Brier=vals['augmented']['Brier']-vals['baseline']['Brier']))
    csv(root,'06_REPEAT_STABILITY.csv',repeatrows);csv(root,'06b_REPEAT_INCREMENTAL_STABILITY.csv',repeatdelta)
    performance=[];increment=[];subperformance=[];subincrement=[];audits=[]
    for i,row in enumerate(summary.itertuples(index=False)):
        a=averaged[averaged.analysis_set==row.analysis_set]
        base=a[a.role=='baseline'].sort_values('exam_id').reset_index(drop=True)
        aug=a[a.role=='augmented'].sort_values('exam_id').reset_index(drop=True)
        assert base[['exam_id','patient_id','outcome']].equals(aug[['exam_id','patient_id','outcome']])
        assert len(base)==row.n_exams
        y=base.outcome.to_numpy(int);groups=base.patient_id.to_numpy(str)
        p0=base.predicted_probability.to_numpy(float);p1=aug.predicted_probability.to_numpy(float)
        ci,valid,invalid=boot_pair(y,groups,p0,p1,20262531+i*100)
        point={}
        for j,(name,role,p) in enumerate([(row.baseline_model,'baseline',p0),(row.extended_model,'augmented',p1)]):
            point[role]=metric(y,p)
            performance.append(dict(model=name,analysis_set=row.analysis_set,module=row.module,tier=row.tier,role=role,
                n_exam=len(base),n_patient=base.patient_id.nunique(),pregnancy_n=int(y.sum()),non_pregnancy_n=int((y==0).sum()),
                repeated_patient_groups=row.repeated_patient_groups,**point[role],AUC_CI_low=ci[0,j],AUC_CI_high=ci[1,j],
                Brier_CI_low=ci[0,j+3],Brier_CI_high=ci[1,j+3],
                bootstrap_valid=valid,bootstrap_invalid_one_class=invalid,summary='per-exam mean10OOF then cluster bootstrap',
                calibration_CI='not estimated in this first pass; diagnostic point estimates only'))
        delta=point['augmented']['AUC']-point['baseline']['AUC']
        increment.append(dict(comparison=row.module,analysis_set=row.analysis_set,tier=row.tier,baseline_model=row.baseline_model,
            extended_model=row.extended_model,n=len(base),n_patient=base.patient_id.nunique(),
            AUC_baseline=point['baseline']['AUC'],AUC_extended=point['augmented']['AUC'],Delta_AUC=delta,
            Delta_AUC_CI_low=ci[0,2],Delta_AUC_CI_high=ci[1,2],
            Brier_baseline=point['baseline']['Brier'],Brier_extended=point['augmented']['Brier'],
            Delta_Brier=point['augmented']['Brier']-point['baseline']['Brier'],
            Delta_Brier_CI_low=ci[0,5],Delta_Brier_CI_high=ci[1,5],bootstrap_valid=valid,
            interpretation='未检测到明确的增量区分能力' if ci[0,2]<=0<=ci[1,2] else '存在正向点估计；需结合区间精度及探索性质解释' if delta>0 else '当前增量区分能力点估计为负'))
        for s,source in enumerate(['desktop_276','short_35'],1):
            mask=base.long_short_source.eq(source).to_numpy()
            yy=y[mask];gg=groups[mask];pp0=p0[mask];pp1=p1[mask]
            if len(yy)==0 or len(np.unique(yy))<2:
                subincrement.append(dict(analysis_set=row.analysis_set,source=source,n=len(yy),n_patient=len(np.unique(gg)),
                                         status='AUC unavailable: fewer than two outcome classes'))
                continue
            sci,sv,si=boot_pair(yy,gg,pp0,pp1,20262531+i*100+s)
            for j,(name,p) in enumerate([(row.baseline_model,pp0),(row.extended_model,pp1)]):
                subperformance.append(dict(analysis_set=row.analysis_set,source=source,model=name,n_exam=len(yy),n_patient=len(np.unique(gg)),
                    pregnancy_n=int(yy.sum()),non_pregnancy_n=int((yy==0).sum()),AUC=float(roc_auc_score(yy,p)),AUC_CI_low=sci[0,j],AUC_CI_high=sci[1,j],bootstrap_valid=sv))
            subincrement.append(dict(analysis_set=row.analysis_set,source=source,n=len(yy),n_patient=len(np.unique(gg)),
                Delta_AUC=float(roc_auc_score(yy,pp1)-roc_auc_score(yy,pp0)),CI_low=sci[0,2],CI_high=sci[1,2],bootstrap_valid=sv,
                interpretation='same overall OOF scores, no subgroup retraining or selection'))
        audits.append(dict(analysis_set=row.analysis_set,paired_exams_equal=True,paired_patients_equal=True,paired_outcomes_equal=True,
                           every_probability_averages_exactly10_repeats=True,bootstrap_unit='patient',no_model_refit_in_bootstrap=True))
        save(root/'progress.json',dict(stage='metrics_bootstrap',completed_analysis_sets=i+1,total_analysis_sets=len(summary)))
    perf=pd.DataFrame(performance);inc=pd.DataFrame(increment)
    csv(root,'04_MODEL_PERFORMANCE.csv',perf);csv(root,'05_INCREMENTAL_PERFORMANCE.csv',inc)
    csv(root,'10_LONG_SHORT_MODEL_PERFORMANCE.csv',subperformance);csv(root,'11_LONG_SHORT_INCREMENTAL_PERFORMANCE.csv',subincrement)
    csv(root,'PAIRING_AND_BOOTSTRAP_AUDIT.csv',audits)
    # Match first- and averaged-repeat aggregation counts without treating repeats as N.
    assert perf.groupby('analysis_set').n_exam.nunique().eq(1).all()
    assert perf.n_exam.max()<=311
    assert hashlib.sha256(Path(protocol['source_master']).read_bytes()).hexdigest()==protocol['source_master_sha256']
    fitchecks=json.loads((root/'FIT_CHECKS.json').read_text(encoding='utf-8'))
    fitchecks.update(all_paired_case_patient_outcome_sets_equal=True,mean_OOF_uses10_predictions_per_exam=True,
                     fast_weighted_AUC_checked_against_sklearn=True,bootstrap_iterations_requested=2000,
                     repetitions_not_stacked_as_independent_samples=True,performance_analysis_sets=len(summary),
                     source_master_unchanged_after_metrics=True)
    save(root/'CHECKS_FINAL.json',fitchecks)
    def delta_table(module):
        f=inc[inc.comparison==module]
        return md(f[['analysis_set','n','n_patient','AUC_baseline','AUC_extended','Delta_AUC','Delta_AUC_CI_low','Delta_AUC_CI_high','Delta_Brier']].round(4))
    rp=pd.DataFrame(repeatdelta)
    stab=rp.groupby('analysis_set').agg(repeat_delta_mean=('Delta_AUC','mean'),repeat_delta_min=('Delta_AUC','min'),repeat_delta_max=('Delta_AUC','max')).reset_index()
    csv(root,'06c_REPEAT_SUMMARY.csv',stab)
    report=f'''# 第一轮Extended6影像增量内部验证

本轮按最新执行指令运行：Ridge Logistic，外层患者分组分层5折、内层患者分组5折、10个固定seed。只调C，不竞争模型、不筛变量、不修改特征或阈值。未加载任何旧纵向频率字段，Core3及旧7项本轮仅预留。

## 第一部分 数据集与患者数

样本库311次检查，临床源值已统一44条漏月及1条BMI缺失政策；桌面源表和旧master/旧结果未改写。身份未核实者保持登记但不假设为新患者。每个影像模块只用已观察影像，临床缺失处理版和完整病例版同时列出；联合模型严格完整病例。

{md(summary[['analysis_set','tier','n_exams','n_unique_patients','pregnancy_n','non_pregnancy_n','repeated_patient_groups','n_with_missing_clinical','long_n','short_n']])}

01_ANALYSIS_COHORTS.csv逐例记录纳入和缺失原因。配对双方病例、患者及标签完全一致，检查通过后才计算ΔAUC。指标为检查/周期级；同患者周期一起分折和bootstrap。

## 第二部分 Clinical Extended6表现

固定年龄、实际胚胎数、囊胚指示、BMI、含月总年限、内膜厚度。各影像子集均重新获得自身临床参考，不能引用全体临床AUC作差。没有单一跨所有子集通用的临床AUC。

{md(perf[perf.role=='baseline'][['analysis_set','n_exam','n_patient','AUC','AUC_CI_low','AUC_CI_high','Brier','calibration_intercept','calibration_slope']].round(4))}

## 第三部分 Primary Extended6与ER-MSE

{delta_table('MSE')}

本轮唯一主要影像比较。MSE连续输入，未使用分解项、ER均值或事后解释变量。区间包含0时，写“未检测到明确的增量区分能力”，不写“特征无效”。

## 第四部分 Extended6与ActivityRate

{delta_table('ACT')}

活动率保持连续，具有支持而无检出时保留算法0；无支持和处理失败保持缺失。建模前无结局分布与支持检查见ACTIVITY_PREFLIGHT_NO_OUTCOME.json，未自动修值。它仍是关键次要探索指标，不升级为真实生理波频率。

## 第五部分 Extended6与双壁幅度

{delta_table('AMP')}

两壁按预设2参数块共同加入，未比较单壁后择优删特征。这里评价预测增量，不对调参后的Ridge系数进行普通Wald/LRT，也不把ΔAUC区间冒充βAmpA=βAmpP=0的参数检验。

## 第六部分 Extended6与ER-CC

{delta_table('CC')}

CC是独立次要解释比较，不与MSE竞争后择优替代。

## 第七部分 探索性多影像联合模型

{delta_table('IMG')}

{delta_table('IMG_ACT')}

两者均标记exploratory_multimodal_model，分别10/11预测参数，只使用临床及全部所需影像完整的病例。无论结果如何均不替代主MSE比较。

## 第八部分 长短视频敏感性

{md(pd.DataFrame(subincrement).round(4))}

分层直接使用整体模型的同一平均OOF预测，没有分别训练或按结果挑来源。短视频约45–48秒，样本少时CI较宽；缺少两个结局类别时明确标为无法计算，不改病例。各模型分层AUC与CI见10_LONG_SHORT_MODEL_PERFORMANCE.csv。

## 第九部分 校准和Brier

{md(perf[['analysis_set','model','Brier','Brier_CI_low','Brier_CI_high','calibration_intercept','calibration_slope','calibration_status']].round(4))}

ΔBrier=增强−基线，负值表示更小预测误差。calibration_intercept是固定斜率1的校准总体偏差；slope来自带自由截距的OOF logit诊断回归。它们只作诊断，不在测试数据上校准原预测后重报性能。本轮给校准点估计，不声称bootstrap了校准区间。概率仅在校准logit计算中作1e-12数值裁剪。

## 第十部分 限制与复核

主汇总先对每次检查10轮OOF概率求均值，再以患者聚类bootstrap2000次得到AUC/Brier及配对差的95%百分位区间。没有把10轮叠成10倍样本；单轮结果、范围见06系列。区间条件于固定OOF概率，不是重新拟合完整嵌套流程的区间，未覆盖全部训练与选惩罚不确定性。

{md(stab.round(4))}

所有插补、均值、SD在inner训练内重新拟合，outer refit只用outer training；二元阶段用训练众数、保持0/1，临床连续量用训练中位数，影像不插补。所有模型内层以负log-loss选同一个预设C网格，不能按outer AUC改网格。系数、C边界及预处理核查均保存。

患者重复、样本有限、支持相关缺失、测量误差及来源异质性仍存在。Ridge和拆块不能使Riley样本量限制消失。少量临床缺失处理结果不能替代缺失机制研究。已知一次目录/临床标签冲突沿用最终临床表，旧标签保留，未按本次AUC决定标签。短视频既往已做探索，现合入开发；本轮是按新冻结方案完成的探索性内部验证增量分析，不是全新外部验证，更不是可部署的绝对风险模型证明。

无单变量筛选、无LASSO/SVM/RF等竞争、无检测器调参、无按AUC删患者或改长短来源。全部正负结果保留。CHECKS_FINAL.json记录源码来源未变、患者隔离、配对和10轮汇总检查。
'''
    (root/'AUC_FIRST_PASS_REPORT.md').write_text(report,encoding='utf-8')
    deliver=['05_INCREMENTAL_PERFORMANCE.csv','04_MODEL_PERFORMANCE.csv','AUC_FIRST_PASS_REPORT.md']
    with zipfile.ZipFile(root/'第一轮内部验证_三份审查材料.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in deliver:z.write(root/f,arcname=f)
    save(root/'progress.json',dict(stage='completed',analysis_sets=len(summary),model_performance_rows=len(perf),increment_rows=len(inc)))
    print(inc[['analysis_set','n','n_patient','Delta_AUC','Delta_AUC_CI_low','Delta_AUC_CI_high','Delta_Brier']].round(4).to_string(index=False),flush=True)

if __name__=='__main__':
    import sys
    summarize(Path(sys.argv[1]))
