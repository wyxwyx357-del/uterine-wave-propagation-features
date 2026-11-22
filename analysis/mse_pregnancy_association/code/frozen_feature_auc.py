"""Prespecified grouped-CV AUC of frozen amplitudes, without feature retuning.

Clinical outcomes are authorized only for this separate analysis. Patient data,
predictions and all model outputs remain local. No window is an independent
patient, and no new patient-level ER/coordination average is constructed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLINICAL = ['年龄','BMI','不孕年限_年','移植胚胎数_多胚胎','内膜厚度_mm','胚胎种类_含囊胚','内膜分型_C型']
AMPLITUDE = ['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude']
MASK = ['anterior_amplitude_missing','posterior_amplitude_missing']
COORD = ['single_block_CC','single_block_MSE']
OUTCOME = '结局_临床妊娠'
GROUP = '患者聚类标识'
EXAM = '临床检查编号'
VIDEO = '视频研究编号'
SEED = 20260531


def save(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model():
    # Same imputation/scaling and almost-unpenalized clinical Logistic as the
    # existing baseline. No hyperparameter or variable selection in this trial.
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
        StandardScaler(),LogisticRegression(C=1e6,penalty='l2',solver='liblinear',
                                           max_iter=5000,random_state=SEED))


def assemble(args):
    features_path = args.features/'patient_results.csv'
    features = pd.read_csv(features_path)
    mapping = pd.read_excel(args.cohort_map,usecols=[EXAM,VIDEO,'视频完整路径'])
    mapping['case_id'] = mapping['视频完整路径'].map(lambda p:Path(str(p)).parent.name)
    clinical = pd.read_excel(args.clinical,sheet_name='建模输入',
                            usecols=[EXAM,VIDEO,GROUP,OUTCOME]+CLINICAL)
    if clinical[EXAM].duplicated().any() or mapping['case_id'].duplicated().any():
        raise ValueError('Exam/clip mapping is not one-to-one; do not guess a label')
    joined = features.merge(mapping[['case_id',EXAM,VIDEO]],on='case_id',how='left',validate='one_to_one')
    joined = joined.merge(clinical,on=EXAM,how='left',suffixes=('_map','_clinical'),
                          validate='many_to_one',indicator=True)
    reasons = []
    for _, row in joined.iterrows():
        bad=[]
        if row['_merge']!='both':bad.append('clinical_exam_not_matched')
        if row['_merge']=='both' and (pd.isna(row[VIDEO+'_map']) or pd.isna(row[VIDEO+'_clinical']) or
            float(row[VIDEO+'_map'])!=float(row[VIDEO+'_clinical'])):bad.append('video_identifier_conflict')
        if row.get(OUTCOME) not in (0,1):bad.append('outcome_not_binary')
        if pd.isna(row.get(GROUP)) or not str(row.get(GROUP)).strip():bad.append('patient_group_missing')
        reasons.append(';'.join(bad))
    joined['mapping_exclusion_reason'] = reasons
    joined.to_csv(args.output/'clinical_matching_audit.csv',index=False,encoding='utf-8-sig')
    data = joined.loc[joined['mapping_exclusion_reason'].eq('')].copy().reset_index(drop=True)
    if data[EXAM].duplicated().any():
        raise ValueError('Multiple clips map to one examination; no frozen clip aggregation exists')
    data[GROUP] = data[GROUP].astype(str).str.strip()
    data[OUTCOME] = data[OUTCOME].astype(int)
    for c in CLINICAL+AMPLITUDE:
        data[c] = pd.to_numeric(data[c],errors='coerce')
    for c, name in zip(AMPLITUDE,MASK):data[name]=data[c].isna().astype(float)
    data[COORD[0]]=np.nan;data[COORD[1]]=np.nan
    hashes={str(features_path.resolve()):digest(features_path),str(args.clinical.resolve()):digest(args.clinical),
            str(args.cohort_map.resolve()):digest(args.cohort_map)}
    block_counts=[]
    for i,row in data.iterrows():
        path=args.features/row['case_id']/'features.json'
        hashes[str(path.resolve())]=digest(path)
        measured=json.loads(path.read_text(encoding='utf-8'))
        blocks=measured['modes']['coordination'];block_counts.append(len(blocks))
        # Reuse an already existing scalar ONLY if there is exactly one block.
        # Multi-block patients are not averaged or selected for outcome fit.
        if len(blocks)==1:
            data.loc[i,COORD[0]]=blocks[0]['zero_lag_ER_Pearson_CC']
            data.loc[i,COORD[1]]=blocks[0]['ER_MSE']
    data['coordination_block_count_original']=block_counts
    data.to_csv(args.output/'analysis_cohort.csv',index=False,encoding='utf-8-sig')
    return data,hashes,len(joined)


def grouped_oof(data, models, folds, seed):
    y=data[OUTCOME].to_numpy(int);groups=data[GROUP].to_numpy(str)
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    predictions={name:np.full(len(data),np.nan) for name in models}
    assignment=np.full(len(data),-1,int);audit=[]
    for fold,(train,test) in enumerate(splitter.split(np.zeros(len(data)),y,groups),1):
        assert not set(groups[train])&set(groups[test])
        assert (assignment[test]==-1).all()
        assignment[test]=fold
        row=dict(fold=fold,train_exams=len(train),test_exams=len(test),
                 train_patients=len(set(groups[train])),test_patients=len(set(groups[test])),
                 test_positive=int(y[test].sum()),patient_overlap=0)
        for name, columns in models.items():
            model=make_model().fit(data.iloc[train][columns],y[train])
            p=model.predict_proba(data.iloc[test][columns])[:,1]
            predictions[name][test]=p
            row[name+'_AUC']=float(roc_auc_score(y[test],p)) if len(np.unique(y[test]))==2 else None
        audit.append(row)
    assert (assignment>0).all() and all(np.isfinite(p).all() for p in predictions.values())
    return predictions,assignment,audit


def bootstrap_oof(data,predictions,iterations,seed):
    y=data[OUTCOME].to_numpy(int)
    groups,inverse=np.unique(data[GROUP].to_numpy(str),return_inverse=True)
    rng=np.random.default_rng(seed)
    draws={k:[] for k in predictions}
    for _ in range(iterations):
        counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        weights=counts[inverse]
        if not weights[y==0].sum() or not weights[y==1].sum():continue
        for name,p in predictions.items():draws[name].append(float(roc_auc_score(y,p,sample_weight=weights)))
    return {k:np.asarray(v) for k,v in draws.items()}


def evaluate(name,data,models,args):
    folder=args.output/name;folder.mkdir(exist_ok=True)
    all_metrics=[];primary=None;primary_folds=None
    for repeat in range(args.repeats):
        predictions,folds,audit=grouped_oof(data,models,args.folds,SEED+repeat)
        if repeat==0:
            primary=predictions;primary_folds=folds
            pd.DataFrame(audit).to_csv(folder/'fold_audit.csv',index=False,encoding='utf-8-sig')
        for model,p in predictions.items():
            all_metrics.append(dict(repeat=repeat,seed=SEED+repeat,model=model,
                AUC=float(roc_auc_score(data[OUTCOME],p)),Brier=float(brier_score_loss(data[OUTCOME],p))))
    repeated=pd.DataFrame(all_metrics)
    repeated.to_csv(folder/'repeat_metrics.csv',index=False,encoding='utf-8-sig')
    oof=data[['case_id',EXAM,GROUP,OUTCOME]].copy();oof['fold']=primary_folds
    for model,p in primary.items():oof[model]=p
    oof.to_csv(folder/'out_of_fold_predictions.csv',index=False,encoding='utf-8-sig')
    draws=bootstrap_oof(data,primary,args.bootstrap,SEED+1000)
    metrics=[]
    for model,p in primary.items():
        values=repeated.loc[repeated['model']==model,'AUC']
        low,high=np.quantile(draws[model],[.025,.975])
        metrics.append(dict(analysis=name,model=model,n=len(data),patients=data[GROUP].nunique(),
            positive=int(data[OUTCOME].sum()),negative=int((1-data[OUTCOME]).sum()),
            primary_OOF_AUC=float(roc_auc_score(data[OUTCOME],p)),CI_low=float(low),CI_high=float(high),
            repeat_mean_AUC=float(values.mean()),repeat_min_AUC=float(values.min()),repeat_max_AUC=float(values.max())))
    differences=[]
    baseline='clinical'
    pairs=[(baseline,k) for k in models if k not in ('clinical','amplitude','coordination')]
    if 'clinical_amp_missing' in models:pairs.append(('clinical_missing','clinical_amp_missing'))
    if 'clinical_amp_coordination' in models:pairs.append(('clinical_amp','clinical_amp_coordination'))
    for a,b in pairs:
        dist=draws[b]-draws[a]
        low,high=np.quantile(dist,[.025,.975])
        differences.append(dict(analysis=name,comparison=b+' minus '+a,n=len(data),
            delta_AUC=float(roc_auc_score(data[OUTCOME],primary[b])-roc_auc_score(data[OUTCOME],primary[a])),
            CI_low=float(low),CI_high=float(high),
            repeat_mean_delta_AUC=float(repeated.loc[repeated['model']==b,'AUC'].mean()-repeated.loc[repeated['model']==a,'AUC'].mean())))
    pd.DataFrame(metrics).to_csv(folder/'AUC_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(differences).to_csv(folder/'paired_AUC_differences.csv',index=False,encoding='utf-8-sig')
    return metrics,differences,primary


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(primary='amplitude_main',clinical_features=CLINICAL,amplitudes=AMPLITUDE,
        model='same clinical baseline Logistic C=1e6 liblinear; no model/feature selection',
        outcome='reviewed clinical pregnancy from final 建模输入 sheet; not video directory label',
        patient_group=GROUP,folds=args.folds,repeats=args.repeats,first_seed=SEED,bootstrap=args.bootstrap,
        preprocessing='median imputation and standard scaling fitted only in each training fold',
        uncertainty='patient-cluster bootstrap of fixed primary OOF predictions; no refit in bootstrap; split sensitivity separately reported',
        missingness_controls=True,complete_amplitude_sensitivity=True,
        coordination_secondary='exactly one existing block with finite CC/MSE; no block mean or best-block choice',
        excluded_window_features='ER, directional RMS and energy fraction have no frozen patient aggregation and are not pooled',
        speed_or_events=False,feature_algorithms_changed=False,
        python=platform.python_version(),sklearn=sklearn.__version__,pandas=pd.__version__,
        references=['https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html',
                    'https://scikit-learn.org/stable/common_pitfalls.html'])
    save(args.output/'protocol_before_fit.json',protocol)
    data,hashes,input_rows=assemble(args)
    all_models=dict(clinical=CLINICAL,amplitude=AMPLITUDE,clinical_amp=CLINICAL+AMPLITUDE,
                    clinical_missing=CLINICAL+MASK,clinical_amp_missing=CLINICAL+AMPLITUDE+MASK)
    metrics,deltas,predictions=evaluate('amplitude_main',data,all_models,args)
    complete=data.loc[data[AMPLITUDE].notna().all(axis=1)].reset_index(drop=True)
    m,d,_=evaluate('amplitude_complete_cases',complete,
                   dict(clinical=CLINICAL,amplitude=AMPLITUDE,clinical_amp=CLINICAL+AMPLITUDE),args)
    metrics+=m;deltas+=d
    single=data.loc[data[COORD].notna().all(axis=1)].reset_index(drop=True)
    m,d,_=evaluate('coordination_single_block_exploratory',single,
        dict(clinical=CLINICAL,amplitude=AMPLITUDE,coordination=COORD,clinical_amp=CLINICAL+AMPLITUDE,
             clinical_amp_coordination=CLINICAL+AMPLITUDE+COORD),args)
    metrics+=m;deltas+=d
    assert all(digest(Path(p))==h for p,h in hashes.items())
    metrics_df=pd.DataFrame(metrics);deltas_df=pd.DataFrame(deltas)
    metrics_df.to_csv(args.output/'all_AUC_metrics.csv',index=False,encoding='utf-8-sig')
    deltas_df.to_csv(args.output/'all_paired_differences.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(args.output/'冻结幅度_AUC测试.xlsx',engine='openpyxl') as writer:
        metrics_df.to_excel(writer,sheet_name='AUC',index=False)
        deltas_df.to_excel(writer,sheet_name='配对增益',index=False)
        data[['case_id',EXAM,GROUP,OUTCOME]+CLINICAL+AMPLITUDE+MASK+COORD].to_excel(writer,sheet_name='分析队列',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for model in ('clinical','amplitude','clinical_amp'):
        fpr,tpr,_=roc_curve(data[OUTCOME],predictions[model])
        ax.plot(fpr,tpr,label=f"{model}: {roc_auc_score(data[OUTCOME],predictions[model]):.3f}")
    ax.plot([0,1],[0,1],'--',color='gray',linewidth=.8)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'Grouped {args.folds}-fold out-of-fold ROC')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(args.output/'OOF_ROC.png',dpi=180);plt.close(fig)
    save(args.output/'summary.json',dict(input_exams=input_rows,matched_exams=len(data),patients=int(data[GROUP].nunique()),
        positive=int(data[OUTCOME].sum()),negative=int((1-data[OUTCOME]).sum()),
        complete_amplitude_exams=len(complete),single_coordination_exams=len(single),
        primary_metrics=[r for r in metrics if r['analysis']=='amplitude_main'],
        primary_differences=[r for r in deltas if r['analysis']=='amplitude_main'],
        source_SHA256_unchanged=hashes,internal_validation_only=True))
    lines=['# 冻结特征的临床妊娠AUC测试','',
        f'{input_rows}例来源中匹配{len(data)}次检查、{data[GROUP].nunique()}名患者；阳性{int(data[OUTCOME].sum())}，阴性{int((1-data[OUTCOME]).sum())}。',
        '主要分析使用已冻结的前、后壁径向伸缩幅度。临床基线沿用既有7变量及Logistic参数，在相同患者、相同分组折上重新拟合。',
        f'{args.folds}折按患者分组，{args.repeats}个预设种子重复，不选最高分；插补、标准化仅在训练折拟合。主结果为第一个预设种子的折外AUC，另列重复均值/范围。',
        f'区间为{args.bootstrap}次患者聚类bootstrap的折外预测区间；每次bootstrap不重训模型，未覆盖全部训练不确定性。',
        '', f'| 模型 | 折外AUC | 95%区间 | {args.repeats}次重复均值 |','|---|---:|---|---:|']
    for r in metrics:
        if r['analysis']=='amplitude_main':lines.append(f"|{r['model']}|{r['primary_OOF_AUC']:.4f}|{r['CI_low']:.4f}–{r['CI_high']:.4f}|{r['repeat_mean_AUC']:.4f}|")
    lines += ['', '## 配对增益']
    for r in deltas:
        if r['analysis']=='amplitude_main':lines.append(f"- {r['comparison']}: ΔAUC={r['delta_AUC']:+.4f}，95%区间{r['CI_low']:+.4f}至{r['CI_high']:+.4f}。")
    lines += ['',f'双壁幅度均可用的{len(complete)}例另作完整病例敏感性分析；不据此排除主分析中的缺失者。',
        f'协调性仅在{len(single)}例恰好一个既有区块、CC/MSE数值明确者作探索性对照。同一子集内重建基线比较，不能与主队列结果跨样本比增益。',
        '没有把多窗口ER、RMS或energy fraction自行平均为新患者特征，也没有平均多块CC/MSE。无自动次数/速度或原始谱能量主变量。',
        '缺失指示对照用于区分活动值与可分析性信息。未调0.6、追踪、ROI、频带或任何冻结特征；未按结局改代码参数。',
        '数据已参与开发与复核，本次仅为内部探索性判别能力测试，不是外部验证。',
        '', '方法参考：', *protocol['references']]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(metrics_df[['analysis','model','n','primary_OOF_AUC','repeat_mean_AUC']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--cohort-map',type=Path,required=True)
    parser.add_argument('--clinical',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--bootstrap',type=int,default=1000)
    run(parser.parse_args())
