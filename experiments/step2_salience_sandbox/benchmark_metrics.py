"""Paired benchmark metrics; count intervals remain intervals."""
import numpy as np
from .core import retain,evaluate_threshold
from .matching import upstream_predictions,activity_predictions,match_events,upstream_recall


def selection(shared,branch,tau):
    upstream={p['prediction_id']:p for p in upstream_predictions(shared)}
    result=[];signature=[]
    for wall in branch['walls']:
        for proposal in wall['proposals']:
            kept=retain(proposal['event'],proposal['sections'],tau)
            if not kept['retained_sections']:continue
            key=f"wall{wall['wall']}/"+proposal['event']['event_proposal_id']
            result.append(dict(upstream[key],sections=kept['retained_sections']))
            signature.append((key,tuple(kept['retained_sections'])))
    return result,tuple(signature)


def interval_error(low,high,truth):
    if low is None or high is None:return dict(coverage=None,distance=None,midpoint_error=None,width=None)
    return dict(coverage=bool(low<=truth<=high),distance=float(max(low-truth,truth-high,0)),
                midpoint_error=float(abs((low+high)/2-truth)),width=float(high-low))


def evaluate(record,config,tau,force=False):
    shared=record['shared'];branch=record['branches'][config];truth=record['truth']
    salience,signature=selection(shared,branch,tau)
    if force or signature not in record['downstream_cache']:
        outcome=evaluate_threshold(shared,branch,tau)['result']
        record['downstream_cache'][signature]=outcome
    else:outcome=record['downstream_cache'][signature]
    duration=shared['exposure']['seconds']
    predicted=activity_predictions(outcome)
    final=match_events(truth,predicted,shared['span'],duration)
    salience_match=match_events(truth,salience,shared['span'],duration)
    matched={m['truth_id'] for m in final['matches']};kept={m['truth_id'] for m in salience_match['matches']}
    represented=set(record['upstream']['upstream_represented_truth_ids'])
    strata={}
    for label in ('clear','medium','structured_weak'):
        ids={t['truth_event_id'] for t in truth if t['truth_amplitude_class']==label}
        strata[label]=dict(truth=len(ids),upstream=len(ids&represented),salience=len(ids&represented&kept),final=len(ids&matched))
    directions=dict(C2F_total=0,C2F_correct=0,F2C_total=0,F2C_correct=0,unresolved=0)
    lookup={p['prediction_id']:p for p in predicted};tlookup={t['truth_event_id']:t for t in truth}
    for edge in final['matches']:
        a=tlookup[edge['truth_id']]['truth_direction'];b=lookup[edge['prediction_id']]['direction']
        directions[a+'_total']+=1;directions[a+'_correct']+=int(a==b);directions['unresolved']+=int(b not in ('C2F','F2C'))
    low=outcome.get('significant_propagation_activity_count_min');high=outcome.get('significant_propagation_activity_count_max')
    return dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],scenario=record['scenario'],
        config=config,threshold=float(tau),pool_hash=shared['pool_hash'],observable_seconds=duration,
        truth=truth,truth_count=len(truth),upstream=record['upstream'],strata=strata,
        matched_truth_ids=sorted(matched),salience_matched_truth_ids=sorted(kept),
        core_detected='core' in matched,core_salient='core' in kept,
        TP=final['TP'],FP=final['FP'],FN=final['FN'],direction_counts=directions,
        count_min=low,count_max=high,count_exact=outcome.get('significant_propagation_activity_count_exact'),
        frequency_min=outcome.get('significant_propagation_activity_frequency_cpm_min'),
        frequency_max=outcome.get('significant_propagation_activity_frequency_cpm_max'),
        count_interval_error=interval_error(low,high,len(truth)),predictions=predicted,matches=final['matches'])


def ratio(a,b):return a/b if b else None
def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def unique_controls(rows):
    unique={}
    for row in rows:
        if not row['truth_count']:unique.setdefault((row['texture_id'],row['pool_hash']),row)
    return list(unique.values())


def aggregate(rows):
    tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows)
    direction={k:sum(r['direction_counts'][k] for r in rows) for k in ('C2F_total','C2F_correct','F2C_total','F2C_correct','unresolved')}
    matched=direction['C2F_total']+direction['F2C_total']
    strata={}
    for label in ('clear','medium','structured_weak'):
        counts={k:sum(r['strata'][label][k] for r in rows) for k in ('truth','upstream','salience','final')}
        strata[label]=dict(**counts,upstream_recall=ratio(counts['upstream'],counts['truth']),
                          conditional_salience_recall=ratio(counts['salience'],counts['upstream']),
                          end_to_end_recall=ratio(counts['final'],counts['truth']))
    controls=unique_controls(rows);minutes=sum(r['observable_seconds'] for r in controls)/60
    control_fp=sum(r['FP'] for r in controls)
    intervals=[r['count_interval_error'] for r in rows]
    return dict(samples=len(rows),truth_events=sum(r['truth_count'] for r in rows),TP=tp,FP=fp,FN=fn,
        Precision=ratio(tp,tp+fp),Recall=ratio(tp,tp+fn),F1=ratio(2*tp,2*tp+fp+fn),
        total_observation_minutes=sum(r['observable_seconds'] for r in rows)/60,
        FP_per_min=ratio(fp,sum(r['observable_seconds'] for r in rows)/60),
        safety=dict(unique_controls=len(controls),FP=control_fp,minutes=minutes,FP_per_min=ratio(control_fp,minutes),
                    minimum_nonzero_FP_per_min=ratio(1,minutes),positive_case_fraction=ratio(sum(r['FP']>0 for r in controls),len(controls))),
        strata=strata,interval_coverage=mean([float(x['coverage']) if x['coverage'] is not None else None for x in intervals]),
        distance_to_interval=mean([x['distance'] for x in intervals]),midpoint_MAE_auxiliary=mean([x['midpoint_error'] for x in intervals]),
        count_unavailable_cases=sum(x['coverage'] is None for x in intervals),interval_width=mean([x['width'] for x in intervals]),
        direction=dict(overall_accuracy=ratio(direction['C2F_correct']+direction['F2C_correct'],matched),
                       C2F_accuracy=ratio(direction['C2F_correct'],direction['C2F_total']),
                       F2C_accuracy=ratio(direction['F2C_correct'],direction['F2C_total']),
                       unresolved_rate=ratio(direction['unresolved'],matched),counts=direction))


def detailed_summary(rows):
    result=aggregate(rows)
    positives=[r for r in rows if r['truth_count'] and all(t['truth_amplitude_class'] in ('clear','medium') for t in r['truth'])]
    result['clear_medium']=aggregate(positives)
    result['zero']=aggregate([r for r in rows if r['scenario']['family']=='zero'])
    result['artifact_negative']=aggregate([r for r in rows if r['scenario']['family']=='artifact' and not r['truth_count']])
    lookup={(r['texture_id'],r['scenario']['scenario']):r for r in rows}
    coexist=[];durations=[];artifacts=[]
    for row in rows:
        scenario=row['scenario'];source=row['texture_id']
        if scenario['family']=='coexist':
            label='medium' if scenario['amplitude']==.04 else 'structured_weak'
            alone=lookup[(source,label+'_C2F')]
            coexist.append(dict(texture_id=source,amplitude_class=label,ratio=scenario['ratio'],
                Recall_alone=int(alone['core_detected']),Recall_with_strong=int(row['core_detected']),
                delta_recall=int(row['core_detected'])-int(alone['core_detected'])))
        if scenario['family']=='duration':
            anchor=lookup[(source,'medium_C2F')]
            lo=row['count_min'];al=anchor['count_min'];hi=row['count_max'];ah=anchor['count_max']
            durations.append(dict(texture_id=source,duration=scenario['duration'],
                event_agreement=row['core_detected']==anchor['core_detected'],
                event_detected=row['core_detected'],anchor_detected=anchor['core_detected'],
                count_bounds_difference=[lo-al,hi-ah] if None not in (lo,al,hi,ah) else None,
                frequency_difference=[row['frequency_min']-anchor['frequency_min'],row['frequency_max']-anchor['frequency_max']]
                    if None not in (row['frequency_min'],anchor['frequency_min']) else None,
                exposure_corrected_frequency_difference=[row['frequency_min']*row['observable_seconds']/anchor['observable_seconds']-anchor['frequency_min'],
                    row['frequency_max']*row['observable_seconds']/anchor['observable_seconds']-anchor['frequency_max']]
                    if None not in (row['frequency_min'],anchor['frequency_min']) and anchor['observable_seconds']>0 else None,
                frequency_note='Expected exposure dependence: identical count has different cpm at 30/60/90s; not automatically measurement error'))
        if scenario['family']=='artifact' and row['truth_count']:
            anchor=lookup[(source,'medium_C2F')]
            f1=lambda r:ratio(2*r['TP'],2*r['TP']+r['FP']+r['FN'])
            artifacts.append(dict(texture_id=source,artifact=scenario['artifact'],Recall_drop=int(anchor['core_detected'])-int(row['core_detected']),
                F1_drop=(f1(anchor)-f1(row)) if f1(anchor) is not None and f1(row) is not None else None,
                interval_distance_change=row['count_interval_error']['distance']-anchor['count_interval_error']['distance']
                    if row['count_interval_error']['distance'] is not None and anchor['count_interval_error']['distance'] is not None else None))
    result['coexistence']=dict(pairs=coexist,delta_recall=mean([r['delta_recall'] for r in coexist]),
        by_ratio={str(k):mean([r['delta_recall'] for r in coexist if r['ratio']==k]) for k in (2,3,5)},
        by_amplitude={k:mean([r['delta_recall'] for r in coexist if r['amplitude_class']==k]) for k in ('medium','structured_weak')})
    result['duration']=dict(pairs=durations,event_agreement=mean([int(r['event_agreement']) for r in durations]),
                           joint_detection_fraction=mean([int(r['event_detected'] and r['anchor_detected']) for r in durations]))
    result['artifact']=dict(pairs=artifacts,Recall_drop=mean([r['Recall_drop'] for r in artifacts]),
        blob_Recall_drop=mean([r['Recall_drop'] for r in artifacts if r['artifact']=='blob']))
    gradient={}
    for noise in (1,3,8):
        names=['zero_rigid','clear_C2F','medium_C2F','structured_weak_C2F'] if noise==1 else [f'zero_noise{noise}']+[f'gradient_{label}_{noise}' for label in ('clear','medium','structured_weak')]
        gradient[str(noise)]=aggregate([r for r in rows if r['scenario']['scenario'] in names])
    result['noise_gradient']=gradient
    return result


def qualifies(dev,val,base_dev,base_val):
    safety=all(val[group]['safety'][key]<=base_val[group]['safety'][key] for group in ('zero','artifact_negative') for key in ('FP','positive_case_fraction'))
    structured=all(val['strata'][k]['end_to_end_recall']>=base_val['strata'][k]['end_to_end_recall'] for k in ('clear','medium')) and val['clear_medium']['F1']>=base_val['clear_medium']['F1']
    def defects(a,b):
        return dict(coexistence=a['coexistence']['delta_recall']-b['coexistence']['delta_recall'],
                    blob=b['artifact']['blob_Recall_drop']-a['artifact']['blob_Recall_drop'],
                    duration=a['duration']['joint_detection_fraction']-b['duration']['joint_detection_fraction'],
                    high_noise=mean([a['noise_gradient'][k]['F1']-b['noise_gradient'][k]['F1'] for k in ('3','8')]))
    dd=defects(dev,base_dev);vd=defects(val,base_val)
    improved=[k for k in vd if vd[k]>=.10-1e-12 and dd[k]>=.10-1e-12]
    not_sacrificed=all(v>=-.05-1e-12 for v in vd.values()) and val['distance_to_interval']<=base_val['distance_to_interval']+1e-12
    return dict(qualifies=bool(safety and structured and improved and not_sacrificed),safety_pass=bool(safety),
                clear_medium_pass=bool(structured),Dev_defect_differences=dd,Val_defect_differences=vd,
                improved_in_both=improved,no_key_regression=bool(not_sacrificed))
