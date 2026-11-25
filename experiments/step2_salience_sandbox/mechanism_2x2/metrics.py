"""Frozen truth matching; separate formation, fixed-scorer and phase diagnostics."""
from collections import Counter
import numpy as np
from analysis.mse_pregnancy_association.code.patient_activity_aggregation import candidate_nodes
from analysis.mse_pregnancy_association.code.activity_counting import count_activities
from ..matching import activity_predictions,match_events
from ..core import BaselineMinMaxScorer,continuous_branch,evaluate_threshold
from ..scorers import OutOfBandMADScorer
from ..diagnostic.attribution_rules import relation,distribution,consecutive_count
from ..diagnostic.provenance import phase_kind
from ..diagnostic.formation import reference_phases,ridge_main_support
from ..diagnostic.waveforms import truth_component
from .mechanisms import filter_only,partial,strength

SCORERS={'S1':('baseline_minmax',.6),'S2':('mad_oob',1.2065533477611825)}
COUNT='significant_propagation_activity_count'

def outcome(record,result,nodes,layer):
    shared=record['shared'];truth=record['truth'];predictions=[p for p in activity_predictions(result) if p['propagation_supported']]
    match=match_events(truth,predictions,shared['span'],shared['exposure']['seconds'])
    by_node={n['node_id']:n for n in nodes};activity={a['activity_id']:a for a in result['activities']}
    for p in predictions:
        group=[by_node[n] for n in activity[p['prediction_id']]['source_nodes']]
        p['walls']=sorted({n['wall'] for n in group})
        kinds={phase_kind(shared['walls'][n['wall']]['reconciliation']['source_evidence'][n['canonical_source']]) for n in group}
        p['phase_origin']=next(iter(kinds)) if len(kinds)==1 else 'mixed'
        p['stable_identity']=sorted((n['wall'],n['canonical_source']) for n in group)
        p['relations']=[relation(p,t,shared['span']) for t in truth]
    actual={(e['prediction_id'],e['truth_id']) for e in match['allowed_edges']}
    assert actual=={(p['prediction_id'],r['truth_id']) for p in predictions for r in p['relations'] if r['matcher_eligible']}
    by_id={p['prediction_id']:p for p in predictions};tp={e['truth_id']:by_id[e['prediction_id']] for e in match['matches']}
    extras=[]
    for p in predictions:
        if p['prediction_id'] not in match['unmatched_prediction_ids'] or not truth:continue
        eligible=[r for r in p['relations'] if r['matcher_eligible']]
        if eligible:
            assert all(r['truth_id'] in tp for r in eligible)
            subtype='A1' if any(not set(p['walls'])&set(tp[r['truth_id']]['walls']) for r in eligible) else 'A2'
        else:subtype='other_unmatched'
        extras.append(dict(p,subtype=subtype))
    truth_rows=[];matched_truth={e['truth_id'] for e in match['matches']}
    for t in truth:
        truth_rows.append(dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],scenario=record['scenario']['scenario'],
            layer=layer,truth_id=t['truth_event_id'],amplitude_class=t['truth_amplitude_class'],matched=t['truth_event_id'] in matched_truth,
            matcher_eligible_count=sum(any(r['truth_id']==t['truth_event_id'] and r['matcher_eligible'] for r in p['relations']) for p in predictions)))
    low=result.get(COUNT+'_min');high=result.get(COUNT+'_max');n=len(truth)
    coverage=bool(low<=n<=high) if low is not None and high is not None else None
    distance=float(max(low-n,n-high,0)) if low is not None and high is not None else None
    return dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],scenario=record['scenario'],layer=layer,
        input_hash=record['input_hash'],truth_count=n,truth=truth,observable_seconds=shared['exposure']['seconds'],
        TP=match['TP'],FP=match['FP'],FN=match['FN'],precision=match['Precision'],recall=match['Recall'],F1=match['F1'],
        A2=sum(p['subtype']=='A2' for p in extras),paired_phase_A2=sum(p['subtype']=='A2' and p['phase_origin']=='paired_phase' for p in extras),
        A1=sum(p['subtype']=='A1' for p in extras),positive_unmatched=len(extras),
        count_min=low,count_max=high,count_interval_coverage=coverage,distance_to_interval=distance,
        predictions=predictions,unmatched=extras,truth_rows=truth_rows,match=match)

def formation_result(record):
    shared=record['shared'];walls=[]
    for wall in shared['walls']:
        walls.append(dict(wall=wall['wall'],scores=[dict(event_proposal_id=e['event_proposal_id'],decision='salient_support',
            retained_sections=e['supporting_sections']) for e in wall['reconciliation']['event_proposals']]))
    nodes=candidate_nodes(dict(walls=walls),dict(walls=[dict(reconciliation=w['reconciliation']) for w in shared['walls']]))
    if shared['observable']:
        result=count_activities(nodes,shared['s'],shared['span'],shared['dt'],shared['exposure']['seconds'])
    else:result=dict(activities=[],**{COUNT+suffix:None for suffix in ('_min','_max')})
    return outcome(record,result,nodes,'formation_no_salience')

def fixed_results(record):
    shared=record['shared'];rows={};evaluations={}
    scorers={'S1':BaselineMinMaxScorer(shared),'S2':OutOfBandMADScorer(shared,(10,20))}
    for name,scorer in scorers.items():
        branch=continuous_branch(shared,scorer);ev=evaluate_threshold(shared,branch,SCORERS[name][1])
        evaluations[name]=ev;rows[name]=outcome(record,ev['result'],ev['nodes'],name)
    return rows,evaluations,scorers['S2'].scales

def unique_negative(rows):
    groups={}
    for row in rows:
        if row['truth_count']:continue
        if row['input_hash'] in groups:
            other=groups[row['input_hash']]
            assert all(row[k]==other[k] for k in ('TP','FP','FN','count_min','count_max','observable_seconds'))
        else:groups[row['input_hash']]=row
    return list(groups.values())

def aggregate(rows):
    TP=sum(r['TP'] for r in rows);FP=sum(r['FP'] for r in rows);FN=sum(r['FN'] for r in rows)
    precision=TP/(TP+FP) if TP+FP else None;recall=TP/(TP+FN) if TP+FN else None
    controls=unique_negative(rows);minutes=sum(r['observable_seconds'] for r in controls)/60
    low=sum(r['count_min'] for r in controls) if all(r['count_min'] is not None for r in controls) else None
    high=sum(r['count_max'] for r in controls) if all(r['count_max'] is not None for r in controls) else None
    truth_rows=[t for r in rows for t in r['truth_rows']]
    return dict(cases=len(rows),truth_count=TP+FN,propagation_predictions=TP+FP,predictions_per_truth=(TP+FP)/(TP+FN) if TP+FN else None,
        duplicate_FP=sum(r['A1']+r['A2'] for r in rows),TP=TP,FP=FP,FN=FN,precision=precision,recall=recall,F1=2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else None,
        A2=sum(r['A2'] for r in rows),paired_phase_A2=sum(r['paired_phase_A2'] for r in rows),A1=sum(r['A1'] for r in rows),
        positive_unmatched=sum(r['positive_unmatched'] for r in rows),
        matcher_eligible_count=distribution([t['matcher_eligible_count'] for t in truth_rows]),
        strata={label:dict(truth=sum(t['amplitude_class']==label for t in truth_rows),TP=sum(t['amplitude_class']==label and t['matched'] for t in truth_rows),
            recall=sum(t['amplitude_class']==label and t['matched'] for t in truth_rows)/sum(t['amplitude_class']==label for t in truth_rows)
            if any(t['amplitude_class']==label for t in truth_rows) else None) for label in ('clear','medium','structured_weak')},
        negative=dict(raw_rows=sum(not r['truth_count'] for r in rows),unique_inputs=len(controls),minutes=minutes,FP_min=low,FP_max=high,
            FP_per_min_min=low/minutes if low is not None and minutes else None,FP_per_min_max=high/minutes if high is not None and minutes else None,
            ledger_FP=sum(r['FP'] for r in controls)),
        count_interval_coverage=sum(r['count_interval_coverage'] is True for r in rows)/len(rows) if rows else None,
        count_interval_missing=sum(r['count_interval_coverage'] is None for r in rows),
        distance_to_interval_mean=float(np.mean([r['distance_to_interval'] for r in rows if r['distance_to_interval'] is not None])) if rows else None)

def phase_diagnostics(record,model,logs):
    """Truth is used only AFTER the experimental algorithm has returned."""
    shared=record['shared'];components={}
    for truth in record['truth']:
        raw,_,arrival=truth_component(record,truth)
        with filter_only(model):filtered=strength.filtered_runs(raw,shared['valid'],shared['t'])
        components[truth['truth_event_id']]=(raw,filtered,arrival)
    pattern_rows=[];cross=[];filter_rows=[];ringing_pairs=[]
    for log in logs:
        wall=log['wall'];ridges=log['ridges'];by_id={r['id']:r for r in ridges};edges=log['edges']
        selected={rid:e for e in edges if e['selected'] for rid in e['ridge_key']}
        refs={t['truth_event_id']:reference_phases(record,t,components[t['truth_event_id']],wall) for t in record['truth']}
        def labels(rid,ids):
            return [tid for tid,r in refs.items() if consecutive_count(set(ids)&set(ridge_main_support(by_id[rid],r,shared['dt'])))>=3]
        for truth in record['truth']:
            tid=truth['truth_event_id'];main=[]
            for edge in edges:
                a,b=edge['phase_ridge_ids']
                if by_id[a]['sign']!=1:continue
                if tid in labels(a,edge['sections']) and tid in labels(b,edge['sections']):main.append(edge)
            four=False
            for edge in main:
                a,b=edge['phase_ridge_ids'];ea=selected.get(a);eb=selected.get(b)
                if edge['selected'] or ea is None or eb is None or ea['ridge_key']==eb['ridge_key']:continue
                if by_id[ea['phase_ridge_ids'][0]]['sign']==by_id[eb['phase_ridge_ids'][0]]['sign']==-1:four=True
            pattern_rows.append(dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],model=model,wall=wall,truth_id=tid,
                main_pair_available=bool(main),central_main_pair_selected=any(e['selected'] for e in main),four_lobe_flanking_pairs=four,
                main_edges=[{k:e[k] for k in ('phase_ridge_ids','W_old','G','W_new','W_used','selected')} for e in main]))
        for edge in edges:
            if not edge['selected']:continue
            a,b=edge['phase_ridge_ids'];la=labels(a,edge['sections']);lb=labels(b,edge['sections'])
            cross.append(dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],model=model,wall=wall,
                phase_ridge_ids=edge['phase_ridge_ids'],phase_i_truth_labels=la,phase_j_truth_labels=lb,
                cross_truth_pairing_error=len(la)==len(lb)==1 and la!=lb,
                possible_cross_truth=any(x!=y for x in la for y in lb),
                unassigned_phase=not la or not lb))
        for truth in record['truth']:
            tid=truth['truth_event_id'];raw,filtered,arrival=components[tid]
            with filter_only(model):_,columns,_=partial.observed_lobes(raw[:,wall],shared['valid'][:,wall],shared['t'],
                np.broadcast_to(shared['noise_floor'],(len(shared['s']),)))
            ratios=[];lobes=peripheral=0
            for c,col in enumerate(columns):
                for node in col:
                    lobes+=1;ref=refs[tid][c,node['sign']]['filtered_main_peak_s']
                    is_main=ref is not None and node['start_s']-shared['dt']/2<=ref<=node['end_s']+shared['dt']/2
                    if not is_main:
                        peripheral+=1
                        index=int(np.argmin(abs(shared['t']-ref))) if ref is not None else None
                        amplitude=abs(filtered[index,wall,c]) if index is not None else None
                        if amplitude:ratios.append(node['amplitude']/amplitude)
            filter_rows.append(dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],model=model,wall=wall,truth_id=tid,
                truth_component_lobe_count=lobes,peripheral_lobe_count=peripheral,peripheral_main_amplitude_ratios=ratios))
        for edge in edges:
            if not edge['selected'] or not components:continue
            support=[]
            for rid in edge['ridge_key']:
                for node in by_id[rid]['nodes']:
                    if node['column'] not in edge['sections']:continue
                    c=node['column'];index=node['peak_index']
                    if not all(comp[0][index,wall,c]==0 for comp in components.values()):continue
                    if any(comp[1][index,wall,c]*node['sign']>0 for comp in components.values()):support.append(c)
            if consecutive_count(support)>=3:ringing_pairs.append(dict(case_id=record['case_id'],model=model,wall=wall,phase_ridge_ids=edge['phase_ridge_ids']))
    return dict(pattern_rows=pattern_rows,cross_truth=cross,filter_rows=filter_rows,ringing_like_selected_pairs=ringing_pairs)
