"""Maximum cardinality, then minimum total absolute center-time difference."""
import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_SHARED_NATIVE_SECTIONS=3  # Explicit infrastructure event-matching rule, user specified.


def match_events(truth,predictions,measurement_span,observation_seconds):
    if not np.isfinite(measurement_span) or measurement_span<=0:
        raise ValueError('Use actual positive measurement span')
    if not np.isfinite(observation_seconds) or observation_seconds<0:
        raise ValueError('Observation seconds must be finite and nonnegative')
    for row in truth:
        times=[row[k] for k in ('truth_start_s','truth_center_s','truth_end_s')]
        if not np.isfinite(times).all() or not times[0]<=times[1]<=times[2]:
            raise ValueError('Expected finite ordered truth support/center')
    for row in predictions:
        times=[row[k] for k in ('evidence_start_s','evidence_end_s')]
        if not np.isfinite(times).all() or times[1]<times[0]:
            raise ValueError('Expected finite ordered prediction support')
    truth=sorted(truth,key=lambda x:x['truth_event_id'])
    predictions=sorted(predictions,key=lambda x:x['prediction_id'])
    if len({x['truth_event_id'] for x in truth})!=len(truth) or len({x['prediction_id'] for x in predictions})!=len(predictions):
        raise ValueError('Truth and prediction IDs must be unique')
    edges=[];pair_evidence=[]
    for i,t in enumerate(truth):
        truth_sections=set(t['truth_sections'])
        for j,p in enumerate(predictions):
            midpoint=(p['evidence_start_s']+p['evidence_end_s'])/2
            prediction_sections=set(p['sections'])
            shared=truth_sections & prediction_sections
            evidence=dict(truth_index=i,prediction_index=j,truth_id=t['truth_event_id'],
                          prediction_id=p['prediction_id'],intersection_count=len(shared),
                          truth_coverage=len(shared)/len(truth_sections) if truth_sections else None,
                          prediction_coverage=len(shared)/len(prediction_sections) if prediction_sections else None,
                          center_deviation_s=abs(midpoint-t['truth_center_s']),
                          time_gate_passed=bool(t['truth_start_s']-measurement_span<=midpoint<=t['truth_end_s']+measurement_span),
                          spatial_gate_passed=len(shared)>=MIN_SHARED_NATIVE_SECTIONS)
            pair_evidence.append(evidence)
            if evidence['time_gate_passed'] and evidence['spatial_gate_passed']:
                edges.append(evidence)
    matched=[]
    if truth:
        maximum=max((e['center_deviation_s'] for e in edges),default=0)
        # Saving one unmatched penalty dominates every allowed real-edge cost combined.
        penalty=(len(truth)+1)*(maximum+1)
        cost=np.full((len(truth),len(predictions)+len(truth)),penalty)
        cost[:,:len(predictions)]=penalty*(len(truth)+2)
        lookup={}
        for e in edges:
            key=e['truth_index'],e['prediction_index']
            cost[key]=e['center_deviation_s'];lookup[key]=e
        rows,cols=linear_sum_assignment(cost)
        matched=[lookup[(int(i),int(j))] for i,j in zip(rows,cols) if (int(i),int(j)) in lookup]
    matched_t={e['truth_id'] for e in matched};matched_p={e['prediction_id'] for e in matched}
    tp=len(matched);fp=len(predictions)-tp;fn=len(truth)-tp
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    directions=[];propagation=[]
    for edge in matched:
        t=truth[edge['truth_index']];p=predictions[edge['prediction_index']]
        true_prop=t.get('truth_propagation_supported',t['truth_direction'] in ('C2F','F2C'))
        propagation.append((bool(true_prop),bool(p['propagation_supported'])))
        if t['truth_direction'] in ('C2F','F2C'):
            directions.append((t['truth_direction'],p['direction']))
    def accuracy(rows):
        return sum(a==b for a,b in rows)/len(rows) if rows else None
    return dict(TP=tp,FP=fp,FN=fn,Precision=precision,Recall=recall,F1=f1,
                FP_per_min=fp*60/observation_seconds if observation_seconds>0 else None,
                pair_evidence=pair_evidence,allowed_edges=edges,matches=matched,
                unmatched_truth_ids=[t['truth_event_id'] for t in truth if t['truth_event_id'] not in matched_t],
                unmatched_prediction_ids=[p['prediction_id'] for p in predictions if p['prediction_id'] not in matched_p],
                direction_accuracy=accuracy(directions),
                C2F_accuracy=accuracy([pair for pair in directions if pair[0]=='C2F']),
                F2C_accuracy=accuracy([pair for pair in directions if pair[0]=='F2C']),
                unresolved_direction_rate=sum(b not in ('C2F','F2C') for _,b in directions)/len(directions) if directions else None,
                propagation_supported_accuracy=accuracy(propagation),
                local_propagation_confusion={f'truth_{int(a)}_prediction_{int(b)}':sum(x==a and y==b for x,y in propagation)
                                             for a in (False,True) for b in (False,True)})


def upstream_predictions(shared):
    result=[]
    for wall in shared['walls']:
        ownership=wall['reconciliation']
        for event in ownership['event_proposals']:
            entity=ownership['source_evidence'][event['canonical_source_entity_id']]
            phases=[p for ps in entity['profile'].values() for p in ps]
            result.append(dict(prediction_id=f"wall{wall['wall']}/"+event['event_proposal_id'],
                               evidence_start_s=min(p['start_s'] for p in phases),
                               evidence_end_s=max(p['end_s'] for p in phases),
                               sections=event['supporting_sections'],direction=event['direction'],
                               propagation_supported=event['direction'] in ('C2F','F2C')))
    return result


def activity_predictions(result):
    rows=[]
    for event in result['activities']:
        evidence=event['block_evidence']
        directions={e['measurement']['classification'] for e in evidence if e['propagation_supported']}
        rows.append(dict(prediction_id=event['activity_id'],evidence_start_s=event['evidence_start_s'],
                         evidence_end_s=event['evidence_end_s'],
                         sections=sorted({c for e in evidence for c in e['sections']}),
                         direction=next(iter(directions)) if len(directions)==1 else None,
                         propagation_supported=event['propagation_supported']))
    return rows


def upstream_recall(shared,truth):
    matched=match_events(truth,upstream_predictions(shared),shared['span'],shared['exposure']['seconds'])
    represented={m['truth_id'] for m in matched['matches']}
    strata={}
    for label in ('clear','medium','structured_weak'):
        selected=[t for t in truth if t['truth_amplitude_class']==label]
        found=sum(t['truth_event_id'] in represented for t in selected)
        strata[label]=dict(truth_event_count=len(selected),truth_events_represented_in_upstream=found,
                           upstream_recall=found/len(selected) if selected else None)
    return dict(truth_event_count=len(truth),truth_events_represented_in_upstream=len(represented),
                upstream_recall=len(represented)/len(truth) if truth else None,
                upstream_missed_truth_ids=matched['unmatched_truth_ids'],
                upstream_represented_truth_ids=sorted(represented),strata=strata)


def miss_attribution(upstream,final_match):
    missed=set(final_match['unmatched_truth_ids'])
    return dict(upstream_missed_truth_ids=upstream['upstream_missed_truth_ids'],
                salience_and_downstream_missed_represented_truth_ids=sorted(missed & set(upstream['upstream_represented_truth_ids'])),
                note='Upstream misses are not assigned to any salience scorer; remaining misses may arise in salience or downstream classification.')
