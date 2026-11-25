"""User-specified relation classes; phase/split provenance never selects class."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

ORDER=('A','E','F','B','D')


def consecutive_count(ids):
    ids=sorted(set(ids));best=run=0;last=None
    for c in ids:
        run=run+1 if last is not None and c==last+1 else 1
        best=max(best,run);last=c
    return best


def overlap(a,b,c,d):
    common=max(0.,min(b,d)-max(a,c))
    union=max(b,d)-min(a,c)
    return common,common/union if union>0 else None


def stable_identity(nodes):
    return json.dumps(sorted({(int(n['wall']),n['canonical_source']) for n in nodes}),separators=(',',':'))


def relation(pred,truth,span):
    start,end=pred['evidence_start_s'],pred['evidence_end_s'];mid=(start+end)/2
    lo,hi=truth['truth_start_s'],truth['truth_end_s']
    shared=sorted(set(pred['sections'])&set(truth['truth_sections']))
    seconds,iou=overlap(start,end,lo,hi)
    time_ok=lo-span<=mid<=hi+span
    outside=max(lo-span-mid,mid-hi-span,0.)
    direction='unresolved' if pred['direction'] not in ('C2F','F2C') else 'same' if pred['direction']==truth['truth_direction'] else 'opposite'
    return dict(truth_id=truth['truth_event_id'],midpoint_difference_s=mid-truth['truth_center_s'],
        start_difference_s=start-lo,end_difference_s=end-hi,interval_overlap_seconds=seconds,interval_IoU=iou,
        closest_truth_boundary_distance_s=min(abs(mid-lo),abs(mid-hi)),
        interval_gap_s=max(lo-end,start-hi,0.),inside_truth_support=bool(lo<=mid<=hi),
        inside_truth_expanded_by_measurement_span=bool(time_ok),outside_matcher_time_boundary_s=outside,
        shared_sections=shared,shared_section_count=len(shared),
        truth_coverage=len(shared)/len(truth['truth_sections']) if truth['truth_sections'] else None,
        prediction_coverage=len(shared)/len(pred['sections']) if pred['sections'] else None,
        contiguous_shared_count=consecutive_count(shared),direction_relation=direction,
        temporal_side='before_truth' if mid<lo else 'after_truth' if mid>hi else 'inside_truth',
        matcher_eligible=bool(time_ok and len(shared)>=3),
        E_space=bool(time_ok and len(shared)==2),
        E_time_fixed=bool(not time_ok and len(shared)>=3 and outside<=2.),
        E_time_auxiliary=bool(not time_ok and len(shared)>=3 and outside<=1.5*span),
        within_15s_path_overlap=bool(min(abs(mid-lo),abs(mid-hi))<=15. and len(shared)>=2))


def artifact_definition(source,scenario,shared,protocol,interval_start,interval_end):
    kind=scenario.get('artifact')
    if scenario['family']!='artifact' or scenario['amplitude'] is None:return None
    t=shared['t'];n=len(shared['s']);dt=shared['dt']
    if kind=='brightness':
        return dict(type=kind,start_s=float(interval_start[0]),end_s=float(interval_end[-1]),
                    walls=[0,1],sections=list(range(n)),per_section_support={},temporally_localized=False,
                    interpretation='Whole-video photometric condition; temporal coincidence alone cannot identify causation')
    at=protocol['timing']['artifact_Dev_center']+source['event_jitter']
    middle=int(round(protocol['timing']['artifact_Dev_position_fraction']*(n-1)))+source['artifact_jitter']
    columns=list(range(max(0,middle-2),min(n,middle+3)));support={}
    for c in columns:
        ids=np.array([int(np.argmin(abs(t-at)))]) if kind=='spike' else np.flatnonzero(abs(t-at-.08*(c-middle))<=2.)
        support[c]=[float(t[ids[0]]-dt/2),float(t[ids[-1]]+dt/2)]
    return dict(type=kind,start_s=min(x[0] for x in support.values()),end_s=max(x[1] for x in support.values()),
                walls=[0],sections=columns,per_section_support=support,temporally_localized=True,
                interpretation='Exact injected A-sample support, not filtered response support')


def artifact_relation(pred,artifact):
    if artifact is None:return None
    seconds,iou=overlap(pred['evidence_start_s'],pred['evidence_end_s'],artifact['start_s'],artifact['end_s'])
    shared=sorted(set(pred['sections'])&set(artifact['sections']))
    return dict(**artifact,overlap_seconds=seconds,interval_IoU=iou,shared_sections=shared,
                spatial_overlap_count=len(shared),affected_wall_overlap=bool(set(pred['walls'])&set(artifact['walls'])),
                direct_time_overlap=bool(seconds>0),propagation_direction=pred['direction'])


def primary_class(relations,artifact):
    if any(r['matcher_eligible'] for r in relations):return 'A'
    if any(r['E_space'] or r['E_time_fixed'] for r in relations):return 'E'
    if artifact is not None and artifact['direct_time_overlap']:return 'F'
    if any(r['within_15s_path_overlap'] for r in relations):return 'B'
    return 'D'


def maximum_matching(left,right,edges,marked=frozenset(),maximize_marked=True):
    """Diagnostics only: maximum cardinality then max/min marked A1 links."""
    left=sorted(left);right=sorted(right)
    if not left:return dict(cardinality=0,marked_pairs=0,pairs=[])
    reward=min(len(left),len(right))+2
    cost=np.zeros((len(left),len(right)+len(left)))
    cost[:,:len(right)]=reward*(len(left)+2)
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if (a,b) in edges:
                cost[i,j]=-reward+((-1 if maximize_marked else 1) if (a,b) in marked else 0)
    ii,jj=linear_sum_assignment(cost)
    pairs=[(left[i],right[j]) for i,j in zip(ii,jj) if j<len(right) and (left[i],right[j]) in edges]
    return dict(cardinality=len(pairs),marked_pairs=sum(pair in marked for pair in pairs),pairs=pairs)


def edge_in_some_maximum(left,right,edges,edge,maximum):
    a,b=edge
    if edge not in edges:return False
    reduced={(x,y) for x,y in edges if x!=a and y!=b}
    return 1+maximum_matching(set(left)-{a},set(right)-{b},reduced)['cardinality']==maximum


def distribution(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
                P90=float(np.percentile(x,90)) if len(x) else None,
                ge2_fraction=float((x>=2).mean()) if len(x) else None,ge3_fraction=float((x>=3).mean()) if len(x) else None)
