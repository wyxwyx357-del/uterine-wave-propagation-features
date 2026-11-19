"""One patient-level proposal ledger; cross-wall identity is not event truth.

Reuse native sections and span+one-frame timing from existing ownership.
Walls are separate measurement channels, not separate patient wave counts.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np

from experiments.direction_identity_trial import reconcile_identity_then_direction


def longest_adjacent(ids):
    ids=np.asarray(sorted(set(ids)),int)
    return max((len(b) for b in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)),default=0)


def candidate_nodes(salience, ownership):
    nodes=[]
    for wall_record in salience['walls']:
        wall=wall_record['wall'];source=ownership['walls'][wall]['reconciliation']
        lookup={e['event_proposal_id']:e for e in source['event_proposals']}
        for score in wall_record['scores']:
            if score['decision']!='salient_support':continue
            e=lookup[score['event_proposal_id']];entity=source['source_evidence'][e['canonical_source_entity_id']]
            arrival=dict(zip(e['supporting_sections'],e['arrival_time_s']))
            ids=score['retained_sections']
            profile={c:deepcopy(entity['profile'][c] if c in entity['profile'] else entity['profile'][str(c)]) for c in ids}
            nodes.append(dict(node_id=f'wall{wall}/'+e['event_proposal_id'],wall=wall,
                source_proposal_id=e['event_proposal_id'],canonical_source=e['canonical_source_entity_id'],
                sections=ids,arrival={c:arrival[c] for c in ids},profile=profile,
                direction=e['direction'],original_classification=e['canonical_measurement'].get('classification'),
                arrival_marker=e['canonical_measurement'].get('arrival_marker'),
                original_ownership_status=e['ownership_status']))
    return nodes


def compare_walls(a,b,span,dt):
    if a['wall']==b['wall']:raise ValueError('Cross-wall comparison requires different walls')
    shared=sorted(set(a['sections'])&set(b['sections']));tolerance=span+dt
    matched=[];near=[];conflict=[]
    for c in shared:
        left,right=a['profile'][c],b['profile'][c]
        same_sequence=[p['sign'] for p in left]==[p['sign'] for p in right]
        pairs=[(p,q) for p in left for q in right if p['sign']==q['sign']]
        peak_near=any(abs(p['peak_s']-q['peak_s'])<=tolerance for p,q in pairs)
        comparable_arrival=a['arrival_marker']==b['arrival_marker']
        arrival_near=comparable_arrival and abs(a['arrival'][c]-b['arrival'][c])<=tolerance
        if peak_near or arrival_near:near.append(c)
        if not same_sequence or not comparable_arrival:
            if peak_near or arrival_near:conflict.append(c)
            continue
        # Each wall has its own continuity bounds. Their run indices need not
        # equal: they describe different tissue, not the same material point.
        valid=(all(p.get('epoch') is not None and q.get('epoch') is not None for p,q in pairs) and
            all(abs(p['peak_s']-q['peak_s'])<=tolerance and
                q['start_s']-dt<=p['peak_s']<=q['end_s']+dt and
                p['start_s']-dt<=q['peak_s']<=p['end_s']+dt for p,q in pairs) and arrival_near)
        if valid:matched.append(c)
        elif peak_near or arrival_near:conflict.append(c)
    direction_compatible=(a['direction']==b['direction'])
    relation='shared_identity_supported' if (longest_adjacent(matched)>=3 and
        len(matched)==len(shared) and not conflict and direction_compatible) else (
        'unresolved' if near else 'no_shared_identity_evidence')
    return dict(node_a=a['node_id'],node_b=b['node_id'],relation=relation,
        shared_sections=shared,matched_sections=matched,near_sections=near,
        phase_or_marker_conflict_sections=conflict,directions_compatible=direction_compatible,
        tolerance_s=tolerance,reason=('phase order, peaks, local arrival and direction agree' if relation=='shared_identity_supported'
            else 'partial/conflicting cross-wall evidence' if relation=='unresolved'
            else 'insufficient corresponding native phase/arrival evidence'))


def aggregate_nodes(nodes,span,dt):
    comparisons=[compare_walls(a,b,span,dt) for a,b in combinations(nodes,2) if a['wall']!=b['wall']]
    choices={n['node_id']:[] for n in nodes};uncertain={n['node_id']:[] for n in nodes}
    for edge in comparisons:
        a,b=edge['node_a'],edge['node_b']
        if edge['relation']=='shared_identity_supported':choices[a].append(b);choices[b].append(a)
        elif edge['relation']=='unresolved':uncertain[a].append(b);uncertain[b].append(a)
    # No transitive or one-to-many merging of consecutive activities.
    pairs={tuple(sorted((a,bs[0]))) for a,bs in choices.items() if len(bs)==1 and
           choices[bs[0]]==[a] and not uncertain[a] and not uncertain[bs[0]]}
    lookup={n['node_id']:n for n in nodes};used=set();groups=[]
    for a,b in sorted(pairs):groups.append([lookup[a],lookup[b]]);used.update((a,b))
    groups.extend([[n] for n in nodes if n['node_id'] not in used])
    groups.sort(key=lambda group:(min(float(np.median(list(n['arrival'].values()))) for n in group),group[0]['node_id']))
    ledger=[]
    for index,group in enumerate(groups,1):
        members=[n['node_id'] for n in group];known={n['direction'] for n in group if n['direction'] is not None}
        direction=next(iter(known)) if len(known)==1 else None
        unresolved=sorted({k for node in members for k in uncertain[node]+choices[node] if k not in members})
        phases=[p for n in group for ps in n['profile'].values() for p in ps]
        ledger.append(dict(patient_candidate_id=f'activity_{index:03d}',source_nodes=members,
            walls=sorted({n['wall'] for n in group}),classification='propagation_candidate' if direction else 'local_or_unresolved_candidate',
            direction=direction,source_measurements=deepcopy(group),cross_wall_merged=len(group)==2,
            ownership_status='unresolved_cross_wall' if unresolved else 'shared_identity_proposal' if len(group)==2 else 'single_wall_proposal',
            unresolved_related_nodes=unresolved,evidence_start_s=min(p['start_s'] for p in phases),
            evidence_end_s=max(p['end_s'] for p in phases),reviewed=False))
    # Conditional count range: how many identities remain if unresolved
    # cross-wall pairs are eventually shown to be shared? Not a confidence
    # interval and not permission to merge them now.
    graph={n['node_id']:[] for n in nodes if n['wall']==0 and n['node_id'] not in used}
    for edge in comparisons:
        a,b=edge['node_a'],edge['node_b']
        if a in used or b in used or edge['relation']=='no_shared_identity_evidence':continue
        if lookup[a]['wall']==1:a,b=b,a
        graph[a].append(b)
    owner={}
    def augment(a,seen):
        for b in graph[a]:
            if b in seen:continue
            seen.add(b)
            if b not in owner or augment(owner[b],seen):owner[b]=a;return True
        return False
    possible=sum(augment(a,set()) for a in graph)
    return dict(activities=ledger,cross_wall_evidence=comparisons,
        input_wall_proposal_rows=len(nodes),patient_candidate_rows=len(ledger),
        cross_wall_pairs_merged=len(pairs),unresolved_cross_wall_rows=sum(bool(r['unresolved_related_nodes']) for r in ledger),
        conditional_identity_count_range=[len(ledger)-possible,len(ledger)],
        conditional_range_semantics='if recorded unresolved cross-wall matches are shared; not a statistical confidence interval',
        formal_event_count=None,formal_event_frequency_cpm=None)


def observation_union(maps):
    """Use union, never two wall durations or just salient activity duration.

    Local exposure requires the existing minimum of three adjacent measurable
    radial sections in at least one wall. This is not full-cavity coverage.
    """
    v=np.asarray(maps['valid_mask'],bool)&np.isfinite(maps['A'])
    begin=np.asarray(maps['interval_start_s'],float);end=np.asarray(maps['interval_end_s'],float)
    if v.ndim!=3 or v.shape[1]!=2 or len(v)!=len(begin) or end.shape!=begin.shape or np.any(end<=begin):
        raise ValueError('Expected physical radial interval maps')
    supported=(v[:,:,:-2]&v[:,:,1:-1]&v[:,:,2:]).any((1,2)) if v.shape[2]>=3 else np.zeros(len(v),bool)
    intervals=[]
    for a,b in zip(begin[supported],end[supported]):
        if intervals and a<=intervals[-1][1]+1e-9:intervals[-1][1]=max(intervals[-1][1],float(b))
        else:intervals.append([float(a),float(b)])
    return dict(seconds=sum(b-a for a,b in intervals),intervals_s=intervals,
        rule='union of measurement intervals with >=3 adjacent valid native sections in either wall',
        coverage_scope='locally observable radial activity, not proof of full-cavity visibility')


def aggregate_by_direction(nodes,span,dt,seconds):
    """Direction first; tissue wall never becomes the reporting category.

    Same direction is necessary, not sufficient, for shared identity. Local,
    complex and unknown-direction proposals stay outside C2F/F2C frequency.
    """
    buckets={}
    for label in ('C2F','F2C','other'):
        selected=[n for n in nodes if n['direction']==label] if label!='other' else [n for n in nodes if n['direction'] not in ('C2F','F2C')]
        result=aggregate_nodes(selected,span,dt)
        for i,event in enumerate(result['activities'],1):event['patient_candidate_id']=f'{label}_{i:03d}'
        low,high=result['conditional_identity_count_range']
        result['candidate_rate_range_cpm']=[low*60/seconds,high*60/seconds] if seconds>0 else [None,None]
        result['direction_bucket']=label
        result['direction_source']='original arrival-order proposal, not radial strain sign or wall label'
        buckets[label]=result
    return buckets


def _identity_first_buckets(reconciled,seconds):
    """Adapt identity-first activities to the existing patient report interface."""
    buckets={}
    for label in ('C2F','F2C','other'):
        activities=[deepcopy(activity) for activity in reconciled['activities']
                    if (activity['direction']==label if label!='other'
                        else activity['direction'] not in ('C2F','F2C'))]
        for index,activity in enumerate(activities,1):
            activity['patient_candidate_id']=f'{label}_{index:03d}'
        count=len(activities);rate=count*60/seconds if seconds>0 else None
        buckets[label]=dict(activities=activities,patient_candidate_rows=count,
            conditional_identity_count_range=[count,count],
            candidate_rate_range_cpm=[rate,rate],direction_bucket=label,
            direction_source=('arrival refit on each contiguous salience-retained block; '
                              'cross-wall identity reconciled before direction'),
            formal_event_count=None,formal_event_frequency_cpm=None)
    return buckets


def patient_summary(salience,ownership,maps):
    dt=float(np.median(np.diff(maps['interval_center_s'])));span=float(np.median(maps['interval_dt_s']))
    nodes=candidate_nodes(salience,ownership)
    exposure=observation_union(maps);seconds=exposure['seconds']
    legacy=aggregate_nodes(nodes,span,dt)
    reconciled=reconcile_identity_then_direction(nodes,np.asarray(maps['s'],float),span,dt)
    activities=reconciled['activities'];count=len(activities)
    directional=sum(a['direction'] in ('C2F','F2C') for a in activities)
    unresolved_cross_wall=sum(bool(a['unresolved_related_nodes']) for a in activities)
    result=dict(activities=activities,cross_wall_evidence=reconciled['identity_comparisons'],
        input_wall_proposal_rows=len(nodes),patient_candidate_rows=count,
        cross_wall_pairs_merged=reconciled['cross_wall_identity_pairs'],
        unresolved_cross_wall_rows=unresolved_cross_wall,
        conditional_identity_count_range=[count,count],
        conditional_range_semantics='exact output of current identity-first algorithm; not a confidence interval',
        formal_event_count=None,formal_event_frequency_cpm=None)
    result.update(exposure=exposure,candidate_activity_rate_cpm=count*60/seconds if seconds>0 else None,
        unresolved_ledger_row_density_cpm=count*60/seconds if seconds>0 else None,
        conditional_candidate_rate_range_cpm=[count*60/seconds,count*60/seconds] if seconds>0 else [None,None],
        directional_candidate_rows=directional,local_or_unresolved_rows=count-directional,
        candidate_propagation_rate_cpm=directional*60/seconds if seconds>0 else None,
        directional_candidate_counts={d:sum(a['direction']==d for a in activities) for d in ('C2F','F2C')},
        rates_are_exploratory_candidate_densities=True,timing_tolerance_s=span+dt)
    buckets=_identity_first_buckets(reconciled,seconds)
    result.update(primary_reporting='identity_first_direction_second',directional_summaries={d:buckets[d] for d in ('C2F','F2C')},
        non_directional_summary=buckets['other'],
        directional_candidate_count_range=[sum(buckets[d]['conditional_identity_count_range'][i] for d in ('C2F','F2C')) for i in (0,1)],
        combined_activity_fields_are_secondary_QC=True,
        legacy_direction_first_QC=dict(patient_candidate_rows=legacy['patient_candidate_rows'],
            conditional_identity_count_range=legacy['conditional_identity_count_range'],
            cross_wall_pairs_merged=legacy['cross_wall_pairs_merged']),
        retained_direction_refit=True,identity_before_direction=True)
    return result
