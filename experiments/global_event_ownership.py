"""Sandbox global ownership of frozen parents and already-local split children.

No image/signal detection, filter changes, injected truth or patient counts.
Global means all candidates in ONE wall. Cross-wall event pairing is separate.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)


def build_entities(report,proposals,time_s,valid):
    t=np.asarray(time_s,float);m=np.asarray(valid,bool)
    if m.ndim!=2 or len(t)!=len(m) or len(t)<3 or np.any(np.diff(t)<=0):
        raise ValueError('Expected native time x section validity for one wall')
    epochs={c:_runs(m[:,c]) for c in range(m.shape[1])}
    def phase_node(col,p):
        index=int(np.argmin(abs(t-p['peak_s'])))
        epoch=next(((int(a),int(b)) for a,b in epochs[col] if a<=index<b),None)
        return dict(sign=int(p['sign']),peak_s=float(p['peak_s']),
            start_s=float(p['observed_start_s']),end_s=float(p['observed_end_s']),epoch=epoch)
    fragments={f['fragment_id']:f for f in report['fragments']}
    parents={e['event_id']:e for e in report['events']}
    entities={};parent_to_base={}
    for group in report['proposed_events']:
        eid='base:'+group['proposal_id'];profile={};arrival={}
        members=[parents[k] for k in group['fragment_ids']]
        for fid in group['fragment_ids']:
            f=fragments[fid];parent_to_base[fid]=eid
            for p in f['local_phase_support']:
                c=int(p['section']);profile.setdefault(c,[]).append(phase_node(c,p))
            arrival.update(zip(f['sections'],f['arrival_time_s']))
        for c in profile:
            unique={(p['sign'],p['peak_s']):p for p in profile[c]}
            profile[c]=sorted(unique.values(),key=lambda p:p['peak_s'])
        canonical=max(members,key=lambda e:(e['evidence_kind']=='paired_phase',len(e['sections'])))
        entities[eid]=dict(entity_id=eid,source_kind='existing',source_parents=list(group['fragment_ids']),
            complete=all(len(ps)==2 for ps in profile.values()),profile=profile,arrival=arrival,
            direction=canonical['classification'] if canonical['classification'] in ('C2F','F2C') else None,
            canonical_measurement=deepcopy(canonical),original_measurements=deepcopy(members),
            phase_ridge_provenance=[dict(parent_id=e['event_id'],phase_ridge_ids=e['phase_ridge_ids']) for e in members])
    splits=[]
    for proposal in proposals:
        if proposal['split_status']!='split_two':
            continue
        children=[]
        for child in (proposal['child_1'],proposal['child_2']):
            eid='child:'+child['event_id'];profile={}
            for row in child['supporting_phase_evidence']:
                c=int(row['section']);profile[c]=[phase_node(c,p) for p in row['phases']]
            entities[eid]=dict(entity_id=eid,source_kind='child',source_parents=[proposal['parent_candidate_id']],
                complete=True,profile=profile,arrival=dict(zip(child['sections'],child['arrival_time_s'])),
                direction=child['direction'],canonical_measurement=deepcopy(child),original_measurements=[],
                phase_ridge_provenance=[dict(parent_id=proposal['parent_candidate_id'],
                    source='parent_local_auxiliary',phase_ridge_ids=child['phase_ridge_ids'])])
            children.append(eid)
        splits.append(dict(parent_id=proposal['parent_candidate_id'],base_id=parent_to_base[proposal['parent_candidate_id']],
                           children=children))
    return entities,splits


def compare_identity(a,b,*,sample_dt_s,measurement_span_s):
    """Prioritized evidence, not a scalar score or fixed global-time radius.

    Existing 3-section connector support is reused. Both signs/sequence,
    physical continuity, local arrival and peak trajectories must agree.
    Uncomparable/partial evidence is retained rather than forcibly merged.
    """
    shared=sorted(set(a['profile'])&set(b['profile']))
    tolerance=measurement_span_s+sample_dt_s
    evidence=dict(shared_sections=shared,tolerance_s=tolerance,matched_sections=[],
                  exact_phase_nodes=0,phase_sequence_conflict=False)
    if not shared:
        return dict(relation='unrelated',priority=0,reason='no shared native support',**evidence)
    paired=a['complete'] and b['complete']
    exact=0;matched=[];near=[];contradictions=[]
    for c in shared:
        left,right=a['profile'][c],b['profile'][c]
        signs_a=[p['sign'] for p in left];signs_b=[p['sign'] for p in right]
        paired_here=len(left)==2 and len(right)==2
        if paired_here and signs_a!=signs_b:
            evidence['phase_sequence_conflict']=True
            continue
        pairs=[]
        for p in left:
            options=[q for q in right if q['sign']==p['sign']]
            if len(options)==1:
                pairs.append((p,options[0]))
        if not pairs:
            continue
        distances=[abs(p['peak_s']-q['peak_s']) for p,q in pairs]
        if min(distances)<=tolerance:
            near.append(c)
        epoch_ok=all(p['epoch'] is not None and p['epoch']==q['epoch'] for p,q in pairs)
        bounds_ok=all(q['start_s']-sample_dt_s<=p['peak_s']<=q['end_s']+sample_dt_s and
                      p['start_s']-sample_dt_s<=q['peak_s']<=p['end_s']+sample_dt_s for p,q in pairs)
        peaks_ok=max(distances)<=tolerance
        arrival_ok=not paired_here or abs(a['arrival'][c]-b['arrival'][c])<=tolerance
        if epoch_ok and bounds_ok and peaks_ok and arrival_ok:
            matched.append(c)
            exact+=sum(p['peak_s']==q['peak_s'] for p,q in pairs)
        elif min(distances)<=tolerance:
            contradictions.append(c)
    evidence.update(matched_sections=matched,exact_phase_nodes=exact)
    if a['direction'] and b['direction'] and a['direction']!=b['direction']:
        return dict(relation='ambiguous' if near else 'distinct',priority=0,
                    reason='opposing known propagation directions',**evidence)
    if evidence['phase_sequence_conflict']:
        return dict(relation='distinct',priority=0,reason='different ordered phase sequences',**evidence)
    blocks=np.split(np.asarray(matched,int),np.flatnonzero(np.diff(matched)!=1)+1)
    enough=max((len(x) for x in blocks),default=0)>=3
    if enough and not contradictions and len(matched)==len(shared):
        relation='duplicate' if paired else 'partial_duplicate'
        priority=3 if paired and exact==2*len(shared) else 2 if paired else 1
        return dict(relation=relation,priority=priority,reason='same local phase and arrival structure',**evidence)
    if near:
        return dict(relation='ambiguous',priority=0,reason='partial or conflicting local phase support',**evidence)
    return dict(relation='distinct',priority=0,reason='separate local phase timing',**evidence)


def _rank(e):
    # Complete evidence first, then larger native support; existing evidence
    # wins ties. A smaller local child cannot overwrite a wider complete parent.
    return (e['complete'],len(e['profile']),
            e['source_kind']=='existing',e['entity_id'])


def reconcile(report,proposals,time_s,valid,*,measurement_span_s):
    entities,splits=build_entities(report,proposals,time_s,valid)
    dt=float(np.median(np.diff(time_s)))
    relations={}
    def compare(x,y):
        key=tuple(sorted((x,y)))
        if key not in relations:
            relations[key]=compare_identity(entities[x],entities[y],sample_dt_s=dt,
                                            measurement_span_s=measurement_span_s)
        return relations[key]
    def duplicate(x,y):
        return compare(x,y)['relation'] in ('duplicate','partial_duplicate')
    bases=[k for k,e in entities.items() if e['source_kind']=='existing']
    base_ambiguity={}
    for x,y in combinations(bases,2):
        if compare(x,y)['relation']=='ambiguous':
            for k,other in ((x,y),(y,x)):
                base_ambiguity.setdefault(k,dict(reason='conflicting original ownership evidence',possible_owners=[]))['possible_owners'].append(other)
    for k in bases:
        choices=[p for p in bases if p!=k and entities[p]['complete'] and duplicate(k,p)]
        if any(not duplicate(a,b) for a,b in combinations(choices,2)):
            previous=base_ambiguity.get(k,{}).get('possible_owners',[])
            base_ambiguity[k]=dict(reason='original evidence matches conflicting complete events',possible_owners=sorted(set(previous+choices)))
    # Reconcile original evidence first. No child can serve as a transitive
    # bridge that merges otherwise conflicting original event identities.
    base_clusters={k:[k] for k in bases};base_owner={k:k for k in bases}
    for x,y in combinations(sorted(bases,key=lambda k:_rank(entities[k]),reverse=True),2):
        cx,cy=base_owner[x],base_owner[y]
        if cx==cy or x in base_ambiguity or y in base_ambiguity or not duplicate(x,y):
            continue
        union=base_clusters[cx]+base_clusters[cy]
        anchor=max(union,key=lambda k:_rank(entities[k]))
        if not all(k==anchor or duplicate(anchor,k) for k in union):
            continue
        if any(compare(p,q)['relation'] in ('ambiguous','distinct') for p,q in combinations(union,2)):
            continue
        base_clusters.pop(cx);base_clusters.pop(cy);base_clusters[anchor]=union
        for k in union:base_owner[k]=anchor
    owners={};uncertain={};attachments={k:[] for k in base_clusters}
    for k,e in entities.items():
        if e['source_kind']!='child':continue
        definite=[];possible=[]
        for anchor,members in base_clusters.items():
            relation=compare(k,anchor)
            if relation['relation'] in ('duplicate','partial_duplicate'):
                definite.append((anchor,relation['priority']))
            elif relation['relation']=='ambiguous':
                possible.append(anchor)
        highest=max((priority for _,priority in definite),default=0)
        strongest=[anchor for anchor,priority in definite if priority==highest]
        # Complete/exact evidence takes precedence over incomplete secondary
        # overlap. A partial match alone cannot resolve competing owners.
        if len(strongest)==1 and (highest>=2 or not any(p!=strongest[0] for p in possible)):
            owners[k]=strongest[0]
        elif definite or possible:
            uncertain[k]=dict(reason='multiple or incomplete existing ownership',
                              possible_owners=sorted(set([a for a,_ in definite]+possible)))
    must_separate={frozenset(s['children']) for s in splits}
    viable=[];actions=[]
    for split in splits:
        a,b=split['children']
        if a in uncertain or b in uncertain or (a in owners and b in owners and owners[a]==owners[b]):
            actions.append(dict(**split,status='ambiguous',reason='conflicting child ownership',child_owners=None))
        else:viable.append(split)
    unmatched=sorted({c for s in viable for c in s['children'] if c not in owners},
                     key=lambda k:_rank(entities[k]),reverse=True)
    new_clusters={};new_owner={}
    for child in unmatched:
        candidates=[anchor for anchor,members in new_clusters.items()
            if duplicate(child,anchor) and all(frozenset((child,k)) not in must_separate for k in members)
            and not any(compare(child,k)['relation'] in ('ambiguous','distinct') for k in members)]
        possible=[anchor for anchor in new_clusters if compare(child,anchor)['relation']=='ambiguous']
        if len(candidates)==1 and not possible:
            anchor=candidates[0];new_clusters[anchor].append(child);new_owner[child]=anchor
        elif candidates or possible:
            uncertain[child]=dict(reason='conflicting new-child ownership',possible_owners=sorted(set(candidates+possible)))
        else:
            new_clusters[child]=[child];new_owner[child]=child
    accepted=[]
    for split in viable:
        a,b=split['children']
        if a in uncertain or b in uncertain:
            actions.append(dict(**split,status='ambiguous',reason='new-child identity ambiguity',child_owners=None));continue
        assigned=[owners[c] if c in owners else new_owner[c] for c in (a,b)]
        if assigned[0]==assigned[1]:
            actions.append(dict(**split,status='ambiguous',reason='two children cannot share one identity',child_owners=None));continue
        action=dict(**split,status='reuse_existing' if all(c in owners for c in (a,b)) else 'accept_split',
                    reason='one identity per supported activity',child_owners=assigned)
        actions.append(action);accepted.append(action)
    # Resolve transaction dependencies monotonically: a rejected split cannot
    # erase an existing group or act as the sole bridge for another split.
    while True:
        active_children={c for a in accepted for c in a['children']}
        referenced={owner for a in accepted for owner in a['child_owners']}
        split_parents={a['parent_id'] for a in accepted}
        retired=set()
        for anchor,members in base_clusters.items():
            source_parents={p for k in members for p in entities[k]['source_parents']}
            complete_parents={m['event_id'] for k in members for m in entities[k]['original_measurements']
                              if m['evidence_kind']=='paired_phase'}
            # A remaining unpaired fragment is evidence, not an independently
            # complete event that can lock a decomposed container forever.
            # Preserve it explicitly as unresolved, never silently discard it.
            if (source_parents&split_parents and complete_parents<=split_parents and anchor not in referenced):
                retired.add(anchor)
        reject=set()
        for a in accepted:
            source=base_owner[a['base_id']]
            if source not in a['child_owners'] and source not in retired:
                reject.add(a['parent_id'])
                a['reason']='source group is still independently owned; cannot add two more identities'
        for anchor,members in new_clusters.items():
            active=[c for c in members if c in active_children]
            if not active:continue
            canonical=max(active,key=lambda k:_rank(entities[k]))
            if any(c!=canonical and not duplicate(c,canonical) for c in active):
                for a in accepted:
                    if anchor in a['child_owners']:
                        reject.add(a['parent_id']);a['reason']='new identity depends on rejected evidence bridge'
        if not reject:break
        for a in accepted:
            if a['parent_id'] in reject:a['status']='ambiguous';a['child_owners']=None
        accepted=[a for a in accepted if a['parent_id'] not in reject]
    finals=[];source_to_final={}
    def emit(anchor,members,kind):
        canonical=max(members,key=lambda k:_rank(entities[k]));e=entities[canonical]
        eid=f'event_proposal_{len(finals)+1:03d}'
        source_parents=sorted({p for k in members for p in entities[k]['source_parents']})
        finals.append(dict(event_proposal_id=eid,ownership_status='merged_evidence' if len(members)>1 else kind,
            canonical_source_entity_id=canonical,source_entity_ids=sorted(members),source_parent_ids=source_parents,
            from_split=any(entities[k]['source_kind']=='child' for k in members),
            supporting_sections=sorted(e['profile']),arrival_time_s=[e['arrival'][c] for c in sorted(e['profile'])],
            direction=e['direction'],canonical_measurement=deepcopy(e['canonical_measurement']),
            preserved_original_measurements=[deepcopy(m) for k in members for m in entities[k]['original_measurements']],
            phase_ridge_provenance=[deepcopy(p) for k in members for p in entities[k]['phase_ridge_provenance']],
            ownership_conflicts=[],event_count=None,event_frequency=None,reviewed=False))
        for k in members:source_to_final[k]=eid
    for anchor,members in sorted(base_clusters.items()):
        if anchor in retired:continue
        children=[c for c in active_children if owners.get(c)==anchor]
        emit(anchor,members+sorted(children),'retained_existing')
    for anchor,members in sorted(new_clusters.items()):
        active=[c for c in members if c in active_children]
        if active:emit(anchor,active,'retained_new_child')
    for action in actions:
        action['final_event_ids']=[source_to_final.get(c) for c in action['children']] if action['status']!='ambiguous' else []
    ambiguity=[dict(source_entity_id=k,**v) for k,v in sorted({**base_ambiguity,**uncertain}.items())]
    for event in finals:
        event['ownership_conflicts']=[a['source_entity_id'] for a in ambiguity
            if a['source_entity_id'] in event['source_entity_ids'] or any(owner in event['source_entity_ids'] for owner in a['possible_owners'])]
        event['ownership_conflicts'] += ['split:'+a['parent_id'] for a in actions if a['status']=='ambiguous'
                                        and a['parent_id'] in event['source_parent_ids']]
        if event['ownership_conflicts']:event['ownership_status']='ambiguous_preserved_existing'
    return dict(event_proposals=finals,split_actions=actions,ambiguous_ownership=ambiguity,
        source_to_final=source_to_final,retired_parent_groups=sorted(retired),
        unresolved_retired_fragments=[dict(parent_candidate_id=m['event_id'],source_group=anchor,
            original_measurement=deepcopy(m),status='unassigned_original_fragment',
            reason='Container split; partial fragment has no independently complete identity')
            for anchor in sorted(retired) for k in base_clusters[anchor] for m in entities[k]['original_measurements']
            if m['event_id'] not in split_parents],
        source_evidence=deepcopy(entities),
        pairwise_evidence=[dict(source_a=x,source_b=y,**value) for (x,y),value in sorted(relations.items())
                           if value['relation'] not in ('unrelated','distinct')],
        matching_rules=dict(native_sample_dt_s=dt,measurement_span_s=measurement_span_s,
            timing_tolerance_s=measurement_span_s+dt,minimum_shared_adjacent_sections=3,
            priority=['exact ordered phase nodes','complete phase and arrival structure','unique partial phase ownership'],
            no_transitive_bridge=True),event_count=None,event_frequency=None)
