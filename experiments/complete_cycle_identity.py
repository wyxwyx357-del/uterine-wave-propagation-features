"""Sandbox complete-cycle proposals; ambiguity is retained, never force paired.

Both phase orders are allowed. Three native positions and span+dt are existing
engineering supports, not new clinical cutoffs. No ground truth is read here.
"""
from itertools import combinations
import numpy as np
from uterine_wave.radial_events import _arrival_evidence, _describe_arrival, RadialWaveConfig


def blocks(ids):
    ids=np.asarray(sorted(set(ids)),int)
    return [b.tolist() for b in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1) if len(b)>=3]


def phase_ridges(nodes):
    """One unique signed trajectory per source phase; retain all aliases."""
    ridges=[];lookup={}
    for node in nodes:
        for sign in (1,-1):
            # Multiple same-sign peaks are not arbitrarily reduced to one.
            profile={c:values[0] for c in node['sections']
                     if len(values:=[p for p in node['profile'][c] if p['sign']==sign])==1}
            if not blocks(profile):continue
            key=(node['wall'],sign,tuple((c,p['peak_s'],tuple(p['epoch']) if p['epoch'] is not None else None)
                                        for c,p in sorted(profile.items())))
            if key in lookup:
                ridges[lookup[key]]['source_nodes'].append(node['node_id']);continue
            lookup[key]=len(ridges)
            ridges.append(dict(ridge_id=f'phase_{len(ridges):03d}',wall=node['wall'],
                sign=sign,profile=profile,source_nodes=[node['node_id']]))
    return ridges


def propose_cycles(nodes,s,span,dt):
    ridges=phase_ridges(nodes);proposals=[];rejected=[]
    tolerance=span+dt
    for left,right in combinations(ridges,2):
        if left['wall']!=right['wall'] or left['sign']==right['sign']:continue
        shared=sorted(set(left['profile'])&set(right['profile']))
        for order in (1,-1):
            first,second=(left,right) if order==1 else (right,left)
            good=[];diagnostics=[]
            for c in shared:
                a,b=first['profile'][c],second['profile'][c]
                epoch=a['epoch'] is not None and a['epoch']==b['epoch']
                ordered=a['peak_s']<b['peak_s']
                boundary_gap=b['start_s']-a['end_s']
                continuous=-tolerance<=boundary_gap<=tolerance
                # A third observed phase strictly between these peaks prevents
                # an adjacent-cycle interpretation; near-alias peaks stay QC.
                intervening=[]
                for other in ridges:
                    if other['wall']!=first['wall'] or other['ridge_id'] in (first['ridge_id'],second['ridge_id']):continue
                    p=other['profile'].get(c)
                    if p and a['peak_s']+tolerance<p['peak_s']<b['peak_s']-tolerance:
                        intervening.append(other['ridge_id'])
                valid=epoch and ordered and continuous and not intervening
                if valid:good.append(c)
                diagnostics.append(dict(section=c,epoch_compatible=epoch,time_order=ordered,
                    boundary_gap_s=boundary_gap,intervening_ridges=intervening,eligible=valid))
            supported=blocks(good)
            if not supported:
                rejected.append(dict(first=first['ridge_id'],second=second['ridge_id'],
                    reason='no three-adjacent ordered contiguous same-epoch supports',sections=diagnostics));continue
            for block in supported:
                x=np.asarray(s)[block]
                phase_fits=[_arrival_evidence(x,np.array([r['profile'][c]['peak_s'] for c in block]),dt)
                            for r in (first,second)]
                # Do not call this an exact zero crossing: only phase bounds
                # are available at this layer; midpoint uncertainty is stored.
                arrival=np.array([(first['profile'][c]['end_s']+second['profile'][c]['start_s'])/2 for c in block])
                m=_describe_arrival(np.array(block),arrival,np.asarray(s),dt,phase_fits,RadialWaveConfig())
                phase_conflict=phase_fits[0]['slope']*phase_fits[1]['slope']<0
                proposal=dict(cycle_id=f'cycle_{len(proposals):03d}',wall=first['wall'],
                    first_phase=first['ridge_id'],second_phase=second['ridge_id'],
                    phase_order=[first['sign'],second['sign']],sections=block,
                    arrival_time_s=arrival.tolist(),arrival_marker='phase_boundary_midpoint',
                    boundary_bracket_width_s=[abs(first['profile'][c]['end_s']-second['profile'][c]['start_s']) for c in block],
                    phase_fits=phase_fits,phase_trajectory_conflict=bool(phase_conflict),
                    propagation_measurement=m,
                    propagation_supported=m['classification'] in ('C2F','F2C') and not phase_conflict,
                    ownership_status='proposal',competitors=[],
                    source_nodes=sorted(set(first['source_nodes']+second['source_nodes'])))
                proposals.append(proposal)
    # Disconnected observations of the SAME phase pair are one proposal with
    # separate blocks, never two cycles competing with themselves. No fit is
    # computed across the gap.
    grouped={}
    for p in proposals:
        key=(p['first_phase'],p['second_phase'])
        observation=dict(sections=p['sections'],arrival_time_s=p['arrival_time_s'],
            boundary_bracket_width_s=p['boundary_bracket_width_s'],phase_fits=p['phase_fits'],
            propagation_measurement=p['propagation_measurement'],
            propagation_supported=p['propagation_supported'],
            phase_trajectory_conflict=p['phase_trajectory_conflict'])
        if key not in grouped:
            grouped[key]=dict(p,block_evidence=[observation])
        else:
            q=grouped[key];q['block_evidence'].append(observation)
            points=dict(zip(q['sections'],q['arrival_time_s']))
            points.update(zip(p['sections'],p['arrival_time_s']))
            q['sections']=sorted(points);q['arrival_time_s']=[points[c] for c in q['sections']]
            q['propagation_supported']=q['propagation_supported'] or p['propagation_supported']
    proposals=list(grouped.values())
    for index,p in enumerate(proposals):p['cycle_id']=f'cycle_{index:03d}'
    # Compare phase occupancy using physical nodes, not only ridge labels.
    # Aliases with nearby peaks in the same epoch are possible shared phases.
    by_id={r['ridge_id']:r for r in ridges}
    def same_phase(a,b):
        if a['wall']!=b['wall'] or a['sign']!=b['sign']:return False
        ids=[c for c in set(a['profile'])&set(b['profile'])
             if a['profile'][c]['epoch'] is not None and a['profile'][c]['epoch']==b['profile'][c]['epoch']
             and abs(a['profile'][c]['peak_s']-b['profile'][c]['peak_s'])<=tolerance]
        return bool(blocks(ids))
    for a,b in combinations(proposals,2):
        if a['wall']!=b['wall']:continue
        if any(same_phase(by_id[p],by_id[q]) for p in (a['first_phase'],a['second_phase'])
               for q in (b['first_phase'],b['second_phase'])):
            a['competitors'].append(b['cycle_id']);b['competitors'].append(a['cycle_id'])
    for p in proposals:
        p['ownership_status']='competition' if p['competitors'] else 'confirmed_complete_cycle'
    # Only full two-phase, same-section evidence can deduplicate opposite walls.
    definite=[p for p in proposals if not p['competitors']]
    matches={p['cycle_id']:[] for p in definite}
    for a,b in combinations(definite,2):
        if a['wall']==b['wall'] or a['phase_order']!=b['phase_order']:continue
        good=[]
        for c in set(a['sections'])&set(b['sections']):
            if all(abs(by_id[a[k]]['profile'][c]['peak_s']-by_id[b[k]]['profile'][c]['peak_s'])<=tolerance
                   for k in ('first_phase','second_phase')):good.append(c)
        if blocks(good):matches[a['cycle_id']].append(b['cycle_id']);matches[b['cycle_id']].append(a['cycle_id'])
    used=set();groups=[]
    for p in definite:
        cid=p['cycle_id']
        if cid in used:continue
        options=matches[cid]
        if len(options)==1 and matches[options[0]]==[cid]:members=[cid,options[0]]
        else:members=[cid]
        groups.append(members);used.update(members)
    confirmed_by_id={p['cycle_id']:p for p in definite}
    final=sum(any(confirmed_by_id[c]['propagation_supported'] for c in group) for group in groups)
    return dict(ridges=ridges,cycle_proposals=proposals,rejected_pairs=rejected,
        complete_cycle_proposal_count=len(proposals),
        competing_cycle_count=sum(bool(p['competitors']) for p in proposals),
        confirmed_cycle_groups=groups,experimental_propagation_count=final,
        unassigned_phase_ids=[r['ridge_id'] for r in ridges if not any(r['ridge_id'] in (p['first_phase'],p['second_phase']) for p in proposals)],
        direction_used_for_counting=False,no_new_thresholds=True,
        formal_results_changed=False)
