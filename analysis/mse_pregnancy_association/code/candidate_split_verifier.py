"""Sandbox: split evidence strictly inside existing periodic/partial parents.

No parent detection, global auxiliary scan, truth access or patient counting.
Local phase nodes are passed to the unchanged ridge/pair measurement code.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np
from uterine_wave import radial_events as core
from analysis.mse_pregnancy_association.code.event_input_smoothing import measurement_span_mean


def _blocks(ids):
    ids=np.asarray(ids,int)
    return [x for x in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1) if len(x)]


def _local_nodes(signal,valid,t,s,parent,span,floor):
    context=np.flatnonzero((t>=parent['start_s']-span)&(t<=parent['end_s']+span))
    times=t[context]
    values=np.full((len(times),len(s)),np.nan)
    columns=[[] for _ in s]
    for col in parent['sections']:
        supported=valid[context,col]&np.isfinite(signal[context,col])
        for run_id,(begin,end) in enumerate(core._runs(supported)):
            if end-begin<3:
                continue
            samples=int(2*np.floor(span/np.median(np.diff(times))/2)+1)
            if end-begin<samples:
                continue
            # Only this parent's buffered slice is read by the auxiliary filter.
            y=measurement_span_mean(signal[context[begin:end],col],times[begin:end],span_s=span)
            values[begin:end,col]=y
            edges=np.r_[0,np.flatnonzero(np.diff(y>=0))+1,len(y)]
            for sequence,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
                peak=a+int(np.argmax(abs(y[a:b])))
                amplitude=float(abs(y[peak]))
                if amplitude<=max(float(floor[col]),1e-12) or peak in (a,b-1):
                    continue
                columns[col].append(dict(column=col,run_id=run_id,sequence=sequence,
                    sign=1 if y[peak]>0 else -1,peak_index=int(begin+peak),
                    peak_time_s=float(times[begin+peak]),amplitude=amplitude,
                    first=int(begin+a),last=int(begin+b-1),
                    start_s=float(times[begin+a]),end_s=float(times[begin+b-1]),
                    left_censored=bool(a==0),right_censored=bool(b==len(y))))
    return context,times,values,columns


def prepare_parent(signal,valid,time_s,space,parent,*,noise_floor,measurement_span_s):
    """Compute local evidence once; support=3/4/5 uses the SAME observations.

    The parent uses its original time/section bounding box. Context is one
    existing measurement span; child ARRIVALS must remain in the unexpanded
    box. No neighboring sections or new parent envelopes are introduced.
    """
    y=np.asarray(signal,float);m=np.asarray(valid,bool)
    t=np.asarray(time_s,float);s=np.asarray(space,float)
    ids=np.asarray(parent['sections'],int)
    if (y.shape!=m.shape or y.shape!=(len(t),len(s)) or len(t)<3 or
        np.any(np.diff(t)<=0) or not np.allclose(np.diff(t),np.median(np.diff(t)),rtol=1e-5,atol=1e-9) or
        len(ids)<1 or np.any(np.diff(ids)<=0) or ids.min()<0 or ids.max()>=len(s) or
        not np.isfinite(measurement_span_s) or measurement_span_s<=0):
        raise ValueError('Expected native uniform maps, ordered parent sections and positive measurement span')
    floor=np.broadcast_to(np.asarray(noise_floor,float),(len(s),))
    if not np.isfinite(floor).all() or np.any(floor<0) or parent['start_s']>parent['end_s']:
        raise ValueError('Invalid frozen noise reference or parent time bounds')
    context,times,values,columns=_local_nodes(y,m,t,s,parent,measurement_span_s,floor)
    ridges=core._ridges(columns,times,s) if len(times)>=3 else []
    edges=core._phase_edges(ridges)
    config=core.RadialWaveConfig()
    measured=[]
    # Keep both physiological polarities available in the observations. This
    # trial verifies two positive->negative maxima, the already frozen raw
    # length-bump reference. It does not declare negative-first activity false.
    for key,edge in edges.items():
        if ridges[edge['first']]['sign']!=1:
            continue
        event=core._measure_pair(dict(edge),ridges,values,times,s,config)
        kept=[col for col,arrival in zip(event['sections'],event['arrival_time_s'])
              if parent['start_s']<=arrival<=parent['end_s']]
        for block in _blocks(kept):
            if len(block)<3:
                continue
            local_edge=dict(edge,columns=block.tolist())
            item=core._measure_pair(local_edge,ridges,values,times,s,config)
            item.update(_edge=local_edge,_key=key)
            measured.append(item)
    # Local event rank is derived from actual eligible successive phases, not
    # the injected truth or an optimized lag. It exposes identity switching.
    ordinals={};turn_counts={}
    for col,column in enumerate(columns):
        turns=[]
        for first,second in zip(column[:-1],column[1:]):
            if (first['sign']!=1 or second['sign']!=-1 or first['run_id']!=second['run_id'] or
                second['sequence']-first['sequence']!=1):
                continue
            k,k2=first['last'],second['first']
            crossing=times[k]-values[k,col]/(values[k2,col]-values[k,col])*(times[k2]-times[k])
            if parent['start_s']<=crossing<=parent['end_s']:
                turns.append((first,second))
        turn_counts[col]=len(turns)
        for rank,(first,second) in enumerate(turns):
            ordinals[(col,first['run_id'],first['sequence'],second['sequence'])]=rank
    node_maps={r['id']:{n['column']:n for n in r['nodes']} for r in ridges}
    options=[]
    for left,right in combinations(measured,2):
        if set(left['phase_ridge_ids'])&set(right['phase_ridge_ids']):
            continue
        common=sorted(set(left['sections'])&set(right['sections']))
        for block in _blocks(common):
            if len(block)<3:
                continue
            a,b=left,right
            arrivals_a=dict(zip(a['sections'],a['arrival_time_s']))
            arrivals_b=dict(zip(b['sections'],b['arrival_time_s']))
            differences=np.array([arrivals_b[c]-arrivals_a[c] for c in block])
            if np.all(differences<0):
                a,b=b,a
                differences=-differences
            conflicts=[]
            if not np.all(differences>0):
                conflicts.append('children_cross_in_time')
            ranks=[[],[]]
            for col in block:
                nodes=[node_maps[rid][int(col)] for item in (a,b) for rid in item['phase_ridge_ids']]
                if len({n['run_id'] for n in nodes})!=1:
                    conflicts.append('different_continuous_runs')
                    continue
                if not nodes[1]['sequence']<nodes[2]['sequence']:
                    conflicts.append('interleaved_phase_sequences')
                for index in (0,1):
                    first,second=nodes[2*index:2*index+2]
                    rank=ordinals.get((int(col),first['run_id'],first['sequence'],second['sequence']))
                    if rank is None:
                        conflicts.append('unresolved_local_turn_identity')
                    else:
                        ranks[index].append(rank)
            if any(len(set(r))!=1 for r in ranks):
                conflicts.append('local_event_rank_switch')
            if all(ranks) and any(y!=x+1 for x,y in zip(*ranks)):
                conflicts.append('additional_intervening_activity')
            children=[]
            for item in (a,b):
                edge=dict(item['_edge'],columns=block.tolist())
                child=core._measure_pair(edge,ridges,values,times,s,config)
                if any(flag in child['quality_flags'] for flag in ('nonmonotone_arrivals','phase_direction_conflict')):
                    conflicts.append('inconsistent_child_arrival_order')
                selected,_=core._pair_phases(ridges,{item['_key']:edge})
                propagation=bool(selected and child['classification'] in ('C2F','F2C') and
                    'phase_direction_conflict' not in child['quality_flags'] and
                    'nonmonotone_arrivals' not in child['quality_flags'])
                child.update(parent_candidate_id=parent['event_id'],activity_split_supported=True,
                    propagation_supported=propagation,
                    fitted_classification=child['classification'],
                    classification=child['classification'] if propagation else 'activity_only',
                    direction=child['classification'] if propagation else None,
                    fitted_velocity_paths_per_s=child['velocity_paths_per_s'],
                    velocity_paths_per_s=child['velocity_paths_per_s'] if propagation else None,
                    speed_candidate_paths_per_s=child['velocity_paths_per_s'] if propagation else None,
                    supporting_phase_evidence=[dict(section=int(c),
                        phases=[dict(sign=node_maps[r][int(c)]['sign'],
                            peak_s=node_maps[r][int(c)]['peak_time_s'],
                            observed_start_s=node_maps[r][int(c)]['start_s'],
                            observed_end_s=node_maps[r][int(c)]['end_s'],
                            run_id=node_maps[r][int(c)]['run_id'],
                            amplitude=node_maps[r][int(c)]['amplitude']) for r in child['phase_ridge_ids']]) for c in block],
                    propagation_source='Local nodes through unchanged ridge/pair measurement and pairing',
                    reviewed=False)
                children.append(child)
            options.append(dict(identity=tuple(a['phase_ridge_ids']+b['phase_ridge_ids']),
                                common_sections=block.tolist(),children=children,
                                conflicts=sorted(set(conflicts))))
    return dict(parent=deepcopy(parent),context_time_range_s=[float(times[0]),float(times[-1])] if len(times) else None,
        context_expansion_s=float(measurement_span_s),turn_counts=turn_counts,options=options,
        local_phase_ridges=[dict(ridge_id=r['id'],sign=r['sign'],sections=[n['column'] for n in r['nodes']],
                                peak_time_s=[n['peak_time_s'] for n in r['nodes']]) for r in ridges],
        local_observations=dict(time_s=times,sections=ids,smoothed_A=values[:,ids],
                                valid_mask=m[context][:,ids],auxiliary_valid_mask=np.isfinite(values[:,ids]),
                                raw_A=y[context][:,ids]))


def decide_split(prepared,*,minimum_adjacent_support=3,include_signals=False):
    if minimum_adjacent_support not in (3,4,5):
        raise ValueError('This fixed experiment permits adjacent support 3, 4 or 5')
    parent=prepared['parent']
    eligible=[p for p in prepared['options'] if len(p['common_sections'])>=minimum_adjacent_support]
    identities={p['identity'] for p in eligible if not p['conflicts']}
    if len(identities)==1 and not any(p['conflicts'] for p in eligible):
        selected=max((p for p in eligible if not p['conflicts']),key=lambda p:len(p['common_sections']))
        status='split_two';reason='two ordered complete local phase sequences on adjacent native sections'
        children=deepcopy(selected['children'])
        for index,child in enumerate(children,1):
            child['event_id']=f"{parent['event_id']}_child_{index}"
    else:
        children=[]
        tentative=any(n>=2 for n in prepared['turn_counts'].values())
        status='split_ambiguous' if tentative or eligible else 'keep_single'
        reason=('competing identities or inconsistent local order' if eligible else
                'second-turn evidence lacks shared adjacent support' if tentative else 'no supported internal second activity')
    result=dict(parent_candidate_id=parent['event_id'],parent=deepcopy(parent),
        split_status=status,reason=reason,minimum_adjacent_support=minimum_adjacent_support,
        context_time_range_s=prepared['context_time_range_s'],
        context_expansion_s=prepared['context_expansion_s'],
        child_1=children[0] if children else None,child_2=children[1] if children else None,
        tentative_pair_count=len(eligible),
        conflict_reasons=sorted({flag for p in eligible for flag in p['conflicts']}),
        local_turn_counts=prepared['turn_counts'],event_count=None,event_frequency=None)
    if include_signals:
        result['local_observations']=prepared['local_observations']
    return result
