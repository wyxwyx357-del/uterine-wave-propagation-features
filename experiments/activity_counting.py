"""Significant propagation counts independent of anatomical direction labels.

Existing adjacent support and arrival classifier are reused without new cutoffs.
Direction is descriptive QC only. Unresolved ownership remains explicit.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np
from experiments.direction_identity_trial import compare_identity_without_direction
from uterine_wave.radial_events import _describe_arrival, _arrival_evidence, RadialWaveConfig


def retained_phase_fits(node,block,s,dt):
    """Fit signed phase peaks only where each section has one unambiguous peak."""
    fits=[];availability=[]
    for sign in (1,-1):
        peaks=[[p['peak_s'] for p in node['profile'][int(c)] if p['sign']==sign]
               for c in block]
        complete=all(len(values)==1 for values in peaks)
        availability.append(dict(sign=sign,complete=complete,
                                 peak_counts=[len(values) for values in peaks]))
        if complete:
            fits.append(dict(_arrival_evidence(np.asarray(s)[block],
                             np.array([values[0] for values in peaks]),dt),sign=sign))
    return fits,availability


def count_activities(nodes,s,span,dt,observation_seconds):
    edges=[compare_identity_without_direction(a,b,span,dt)
           for a,b in combinations(nodes,2) if a['wall']!=b['wall']]
    choices={n['node_id']:[] for n in nodes};uncertain={n['node_id']:[] for n in nodes}
    for edge in edges:
        target=choices if edge['relation']=='shared_identity_supported' else uncertain if edge['relation']=='unresolved' else None
        if target is not None:
            a,b=edge['node_a'],edge['node_b'];target[a].append(b);target[b].append(a)
    pairs={tuple(sorted((a,bs[0]))) for a,bs in choices.items() if len(bs)==1 and choices[bs[0]]==[a]
           and not uncertain[a] and not uncertain[bs[0]]}
    lookup={n['node_id']:n for n in nodes};used=set();groups=[]
    for a,b in sorted(pairs):groups.append([lookup[a],lookup[b]]);used.update((a,b))
    groups.extend([[n] for n in nodes if n['node_id'] not in used])
    activities=[]
    for i,group in enumerate(groups,1):
        evidence=[]
        for node in group:
            ids=np.asarray(sorted(node['sections']),int)
            for block in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1):
                if len(block)<3:continue
                arrival=np.array([node['arrival'][int(c)] for c in block])
                phase_fits,availability=retained_phase_fits(node,block,s,dt)
                m=_describe_arrival(block,arrival,np.asarray(s),dt,phase_fits,RadialWaveConfig())
                evidence.append(dict(node_id=node['node_id'],sections=block.tolist(),measurement=m,
                    phase_fit_availability=availability,
                    propagation_supported=m['classification'] in ('C2F','F2C')))
        members=[n['node_id'] for n in group]
        phases=[p for n in group for ps in n['profile'].values() for p in ps]
        activities.append(dict(activity_id=f'activity_{i:03d}',source_nodes=members,
            walls=sorted({n['wall'] for n in group}),
            propagation_supported=any(e['propagation_supported'] for e in evidence),
            block_evidence=evidence,direction_used_for_counting=False,
            direction_conflict_QC=len({n['direction'] for n in group if n['direction']})>1,
            unresolved_related_nodes=sorted({k for a in members for k in choices[a]+uncertain[a] if k not in members}),
            evidence_start_s=min(p['start_s'] for p in phases),evidence_end_s=max(p['end_s'] for p in phases)))
    count=sum(a['propagation_supported'] for a in activities)
    return dict(activities=activities,identity_comparisons=edges,
        significant_propagation_activity_count=count,
        significant_propagation_activity_frequency_cpm=count*60/observation_seconds if observation_seconds>0 else None,
        other_significant_activity_count=len(activities)-count,
        effective_observation_seconds=observation_seconds,
        direction_used_for_counting=False,cutoff=.6,
        interpretation='algorithm-defined significant propagation activity; weak activity may be missed')
