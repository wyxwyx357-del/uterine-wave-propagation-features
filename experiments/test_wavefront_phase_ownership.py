"""Sandbox phase-ownership feasibility; preserve counts and ambiguous evidence.

No polarity is forbidden and a shared phase is not automatically one event.
Only existing span+dt and three-adjacent-position support are used.
"""
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_activity_strength_trial import read,save
from experiments.audit_wave_evidence import control_arrays


def longest(ids):
    ids=np.array(sorted(ids),int)
    return max((len(b) for b in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)),default=0)


def evidence_graph(counted,original,tolerance):
    nodes=[]
    for activity in counted['activities']:
        for node_id in activity['source_nodes']:
            wall=int(node_id[4]);eid=node_id.split('/')[1]
            rec=original['walls'][wall]['reconciliation']
            event=next(e for e in rec['event_proposals'] if e['event_proposal_id']==eid)
            profile=rec['source_evidence'][event['canonical_source_entity_id']]['profile']
            retained=sorted({c for block in activity['block_evidence'] if block['node_id']==node_id for c in block['sections']})
            sequences=sorted({tuple(p['sign'] for p in profile[str(c)]) for c in retained})
            nodes.append(dict(node_id=node_id,wall=wall,source=event['canonical_source_entity_id'],
                retained_sections=retained,profile=profile,
                phase_sequences=[list(seq) for seq in sequences],
                propagation_supported=activity['propagation_supported']))
    edges=[]
    for a,b in combinations(nodes,2):
        # Context can expose shared phases, but is never promoted to salient support.
        shared=sorted(set(a['profile'])&set(b['profile']),key=int)
        supports=[]
        for sign in (1,-1):
            matched=[]
            for col in shared:
                left=[p for p in a['profile'][col] if p['sign']==sign]
                right=[p for p in b['profile'][col] if p['sign']==sign]
                if len(left)!=1 or len(right)!=1:continue
                p,q=left[0],right[0]
                continuity=p['epoch'] is not None and q['epoch'] is not None
                if a['wall']==b['wall']:continuity=continuity and p['epoch']==q['epoch']
                if continuity and abs(p['peak_s']-q['peak_s'])<=tolerance:
                    matched.append(int(col))
            if longest(matched)>=3:supports.append(dict(sign=sign,sections=matched))
        if supports:
            edges.append(dict(a=a['node_id'],b=b['node_id'],shared_phase_support=supports,
                relation='phase_reuse_or_same_activity_unresolved',
                both_polarities_shared=len(supports)==2,
                automatic_merge=False))
    return dict(nodes=nodes,edges=edges,accepted_new_wavefronts=[],
                original_count=counted['significant_propagation_activity_count'],
                automatic_count_change=0,ownership_not_resolved=True)


def run():
    root=ROOT/'outputs';target=root/'wavefront_phase_ownership_trial_20261119';target.mkdir(exist_ok=True)
    records=[]
    for path in sorted((root/'significant_activity_counts_20261119_v2').glob('texture*.json')):
        counted=read(path);original=read(root/'global_ownership_trial_20261119_v1'/path.name)
        _,_,t,_,span=control_arrays(root/'p3_outer_ideal_prior_20261117_v2'/(path.stem.split('_')[0]+'_rigid.npz'))
        result=evidence_graph(counted,original,float(span+np.median(np.diff(t))))
        save(target/path.name,result)
        records.append(dict(source=path.stem,phase_reuse_edges=len(result['edges']),
                            original_count=result['original_count'],automatic_count_change=0))
        if path.stem in ('texture0_local_once','texture0_double_F2C_0.04_gap10'):
            print(path.stem,'nodes',[(n['node_id'],n['phase_sequences']) for n in result['nodes']],
                  'edges',result['edges'])
    save(target/'summary.json',dict(conditions=len(records),records=records,
         no_parameter_changes=True,truth_not_used=True,counts_not_changed=True,
         conclusion='phase-role graph exposes ambiguous reuse; evidence alone does not establish a unique count'))
    print('CONDITIONS',len(records),'conditions_with_reuse',sum(r['phase_reuse_edges']>0 for r in records))


if __name__=='__main__':run()
