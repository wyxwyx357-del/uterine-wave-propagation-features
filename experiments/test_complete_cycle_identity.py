"""Fixed 87-control complete-cycle trial; truth attached after proposals only."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.complete_cycle_identity import propose_cycles,blocks
from experiments.audit_wave_evidence import control_arrays
from experiments.test_activity_strength_trial import read,save
from experiments.test_double_event_signals import double_fixture,raw_truth,phase_identity
from experiments.test_weak_wave_signals import radial_fixture


def evaluate(result,truth,tolerance):
    ridges={r['ridge_id']:r for r in result['ridges']};labels={};mixed=[];additional=[]
    for p in result['cycle_proposals']:
        matches={};cross=[];phase_labels=[]
        for c in p['sections']:
            ids=[phase_identity(c,ridges[p[k]]['sign'],ridges[p[k]]['profile'][c]['peak_s'],truth)
                 for k in ('first_phase','second_phase')]
            phase_labels.append(dict(section=c,truth_phase_ids=ids))
            if None not in ids and ids[0]!=ids[1]:cross.append(c)
            if None not in ids and ids[0]==ids[1] and p['phase_order']==[1,-1]:
                matches.setdefault(ids[0],[]).append(c)
        supported=[identity for identity,cols in matches.items() if blocks(cols)]
        labels[p['cycle_id']]=supported
        if cross:mixed.append(p['cycle_id'])
        if not supported:additional.append(p['cycle_id'])
        p['evaluation_only']=dict(supported_truth_ids=supported,cross_truth_sections=cross,phase_labels=phase_labels,
                                  correct_role_for_known_length_bump=p['phase_order']==[1,-1])
    represented=Counter();false_merges=[]
    for group in result['confirmed_cycle_groups']:
        # Count only propagating groups, and expose distinct-truth collapse.
        members=[p for p in result['cycle_proposals'] if p['cycle_id'] in group]
        if not any(p['propagation_supported'] for p in members):continue
        identities={i for cid in group for i in labels[cid]}
        if len(identities)>1:false_merges.append(group)
        for identity in identities:represented[identity]+=1
    return dict(proposals_per_truth={str(i):sum(i in ids for ids in labels.values()) for i in range(truth['uniform_observable_count'])},
                representations_per_truth={str(i):represented[i] for i in range(truth['uniform_observable_count'])},
                mixed_truth_cycle_proposals=mixed,additional_or_unassigned_proposals=additional,
                confirmed_groups_merging_distinct_truths=false_merges)


def run():
    root=ROOT/'outputs';output=root/'complete_cycle_identity_trial_20261119';output.mkdir(exist_ok=True)
    rows=[]
    for path in sorted((root/'significant_activity_counts_20261119_v2').glob('texture*.json')):
        counted=read(path);original=read(root/'global_ownership_trial_20261119_v1'/path.name)
        parent=read(root/'event_input_smoothing_20261118_v1'/path.name)
        texture=path.stem.split('_')[0]
        _,_,t,s,span=control_arrays(root/'p3_outer_ideal_prior_20261117_v2'/(texture+'_rigid.npz'))
        nodes=[]
        for a in counted['activities']:
            for nid in a['source_nodes']:
                w=int(nid[4]);eid=nid.split('/')[1];rec=original['walls'][w]['reconciliation']
                e=next(e for e in rec['event_proposals'] if e['event_proposal_id']==eid)
                profile=rec['source_evidence'][e['canonical_source_entity_id']]['profile']
                ids=sorted({c for b in a['block_evidence'] if b['node_id']==nid for c in b['sections']})
                nodes.append(dict(node_id=nid,wall=w,sections=ids,profile={c:profile[str(c)] for c in ids}))
        result=propose_cycles(nodes,s,span,float(np.median(np.diff(t))))
        truth=None;direction=parent['expected_direction']
        if parent['family']=='double':
            truth=raw_truth(double_fixture(t,s,span,direction,parent['gap_s'],parent['second_amplitude'])[0],t)
        elif parent['family']=='single_or_negative' and direction:
            kind,amp=path.stem[len(texture+'_single_'):].rsplit('_',1)
            truth=raw_truth(radial_fixture(t,s,span,kind,float(amp))[0],t)
        result['evaluation_only']=evaluate(result,truth,span+float(np.median(np.diff(t)))) if truth else None
        result.update(source=path.stem,old_count=counted['significant_propagation_activity_count'])
        save(output/path.name,result)
        rows.append(dict(source=path.stem,old_count=result['old_count'],
            proposals=result['complete_cycle_proposal_count'],competition=result['competing_cycle_count'],
            final_count=result['experimental_propagation_count'],evaluation=result['evaluation_only']))
    categories={
        'local_once':lambda n:n.endswith('_local_once'),
        'single_strong':lambda n:'_single_' in n and n.endswith(('C2F_0.04','F2C_0.04')),
        'double_15_strong':lambda n:'_double_' in n and n.endswith('_0.04_gap15'),
        'double_20_strong':lambda n:'_double_' in n and n.endswith('_0.04_gap20'),
        'double_10_strong':lambda n:'_double_' in n and n.endswith('_0.04_gap10'),
        'double_15_weak':lambda n:'_double_' in n and n.endswith('_0.01_gap15'),
        'double_20_weak':lambda n:'_double_' in n and n.endswith('_0.01_gap20'),
        'synchronous':lambda n:'synchronous' in n,
        'rigid':lambda n:n.endswith('_rigid') or 'rigid_only' in n}
    aggregates=[]
    for name,select in categories.items():
        subset=[r for r in rows if select(r['source'])]
        record=dict(condition=name,conditions=len(subset),**{k:sum(r[k] for r in subset) for k in ('old_count','proposals','competition','final_count')})
        aggregates.append(record);print(record)
    if len(rows)!=87:raise ValueError('Expected existing 87 conditions')
    save(output/'summary.json',dict(conditions=len(rows),rows=rows,aggregates=aggregates,
         formal_results_changed=False,parameter_sweep=False,truth_used_for_proposals=False,
         conclusion='cycle identity remains non-unique; positive double-wave nonregression failed; stop without tuning'))
    print('TARGET',next(r for r in rows if r['source']=='texture0_double_F2C_0.04_gap10'))


if __name__=='__main__':run()
