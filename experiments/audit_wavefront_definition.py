"""Read-only diagnostic of two saved counting failures; no detector changes."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.audit_wave_evidence import control_arrays
from experiments.test_activity_strength_trial import read,save
from experiments.test_double_event_signals import double_fixture,raw_truth,phase_identity


def audit(name):
    root=ROOT/'outputs'
    counted=read(root/'significant_activity_counts_20261119_v2'/(name+'.json'))
    original=read(root/'global_ownership_trial_20261119_v1'/(name+'.json'))
    salience=read(root/'normalized_activity_trial_20261119_v1'/(name+'.json'))
    is_double='double' in name
    _,valid,t,s,span=control_arrays(root/'p3_outer_ideal_prior_20261117_v2'/
        ('texture0_rigid.npz' if is_double else 'texture0_local_once.npz'))
    truth=raw_truth(double_fixture(t,s,span,'F2C',10.,.04)[0],t) if is_double else None
    labels={}
    for wall in salience['walls']:
        for i,match in enumerate(wall['metrics'].get('after_matches',[])):
            if wall['metrics']['after_supported_truth'][i]:
                labels[f"wall{wall['wall']}/"+match['group_id']]=i+1
    records=[]
    for activity in counted['activities']:
        if not activity['propagation_supported']:continue
        for node_id in activity['source_nodes']:
            wall=int(node_id[4]);eid=node_id.split('/')[1]
            rec=original['walls'][wall]['reconciliation']
            event=next(e for e in rec['event_proposals'] if e['event_proposal_id']==eid)
            entity=rec['source_evidence'][event['canonical_source_entity_id']]
            blocks=[]
            for block in activity['block_evidence']:
                if block['node_id']!=node_id:continue
                ids=block['sections'];a=activity['evidence_start_s'];b=activity['evidence_end_s']
                window=(t>=a)&(t<=b)
                visible=np.flatnonzero(valid[window,wall].all(0)) if window.any() else np.array([],int)
                observable_span=float(np.ptp(s[visible])) if len(visible)>1 else None
                points=[]
                for c in ids:
                    phases=entity['profile'][str(c)]
                    points.append(dict(section=c,s=float(s[c]),arrival_s=entity['arrival'][str(c)],
                        phases=[dict(p,truth_phase_id=(phase_identity(c,p['sign'],p['peak_s'],truth)
                                                     if truth else None)) for p in phases]))
                spatial_span=float(s[ids[-1]]-s[ids[0]])
                blocks.append(dict(block,points=points,spatial_span=spatial_span,
                    observable_sections_full_phase_window=visible.tolist(),
                    observable_span=observable_span,
                    coverage_fraction=spatial_span/observable_span if observable_span else None,
                    native_section_fraction=len(ids)/len(visible) if len(visible) else None,
                    coverage_definition='same-wall valid at every sample in recorded phase footprint; span is envelope, not filled gaps',
                    no_coverage_gate_applied=True))
            records.append(dict(activity_id=activity['activity_id'],source_node=node_id,wall=wall,
                source=event['canonical_source_entity_id'],source_parent_ids=event['source_parent_ids'],
                evidence_kind=event['canonical_measurement']['evidence_kind'],
                arrival_marker=event['canonical_measurement']['arrival_marker'],
                truth_id=labels.get(node_id),start_s=activity['evidence_start_s'],
                end_s=activity['evidence_end_s'],blocks=blocks,
                unresolved_related_nodes=activity['unresolved_related_nodes']))
    return dict(source=name,records=records,identity_comparisons=counted['identity_comparisons'],
                diagnostic_only=True,no_algorithm_changes=True,truth_used_only_for_audit=True)


if __name__=='__main__':
    output=ROOT/'outputs/wavefront_definition_audit_20261119';output.mkdir(exist_ok=True)
    for name in ('texture0_local_once','texture0_double_F2C_0.04_gap10'):
        result=audit(name);save(output/(name+'.json'),result)
        for r in result['records']:
            print(name,r['source_node'],r['source'],r['evidence_kind'],'truth',r['truth_id'],
                  [(b['sections'],round(b['spatial_span'],4),
                    [p['arrival_s'] for p in b['points']]) for b in r['blocks']])
