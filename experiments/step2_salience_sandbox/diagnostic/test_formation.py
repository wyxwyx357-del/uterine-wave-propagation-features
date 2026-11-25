"""Formation audit invariants; core pairing behavior uses production functions."""
import csv
import json
import numpy as np
from .formation import partial,lobe_key,OUTPUT
from ..benchmark_data import load,sha

def test_production_pairing_can_prefer_two_flanks_to_one_stronger_central_pair():
    # This is the actual production assignment, not a second implementation.
    # The central edge is best individually, but its two alternatives sum higher.
    ridges=[dict(sign=-1),dict(sign=1),dict(sign=-1),dict(sign=1)]
    edges={(0,1):dict(association_weight=11.),(1,2):dict(association_weight=13.),(2,3):dict(association_weight=12.)}
    selected,ambiguous=partial.baseline._pair_phases(ridges,edges)
    assert set(selected)=={(0,1),(2,3)}
    assert len({rid for pair in selected for rid in pair})==4
    assert set(ambiguous)=={1,2}

def test_production_pairing_preserves_central_pair_when_its_weight_is_larger_than_both_flanks():
    ridges=[dict(sign=-1),dict(sign=1),dict(sign=-1),dict(sign=1)]
    edges={(0,1):dict(association_weight=1.),(1,2):dict(association_weight=13.),(2,3):dict(association_weight=2.)}
    selected,_=partial.baseline._pair_phases(ridges,edges)
    assert set(selected)=={(1,2)}

def test_lobe_identity_includes_input_wall_section_run_and_sequence():
    n=dict(column=1,run_id=0,sequence=2,sign=1)
    key=lobe_key('texture/scenario',0,n)
    assert key!=lobe_key('texture/scenario',1,n)
    assert key!=lobe_key('texture/other',0,n)
    assert key!=lobe_key('texture/scenario',0,dict(n,run_id=1))

def test_known_truth_reference_uses_the_actual_wall_filter_support():
    from .formation import reference_phases,ridge_main_support
    t=np.arange(5.);raw=np.ones((5,2,1));filtered=np.zeros_like(raw)
    filtered[1,0,0]=4.;filtered[2,1,0]=4.;filtered[4,:,:]=-4.
    component=raw,filtered,np.array([2.])
    record=dict(shared=dict(t=t,s=np.array([.5])))
    a=reference_phases(record,{},component,0);b=reference_phases(record,{},component,1)
    assert a[0,1]['filtered_main_peak_s']==1.
    assert b[0,1]['filtered_main_peak_s']==2.
    ridge=dict(nodes=[dict(column=0,sign=1,start_s=.9,end_s=1.1)])
    assert ridge_main_support(ridge,a,.1)==[0] and ridge_main_support(ridge,b,.1)==[]

def test_formation_artifacts_are_complete_and_provenance_scopes_remain_explicit():
    s=load(OUTPUT/'FORMATION_SUMMARY.json')
    expected={'baseline':(100,73,70,3),'mad_low':(120,90,81,2),'mad_high':(110,85,81,1)}
    with (OUTPUT/'A_DUPLICATE_FORMATION_LEDGER.csv').open(encoding='utf-8-sig',newline='') as f:events=list(csv.DictReader(f))
    with (OUTPUT/'A2_SAME_WALL_PAIR_TRACE.csv').open(encoding='utf-8-sig',newline='') as f:pairs=list(csv.DictReader(f))
    for name,(A,A2,base,child) in expected.items():
        result=s['summary'][name]
        assert (result['A_count'],result['A2_count'],result['A2_exact_final_bases_already_before_reconcile'],result['A2_matched_TP_has_child_source'])==(A,A2,base,child)
        assert len([r for r in events if r['point']==name])==A
        assert result['A2_original_lobe_reuse']==0
        assert result['A2_ownership_duplicate_criteria_pass']==0
        assert len(s['by_texture'][name])==6
        assert sum(v['A2_count'] for v in s['by_texture'][name].values())==A2
    for row in pairs:
        relations=json.loads(row['parent_pair_relations'])
        assert json.loads(row['ownership_identity_comparison'])['relation']=='distinct'
        for r in relations:
            assert not r['shared_original_lobe_ids']
            if r['main_pair_partitioned_into_flanking_pairs']:
                bridges=[e for e in r['existing_bridge_pairs'] if e['connects_main_positive_and_negative']]
                assert bridges and all(not e['selected'] and e['local_exchange_objective_change']<0 for e in bridges)

def test_formation_audit_hashes_and_scope_flags():
    a=load(OUTPUT/'FORMATION_READ_ONLY_AUDIT.json')
    from .attribute import FLAGS
    assert all(a[k] is False for k in FLAGS)
    assert all(sha(OUTPUT/name)==h for name,h in a['output_sha256'].items())
    assert a['previous_artifacts_unchanged'] and a['all_traced_walls_exact_reconstruction']
