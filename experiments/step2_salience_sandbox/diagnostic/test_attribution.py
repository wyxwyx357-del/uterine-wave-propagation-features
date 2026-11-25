"""Diagnostic contracts, including joint matching competition and data isolation."""
import itertools
import json
from pathlib import Path
import pytest
from .attribution_rules import relation, primary_class, stable_identity, maximum_matching, edge_in_some_maximum
from .provenance import phase_kind
from ..diagnostics.dev_propagation import DevDataAccess

def prediction(lo,hi,sections=(0,1,2)):
    return dict(evidence_start_s=lo,evidence_end_s=hi,sections=list(sections),direction='F2C')

TRUTH=dict(truth_event_id='truth',truth_start_s=10.,truth_end_s=20.,truth_center_s=15.,
           truth_sections=[0,1,2,3],truth_direction='C2F')

def test_relation_direction_is_not_a_matcher_gate():
    r=relation(prediction(13,17),TRUTH,2.)
    assert r['matcher_eligible'] and r['direction_relation']=='opposite'

def test_fixed_edge_and_auxiliary_edge_are_distinct():
    r=relation(prediction(24,25),TRUTH,2.)
    assert not r['E_time_fixed'] and r['E_time_auxiliary']
    assert primary_class([r],None)=='B'

@pytest.mark.parametrize('classes,expected',[(('A','E','F','B'),'A'),(('E','F','B'),'E'),(('F','B'),'F'),(('B',),'B'),((),'D')])
def test_priority_and_phase_orthogonality(classes,expected):
    r=dict(matcher_eligible='A' in classes,E_space='E' in classes,E_time_fixed=False,within_15s_path_overlap='B' in classes)
    for phase in ('single_phase','paired_phase','mixed'):
        r['phase_origin']=phase
        assert primary_class([r],dict(direct_time_overlap='F' in classes))==expected

def test_midpoint_boundary_distance_and_interval_gap_are_not_conflated():
    r=relation(prediction(0,12),TRUTH,2.)
    assert r['closest_truth_boundary_distance_s']==4. and r['interval_gap_s']==0.

def test_stable_identity_ignores_node_order_and_dynamic_activity_ids():
    nodes=[dict(wall=0,canonical_source='base:p'),dict(wall=1,canonical_source='child:p')]
    assert stable_identity(nodes)==stable_identity(nodes[::-1])
    assert stable_identity(nodes)!=stable_identity(nodes[:1])

def test_individually_foldable_is_not_jointly_additive():
    left={'a'};right={'b','c'};edges={('a','b'),('a','c')}
    assert all(edge_in_some_maximum(left,right,edges,e,1) for e in edges)
    assert maximum_matching(left,right,edges,edges)['marked_pairs']==1

def test_joint_marked_range_preserves_global_cardinality():
    edges={('a','b'),('a','d'),('c','b'),('c','d')};marked={('a','b')}
    assert maximum_matching({'a','c'},{'b','d'},edges,marked,False)['marked_pairs']==0
    assert maximum_matching({'a','c'},{'b','d'},edges,marked,True)['marked_pairs']==1

def test_assignment_diagnostic_matches_exhaustive_small_graphs():
    left={'a','c'};right={'b','d','f'};possible=list(itertools.product(sorted(left),sorted(right)))
    for mask in range(1<<len(possible)):
        edges={e for i,e in enumerate(possible) if mask&(1<<i)};marked=edges&{('a','b'),('c','d')}
        matchings=[set()]
        for count in (1,2):
            matchings.extend(set(comb) for comb in itertools.combinations(edges,count) if
                             len({a for a,b in comb})==count and len({b for a,b in comb})==count)
        m=max(map(len,matchings));maximum=[s for s in matchings if len(s)==m]
        for want_max in (False,True):
            got=maximum_matching(left,right,edges,marked,want_max)
            expected=(max if want_max else min)(len(s&marked) for s in maximum)
            assert (got['cardinality'],got['marked_pairs'])==(m,expected)

def test_phase_kind_uses_actual_canonical_evidence():
    assert phase_kind(dict(original_measurements=[dict(evidence_kind='paired_phase')],profile={}))=='paired_phase'
    assert phase_kind(dict(original_measurements=[dict(evidence_kind='paired_phase'),dict(evidence_kind='single_phase')],profile={}))=='mixed'
    assert phase_kind(dict(original_measurements=[],profile={0:[dict(sign=1),dict(sign=-1)]}))=='paired_phase'

def test_access_guard_blocks_heldout_and_private_sources(tmp_path):
    work=tmp_path/'work';output=work/'diagnostic';guard=DevDataAccess(work,output,['case_001'])
    guard.check(work/'bases/case_001/rigid.npz')
    guard.check(output/'waveforms/a.npz','wb')
    for path in (work/'STEP2_VAL_RESULTS.json',work/'bases/case_003/rigid.npz',tmp_path/'SELECTED_IDENTITY_LOCAL.json',tmp_path/'patient.csv'):
        with pytest.raises(PermissionError):guard.check(path)

def test_artifact_temporal_coincidence_has_no_hidden_spatial_gate():
    from .attribution_rules import artifact_relation
    p=dict(prediction(1,5),walls=[1]);artifact=dict(start_s=2.,end_s=3.,sections=[20],walls=[0])
    result=artifact_relation(p,artifact)
    assert result['direct_time_overlap'] and result['spatial_overlap_count']==0 and not result['affected_wall_overlap']
    assert primary_class([],result)=='F'

def test_completed_artifacts_reconcile_if_present():
    from .replay import OUTPUT,POINTS
    from ..benchmark_data import load
    path=OUTPUT/'DEV_POSITIVE_FP_ATTRIBUTION_SUMMARY.json'
    if not path.exists():pytest.skip('Runtime Dev diagnostic artifacts not supplied')
    import csv
    summary=load(path)
    with (OUTPUT/'DEV_POSITIVE_FP_EVENT_LEDGER.csv').open(encoding='utf-8-sig',newline='') as f:events=list(csv.DictReader(f))
    with (OUTPUT/'DEV_POSITIVE_FP_BY_TEXTURE.csv').open(encoding='utf-8-sig',newline='') as f:textures=list(csv.DictReader(f))
    for point in POINTS:
        rows=[r for r in events if r['point']==point]
        assert len(rows)==summary['ledgers'][point]['positive_unmatched']
        assert sum(v['count'] for v in summary['attribution'][point]['primary'].values())==len(rows)
        assert len([r for r in textures if r['point']==point])==6
        assert sum(int(r['unmatched']) for r in textures if r['point']==point)==len(rows)
        for row in rows:
            links=json.loads(row['duplicate_links']);rels=json.loads(row['truth_relations'])
            assert (row['primary_class']=='A')==any(r['matcher_eligible'] for r in rels)
            if row['primary_class']=='A':assert links
    increments=load(OUTPUT/'DEV_MAD_INCREMENTAL_FP_ATTRIBUTION.json')
    for name,x in increments.items():
        sets=x['unmatched_sets']
        assert sets['MAD_only']-sets['baseline_only']==x['unmatched_net_change']
        assert sum(x['source_type_counts'].values())==sets['MAD_only']
