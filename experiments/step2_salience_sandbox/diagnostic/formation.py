"""Read-only lobe -> phase-pair -> base -> ownership lineage for existing Dev A.

No detector parameters, pairing weights, selection rules, or outputs are changed.
All local phase IDs are scoped to input and wall; child auxiliary IDs are separate.
"""
import csv
import inspect
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK, ROOT, load, save, sha, verify
from ..upstream import stable_hash,pool_hash
from ..matching import match_events
from analysis.mse_pregnancy_association.code import partial_phase_events as partial
from analysis.mse_pregnancy_association.code import global_event_ownership as ownership
from .replay import OUTPUT as PRIOR, POINTS, gate, guarded
from .attribute import write_csv, FLAGS
from .attribution_rules import relation, overlap, consecutive_count
from .waveforms import truth_component

OUTPUT=PRIOR/'paired_phase_formation'

def lobe_key(case,wall,node):
    return f"{case}/w{wall}/c{node['column']}/run{node['run_id']}/seq{node['sequence']}/sign{node['sign']}"

def reconstruct_wall(record,wall):
    """Call production functions, then demand exact equality with saved reports."""
    s=record['shared'];report=s['walls'][wall]['parent_report'];baseline=partial.baseline
    replayed=partial.detect_partial(s['A'][:,wall],s['valid'][:,wall],s['t'],s['s'],noise_floor=s['noise_floor'])
    replayed.pop('filtered_A')
    assert stable_hash(replayed)==stable_hash(report),'Production parent report replay differs'
    filtered,columns,unresolved=partial.observed_lobes(s['A'][:,wall],s['valid'][:,wall],s['t'],report['noise_floor_rate'])
    np.testing.assert_allclose(filtered,s['filtered'][:,wall],rtol=0,atol=0,equal_nan=True)
    ridges=baseline._ridges(columns,s['t'],s['s'])
    compact=[dict(ridge_id=r['id'],sign=r['sign'],sections=[n['column'] for n in r['nodes']],
                  peak_time_s=[n['peak_time_s'] for n in r['nodes']],fit=r['fit']) for r in ridges]
    assert stable_hash(compact)==stable_hash(report['phase_ridges'])
    edges=baseline._phase_edges(ridges)
    measured={k:baseline._measure_pair(e,ridges,filtered,s['t'],s['s'],baseline.RadialWaveConfig(**report['config'])) for k,e in edges.items()}
    selected,ambiguous=baseline._pair_phases(ridges,edges)
    alternatives=[dict(measured[k],selected=k in selected) for k in edges]
    # detect_radial_events adds alternative_phase_pairing to selected events'
    # shared quality_flags list AFTER _measure_pair. The full production report
    # above includes and verifies that annotation; compare numerical pair
    # evidence here before that annotation, without modifying the saved report.
    strip_flags=lambda rows:[{k:v for k,v in row.items() if k!='quality_flags'} for row in rows]
    assert stable_hash(strip_flags(alternatives))==stable_hash(strip_flags(report['alternative_phase_pairs']))
    assert ambiguous==report['ambiguous_phase_ridge_ids']
    original={e['event_id']:e for e in report['events']}
    chosen={tuple(sorted(e['phase_ridge_ids'])) for e in original.values() if e['evidence_kind']=='paired_phase'}
    assert chosen==set(selected)
    assert len(chosen)==sum(e['evidence_kind']=='paired_phase' for e in original.values())==len(selected), 'Event construction copied a selected phase pair'
    chosen_ids=[rid for key in selected for rid in key]
    assert len(chosen_ids)==len(set(chosen_ids)),'Production pairing reused a ridge'
    seen=set()
    for ridge in ridges:
        for n in ridge['nodes']:
            key=lobe_key(record['case_id'],wall,n)
            assert key not in seen,'Production ridge construction reused a raw lobe'
            seen.add(key)
    return dict(wall=wall,columns=columns,ridges=ridges,edges=edges,measured=measured,selected=set(selected),
                original=original,filtered=filtered,unresolved=unresolved,report=report,
                reconciliation=s['walls'][wall]['reconciliation'])

def reference_phases(record,truth,component,wall=0):
    """Known injected component; continuous readout, never a new matching gate."""
    raw,filtered,arrival=component;t=record['shared']['t'];result={}
    for c in range(len(record['shared']['s'])):
        raw_indices=np.flatnonzero(abs(raw[:,0,c])>0)
        for sign in (-1,1):
            half=(t<=arrival[c]) if sign>0 else (t>=arrival[c])
            window=np.flatnonzero(half & (t>=t[raw_indices[0]]) & (t<=t[raw_indices[-1]]) & np.isfinite(filtered[:,wall,c]))
            raw_peak=int(np.argmax(sign*raw[:,wall,c]));filtered_peak=int(window[np.argmax(sign*filtered[window,wall,c])]) if len(window) else None
            result[c,sign]=dict(raw_peak_s=float(t[raw_peak]),filtered_main_peak_s=float(t[filtered_peak]) if filtered_peak is not None else None,
                               raw_support_start_s=float(t[raw_indices[0]]),raw_support_end_s=float(t[raw_indices[-1]]))
    return result

def ridge_main_support(ridge,refs,dt):
    return [n['column'] for n in ridge['nodes'] if refs[n['column'],n['sign']]['filtered_main_peak_s'] is not None and
            n['start_s']-dt/2<=refs[n['column'],n['sign']]['filtered_main_peak_s']<=n['end_s']+dt/2]

def edge_main_support(key,trace,refs,dt):
    edge=trace['edges'][key]
    support=set(edge['columns'])
    for rid in key:support &= set(ridge_main_support(trace['ridges'][rid],refs,dt))
    return sorted(support)

def parent_events(entity,trace):
    # A child has auxiliary phase IDs: inspect its original parent, without
    # treating the child profile or local IDs as original FFT-ridge evidence.
    return [trace['original'][p] for p in entity['source_parents']]

def source_trace(record,trace,entity_id):
    entity=trace['reconciliation']['source_evidence'][entity_id];wall=trace['wall']
    originals=parent_events(entity,trace);parents=[]
    for e in originals:
        ids=e['phase_ridge_ids'];sections=set(e['sections'])
        lobes=[dict(lobe_id=lobe_key(record['case_id'],wall,n),ridge_id=rid,**n)
               for rid in ids for n in trace['ridges'][rid]['nodes'] if n['column'] in sections]
        key=tuple(sorted(ids))
        parents.append(dict(event_id=e['event_id'],evidence_kind=e['evidence_kind'],phase_ridge_ids=ids,
            selected_in_production_pairing=key in trace['selected'],phase_pairing_ambiguous=e['phase_pairing_ambiguous'],
            association_weight=e.get('association_weight'),leading_polarity=e['leading_polarity'],
            first_phase_arrival_s=e['first_phase_arrival_s'],second_phase_arrival_s=e['second_phase_arrival_s'],
            sections=e['sections'],start_s=e['start_s'],end_s=e['end_s'],arrival_time_s=e['arrival_time_s'],
            classification=e['classification'],lobes=lobes))
    return dict(case_id=record['case_id'],wall=wall,canonical_source_entity_id=entity_id,
        source_kind=entity['source_kind'],source_parent_ids=entity['source_parents'],original_parents=parents,
        canonical_phase_profile=entity['profile'],canonical_measurement=entity['canonical_measurement'],
        phase_id_scope='global_original_filtered_ridges' if entity['source_kind']=='existing' else 'parent_local_auxiliary_not_global_ridge_IDs',
        source_to_final=trace['reconciliation']['source_to_final'].get(entity_id),
        phase_ridge_provenance=entity['phase_ridge_provenance'])

def compare_parent_pairs(record,trace,a,b,refs):
    """Describe the ACTUAL selected pairs and their preexisting alternatives."""
    ai=set(a['phase_ridge_ids']);bi=set(b['phase_ridge_ids']);both_paired=len(ai)==len(bi)==2
    at={p['lobe_id'] for p in a['lobes']};bt={p['lobe_id'] for p in b['lobes']}
    bridges=[]
    for x in ai:
        for y in bi-ai:
            key=tuple(sorted((x,y)))
            if key not in trace['edges']:continue
            edge=trace['measured'][key];support=edge_main_support(key,trace,refs,record['shared']['dt'])
            selected_weight=sum(trace['measured'][k]['association_weight'] for k in {tuple(sorted(ai)),tuple(sorted(bi))} if k in trace['selected'])
            bridges.append(dict(phase_ridge_ids=edge['phase_ridge_ids'],selected=key in trace['selected'],
                leading_polarity=edge['leading_polarity'],sections=edge['sections'],classification=edge['classification'],
                association_weight=edge['association_weight'],main_reference_shared_sections=support,
                main_reference_contiguous_sections=consecutive_count(support),
                connects_main_positive_and_negative=bool(edge['leading_polarity']==1 and consecutive_count(support)>=3),
                two_actual_selected_pairs_weight=selected_weight,
                local_exchange_objective_change=edge['association_weight']-selected_weight,
                comparison_scope='Remove the two observed selected edges and add this one bridge; not a counterfactual rerun or retuning'))
    seconds,iou=overlap(a['start_s'],a['end_s'],b['start_s'],b['end_s'])
    contained=(a['start_s']<=b['start_s'] and a['end_s']>=b['end_s']) or (b['start_s']<=a['start_s'] and b['end_s']>=a['end_s'])
    partition=both_paired and not ai&bi and a['selected_in_production_pairing'] and b['selected_in_production_pairing'] and any(
        x['connects_main_positive_and_negative'] and not x['selected'] for x in bridges)
    return dict(parent_a=a['event_id'],parent_b=b['event_id'],same_parent=a['event_id']==b['event_id'],
        both_original_parents_paired=both_paired,
        both_distinct_selected_pairs=both_paired and ai!=bi and a['selected_in_production_pairing'] and b['selected_in_production_pairing'],
        shared_ridge_ids=sorted(ai&bi),shared_original_lobe_ids=sorted(at&bt),
        original_lobe_intersection_count=len(at&bt),original_lobe_union_count=len(at|bt),
        shared_sections=sorted(set(a['sections'])&set(b['sections'])),
        phase_order_a=[trace['ridges'][r]['sign'] for r in a['phase_ridge_ids']],
        phase_order_b=[trace['ridges'][r]['sign'] for r in b['phase_ridge_ids']],
        temporal_relation='nested_or_equal' if contained else 'overlapping' if seconds>0 else 'adjacent_or_separated',
        interval_overlap_s=seconds,interval_IoU=iou,interval_gap_s=max(a['start_s']-b['end_s'],b['start_s']-a['end_s'],0.),
        main_pair_partitioned_into_flanking_pairs=partition,existing_bridge_pairs=bridges,
        earliest_multiple_paired_parent_stage='phase_pairing' if both_paired and ai!=bi and a['selected_in_production_pairing'] and b['selected_in_production_pairing'] else 'not_two_distinct_original_paired_events')

def compare_sources(record,point,truth_id,trace,a_id,b_id,source_cache):
    a=source_cache[record['case_id'],trace['wall'],a_id];b=source_cache[record['case_id'],trace['wall'],b_id]
    rec=trace['reconciliation'];ae=rec['source_evidence'][a_id];be=rec['source_evidence'][b_id]
    evidence=ownership.compare_identity(ae,be,sample_dt_s=record['shared']['dt'],measurement_span_s=record['shared']['span'])
    previous=[e for e in rec['pairwise_evidence'] if {e['source_a'],e['source_b']}=={a_id,b_id}]
    if previous:
        prior={k:v for k,v in previous[0].items() if k not in ('source_a','source_b')}
        assert stable_hash(prior)==stable_hash(evidence)
    refs=trace['references'][truth_id]
    parent_relations=[compare_parent_pairs(record,trace,pa,pb,refs) for pa in a['original_parents'] for pb in b['original_parents']]
    final_a=rec['source_to_final'].get(a_id);final_b=rec['source_to_final'].get(b_id)
    originals_distinct=any(x['both_distinct_selected_pairs'] for x in parent_relations)
    final_original=a['source_kind']==b['source_kind']=='existing'
    return dict(case_id=record['case_id'],texture_id=record['source']['texture_id'],scenario_id=record['scenario']['scenario'],
        point=point,truth_id=truth_id,wall=trace['wall'],unmatched_canonical_source=a_id,TP_canonical_source=b_id,
        unmatched_source_kind=a['source_kind'],TP_source_kind=b['source_kind'],
        final_proposal_a=final_a,final_proposal_b=final_b,remained_separate_after_reconcile=final_a!=final_b,
        both_final_representations_are_original_bases=final_original,
        multiple_paired_parents_before_reconcile=originals_distinct,
        exact_final_phase_pairs_already_before_reconcile=originals_distinct and final_original,
        parent_pair_relations=parent_relations,ownership_identity_comparison=evidence,
        ownership_duplicate_criteria_pass=evidence['relation'] in ('duplicate','partial_duplicate'),
        earliest_original_parent_stage='phase_pairing' if originals_distinct else 'not_two_distinct_original_paired_events',
        source_trace_keys=[f"{record['case_id']}|w{trace['wall']}|{x}" for x in (a_id,b_id)])

def source_refs():
    functions=[partial.observed_lobes,partial.baseline._ridges,partial.baseline._phase_edges,partial.baseline._measure_pair,
               partial.baseline._pair_phases,partial.baseline.detect_radial_events,partial.propose_groups,
               ownership.build_entities,ownership.compare_identity,ownership.reconcile]
    return [dict(file=str(Path(inspect.getsourcefile(f)).resolve().relative_to(ROOT)),function=f.__name__,
                 first_line=inspect.getsourcelines(f)[1],last_line=inspect.getsourcelines(f)[1]+len(inspect.getsourcelines(f)[0])-1,
                 sha256=sha(inspect.getsourcefile(f)),source=inspect.getsource(f)) for f in functions]

def read_prior():
    audit=load(PRIOR/'DEV_POSITIVE_FP_READ_ONLY_AUDIT.json')
    assert sha(PRIOR/'replay.pkl')==audit['replay_cache_sha256']
    assert all(sha(PRIOR/name)==h for name,h in audit['output_sha256'].items())
    with (PRIOR/'replay.pkl').open('rb') as f:data=pickle.load(f)
    with (PRIOR/'DEV_POSITIVE_FP_EVENT_LEDGER.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['primary_class']=='A']
    import json
    for row in rows:
        for key in ('source_chain','duplicate_links','source_nodes','truth_relations','walls'):row[key]=json.loads(row[key])
    assert Counter(r['point'] for r in rows)=={'baseline':100,'mad_low':120,'mad_high':110}
    return audit,data,rows

def run():
    OUTPUT.mkdir(exist_ok=True)
    protocol,access=guarded(PRIOR);before={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)}
    old_audit,data,targets=read_prior();assert stable_hash(protocol)==stable_hash(data['protocol'])
    records={r['case_id']:r for r in data['records']};ledgers=gate(data['records'])
    prior_hashes={f.name:sha(f) for f in PRIOR.glob('DEV_*')}
    traces={};sources={};pair_rows=[];event_rows=[];upstream_rows=[];truth_rows=[];lobe_rows=[]
    for case_id in sorted({r['case_id'] for r in targets}):
        record=records[case_id]
        components={t['truth_event_id']:truth_component(record,t) for t in record['truth']}
        for wall in (0,1):
            trace=reconstruct_wall(record,wall)
            trace['references']={t['truth_event_id']:reference_phases(record,t,components[t['truth_event_id']],wall) for t in record['truth']}
            trace['components']=components
            traces[case_id,wall]=trace
            for entity_id in trace['reconciliation']['source_evidence']:
                sources[case_id,wall,entity_id]=source_trace(record,trace,entity_id)
            for column in trace['columns']:
                for lobe in column:
                    lobe_rows.append(dict(case_id=case_id,wall=wall,lobe_id=lobe_key(case_id,wall,lobe),**lobe))
            for event in trace['report']['events']:
                if event['evidence_kind']!='paired_phase':continue
                basic=dict(evidence_start_s=event['start_s'],evidence_end_s=event['end_s'],sections=event['sections'],direction=event['classification'])
                relations=[relation(basic,t,record['shared']['span']) for t in record['truth']]
                key=tuple(sorted(event['phase_ridge_ids']))
                upstream_rows.append(dict(case_id=case_id,texture_id=record['source']['texture_id'],wall=wall,
                    event_id=event['event_id'],phase_ridge_ids=event['phase_ridge_ids'],selected=key in trace['selected'],
                    first_phase_arrival_s=event['first_phase_arrival_s'],second_phase_arrival_s=event['second_phase_arrival_s'],
                    leading_polarity=event['leading_polarity'],start_s=event['start_s'],end_s=event['end_s'],sections=event['sections'],
                    classification=event['classification'],association_weight=event['association_weight'],
                    alternative_pairing=event['phase_pairing_ambiguous'],truth_relations=relations))
            # Count ORIGINAL selected pairs per truth before salience. These
            # counts use raw event bounds, not final retained-profile bounds.
            for truth in record['truth']:
                refs=trace['references'][truth['truth_event_id']]
                selected_eligible=[r for r in upstream_rows if r['case_id']==case_id and r['wall']==wall and
                                   any(rel['truth_id']==truth['truth_event_id'] and rel['matcher_eligible'] for rel in r['truth_relations'])]
                main_edges=[dict(phase_ridge_ids=trace['measured'][key]['phase_ridge_ids'],selected=key in trace['selected'],
                    main_reference_sections=support,association_weight=trace['measured'][key]['association_weight'])
                    for key in trace['edges'] for support in [edge_main_support(key,trace,refs,record['shared']['dt'])]
                    if trace['measured'][key]['leading_polarity']==1 and consecutive_count(support)>=3]
                truth_rows.append(dict(case_id=case_id,texture_id=record['source']['texture_id'],wall=wall,truth_id=truth['truth_event_id'],
                    original_selected_paired_events_eligible=len(selected_eligible),original_event_ids=[r['event_id'] for r in selected_eligible],
                    main_positive_negative_pair_edges=main_edges))
    print('Verified production lobes, ridges, pairing weights and selected pairs for',len(traces),'Dev walls',flush=True)
    for target in targets:
        record=records[target['case_id']];point=record['points'][target['point']]
        nodes={n['node_id']:n for n in point['evaluation']['nodes']}
        comparisons=[];a1_cross=[]
        for link in target['duplicate_links']:
            for node_a in target['source_nodes']:
                for node_b in link['TP_source_nodes']:
                    a=nodes[node_a];b=nodes[node_b]
                    if a['wall']!=b['wall']:
                        if link['subtype']=='A1':a1_cross.append(dict(truth_id=link['truth_id'],wall_a=a['wall'],wall_b=b['wall'],canonical_a=a['canonical_source'],canonical_b=b['canonical_source']))
                        continue
                    row=compare_sources(record,target['point'],link['truth_id'],traces[target['case_id'],a['wall']],a['canonical_source'],b['canonical_source'],sources)
                    row['target_event_key']=target['event_key'];row['link_subtype']=link['subtype'];pair_rows.append(row);comparisons.append(row)
        parent_relations=[rel for row in comparisons for rel in row['parent_pair_relations']]
        event_rows.append(dict(event_key=target['event_key'],case_id=target['case_id'],texture_id=target['texture_id'],scenario_id=target['scenario_id'],
            point=target['point'],subtype=target['subtype'],stable_source_identity=target['stable_source_identity'],
            phase_origin=target['phase_origin'],split_origin=target['split_origin'],
            unmatched_source_trace_keys=[f"{record['case_id']}|w{nodes[n]['wall']}|{nodes[n]['canonical_source']}" for n in target['source_nodes']],
            TP_source_trace_keys=sorted({f"{record['case_id']}|w{nodes[n]['wall']}|{nodes[n]['canonical_source']}" for l in target['duplicate_links'] for n in l['TP_source_nodes']}),
            same_wall_source_comparisons=len(comparisons),A1_cross_wall_links=a1_cross,
            original_distinct_paired_parents_before_reconcile=any(r['both_distinct_selected_pairs'] for r in parent_relations),
            exact_final_base_pairs_before_reconcile=any(r['exact_final_phase_pairs_already_before_reconcile'] for r in comparisons),
            matched_TP_has_child_source=any(r['TP_source_kind']=='child' for r in comparisons),
            any_original_lobe_reuse=any(r['original_lobe_intersection_count'] for r in parent_relations),
            main_pair_partitioned_into_flanking_pairs=any(r['main_pair_partitioned_into_flanking_pairs'] for r in parent_relations),
            ownership_would_merge_under_existing_rules=any(r['ownership_duplicate_criteria_pass'] for r in comparisons),
            comparisons=comparisons))
    summary={name:summarize_events([r for r in event_rows if r['point']==name],[r for r in pair_rows if r['point']==name]) for name in POINTS}
    print('Formation findings:',summary,flush=True)
    write_csv(OUTPUT/'A_DUPLICATE_FORMATION_LEDGER.csv',event_rows)
    write_csv(OUTPUT/'A2_SAME_WALL_PAIR_TRACE.csv',pair_rows)
    write_csv(OUTPUT/'UPSTREAM_PHASE_PAIR_LEDGER.csv',upstream_rows)
    write_csv(OUTPUT/'ORIGINAL_LOBE_LEDGER.csv',lobe_rows)
    write_csv(OUTPUT/'TRUTH_UPSTREAM_PHASE_PAIRS.csv',truth_rows)
    save(OUTPUT/'SOURCE_LINEAGE.json',{'|'.join(map(str,k)):v for k,v in sources.items()})
    wall_graphs=[dict(case_id=case,wall=wall,selected_pairs=sorted(trace['selected']),
        ridges=trace['ridges'],edges=[dict(ridge_key=key,selected=key in trace['selected'],**trace['measured'][key]) for key in trace['edges']],
        proposed_groups_before_reconcile=trace['report']['proposed_events'],
        source_to_final=trace['reconciliation']['source_to_final'],
        ownership_matching_rules=trace['reconciliation']['matching_rules']) for (case,wall),trace in traces.items()]
    save(OUTPUT/'PRODUCTION_PAIRING_GRAPHS.json',wall_graphs)
    texture_summary={
        name:{texture:summarize_events([r for r in event_rows if r['point']==name and r['texture_id']==texture],
            [r for r in pair_rows if r['point']==name and r['texture_id']==texture]) for texture in sorted({r['texture_id'] for r in event_rows})} for name in POINTS}
    from .formation_figures import produce_figures,component_readout
    component_evidence=component_readout(records,traces,pair_rows)
    write_csv(OUTPUT/'PHASE_LOBE_COMPONENT_READOUT.csv',component_evidence)
    component_summary={}
    for role in ('main','peripheral'):
        rows=[r for r in component_evidence if r['role_relative_to_unselected_main_pair']==role]
        fractions=[r['signed_component_fraction'] for r in rows if r['signed_component_fraction'] is not None]
        component_summary[role]=dict(native_lobe_samples=len(rows),distinct_input_walls=len({(r['case_id'],r['wall']) for r in rows}),
            truth_raw_component_zero_count=sum(r['truth_raw_component_zero'] for r in rows),
            truth_raw_component_zero_fraction=sum(r['truth_raw_component_zero'] for r in rows)/len(rows) if rows else None,
            filtered_truth_sign_agreement_count=sum(r['filtered_truth_component_sign_agrees'] for r in rows),
            filtered_truth_component_fraction_median=float(np.median(fractions)) if fractions else None,
            filtered_truth_component_fraction_P10_P90=np.percentile(fractions,[10,90]).tolist() if fractions else [],
            statistical_unit_note='Descriptive native lobe samples deduplicated by input/wall/truth/ridge/section/role, not independent patients or confidence intervals')
    figures=produce_figures(records,traces,sources,pair_rows,OUTPUT)
    # Save a single array pack per traced input, including ALL truth components.
    waveform_dir=OUTPUT/'waveforms';waveform_dir.mkdir(exist_ok=True)
    waveform_files=[]
    for case in sorted({case for case,wall in traces}):
        record=records[case];shared=record['shared'];components=traces[case,0]['components']
        arrays={key:shared[key] for key in ('A','filtered','valid','t','s')}
        arrays.update(interval_start_s=record['interval_start'],interval_end_s=record['interval_end'])
        for tid,(raw,filtered,arrival) in components.items():arrays[tid+'_raw']=raw;arrays[tid+'_filtered']=filtered;arrays[tid+'_arrival']=arrival
        from .attribute import filtered_runs_check
        error=filtered_runs_check(record,shared['A']-sum(c[0] for c in components.values()),components)
        path=waveform_dir/(case.replace('/','__')+'.npz');np.savez_compressed(path,**arrays)
        waveform_files.append(dict(case_id=case,path=str(path.relative_to(OUTPUT)),truth=record['truth'],linear_component_reconstruction_max_abs_error=error))
    save(OUTPUT/'FORMATION_SUMMARY.json',dict(ledgers=ledgers,summary=summary,by_texture=texture_summary,figures=figures,waveforms=waveform_files,
        analytical_component_readout=component_summary,
        verified_input_walls=len(traces),source_references=source_refs(),
        main_phase_reference='Known injected truth component, through SAME production filter/valid runs. Positive maximum before deformation center, negative minimum after center within its original A support. Actual lobe interval containment, no new amplitude/fit veto.',
        limitation='A precedes B in prior primary classification. Small primary B count does not exclude filter-related lobe structure within A. Structural phase-pair localization is not a filter-intervention causal proof.',
        localization=dict(first_independent_paired_event_decision='radial_events._pair_phases',
            emission='detect_radial_events emits one event per selected edge; no extra copy at emission',
            ownership='Existing compare_identity finds distinct local phase identities, so reconcile retains them. No merge-eligible missed duplicate in the traced same-wall pairs.',
            child_scope='Original parent-pair evidence is separated from final auxiliary split-child phases; it does not equate their lobe IDs.',
            causal_limit='This is an event-representation finding. Filtered peripheral lobes can participate in A; the audit does not establish the effect of changing the filter or justify a particular replacement pairing rule.'),
        next_action='Stop for review; no algorithm changes'))
    assert before=={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)}
    assert prior_hashes=={f.name:sha(f) for f in PRIOR.glob('DEV_*')}
    assert all(pool_hash(record['shared'])==record['shared']['pool_hash'] for record in records.values()),'Cached upstream was mutated in memory'
    verify(WORK)
    save(OUTPUT/'FORMATION_READ_ONLY_AUDIT.json',dict(**{k:False for k in FLAGS},
        previous_artifacts_unchanged=True,previous_input_hashes_unchanged=True,
        frozen_replay_sha256=sha(PRIOR/'replay.pkl'),source_sha256=verify(WORK)['protected_source_sha256'],
        diagnostic_source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        all_three_ledger_gates_passed=True,all_traced_walls_exact_reconstruction=True,
        no_original_ridge_or_lobe_reuse_in_selected_pairs=True,data_reads=sorted(access.accesses),
        cached_upstream_unchanged_in_memory=True,
        output_sha256={str(f.relative_to(OUTPUT)):sha(f) for f in OUTPUT.rglob('*') if f.is_file() and
                       f.name not in ('FORMATION_READ_ONLY_AUDIT.json','FORMATION_TESTS.xml')}))

def summarize_events(events,pairs):
    a2=[r for r in events if r['subtype']=='A2']
    return dict(A_count=len(events),A1_count=sum(r['subtype']=='A1' for r in events),A2_count=len(a2),
        A2_phase_origin=dict(Counter(r['phase_origin'] for r in a2)),
        A2_with_two_distinct_original_paired_parents=sum(r['original_distinct_paired_parents_before_reconcile'] for r in a2),
        A2_exact_final_bases_already_before_reconcile=sum(r['exact_final_base_pairs_before_reconcile'] for r in a2),
        A2_matched_TP_has_child_source=sum(r['matched_TP_has_child_source'] for r in a2),
        A2_original_lobe_reuse=sum(r['any_original_lobe_reuse'] for r in a2),
        A2_main_pair_partitioned_into_flanking_pairs=sum(r['main_pair_partitioned_into_flanking_pairs'] for r in a2),
        A2_ownership_duplicate_criteria_pass=sum(r['ownership_would_merge_under_existing_rules'] for r in a2),
        same_wall_comparisons=len(pairs),ownership_relations=dict(Counter(r['ownership_identity_comparison']['relation'] for r in pairs)),
        original_parent_stages=dict(Counter(r['earliest_original_parent_stage'] for r in pairs)),
        parent_temporal_relations=dict(Counter(rel['temporal_relation'] for r in pairs for rel in r['parent_pair_relations'])))

if __name__=='__main__':run()
