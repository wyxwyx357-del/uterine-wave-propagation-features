"""Dev-only attribution of three PREEXISTING points; no fitted parameters."""
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK, ROOT, load, save, sha, verify
from ..upstream import stable_hash
from .replay import OUTPUT, POINTS, COUNT, guarded, gate
from .attribution_rules import ORDER, relation, artifact_definition, artifact_relation, primary_class, distribution
from .provenance import enrich, graph, duplicate_links, graph_accounting
from .waveforms import truth_component, metrics, finalize_recurrence, filter_sources, make_plot
from .incremental import compare_case, summarize as incremental_summary

FLAGS=('Val_read','pregnancy_outcome_read','clinical_AUC_calculated','production_changed','threshold_changed',
       'truth_matcher_changed','filter_changed','split_changed','cross_wall_identity_changed','Step3_started','400plus_cohort_run')
DIMENSIONS=('phase_origin','split_origin','wall_status')

def write_csv(path,rows):
    from analysis.mse_pregnancy_association.code.measurement.io import jsonable
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(jsonable(v),ensure_ascii=False,allow_nan=False,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})

def composition(events):
    n=len(events)
    return dict(n=n,primary={c:dict(count=sum(p['primary_class']==c for p in events),percentage=100*sum(p['primary_class']==c for p in events)/n if n else 0.) for c in ORDER},
        provenance={field:dict(Counter(p[field] for p in events)) for field in DIMENSIONS},
        provenance_by_primary={c:{field:dict(Counter(p[field] for p in events if p['primary_class']==c)) for field in DIMENSIONS} for c in ORDER},
        A_subtypes=dict(Counter(p['subtype'] for p in events if p['primary_class']=='A')),
        provenance_rates_by_primary={c:dict(n=len(rows),single_phase_fraction=sum(p['phase_origin']=='single_phase' for p in rows)/len(rows) if rows else None,
            paired_phase_fraction=sum(p['phase_origin']=='paired_phase' for p in rows)/len(rows) if rows else None,
            split_child_fraction=sum(p['split_origin']=='split_child' for p in rows)/len(rows) if rows else None,
            shared_parent_with_TP_fraction=sum(p['shared_parent_with_TP'] for p in rows)/len(rows) if rows else None,
            cross_wall_unresolved_fraction=sum(p['wall_status']=='single_wall_unresolved_cross_wall' for p in rows)/len(rows) if rows else None)
            for c in ORDER for rows in [[p for p in events if p['primary_class']==c]]})

def groups(s):
    name=s['scenario'];result=[]
    if s['family']=='single':result.append('single_'+('structured_weak' if name.startswith('structured_weak') else name.split('_')[0]))
    if s['family']=='coexist':result.append('coexist_'+str(s['ratio'])+':1')
    if s['family']=='artifact' and s['amplitude'] is not None:result.append('artifact_'+s['artifact']+' + wave')
    if s['base']=='brightness':result.append('brightness-related')
    if s['noise'] in (3,8):result.append('high-noise_'+str(s['noise'])+'x')
    if s['family']=='duration':result.append('duration_'+str(int(s['duration'])))
    if name=='medium_C2F':result.append('baseline-duration reference (medium_C2F)')
    if s['amplitude'] is None:result.append('negative controls')
    return result

def aggregate_rows(records,events,truth_rows,selector,label,kind):
    out=[]
    for name in POINTS:
        selected=[r for r in records if selector(r)];ids={r['case_id'] for r in selected}
        e=[p for p in events if p['point']==name and p['case_id'] in ids]
        t=[p for p in truth_rows if p['point']==name and p['case_id'] in ids]
        negatives={r['shared']['pool_hash']:r for r in selected if not r['truth']}
        truth=sum(len(r['truth']) for r in selected)
        out.append(dict(point=name,group_type=kind,group=label,cases=len(selected),truth_count=truth,
            TP=sum(r['points'][name]['match']['TP'] for r in selected),ledger_FP=sum(r['points'][name]['match']['FP'] for r in selected),
            FN=sum(r['points'][name]['match']['FN'] for r in selected),
            negative_control_v2_min=sum(r['points'][name]['evaluation']['result'][COUNT+'_min'] for r in negatives.values()),
            negative_control_v2_max=sum(r['points'][name]['evaluation']['result'][COUNT+'_max'] for r in negatives.values()),
            unmatched=len(e),unmatched_per_truth=len(e)/truth if truth else None,
            **{c:sum(p['primary_class']==c for p in e) for c in ORDER},
            **{field:dict(Counter(p[field] for p in e)) for field in DIMENSIONS},
            matcher_eligible_count=distribution([p['matcher_eligible_count'] for p in t]),
            within_15s_path_overlap_count=distribution([p['within_15s_path_overlap_count'] for p in t])))
    return out

def paired_scenarios(records):
    out=[]
    for texture in sorted({r['source']['texture_id'] for r in records}):
        lookup={r['scenario']['scenario']:r for r in records if r['source']['texture_id']==texture}
        for name in POINTS:
            for weaker in ('medium','structured_weak'):
                single=lookup[weaker+'_C2F'];clear=lookup['clear_C2F']
                for ratio in (2,3,5):
                    coexist=lookup[f'coexist_{weaker}_{ratio}']
                    out.append(dict(texture_id=texture,point=name,weaker_class=weaker,ratio=ratio,
                        single_scenario=single['scenario']['scenario'],coexist_scenario=coexist['scenario']['scenario'],
                        single_unmatched=single['points'][name]['match']['FP'],clear_single_unmatched=clear['points'][name]['match']['FP'],
                        coexist_unmatched=coexist['points'][name]['match']['FP'],
                        change=coexist['points'][name]['match']['FP']-single['points'][name]['match']['FP'],
                        zero_controls={s:r['points'][name]['match']['FP'] for s,r in lookup.items() if r['scenario']['family']=='zero'}))
    return out

def choose_plots(events,all_predictions,records):
    by_case={r['case_id']:r for r in records};textures=sorted({p['texture_id'] for p in events})
    selectors=[('normal_single_TP',lambda p:p['matched_TP'] and by_case[p['case_id']]['scenario']['family']=='single' and
                sum(q['propagation_supported'] for q in all_predictions[p['case_id'],p['point']])==1),
        ('A1',lambda p:p.get('subtype')=='A1'),('A2',lambda p:p.get('subtype')=='A2'),
        ('B',lambda p:p.get('primary_class')=='B'),('F',lambda p:p.get('primary_class')=='F'),
        ('BD_single_phase',lambda p:p.get('primary_class') in ('B','D') and p['phase_origin']=='single_phase'),
        ('BD_split_child',lambda p:p.get('primary_class') in ('B','D') and p['split_origin']=='split_child'),
        ('D',lambda p:p.get('primary_class')=='D'),('E',lambda p:p.get('primary_class')=='E')]
    candidates=[p for rows in all_predictions.values() for p in rows if p['propagation_supported']]
    result=[];folder=OUTPUT/'plots';folder.mkdir(exist_ok=True)
    for i,(label,select) in enumerate(selectors):
        pool=[p for p in candidates if select(p)];rank=textures[i%len(textures):]+textures[:i%len(textures)]
        pool.sort(key=lambda p:(rank.index(p['texture_id']),p['case_id'],list(POINTS).index(p['point']),p['stable_source_identity']))
        if not pool:result.append(dict(category=label,available=0));continue
        p=pool[0];path=folder/(label+'.png')
        make_plot(p,by_case[p['case_id']],all_predictions[(p['case_id'],p['point'])],path)
        result.append(dict(category=label,available=len(pool),event_key=p['event_key'],file=str(path.relative_to(OUTPUT))))
    return result

def run():
    p,access=guarded();before={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)}
    with (OUTPUT/'replay.pkl').open('rb') as f:replayed=pickle.load(f)
    assert stable_hash(p)==stable_hash(replayed['protocol'])
    records=replayed['records'];ledgers=gate(records) # mandatory BEFORE ANY classification
    gate_record=load(OUTPUT/'REPLAY_GATE.json');assert before==gate_record['input_sha256']
    events=[];truth_rows=[];cross=[];all_predictions={};waveform_files=[]
    for r in records:
        if not r['truth']:continue
        artifact=artifact_definition(r['source'],r['scenario'],r['shared'],p,r['interval_start'],r['interval_end'])
        components={t['truth_event_id']:truth_component(r,t) for t in r['truth']}
        save_waveform=False
        for name,(scorer,tau,_) in POINTS.items():
            point=r['points'][name];enriched=enrich(r,name);g=graph(point)
            for pred in enriched:
                pred.update(case_id=r['case_id'],texture_id=r['source']['texture_id'],scenario_id=r['scenario']['scenario'],
                            point=name,scorer=scorer,threshold=tau,measurement_span_s=r['shared']['span'],
                            event_key=r['case_id']+'|'+name+'|'+pred['stable_source_identity'],
                            noise_multiplier=r['scenario']['noise'],scenario_family=r['scenario']['family'],
                            duration_s=pred['evidence_end_s']-pred['evidence_start_s'])
                pred['truth_relations']=[relation(pred,t,r['shared']['span']) for t in r['truth']]
            propagated=[q for q in enriched if q['propagation_supported']]
            # No copied matching implementation: relation readouts must equal frozen matcher edges.
            actual={(e['prediction_id'],e['truth_id']) for e in point['match']['allowed_edges']}
            calculated={(q['prediction_id'],rel['truth_id']) for q in propagated for rel in q['truth_relations'] if rel['matcher_eligible']}
            assert actual==calculated
            lookup={q['prediction_id']:q for q in propagated}
            tp={m['truth_id']:lookup[m['prediction_id']] for m in point['match']['matches']}
            unmatched=[]
            for pred in propagated:
                if pred['matched_TP']:continue
                rels=pred['truth_relations'];ar=artifact_relation(pred,artifact)
                klass=primary_class(rels,ar);links=duplicate_links(pred,rels,tp,point,g) if klass=='A' else []
                subtype='A1' if any(l['subtype']=='A1' for l in links) else 'A2' if klass=='A' else ''
                eligible=[rel for rel in rels if rel['shared_section_count']>=2] or rels
                nearest=min(eligible,key=lambda rel:(rel['closest_truth_boundary_distance_s'],abs(rel['midpoint_difference_s'])))
                pred.update(primary_class=klass,subtype=subtype,duplicate_links=links,artifact_relation=ar,
                    evaluation_edge_case=klass=='E',E_space=any(rel['E_space'] for rel in rels),
                    E_time_fixed=any(rel['E_time_fixed'] for rel in rels),E_time_auxiliary=any(rel['E_time_auxiliary'] for rel in rels),
                    nearest_truth_relation=nearest,Step1_count_min=point['evaluation']['result'][COUNT+'_min'],
                    Step1_count_max=point['evaluation']['result'][COUNT+'_max'],
                    Step1_identity_reduction=g[3])
                if klass=='B':
                    truth=next(t for t in r['truth'] if t['truth_event_id']==nearest['truth_id'])
                    pred['waveform']=metrics(pred,r,truth,components[truth['truth_event_id']]);save_waveform=True
                else:pred['waveform']=None
                events.append(pred);unmatched.append(pred)
            for truth in r['truth']:
                tid=truth['truth_event_id'];rels=[next(rel for rel in q['truth_relations'] if rel['truth_id']==tid) for q in propagated]
                truth_rows.append(dict(case_id=r['case_id'],texture_id=r['source']['texture_id'],scenario_id=r['scenario']['scenario'],point=name,
                    truth_event_id=tid,truth_amplitude_class=truth['truth_amplitude_class'],
                    matcher_eligible_count=sum(rel['matcher_eligible'] for rel in rels),
                    within_15s_path_overlap_count=sum(rel['within_15s_path_overlap'] for rel in rels)))
            cross.append(graph_accounting(r,name,unmatched,g));all_predictions[r['case_id'],name]=enriched
        if save_waveform:
            folder=OUTPUT/'waveforms';folder.mkdir(exist_ok=True);path=folder/(r['case_id'].replace('/','__')+'.npz')
            arrays={k:r['shared'][k] for k in ('A','filtered','valid','t','s')}
            for tid,(raw,filtered,_) in components.items():arrays[tid+'_raw']=raw;arrays[tid+'_filtered']=filtered
            total=sum(c[0] for c in components.values());residual=r['shared']['A']-total
            reconstruction=filtered_runs_check(r,residual,components)
            np.savez_compressed(path,**arrays)
            waveform_files.append(dict(case_id=r['case_id'],file=str(path.relative_to(OUTPUT)),truth=r['truth'],
                matched_and_unmatched_intervals=[dict(point=name,stable_identity=q['stable_source_identity'],matched_TP=q['matched_TP'],
                    start_s=q['evidence_start_s'],end_s=q['evidence_end_s']) for name in POINTS for q in all_predictions[r['case_id'],name] if q['propagation_supported']],
                linear_decomposition_max_abs_error=reconstruction))
    finalize_recurrence(events)
    assert {n:sum(q['point']==n for q in events) for n in POINTS}=={n:v['positive_unmatched'] for n,v in ledgers.items()}
    print('Primary attribution:',{n:composition([q for q in events if q['point']==n])['primary'] for n in POINTS},flush=True)
    summary={name:composition([q for q in events if q['point']==name]) for name in POINTS}
    cross_summary={name:{field:sum(row[field] for row in cross if row['point']==name) for field in (
        'ledger_unmatched','count_min','count_max','min_count_equivalent_excess','max_count_equivalent_excess','identity_reduction',
        'A1_count','A1_with_unresolved_relation','A1_individually_foldable','A1_jointly_absorbable_min','A1_jointly_absorbable_max')} for name in POINTS}
    multiplicity={name:{metric:distribution([q[metric] for q in truth_rows if q['point']==name]) for metric in (
        'matcher_eligible_count','within_15s_path_overlap_count')} for name in POINTS}
    scenarios=[]
    for s in p['scenarios']:
        label=s['scenario'];scenarios+=aggregate_rows(records,events,truth_rows,lambda r,label=label:r['scenario']['scenario']==label,label,'actual_scenario')
    mapping={s['scenario']:groups(s) for s in p['scenarios']}
    for label in sorted({g for row in mapping.values() for g in row}):
        scenarios+=aggregate_rows(records,events,truth_rows,lambda r,label=label:label in mapping[r['scenario']['scenario']],label,'logical_group_overlapping')
    textures=[]
    for texture in sorted({r['source']['texture_id'] for r in records}):
        textures+=aggregate_rows(records,events,truth_rows,lambda r,texture=texture:r['source']['texture_id']==texture,texture,'texture')
    incremental={name:incremental_summary([compare_case(r,{n:all_predictions[r['case_id'],n] for n in POINTS},name)
        for r in records if r['truth']]) for name in ('mad_low','mad_high')}
    phase_report={name:{status:dict(n=len(rows),**{field:dict(Counter(q[field] for q in rows)) for field in DIMENSIONS},
        shared_parent_with_TP=sum(q['shared_parent_with_TP'] for q in rows),
        has_reconciled_child_evidence=sum(q['has_reconciled_child_evidence'] for q in rows))
        for status,rows in [('matched',[q for (cid,n),ps in all_predictions.items() if n==name for q in ps if q['matched_TP']]),
                            ('unmatched',[q for q in events if q['point']==name])]} for name in POINTS}
    crosstab=[]
    for name in POINTS:
        for klass in ORDER:
            rows=[q for q in events if q['point']==name and q['primary_class']==klass]
            for dimension in DIMENSIONS:
                for value,count in sorted(Counter(q[dimension] for q in rows).items()):
                    crosstab.append(dict(point=name,primary_class=klass,dimension=dimension,value=value,count=count,within_primary_percentage=100*count/len(rows)))
    plots=choose_plots(events,all_predictions,records)
    ring=[q for q in events if q['primary_class']=='B']
    ringing=dict(grade='suggestive but not proven' if ring else 'little evidence',filter_source=filter_sources(),
        primary_B_count_by_point={n:sum(q['point']==n for q in ring) for n in POINTS},
        ringing_like_by_point={n:sum(q['point']==n and q['waveform']['ringing_like'] for q in ring) for n in POINTS},
        rule='B; actual phase peaks match same-sign filtered analytical truth extrema within actual measurement span outside local raw truth support on >=3 adjacent sections, offset spread<=span, similar sign/side/offset recurring in another texture. Diagnostic label only; no pipeline rejection.',
        events=[{k:q[k] for k in ('event_key','point','case_id','nearest_truth_relation','phase_origin','split_origin','waveform')} for q in ring],
        waveform_files=waveform_files,plots=plots,
        limitations='Deterministic analytical component correspondence can support consistency with side lobes, not a causal proof or a clinical extrapolation.')
    artifact_events=[q for q in events if q['artifact_relation'] is not None]
    artifacts=dict(primary_F={n:sum(q['point']==n and q['primary_class']=='F' for q in events) for n in POINTS},
        events=[{k:q[k] for k in ('event_key','point','primary_class','artifact_relation')} for q in artifact_events],
        rule='Primary F after A/E, direct temporal overlap with exact injected support. Spatial overlap is recorded but is not an extra exclusion.',
        brightness_limitation='Whole-video photometric condition is nonlocalized: coincidence does not identify cause.')
    A2_phase_order={name:dict(A2_count=sum(q['point']==name and q['subtype']=='A2' for q in events),
        has_reverse_ordered_biphasic_same_wall_link=sum(q['point']==name and q['subtype']=='A2' and any(
            pair['all_shared_sections_reverse_biphasic_order'] for link in q['duplicate_links'] for pair in link['same_wall_phase_comparisons']) for q in events)) for name in POINTS}
    overall=dict(ledgers=ledgers,primary_priority=list(ORDER),attribution=summary,cross_wall=cross_summary,
        predictions_per_truth=multiplicity,scenario_mapping=mapping,logical_groups_overlap=True,
        boundary_distance_convention='Prediction midpoint to nearest physical truth start/end; interval gap also retained separately.',
        A_subtype_rule='Preserve all eligible occupied truths; A1 if any disjoint-wall TP link, otherwise A2. Eligibility is not proof of phase identity.',
        paired_within_texture=paired_scenarios(records),representative_plots=plots,
        phase_provenance_comparison=phase_report,ringing_grade=ringing['grade'],
        split_phase_grade='suggestive but not proven',split_specific_grade='little evidence',cross_wall_grade='suggestive but not proven',
        A2_same_wall_phase_order=A2_phase_order,
        next_direction='5. 暂不能定位，需要补充诊断',
        next_direction_reason='One truth has multiple eligible paired-phase unsplit bases across all textures. This localizes the ledger structure, but neither single-phase/split origin nor cross-wall ambiguity explains the majority. A1 eligibility alone is not identity proof; B waveform evidence concerns only a few events. Further mechanism identification is needed before choosing an algorithm change.',
        scope='Only Dev synthetic/semi-synthetic benchmark; no claim about patient mechanisms or superiority of a scorer')
    save(OUTPUT/'DEV_POSITIVE_FP_ATTRIBUTION_SUMMARY.json',overall)
    write_csv(OUTPUT/'DEV_POSITIVE_FP_EVENT_LEDGER.csv',events)
    write_csv(OUTPUT/'DEV_POSITIVE_FP_PRIMARY_X_PROVENANCE.csv',crosstab)
    write_csv(OUTPUT/'DEV_POSITIVE_FP_BY_SCENARIO.csv',scenarios)
    write_csv(OUTPUT/'DEV_POSITIVE_FP_BY_TEXTURE.csv',textures)
    write_csv(OUTPUT/'DEV_PREDICTIONS_PER_TRUTH.csv',truth_rows)
    save(OUTPUT/'DEV_CROSS_WALL_DUPLICATE_DIAGNOSTIC.json',dict(summary=cross_summary,cases=cross,
        A1_events=[q for q in events if q['subtype']=='A1'],
        interpretation='Individually foldable A1 edges may compete. Jointly absorbable min/max count marked A1 links among GLOBAL maximum-cardinality matchings; these are potential identity reductions, not proven duplicates. Count-equivalent excess is SUM over truth-containing cases of max(v2 bound - case truth count,0), not pooled subtraction; it also depends on missed truths.'))
    save(OUTPUT/'DEV_FILTER_RINGING_DIAGNOSTIC.json',ringing)
    save(OUTPUT/'DEV_SPLIT_PHASE_PROVENANCE.json',dict(matched_vs_unmatched=phase_report,by_primary={n:summary[n]['provenance_by_primary'] for n in POINTS},
        rates_by_primary={n:summary[n]['provenance_rates_by_primary'] for n in POINTS},
        definition='Canonical-source phase kind; all contributor phase kinds preserved separately. Canonical child versus attached child evidence are separate. Parent/ridge IDs are wall-qualified.',
        remainder_activities_emitted=0,events=[{k:q[k] for k in ('event_key','phase_origin','split_origin','wall_status','ownership_statuses','source_chain','shared_parent_with_TP','sibling_prediction_count')} for q in events]))
    save(OUTPUT/'DEV_INJECTED_ARTIFACT_DIAGNOSTIC.json',artifacts)
    save(OUTPUT/'DEV_MAD_INCREMENTAL_FP_ATTRIBUTION.json',incremental)
    verify(WORK);after={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)};assert before==after
    files=sorted(f for f in OUTPUT.glob('DEV_*') if f.name!='DEV_POSITIVE_FP_READ_ONLY_AUDIT.json')
    import subprocess
    import xml.etree.ElementTree as ET
    def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    tracked_changes=git('diff','--name-only').splitlines()
    assert not tracked_changes,'Read-only audit found unexpected tracked code changes'
    tests=ET.parse(OUTPUT/'DIAGNOSTIC_TESTS.xml').getroot().find('testsuite') if (OUTPUT/'DIAGNOSTIC_TESTS.xml').exists() else None
    save(OUTPUT/'DEV_POSITIVE_FP_READ_ONLY_AUDIT.json',dict(**{k:False for k in FLAGS},
        gate_passed=True,all_186_frozen_pool_hashes_verified=True,all_baseline_predictions_equal_historical=True,
        attribution_relations_equal_production_matcher_edges=True,ambiguity_graph_cardinality_equals_production=True,
        frozen_input_sha256_unchanged=before==after,input_sha256=before,
        protected_source_sha256=verify(WORK)['protected_source_sha256'],
        benchmark_source_sha256=verify(WORK)['benchmark_source_sha256'],
        data_reads_replay=gate_record['data_reads'],data_reads_attribution=sorted(access.accesses),
        output_sha256={f.name:sha(f) for f in files},no_threshold_search=True,
        replay_cache_sha256=sha(OUTPUT/'replay.pkl'),
        permitted_new_code_directory='experiments/step2_salience_sandbox/diagnostic/',
        diagnostic_source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        HEAD=git('rev-parse','HEAD'),branch=git('branch','--show-current'),tracked_code_changes=tracked_changes,
        validation=dict(tests=int(tests.get('tests')),failed=int(tests.get('failures')),errors=int(tests.get('errors')),skipped=int(tests.get('skipped')),
                        junit_sha256=sha(OUTPUT/'DIAGNOSTIC_TESTS.xml')) if tests is not None else None,
        strict_read_only_meaning='Frozen code, inputs, parameters and benchmark decisions unchanged; new diagnostic code/artifacts only'))
    print('Completed all 12 requested artifacts. Cross-wall:',cross_summary,flush=True)

def filtered_runs_check(record,residual,components):
    from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
    recomposed=filtered_runs(residual,record['shared']['valid'],record['shared']['t'])+sum(c[1] for c in components.values())
    np.testing.assert_allclose(recomposed,record['shared']['filtered'],rtol=1e-10,atol=1e-12,equal_nan=True)
    return float(np.nanmax(abs(recomposed-record['shared']['filtered'])))

if __name__=='__main__':run()
