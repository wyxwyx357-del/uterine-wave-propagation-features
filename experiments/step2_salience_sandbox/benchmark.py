"""One prelocked Dev fit, then one untouched Val evaluation; no production writes."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

from .benchmark_data import WORK,HERE,ROOT,CONFIGS,verify,load,save,sha,make_case
from .benchmark_metrics import evaluate,aggregate,detailed_summary,qualifies,selection,mean
from .core import BaselineMinMaxScorer,continuous_branch,evaluate_threshold,baseline_equivalence_gate
from .upstream import extract_shared_upstream,stable_hash
from .scorers import DevRigidBank,PerSectionRigidReferenceScorer,OutOfBandMADScorer,InBandMADScorer,PercentileScorer
from .matching import upstream_recall,match_events,upstream_predictions


def bank_from_dev(work,p):
    controls=[]
    for source in p['sources']:
        if source['split']!='Dev':continue
        for condition in ('static','rigid'):
            with np.load(work/'bases'/source['texture_id']/(condition+'.npz'),allow_pickle=False) as z:maps=dict(z)
            controls.append(dict(control_id=source['texture_id']+'_'+condition,split='Dev',
                                 condition='zero' if condition=='static' else 'rigid',maps=maps))
    return DevRigidBank(controls)


def build_records(work,p,split,bank):
    records=[];gates=[]
    for source in p['sources']:
        if source['split']!=split:continue
        bases={}
        for condition in ('static','rigid','brightness'):
            with np.load(work/'bases'/source['texture_id']/(condition+'.npz'),allow_pickle=False) as z:bases[condition]=dict(z)
        for scenario in p['scenarios']:
            maps,truth=make_case(source,scenario,bases,p)
            shared=extract_shared_upstream(maps,p['upstream_noise_floor'])
            baseline=continuous_branch(shared,BaselineMinMaxScorer(shared))
            evaluation=evaluate_threshold(shared,baseline,.6)
            gate=baseline_equivalence_gate(maps,p['upstream_noise_floor'],shared,baseline,evaluation)
            gate.update(texture_id=source['texture_id'],scenario=scenario['scenario']);gates.append(gate)
            # No alternative scorer for this case is constructed before equivalence passes.
            scorers=[PerSectionRigidReferenceScorer(bank),OutOfBandMADScorer(shared,p['S3a_band_cpm']),
                     InBandMADScorer(shared),PercentileScorer(shared,95),PercentileScorer(shared,99)]
            branches=dict(zip(CONFIGS,[baseline]+[continuous_branch(shared,s) for s in scorers]))
            if any(b['pool_hash']!=shared['pool_hash'] for b in branches.values()):raise RuntimeError('Candidate pool changed')
            records.append(dict(case_id=source['texture_id']+'/'+scenario['scenario'],source=source,scenario=scenario,
                shared=shared,truth=truth,branches=branches,upstream=upstream_recall(shared,truth),downstream_cache={}))
        print(split,source['texture_id'],'shared upstream + six scorer branches complete',flush=True)
    save(work/(split.upper()+'_BASELINE_GATES.json'),dict(passed=True,cases=gates))
    return records


def threshold_grid(records,config):
    values=[r['score'] for record in records for wall in record['branches'][config]['walls']
            for event in wall['proposals'] for r in event['sections'] if r['score'] is not None and np.isfinite(r['score'])]
    if not values:return [0.]
    return sorted(set([0.,float(np.nextafter(min(values),-np.inf))]+np.quantile(values,np.linspace(0,1,21)).tolist()))


def curve_data(records,config,grid):
    points=[];distributions={label:[] for label in ('clear','medium','structured_weak','artifact','zero')}
    for record in records:
        predictions=upstream_predictions(record['shared'])
        edges=match_events(record['truth'],predictions,record['shared']['span'],record['shared']['exposure']['seconds'])['allowed_edges']
        by_pred={}
        for edge in edges:
            key=edge['prediction_id']
            if key not in by_pred or edge['center_deviation_s']<by_pred[key]['center_deviation_s']:by_pred[key]=edge
        truth_lookup={t['truth_event_id']:t for t in record['truth']}
        for wall in record['branches'][config]['walls']:
            for event in wall['proposals']:
                key=f"wall{wall['wall']}/"+event['event']['event_proposal_id']
                label=truth_lookup[by_pred[key]['truth_id']]['truth_amplitude_class'] if key in by_pred else 'artifact' if record['scenario']['artifact'] else 'zero'
                if event['S_crit'] is not None:distributions[label].append(event['S_crit'])
    for tau in grid:
        tp=fp=fn=0;minutes=0.
        for record in records:
            predictions,_=selection(record['shared'],record['branches'][config],tau)
            match=match_events(record['truth'],predictions,record['shared']['span'],record['shared']['exposure']['seconds'])
            tp+=match['TP'];fp+=match['FP'];fn+=match['FN'];minutes+=record['shared']['exposure']['seconds']/60
        points.append(dict(threshold=float(tau),TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp) if tp+fp else None,
                           Recall=tp/(tp+fn) if tp+fn else None,FP_per_min=fp/minutes if minutes else None))
    pos=np.asarray(distributions['clear']+distributions['medium']);neg=np.asarray(distributions['artifact']+distributions['zero'])
    delta=float(np.sign(pos[:,None]-neg).mean()) if len(pos) and len(neg) else None
    return dict(scope='Proposal-level salience + >=3 native section curves only; cross-wall duplicates are not patient activities',
                PR_FROC=points,S_crit_distributions=distributions,Cliffs_delta_clear_medium_vs_negative=delta)


def add_score_drift(summary,records,config):
    anchors={}
    for record in records:
        # Core score = maximum Scrit among candidates time/spatially eligible for core truth.
        upstream=upstream_predictions(record['shared'])
        edges=match_events(record['truth'],upstream,record['shared']['span'],record['shared']['exposure']['seconds'])['allowed_edges']
        ids={e['prediction_id'] for e in edges if e['truth_id']=='core'}
        scores=[event['S_crit'] for wall in record['branches'][config]['walls'] for event in wall['proposals']
                if f"wall{wall['wall']}/"+event['event']['event_proposal_id'] in ids and event['S_crit'] is not None]
        anchors[(record['source']['texture_id'],record['scenario']['scenario'])]=max(scores) if scores else None
    for pair in summary['duration']['pairs']:
        name='duration_'+str(int(pair['duration']))
        current=anchors[(pair['texture_id'],name)];base=anchors[(pair['texture_id'],'medium_C2F')]
        pair.update(core_S_crit=current,anchor_S_crit=base,absolute_score_drift=current-base if current is not None and base is not None else None,
                    relative_score_drift=(current-base)/abs(base) if current is not None and base not in (None,0) else None)


def fit_dev(work):
    p=verify(work)
    save(work/'DEV_EVALUATION_STARTED.json',dict(prelock=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json')),exclusive=True)
    bank=bank_from_dev(work,p);records=build_records(work,p,'Dev',bank)
    baseline_rows=[evaluate(r,'baseline_minmax',.6,force=True) for r in records]
    baseline=detailed_summary(baseline_rows)
    thresholds={'baseline_minmax':.6};traces={};curves={};summaries={'baseline_minmax':baseline};selected={'baseline_minmax':baseline_rows}
    for config in CONFIGS:
        grid=sorted(set(threshold_grid(records,config)+([.6] if config=='baseline_minmax' else [])))
        curves[config]=curve_data(records,config,grid)
        if config=='baseline_minmax':continue
        results=[]
        for tau in grid:
            rows=[evaluate(r,config,tau) for r in records]
            summary=aggregate(rows)
            cm=aggregate([r for r in rows if r['truth_count'] and all(t['truth_amplitude_class'] in ('clear','medium') for t in r['truth'])])
            safety=summary['safety']
            feasible=(safety['FP_per_min']<=.05+1e-12) if safety['minutes']>=20 else safety['FP']<=baseline['safety']['FP']
            key=(cm['F1'] if cm['F1'] is not None else -1,
                 -(safety['FP_per_min'] or 0),cm['Recall'] if cm['Recall'] is not None else -1,
                 -(summary['distance_to_interval'] if summary['distance_to_interval'] is not None else np.inf),
                 summary['strata']['structured_weak']['end_to_end_recall'] or 0,float(tau))
            results.append(dict(threshold=float(tau),feasible=bool(feasible),selection_key=key,
                                safety=safety,clear_medium=cm,all_cases=summary))
        feasible=[r for r in results if r['feasible']]
        if not feasible:raise RuntimeError('No threshold meets locked safety gate: '+config)
        winner=max(feasible,key=lambda r:r['selection_key'])
        tau=winner['threshold'];thresholds[config]=tau
        # Required full downstream rerun at every finally selected threshold.
        rows=[evaluate(r,config,tau,force=True) for r in records]
        summary=detailed_summary(rows);add_score_drift(summary,records,config)
        selected[config]=rows;summaries[config]=summary;traces[config]=results
        print('DEV LOCK',config,tau,'F1',summary['clear_medium']['F1'],'safety',summary['safety']['FP'],summary['safety']['minutes'],flush=True)
    add_score_drift(baseline,records,'baseline_minmax')
    for config in CONFIGS:
        for source in p['dev_texture_ids']:
            summaries[config].setdefault('per_texture',{})[source]=aggregate([r for r in selected[config] if r['texture_id']==source])
    save(work/'STEP2_DEV_RESULTS.json',dict(split='Dev',summaries=summaries,cases=selected,selection_traces=traces,
        continuous_score_curves=curves,reference_bank_provenance=bank.provenance,reference_bank_hash=bank.bank_hash,
        safety_exposure_note='Identical source/input negative scenarios counted once in pooled safety minutes; zero and artifact subgroups separately reported'))
    verify(work)
    save(work/'STEP2_DEV_THRESHOLDS.json',dict(thresholds=thresholds,prelock_sha256=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'),
        band_cpm=p['S3a_band_cpm'],reference_bank_hash=bank.bank_hash,Dev_only=True,Val_accessed=False,
        grid_rule=p['threshold_selection'],scorer_source_sha256=sha(HERE/'scorers.py')),exclusive=True)
    (work/'DEV_THRESHOLDS.sha256').write_text(sha(work/'STEP2_DEV_THRESHOLDS.json'),encoding='ascii')
    print('DEV THRESHOLDS FROZEN',flush=True)


def run_val(work):
    p=verify(work);threshold_path=work/'STEP2_DEV_THRESHOLDS.json'
    if sha(threshold_path)!=(work/'DEV_THRESHOLDS.sha256').read_text().strip():raise RuntimeError('Dev thresholds changed')
    started=load(work/'VAL_RUN_STARTED.json')
    if started['threshold_sha256']!=sha(threshold_path):raise RuntimeError('Val began with different thresholds')
    if not (work/'VAL_GENERATION_COMPLETE.json').exists():raise RuntimeError('Val controls are incomplete')
    save(work/'VAL_EVALUATION_STARTED.json',dict(threshold_sha256=sha(threshold_path)),exclusive=True)
    frozen=load(threshold_path);bank=bank_from_dev(work,p)
    if bank.bank_hash!=frozen['reference_bank_hash']:raise RuntimeError('Reference bank changed')
    records=build_records(work,p,'Val',bank);rows={};summaries={}
    for config in CONFIGS:
        rows[config]=[evaluate(r,config,frozen['thresholds'][config],force=True) for r in records]
        summary=detailed_summary(rows[config]);add_score_drift(summary,records,config)
        summary['per_texture']={source:aggregate([r for r in rows[config] if r['texture_id']==source]) for source in p['val_texture_ids']}
        summaries[config]=summary
    save(work/'STEP2_VAL_RESULTS.json',dict(split='Val',one_formal_run=True,threshold_sha256=sha(threshold_path),
        prelock_sha256=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'),summaries=summaries,cases=rows))
    dev=load(work/'STEP2_DEV_RESULTS.json');comparisons={}
    for config in CONFIGS[1:]:
        comparisons[config]=qualifies(dev['summaries'][config],summaries[config],dev['summaries']['baseline_minmax'],summaries['baseline_minmax'])
        comparisons[config]['paired_Val_source_F1_difference']={source:summaries[config]['per_texture'][source]['F1']-summaries['baseline_minmax']['per_texture'][source]['F1'] for source in p['val_texture_ids']}
    candidates=[config for config,comparison in comparisons.items() if comparison['qualifies']]
    baseline=summaries['baseline_minmax']
    decision='new scorer qualifies as production-replacement candidate' if candidates else 'retain production Min-Max + 0.6'
    if not candidates and any(c['safety_pass'] and c['clear_medium_pass'] and not c['no_key_regression'] for c in comparisons.values()):
        decision='no method clearly superior'
    save(work/'STEP2_METHOD_COMPARISON.json',dict(comparisons=comparisons,Dev=dev['summaries'],Val=summaries,
        no_P_value_search=True,count_error='distance to interval is primary; midpoint error is auxiliary'))
    save(work/'STEP2_FINAL_DECISION.json',dict(decision=decision,production_baseline_retained=True,candidate_replacement=candidates,
        no_clear_winner=decision=='no method clearly superior',production_modified=False,Val_retuned=False,
        reason='Candidates must pass every prelocked safety/structured/regression gate and improve a specified defect by >=.10 in both Dev and Val.',
        any_gate_failed=any(not c['qualifies'] for c in comparisons.values()),gates=comparisons,
        limitations=['Six or fewer patient-separated source textures per split; generated observations are paired, not independent patients.',
            'Pixel warps and signal-layer injections are not acoustic or clinical validation; exact poses omit background-estimation error.',
            'Historical upstream noise floor is fixed and not independently validated here; S3a was chosen only on paired synthetic Step4a zero controls.',
            'Two Dev-source technical paths incidentally exposed outcome-bearing folder names; no outcome table, extracted label or clinical AUC was used.',
            'Ledger event metrics are conditional on strict-pair/single identity accounting; v2 count bounds are evaluated separately.'],
        pregnancy_outcome_table_read=False,clinical_AUC_calculated=False,formal_patient_cohort_run=False,
        prelock_sha256=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'),threshold_sha256=sha(threshold_path)))
    verify(work)
    if sha(threshold_path)!=(work/'DEV_THRESHOLDS.sha256').read_text().strip():raise RuntimeError('Thresholds changed during Val')
    save(work/'VAL_COMPLETE.json',dict(completed=True,decision=decision),exclusive=True)
    print('VAL COMPLETE',decision,flush=True)


def plots(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dev=load(work/'STEP2_DEV_RESULTS.json');val=load(work/'STEP2_VAL_RESULTS.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for config in CONFIGS:
        points=dev['continuous_score_curves'][config]['PR_FROC']
        axes[0].plot([r['Recall'] for r in points],[r['Precision'] for r in points],'.-',label=config)
        axes[1].plot([r['FP_per_min'] for r in points],[r['Recall'] for r in points],'.-')
    axes[0].set(xlabel='Proposal recall',ylabel='Proposal precision',title='Dev salience-layer PR (not final counts)')
    axes[1].set(xlabel='Proposal FP/min',ylabel='Proposal recall',title='Dev salience-layer FROC')
    axes[0].legend(fontsize=6);fig.savefig(work/'DEV_PR_FROC.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for config in CONFIGS:
        s=val['summaries'][config];noise=s['noise_gradient']
        axes[0].plot([1,3,8],[noise[str(k)]['F1'] for k in (1,3,8)],'o-',label=config)
        axes[1].plot([2,3,5],[s['coexistence']['by_ratio'][str(k)] for k in (2,3,5)],'o-')
        pairs=s['duration']['pairs']
        axes[2].plot([30,60,90],[mean([int(r['event_detected']) for r in pairs if r['duration']==30]),
                     mean([int(r['anchor_detected']) for r in pairs]),mean([int(r['event_detected']) for r in pairs if r['duration']==90])],'o-')
    axes[0].set(xlabel='Residual noise multiplier',ylabel='Event F1',title='Held-out noise gradient')
    axes[1].set(xlabel='Strong/weaker ratio',ylabel='Recall with strong - alone',title='Held-out coexistence masking')
    axes[2].set(xlabel='Duration (s)',ylabel='Core event recall',title='Same core content, different duration')
    axes[0].legend(fontsize=6);fig.savefig(work/'VAL_FAILURE_MODES.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('dev','val','plots'));parser.add_argument('--work',type=Path,default=WORK)
    args=parser.parse_args()
    {'dev':fit_dev,'val':run_val,'plots':plots}[args.mode](args.work)
