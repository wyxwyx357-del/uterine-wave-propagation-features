"""Phase 1: immutable source, RNG, generated input, truth, scope and gate locks."""
import argparse,datetime,inspect,sys
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK,ROOT,load,save,sha,verify,make_case
from ..upstream import stable_hash
from ..diagnostic.attribution_rules import artifact_definition
from .sources import OUTPUT,GLOBAL_SEED
from .scope import audit_scope,ATTEMPT
from .evaluate import MERGE_DEFINITION
from .gates import DEFINITIONS

HERE=Path(__file__).resolve().parent
FLAGS=('old_Dev_used_for_tuning_this_round','old_Val_read','validation_texture_reused_after_unlock','validation_texture_retuned',
       'M1_formula_changed','M1_scope_changed_from_attempt_002','candidate_split_verifier_intercepted','filter_changed',
       'MinMax_threshold_changed','MAD_used_as_formal_endpoint','truth_matcher_changed','Step1_changed','scenario_protocol_changed',
       'RNG_changed_after_prelock','truth_changed_after_prelock','pregnancy_outcome_read','clinical_AUC_calculated','production_changed','400plus_cohort_run')

def source_hashes():
    hashes={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    for folder in ('mechanism_2x2','diagnostic'):
        hashes.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.parent.joinpath(folder).glob('*.py')})
    return hashes

class Access:
    def __init__(self):
        self.reads=set();self.work=WORK.resolve();self.output=OUTPUT.resolve()
        self.allowed={WORK/name for name in ('STEP2_BENCHMARK_PROTOCOL_PRELOCK.json','PRELOCK.sha256')}
        self.allowed.update(ATTEMPT/name for name in ('PAIRING_FILTER_2X2_PRELOCK.json','PRELOCK.sha256','PAIRING_FILTER_2X2_IMPLEMENTATION_AUDIT.json'))
    def hook(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve();mode=args[1]
        if path==self.output or self.output in path.parents:
            if mode and not any(c in mode for c in 'wax+'):self.reads.add(str(path.relative_to(self.output)))
            return
        if path==self.work or self.work in path.parents:
            if path not in self.allowed:raise PermissionError('Validation denies old experiment data: '+path.name)
            if mode and any(c in mode for c in 'wax+'):raise PermissionError('Old exclusion/prelock metadata is read-only')
        elif path.suffix.lower() in ('.npz','.pkl','.csv','.xlsx','.xls','.mp4','.avi','.dcm','.png','.jpg'):
            raise PermissionError('Validation only reads its anonymous frozen inputs')

def validate():
    p=load(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json')
    assert sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json')==(OUTPUT/'PRELOCK.sha256').read_text().strip()
    assert p['source_sha256']==source_hashes(),'STOP: code differs from unseen validation prelock'
    for name,h in p['frozen_artifact_sha256'].items():assert sha(OUTPUT/name)==h,'Frozen artifact changed: '+name
    verify(WORK)
    return p

def freeze():
    if (OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json').exists():raise RuntimeError('Prelock already exists; never overwrite')
    if (OUTPUT/'USED_VALIDATION_DATA.json').exists():raise RuntimeError('Already used validation data')
    from .sources import complete_saved_hash_audit
    complete_saved_hash_audit()
    access=Access();sys.addaudithook(access.hook)
    base=verify(WORK);manifest=load(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')
    assert sha(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')==(OUTPUT/'INPUT_MANIFEST.sha256').read_text().strip()
    independence=load(OUTPUT/'M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json');assert independence['final'] and independence['all_intersections_zero']
    sources=manifest['sources'];assert len({s['patient_group_sha256'] for s in sources})==len(sources)
    scope=audit_scope();save(OUTPUT/'M1_UNSEEN_TEXTURE_SCOPE_AUDIT.json',scope)
    samples=[];truths=[];seeds=[]
    for source in sources:
        bases={}
        for condition in ('static','rigid','brightness'):
            with np.load(OUTPUT/'bases'/source['texture_id']/(condition+'.npz'),allow_pickle=False) as z:bases[condition]=dict(z)
        for scenario in base['scenarios']:
            # Exact frozen generator. 'Dev' here is ONLY its timing-key adapter;
            # all texture pixels/maps come from the independent source above.
            maps,truth=make_case(dict(source,split='Dev'),scenario,bases,base)
            sample=source['texture_id']+'/'+scenario['scenario'];path=OUTPUT/'inputs'/(sample.replace('/','__')+'.npz')
            path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**maps)
            shim=dict(t=maps['interval_center_s'],s=maps['s'],dt=float(np.median(np.diff(maps['interval_center_s']))))
            artifact=artifact_definition(source,dict(scenario,amplitude=0.) if scenario['family']=='artifact' else scenario,
                shim,base,maps['interval_start_s'],maps['interval_end_s'])
            samples.append(dict(sample_id=sample,texture_id=source['texture_id'],scenario=scenario,input_file=str(path.relative_to(OUTPUT)),
                input_file_sha256=sha(path),generated_sample_array_hash=stable_hash(maps),truth_hash=stable_hash(truth)))
            truths.append(dict(sample_id=sample,truth=truth,wall_support={t['truth_event_id']:[0,1] for t in truth},
                coexist_identity={t['truth_event_id']:t['truth_event_id'] for t in truth},artifact=artifact))
            seeds.append(dict(sample_id=sample,global_seed=GLOBAL_SEED,texture_seed=source['seed'],scenario_seed=source['seed'],
                generated_sample_hash=stable_hash(maps)))
    save(OUTPUT/'GENERATED_SAMPLE_MANIFEST.json',dict(samples=samples,generated_before_any_event_result=True))
    save(OUTPUT/'M1_UNSEEN_TEXTURE_TRUTH_MANIFEST.json',dict(samples=truths,original_truth_generator_unchanged=True))
    save(OUTPUT/'RNG_MANIFEST.json',dict(global_seed=GLOBAL_SEED,samples=seeds,
        common_randomness='All31 scenario seeds equal their texture seed by design, preserving the frozen Dev shared phase/jitter/artifact-jitter realization across single/coexist/duration. No scenario RNG draws or resampling exist in frozen make_case.'))
    frozen_names=['M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json','M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json',
        'M1_UNSEEN_TEXTURE_TRUTH_MANIFEST.json','M1_UNSEEN_TEXTURE_SCOPE_AUDIT.json','GENERATED_SAMPLE_MANIFEST.json','RNG_MANIFEST.json']
    artifact_hashes={name:sha(OUTPUT/name) for name in frozen_names}
    artifact_hashes.update({r['input_file']:r['input_file_sha256'] for r in samples})
    prelock=dict(phase='Phase1 complete; no event results generated',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        texture_count=len(sources),formal_validation_eligible=len(sources)>=8,texture_ids=[s['texture_id'] for s in sources],
        scenario_ids=[s['scenario'] for s in base['scenarios']],scenario_definition_hash=stable_hash(base['scenarios']),scenarios=base['scenarios'],
        global_seed=GLOBAL_SEED,RNG_manifest_hash=sha(OUTPUT/'RNG_MANIFEST.json'),
        truth_manifest_hash=sha(OUTPUT/'M1_UNSEEN_TEXTURE_TRUTH_MANIFEST.json'),frozen_artifact_sha256=artifact_hashes,
        source_sha256=source_hashes(),protected_source_sha256=base['protected_source_sha256'],
        M0='Unchanged production chain + Min-Max0.6 + Step1v2',M1='Exact attempt_002 mechanisms.intervention(M1), global detector only; same Min-Max0.6 + Step1v2',
        M1_mechanism_file_hash=scope['current_mechanisms_sha256'],M1_scope_hash_matches_attempt002=True,
        salience=dict(scorer='Min-Max',threshold=.6,MAD_formal_endpoint=False),noise_floor=base['upstream_noise_floor'],
        A2_definition='Exact prior S1 final propagation-ledger A>E... taxonomy: unmatched propagation prediction eligible for occupied truth; A1 if any disjoint-wall TP link, otherwise same-wall A2. No direction TP gate.',
        merge_two_truth_error_definition=MERGE_DEFINITION,
        phase_assignment='Unchanged attempt_002 phase_diagnostics, existing truth-component lobe containment on >=3 adjacent sections; ambiguous/unassigned labels separate',
        single_phase_flow='Main formed unit: canonical single-phase event proposals before salience. Passed/propagating are wall nodes; final unmatched are cross-wall activity units. Raw detector single-phase count and truth-bearing-only flow also saved.',
        ledger_population='All31 raw scenarios per texture for event ledger/F1; unique-input dedup only for negative safety. Frozen brightness controls overlap between zero/artifact-negative subgroups, whose counts are not additive.',
        gate_definitions=DEFINITIONS,gate_arithmetic='Exact rational count comparisons, no tolerance or rounding',
        interval_analysis='Coverage among evaluable samples with missing count; coverage and distance paired transitions separately; no significance claim',
        bootstrap='Not performed; per-texture, signed contributions and leave-one-texture-out are prelocked robustness checks',
        phase2='Only M0/M1 raw outputs stored, no metrics revealed, no configuration changes',phase3='One unlock; register used validation data before aggregating any metrics',
        after_PASS='Freeze M1 candidate only; separately preregister small real-clinical-video two-rater blinded review agreement study; no400plus cohort',
        after_FAIL='No production adoption; permanently retire this panel from tuning; report failure mechanism and stop',
        **{k:False for k in FLAGS})
    save(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json',prelock,exclusive=True)
    (OUTPUT/'PRELOCK.sha256').write_text(sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json')+'\n',encoding='ascii')
    print('FROZEN',len(sources),'textures',len(samples),'samples; PRELOCK',sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json'),flush=True)

if __name__=='__main__':freeze()
