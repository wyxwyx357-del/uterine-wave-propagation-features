"""Phase2 stores blinded raw results; Phase3 unlocks exactly once and retires data."""
import argparse,datetime,pickle,subprocess,sys
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK,ROOT,load,save,sha,verify
from ..upstream import extract_shared_upstream,stable_hash
from ..core import BaselineMinMaxScorer,continuous_branch,evaluate_threshold
from ..mechanism_2x2.mechanisms import intervention
from .sources import OUTPUT
from .freeze import validate,Access,FLAGS
from .evaluate import evaluate_saved
from .report import write

def run():
    access=Access();sys.addaudithook(access.hook);lock=validate()
    if (OUTPUT/'USED_VALIDATION_DATA.json').exists():raise RuntimeError('Used validation panel; no reruns')
    save(OUTPUT/'PHASE2_STARTED.json',dict(prelock_sha256=sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json'),results_not_unlocked=True),exclusive=True)
    sources={s['texture_id']:s for s in load(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')['sources']}
    samples=load(OUTPUT/'GENERATED_SAMPLE_MANIFEST.json')['samples'];truths={r['sample_id']:r for r in load(OUTPUT/'M1_UNSEEN_TEXTURE_TRUTH_MANIFEST.json')['samples']}
    files=[]
    for index,sample in enumerate(samples):
        with np.load(OUTPUT/sample['input_file'],allow_pickle=False) as z:maps=dict(z)
        assert stable_hash(maps)==sample['generated_sample_array_hash']
        truth=truths[sample['sample_id']]['truth'];assert stable_hash(truth)==sample['truth_hash']
        for model in ('M0','M1'):
            logs=[]
            with intervention(model,logs):shared=extract_shared_upstream(maps,lock['noise_floor'])
            branch=continuous_branch(shared,BaselineMinMaxScorer(shared));evaluation=evaluate_threshold(shared,branch,.6)
            record=dict(case_id=sample['sample_id'],source=sources[sample['texture_id']],scenario=sample['scenario'],truth=truth,
                shared=shared,interval_start=maps['interval_start_s'],interval_end=maps['interval_end_s'],input_hash=sample['generated_sample_array_hash'])
            destination=OUTPUT/'blinded_results'/model/(sample['sample_id'].replace('/','__')+'.pkl');destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:pickle.dump(dict(record=record,evaluation=evaluation,pairing_logs=logs),stream,protocol=5)
            files.append(dict(model=model,sample_id=sample['sample_id'],path=str(destination.relative_to(OUTPUT)),sha256=sha(destination)))
            assert stable_hash(maps)==sample['generated_sample_array_hash'],'Shared generated input was mutated'
        if (index+1)%31==0:print('Blinded M0/M1 completed',sample['texture_id'],'31 locked scenarios; no metrics disclosed',flush=True)
    validate()
    save(OUTPUT/'PHASE2_COMPLETE.json',dict(prelock_sha256=sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json'),sample_count=len(samples),raw_results=files),exclusive=True)
    print('Phase2 complete. All frozen inputs finished; no partial metric review.',flush=True)

def unlock():
    access=Access();sys.addaudithook(access.hook);lock=validate()
    complete=load(OUTPUT/'PHASE2_COMPLETE.json')
    assert complete['prelock_sha256']==sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json')
    manifest=load(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')
    # Retire the panel BEFORE the first result is opened/aggregated. Even a
    # later reporting failure cannot make these textures unused again.
    used=dict(status='used validation data',unlocked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        permanently_excluded_from_tuning=True,sources=manifest['sources'],prelock_sha256=sha(OUTPUT/'M1_UNSEEN_TEXTURE_PRELOCK.json'))
    save(OUTPUT/'USED_VALIDATION_DATA.json',used,exclusive=True)
    results={'M0':[],'M1':[]}
    for item in complete['raw_results']:
        assert sha(OUTPUT/item['path'])==item['sha256']
        with (OUTPUT/item['path']).open('rb') as stream:payload=pickle.load(stream)
        results[item['model']].append(evaluate_saved(payload,item['model']))
    assert len(results['M0'])==len(results['M1'])==complete['sample_count']
    report=write(OUTPUT,results,lock['formal_validation_eligible'])
    status=report['gate']['M1_UNSEEN_VALIDATION']
    # A candidate-version record is not a production deployment or a Git tag.
    if status=='PASS':
        save(OUTPUT/'M1_CANDIDATE_FREEZE.json',dict(candidate='M1 global phase-pairing amplitude weight',
            implementation='exact attempt_002',mechanisms_sha256=lock['M1_mechanism_file_hash'],prelock_sha256=used['prelock_sha256'],
            validation_scope='M1 independently validated on unseen texture-based controlled synthetic/semi-synthetic benchmark.',
            production_adopted=False,next_stage='Separately preregister small real-clinical-video two-rater independent blinded review and agreement study; no400plus cohort',
            review_protocol_requirements=['Raters blind to algorithm results and pregnancy outcomes','onset/offset consensus','propagation identity',
                'disagreement adjudication','patient/video sampling','observer agreement estimands; ICC model/type/definition/95%CI only if preregistered']))
    validate()
    assert subprocess.run(['git','diff','--name-only'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()==''
    audit=dict(**{k:False for k in FLAGS},M1_UNSEEN_VALIDATION=status,prelock_sha256=used['prelock_sha256'],
        frozen_sources_and_truth_unchanged=True,one_time_unlock=True,permanent_used_validation_data=True,
        old_Val_exclusion_hash_metadata_read=True,old_Val_images_inputs_or_results_read=False,
        old_Dev_inputs_or_results_read=False,MAD_instantiated_as_endpoint=False,models_run=['M0','M1'],
        production_source_sha256=verify(WORK)['protected_source_sha256'],
        data_reads=sorted(access.reads),production_adopted=False,clinical_review_started=False,
        output_sha256={str(f.relative_to(OUTPUT)):sha(f) for f in OUTPUT.rglob('*') if f.is_file() and
            'private' not in f.relative_to(OUTPUT).parts and f.name!='M1_UNSEEN_TEXTURE_READ_ONLY_AUDIT.json'})
    save(OUTPUT/'M1_UNSEEN_TEXTURE_READ_ONLY_AUDIT.json',audit)
    print('ONE-TIME UNLOCK COMPLETE:',status,flush=True)
    print('GATE DECISION',report['gate'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('run','unlock'));args=parser.parse_args()
    run() if args.phase=='run' else unlock()
