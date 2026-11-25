"""Scope regression on analytical fixtures only, no new/old real textures."""
import inspect,sys
from pathlib import Path
from analysis.mse_pregnancy_association.code import candidate_split_verifier as split
from ..benchmark_data import ROOT,WORK,load,sha,save
from ..fixtures import case,FIXED_NOISE_FLOOR
from ..upstream import extract_shared_upstream,stable_hash
from ..mechanism_2x2 import mechanisms as frozen

ATTEMPT=WORK/'pairing_filter_2x2/attempt_002'

def audit_scope():
    lock=load(ATTEMPT/'PAIRING_FILTER_2X2_PRELOCK.json')
    path=Path(inspect.getsourcefile(frozen))
    assert sha(path)==lock['experimental_source_sha256']['mechanisms.py'],'STOP: M1 implementation differs from attempt_002'
    maps,_=case('two');states={};calls={};results={};logs_by_model={}
    measure=frozen.CORE._measure_pair;pair_code=frozen.ORIGINAL_PAIR.__code__
    parent=dict(event_id='analytical_scope_container',start_s=float(maps['interval_center_s'][0]),
        end_s=float(maps['interval_center_s'][-1]),sections=list(range(len(maps['s']))))
    for model in ('M0','M1'):
        paths=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is pair_code:
                chain=[];caller=frame.f_back
                while caller is not None and len(chain)<12:
                    chain.append((Path(caller.f_code.co_filename).name,caller.f_code.co_name));caller=caller.f_back
                paths.append(chain)
        sys.setprofile(profile)
        try:
            logs=[]
            with frozen.intervention(model,logs):
                states[model]=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
                assert frozen.CORE._measure_pair is measure
                results[model]=split.prepare_parent(maps['A'][:,0],maps['valid_mask'][:,0],maps['interval_center_s'],maps['s'],parent,
                    noise_floor=FIXED_NOISE_FLOOR,measurement_span_s=2.04)
            logs_by_model[model]=logs
        finally:sys.setprofile(None)
        calls[model]=paths
    assert stable_hash(results['M0'])==stable_hash(results['M1']),'STOP: fixed local split verifier changed'
    assert [x['filtered_signature'] for x in logs_by_model['M0']]==[x['filtered_signature'] for x in logs_by_model['M1']]
    assert [x['raw_edge_signature'] for x in logs_by_model['M0']]==[x['raw_edge_signature'] for x in logs_by_model['M1']]
    global_paths=[p for p in calls['M1'] if any(name=='weighted_pair' for _,name in p)]
    local_paths=[p for p in calls['M1'] if any(file=='candidate_split_verifier.py' for file,_ in p)]
    assert global_paths and local_paths,'Scope fixture did not exercise both actual paths'
    assert all(not any(name=='weighted_pair' for _,name in p) for p in local_paths),'STOP: split pairing intercepted by M1'
    assert all(e['W_used']==e['W_new'] for log in logs_by_model['M1'] for e in log['edges'])
    return dict(passed=True,attempt_002_mechanisms_sha256=lock['experimental_source_sha256']['mechanisms.py'],
        current_mechanisms_sha256=sha(path),scope_hash_identical=True,
        intercept_path=global_paths[0],unmodified_split_path=local_paths[0],
        global_pair_call_count=len(global_paths),local_split_pair_call_count=len(local_paths),
        fixed_local_prepare_parent_output_hash=stable_hash(results['M1']),
        _measure_pair_unchanged=True,candidate_split_verifier_intercepted=False,analytical_fixture_only=True,
        no_MAD_no_filter_intervention_no_real_texture_access=True)

if __name__=='__main__':
    from .sources import OUTPUT
    save(OUTPUT/'M1_UNSEEN_TEXTURE_SCOPE_AUDIT.json',audit_scope())
    print('M1 scope audit passed: exact attempt_002 global-only scope; local split unchanged')
