"""Frozen three-point replay, with a mandatory aggregate gate before attribution."""
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK, load, save, sha, verify, make_case
from ..diagnostics.dev_propagation import DevDataAccess
from ..core import BaselineMinMaxScorer, continuous_branch, evaluate_threshold
from ..scorers import OutOfBandMADScorer
from ..upstream import extract_shared_upstream, stable_hash
from ..matching import activity_predictions, match_events

OUTPUT=WORK/'positive_unmatched_attribution'
POINTS={'baseline':('baseline_minmax',.6,(125,121,49,5,116)),
        'mad_low':('mad_oob',1.2065533477611825,(144,148,30,2,146)),
        'mad_high':('mad_oob',1.3897888189122913,(133,130,41,2,128))}
COUNT='significant_propagation_activity_count'

def guarded(output=OUTPUT):
    p=load(WORK/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json')
    p={**p,'sources':[s for s in p['sources'] if s['split']=='Dev'],'val_texture_ids':[]}
    guard=DevDataAccess(WORK,output,[s['texture_id'] for s in p['sources']])
    sys.addaudithook(guard.hook)
    verify(WORK)
    return p,guard

def gate(records):
    report={}
    for name,(_,tau,expected) in POINTS.items():
        total={k:sum(r['points'][name]['match'][k] for r in records) for k in ('TP','FP','FN')}
        negative=[r for r in records if not r['truth']];pools=defaultdict(list)
        for r in negative:pools[r['shared']['pool_hash']].append(r)
        duplicates=[]
        for h,rows in pools.items():
            outcomes=[stable_hash(r['points'][name]) for r in rows]
            assert len(set(outcomes))==1,'Same negative input gave different output'
            if len(rows)>1:duplicates.append(dict(pool_hash=h,cases=[r['case_id'] for r in rows],identical_outputs=True,
                                                 ledger_FP=rows[0]['points'][name]['match']['FP']))
        unique=[v[0] for v in pools.values()]
        lo=sum(r['points'][name]['evaluation']['result'][COUNT+'_min'] for r in unique)
        hi=sum(r['points'][name]['evaluation']['result'][COUNT+'_max'] for r in unique)
        positive=sum(r['points'][name]['match']['FP'] for r in records if r['truth'])
        actual=(total['TP'],total['FP'],total['FN'],hi,positive)
        assert actual==expected,(name,'MANDATORY LEDGER MISMATCH',actual,expected)
        assert lo==hi==expected[3]
        assert len(negative)==48 and len(unique)==42 and len(records)==186
        assert positive==total['FP']-hi,'Ledger/unique-negative accounting identity failed'
        report[name]=dict(threshold=tau,**total,raw_negative_rows=len(negative),unique_negative_inputs=len(unique),
                          duplicate_pool_hashes=duplicates,unique_negative_v2_min=lo,unique_negative_v2_max=hi,
                          raw_negative_ledger_FP=sum(r['points'][name]['match']['FP'] for r in negative),
                          positive_unmatched=positive,gate_passed=True)
    return report

def run():
    OUTPUT.mkdir(exist_ok=True)
    p,guard=guarded();before={str(f.relative_to(WORK)):sha(f) for f in sorted(guard.allowed)}
    old=load(WORK/'STEP2_DEV_RESULTS.json')
    historic={r['case_id']:r for r in old['cases']['baseline_minmax']}
    records=[]
    for source in p['sources']:
        bases={}
        for condition in ('static','rigid','brightness'):
            with np.load(WORK/'bases'/source['texture_id']/(condition+'.npz'),allow_pickle=False) as z:bases[condition]=dict(z)
        for scenario in p['scenarios']:
            maps,truth=make_case(source,scenario,bases,p)
            shared=extract_shared_upstream(maps,p['upstream_noise_floor'])
            case_id=source['texture_id']+'/'+scenario['scenario']
            assert shared['pool_hash']==historic[case_id]['pool_hash'],case_id
            branches={'baseline_minmax':continuous_branch(shared,BaselineMinMaxScorer(shared)),
                      'mad_oob':continuous_branch(shared,OutOfBandMADScorer(shared,p['S3a_band_cpm']))}
            points={}
            for name,(scorer,tau,_) in POINTS.items():
                ev=evaluate_threshold(shared,branches[scorer],tau)
                predictions=activity_predictions(ev['result'])
                matched=match_events(truth,[q for q in predictions if q['propagation_supported']],shared['span'],shared['exposure']['seconds'])
                points[name]=dict(evaluation=ev,predictions=predictions,match=matched)
            # Historical baseline prediction equality is an independent production-chain replay gate.
            assert stable_hash(points['baseline']['predictions'])==stable_hash(historic[case_id]['predictions']),case_id
            records.append(dict(case_id=case_id,source=source,scenario=scenario,shared=shared,truth=truth,
                                interval_start=maps['interval_start_s'],interval_end=maps['interval_end_s'],
                                branches=branches,points=points))
        print('Verified frozen Dev pools and baseline predictions:',source['texture_id'],flush=True)
    ledgers=gate(records)
    after={str(f.relative_to(WORK)):sha(f) for f in sorted(guard.allowed)}
    assert before==after
    save(OUTPUT/'REPLAY_GATE.json',dict(ledgers=ledgers,input_sha256=before,data_reads=sorted(guard.accesses)))
    with (OUTPUT/'replay.pkl').open('wb') as f:pickle.dump(dict(protocol=p,records=records),f,protocol=5)
    print('ALL THREE MANDATORY LEDGER GATES PASSED',ledgers,flush=True)

if __name__=='__main__':run()
