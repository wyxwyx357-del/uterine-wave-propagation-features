"""Execute only the four prelocked mechanisms and two existing fixed scorers."""
import argparse,pickle,unittest,subprocess
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK,ROOT,save,sha,verify,load
from ..upstream import extract_shared_upstream,stable_hash
from ..diagnostic.replay import OUTPUT as PRIOR
from .protocol import ROOT_OUTPUT,access_data,validate,maps_from,FLAGS,code_hashes
from .mechanisms import MODELS,intervention,run_inventory
from .metrics import fixed_results,formation_result,phase_diagnostics,aggregate
from .preflight import ImplementationTests
from .report import produce,plot_cases

def asymmetric_fixture(ratio):
    begin=np.arange(751)*.12;end=begin+2.04;keep=end<=90.+1e-9;begin=begin[keep];end=end[keep];t=(begin+end)/2
    s=np.linspace(.1,.9,8);arrival=40+(s-.5)/.12;A=np.zeros((len(t),2,len(s)))
    for c,a in enumerate(arrival):
        pos=(t>=a-6.4)&(t<a);neg=(t>=a)&(t<=a+6.4)
        p=np.exp(-.5*((t[pos]-a+3.2)/1.6)**2);n=np.exp(-.5*((t[neg]-a-3.2)/1.6)**2)
        A[pos,:,c]=(.02*p/p.max())[:,None];A[neg,:,c]=(-.02*ratio*n/n.max())[:,None]
    truth=[dict(truth_event_id='asymmetric',truth_start_s=float(arrival.min()-6.4),truth_end_s=float(arrival.max()+6.4),truth_center_s=40.,
        truth_direction='C2F',truth_sections=list(range(8)),truth_amplitude_class='fixture_only',truth_amplitude=.02,truth_speed=.12)]
    return dict(A=A,valid_mask=np.ones_like(A,bool),s=s,interval_center_s=t,interval_start_s=begin,interval_end_s=end,interval_dt_s=end-begin),truth

def fixture_safety(protocol):
    result=[]
    for ratio in protocol['asymmetric_fixture']['ratios']:
        maps,truth=asymmetric_fixture(ratio)
        actual_ratio=(-maps['A'][:,0].min(axis=0))/maps['A'][:,0].max(axis=0)
        np.testing.assert_allclose(actual_ratio,ratio,rtol=0,atol=1e-15)
        for model in MODELS:
            logs=[]
            with intervention(model,logs):shared=extract_shared_upstream(maps,protocol['common_configuration']['noise_floor'])
            record=dict(case_id='fixture_asym_'+str(ratio),source=dict(texture_id='fixture_only'),scenario=dict(scenario='fixture_only'),
                shared=shared,truth=truth,input_hash=stable_hash(maps))
            formation=formation_result(record);downstream,_,_=fixed_results(record)
            result.append(dict(ratio=ratio,actual_raw_peak_ratio=actual_ratio.tolist(),model=model,formation=formation,
                S1=downstream['S1'],S2=downstream['S2'],excluded_from_benchmark=True))
    return result

def main(output):
    protocol=validate(output);p,access,old_records=access_data();before={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)}
    assert before==protocol['source_file_sha256']
    for record in old_records:
        assert stable_hash(maps_from(record))==protocol['anonymous_input_sha256'][record['case_id']]
        assert stable_hash(record['truth'])==protocol['truth_sha256'][record['case_id']]
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ImplementationTests))
    implementation=dict(tests_run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),passed=tests.wasSuccessful(),
        common_configuration_hash=stable_hash(protocol['common_configuration']),
        primary_scope=protocol['endpoints']['primary_A2'],source_sha256=code_hashes(),
        only_scoped_runtime_interventions=True,split_measurement_and_MAD_OOB_filter_unchanged=True)
    save(output/'PAIRING_FILTER_2X2_IMPLEMENTATION_AUDIT.json',implementation)
    if not tests.wasSuccessful():raise RuntimeError('Prelocked implementation test failed; stop')
    rows={m:{s:[] for s in ('S1','S2')} for m in MODELS};formation={m:[] for m in MODELS}
    phases={m:{k:[] for k in ('pattern_rows','cross_truth','filter_rows','ringing_like_selected_pairs')} for m in MODELS}
    pairing_audit=[];runs=[];unsupported=[];fixture_rows=[];cache={};scales={};inventory={}
    plot_ids={source['texture_id']+'/medium_C2F' for source in p['sources']}|{'case_001/coexist_medium_3'}
    for model in MODELS:
        for index,old in enumerate(old_records):
            maps=maps_from(old);case_id=old['case_id']
            if model in ('M2','M3'):
                if case_id not in inventory:
                    inventory[case_id]=run_inventory(maps,execute=True)
                    runs.extend(dict(r,case_id=case_id) for r in inventory[case_id])
                if any(r['M2_filter_unsupported'] for r in inventory[case_id]):
                    unsupported.append(dict(model=model,case_id=case_id,truth_bearing=bool(old['truth'])))
                    continue
            logs=[]
            with intervention(model,logs):shared=extract_shared_upstream(maps,p['upstream_noise_floor'])
            record=dict(case_id=case_id,source=old['source'],scenario=old['scenario'],truth=old['truth'],shared=shared,
                interval_start=old['interval_start'],interval_end=old['interval_end'],input_hash=protocol['anonymous_input_sha256'][case_id])
            assert stable_hash(shared['valid'])==stable_hash(old['shared']['valid'])
            assert stable_hash(shared['t'])==stable_hash(old['shared']['t'])
            assert stable_hash(shared['exposure'])==stable_hash(old['shared']['exposure']),'Frozen observation denominator changed'
            downstream,evaluations,mad_scales=fixed_results(record)
            if model=='M0':
                assert shared['pool_hash']==old['shared']['pool_hash'],case_id
                for scorer,prior in [('S1','baseline'),('S2','mad_low')]:
                    assert stable_hash(evaluations[scorer])==stable_hash(old['points'][prior]['evaluation']),case_id+'/'+scorer
                scales[case_id]=stable_hash(mad_scales)
            else:assert stable_hash(mad_scales)==scales[case_id],'MAD OOB denominator changed'
            formation[model].append(formation_result(record))
            for s in ('S1','S2'):rows[model][s].append(downstream[s])
            diagnostic=phase_diagnostics(record,model,logs)
            for key in phases[model]:phases[model][key].extend(diagnostic[key])
            for log in logs:
                pairing_audit.append(dict(case_id=case_id,texture_id=old['source']['texture_id'],model=model,wall=log['wall'],
                    edges=log['edges'],selected=log['selected'],raw_edge_signature=log['raw_edge_signature'],filtered_signature=log['filtered_signature']))
            if case_id in plot_ids:cache[case_id,model]=(record,logs)
            if (index+1)%31==0:print(model,old['source']['texture_id'],'31 frozen scenarios complete',flush=True)
        if model=='M0':
            a=aggregate(rows['M0']['S1']);b=aggregate(rows['M0']['S2'])
            assert (a['TP'],a['FP'],a['FN'],a['A2'],a['positive_unmatched'],a['negative']['FP_max'])==(125,121,49,73,116,5)
            assert (b['TP'],b['FP'],b['FN'],b['A2'],b['positive_unmatched'],b['negative']['FP_max'])==(144,148,30,90,146,2)
            implementation.update(all186_M0_reproduced=True,M0_S1=a,M0_S2=b)
            save(output/'PAIRING_FILTER_2X2_IMPLEMENTATION_AUDIT.json',implementation)
            print('M0 EXACT REPRODUCTION GATE PASSED; intervention Dev execution now allowed',flush=True)
        validate(output)
    fixture_rows=fixture_safety(protocol)
    result=produce(output,rows,formation,phases,pairing_audit,runs,fixture_rows,unsupported,implementation['common_configuration_hash'])
    figures=plot_cases(output,cache);save(output/'REPRESENTATIVE_FIGURES.json',figures)
    with (output/'REPRESENTATIVE_INPUTS.pkl').open('wb') as f:pickle.dump(cache,f,protocol=5)
    after={str(f.relative_to(WORK)):sha(f) for f in sorted(access.allowed)};assert before==after;validate(output)
    assert subprocess.run(['git','diff','--name-only'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()==''
    save(output/'PAIRING_FILTER_2X2_READ_ONLY_AUDIT.json',dict(**{k:False for k in FLAGS},
        prelock_sha256=sha(output/'PAIRING_FILTER_2X2_PRELOCK.json'),prelock_unchanged=True,input_hashes_unchanged=True,
        all186_M0_exact=True,common_configuration_hash=implementation['common_configuration_hash'],source_sha256=code_hashes(),
        protected_source_sha256=verify(WORK)['protected_source_sha256'],data_reads=sorted(access.accesses),
        unsupported_inputs=unsupported,formal_winner_allowed_from_filter_support=not any(r['truth_bearing'] for r in unsupported),
        interpretation_status='pending evidence review',
        output_sha256={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file() and f.name!='PAIRING_FILTER_2X2_READ_ONLY_AUDIT.json'}))
    print('COMPLETE PRIMARY RESULTS',result['fixed_S1_A2'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args();main(ROOT_OUTPUT/args.attempt)
