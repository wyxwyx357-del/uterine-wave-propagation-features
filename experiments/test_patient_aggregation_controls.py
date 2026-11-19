"""Minimal cross-wall identity check on twelve existing known conditions."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_activity_strength_trial import read,save,digest
from experiments.audit_wave_evidence import control_arrays
from experiments.patient_activity_aggregation import patient_summary


def run(output):
    output.mkdir(parents=True,exist_ok=True);rows=[];sources={}
    for texture in range(3):
        control=ROOT/f'outputs/p3_outer_ideal_prior_20261117_v2/texture{texture}_rigid.npz'
        a,v,t,s,span=control_arrays(control);sources[str(control)]=digest(control)
        maps=dict(A=a,valid_mask=v,interval_center_s=t,interval_dt_s=np.full(len(t),span),
                  interval_start_s=t-span/2,interval_end_s=t+span/2)
        for direction in ('C2F','F2C'):
            for family,expected in [('single',1),('double',2)]:
                name=f'texture{texture}_{family}_{direction}_0.04'+('_gap20' if family=='double' else '')
                sp=ROOT/'outputs/normalized_activity_trial_20261119_v1'/(name+'.json')
                op=ROOT/'outputs/global_ownership_trial_20261119_v1'/(name+'.json')
                selected=read(sp);original=read(op)
                sources[str(sp)]=digest(sp);sources[str(op)]=digest(op)
                result=patient_summary(selected,original,maps)
                # Truth association already evaluated independently in the
                # fixed salience experiment; it is not passed to aggregation.
                labels={}
                for w in selected['walls']:
                    for i,m in enumerate(w['metrics']['after_matches']):
                        if w['metrics']['after_supported_truth'][i]:labels[f"wall{w['wall']}/"+m['group_id']]=i
                represented={i:[] for i in range(expected)};wrong=[]
                for event in result['activities']:
                    ids={labels[n] for n in event['source_nodes'] if n in labels}
                    if len(ids)>1:wrong.append(event['patient_candidate_id'])
                    for i in ids:represented[i].append(event['patient_candidate_id'])
                row=dict(source=name,input_rows=result['input_wall_proposal_rows'],patient_rows=result['patient_candidate_rows'],
                    pairs_merged=result['cross_wall_pairs_merged'],expected_patient_activities=expected,
                    represented_per_truth={str(i):len(set(ids)) for i,ids in represented.items()},
                    merged_distinct_truths=wrong,conditional_count_range=result['conditional_identity_count_range'])
                row['passed']=result['patient_candidate_rows']==expected and not wrong and all(len(set(ids))==1 for ids in represented.values())
                rows.append(row);save(output/(name+'.json'),dict(summary=row,result=result));print(row,flush=True)
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Control inputs changed')
    summary=dict(conditions=len(rows),passed_conditions=sum(r['passed'] for r in rows),passed=all(r['passed'] for r in rows),
        source_files=sources,rows=rows,no_new_fixtures=True,no_threshold_tuning=True,
        code_sha256=digest(ROOT/'experiments/patient_activity_aggregation.py'))
    save(output/'summary.json',summary)
    print('FINAL',summary['conditions'],summary['passed_conditions'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
