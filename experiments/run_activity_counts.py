"""Recompute ownership and fixed-.6 significant counts on saved 87+12 inputs."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activity_counting import count_activities
from experiments.global_event_ownership import reconcile
from experiments.patient_activity_aggregation import candidate_nodes,observation_union
from experiments.normalized_activity import normalize_magnitude,score_salience
from experiments.activity_strength import filtered_runs
from experiments.audit_wave_evidence import control_arrays
from experiments.test_double_event_signals import double_fixture
from experiments.test_weak_wave_signals import radial_fixture
from experiments.test_activity_strength_trial import read,save


def run():
    root=ROOT/'outputs';out=root/(sys.argv[1] if len(sys.argv)>1 else 'significant_activity_counts_20261119_v2');out.mkdir(exist_ok=True)
    paths=list(sorted((root/'global_ownership_trial_20261119_v1').glob('texture*.json')))
    paths+=list(sorted((root/'global_ownership_real_20261119_v1').glob('case_*.json')))
    summary=[]
    for path in paths:
        old=read(path);real=path.stem.startswith('case_')
        if real:
            with np.load(root/'cohort_wave_review_20261118_v1'/path.stem/'maps.npz') as z:maps=dict(z)
            a,v,t,s=maps['A'],maps['valid_mask'],maps['interval_center_s'],maps['s'];span=float(np.median(maps['interval_dt_s']))
            source_walls=old['walls']
            reports=[w['parent_report'] for w in source_walls];splits=[w['local_split_proposals'] for w in source_walls]
        else:
            parent=read(root/'event_input_smoothing_20261118_v1'/path.name)
            local=read(root/'candidate_split_trial_20261119_v1'/path.name)
            texture=parent['texture'];family=old['family']
            cp=root/'p3_outer_ideal_prior_20261117_v2'/(path.stem+'.npz' if family=='image_control' else texture+'_rigid.npz')
            a,v,t,s,span=control_arrays(cp)
            if family=='double':a=a+double_fixture(t,s,span,parent['expected_direction'],parent['gap_s'],parent['second_amplitude'])[0][:,None]
            elif family=='single_or_negative':
                kind,amp=path.stem[len(texture+'_single_'):].rsplit('_',1)
                a=a+radial_fixture(t,s,span,kind,float(amp))[0][:,None]
            maps=dict(A=a,valid_mask=v,interval_start_s=t-span/2,interval_end_s=t+span/2)
            reports=[w['report'] for w in parent['methods']['periodic_frozen']['walls']]
            splits=[w['versions']['3']['proposals'] for w in local['walls']]
        normalized,metadata=normalize_magnitude(filtered_runs(a,v,t),v)
        ownership=dict(walls=[]);selected=dict(walls=[])
        for w in (0,1):
            result=reconcile(reports[w],splits[w],t,v[:,w],measurement_span_s=span)
            ownership['walls'].append(dict(reconciliation=result))
            selected['walls'].append(dict(wall=w,scores=[score_salience(e,result['source_evidence'],normalized[:,w],v[:,w],t) for e in result['event_proposals']]))
        nodes=candidate_nodes(selected,ownership)
        result=count_activities(nodes,s,span,float(np.median(np.diff(t))),observation_union(maps)['seconds'])
        result.update(source=path.stem,real=real,normalization=metadata)
        save(out/path.name,result)
        summary.append({k:result[k] for k in ('source','real','significant_propagation_activity_count','significant_propagation_activity_frequency_cpm','other_significant_activity_count')})
    save(out/'summary.json',dict(rows=summary,cutoff=.6,direction_used_for_counting=False))
    for r in summary:
        if r['real'] or 'rigid' in r['source'] or 'local_once' in r['source'] or 'synchronous' in r['source'] or r['source']=='texture0_double_F2C_0.04_gap10':print(r,flush=True)


if __name__=='__main__':run()
