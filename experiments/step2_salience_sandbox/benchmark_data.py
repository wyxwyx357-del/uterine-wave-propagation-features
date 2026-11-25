"""Prelocked, source-separated real-texture controls and signal-level injections."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=HERE/'benchmark_run'
CONFIGS=('baseline_minmax','rigid_reference','mad_oob','mad_inband','percentile_p95','percentile_p99')
DT=.12
SPAN=2.04
NOISE_FLOOR=.0007509593468916695


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value,exclusive=False):
    from analysis.mse_pregnancy_association.code.measurement.io import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as stream:
        json.dump(jsonable(value),stream,ensure_ascii=False,indent=2,allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(work=WORK):
    path=work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'
    if sha(path)!=(work/'PRELOCK.sha256').read_text().strip():
        raise RuntimeError('Prelock changed; stop and invalidate this run')
    p=load(path)
    for name,h in p['benchmark_source_sha256'].items():
        if sha(HERE/name)!=h:
            raise RuntimeError('Sandbox code changed after prelock: '+name)
    for name,h in p['protected_source_sha256'].items():
        if sha(ROOT/name)!=h:
            raise RuntimeError('Protected implementation changed: '+name)
    return p


def scenarios():
    result=[]
    def add(name):
        result.append(dict(scenario=name,duration=60.,noise=1.,family='single',base='rigid',
                           amplitude=None,direction='C2F',artifact=None,ratio=None))
    # Build explicitly to avoid accidental duplicated keyword overrides.
    for name,base,noise in [('static','static',1),('rigid','rigid',1),('brightness','brightness',1),('noise3','rigid',3),('noise8','rigid',8)]:
        add('zero_'+name);result[-1].update(family='zero',base=base,noise=noise)
    for label,amplitude in [('clear',.08),('medium',.04),('structured_weak',.02)]:
        for direction in ('C2F','F2C'):
            add(f'{label}_{direction}');result[-1].update(amplitude=amplitude,direction=direction)
    for label,amplitude in [('medium',.04),('structured_weak',.02)]:
        for ratio in (2,3,5):
            add(f'coexist_{label}_{ratio}');result[-1].update(family='coexist',amplitude=amplitude,ratio=ratio)
    for artifact in ('spike','blob','brightness'):
        for wave in (False,True):
            add(f'artifact_{artifact}_{int(wave)}')
            result[-1].update(family='artifact',artifact=artifact,base='brightness' if artifact=='brightness' else 'rigid',amplitude=.04 if wave else None)
    for duration in (30.,90.):
        add(f'duration_{int(duration)}');result[-1].update(family='duration',duration=duration,amplitude=.04)
    for label,amplitude in [('clear',.08),('medium',.04),('structured_weak',.02)]:
        for noise in (3,8):
            add(f'gradient_{label}_{noise}');result[-1].update(family='noise',amplitude=amplitude,noise=noise)
    assert len(result)==31
    return result


def prepare_sources(work):
    """Private paths are used only to open the first frame; no outcome column is loaded."""
    import cv2
    full=ROOT.parent/'uterine-wave-propagation-features'
    records=load(full/'outputs/tracking_visual_qc_20261123/metadata/SELECTED_IDENTITY_LOCAL.json')
    by_id={r['anonymous_case_id']:r for r in records}
    # Prior recovery-development sources and the two incidentally exposed paths stay Dev.
    dev_order=[1,2,9,10,13,15,11,12,14]
    val_order=[3,4,5,6,7,8]
    patient=lambda r:hashlib.sha256(r['case_id'].rsplit('-',1)[0].split('-',1)[-1].upper().encode()).hexdigest()
    prior_groups={patient(by_id[f'case_{i:03d}']) for i in dev_order}
    val=[];groups=set()
    for i in val_order:
        r=by_id[f'case_{i:03d}'];g=patient(r)
        if g not in prior_groups and g not in groups:
            val.append(r);groups.add(g)
    target=min(6,len(val));dev=[];dev_groups=set()
    for i in dev_order:
        r=by_id[f'case_{i:03d}'];g=patient(r)
        if g not in groups and g not in dev_groups and len(dev)<target:
            dev.append(r);dev_groups.add(g)
    target=min(len(dev),len(val));dev=dev[:target];val=val[:target]
    if target<2:
        raise RuntimeError('Insufficient independent source groups for texture-separated development/validation')
    sources=[]
    for split,rows,seed0 in [('Dev',dev,2026112500),('Val',val,2027112500)]:
        for i,r in enumerate(rows):
            source_id=r['anonymous_case_id'];cap=cv2.VideoCapture(r['video_path'])
            ok,frame=cap.read();cap.release()
            if not ok:
                raise ValueError('First frame unavailable: '+source_id)
            frame=cv2.resize(frame,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            with np.load(Path(r['folder'])/'layout.npz',allow_pickle=False) as z:
                grid={k:z[k] for k in ('u','A_points','A_pair_ids','B_points','B_curve_ids','arc_positions','B_arc_positions')}
            lengths=np.linalg.norm(grid['A_points'][grid['A_pair_ids'][:,:,1]]-grid['A_points'][grid['A_pair_ids'][:,:,0]],axis=-1)
            np.testing.assert_allclose(lengths,21.,atol=5e-5,rtol=0)
            source_file=work/'private'/f'{source_id}.npz';source_file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(source_file,gray=gray,**grid)
            seed=seed0+i; rng=np.random.default_rng(seed)
            sources.append(dict(texture_id=source_id,split=split,seed=seed,patient_group_sha256=patient(r),
                source_key_sha256=hashlib.sha256(r['case_id'].encode()).hexdigest(),
                first_frame_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),source_npz_sha256=sha(source_file),
                shape=gray.shape,native_sections=len(grid['u']),
                phase=float(rng.uniform(.1,.5)),event_jitter=float(rng.uniform(-.2,.2)),
                artifact_jitter=int(rng.integers(-1,2)),
                provenance='one native first frame and frozen 21px layout per source examination; physical patient-name grouping hashed locally; no video/outcome path in public protocol'))
    d=[r for r in sources if r['split']=='Dev'];v=[r for r in sources if r['split']=='Val']
    for key in ('patient_group_sha256','source_key_sha256','first_frame_sha256'):
        if {r[key] for r in d}&{r[key] for r in v}:
            raise RuntimeError('Texture/source leakage: '+key)
    save(work/'SOURCE_PROVENANCE.json',dict(sources=sources,texture_leakage=0,
        selection='Dev: exposed or prior recovery-development groups first; Val: anonymous 003-008, excluding all known Dev groups; balanced without outcome/feature ranking',
        incidental_path_exposure='Two technical-list video paths incidentally displayed outcome-bearing directory names before prelock. Both assigned Dev. No outcome table, label extraction or outcome-dependent scoring.',
        limitation='Held out for the current salience benchmark; not independent clinical validation of A0 or the historical upstream noise floor.'))
    return sources


def select_band(work):
    records=load(HERE/'artifacts/STEP2_SCORER_DEFINITIONS.json')['Step4a_out_of_band_residual_stability']
    bands=((10.,20.),(10.,25.),(15.,25.));rows=[]
    # Selection is based on the complete, paired synthetic zero-deformation panel.
    # The three historical real controls lack patient provenance here and cannot influence held-out selection.
    for band in bands:
        chosen=[r for r in records if tuple(r['band_cpm'])==band]
        paired={condition:[r for r in chosen if r['control'].startswith('synthetic_') and f'_{condition}_' in r['control']]
                for condition in ('rigid','brightness')}
        values={c:np.array([r['scale_median'] for r in rs]) for c,rs in paired.items()}
        if any(len(x)!=3 for x in values.values()):
            raise ValueError('Expected the three paired Step4a synthetic textures')
        cv={c:float(x.std(ddof=1)/x.mean()) if x.mean()>0 else None for c,x in values.items()}
        mismatch=float(abs(np.log(values['rigid'].mean()/values['brightness'].mean())))
        all_scales=np.array([x for r in chosen for wall in r['section_scales'] for x in wall if x is not None])
        expected=sum(len(wall) for r in chosen for wall in r['section_scales'])
        rows.append(dict(band_cpm=band,P50_scale=float(np.percentile(all_scales,50)),P95_scale=float(np.percentile(all_scales,95)),
            CV=cv,rigid_brightness_log_ratio=mismatch,missing_fraction=1-len(all_scales)/expected,
            selection_cost=float(np.mean(list(cv.values()))+.5*mismatch),
            distance_above_5_cpm=band[0]-5,distance_below_first_difference_zero_cpm=60/SPAN-band[1],
            selection_panel='three paired synthetic Dev controls; static controls and historical unknown-provenance real textures reported only as diagnostics'))
    minimum=min(r['selection_cost'] for r in rows)
    selected=next(r for r in rows if r['selection_cost']<=minimum*1.05+1e-15)
    result=dict(candidates=rows,chosen_band=selected['band_cpm'],
        rule='Before inspection: cost=mean within-condition cross-texture CV + 0.5*abs(log(rigid/brightness mean scale)); choose first band in [(10,20),(10,25),(15,25)] within 5% of minimum. Require finite positive paired scales, complete coverage and bands between passband and first difference zero.',
        no_wave_recall_F1_or_Val_used=True,scope='Step4a zero-deformation residuals only; synthetic paired panel controls the choice to prevent unknown historical-source overlap with Val')
    save(work/'STEP2_OOB_BAND_SELECTION.json',result,exclusive=True)
    return result


def prelock(work):
    from .audit import protected_snapshot
    work.mkdir(parents=True,exist_ok=True)
    if (work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json').exists():
        raise RuntimeError('Prelock already exists; it must not be overwritten')
    # This rule file is created before inspecting any Step4a band values.
    save(work/'OOB_SELECTION_RULE_BEFORE_INSPECTION.json',dict(cost='mean CV + .5*absolute log rigid/brightness mean ratio',
        near_tie_relative=.05,band_order=[[10,20],[10,25],[15,25]],selection_controls='paired synthetic Step4a controls only'),exclusive=True)
    band=select_band(work);sources=prepare_sources(work)
    p=dict(version=1,base_commit='8ad7f133df7009de845263ae4689472e41f8fb81',
        sources=sources,dev_texture_ids=[s['texture_id'] for s in sources if s['split']=='Dev'],
        val_texture_ids=[s['texture_id'] for s in sources if s['split']=='Val'],texture_leakage=0,
        scenarios=scenarios(),configs=CONFIGS,seeds=[s['seed'] for s in sources],
        dt_s=DT,nominal_interval_s=2.,actual_interval_s=SPAN,upstream_noise_floor=NOISE_FLOOR,
        amplitude_classes=dict(clear_min_stretch=.08,medium_min_stretch=.04,structured_weak_min_stretch=.02,
            sigma_s=1.6,finite_support_sigma=4,speed_full_path_per_s=.12,
            structured_weak_clean_filtered_RMS_minimum=3*.000222,
            residual_reference='Step4a upper real-texture filtered RMS rounded to .000222 s^-1; amplitude requirement is an engineering fixture definition, not a physiological detection cutoff'),
        timing=dict(Dev_center=16.,Val_center=17.,strong_center_offset=27.,artifact_Dev_center=42.,artifact_Val_center=45.,
                    artifact_Dev_position_fraction=.45,artifact_Val_position_fraction=.65),
        artifacts=dict(spike_A=.08,spike_frames=1,blob_A=.03,blob_support_s=4.,blob_sigma_s=.7,
                       blob_adjacent_sections=5,blob_section_delay_s=.08,brightness_gain_amplitude=.10,brightness_period_s=20.),
        noise_levels=[1,3,8],noise_definition='multiply same real-texture rigid A0 residual, preserving temporal/spatial correlation; experimental stress levels, not clinical distribution',
        zero_control_generation=dict(duration_s=90.,native_frames=751,conditions=['static','rigid','brightness'],
             rigid_dx='3*(sin(2*pi*t/30+phase)-sin(phase))',rigid_dy='2*(cos(2*pi*t/30+phase)-cos(phase))',
             brightness='gain=1+.10*(sin(2*pi*t/20+phase)-sin(phase)); round/clip uint8',
             pose='exact inverse analytic translations; pose estimator is not evaluated',cv2_threads=1),
        scorers_definition_sha256=sha(HERE/'artifacts/STEP2_SCORER_DEFINITIONS.json'),
        S3a_band_cpm=band['chosen_band'],band_selection_sha256=sha(work/'STEP2_OOB_BAND_SELECTION.json'),
        matching=dict(implementation='locked sandbox matching.py',time='inclusive truth support +/- actual measurement span',
                      shared_native_sections_min=3,objective='maximum cardinality then minimum total center deviation',direction_gate=False),
        threshold_selection=dict(baseline=.6,baseline_reoptimized=False,safety_unique_input_dedup=True,
             nonbaseline_grid='Dev-only section-score quantiles at 0,.05,...,1 plus nextafter(min,-inf) and 0; no adaptive refinement',
             gate1='pooled Dev zero and no-wave artifact FP/min <= .05 if total minutes>=20; otherwise exact FP count <= baseline with all counts/minutes disclosed',
             gate2='maximize pooled clear+medium-only case event F1',
             tie_break=['lower FP/min','higher clear+medium recall','lower mean distance-to-count-interval','higher structured-weak recall','higher/conservative tau'],
             tie_tolerance=1e-12,conditional_recall='salience-only matching among upstream-represented truth IDs; downstream and upstream misses reported separately'),
        winner_rules=dict(safety='Val zero and no-wave artifact FP counts and positive-case fractions must each be <= fixed production baseline',
             structured='Val clear recall, medium recall and pooled clear+medium event F1 must each be >= baseline',
             minimum_defect_improvement=.10,improvement_both_Dev_and_Val=True,
             defects=['coexistence delta recall','blob artifact recall-drop reduction','duration joint correct detection, not agreement of two misses','mean high-noise event F1'],
             critical_regression_margin=.05,interval_distance_must_not_increase=True,
             inference='paired source/scenario differences and Cliff delta; no P-value winner search; no population CI from six textures',
             final_decision='any qualified -> replacement candidate; otherwise safety+structured pass but key regression -> no method clearly superior; otherwise retain fixed production baseline'),
        metrics=['exact FP count/control minutes','FP/min','positive-control-case fraction','upstream recall','conditional salience recall',
                 'end-to-end recall by amplitude','Precision','Recall','F1','interval coverage','distance-to-interval','auxiliary midpoint MAE',
                 'direction accuracy/C2F/F2C/unresolved','coexistence paired delta recall','artifact deltas','duration agreement/count/frequency/score drift',
                 'noise gradient','proposal-level S_crit PR/FROC','Cliff delta'],
        count_interpretation='Step1 min/max/exact unchanged; ledger-level event metrics never replace identity-ambiguity count bounds',
        pregnancy_outcomes_read=False,clinical_AUC_calculated=False,production_modified=False,
        incidental_path_label_exposure=True,exposure_scope='two Dev-source directory names only; no outcome table or labels enter the benchmark',
        independent_validation_scope='texture-held-out salience development experiment, not acoustic resimulation, independent clinical validation, or A0/noise-floor validation',
        benchmark_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        bug_policy='After formal start, code/protocol bug -> stop, invalidate old results, create a new protocol version before restarting; never edit this prelock',
        protected_source_sha256=protected_snapshot(),
        benchmark_source_sha256={f.name:sha(f) for f in sorted(HERE.glob('*.py'))})
    save(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json',p,exclusive=True)
    (work/'PRELOCK.sha256').write_text(sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'),encoding='ascii')
    print('PRELOCK',len(p['dev_texture_ids']),len(p['val_texture_ids']),p['S3a_band_cpm'],flush=True)


def generate(work,split):
    import cv2
    from analysis.mse_pregnancy_association.code.measurement.tracking import track_segments
    from analysis.mse_pregnancy_association.code.measurement.deformation import compute_spacetime
    p=verify(work)
    if split=='Val' and not (work/'STEP2_DEV_THRESHOLDS.json').exists():
        raise RuntimeError('Dev thresholds must be frozen before any Val controls are generated')
    save(work/(split.upper()+'_GENERATION_STARTED.json'),dict(prelock_sha256=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json')),exclusive=True)
    if split=='Val':
        save(work/'VAL_RUN_STARTED.json',dict(prelock_sha256=sha(work/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json'),threshold_sha256=sha(work/'STEP2_DEV_THRESHOLDS.json')),exclusive=True)
    if cv2.cuda.getCudaEnabledDeviceCount()<1:
        raise RuntimeError('Frozen CUDA A0 required; no fallback')
    cv2.setNumThreads(1)
    t=np.arange(751)*DT
    for source in p['sources']:
        if source['split']!=split:continue
        path=work/'private'/(source['texture_id']+'.npz')
        if sha(path)!=source['source_npz_sha256']:raise RuntimeError('Source changed')
        with np.load(path,allow_pickle=False) as z:raw=dict(z)
        gray=raw.pop('gray');grid=raw;initial=np.vstack((grid['A_points'],grid['B_points']))
        for condition in ('static','rigid','brightness'):
            frames=[];poses=np.repeat(np.eye(3)[None],len(t),axis=0)
            for i,time in enumerate(t):
                phi=source['phase']
                dx=3*(np.sin(2*np.pi*time/30+phi)-np.sin(phi)) if condition=='rigid' else 0
                dy=2*(np.cos(2*np.pi*time/30+phi)-np.cos(phi)) if condition=='rigid' else 0
                frame=cv2.warpAffine(gray,np.array([[1,0,dx],[0,1,dy]],float),(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101) if condition=='rigid' else gray
                if condition=='brightness':
                    gain=1+.10*(np.sin(2*np.pi*time/20+phi)-np.sin(phi))
                    frame=np.uint8(np.clip(np.rint(gray.astype(float)*gain),0,255))
                frames.append(frame);poses[i,:2,2]=[-dx,-dy]
            tracks=track_segments(frames,initial,t,poses)
            maps=compute_spacetime(tracks,grid,interval_s=2.)
            destination=work/'bases'/source['texture_id'];destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/(condition+'.npz'),**maps)
            np.savez_compressed(destination/(condition+'_tracks.npz'),**tracks)
            save(destination/(condition+'_audit.json'),dict(source_sha256=source['source_npz_sha256'],
                 split=split,condition=condition,truth_A=0.,valid_fraction=float(maps['valid_mask'].mean()),
                 reanchors=int(tracks['reanchor'].sum()),map_sha256=sha(destination/(condition+'.npz')),
                 tracks_sha256=sha(destination/(condition+'_tracks.npz')),cv2_version=cv2.__version__,
                 truth_basis='constant initial radial lengths under analytic translations or photometric-only perturbation'))
            print(split,source['texture_id'],condition,'complete',flush=True)
    verify(work)
    save(work/(split.upper()+'_GENERATION_COMPLETE.json'),dict(completed=True),exclusive=True)


def amplitude_class(amplitude):
    return 'clear' if amplitude>=.08 else 'medium' if amplitude>=.04 else 'structured_weak'


def make_case(source,scenario,bases,p):
    base=bases[scenario['base']]
    keep=base['interval_end_s']<=scenario['duration']+1e-9
    maps={k:(v[keep].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v)
          for k,v in base.items()}
    maps['A']=maps['A']*scenario['noise']
    truth=[];s=maps['s'];t=maps['interval_center_s'];span=maps['interval_dt_s']
    first=p['timing'][source['split']+'_center']+source['event_jitter']
    def add(event_id,center,amplitude,direction):
        arrival=center+(1 if direction=='C2F' else -1)*(s-.5)/.12
        def length(time):
            offset=time[:,None]-arrival
            return 1+amplitude*np.where(abs(offset)<=6.4,np.exp(-.5*(offset/1.6)**2),0)
        begin=length(maps['interval_start_s']);end=length(maps['interval_end_s'])
        signal=(end-begin)/begin/span[:,None]
        maps['A']+=signal[:,None,:]
        truth.append(dict(truth_event_id=event_id,truth_start_s=float(arrival.min()-6.4),truth_end_s=float(arrival.max()+6.4),
            truth_center_s=float(center),truth_direction=direction,truth_sections=list(range(len(s))),
            truth_amplitude_class=amplitude_class(amplitude),truth_amplitude=float(amplitude),truth_speed=.12))
    if scenario['amplitude'] is not None:
        add('core',first,scenario['amplitude'],scenario['direction'])
        if scenario['family']=='coexist':add('strong',first+27,scenario['amplitude']*scenario['ratio'],'F2C')
    artifact=scenario['artifact']
    if artifact in ('spike','blob'):
        at=p['timing']['artifact_'+source['split']+'_center']+source['event_jitter']
        middle=int(round(p['timing']['artifact_'+source['split']+'_position_fraction']*(len(s)-1)))+source['artifact_jitter']
        columns=np.arange(max(0,middle-2),min(len(s),middle+3))
        for c in columns:
            if artifact=='spike':maps['A'][int(np.argmin(abs(t-at))),0,c]+=.08
            else:
                offset=t-at-.08*(c-middle)
                maps['A'][:,0,c]+=.03*np.where(abs(offset)<=2,np.exp(-.5*(offset/.7)**2),0)
    return maps,truth


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prelock','generate'))
    parser.add_argument('--split',choices=('Dev','Val'));parser.add_argument('--work',type=Path,default=WORK)
    args=parser.parse_args()
    prelock(args.work) if args.mode=='prelock' else generate(args.work,args.split)
