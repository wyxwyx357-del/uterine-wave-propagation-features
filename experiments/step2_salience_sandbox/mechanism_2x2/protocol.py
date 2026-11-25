"""Generate immutable attempt protocols BEFORE running any intervention."""
import argparse,pickle,subprocess,sys
from pathlib import Path
import scipy
from ..benchmark_data import WORK,ROOT,load,save,sha,verify
from ..diagnostics.dev_propagation import DevDataAccess
from ..diagnostic.replay import OUTPUT as PRIOR
from ..upstream import stable_hash

HERE=Path(__file__).resolve().parent
ROOT_OUTPUT=WORK/'pairing_filter_2x2'
FLAGS=('Val_read','pregnancy_outcome_read','clinical_AUC_calculated','production_changed','production_filter_changed',
       'production_pairing_changed','threshold_retuned','new_filter_searched','new_pairing_formula_searched','Step3_started','400plus_cohort_run')

def access_data():
    p=load(WORK/'STEP2_BENCHMARK_PROTOCOL_PRELOCK.json');p={**p,'sources':[s for s in p['sources'] if s['split']=='Dev'],'val_texture_ids':[]}
    guard=DevDataAccess(WORK,ROOT_OUTPUT,[s['texture_id'] for s in p['sources']])
    guard.allowed.update(PRIOR/name for name in ('replay.pkl','DEV_POSITIVE_FP_READ_ONLY_AUDIT.json','DEV_POSITIVE_FP_ATTRIBUTION_SUMMARY.json'))
    sys.addaudithook(guard.hook);verify(WORK)
    audit=load(PRIOR/'DEV_POSITIVE_FP_READ_ONLY_AUDIT.json');assert sha(PRIOR/'replay.pkl')==audit['replay_cache_sha256']
    with (PRIOR/'replay.pkl').open('rb') as f:data=pickle.load(f)
    assert stable_hash(p)==stable_hash(data['protocol'])
    return p,guard,data['records']

def maps_from(record):
    s=record['shared']
    return dict(A=s['A'],valid_mask=s['valid'],s=s['s'],interval_center_s=s['t'],interval_start_s=record['interval_start'],
                interval_end_s=record['interval_end'],interval_dt_s=record['interval_end']-record['interval_start'])

def code_hashes():return {f.name:sha(f) for f in sorted(HERE.glob('*.py'))}

def validate(attempt):
    protocol=load(attempt/'PAIRING_FILTER_2X2_PRELOCK.json')
    assert sha(attempt/'PAIRING_FILTER_2X2_PRELOCK.json')==(attempt/'PRELOCK.sha256').read_text().strip()
    assert protocol['experimental_source_sha256']==code_hashes(),'Implementation changed: invalidate attempt and restart at M0 with a new prelock'
    verify(WORK)
    return protocol

def prepare(attempt):
    p,guard,records=access_data()
    attempt.mkdir(parents=True,exist_ok=False)
    # A later attempt invalidates prior results, never overwrites their prelock.
    for previous in ROOT_OUTPUT.glob('attempt_*'):
        if previous!=attempt and (previous/'PAIRING_FILTER_2X2_PRELOCK.json').exists() and not (previous/'INVALIDATED.json').exists():
            save(previous/'INVALIDATED.json',dict(reason='Implementation revision; prior results invalid; new attempt starts from M0',replaced_by=attempt.name))
    baseline=verify(WORK)
    protocol=dict(experiment='Prelocked Dev-only pairing x filter 2x2',HEAD=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
        experimental_source_sha256=code_hashes(),frozen_benchmark_source_sha256=baseline['benchmark_source_sha256'],
        protected_source_sha256=baseline['protected_source_sha256'],scipy_version=scipy.__version__,
        dev_texture_ids=[s['texture_id'] for s in p['sources']],scenario_ids=[s['scenario'] for s in p['scenarios']],
        anonymous_input_sha256={r['case_id']:stable_hash(maps_from(r)) for r in records},truth_sha256={r['case_id']:stable_hash(r['truth']) for r in records},
        source_file_sha256={str(f.relative_to(WORK)):sha(f) for f in sorted(guard.allowed)},
        M0=dict(filter='unchanged production 0.5-5 cpm FFT rectangular mask',pairing='unchanged production additive association weights'),
        M1=dict(filter='M0',pairing='original _pair_phases solver with W_old * G'),
        M2=dict(filter='one Butterworth order 2, 0.5-5 cpm, scipy sosfiltfilt defaults',pairing='M0'),
        M3=dict(filter='M2',pairing='M1'),
        filter_parameters=dict(order=2,band_Hz=[.5/60,5/60],sampling='1/median(diff(actual run times))',output='sos',
            zero_phase='sosfiltfilt',axis=0,padtype='odd',padlen=None,demean_before_filter=False,minimum_run='unchanged max(3,ceil(12/median_dt))',
            unsupported='ValueError logged per native run; no parameter fallback, no fair M2/M3 winner if any truth-bearing input unsupported'),
        amplitude_factor=dict(formula='G = median over original edge sections of min(a_i/R, a_j/R); W_new=W_old*G',
            amplitudes='max abs(actual detector filtered A) over inclusive original lobe [first,last]',
            R='same video/wall/pre-weight connected pairing component/section max over all participating ridges having a lobe there',
            zero_R_factor=0,epsilon=None,offset=None,lambda_parameter=None,normalization_global=False,
            missing='No section exclusion: production lobe amplitude invariants must hold or stop',
            scope='global phase pairing only; original _measure_pair, local split measurement_span_mean and split decisions unchanged'),
        downstream=dict(S1=dict(scorer='unchanged Min-Max',threshold=.6),S2=dict(scorer='unchanged MAD OOB',band_cpm=[10,20],threshold=1.2065533477611825),
            M2_scoring_numerator='same experimental filtered A',MAD_denominator='unchanged FFT raw-A 10-20 cpm; never retuned'),
        common_configuration=dict(noise_floor=p['upstream_noise_floor'],support_sections=3,matcher='unchanged production time gate + >=3 shared sections; direction not TP gate',
            continuous_runs='original per-wall/section valid & finite A',observation='frozen Step1 filtered-support denominator',
            split='unchanged prepare_parent/decide_split, same minimum support 3',ownership='unchanged reconcile',cross_wall='unchanged count_activities identity',propagation='unchanged retained-node classifier'),
        endpoints=dict(primary_A2='Same exact prior A>A1/A2 definition among S1 final propagation predictions; D0 must equal 73. Formation-no-salience A2 reported separately, with same production nodes/propagation/identity and full native proposal sections.',
            primary_multiplicity='All 174 truths, number of propagation predictions passing frozen matcher, separately formation/S1/S2',
            primary_four_lobe='Per truth x wall (348 rows): main-positive/main-negative candidate edge, identified by lobe containment of known truth component main extrema, unselected while its two ridges belong to distinct selected negative-first pairs',
            central_selection='All 348 truth-wall rows denominator; edge availability also reported',
            cross_truth_pairing_error='Selected edge whose two phase ridges each uniquely contain main extrema of DIFFERENT known truths on >=3 adjacent original edge sections. Missing/ambiguous assignment separately reported; not assumed safe.',
            truth_reference='Diagnostic only, AFTER algorithm: known injected component through same condition filter, sign-specific main extremum before/after deformation center within original raw-component support; no truth enters pairing/filter/scoring',
            filter_peripheral='Pure truth component lobes from unchanged observed_lobes/noise floor; those not containing sign-specific main extremum; ratios relative to that main amplitude',
            ringing_like='Selected native-wall phase pair with >=3 adjacent sections having phase peaks outside ALL raw truth support and same-sign filtered truth component contribution; structural flag, not causal proof',
            secondary=['TP','FP','FN','precision','recall','F1','negative unique-input Step1 FP/min min/max','positive unmatched','count interval coverage','distance to interval'],
            safety=['clear/medium/structured weak recall','coexist 2/3/5 matched TP FN duplicate FP precision recall predictions/truth','cross-truth pairing','per-texture consistency']),
        negative_dedup='Raw A/valid/native axes/measurement intervals hash, not scenario name; expect 48 rows, 42 unique; duplicate outputs must agree',
        asymmetric_fixture=dict(ratios=[1.,.7,.4],duration_s=90.,dt=.12,sections=8,s_range=[.1,.9],
            signal='Disjoint positive Gaussian on arrival-6.4<=t<arrival, center arrival-3.2; negative Gaussian on arrival<=t<=arrival+6.4, center arrival+3.2; sigma1.6. Normalize each native sampled half to peak1, then multiply by +0.02 and -0.02*ratio. No overlap; actual raw peak ratios exactly 1:ratio.',
            arrival='40+(s-.5)/.12',measurement_span_s=2.04,truth_support='arrival +/- 6.4',
            role='Only minimal mechanism safety fixture, excluded from benchmark interpretation; existing scenarios have no deliberately fixed 1:.7 or 1:.4 phase-asymmetry strata'),
        implementation_tests=['M0 identical fixture and all186 replay','G=1 exact degeneration','four-lobe manual weight check','symmetric sign invariant G',
            '1:-.4 absolute ratio','constant/zero finite M2','invalid gap independent M2 states','common config and original split/classifier interfaces unchanged'],
        effect_formulas=dict(R_P='(D0-D1)/D0',R_F='(D0-D2)/D0',R_C='(D0-D3)/D0',R_F_given_P='(D1-D3)/D0'),
        interpretation='Only user cases A-E, jointly assess duplication/recall/coexist/negative safety/texture; no new numeric veto or automatic F1 winner',
        stop_rules=['Any implementation test failure stops before intervention Dev run','M0 mismatch stops','Changed code/protocol invalidates attempt and restarts from M0',
            'No threshold/filter/formula search','Only 4 models and 2 fixed scorers','No added mechanism fixture ratios','Stop after outputs; no production adoption'],
        **{k:False for k in FLAGS},pregnancy_read=False)
    save(attempt/'PAIRING_FILTER_2X2_PRELOCK.json',protocol,exclusive=True)
    (attempt/'PRELOCK.sha256').write_text(sha(attempt/'PAIRING_FILTER_2X2_PRELOCK.json')+'\n',encoding='ascii')
    print('PRELOCK CREATED',attempt.name,sha(attempt/'PAIRING_FILTER_2X2_PRELOCK.json'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args();prepare(ROOT_OUTPUT/args.attempt)
