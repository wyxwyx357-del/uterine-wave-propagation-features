"""Run infrastructure gates and write the five requested audit artifacts.

Run: python -m experiments.step2_salience_sandbox.audit
No alternative-scorer thresholds, PR/FROC scans, benchmark matrix or winner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np

from analysis.mse_pregnancy_association.code.measurement.io import jsonable
from analysis.mse_pregnancy_association.code.step4a_noise_calibration import cached_maps
from .fixtures import KINDS,case,FIXED_NOISE_FLOOR
from .upstream import extract_shared_upstream,stable_hash,pool_hash
from .core import (BaselineMinMaxScorer,continuous_branch,evaluate_threshold,
                   baseline_equivalence_gate,BaselineEquivalenceError)
from .scorers import (DevRigidBank,PerSectionRigidReferenceScorer,OutOfBandMADScorer,
                      InBandMADScorer,PercentileScorer,mad_scale)
from .matching import (MIN_SHARED_NATIVE_SECTIONS,match_events,activity_predictions,
                       upstream_recall,miss_attribution)

ROOT=Path(__file__).resolve().parents[2]
SANDBOX=Path(__file__).resolve().parent
OOB_BANDS=((10.,20.),(10.,25.),(15.,25.))  # Inspection candidates only, no selected band.


def save(path,value):
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def protected_snapshot():
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    prefix=SANDBOX.relative_to(ROOT).as_posix()+'/'
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names
            if not p.startswith(prefix) and (p.endswith('.py') or p=='docs/ACTIVITY_COUNT_V2_STEP1_FREEZE.json')}
    changes=subprocess.check_output(['git','diff','881d87a','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(not p.startswith(prefix) for p in changes):
        raise RuntimeError('Production, Step 1 or Step 4a changed relative to the supplied base')
    return hashes


def branch_scores(branch):
    return [dict(wall=w['wall'],event_proposal_id=p['event']['event_proposal_id'],
                 canonical_source=p['event']['canonical_source_entity_id'],S_crit=p['S_crit'],
                 sections=[{k:v for k,v in row.items() if k!='sample_scores'} for row in p['sections']])
            for w in branch['walls'] for p in w['proposals']]


class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.success_ids=[]

    def addSuccess(self,test):
        super().addSuccess(test);self.success_ids.append(test.id())


def definitions(bank):
    return dict(
        comparison_semantics='Complete salience scales, not denominator-only comparison: peak and RMS numerators differ.',
        numerator_audit=[dict(scorer='S1 Min-Max',numerator_statistic='production peak'),
                         dict(scorer='S2 RigidReference',numerator_statistic='event-section filtered-A RMS'),
                         dict(scorer='S3a out-of-band MAD',numerator_statistic='event-section filtered-A RMS'),
                         dict(scorer='S3b in-band MAD',numerator_statistic='event-section filtered-A RMS'),
                         dict(scorer='S4 P95/P99',numerator_statistic='event-section peak magnitude minus video P05')],
        families=5,configurations=6,
        S1=dict(class_name='BaselineMinMaxScorer',numerator='production peak of normalized magnitude in complete event-section phase window',
                denominator='whole-video both-wall global magnitude max-min; production normalization reused',threshold=.6),
        S2=dict(class_name='PerSectionRigidReferenceScorer',numerator='event-section filtered-A RMS',
                denominator='maximum matched-inclusive-duration single-section RMS across Dev zero/rigid bank native windows/sections/walls',
                reference_controls=bank.provenance,reference_bank_hash=bank.bank_hash,threshold=None,hybrid=False),
        S3a=dict(class_name='OutOfBandMADScorer',numerator='event-section filtered-A RMS',
                 denominator='1.4826 * median(abs(out_of_band_residual - median(out_of_band_residual)))',
                 scope='whole-video time, separately per wall and native section, on detector-eligible complete runs',
                 scale_name='out_of_band_residual_scale',noise_interpretation='proxy; band may include non-noise signal components',
                 inspection_bands_cpm=OOB_BANDS,
                 band_selected=False,threshold=None),
        S3b=dict(class_name='InBandMADScorer',numerator='event-section filtered-A RMS',
                 denominator='1.4826 * median(abs(filtered_A - median(filtered_A)))',
                 scope='same whole-video-time per-wall per-section support as S3a; in-band 0.5-5 cpm',
                 role='mechanism control, not prioritized production candidate',threshold=None),
        S4=dict(class_name='PercentileScorer',numerator='event-section peak |filtered A| minus video P05',
                denominator='video P95-P05 or P99-P05; whole-video both-wall shared filtered support',
                variants=[95,99],clipped=False,threshold=None),
        undefined_scale='zero/missing denominator -> unscored None, with numerator/denominator/reason preserved; no epsilon or hybrid floor',
        S_crit='max of adjacent native triple minima equals max of minima over all native blocks of length>=3; null for no fully scored triple',
        S_crit_scope='salience continuous summaries only; any later threshold always reruns full retained sections -> identity -> propagation -> Step1 count',
        threshold_predicate='strict score > tau; incomplete/nonfinite section scores break support')


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    before=protected_snapshot()
    protocol=dict(
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        base_commit='881d87a86924eedcff5852bef9994e00ad9618cd',
        step1_tag='activity-count-v2-step1-freeze',step1_commit='587e32ce885fd46fddd77999c74223cd48b3aa7f',
        pregnancy_outcomes_read=False,clinical_AUC_calculated=False,production_implementation_modified=False,
        Step1_modified=False,Step4a_modified=False,thresholds_selected=False,benchmark_winner_selected=False,
        formal_patient_cohort_run=False,full_Dev_Val_benchmark_generated=False,
        candidate_pool='one extract_shared_upstream call per tiny case, all scorers share the same object and SHA256; production baseline reference runs separately for gate only',
        tiny_cases=list(KINDS),nominal_measurement_span_s=2.,actual_measurement_span_s=2.04,
        noise_floor=FIXED_NOISE_FLOOR,noise_floor_source='unchanged historical Step4a all_texture_threshold',
        noise_floor_tuned=False,minimum_adjacent_sections=3,baseline_threshold=.6,
        scorer_families=5,scorer_configurations=6,alternative_scorers_thresholds=None,
        numerator_confound='Compare complete salience scales, not solely denominator. No numerator sensitivity search this round.',
        matching=dict(time='prediction midpoint inside inclusive [truth_start - actual measurement_span, truth_end + actual measurement_span]',
                      space='intersection count >= MIN_SHARED_NATIVE_SECTIONS (count, no additional adjacency condition)',
                      MIN_SHARED_NATIVE_SECTIONS=MIN_SHARED_NATIVE_SECTIONS,
                      objective='maximum cardinality then minimum total absolute prediction-midpoint vs truth-center deviation',
                      direction_required_for_event_TP=False,propagation_required_for_event_TP=False,
                      prediction_unit='Step1 strict-pair/single activity ledger; v2 identity count bounds reported separately, no forced ambiguity merging',
                      metric_scope='Ledger-level event metrics; do not substitute these point metrics for ambiguous v2 count bounds'),
        out_of_band_inspection_bands_cpm=OOB_BANDS,out_of_band_selected=False,
        finite_interval_response='Approximate H(f) proportional to (1-exp(-i*2*pi*f*T))/T; zeros f=n/T; first nonzero zero at 60/2.04=29.4118 cpm, not all >5 cpm is pure noise',
        scale_partition='Only declared Dev zero/rigid controls enter RigidReference bank; Val rejected by constructor',
        reference_input_root=str(args.controls),step4a_input_root=str(args.step4a),
        protected_source_sha256=before)
    save(args.output/'STEP2_SANDBOX_PROTOCOL.json',protocol)
    cases={};gates={}
    # BLOCKING FIRST PHASE: no alternative scorer constructed before every gate passes.
    try:
        for name in KINDS:
            maps,truth=case(name)
            shared=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
            baseline=BaselineMinMaxScorer(shared)
            branch=continuous_branch(shared,baseline)
            evaluation=evaluate_threshold(shared,branch,.6)
            gates[name]=baseline_equivalence_gate(maps,FIXED_NOISE_FLOOR,shared,branch,evaluation)
            gates[name].update(normalized_value_hash=stable_hash(branch['normalized_values']),
                               proposals_and_sections=branch_scores(branch),
                               retained_selection=evaluation['selection'],
                               nodes_hash=stable_hash(evaluation['nodes']),
                               activity_ledger_hash=stable_hash(evaluation['result']['activities']),
                               v2={k:v for k,v in evaluation['result'].items()
                                   if k.startswith('significant_propagation_activity_count') or k.startswith('significant_propagation_activity_frequency')})
            cases[name]=(shared,truth,branch,evaluation)
    except BaselineEquivalenceError as error:
        save(args.output/'STEP2_BASELINE_EQUIVALENCE.json',dict(passed=False,failed_case=name,layer=str(error),completed=gates))
        raise
    save(args.output/'STEP2_BASELINE_EQUIVALENCE.json',dict(passed=True,cases=gates))

    from . import test_sandbox
    tests=unittest.TextTestRunner(verbosity=1,resultclass=RecordingResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_sandbox))
    test_summary=dict(passed=len(tests.success_ids),failed=len(tests.failures),errors=len(tests.errors),
                      skipped=len(tests.skipped),successful_test_ids=tests.success_ids)
    save(args.output/'STEP2_TRUTH_MATCHER_TESTS.json',dict(
        unit_tests=test_summary,examples=test_sandbox.matcher_examples(),
        minimum_shared_native_sections=MIN_SHARED_NATIVE_SECTIONS,direction_is_not_an_event_matching_gate=True))
    if not tests.wasSuccessful():
        raise RuntimeError('Infrastructure test failure; alternative evaluation stopped')

    references=[];reference_hashes={}
    for i in range(3):
        path=args.controls/f'texture{i}_rigid.npz'
        references.append(dict(control_id=f'Dev_cached_texture{i}_rigid',split='Dev',condition='rigid',maps=cached_maps(path)))
        reference_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    bank=DevRigidBank(references)
    scores={};recalls={};pool_records={}
    for name,(shared,truth,baseline,evaluation) in cases.items():
        recalls[name]=dict(truth=truth,**upstream_recall(shared,truth))
        final_match=match_events(truth,activity_predictions(evaluation['result']),shared['span'],shared['exposure']['seconds'])
        recalls[name]['baseline_event_detection']=final_match
        recalls[name]['baseline_miss_attribution']=miss_attribution(recalls[name],final_match)
        branches=[baseline]
        alternatives=[PerSectionRigidReferenceScorer(bank),OutOfBandMADScorer(shared,(10,25)),
                      InBandMADScorer(shared),PercentileScorer(shared,95),PercentileScorer(shared,99)]
        branches.extend(continuous_branch(shared,scorer) for scorer in alternatives)
        signatures=[]
        scores[name]={}
        for branch in branches:
            signatures.append([(w['wall'],p['event']['event_proposal_id'],p['event']['canonical_source_entity_id'])
                               for w in branch['walls'] for p in w['proposals']])
            if branch['pool_hash']!=shared['pool_hash'] or pool_hash(shared)!=shared['pool_hash']:
                raise RuntimeError('Candidate pool mismatch')
            scores[name][branch['scorer']]=dict(pool_hash=branch['pool_hash'],scores=branch_scores(branch),
                                               threshold=.6 if branch['scorer']=='S1_baseline_minmax' else None)
        if not all(signature==signatures[0] for signature in signatures):
            raise RuntimeError('Proposal IDs differ between branches')
        pool_records[name]=dict(unique_pool_hash=shared['pool_hash'],all_6_configurations_equal=True,
                                proposal_and_canonical_source_ids=signatures[0],
                                ownership_and_source_evidence_hash=stable_hash([w['reconciliation'] for w in shared['walls']]))
    save(args.output/'STEP2_UPSTREAM_RECALL_TESTS.json',dict(cases=recalls,
         upstream_misses_not_assigned_to_salience_scorers=True,
         note='Baseline conditional misses may arise in salience and downstream classification; no alternate-scorer detection metrics or thresholds computed.'))

    # Residual stability only: all 12 existing Step4a controls, no event threshold scans.
    stability=[]
    controls=[(r['control_id'],r['maps']) for r in references]
    generated=sorted((args.step4a/'generated').glob('synthetic_*_maps.npz'))
    if len(generated)!=9:
        raise ValueError('Expected the nine fixed Step4a generated controls')
    for path in generated:
        with np.load(path,allow_pickle=False) as z:
            controls.append((path.stem,dict(z)))
        reference_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,maps in controls:
        shared=dict(A=np.asarray(maps['A'],float),valid=np.asarray(maps['valid_mask'],bool)&np.isfinite(maps['A']),
                    t=np.asarray(maps['interval_center_s']),s=np.asarray(maps['s']),
                    dt=float(np.median(np.diff(maps['interval_center_s']))))
        for band in OOB_BANDS:
            scorer=OutOfBandMADScorer(shared,band)
            scales=np.asarray(scorer.scales,float)
            finite=scales[np.isfinite(scales)]
            stability.append(dict(control=name,band_cpm=band,section_scales=scorer.scales,
                                  scale_median=float(np.median(finite)) if len(finite) else None,
                                  scale_p05=float(np.percentile(finite,5)) if len(finite) else None,
                                  scale_p95=float(np.percentile(finite,95)) if len(finite) else None,
                                  zero_scale_sections=int((finite==0).sum()),
                                  finite_section_count=int(len(finite))))
    save(args.output/'STEP2_SCORER_DEFINITIONS.json',dict(
        definitions=definitions(bank),shared_pool_verification=pool_records,continuous_scores=scores,
        Step4a_out_of_band_residual_stability=stability,reference_input_sha256=reference_hashes,
        thresholds_selected=False,band_selected=False,winner_selected=False,unit_tests=test_summary))
    if protected_snapshot()!=before:
        raise RuntimeError('Protected code changed during experiment')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in reference_hashes.items()):
        raise RuntimeError('Control inputs changed during audit')
    protocol['protected_hashes_unchanged_after_run']=True
    protocol['sandbox_source_sha256']={str(p.relative_to(SANDBOX)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(SANDBOX.glob('*.py'))}
    protocol['all_basic_tests_passed']=True
    save(args.output/'STEP2_SANDBOX_PROTOCOL.json',protocol)
    print('Baseline gates passed; infrastructure tests passed; five artifacts written. No winner or alternative threshold selected.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SANDBOX/'artifacts')
    parser.add_argument('--controls',type=Path,default=ROOT.parent/'uterine-wave-propagation-features/outputs/p3_outer_ideal_prior_20261117_v2')
    parser.add_argument('--step4a',type=Path,default=ROOT/'outputs/step4a_noise_calibration_20261125')
    run(parser.parse_args())
