"""Fixed double-activity experiment with filter-independent observable truth.

Two latent Gaussian length changes need not make two observable contractions.
Keep latent centers and raw-composite maxima separately. Truth is never used
by detection or ownership. No new detector threshold is introduced.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave import radial_events
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable
from experiments.audit_event_filtering import reflected_fft
from experiments.audit_wave_evidence import control_arrays
from experiments.partial_phase_events import detect_partial, observed_lobes

FILTERS = dict(periodic_fft=band_limited_signal, reflected_fft=reflected_fft)
GAPS = (5.,10.,15.,20.)  # 20 s reproduces the previously observed weak-tail failure.


def double_fixture(t, s, interval, direction, gap, second_amplitude):
    """Apply the finite-interval strain formula to the SUM of length changes.

    Do not add independently normalized rates. Width=4s and speed=.04 paths/s
    are frozen from the previous single-activity experiment.
    """
    before = np.ones((len(t), len(s))); after = before.copy()
    arrivals = []
    for center, amplitude in [(30-gap/2, .04), (30+gap/2, second_amplitude)]:
        arrival = center+(1 if direction == 'C2F' else -1)*(s-.5)/.04
        offset = t[:, None]-arrival
        before += amplitude*np.exp(-.5*((offset-interval/2)/4)**2)
        after += amplitude*np.exp(-.5*((offset+interval/2)/4)**2)
        arrivals.append(arrival)
    return (after-before)/before/interval, np.asarray(arrivals)


def raw_truth(y, t):
    """Observable positive-to-negative rate transitions and signed raw peaks.

    No bandpass, amplitude floor or tested-filter reference is used. Peaks are
    taken from the positive/negative lobes adjacent to each maximum of length.
    """
    columns = []
    for col in range(y.shape[1]):
        z = y[:, col]
        roots = np.flatnonzero((z[:-1] > 0) & (z[1:] <= 0))
        edges = np.r_[0, np.flatnonzero(np.diff(z > 0))+1, len(t)]
        events = []
        for k in roots:
            positive = next((a, b) for a, b in zip(edges[:-1], edges[1:]) if a <= k < b)
            negative = next((a, b) for a, b in zip(edges[:-1], edges[1:]) if a <= k+1 < b)
            peak_plus = positive[0]+int(np.argmax(z[slice(*positive)]))
            peak_minus = negative[0]+int(np.argmin(z[slice(*negative)]))
            arrival = t[k]-z[k]/(z[k+1]-z[k])*(t[k+1]-t[k])
            events.append(dict(arrival_s=float(arrival),
                positive_peak_s=float(t[peak_plus]), negative_peak_s=float(t[peak_minus]),
                positive_peak=float(z[peak_plus]), negative_peak=float(abs(z[peak_minus])),
                positive_phase_indices=[int(positive[0]),int(positive[1])],
                negative_phase_indices=[int(negative[0]),int(negative[1])],
                positive_phase_time_s=[float(t[positive[0]]),float(t[positive[1]-1])],
                negative_phase_time_s=[float(t[negative[0]]),float(t[negative[1]-1])]))
        columns.append(events)
    counts = [len(c) for c in columns]
    # The fixed same-direction fixture has the same intrinsic shape everywhere.
    # A mismatch is an input/visibility finding, not permission to fill a gap.
    return dict(columns=columns, observable_counts_per_section=counts,
                uniform_observable_count=counts[0] if len(set(counts)) == 1 else None)


def phase_identity(col, sign, time, truth):
    refs = truth['columns'][col]
    if not refs:
        return None
    key = 'positive_phase_time_s' if sign > 0 else 'negative_phase_time_s'
    choices=[i for i,event in enumerate(refs) if event[key][0]<=time<=event[key][1]]
    return choices[0] if len(choices)==1 else None


def filter_phase_evidence(filtered, truth, t, floor):
    """Measure clean-filter distortion within each independent RAW phase.

    The raw peak and phase bounds stay fixed. A boundary maximum or wrong-sign
    value is not relabeled a preserved internal peak. The existing threshold
    is recorded as an explanatory scale, never retuned using these fixtures.
    """
    rows = []
    for col, events in enumerate(truth['columns']):
        for identity, event in enumerate(events):
            for prefix, sign in [('positive',1),('negative',-1)]:
                a,b = event[prefix+'_phase_indices']
                projected = sign*filtered[a:b,col]
                peak = a+int(np.argmax(projected))
                amplitude = float(sign*filtered[peak,col])
                raw_peak = event[prefix+'_peak']
                rows.append(dict(section=col, truth_event=identity, polarity=sign,
                    raw_peak_s=event[prefix+'_peak_s'], raw_peak_rate=raw_peak,
                    best_filtered_signed_peak_s=float(t[peak]),
                    best_filtered_signed_rate=amplitude,
                    peak_error_s=float(t[peak]-event[prefix+'_peak_s']),
                    raw_phase_internal_filtered_peak=bool(a<peak<b-1 and amplitude>0),
                    peak_retention_ratio=amplitude/raw_peak,
                    raw_peak_below_original_floor=bool(raw_peak<=floor),
                    filtered_peak_below_original_floor=bool(amplitude<=floor),
                    distinguishing_inner_phase=len(events)==2 and
                        ((identity==0 and sign<0) or (identity==1 and sign>0))))
    return rows


def layer_trace(signal, valid, t, s, floor, truth, function):
    """Observe the unchanged lobe/ridge stages; do not feed truth into them."""
    with patch.object(radial_events,'band_limited_signal',function):
        _,columns,_=observed_lobes(signal,valid,t,np.full(len(s),floor))
    ridges=radial_events._ridges(columns,t,s)
    nodes=[dict(section=col,sign=n['sign'],peak_s=n['peak_time_s'],
                truth_identity=phase_identity(col,n['sign'],n['peak_time_s'],truth))
           for col,column in enumerate(columns) for n in column]
    support=[]
    for identity in range(truth['uniform_observable_count'] or 0):
        phases={}
        for sign in (1,-1):
            before={n['section'] for n in nodes if n['sign']==sign and n['truth_identity']==identity}
            paths=[sorted(n['column'] for n in r['nodes']
                          if phase_identity(n['column'],sign,n['peak_time_s'],truth)==identity)
                   for r in ridges if r['sign']==sign]
            phases[str(sign)]=dict(before_ridges_sections=sorted(before),
                longest_single_ridge_sections=max(paths,key=len,default=[]),
                retained_ridge_sections=sorted({c for path in paths for c in path}))
        support.append(dict(truth_event=identity,phases=phases))
    return dict(observed_nodes=nodes,support=support,
        ridges=[dict(sign=r['sign'],sections=[n['column'] for n in r['nodes']],
                     peak_s=[n['peak_time_s'] for n in r['nodes']]) for r in ridges])


def describe_piece(piece, ridges, truth):
    """Attribute measured phase nodes to RAW observable events for evaluation.

    Raw same-sign phase intervals define identity, without a tuned time gate.
    Out-of-phase nodes remain unassigned. Large timing errors remain visible
    and are never called successful recovery.
    """
    phases = [ridges[i] for i in piece['phase_ridge_ids']]
    paired = len(phases) == 2
    by_phase = [dict(zip(p['sections'], p['peak_time_s'])) for p in phases]
    assignments, cross = {}, []
    for col, time in zip(piece['sections'], piece['arrival_time_s']):
        labels = [phase_identity(col, p['sign'], peaks[col], truth)
                  for p, peaks in zip(phases, by_phase)]
        if None not in labels and len(set(labels)) > 1:
            cross.append(col)
        # In this known length-bump fixture, negative->positive is the trough
        # between activities, not another maximum of length. It is retained as
        # candidate evidence, never prohibited as a real physiological shape.
        correct_role = not paired or (phases[0]['sign'] > 0 and phases[1]['sign'] < 0)
        if labels and labels[0] is not None and len(set(labels)) == 1 and correct_role:
            identity = labels[0]
            marker = 'arrival_s' if paired else ('positive_peak_s' if phases[0]['sign'] > 0 else 'negative_peak_s')
            expected = truth['columns'][col][identity][marker]
            assignments.setdefault(identity, []).append(dict(section=col, measured_s=float(time),
                                                             truth_s=expected, error_s=float(time-expected)))
    labels = {phase_identity(c, p['sign'], tm, truth) for p, peaks in zip(phases, by_phase)
              for c, tm in peaks.items() if c in piece['sections']}
    return dict(fragment_id=piece['event_id'], assignments=assignments,
                cross_event_pair_sections=cross, phase_truth_labels=sorted(x for x in labels if x is not None),
                between_activity_pair=paired and phases[0]['sign'] < 0,
                classification=piece['classification'])


def one_to_one(groups, number, n_sections):
    """Maximize native-section support, then minimize timing error, no cutoffs.

    Assign each group at most once. Missing identities remain unmatched. Output
    matched support and timing, not a thresholded recall/accuracy statistic.
    """
    options = []
    for identity in range(number):
        choices = [None]
        for index, group in enumerate(groups):
            points = group['assignments'].get(identity, [])
            if points:
                choices.append((index, points))
        options.append(choices)
    best, best_key = None, None
    for combination in product(*options):
        selected = [x[0] for x in combination if x is not None]
        if len(selected) != len(set(selected)):
            continue
        support = sum(len({p['section'] for p in x[1]}) for x in combination if x is not None)
        error = sum(float(np.mean([p['error_s']**2 for p in x[1]])) for x in combination if x is not None)
        key = (-support, error)
        if best_key is None or key < best_key:
            best, best_key = combination, key
    output = []
    for identity, choice in enumerate(best or []):
        if choice is None:
            output.append(dict(truth_event=identity, matched=False, group_id=None,
                               native_section_fraction=0., arrival_RMSE_s=None))
            continue
        index, points = choice
        output.append(dict(truth_event=identity, matched=True, group_id=groups[index]['group_id'],
            native_section_fraction=len({p['section'] for p in points})/n_sections,
            arrival_RMSE_s=float(np.sqrt(np.mean([p['error_s']**2 for p in points]))),
            direction=groups[index]['classification'], measurements=points))
    return output


def evaluate(report, truth):
    ridges = {p['ridge_id']:p for p in report['phase_ridges']}
    events = {e['event_id']:e for e in report['events']}
    # Evaluate the measured attachment, if ownership proposed a complementary
    # pair, rather than pretending it kept the original single-phase arrival.
    fragments = [describe_piece(dict(events[f['fragment_id']],
        sections=f['sections'], arrival_time_s=f['arrival_time_s'],
        phase_ridge_ids=f['phase_ridge_ids'], classification=f['direction']), ridges, truth)
        for f in report['fragments']]
    by_id = {p['fragment_id']:p for p in fragments}
    groups = []
    for group in report['proposed_events']:
        parts = [by_id[k] for k in group['fragment_ids']]
        combined = {}
        for part in parts:
            for identity, points in part['assignments'].items():
                combined.setdefault(identity, []).extend(points)
        groups.append(dict(group_id=group['proposal_id'], assignments=combined,
            classification=events[group['fragment_ids'][0]]['classification'],
            fragment_ids=group['fragment_ids'], multiple_truth_labels=len(combined) > 1,
            ownership_joined_distinct_pure_fragments=len(parts) > 1 and
                all(len(p['assignments']) == 1 for p in parts) and len(combined) > 1))
    switches = 0
    for ridge in ridges.values():
        identities = {phase_identity(c, ridge['sign'], tm, truth)
                      for c, tm in zip(ridge['sections'], ridge['peak_time_s'])}
        switches += len(identities-{None}) > 1
    number = truth['uniform_observable_count']
    matches = one_to_one(groups, number, len(truth['columns'])) if number is not None else None
    return dict(fragments=fragments, groups=groups, matches=matches,
        candidate_count=report['candidate_count'], direction_structures=report['traveling_candidate_count'],
        cross_event_pair_candidates=sum(bool(p['cross_event_pair_sections']) for p in fragments),
        mixed_identity_ridges=switches,
        between_activity_pair_candidates=sum(p['between_activity_pair'] for p in fragments),
        ownership_cross_event_joins=sum(g['ownership_joined_distinct_pure_fragments'] for g in groups),
        multiple_group_support=[sum(bool(g['assignments'].get(i)) for g in groups) for i in range(number or 0)],
        formal_event_count=None)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = read(args.noise_reference)
    frozen = read(args.source_protocol)['source_files']
    sources, rows = {}, []
    for path in sorted(args.controls_root.glob('texture*_rigid.npz')):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if frozen.get(str(path)) != sha:
            raise ValueError('Frozen residual cache changed or is not in protocol')
        sources[str(path)] = sha
        noise, valid, t, s, interval = control_arrays(path)
        if not valid.all():
            raise ValueError('These finite raw-truth controls require the original complete rigid residuals')
        texture = path.stem.split('_')[0]
        floor = calibration['leave_one_texture_out_thresholds'][texture[7:]]
        for direction, second, gap in product(('C2F','F2C'), (.04,.01), GAPS):
            clean, latent = double_fixture(t, s, interval, direction, gap, second)
            truth = raw_truth(clean, t)
            row = dict(source=f'{texture}_{direction}_{second:.2f}_gap{gap:g}', texture=texture,
                direction=direction, second_amplitude=second, gap_s=gap, latent_event_count=2,
                latent_arrival_s=latent, raw_truth=truth, methods={})
            row['clean_only_causal_check'] = {}
            # Remove residual noise as a causal control, not a proposed patient
            # method. If RAW clean works but bandpassed clean fails, modifying
            # phase ownership cannot restore the removed measurement evidence.
            for name,function in [*FILTERS.items(),('raw_no_filter',lambda y,t:y.copy())]:
                with patch.object(radial_events,'band_limited_signal',function):
                    report=detect_partial(clean,valid[:,0],t,s,noise_floor=floor)
                report.pop('filtered_A')
                row['clean_only_causal_check'][name]=evaluate(report,truth)
            for name, function in FILTERS.items():
                walls = []
                clean_filtered = function(clean, t)
                # Keep filtered truth-like structures as observations only;
                # they NEVER replace the independent RAW reference above.
                filter_observations = raw_truth(clean_filtered, t)
                with patch.object(radial_events, 'band_limited_signal', function):
                    for wall in (0,1):
                        report = detect_partial(noise[:,wall]+clean, valid[:,wall], t, s, noise_floor=floor)
                        filtered = report.pop('filtered_A')
                        evaluation = evaluate(report, truth)
                        walls.append(dict(wall=wall, report=report, evaluation=evaluation))
                        np.savez_compressed(args.output/(row['source']+f'_{name}_wall{wall}_signals.npz'),
                            raw_clean_A=clean, filtered_clean_A=clean_filtered, filtered_noisy_A=filtered,
                            time_s=t, s=s, latent_arrival_s=latent)
                row['methods'][name] = dict(walls=walls,
                    filtered_clean_maxima_per_section=filter_observations['observable_counts_per_section'],
                    clean_filter_phase_evidence=filter_phase_evidence(clean_filtered,truth,t,floor),
                    clean_layer_trace=layer_trace(clean,valid[:,0],t,s,floor,truth,function),
                    noisy_layer_traces=[layer_trace(noise[:,w]+clean,valid[:,w],t,s,floor,truth,function)
                                        for w in (0,1)])
            save(args.output/(row['source']+'.json'), row)
            rows.append(row)
            print(row['source'], truth['uniform_observable_count'], flush=True)
    if len(sources) != 3 or len(rows) != 48:
        raise ValueError('Expected three textures and 48 fixed combinations')
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path,sha in sources.items()):
        raise RuntimeError('Source data changed')
    def summarize(selected, method):
        evaluations = [w['evaluation'] for r in selected for w in r['methods'][method]['walls']]
        matches = [m for e in evaluations for m in e['matches'] or []]
        actual = [m for m in matches if m['matched']]
        inner = [p for r in selected for p in r['methods'][method]['clean_filter_phase_evidence']
                 if p['distinguishing_inner_phase']]
        return dict(wall_conditions=len(evaluations), observable_truth_events=len(matches),
            assigned_observable_events=len(actual),
            direction_agreement=sum(m['direction'] == r['direction'] for r in selected
                for w in r['methods'][method]['walls'] for m in w['evaluation']['matches'] or [] if m['matched']),
            minimum_native_section_fraction=min((m['native_section_fraction'] for m in matches), default=None),
            median_arrival_RMSE_s=float(np.median([m['arrival_RMSE_s'] for m in actual])) if actual else None,
            maximum_arrival_RMSE_s=max((m['arrival_RMSE_s'] for m in actual), default=None),
            cross_event_pair_candidates=sum(e['cross_event_pair_candidates'] for e in evaluations),
            mixed_identity_ridges=sum(e['mixed_identity_ridges'] for e in evaluations),
            ownership_cross_event_joins=sum(e['ownership_cross_event_joins'] for e in evaluations),
            possible_split_truth_events=sum(n > 1 for e in evaluations for n in e['multiple_group_support']),
            distinguishing_phase_peak_retention_median=float(np.median([p['peak_retention_ratio'] for p in inner])) if inner else None,
            raw_distinguishing_phase_below_floor_fraction=float(np.mean([p['raw_peak_below_original_floor'] for p in inner])) if inner else None,
            filtered_distinguishing_phase_below_floor_fraction=float(np.mean([p['filtered_peak_below_original_floor'] for p in inner])) if inner else None,
            direction_structures=sum(e['direction_structures'] for e in evaluations))
    summary = dict(by_condition=[dict(gap_s=gap, second_amplitude=amp,
        raw_observable_counts=sorted({r['raw_truth']['uniform_observable_count'] for r in rows
                                     if r['gap_s']==gap and r['second_amplitude']==amp}),
        methods={method:summarize([r for r in rows if r['gap_s']==gap and r['second_amplitude']==amp], method)
                 for method in FILTERS}) for gap,amp in product(GAPS,(.04,.01))],
        raw_resolvable_two={method:summarize([r for r in rows if r['raw_truth']['uniform_observable_count']==2],method)
                            for method in FILTERS},
        clean_only_causal_check=[dict(gap_s=gap,second_amplitude=amp,
            methods={method:dict(
                observable_truth_events=sum(len(r['clean_only_causal_check'][method]['matches'] or []) for r in rows
                    if r['gap_s']==gap and r['second_amplitude']==amp),
                native_full_support_assignments=sum(m['native_section_fraction']==1. for r in rows
                    if r['gap_s']==gap and r['second_amplitude']==amp
                    for m in r['clean_only_causal_check'][method]['matches'] or []),
                maximum_arrival_RMSE_s=max((m['arrival_RMSE_s'] for r in rows
                    if r['gap_s']==gap and r['second_amplitude']==amp
                    for m in r['clean_only_causal_check'][method]['matches'] or [] if m['matched']),default=None))
                for method in (*FILTERS,'raw_no_filter')}) for gap,amp in product(GAPS,(.04,.01))],
        formal_event_count=None, patient_accuracy=None, source_files_unchanged=True)
    save(args.output/'summary.json', summary)
    save(args.output/'protocol.json', dict(source_files=sources,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.noise_reference,args.source_protocol)},
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'uterine_wave/radial_events.py','uterine_wave/amplitude.py',
            'experiments/partial_phase_events.py','experiments/audit_event_filtering.py')},
        source_protocol=str(args.source_protocol), interval_s=interval,
        truth='Unfiltered finite-interval rate of summed Gaussian lengths; latent centers kept separately',
        widths_s=4., speed_paths_per_s=.04, gaps_s=list(GAPS), first_amplitude=.04,
        second_amplitudes=[.04,.01], thresholds='Original leave-one-texture-out absolute noise thresholds',
        matching='One-to-one groups; maximize raw-identity section support, then minimize RAW-reference timing error',
        phase_labels='Membership in raw-composite same-sign phase intervals, no tuned time gate; outside remains unassigned',
        adoption='Periodic FFT + partial-phase main experiment; mirrored FFT not default'))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-root',type=Path,required=True)
    parser.add_argument('--noise-reference',type=Path,required=True)
    parser.add_argument('--source-protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
