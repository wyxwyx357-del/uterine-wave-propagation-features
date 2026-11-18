"""Paired review of saved radial candidates and the existing four-condition controls.

Keep all detector parameters and patient outputs unchanged. Clinical reference
rates, if supplied, are descriptive only and never determine a candidate label.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave.candidate_evidence import analyze_candidate_evidence
from uterine_wave.deformation import rate_A, interval_support
from uterine_wave.radial_events import detect_radial_events
from uterine_wave.io import jsonable


def load(path):
    with np.load(path, allow_pickle=False) as data: return dict(data)


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(evidence):
    events = [e for e in evidence['events'] if e['classification'] in ('C2F', 'F2C')]
    ratios = [e['activity']['activity_to_context_rms_ratio'] for e in events]
    ratios = [x for x in ratios if x is not None]
    correlations = [e['continuity']['aligned_correlation']['mean'] for e in events]
    correlations = [x for x in correlations if x is not None]
    return dict(directional_candidates=len(events),
        paired=sum(e['evidence_kind'] == 'paired_phase' for e in events),
        single=sum(e['evidence_kind'] == 'single_phase' for e in events),
        activity_context_ratio_median=float(np.median(ratios)) if ratios else None,
        ratio_at_or_below_one=sum(x <= 1 for x in ratios),
        aligned_waveform_correlation_mean=float(np.mean(correlations)) if correlations else None,
        candidates_with_adjacent_reversal=sum(e['continuity']['reversed_adjacent_steps'] > 0 for e in events),
        candidates_with_phase_direction_conflict=sum(
            any(not x for x in e['continuity']['phase_direction_agreement']) for e in events),
        exact_same_wall_phase_reuses=evidence['exact_same_wall_phase_reuses'],
        confirmed_event_count=None)


def control_arrays(path):
    """Read the frozen A0 cache format used in the prior 12 paired experiments."""
    data = load(path)
    time = data['time_s']
    points = data['A0_positions']
    if points.shape[1] % 7:
        raise ValueError('Expected cached [4N radial, 3N band] points')
    n = points.shape[1]//7
    ids = np.arange(4*n).reshape(2, n, 2)
    hop = int(np.floor(2/np.median(np.diff(time))+.5))
    track = dict(time_s=time, target_valid=data['A0_target_valid'],
                 step_valid=data['A0_step_valid'], segment_id=data['A0_segment_id'])
    valid = interval_support(track, hop)[:, :4*n][:, ids].all(3)
    values = rate_A(points[:, :4*n], dict(A_pair_ids=ids), time, hop)
    # Same original 21 px endpoint margin and full-path normalization as the
    # frozen control evaluation; do not stretch the retained region to 0..1.
    inner = points[0, :4*n].reshape(2, n, 2, 2)[:, :, 0]
    retained = np.linalg.norm(np.diff(inner, axis=1), axis=2).sum(1).min()
    trim = 21/(retained+42)
    return values, valid, (time[hop:]+time[:-hop])/2, np.linspace(trim, 1-trim, n), float(np.median(time[hop:]-time[:-hop]))


def control_report(path, calibration):
    a, valid, t, s, actual_interval = control_arrays(path)
    texture, condition = path.stem.split('_', 1)
    threshold = calibration['leave_one_texture_out_thresholds'][texture[7:]]
    walls, filtered = [], np.full_like(a, np.nan)
    for wall in (0, 1):
        report = detect_radial_events(a[:, wall], valid[:, wall], t, s, noise_floor=threshold)
        filtered[:, wall] = report.pop('filtered_A')
        for event in report['events']: event['event_id'] = f'wall{wall}_'+event['event_id']
        walls.append(dict(wall=wall, **report))
    arrays = dict(filtered_A=filtered, candidate_valid_mask=valid & np.isfinite(filtered),
                  interval_center_s=t, s=s)
    result = analyze_candidate_evidence(dict(walls=walls), arrays)
    return dict(source=path.stem, condition=condition, source_sha256=digest(path),
                nominal_interval_s=2., actual_interval_s=actual_interval,
                noise_floor_rate=threshold, summary=summary(result), evidence=result,
                correct_direction_walls=sum(any(e['classification'] == condition for e in w['events'])
                                            for w in walls) if condition in ('C2F', 'F2C') else None)


def plot_case(path, candidates, arrays, evidence, scale):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, s, a = arrays['interval_center_s'], arrays['s'], arrays['filtered_A']
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for wall in (0, 1):
        ax = axes[0, wall]
        ax.pcolormesh(t, s, np.ma.masked_invalid(a[:, wall].T), cmap='RdBu_r',
                      vmin=-scale, vmax=scale, shading='nearest')
        ax.set_facecolor('.7'); ax.set_ylim(1, 0)
        ax.set_title(('Anterior' if wall == 0 else 'Posterior')+' / saved candidates')
        ax.set_xlabel('Acquisition time (s)'); ax.set_ylabel('Cervix -> fundus s')
        for event in candidates['walls'][wall]['events']:
            if event['classification'] not in ('C2F', 'F2C'): continue
            ax.plot(event['arrival_time_s'], s[event['sections']], color='.15', alpha=.65, lw=.8)
    events = [dict(e, wall=w['wall']) for w in candidates['walls'] for e in w['events']
              if e['classification'] in ('C2F', 'F2C')]
    if events:
        # Same display-only choice in every case: largest spatial support, then
        # earliest saved event ID. No clinical value, amplitude or tuning used.
        event = sorted(events, key=lambda e: (-len(e['sections']), e['event_id']))[0]
        row = next(e for e in evidence['events'] if e['event_id'] == event['event_id'])
        for j, col in enumerate(event['sections']):
            detail = row['activity']['per_section'][j]
            use = (t >= detail['activity_start_s']) & (t <= detail['activity_end_s'])
            color = plt.get_cmap('viridis')(s[col])
            axes[1, 0].plot(t[use], a[use, event['wall'], col], color=color, alpha=.8)
            axes[1, 1].plot(t[use]-event['arrival_time_s'][j], a[use, event['wall'], col], color=color, alpha=.8)
        axes[1, 0].set_title(event['event_id']+' / same signed phases before alignment')
        axes[1, 1].set_title('After arrival alignment / native samples, no interpolation')
    for ax in axes[1]: ax.set_ylabel('Filtered radial A (1/s)'); ax.grid(alpha=.2)
    axes[1, 0].set_xlabel('Acquisition time (s)'); axes[1, 1].set_xlabel('Time relative to arrival (s)')
    fig.suptitle(path.stem+' / review evidence only; common A color scale across all cases')
    fig.savefig(path, dpi=130); plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.review_root/'screening_summary.json').read_text(encoding='utf-8'))
    calibration = json.loads(args.noise_reference.read_text(encoding='utf-8'))
    reference = {}
    if args.recorded_reference:
        for row in json.loads(args.recorded_reference.read_text(encoding='utf-8')):
            matches = row['matches']
            if len(matches) == 1 and row['clinical_record_unit'] == 'per_minute':
                reference[row['review_id']] = dict(positive_cpm=matches[0]['positive_value'],
                    negative_cpm=matches[0]['negative_value'],
                    direction_mapping=row['positive_direction_definition'],
                    clinical_observation_matches_video=None)
    values = []
    for row in cases:
        a = load(args.review_root/row['review_id']/'candidate_maps.npz')['filtered_A']
        values.append(abs(a[np.isfinite(a)]))
    display_scale = float(np.quantile(np.concatenate(values), .97))
    save(args.output/'method.json', dict(source_code_sha256={name: digest(ROOT/name) for name in
        ('uterine_wave/candidate_evidence.py', 'uterine_wave/radial_events.py', 'experiments/audit_wave_evidence.py')},
        noise_reference_sha256=digest(args.noise_reference), no_new_decision_threshold=True,
        context_is_not_noise=True, tracking_changed=False, common_display_scale=display_scale,
        example_choice='largest spatial support then earliest event ID; display only'))
    summaries, details = [], []
    for case in cases:
        review = case['review_id']; source = args.review_root/review
        paths = [source/'candidates.json', source/'candidate_maps.npz', source/'maps.npz']
        hashes = {p.name: digest(p) for p in paths}
        candidates = json.loads(paths[0].read_text(encoding='utf-8'))
        arrays = load(paths[1]); maps = load(paths[2])
        if (not np.array_equal(arrays['s'], maps['s']) or
                not np.array_equal(arrays['interval_center_s'], maps['interval_center_s']) or
                not np.array_equal(arrays['candidate_valid_mask'], maps['valid_mask'] & np.isfinite(arrays['filtered_A']))):
            raise ValueError('Candidate source axes/support differ from the original radial maps')
        evidence = analyze_candidate_evidence(candidates, arrays)
        destination = args.output/review; destination.mkdir(exist_ok=True)
        save(destination/'evidence.json', evidence)
        record = dict(review_id=review, **summary(evidence), source_sha256=hashes,
                      nominal_interval_s=float(maps['nominal_interval_s']),
                      actual_interval_s=float(np.median(maps['interval_dt_s'])),
                      recorded_reference=reference.get(review))
        summaries.append(record)
        for event in evidence['events']:
            details.append(dict(review_id=review, event_id=event['event_id'], wall=event['wall'],
                direction=event['classification'], kind=event['evidence_kind'],
                activity_context_ratio=event['activity']['activity_to_context_rms_ratio'],
                activity_rms=event['activity']['footprint_rms'],
                context_rms=event['activity']['same_run_context_rms'],
                aligned_waveform_correlation=event['continuity']['aligned_correlation']['mean'],
                alignment_gain=event['continuity']['alignment_gain']['mean'],
                adjacent_reversals=event['continuity']['reversed_adjacent_steps'],
                phase_direction_conflicts=sum(not v for v in event['continuity']['phase_direction_agreement'])))
        if args.plots: plot_case(args.output/(review+'.png'), candidates, arrays, evidence, display_scale)
        if hashes != {p.name: digest(p) for p in paths}: raise RuntimeError('Original results changed during audit')
        print(review, json.dumps({k: record[k] for k in ('directional_candidates',
              'activity_context_ratio_median', 'candidates_with_adjacent_reversal',
              'candidates_with_phase_direction_conflict')}), flush=True)
    control_files = sorted(args.controls_root.glob('texture*_*.npz'))
    if not control_files: raise ValueError('No saved control caches found')
    controls = [control_report(p, calibration) for p in control_files]
    known = [e for row in controls if row['condition'] in ('C2F', 'F2C')
             for e in row['evidence']['events'] if e['classification'] in ('C2F', 'F2C')]
    known_ratios = [e['activity']['activity_to_context_rms_ratio'] for e in known
                    if e['activity']['activity_to_context_rms_ratio'] is not None]
    verified = dict(known_direction_walls=sum(row['correct_direction_walls'] for row in controls
                    if row['correct_direction_walls'] is not None),
        expected_direction_walls=2*sum(row['condition'] in ('C2F', 'F2C') for row in controls),
        known_direction_candidates=len(known),
        known_candidates_ratio_at_or_below_one=sum(v <= 1 for v in known_ratios),
        known_candidates_with_adjacent_reversal=sum(e['continuity']['reversed_adjacent_steps'] > 0 for e in known),
        negative_direction_candidates=sum(row['summary']['directional_candidates'] for row in controls
                                         if row['condition'] in ('rigid', 'local_once')),
        no_detector_rule_changed=True)
    save(args.output/'verification.json', verified)
    save(args.output/'summary.json', summaries); save(args.output/'controls.json', controls)
    with (args.output/'candidate_evidence.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]) if details else ['review_id'])
        writer.writeheader(); writer.writerows(details)
    total = lambda name: sum(row[name] for row in summaries)
    lines = ['# 活动、连续传播与同一次活动：配对审查', '',
        '保留原A0、约2秒径向图、滤波、噪声门槛、候选分类及所有原文件。候选不是已确认次数。',
        '活动范围按候选自己的每列正/负阶段恢复；背景对比仅用同一连续段中其余时间，不跨缺失。',
        '其余时间可能有真波，因此比值不是SNR，也不以比值≤1排除候选。',
        '波形按已保存到达时间对齐，只使用原生样本；相位或零交叉作为对齐点的含义分别保留。',
        '相邻顺序逆转、两相方向冲突分别记录，不以单个指标删除测量。',
        '重叠仅表示可能需要复核的关系；跨壁同号ridge编号不代表同一相位。不同到达标记/极性不直接比较。', '',
        f"方向候选共{total('directional_candidates')}个；其中{total('ratio_at_or_below_one')}个的活动/上下文RMS≤1；",
        f"{total('candidates_with_adjacent_reversal')}个含可分辨的相邻到达逆序；{total('candidates_with_phase_direction_conflict')}个有两相方向冲突。",
        f"同壁同一phase被直接复用：{total('exact_same_wall_phase_reuses')}处。此项不能证明生理事件已去重。", '',
        '|病例|方向候选|单相|活动/上下文RMS中位比|相邻逆序|两相方向冲突|',
        '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        ratio = row['activity_context_ratio_median']
        text = f'{ratio:.3f}' if ratio is not None else '无可比上下文'
        lines.append(f"|{row['review_id']}|{row['directional_candidates']}|{row['single']}|{text}|"
                     f"{row['candidates_with_adjacent_reversal']}|{row['candidates_with_phase_direction_conflict']}|")
    lines += ['', '## 既有半合成对照（按源缓存条件）', '',
        '|来源|已知条件|方向候选|相邻逆序|两相方向冲突|', '|---|---|---:|---:|---:|']
    for row in controls:
        q = row['summary']
        lines.append(f"|{row['source']}|{row['condition']}|{q['directional_candidates']}|"
                     f"{q['candidates_with_adjacent_reversal']}|{q['candidates_with_phase_direction_conflict']}|")
    lines += ['', '临床参考仅用于检查级对照，未用于检测、排序或改参数；记录单位为次/分，观察时段和方向映射仍未确认。',
              f"已知传播保持{verified['known_direction_walls']}/{verified['expected_direction_walls']}壁方向正确；其中{verified['known_candidates_ratio_at_or_below_one']}个正确方向候选的活动/上下文RMS≤1，",
              f"{verified['known_candidates_with_adjacent_reversal']}个也有局部逆序。两项均不能单独作为删波规则。",
              '相位经频带滤波且按数据提取的到达标记对齐，相关较高不等于独立证明有生理蠕动。',
              '本轮不把候选重叠直接合并，不计算正式事件频率，也不声称完成真实准确性验证。', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--recorded-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__': main()
