"""Audit saved native-step PCC for all existing cohort review cases.

No video decoding, retracking, clinical-table input, threshold tuning or event
reclassification. Keep patient outputs local under an explicit output folder.
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
from uterine_wave.quality import (tracking_quality, radial_quality_maps, candidate_quality,
                                  LITERATURE_PCC_REFERENCE)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(args):
    cases = json.loads((args.review_root/'screening_summary.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/'method.json', dict(
        code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                     ('uterine_wave/quality.py', 'uterine_wave/_lk.py', 'experiments/audit_tracking_quality.py')},
        reference_pcc=LITERATURE_PCC_REFERENCE, reference_is_decision_threshold=False,
        tracking_changed=False, clinical_or_outcome_inputs_used=False,
        native_sampling=True, interval_alignment='same source start/end as radial A'))
    summaries = []
    for case in cases:
        review = case['review_id']
        source = args.review_root/review
        names = ('tracks.npz', 'grid.npz', 'maps.npz', 'candidates.json')
        hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
        tracks, grid, maps = [load(source/name) for name in names[:3]]
        if 'target_ncc' not in tracks:
            summaries.append(dict(review_id=review, status='saved_native_pcc_unavailable', source_sha256=hashes))
            continue
        quality = tracking_quality(tracks, grid)
        qmaps = radial_quality_maps(tracks, grid, maps)
        candidates = json.loads((source/'candidates.json').read_text(encoding='utf-8'))
        events = candidate_quality(qmaps, candidates)
        destination = args.output/review
        destination.mkdir(exist_ok=True)
        save(destination/'tracking_quality.json', quality)
        save(destination/'candidate_quality.json', events)
        np.savez_compressed(destination/'quality_maps.npz', **qmaps)
        directional = [row for row in events if row['classification'] in ('C2F', 'F2C')]
        means = [row['interval_pcc']['mean'] for row in directional if row['interval_pcc']['mean'] is not None]
        summary = dict(review_id=review, status='audited', source_sha256=hashes,
                       radial_points=quality['radial_points'], walls=quality['walls'],
                       directional_candidate_count=len(directional),
                       directional_candidates_with_quality=len(means),
                       minimum_candidate_region_pcc_mean=min(means) if means else None,
                       candidates_below_reference=sum(value < LITERATURE_PCC_REFERENCE for value in means),
                       official_wave_count=None, wave_accuracy=None)
        summaries.append(summary)
        # Verify that the audit did not replace any original result.
        if hashes != {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}:
            raise RuntimeError('Source results changed during audit')
        mean = quality['radial_points']['mean']
        print(review, 'PCC', round(mean, 6) if mean is not None else None,
              'coverage', round(quality['radial_points']['pcc_coverage_fraction'], 6), flush=True)
    save(args.output/'summary.json', summaries)
    with (args.output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['review_id', 'radial_PCC_conditional_mean', 'PCC_coverage',
                         'anterior_inner_mean', 'anterior_outer_mean', 'posterior_inner_mean',
                         'posterior_outer_mean', 'direction_candidates_not_waves',
                         'minimum_candidate_region_PCC_mean', 'candidates_below_0p8_reference'])
        for row in summaries:
            if row['status'] != 'audited': continue
            writer.writerow([row['review_id'], row['radial_points']['mean'],
                row['radial_points']['pcc_coverage_fraction'],
                *[wall[part]['mean'] for wall in row['walls'] for part in ('inner', 'outer')],
                row['directional_candidate_count'], row['minimum_candidate_region_pcc_mean'],
                row['candidates_below_reference']])
    lines = ['# 已保存轨迹的PCC质量审查', '',
             'NCC为去均值后的局部Pearson相关；这里只汇总已支持的原生相邻帧步骤。',
             '排除首帧种子、重定位跳跃、段/位姿阶段边界；同时保留覆盖率和拒绝原因。',
             '分数经过现有LK/FB/NCC筛选，不能声称独立复现Huang整段PCC纳入标准。',
             '0.8仅作文献参照，不排除患者、不删除候选、不确认波。',
             '质量图覆盖径向形变同一原生时间区间的inner/outer；不把2秒间隔当成逐帧间隔。',
             '候选质量为其时空包围区域内有效形变区间的描述；重叠区间不是独立样本。',
             '原追踪、A/epsilon、滤波、ER和候选文件哈希前后不变；本次不计算准确率。', '',
             '|病例|径向PCC均值|PCC覆盖|候选区域最低均值|',
             '|---|---:|---:|---:|']
    for row in summaries:
        if row['status'] != 'audited':
            lines.append(f"|{row['review_id']}|未保存PCC|—|—|")
            continue
        local = row['minimum_candidate_region_pcc_mean']
        local_text = f'{local:.5f}' if local is not None else '—'
        mean = row['radial_points']['mean']
        mean_text = f'{mean:.5f}' if mean is not None else '无有效PCC'
        lines.append(f"|{row['review_id']}|{mean_text}|"
                     f"{row['radial_points']['pcc_coverage_fraction']:.2%}|{local_text}|")
    lines += ['', '高PCC表示相邻块外观相似，不证明局部形变就是生理蠕动，也不证明已识别独立事件。',
              '若争议候选的PCC同样很高，则继续提高PCC门槛不能针对性解决事件误报。', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__': main()
