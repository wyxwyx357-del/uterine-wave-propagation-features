"""Local cohort input inventory. Only approved spreadsheet columns are read.

Legacy background caches are adapters for pose data, never feature inputs.
Patient identifiers and paths belong in local manifests, not source control.
"""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np

from .geometry import read_labelme
from .io import jsonable

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
HEADERS = {
    '视频研究编号': 'video_id', '视频完整路径': 'video_path',
    '视频检查时间（首帧）': 'exam_time', '视频时长（秒，CSV估计）': 'listed_duration_s',
    '视频帧数': 'listed_frame_count', '视频帧率': 'listed_fps',
    '临床患者姓名': 'subject_name_for_matching', '患者姓名拼音（大写）': 'pinyin_for_matching',
}
REQUIRED_LABELS = {'ant_wall', 'post_wall', 'cervix_point', 'fundus_point'}
BASE_COMMIT = '4aafb8a9fbb24fb9cf256a25d66eb6f14d911384'


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _sheet_rows(archive, sheet, columns=None):
    with archive.open(sheet) as stream:
        for _, row in ET.iterparse(stream, events=('end',)):
            if row.tag != NS+'row':
                continue
            values = {}
            for cell in row:
                column = re.sub(r'\d', '', cell.get('r', ''))
                if columns is not None and column not in columns:
                    continue
                value = cell.find(NS+'v')
                raw = value.text if value is not None else ''
                if cell.get('t') == 'inlineStr':
                    raw = ''.join(cell.find(NS+'is').itertext())
                values[column] = (cell.get('t'), raw)
            yield int(row.get('r')), values
            row.clear()


def _shared_strings(archive, indices):
    """Resolve only strings referenced by approved cells, not clinical values."""
    selected = {}
    if not indices:
        return selected
    with archive.open('xl/sharedStrings.xml') as stream:
        index = 0
        for _, item in ET.iterparse(stream, events=('end',)):
            if item.tag == NS+'si':
                if index in indices:
                    selected[index] = ''.join(t.text or '' for t in item.iter(NS+'t'))
                index += 1
                item.clear()
    return selected


def read_cohort(path):
    """Read identity, acquisition and video fields; never outcome/activity cells."""
    with ZipFile(path) as z:
        sheet = next(n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n))
        _, header = next(_sheet_rows(z, sheet))
        strings = _shared_strings(z, {int(v) for kind, v in header.values() if kind == 's'})
        decode = lambda kind, value: strings[int(value)] if kind == 's' else value
        columns = {c: HEADERS[decode(*value)] for c, value in header.items()
                   if decode(*value) in HEADERS}
        if 'video_path' not in columns.values() or 'video_id' not in columns.values():
            raise ValueError('Cohort sheet requires video identifier and complete video path columns')
        rows = [(n, r) for n, r in _sheet_rows(z, sheet, columns) if n != 1]
        strings = _shared_strings(z, {int(v) for _, r in rows for kind, v in r.values() if kind == 's'})
        selected = []
        for number, cells in rows:
            row = {columns[c]: decode(*value) for c, value in cells.items()}
            if row.get('video_path'):
                row.update(case_id=Path(row['video_path']).parent.name, spreadsheet_row=number)
                selected.append(row)
    grouped = {}
    for row in selected:
        entry = grouped.setdefault(row['video_path'], dict(row, spreadsheet_rows=[]))
        entry['spreadsheet_rows'].append(row['spreadsheet_row'])
    return dict(source_rows=len(selected), read_columns=list(columns.values()),
                cases=list(grouped.values()), outcomes_read=False)


@contextmanager
def local_video_name(path):
    """OpenCV on Windows can read an ASCII basename in a Unicode directory."""
    path = Path(path).resolve()
    previous = Path.cwd()
    if path.name.isascii():
        try:
            os.chdir(path.parent)
            yield path.name
        finally:
            os.chdir(previous)
    else:
        with tempfile.TemporaryDirectory(prefix='frozen_video_') as folder:
            staged = Path(folder)/('video'+path.suffix)
            shutil.copyfile(path, staged)
            yield str(staged)


def probe_video(path, time_factor):
    import cv2
    with local_video_name(path) as name:
        cap = cv2.VideoCapture(name)
        try:
            if not cap.isOpened():
                raise ValueError('video_open_failed')
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            ok, first = cap.read()
            initial_time = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not ok or n < 2 or not np.isfinite(fps) or fps <= 0:
                raise ValueError('video_metadata_or_first_frame_invalid')
            cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
            ok, last = cap.read()
            end_time = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not ok or last.shape != first.shape:
                raise ValueError('video_last_frame_unreadable_or_size_changed')
            return dict(video_frame_count=n, video_fps=fps,
                video_width=first.shape[1], video_height=first.shape[0],
                video_duration_s=n/fps*time_factor, source_presentation_start_s=initial_time,
                decoded_endpoint_span_s=(end_time-initial_time)*time_factor,
                full_decode_checked=False), first
        finally:
            cap.release()


def _text_key(text):
    return ''.join(c for c in str(text).upper() if c.isalnum())


def annotation_catalog(roots):
    """Only first-frame geometry; ignore clinical/case metadata JSON files."""
    records = []
    for root in roots:
        root = Path(root).resolve()
        if root.suffix.lower() == '.zip':
            with ZipFile(root) as z:
                names = set(z.namelist())
                candidates = [n for n in names if n.endswith('.json') and
                    ('__start__f0000' in n or ('/labels/' not in n and n.rsplit('/', 1)[-1] != 'case_metadata.json'))]
                for name in sorted(candidates):
                    document = json.loads(z.read(name).decode('utf-8-sig'))
                    if not isinstance(document, dict) or not isinstance(document.get('shapes'), list):
                        continue
                    image = str(Path(name).with_suffix('.png')).replace('\\', '/')
                    records.append(dict(path=str(root), entry=name, document=document,
                        reference_entry=image if image in names else None,
                        match_key=_text_key(name)))
        else:
            for path in sorted(root.rglob('*.json')):
                if '__start__f0000' not in path.name and path.stem != path.parent.name:
                    continue
                document = json.loads(path.read_text(encoding='utf-8-sig'))
                if isinstance(document, dict) and isinstance(document.get('shapes'), list):
                    records.append(dict(path=str(path), entry=None, document=document,
                        reference_entry=str(path.with_suffix('.png')) if path.with_suffix('.png').exists() else None,
                        match_key=_text_key(str(path))))
    return records


def match_annotation(row, catalog):
    match = re.fullmatch(r'([^-]+)-(.+)-(\d{8})', row['case_id'])
    if match is None:
        raise ValueError('case_identifier_has_no_exam_date')
    number, name, date = match.groups()
    identities = [_text_key(row['case_id']), _text_key(name+'_'+number+'_'+date)]
    # Some rechecked LabelMe folders use date + Chinese name, without an ID.
    person = row.get('subject_name_for_matching', '')
    aliases = [_text_key(date+'-'+name)]
    if person:
        aliases.append(_text_key(date+'-'+person))
    candidates = [r for r in catalog if any(k in r['match_key'] for k in identities+aliases)]
    if not candidates:
        raise ValueError('annotation_not_found')
    geometry = lambda r: json.dumps(sorted((s['label'], s['points']) for s in r['document']['shapes']
                                          if s['label'] in REQUIRED_LABELS), sort_keys=True)
    if len({geometry(r) for r in candidates}) > 1:
        raise ValueError('annotation_ambiguous_multiple_different_first_frame_geometries')
    return candidates[0], len(candidates)


def annotation_bytes(record):
    if record['entry'] is None:
        return Path(record['path']).read_bytes()
    with ZipFile(record['path']) as z:
        return z.read(record['entry'])


def reference_bytes(record):
    if not record['reference_entry']:
        return None
    if record['entry'] is None:
        return Path(record['reference_entry']).read_bytes()
    with ZipFile(record['path']) as z:
        return z.read(record['reference_entry'])


def bind_annotation(row, catalog, case_dir, first):
    """Bind an existing first-frame annotation by exam identity and dimensions."""
    import cv2
    annotation, aliases = match_annotation(row, catalog)
    document = annotation['document']
    destination = case_dir/'first_frame.json'
    destination.write_bytes(annotation_bytes(annotation))
    read_labelme(destination, coordinate_scale=.5)
    result = dict(annotation_path=str(destination), annotation_source=dict(
        path=annotation['path'], entry=annotation['entry']), annotation_sha256=sha256(destination),
        annotation_identical_alias_count=aliases)
    if first is not None and (document.get('imageWidth'), document.get('imageHeight')) != (first.shape[1], first.shape[0]):
        raise ValueError('annotation_video_dimensions_mismatch')
    raw = reference_bytes(annotation)
    if raw is not None and first is not None:
        reference = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if reference is None or reference.shape != first.shape:
            raise ValueError('annotation_reference_image_dimensions_mismatch')
        (case_dir/'first_frame.png').write_bytes(raw)
        a, b = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(float).ravel() for x in (reference, first)]
        result['first_frame_reference_QC'] = dict(pixels_equal=bool(np.array_equal(reference, first)),
            grayscale_PCC=float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None,
            matching_basis='unique full exam identity or name plus exam date; pixel metrics descriptive only')
    return result


def pose_catalog(roots, index=None):
    if index is not None:
        with Path(index).open(encoding='utf-8-sig', newline='') as f:
            return {r['case_id']: [Path(r['cache_path']).resolve()] for r in csv.DictReader(f)}
    result = {}
    for root in roots:
        for p in Path(root).rglob('*.npz'):
            key = p.stem if re.fullmatch(r'[^-]+-.+-\d{8}', p.stem) else p.parent.name
            result.setdefault(key, []).append(p.resolve())
    return result


def read_pose(path, row, probe, time_factor, allow_legacy_epoch):
    with np.load(path, allow_pickle=False) as z:
        epoch_key = 'pose_epoch' if 'pose_epoch' in z else 'epoch' if allow_legacy_epoch and 'epoch' in z else None
        if epoch_key is None or not {'time_s', 'poses'}.issubset(z.files):
            raise ValueError('pose_missing_time_s_poses_or_pose_epoch')
        data = {k: z[k].copy() for k in ('time_s', 'poses')}
        data['pose_epoch'] = z[epoch_key].copy()
        if 'pose_valid' in z:
            data['pose_valid'] = z['pose_valid'].copy()
        case = str(z['case_id'].reshape(-1)[0]) if 'case_id' in z else path.parent.name if path.stem in ('pose', 'poses') else path.stem
        if case != row['case_id']:
            raise ValueError('pose_case_id_mismatch')
        if 'video_size' in z and int(z['video_size'].reshape(-1)[0]) != Path(row['video_path']).stat().st_size:
            raise ValueError('pose_source_video_size_mismatch')
        if 'video_path' in z and Path(str(z['video_path'].reshape(-1)[0])).parent.name != row['case_id']:
            raise ValueError('pose_source_video_exam_mismatch')
    t, poses, epoch = data['time_s'], data['poses'], data['pose_epoch']
    n = probe['video_frame_count']
    if (t.shape != (n,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or
            not np.allclose(t, np.arange(n)/probe['video_fps']*time_factor, atol=1e-6, rtol=0) or
            not np.isclose(t[-1]-t[0], probe['decoded_endpoint_span_s'], atol=1e-6, rtol=0)):
        raise ValueError('pose_video_timeline_mismatch')
    if (poses.shape != (n, 3, 3) or not np.isfinite(poses).all() or epoch.shape != (n,) or
            not np.allclose(poses[:, 2], [0, 0, 1]) or np.any(abs(np.linalg.det(poses)) < 1e-12)):
        raise ValueError('pose_shape_or_matrix_invalid')
    if 'pose_valid' in data and data['pose_valid'].shape != (n,):
        raise ValueError('pose_valid_shape_mismatch')
    return data, epoch_key


def audit_inputs(cohort, *, data_root, annotation_roots, pose_roots, output,
                 time_factor, pose_index=None, allow_legacy_epoch=False):
    import cv2
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = read_cohort(cohort)
    catalog = annotation_catalog(annotation_roots)
    poses = pose_catalog(pose_roots, pose_index)
    seen = set()
    records = []
    for index, row in enumerate(table['cases'], 1):
        record = dict(row, processing_status='input_invalid', failure_reason=None,
            failures=[], annotation_path=None, pose_path=None, time_factor=time_factor,
            coordinate_scale=.5, tracking_success=None, spacetime_success=None, feature_success=None)
        case_dir = output/'inputs'/row['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            if row['case_id'] in seen:
                raise ValueError('multiple_videos_for_same_exam_identifier')
            seen.add(row['case_id'])
            video = Path(row['video_path']).resolve()
            if not video.is_relative_to(data_root) or not video.is_file():
                raise ValueError('video_missing_or_outside_data_root')
            if row.get('video_id') and str(row['video_id']).split('.')[0] != row['case_id'].split('-')[0]:
                raise ValueError('spreadsheet_video_id_path_mismatch')
            probe, first = probe_video(video, time_factor)
            record.update(probe)
            record['video_size_bytes'] = video.stat().st_size
            record['video_binding'] = 'spreadsheet_video_id_and_full_exam_directory'
        except (ValueError, OSError, cv2.error) as e:
            record['failures'].append(str(e))
            first = None
        try:
            record.update(bind_annotation(row, catalog, case_dir, first))
        except (ValueError, OSError, KeyError, cv2.error) as e:
            record['failures'].append(str(e))
        try:
            candidates = [p for p in poses.get(row['case_id'], []) if p.is_file()]
            if not candidates:
                raise ValueError('pose_not_found')
            if first is None:
                raise ValueError('pose_timeline_not_checked_without_readable_video')
            resolved = [(p, *read_pose(p, row, record, time_factor, allow_legacy_epoch)) for p in candidates]
            if len(resolved) > 1 and any(any(not np.array_equal(d[k], resolved[0][1].get(k)) for k in d)
                                        or set(d) != set(resolved[0][1]) for _, d, _ in resolved[1:]):
                raise ValueError('pose_ambiguous_multiple_background_solutions')
            source, data, epoch_key = resolved[0]
            np.savez_compressed(case_dir/'poses.npz', **data, case_id=np.array(row['case_id']),
                                coordinate_scale=np.array(.5))
            record.update(pose_path=str(case_dir/'poses.npz'), pose_source_path=str(source),
                pose_source_sha256=sha256(source), pose_sha256=sha256(case_dir/'poses.npz'),
                pose_source_epoch_key=epoch_key, pose_valid_present='pose_valid' in data,
                pose_timeline_check='all_cached_times_vs_video_fps_and_decoded_endpoints; full_decode_at_run')
        except (ValueError, OSError, KeyError) as e:
            record['failures'].append(str(e))
        record['failure_reason'] = '; '.join(record['failures']) or None
        if not record['failures']:
            record['processing_status'] = 'ready_for_tracking'
        records.append(record)
        save_json(output/'input_manifest.json', dict(baseline_commit=BASE_COMMIT,
            cohort_read_columns=table['read_columns'], outcomes_read=False, cases=records,
            source_rows=table['source_rows'], annotation_candidates=len(catalog)))
        if index % 20 == 0 or index == len(table['cases']):
            print('AUDIT', index, '/', len(table['cases']), 'ready', sum(
                r['processing_status'] == 'ready_for_tracking' for r in records), flush=True)
    return records
