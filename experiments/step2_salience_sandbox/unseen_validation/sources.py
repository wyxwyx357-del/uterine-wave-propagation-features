"""Phase 1 technical source audit/generation, with no event algorithms or outcomes.

Private acquisition paths stay inside this process. Public artifacts contain only
anonymous IDs and hashes. Old Val is accessed solely as exclusion-hash metadata.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from ..benchmark_data import WORK,ROOT,load,save,sha,verify
from ..upstream import stable_hash

OUTPUT=WORK/'m1_unseen_texture_validation_v2'
FULL=ROOT.parent/'uterine-wave-propagation-features'/'outputs'
GLOBAL_SEED=2026112509
TARGET=8
GEOMETRY_KEYS=('u','A_points','A_pair_ids','B_points','B_curve_ids','arc_positions','B_arc_positions')

def digest(value):return hashlib.sha256(str(value).encode('utf-8')).hexdigest()
def normalized(value):return ''.join(c for c in str(value).upper() if c.isalnum())
def group_token(case):return normalized(str(case).rsplit('-',1)[0].split('-',1)[-1])
def prior_group_hash(case):return digest(str(case).rsplit('-',1)[0].split('-',1)[-1].upper())
def file_identity(path):
    path=Path(path).resolve();info=path.stat()
    return digest((info.st_dev,info.st_ino,info.st_size))

def inventory():
    def projected_csv(path,columns):
        # Select identity/acquisition columns by header, never access clinical
        # cells or construct a full clinical record.
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream);header=next(reader);indices={c:header.index(c) for c in columns}
            return [{c:row[i] for c,i in indices.items()} for row in reader]
    old=load(FULL/'frozen_feature_pipeline_20261119/input_manifest.json')['cases']
    reg=projected_csv(FULL/'expanded_sample_registry_311_20261120/MODEL_INPUT_MASTER_TABLE_311.csv',
        ['exam_id','case_id','case_id_desktop','patient_name_key'])
    queue=projected_csv(FULL/'existing_data_completion_20261124/FINAL_COMPLETION_QUEUE_LOCAL.csv',
        ['exam_id','case_id','video_path','geometry_candidate_paths'])
    selected=load(FULL/'tracking_visual_qc_20261123/metadata/SELECTED_IDENTITY_LOCAL.json')
    p=verify(WORK) # Includes Dev/old-Val exclusion hashes, NEVER their results/images.
    old_cases={r['case_id'] for r in old}|{r['case_id'] for r in selected}
    for column in ('case_id','case_id_desktop'):old_cases|={r[column] for r in reg if r[column]}
    excluded_groups={group_token(x) for x in old_cases if x}
    excluded_groups|={normalized(r['patient_name_key']) for r in reg if r['patient_name_key']}
    for r in old:
        excluded_groups.update(normalized(r[k]) for k in ('subject_name_for_matching','pinyin_for_matching') if r.get(k))
    old_paths={Path(r['video_path']).resolve() for r in old+selected if r.get('video_path')}
    old_file_ids={file_identity(path) for path in old_paths if path.is_file()}
    old_source_keys={digest(x) for x in old_cases}
    old_frames={r['first_frame_sha256'] for r in p['sources']}
    old_private_sources={r['source_npz_sha256'] for r in p['sources']}
    # Previously saved PNG file hashes add coverage without reading ANY old image.
    images=load(FULL/'nnunet_labelme_completion_490_20261124/INPUT_MANIFEST_LOCAL.json')
    by_exam={str(r['exam_id']):r for r in queue}
    old_png_hashes={r['image_sha256'] for r in images if str(r['exam_id']) in by_exam and group_token(by_exam[str(r['exam_id'])]['case_id']) in excluded_groups}
    used_groups=set()
    for registry in WORK.glob('m1_unseen_texture_validation*/USED_VALIDATION_DATA.json'):
        for source in load(registry)['sources']:
            used_groups.add(source['patient_group_sha256']);old_source_keys.add(source['source_key_sha256'])
            old_frames.add(source['first_frame_sha256']);old_file_ids.add(source['original_file_identity_sha256'])
    candidates=queue
    candidates.sort(key=lambda r:digest(f"{GLOBAL_SEED}|source-order|{r['case_id']}"))
    exclusions=dict(patient_groups=excluded_groups,source_keys=old_source_keys,original_file_ids=old_file_ids,
        gray_frame_hashes=old_frames,source_cache_hashes=old_private_sources,png_file_hashes=old_png_hashes,
        dev_patient_hashes={r['patient_group_sha256'] for r in p['sources'] if r['split']=='Dev'},
        val_patient_hashes={r['patient_group_sha256'] for r in p['sources'] if r['split']=='Val'},
        dev_source_keys={r['source_key_sha256'] for r in p['sources'] if r['split']=='Dev'},
        val_source_keys={r['source_key_sha256'] for r in p['sources'] if r['split']=='Val'},used_validation_group_hashes=used_groups)
    metadata=dict(technical_queue_rows=len(candidates),conservatively_excluded_historical_input_cohort_rows=len(old),
        conservatively_excluded_registry_rows=len(reg),prior_selected_technical_cases=len(selected),
        metadata_only_columns=['exam_id','case_id','case_id_desktop','patient_name_key','video_path','geometry_candidate_paths'],
        no_clinical_predictor_or_outcome_columns_read=True,old_Val_results_or_input_arrays_read=False,
        old_Val_exclusion_hash_metadata_read=True,
        historical_exclusion_scope='Entire historical 276 acquisition-input cohort, entire 311 sample registry, all15 tracking-QC selections, all frozen salience Dev and old-Val source hashes; covers the narrower attribution/formation/2x2 panels by containment',
        original_file_comparison='Filesystem original-file identity (device/inode/size) and canonical acquisition/source keys. Old Val video bytes are not opened. New original-video SHA256 also retained.',
        image_comparison='Prior saved gray first-frame hashes and saved PNG file hashes only; no old Val images decoded.')
    return candidates,exclusions,metadata

def generate_controls(source,gray,grid):
    import cv2
    from analysis.mse_pregnancy_association.code.measurement.tracking import track_segments
    from analysis.mse_pregnancy_association.code.measurement.deformation import compute_spacetime
    times=np.arange(751)*.12;initial=np.vstack((grid['A_points'],grid['B_points']))
    folder=OUTPUT/'bases'/source['texture_id'];folder.mkdir(parents=True,exist_ok=True)
    for condition in ('static','rigid','brightness'):
        frames=[];poses=np.repeat(np.eye(3)[None],len(times),axis=0)
        for i,t in enumerate(times):
            phase=source['phase'];dx=3*(np.sin(2*np.pi*t/30+phase)-np.sin(phase)) if condition=='rigid' else 0
            dy=2*(np.cos(2*np.pi*t/30+phase)-np.cos(phase)) if condition=='rigid' else 0
            image=cv2.warpAffine(gray,np.array([[1,0,dx],[0,1,dy]],float),(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101) if condition=='rigid' else gray
            if condition=='brightness':
                gain=1+.10*(np.sin(2*np.pi*t/20+phase)-np.sin(phase));image=np.uint8(np.clip(np.rint(gray.astype(float)*gain),0,255))
            frames.append(image);poses[i,:2,2]=[-dx,-dy]
        tracks=track_segments(frames,initial,times,poses);maps=compute_spacetime(tracks,grid,interval_s=2.)
        np.savez_compressed(folder/(condition+'.npz'),**maps)
        save(folder/(condition+'_TECHNICAL.json'),dict(map_sha256=sha(folder/(condition+'.npz')),map_array_hash=stable_hash(maps),
            valid_fraction=float(maps['valid_mask'].mean()),event_algorithms_called=False,technical_tracking_measurement_completed=True))
        print('Technical A0 controls',source['texture_id'],condition,'complete',flush=True)

def prepare():
    import cv2
    from analysis.mse_pregnancy_association.code.measurement.geometry import read_labelme,layout
    from analysis.mse_pregnancy_association.code.measurement.input_inventory import local_video_name
    if (OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json').exists():raise RuntimeError('Final texture manifest already frozen')
    OUTPUT.mkdir(exist_ok=True);cv2.setNumThreads(1)
    assert cv2.cuda.getCudaEnabledDeviceCount()>0,'Same frozen CUDA A0 backend required; no alternative tracker'
    candidates,old,metadata=inventory()
    save(OUTPUT/'SOURCE_SELECTION_PLAN.json',dict(global_seed=GLOBAL_SEED,target=TARGET,
        order='SHA256(global_seed|source-order|case_id); no event result available',
        technical_rules='Video first frame readable, existing first-frame four-label geometry accepted by unchanged geometry/layout, native baseline21px, frozen tracking/measurement runnable; no event/count/amplitude selection',
        exclude_all_historical_276_and_311_groups=True))
    accepted=[];rejected=[];seen_groups=set();seen_frames=set();private=[]
    for row in candidates:
        token=group_token(row['case_id']);source_key=digest(row['case_id']);key=source_key[:16]
        reason=None
        if not token:reason='missing_patient_group_identity'
        elif token in old['patient_groups'] or digest(token) in old['used_validation_group_hashes']:reason='historical_or_used_validation_patient_group'
        elif token in seen_groups:reason='within_new_panel_patient_repeat'
        elif source_key in old['source_keys']:reason='historical_source_key'
        video=Path(row['video_path'])
        if reason is None and not video.is_file():reason='video_not_available'
        if reason is None and file_identity(video) in old['original_file_ids']:reason='historical_original_file_identity'
        paths=[Path(p.strip()) for p in row['geometry_candidate_paths'].split(';') if p.strip()]
        paths=[p for p in paths if p.is_file()];paths.sort(key=lambda p:sha(p))
        if reason is None and not paths:reason='no_available_geometry_file'
        if reason:
            rejected.append(dict(source_key_hash=source_key,reason=reason));continue
        temporary=OUTPUT/'private';temporary.mkdir(exist_ok=True);staged=temporary/('source_'+key+video.suffix)
        shutil.copyfile(video,staged)
        with local_video_name(staged) as video_name:
            cap=cv2.VideoCapture(video_name);ok,frame=cap.read();cap.release()
        if not ok:
            rejected.append(dict(source_key_hash=source_key,reason='first_frame_decode_failed'));continue
        gray=cv2.cvtColor(cv2.resize(frame,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        frame_hash=hashlib.sha256(gray.tobytes()).hexdigest()
        if frame_hash in old['gray_frame_hashes'] or frame_hash in seen_frames:
            rejected.append(dict(source_key_hash=source_key,reason='prior_or_repeat_image_content'));continue
        chosen=None;failures=[]
        for path in paths:
            try:
                shapes=read_labelme(path,coordinate_scale=.5);grid_full=layout(shapes)
            except ValueError as error:
                failures.append(type(error).__name__);continue
            grid={k:grid_full[k] for k in GEOMETRY_KEYS};chosen=path;break
        if chosen is None:
            rejected.append(dict(source_key_hash=source_key,reason='geometry_unusable',technical_failures=failures));continue
        lengths=np.linalg.norm(grid['A_points'][grid['A_pair_ids'][:,:,1]]-grid['A_points'][grid['A_pair_ids'][:,:,0]],axis=-1)
        np.testing.assert_allclose(lengths,21.,atol=5e-5,rtol=0)
        texture='unseen_'+str(len(accepted)+1).zfill(3)
        seed=int(digest(f'{GLOBAL_SEED}|texture|{source_key}')[:8],16);rng=np.random.default_rng(seed)
        source=dict(texture_id=texture,source_id='source_'+key,source_key_sha256=source_key,patient_group_sha256=digest(token),
            prior_group_convention_sha256=prior_group_hash(row['case_id']),original_file_identity_sha256=file_identity(video),
            original_video_sha256=sha(staged),first_frame_sha256=frame_hash,seed=seed,
            phase=float(rng.uniform(.1,.5)),event_jitter=float(rng.uniform(-.2,.2)),artifact_jitter=int(rng.integers(-1,2)),
            shape=list(gray.shape),native_sections=len(grid['u']),annotation_sha256=sha(chosen),
            eligibility_reason='Independent group/source/image; unchanged first-frame geometry and21px layout; frozen tracking/measurement runnable')
        raw_path=temporary/(texture+'.npz');np.savez_compressed(raw_path,gray=gray,**grid)
        source['source_cache_sha256']=sha(raw_path);source['texture_array_pool_hash']=stable_hash(dict(gray=gray,**grid))
        if source['source_cache_sha256'] in old['source_cache_hashes']:
            rejected.append(dict(source_key_hash=source_key,reason='historical_source_cache_hash'));continue
        # No source may enter either model until independence audit exists.
        tentative=accepted+[source]
        audit=make_audit(tentative,old,metadata,rejected,final=False)
        save(OUTPUT/'M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json',audit)
        try:generate_controls(source,gray,grid)
        except (ValueError,RuntimeError,cv2.error) as error:
            rejected.append(dict(source_key_hash=source_key,reason='tracking_or_measurement_failure',exception_type=type(error).__name__))
            continue
        accepted.append(source);seen_groups.add(token);seen_frames.add(frame_hash)
        private.append(dict(texture_id=texture,source_key_sha256=source_key,video_path=str(video),annotation_path=str(chosen)))
        save(OUTPUT/'TECHNICAL_SOURCE_PROGRESS.json',dict(accepted=accepted,rejections=rejected))
        if len(accepted)>=TARGET:break
    audit=make_audit(accepted,old,metadata,rejected,final=True)
    save(OUTPUT/'M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json',audit)
    save(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json',dict(sources=accepted,global_seed=GLOBAL_SEED,selection_complete=True,
        texture_count=len(accepted),formal_validation_eligible=len(accepted)>=TARGET,no_event_results_used_for_selection=True))
    (OUTPUT/'INPUT_MANIFEST.sha256').write_text(sha(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')+'\n',encoding='ascii')
    save(OUTPUT/'private/SOURCE_PATHS_PRIVATE.json',private)
    print('Frozen independent texture count',len(accepted),'formal eligible',len(accepted)>=TARGET,flush=True)

def make_audit(sources,old,metadata,rejected,final):
    values=dict(patient_group_overlap=sum(s['patient_group_sha256'] in {digest(t) for t in old['patient_groups']} for s in sources),
        Dev_patient_overlap=sum(s['prior_group_convention_sha256'] in old['dev_patient_hashes'] for s in sources),
        old_Val_patient_overlap=sum(s['prior_group_convention_sha256'] in old['val_patient_hashes'] for s in sources),
        source_key_overlap=sum(s['source_key_sha256'] in old['source_keys']|old['dev_source_keys']|old['val_source_keys'] for s in sources),
        original_file_identity_overlap=sum(s['original_file_identity_sha256'] in old['original_file_ids'] for s in sources),
        gray_image_content_overlap=sum(s['first_frame_sha256'] in old['gray_frame_hashes'] for s in sources),
        source_cache_file_hash_overlap=sum(s['source_cache_sha256'] in old['source_cache_hashes'] for s in sources))
    values['previous_used_validation_group_overlap']=sum(s['patient_group_sha256'] in old['used_validation_group_hashes'] for s in sources)
    assert not any(values.values()),'Independence overlap: stop before events'
    return dict(final=final,texture_count=len(sources),intersection_counts=values,all_intersections_zero=True,
        source_metadata_audit=metadata,exclusion_hash_counts={k:len(v) for k,v in old.items()},
        exclusions=[dict(source_key_hash=r['source_key_hash'],reason=r['reason']) for r in rejected],
        no_M0_or_M1_event_results_generated=True,old_Val_results_read=False,pregnancy_outcome_read=False,
        within_new_texture_unique_groups=len({s['patient_group_sha256'] for s in sources}),
        within_new_texture_unique_gray_frames=len({s['first_frame_sha256'] for s in sources}))

def complete_saved_hash_audit():
    """Supplement pixel/cache comparisons using saved PNG provenance only."""
    candidates,old,_=inventory();manifest=load(OUTPUT/'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')
    by_source={digest(r['case_id']):r for r in candidates}
    images={str(r['exam_id']):r for r in load(FULL/'nnunet_labelme_completion_490_20261124/INPUT_MANIFEST_LOCAL.json')}
    available=[]
    for source in manifest['sources']:
        row=by_source[source['source_key_sha256']];image=images.get(str(row['exam_id']))
        if image is not None:available.append(image['image_sha256'])
    overlap=sum(h in old['png_file_hashes'] for h in available)
    assert overlap==0,'Saved image-content hash overlaps old panel; STOP before event generation'
    audit=load(OUTPUT/'M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json')
    audit['intersection_counts']['saved_PNG_file_hash_overlap']=overlap
    audit['additional_image_hash_coverage']=dict(new_sources_with_saved_PNG_hash=len(available),total_new_sources=len(manifest['sources']),
        old_saved_PNG_hashes=len(old['png_file_hashes']),old_gray_hashes=len(old['gray_frame_hashes']),
        stable_source_array_pool_hashes_saved_for_all_new_sources=True,
        note='Compare all available old saved hashes, original-file identities and group/source keys; do not decode old Val images to create new hashes.')
    audit['intersection_counts']['image_content_or_source_pool_hash_overlap']=sum(audit['intersection_counts'][k] for k in (
        'gray_image_content_overlap','source_cache_file_hash_overlap','saved_PNG_file_hash_overlap'))
    save(OUTPUT/'M1_UNSEEN_TEXTURE_INDEPENDENCE_AUDIT.json',audit)

if __name__=='__main__':prepare()
