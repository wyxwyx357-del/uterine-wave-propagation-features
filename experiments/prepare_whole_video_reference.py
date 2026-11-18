"""Prepare a small complete-video manual reference task from existing videos.

No tracking, prediction, outcome access or automatic accuracy calculation.
The first three cases are purposive DEVELOPMENT checks, not an independent
clinical validation sample. All original frames/files stay unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=('case_1040','case_955','case_1209')
COMMIT='884c94114255f996824e10be9e9ea1729924e8e2'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def build(args):
    args.output.mkdir(parents=True,exist_ok=True)
    key=read(args.video_review/'review_key_private.json')
    sources={r['review_id']:r for r in key['sources']}
    text=(args.video_review/'review_data.js').read_text(encoding='utf-8')
    old=json.loads(text.split('=',1)[1].strip().rstrip(';'))
    available=[r['case'] for r in read(args.ownership/'summary.json')['cases']]
    if len(available)!=12 or not set(PRIMARY)<=set(available):
        raise ValueError('Expected the same 12 ownership-review cases')
    ordered=list(PRIMARY)+sorted(set(available)-set(PRIMARY),key=lambda x:int(x.split('_')[-1]))
    private=[];items=[];machine={};hashes={}
    def relative(path):return os.path.relpath(path,args.output).replace(os.sep,'/')
    for number,case in enumerate(ordered,1):
        source=sources[case];alias=source['case_alias'];meta=old['cases'][alias]
        video=args.video_review/meta['video']
        hashes[str(video)]=digest(video)
        if hashes[str(video)]!=source['source_video_sha256']:
            raise ValueError('Original video differs from its frozen review copy')
        token=f'V{number:02d}'
        item=dict(token=token,primary=number<=len(PRIMARY),duration_s=meta['duration_s'],fps=meta['fps'],
            video=relative(video),poster=relative(args.video_review/meta['poster']),
            data=relative(args.video_review/meta['data']),data_alias=alias,
            comparison_map=relative(args.ownership/(case+'.png')))
        items.append(item)
        result_path=args.ownership/(case+'.json');hashes[str(result_path)]=digest(result_path)
        result=read(result_path);proposals=[]
        for wall in result['walls']:
            for event in wall['reconciliation']['event_proposals']:
                arrivals=event['arrival_time_s'];measurement=event['canonical_measurement']
                proposals.append(dict(event_id=event['event_proposal_id'],wall=wall['wall'],
                    sections=event['supporting_sections'],arrival_time_s=arrivals,direction=event['direction'],
                    start_s=measurement['start_s'],end_s=measurement['end_s'],
                    new_child=event['canonical_source_entity_id'].startswith('child:'),
                    ownership_status=event['ownership_status'],source_parents=event['source_parent_ids']))
        machine[token]=proposals
        private.append(dict(token=token,review_id=case,case_alias=alias,
            purpose=('changed_split' if number==1 else 'previous_visible_propagation' if number==2
                     else 'previous_motion_interference' if number==3 else 'optional_existing_case'),
            source_video_sha256=hashes[str(video)]))
    bundle_id=hashlib.sha256(json.dumps(private,sort_keys=True).encode()).hexdigest()[:16]
    bundle=dict(bundle_id=bundle_id,algorithm_commit=COMMIT,cases=items,
                primary_total_duration_s=sum(x['duration_s'] for x in items if x['primary']))
    (args.output/'review_data.js').write_text('window.wholeVideoReference='+json.dumps(bundle,ensure_ascii=False)+';\n'+
        'window.wholeVideoMachine='+json.dumps(machine,ensure_ascii=False)+';\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('whole_video_reference.html'),args.output/'index.html')
    save(args.output/'review_key_private.json',dict(bundle_id=bundle_id,sources=private))
    save(args.output/'protocol.json',dict(bundle_id=bundle_id,algorithm_commit=COMMIT,
        selection='First changed case + prior visible propagation + prior motion-interference case; other nine optional',
        primary_cases=list(PRIMARY),source_files=hashes,
        complete_original_video=True,video_reencoded=False,default_prediction_hidden=True,
        outcome_read=False,independent_clinical_validation=False,formal_event_count=None,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),Path(__file__).with_name('whole_video_reference.html'))}))
    (args.output/'README.md').write_text(
        '# 完整视频人工参考\n\n首轮3段全片约3分19秒，其余9例可选。\n\n'
        '先不展开程序，记录你能看见的活动：明确传播、局部活动或不确定；方向使用宫颈→宫底/宫底→宫颈。'
        '同一活动在前后壁都看到只记一次，大致时刻足够，起止边界可留空。\n\n'
        '整段看完后明确选择“活动已记录”“未见明确传播”或“看不清”。未复核、看不清不当零次。'
        '展开程序前的标注会单独保存，后续修改也保留以供对照。\n\n'
        '最后导出JSON给我；CSV作为可选备份。保持目录结构，在本机浏览器打开index.html。'
        '视频仍指向原一倍速文件，未补帧或重新编码。\n\n'
        '这是针对已有开发病例的人工参考，不是独立临床准确率验证。本页不计算准确率或正式波频率。\n',encoding='utf-8')
    print(json.dumps(dict(bundle_id=bundle_id,primary_cases=3,optional_cases=9,
                         primary_total_duration_s=bundle['primary_total_duration_s'],output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-review',type=Path,required=True)
    p.add_argument('--ownership',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())
