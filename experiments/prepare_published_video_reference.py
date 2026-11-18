"""Acquire author-described public ultrasound examples, not patient labels.

Only published supplementary media/legends are used. No annotation request,
outcome access, algorithm tuning or claim of completed detection validation.
"""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
from urllib.request import Request,urlopen
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ARTICLE='https://www.rbmojournal.com/article/S1472-6483%2825%2900562-0/fulltext'
DOI='10.1016/j.rbmo.2025.105355'
ASSETS=[('mmc1.docx','legend',None),('mmc2.mp4','author_example_C2F','C2F'),
        ('mmc3.mp4','author_example_quiescence',None),('mmc4.mp4','author_example_F2C','F2C')]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def build(args):
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for suffix,reference,direction in ASSETS:
        url='https://ars.els-cdn.com/content/image/1-s2.0-S1472648325005620-'+suffix
        path=args.output/('nicoli_2025_'+suffix)
        if not path.exists():
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as response:
                length=response.headers.get('Content-Length')
                with path.open('wb') as stream:
                    for chunk in iter(lambda:response.read(1024*1024),b''):stream.write(chunk)
                if length and path.stat().st_size!=int(length):raise ValueError('Incomplete published file: '+suffix)
        records.append(dict(file=path.name,url=url,bytes=path.stat().st_size,sha256=digest(path),
            reference_kind=reference,author_direction=direction))
    with zipfile.ZipFile(args.output/'nicoli_2025_mmc1.docx') as archive:
        xml=ET.fromstring(archive.read('word/document.xml'))
    ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    legend='\n'.join(''.join(node.text or '' for node in para.iter(ns+'t')) for para in xml.iter(ns+'p'))
    (args.output/'nicoli_2025_legend.txt').write_text(legend,encoding='utf-8')
    expected=('VIDEO 1','VIDEO 2','VIDEO 3','cervix to the fundus','quiescence','fundus to the cervix')
    if not all(text in legend for text in expected):raise ValueError('Published legend differs; inspect mapping before use')
    # Keep this handle alive while OpenCV decodes; Unicode filenames are
    # avoided by relative paths after changing to the output directory.
    dll=os.add_dll_directory(str(args.opencv_bin)) if args.opencv_bin else None
    import cv2
    previous=Path.cwd()
    try:
        os.chdir(args.output.resolve())
        for record in records:
            if not record['file'].endswith('.mp4'):continue
            path=Path(record['file']);cap=cv2.VideoCapture(str(path))
            fps=float(cap.get(cv2.CAP_PROP_FPS));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok,image=cap.read();cap.release()
            if not ok or fps<=0 or frames<1:raise ValueError('Cannot decode published video: '+path.name)
            poster=path.stem+'_first.png';cv2.imencode('.png',image)[1].tofile(poster)
            record.update(encoded_fps=fps,frames=frames,width=width,height=height,encoded_duration_s=frames/fps,
                author_video_number={'mmc2.mp4':1,'mmc3.mp4':2,'mmc4.mp4':3}[path.name.split('nicoli_2025_',1)[1]],
                original_acquisition_time_factor=None,exact_event_count=None,exact_event_arrivals_s=None,
                detector_evaluation_status='not_run',poster=poster)
    finally:os.chdir(previous)
    save(args.output/'published_reference_manifest.json',dict(article_url=ARTICLE,doi=DOI,
        title='Decoding uterine contractility: from physiology to pathology, through emerging technologies',
        reference_source='Author supplementary legend, not user or model medical annotation',assets=records,
        allowed_first_checks=['presence of propagation','anatomical direction','false detections in quiescent example'],
        not_yet_supported=['clinical sensitivity/specificity','exact wave count without event reference',
                           'cycles/min or physical speed without acquisition time scale'],
        methods_changed=False,patient_outcomes_read=False,accuracy=None,event_count=None))
    descriptions={'author_example_C2F':'作者视频1：宫颈 → 宫底传播',
                  'author_example_quiescence':'作者视频2：静息示例',
                  'author_example_F2C':'作者视频3：宫底 → 宫颈传播'}
    cards=[]
    for record in records:
        if not record['file'].endswith('.mp4'):continue
        cards.append('<section><h2>'+descriptions[record['reference_kind']]+'</h2><video controls preload="metadata" poster="'+
            escape(record['poster'])+'" src="'+escape(record['file'])+'"></video><p>文件播放时长 '+
            f"{record['encoded_duration_s']:.2f}"+' 秒；原始采集倍率未确认。作者未提供逐波时间真值，当前未跑程序评价。</p></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>期刊公开传播参考</title><style>body{font:17px/1.7 system-ui;max-width:1050px;margin:30px auto;padding:0 20px;color:#1a2937}video{width:100%;max-height:560px;background:#111}section{padding:12px 0;border-top:1px solid #ccd4df}a{color:#1658a0}</style><h1>期刊公开超声参考</h1><p>无需你做医学标注。以下传播/静息类别直接来自作者配套图例。用于后续外部对照，尚不是程序已通过验证。</p><p><a href="'+escape(ARTICLE)+'">期刊原文</a> · <a href="nicoli_2025_legend.txt">作者视频图例</a> · <a href="published_reference_manifest.json">文件和参考记录</a></p>'+''.join(cards)+'<p>Wang 2022补充S1另存为曲线演示：下载文件没有原始超声画面，不能作为追踪输入；图例所述3次C2F、2次F2C不能当成本程序结果。</p></html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(ultrasound_examples=3,author_labels_verified=True,detector_validation_completed=False,
                         original_time_scale_confirmed=False,output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--opencv-bin',type=Path)
    build(p.parse_args())
