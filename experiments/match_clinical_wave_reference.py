"""Read recorded wave quantities and match existing review cases, without outcomes.

This produces a clinical-reference audit, not algorithm accuracy: observation
duration, direction convention and event identity still require confirmation.
The workbook is read-only and patient-level outputs must remain local.
"""
import argparse
from collections import Counter
from datetime import datetime,timedelta
import csv
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
FIELDS={'study_number':'视频研究编号','video_datetime':'视频检查时间（首帧）',
        'video_duration_recorded_s':'视频时长（秒，CSV估计）','clinical_check_id':'临床检查编号',
        'clinical_datetime':'临床_容受检查时间','positive_raw':'临床_蠕动（正向）',
        'negative_raw':'临床_蠕动（反向）'}


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def numeric(value):
    if not str(value).strip():return None
    try:result=float(value)
    except ValueError:return None
    return result if math.isfinite(result) else None


def read_reference(path):
    """Decode only explicitly allowed body columns on the named final sheet."""
    with zipfile.ZipFile(path)as archive:
        shared=[]
        if 'xl/sharedStrings.xml'in archive.namelist():
            shared=[''.join(node.itertext())for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si',NS)]
        relationships={r.attrib['Id']:r.attrib['Target']for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        workbook=ET.fromstring(archive.read('xl/workbook.xml'))
        sheet=next(s for s in workbook.findall('m:sheets/m:sheet',NS)if s.attrib['name']=='最终队列')
        target=relationships[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
        target=target.lstrip('/')if target.startswith('/')else posixpath.normpath('xl/'+target)
        rows=ET.fromstring(archive.read(target)).findall('m:sheetData/m:row',NS)
        def value(cell):
            node=cell.find('m:v',NS)
            if cell.attrib.get('t')=='s':return shared[int(node.text)]if node is not None else ''
            if cell.attrib.get('t')=='inlineStr':
                node=cell.find('m:is',NS)
                return ''.join(node.itertext())if node is not None else ''
            return node.text if node is not None else ''
        header={re.sub(r'\d','',c.attrib['r']):value(c)for c in rows[0]}
        columns={col:key for col,name in header.items()for key,title in FIELDS.items()if name==title}
        if set(columns.values())!=set(FIELDS):raise ValueError('Required clinical-reference columns are missing')
        records=[]
        for row in rows[1:]:
            record={key:'' for key in FIELDS};record['excel_row']=int(row.attrib['r'])
            for cell in row:
                col=re.sub(r'\d','',cell.attrib['r'])
                if col in columns:record[columns[col]]=value(cell)
            if not any(record[key]for key in FIELDS):continue
            record['positive_value']=numeric(record['positive_raw'])
            record['negative_value']=numeric(record['negative_raw'])
            record['video_duration_recorded_s']=numeric(record['video_duration_recorded_s'])
            serial=numeric(record['clinical_datetime'])
            if serial is not None:
                record['clinical_datetime']=(datetime(1899,12,30)+timedelta(days=serial)).isoformat(sep=' ',timespec='seconds')
            record['reference_kind']=('numeric_pair'if record['positive_value']is not None and record['negative_value']is not None
                                      else 'missing_or_non_numeric')
            records.append(record)
    return records


def run(args):
    references=read_reference(args.workbook)
    for record in references:
        record['record_unit']=args.record_unit
        record['positive_direction_definition']=args.positive_direction
        if args.record_unit=='per_minute':
            record['recorded_positive_rate_cpm']=record['positive_value']
            record['recorded_negative_rate_cpm']=record['negative_value']
            record['recorded_directional_sum_cpm']=(record['positive_value']+record['negative_value']
                if record['reference_kind']=='numeric_pair'else None)
    protocol=json.loads((args.cohort_review/'protocol.json').read_text(encoding='utf-8'))
    screened=json.loads((args.cohort_review/'screening_summary.json').read_text(encoding='utf-8'))
    lookup={row['review_id']:row for row in screened};matched=[]
    for selected in protocol['selected']:
        study=selected['case_id'].split('-')[0]
        date=datetime.strptime(selected['case_id'].rsplit('-',1)[1],'%Y%m%d').date()
        candidates=[r for r in references if r['study_number'].strip()==study and
                    r['video_datetime'][:10]==date.isoformat()]
        row=dict(review_id=selected['review_id'],study_number=study,
                 match_status='unique_study_number_and_video_date'if len(candidates)==1 else 'unresolved',
                 matches=candidates,observation_duration_s=None,positive_direction_definition=args.positive_direction,
                 clinical_record_unit=args.record_unit,
                 algorithm_event_count=None)
        if selected['review_id']in lookup:
            result=lookup[selected['review_id']]
            row.update(actual_video_duration_s=result.get('duration_s'),
                       directional_structures_per_wall=result.get('directional_structures_per_wall'),
                       algorithm_structures_are_not_event_counts=True)
        matched.append(row)
    numeric_rows=[r for r in references if r['reference_kind']=='numeric_pair']
    zero_rows=[r for r in numeric_rows if r['positive_value']==0 and r['negative_value']==0]
    matched_zero=[r['review_id']for r in matched if len(r['matches'])==1 and
                  r['matches'][0]['positive_value']==0 and r['matches'][0]['negative_value']==0]
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'recorded_reference.json',references);save(args.output/'review_case_matches.json',matched)
    summary=dict(source_workbook=str(args.workbook),source_sha256=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        source_sheet='最终队列',body_columns_read=list(FIELDS.values()),records=len(references),
        numeric_pairs=len(numeric_rows),both_recorded_zero=len(zero_rows),
        non_numeric_or_missing=len(references)-len(numeric_rows),
        positive_value_distribution=dict(Counter(r['positive_raw']for r in references)),
        negative_value_distribution=dict(Counter(r['negative_raw']for r in references)),
        reviewed_cases=len(matched),uniquely_matched=sum(r['match_status'].startswith('unique')for r in matched),
        matched_both_zero=matched_zero,outcomes_used=False,
        clinical_record_unit=args.record_unit,record_unit_confirmed=args.record_unit!='unconfirmed',
        clinical_observation_duration_confirmed=False,direction_definition_confirmed=args.positive_direction!='unconfirmed',
        algorithm_accuracy=None)
    save(args.output/'summary.json',summary)
    with (args.output/'review_case_matches.csv').open('w',encoding='utf-8-sig',newline='')as stream:
        suffix='_cpm'if args.record_unit=='per_minute'else '_value'
        writer=csv.writer(stream);writer.writerow(['review_id','video_date','clinical_positive_recorded'+suffix,'clinical_negative_recorded'+suffix,
            'actual_video_duration_s','anterior_direction_structures_not_waves','posterior_direction_structures_not_waves','excel_row','status'])
        for r in matched:
            ref=r['matches'][0]if len(r['matches'])==1 else {}
            directions=r.get('directional_structures_per_wall')or['','']
            writer.writerow([r['review_id'],ref.get('video_datetime',''),ref.get('positive_raw',''),ref.get('negative_raw',''),
                r.get('actual_video_duration_s',''),*directions,ref.get('excel_row',''),r['match_status']])
    lines=['# 临床蠕动记录匹配','',
        f"参考表中读取{len(references)}条记录；{len(numeric_rows)}条正反均为数值，{len(zero_rows)}条为0/0。",
        f"本次{len(matched)}例视频匹配{summary['uniquely_matched']}例；0/0病例：{', '.join(matched_zero)}。",' ',
        '仅使用视频编号、时间、视频时长、临床检查编号和正反向原始记录，未使用妊娠结局。',
        '临床观察时段对应关系与正向定义仍需说明；程序尚无正式事件频率，本次不计算准确率或ICC。',
        f'用户确认的记录单位：{args.record_unit}；正向解剖定义：{args.positive_direction}。',
        '次/分记录不自动乘视频时长变成整数次数；方向定义未知时保留正/反原标签。',
        '正反频率合计仅指两列之和，不自动等于去重后的独立生理事件频率。',
        '视频时长不自动等于临床观察时长；同日匹配不证明临床记录针对保存视频。',
        '程序当前输出方向结构，不是经事件关联去重的波次数。0/0病例有候选时，优先检查假候选及统计时段对应关系。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k]for k in('records','numeric_pairs','both_recorded_zero','non_numeric_or_missing','reviewed_cases','uniquely_matched','matched_both_zero')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook',type=Path,required=True)
    parser.add_argument('--cohort-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record-unit',choices=('unconfirmed','per_minute'),default='unconfirmed')
    parser.add_argument('--positive-direction',choices=('unconfirmed','C2F','F2C'),default='unconfirmed')
    run(parser.parse_args())


if __name__=='__main__':main()
