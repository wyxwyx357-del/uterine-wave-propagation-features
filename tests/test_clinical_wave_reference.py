"""Synthetic workbook checks; no patient or outcome data."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from xml.sax.saxutils import escape
import zipfile
from experiments.match_clinical_wave_reference import read_reference,run


def workbook(path,positive='2',negative='1'):
    columns={'F':'视频研究编号','H':'视频检查时间（首帧）','I':'视频时长（秒，CSV估计）',
             'K':'临床检查编号','T':'临床_容受检查时间','X':'临床_蠕动（正向）','Y':'临床_蠕动（反向）','P':'UNUSED_COLUMN'}
    def inline(col,row,text):return f'<c r="{col}{row}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
    header=''.join(inline(c,1,t)for c,t in columns.items())
    body=''.join(inline(c,2,t)for c,t in {'F':'9001','H':'2024-01-02 11:59:00','K':'DEMO','X':positive,'Y':negative,'P':'UNUSED_SENTINEL'}.items())
    body+='<c r="I2"><v>120</v></c><c r="T2"><v>45293.5</v></c>'
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(path,'w')as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="最终队列" sheetId="2" r:id="r2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r2" Target="worksheets/sheet2.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet2.xml',f'<worksheet xmlns="{ns}"><sheetData><row r="1">{header}</row><row r="2">{body}</row></sheetData></worksheet>')


class ClinicalReference(unittest.TestCase):
    def test_allowed_columns_named_sheet_and_excel_date(self):
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'reference.xlsx';workbook(p);rows=read_reference(p)
            self.assertEqual(rows[0]['clinical_datetime'],'2024-01-02 12:00:00')
            self.assertEqual(rows[0]['positive_value'],2.)
            self.assertNotIn('UNUSED_SENTINEL',json.dumps(rows))

    def test_zero_missing_and_non_numeric_are_different(self):
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'reference.xlsx';workbook(p,'0','0');r=read_reference(p)[0]
            self.assertEqual(r['reference_kind'],'numeric_pair');self.assertEqual(r['negative_value'],0.)
            workbook(p,'杂乱无序','');r=read_reference(p)[0]
            self.assertEqual(r['positive_raw'],'杂乱无序')
            self.assertIsNone(r['positive_value']);self.assertIsNone(r['negative_value'])

    def test_per_minute_is_not_multiplied_by_video_duration_or_direction_guessed(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);p=root/'reference.xlsx';workbook(p);source=root/'review';source.mkdir()
            (source/'protocol.json').write_text(json.dumps({'selected':[{'case_id':'9001-DEMO-20240102','review_id':'case_demo'}]}))
            (source/'screening_summary.json').write_text(json.dumps([{'review_id':'case_demo','duration_s':120,'directional_structures_per_wall':[9,8]}]))
            with redirect_stdout(io.StringIO()):
                run(SimpleNamespace(workbook=p,cohort_review=source,output=root/'out',record_unit='per_minute',positive_direction='unconfirmed'))
            reference=json.loads((root/'out/recorded_reference.json').read_text(encoding='utf-8'))[0]
            self.assertEqual(reference['recorded_positive_rate_cpm'],2.)
            self.assertEqual(reference['recorded_negative_rate_cpm'],1.)
            self.assertEqual(reference['recorded_directional_sum_cpm'],3.)
            self.assertEqual(reference['positive_direction_definition'],'unconfirmed')
            match=json.loads((root/'out/review_case_matches.json').read_text(encoding='utf-8'))[0]
            self.assertIsNone(match['algorithm_event_count'])


if __name__=='__main__':unittest.main()
