import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from analysis.mse_pregnancy_association.code import replay_measurement as replay_entry
from analysis.mse_pregnancy_association.code import record_mse_core as core
from analysis.mse_pregnancy_association.code import record_mse_measurement as entry
from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows as historical
from analysis.mse_pregnancy_association.code.mse_outcome_association import join_measurements
from analysis.tests.fixtures import paired


def payload():
    return dict(case_id='synthetic_measurement',
        paired_ER_windows=[paired(20,.2,.1),paired(21,.4,-.1),paired(50,.1,-.4),paired(51,.2,.3)],
        measurement_metadata=copy.deepcopy(entry.EXPECTED_METADATA))


class RecordMSEBoundary(unittest.TestCase):
    def test_numeric_core_is_verbatim_historical_function_AST(self):
        directory=Path(core.__file__).parent
        original=ast.parse((directory/'complete_record_coordination.py').read_text(encoding='utf-8'))
        copied=ast.parse(Path(core.__file__).read_text(encoding='utf-8'))
        for name in ['_segments','_coordination','_decompose','summarize_paired_windows']:
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in copied.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

    def test_unchanged_outputs_for_gaps_multiple_blocks_and_missing_support(self):
        samples=[payload()['paired_ER_windows'],[],
            [paired(20,1.,0.),paired(20,-1.,0.,7,10),paired(21,.3,-.1)]]
        for rows in samples:
            source=payload();source['paired_ER_windows']=rows
            measured=entry.measure_record(source)
            self.assertEqual(measured['record'],historical(rows))
            self.assertEqual(set(measured),{'case_id','measurement_metadata','record'})

    def test_labels_and_clinical_fields_are_rejected_at_every_input_level(self):
        for level in ['top','metadata','window']:
            for field in ['outcome','pregnancy','clinical','age','BMI']:
                with self.subTest(level=level,field=field):
                    source=payload()
                    target=source if level=='top' else source['measurement_metadata'] if level=='metadata' else source['paired_ER_windows'][0]
                    target[field]=1
                    with self.assertRaises(ValueError):entry.measure_record(source)

    def test_each_protocol_change_fails_before_any_aggregation(self):
        changes=dict(definition_version='different_definition',requested_s_range=[.1,.9],
            nominal_interval_s=.96,nominal_window_s=30.,nominal_step_s=20.,band_cpm=[1.,5.])
        for key,value in changes.items():
            with self.subTest(key=key):
                source=payload();source['measurement_metadata'][key]=value
                with patch.object(entry,'summarize_paired_windows',side_effect=AssertionError('must not aggregate')):
                    with self.assertRaisesRegex(ValueError,'Protocol mismatch'):entry.measure_record(source)

    def test_missing_metadata_and_boolean_aliases_are_not_inferred(self):
        for key in entry.EXPECTED_METADATA:
            source=payload();del source['measurement_metadata'][key]
            with self.assertRaises(ValueError):entry.measure_record(source)
        source=payload();source['measurement_metadata']['nominal_step_s']=True
        with self.assertRaises(ValueError):entry.measure_record(source)

    def test_nonfinite_values_fail_without_turning_missing_into_zero(self):
        source=payload();source['paired_ER_windows'][0]['anterior_ER']=np.nan
        with self.assertRaises(ValueError):entry.measure_record(source)
        source=payload();source['paired_ER_windows']=[]
        self.assertIsNone(entry.measure_record(source)['record']['record_MSE_time_equal'])

    def test_fresh_measurement_import_has_no_clinical_or_model_dependency(self):
        code='import sys; import analysis.mse_pregnancy_association.code.record_mse_measurement; '+\
             'assert not any(x in sys.modules for x in ["pandas","sklearn","scipy",'+\
             '"analysis.mse_pregnancy_association.code.complete_record_coordination"])'
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)

    def test_CLI_only_writes_valid_measurement_output_and_never_overwrites(self):
        module='analysis.mse_pregnancy_association.code.record_mse_measurement'
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'synthetic_input.json';out=Path(tmp)/'synthetic_output.json'
            bad=payload();bad['outcome']=1;source.write_text(json.dumps(bad),encoding='utf-8')
            command=[sys.executable,'-m',module,'--input',str(source),'--output',str(out)]
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertFalse(out.exists())
            source.write_text(json.dumps(payload()),encoding='utf-8')
            subprocess.run(command,check=True,capture_output=True)
            before=out.read_bytes()
            self.assertEqual(json.loads(before)['record'],historical(payload()['paired_ER_windows']))
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(out.read_bytes(),before)

    def test_phase2_exact_left_join_preserves_missing_and_unmatched_records(self):
        first=entry.measure_record(payload())
        other=payload();other['case_id']='synthetic_unmatched';other['paired_ER_windows']=[]
        second=entry.measure_record(other)
        clinical=pd.DataFrame(dict(case_id=['synthetic_measurement'],patient_id=['synthetic_group'],outcome=[1],age=[30]))
        joined=join_measurements([first,second],clinical)
        self.assertEqual(joined.case_id.tolist(),['synthetic_measurement','synthetic_unmatched'])
        self.assertEqual(joined.clinical_join.astype(str).tolist(),['both','left_only'])
        self.assertTrue(pd.isna(joined.loc[1,'record_MSE_time_equal']))
        self.assertTrue(pd.isna(joined.loc[1,'outcome']))

    def test_saved_tracks_CLI_emits_the_strict_phase1_schema(self):
        measured=entry.measure_record(payload())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.savez(root/'synthetic_tracks.npz',dummy=np.zeros(1))
            (root/'synthetic_grid.json').write_text('{}',encoding='utf-8')
            args=['replay','--case-id','synthetic_measurement','--tracks',str(root/'synthetic_tracks.npz'),
                '--grid',str(root/'synthetic_grid.json'),'--output',str(root/'synthetic_result.json')]
            with patch.object(sys,'argv',args),patch.object(replay_entry,'replay',return_value=dict(measurement=measured)):
                replay_entry.main()
            self.assertEqual(json.loads((root/'synthetic_result.json').read_text(encoding='utf-8')),measured)

    def test_phase2_refuses_ambiguous_identity_and_label_recoding(self):
        measured=entry.measure_record(payload())
        clinical=pd.DataFrame(dict(case_id=['synthetic_measurement'],patient_id=['synthetic_group'],outcome=[1]))
        with self.assertRaises(ValueError):join_measurements([measured,measured],clinical)
        with self.assertRaises(ValueError):join_measurements([measured],pd.concat([clinical,clinical]))
        clinical['outcome']='-'
        with self.assertRaises(ValueError):join_measurements([measured],clinical)
