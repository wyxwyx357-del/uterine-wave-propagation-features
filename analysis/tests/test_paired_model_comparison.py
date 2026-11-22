import unittest
import numpy as np
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores
from analysis.tests.fixtures import oof


class PairedComparison(unittest.TestCase):
    def test_identical_rankings_have_zero_paired_delta_in_all_cluster_draws(self):
        result=evaluate_frozen_scores(oof(),['baseline','augmented'],iterations=16)
        np.testing.assert_array_equal(result['draws'][:,1]-result['draws'][:,0],np.zeros(16))
        self.assertEqual(result['n_unique_patients'],30)

    def test_unpaired_missing_scores_are_rejected(self):
        data=oof();data.loc[0,'augmented']=np.nan
        with self.assertRaises(AssertionError):evaluate_frozen_scores(data,['baseline','augmented'],iterations=2)

    def test_patient_crossing_outer_folds_is_rejected(self):
        data=oof();data.loc[data.exam_id.eq('fixture_2_0'),'patient_id']='synthetic_1_0'
        with self.assertRaises(AssertionError):evaluate_frozen_scores(data,['baseline','augmented'],iterations=2)
