import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores
from analysis.tests.fixtures import oof


class AUCAggregation(unittest.TestCase):
    def test_fold_scaling_does_not_create_cross_fold_primary_ranking(self):
        data=oof();result=evaluate_frozen_scores(data,['baseline'],iterations=12)
        self.assertAlmostEqual(result['points'][0],1.)
        first=data[data['repeat'].eq(0)]
        self.assertLess(roc_auc_score(first.outcome,first.baseline),1.)

    def test_positive_negative_pair_weights_not_simple_fold_average(self):
        data=oof();data.loc[data.fold.eq(1),'baseline']=1-data.loc[data.fold.eq(1),'baseline']
        # Remove two observations in one fold consistently across repeats; unequal pair counts.
        data=data[~(data.fold.eq(1)&data.exam_id.isin(['fixture_1_4','fixture_1_5']))]
        r=evaluate_frozen_scores(data,['baseline'],iterations=12)
        expected=36/(36+4)
        self.assertAlmostEqual(r['points'][0],expected)
        self.assertNotAlmostEqual(r['points'][0],.8)
