import unittest
import numpy as np
from experiments.all_features_auc import grouped_splits, pipeline


class AllFeatureModelTests(unittest.TestCase):
    def test_inner_groups_are_disjoint_from_outer_test(self):
        groups = np.repeat(np.arange(40), 2)
        y = np.repeat(np.arange(40) % 2, 2)
        for train, test in grouped_splits(y, groups, 5, 20260531):
            for a, b in grouped_splits(y[train], groups[train], 3, 20261531):
                self.assertFalse(set(groups[train[a]]) & set(groups[train[b]]))
                self.assertFalse(set(groups[test]) & set(groups[train[a]]) )
                self.assertFalse(set(groups[test]) & set(groups[train[b]]) )

    def test_classification_lasso_and_svm_keep_preprocessing_inside_pipeline(self):
        for kind in ('lasso','svm'):
            model = pipeline(kind)
            self.assertEqual(list(model.named_steps), ['imputer','scale','model'])
        self.assertEqual(pipeline('lasso').named_steps['model'].penalty,'l1')
        self.assertFalse(pipeline('svm').named_steps['model'].probability)


if __name__=='__main__':
    unittest.main()
