import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('sklearn','pandas'))
if AVAILABLE:
    import numpy as np
    import pandas as pd
    from experiments.frozen_feature_auc import grouped_oof, make_model, OUTCOME, GROUP


@unittest.skipUnless(AVAILABLE, 'Optional AUC dependencies are not installed')
class FrozenAUCTests(unittest.TestCase):
    def test_repeat_examinations_never_cross_patient_folds(self):
        groups=np.repeat(np.arange(20),2)
        data=pd.DataFrame({GROUP:groups.astype(str),OUTCOME:np.repeat(np.arange(20)%2,2),
                           'x':np.sin(np.arange(40))})
        predictions,folds,audit=grouped_oof(data,{'example':['x']},5,20260531)
        self.assertTrue(np.isfinite(predictions['example']).all())
        for g in np.unique(groups):self.assertEqual(len(set(folds[groups==g])),1)
        self.assertTrue(all(r['patient_overlap']==0 for r in audit))

    def test_imputation_remains_fitted_on_training_values(self):
        x=pd.DataFrame({'x':[1.,np.nan,3.,4.]})
        model=make_model().fit(x,np.array([0,1,0,1]))
        before=model.named_steps['simpleimputer'].statistics_.copy()
        model.predict_proba(pd.DataFrame({'x':[10000.,np.nan]}))
        self.assertEqual(float(before[0]),3.)
        np.testing.assert_array_equal(model.named_steps['simpleimputer'].statistics_,before)


if __name__=='__main__':unittest.main()
