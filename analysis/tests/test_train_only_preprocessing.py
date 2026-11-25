import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis.pregnancy_prediction.code.clinical9_models import pipeline,preprocessing_state
from analysis.pregnancy_prediction.code import clinical9_frozen_mse as engine
from analysis.tests.fixtures import clinical


class TrainOnlyPreprocessing(unittest.TestCase):
    def test_test_extremes_and_missing_values_do_not_change_training_state(self):
        train=clinical();train.loc[0,'age']=np.nan
        fitted=pipeline('Clinical9_MSE',.1).fit(train,train.outcome)
        before,names=preprocessing_state(fitted,train,'Clinical9_MSE')
        test=clinical(8);test.loc[:,'age']=1e9;test.loc[:,'bmi']=np.nan
        p=fitted.predict_proba(test)[:,1]
        after,_=preprocessing_state(fitted,train,'Clinical9_MSE')
        self.assertEqual(before,after)
        self.assertAlmostEqual(before['medians'][0],train.age.median())
        self.assertTrue(np.isfinite(p).all())

    def test_actual_archived_inner_selection_and_outer_refit_on_synthetic_only(self):
        data=clinical(80)
        train=list(range(60));test=list(range(60,80))
        assignment=np.repeat(np.arange(30)%5+1,2).tolist()
        job=dict(key='synthetic',cohort='MSE185',repeat=0,fold=1,train=train,test=test,inner_assignment=assignment)
        original=(engine.OUT,engine._DATA,engine.MSE_MODELS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                engine.OUT=Path(tmp);(engine.OUT/'jobs').mkdir();engine._DATA=data
                engine.MSE_MODELS=['Clinical9','Clinical9_MSE']
                engine.fit_job(job)
                result=json.loads((engine.OUT/'jobs/synthetic.json').read_text(encoding='utf-8'))
                self.assertEqual(len(result['results']),2)
                for fit in result['results']:
                    self.assertEqual(len(fit['inner_preprocessing']),5)
                    for split in fit['inner_preprocessing']:
                        expected=sum(f!=split['inner_fold'] for f in assignment)
                        self.assertEqual(split['preprocessing']['n_fit'],expected)
                    best=min(fit['inner_scores'],key=lambda s:s['mean_logloss'])['C']
                    self.assertEqual(fit['selected_C'],best)
                    self.assertEqual(fit['classes'],[0,1])
        finally:
            engine.OUT,engine._DATA,engine.MSE_MODELS=original
