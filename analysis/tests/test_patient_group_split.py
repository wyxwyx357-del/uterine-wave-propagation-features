import unittest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class PatientGroupSplit(unittest.TestCase):
    def test_original_seed_scheme_keeps_all_cycles_together(self):
        group=np.repeat(np.arange(80),2);y=np.repeat(np.arange(80)%2,2)
        for repeat in range(2):
            outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260531+repeat)
            for fold,(train,test) in enumerate(outer.split(np.zeros(len(y)),y,group),1):
                self.assertFalse(set(group[train])&set(group[test]))
                inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260531+1000+repeat*100+fold)
                for a,b in inner.split(np.zeros(len(train)),y[train],group[train]):
                    self.assertFalse(set(group[train][a])&set(group[train][b]))

    def test_fixed_seed_and_order_reproduce_assignments(self):
        group=np.repeat(np.arange(40),2);y=group%2
        def split():return [test.tolist() for _,test in StratifiedGroupKFold(5,shuffle=True,random_state=20260531).split(np.zeros(len(y)),y,group)]
        self.assertEqual(split(),split())
