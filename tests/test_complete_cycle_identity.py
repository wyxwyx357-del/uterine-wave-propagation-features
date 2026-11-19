import unittest
from copy import deepcopy
import numpy as np
from experiments.complete_cycle_identity import propose_cycles


def phase(name,sign,offset,epoch=(0,100)):
    return dict(node_id=name,wall=0,sections=[0,1,2],profile={c:[dict(
        sign=sign,peak_s=offset+c,start_s=offset+c-2,end_s=offset+c+2,epoch=epoch)]
        for c in range(3)})


class CompleteCycleTests(unittest.TestCase):
    def test_ordered_pair_produces_one_cycle_without_truth(self):
        r=propose_cycles([phase('p',1,2),phase('n',-1,6)],np.arange(3),2.,.1)
        self.assertEqual(r['complete_cycle_proposal_count'],1)
        self.assertEqual(r['experimental_propagation_count'],1)

    def test_neighboring_half_cycles_cause_competition_not_nearest_pairing(self):
        r=propose_cycles([phase('p',1,2),phase('n',-1,6),phase('p2',1,10)],np.arange(3),2.,.1)
        self.assertEqual(r['complete_cycle_proposal_count'],2)
        self.assertEqual(r['competing_cycle_count'],2)
        self.assertEqual(r['experimental_propagation_count'],0)

    def test_epoch_boundary_cannot_be_spliced(self):
        r=propose_cycles([phase('p',1,2),phase('n',-1,6,(100,200))],np.arange(3),2.,.1)
        self.assertEqual(r['complete_cycle_proposal_count'],0)

    def test_identical_alias_does_not_create_extra_cycle(self):
        p=phase('p',1,2);alias=deepcopy(p);alias['node_id']='alias'
        r=propose_cycles([p,alias,phase('n',-1,6)],np.arange(3),2.,.1)
        self.assertEqual(r['complete_cycle_proposal_count'],1)

    def test_reverse_polarity_is_not_prohibited(self):
        r=propose_cycles([phase('n',-1,2),phase('p',1,6)],np.arange(3),2.,.1)
        self.assertEqual(r['complete_cycle_proposal_count'],1)
        self.assertEqual(r['cycle_proposals'][0]['phase_order'],[-1,1])


if __name__=='__main__':unittest.main()
