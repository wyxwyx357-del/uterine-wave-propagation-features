import unittest
from experiments.compare_clinical_salience import comparison_row


class ClinicalSalienceComparison(unittest.TestCase):
    def test_zero_record_and_local_proposal_do_not_fabricate_wave_rate(self):
        ref=dict(review_id='case_test',match_status='unique',actual_video_duration_s=60.,
                 matches=[dict(positive_value=0.,negative_value=0.,positive_raw='0',negative_raw='0')])
        pred=dict(walls=[dict(metrics=dict(after_proposals=1,after_directional=0)),
                         dict(metrics=dict(after_proposals=0,after_directional=0))])
        row=comparison_row(ref,pred)
        self.assertFalse(row['clinical_has_recorded_activity'])
        self.assertTrue(row['has_salient_proposal'])
        self.assertFalse(row['has_directional_salient_proposal'])
        self.assertIsNone(row['algorithm_event_rate_cpm'])
        self.assertEqual(row['observation_period_correspondence'],'user_unconfirmed')

    def test_missing_or_ambiguous_reference_is_not_zero_activity(self):
        pred=dict(walls=[dict(metrics=dict(after_proposals=0,after_directional=0)) for _ in range(2)])
        for matches in ([],[dict(positive_value=1.,negative_value=None)],[{},{}]):
            row=comparison_row(dict(review_id='test',match_status='unresolved',matches=matches),pred)
            self.assertIsNone(row['clinical_has_recorded_activity'])
            self.assertIsNone(row['clinical_column_sum_cpm'])


if __name__=='__main__':unittest.main()
