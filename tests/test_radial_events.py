import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from uterine_wave.radial_events import detect_radial_events


class RadialWaveCandidates(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(751)*.12
        self.s=np.linspace(.2,.8,17)

    def pulse(self,direction=1,polarity=1):
        arrival=25+direction*(self.s-.5)/.04
        x=self.t[:,None]-arrival[None]
        return polarity*(-.03*x/9)*np.exp(-.5*(x/3)**2)

    def detect(self,y,valid=None):
        return detect_radial_events(y,np.ones_like(y,bool) if valid is None else valid,
                                    self.t,self.s,noise_floor=.001)

    def test_isolated_biphasic_wave_is_one_event_for_both_polarities(self):
        for direction in (1,-1):
            for polarity in (1,-1):
                with self.subTest(direction=direction,polarity=polarity):
                    result=self.detect(self.pulse(direction,polarity))
                    self.assertEqual(result['traveling_candidate_count'],1)
                    event=result['events'][0]
                    self.assertEqual(event['classification'],'C2F' if direction==1 else 'F2C')
                    self.assertAlmostEqual(event['velocity_paths_per_s'],direction*.04,places=5)
                    self.assertGreater(event['spatial_span'],.59)
                    self.assertFalse(event['reviewed'])
                    self.assertIsNone(result['event_frequency_cpm'])

    def test_regular_motion_does_not_require_an_outlier_z_score(self):
        y=.006*np.cos(2*np.pi*(self.t[:,None]/30-self.s[None]))
        result=self.detect(y)
        self.assertGreaterEqual(result['traveling_candidate_count'],2)
        # A phase ridge can be used by only one event, never double counted.
        ridges=[j for e in result['events'] for j in e['phase_ridge_ids']]
        self.assertEqual(len(ridges),len(set(ridges)))

    def test_synchronous_local_and_low_amplitude_are_not_propagating_waves(self):
        sync=.005*np.cos(2*np.pi*self.t[:,None]/30)*np.ones((1,len(self.s)))
        self.assertEqual(self.detect(sync)['traveling_candidate_count'],0)
        local=self.pulse();local[:,:7]=0;local[:,10:]=0
        self.assertEqual(self.detect(local)['traveling_candidate_count'],0)
        self.assertEqual(self.detect(.01*self.pulse())['candidate_count'],0)
        self.assertEqual(self.detect(np.zeros_like(sync))['candidate_count'],0)

    def test_invalid_gap_is_not_filtered_across_or_used_for_phase_pairing(self):
        y=self.pulse();valid=np.ones_like(y,bool)
        gap=(self.t>=22)&(self.t<=27);valid[gap]=False;y[gap]=np.nan
        result=self.detect(y,valid)
        self.assertTrue(np.isnan(result['filtered_A'][gap]).all())
        self.assertTrue(all(not(e['start_s']<22 and e['end_s']>27) for e in result['events']))

    def test_speed_keeps_full_path_units_on_a_cropped_spatial_interval(self):
        event=self.detect(self.pulse())['events'][0]
        self.assertAlmostEqual(event['velocity_paths_per_s'],.04,places=5)
        self.assertNotAlmostEqual(event['velocity_paths_per_s'],.04/.6,places=3)

    def test_no_mutation_and_native_timestamps_required(self):
        y=self.pulse();valid=np.ones_like(y,bool);before=y.copy()
        self.detect(y,valid);np.testing.assert_array_equal(y,before);self.assertTrue(valid.all())
        with self.assertRaises(ValueError):
            detect_radial_events(y,valid,self.t+np.arange(len(self.t))**2*.00001,
                                 self.s,noise_floor=.001)

    def test_cli_native_section_ids_and_review_gate(self):
        from uterine_wave.cli import main
        t=self.t;s=np.linspace(0,1,21)
        x=t[:,None]-(25+(s-.5)/.04)[None]
        wave=-.03*x/9*np.exp(-.5*(x/3)**2)
        a=np.stack((wave,wave),axis=1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            np.savez(root/'maps.npz',A=a,valid_mask=np.ones_like(a,bool),s=s,
                     interval_center_s=t,interval_dt_s=np.full(len(t),2.),nominal_interval_s=2.)
            main(['wave-candidates','--maps',str(root/'maps.npz'),'--noise-floor','.001',
                  '--s-range','.2','.8','--output',str(root/'candidates.json')])
            result=json.loads((root/'candidates.json').read_text(encoding='utf-8'))
            self.assertIsNone(result['event_count']);self.assertIsNone(result['event_frequency_cpm'])
            event=result['walls'][0]['events'][0]
            self.assertGreaterEqual(min(event['sections']),4)
            self.assertEqual(event['wall'],0);self.assertFalse(event['reviewed'])
            with np.load(root/'candidates.npz') as data:
                self.assertTrue(np.isnan(data['filtered_A'][:,:,:4]).all())

    def test_candidates_cannot_enter_reviewed_feature_measurements_unchecked(self):
        from uterine_wave.radial_events import radial_candidates_from_maps
        from uterine_wave.events import measure_event
        a=np.stack((self.pulse(),self.pulse()),axis=1)
        maps=dict(A=a,epsilon=2*a,valid_mask=np.ones_like(a,bool),s=self.s,
                  interval_center_s=self.t,interval_dt_s=np.full(len(self.t),2.))
        report,_=radial_candidates_from_maps(maps,noise_floor=.001)
        candidate=report['walls'][0]['events'][0]
        result=measure_event(maps,candidate)
        self.assertEqual(result['status'],'needs_review')
        self.assertIsNone(result['velocity_paths_per_s'])

    def test_missing_spatial_section_does_not_bridge_two_local_activities(self):
        y=self.pulse();valid=np.ones_like(y,bool)
        valid[:,8]=False
        result=self.detect(y,valid)
        for event in result['events']:
            self.assertTrue(max(event['sections'])<8 or min(event['sections'])>8)

    def test_competing_phase_pairs_keep_supported_candidates_without_reusing_phases(self):
        from uterine_wave.radial_events import _pair_phases
        ridges=[dict(sign=-1),dict(sign=1),dict(sign=-1),dict(sign=1)]
        supports={(0,1):9,(0,3):4,(1,2):5,(2,3):7}
        edges={key:dict(columns=list(range(n)),score=(1,n,0)) for key,n in supports.items()}
        chosen,ambiguous=_pair_phases(ridges,edges)
        self.assertEqual(set(chosen),{(0,1),(2,3)})
        self.assertEqual(set(ambiguous),{0,1,2,3})


if __name__=='__main__':unittest.main()
