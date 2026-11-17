import unittest
from pathlib import Path
import tempfile
import numpy as np
from uterine_wave.controls import synthetic_geometry, reference_tracks
from uterine_wave.geometry import layout
from uterine_wave.deformation import compute_spacetime, interval_support


class Spacetime(unittest.TestCase):
    def test_cli_defaults_to_physical_interval_and_preserves_pose_mask(self):
        from uterine_wave.cli import main
        grid=layout(synthetic_geometry())
        for fps, expected, extra in [(31,62,[]),(35,70,[]),
                (31,30,['--interval-s','0.96']),(35,34,['--interval-s','0.96'])]:
            with self.subTest(fps=fps,extra=extra), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);t=np.arange(180)/fps
                points=np.repeat(np.vstack((grid['A_points'],grid['B_points']))[None],len(t),axis=0)
                tracks=reference_tracks(points,t)
                tracks['pose_valid']=np.ones(len(t),bool);tracks['pose_valid'][80]=False
                np.savez(root/'tracks.npz',**tracks);np.savez(root/'grid.npz',**grid)
                main(['features','--tracks',str(root/'tracks.npz'),'--layout',str(root/'grid.npz'),
                      '--output',str(root/'result.npz')]+extra)
                with np.load(root/'result.npz') as result:
                    self.assertEqual(int(result['hop']),expected)
                    np.testing.assert_allclose(result['interval_dt_s'],expected/fps)
                    self.assertFalse(result['valid_mask'][80-expected:81].any())
                    self.assertTrue(result['valid_mask'][:80-expected].all())
                    self.assertTrue(result['valid_mask'][81:].all())
                    self.assertIn('interval_center_s',result)

    def test_physical_interval_at_31fps_and_known_extension(self):
        grid=layout(synthetic_geometry());t=np.arange(80)/31
        seeds=np.vstack((grid['A_points'],grid['B_points']))
        scale=1+.02*t
        points=seeds[None]*scale[:,None,None]
        result=compute_spacetime(reference_tracks(points,t),grid)
        self.assertEqual(result['hop'],62)
        np.testing.assert_allclose(result['interval_dt_s'],2.)
        np.testing.assert_allclose(result['interval_center_s'],(t[:-62]+t[62:])/2)
        expected=.02*2/(1+.02*t[:-62])
        np.testing.assert_allclose(result['epsilon'],np.broadcast_to(expected[:,None,None],result['epsilon'].shape),atol=1e-13)
        np.testing.assert_array_equal(result['s'],grid['u'])
        self.assertGreater(result['s'][0],0)
        self.assertLess(result['s'][-1],1)

    def test_mid_interval_pose_failure_invalidates_without_mutating_tracks(self):
        grid=layout(synthetic_geometry());t=np.arange(45)*.12
        p=np.repeat(np.vstack((grid['A_points'],grid['B_points']))[None],len(t),axis=0)
        tracks=reference_tracks(p,t);before=tracks['target_valid'].copy()
        tracks['pose_valid']=np.ones(len(t),bool);tracks['pose_valid'][12]=False
        result=compute_spacetime(tracks,grid,interval_s=.96)
        self.assertFalse(result['valid_mask'][4:13].any())
        self.assertTrue(np.isnan(result['A'][4:13]).all())
        self.assertTrue(result['valid_mask'][:4].all())
        self.assertTrue(result['valid_mask'][13:].all())
        np.testing.assert_array_equal(tracks['target_valid'],before)
        tracks['pose_valid']=np.ones(2,bool)
        with self.assertRaises(ValueError):interval_support(tracks,8)


if __name__=='__main__':unittest.main()
