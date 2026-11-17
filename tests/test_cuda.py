"""Opt-in CUDA smoke test; numerical tests do not require importing OpenCV."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('UTERINE_WAVE_CUDA_TESTS') == '1', 'requires explicit CUDA test environment')
class CudaTracking(unittest.TestCase):
    def test_video_keeps_frames_and_real_time_factor(self):
        import cv2
        from uterine_wave.io import read_video
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25., (80, 60))
            self.assertTrue(writer.isOpened())
            for i in range(16):
                writer.write(np.full((60, 80, 3), 30+i, np.uint8))
            writer.release()
            frames, time_s = read_video(path, time_factor=3)
            self.assertEqual(len(frames), 16)
            self.assertEqual(frames[0].shape, (30, 40))
            np.testing.assert_allclose(time_s, np.arange(16)*.12, atol=1e-9)

    def test_static_and_rigid_shapes_and_flags(self):
        from uterine_wave.controls import synthetic_geometry, synthetic_texture, known_motion
        from uterine_wave.geometry import layout
        from uterine_wave.tracking import track_segments
        from uterine_wave.deformation import compute_maps
        shapes, gray = synthetic_geometry(), synthetic_texture()
        grid = layout(shapes)
        points = np.vstack((grid['A_points'], grid['B_points']))
        t = np.arange(20)*.12
        for condition in ('static', 'rigid'):
            frames, truth, poses = known_motion(gray, shapes, grid, t, condition)
            pose_valid=np.ones(len(t),bool);pose_valid[10]=False
            track = track_segments(frames, points, t, poses, pose_valid=pose_valid)
            np.testing.assert_array_equal(track['pose_valid'],pose_valid)
            self.assertEqual(track['positions'].shape, truth.shape)
            self.assertFalse(track['step_valid'][0].any())
            self.assertTrue(track['target_valid'][1:].any())
            maps = compute_maps(track, grid)
            self.assertFalse(maps['A_valid'][2:11].any())
            self.assertEqual(len(maps['time_s']), len(t)-8)
            if condition == 'static':
                np.testing.assert_allclose(track['corrected'], truth, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
