"""Verify that calibration reaches the frozen measurement and continuity rules."""
import numpy as np
import pytest

from analysis.mse_pregnancy_association.code.step4a_noise_calibration import cached_maps


def cache_fixture(path, *, broken_step=False, baseline=21.):
    t = np.arange(70)*.12
    n = 3
    seeds = np.zeros((7*n,2))
    for wall in (0,1):
        for c in range(n):
            first = (wall*n+c)*2
            seeds[first] = [50+20*c, 50+80*wall]
            seeds[first+1] = seeds[first]+[0,baseline]
    p = np.broadcast_to(seeds,(len(t),)+seeds.shape).copy()
    p += np.column_stack((np.sin(t),np.cos(t)-1))[:,None,:]
    valid = np.ones(p.shape[:2],bool)
    step = valid.copy(); step[0] = False
    if broken_step:
        step[25,0] = False
    np.savez(path,time_s=t,A0_positions=p,A0_target_valid=valid,A0_step_valid=step,
             A0_segment_id=np.zeros_like(valid,int),P3_outer_positions=np.full_like(p,np.nan))


def test_zero_strain_rigid_cache_preserves_native_two_second_definition(tmp_path):
    path = tmp_path/'cache.npz'
    cache_fixture(path)
    before = path.read_bytes()
    maps = cached_maps(path)
    np.testing.assert_allclose(maps['A'],0,atol=1e-15,rtol=0)
    np.testing.assert_allclose(maps['interval_dt_s'],2.04,atol=1e-12,rtol=0)
    assert maps['valid_mask'].all()
    assert path.read_bytes() == before


def test_cached_control_never_bridges_invalid_native_step(tmp_path):
    path = tmp_path/'cache.npz'
    cache_fixture(path,broken_step=True)
    maps = cached_maps(path)
    assert not maps['valid_mask'][8:25,0,0].any()
    assert maps['valid_mask'][:8,0,0].all()
    assert maps['valid_mask'][25:,0,0].all()
    assert maps['valid_mask'][:,:,1:].all()


def test_calibration_rejects_a_different_radial_baseline(tmp_path):
    path = tmp_path/'cache.npz'
    cache_fixture(path,baseline=25.)
    with pytest.raises(ValueError,match='21 px'):
        cached_maps(path)
