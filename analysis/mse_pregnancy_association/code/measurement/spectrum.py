"""Directional spectra; raw FFT power is QC, not cross-grid physical energy."""
import numpy as np


def _spectrum(y, time_s, space, band_cpm=(.5, 5.)):
    """ER>0 means cervix-to-fundus; space must increase anatomically C to F.

    No temporal filling or spatial filling. This diagnostic operates on a
    complete contiguous block. The numerical energy floor is not a noise gate.
    """
    y, t, space = np.asarray(y, float), np.asarray(time_s, float), np.asarray(space, float)
    if y.shape != (len(t), len(space)) or len(t) < 3 or len(space) < 3 or not np.isfinite(y).all():
        raise ValueError("Expected a finite time x space map with at least 3 samples per axis")
    if not np.isfinite(t).all() or not np.isfinite(space).all() or np.any(np.diff(space) <= 0):
        raise ValueError("Coordinates must be finite and ordered")
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-9):
        raise ValueError("FFT requires uniform real timestamps; do not fill missing frames")
    if not 0 < band_cpm[0] < band_cpm[1]:
        raise ValueError("Invalid temporal frequency band")
    ft = np.fft.fftfreq(len(t), np.median(dt))
    g = np.diff(space)
    weights = np.r_[g[0], (g[:-1]+g[1:])/2, g[-1]]
    weights /= weights.mean()
    fx = np.fft.fftfreq(len(space), d=(space[-1]-space[0])/(len(space)-1))
    transformed = np.fft.fft(y-y.mean(0), axis=0)*weights
    spatial = np.exp(-2j*np.pi*(space-space[0])[:, None]*fx[None, :])
    power = np.abs(transformed @ spatial)**2
    band = (np.abs(ft) >= band_cpm[0]/60) & (np.abs(ft) <= band_cpm[1]/60)
    return y, t, space, ft, fx, power, band


def _direction_summary(y, t, space, ft, fx, power, band, band_cpm):
    dt = np.diff(t)
    ef = float(power[band[:, None] & (ft[:, None]*fx[None, :] < 0)].sum())
    eb = float(power[band[:, None] & (ft[:, None]*fx[None, :] > 0)].sum())
    total = float(power.sum())
    band_energy = float(power[band].sum())
    directional_energy = ef+eb
    er = (ef-eb)/(ef+eb) if ef+eb > max(1e-25, total*1e-12) else None
    return dict(ER=er, forward_energy=ef, backward_energy=eb,
                directional_energy=directional_energy,
                spectral_energy_normalization='unnormalized_fft_power',
                spectral_energy_role='QC_only',
                spectral_energy_cross_grid_comparable=False,
                directional_rms=float(np.sqrt(directional_energy)/(len(t)*len(space))),
                directional_band_energy_fraction=(directional_energy/band_energy if band_energy > 1e-25 else None),
                time_resolution_cpm=60/(len(t)*np.median(dt)),
                duration_s=float(t[-1]-t[0]), spatial_columns=len(space),
                band_cpm=list(band_cpm), event_detected=None)


def direction(y, time_s, space, band_cpm=(.5, 5.)):
    """Unchanged ER convention: positive C2F; this does not detect events."""
    return _direction_summary(*_spectrum(y, time_s, space, band_cpm), band_cpm)


def mode_velocity(y, time_s, space, band_cpm=(.5, 5.), *, refine=False):
    """Directional spectral peak speed, not individual-wave arrival speed.

    v=-ft/fs for the Fourier convention used here. When space is the original
    full-wall normalized arc, units are full-wall paths per second. Both
    directions are retained, including standing/balanced activity. No new
    physiological activity threshold or smoothing. The coarse grid peak is
    archived when refine=True requests the continuous candidate. The default
    preserves the baseline pending real-video validation of the candidate.
    """
    arrays = _spectrum(y, time_s, space, band_cpm)
    y, t, space, ft, fx, power, band = arrays
    result = _direction_summary(*arrays, band_cpm)
    result['space_frequency_resolution'] = float(abs(fx[1]))
    result['positive_temporal_band_bins'] = int((band & (ft > 0)).sum())
    floor = max(1e-25, float(power.sum())*1e-12)
    for name, sign in [('C2F', -1), ('F2C', 1)]:
        mask = band[:, None] & (ft[:, None] > 0) & (ft[:, None]*fx[None, :]*sign > 0)
        energy = float(power[mask].sum())
        if not mask.any() or energy <= floor:
            result[name] = dict(status='no_resolved_directional_energy', velocity=None)
            continue
        peak = np.where(mask, power, -np.inf)
        i, j = np.unravel_index(np.argmax(peak), peak.shape)
        result[name] = dict(status='spectral_mode_candidate', velocity=float(-ft[i]/fx[j]),
            temporal_frequency_hz=float(ft[i]), spatial_frequency=float(fx[j]),
            peak_energy_fraction=float(power[i, j]/energy), directional_energy=energy,
            available_spectral_cells=int(mask.sum()),
            lowest_nonzero_spatial_peak=bool(np.isclose(abs(fx[j]),abs(fx[1]))),
            spatial_nyquist_peak=bool(len(fx)%2 == 0 and j == len(fx)//2))
    if refine:
        from .wavefit import fit_wave_modes
        fit = fit_wave_modes(y, t, space, band_cpm)
        result['velocity_fit'] = {key: value for key, value in fit.items() if key != 'components'}
        for name in ('C2F', 'F2C'):
            coarse = result[name]
            component = fit['components'][name]
            result[name] = dict(coarse, grid_peak=coarse.copy())
            result[name].update(component)
            result[name]['velocity_method'] = 'continuous_two_direction_fit'
    return result


def describe(measured, reference, valid, time_s, arc):
    """Matched-support error; direction uses the longest complete spatial run."""
    measured, reference, valid = np.asarray(measured), np.asarray(reference), np.asarray(valid, bool)
    valid = valid & np.isfinite(measured) & np.isfinite(reference)
    a, b = measured[valid], reference[valid]
    baseline = float(np.sqrt(np.mean(b*b))) if len(b) else None
    row = dict(valid_fraction=float(valid.mean()),
               rmse=float(np.sqrt(np.mean((a-b)**2))) if len(a) else None,
               reference_rms=baseline, measured_rms=float(np.sqrt(np.mean(a*a))) if len(a) else None)
    if baseline is not None and baseline > 1e-10:
        row.update(normalized_rmse=row['rmse']/baseline, rms_ratio=row['measured_rms']/baseline,
                   correlation=float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-20 and np.std(b) > 1e-20 else None)
    complete = valid.all(0)
    bounds = np.flatnonzero(np.diff(np.r_[False, complete, False])).reshape(-1, 2)
    choices = [(int(u), int(v)) for u, v in bounds if v-u >= 3]
    if choices:
        u, v = max(choices, key=lambda p: arc[p[1]-1]-arc[p[0]])
        row.update(direction_columns=[u, v],
                   measured_ER=direction(measured[:, u:v], time_s, arc[u:v])['ER'],
                   reference_ER=direction(reference[:, u:v], time_s, arc[u:v])['ER'])
    return row
