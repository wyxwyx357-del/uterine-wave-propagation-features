"""Directional spectral energy. ER describes a pattern, not an event detector."""
import numpy as np


def direction(y, time_s, space, band_cpm=(.5, 5.)):
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
    ef = float(power[band[:, None] & (ft[:, None]*fx[None, :] < 0)].sum())
    eb = float(power[band[:, None] & (ft[:, None]*fx[None, :] > 0)].sum())
    total = float(power.sum())
    er = (ef-eb)/(ef+eb) if ef+eb > max(1e-25, total*1e-12) else None
    return dict(ER=er, forward_energy=ef, backward_energy=eb,
                time_resolution_cpm=60/(len(t)*np.median(dt)),
                duration_s=float(t[-1]-t[0]), spatial_columns=len(space),
                band_cpm=list(band_cpm), event_detected=None)


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
