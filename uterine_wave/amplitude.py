"""Amplitude-related summaries for complete radial-strain time blocks."""
import numpy as np


def band_limited_signal(values, time_s, band_cpm=(.5, 5.)):
    """Return a zero-phase FFT band pass of one complete time block.

    ``values`` may have any trailing spatial dimensions.  The result has those
    trailing dimensions.  This is an interval-strain variability measure, not
    a single-event peak amplitude.  Missing samples are not filled.
    """
    values = np.asarray(values, float)
    time_s = np.asarray(time_s, float)
    if values.ndim < 1 or values.shape[0] != len(time_s) or len(time_s) < 3:
        raise ValueError('Expected time-first values with at least three samples')
    if not np.isfinite(values).all() or not np.isfinite(time_s).all():
        raise ValueError('Band-limited SD requires a complete finite block')
    dt = np.diff(time_s)
    if np.any(dt <= 0) or not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-9):
        raise ValueError('FFT amplitude requires uniform real timestamps')
    if not 0 < band_cpm[0] < band_cpm[1]:
        raise ValueError('Invalid temporal frequency band')
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(len(time_s), np.median(dt))
    keep = (frequencies >= band_cpm[0]/60) & (frequencies <= band_cpm[1]/60)
    spectrum[~keep] = 0
    return np.fft.irfft(spectrum, n=len(time_s), axis=0)


def band_limited_sd(values, time_s, band_cpm=(.5, 5.)):
    """Return per-column SD after band limiting a complete time block."""
    return np.std(band_limited_signal(values, time_s, band_cpm), axis=0)


def summarize_interval_strain(values, time_s, band_cpm=(.5, 5.)):
    """Raw and physiological-band SD for one complete interval-strain block."""
    values = np.asarray(values, float)
    band_sd = band_limited_sd(values, time_s, band_cpm)
    raw_sd = np.std(values, axis=0)
    resolution = 60/(len(time_s)*np.median(np.diff(np.asarray(time_s, float))))
    bins = np.fft.rfftfreq(len(time_s), np.median(np.diff(time_s)))
    positive_bins = int(((bins >= band_cpm[0]/60) & (bins <= band_cpm[1]/60)).sum())
    return dict(
        definition='band_limited_interval_radial_strain_SD',
        interpretation='radial_stretch_activity_amplitude',
        units='dimensionless',
        depends_on_measurement_interval_and_motion_rate=True,
        raw_SD_per_pair=raw_sd,
        raw_SD_mean=float(raw_sd.mean()),
        band_SD_per_pair=band_sd,
        band_SD_mean=float(band_sd.mean()),
        band_cpm=list(band_cpm),
        time_resolution_cpm=float(resolution),
        positive_frequency_bins=positive_bins,
        duration_s=float(time_s[-1]-time_s[0]),
    )
