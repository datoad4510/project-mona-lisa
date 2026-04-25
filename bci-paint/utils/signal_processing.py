"""Digital signal-processing helpers used by the BCI pipeline."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

import config


# ── Band-pass filter ──────────────────────────────────────────────────────────

def butter_bandpass(
    data:   np.ndarray,
    lowcut: float,
    highcut: float,
    fs:     float = config.SAMPLE_RATE,
    order:  int   = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter along the last axis."""
    nyq  = fs / 2.0
    sos  = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data)


# ── Band-power estimation ─────────────────────────────────────────────────────

def bandpower(
    data:   np.ndarray,
    band:   tuple[float, float],
    fs:     float = config.SAMPLE_RATE,
    nperseg: int  = None,
) -> float:
    """Mean PSD power (µV²/Hz) within *band* averaged over all channels.

    *data* can be 1-D (single channel) or 2-D (channels × samples).
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if nperseg is None:
        nperseg = min(data.shape[-1], fs * 2)   # 2-second windows

    powers = []
    for ch in data:
        freqs, psd = welch(ch, fs=fs, nperseg=int(nperseg))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.mean(powers))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(epoch: np.ndarray) -> np.ndarray:
    """Extract a flat feature vector from a (N_CH × N_SAMPLES) EEG epoch.

    Features (per channel):
        • relative band power for each of the 5 standard bands
        • log-variance of the raw signal
    Total length = N_CH × 6 features.
    """
    n_ch = epoch.shape[0]
    feats = []
    for ch in range(n_ch):
        signal = epoch[ch].astype(np.float64)
        total  = np.var(signal) + 1e-12

        band_powers = []
        for band in config.BANDS.values():
            bp = bandpower(signal, band)
            band_powers.append(bp / total)            # relative power

        log_var = np.log(total + 1e-12)
        feats.extend(band_powers + [log_var])

    return np.array(feats, dtype=np.float32)
