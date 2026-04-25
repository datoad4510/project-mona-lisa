"""SSVEP (Steady-State Visual Evoked Potential) frequency detector.

The canvas regions flicker at distinct frequencies.  When the participant
stares at one region their occipital EEG acquires a strong sinusoidal
component at that frequency.

Algorithm
---------
1. Maintain a rolling buffer of the last SSVEP_WINDOW_SAMPLES samples from
   the PO3/PO4 channels.
2. Every time new samples arrive, compute the FFT over the window.
3. For each target frequency f compute an SNR score:
       SNR(f) = power(f) / median(power in ±2 Hz neighbourhood excluding f)
4. Return the frequency with the highest SNR if it exceeds SSVEP_SNR_THRESHOLD,
   otherwise return None (no clear target detected).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import welch

import config


class SSVEPDetector:
    """Thread-safe rolling SSVEP detector."""

    def __init__(self) -> None:
        cap = config.SSVEP_WINDOW_SAMPLES + config.SAMPLE_RATE  # extra headroom
        # Separate buffer per SSVEP channel
        self._buffers = [
            deque(maxlen=cap) for _ in config.SSVEP_CHANNEL_INDICES
        ]
        self._lock = __import__("threading").Lock()

    # ------------------------------------------------------------------
    def add_chunk(self, chunk: np.ndarray) -> None:
        """Feed a (N_CH × N_SAMPLES) array; only SSVEP channels are stored."""
        with self._lock:
            for buf_idx, ch_idx in enumerate(config.SSVEP_CHANNEL_INDICES):
                self._buffers[buf_idx].extend(chunk[ch_idx].tolist())

    # ------------------------------------------------------------------
    def detect(self) -> Optional[float]:
        """Return the detected SSVEP frequency (Hz) or None."""
        with self._lock:
            min_len = min(len(b) for b in self._buffers)
            if min_len < config.SSVEP_WINDOW_SAMPLES:
                return None
            data = np.array(
                [list(b)[-config.SSVEP_WINDOW_SAMPLES:] for b in self._buffers],
                dtype=np.float32,
            )                                        # (n_ssvep_ch, window)

        # Average power spectrum across selected channels
        avg_snr  = self._compute_snr(data)
        best_idx = int(np.argmax(avg_snr))
        best_snr = avg_snr[best_idx]

        if best_snr >= config.SSVEP_SNR_THRESHOLD:
            return config.SSVEP_FREQUENCIES[best_idx]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_snr(self, data: np.ndarray) -> np.ndarray:
        """Return SNR value for each target frequency, averaged over channels."""
        sr   = config.SAMPLE_RATE
        nperseg = min(config.SSVEP_WINDOW_SAMPLES, sr * 2)
        snr_per_channel = []

        for ch_signal in data:
            freqs, psd = welch(ch_signal, fs=sr, nperseg=nperseg)
            ch_snr = []
            for f_target in config.SSVEP_FREQUENCIES:
                # power at target bin
                target_idx = int(np.argmin(np.abs(freqs - f_target)))
                target_pwr = psd[target_idx]

                # noise floor: bins 1–3 Hz away (exclude ±0.5 Hz)
                noise_mask = (
                    (freqs >= f_target - 3) & (freqs <= f_target + 3)
                    & (np.abs(freqs - f_target) > 0.5)
                )
                noise_pwr = np.median(psd[noise_mask]) if noise_mask.any() else 1e-9
                ch_snr.append(target_pwr / (noise_pwr + 1e-9))

            snr_per_channel.append(ch_snr)

        return np.mean(snr_per_channel, axis=0)   # (n_frequencies,)
