"""g.tec Unicorn Hybrid Black wrapper with a mock/demo mode.

Real mode  → connects to a paired Unicorn over Bluetooth using the UnicornPy
             native SDK and streams live EEG.
Mock mode  → spawns a background thread that generates synthetic EEG whose
             spectral content can be controlled at runtime (used for UI testing
             and demos when the physical headset is absent).

UnicornPy installation
----------------------
UnicornPy ships as part of the **g.tec Unicorn Suite** installer (not on PyPI).
After installing Unicorn Suite, copy ``UnicornPy.so`` (Linux/macOS) or
``UnicornPy.pyd`` (Windows) from the SDK folder into your project root or
virtual-environment ``site-packages``.

Usage
-----
    headset = Headset(mock=True)
    headset.start(callback)   # callback(chunk: np.ndarray) shape (N_CH, N_SAMPLES)
    ...
    headset.stop()
"""

from __future__ import annotations

import logging
import os
import math
import time
import threading
import numpy as np
from typing import Callable, Optional

import config

log = logging.getLogger("headset")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class Headset:
    """Unified interface over the g.tec Unicorn Hybrid Black (real or mock)."""

    def __init__(self, mock: bool = False, device_serial: Optional[str] = None):
        self.mock = mock
        # Optional: pass the Unicorn serial (e.g. "UN-2019.05.51") to target a
        # specific device when multiple are paired.  If None, the first paired
        # device is used.
        self.device_serial = device_serial or os.getenv("UNICORN_DEVICE_SERIAL")

        self._callback: Optional[Callable] = None
        self._running   = False
        self._thread: Optional[threading.Thread] = None

        # Mock controls (set these before or during streaming)
        # ssvep_freq  – inject a sinusoidal component at this Hz into PO7/PO8
        # color_state – 0..N_COLORS-1 to inject colour-imagery pattern
        self.mock_ssvep_freq:  Optional[float] = None
        self.mock_color_state: Optional[int]   = None

        # Real SDK handle
        self._device = None

    # ------------------------------------------------------------------
    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Begin streaming EEG.  *callback* receives (N_CH × N_SAMPLES) arrays."""
        self._callback = callback
        self._running  = True
        mode = "MOCK" if self.mock else "REAL"
        log.info("[headset] Starting in %s mode", mode)
        if self.mock:
            self._thread = threading.Thread(target=self._mock_loop, daemon=True)
            self._thread.start()
        else:
            self._thread = threading.Thread(target=self._real_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        log.info("[headset] Stopping")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._device is not None:
            try:
                self._device.StopAcquisition()
                log.info("[headset] Acquisition stopped")
            except Exception:
                pass
            self._device = None

    # ------------------------------------------------------------------
    # Real SDK  (UnicornPy)
    # ------------------------------------------------------------------

    # The Unicorn SDK delivers data via a synchronous poll call rather than a
    # callback.  We run the poll in a dedicated daemon thread and call the user
    # callback after each read, preserving the same interface as mock mode.
    _CHUNK      = 16   # samples per read ≈ 64 ms at 250 Hz
    _CHUNK_SECS = _CHUNK / config.SAMPLE_RATE

    def _real_loop(self) -> None:
        try:
            import UnicornPy  # type: ignore  # native SDK, not on PyPI
        except ImportError as exc:
            raise RuntimeError(
                "UnicornPy not found.  Install the g.tec Unicorn Suite and "
                "copy UnicornPy.so / UnicornPy.pyd into your Python path."
            ) from exc

        try:
            log.info("[headset] Scanning for paired devices…")
            available = UnicornPy.GetAvailableDevices(True)   # True = paired only
            log.info("[headset] Paired devices found: %s", available)
            if not available:
                raise RuntimeError(
                    "No paired Unicorn device found.  Pair the headset via "
                    "the Unicorn Suite or OS Bluetooth settings first."
                )

            target = self.device_serial
            if target and target in available:
                serial = target
            else:
                serial = available[0]
                if target:
                    log.warning("[headset] Requested serial %s not found; falling back to %s", target, serial)

            log.info("[headset] Connecting to %s…", serial)
            self._device = UnicornPy.Unicorn(serial)
            self._device.StartAcquisition(False)   # False = real signal (not test)
            log.info("[headset] ✓ Connected to %s — streaming started", serial)

            # Allocate a reusable receive buffer:
            # each frame = TotalChannelsCount (17) float32 values = 17 × 4 bytes
            n_total = UnicornPy.TotalChannelsCount   # 17 in SDK v4
            buf_len = self._CHUNK * n_total * 4
            recv_buf = bytearray(buf_len)

            chunks_received = 0
            samples_received = 0
            t_start = time.monotonic()
            t_last_log = t_start

            while self._running:
                self._device.GetData(self._CHUNK, recv_buf, buf_len)

                # Parse: shape (CHUNK, 17), EEG is the first 8 columns
                raw = np.frombuffer(recv_buf, dtype=np.float32)
                raw = raw.reshape((self._CHUNK, n_total))
                # Transpose to (N_CH, N_SAMPLES) — the rest of the app's convention
                chunk = raw[:, :config.N_CHANNELS].T.copy()

                chunks_received  += 1
                samples_received += self._CHUNK

                if self._callback:
                    self._callback(chunk)

                # Log a stats line every 5 seconds
                now = time.monotonic()
                if now - t_last_log >= 5.0:
                    elapsed   = now - t_start
                    actual_hz = samples_received / elapsed
                    eeg_mean  = float(np.mean(np.abs(chunk)))
                    eeg_max   = float(np.max(np.abs(chunk)))
                    log.info(
                        "[headset] ✓ streaming  chunks=%d  samples=%d  "
                        "rate=%.1f Hz  |EEG| mean=%.2f µV  max=%.2f µV",
                        chunks_received, samples_received, actual_hz,
                        eeg_mean, eeg_max,
                    )
                    t_last_log = now

        except Exception as exc:
            log.error("[headset] ✗ Error: %s", exc)
            raise RuntimeError(f"Failed to stream from Unicorn: {exc}") from exc
        finally:
            if self._device is not None:
                try:
                    self._device.StopAcquisition()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Mock generator
    # ------------------------------------------------------------------
    # Unicorn delivers 250 Hz; we emit 16 samples per callback ≈ 64 ms per chunk.

    def _mock_loop(self) -> None:
        t = 0.0
        rng = np.random.default_rng(42)
        sr  = config.SAMPLE_RATE

        chunks_received = 0
        samples_received = 0
        t_start = time.monotonic()
        t_last_log = t_start

        log.info("[headset] Mock streaming started (%d Hz, %d channels)", sr, config.N_CHANNELS)

        while self._running:
            t_samples = np.arange(self._CHUNK) / sr + t
            chunk = rng.normal(0, 8, (config.N_CHANNELS, self._CHUNK)).astype(np.float32)

            # ── inject SSVEP signal into PO7/PO8 ────────────────────────────
            if self.mock_ssvep_freq is not None:
                f = self.mock_ssvep_freq
                wave = (30 * np.sin(2 * math.pi * f * t_samples)).astype(np.float32)
                for idx in config.SSVEP_CHANNEL_INDICES:
                    chunk[idx] += wave

            # ── inject colour-imagery pattern into frontal channels ──────────
            if self.mock_color_state is not None:
                state = self.mock_color_state
                alpha_f = 10.0 + state * 1.5          # shift dominant alpha
                amp     = 15 + state * 5
                wave = (amp * np.sin(2 * math.pi * alpha_f * t_samples)).astype(np.float32)
                for idx in config.COLOR_CHANNEL_INDICES:
                    chunk[idx] += wave

            if self._callback:
                self._callback(chunk)

            chunks_received  += 1
            samples_received += self._CHUNK
            t += self._CHUNK_SECS

            now = time.monotonic()
            if now - t_last_log >= 5.0:
                elapsed   = now - t_start
                actual_hz = samples_received / elapsed
                log.info(
                    "[headset] mock  chunks=%d  samples=%d  rate=%.1f Hz  "
                    "ssvep_freq=%s  color_state=%s",
                    chunks_received, samples_received, actual_hz,
                    self.mock_ssvep_freq, self.mock_color_state,
                )
                t_last_log = now

            time.sleep(self._CHUNK_SECS * 0.95)   # slight underslip prevents drift
