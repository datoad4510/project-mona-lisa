"""Mental-imagery colour classifier.

Training phase
--------------
Each training trial shows the participant a solid colour patch for
TRAINING_STIMULUS_MS ms.  During (or just after) that window the app
records a COLOR_EPOCH_SAMPLES-long EEG segment and stores it with the
colour label.

After all trials are collected, `train()` is called which:
  1. Extracts features from every epoch (band powers across frequency bands
     and channels).
  2. Fits an LDA classifier (fast, works well with small datasets).

Inference phase
---------------
During painting the app continuously collects RECALL_SAMPLES of EEG while
the participant mentally recalls a colour.  `predict()` returns the colour
index with the highest posterior probability.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from collections import deque
import threading

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
from utils.signal_processing import extract_features


class ColorClassifier:
    """Trains on short EEG epochs labelled by colour and classifies recall."""

    def __init__(self, n_colors: int = config.N_TRAINING_COLORS) -> None:
        self.n_colors   = n_colors
        self._pipeline: Optional[Pipeline] = None
        self._trained   = False

        # Inference rolling buffer (frontal channels only)
        cap = config.RECALL_SAMPLES + config.SAMPLE_RATE
        self._inf_buffers = [
            deque(maxlen=cap) for _ in config.COLOR_CHANNEL_INDICES
        ]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: List[np.ndarray],
        labels: List[int],
    ) -> float:
        """Fit the classifier.

        Parameters
        ----------
        epochs : list of (N_CH × COLOR_EPOCH_SAMPLES) arrays
        labels : integer colour index per epoch

        Returns
        -------
        float
            Leave-one-out cross-validation accuracy (0–1).
        """
        # Use only the colour-relevant channels so training and inference
        # operate on the same feature space.
        ch_idx = config.COLOR_CHANNEL_INDICES
        X = np.array([
            extract_features(ep[ch_idx] if ep.shape[0] > len(ch_idx) else ep)
            for ep in epochs
        ])
        y = np.array(labels, dtype=int)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis()),
        ])

        # LOO cross-validation accuracy
        from sklearn.model_selection import cross_val_score, LeaveOneOut
        if len(set(y)) < 2 or len(y) < 4:
            # Not enough data for CV – just fit on all
            self._pipeline.fit(X, y)
            self._trained = True
            return float(np.mean(self._pipeline.predict(X) == y))

        scores = cross_val_score(self._pipeline, X, y, cv=LeaveOneOut())
        self._pipeline.fit(X, y)
        self._trained = True
        return float(scores.mean())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Feed a (N_CH × N_SAMPLES) array; only frontal channels stored."""
        with self._lock:
            for buf_idx, ch_idx in enumerate(config.COLOR_CHANNEL_INDICES):
                self._inf_buffers[buf_idx].extend(chunk[ch_idx].tolist())

    def predict(self) -> Optional[Tuple[int, float]]:
        """Return (color_index, confidence) or None if not ready."""
        if not self._trained or self._pipeline is None:
            return None

        with self._lock:
            min_len = min(len(b) for b in self._inf_buffers)
            if min_len < config.RECALL_SAMPLES:
                return None
            data = np.array(
                [list(b)[-config.RECALL_SAMPLES:] for b in self._inf_buffers],
                dtype=np.float32,
            )

        features = extract_features(data).reshape(1, -1)
        proba    = self._pipeline.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        return best_idx, float(proba[best_idx])
