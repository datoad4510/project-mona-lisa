"""Colour-training screen.

Flow
----
1. Display an instruction banner: "We will show you colours. Please focus on
   each colour as it appears."
2. For each trial:
     a. Show a full-screen colour patch for TRAINING_STIMULUS_MS.
     b. During the patch the app records EEG into `_current_epoch_buffer`.
     c. After TRAINING_STIMULUS_MS hide the patch (grey ISI screen) and
        save the epoch.
3. After all trials emit `training_complete(epochs, labels)`.

The actual EEG recording is driven externally: the main app feeds raw EEG
chunks via `add_chunk()` which this widget buffers.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal
from PyQt5.QtGui     import QColor, QPainter, QFont, QBrush
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

import config


class TrainingScreen(QWidget):
    """Full-window training stimulus and EEG collection."""

    # Emitted when all trials are done
    training_complete = pyqtSignal(list, list)   # epochs: list[np.ndarray], labels: list[int]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        self._epochs:  List[np.ndarray] = []
        self._labels:  List[int]        = []

        # Trial sequencing
        self._trial_list: List[int] = []   # sequence of colour indices
        self._trial_idx   = 0
        self._in_stimulus = False

        # EEG epoch buffer (all channels)
        self._epoch_buf = deque(maxlen=config.COLOR_EPOCH_SAMPLES + config.SAMPLE_RATE)
        self._collecting = False

        # UI state
        self._current_color: Optional[QColor] = None
        self._phase = "idle"    # "idle" | "instruction" | "stimulus" | "isi" | "done"
        self._instruction_text = ""
        self._countdown = 0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._next_phase)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_training(self) -> None:
        """Build the trial list and begin."""
        colors = config.TRAINING_COLORS[:config.N_TRAINING_COLORS]
        self._trial_list = [
            color_idx
            for color_idx in range(len(colors))
            for _ in range(config.TRAINING_TRIALS_PER_COLOR)
        ]
        # Shuffle so colours alternate
        import random
        random.shuffle(self._trial_list)

        self._trial_idx = 0
        self._epochs.clear()
        self._labels.clear()
        self._phase = "instruction"
        self._instruction_text = (
            "Colour Training\n\n"
            "You will see coloured patches.\n"
            "Focus your attention on each colour.\n\n"
            "Starting in 3 seconds…"
        )
        self.update()
        self._timer.start(3000)

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Feed a raw EEG chunk (N_CH × N_SAMPLES) during training."""
        if self._collecting:
            for sample_col in range(chunk.shape[1]):
                self._epoch_buf.append(chunk[:, sample_col])

    # ------------------------------------------------------------------
    # Internal sequencing
    # ------------------------------------------------------------------

    def _next_phase(self) -> None:
        if self._phase == "instruction":
            self._show_stimulus()
        elif self._phase == "stimulus":
            self._show_isi()
        elif self._phase == "isi":
            if self._trial_idx < len(self._trial_list):
                self._show_stimulus()
            else:
                self._finish()

    def _show_stimulus(self) -> None:
        color_idx = self._trial_list[self._trial_idx]
        self._trial_idx   += 1
        color_info         = config.TRAINING_COLORS[color_idx]
        self._current_color = QColor(*color_info["rgb"])
        self._phase         = "stimulus"
        self._collecting    = True
        self._epoch_buf.clear()
        self.update()
        self._timer.start(config.TRAINING_STIMULUS_MS)

    def _show_isi(self) -> None:
        self._collecting = False
        # Save epoch
        color_idx = self._trial_list[self._trial_idx - 1]
        if len(self._epoch_buf) >= config.COLOR_EPOCH_SAMPLES:
            arr = np.array(list(self._epoch_buf)[-config.COLOR_EPOCH_SAMPLES:]).T
            self._epochs.append(arr.astype(np.float32))
            self._labels.append(color_idx)
        self._current_color = None
        self._phase         = "isi"
        self.update()
        self._timer.start(config.TRAINING_ISI_MS)

    def _finish(self) -> None:
        self._phase = "done"
        self._instruction_text = (
            "Training complete!\n\n"
            f"Collected {len(self._epochs)} epochs.\n"
            "Building colour model…"
        )
        self.update()
        self.training_complete.emit(self._epochs, self._labels)

    # ------------------------------------------------------------------
    # Qt painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._phase == "stimulus" and self._current_color:
            # Full-screen colour patch
            painter.fillRect(self.rect(), QBrush(self._current_color))
            # Progress bar at bottom
            total   = len(self._trial_list)
            done    = self._trial_idx
            bar_w   = int(self.width() * done / max(total, 1))
            painter.fillRect(0, self.height() - 8, bar_w, 8,
                             QBrush(QColor(255, 255, 255, 160)))
        else:
            # Instruction / ISI / done — grey background with text
            painter.fillRect(self.rect(), QBrush(QColor(40, 40, 40)))
            painter.setPen(QColor(220, 220, 220))
            font_pt = max(10, int(self.height() * 0.029))
            font = QFont("Arial", font_pt)
            font.setWordSpacing(2)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self._instruction_text)
