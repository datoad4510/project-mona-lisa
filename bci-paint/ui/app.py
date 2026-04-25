"""Main application window.

State machine
-------------
CONNECTING  →  TRAINING  →  READY  →  PAINTING

CONNECTING
    Show a loading splash.  Headset connects in background.
    → TRAINING once headset is live.

TRAINING
    TrainingScreen shown.  EEG chunks fed to both SSVEPDetector and
    TrainingScreen's epoch buffer.  Classifier trained at the end.
    → READY once training_complete signal fires and model is fitted.

READY
    Brief confirmation screen ("Model ready — press Space to start painting").
    → PAINTING on Space or button press.

PAINTING
    PaintCanvas shown with SSVEP overlays flickering.
    Two parallel BCI loops driven by a QTimer (100 ms tick):

    1. Region selection (SSVEP)
       SSVEPDetector accumulates chunks.  Every tick we call detect().
       If it returns a frequency we map it to a region.  A running stare
       timer (_stare_timer) tracks how long the same region has been
       detected consecutively.  After STARE_CONFIRM_SEC the region is
       highlighted, then confirmed and coloured.

    2. Colour selection (mental imagery)
       ColorClassifier accumulates chunks.  Every tick we call predict().
       The current colour is displayed in the status bar.  As soon as a
       region is confirmed we apply the current colour.

    A "Retrain" button drops back to TRAINING.  A "Reset" button clears the
    canvas.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np

from PyQt5.QtCore    import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui     import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QSizePolicy,
)

import config
from bci            import Headset, SSVEPDetector, ColorClassifier
from ui.canvas      import PaintCanvas
from ui.training_screen import TrainingScreen
from ui.ssvep_overlay   import SSVEPOverlay


# ---------------------------------------------------------------------------
# EEG relay — forwards headset chunks to registered consumers via Qt signals
# (Qt signals are thread-safe; the headset callback runs on a worker thread)
# ---------------------------------------------------------------------------

class _EEGRelay(QObject):
    chunk_received = pyqtSignal(object)   # np.ndarray


# ---------------------------------------------------------------------------
# Frequency → region mapping
# ---------------------------------------------------------------------------

_FREQ_TO_REGION: Dict[float, int] = {
    r["ssvep_freq"]: r["id"] for r in config.CANVAS_REGIONS
}


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BCIPaintApp(QMainWindow):
    """Root window; orchestrates all states."""

    def __init__(self, mock: bool = False, debug_click: bool = False):
        super().__init__()
        self.setWindowTitle("BCI Paint — Neurosity Crown")
        self.mock        = mock
        self.debug_click = debug_click

        # BCI components
        self._headset     = Headset(mock=mock)
        self._ssvep       = SSVEPDetector()
        self._classifier  = ColorClassifier(n_colors=config.N_TRAINING_COLORS)
        self._relay       = _EEGRelay()
        self._relay.chunk_received.connect(self._on_chunk_main_thread)

        # State
        self._state            = "CONNECTING"
        self._current_color_idx: Optional[int] = None
        self._stare_start_ms:    Optional[int] = None
        self._stare_region_id:   Optional[int] = None

        # Tick timer (drives BCI inference loop)
        self._tick = QTimer(self)
        self._tick.setInterval(100)   # 100 ms
        self._tick.timeout.connect(self._bci_tick)

        self._setup_ui()
        self.setFocusPolicy(Qt.StrongFocus)   # ensure main window captures key events
        self._connect_headset()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setFixedSize(config.CANVAS_W + 220, config.CANVAS_H + 80)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Main stacked area ───────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setFixedSize(config.CANVAS_W, config.CANVAS_H)

        # Page 0 – connecting splash
        self._page_connecting = self._make_splash("Connecting to Neurosity Crown…")
        self._stack.addWidget(self._page_connecting)

        # Page 1 – training screen
        self._training_screen = TrainingScreen()
        self._training_screen.training_complete.connect(self._on_training_complete)
        self._stack.addWidget(self._training_screen)

        # Page 2 – ready screen
        self._page_ready = self._make_ready_screen()
        self._stack.addWidget(self._page_ready)

        # Page 3 – paint canvas
        self._canvas  = PaintCanvas(debug_click=self.debug_click)
        self._overlay = SSVEPOverlay(self._canvas)
        if self.debug_click:
            self._canvas.region_clicked.connect(self._debug_fill_region)
        self._stack.addWidget(self._canvas)

        root_layout.addWidget(self._stack)

        # ── Right side-panel ────────────────────────────────────────────
        panel = self._build_side_panel()
        root_layout.addWidget(panel)

        # ── Bottom status bar ────────────────────────────────────────────
        self._status_bar = QLabel("  Status: Connecting…")
        self._status_bar.setFixedHeight(30)
        self._status_bar.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; font-size: 12px; padding-left: 8px;"
        )
        main_frame = QWidget()
        main_frame_layout = QVBoxLayout(main_frame)
        main_frame_layout.setContentsMargins(0, 0, 0, 0)
        main_frame_layout.setSpacing(0)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self._stack)
        h.addWidget(panel)
        main_frame_layout.addLayout(h)
        main_frame_layout.addWidget(self._status_bar)
        self.setCentralWidget(main_frame)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        panel.setStyleSheet("background: #16213e;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignTop)

        # Title
        title = QLabel("BCI Paint")
        title.setStyleSheet("color: #e94560; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # Current colour indicator
        layout.addWidget(self._section_label("Active Colour"))
        self._color_display = QLabel()
        self._color_display.setFixedHeight(50)
        self._color_display.setAlignment(Qt.AlignCenter)
        self._color_display.setStyleSheet(
            "background: #888; border-radius: 6px; color: white; font-size: 13px;"
        )
        self._color_display.setText("—")
        layout.addWidget(self._color_display)

        # Colour confidence
        layout.addWidget(self._section_label("Colour confidence"))
        self._color_confidence = QProgressBar()
        self._color_confidence.setRange(0, 100)
        self._color_confidence.setValue(0)
        self._color_confidence.setStyleSheet(self._progress_style("#e94560"))
        layout.addWidget(self._color_confidence)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #333;")
        layout.addWidget(sep2)

        # Stare region
        layout.addWidget(self._section_label("Stare target"))
        self._stare_label = QLabel("—")
        self._stare_label.setStyleSheet("color: #a8dadc; font-size: 14px;")
        layout.addWidget(self._stare_label)

        # Stare progress
        layout.addWidget(self._section_label(f"Stare progress ({config.STARE_CONFIRM_SEC:.0f} s)"))
        self._stare_bar = QProgressBar()
        self._stare_bar.setRange(0, 100)
        self._stare_bar.setValue(0)
        self._stare_bar.setStyleSheet(self._progress_style("#a8dadc"))
        layout.addWidget(self._stare_bar)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("color: #333;")
        layout.addWidget(sep3)

        # Buttons
        btn_reset = QPushButton("Reset Canvas")
        btn_reset.clicked.connect(self._reset_canvas)
        btn_reset.setStyleSheet(self._btn_style("#457b9d"))
        layout.addWidget(btn_reset)

        btn_retrain = QPushButton("Retrain Colours")
        btn_retrain.clicked.connect(self._go_training)
        btn_retrain.setStyleSheet(self._btn_style("#e76f51"))
        layout.addWidget(btn_retrain)

        if self.mock:
            sep4 = QFrame()
            sep4.setFrameShape(QFrame.HLine)
            sep4.setStyleSheet("color: #333;")
            layout.addWidget(sep4)
            layout.addWidget(self._section_label("Mock Controls"))
            self._mock_controls = self._build_mock_controls()
            layout.addWidget(self._mock_controls)

        layout.addStretch()
        return panel

    def _build_mock_controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # SSVEP simulation buttons
        v.addWidget(self._section_label("Simulate stare at:"))
        for region in config.CANVAS_REGIONS:
            btn = QPushButton(region["label"])
            btn.setCheckable(True)
            freq = region["ssvep_freq"]
            btn.clicked.connect(
                lambda checked, f=freq: self._mock_set_ssvep(f if checked else None)
            )
            btn.setStyleSheet(self._btn_style("#2d6a4f", small=True))
            v.addWidget(btn)

        # Colour simulation buttons
        v.addWidget(self._section_label("Simulate think of:"))
        for i, c in enumerate(config.TRAINING_COLORS[:config.N_TRAINING_COLORS]):
            btn = QPushButton(c["name"])
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda checked, idx=i: self._mock_set_color(idx if checked else None)
            )
            btn.setStyleSheet(self._btn_style(
                "#{:02x}{:02x}{:02x}".format(*c["rgb"]), small=True
            ))
            v.addWidget(btn)

        return w

    # ------------------------------------------------------------------
    # Splash helper
    # ------------------------------------------------------------------

    def _make_splash(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; font-size: 18px; padding: 40px;"
        )
        lbl.setFixedSize(config.CANVAS_W, config.CANVAS_H)
        return lbl

    def _make_ready_screen(self) -> QWidget:
        """Ready screen with a prominent Start Painting button."""
        w = QWidget()
        w.setFixedSize(config.CANVAS_W, config.CANVAS_H)
        w.setStyleSheet("background: #1a1a2e;")
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(24)

        icon_lbl = QLabel("✓")
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("color: #2ecc71; font-size: 64px; background: transparent;")
        layout.addWidget(icon_lbl)

        title = QLabel("Model Trained!")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #e0e0e0; font-size: 26px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        self._ready_acc_label = QLabel("")
        self._ready_acc_label.setAlignment(Qt.AlignCenter)
        self._ready_acc_label.setStyleSheet("color: #aaa; font-size: 14px; background: transparent;")
        layout.addWidget(self._ready_acc_label)

        hint = QLabel("Press  Space  or click the button below to start painting")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 14px; background: transparent;")
        layout.addWidget(hint)

        btn = QPushButton("▶  Start Painting")
        btn.setFixedSize(220, 52)
        btn.clicked.connect(self._go_painting)
        btn.setStyleSheet(
            "QPushButton { background: #e94560; color: white; border-radius: 8px; "
            "font-size: 17px; font-weight: bold; }"
            "QPushButton:hover { background: #c73652; }"
        )
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        return w

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        return lbl

    @staticmethod
    def _progress_style(color: str) -> str:
        return (
            f"QProgressBar {{ background: #0f3460; border-radius: 4px; height: 14px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )

    @staticmethod
    def _btn_style(color: str, small: bool = False) -> str:
        size = "11px" if small else "13px"
        return (
            f"QPushButton {{ background: {color}; color: white; border-radius: 5px; "
            f"padding: {'4px 6px' if small else '8px 10px'}; font-size: {size}; }}"
            f"QPushButton:hover {{ opacity: 0.85; }}"
            f"QPushButton:checked {{ border: 2px solid white; }}"
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _go_connecting(self) -> None:
        self._state = "CONNECTING"
        self._stack.setCurrentIndex(0)
        self._update_status("Connecting to headset…")

    def _go_training(self) -> None:
        self._state = "TRAINING"
        self._tick.stop()
        self._overlay.stop_all()
        self._stack.setCurrentIndex(1)
        self._training_screen.start_training()
        self._update_status("Training in progress…")

    def _go_ready(self) -> None:
        self._state = "READY"
        self._stack.setCurrentIndex(2)
        self._update_status("Model ready — press Space or click 'Start Painting'")
        self.setFocus()   # reclaim keyboard focus from child widgets

    def _go_painting(self) -> None:
        self._state = "PAINTING"
        self._stack.setCurrentIndex(3)
        self._overlay.start_all()
        self._stare_start_ms   = None
        self._stare_region_id  = None
        self._tick.start()
        self._update_status("Painting — stare at a region to select it")

    # ------------------------------------------------------------------
    # Headset connection
    # ------------------------------------------------------------------

    def _connect_headset(self) -> None:
        def _cb(chunk: np.ndarray) -> None:
            self._relay.chunk_received.emit(chunk)

        try:
            self._headset.start(_cb)
            # Small delay then transition to training
            QTimer.singleShot(1500, self._go_training)
        except RuntimeError as exc:
            self._page_connecting.setText(f"Connection error:\n{exc}")

    # ------------------------------------------------------------------
    # EEG chunk handler (main thread, via signal)
    # ------------------------------------------------------------------

    def _on_chunk_main_thread(self, chunk: np.ndarray) -> None:
        # Feed both detectors regardless of state
        self._ssvep.add_chunk(chunk)
        self._classifier.add_chunk(chunk)

        # Feed training screen during training
        if self._state == "TRAINING":
            self._training_screen.add_chunk(chunk)

    # ------------------------------------------------------------------
    # Training complete
    # ------------------------------------------------------------------

    def _on_training_complete(
        self, epochs: list, labels: list
    ) -> None:
        if not epochs:
            self._go_ready()
            return
        acc = self._classifier.train(epochs, labels)
        self._ready_acc_label.setText(f"Classifier accuracy (LOO): {acc:.1%}")
        self._update_status(f"Classifier trained — LOO accuracy: {acc:.1%}")
        QTimer.singleShot(1200, self._go_ready)

    # ------------------------------------------------------------------
    # BCI inference tick (100 ms)
    # ------------------------------------------------------------------

    def _bci_tick(self) -> None:
        self._update_color_from_classifier()
        self._update_stare_from_ssvep()

    def _update_color_from_classifier(self) -> None:
        result = self._classifier.predict()
        if result is None:
            return
        color_idx, confidence = result
        self._current_color_idx = color_idx
        colors = config.TRAINING_COLORS[:config.N_TRAINING_COLORS]
        if color_idx < len(colors):
            color_info = colors[color_idx]
            qc  = QColor(*color_info["rgb"])
            self._color_display.setStyleSheet(
                f"background: rgb({color_info['rgb'][0]},{color_info['rgb'][1]},"
                f"{color_info['rgb'][2]}); border-radius: 6px; color: white; font-size: 13px;"
            )
            self._color_display.setText(color_info["name"])
            self._color_confidence.setValue(int(confidence * 100))

    def _update_stare_from_ssvep(self) -> None:
        detected_freq = self._ssvep.detect()
        detected_region = _FREQ_TO_REGION.get(detected_freq) if detected_freq else None

        now_ms = self._elapsed_ms()

        if detected_region is None:
            # No signal — reset stare tracker
            self._stare_start_ms  = None
            self._stare_region_id = None
            self._stare_bar.setValue(0)
            self._stare_label.setText("—")
            self._canvas.set_highlight(None)
            return

        if detected_region != self._stare_region_id:
            # New region — reset timer
            self._stare_region_id = detected_region
            self._stare_start_ms  = now_ms

        region_label = config.CANVAS_REGIONS[detected_region]["label"]
        elapsed_sec  = (now_ms - (self._stare_start_ms or now_ms)) / 1000.0
        progress     = min(elapsed_sec / config.STARE_CONFIRM_SEC, 1.0)

        self._stare_label.setText(region_label)
        self._stare_bar.setValue(int(progress * 100))
        self._canvas.set_highlight(detected_region)

        if progress >= 1.0:
            self._confirm_region(detected_region)

    def _confirm_region(self, region_id: int) -> None:
        """Apply current colour to region; reset stare tracker."""
        color = self._get_current_qcolor()
        self._canvas.color_region(region_id, color)

        # Update status
        region_label = config.CANVAS_REGIONS[region_id]["label"]
        color_name   = (
            config.TRAINING_COLORS[self._current_color_idx]["name"]
            if self._current_color_idx is not None
            else "default"
        )
        self._update_status(f"Filled: {region_label} → {color_name}")

        # Reset stare
        self._stare_start_ms  = None
        self._stare_region_id = None
        self._stare_bar.setValue(0)
        self._canvas.set_highlight(None)

    def _get_current_qcolor(self) -> QColor:
        if self._current_color_idx is not None:
            colors = config.TRAINING_COLORS[:config.N_TRAINING_COLORS]
            if self._current_color_idx < len(colors):
                return QColor(*colors[self._current_color_idx]["rgb"])
        return QColor(128, 128, 128)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_status(self, msg: str) -> None:
        self._status_bar.setText(f"  {msg}")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _reset_canvas(self) -> None:
        self._canvas.reset()
        self._update_status("Canvas reset")

    # ------------------------------------------------------------------
    # Debug click (demo mode)
    # ------------------------------------------------------------------

    def _debug_fill_region(self, region_id: int) -> None:
        color = self._get_current_qcolor()
        self._canvas.color_region(region_id, color)

    # ------------------------------------------------------------------
    # Mock controls
    # ------------------------------------------------------------------

    def _mock_set_ssvep(self, freq: Optional[float]) -> None:
        self._headset.mock_ssvep_freq = freq

    def _mock_set_color(self, idx: Optional[int]) -> None:
        self._headset.mock_color_state = idx

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and self._state == "READY":
            self._go_painting()
        elif event.key() == Qt.Key_R and self._state == "PAINTING":
            self._reset_canvas()
        elif event.key() == Qt.Key_T:
            self._go_training()
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._tick.stop()
        self._overlay.stop_all()
        self._headset.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms() -> int:
        import time
        return int(time.monotonic() * 1000)
