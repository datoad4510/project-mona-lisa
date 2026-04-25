"""SSVEP flickering-stimulus overlay.

Each canvas region is covered by a semi-transparent white overlay that
flickers on/off at the region's target frequency.  The overlay is a child
widget of the canvas so it auto-sizes/moves with it.

Timing approach
---------------
A single master QTimer fires every MASTER_INTERVAL_MS (≈8 ms, ~120 fps).
On each tick every FlickerWidget repaints itself.  Inside paintEvent the
on/off state is computed from ``time.monotonic()`` relative to the start
time and the ideal cycle period.  This decouples visual state from QTimer
granularity (~15 ms on Windows) and gives accurate frequencies at all
supported rates (8, 10, 12, 15 Hz).
"""

from __future__ import annotations

import time
from typing import Dict

from PyQt5.QtCore    import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui     import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QWidget

import config

MASTER_INTERVAL_MS = 8   # repaint budget ≈ 120 fps — well above 2× Nyquist of 15 Hz


class _FlickerWidget(QWidget):
    """Full-canvas overlay for one circle; paints only inside the ellipse."""

    def __init__(self, region: dict, parent: QWidget):
        super().__init__(parent)
        self._region  = region
        self._period  = 1.0 / region["ssvep_freq"]   # full cycle in seconds
        self._active  = False
        self._t0: float = 0.0

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, parent.width(), parent.height())

    def start(self) -> None:
        self._t0     = time.monotonic()
        self._active = True
        self.update()

    def stop(self) -> None:
        self._active = False
        self.update()

    def paintEvent(self, _event) -> None:
        if not self._active:
            return

        # Compute phase within current cycle (0.0 – 1.0).
        # ON for the first half, OFF for the second half → square wave.
        elapsed   = time.monotonic() - self._t0
        phase     = (elapsed % self._period) / self._period
        if phase >= 0.5:
            return   # OFF half — nothing to draw

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, config.FLICKER_OVERLAY_ALPHA)))
        x0, y0, x1, y1 = self._region["coords"]
        w, h = self.width(), self.height()
        painter.drawEllipse(QRectF(
            QPointF(x0 * w, y0 * h),
            QPointF(x1 * w, y1 * h),
        ))


# ---------------------------------------------------------------------------

class SSVEPOverlay:
    """Manages all flicker widgets over a PaintCanvas."""

    def __init__(self, canvas: QWidget):
        self._widgets: Dict[int, _FlickerWidget] = {}
        for region in config.CANVAS_REGIONS:
            w = _FlickerWidget(region, canvas)
            w.show()
            self._widgets[region["id"]] = w

        # Single master timer drives all repaints
        self._master = QTimer(canvas)
        self._master.setInterval(MASTER_INTERVAL_MS)
        self._master.timeout.connect(self._tick)

    def _tick(self) -> None:
        for w in self._widgets.values():
            if w._active:
                w.update()

    def start_all(self) -> None:
        for w in self._widgets.values():
            w.start()
        self._master.start()

    def stop_all(self) -> None:
        self._master.stop()
        for w in self._widgets.values():
            w.stop()

    def start_region(self, region_id: int) -> None:
        if region_id in self._widgets:
            self._widgets[region_id].start()

    def stop_region(self, region_id: int) -> None:
        if region_id in self._widgets:
            self._widgets[region_id].stop()
