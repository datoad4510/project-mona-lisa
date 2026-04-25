"""SSVEP flickering-stimulus overlay.

Each canvas region is covered by a semi-transparent white overlay that
flickers on and off at the region's target frequency.  The overlay is a
child widget of the canvas so it auto-sizes/moves with it.
"""

from __future__ import annotations

from typing import Dict

from PyQt5.QtCore    import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui     import QColor, QPainter, QBrush, QPainterPath
from PyQt5.QtWidgets import QWidget

import config


class _FlickerWidget(QWidget):
    """Semi-transparent overlay over one circle that flickers at *freq* Hz."""

    def __init__(self, region: dict, parent: QWidget):
        super().__init__(parent)
        self._region    = region
        self._visible_f = True
        self._active    = False

        freq_hz     = region["ssvep_freq"]
        half_period = int(1000 / (2 * freq_hz))

        self._timer = QTimer(self)
        self._timer.setInterval(half_period)
        self._timer.timeout.connect(self._toggle)

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, parent.width(), parent.height())

    def start(self) -> None:
        self._active    = True
        self._visible_f = True
        self._timer.start()
        self.update()

    def stop(self) -> None:
        self._active = False
        self._timer.stop()
        self.update()

    def _toggle(self) -> None:
        self._visible_f = not self._visible_f
        self.update()

    def paintEvent(self, _event) -> None:
        if not self._active or not self._visible_f:
            return
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

    def start_all(self) -> None:
        for w in self._widgets.values():
            w.start()

    def stop_all(self) -> None:
        for w in self._widgets.values():
            w.stop()

    def start_region(self, region_id: int) -> None:
        if region_id in self._widgets:
            self._widgets[region_id].start()

    def stop_region(self, region_id: int) -> None:
        if region_id in self._widgets:
            self._widgets[region_id].stop()
