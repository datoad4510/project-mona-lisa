"""Paint canvas widget.

Four distinct circles arranged in a 2 × 2 grid, each colourable via BCI.
Circles never touch or overlap.  Each circle is annotated with its SSVEP
stimulus frequency so the participant knows which one to stare at.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore    import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui     import (
    QColor, QPainter, QPen, QBrush, QPainterPath, QFont,
)
from PyQt5.QtWidgets import QWidget

import config


Region = dict


class PaintCanvas(QWidget):
    """Four-circle colourable canvas."""

    region_clicked = pyqtSignal(int)

    def __init__(self, parent=None, debug_click: bool = False):
        super().__init__(parent)
        self.setFixedSize(config.CANVAS_W, config.CANVAS_H)
        self.debug_click = debug_click

        self._fill: Dict[int, QColor] = {
            r["id"]: QColor(*r["default_color"])
            for r in config.CANVAS_REGIONS
        }
        self._highlighted: Optional[int] = None
        self._confirmed:   Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def color_region(self, region_id: int, color: QColor) -> None:
        self._fill[region_id] = color
        self._confirmed = region_id
        self.update()

    def set_highlight(self, region_id: Optional[int]) -> None:
        if self._highlighted != region_id:
            self._highlighted = region_id
            self.update()

    def reset(self) -> None:
        for r in config.CANVAS_REGIONS:
            self._fill[r["id"]] = QColor(*r["default_color"])
        self._highlighted = None
        self._confirmed   = None
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QBrush(QColor(28, 28, 40)))

        for region in config.CANVAS_REGIONS:
            rid  = region["id"]
            path = self._build_path(region)

            # Fill
            painter.fillPath(path, QBrush(self._fill[rid]))

            # Highlight pulse (white overlay) while user is staring
            if rid == self._highlighted:
                painter.fillPath(path, QBrush(QColor(255, 255, 255, 70)))

            # Border — thicker + lighter when this region was last confirmed
            if rid == self._confirmed:
                painter.setPen(QPen(QColor(255, 255, 255), 4))
            elif rid == self._highlighted:
                painter.setPen(QPen(QColor(200, 200, 255), 3))
            else:
                painter.setPen(QPen(QColor(100, 100, 120), 2))
            painter.drawPath(path)

            # Labels inside the circle
            self._draw_labels(painter, region, path)

        # Fixed black fixation dot at the centre of the canvas.
        # Radius = 1/4 of the average circle radius.
        r0 = config.CANVAS_REGIONS[0]
        x0, y0, x1, y1 = r0["coords"]
        circle_r_px = ((x1 - x0) * self.width() + (y1 - y0) * self.height()) / 4.0
        dot_r = circle_r_px / 4.0
        cx = self.width()  / 2.0
        cy = self.height() / 2.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.drawEllipse(QRectF(cx - dot_r, cy - dot_r, dot_r * 2, dot_r * 2))

    def mousePressEvent(self, event) -> None:
        if not self.debug_click:
            return
        pos = QPointF(event.pos())
        for region in config.CANVAS_REGIONS:
            if self._build_path(region).contains(pos):
                self.region_clicked.emit(region["id"])
                break

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _scale(self, nx: float, ny: float) -> QPointF:
        return QPointF(nx * self.width(), ny * self.height())

    def _build_path(self, region: Region) -> QPainterPath:
        x0, y0, x1, y1 = region["coords"]
        path = QPainterPath()
        path.addEllipse(QRectF(self._scale(x0, y0), self._scale(x1, y1)))
        return path

    def _draw_labels(
        self, painter: QPainter, region: Region, path: QPainterPath
    ) -> None:
        bbox = path.boundingRect()
        cx   = bbox.center().x()
        cy   = bbox.center().y()

        painter.save()

        # Frequency label (large, top half of circle)
        freq_font = QFont("Arial", 18, QFont.Bold)
        painter.setFont(freq_font)
        painter.setPen(QPen(QColor(255, 255, 255, 210)))
        freq_rect = QRectF(bbox.left(), cy - bbox.height() * 0.30,
                           bbox.width(), bbox.height() * 0.40)
        painter.drawText(freq_rect, Qt.AlignCenter,
                         f"{region['ssvep_freq']:.4g} Hz")

        # Name label (smaller, lower)
        name_font = QFont("Arial", 10)
        painter.setFont(name_font)
        painter.setPen(QPen(QColor(200, 200, 200, 160)))
        name_rect = QRectF(bbox.left(), cy + bbox.height() * 0.12,
                           bbox.width(), bbox.height() * 0.28)
        painter.drawText(name_rect, Qt.AlignCenter, region["label"])

        painter.restore()
