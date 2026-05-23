"""
MinimapWidget — top-down tactical pitch view.

Draws an accurate football pitch (105 × 68 m) using QPainter and overlays
coloured player/ball dots from world-coordinate positions emitted by the
HomographyProcessor pipeline.

Object-id → colour mapping (matches detection_utils.LABELS):
  0  Player-L   blue
  1  Player-R   red
  2  GK-L       green
  3  GK-R       orange
  4  Ball       white
  5+ Officials  grey
"""

from __future__ import annotations

from typing import List, Dict, Any

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRectF, QPointF, QSize
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient,
)

# ── Pitch dimensions (metres) ──────────────────────────────────────────────────
FIELD_W = 105.0   # length (goal-line to goal-line)
FIELD_H = 68.0    # width  (touch-line to touch-line)

# ── Detection-class colours (RGB) ─────────────────────────────────────────────
_TEAM_COLORS: Dict[int, QColor] = {
    0: QColor(29,  78, 216),    # Player-L  — team-A blue
    1: QColor(220, 38,  38),    # Player-R  — team-B red
    2: QColor(16,  185, 129),   # GK-L      — green
    3: QColor(245, 158,  11),   # GK-R      — amber
    4: QColor(255, 255, 255),   # Ball      — white
}
_DEFAULT_COLOR = QColor(148, 163, 184)   # slate-400 for refs / staff

_PITCH_GREEN   = QColor(22, 101, 52)     # dark green grass
_PITCH_GREEN2  = QColor(20,  91, 47)     # stripe alternate
_LINE_COLOR    = QColor(255, 255, 255, 220)
_GOAL_COLOR    = QColor(0,   230, 255, 230)   # cyan goals
_SHADOW        = QColor(0, 0, 0, 100)


class MinimapWidget(QWidget):
    """
    Top-down football pitch widget.

    Call ``update_positions(positions)`` with a list of dicts, each containing:
      - ``object_id``        int  (YOLO class id)
      - ``world_x_meters``   float
      - ``world_y_meters``   float

    The widget re-renders automatically on each call.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._positions: List[Dict[str, Any]] = []
        self._confidence: float = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(220, 144)

    # ── Public slots ──────────────────────────────────────────────────────────

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Receive world-coordinate positions and repaint."""
        self._positions = positions
        self.update()

    def update_confidence(self, confidence: float) -> None:
        """Store the latest homography confidence (0-1) for overlay display."""
        self._confidence = confidence
        self.update()

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def sizeHint(self) -> QSize:
        return QSize(360, 234)   # ~105:68 aspect ratio

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        field_rect = self._compute_field_rect(w, h)

        self._draw_pitch(painter, field_rect)
        self._draw_players(painter, field_rect)
        self._draw_confidence(painter, w, h)

        painter.end()

    # ── Private: layout ───────────────────────────────────────────────────────

    def _compute_field_rect(self, w: int, h: int) -> QRectF:
        """Return the largest rect that fits the pitch aspect ratio with 8 px margin."""
        margin = 8
        aw, ah = w - 2 * margin, h - 2 * margin
        aspect = FIELD_W / FIELD_H
        if aw / ah > aspect:
            ph = ah
            pw = ph * aspect
        else:
            pw = aw
            ph = pw / aspect
        ox = (w - pw) / 2
        oy = (h - ph) / 2
        return QRectF(ox, oy, pw, ph)

    def _fp(self, wx: float, wy: float, rect: QRectF) -> QPointF:
        """World metres → widget pixels."""
        px = rect.left() + (wx / FIELD_W) * rect.width()
        py = rect.top()  + (wy / FIELD_H) * rect.height()
        return QPointF(px, py)

    def _fp_rect(
        self, wx1: float, wy1: float, wx2: float, wy2: float, rect: QRectF
    ) -> QRectF:
        """Two world-space corners → QRectF in widget space."""
        p1 = self._fp(wx1, wy1, rect)
        p2 = self._fp(wx2, wy2, rect)
        return QRectF(p1.x(), p1.y(), p2.x() - p1.x(), p2.y() - p1.y())

    # ── Private: drawing ──────────────────────────────────────────────────────

    def _draw_pitch(self, painter: QPainter, rect: QRectF) -> None:
        # ── Grass with alternating stripes ────────────────────────────────────
        painter.setClipRect(rect)
        stripe_w = rect.width() / 10
        for i in range(10):
            c = _PITCH_GREEN if i % 2 == 0 else _PITCH_GREEN2
            painter.fillRect(
                QRectF(rect.left() + i * stripe_w, rect.top(), stripe_w, rect.height()), c
            )
        painter.setClipping(False)

        # ── White pitch lines ─────────────────────────────────────────────────
        pen = QPen(_LINE_COLOR, 1.2)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        fp = lambda wx, wy: self._fp(wx, wy, rect)          # noqa: E731
        fpr = lambda x1, y1, x2, y2: self._fp_rect(x1, y1, x2, y2, rect)  # noqa: E731

        # Outer boundary
        painter.drawRect(rect.adjusted(0, 0, -0.5, -0.5))

        # Centre line
        painter.drawLine(fp(52.5, 0), fp(52.5, 68))

        # Centre circle  (r = 9.15 m)
        cx = fp(52.5, 34)
        rx = (9.15 / FIELD_W) * rect.width()
        ry = (9.15 / FIELD_H) * rect.height()
        painter.drawEllipse(cx, rx, ry)

        # Centre spot
        painter.setBrush(QBrush(_LINE_COLOR))
        painter.drawEllipse(cx, 1.5, 1.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Penalty areas — left
        painter.drawRect(fpr(0, 13.84, 16.5, 54.16))   # big box
        painter.drawRect(fpr(0, 24.84,  5.5, 43.16))   # small box
        # Penalty areas — right
        painter.drawRect(fpr(88.5, 13.84, 105.0, 54.16))
        painter.drawRect(fpr(99.5, 24.84, 105.0, 43.16))

        # Penalty spots
        painter.setBrush(QBrush(_LINE_COLOR))
        painter.drawEllipse(fp(11.0, 34.0), 1.5, 1.5)
        painter.drawEllipse(fp(94.0, 34.0), 1.5, 1.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Penalty arcs (the D shapes outside the penalty area)
        self._draw_penalty_arc(painter, rect, left=True)
        self._draw_penalty_arc(painter, rect, left=False)

        # ── Goals ─────────────────────────────────────────────────────────────
        goal_pen = QPen(_GOAL_COLOR, 2.0)
        painter.setPen(goal_pen)
        # Left goal
        painter.drawLine(fp(0, 30.34), fp(-1.5, 30.34))
        painter.drawLine(fp(-1.5, 30.34), fp(-1.5, 37.66))
        painter.drawLine(fp(-1.5, 37.66), fp(0, 37.66))
        # Right goal
        painter.drawLine(fp(105, 30.34), fp(106.5, 30.34))
        painter.drawLine(fp(106.5, 30.34), fp(106.5, 37.66))
        painter.drawLine(fp(106.5, 37.66), fp(105, 37.66))

    def _draw_penalty_arc(
        self, painter: QPainter, rect: QRectF, left: bool
    ) -> None:
        """Draw the D arc outside a penalty area."""
        # Penalty spot world position
        spot_x = 11.0 if left else 94.0
        spot_y = 34.0
        # Radius = 9.15 m
        rx = (9.15 / FIELD_W) * rect.width()
        ry = (9.15 / FIELD_H) * rect.height()

        cx, cy = self._fp(spot_x, spot_y, rect).x(), self._fp(spot_x, spot_y, rect).y()

        # Draw a full ellipse as a path then clip to the part outside the box
        path = QPainterPath()
        arc_rect = QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)
        path.addEllipse(arc_rect)

        # Box boundary in widget x
        box_x = self._fp(16.5 if left else 88.5, 0, rect).x()

        # Clip rect: the half outside the box
        if left:
            clip = QRectF(box_x, rect.top() - 20, rect.right() - box_x + 20, rect.height() + 40)
        else:
            clip = QRectF(rect.left() - 20, rect.top() - 20, box_x - rect.left() + 20, rect.height() + 40)

        old_clip = painter.clipRegion()
        painter.setClipRect(clip)
        painter.drawPath(path)
        painter.setClipping(False)

    def _draw_players(self, painter: QPainter, rect: QRectF) -> None:
        """Render each player/ball as a circle."""
        dot_r = max(4.0, rect.width() * 0.024)  # scales with widget size

        for pos in self._positions:
            wx = float(pos.get("world_x_meters", -999))
            wy = float(pos.get("world_y_meters", -999))

            # Skip obviously out-of-bounds positions
            if not (-5 <= wx <= FIELD_W + 5 and -5 <= wy <= FIELD_H + 5):
                continue

            oid = int(pos.get("object_id", -1))
            color = _TEAM_COLORS.get(oid, _DEFAULT_COLOR)

            pt = self._fp(max(0, min(wx, FIELD_W)), max(0, min(wy, FIELD_H)), rect)

            # Drop shadow
            painter.setBrush(QBrush(_SHADOW))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(pt.x() + 1, pt.y() + 1), dot_r, dot_r)

            # Coloured dot
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.0))
            painter.drawEllipse(pt, dot_r, dot_r)

            # Ball gets an extra highlight ring
            if oid == 4:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 220, 0, 200), 1.5))
                painter.drawEllipse(pt, dot_r + 2, dot_r + 2)

    def _draw_confidence(self, painter: QPainter, w: int, h: int) -> None:
        """Draw a small homography-confidence badge in the bottom-right corner."""
        if self._confidence <= 0:
            return

        text = f"H: {self._confidence:.0%}"
        pct = self._confidence

        # Color: red → yellow → green
        if pct < 0.4:
            color = QColor(220, 38, 38)
        elif pct < 0.7:
            color = QColor(245, 158, 11)
        else:
            color = QColor(16, 185, 129)

        painter.setFont(QFont("monospace", 8))
        painter.setPen(QPen(color))
        painter.drawText(
            QRectF(w - 60, h - 18, 56, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            text,
        )
