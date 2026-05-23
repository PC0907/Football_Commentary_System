"""
Football Analysis System — main UI
===================================
Broadcast-style dark interface with:
  • Live video preview during processing
  • Real-time tactical minimap (homography-based)
  • Scrolling live commentary feed
  • Match-stats panel (possession bar, shots, passes, fouls)
  • Input / output video players for pre/post review
"""

from __future__ import annotations

import sys
import os
import cv2
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QFrame, QSizePolicy,
    QComboBox, QStackedWidget, QSlider,
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSlot
from PyQt6.QtGui import (
    QFont, QPixmap, QImage, QColor, QPainter, QBrush,
    QPen, QLinearGradient, QIcon,
)

from components.video_player    import VideoPlayer
from components.team_sheet      import TeamSheetDialog
from components.processor       import VideoProcessor
from components.minimap_widget  import MinimapWidget
from themes                     import ThemeManager


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="app.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Colour tokens (broadcast dark theme) ─────────────────────────────────────
BG          = "#0B0F1A"
PANEL       = "#141927"
CARD        = "#1C2438"
BORDER      = "#2A3347"
ACCENT      = "#3B82F6"
TEAM_A      = "#1D4ED8"
TEAM_B      = "#DC2626"
TEXT        = "#F1F5F9"
TEXT_DIM    = "#94A3B8"
TEXT_MUTED  = "#475569"
SUCCESS     = "#10B981"
WARNING     = "#F59E0B"
DANGER      = "#EF4444"

# ── Stylesheet ────────────────────────────────────────────────────────────────
_QSS = f"""
/* ── Global ──────────────────────────────────────────────────────── */
* {{
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: {TEXT};
}}
QMainWindow, QWidget {{
    background-color: {BG};
}}

/* ── Panels / cards ──────────────────────────────────────────────── */
QFrame#panel {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ── Section labels ──────────────────────────────────────────────── */
QLabel#section_label {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: #2563EB;
}}
QPushButton:disabled {{
    background-color: {PANEL};
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}
QPushButton#primary {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton#primary:hover {{
    background-color: #2563EB;
}}
QPushButton#danger {{
    background-color: #7F1D1D;
    border-color: {DANGER};
}}
QPushButton#danger:hover {{
    background-color: {DANGER};
}}

/* ── Progress bar ────────────────────────────────────────────────── */
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {CARD};
    height: 6px;
    text-align: center;
    font-size: 11px;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

/* ── Commentary feed ─────────────────────────────────────────────── */
QTextEdit#commentary {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px;
    font-size: 12px;
    line-height: 1.6;
    color: {TEXT};
}}

/* ── Combo box ───────────────────────────────────────────────────── */
QComboBox {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    color: {TEXT};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

/* ── Slider ──────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {CARD};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: none;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper widgets
# ═══════════════════════════════════════════════════════════════════════════════

class LiveFrameDisplay(QWidget):
    """
    Simple widget that shows BGR numpy frames as they arrive from
    VideoProcessor.frame_ready.  Shown during processing instead of the
    VideoPlayer.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._label)

        self._info = QLabel("Waiting for first frame…", self)
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._info)

        # Blank placeholder
        blank = QPixmap(640, 360)
        blank.fill(QColor(BG))
        self._label.setPixmap(blank)

    @pyqtSlot(object)
    def show_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._label.width(), self._label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pix)
        self._info.hide()

    def set_info(self, text: str) -> None:
        self._info.setText(text)
        self._info.show()


class PossessionBar(QWidget):
    """Two-colour horizontal bar showing A% vs B%."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._a = 50
        self._b = 50
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_possession(self, team_a: int, team_b: int) -> None:
        total = team_a + team_b
        if total > 0:
            self._a = int(100 * team_a / total)
            self._b = 100 - self._a
        else:
            self._a = self._b = 50
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        split = int(w * self._a / 100)

        # Team-A side
        p.setBrush(QBrush(QColor(TEAM_A)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, split, h, 4, 4)

        # Team-B side
        p.setBrush(QBrush(QColor(TEAM_B)))
        p.drawRoundedRect(split, 0, w - split, h, 4, 4)

        # Labels
        p.setPen(QPen(QColor(TEXT)))
        font = QFont("monospace", 9, QFont.Weight.Bold)
        p.setFont(font)
        if split > 32:
            p.drawText(8, 0, split - 10, h, Qt.AlignmentFlag.AlignVCenter, f"{self._a}%")
        if w - split > 32:
            p.drawText(split + 4, 0, w - split - 8, h,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       f"{self._b}%")
        p.end()


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("section_label")
    lbl.setStyleSheet(
        f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700;"
        f" letter-spacing: 1.5px; text-transform: uppercase;"
    )
    return lbl


def _stat_row(label: str, val_a: str, val_b: str,
              color_a: str = TEAM_A, color_b: str = TEAM_B) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    def _mk(text, color, align):
        l = QLabel(text)
        l.setFont(QFont("monospace", 11, QFont.Weight.Bold))
        l.setStyleSheet(f"color: {color};")
        l.setAlignment(align)
        l.setFixedWidth(36)
        return l
    row.addWidget(_mk(val_a, color_a, Qt.AlignmentFlag.AlignRight))
    mid = QLabel(label)
    mid.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
    mid.setAlignment(Qt.AlignmentFlag.AlignCenter)
    row.addWidget(mid, stretch=1)
    row.addWidget(_mk(val_b, color_b, Qt.AlignmentFlag.AlignLeft))
    return row


def _panel(contents: QWidget) -> QFrame:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    layout.addWidget(contents)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main application window
# ═══════════════════════════════════════════════════════════════════════════════

class FootballAnalysisApp(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.theme_manager      = ThemeManager("Dark")
        self.input_video_path: str | None = None
        self.team_sheet_data    = self._load_team_sheet()
        self.processor: VideoProcessor | None = None
        self._processing        = False

        self._init_ui()
        self._connect_controls()
        log.info("Application started")

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        self.setWindowTitle("⚽  Football Analysis System")
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(_QSS)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([820, 380])
        root_layout.addWidget(splitter, stretch=1)

        root_layout.addWidget(self._build_commentary_panel())
        root_layout.addWidget(self._build_control_bar())

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("panel")
        header.setFixedHeight(60)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)

        # Logo + title
        title = QLabel("⚽  FOOTBALL ANALYSIS SYSTEM")
        title.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; letter-spacing: 1px;")
        layout.addWidget(title)

        layout.addStretch()

        # Score bug ── Team A [0] – [0] Team B
        self._lbl_team_a = QLabel(self._team_name("team_a"))
        self._lbl_team_a.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        self._lbl_team_a.setStyleSheet(f"color: {TEAM_A};")

        self._lbl_score_a = QLabel("0")
        self._lbl_score_b = QLabel("0")
        for s in (self._lbl_score_a, self._lbl_score_b):
            s.setFont(QFont("monospace", 18, QFont.Weight.Bold))
            s.setStyleSheet(f"color: {TEXT};")
            s.setFixedWidth(28)
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sep = QLabel("—")
        sep.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 16px;")

        self._lbl_team_b = QLabel(self._team_name("team_b"))
        self._lbl_team_b.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        self._lbl_team_b.setStyleSheet(f"color: {TEAM_B};")

        for w in (self._lbl_team_a, self._lbl_score_a, sep,
                  self._lbl_score_b, self._lbl_team_b):
            layout.addWidget(w)

        layout.addStretch()

        # Theme selector
        theme_lbl = QLabel("Theme")
        theme_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light", "Blue", "Green"])
        self._theme_combo.setCurrentText("Dark")
        self._theme_combo.setFixedWidth(90)
        layout.addWidget(theme_lbl)
        layout.addWidget(self._theme_combo)

        return header

    # ── Left: video panel ─────────────────────────────────────────────────────

    def _build_video_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(_section_label("VIDEO FEED"))

        # Stack: 0 = input player, 1 = live processing display, 2 = output player
        self._video_stack = QStackedWidget()

        self._input_player  = VideoPlayer("Input")
        self._live_display  = LiveFrameDisplay()
        self._output_player = VideoPlayer("Output")

        self._video_stack.addWidget(self._input_player)   # 0
        self._video_stack.addWidget(self._live_display)   # 1
        self._video_stack.addWidget(self._output_player)  # 2

        layout.addWidget(self._video_stack, stretch=1)
        return panel

    # ── Right panel (minimap + stats) ─────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_minimap_section(), stretch=2)
        layout.addWidget(self._build_stats_section(),   stretch=3)

        return panel

    def _build_minimap_section(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_label("TACTICAL MAP"))
        hdr.addStretch()
        self._hconf_label = QLabel("Homography: –")
        self._hconf_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        hdr.addWidget(self._hconf_label)
        lay.addLayout(hdr)

        self._minimap = MinimapWidget()
        lay.addWidget(self._minimap, stretch=1)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(10)
        for color, name in [(TEAM_A, "Team A"), (TEAM_B, "Team B"),
                            ("#10B981", "GK"), ("#FFFFFF", "Ball")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
            legend.addWidget(dot)
            legend.addWidget(lbl)
        legend.addStretch()
        lay.addLayout(legend)

        return card

    def _build_stats_section(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lay.addWidget(_section_label("MATCH STATS"))

        # Team-name row
        names = QHBoxLayout()
        self._stats_name_a = QLabel(self._team_name("team_a"))
        self._stats_name_b = QLabel(self._team_name("team_b"))
        for lbl, color in ((self._stats_name_a, TEAM_A), (self._stats_name_b, TEAM_B)):
            lbl.setFont(QFont("Inter", 10, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color: {color};")
        self._stats_name_a.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._stats_name_b.setAlignment(Qt.AlignmentFlag.AlignLeft)
        names.addWidget(self._stats_name_a, stretch=1)
        names.addSpacing(8)
        names.addWidget(self._stats_name_b, stretch=1)
        lay.addLayout(names)

        # Possession bar
        lay.addWidget(_section_label("POSSESSION"))
        self._possession_bar = PossessionBar()
        lay.addWidget(self._possession_bar)

        lay.addWidget(self.__divider())

        # Stat rows
        self._stat_shots_a  = QLabel("0"); self._stat_shots_b  = QLabel("0")
        self._stat_passes_a = QLabel("0"); self._stat_passes_b = QLabel("0")
        self._stat_fouls_a  = QLabel("0"); self._stat_fouls_b  = QLabel("0")

        for row_data in [
            ("SHOTS",  self._stat_shots_a,  self._stat_shots_b),
            ("PASSES", self._stat_passes_a, self._stat_passes_b),
            ("FOULS",  self._stat_fouls_a,  self._stat_fouls_b),
        ]:
            lay.addLayout(self._stat_row_widget(*row_data))

        lay.addStretch()
        return card

    @staticmethod
    def _stat_row_widget(label: str, val_a: QLabel, val_b: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        for lbl, color, align in [
            (val_a, TEAM_A, Qt.AlignmentFlag.AlignRight),
            (val_b, TEAM_B, Qt.AlignmentFlag.AlignLeft),
        ]:
            lbl.setFont(QFont("monospace", 13, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color: {color};")
            lbl.setAlignment(align)
            lbl.setFixedWidth(38)

        mid = QLabel(label)
        mid.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        mid.setAlignment(Qt.AlignmentFlag.AlignCenter)

        row.addWidget(val_a)
        row.addWidget(mid, stretch=1)
        row.addWidget(val_b)
        return row

    @staticmethod
    def __divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    # ── Commentary panel ──────────────────────────────────────────────────────

    def _build_commentary_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("panel")
        card.setFixedHeight(148)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_label("📢  LIVE COMMENTARY"))
        hdr.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(54, 22)
        clear_btn.setStyleSheet(
            f"font-size: 10px; padding: 2px 6px; color: {TEXT_MUTED};"
            f" background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px;"
        )
        clear_btn.clicked.connect(self._clear_commentary)
        hdr.addWidget(clear_btn)
        lay.addLayout(hdr)

        self._commentary = QTextEdit()
        self._commentary.setObjectName("commentary")
        self._commentary.setReadOnly(True)
        self._commentary.setFont(QFont("monospace", 11))
        lay.addWidget(self._commentary, stretch=1)

        return card

    # ── Control bar ───────────────────────────────────────────────────────────

    def _build_control_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("panel")
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_upload  = QPushButton("⬆  Upload Video")
        self._btn_team    = QPushButton("👥  Team Sheet")
        self._btn_process = QPushButton("▶  Process")
        self._btn_process.setObjectName("primary")
        self._btn_export  = QPushButton("💾  Export")
        self._btn_cancel  = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_playpause = QPushButton("⏵  Play / Pause")

        self._btn_process.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        self._btn_playpause.setEnabled(False)

        for btn in (self._btn_upload, self._btn_team, self._btn_process,
                    self._btn_playpause, self._btn_export, self._btn_cancel):
            btn.setMinimumHeight(36)
            btn_row.addWidget(btn)

        lay.addLayout(btn_row)

        # Progress row
        prog_row = QHBoxLayout()
        prog_row.setSpacing(10)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)

        self._status = QLabel("Ready")
        self._status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self._status.setFixedWidth(300)

        prog_row.addWidget(self._progress, stretch=1)
        prog_row.addWidget(self._status)
        lay.addLayout(prog_row)

        return bar

    # ══════════════════════════════════════════════════════════════════════════
    # Signal connections
    # ══════════════════════════════════════════════════════════════════════════

    def _connect_controls(self) -> None:
        self._btn_upload.clicked.connect(self._upload_video)
        self._btn_team.clicked.connect(self._open_team_sheet)
        self._btn_process.clicked.connect(self._start_processing)
        self._btn_export.clicked.connect(self._export_video)
        self._btn_cancel.clicked.connect(self._cancel_processing)
        self._btn_playpause.clicked.connect(self._toggle_play)
        self._theme_combo.currentTextChanged.connect(self._change_theme)

    # ══════════════════════════════════════════════════════════════════════════
    # Action handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _upload_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v)"
        )
        if not path:
            return
        self.input_video_path = path
        self._input_player.load_video(path)
        self._video_stack.setCurrentIndex(0)
        self._btn_playpause.setEnabled(True)
        self._status.setText(f"Loaded: {Path(path).name}")
        if self.team_sheet_data:
            self._btn_process.setEnabled(True)
        log.info("Video loaded: %s", path)

    def _open_team_sheet(self) -> None:
        dlg = TeamSheetDialog(self.team_sheet_data, self.theme_manager, self)
        if dlg.exec():
            self.team_sheet_data = dlg.get_team_data()
            self._save_team_sheet(self.team_sheet_data)
            self._refresh_team_names()
            self._status.setText("Team sheet updated")
            if self.input_video_path:
                self._btn_process.setEnabled(True)

    def _start_processing(self) -> None:
        if not self.input_video_path or not self.team_sheet_data:
            QMessageBox.warning(self, "Missing data",
                                "Please load a video and team sheet first.")
            return

        self._processing = True
        self._btn_upload.setEnabled(False)
        self._btn_team.setEnabled(False)
        self._btn_process.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        # Switch to live-preview pane
        self._video_stack.setCurrentIndex(1)
        self._live_display.set_info("Initialising models…")
        self._commentary.clear()
        self._progress.setValue(0)
        self._status.setText("Processing…")

        self.processor = VideoProcessor(self.input_video_path, self.team_sheet_data)

        # Wire all signals
        self.processor.progress_updated.connect(self._on_progress)
        self.processor.processing_complete.connect(self._on_complete)
        self.processor.frame_ready.connect(self._live_display.show_frame)
        self.processor.minimap_updated.connect(self._minimap.update_positions)
        self.processor.homography_confidence.connect(self._on_hconf)
        self.processor.commentary_generated.connect(self._append_commentary)
        self.processor.stats_updated.connect(self._on_stats)

        self.processor.start()
        log.info("Processing started")

    def _cancel_processing(self) -> None:
        if self.processor and self.processor.isRunning():
            self.processor.canceled = True
            self._status.setText("Cancelling…")

    def _export_video(self) -> None:
        if not self.processor or not getattr(self.processor, "output_path", None):
            QMessageBox.warning(self, "No output", "No processed video to export.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "MP4 Video (*.mp4)"
        )
        if save_path:
            try:
                shutil.copy2(self.processor.output_path, save_path)
                self._status.setText(f"Exported → {Path(save_path).name}")
                QMessageBox.information(self, "Export OK",
                                        f"Saved to:\n{save_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

    def _toggle_play(self) -> None:
        idx = self._video_stack.currentIndex()
        if idx == 0:
            player = self._input_player
        elif idx == 2:
            player = self._output_player
        else:
            return
        if player.is_playing():
            player.pause()
        else:
            player.play()

    def _clear_commentary(self) -> None:
        self._commentary.clear()

    def _change_theme(self, name: str) -> None:
        self.theme_manager.set_theme(name)

    # ══════════════════════════════════════════════════════════════════════════
    # Signal slots (from VideoProcessor)
    # ══════════════════════════════════════════════════════════════════════════

    @pyqtSlot(int, str)
    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status.setText(msg)

    @pyqtSlot(str, bool)
    def _on_complete(self, output_path: str, success: bool) -> None:
        self._processing = False
        self._btn_upload.setEnabled(True)
        self._btn_team.setEnabled(True)
        self._btn_process.setEnabled(True)
        self._btn_cancel.setEnabled(False)

        if success:
            self._progress.setValue(100)
            self._status.setText("Processing complete ✔")
            self._output_player.load_video(output_path)
            self._video_stack.setCurrentIndex(2)      # switch to output player
            self._btn_export.setEnabled(True)
            self._btn_playpause.setEnabled(True)
            self._append_commentary(
                f"── Processing complete — output: {Path(output_path).name} ──"
            )
            log.info("Processing complete: %s", output_path)
        else:
            self._progress.setValue(0)
            self._status.setText("Processing failed ✘")
            self._video_stack.setCurrentIndex(0)
            QMessageBox.critical(self, "Error",
                                 "Processing failed — check app.log for details.")

    @pyqtSlot(float)
    def _on_hconf(self, conf: float) -> None:
        self._minimap.update_confidence(conf)
        if conf <= 0:
            self._hconf_label.setText("Homography: –")
            self._hconf_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        else:
            pct = f"{conf:.0%}"
            color = SUCCESS if conf >= 0.7 else (WARNING if conf >= 0.4 else DANGER)
            self._hconf_label.setText(f"Homography: {pct}")
            self._hconf_label.setStyleSheet(f"color: {color}; font-size: 10px;")

    @pyqtSlot(str)
    def _append_commentary(self, line: str) -> None:
        # Detect if line has a timestamp prefix "MM:SS  —  text"
        if "  —  " in line:
            ts, text = line.split("  —  ", 1)
            html = (
                f'<span style="color:{ACCENT}; font-family:monospace;">{ts}</span>'
                f'&nbsp;&nbsp;<span style="color:{TEXT};">{text}</span>'
            )
        else:
            html = f'<span style="color:{TEXT_MUTED}; font-style:italic;">{line}</span>'

        self._commentary.append(html)
        sb = self._commentary.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(dict)
    def _on_stats(self, data: dict) -> None:
        m = data.get("match", {})

        # Score
        score = m.get("score", {})
        self._lbl_score_a.setText(str(score.get("team_a", 0)))
        self._lbl_score_b.setText(str(score.get("team_b", 0)))

        # Possession bar
        poss = m.get("possession", {})
        self._possession_bar.set_possession(
            poss.get("team_a", 50), poss.get("team_b", 50)
        )

        # Stat counters
        shots  = m.get("shots",  {})
        passes = m.get("passes", {})
        fouls  = m.get("fouls",  {})

        self._stat_shots_a.setText(str(shots.get("team_a",  0)))
        self._stat_shots_b.setText(str(shots.get("team_b",  0)))
        self._stat_passes_a.setText(str(passes.get("team_a", 0)))
        self._stat_passes_b.setText(str(passes.get("team_b", 0)))
        self._stat_fouls_a.setText(str(fouls.get("team_a",  0)))
        self._stat_fouls_b.setText(str(fouls.get("team_b",  0)))

    # ══════════════════════════════════════════════════════════════════════════
    # Team-sheet persistence helpers
    # ══════════════════════════════════════════════════════════════════════════

    _CONFIG_DIR  = Path.home() / ".football_commentary"
    _SHEET_FILE  = _CONFIG_DIR / "default_team_sheet.json"
    _DEFAULT_SHEET = {
        "team_a": {
            "team_name": "Team A",
            "players": [
                {"number": 1,  "name": "Goalkeeper A",   "position": "GK"},
                {"number": 9,  "name": "Striker A",      "position": "FW"},
                {"number": 10, "name": "Playmaker A",    "position": "MF"},
            ],
        },
        "team_b": {
            "team_name": "Team B",
            "players": [
                {"number": 1,  "name": "Goalkeeper B",   "position": "GK"},
                {"number": 9,  "name": "Striker B",      "position": "FW"},
                {"number": 10, "name": "Playmaker B",    "position": "MF"},
            ],
        },
    }

    def _load_team_sheet(self) -> dict:
        try:
            self._CONFIG_DIR.mkdir(exist_ok=True)
            if self._SHEET_FILE.exists():
                with open(self._SHEET_FILE) as f:
                    return json.load(f)
        except Exception as exc:
            log.warning("Could not load team sheet: %s", exc)
        return self._DEFAULT_SHEET

    def _save_team_sheet(self, data: dict) -> None:
        try:
            self._CONFIG_DIR.mkdir(exist_ok=True)
            with open(self._SHEET_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("Could not save team sheet: %s", exc)

    def _team_name(self, key: str) -> str:
        return (self.team_sheet_data or {}).get(key, {}).get("team_name", key.upper())

    def _refresh_team_names(self) -> None:
        a = self._team_name("team_a")
        b = self._team_name("team_b")
        self._lbl_team_a.setText(a)
        self._lbl_team_b.setText(b)
        self._stats_name_a.setText(a)
        self._stats_name_b.setText(b)

    # ══════════════════════════════════════════════════════════════════════════
    # Close
    # ══════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.processor and self.processor.isRunning():
            self.processor.canceled = True
            self.processor.wait(3000)
        log.info("Application closed")
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent cross-platform rendering

    # High-DPI
    try:
        app.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass

    window = FootballAnalysisApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
