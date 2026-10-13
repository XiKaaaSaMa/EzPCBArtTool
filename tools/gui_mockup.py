#!/usr/bin/env python
"""Static layout preview of the planned M1 interface.

Every widget here is a placeholder: nothing is connected to the pipeline, and
every number shown is made up. The point is to review the layout, information
density and visual style before any of it is implemented, so that decisions
about the real interface are made against something visible rather than a
description.

The canvas is painted by hand rather than rendered from a board file, so what
you see is interface chrome -- grid, rulers, dimension annotation -- plus a
static ring-with-hole shape standing in for the artwork.

Usage::

    python tools/gui_mockup.py
    python tools/gui_mockup.py --screenshot preview.png   # render and exit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import uuid
import warnings
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

# A packaged build has no source tree to walk up out of, so the project root is
# only added when this file is actually sitting in one. PyInstaller resolves
# both `pcbart` and `design_canvas` from its own bundle instead.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    DEFAULT_IMAGE_LONG_EDGE_MM,
    DEFAULT_TAB_HEIGHT_MM,
    DEFAULT_TAB_WIDTH_MM,
    IRREGULAR_BASES,
    OUTLINE_SHAPES,
    SHAPE_CHOICES,
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    IrregularBase,
    OutlineShape,
    build_template,
    placement_for_pixels,
    template_names,
)
from pcbart.boardart import (  # noqa: E402
    TONE_THRESHOLD,
    BoardArtParams,
    board_art,
    board_art_preview,
    layer_keys as board_art_layer_keys,
    tone_to_ink,
)
from pcbart import badge_card  # noqa: E402
from pcbart import badge_content  # noqa: E402
from pcbart.design_layers import LAYERS, LAYER_KEYS, LayerStack, layer  # noqa: E402
from pcbart.draft import (  # noqa: E402
    DRAFT_FILTER,
    DRAFT_SUFFIX,
    DraftInfo,
    DraftStore,
    apply_board_dict,
    board_to_dict,
    default_draft_path,
    fingerprint,
    load_draft,
    save_draft,
)
from pcbart import operators  # noqa: E402
from pcbart.projects import UNTITLED, Project, Workspace  # noqa: E402
from pcbart.ui_theme import (  # noqa: E402
    PRESETS,
    ROLE_LABELS,
    Scheme,
    scheme_from_roles,
    scheme_from_seed,
    scheme_from_settings,
    settings_for,
    save_settings,
    save_settings_keeping,
)

# The Qt half of the theme: icon rasterising, the control style and the
# application-wide stylesheet. Imported the same careful way as the canvas,
# because a packaged build resolves `tools.*` from its own bundle.
try:
    from tools.qt_theme import (  # noqa: E402
        apply_scheme as qt_apply_scheme,
        icon as qt_theme_icon,
        style_window,
    )
except ImportError:  # running as a script, with tools/ itself on the path
    from qt_theme import (  # noqa: E402
        apply_scheme as qt_apply_scheme,
        icon as qt_theme_icon,
        style_window,
    )

# The canvas lives beside this file rather than inside it: it is a substantial
# widget with its own geometry, and keeping it separate keeps this module
# focused on the surrounding layout.
#
# Imported by full path (`tools.design_canvas`) rather than by bare name so a
# packaged build can resolve it without a sys.path entry pointing at a source
# directory. A source run falls back to the sibling import, which is what the
# CLI and the tests have always used.
try:
    from tools.design_canvas import (  # noqa: E402
        CANVAS_BG,
        GRID_MAJOR,
        GRID_MINOR,
        RULER_BG,
        DesignCanvas,
    )
except ImportError:  # running as a script, with tools/ itself on the path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from design_canvas import (  # noqa: E402
        CANVAS_BG,
        GRID_MAJOR,
        GRID_MINOR,
        RULER_BG,
        DesignCanvas,
    )


from PySide6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ----------------------------------------------------------------- palette
#
# The colours below are module-level names, but they are **not constants**: the
# user picks the colour scheme at runtime, so these are re-bound whenever the
# theme changes. Every call site in this file interpolates them into a
# stylesheet string at the moment it builds a widget, which is why re-binding
# them is enough for the widgets built afterwards -- and why the theme switch
# rebuilds the window rather than trying to mutate live widgets.
#
# They are read through ``_PALETTE`` so a lookup happens at call time rather
# than an import-time copy being frozen into the module. A plain
# ``WINDOW_BG = scheme["window"]`` would keep whichever scheme was active when
# this module was imported, which is exactly the class of bug that already bit
# this file once (a duplicated colour table that kept the old white canvas).

_PALETTE: "Scheme" = None  # type: ignore[assignment]


def set_palette(scheme) -> None:
    """Re-bind the module's colour names to ``scheme``.

    Called during startup and again on every theme switch. Widgets built
    *after* this call pick up the new values directly; widgets built earlier
    are brought along by :func:`restyle`, which re-runs the style builder each
    of them registered through :func:`themed`.
    """
    global _PALETTE, WINDOW_BG, PANEL_BG, PANEL_ALT, BORDER, TEXT, MUTED
    global ACCENT, ACCENT_TEXT, ACCENT_SOFT, ACCENT_SOFT_TEXT, HOVER, ACTIVE
    global SUCCESS, WARNING, DANGER, ON_ACCENT, DISABLED_BG, DISABLED_TEXT
    global WARNING_BG, WARNING_FG, WARNING_EDGE
    global TITLE_BAR_BG, OUTLINE, CUTOUT, DIM

    _PALETTE = scheme
    WINDOW_BG = scheme["window"]
    PANEL_BG = scheme["panel"]
    PANEL_ALT = scheme["panel_alt"]
    BORDER = scheme["border"]
    TEXT = scheme["text"]
    MUTED = scheme["muted"]
    ACCENT = scheme["accent"]
    ACCENT_TEXT = scheme["accent_text"]
    ACCENT_SOFT = scheme["accent_soft"]
    ACCENT_SOFT_TEXT = scheme["accent_soft_text"]
    HOVER = scheme["hover"]
    ACTIVE = scheme["active"]
    SUCCESS = scheme["success"]
    WARNING = scheme["warning"]
    DANGER = scheme["danger"]
    ON_ACCENT = scheme["on_accent"]
    DISABLED_BG = scheme["disabled_bg"]
    DISABLED_TEXT = scheme["disabled_text"]

    # The warning banner's three colours come from the scheme's container pair
    # plus the warning hue for its edge. Derived here once rather than spelled
    # out at each call site: that is exactly how the old cream banner ended up
    # frozen into a dark theme.
    WARNING_BG = scheme["warning_container"]
    WARNING_FG = scheme["on_warning_container"]
    WARNING_EDGE = scheme["warning"]

    TITLE_BAR_BG = WINDOW_BG
    """Title-bar fill for an in-app floating panel: the window colour.

    The bar is an explicit colour rather than transparent. Only the title
    *text* carries no background, so it never shows a box of its own.
    """

    # Not part of the generated scheme: these three describe the *board*, not
    # the interface, and follow the EDA layer table the canvas is drawn from.
    # Re-stating them here keeps every colour name in one place.
    OUTLINE = "#c026d3"
    CUTOUT = "#b91c1c"
    DIM = "#2563eb"


def current_scheme():
    """The scheme in force, loading the saved one on first use."""
    global _PALETTE
    if _PALETTE is None:
        from pcbart.ui_theme import scheme_from_settings

        set_palette(scheme_from_settings())
    return _PALETTE


# Bound at import so a module-level interpolation before the first
# ``set_palette`` call still finds a colour rather than a NameError.
set_palette(current_scheme())

PANEL_WIDTH = 316
SIDE_WIDTH = 296

PREVIEW_DEBOUNCE_MS = 90
"""How long the preview waits after the last tuning change before converting.

Every step of a slider drag emits its own value. A conversion costs long enough
that converting each step makes the window fall behind the handle, so the work
is coalesced onto a short single-shot timer and only the value the user stopped
on is computed. Short enough that a single click still feels immediate; long
enough that a continuous drag produces one conversion instead of dozens.
"""


# ------------------------------------------------------------- themed styles
#
# Every per-widget stylesheet in this file interpolates theme colours, and a
# widget's stylesheet is a *string*, not a live binding -- assign it once and
# it keeps those colours forever. That is why the first version of the theme
# switch rebuilt the whole window instead of restyling it.
#
# ``themed`` fixes that at the source: the widget keeps the *builder* rather
# than only its result, so :func:`restyle` can ask it again after the palette
# moves. Call sites that used to read
#
#     label.setStyleSheet(f"color: {MUTED}; ...")
#
# now read
#
#     themed(label, lambda: f"color: {MUTED}; ...")
#
# and the difference is that the f-string is re-evaluated on demand instead of
# being frozen at construction.


#: Attribute name the builder is stored under. Prefixed to stay clear of Qt's
#: own properties and of this module's ``_``-prefixed private state.
_BUILDER_ATTR = "_pcbart_style_builder"


def themed(widget: QWidget, builder) -> QWidget:
    """Give ``widget`` a stylesheet that is rebuilt when the theme changes.

    ``builder`` is a zero-argument callable returning the stylesheet string.
    It is invoked now and again by :func:`restyle`, which is what makes a
    switch instant and complete instead of instant and partial.

    ⚠️ **The builder is re-run, so it must read current state rather than
    capture a decision made when the widget was built.** A builder that closes
    over a boolean computed once will keep painting that old decision in the
    new colours -- which looks like the theme half-applied.
    """
    setattr(widget, _BUILDER_ATTR, builder)
    widget.setStyleSheet(builder())
    return widget


def muted(widget: QWidget, *, size: int = 11) -> QWidget:
    """Style ``widget`` as secondary text: the muted colour at a small size.

    A named helper rather than repeating the f-string forty times, and the one
    place to change if the muted treatment ever does. It also makes the intent
    legible at the call site -- ``muted(note)`` says *why* the widget is grey,
    where a stylesheet says only *that* it is.
    """
    return themed(widget, lambda: f"color: {MUTED}; font-size: {size}px;")


def note_text(label: QLabel, message: str) -> None:
    """Set a note label's message, hiding the label while it has none.

    These labels carry outcomes and errors now. The instructions that used to
    live in them -- telling the user to tune parameters first, or what a
    setting will do -- were removed at the user's request, so an empty message
    must leave no gap behind.
    """
    label.setText(message)
    label.setVisible(bool(message))


#: Semantic text tones a status label can carry. Named rather than passed as a
#: colour so the builder can look the colour up *again* after a theme change --
#: capturing the value instead would replay the old one on every restyle.
TONES = {
    "muted": lambda: MUTED,
    "text": lambda: TEXT,
    "success": lambda: SUCCESS,
    "warning": lambda: WARNING,
    "danger": lambda: DANGER,
    "accent": lambda: ACCENT,
}


def tone(widget: QWidget, name: str, *, size: int = 11) -> QWidget:
    """Colour ``widget`` by semantic tone, and keep it correct across themes.

    The distinction from :func:`muted` matters for any label whose colour is a
    *decision*: a validation note that turns red, an import note that turns
    green. Writing those as ``f"color: {DANGER}"`` bakes in the shade, so a
    later theme change leaves the note in the old palette's red.

    Here the widget remembers the **tone** ("this is an error") and the colour
    is looked up whenever the stylesheet is built -- including on every later
    theme change.
    """
    if name not in TONES:
        raise KeyError(f"未知的文字色调: {name}（可用: {', '.join(TONES)}）")
    lookup = TONES[name]

    def build() -> str:
        return f"color: {lookup()}; font-size: {size}px;"

    return themed(widget, build)


def warn_surface(
    widget: QWidget, *, size: int | None = None, padding: str = "5px 8px"
) -> QWidget:
    """A banner or status box in the warning container colours.

    Named as a *surface* rather than a colour because the three values
    (background, text, edge) have to move together. The warning hue is a
    mid-tone in a light theme and near-white in a dark one, so pairing a new
    background with an old text colour is exactly how a banner becomes
    unreadable after a theme change.
    """

    def build() -> str:
        size_part = f" font-size: {size}px;" if size else ""
        return (
            f"background: {WARNING_BG}; color: {WARNING_FG};"
            f" border: 1px solid {WARNING_EDGE};"
            f" border-radius: 4px; padding: {padding};{size_part}"
        )

    return themed(widget, build)


def info_surface(widget: QWidget, *, padding: str = "6px 8px") -> QWidget:
    """A plain boxed readout: panel background, hairline edge, body text.

    For panel status lines that are informational rather than warnings, so an
    announcement like "当前：异形板" does not shout in the warning colour when
    nothing is wrong.
    """

    def build() -> str:
        return (
            f"background: {PANEL_BG}; border: 1px solid {BORDER};"
            f" border-radius: 4px; padding: {padding}; font-size: 11px;"
        )

    return themed(widget, build)


def restyle(root: QWidget) -> int:
    """Re-run every registered style builder under ``root``.

    Returns how many widgets were restyled, so a test can assert that the walk
    actually found something rather than passing because both sides are empty.

    Widgets are collected with :meth:`QWidget.findChildren`, which walks the
    live tree -- so a deleted panel is simply absent rather than a dangling
    reference to guard against.
    """
    count = 0
    candidates = [root, *root.findChildren(QWidget)]
    for widget in candidates:
        builder = getattr(widget, _BUILDER_ATTR, None)
        if builder is None:
            continue
        try:
            widget.setStyleSheet(builder())
        except RuntimeError:
            # The C++ object is gone but Python still holds a wrapper. Not an
            # error worth surfacing: the widget is no longer on screen.
            continue
        count += 1
    return count


#: Keyboard shortcuts worth teaching. Every binding here is one the window
#: actually installs -- the 编辑 menu table, the file-level shortcuts the
#: window sets up, or the canvas's own gestures. A row naming a key nothing
#: listens to would be worse than no cheat sheet at all, so adding one here
#: means adding the binding first.
KEY_CHIPS: tuple[tuple[str, str], ...] = (
    ("Ctrl+Tab", "切换板子"),
    ("Ctrl+N", "新建板子"),
    ("Ctrl+S", "保存草稿"),
    ("Ctrl+Shift+S", "另存为草稿"),
    ("Ctrl+O", "打开草稿"),
    ("Ctrl+E", "另存为工程"),
    ("Ctrl+Shift+E", "另存为下单文件"),
    ("Ctrl+Z", "撤销"),
    ("Ctrl+C / V / X", "复制 / 粘贴 / 剪切"),
    ("Del", "删除"),
    ("R", "旋转 90°"),
    ("Ctrl+G / Ctrl+Shift+G", "组合 / 取消组合"),
)

#: The canvas gestures, listed under the keys in the same cheat sheet.
MOUSE_CHIPS: tuple[tuple[str, str], ...] = (
    ("滚轮缩放", "中键/空格拖动平移"),
    ("双击画布", "整板回到视野"),
    ("Ctrl+单击 / Shift+单击", "加选 / 选一段"),
)


def shortcut_lines() -> list[str]:
    """The cheat sheet as plain lines, for 【帮助 → 快捷键】.

    Assembled from the same tuples the bindings are checked against, so the
    menu page and the keys the window really installs cannot drift apart.
    """
    lines = ["快捷键", ""]
    lines += [f"{key}　{action}" for key, action in KEY_CHIPS]
    lines += ["", "鼠标手势", ""]
    lines += [f"{key}　{action}" for key, action in MOUSE_CHIPS]
    return lines


def section(title: str, compact: bool = False) -> QGroupBox:
    """A titled group box. Sections are the only container in this layout.

    ``compact`` tightens the frame for dense panels; the main window keeps the
    roomier default.

    The title sits *inside* the frame. Qt's default places it on the border
    (``subcontrol-origin: margin``), which reads as a label floating over the
    box edge; anchoring it to the padding box keeps it contained.
    """
    padding_top = 20 if compact else 22
    padding = (
        f"{padding_top}px 7px 5px 7px" if compact else f"{padding_top}px 9px 7px 9px"
    )
    box = QGroupBox(title)
    themed(
        box,
        lambda: f"""
                    QGroupBox {{
                        background: {PANEL_BG};
                        border: 1px solid {BORDER};
                        border-radius: 4px;
                        margin-top: 0px;
                        padding: {padding};
                        font-weight: 600;
                    }}
                    QGroupBox::title {{
                        subcontrol-origin: padding;
                        subcontrol-position: top left;
                        left: 2px;
                        top: 2px;
                        padding: 0;
                        color: {TEXT};
                    }}
        """,
    )
    return box


class WheelIgnoring:
    """Mixin: refuse wheel input so scrolling cannot edit a value.

    A slider or combo box that happens to sit under the pointer changes value
    when the user is only trying to scroll the panel, silently altering a
    setting they never touched. Ignoring the event hands it back to Qt, which
    propagates it to the surrounding scroll area -- so scrolling still works,
    it just no longer edits anything.
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.ignore()


class NoWheelComboBox(WheelIgnoring, QComboBox):
    """A combo box that ignores wheel input."""


class NoWheelSlider(WheelIgnoring, QSlider):
    """A slider that ignores wheel input."""


class NoWheelSpinBox(WheelIgnoring, QSpinBox):
    """An integer spin box that ignores wheel input."""


class NoWheelDoubleSpinBox(WheelIgnoring, QDoubleSpinBox):
    """A decimal spin box that ignores wheel input."""


def ndarray_to_qimage(array: np.ndarray) -> QImage:
    """Copy a greyscale or BGR numpy array into a ``QImage``.

    The buffer is copied because ``QImage`` does not take ownership of it:
    wrapping a temporary produces a corrupt image the moment it is freed.
    """
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    contiguous = np.ascontiguousarray(array)
    image = QImage(
        contiguous.data,
        contiguous.shape[1],
        contiguous.shape[0],
        contiguous.strides[0],
        QImage.Format_BGR888,
    )
    return image.copy()


def row(label: str, widget: QWidget, hint: str = "") -> QWidget:
    """One labelled control row; the hint sits under the control in muted text."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 1, 0, 1)
    layout.setSpacing(1)

    line = QWidget()
    line_layout = QHBoxLayout(line)
    line_layout.setContentsMargins(0, 0, 0, 0)
    line_layout.setSpacing(8)

    name = QLabel(label)
    name.setMinimumWidth(96)
    name.setWordWrap(False)
    line_layout.addWidget(name)
    line_layout.addWidget(widget, 1)

    layout.addWidget(line)
    if hint:
        note = QLabel(hint)
        muted(note)
        note.setWordWrap(True)
        note.setContentsMargins(104, 0, 0, 0)
        layout.addWidget(note)
    return holder


def colour_row(label: str, swatch: QWidget) -> QWidget:
    """A label with its colour chip immediately beside it.

    Separate from :func:`row` because that helper stretches its widget to fill
    the line, which is right for a combo box and wrong for a fixed-size chip:
    stretched, the chip drifts to the far right edge and the eye has to travel
    across the whole panel to pair a name with its colour.
    """
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(8)

    name = QLabel(label)
    name.setMinimumWidth(96)
    layout.addWidget(name)
    layout.addWidget(swatch)
    layout.addStretch(1)
    return holder


def combo(items: list[str], current: int = 0) -> QComboBox:
    """A combo box that can shrink, and that ignores wheel input.

    By default Qt sizes a ``QComboBox`` from its longest entry, so the layer
    list -- which includes "顶层 + 顶层丝印层 + 顶层阻焊层" -- imposes a
    223 px floor and forces the whole panel wider than its viewport. Adjusting
    to a short contents length lets the control be narrower than its text,
    which is normal behaviour for a combo box.
    """
    box = NoWheelComboBox()
    box.addItems(items)
    box.setCurrentIndex(current)
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(6)
    box.setMinimumWidth(0)
    box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return box


def spin(
    value: float,
    lo: float,
    hi: float,
    step: float,
    suffix: str = "",
    decimals: int = 2,
) -> QDoubleSpinBox:
    box = NoWheelDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    return box


def int_spin(value: int, lo: int, hi: int, suffix: str = "") -> QSpinBox:
    box = NoWheelSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


def slider(value: int) -> QSlider:
    bar = NoWheelSlider(Qt.Horizontal)
    bar.setRange(0, 100)
    bar.setValue(value)
    return bar


def checkbox(text: str, checked: bool = False) -> QCheckBox:
    box = QCheckBox(text)
    box.setChecked(checked)
    return box


# ---------------------------------------------------------------- canvas


class CanvasPlaceholder(QWidget):
    """Interface chrome only: rulers, millimetre grid, one stand-in shape."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(CANVAS_BG))

        ruler = 22
        plot = QRectF(
            ruler, ruler, self.width() - ruler, self.height() - ruler
        )
        if plot.width() <= 10 or plot.height() <= 10:
            return

        # 60 mm of artwork across the short side, snapped to a round scale.
        px_per_mm = min(plot.width(), plot.height()) / 78.0
        step = 5.0
        while step * px_per_mm < 14:
            step *= 2
        major_every = 2

        cx = plot.center().x()
        cy = plot.center().y()

        def grid_x(mm: float) -> float:
            return cx + mm * px_per_mm

        def grid_y(mm: float) -> float:
            return cy - mm * px_per_mm

        painter.setClipRect(plot)
        painter.setPen(QPen(QColor(GRID_MINOR), 1))
        span_mm = max(plot.width(), plot.height()) / px_per_mm
        n = int(span_mm / step) + 2
        for i in range(-n, n + 1):
            mm = i * step
            pen = QPen(QColor(GRID_MAJOR if i % major_every == 0 else GRID_MINOR), 1)
            painter.setPen(pen)
            painter.drawLine(QPointF(grid_x(mm), plot.top()), QPointF(grid_x(mm), plot.bottom()))
            painter.drawLine(QPointF(plot.left(), grid_y(mm)), QPointF(plot.right(), grid_y(mm)))

        # Stand-in artwork: a 60 mm ring with a 36.6 mm circular cutout.
        board_r = 30.0 * px_per_mm
        hole_r = 36.576 / 2.0 * px_per_mm
        board = QRectF(cx - board_r, cy - board_r, board_r * 2, board_r * 2)
        hole = QRectF(cx - hole_r, cy - hole_r, hole_r * 2, hole_r * 2)

        painter.setPen(QPen(QColor(OUTLINE), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(board)

        painter.setPen(QPen(QColor(CUTOUT), 1.5, Qt.DashLine))
        painter.setBrush(QColor(CANVAS_BG))
        painter.drawEllipse(hole)

        # Dimension annotation under the board.
        y = board.bottom() + 16
        painter.setPen(QPen(QColor(DIM), 1))
        painter.drawLine(QPointF(board.left(), y), QPointF(board.right(), y))
        for x in (board.left(), board.right()):
            painter.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))

        painter.setPen(QColor(DIM))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        label = f"{board_r * 2 / px_per_mm:.2f} mm"
        metrics = QFontMetrics(font)
        painter.fillRect(
            QRectF(
                cx - metrics.horizontalAdvance(label) / 2 - 4,
                y - 8,
                metrics.horizontalAdvance(label) + 8,
                15,
            ),
            QColor(CANVAS_BG),
        )
        painter.drawText(
            QPointF(cx - metrics.horizontalAdvance(label) / 2, y + 4), label
        )

        painter.setClipping(False)

        # Rulers.
        painter.fillRect(QRectF(0, 0, self.width(), ruler), QColor(RULER_BG))
        painter.fillRect(QRectF(0, 0, ruler, self.height()), QColor(RULER_BG))
        painter.setPen(QColor(BORDER))
        painter.drawLine(0, ruler, self.width(), ruler)
        painter.drawLine(ruler, 0, ruler, self.height())

        painter.setPen(QColor(MUTED))
        font.setPointSize(7)
        painter.setFont(font)
        for i in range(-n, n + 1):
            mm = i * step
            if i % major_every:
                continue
            x = grid_x(mm)
            if plot.left() <= x <= plot.right():
                painter.drawLine(QPointF(x, ruler - 4), QPointF(x, ruler))
                painter.drawText(QPointF(x + 2, ruler - 6), f"{mm:.0f}")
            yy = grid_y(mm)
            if plot.top() <= yy <= plot.bottom():
                painter.drawLine(QPointF(ruler - 4, yy), QPointF(ruler, yy))
                painter.drawText(QPointF(3, yy + 8), f"{mm:.0f}")

        painter.setPen(QColor(MUTED))
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QRectF(plot.left(), plot.bottom() - 18, plot.width(), 16),
            Qt.AlignRight,
            "单位 mm　·　静态示意图，非真实几何",
        )


# ------------------------------------------------- image preprocessing window

COLOR_STYLE = "彩色丝印"
"""Style that keeps the artwork's own colours and locks it to silkscreen.

Colour is only reproducible on the silkscreen layer, so this combination is a
hard rule rather than a default the user can override. It is also the *only*
style that shows the original picture: everywhere else the canvas shows the
processed line art, tinted in its layer's colour.
"""

BOARD_ART_STYLE = "板画"
"""Style that splits the picture into two tones instead of tracing lines.

White goes to silkscreen and black to copper, and both are solid filled
regions. Which layers it uses is fixed -- silk is the only layer that prints
white, copper the only one that gives the dark field a real extent -- so this
style changes what 呈现方式 offers rather than being overridden by it.
"""

STYLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("线稿", "提取线条，输出黑白线稿，按图层颜色显示"),
    (
        BOARD_ART_STYLE,
        "轮廓白线 + 亮部白块、暗部与背景黑块；白色走丝印、黑色走铜",
    ),
    (COLOR_STYLE, "保留原图配色：只有丝印能表达颜色，层已锁定"),
)


#: What ``_committed_style`` holds when 处理图像 was off at import. It is not a
#: style -- it marks that the picture went to the board untouched, so the
#: publishing guard and the preview both know not to hand it a conversion.
RAW_STYLE = "__raw__"


#: How the 线稿风格 reduces a picture to black and white, in the order the
#: combo shows them. ``(key, label, description)``.
#:
#: The readings answer different questions and a picture has to pick one, so
#: this is a choice rather than a single "best" default:
#:
#: * ``fill`` splits the picture at one level, so *everything* is either black
#:   or white. It is right for flat two-tone drawings and it is what the
#:   project calibrated against a hand-made Photoshop reference.
#: * ``outline`` adds a contour pass on top of the same split, so boundaries
#:   that are neither dark nor light enough to cross the level -- eyebrows, the
#:   links of a chain, cloth weave -- survive as lines instead of being
#:   swallowed into a block. It is right for soft-shaded illustrations, which
#:   is what the asset library contains.
#: * ``dodge`` asks "is this pixel darker than its neighbourhood?" and ignores
#:   absolute brightness, so a dark costume that ``fill`` would flood into one
#:   black mass comes out as its contour lines. Measured on the dark
#:   illustration that prompted it: fill 15.2% ink as a black block, dodge 11.5%
#:   as pure line work with the costume lines, the leg rings and the gun intact.
#:   It is not better across the board -- on a pale flat drawing its ratio
#:   response turns texture noise into a mesh of hairlines, which is what 降噪
#:   is for -- so the description below names both audiences rather than
#:   claiming a winner.
#:
#: ⚠️ ``dodge`` **must stay at the end of the tuple**. ``_reset_tuning``
#: returns to index 0 and several tests look options up by key, but the default
#: reading is a promise as well: inserting above ``fill`` would silently change
#: what a fresh import does.
READING_FILL = "fill"
READING_OUTLINE = "outline"
READING_DODGE = "dodge"

READING_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        READING_FILL,
        "填充色块（默认）",
        "按一个明暗分界分成黑白两块：适合纯色线稿，细节少但干净",
    ),
    (
        READING_OUTLINE,
        "描边 + 填充",
        "先找轮廓，再补暗部色块：适合有明暗过渡的立绘，眉毛、链条、布料纹理能保留",
    ),
    (
        READING_DODGE,
        "纯线条",
        "只描「比周围暗」的边，深色立绘不涂黑块；浅色平涂图易出碎网，可先降噪",
    ),
)


def layer_options(side: str) -> list[tuple[str, tuple[str, ...], str]]:
    """How the artwork is realised in physical layers, for one board side.

    Returns ``(label, layer keys, description)``. The keys matter: this choice
    decides where an imported picture actually lands, so it has to name real
    layers rather than only describe them.

    A combined entry means one object per layer. The same picture on copper
    and on silkscreen is two objects with independent geometry, which is how
    the editor models it too, and the canvas draws them in their own colours.

    Only copper, silkscreen and solder mask appear here. The paste and
    assembly layers are deliberately absent: this tool does not model
    assembly, and offering them would imply a capability that is not there.
    """
    prefix = "top" if side == "顶层" else "bottom"
    copper, silk, mask = f"{prefix}_copper", f"{prefix}_silk", f"{prefix}_mask"
    return [
        (f"{side}", (copper,), "铜箔浮雕：凸起的金属线条，无阻焊覆盖"),
        (f"{side}丝印层", (silk,), "黑白丝印：最常见的图案表达"),
        (f"{side}阻焊层", (mask,), "阻焊开窗：露出板材本色，呈黄色"),
        (f"{side} + {side}丝印层", (copper, silk), "铜箔做凸起，丝印补充细节"),
        (
            f"{side} + {side}阻焊层",
            (copper, mask),
            "金属色图案：开窗处的铜面，通常是金色",
        ),
        (
            f"{side} + {side}丝印层 + {side}阻焊层",
            (copper, silk, mask),
            "三层叠加，细节最丰富",
        ),
    ]


BOARD_ART_SIDE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("top", "顶面", "白色轮廓与亮部印在顶层丝印，黑色块面铺在顶层铜"),
    ("bottom", "底面", "白色轮廓与亮部印在底层丝印，黑色块面铺在底层铜"),
    ("both", "两面", "同样的图案正反各一份，做双面挂件时用"),
)
"""Side choices while 板画 is selected.

A shorter list than :func:`layer_options` on purpose. That list names physical
layer combinations, which is what 线稿 needs because it can print on copper,
silk or mask. A board painting is always the same *two* layers, so offering
six combinations would only invite a choice that cannot take effect.
"""


def segmented(options: list[str], current: int = 0) -> QWidget:
    """A two-state segmented control, for top/bottom style either-or choices."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    group = QButtonGroup(holder)
    buttons: list[QPushButton] = []
    last = len(options) - 1
    for index, text in enumerate(options):
        button = QPushButton(text)
        button.setCheckable(True)
        button.setChecked(index == current)
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(26)
        # Round only the outer ends of the pair. Qt has no :first/:last
        # selector, so the radius is decided here rather than in the sheet.
        left = 3 if index == 0 else 0
        right = 3 if index == last else 0
        themed(
            button,
            lambda: f"""
                            QPushButton {{
                                background: {PANEL_BG};
                                border: 1px solid {BORDER};
                                padding: 3px 14px;
                                color: {TEXT};
                                border-top-left-radius: {left}px;
                                border-bottom-left-radius: {left}px;
                                border-top-right-radius: {right}px;
                                border-bottom-right-radius: {right}px;
                            }}
                            QPushButton:checked {{
                                background: {ACCENT};
                                border-color: {ACCENT};
                                color: white;
                                font-weight: 600;
                            }}
            """,
        )
        group.addButton(button, index)
        layout.addWidget(button)
        buttons.append(button)

    holder.buttons = buttons  # type: ignore[attr-defined]
    holder.group = group  # type: ignore[attr-defined]
    return holder


class DropZone(QWidget):
    """Upload target: click to browse, or drag a file onto it.

    Local files only. Once an image is loaded the zone shows the picture
    itself, scaled to fit, with the filename underneath.
    """

    image_selected = Signal(str)
    """Emitted with the chosen path, after the user picks or drops a file."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(168)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptDrops(True)
        self._dragging = False
        self._pixmap: QPixmap | None = None
        self._caption = ""
        self._detail = ""

    # ------------------------------------------------------------ file input

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.browse()
            event.accept()

    def browse(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;所有文件 (*)",
        )
        if path:
            self.image_selected.emit(path)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls() and self._first_local_file(event):
            self._dragging = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.update()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.update()
        path = self._first_local_file(event)
        if path:
            event.acceptProposedAction()
            self.image_selected.emit(path)

    @staticmethod
    def _first_local_file(event) -> str | None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                return url.toLocalFile()
        return None

    # -------------------------------------------------------------- display

    def show_image(self, pixmap: QPixmap, caption: str, detail: str) -> None:
        self._pixmap = pixmap
        self._caption = caption
        self._detail = detail
        self.update()

    def show_array(self, array_bgr: np.ndarray, caption: str, detail: str) -> None:
        """Show a processed image held as a numpy array.

        An array is copied before hand-off: ``QImage`` does not take ownership
        of the buffer, so the data has to outlive the call.
        """
        height, width = array_bgr.shape[:2]
        if array_bgr.ndim == 2:
            array_bgr = cv2.cvtColor(array_bgr, cv2.COLOR_GRAY2BGR)
        contiguous = np.ascontiguousarray(array_bgr)
        image = QImage(
            contiguous.data, width, height, contiguous.strides[0], QImage.Format_BGR888
        )
        self.show_image(QPixmap.fromImage(image.copy()), caption, detail)

    def clear_image(self) -> None:
        self._pixmap = None
        self._caption = ""
        self._detail = ""
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        border = ACCENT if self._dragging else BORDER
        fill = "#eef4ff" if self._dragging else "#fbfcfd"

        box = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        painter.setBrush(QColor(fill))
        pen = QPen(QColor(border), 2, Qt.DashLine)
        pen.setDashPattern([7, 5])
        painter.setPen(pen)
        painter.drawRoundedRect(box, 12, 12)

        if self._pixmap is None:
            self._paint_prompt(painter, box)
        else:
            self._paint_preview(painter, box)

    def _paint_prompt(self, painter: QPainter, box: QRectF) -> None:
        centre = box.center()
        plus_size = 20
        plus_y = centre.y() - min(30.0, box.height() * 0.16)
        painter.setPen(QPen(QColor(MUTED), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(
            QPointF(centre.x() - plus_size / 2, plus_y),
            QPointF(centre.x() + plus_size / 2, plus_y),
        )
        painter.drawLine(
            QPointF(centre.x(), plus_y - plus_size / 2),
            QPointF(centre.x(), plus_y + plus_size / 2),
        )

        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        painter.drawText(
            QRectF(box.left(), plus_y + 18, box.width(), 22),
            Qt.AlignCenter,
            "点击或拖拽图片到此处",
        )

        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(box.left(), plus_y + 42, box.width(), 20),
            Qt.AlignCenter,
            "支持 PNG / JPG / WEBP / BMP",
        )

    def _paint_preview(self, painter: QPainter, box: QRectF) -> None:
        """Draw the loaded image scaled to fit, with its caption beneath."""
        footnote = 34.0
        area = box.adjusted(8, 8, -8, -footnote)

        pixmap = self._pixmap
        scaled = pixmap.scaled(
            QSize(int(area.width()), int(area.height())),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        target = QRectF(
            area.center().x() - scaled.width() / 2,
            area.center().y() - scaled.height() / 2,
            scaled.width(),
            scaled.height(),
        )

        # Checkerboard behind the image, so transparent regions read as
        # transparent instead of showing whatever colour happens to be there.
        if pixmap.hasAlphaChannel():
            self._paint_checkerboard(painter, target, scaled)

        painter.drawPixmap(target.toRect(), scaled)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        painter.drawText(
            QRectF(box.left() + 6, box.bottom() - footnote + 2, box.width() - 12, 15),
            Qt.AlignCenter,
            self._caption,
        )

        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(box.left() + 6, box.bottom() - footnote + 17, box.width() - 12, 14),
            Qt.AlignCenter,
            self._detail,
        )

    @staticmethod
    def _paint_checkerboard(painter: QPainter, target: QRectF, scaled) -> None:
        cell = 8
        light = QColor("#ffffff")
        dark = QColor("#dfe3e8")
        painter.save()
        painter.setClipRect(target)
        painter.fillRect(target, light)
        cols = scaled.width() // cell + 1
        rows = scaled.height() // cell + 1
        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2:
                    continue
                painter.fillRect(
                    QRectF(
                        target.left() + col * cell,
                        target.top() + row * cell,
                        cell,
                        cell,
                    ),
                    dark,
                )
        painter.restore()


class CaptionButton(QPushButton):
    """A title-bar button that paints its own glyph.

    The characters ``—``, ``□`` and ``×`` come from whatever font is in use
    and land at different sizes and baselines, so the three controls never line
    up. Drawing the glyph geometrically keeps them identical, and makes the
    same button usable for both maximise and restore.
    """

    SIZE = 26

    def __init__(self, kind: str, tooltip: str, on_click) -> None:
        super().__init__()
        self._kind = kind
        self.setToolTip(tooltip)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        themed(
            self,
            lambda: f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 3px;
            }}
            QPushButton:hover {{ background: {HOVER}; }}
            QPushButton:pressed {{ background: {ACTIVE}; }}
            """,
        )
        self.clicked.connect(on_click)

    def set_kind(self, kind: str) -> None:
        if kind != self._kind:
            self._kind = kind
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colour = QColor(DANGER if self._kind == "close" and self.underMouse() else MUTED)
        painter.setPen(QPen(colour, 1.4))
        painter.setBrush(Qt.NoBrush)

        c = self.rect().center()
        half = 4
        cx, cy = c.x() + 0.5, c.y() + 0.5

        if self._kind == "minimise":
            painter.drawLine(QPointF(cx - half, cy + 3), QPointF(cx + half, cy + 3))
        elif self._kind == "maximise":
            painter.drawRect(QRectF(cx - half, cy - half, half * 2, half * 2))
        elif self._kind == "restore":
            # Two offset squares, the usual "restore down" glyph.
            painter.drawRect(QRectF(cx - half, cy - half + 2, half * 2 - 2, half * 2 - 2))
            painter.drawLine(
                QPointF(cx - half + 2, cy - half), QPointF(cx + half, cy - half)
            )
            painter.drawLine(
                QPointF(cx + half, cy - half), QPointF(cx + half, cy + half - 2)
            )
        else:  # close
            painter.drawLine(
                QPointF(cx - half, cy - half), QPointF(cx + half, cy + half)
            )
            painter.drawLine(
                QPointF(cx - half, cy + half), QPointF(cx + half, cy - half)
            )


class PanelTitleBar(QWidget):
    """Title bar for an in-app floating panel: drag, minimise, maximise, close."""

    def __init__(self, panel: "FloatingPanel", title: str) -> None:
        super().__init__(panel)
        self._panel = panel
        self._press: QPoint | None = None
        self._origin: QPoint | None = None
        self.setFixedHeight(FloatingPanel.TITLE_HEIGHT)
        self.setCursor(Qt.SizeAllCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Painted in paintEvent rather than via a stylesheet. A stylesheet
        # border-radius on this child did not clip its fill, which left the
        # panel's top two corners square while the bottom two stayed round.
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(2)

        self.label = QLabel(title)
        themed(
            self.label,
            lambda: f"color: {TEXT}; font-weight: 600;"
            "background: transparent; border: none;",
        )
        layout.addWidget(self.label)
        layout.addStretch(1)

        self.minimise_button = CaptionButton("minimise", "最小化", panel.collapse)
        self.maximise_button = CaptionButton("maximise", "最大化 / 还原", panel.toggle_maximised)
        self.close_button = CaptionButton("close", "关闭", panel.close_panel)
        for button in (self.minimise_button, self.maximise_button, self.close_button):
            layout.addWidget(button)

    def set_title(self, title: str) -> None:
        self.label.setText(title)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Fill the bar grey with rounded top corners, plus a bottom hairline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        inner = max(0, FloatingPanel.RADIUS - 1)
        path = QPainterPath()
        path.moveTo(0, self.height())
        path.lineTo(0, inner)
        path.quadTo(0, 0, inner, 0)
        path.lineTo(self.width() - inner, 0)
        path.quadTo(self.width(), 0, self.width(), inner)
        path.lineTo(self.width(), self.height())
        path.closeSubpath()
        painter.fillPath(path, QColor(TITLE_BAR_BG))

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

    # Dragging moves the panel inside its parent, never onto the desktop.
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self._press = event.globalPosition().toPoint()
            self._origin = self._panel.pos()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is None or self._origin is None:
            return
        if not event.buttons() & Qt.LeftButton:
            return
        delta = event.globalPosition().toPoint() - self._press
        self._panel.move(self._origin + delta)
        self._panel.clamp_inside_parent()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._press = None
        self._origin = None


class ResizeGrip(QWidget):
    """One visible handle on the panel's border: an edge pill or a corner arrow.

    The panel is resizable from all eight directions, but the resize zone is a
    12 px border and the only hint was the cursor changing when the pointer
    happened to land on it, so users did not find it. Worse, two edges were not
    reachable at all: the scroll bar sits on the right border and the title bar
    on the top one, so presses there never reached the panel.

    Each grip now occupies a zone in a border strip that the layout reserves,
    which both makes it visible and keeps it clear of the content. A corner
    draws a diagonal double arrow; an edge draws a short pill, the way a
    splitter handle looks, because a diagonal arrow on a straight edge promises
    the wrong directions.

    It drives the panel's own resize code rather than reimplementing it, so
    every grip behaves identically, including the minimum-size floor and the
    parent-window clamping.
    """

    #: Which edge or corner this grip drives, in the panel's own vocabulary.
    ZONES = ("n", "s", "e", "w", "ne", "nw", "se", "sw")

    CURSORS = {
        "n": Qt.SizeVerCursor,
        "s": Qt.SizeVerCursor,
        "e": Qt.SizeHorCursor,
        "w": Qt.SizeHorCursor,
        "ne": Qt.SizeBDiagCursor,
        "sw": Qt.SizeBDiagCursor,
        "nw": Qt.SizeFDiagCursor,
        "se": Qt.SizeFDiagCursor,
    }

    LABELS = {
        "n": "拖动调整高度",
        "s": "拖动调整高度",
        "e": "拖动调整宽度",
        "w": "拖动调整宽度",
        "ne": "拖动调整大小",
        "nw": "拖动调整大小",
        "se": "拖动调整大小",
        "sw": "拖动调整大小",
    }

    def __init__(self, panel: "FloatingPanel", zone: str) -> None:
        super().__init__(panel)
        if zone not in self.ZONES:
            raise ValueError(f"未知的缩放手柄位置: {zone}")
        self._panel = panel
        self.zone = zone
        self.setCursor(self.CURSORS[zone])
        self.setToolTip(self.LABELS[zone])
        self.setMouseTracking(True)
        self.setFixedSize(*self._size_for(zone))

    @staticmethod
    def _size_for(zone: str) -> tuple[int, int]:
        edge = FloatingPanel.EDGE
        long = FloatingPanel.EDGE_PILL_LONG
        return {
            "n": (long, edge),
            "s": (long, edge),
            "e": (edge, long),
            "w": (edge, long),
            "ne": (edge, edge),
            "nw": (edge, edge),
            "se": (edge, edge),
            "sw": (edge, edge),
        }[zone]

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen(QColor(MUTED), 1.3)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        if self.zone in ("ne", "nw", "se", "sw"):
            self._paint_corner(painter)
        else:
            self._paint_edge(painter)

    def _paint_corner(self, painter: QPainter) -> None:
        """One double-headed arrow along the diagonal.

        Two arrows or a stack of diagonal lines crowd a 12 px box and read as
        noise. A single arrow with a head on each end survives the size and says
        both things that matter: this is a handle, and it moves both ways.

        The arrow points the way the corner moves when the panel *grows*, so
        the east corners mirror the west ones -- an identical glyph at all four
        corners would point the wrong way on half of them.
        """
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 150))
        painter.drawRoundedRect(self.rect(), 3, 3)

        pen = QPen(QColor(MUTED), 1.3)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        size = self.width()
        low, high = 3, size - 3
        head = 3
        east = "e" in self.zone
        north = "n" in self.zone

        # Work in "grows towards the bottom-right" and mirror the coordinate
        # that the zone flips. Writing each corner out separately is how the
        # barbs ended up pointing away from the shaft the first time.
        def point(x: float, y: float) -> tuple[int, int]:
            px = x if east else size - x
            py = y if north else size - y
            return int(px), int(py)

        painter.drawLine(*point(low, high), *point(high, low))
        painter.drawLine(*point(low, high), *point(low + head, high))
        painter.drawLine(*point(low, high), *point(low, high - head))
        painter.drawLine(*point(high, low), *point(high - head, low))
        painter.drawLine(*point(high, low), *point(high, low + head))

    def _paint_edge(self, painter: QPainter) -> None:
        """A short rounded pill along the edge it moves.

        A pill rather than an arrow: this handle moves one axis only, and a
        two-headed diagonal on a straight edge would promise both.
        """
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(MUTED))
        radius = min(rect.width(), rect.height()) / 2.0
        painter.drawRoundedRect(rect, radius, radius)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self._panel.begin_resize(self.zone, event.globalPosition().toPoint())
            # The panel grows out from under the pointer immediately, so
            # without the grab the rest of the drag would be delivered
            # somewhere else and the resize would stop after one step.
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panel.is_resizing():
            self._panel.apply_resize(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panel.is_resizing():
            self._panel.end_resize()
            self.releaseMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)


def chrome_height(window: QWidget) -> int:
    """Height of everything above the working area: menu, toolbar, tabs.

    Measured from the widgets rather than assumed: the tab row and the toolbar
    change height with the platform's font metrics, and a hard-coded offset
    would put the floating panels back on top of the tab row exactly then.

    ⚠️ **A widget that has never been laid out reports a meaningless height.**
    Before the window is shown, ``QToolBar.height()`` and friends answer with
    Qt's default widget size -- 480 px here -- rather than their real one, so
    this summed to 1470 on a 900 px window and put every floating panel below
    the bottom edge. The panel was open, populated and correct, and completely
    invisible; only the screenshots taken before ``show()`` could see it, which
    is exactly the state a user must never be able to reach.

    Unshown widgets therefore contribute their layout's own size hint. That is
    approximate, but it is the right order of magnitude, so a panel lands on
    screen and is re-placed precisely by the next resize or interaction.
    """
    def laid_out(widget) -> int:
        return widget.height() if widget.isVisible() else widget.sizeHint().height()

    total = laid_out(window.menuBar()) if hasattr(window, "menuBar") else 0

    for bar in window.findChildren(QToolBar):
        total += laid_out(bar)

    row = window.findChild(QWidget, "ProjectTabRow")
    if row is not None:
        total += laid_out(row)

    return total


class FloatingPanel(QWidget):
    """An in-app floating panel, not an operating-system window.

    Staying parented to the main window is the whole point: the panel cannot
    be opened from the desktop, takes no taskbar slot, is clipped to the
    application, and always sits on top of the canvas. That also means
    "minimise" has to collapse it to its title bar rather than iconify it.

    The panel is freely resizable by dragging any edge or corner. Qt's own
    ``QSizeGrip`` only offers the bottom-right corner and only grows, so the
    edges are handled here instead -- with :class:`ResizeGrip` drawn in the
    corner so the capability is visible rather than merely true.
    """

    MARGIN = 12
    TITLE_HEIGHT = 30

    #: Width of the border strip the layout reserves for the resize handles.
    #: Without it the content covers the border: the scroll bar sits on the
    #: right edge and the title bar on the top one, so presses there never
    #: reached the panel and those two edges could not be dragged at all.
    EDGE = 12

    #: Length of the pill drawn on a straight edge.
    EDGE_PILL_LONG = 30

    MIN_WIDTH = 330
    MIN_HEIGHT = 240
    RADIUS = 6

    #: Kept for the tests and for callers that sized themselves against it.
    GRIP = EDGE
    GRIP_SIZE = EDGE

    def __init__(self, parent: QWidget, title: str, width: int = 420) -> None:
        super().__init__(parent)
        self.setObjectName("FloatingPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        themed(
            self,
            lambda: f"""
            QWidget#FloatingPanel {{
                background: {WINDOW_BG};
                border: 1px solid {BORDER};
                border-radius: {self.RADIUS}px;
            }}
            """,
        )
        # The panel chooses its own size; only a floor is imposed.
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # A border strip is reserved on all four sides for the resize handles,
        # rather than letting them float over the content. Overlaid, they would
        # sit on whatever a panel puts near its edges -- the preprocessing
        # panel's 导入图片 button is at its bottom-right -- and swallowing a
        # click on a working control to start a resize is worse than having no
        # handle at all.
        #
        # The strip is also what makes the right and top edges draggable: the
        # scroll bar and the title bar are inside it, so presses on the border
        # reach the panel itself.
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(self.EDGE, self.EDGE, self.EDGE, self.EDGE)
        self._outer.setSpacing(0)
        self.title_bar = PanelTitleBar(self, title)
        self._outer.addWidget(self.title_bar)

        # Content scrolls rather than being crushed. Without this, dragging the
        # panel small squeezes every section until controls overlap.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            # The bar is painted 6 px away from the scroll area's own right
            # edge, which would put it straight through the right-hand resize
            # handle. Pulling it inwards keeps the two apart.
            "QScrollBar:vertical { margin-right: 6px; }"
        )
        self._outer.addWidget(self._scroll, 1)

        self._footer = QWidget()
        self._footer.setObjectName("PanelFooter")
        self._footer.setAttribute(Qt.WA_StyledBackground, True)
        themed(
            self._footer,
            lambda: f"""
            QWidget#PanelFooter {{
                background: {WINDOW_BG};
                border-bottom-left-radius: {max(0, self.RADIUS - 1)}px;
                border-bottom-right-radius: {max(0, self.RADIUS - 1)}px;
            }}
            """,
        )
        self._footer_layout = QVBoxLayout(self._footer)
        self._footer_layout.setContentsMargins(6, 4, 6, 6)
        self._footer_layout.setSpacing(3)
        self._footer.setVisible(False)
        self._outer.addWidget(self._footer)
        self._footer_has_widgets = False

        self._content = QWidget()
        self._content.setObjectName("PanelContent")
        self._content.setAttribute(Qt.WA_StyledBackground, True)
        # Rounded at the bottom, because this fills the panel's lower area and
        # would otherwise square off the corners the panel rounds.
        themed(
            self._content,
            lambda: f"""
            QWidget#PanelContent {{
                background: {WINDOW_BG};
                border-bottom-left-radius: {max(0, self.RADIUS - 1)}px;
                border-bottom-right-radius: {max(0, self.RADIUS - 1)}px;
            }}
            """,
        )
        self._scroll.setWidget(self._content)

        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        # Sections are separated by a single hairline, so the frame itself
        # reads as the divider.
        self.content_layout.setSpacing(3)

        self.resize(width, 560)

        self._collapsed = False
        self._maximised = False
        self._expanded_height = 560
        self._restore_geometry: QRect | None = None

        self._resize_edge = ""
        self._resize_press: QPoint | None = None
        self._resize_origin: QRect | None = None
        self._cursor_edge = ""
        """The border zone the cursor is currently showing, so it is only
        changed when it actually changes -- see :meth:`_apply_edge_cursor`."""
        self.setMouseTracking(True)

        # Created last so they sit above the scroll area and the footer in the
        # child stacking order, in the strip the layout reserved for them.
        self._grips: dict[str, ResizeGrip] = {}
        for zone in ResizeGrip.ZONES:
            grip = ResizeGrip(self, zone)
            self._grips[zone] = grip
        # The bottom-right handle keeps its own name: it is the one users
        # reach for, and tests and callers refer to it directly.
        self.resize_grip = self._grips["se"]
        self._place_grips()
        for grip in self._grips.values():
            grip.raise_()

        parent.installEventFilter(self)

        # Watch pointer moves application-wide so a border cursor is cleared
        # even when the move lands on a child. See ``watch_pointer``.
        self._pointer_watch_installed = False
        application = QApplication.instance()
        if application is not None:
            self.watch_pointer(application)

    def grips(self) -> list[ResizeGrip]:
        """Every resize handle, in a stable order."""
        return [self._grips[zone] for zone in ResizeGrip.ZONES]

    def _set_grips_visible(self, visible: bool) -> None:
        """Show or hide all eight handles together.

        One place decides, rather than every state transition remembering to
        touch each handle: forgetting one leaves a handle that looks usable on
        a panel whose edges are disabled.
        """
        for grip in self._grips.values():
            grip.setVisible(visible)

    def _place_grips(self) -> None:
        """Lay the eight handles out along the reserved border strip.

        Corners take the corners and the straight edges take the middle of
        their side. Each one is centred in the strip so it reads as part of the
        frame rather than as something resting on the content.
        """
        edge = self.EDGE
        width, height = self.width(), self.height()
        long = self.EDGE_PILL_LONG
        half_long = long // 2

        positions = {
            "n": ((width - long) // 2, 0),
            "s": ((width - long) // 2, height - edge),
            "w": (0, (height - long) // 2),
            "e": (width - edge, (height - long) // 2),
            "nw": (0, 0),
            "ne": (width - edge, 0),
            "sw": (0, height - edge),
            "se": (width - edge, height - edge),
        }
        _ = half_long  # kept with the arithmetic it documents

        for zone, (x, y) in positions.items():
            self._grips[zone].move(max(0, x), max(0, y))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_grips()

    # -------------------------------------------------------- panel mechanics

    def set_footer(self, widget: QWidget) -> None:
        """Pin a widget below the scrolling content, always visible.

        For an action that commits the panel's work. In the scrolling area it
        sat ~380 px below the fold on this panel, so the button that actually
        applies the settings was the one thing the user could not see.
        """
        self._footer_layout.addWidget(widget)
        self._footer_has_widgets = True
        self._footer.setVisible(not self._collapsed)

    def collapse(self) -> None:
        """Minimise to the title bar, staying visible inside the app."""
        if self._collapsed:
            return
        self._expanded_height = self.height()
        self._collapsed = True
        self._scroll.setVisible(False)
        self._footer.setVisible(False)
        # Collapsed to the title bar there is nothing to resize, and keeping
        # the strip would leave a grey band around the title.
        self._set_grips_visible(False)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(self.MIN_WIDTH, self.TITLE_HEIGHT + 2)
        self.resize(self.width(), self.TITLE_HEIGHT + 2)
        self._sync_maximise_button()

    def expand(self) -> None:
        """Restore from the collapsed state, back to the previous size."""
        if not self._collapsed:
            return
        self._collapsed = False
        self._scroll.setVisible(True)
        self._footer.setVisible(self._footer_has_widgets)
        self._outer.setContentsMargins(self.EDGE, self.EDGE, self.EDGE, self.EDGE)
        # A maximised panel fills the window and its edges are disabled, so
        # there is nothing for the handles to do.
        self._set_grips_visible(not self._maximised)
        self._place_grips()
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        if self._maximised:
            self.maximise()
        else:
            target_h = max(self.MIN_HEIGHT, self._expanded_height)
            self.resize(self.width(), target_h)
            self.clamp_inside_parent()
        self._sync_maximise_button()

    def toggle_maximised(self) -> None:
        """Maximise when open, restore when maximised, restore when collapsed.

        All three behaviours are expected of the same title-bar button, so the
        collapsed case is handled first: pressing it while minimised reopens
        the panel at the size it had before minimising.
        """
        if self._collapsed:
            self.expand()
            return
        if self._maximised:
            self.restore()
        else:
            self.maximise()

    def maximise(self) -> None:
        """Fill the main window, still as an in-app panel."""
        parent = self.parentWidget()
        if parent is None:
            return
        if not self._maximised:
            # Remember the user's size once; re-maximising must not overwrite
            # it with the maximised geometry.
            self._restore_geometry = self.geometry()
        self._maximised = True
        self._set_grips_visible(False)
        self.setMaximumSize(16777215, 16777215)
        self.setGeometry(
            self.MARGIN,
            self.MARGIN,
            max(self.MIN_WIDTH, parent.width() - 2 * self.MARGIN),
            max(self.MIN_HEIGHT, parent.height() - 2 * self.MARGIN),
        )
        self._sync_maximise_button()

    def restore(self) -> None:
        self._maximised = False
        self._set_grips_visible(True)
        self.setMaximumSize(16777215, 16777215)
        if self._restore_geometry is not None:
            self.setGeometry(self._restore_geometry)
        else:
            self.resize(420, 560)
            self.place_top_right()
        self.clamp_inside_parent()
        self._sync_maximise_button()

    def close_panel(self) -> None:
        self.hide()

    def _sync_maximise_button(self) -> None:
        """Show the restore glyph whenever pressing it would shrink the panel."""
        self.title_bar.maximise_button.set_kind(
            "restore" if self._maximised else "maximise"
        )

    def place_top_right(self, margin: int = MARGIN) -> None:
        """Anchor to the top-right of the *work area*, under the fixed chrome.

        Not the top-right of the window: the project tab row lives there now,
        and a panel that lands on top of it hides every open board but the
        first -- which is the one piece of navigation the user needs while
        working in the panel. The offset is whatever the chrome above the
        canvas actually measures, so it stays right if the banner wraps to two
        lines on a narrow window.
        """
        parent = self.parentWidget()
        if parent is None:
            return
        top = max(margin, chrome_height(parent) + margin)
        self.move(max(0, parent.width() - self.width() - margin), top)

    def clamp_inside_parent(self) -> None:
        """Keep the panel fully inside the application window."""
        parent = self.parentWidget()
        if parent is None:
            return
        x = max(0, min(self.x(), max(0, parent.width() - self.width())))
        y = max(0, min(self.y(), max(0, parent.height() - self.height())))
        self.move(x, y)

    # ------------------------------------------------------------ resizing

    def _edge_at(self, pos: QPoint) -> str:
        """Which edge or corner the point is over, as ``n``/``s``/``e``/``w``."""
        if self._maximised or self._collapsed:
            return ""
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        edge = ""
        if y <= self.GRIP:
            edge += "n"
        elif y >= h - self.GRIP:
            edge += "s"
        if x <= self.GRIP:
            edge += "w"
        elif x >= w - self.GRIP:
            edge += "e"
        return edge

    @staticmethod
    def _cursor_for(edge: str):
        return {
            "n": Qt.SizeVerCursor,
            "s": Qt.SizeVerCursor,
            "e": Qt.SizeHorCursor,
            "w": Qt.SizeHorCursor,
            "ne": Qt.SizeBDiagCursor,
            "sw": Qt.SizeBDiagCursor,
            "nw": Qt.SizeFDiagCursor,
            "se": Qt.SizeFDiagCursor,
        }.get(edge, Qt.ArrowCursor)

    # The resize itself lives here rather than in the mouse handlers, because
    # two things drive it: the edges of this widget and the visible corner
    # grip. One implementation means dragging either one lands on exactly the
    # same geometry rules.

    def begin_resize(self, edge: str, global_pos: QPoint) -> None:
        self._resize_edge = edge
        self._resize_press = global_pos
        self._resize_origin = self.geometry()

    def is_resizing(self) -> bool:
        return bool(self._resize_edge and self._resize_press and self._resize_origin)

    def apply_resize(self, global_pos: QPoint) -> None:
        """Resize towards a pointer position, honouring the floor and parent."""
        if not self.is_resizing():
            return
        delta = global_pos - self._resize_press
        origin = self._resize_origin
        left, top = origin.left(), origin.top()
        width, height = origin.width(), origin.height()

        if "e" in self._resize_edge:
            width = origin.width() + delta.x()
        if "w" in self._resize_edge:
            width = origin.width() - delta.x()
            left = origin.left() + delta.x()
        if "s" in self._resize_edge:
            height = origin.height() + delta.y()
        if "n" in self._resize_edge:
            height = origin.height() - delta.y()
            top = origin.top() + delta.y()

        # Hold the floor, and push back when a side would leave the parent.
        parent = self.parentWidget()
        if width < self.MIN_WIDTH:
            if "w" in self._resize_edge:
                left -= self.MIN_WIDTH - width
            width = self.MIN_WIDTH
        if height < self.MIN_HEIGHT:
            if "n" in self._resize_edge:
                top -= self.MIN_HEIGHT - height
            height = self.MIN_HEIGHT
        if parent is not None:
            if left < 0:
                width += left
                left = 0
            if top < 0:
                height += top
                top = 0
            if left + width > parent.width():
                width = parent.width() - left
            if top + height > parent.height():
                height = parent.height() - top

        self.setGeometry(
            left, top, max(self.MIN_WIDTH, width), max(self.MIN_HEIGHT, height)
        )

    def end_resize(self) -> None:
        self._resize_edge = ""
        self._resize_press = None
        self._resize_origin = None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        edge = self._edge_at(event.position().toPoint())
        if edge and event.button() == Qt.LeftButton:
            self.begin_resize(edge, event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.is_resizing():
            self.apply_resize(event.globalPosition().toPoint())
            event.accept()
            return

        self._apply_edge_cursor(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.is_resizing():
            self.end_resize()
            self._apply_edge_cursor(event.position().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _apply_edge_cursor(self, pos: QPoint) -> None:
        """Show a resize cursor on the border, and nothing anywhere else.

        ⚠️ **The fix is ``unsetCursor``, not ``setCursor(Qt.ArrowCursor)``.**
        Both make the panel itself read as an arrow, so both look right when you
        check ``panel.cursor()`` -- which is why the first version of this was
        wrong and the first version of its test passed anyway.

        What actually happens with an explicit arrow: Qt resolves a child's
        cursor by walking *up* the parent chain until it finds a widget that has
        one set. An explicit ``ArrowCursor`` **is** a set cursor, so it satisfies
        that walk -- and every widget inside the panel which never set a cursor
        of its own (a group box, a plain container) inherits the arrow from the
        panel instead of falling through to the default. That is harmless on its
        own, but the *previous* cursor is the problem: while the pointer is on
        the border the panel holds e.g. ``SizeHorCursor``, and the moment the
        pointer crosses into the content the panel stops receiving moves, so its
        stale resize cursor keeps being inherited by everything inside.

        ``unsetCursor`` removes the panel from that walk entirely, so each child
        resolves its own cursor -- a grip its arrow, a button its pointing hand,
        a text box its I-beam, and everything else the plain default.
        """
        edge = self._edge_at(pos)
        if edge == self._cursor_edge:
            return
        self._cursor_edge = edge
        if edge:
            self.setCursor(self._cursor_for(edge))
        else:
            self.unsetCursor()

    def clear_border_cursor(self) -> None:
        """Drop a resize cursor that is still set after the pointer left.

        ⚠️ **Needed because the panel only sees moves while the pointer is over
        the panel itself.** Once the pointer crosses into the content, the move
        events go to the child widget and the panel's ``mouseMoveEvent`` never
        runs -- so if the panel was left holding a resize cursor from the border
        it keeps holding it, and every child that has no cursor of its own
        inherits the arrow.

        Whoever sees the pointer move anywhere else in the window calls this, so
        the stale cursor is cleared as soon as the pointer is known to be off
        the panel.
        """
        if self._cursor_edge or self.cursor().shape() != Qt.ArrowCursor:
            self._cursor_edge = ""
            self.unsetCursor()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        # Application-wide pointer moves: clear a border cursor once the pointer
        # has moved onto the *content*. The panel cannot do this itself, because
        # the moment the pointer leaves the border the events go to a child and
        # the panel's own mouseMoveEvent stops running -- which is exactly how a
        # resize cursor ended up stuck across the whole panel body.
        #
        # ``watched is not self`` matters: an event delivered to the panel is
        # the border case, and must be left to the panel's own handler or the
        # cursor would be cleared in the same breath it was set.
        if (
            event.type() == event.Type.MouseMove
            and getattr(self, "_pointer_watch_installed", False)
            and watched is not self
            and not isinstance(watched, ResizeGrip)
            and self._cursor_edge
            and not self.is_resizing()
        ):
            local = self._local_point(event)
            # Two cases both mean "the border cursor is no longer wanted":
            # a child inside the panel took the move (the pointer is over the
            # content), or the move happened outside the panel entirely.
            on_content = isinstance(watched, QWidget) and self.isAncestorOf(watched)
            off_panel = local is None or not self.rect().contains(local)
            if on_content or off_panel:
                self.clear_border_cursor()

        # Follow the application window: a maximised panel keeps filling it and
        # a floating one keeps its distance from the right edge.
        if watched is self.parentWidget() and event.type() == event.Type.Resize:
            if self._maximised:
                self.setGeometry(
                    self.MARGIN,
                    self.MARGIN,
                    max(self.MIN_WIDTH, watched.width() - 2 * self.MARGIN),
                    max(self.MIN_HEIGHT, watched.height() - 2 * self.MARGIN),
                )
            else:
                self.clamp_inside_parent()
                self.place_top_right()
        return super().eventFilter(watched, event)

    def _local_point(self, event) -> QPoint | None:
        """Where a mouse event happened, in this panel's own coordinates."""
        try:
            return self.mapFromGlobal(event.globalPosition().toPoint())
        except AttributeError:  # pragma: no cover - event without a position
            return None

    def watch_pointer(self, app) -> None:
        """Clear a lingering border cursor when the pointer leaves the panel.

        The panel's own ``mouseMoveEvent`` only fires while the pointer is over
        the panel; the moment it crosses into the content, the child widget
        receives the move and the panel hears nothing more. Without this, a
        resize cursor picked up on the border stays set -- and every child
        without a cursor of its own inherits it.

        An application-wide event filter is the smallest hook that sees every
        move regardless of which widget consumes it. The filter only inspects
        the position and clears this panel's own cursor, so it is cheap and
        cannot swallow the event.

        ⚠️ **The filter holds this panel alive and is called for every event in
        the application**, so it has to be removed when the panel goes away.
        It is not a leak in normal use -- a panel lives as long as its window --
        but a long-lived process that opens and closes windows would accumulate
        filters, and every one of them runs on every event. ``destroyed`` is the
        hook that cannot be forgotten, because Qt emits it however the widget
        dies.
        """
        self._pointer_watch_app = app
        app.installEventFilter(self)
        self._pointer_watch_installed = True
        self.destroyed.connect(self._drop_pointer_watch)

    def _drop_pointer_watch(self, *_args) -> None:
        """Detach the application-wide filter when this panel is destroyed."""
        app = getattr(self, "_pointer_watch_app", None)
        if app is None or not getattr(self, "_pointer_watch_installed", False):
            return
        try:
            app.removeEventFilter(self)
        except (RuntimeError, TypeError):
            # Either side already gone; nothing left to detach.
            pass
        self._pointer_watch_installed = False



class ImageView(QWidget):
    """Zoomable, pannable view of one image.

    Zoom follows the pointer, so whatever detail is under the cursor stays
    under it while scaling -- the behaviour every image viewer has, and what
    makes inspecting a line drawing practical.

    Scale is set to fit when a new image arrives, then left alone: replacing
    the image during tuning must not throw away the zoom the user chose.
    """

    scale_changed = Signal(float)

    MIN_SCALE = 0.02
    MAX_SCALE = 32.0

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._pan_from: QPointF | None = None
        self._pan_offset: QPointF | None = None
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("滚轮缩放　·　拖动平移　·　双击适应窗口")

    # ------------------------------------------------------------ contents

    def has_image(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def set_image(self, image: QImage, keep_view: bool = True) -> None:
        first = not self.has_image()
        self._image = image
        if first or not keep_view:
            self.fit()
        else:
            self._clamp()
        self.update()

    def clear(self) -> None:
        self._image = None
        self.update()

    # -------------------------------------------------------------- zoom

    def fit(self) -> None:
        if not self.has_image():
            return
        width = max(1, self.width())
        height = max(1, self.height())
        self._scale = min(
            width / self._image.width(), height / self._image.height()
        ) * 0.97
        self._center()
        self._emit_scale()
        self.update()

    def actual_size(self) -> None:
        """Show one image pixel per screen pixel."""
        if not self.has_image():
            return
        self._scale = 1.0
        self._center()
        self._emit_scale()
        self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        if not self.has_image():
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)

        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
        if abs(new_scale - self._scale) < 1e-9:
            return

        # Keep the image point currently under `anchor` exactly there.
        image_point = (anchor - self._offset) / self._scale
        self._scale = new_scale
        self._offset = anchor - image_point * self._scale
        self._clamp()
        self._emit_scale()
        self.update()

    def _emit_scale(self) -> None:
        self.scale_changed.emit(self._scale)

    def _center(self) -> None:
        width = float(self.width())
        height = float(self.height())
        self._offset = QPointF(
            (width - self._image.width() * self._scale) / 2.0,
            (height - self._image.height() * self._scale) / 2.0,
        )

    def _clamp(self) -> None:
        """Centre an axis that fits, otherwise keep the image on screen."""
        if not self.has_image():
            return
        width = float(self.width())
        height = float(self.height())
        image_w = self._image.width() * self._scale
        image_h = self._image.height() * self._scale

        if image_w <= width:
            self._offset.setX((width - image_w) / 2.0)
        else:
            self._offset.setX(min(0.0, max(width - image_w, self._offset.x())))

        if image_h <= height:
            self._offset.setY((height - image_h) / 2.0)
        else:
            self._offset.setY(min(0.0, max(height - image_h, self._offset.y())))

    # ------------------------------------------------------------- events

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.18**steps, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self.has_image():
            self._pan_from = event.position()
            self._pan_offset = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_from is not None and self._pan_offset is not None:
            delta = event.position() - self._pan_from
            self._offset = self._pan_offset + delta
            self._clamp()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._pan_from is not None:
            self._pan_from = None
            self._pan_offset = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.has_image():
            self._clamp()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f8fa"))

        if not self.has_image():
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "载入图片后在此查看")
            return

        target = QRectF(
            self._offset.x(),
            self._offset.y(),
            self._image.width() * self._scale,
            self._image.height() * self._scale,
        )
        # Smooth when shrinking (less aliasing), sharp when magnifying so the
        # pixel structure of a line drawing stays legible.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self._scale < 1.0)
        painter.drawImage(target, self._image)

        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setPen(QColor(BORDER))
        painter.drawRect(target)

        painter.setPen(QColor(MUTED))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QRectF(6, self.height() - 20, self.width() - 12, 16),
            Qt.AlignLeft,
            f"{self._scale * 100:.0f}%",
        )


class PreviewPanel(FloatingPanel):
    """A window onto the processed image, for inspecting detail.

    Deliberately separate from the upload box. That box is a control fixed in
    the settings layout and sized to the panel; this is an independent floating
    window that can be moved, resized and zoomed into -- the same mechanics as
    the preprocessing panel itself.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, "图像预览")
        self.resize(620, 760)
        self._image_size: tuple[int, int] | None = None
        self._is_full = False
        self._note = ""
        self._recompute = None

        layout = self.content_layout
        layout.setSpacing(4)

        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(6)

        self.fit_button = QPushButton("适应窗口")
        self.fit_button.setFixedHeight(24)
        self.actual_button = QPushButton("1:1")
        self.actual_button.setFixedWidth(52)
        self.actual_button.setFixedHeight(24)
        self.full_button = QPushButton("全分辨率")
        self.full_button.setFixedHeight(24)
        self.full_button.setToolTip("按原始尺寸重算；拖动滑块时为快速预览")
        bar_layout.addWidget(self.fit_button)
        bar_layout.addWidget(self.actual_button)
        bar_layout.addWidget(self.full_button)
        bar_layout.addStretch(1)

        self.info_label = QLabel("尚未载入")
        muted(self.info_label)
        bar_layout.addWidget(self.info_label)
        layout.addWidget(bar)

        self.view = ImageView()
        layout.addWidget(self.view, 1)

        self.fit_button.clicked.connect(self.view.fit)
        self.actual_button.clicked.connect(self.view.actual_size)
        self.full_button.clicked.connect(self._on_recompute)
        self.view.scale_changed.connect(self._on_scale)

    def set_recompute_callback(self, callback) -> None:
        self._recompute = callback

    def _on_recompute(self) -> None:
        if self._recompute is not None:
            self._recompute()

    def _on_scale(self, scale: float) -> None:
        if self._image_size is None:
            return
        width, height = self._image_size
        quality = "全分辨率" if self._is_full else "快速预览"
        suffix = f"　·　{self._note}" if self._note else ""
        self.info_label.setText(
            f"{width} × {height} px　·　{quality}　·　显示 {scale * 100:.0f}%{suffix}"
        )
    def show_array(
        self,
        array: np.ndarray,
        size: tuple[int, int],
        is_full_resolution: bool = False,
        note: str = "",
    ) -> None:
        """Display an image, keeping the user's zoom if one was already set."""
        self._image_size = size
        self._is_full = is_full_resolution
        self._note = note
        self.view.set_image(ndarray_to_qimage(array), keep_view=self.view.has_image())
        self._on_scale(self.view._scale)


class PreprocessPanel(FloatingPanel):
    """Image preprocessing as a panel floating inside the main window.

    Only the colour-style rule is live: choosing the colour style pins the
    artwork to a silkscreen layer and disables the other combinations. Every
    other control is a placeholder, as is the denoise section, which waits for
    the PCB_lightgraph integration.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, "图像预处理")
        # No setStyleSheet() here on purpose: a stylesheet set on this widget
        # replaces the panel's own rule outright, which silently removed the
        # rounded corners and border. Text colour comes from the application
        # stylesheet instead.
        self._source_bgr: np.ndarray | None = None
        self._source_alpha: np.ndarray | None = None
        self._source_path: str | None = None
        self._source_size: tuple[int, int] | None = None
        self._preview_panel: PreviewPanel | None = None
        self._inline_processed: np.ndarray | None = None
        self._full_processed: np.ndarray | None = None
        self._published_digest: str | None = None
        """Hash of the line art already written to disk.

        The canvas is republished on every slider tick, and most ticks produce
        the same pixels; this skips the rewrite when nothing changed.
        """

        self._published_digests: dict[str, str] = {}
        """Per-file hashes for 板画, which writes two files rather than one.

        A single hash cannot serve both tones: the white and black images
        differ, so whichever was written second would look like a change to the
        first and force a rewrite on every tick.
        """

        self._imported_path: str | None = None
        """The picture currently on the canvas, if any.

        Uploading and importing are separate steps, so "there is a source
        loaded" is not the same as "the board has this picture". Presentation
        changes must only rearrange a picture that was actually imported, or
        choosing a layer would quietly import one the user has not committed.
        """

        self._import_id: str = ""
        """Identifies the objects created by the *last* import.

        Every press of 导入图片 gets a fresh id, so the same file can be
        imported again with different settings and exist alongside the first
        import instead of taking it over. Empty until something is imported.
        """

        self._two_tone_source: bool = False
        """Whether the loaded picture was detected as two-colour."""

        self._soft_shaded_source: bool = False
        """Whether the loaded picture looked softly shaded, so 降噪 was ticked
        for it. Recorded so the panel can say why it changed a setting on its
        own rather than leaving the user to notice."""

        self._committed_style: str = ""
        self._raw_snapped: bool = False
        """Whether the raw (unprocessed) import was snapped to pure black and
        white. Only a two-colour picture is, and only then is there anything to
        publish: a colour picture left unprocessed goes to the board exactly as
        the file has it."""

        self._trace_cache = None
        """Keeps the picture-derived half of the line-art pass between ticks.

        A slider drag recomputes the conversion on every step, and most of that
        work -- compositing, desaturating, building the subject mask, reading
        the Otsu level -- is a function of the picture and not of any setting on
        the panel. Measured on 2000 px library artwork: 3.4 s per tick without
        this, about 0.11 s with it. That difference is the "调参数非常卡" the
        user reported. See :class:`pcbart.lineart.TraceCache`."""

        self._preview_timer: QTimer | None = None
        """Coalesces a burst of slider values into one conversion.

        Each step of a drag emits its own signal, and a preview takes long
        enough that computing every step queues work faster than it drains --
        the window then lags behind the handle. Restarting a short single-shot
        timer per step means only the value the user stopped on is computed."""

        from pcbart.lineart import TraceCache

        self._trace_cache = TraceCache()

        self._source_description: str = ""
        """The picture's own description, without the board suffix.

        Kept so the label can be rebuilt from scratch each time instead of
        having its previous text parsed back out.
        """
        """The 风格 the current elements were imported with.

        Kept separate from the combo box, which now describes the *next*
        import. Reading the combo while republishing line art would let a
        pending 彩色丝印 selection strip the line art off an element that was
        imported as 线稿 and has not been re-imported.
        """

        layout = self.content_layout

        upload = section("1 · 上传图片", compact=True)
        upload_layout = QVBoxLayout(upload)
        upload_layout.setContentsMargins(0, 0, 0, 0)
        self.drop_zone = DropZone()
        self.drop_zone.image_selected.connect(self.load_image)
        upload_layout.addWidget(self.drop_zone)
        self.source_note = QLabel("尚未载入图片")
        self.source_note.setWordWrap(True)
        muted(self.source_note)
        upload_layout.addWidget(self.source_note)
        layout.addWidget(upload, 1)

        # 处理图像 gates everything below it. A picture that is already flat
        # black and white does not need tracing -- and tracing it is not
        # harmless: the line-art pass thickens strokes and adds a silhouette
        # ring the artwork never had. The switch is turned off for such a
        # picture, so pressing 导入图片 puts the artwork on the board as it is.
        process_group = section("2 · 处理图像", compact=True)
        process_layout = QVBoxLayout(process_group)
        process_layout.setContentsMargins(0, 0, 0, 0)
        process_layout.setSpacing(2)
        self.process_check = QCheckBox("处理图像（按下面的风格与参数）")
        self.process_check.setChecked(True)
        self.process_check.setToolTip(
            "勾选：先按下面的风格与参数处理，再放到画板上；取消：原样放置。"
        )
        self.process_check.stateChanged.connect(self._on_process_toggled)
        process_layout.addWidget(self.process_check)
        self.process_note = QLabel()
        self.process_note.setWordWrap(True)
        muted(self.process_note)
        self.process_note.setVisible(False)
        process_layout.addWidget(self.process_note)
        layout.addWidget(process_group)
        self.process_group = process_group

        style_group = section("3 · 风格", compact=True)
        style_layout = QVBoxLayout(style_group)
        style_layout.setContentsMargins(0, 0, 0, 0)
        style_layout.setSpacing(2)
        self.style_combo = combo([name for name, _ in STYLE_OPTIONS])
        style_layout.addWidget(row("风格", self.style_combo))
        layout.addWidget(style_group)
        self.style_group = style_group

        layer_group = section("4 · 呈现方式", compact=True)
        layer_layout = QVBoxLayout(layer_group)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.setSpacing(2)

        self.side_control = segmented(["顶层", "底层"])
        self.side_row = row("板面", self.side_control)
        layer_layout.addWidget(self.side_row)
        for button in self.side_control.buttons:  # type: ignore[attr-defined]
            button.clicked.connect(self._on_side_changed)

        # 板画's own side choice, shown in place of the two rows above. Three
        # options rather than the top/bottom pair because a painting can go on
        # both faces at once, which the pair cannot express.
        self.board_art_side_combo = combo([label for _key, label, _hint in BOARD_ART_SIDE_OPTIONS])
        self.board_art_side_row = row("板面", self.board_art_side_combo)
        self.board_art_side_row.setVisible(False)
        layer_layout.addWidget(self.board_art_side_row)
        self.board_art_side_combo.currentIndexChanged.connect(self._on_board_art_side_changed)

        self.layer_combo = combo([])
        self.presentation_row = row("呈现", self.layer_combo)
        layer_layout.addWidget(self.presentation_row)
        self.layer_group = layer_group
        layout.addWidget(layer_group)

        # The tuning panel: every control here drives a live preview, so the
        # effect of a change is visible immediately instead of after a round
        # trip through the editor.
        tune = section("5 · 线稿调试", compact=True)
        self.tuning_section = tune
        tune_layout = QVBoxLayout(tune)
        tune_layout.setContentsMargins(0, 0, 0, 0)
        tune_layout.setSpacing(3)

        # 读法 first: it changes which of the controls below mean anything, so
        # offering them in the other order invites the user to tune a slider
        # that the current reading ignores -- the "dead control" problem this
        # panel has already been reported for twice.
        self.reading_combo = combo([label for _key, label, _hint in READING_OPTIONS])
        self.reading_combo.currentIndexChanged.connect(self._on_reading_changed)
        tune_layout.addWidget(row("读法", self.reading_combo))

        self.strength_slider = NoWheelSlider(Qt.Horizontal)
        self.strength_slider.setRange(20, 400)
        self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(self._on_tuning_changed)
        self.strength_row = row(
            "线条粗细", self.strength_slider, "纯线条读法下的笔画粗细；1.00 为标定值"
        )
        tune_layout.addWidget(self.strength_row)

        self.threshold_slider = NoWheelSlider(Qt.Horizontal)
        self.threshold_slider.setRange(2, 98)
        self.threshold_slider.setValue(46)
        self.threshold_slider.valueChanged.connect(self._on_tuning_changed)
        self.threshold_row = row(
            "墨量阈值",
            self.threshold_slider,
            "纯线条读法下越高墨越多、细碎线也越多；输出恒为纯黑白",
        )
        tune_layout.addWidget(self.threshold_row)

        # The contour controls, shown only for the 描边读法. The width is a
        # millimetre value because that is what the board sees; the detail
        # scale is a picture fraction because it selects what counts as a line.
        self.outline_width_slider = NoWheelSlider(Qt.Horizontal)
        self.outline_width_slider.setRange(10, 100)
        self.outline_width_slider.setValue(20)
        self.outline_width_slider.valueChanged.connect(self._on_tuning_changed)
        self.outline_width_row = row(
            "描边线宽",
            self.outline_width_slider,
            "描边在成品上的宽度（0.10–1.00 mm）；低于 0.10 mm 嘉立创无法生产",
        )
        tune_layout.addWidget(self.outline_width_row)

        self.outline_detail_slider = NoWheelSlider(Qt.Horizontal)
        self.outline_detail_slider.setRange(4, 60)
        self.outline_detail_slider.setValue(16)
        self.outline_detail_slider.valueChanged.connect(self._on_tuning_changed)
        self.outline_detail_row = row(
            "细节尺度",
            self.outline_detail_slider,
            "越小细线越清楚，也越容易把纹理当线；越大越干净",
        )
        tune_layout.addWidget(self.outline_detail_row)

        self.outline_density_slider = NoWheelSlider(Qt.Horizontal)
        self.outline_density_slider.setRange(10, 250)
        self.outline_density_slider.setValue(60)
        self.outline_density_slider.valueChanged.connect(self._on_tuning_changed)
        self.outline_density_row = row(
            "描边密度",
            self.outline_density_slider,
            "保留多少轮廓：小=只留最明显的边，大=细节更多、墨也更多",
        )
        tune_layout.addWidget(self.outline_density_row)

        self.tune_note = QLabel()
        self.tune_note.setWordWrap(True)
        muted(self.tune_note)
        self.tune_note.setVisible(False)
        tune_layout.addWidget(self.tune_note)

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 2, 0, 0)
        preview_layout.setSpacing(6)
        self.preview_toggle = QCheckBox("显示处理结果")
        self.preview_toggle.setChecked(True)
        self.preview_toggle.stateChanged.connect(self._on_tuning_changed)
        preview_layout.addWidget(self.preview_toggle)
        preview_layout.addStretch(1)
        self.reset_button = QPushButton("恢复默认")
        self.reset_button.setFixedHeight(24)
        self.reset_button.clicked.connect(self._reset_tuning)
        preview_layout.addWidget(self.reset_button)
        tune_layout.addWidget(preview_row)

        layout.addWidget(tune)

        denoise = section("6 · 降噪", compact=True)
        self.denoise_section = denoise
        denoise_layout = QVBoxLayout(denoise)
        denoise_layout.setContentsMargins(0, 0, 0, 0)
        denoise_layout.setSpacing(3)

        self.denoise_check = QCheckBox("启用降噪（独立步骤，先于线稿提取）")
        self.denoise_check.setToolTip(
            "干净矢量图用不到，且会软化细节；扫描件与照片建议开启"
        )
        self.denoise_check.stateChanged.connect(self._on_tuning_changed)
        denoise_layout.addWidget(self.denoise_check)

        self.denoise_slider = NoWheelSlider(Qt.Horizontal)
        self.denoise_slider.setRange(3, 31)
        self.denoise_slider.setValue(5)
        self.denoise_slider.valueChanged.connect(self._on_tuning_changed)
        denoise_layout.addWidget(
            row("模糊核", self.denoise_slider, "越大去噪越强，细节损失也越多")
        )

        self.despeckle_slider = NoWheelSlider(Qt.Horizontal)
        self.despeckle_slider.setRange(0, 60)
        self.despeckle_slider.setValue(0)
        self.despeckle_slider.valueChanged.connect(self._on_tuning_changed)
        denoise_layout.addWidget(
            row("去杂点", self.despeckle_slider, "清除小于该像素数的孤立墨点（0=关）")
        )

        silhouette_row = QWidget()
        silhouette_layout = QHBoxLayout(silhouette_row)
        silhouette_layout.setContentsMargins(0, 2, 0, 0)
        silhouette_layout.setSpacing(6)
        self.silhouette_check = QCheckBox("补主体外轮廓")
        # ⚠️ **Off by default, at the user's request.** This draws a contour
        # around the subject's outer edge. It helps a cut-out figure read as a
        # single shape, but it is also ink the picture does not have -- and on
        # an illustration whose own outline is already closed it only thickens
        # the silhouette. 「补主体外轮廓」 is therefore an opt-in.
        #
        # It affects **only the line-art image**. An irregular board outline is
        # cut from ``image_prep.subject_mask`` (see ``pcbart.compose``), which
        # does not read this flag, so switching it off cannot change the board.
        self.silhouette_check.setChecked(False)
        self.silhouette_check.setToolTip(
            "线稿算法看不到主体的外边界（浅色主体对浅色背景时尤其明显），"
            "勾选后补上主体外边界。只影响线稿图案；异形板框走自己的主体掩膜。"
        )
        self.silhouette_check.stateChanged.connect(self._on_tuning_changed)
        silhouette_layout.addWidget(self.silhouette_check)
        self.silhouette_slider = NoWheelSlider(Qt.Horizontal)
        self.silhouette_slider.setRange(1, 31)
        self.silhouette_slider.setValue(5)
        self.silhouette_slider.valueChanged.connect(self._on_tuning_changed)
        self.silhouette_slider.setFixedWidth(110)
        silhouette_layout.addWidget(self.silhouette_slider)
        silhouette_layout.addStretch(1)
        denoise_layout.addWidget(silhouette_row)

        # 弧线平滑 is its own switch, separate from every style, because it is
        # the one step whose cost is measured in seconds rather than
        # milliseconds. See `clean_strokes_with_arcs` for the trade.
        self.arc_smooth_check = QCheckBox("弧线平滑（清毛刺）")
        self.arc_smooth_check.setToolTip(
            "线稿的锯齿边缘重描成平滑曲线；\n"
            "实测碎块减少 23–43%，细节总量基本不变；\n"
            "这一步要跑几秒，只在需要时打开。"
        )
        self.arc_smooth_check.stateChanged.connect(self._on_tuning_changed)
        denoise_layout.addWidget(self.arc_smooth_check)

        layout.addWidget(denoise)

        self.board_art_section = self._build_board_art_section()
        layout.addWidget(self.board_art_section)

        # The preview window belongs to every style that processes a picture,
        # so it sits outside 线稿调试. It used to live inside that section,
        # which meant picking 板画 hid the only way to open it -- the styles
        # that change the picture most had no way to inspect the result.
        preview_group = section("8 · 预览", compact=True)
        self.preview_section = preview_group
        preview_layout_group = QVBoxLayout(preview_group)
        preview_layout_group.setContentsMargins(0, 0, 0, 0)
        preview_layout_group.setSpacing(3)
        self.open_preview_button = QPushButton("打开大图预览")
        self.open_preview_button.setMinimumHeight(28)
        self.open_preview_button.clicked.connect(self.open_preview_window)
        preview_layout_group.addWidget(self.open_preview_button)
        layout.addWidget(preview_group)

        self.redraw_section = self._build_redraw_section()
        layout.addWidget(self.redraw_section)

        # Importing is a deliberate step, so it gets its own action. This is
        # the click that turns the preview into board geometry, so it is pinned
        # to the bottom of the panel rather than living in the scrolling area
        # where it sat 380 px below the fold.
        self.import_button = QPushButton("导入图片")
        self.import_button.setMinimumHeight(32)
        self.import_button.setEnabled(False)
        self.import_button.setToolTip("载入图片后可导入")
        themed(
            self.import_button,
            lambda: f"""
                            QPushButton {{
                                background: {ACCENT}; color: white; border: none;
                                border-radius: 3px; font-weight: 600;
                            }}
                            QPushButton:hover {{ background: #1d4ed8; }}
                            QPushButton:pressed {{ background: #1e40af; }}
                            QPushButton:disabled {{
                                background: #c3ccd8; color: #f2f4f7;
                            }}
            """,
        )
        self.import_button.clicked.connect(self.import_to_canvas)

        self.import_note = QLabel()
        self.import_note.setWordWrap(True)
        muted(self.import_note)
        self.import_note.setVisible(False)
        self.set_footer(self._build_import_footer())

        # Changing 风格 moves the imported picture to the layers that choice
        # names, and decides which of the controls below are live.
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        self._refresh_layers()

    def _build_redraw_section(self) -> QGroupBox:
        """The entry point for the optional, externally supplied model.

        The redraw reading needs weights that are deliberately **not** shipped
        (183 MB against a 178 MB installer, and a CPU run peaks at 4.6 GB), so
        the capability has to advertise itself or nobody would know it exists.
        This section is that advertisement, and it is honest about which of the
        three things it needs is missing: the model folder, or PyTorch, or
        nothing.
        """
        box = section("9 · 线稿重绘（外接模型）", compact=True)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.redraw_status = QLabel()
        self.redraw_status.setWordWrap(True)
        layout.addWidget(self.redraw_status)

        self.redraw_button = QPushButton("外接模型设置与说明…")
        self.redraw_button.setMinimumHeight(28)
        self.redraw_button.setToolTip(
            "重绘用外接 Anime2Sketch 模型；这里查看状态、指定模型文件夹"
        )
        self.redraw_button.clicked.connect(self._on_external_redraw)
        layout.addWidget(self.redraw_button)
        self._refresh_redraw_section()
        return box

    def _redraw_status(self) -> tuple[str, str]:
        """``(tone, sentence)`` for the external-model section.

        Split out from the widget so the wording is testable without a window,
        and so the three states are written in one place: ready, model missing,
        runtime missing.
        """
        from pcbart.model_paths import (
            DEFAULT_VARIANT,
            check_model_dir,
            resolve_model_dir,
            torch_available,
        )

        directory = resolve_model_dir()
        report = check_model_dir(directory, DEFAULT_VARIANT)
        if not report["ok"]:
            return (
                "warning",
                f"⚠️ 尚未安装：缺少 {'；'.join(report['missing'])}。"
                f"当前文件夹：{directory}",
            )
        if not torch_available():
            return (
                "warning",
                f"⚠️ 模型已就位，但缺少运行库 PyTorch："
                f"pip install torch\n模型文件夹：{directory}",
            )
        return "success", f"✅ 已就绪（{DEFAULT_VARIANT}）：{directory}"

    def _refresh_redraw_section(self) -> None:
        tone_name, sentence = self._redraw_status()
        self.redraw_status.setText(sentence)
        tone(self.redraw_status, tone_name)

    def _on_external_redraw(self) -> None:
        """Explain the external model and let the user point at one.

        Deliberately a dialog rather than a silent jump to the settings panel:
        the model is unusual enough (an optional ~190 MB download the installer
        does not carry) that a user meeting this button needs the *why* as well
        as the where.

        ⚠️ **The button does not claim to redraw anything.** The workspace for
        it is still a standalone tool, so the dialog says that plainly and
        names the command. Offering an "开始重绘" button here would be the
        placeholder problem this project has been burned by: a control that
        looks like it runs the feature and does not.
        """
        lines = [
            "线稿重绘用 Anime2Sketch（MIT）模型**重新画线稿**，不是在图上找边。"
            "填充读法糊成色块的细眉毛、链条、衣褶，模型能描成线条。",
            "",
            "模型不随软件分发：权重约 190 MB（安装包约 178 MB）；"
            "纯 CPU 跑一次约需 4.6 GB 内存，是否安装由你决定。",
            "",
            self._redraw_status()[1],
        ]
        if not self._redraw_ready():
            lines += [
                "",
                "获取：python tools\\fetch_anime2sketch.py",
                "也可在下面【选择模型文件夹…】里指向已下载好的那份。",
            ]
        else:
            lines += [
                "",
                "用法：在【重绘线稿（探针）】窗口里选图、调两个滑块，"
                "导出 PNG 后再用本窗口上传。",
                "启动：python tools\\line_redraw_panel.py",
            ]

        box = QMessageBox(self)
        box.setWindowTitle("线稿重绘（外接模型）")
        box.setIcon(QMessageBox.Information)
        box.setText("\n".join(lines).replace("**", ""))
        pick = box.addButton("选择模型文件夹…", QMessageBox.ActionRole)
        box.addButton("关闭", QMessageBox.RejectRole)
        box.exec()

        if box.clickedButton() is pick:
            self._open_model_settings()

    def _redraw_ready(self) -> bool:
        from pcbart.model_paths import DEFAULT_VARIANT, check_model_dir, resolve_model_dir, torch_available

        return (
            check_model_dir(resolve_model_dir(), DEFAULT_VARIANT)["ok"]
            and torch_available()
        )

    def _open_model_settings(self) -> None:
        """Open 界面设置 at the model section of this panel's window."""
        window = self.parentWidget() or self
        while window is not None and not hasattr(window, "design_canvas"):
            window = window.parentWidget()
        if window is None:
            return
        panel = show_settings(window)
        if panel is not None:
            panel.focus_model_section()

    def _build_board_art_section(self) -> QGroupBox:
        """The 板画 controls, hidden unless that style is selected.

        Shown only for 板画 rather than always visible: these knobs do nothing
        to a line drawing, and a control that silently has no effect is the
        placeholder problem this project keeps running into.

        Every slider is a *display* scale. The values the model really uses are
        floats and millimetres, so the widgets carry integers and the
        conversion happens in :meth:`_board_art_params`.
        """
        box = section("7 · 板画设置", compact=True)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.tone_slider = NoWheelSlider(Qt.Horizontal)
        self.tone_slider.setRange(30, 90)
        self.tone_slider.setValue(int(TONE_THRESHOLD * 100))
        self.tone_slider.valueChanged.connect(self._on_tuning_changed)
        layout.addWidget(
            row("明暗分界", self.tone_slider, "越低白色越多；0.62 为标定值")
        )

        self.outline_mm_slider = NoWheelSlider(Qt.Horizontal)
        self.outline_mm_slider.setRange(5, 150)
        self.outline_mm_slider.setValue(46)
        self.outline_mm_slider.valueChanged.connect(self._on_tuning_changed)
        layout.addWidget(
            row(
                "轮廓宽度",
                self.outline_mm_slider,
                "角色外圈白线的粗细，单位毫米；随成品尺寸换算成像素",
            )
        )

        self.min_feature_slider = NoWheelSlider(Qt.Horizontal)
        self.min_feature_slider.setRange(0, 50)
        self.min_feature_slider.setValue(10)
        self.min_feature_slider.valueChanged.connect(self._on_tuning_changed)
        layout.addWidget(
            row(
                "最小特征",
                self.min_feature_slider,
                "低于此宽度的线与点会被去掉；调低保细节，代价是细线可能印成毛边",
            )
        )

        options = QWidget()
        options_layout = QHBoxLayout(options)
        options_layout.setContentsMargins(0, 2, 0, 0)
        options_layout.setSpacing(10)

        self.background_check = QCheckBox("背景填黑")
        self.background_check.setChecked(True)
        self.background_check.setToolTip(
            "勾选：整块背景铺铜，做出实心暗场；"
            "取消：只画角色，背景留出板子本色"
        )
        self.background_check.stateChanged.connect(self._on_tuning_changed)
        options_layout.addWidget(self.background_check)

        self.flip_vertical_check = QCheckBox("上下翻转")
        self.flip_vertical_check.setToolTip("镜像上下；预览与落图同时生效")
        self.flip_vertical_check.stateChanged.connect(self._on_tuning_changed)
        options_layout.addWidget(self.flip_vertical_check)

        self.flip_horizontal_check = QCheckBox("左右翻转")
        self.flip_horizontal_check.setToolTip("镜像左右；预览与落图同时生效")
        self.flip_horizontal_check.stateChanged.connect(self._on_tuning_changed)
        options_layout.addWidget(self.flip_horizontal_check)

        self.invert_check = QCheckBox("反相")
        self.invert_check.setToolTip(
            "黑白对调：深色块面改印丝印、亮部与轮廓留成铜面，得到负片效果"
        )
        self.invert_check.stateChanged.connect(self._on_tuning_changed)
        options_layout.addWidget(self.invert_check)
        options_layout.addStretch(1)
        layout.addWidget(options)

        box.setVisible(False)
        return box

    def _build_import_footer(self) -> QWidget:
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(3)
        holder_layout.addWidget(self.import_button)
        holder_layout.addWidget(self.import_note)
        return holder

    # ------------------------------------------------------------ behaviour

    def open_preview_window(self) -> PreviewPanel:
        """Show the zoomable preview window and convert at full resolution.

        The inline box shows a bounded-size preview so slider drags stay
        responsive; this window is for inspecting detail, so it converts the
        original at full resolution. The radius rule makes the two agree.
        """
        panel = self._ensure_preview_panel()
        panel.show()
        panel.raise_()
        self.refresh_preview_full()

        if self._source_bgr is None:
            QMessageBox.information(
                self, "尚未载入图片", "请先在上传区域选择一张图片。"
            )
        return panel

    def refresh_preview_full(self) -> None:
        """Recompute the preview window's image at full resolution."""
        if self._source_bgr is None or self._preview_panel is None:
            return

        from pcbart.work_scale import prepare_working_image

        full_bgr, full_alpha, full_scale = prepare_working_image(
            self._source_bgr, self._source_alpha
        )
        start = time.perf_counter()
        try:
            processed = self._process(
                full_bgr, full_alpha, param_scale=full_scale
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            import traceback

            traceback.print_exc()
            QMessageBox.warning(
                self, "处理失败", f"无法生成全分辨率预览：\n{exc}"
            )
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._full_processed = processed
        display = self._display_image(processed)
        self._preview_panel.show_array(
            display,
            (display.shape[1], display.shape[0]),
            is_full_resolution=True,
            note=f"{elapsed_ms:.0f} ms",
        )

        # Hand the finished line art to the document so the canvas can show it
        # on its layer. Done here rather than on every slider tick: the canvas
        # should hold the picture the user settled on, not each intermediate
        # value the drag passed through.
        self._publish_line_art(processed)

    def _ensure_preview_panel(self) -> PreviewPanel:
        if self._preview_panel is None:
            parent = self.parentWidget() or self
            self._preview_panel = PreviewPanel(parent)
            self._preview_panel.resize(640, 780)
            self._preview_panel.set_recompute_callback(self.refresh_preview_full)
            self._preview_panel.place_top_right(PreviewPanel.MARGIN)
            self._preview_panel.show()
            self._preview_panel.raise_()
        return self._preview_panel

    def _update_preview_panel(self) -> None:
        """Push the current bounded preview to the window, if it is open.

        Slider drags push the fast preview so the window tracks the controls.
        Converting the original at full resolution on every tick would make
        dragging stutter, so the window offers a button for that instead.
        """
        if self._preview_panel is None or not self._preview_panel.isVisible():
            return
        if self._inline_processed is None:
            return
        # Through _display_image, not straight through: 板画 holds two masks
        # rather than a picture, and handing those to the window would draw the
        # tones instead of the artwork.
        display = self._display_image(self._inline_processed)
        height, width = display.shape[:2]
        self._preview_panel.show_array(
            display, (width, height), is_full_resolution=False
        )

    def _place_on_canvas(self, path: str, width_px: int, height_px: int) -> None:
        """Put the imported artwork on the canvas, on the chosen layers.

        Only called from 导入图片. Uploading deliberately stops at the preview,
        so the user's parameters are settled before any geometry exists.

        Choosing a board is a separate decision again: no outline is created
        here and no panel opens. An already-configured board is left alone --
        importing a picture must not silently throw away a board the user
        sized.
        """
        self._source_size = (width_px, height_px)
        self._imported_path = path
        # A fresh identity per import, so the same file imported twice produces
        # two independent elements rather than the second taking over the
        # first. See ImagePlacement.import_id.
        self._import_id = uuid.uuid4().hex[:8]
        # 处理图像 off means the board holds the picture itself, so the
        # committed "style" records that rather than a style the conversion
        # never ran. The publishing guard reads this, and treating a raw import
        # as a styled one would let a later slider tick repaint it.
        self._committed_style = (
            self._style() if self.process_check.isChecked() else RAW_STYLE
        )
        self._sync_placements()
        self._refresh_board_note()

    # ------------------------------------------------------ layer placement

    def _style(self) -> str:
        return STYLE_OPTIONS[self.style_combo.currentIndex()][0]

    def _presentation_keys(self) -> list[str]:
        """Which layers the current 风格 + 呈现方式 selection means.

        This is where "the picture goes on the layer picked in preprocessing"
        is actually decided. Two styles override the 呈现方式 list rather than
        being one of its entries:

        彩色丝印 pins the picture to one silkscreen layer, because colour can
        only be printed on silk. 板画 always needs *two* layers -- white on
        silk, black on copper -- so it offers its own 顶面/底面/两面 choice and
        both of its tones are returned here.
        """
        if self._style() == BOARD_ART_STYLE:
            keys = board_art_layer_keys(self._board_art_side())
            return keys["white"] + keys["black"]

        side = self._side()
        prefix = "top" if side == "顶层" else "bottom"
        if self._style() == COLOR_STYLE:
            return [f"{prefix}_silk"]

        options = layer_options(side)
        index = self.layer_combo.currentIndex()
        if not 0 <= index < len(options):
            index = 0
        return list(options[index][1])

    def _tone_by_layer_key(self) -> dict[str, str]:
        """Layer key to tone, for the style currently selected.

        Empty for every style but 板画. The pairing is fixed -- white is always
        the silkscreen copy and black always the copper one -- so the writer can
        rely on it rather than having to guess which two objects go together.
        """
        if self._style() != BOARD_ART_STYLE:
            return {}
        keys = board_art_layer_keys(self._board_art_side())
        return {key: "white" for key in keys["white"]} | {
            key: "black" for key in keys["black"]
        }

    def _board_art_side(self) -> str:
        """The 板画 side key: ``top``, ``bottom`` or ``both``."""
        index = self.board_art_side_combo.currentIndex() if hasattr(
            self, "board_art_side_combo"
        ) else 0
        if not 0 <= index < len(BOARD_ART_SIDE_OPTIONS):
            index = 0
        return BOARD_ART_SIDE_OPTIONS[index][0]

    def _belongs_to_import(self, placement) -> bool:
        """Whether a placement was created by the import currently being edited.

        Matched on the import's identity, **not** on the file path. Matching by
        path made every object from the same file look like part of one import,
        so importing a picture a second time -- say once as silk and once as
        colour silk -- moved or deleted the first one and only the latest
        presentation survived.
        """
        return bool(self._import_id) and placement.import_id == self._import_id

    def _is_imported(self) -> bool:
        """Whether the picture currently loaded has been committed to the canvas.

        Tracked through the *upload*, not the file path: uploading a picture
        again -- even the same file -- starts a new, uncommitted import, so the
        identity is reset on load and only 导入图片 assigns one.
        """
        return bool(self._import_id) and self._imported_path is not None

    def _sync_placements(self) -> None:
        """Make the document hold one object per selected layer for this picture.

        Called only when 导入图片 is pressed, and whenever the presentation of
        the *current* import is changed afterwards. A combined choice such as
        顶层 + 顶层丝印层 creates one object per layer, because that is what it
        says on screen.

        Deliberately **not** driven by the presentation controls while an
        import is pending. Those controls describe the *next* import, so acting
        on them here moved elements the user had already placed: selecting
        彩色丝印 and then importing a second copy dragged the first copy onto
        the silk layer. Only 导入图片 rearranges anything.
        """
        main = self.parentWidget()
        if main is None or not hasattr(main, "design_canvas"):
            return
        if not self._is_imported():
            return

        canvas = main.design_canvas
        board = canvas.board
        keys = self._presentation_keys()

        siblings = [item for item in board.placements if self._belongs_to_import(item)]
        kept: dict[str, object] = {}
        for placement in siblings:
            key = placement.layer_key
            if key in keys and key not in kept:
                # Keep the object the user may already have positioned.
                kept[key] = placement
            else:
                board.remove_placement(placement)

        # Geometry for anything created now comes from an element that was
        # already on the board, so re-importing onto another layer preserves
        # the position and size the user already chose. Siblings of this import
        # are preferred; failing that, the most recent import of the same file
        # is the natural starting point.
        template = next(iter(kept.values()), None)
        if template is None and siblings:
            template = siblings[0]
        if template is None:
            template = self._latest_import_of_source(board)

        tone_map = self._tone_by_layer_key()

        for key in keys:
            if key in kept:
                # Keep an existing object's geometry, but restate its tone: the
                # side may have changed when it was re-imported.
                kept[key].tone = tone_map.get(key, "")
                continue
            placement = self._new_placement(template)
            placement.layer = layer(key).name
            # Stamp the import this object belongs to; that, not the path, is
            # what keeps separate imports of one file apart.
            placement.import_id = self._import_id
            placement.tone = tone_map.get(key, "")
            board.add_placement(placement)
            kept[key] = placement

        selected = next(iter(kept.values()), None)
        canvas.selected = selected
        canvas.selection_changed.emit(selected)
        # Keep the layer panel's "current" in step, so the two places a layer
        # can be chosen from do not disagree.
        if keys:
            board.layers.set_current(keys[0])

        # ⚠️ A 板画 is two placements, and it is **one element** to whoever
        # drew it -- the white tone on silkscreen and the black on copper are
        # two halves of one picture. Grouping them here is what makes that true
        # on the canvas and in the exported file: before this they were two
        # rows in the layer list and two things to move, and keeping them
        # registered was entirely on the user.
        self._group_painting(board)

        # A picture that has already been processed must keep showing the line
        # art on its layer rather than falling back to the colour original.
        if self._inline_processed is not None:
            self._publish_line_art(self._inline_processed)
        canvas.update()

    def _group_painting(self, board) -> None:
        """Put this import's two board-painting tones into one named group.

        Does nothing unless the import produced the pair: a single-tone import
        -- a line drawing, a colour picture -- is one object, and
        :meth:`ArtBoard.make_group` would refuse it anyway. The guard is
        explicit so the intent is visible rather than resting on a rejection.

        Named after the source file, because the group shows up in the layer
        list and in the editor -- where 「组合 3」 says nothing about which
        picture it is.
        """
        members = [
            item
            for item in board.placements
            if self._belongs_to_import(item) and item.tone in ("white", "black")
        ]
        if len(members) < 2:
            return
        name = Path(self._source_path or "").stem or "板画"
        board.make_group(members, name=name)

    def _latest_import_of_source(self, board):
        """The newest existing element made from the picture being imported.

        Used as the geometry template for a fresh import of the same file, so
        the second copy starts where the first one was placed rather than at
        the origin with default size.
        """
        if not self._source_path:
            return None
        try:
            wanted = Path(self._source_path).resolve()
        except OSError:
            return None
        for placement in reversed(board.placements):
            if not placement.path:
                continue
            try:
                if Path(placement.path).resolve() == wanted:
                    return placement
            except OSError:
                continue
        return None

    def _new_placement(self, template):
        """A placement for this picture, copying ``template``'s geometry.

        The template is another object of the *same import* -- the copper copy
        of a drawing that is also on silk -- so its geometry is what the new
        copy should inherit. ``import_id`` is set by the caller, not here.
        """
        if template is not None:
            return replace(template)
        width_px, height_px = self._source_size or (0, 0)
        return placement_for_pixels(self._source_path or "", width_px, height_px)

    def _on_process_toggled(self) -> None:
        """The 处理图像 switch moved: show or hide everything it gates."""
        self._refresh_layers()
        self._refresh_preview()
        self._repaint_now()

    def _apply_two_tone_default(self, path: str, bgr, alpha) -> None:
        """Turn 处理图像 off by itself when the picture needs no processing.

        A picture that is already flat black and white should go onto the board
        as it is. Tracing it is not merely redundant: the line-art pass thickens
        every stroke and adds a silhouette ring the artwork never had, and on a
        drawing whose glyphs the tracer misreads (unusual letterforms, text with
        counters) it loses parts of the picture outright.

        ⚠️ **Suggestion, not a lock.** The switch is left live, so a user who
        wants the traced version of a two-colour picture can simply tick it
        again. Silently forcing a mode the user cannot override would be worse
        than the problem being solved.

        The picture is not altered here. Snapping it to pure black and white
        happens at import, so what the canvas gets is exactly two values while
        the uploaded file stays untouched.
        """
        from pcbart.image_prep import is_two_tone
        from pcbart.lineart import needs_denoising

        try:
            detected = is_two_tone(bgr)
        except Exception:  # noqa: BLE001 - a detection failure must not block loading
            detected = False

        self._two_tone_source = detected
        self.process_check.blockSignals(True)
        self.process_check.setChecked(not detected)
        self.process_check.blockSignals(False)

        # A softly shaded picture traces better after a blur: all the
        # per-pixel readings are edge detectors, and across a smooth gradient
        # they answer with hundreds of irregular rings instead of lines.
        #
        # ⚠️ **It is offered, not applied.** This used to tick 降噪 by itself,
        # and the user asked for the default to be off: an import that silently
        # changes a processing switch makes the result hard to predict, and the
        # smoothing it applies is visible in the line art. The note below says
        # the picture would benefit and names the control; the user decides.
        # The slider is still pre-sized to the recommended kernel so that
        # ticking the box lands on a sensible value straight away.
        soft = not detected
        if soft:
            try:
                soft = needs_denoising(bgr)
            except Exception:  # noqa: BLE001 - never block a load on a hint
                soft = False
        if soft:
            # The slider's value means "pixels of the user's picture", so the
            # useful kernel is a fraction of the picture rather than a fixed
            # number: the same 5 px that cleans a 1400 px drawing does almost
            # nothing on a 4000 px one. 0.65% of the short side lands on 9 px
            # for the 1378 px drawing this was tuned against, which is where the
            # fragments stop and the fine strands are still intact -- 5 px left
            # 74 of them, 15 px started erasing hair.
            height, width = bgr.shape[:2]
            wanted = int(round(min(height, width) * SOFT_SHADING_KERNEL_SHARE))
            self.denoise_slider.blockSignals(True)
            self.denoise_slider.setValue(
                max(self.denoise_slider.minimum(),
                    min(self.denoise_slider.maximum(), wanted))
            )
            self.denoise_slider.blockSignals(False)
        self._soft_shaded_source = soft

        if detected:
            note_text(
                self.process_note,
                "检测到只有两种颜色，已取消「处理图像」：图片原样放到画板上；"
                "需要线稿或板画时勾选上方开关。",
            )
            tone(self.process_note, "success")
        elif soft:
            note_text(
                self.process_note,
                "检测到柔和明暗渐变：可勾选「降噪」"
                f"（已按图大小预设为 {self.denoise_slider.value()}px），"
                "默认不勾选。",
            )
            tone(self.process_note, "success")
        else:
            note_text(self.process_note, "")
        self._refresh_layers()

    def _on_style_changed(self) -> None:
        """The 风格 combo moved: swap the preview over to the new style.

        The controls describe what the *next* 导入图片 will do, so nothing on
        the board is rearranged here -- re-syncing used to move elements the
        user had already positioned.

        The **preview**, though, has to follow immediately. 线稿 and 板画 are
        two different pictures made from the same source, and leaving the old
        one on screen until some other control is touched shows the user the
        wrong result for the style they just picked.
        """
        self._refresh_layers()
        self._refresh_preview()
        self._repaint_now()

    def _repaint_now(self) -> None:
        """Force the preview widgets to repaint before the next input event.

        A combo box's signal fires from inside its own event handling, and Qt
        only schedules a repaint for after that handler returns. Showing the
        new picture without this leaves the old one on screen for a beat, which
        reads as "the style did not take" and invites the user to poke another
        control to make it update.
        """
        self.drop_zone.update()
        if self._preview_panel is not None and self._preview_panel.isVisible():
            self._preview_panel.repaint()

    def _on_side_changed(self) -> None:
        """The top/bottom switch moved: the layer list changes side with it."""
        self._refresh_layers()

    def _on_board_art_side_changed(self) -> None:
        """The 板画 顶面/底面/两面 choice moved: describe the *next* import.

        Deliberately does not rearrange anything already on the board. 板面 is
        the same kind of control as 呈现方式 -- it says where the next 导入图片
        will go -- and acting on it here moved elements the user had already
        positioned. Only 导入图片 rearranges.
        """
        self._refresh_layers()

    def _refresh_board_note(self) -> None:
        """Show the source description plus the current board size.

        Built from :attr:`_source_description`, not by re-parsing the label's
        own text. The old version split that text on ``"　·　板框"`` to recover
        the description, but the not-yet-configured branch writes
        ``"　·　尚未设置板框"``, which does not contain that separator -- so every
        refresh appended another copy and the label grew to
        "尚未设置板框 · 尚未设置板框 · …".
        """
        main = self.parentWidget()
        if main is None or not hasattr(main, "design_canvas"):
            return
        if not self._source_description:
            return
        board = main.design_canvas.board
        if not board.outline_ready or board.image is None:
            self.source_note.setText(
                f"{self._source_description}　·　尚未设置板框"
            )
            return
        size = board.size_mm()
        self.source_note.setText(
            f"{self._source_description}"
            f"　·　板框 {size[0]:.1f} × {size[1]:.1f} mm"
        )

    def load_image(self, path: str) -> bool:
        """Load a local image and prepare it for processing.

        **Preview only.** The picture does not reach the canvas here: the user
        tunes the parameters first and then presses 导入图片, which is what
        turns the preview into board geometry. That order is the point --
        parameters chosen before anything lands are the parameters the board
        actually gets, rather than a first guess the user then has to correct.

        Returns ``True`` on success. Failures surface to the user rather than
        as a silent no-op, because "nothing happened" is the hardest kind of
        bug to report -- and a Qt signal handler that raises shows exactly
        that way.

        State is committed only after the first conversion succeeds, so a
        failure cannot leave the panel holding half-loaded data.
        """
        from pcbart.image_prep import imread_unicode

        try:
            raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        except ValueError as exc:
            QMessageBox.warning(self, "无法载入图片", str(exc))
            return False

        # Keep the original pixels: the preview is recomputed from source each
        # time a control moves, never from the previous preview.
        if raw.ndim == 3 and raw.shape[2] == 4:
            bgr, alpha = raw[:, :, :3], raw[:, :, 3]
        elif raw.ndim == 3:
            bgr, alpha = raw, None
        else:
            bgr, alpha = raw, None

        if bgr.size == 0:
            QMessageBox.warning(self, "无法载入图片", f"图片尺寸为空: {path}")
            return False

        previous = (
            self._source_bgr,
            self._source_alpha,
            self._source_path,
            self._source_size,
            self._inline_processed,
            self._full_processed,
            self._published_digest,
            self._imported_path,
            self._import_id,
            self._committed_style,
        )
        self._source_bgr, self._source_alpha, self._source_path = bgr, alpha, path
        self._source_size = (bgr.shape[1], bgr.shape[0])
        # The cached conversion belongs to the *previous* picture, so it is
        # dropped rather than carried over.
        self._inline_processed = None
        self._full_processed = None
        self._published_digest = None
        # ⚠️ A *fresh* cache, not ``None``. Setting it to None here turned the
        # cache off for the rest of the session -- nothing recreates it -- and
        # the symptom was exactly the lag the cache exists to remove: measured
        # 900 ms per slider tick on a 1024 px picture instead of 100 ms. The
        # cache is keyed on the picture's own bytes, so a new object is all that
        # is needed to forget the previous file.
        from pcbart.lineart import TraceCache

        self._trace_cache = TraceCache()
        # A fresh upload starts a new, uncommitted import. Without this,
        # re-uploading the same file would look "already imported" and the
        # first press of 导入图片 would silently move the earlier element.
        self._imported_path = None
        self._import_id = ""
        self._committed_style = ""
        self._apply_two_tone_default(path, bgr, alpha)
        try:
            self._refresh_preview()
        except Exception as exc:  # noqa: BLE001 - report and roll back
            import traceback

            traceback.print_exc()
            (
                self._source_bgr,
                self._source_alpha,
                self._source_path,
                self._source_size,
                self._inline_processed,
                self._full_processed,
                self._published_digest,
                self._imported_path,
                self._import_id,
                self._committed_style,
            ) = previous
            QMessageBox.warning(
                self, "无法处理图片", f"{Path(path).name} 载入后处理失败：\n{exc}"
            )
            return False

        height, width = bgr.shape[:2]
        filename = Path(path).name
        self._source_description = (
            f"{filename}　·　{width} × {height} px"
            + ("　·　含透明通道" if alpha is not None else "")
        )
        self.source_note.setText(self._source_description)
        tone(self.source_note, "success")
        self.import_button.setEnabled(True)
        self.import_button.setToolTip("按所选图层导入当前线稿")
        note_text(self.import_note, "")
        self._refresh_board_note()
        return True

    # -------------------------------------------------------------- importing

    def import_to_canvas(self) -> bool:
        """Put the previewed picture on the canvas, on the chosen layers.

        This is the second half of the two-step flow: uploading only read the
        file, and this is the click that commits it as board geometry. It runs
        with whatever the parameters currently are, and publishes the line art
        it just computed, so what lands is exactly what the preview showed.

        Importing the same picture twice replaces it rather than stacking a
        duplicate: the button means "this is the picture I want", not "add
        another copy".
        """
        if self._source_path is None or self._source_bgr is None:
            QMessageBox.information(
                self, "尚未载入图片", "请先在上传区域选择一张图片。"
            )
            return False

        main = self.parentWidget()
        if main is None or not hasattr(main, "design_canvas"):
            QMessageBox.information(
                self, "暂不可用", "主窗口尚未准备好设计画布。"
            )
            return False

        # Recomputed from the source, not reused from the preview, so the
        # committed artwork is a fresh conversion of the picture the user
        # loaded. What it is *not* is a full-resolution conversion: a phone
        # photo or a scan is far finer than the artwork will ever print, and
        # tracing 8000 px cost 6.9 s of frozen window per import. The working
        # resolution keeps the geometry -- measured at 81 vectorised vertices
        # against 107, the difference being hairline noise the 0.35 mm
        # simplification removes anyway -- while being several times faster.
        #
        # `alpha` and any pixel-measured settings are carried across the
        # reduction so "5 px of outline" still means five pixels of the picture
        # the user chose.
        from pcbart.work_scale import prepare_working_image

        working_bgr, working_alpha, work_scale = prepare_working_image(
            self._source_bgr, self._source_alpha
        )
        if not self.process_check.isChecked():
            # 处理图像 off: no tracing, no tone split. A two-colour picture is
            # snapped to exactly black and white -- its edges carry JPEG or
            # resize ringing, and those in-between values would otherwise reach
            # the board as grey the silkscreen cannot print.
            from pcbart.image_prep import is_two_tone, to_pure_two_tone

            self._raw_snapped = bool(is_two_tone(working_bgr))
            if self._raw_snapped:
                processed = to_pure_two_tone(working_bgr, working_alpha)
                if processed.ndim == 3 and processed.shape[2] == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                # Left exactly as the file has it. Converting to grey here
                # would be a silent edit to a picture the user asked not to
                # touch.
                processed = working_bgr
            self._full_processed = processed
            self._published_digest = None
            self._place_on_canvas(
                self._source_path,
                self._source_bgr.shape[1],
                self._source_bgr.shape[0],
            )
            self._publish_line_art(processed)
            self._refresh_board_note()

            canvas = main.design_canvas
            # Same framing as the processed path: 处理图像 off means "do not
            # trace it", not "ignore the card's opening". A two-colour drawing
            # put on the card still has to land in the illustration window.
            canvas.frame_into_card_region(canvas.selected)
            canvas.record_external_change()
            canvas.fit_view()
            note_text(self.import_note, "已原样导入画布（未处理）。")
            tone(self.import_note, "success")
            return True

        self._raw_snapped = False
        try:
            processed = self._process(
                working_bgr, working_alpha, param_scale=work_scale
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
            import traceback

            traceback.print_exc()
            QMessageBox.warning(self, "导入失败", f"无法生成线稿：\n{exc}")
            return False

        self._full_processed = processed
        # Objects first, then publish: publishing targets the placements that
        # exist for this picture, and it refuses to run before the picture is
        # imported, so the order here is load-bearing.
        self._published_digest = None
        self._place_on_canvas(
            self._source_path,
            self._source_bgr.shape[1],
            self._source_bgr.shape[0],
        )
        self._publish_line_art(processed)
        self._refresh_board_note()

        canvas = main.design_canvas
        # ⚠️ The card's illustration opening is applied here, on the import, and
        # its absence was the reported defect: nothing in the panel ever called
        # ``start_framing``, so a picture imported onto a 通行证 landed at its
        # default size with no crop and neither writer clipped it. Doing it here
        # rather than in each style is deliberate -- 线稿, 板画 and 彩色丝印 are
        # three conversions of *one* placement, so framing once covers all three
        # and the "which style was missed" class of bug cannot recur.
        framed = canvas.frame_into_card_region(canvas.selected)
        canvas.record_external_change()
        canvas.fit_view()
        if framed:
            note_text(
                self.import_note,
                "已导入画布并放入立绘区；再次点击替换画布上的图。",
            )
        else:
            note_text(self.import_note, "已导入画布；改参数后再次点击替换画布上的图。")
        tone(self.import_note, "success")
        return True

    def _reading(self) -> str:
        """The 读法 combo's key, not its display label."""
        index = self.reading_combo.currentIndex()
        if 0 <= index < len(READING_OPTIONS):
            return READING_OPTIONS[index][0]
        return READING_FILL

    def reading_label(self) -> str:
        """The selected reading's display label.

        Callers that report *what ran* read the label from the same tuple the
        combo is built from, so a reading can never be reported under another
        one's name.
        """
        index = self.reading_combo.currentIndex()
        if 0 <= index < len(READING_OPTIONS):
            return READING_OPTIONS[index][1]
        return READING_OPTIONS[0][1]

    def _on_reading_changed(self) -> None:
        """The reading moved: show only the controls that reading uses.

        A slider the current reading ignores is worse than a missing one -- the
        panel has already been reported twice for controls that looked live and
        did nothing. 线条粗细 and 墨量阈值 are read by 纯线条 (``mode="dodge"``)
        and by nothing else, so only there are they shown.
        """
        self._refresh_reading_rows()
        self._on_tuning_changed()

    def _refresh_reading_rows(self) -> None:
        """Show the controls that belong to the selected reading.

        线条粗细 and 墨量阈值 are read by ``mode="dodge"`` and by nothing else:
        填充色块 takes its split from Otsu and 描边+填充 from the contour
        density, so in those two readings the sliders would move without
        changing a single pixel. That is the exact defect this panel has been
        reported for before -- a control that looks live and does nothing --
        so they are hidden instead of merely documented.
        """
        reading = self._reading()
        for widget in (
            self.outline_width_row,
            self.outline_detail_row,
            self.outline_density_row,
        ):
            widget.setVisible(reading == READING_OUTLINE)
        for widget in (self.strength_row, self.threshold_row):
            widget.setVisible(reading == READING_DODGE)

    def _lineart_params(self, param_scale: float = 1.0):
        """Line-extraction settings only.

        Denoising is a separate step and is not part of these parameters; the
        two are applied in sequence in :meth:`_process`.

        ``param_scale`` carries the pixel-measured 轮廓宽度 across a reduction;
        粗细 and 阈值 are ratios and do not need it.
        """
        from pcbart.lineart import LineArtParams
        from pcbart.work_scale import scaled_param

        return LineArtParams(
            mode=self._reading(),
            strength=self.strength_slider.value() / 100.0,
            threshold=self.threshold_slider.value() / 100.0,
            silhouette=self.silhouette_check.isChecked(),
            silhouette_width=max(
                1,
                int(
                    round(
                        scaled_param(
                            self.silhouette_slider.value(), param_scale, minimum=1.0
                        )
                    )
                ),
            ),
            # 描边密度 is a share of the subject, so it is resolution-independent
            # too and is *not* scaled. 描边线宽 is in millimetres of the
            # finished board, and :func:`line_art` turns it into pixels with
            # ``px_per_mm``.
            outline_sigma_ratio=self.outline_detail_slider.value() / 10000.0,
            outline_density=self.outline_density_slider.value() / 1000.0,
            outline_width_mm=self.outline_width_slider.value() / 100.0,
        )

    def _px_per_mm(self, image) -> float | None:
        """How many traced pixels make a millimetre on the finished board.

        The contour's width is specified in millimetres, so something has to
        connect the two -- and it is the *board's* size, not the source's.
        Uses the same placement width the 板画 conversion does, so the two
        styles agree about how big the picture will be.
        """
        width_mm = self._placement_width_mm()
        if width_mm <= 0 or image is None or image.shape[1] <= 0:
            return None
        return image.shape[1] / width_mm

    def _board_art_params(self, param_scale: float = 1.0) -> BoardArtParams:
        """The 板画 controls as model parameters.

        Sliders carry integers for Qt's sake; the model wants a 0..1 threshold
        and millimetres, so the conversion lives here and nowhere else.

        最小特征 is in millimetres, so it is resolution-independent already --
        ``param_scale`` is accepted for symmetry with the line-art builder and
        deliberately does not touch it.
        """
        _ = param_scale
        params = BoardArtParams(
            tone_threshold=self.tone_slider.value() / 100.0,
            min_feature_mm=self.min_feature_slider.value() / 100.0,
            black_background=self.background_check.isChecked(),
            flip_vertical=self.flip_vertical_check.isChecked(),
            flip_horizontal=self.flip_horizontal_check.isChecked(),
            invert=self.invert_check.isChecked(),
        )
        # The contour is a width in millimetres, and the model takes a fraction
        # of the short side, so the board's own size is what ties the two
        # together. Without a board yet, the placement default is used -- the
        # same size the picture will be imported at.
        width_mm = self._placement_width_mm()
        if width_mm > 0:
            short_mm = min(width_mm, self._placement_height_mm(width_mm))
            params.outline_ratio = (
                self.outline_mm_slider.value() / 1000.0 / short_mm
                if short_mm > 0
                else 0.0
            )
        return params

    def _placement_width_mm(self) -> float:
        """The width the picture will be imported at, in millimetres."""
        main = self.parentWidget()
        canvas = getattr(main, "design_canvas", None)
        if canvas is None:
            return DEFAULT_IMAGE_LONG_EDGE_MM
        board = canvas.board
        if board.image is not None and board.image.width_mm > 0:
            return float(board.image.width_mm)
        return DEFAULT_IMAGE_LONG_EDGE_MM

    def _placement_height_mm(self, width_mm: float) -> float:
        """Height matching ``width_mm`` at the source's aspect ratio."""
        if self._source_bgr is None or width_mm <= 0:
            return width_mm
        height_px, width_px = self._source_bgr.shape[:2]
        if width_px <= 0:
            return width_mm
        return width_mm * height_px / width_px

    def _process(
        self, image, alpha, max_side: int | None = None, param_scale: float = 1.0
    ):
        """Run the pipeline for the selected style.

        线稿 runs ``denoise -> line_art -> clean_binary`` as three explicit
        steps, so a change in one can be attributed to that step. 板画 runs
        ``board_art`` instead, which resolves the picture straight to two
        tones; denoising has no place there because the tone split is already a
        hard threshold and blurring first would only move its boundary.

        Returns an ink image (线稿), the original colour picture (彩色丝印), or
        a ``BoardArt`` (板画). Those differ in shape, so callers go through
        :meth:`_display_image` rather than guessing from the array.

        彩色丝印 is an identity pass on purpose. Its "processed result" *is* the
        original -- that is what the style means -- and returning it here keeps
        one code path for the preview, the import and the canvas instead of
        three that each special-case the colour style differently.
        """
        if self._style() == COLOR_STYLE:
            return image

        if self._style() == BOARD_ART_STYLE:
            return board_art(
                image,
                alpha,
                self._board_art_params(param_scale),
                px_per_mm=self._board_art_px_per_mm(image),
            )

        from pcbart.denoise import clean_binary, denoise_image
        from pcbart.lineart import clean_strokes_with_arcs, line_art, line_art_preview
        from pcbart.work_scale import scaled_param

        # 降噪核、轮廓宽度、去杂点面积都是「用户那张图」的像素数。降采样之后
        # 原样套用会让同一个数字在更小的图上滤掉更少的东西——
        # 界面上写着 5px，做出来的却不是 5px 的效果。
        blurred = denoise_image(
            image,
            enabled=self.denoise_check.isChecked(),
            kernel_size=max(
                1,
                int(
                    round(
                        scaled_param(
                            self.denoise_slider.value(), param_scale, minimum=1.0
                        )
                    )
                ),
            ),
        )
        params = self._lineart_params(param_scale)
        # The cache is keyed on the blurred plane, so changing 降噪 (or the
        # source) invalidates it on its own and 抠图 stays consistent with the
        # picture actually being traced. Nothing here has to remember to clear
        # it except a change of source, which swaps the whole panel state.
        cache = self._trace_cache
        px_per_mm = self._px_per_mm(image)
        if max_side is None:
            traced = line_art(
                blurred, alpha, params, trace_cache=cache, px_per_mm=px_per_mm
            )
        else:
            traced = line_art_preview(
                blurred,
                alpha,
                params,
                max_side=max_side,
                trace_cache=cache,
                px_per_mm=px_per_mm,
            )
        traced = clean_binary(
            traced,
            min_blob_area=int(
                round(
                    scaled_param(
                        self.despeckle_slider.value(), param_scale, minimum=0.0
                    )
                )
            ),
        )
        if self.arc_smooth_check.isChecked() and max_side is None:
            # ⚠️ **Deliberately not run for the live preview.** Measured at
            # **1.8 s** on a 1024 px picture against 36 ms for the rest of the
            # pass, so running it per slider tick would reintroduce exactly the
            # 调参卡顿 this panel was just fixed for. The preview tracks the
            # sliders at display resolution and this runs on the settled
            # full-resolution conversion, which is the arrangement the earlier
            # VTracer measurement recommended and the same one the 260 ms
            # commit timer already provides. The switch therefore costs one
            # pause after a drag, not a pause per step.
            traced = clean_strokes_with_arcs(traced, px_per_mm=px_per_mm or 1.0)
        return traced

    def _board_art_px_per_mm(self, image) -> float | None:
        """The resolution the painting will print at.

        This is what turns 最小特征 from millimetres into pixels, so it has to
        be the *board's* scale rather than the source's. A preview that was
        resampled down for speed must therefore report the same px/mm as the
        full-resolution conversion, or the two would filter differently and the
        preview would not be what gets imported.
        """
        width_mm = self._placement_width_mm()
        if width_mm <= 0 or image is None or image.shape[1] <= 0:
            return None
        return image.shape[1] / width_mm

    def _is_board_art(self) -> bool:
        return self._style() == BOARD_ART_STYLE

    def _display_image(self, processed):
        """Whatever :meth:`_process` returned, as an image that can be shown.

        板画 resolves to two masks rather than a picture, so showing it means
        compositing them into the black-and-white the board will actually
        carry. Every other style already has an image by this point.

        Returning the composite rather than one of the masks is what makes the
        preview honest: the two tones go to *different* layers on the board, so
        neither mask alone is the board, while the composite is what they look
        like together.
        """
        if self._is_board_art() and hasattr(processed, "white"):
            return board_art_preview(processed)
        return processed

    def _refresh_preview(self) -> None:
        """Recompute the processed preview from the source image.

        Runs at a bounded resolution: the radius rule keeps the result
        scale-invariant, so the preview costs a fraction of the full-size
        conversion while looking the same.

        The canvas is updated here too, at this bounded resolution, so the
        layer preview keeps up with the sliders. ``refresh_preview_full``
        replaces it with the full-resolution result when the user asks for it.
        """
        if self._source_bgr is None:
            return

        from pcbart.work_scale import prepare_working_image

        preview_bgr, preview_alpha, preview_scale = prepare_working_image(
            self._source_bgr, self._source_alpha
        )

        if not self.process_check.isChecked():
            # 处理图像 off: the "result" is the picture itself, snapped to pure
            # black and white when it is a two-colour artwork so the preview
            # shows what will actually be placed.
            from pcbart.image_prep import is_two_tone, to_pure_two_tone

            try:
                if is_two_tone(preview_bgr):
                    processed = to_pure_two_tone(preview_bgr, preview_alpha)
                    if processed.ndim == 3 and processed.shape[2] == 3:
                        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                else:
                    processed = (
                        cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY)
                        if preview_bgr.ndim == 3
                        else preview_bgr
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                note_text(self.tune_note, f"图片读取失败：{exc}")
                tone(self.tune_note, "danger")
                return
            self._inline_processed = processed
            self._show_in_drop_zone(processed, None, processed.shape[1], processed.shape[0], 0.0)
            self._repaint_now()
            return

        start = time.perf_counter()
        try:
            processed = self._process(
                preview_bgr,
                preview_alpha,
                max_side=760,
                param_scale=preview_scale,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            # A failure here must not be silent: the control appears to do
            # nothing, which is indistinguishable from a broken upload.
            import traceback

            traceback.print_exc()
            note_text(self.tune_note, f"处理失败：{exc}")
            tone(self.tune_note, "danger")
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        board_art_result = processed if self._is_board_art() else None
        display = self._display_image(processed)

        height, width = display.shape[:2]
        ink = float((display < 128).mean()) * 100
        self._inline_processed = processed

        self._show_in_drop_zone(display, board_art_result, width, height, ink)

        if board_art_result is not None:
            flips = []
            if self.flip_vertical_check.isChecked():
                flips.append("上下")
            if self.flip_horizontal_check.isChecked():
                flips.append("左右")
            params = self._board_art_params()
            self.tune_note.setText(
                f"分界 {params.tone_threshold:.2f}　·　"
                f"轮廓 {self.outline_mm_slider.value() / 1000:.2f} mm"
                f"（{board_art_result.outline_px} px）　·　"
                f"最小特征 {params.min_feature_mm:.2f} mm"
                f"（{board_art_result.min_feature_px} px）　·　"
                f"{'翻转 ' + '/'.join(flips) if flips else '不翻转'}　·　"
                f"{elapsed_ms:.0f} ms"
            )
            self.tune_note.setVisible(True)
        else:
            denoise_state = (
                f"降噪 {self.denoise_slider.value()}px"
                if self.denoise_check.isChecked()
                else "降噪关"
            )
            if self.despeckle_slider.value():
                denoise_state += f" · 去杂点 {self.despeckle_slider.value()}px"
            if self.arc_smooth_check.isChecked():
                denoise_state += " · 弧线平滑"

            # The reading that produced this preview is named *here*, from the
            # same tuple the combo is built from. Spelling it out per branch
            # meant a new reading silently reported itself as 填充色块.
            reading_state = f"读法 {self.reading_label()}　·　"
            if self._reading() == READING_OUTLINE:
                reading_state += (
                    f"线宽 {self.outline_width_slider.value() / 100:.2f} mm　·　"
                    f"密度 {self.outline_density_slider.value() / 10:.1f}%　·　"
                )
            elif self._reading() == READING_DODGE:
                # The two controls 纯线条 actually reads. Reporting them under
                # 填充色块 would name numbers that changed nothing.
                reading_state += (
                    f"粗细 {self.strength_slider.value() / 100:.2f}　·　"
                    f"阈值 {self.threshold_slider.value() / 100:.2f}　·　"
                )

            self.tune_note.setText(
                f"{reading_state}"
                f"{denoise_state}　·　"
                f"墨量 {ink:.1f}%　·　{elapsed_ms:.0f} ms"
            )
            self.tune_note.setVisible(True)
        tone(self.tune_note, "success")

        # Show the bounded result on the canvas too, so the layer preview
        # tracks the sliders instead of only updating when 大图预览 is opened.
        published_preview = self._publish_line_art(
            processed, note=f"{elapsed_ms:.0f} ms", preview=True
        )
        if published_preview:
            # The canvas is now drawing the display-resolution conversion. That
            # is what makes tuning feel immediate, but it is not what the board
            # should be built from, so the full-resolution one is recomputed
            # once the user stops moving the control.
            self._schedule_committed_publish()
        self._update_preview_panel()

    def _schedule_committed_publish(self) -> None:
        """Recompute the full-resolution artwork shortly after tuning stops.

        Restarting the timer on every tick means one recompute per burst of
        dragging rather than one per tick, so the expensive conversion happens
        when the user has stopped asking for updates.
        """
        timer = getattr(self, "_committed_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(260)
            timer.timeout.connect(self._publish_committed_art)
            self._committed_timer = timer
        timer.start()

    def _publish_committed_art(self) -> None:
        """Write the artwork the board is actually exported from.

        Runs after the preview burst settles. It uses the same source and
        parameters as the preview, at the working resolution rather than the
        display one, so the geometry is several times finer than what is on
        screen while still bounded.
        """
        if self._source_bgr is None or not self._is_imported():
            return
        if self._style() != self._committed_style:
            return

        from pcbart.work_scale import prepare_working_image

        working_bgr, working_alpha, work_scale = prepare_working_image(
            self._source_bgr, self._source_alpha
        )
        try:
            processed = self._process(
                working_bgr, working_alpha, param_scale=work_scale
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
            note_text(self.tune_note, f"线稿写入失败：{exc}")
            tone(self.tune_note, "danger")
            return

        self._full_processed = processed
        # Force the write: the preview already stored *its* digest for this
        # path, so without clearing it the committed conversion would look
        # unchanged and never reach the disk.
        self._published_digest = None
        self._published_digests.clear()
        self._publish_line_art(processed)

    def _show_in_drop_zone(
        self, display, board_art_result, width: int, height: int, ink: float
    ) -> None:
        """Put the current style's result in the upload box, caption included.

        The caption names the style, and it has to. Every style resolves to a
        different picture -- line art, a two-tone painting, or the original
        photograph -- so a caption that always said 线稿预览 labelled two of the
        three wrongly and made a correct preview look like a broken one.
        """
        if not self.preview_toggle.isChecked():
            # The toggle means "show me the source instead", so it wins over
            # every style; the caption says why the two differ.
            pixmap = QPixmap(self._source_path or "")
            if not pixmap.isNull():
                self.drop_zone.show_image(
                    pixmap,
                    Path(self._source_path or "").name,
                    f"{self._source_bgr.shape[1]} × {self._source_bgr.shape[0]} px"
                    "　·　显示原图（「显示处理结果」已关闭）",
                )
            return

        if board_art_result is not None:
            caption = (
                f"{width} × {height} px　·　"
                f"白 {board_art_result.white.mean() * 100:.0f}% / "
                f"黑 {board_art_result.black.mean() * 100:.0f}%"
            )
            label = "板画预览（白=丝印，黑=铜）"
        elif not self.process_check.isChecked():
            caption = f"{width} × {height} px　·　不做处理，原样放置"
            label = "原图预览"
        elif self._style() == COLOR_STYLE:
            caption = (
                f"{width} × {height} px　·　保留原图配色，不做线稿处理"
            )
            label = "彩色丝印预览"
        else:
            caption = f"{width} × {height} px　·　墨量 {ink:.1f}%"
            label = "线稿预览"

        self.drop_zone.show_array(display, label, caption)

    def _publish_line_art(self, processed, note: str = "", preview: bool = False) -> bool:
        """Write the processed picture where the canvas can draw it.

        The canvas shows the line art as a flat black-and-white image tinted in
        the layer's colour, so what it needs is a file on disk rather than an
        array. It is written under a cached name derived from the source, so
        repeated tuning reuses one file instead of filling the disk.

        Every object for this picture is updated, not just the selected one:
        the same artwork on copper and on silkscreen is two objects showing one
        drawing, and leaving one of them on a stale conversion would be a
        silent mismatch.

        彩色丝印 is the one style that keeps the original picture, so nothing is
        published then -- the canvas falls back to the colour source by design.

        Does nothing until the picture has been imported: the preview is not
        board geometry, so tuning a slider must not create one.
        """
        main = self.parentWidget()
        if main is None or not hasattr(main, "design_canvas"):
            return False
        if self._source_path is None or processed is None:
            return False
        if not self._is_imported():
            return False

        # The style combo describes the *next* import -- that is the rule the
        # whole panel follows, and it is why changing it must not rearrange
        # what is already on the canvas. Publishing is part of that rule: the
        # objects on the board were committed under the old style, so feeding
        # them a result computed with the new one writes the wrong kind of
        # picture into them. 板画 resolves to two masks and 线稿 to a single
        # image, so the mismatch is not even the same shape -- switching style
        # after an import raised AttributeError instead of repainting.
        #
        # Slider tweaks keep the style, so this guard is invisible to the
        # behaviour it exists for: tuning still updates the imported artwork.
        # A raw import carries no style, so the style comparison below does not
        # apply to it; what matters is that the picture being published is the
        # one that was placed, which the callers already guarantee.
        if (
            self._committed_style != RAW_STYLE
            and self._style() != self._committed_style
        ):
            return False

        # `preview` marks a display-resolution result. It is still written --
        # that is what makes the canvas track the sliders -- but the caller
        # follows it with a full-resolution rewrite once the drag settles, so
        # the file the board is exported from is never left as a thumbnail.
        _ = preview

        canvas = main.design_canvas
        targets = [
            item for item in canvas.board.placements if self._belongs_to_import(item)
        ]
        if not targets:
            return False

        if self._committed_style == RAW_STYLE:
            # A raw import normally has nothing derived from it, so nothing is
            # written and the canvas falls back to the source file -- the same
            # arrangement 彩色丝印 uses. The exception is a two-colour picture,
            # which was snapped to exactly black and white: that result is what
            # the board should carry, because the source's edge ringing would
            # otherwise reach the vectoriser as extra mid-tones.
            if not self._raw_snapped:
                for placement in targets:
                    placement.processed_path = ""
                    placement.kind = ""
                canvas._tinted_cache.clear()
                canvas.update()
                return True

        if self._committed_style == COLOR_STYLE:
            for placement in targets:
                placement.processed_path = ""
                placement.kind = ""
            canvas._tinted_cache.clear()
            canvas.update()
            return True

        if self._committed_style == BOARD_ART_STYLE:
            return self._publish_board_art(processed, targets, canvas)

        # Skip the rewrite when the pixels have not changed, which is most of
        # slider ticks. Everything below this line invalidates canvas caches
        # and touches the disk, so doing it needlessly makes dragging stutter.
        digest = hashlib.sha1(np.ascontiguousarray(processed).tobytes()).hexdigest()
        target = self._processed_path()
        if target is None:
            return False
        if digest != self._published_digest or not Path(target).exists():
            from pcbart.image_prep import imwrite_unicode

            try:
                # ⚠️ **Written black-on-white, and that is what the canvas
                # needs.** ``design_canvas._tinted`` turns the file into the
                # layer's colour by reading it as *grey*: ink is dark, so it
                # uses ``255 - grey`` as the alpha. An ink-on-transparent file
                # has paper at RGB 0, which reads as grey 0 -- maximum ink --
                # and the whole illustration came out as one solid block of the
                # layer colour. The transparent form is produced at export
                # time instead, by ``pcbart.color_image``, where it is needed
                # so the white paper is not printed onto the card.
                imwrite_unicode(target, processed)
            except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
                note_text(self.tune_note, f"线稿写入失败：{exc}")
                tone(self.tune_note, "danger")
                return False
            self._published_digest = digest

        for placement in targets:
            placement.processed_path = target
            # The kind records that this file is 线稿's own output, so the
            # writer embeds it as a picture instead of tracing it back into
            # filled regions the user cannot edit. See
            # :func:`pcbart.color_image.is_color_artwork`.
            placement.kind = "lineart"

        canvas._pixmaps.pop(target, None)
        canvas._tinted_cache.clear()
        canvas.update()
        return True

    def _publish_board_art(self, art, targets, canvas) -> bool:
        """Write each tone to the file its own layer will draw.

        A painting is two different images on two different layers, so unlike
        line art there is no single file to share. Each placement is matched to
        its tone by layer kind -- silkscreen draws the white one, copper the
        black -- and handed its own cached file.

        Both tones are converted with :func:`tone_to_ink` first. The canvas
        tints *ink* by the layer's colour, so publishing the mask's own
        polarity would tint the empty area instead and draw every shape inside
        out.
        """
        from pcbart.image_prep import imwrite_unicode

        white_ink = tone_to_ink(art.white)
        black_ink = tone_to_ink(art.black)

        for placement in targets:
            kind = layer(placement.layer_key).kind
            image = white_ink if kind == "silk" else black_ink
            suffix = "white" if kind == "silk" else "black"

            digest = hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()
            target = self._processed_path(suffix)
            if target is None:
                return False

            cached = self._published_digests.get(target)
            if cached != digest or not Path(target).exists():
                try:
                    imwrite_unicode(target, image)
                except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
                    note_text(self.tune_note, f"板画写入失败：{exc}")
                    tone(self.tune_note, "danger")
                    return False
                self._published_digests[target] = digest

            placement.processed_path = target
            canvas._pixmaps.pop(target, None)

        canvas._tinted_cache.clear()
        canvas.update()
        return True

    def _processed_path(self, suffix: str = "") -> str | None:
        """Where the processed picture for the current source is cached.

        Kept beside the project rather than in a temp folder: the document
        stores this path, so a temp file that the OS cleans up would leave the
        canvas with nothing to draw.

        ``suffix`` separates the two tones of a board painting, which are two
        files for one source. Without it the second tone would overwrite the
        first and both layers would draw the same image.
        """
        if self._source_path is None:
            return None
        # cache_dir, not a path built from __file__: a packaged build unpacks
        # to a temporary directory that is deleted on exit, and an installed
        # one lives somewhere the user cannot write to.
        from pcbart.paths import cache_dir

        root = cache_dir("lineart_cache")
        stem = Path(self._source_path).stem
        digest = hashlib.sha1(
            str(Path(self._source_path).resolve()).encode("utf-8")
        ).hexdigest()[:8]
        tail = f"_{suffix}" if suffix else ""
        return str(root / f"{stem}_{digest}{tail}.png")

    def _on_tuning_changed(self) -> None:
        """A tuning control moved: recompute the preview, coalescing bursts.

        The debounce is what keeps a drag responsive. Every step of the handle
        emits its own value, and a conversion is long enough that computing each
        one queues work faster than it drains -- the picture on screen then
        falls further and further behind the handle. Restarting a short
        single-shot timer per step means only the value the user stopped on is
        converted, while a single click still updates immediately.

        The timer is created lazily because it needs a live ``self`` as its
        parent, and clearing a slider's value during construction would
        otherwise start one before the panel has finished building.
        """
        timer = self._preview_timer
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(PREVIEW_DEBOUNCE_MS)
            timer.timeout.connect(self._refresh_preview)
            self._preview_timer = timer
        timer.start()

    def _reset_tuning(self) -> None:
        """Restore every style's defaults, including the ones not on screen.

        Both sets are reset rather than only the visible one, so switching
        styles after a reset does not reveal values left over from an earlier
        experiment.
        """
        widgets = (
            self.strength_slider,
            self.threshold_slider,
            self.denoise_slider,
            self.despeckle_slider,
            self.silhouette_slider,
            self.outline_width_slider,
            self.outline_detail_slider,
            self.outline_density_slider,
            self.tone_slider,
            self.outline_mm_slider,
            self.min_feature_slider,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.strength_slider.setValue(100)
        self.threshold_slider.setValue(46)
        self.denoise_slider.setValue(5)
        self.despeckle_slider.setValue(0)
        self.silhouette_slider.setValue(5)
        self.outline_width_slider.setValue(20)
        self.outline_detail_slider.setValue(16)
        self.outline_density_slider.setValue(60)
        self.tone_slider.setValue(int(TONE_THRESHOLD * 100))
        self.outline_mm_slider.setValue(46)
        self.min_feature_slider.setValue(10)
        for widget in widgets:
            widget.blockSignals(False)

        for check, value in (
            (self.denoise_check, False),
            (self.silhouette_check, False),
            (self.arc_smooth_check, False),
            (self.background_check, True),
            (self.flip_vertical_check, False),
            (self.flip_horizontal_check, False),
            (self.invert_check, False),
        ):
            check.blockSignals(True)
            check.setChecked(value)
            check.blockSignals(False)

        self.reading_combo.blockSignals(True)
        self.reading_combo.setCurrentIndex(0)
        self.reading_combo.blockSignals(False)
        self._refresh_reading_rows()

        self._refresh_preview()
        self._repaint_now()

    def _side(self) -> str:
        return "顶层" if self.side_control.buttons[0].isChecked() else "底层"  # type: ignore[attr-defined]

    def _refresh_layers(self) -> None:
        style_index = self.style_combo.currentIndex()
        style = STYLE_OPTIONS[style_index][0]

        side = self._side()
        locked = style == COLOR_STYLE
        painting = style == BOARD_ART_STYLE
        processing = self.process_check.isChecked()

        # 处理图像 off means "put the picture on the board as it is", so there
        # is nothing to configure and nothing to preview. Everything below the
        # switch is hidden rather than disabled: a greyed-out panel of sliders
        # still invites the user to go looking for how to turn them on.
        if not processing:
            for widget in (
                self.style_group,
                self.layer_group,
                self.board_art_section,
                self.tuning_section,
                self.denoise_section,
                self.preview_section,
                self.redraw_section,
            ):
                widget.setVisible(False)
            note_text(self.process_note, "图片原样放到画板上，不做处理。")
            return

        self.style_group.setVisible(True)
        self.layer_group.setVisible(True)

        # Each style shows only the controls that do something to it. A knob
        # that quietly has no effect is the placeholder problem this panel has
        # already been burned by, so the rule is by style rather than by
        # "hide a few of the line-art ones".
        #
        # 彩色丝印 processes nothing at all -- it keeps the original picture --
        # so everything past 呈现方式 is hidden, 预览 included: there is no
        # conversion to inspect.
        self.board_art_section.setVisible(painting)
        self.tuning_section.setVisible(not painting and not locked)
        self.denoise_section.setVisible(not painting and not locked)
        self.preview_section.setVisible(not locked)

        # 线稿重绘 is the third way to turn a picture into line art, so it
        # belongs wherever the built-in line-art reading does -- and *not* under
        # 彩色丝印, which processes nothing. Its status is re-read here because
        # the model folder or PyTorch can have been installed since the panel
        # was built, and a stale "尚未安装" would be a lie the user cannot
        # refresh.
        self.redraw_section.setVisible(not painting and not locked)
        if not painting and not locked:
            self._refresh_redraw_section()
        if not painting and not locked:
            # Which of 线稿调试's rows are live depends on the reading, and a
            # style change is a common way to arrive at this panel -- so the
            # rows are reconciled here rather than only when the combo moves.
            self._refresh_reading_rows()

        # 板画 replaces the layer list with its own three-way choice. The
        # side control above is *not* used then: "both" is a side choice of its
        # own, and leaving 顶层/底层 live beside it would give two controls that
        # disagree about which face is meant.
        self.side_control.setVisible(not painting)
        self.presentation_row.setVisible(not painting)
        self.board_art_side_combo.setVisible(painting)
        self.board_art_side_row.setVisible(painting)

        previous = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        if painting:
            # The layer list is hidden while painting, so it is left empty --
            # but the index is not set, because Qt rejects index 0 on an empty
            # combo. Nothing reads it in this mode either way.
            options = []
            self.layer_combo.setEnabled(False)
        elif locked:
            options = [(f"{side}丝印层", ("",), "彩色丝印：颜色只能印在丝印层")]
            self.layer_combo.setEnabled(False)
        else:
            options = layer_options(side)
            self.layer_combo.setEnabled(True)

        for name, _keys, _hint in options:
            self.layer_combo.addItem(name)

        if options:
            restored = False
            for index, (name, _keys, _hint) in enumerate(options):
                if name == previous or name.replace("顶层", side).replace("底层", side) == previous:
                    self.layer_combo.setCurrentIndex(index)
                    restored = True
                    break
            if not restored:
                self.layer_combo.setCurrentIndex(0)

        self.layer_combo.blockSignals(False)


def live_panel(name: str):
    """The cached panel in ``name``, or ``None`` if it is unusable.

    ⚠️ **Reading the module global is itself the dangerous step.** A panel is
    parented to a window, so it dies with that window -- and the global keeps a
    Python wrapper whose C++ half is gone. Touching that wrapper (reading an
    attribute, comparing it, *printing it*) raises ``RuntimeError``, so a guard
    written as ``if panel is not None and panel.parentWidget() ...`` is already
    too late: the first part is what throws.

    Every reader of one of the panel globals goes through here, which is the
    only way a caller can be sure it is holding a live object.
    """
    module = sys.modules[__name__]
    try:
        panel = getattr(module, name, None)
    except RuntimeError:
        setattr(module, name, None)
        return None
    return panel


def reusable_panel(panel, main_window, factory):
    """Return ``panel`` if it is still usable for ``main_window``, else a new one.

    Three things have to be true for a cached panel to be usable, and asking
    them in this order is what makes the check safe:

    1. **It is not dead.** A parented Qt object lives exactly as long as the
       window that owns it, and when that window is destroyed the module global
       keeps a Python wrapper whose C++ half is gone. *Touching that wrapper --
       reading an attribute, printing it, even ``is None`` on the object rather
       than the name -- raises ``RuntimeError``.* So the liveness check has to
       come first, and it is done by attempting the call and catching that
       error rather than by inspecting anything.
    2. It is parented to the window being asked about, not to a previous one.
    3. It is still in that window's child list.

    Returned as ``(panel, created)``. Extracted from the three ``show_*``
    functions because they all had the same latent crash and only one of them
    happened to avoid it in practice.
    """
    if panel is not None:
        try:
            if panel.parentWidget() is main_window:
                return panel, False
        except RuntimeError:
            # The C++ object is gone; fall through and build a fresh one.
            pass
    made = factory(main_window)
    return made, True


def show_preprocess(main_window: QWidget) -> PreprocessPanel:
    """Show the preprocessing panel inside the main window.

    The panel is a child of the application window by design: it cannot be
    opened from the desktop or appear in the taskbar.
    """
    global _preprocess_panel
    _preprocess_panel, created = reusable_panel(
        _preprocess_panel, main_window, PreprocessPanel
    )
    if created:
        _preprocess_panel.resize(420, 560)
        _preprocess_panel.place_top_right()
    _preprocess_panel.show()
    _preprocess_panel.raise_()
    return _preprocess_panel


_preprocess_panel: PreprocessPanel | None = None


# ---------------------------------------------------------------- panels


class BoardSetupPanel(FloatingPanel):
    """The board-setup panel: preset shape, its size, and the artwork's size.

    Live rather than one-shot. Every control writes straight into the design
    document and the canvas behind the panel redraws, so the board is previewed
    while it is being chosen; the artwork itself can be dragged on the canvas
    with the panel still open, which lets the two halves of the layout be
    settled together. What the panel shows is always what the canvas draws.

    Shape is asked for before size because the shape decides which dimensions
    exist at all -- a circle has one, a rectangle has two. The size rows follow
    the current shape rather than being fixed, so the panel cannot present a
    second dimension for a shape that has only one.

    Irregular outlines are deliberately absent. Tracing the artwork *combines*
    with a preset instead of replacing one, which is a different kind of
    decision, and it will get its own panel.
    """

    changed = Signal()
    """Emitted after an edit, so the canvas and the inspector can refresh."""

    content_requested = Signal()
    """Emitted when the user asks for the template-content panel.

    A signal rather than the panel opening itself: the board-setup panel has no
    business knowing where that panel lives or how it is parented, and the
    window is the one that owns the connection.
    """

    TEMPLATE_OPTION = "套用模板"

    def __init__(self, parent: QWidget, board: ArtBoard, canvas=None) -> None:
        super().__init__(parent, "板框设置")
        self.resize(400, 620)
        self._board = board
        self.canvas = canvas
        """The design canvas, so a template change can re-crop the pictures.

        Optional because the panel is also built in tests and scripts that only
        want the controls; when it is missing the reframing is simply skipped,
        which is the same behaviour as a board that is not a card.
        """
        self._updating = False
        self._run_key: str | None = None
        """Which run of edits the last change belonged to; see
        :meth:`edit_run_key`."""

        # Opening the panel is the decision to have a board, so a first board
        # is seeded from the artwork and goes live immediately. A board the
        # user already configured is left exactly as it was.
        if not board.outline_ready:
            self._seed_from_image()
        board.outline_ready = True

        layout = self.content_layout
        layout.setSpacing(4)

        shape_group = section("1 · 板框形状", compact=True)
        shape_layout = QVBoxLayout(shape_group)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(3)

        labels = [OutlineShape.LABELS[s] for s in SHAPE_CHOICES]
        self.shape_combo = combo(labels + [self.TEMPLATE_OPTION])
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        shape_layout.addWidget(row("形状", self.shape_combo))

        names = template_names()
        self.template_combo = combo(names or ["尚未载入模板"])
        self.template_combo.setEnabled(bool(names))
        # Spell out what each entry is for. The bare name says nothing about
        # which layout it is, and the two are not interchangeable: the short
        # card carries the 阵营 logo block and the tall one a full-height
        # illustration window, so picking the wrong one prints the wrong card.
        for stage, description in badge_card.STAGES:
            index = self.template_combo.findText(badge_card.template_name(stage))
            if index >= 0:
                self.template_combo.setItemData(
                    index, f"{badge_card.template_name(stage)}　·　{description}",
                    Qt.ToolTipRole,
                )
        if names:
            self.template_combo.setToolTip(
                "\n".join(
                    f"{badge_card.template_name(stage)}：{description}"
                    for stage, description in badge_card.STAGES
                )
            )
        if not names:
            # A disabled entry rather than one that can be picked but leads
            # nowhere: the template files have not been supplied yet.
            item = self.shape_combo.model().item(self.shape_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
            self.template_combo.setToolTip("未提供模板文件")
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.template_row = row("模板", self.template_combo)
        shape_layout.addWidget(self.template_row)

        # The way in to the card's wording and its lanyard slot. It sits here,
        # directly under the template picker, because this is where the user
        # already is when they choose 通行证 -- a menu entry somewhere else is
        # a feature they have to be told about, and the first version of this
        # put it in 文件 and left it disabled until a card existed, so it read
        # as a control that had not been built.
        self.content_button = QPushButton("编辑模板内容…")
        self.content_button.setMinimumHeight(26)
        self.content_button.clicked.connect(self.content_requested.emit)
        shape_layout.addWidget(self.content_button)

        # Irregular settings live inside the shape section rather than in a
        # panel of their own: "how the artwork combines with a shape" is part
        # of what the shape is, and the user reaches for it in the same breath
        # as picking 异形板.
        self.irregular_group = section("异形板设置", compact=True)
        irregular_layout = QVBoxLayout(self.irregular_group)
        irregular_layout.setContentsMargins(0, 0, 0, 0)
        irregular_layout.setSpacing(3)

        self.base_combo = combo([IrregularBase.LABELS[b] for b in IRREGULAR_BASES])
        self.base_combo.currentIndexChanged.connect(self._on_irregular_changed)
        irregular_layout.addWidget(row("基础形状", self.base_combo))

        self.base_top_slider = NoWheelSlider(Qt.Horizontal)
        self.base_top_slider.setRange(0, 100)
        self.base_top_slider.setValue(35)
        self.base_top_slider.valueChanged.connect(self._on_irregular_changed)
        self.base_top_row = row(
            "矩形上边界",
            self.base_top_slider,
            "矩形占下半部分的比例，越高底座越大；上半部分只保留图片描边",
        )
        irregular_layout.addWidget(self.base_top_row)

        self.image_x_slider = NoWheelSlider(Qt.Horizontal)
        self.image_x_slider.setRange(-100, 100)
        self.image_x_slider.setValue(0)
        self.image_x_slider.valueChanged.connect(self._on_irregular_changed)
        self.image_x_row = row(
            "图片水平位置",
            self.image_x_slider,
            "图片在矩形底座中的左右位置，0 为居中",
        )
        irregular_layout.addWidget(self.image_x_row)
        shape_layout.addWidget(self.irregular_group)
        layout.addWidget(shape_group)

        size_group = section("2 · 板框尺寸", compact=True)
        size_layout = QVBoxLayout(size_group)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(3)

        self.width_spin = spin(60.0, 3.0, 500.0, 1.0, " mm")
        self.height_spin = spin(60.0, 3.0, 500.0, 1.0, " mm")
        self.diameter_spin = spin(60.0, 3.0, 500.0, 1.0, " mm")
        for widget in (self.width_spin, self.height_spin, self.diameter_spin):
            widget.valueChanged.connect(self._on_board_size_changed)
        self.width_row = row("宽度", self.width_spin)
        self.height_row = row("高度", self.height_spin)
        self.diameter_row = row("直径", self.diameter_spin, "圆形板框只有一个尺寸")
        for holder in (self.width_row, self.height_row, self.diameter_row):
            size_layout.addWidget(holder)

        self.board_aspect_check = QCheckBox("锁定长宽比")
        self.board_aspect_check.setChecked(True)
        self.board_aspect_check.stateChanged.connect(self._on_board_size_changed)
        size_layout.addWidget(self.board_aspect_check)

        self.radius_spin = spin(3.0, 0.0, 250.0, 0.5, " mm", decimals=1)
        self.radius_spin.valueChanged.connect(self._on_board_size_changed)
        self.radius_row = row("圆角半径", self.radius_spin)
        size_layout.addWidget(self.radius_row)

        # The standee tab is part of the board's size rather than of its shape:
        # it hangs off whatever shape is chosen, and the two boards that use it
        # -- a rectangle and an irregular cut-out -- share it unchanged.
        self.standee_check = QCheckBox("立牌")
        self.standee_check.setToolTip("在板框最下方加一个插舌，用于插入立牌底座")
        self.standee_check.stateChanged.connect(self._on_standee_changed)
        size_layout.addWidget(self.standee_check)

        self.standee_group = section("立牌设置", compact=True)
        standee_layout = QVBoxLayout(self.standee_group)
        standee_layout.setContentsMargins(0, 0, 0, 0)
        standee_layout.setSpacing(3)

        self.standee_width_spin = spin(
            DEFAULT_TAB_WIDTH_MM, 0.1, 200.0, 0.1, " mm"
        )
        self.standee_height_spin = spin(
            DEFAULT_TAB_HEIGHT_MM, 0.05, 100.0, 0.1, " mm"
        )
        for widget in (self.standee_width_spin, self.standee_height_spin):
            widget.valueChanged.connect(self._on_standee_size_changed)
        standee_layout.addWidget(
            row("插舌宽度", self.standee_width_spin, "沿板框下边缘方向的长度")
        )
        standee_layout.addWidget(
            row("插舌高度", self.standee_height_spin, "从板框下边缘向下伸出的长度")
        )

        self.standee_x_slider = NoWheelSlider(Qt.Horizontal)
        self.standee_x_slider.setRange(-100, 100)
        self.standee_x_slider.setValue(0)
        self.standee_x_slider.valueChanged.connect(self._on_standee_changed)
        standee_layout.addWidget(
            row(
                "水平位置",
                self.standee_x_slider,
                "插舌沿板框下边缘左右移动，0 为下边缘中点",
            )
        )

        self.standee_note = QLabel()
        self.standee_note.setWordWrap(True)
        muted(self.standee_note)
        standee_layout.addWidget(self.standee_note)
        size_layout.addWidget(self.standee_group)
        layout.addWidget(size_group)

        image_group = section("3 · 图片尺寸", compact=True)
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(3)

        self.image_width_spin = spin(
            DEFAULT_IMAGE_LONG_EDGE_MM, 0.1, 1000.0, 1.0, " mm"
        )
        self.image_height_spin = spin(
            DEFAULT_IMAGE_LONG_EDGE_MM, 0.1, 1000.0, 1.0, " mm"
        )
        self.image_width_spin.valueChanged.connect(self._on_image_size_changed)
        self.image_height_spin.valueChanged.connect(self._on_image_size_changed)
        image_layout.addWidget(row("宽度", self.image_width_spin))
        image_layout.addWidget(row("高度", self.image_height_spin))

        self.image_aspect_check = QCheckBox("锁定长宽比")
        self.image_aspect_check.setChecked(True)
        self.image_aspect_check.stateChanged.connect(self._on_image_size_changed)
        image_layout.addWidget(self.image_aspect_check)

        self.image_note = QLabel()
        self.image_note.setWordWrap(True)
        muted(self.image_note)
        image_layout.addWidget(self.image_note)
        self.image_group = image_group
        layout.addWidget(image_group)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(48)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        spec = self._board.outline
        if spec.shape in SHAPE_CHOICES:
            self.shape_combo.setCurrentIndex(SHAPE_CHOICES.index(spec.shape))
        self._sync_controls()
        self._refresh_status()

        # The canvas edits the same placement this panel does, so a drag there
        # has to show up here. Wired by the panel itself rather than by the
        # factory, so the fields stay correct however the panel was opened.
        canvas = getattr(parent, "design_canvas", None)
        if canvas is not None:
            canvas.placement_changed.connect(self._on_canvas_changed)

    def _on_canvas_changed(self) -> None:
        """Adopt numbers the canvas changed, without echoing them back."""
        self._sync_controls()
        self._refresh_status()

    # -------------------------------------------------------------- seeding

    def _seed_from_image(self) -> None:
        """Size a first board to hold the artwork.

        The panel opens on real numbers instead of an arbitrary default, and
        on a board the artwork actually fits inside -- a circle seeded from a
        rectangle's shorter side would leave most of the picture hanging off
        the board.
        """
        image = self._board.image
        if image is None:
            return
        fitted = self._board.outline.fitted_around(image.width_mm, image.height_mm)
        self._board.outline = replace(
            fitted, width_mm=max(3.0, fitted.width_mm), height_mm=max(3.0, fitted.height_mm)
        )

    # -------------------------------------------------------------- binding

    def _is_template_mode(self) -> bool:
        return self.shape_combo.currentIndex() >= len(SHAPE_CHOICES)

    def _current_shape(self) -> str:
        if self._is_template_mode():
            return self._board.outline.shape
        return SHAPE_CHOICES[self.shape_combo.currentIndex()]

    def _current_base(self) -> str:
        return IRREGULAR_BASES[self.base_combo.currentIndex()]

    def _is_irregular(self) -> bool:
        return self._board.outline.shape == OutlineShape.IRREGULAR

    def _is_trace_base(self) -> bool:
        return self._is_irregular() and self._current_base() == IrregularBase.TRACE

    def _sync_controls(self) -> None:
        """Push the model into the controls, and show the rows it calls for."""
        spec = self._board.outline
        image = self._board.image

        # Under 原图裁切 the board and the artwork are one object, so the board
        # has to adopt the artwork's size. Done before the fields are filled so
        # the panel and the document cannot disagree, and it covers every route
        # into this state: choosing the base, switching shape, and resizing or
        # moving the picture on the canvas.
        if self._is_trace_base() and image is not None:
            spec.width_mm = image.width_mm
            spec.height_mm = image.height_mm

        if spec.standee:
            # Resizing the board can leave the tab hanging past its own edge.
            # Pull it back into the range the slider can express, so the
            # slider, the number and the geometry keep describing one tab
            # instead of three slightly different ones.
            travel = self._board.standee_travel_mm()
            spec.standee_offset_mm = max(
                -travel, min(travel, spec.standee_offset_mm)
            )

        self._updating = True
        try:
            self.width_spin.setValue(spec.width_mm)
            self.height_spin.setValue(spec.height_mm)
            self.diameter_spin.setValue(spec.diameter_mm)
            self.radius_spin.setValue(spec.corner_radius_mm)
            self.board_aspect_check.setChecked(spec.keep_aspect)
            self.base_combo.setCurrentIndex(
                IRREGULAR_BASES.index(spec.irregular_base)
                if spec.irregular_base in IRREGULAR_BASES
                else 0
            )
            self.base_top_slider.setValue(int(round(spec.base_top_ratio * 100)))
            self.standee_check.setChecked(spec.standee)
            self.standee_width_spin.setValue(spec.standee_width_mm)
            self.standee_height_spin.setValue(spec.standee_height_mm)
            self.standee_x_slider.setValue(self._standee_x_percent())
            if image is not None:
                self.image_width_spin.setValue(image.width_mm)
                self.image_height_spin.setValue(image.height_mm)
                self.image_aspect_check.setChecked(image.keep_aspect)
                self.image_x_slider.setValue(self._image_x_percent(image))
        finally:
            self._updating = False

        irregular = self._is_irregular()
        combining = irregular and not self._is_trace_base()
        single = spec.shape in OutlineShape.SQUARE_ONLY

        self.irregular_group.setVisible(irregular)
        # The two sliders describe the base and how the artwork sits in it, so
        # neither means anything for a pure cut-out.
        self.base_top_row.setVisible(combining)
        self.image_x_row.setVisible(combining)
        self.width_row.setVisible(not single)
        self.height_row.setVisible(not single)
        self.board_aspect_check.setVisible(not single)
        self.diameter_row.setVisible(single)
        self.radius_row.setVisible(
            spec.shape == OutlineShape.ROUNDED
            or (irregular and self._current_base() == IrregularBase.ROUNDED)
        )
        self.template_row.setVisible(self._is_template_mode())
        # Under 原图裁切 the board size *is* the artwork size, so a separate
        # artwork-size section would be the same numbers twice.
        self.image_group.setVisible(not self._is_trace_base())
        # The tab's settings only exist while the tab does, the same rule the
        # irregular base's sliders follow.
        self.standee_group.setVisible(spec.standee)
        if spec.standee:
            self._refresh_standee_note()

        if image is None:
            self.image_note.setText("尚未导入图片")
        elif self._is_trace_base():
            self.image_note.setText("")
        elif combining:
            travel = self._image_travel_mm(image)
            self.image_x_slider.setEnabled(travel > 0.05)
            if travel > 0.05:
                self.image_note.setText(f"左右可移动范围 ±{travel:.1f} mm")
            else:
                self.image_note.setText("图片已占满底座宽度，无法左右移动")
        else:
            self.image_note.setText("")

    def _image_travel_mm(self, image) -> float:
        """How far the artwork can slide sideways inside the base.

        This is the real free space, not a fixed percentage of the board: a
        picture as wide as the rectangle has nowhere to go, and saying
        otherwise would put a slider on screen that cannot do anything.
        """
        return max(0.0, (self._board.outline.width_mm - image.width_mm) / 2.0)

    def _image_x_percent(self, image) -> int:
        """The artwork's offset as a signed percentage of its available travel."""
        travel = self._image_travel_mm(image)
        if travel <= 1e-6:
            return 0
        return int(round(max(-1.0, min(1.0, image.x_mm / travel)) * 100))

    # ------------------------------------------------------------- handlers

    def _on_irregular_changed(self) -> None:
        """The base shape, or one of its two sliders, moved."""
        if self._updating:
            return
        spec = self._board.outline
        base = self._current_base()
        image = self._board.image

        became_combining = base != IrregularBase.TRACE and spec.irregular_base == (
            IrregularBase.TRACE
        )
        spec.irregular_base = base
        spec.base_top_ratio = self.base_top_slider.value() / 100.0

        if image is not None:
            if self.sender() is self.image_x_slider:
                travel = self._image_travel_mm(image)
                image.x_mm = self.image_x_slider.value() / 100.0 * travel
            elif became_combining:
                # Switching to a combine mode needs a base worth combining
                # with: the trace-only board was sized to the artwork, which
                # leaves no room around it. Give the base a margin so the
                # character stands on something and can be moved sideways.
                self._seed_base_around_image()

        self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    def _seed_base_around_image(self) -> None:
        """Size a combining base to the artwork plus a margin.

        Only applied when the mode is first entered, so the user's own numbers
        are never overwritten while they are tuning the sliders.
        """
        image = self._board.image
        if image is None:
            return
        spec = self._board.outline
        margin = max(2.0, min(image.width_mm, image.height_mm) * 0.12)
        spec.width_mm = max(3.0, image.width_mm + 2 * margin)
        spec.height_mm = max(3.0, image.height_mm + 2 * margin)

    def _reset_irregular_cache(self) -> None:
        """Drop the cached trace so the next paint re-reads the artwork.

        Edits go through the model rather than replacing it, so the cache --
        which only checks its key on read -- would otherwise be fine; this
        makes the invalidation explicit at the point of change.
        """
        self._board._irregular_cache = None

    def _on_shape_changed(self) -> None:
        if self._updating:
            return
        if self._is_template_mode():
            self._apply_selected_template()
        else:
            image = self._board.image
            artwork = None if image is None else (image.width_mm, image.height_mm)
            shape = self._current_shape()
            self._board.outline = replace(
                self._board.outline.after_shape_change(shape, artwork),
                template=None,
            )
            # A circle is not a card. Leaving the wording and the slot behind
            # would have the content panel offering to edit lines the board no
            # longer draws, and the slot would still be cut out of a round
            # board nobody asked to perforate.
            self._board.texts = []
            self._board.cutouts = []
            if shape == OutlineShape.IRREGULAR:
                # The base's own settings only exist for this shape, and a
                # board that was never irregular has no meaningful ratio yet.
                self._board.outline.base_top_ratio = self.base_top_slider.value() / 100.0
                self._board.outline.irregular_base = self._current_base()
                if self._current_base() != IrregularBase.TRACE:
                    self._seed_base_around_image()
            self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    def _on_template_changed(self) -> None:
        if self._updating or not self._is_template_mode():
            return
        self._apply_selected_template()
        self._sync_controls()
        self._apply()

    def _apply_selected_template(self) -> None:
        """Adopt the chosen template's shape, dimensions **and content**.

        Content too, at the user's request: the cards are mostly wording, and a
        blank template that arrived as a bare outline would send the user back
        to the EDA for the one thing this tool exists to do. Applying it twice
        is harmless -- the content is replaced wholesale from the template
        rather than merged, so re-picking 通行证 on a card whose text was
        mangled restores the template's own wording.

        ⚠️ **The template also decides where the imported pictures are cropped.**
        The user's instruction is that choosing 精0 or 精2 must not leave them
        right-clicking 放入短区／长区: the layout already says which opening it
        has. The reframing therefore happens here, at the one place that knows
        the template changed, rather than at each import -- which matters when
        the picture arrived *before* the card did, the path that used to leave
        it unframed at its default 76.2 mm.
        """
        if not template_names():
            return
        name = self.template_combo.currentText()
        # The layout in force *before* this call decides whether the pictures
        # need moving. Re-picking the card the board already has is how the
        # panel restores mangled wording, and it is also what happens if the
        # user simply walks back into this panel -- neither is a request to
        # redo a crop they may have adjusted by hand.
        previous = self._board.outline.template or ""
        try:
            self._board.outline = build_template(name)
        except KeyError:
            return
        if badge_card.is_badge_template(name):
            # ---- content
            #
            # The user's own edits are replaced, which is what makes re-picking
            # the template a way back to a known state. The chosen operator is
            # kept: it is not part of the template but of *this card*, and
            # dropping it would blank the name every time the shape was
            # re-selected.
            self._board.texts = badge_card.badge_texts()
            self._board.cutouts = badge_card.badge_cutouts()

        # A layout change moves the opening, so every picture the user brought
        # has to be re-framed for the new one -- including one that was imported
        # before the card existed, which is the 「还要右键取景」 path. Pictures
        # that have never been framed at all are caught by the same call, so a
        # card whose artwork arrived first still ends up cropped.
        self._reframe_card_illustrations(layout_changed=previous != name)

    def _reframe_card_illustrations(self, *, layout_changed: bool = True) -> int:
        """Re-crop the card's imported pictures for the template now applied.

        The canvas owns the geometry and the history, so this only asks it and
        reports what came back. A board that is not a card, or a template that
        is not 通行证, leaves everything untouched.

        ``layout_changed`` false means the same card is being re-applied, so
        only pictures that have never been framed are picked up. A picture the
        user deliberately put in the other opening by hand keeps it -- the
        template decides the crop, but it does not overrule a direct choice.

        No separate history entry is recorded here: the canvas records each
        reframing under one run key, and the template swap itself is recorded by
        the panel's own ``changed`` signal. Adding a third recording would only
        push an identical snapshot.
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return 0
        return canvas.frame_illustrations_for_card(reframe=layout_changed)

    def _on_board_size_changed(self) -> None:
        if self._updating:
            return
        spec = self._board.outline
        if spec.shape in OutlineShape.SQUARE_ONLY:
            spec.diameter_mm = self.diameter_spin.value()
        else:
            width = self.width_spin.value()
            height = self.height_spin.value()
            if self.board_aspect_check.isChecked() and spec.width_mm > 0:
                ratio = spec.height_mm / spec.width_mm
                self._updating = True
                try:
                    if self.sender() is self.width_spin:
                        height = width * ratio
                        self.height_spin.setValue(height)
                    elif self.sender() is self.height_spin:
                        width = height / ratio if ratio else height
                        self.width_spin.setValue(width)
                finally:
                    self._updating = False
            spec.width_mm = max(0.1, width)
            spec.height_mm = max(0.1, height)
        spec.corner_radius_mm = self.radius_spin.value()
        spec.keep_aspect = self.board_aspect_check.isChecked()

        # Under 原图裁切 the board is the artwork, so the two sizes are one
        # number and the artwork has to follow the board. Elsewhere the board
        # and the artwork are independent and moving one must not drag the
        # other.
        image = self._board.image
        if self._is_trace_base() and image is not None:
            image.width_mm = spec.width_mm
            image.height_mm = spec.height_mm
            self._updating = True
            try:
                self.image_width_spin.setValue(image.width_mm)
                self.image_height_spin.setValue(image.height_mm)
            finally:
                self._updating = False

        self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    # --------------------------------------------------------------- standee

    def _standee_x_percent(self) -> int:
        """The tab's offset as a signed percentage of how far it can travel.

        A percentage rather than millimetres, so the slider means the same
        thing whatever the board's size: 100 is "as far right as it goes", not
        "one millimetre to the right".
        """
        spec = self._board.outline
        travel = self._board.standee_travel_mm()
        if travel <= 1e-6:
            return 0
        return int(
            round(max(-1.0, min(1.0, spec.standee_offset_mm / travel)) * 100)
        )

    def _refresh_standee_note(self) -> None:
        """Say where the tab is, and whether it is attached at all.

        An unattached tab is reported here rather than drawn as a floating
        rectangle: the canvas shows the board the user actually has, and the
        panel is where the reason belongs.
        """
        travel = self._board.standee_travel_mm()
        # Reading the outline is what attempts the union, so this is the point
        # at which the model knows whether the tab actually attached.
        self._board.outline_polygon()
        error = self._board.standee_error
        self.standee_x_slider.setEnabled(travel > 0.05)
        if error:
            self.standee_note.setText(f"⚠ {error}")
            tone(self.standee_note, "warning")
            return
        contact = self._board.standee_contact_mm()
        width = self._board.outline.standee_width_mm
        muted(self.standee_note)
        if travel > 0.05:
            self.standee_note.setText(f"沿下边缘可左右移动 ±{travel:.1f} mm")
            return
        if width > contact + 0.05:
            # A tab wider than the flat it hangs from is *not* a defect: the
            # tab is sized to the base slot it plugs into, and its corners
            # hanging past a curved bottom is what that looks like. The union
            # still closes into one piece, so the note states the fact without
            # telling the user to shrink the very dimension the base needs.
            self.standee_note.setText(
                f"插舌比下边缘宽（下边缘可用 {contact:.1f} mm），"
                "两端伸到板外；位置固定在中点"
            )
            return
        self.standee_note.setText("下边缘没有可移动的余量，插舌固定在中点")

    def _on_standee_changed(self) -> None:
        """The 立牌 checkbox or its position slider moved."""
        if self._updating:
            return
        spec = self._board.outline
        spec.standee = self.standee_check.isChecked()
        if self.sender() is self.standee_x_slider:
            spec.standee_offset_mm = (
                self.standee_x_slider.value()
                / 100.0
                * self._board.standee_travel_mm()
            )
        # Switching the tab off and on again is a round trip through the same
        # control, not a reset: the size and position the user set are kept,
        # and _sync_controls pulls the position back in range if the board was
        # resized while the tab was off.
        self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    def _on_standee_size_changed(self) -> None:
        """The tab's width or height was typed in."""
        if self._updating:
            return
        spec = self._board.outline
        spec.standee_width_mm = self.standee_width_spin.value()
        spec.standee_height_mm = self.standee_height_spin.value()
        # A resize changes how far the tab can travel, so the offset is kept
        # where the slider says rather than left at a value that is now off
        # the end of the board.
        percent = self.standee_x_slider.value() / 100.0
        spec.standee_offset_mm = percent * self._board.standee_travel_mm()
        self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    def _on_image_size_changed(self) -> None:
        if self._updating:
            return
        image = self._board.image
        if image is None:
            return
        width = self.image_width_spin.value()
        height = self.image_height_spin.value()
        if self.image_aspect_check.isChecked() and image.width_mm > 0:
            ratio = image.height_mm / image.width_mm
            self._updating = True
            try:
                if self.sender() is self.image_width_spin:
                    height = width * ratio
                    self.image_height_spin.setValue(height)
                elif self.sender() is self.image_height_spin:
                    width = height / ratio if ratio else height
                    self.image_width_spin.setValue(width)
            finally:
                self._updating = False
        image.width_mm = max(0.1, width)
        image.height_mm = max(0.1, height)
        image.keep_aspect = self.image_aspect_check.isChecked()

        # A combining base is sized around the artwork, so resizing the
        # artwork has to carry the base with it; otherwise the silhouette walks
        # off the edge of a base that stayed put.
        if self._is_irregular() and not self._is_trace_base():
            self._seed_base_around_image()

        self._reset_irregular_cache()
        self._sync_controls()
        self._apply()

    # --------------------------------------------------------------- result

    def _apply(self) -> None:
        self._board.outline_ready = True
        self._refresh_status()
        self._run_key = self._field_key()
        self.changed.emit()

    def _field_key(self) -> str:
        """Which control produced the current board edit, for undo coalescing.

        Same idea as the image property pane's key: dragging the board width
        spinner is one undo step, not one per increment. When nothing can be
        identified the key falls back to the panel itself, which still merges a
        continuous run rather than pushing a step per motion event.
        """
        sender = self.sender()
        field = "board"
        for name, value in vars(self).items():
            if value is sender:
                field = name
                break
        return f"board:{field}"

    def edit_run_key(self) -> str | None:
        """The run key for the edit just reported."""
        return self._run_key

    def _refresh_status(self) -> None:
        """Describe the board the canvas is now showing."""
        spec = self._board.outline
        problems = spec.validate()
        if problems:
            self.status_label.setText("⚠ " + "；".join(problems))
            warn_surface(self.status_label, size=11, padding="6px 8px")
            return

        label = OutlineShape.LABELS.get(spec.shape, spec.shape)
        if spec.shape in OutlineShape.SQUARE_ONLY:
            text = f"当前：{label}　直径 {spec.diameter_mm:.1f} mm"
        elif spec.shape == OutlineShape.IRREGULAR:
            base = IrregularBase.LABELS.get(spec.irregular_base, spec.irregular_base)
            text = f"当前：{label}　{base}"
            if spec.irregular_base == IrregularBase.TRACE:
                text += "　板框尺寸即图片尺寸"
            else:
                text += (
                    f"　底座 {spec.width_mm:.1f} × {spec.height_mm:.1f} mm"
                    f"　上边界 {spec.base_top_ratio * 100:.0f}%"
                )
        else:
            text = f"当前：{label}　{spec.width_mm:.1f} × {spec.height_mm:.1f} mm"
            if spec.shape == OutlineShape.ROUNDED:
                text += f"　圆角 {spec.corner_radius_mm:.1f} mm"
        if spec.template:
            text += f"　模板 {spec.template}"
        if spec.standee:
            # The board's own size is quoted above; the tab is the extra piece
            # hanging below it, so it is named as such rather than folded into
            # those numbers and left the reader to guess.
            text += (
                f"　立牌插舌 {spec.standee_width_mm:.1f} × "
                f"{spec.standee_height_mm:.1f} mm"
            )
            if spec.standee_offset_mm:
                text += f"（偏移 {spec.standee_offset_mm:+.1f} mm）"
        image = self._board.image
        if image is not None:
            text += f"\n图片 {image.width_mm:.1f} × {image.height_mm:.1f} mm"
        # Say so plainly when there is nothing to trace, rather than letting an
        # empty canvas look like a broken setting.
        if spec.shape == OutlineShape.IRREGULAR and image is None:
            text += "\n尚无图片：请先导入图片后再生成异形板框"
        # A tab that did not attach must say so here as well as in its own
        # section. The canvas is showing a board without the tab the user
        # switched on, and every other line above describes that board as if
        # nothing were wrong.
        if spec.standee and self._board.standee_error:
            text += f"\n⚠ 立牌插舌未接上：{self._board.standee_error}"
        # A trace that failed must say so. Silently drawing the bare base would
        # look like a setting applied successfully.
        error = self._board._irregular_error
        if spec.shape == OutlineShape.IRREGULAR and error:
            text += f"\n⚠ 描边失败：{error}"
            warn_surface(self.status_label, size=11, padding="6px 8px")
            self.status_label.setText(text)
            return
        self.status_label.setText(text)
        info_surface(self.status_label)


class ColourSwatch(QPushButton):
    """One clickable colour chip in the preset row.

    Drawn as a filled rounded square rather than a labelled button because the
    choice *is* the colour: a row of names would make the user read sixteen
    words to find the one they can already see.
    """

    SIZE = 34

    def __init__(self, name: str, colour: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.colour = colour
        self.name = name
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{name}　{colour}")
        self._selected = False
        self.setFocusPolicy(Qt.StrongFocus)

    def set_selected(self, value: bool) -> None:
        if self._selected != value:
            self._selected = value
            self.update()

    def is_selected(self) -> bool:
        return self._selected

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(3, 3, self.width() - 6, self.height() - 6)

        painter.setBrush(QColor(self.colour))
        # The hairline keeps a white swatch visible against a white panel; the
        # double ring below is what marks the current choice, so this one must
        # stay subtle or every swatch looks selected.
        painter.setPen(QPen(QColor(0, 0, 0, 40), 1))
        painter.drawRoundedRect(rect, 9, 9)

        if self._selected:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(TEXT), 2))
            painter.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 11, 11)

        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(ACCENT), 2, Qt.DotLine))
            painter.drawRoundedRect(rect.adjusted(-4.5, -4.5, 4.5, 4.5), 12, 12)


class ThemeSettingsPanel(FloatingPanel):
    """The colour settings: presets, light/dark, and a full custom editor.

    Every change applies immediately. There is no OK button on purpose -- the
    whole point of picking a colour is seeing it, and a dialog that has to be
    dismissed before the change is visible turns choosing into guesswork. The
    change is saved as it is made, so a crash or a forced close cannot lose it.
    """

    #: Wide enough for six colour chips plus their gaps. Narrower and the grid
    #: clips; wider and the labels and colour values drift apart.
    WIDTH = 440

    def __init__(self, parent: QWidget, scheme) -> None:
        super().__init__(parent, "界面设置", width=self.WIDTH)
        self._scheme = scheme
        self._updating = False
        self._swatches: list[ColourSwatch] = []

        layout = self.content_layout

        # ------------------------------------------------------- presets
        preset_group = section("1 · 主题色", compact=True)
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)

        from pcbart.ui_theme import PRESETS

        # Six per row rather than nine: seventeen presets in a nine-wide grid
        # leave a ragged half-empty row, while six gives three tidy rows. The
        # panel is also narrow enough that nine columns makes each chip small
        # enough to be fiddly to hit.
        columns = 6
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for index, (name, colour) in enumerate(PRESETS):
            swatch = ColourSwatch(name, colour)
            swatch.clicked.connect(
                lambda _=False, c=colour: self._choose_seed(c)
            )
            self._swatches.append(swatch)
            grid.addWidget(swatch, index // columns, index % columns)
        # Keep the chips left-aligned when the last row is short, instead of
        # letting Qt spread them across the whole width.
        for column in range(columns):
            grid.setColumnStretch(column, 0)
        grid.setColumnStretch(columns, 1)
        preset_layout.addLayout(grid)

        self.preset_note = QLabel()
        self.preset_note.setWordWrap(True)
        muted(self.preset_note)
        preset_layout.addWidget(self.preset_note)
        layout.addWidget(preset_group)

        # ---------------------------------------------------- light/dark
        mode_group = section("2 · 明暗", compact=True)
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        self.mode_combo = combo(["浅色", "深色"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(row("外观", self.mode_combo))
        layout.addWidget(mode_group)

        # ----------------------------------------------------- advanced
        self.advanced_button = QPushButton("高级：逐个调整颜色")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setIcon(qt_theme_icon("custom", MUTED, size=18))
        self.advanced_button.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.advanced_button)

        self.advanced_group = section("3 · 高级", compact=True)
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)

        from pcbart.ui_theme import ROLE_LABELS

        self._role_buttons: dict[str, QPushButton] = {}
        for role, label in ROLE_LABELS:
            button = QPushButton()
            button.setFixedSize(58, 26)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, r=role: self._edit_role(r))
            self._role_buttons[role] = button
            advanced_layout.addWidget(colour_row(label, button))

        self.reset_button = QPushButton("重置为预设")
        self.reset_button.setIcon(qt_theme_icon("erase", MUTED, size=17))
        self.reset_button.clicked.connect(self._reset_custom)
        advanced_layout.addWidget(self.reset_button)
        self.advanced_group.setVisible(False)
        layout.addWidget(self.advanced_group)

        # ------------------------------------------------- external model
        layout.addWidget(self._build_model_group())

        # ---------------------------------------------------------- updates
        layout.addWidget(self._build_update_group())

        layout.addStretch(1)

        self._sync()
        self._sync_model()

    # ------------------------------------------------------- external model

    def _build_model_group(self) -> QWidget:
        """The external redraw model: pick a folder, or be told what is missing.

        The model is **not shipped** -- 183 MB of weights against a 178 MB
        installer, and a CPU redraw peaks at 4.6 GB of RAM -- so this section
        is the entire delivery mechanism. It therefore has to do three jobs in
        the panel rather than in a log file: show which folder is in force,
        let the user point somewhere else, and say what is missing when the
        folder is not yet a model.
        """
        group = section("4 · 线稿重绘模型（外接）", compact=True)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # The path is shown in a read-only field rather than a label so a long
        # path can be selected and copied -- on the machine that has the model
        # the user often wants to hand it to someone else. The tooltip carries
        # the whole path because the field is narrower than most of them.
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("尚未设置")
        layout.addWidget(row("模型文件夹", self.model_path_edit))

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        self.model_browse_button = QPushButton("选择文件夹…")
        self.model_browse_button.setIcon(qt_theme_icon("open_folder", MUTED, size=17))
        self.model_browse_button.clicked.connect(self._choose_model_dir)
        buttons.addWidget(self.model_browse_button)

        self.model_clear_button = QPushButton("清除")
        self.model_clear_button.setIcon(qt_theme_icon("erase", MUTED, size=17))
        self.model_clear_button.clicked.connect(self._clear_model_dir)
        buttons.addWidget(self.model_clear_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        self.model_hint = QLabel()
        self.model_hint.setWordWrap(True)
        muted(self.model_hint)
        layout.addWidget(self.model_hint)
        return group

    def _build_update_group(self) -> QWidget:
        """Section 5: whether to check for updates, and a manual check button.

        The manual button is deliberately **not** tied to the checkbox. Turning
        automatic checks off must not take away the only way to ask for an
        update -- a feature reachable only through automation is this project's
        recurring "only the script calls it" defect.
        """
        from pcbart import __version__ as current_version

        group = section("5 · 更新", compact=True)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.update_check_box = QCheckBox("启动时检查更新")
        self.update_check_box.setChecked(self._auto_check_enabled())
        self.update_check_box.toggled.connect(self._set_auto_check)
        layout.addWidget(self.update_check_box)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.update_version_label = QLabel(f"当前版本：{current_version}")
        muted(self.update_version_label)
        row.addWidget(self.update_version_label)
        row.addStretch(1)
        self.update_now_button = QPushButton("立即检查更新")
        self.update_now_button.clicked.connect(self._check_now)
        row.addWidget(self.update_now_button)
        layout.addLayout(row)
        return group

    # -------------------------------------------------------------- updates

    @staticmethod
    def _auto_check_enabled() -> bool:
        """Whether startup checks are on. Default on -- it is the point."""
        from pcbart.ui_theme import load_settings

        value = load_settings().get("update_check")
        return True if value is None else bool(value)

    def _set_auto_check(self, enabled: bool) -> None:
        from pcbart.ui_theme import save_settings_keeping

        save_settings_keeping(update_check=bool(enabled))

    def _check_now(self) -> None:
        start_update_check(self.window(), manual=True)

    def focus_model_section(self) -> None:
        """Bring the model controls into view and put the keyboard on them.

        Called when another panel sends the user here (the preprocessing panel's
        external-model button). Landing at the top of a long settings panel and
        leaving them to find section 4 themselves is the kind of small betrayal
        that makes a shortcut worse than none.
        """
        self.model_browse_button.setFocus(Qt.OtherFocusReason)
        area = self.findChild(QScrollArea)
        if area is not None and area.widget() is not None:
            area.ensureWidgetVisible(self.model_browse_button, 24, 24)

    def _choose_model_dir(self) -> None:
        """Let the user point at a downloaded model folder."""
        start = self.model_path_edit.text().strip() or ""
        if not start:
            from pcbart.model_paths import resolve_model_dir

            start = str(resolve_model_dir())
        chosen = QFileDialog.getExistingDirectory(self, "选择 Anime2Sketch 模型文件夹", start)
        if not chosen:
            return
        save_settings_keeping(a2s_model=chosen)
        self._sync_model()

    def _clear_model_dir(self) -> None:
        """Forget the chosen folder, falling back to the default drop-in spot."""
        save_settings_keeping(a2s_model=None)
        self._sync_model()

    def _sync_model(self) -> None:
        """Show where the model is coming from and whether it is complete."""
        from pcbart.model_paths import (
            DEFAULT_VARIANT,
            check_model_dir,
            resolve_model_dir,
        )

        # Deliberately the folder *in force*, even when it turns out to be
        # unusable. Showing a different folder than the one the feature would
        # actually load is the "interface says one thing, the program does
        # another" defect this project treats as a bug; a wrong pick stays on
        # screen next to what is missing from it.
        effective = resolve_model_dir()
        report = check_model_dir(effective, DEFAULT_VARIANT)
        self.model_path_edit.setText(str(effective))
        self.model_path_edit.setToolTip(str(effective))
        self.model_clear_button.setVisible(bool(self._saved_model_setting()))

        if report["ok"]:
            self.model_status.setText(
                f"✅ 可用：架构已就位，权重 {DEFAULT_VARIANT}"
            )
            tone(self.model_status, "success")
        else:
            # Only weight sets that are NOT the one in use are worth naming --
            # "you have netG.pth" is useful when that is why the folder looked
            # fine, and noise otherwise.
            others = [
                name
                for name, here in report["weights"].items()
                if here and name != DEFAULT_VARIANT
            ]
            extra = (
                f"（这个文件夹里有 {'、'.join(others)}，"
                f"但项目用的是 {DEFAULT_VARIANT}）"
                if others
                else ""
            )
            note_text(
                self.model_status,
                f"⚠️ 还不能用：缺少 {'；'.join(report['missing'])}。{extra}",
            )
            tone(self.model_status, "warning")

        note_text(self.model_hint, self._model_hint())

    def _saved_model_setting(self) -> str:
        from pcbart.ui_theme import load_settings

        value = load_settings().get("a2s_model")
        return value if isinstance(value, str) else ""

    def _model_hint(self) -> str:
        """The one fact worth keeping: an environment override outranks this panel.

        ⚠️ The folder picked here is the *third* source consulted, so someone
        who set the environment variable and then wonders why picking a folder
        changes nothing needs to read that here rather than discover it. The
        "where to get a model" prose that used to live here was removed at the
        user's request; the 线稿重绘 dialog in the preprocessing panel carries
        that guidance instead.
        """
        from pcbart.model_paths import MODEL_ENV

        override = os.environ.get(MODEL_ENV)
        if not override:
            return ""
        return f"⚠️ 环境变量 {MODEL_ENV} 已设置，优先于这里的选择：{override}"

    # ------------------------------------------------------------- actions

    def _choose_seed(self, colour: str) -> None:
        from pcbart.ui_theme import scheme_from_seed

        if self._updating:
            return
        # Picking a preset discards any hand edits. Keeping them would produce
        # a scheme that is half the old customisation and half the new preset,
        # which is neither thing the user asked for.
        self._apply(scheme_from_seed(colour, dark=self._scheme.dark))

    def _on_mode_changed(self, index: int) -> None:
        from pcbart.ui_theme import scheme_from_roles, scheme_from_seed

        if self._updating:
            return
        dark = index == 1
        if self._scheme.custom:
            # The hand-picked colours are kept across a light/dark switch --
            # throwing them away because the user also wanted a dark panel
            # would lose work they did not undo.
            self._apply(
                scheme_from_roles(
                    self._scheme.roles, seed=self._scheme.seed, dark=dark
                )
            )
        else:
            self._apply(scheme_from_seed(self._scheme.seed, dark=dark))

    def _edit_role(self, role: str) -> None:
        from PySide6.QtWidgets import QColorDialog

        current = QColor(self._scheme[role])
        chosen = QColorDialog.getColor(
            current, self, f"选择颜色 — {dict(ROLE_LABELS)[role]}"
        )
        if not chosen.isValid():
            return
        from pcbart.ui_theme import scheme_from_roles

        roles = dict(self._scheme.roles)
        roles[role] = chosen.name()
        self._apply(
            scheme_from_roles(roles, seed=self._scheme.seed, dark=self._scheme.dark)
        )

    def _reset_custom(self) -> None:
        from pcbart.ui_theme import scheme_from_seed

        self._apply(scheme_from_seed(self._scheme.seed, dark=self._scheme.dark))

    def _on_advanced_toggled(self, shown: bool) -> None:
        self.advanced_group.setVisible(shown)
        self.resize(self.width(), self.sizeHint().height())

    def _apply(self, scheme) -> None:
        """Hand the new scheme to the window, which owns the switch."""
        window = self.parentWidget()
        while window is not None and not hasattr(window, "apply_colour_scheme"):
            window = window.parentWidget()
        if window is None:
            return
        self._scheme = scheme
        self._sync()
        window.apply_colour_scheme(scheme)

    # ---------------------------------------------------------------- view

    def set_scheme(self, scheme) -> None:
        """Adopt a scheme chosen elsewhere (the panel is rebuilt on switch)."""
        self._scheme = scheme
        self._sync()

    def _sync(self) -> None:
        from pcbart.ui_theme import PRESETS, scheme_from_seed

        self._updating = True
        try:
            for swatch in self._swatches:
                try:
                    same = scheme_from_seed(swatch.colour, self._scheme.dark)
                except ValueError:
                    same = None
                swatch.set_selected(
                    not self._scheme.custom
                    and same is not None
                    and same.roles == self._scheme.roles
                )
            self.mode_combo.setCurrentIndex(1 if self._scheme.dark else 0)

            if self._scheme.custom:
                self.preset_note.setText("当前是自定义配色（已手动改过颜色）。")
            else:
                name = next(
                    (
                        n
                        for n, c in PRESETS
                        if scheme_from_seed(c, self._scheme.dark).roles
                        == self._scheme.roles
                    ),
                    None,
                )
                self.preset_note.setText(
                    f"当前：{name}　{self._scheme.seed}"
                    if name
                    else f"当前：自定义主色　{self._scheme.seed}"
                )

            for role, button in self._role_buttons.items():
                colour = self._scheme[role]
                button.setText(colour.upper())
                button.setStyleSheet(
                    f"QPushButton {{ background: {colour};"
                    f" border: 1px solid {BORDER}; border-radius: 8px;"
                    f" color: {self._scheme['on_accent'] if role == 'accent' else _readable_on(colour)};"
                    f" font-size: 10px; padding: 0; }}"
                )
        finally:
            self._updating = False


def _readable_on(colour: str) -> str:
    """Black or white, whichever reads on ``colour``."""
    from pcbart.ui_theme import _on_colour

    return _on_colour(colour)


class UpdatePanel(FloatingPanel):
    """The "a new version is out" panel: version, changelog, download button.

    Layout follows MAA's update dialog -- the version centred at the top and
    the changelog filling the rest -- because that shape is already familiar
    from the tools this one sits beside, and it puts the two things a user
    actually reads (which version, what changed) above everything else.

    Unlike MAA's dialog this one carries the download button itself. That is
    the whole request: the point is not having to go to the website.

    Notes are rendered as plain text, one label per entry, rather than through
    a Markdown renderer. They come from a manifest this project publishes so
    Markdown would be safe today, but the entries are one-line sentences by
    convention and rich text buys nothing.
    """

    #: Wide enough for a changelog line without wrapping, and tall enough that
    #: a ten-entry release does not need scrolling on a 1080p screen.
    WIDTH = 520
    HEIGHT = 470

    def __init__(self, parent: QWidget, release, *, on_download=None) -> None:
        super().__init__(parent, "发现新版本", width=self.WIDTH)
        self._release = release
        self._on_download = on_download
        self._downloading = False

        layout = self.content_layout

        # ------------------------------------------------- version headline
        # Centred and larger than everything else, the way MAA shows its tag.
        version = QLabel(f"v{release.version}")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(version)

        if release.date:
            date_label = QLabel(release.date)
            date_label.setAlignment(Qt.AlignCenter)
            muted(date_label)
            layout.addWidget(date_label)

        # ------------------------------------------------------ changelog
        #
        # ⚠️ **No nested QScrollArea here.** ``FloatingPanel`` already puts the
        # content in one, and a second one inside it was a real defect: the
        # inner scroller reported its own minimum height, the content widget
        # grew past the panel, and the action buttons ended up below the
        # visible area -- exactly when the user needed them (the download had
        # just started). Long changelogs use the panel's own scrolling.
        notes_group = section("更新内容", compact=True)
        notes_layout = QVBoxLayout(notes_group)
        notes_layout.setContentsMargins(4, 0, 4, 0)
        notes_layout.setSpacing(4)
        for note in release.notes or ("（本次更新没有写说明）",):
            entry = QLabel(f"· {note}")
            entry.setWordWrap(True)
            entry.setTextInteractionFlags(Qt.TextSelectableByMouse)
            notes_layout.addWidget(entry)
        notes_layout.addStretch(1)
        layout.addWidget(notes_group, 1)

        # ------------------------------------------------------- size hint
        if release.size:
            size_label = QLabel(f"安装包 {release.size / 1024 / 1024:.1f} MB")
            muted(size_label)
            layout.addWidget(size_label)

        # --------------------------------------------------- footer (buttons)
        #
        # The status line and the buttons live in the panel's pinned footer,
        # not in the scrolling area. They are the only interactive part of this
        # panel, and a long changelog must not be able to push them out of
        # sight -- ``set_footer`` exists for precisely this case.
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        muted(self.status_label)
        footer_layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addStretch(1)
        self.later_button = QPushButton("稍后")
        self.later_button.clicked.connect(self.close_panel)
        buttons.addWidget(self.later_button)

        self.download_button = QPushButton("下载并安装")
        self.download_button.setDefault(True)
        self.download_button.clicked.connect(self._start_download)
        buttons.addWidget(self.download_button)
        footer_layout.addLayout(buttons)

        self.set_footer(footer)
        self.resize(self.WIDTH, self.HEIGHT)

    def _start_download(self) -> None:
        """Hand off to the downloader, which runs off the UI thread.

        The button is disabled here rather than on the first progress callback:
        a second click before the worker starts would begin a second download
        into the same file.
        """
        if self._downloading:
            return
        self._downloading = True
        self.download_button.setEnabled(False)
        self.status_label.setText("正在下载…")
        if self._on_download is not None:
            self._on_download(self)

    def set_progress(self, done: int, total: int) -> None:
        """Report how much has arrived. Plain counts, not a percentage bar."""
        if total > 0:
            self.status_label.setText(
                f"正在下载… {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
            )
        else:
            self.status_label.setText(f"正在下载… {done / 1024 / 1024:.1f} MB")

    def finish(self, message: str, *, failed: bool = False) -> None:
        """Show the end state, and let the user try again if it failed."""
        self.status_label.setText(message)
        if failed:
            self._downloading = False
            self.download_button.setEnabled(True)
            self.download_button.setText("重试")


#: The open update panel, if any. Module-level because the panel is opened by
#: a background check that has no other handle on the widget.
_update_panel: "UpdatePanel | None" = None


class _UpdateCheckWorker(QThread):
    """Runs the manifest fetch off the UI thread.

    The check is short (3 s timeout) but it is network I/O during startup: on
    a slow or filtered connection, doing it inline would freeze the window
    before it ever painted.
    """

    finished_check = Signal(object)

    def __init__(self, parent, local_version: str) -> None:
        super().__init__(parent)
        self._local_version = local_version

    def run(self) -> None:  # noqa: D102 - QThread entry point
        from pcbart.update_check import check_for_update

        try:
            status = check_for_update(self._local_version)
        except Exception:  # pragma: no cover - the checker never raises by design
            status = None
        self.finished_check.emit(status)


class _UpdateDownloadWorker(QThread):
    """Downloads and verifies the installer, reporting progress by signal."""

    progress = Signal(int, int)
    finished_download = Signal(bool, str, str)  # ok, message, path

    def __init__(self, parent, release, target) -> None:
        super().__init__(parent)
        self._release = release
        self._target = target

    def run(self) -> None:  # noqa: D102 - QThread entry point
        from pcbart.update_check import download, quarantine_partial, verify_download

        ok = download(
            self._release,
            self._target,
            timeout=60.0,
            on_progress=lambda done, total: self.progress.emit(done, total),
        )
        if not ok:
            self.finished_download.emit(False, "下载失败，请检查网络后重试。", "")
            return
        if not verify_download(self._target, self._release.sha256):
            # Move the bad file aside rather than deleting it, and rather than
            # leaving it in place for the next attempt to resume onto.
            quarantine_partial(self._target)
            self.finished_download.emit(False, "下载的文件校验不通过，已丢弃。", "")
            return
        self.finished_download.emit(True, "下载完成，正在启动安装程序…", str(self._target))


def _updates_dir():
    """Where installers are staged. Under the cache root, never beside the exe."""
    from pcbart.paths import cache_dir

    return cache_dir("updates")


def _auto_check_enabled_setting() -> bool:
    """Whether startup update checks are on, read without a panel around.

    Defaults to on: automatic checking is the feature. A missing or malformed
    preference reads as the default rather than as "off", so a hand-edited
    settings file cannot silently disable it.
    """
    from pcbart.ui_theme import load_settings

    value = load_settings().get("update_check")
    return True if value is None else bool(value)


def start_update_check(main_window: QWidget, *, manual: bool = False) -> None:
    """Check for a newer release and, if there is one, offer it.

    ``manual`` changes only what a *negative* result looks like: an automatic
    check says nothing when there is no update, while a manual one has to
    answer the user who just clicked the button.
    """
    from pcbart import __version__

    window = main_window
    worker = _UpdateCheckWorker(window, __version__)
    # Parented to the window but referenced by the connection, so it is not
    # collected while running.
    window._update_check_worker = worker  # type: ignore[attr-defined]

    def on_done(status) -> None:
        window._update_check_worker = None  # type: ignore[attr-defined]
        if status is None or not status.available or status.release is None:
            if manual:
                reason = getattr(status, "reason", "") or "检查失败"
                QMessageBox.information(window, "检查更新", f"没有可用更新：{reason}")
            return
        show_update(window, status.release, on_download=_begin_update_download)

    worker.finished_check.connect(on_done)
    worker.start()


def _begin_update_download(panel: "UpdatePanel") -> None:
    """Download and verify, then launch the installer and quit.

    Kept as a module function rather than a panel method so the panel keeps
    knowing nothing about paths, threads or process launching.
    """
    window = panel.window()
    release = panel._release
    target = staged_path_for(release)

    worker = _UpdateDownloadWorker(window, release, target)
    window._update_download_worker = worker  # type: ignore[attr-defined]

    worker.progress.connect(panel.set_progress)

    def on_finished(ok: bool, message: str, path: str) -> None:
        window._update_download_worker = None  # type: ignore[attr-defined]
        if not ok:
            panel.finish(message, failed=True)
            return
        panel.finish(message)
        _launch_installer(window, path, panel)

    worker.finished_download.connect(on_finished)
    worker.start()


def staged_path_for(release):
    """Where ``release`` will be downloaded to."""
    from pcbart.update_check import staged_path

    return staged_path(release, _updates_dir())


def _launch_installer(window: QWidget, path: str, panel: "UpdatePanel") -> None:
    """Start the installer, then close the application.

    The order matters and so does the delay: Windows will not let an installer
    replace a running executable, so the application has to be gone before the
    setup program starts touching files. The installer is launched first (so a
    failure to launch can still be reported here), then the close is scheduled
    a moment later to give the new process time to appear.
    """
    import subprocess

    try:
        subprocess.Popen([path], close_fds=True)
    except OSError as exc:
        panel.finish(f"无法启动安装程序：{exc}", failed=True)
        return

    panel.finish("安装程序已启动，本程序即将退出。")
    QTimer.singleShot(1200, lambda: window.close())


def show_update(main_window: QWidget, release, *, on_download=None) -> UpdatePanel:
    """Open the update panel, replacing any panel already showing.

    The old panel is always replaced rather than reused: it displays one
    release, and mutating a live panel in place could leave the previous
    release's header on screen while the notes below it are the new ones.

    Closing the old one has to go through a liveness check. After the owning
    window is destroyed the module global still holds a Python wrapper whose
    C++ half is gone, and calling ``close_panel`` on that raises -- the same
    trap ``reusable_panel`` exists for.
    """
    global _update_panel
    previous = live_panel("_update_panel")
    if previous is not None:
        try:
            if previous.parentWidget() is main_window:
                previous.close_panel()
        except RuntimeError:
            pass  # C++ half already gone; nothing to close.
    panel = UpdatePanel(main_window, release, on_download=on_download)
    _update_panel = panel
    panel.show()
    panel.place_top_right()
    return panel


def show_settings(main_window: QWidget) -> "ThemeSettingsPanel | None":
    """Open the colour settings panel, creating it on first use."""
    global _settings_panel
    _settings_panel, created = reusable_panel(
        _settings_panel,
        main_window,
        lambda parent: ThemeSettingsPanel(parent, current_scheme()),
    )
    if created:
        _settings_panel.resize(430, 620)
        _settings_panel.place_top_right()
    _settings_panel.set_scheme(current_scheme())
    _settings_panel.show()
    _settings_panel.raise_()
    return _settings_panel


_settings_panel: "ThemeSettingsPanel | None" = None


def show_board_setup(main_window: QWidget) -> BoardSetupPanel | None:
    """Show the board-setup panel inside the main window.

    A sibling of the preprocessing panel: the same kind of floating panel, on
    the same toolbar row, and just as non-modal. It needs the design canvas, so
    it reports rather than fails silently when the window is not built yet.
    """
    global _board_panel
    canvas = getattr(main_window, "design_canvas", None)
    if canvas is None:
        QMessageBox.information(
            main_window, "暂不可用", "主窗口尚未准备好设计画布。"
        )
        return None

    _board_panel, created = reusable_panel(
        _board_panel,
        main_window,
        lambda parent: BoardSetupPanel(parent, canvas.board, canvas),
    )
    # Refresh the reference even on the reuse path: a window can be re-pointed
    # at another board, and a stale canvas would re-crop the wrong document.
    _board_panel.canvas = canvas
    if created:
        _board_panel.resize(400, 620)
        _board_panel.place_top_right()

        def on_changed() -> None:
            # This panel's subject is the board, and the board is what the view
            # frames, so its edits take the fit again -- the user's own zoom is
            # kept on top of it. Dragging a handle on the canvas is the one
            # resize that must *not* come through here: that would rescale the
            # view under the cursor and the edge would never catch up with it.
            canvas.refit_view()
            canvas.record_external_change(_board_panel.edit_run_key())
            inspector = getattr(main_window, "image_inspector", None)
            if inspector is not None and inspector.isVisible():
                selected = canvas.selected
                if selected is not None:
                    inspector.set_placement(selected)
            preprocess = live_panel("_preprocess_panel")
            if preprocess is not None:
                try:
                    if preprocess.parentWidget() is main_window:
                        preprocess._refresh_board_note()
                except RuntimeError:
                    pass

        _board_panel.changed.connect(on_changed)
        _board_panel.content_requested.connect(
            lambda: open_template_content(main_window)
        )

    _board_panel.show()
    _board_panel.raise_()
    return _board_panel


_board_panel: BoardSetupPanel | None = None


def open_template_content(window) -> "TemplateContentPanel | None":
    """Open 模板内容 for whatever board is on screen.

    One entry point, used by the menu and by the button inside 板框设置, so the
    two can never drift into doing different things.

    **The board does not have to be a card yet.** The panel opens regardless and
    says what is missing, with a button that applies the 通行证 template. The
    first version refused to open unless the board was already a card and
    disabled its menu entry until then -- which meant a user who had not yet
    applied a template saw a grey entry and concluded the feature did not
    exist. Refusing with an explanation is only better than doing nothing if
    the explanation is actually seen.

    Edits are routed into the canvas history here rather than inside the panel:
    the canvas owns the undo stack, and an edit that repaints but is not
    recorded is worse than no undo at all, because the user believes Ctrl+Z
    covers it.
    """
    canvas = getattr(window, "design_canvas", None)
    if canvas is None:
        return None
    panel = show_template_content(window, canvas.board)

    def on_changed() -> None:
        canvas.update()
        canvas.record_external_change()

    # A second call must not stack a second connection, or one edit would be
    # recorded twice and one Ctrl+Z would only undo half of it. Qt has no
    # "is anything connected" query, so the attempt is made and the warning it
    # raises when nothing was connected is swallowed -- that warning is written
    # to stderr by the binding and would otherwise appear every time the panel
    # is opened for the first time.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            panel.changed.disconnect()
        except (RuntimeError, TypeError):
            pass
    panel.changed.connect(on_changed)
    return panel


class TemplateContentPanel(FloatingPanel):
    """Everything the blank card carries, made editable.

    Two lists and a search box, because those are the three things the user
    asked to be able to change: *what the card says*, *where the hole is*, and
    *whose card it is*.

    **The operator search writes into the document, not into the text.** Picking
    someone stores their name and number on the board and the lines keep their
    ``{代号}`` / ``{编号}`` tokens, so switching operator re-renders every line
    at once and the template's own wording survives. Substituting the values
    into the strings instead would make the choice irreversible -- the second
    operator would have to find the first one's name inside ``ARKNIGHTS-R001``
    to replace it.

    The panel is bound to a live board and re-pointed on a tab switch, the same
    as every other panel: after switching tabs it has to edit the board that is
    on screen, not the one it was opened for.
    """

    changed = Signal()
    """Emitted after an edit, so the canvas repaints and the change is undoable."""

    def __init__(self, parent: QWidget, board: ArtBoard, canvas=None) -> None:
        super().__init__(parent, "模板内容")
        self.resize(440, 680)
        self._board = board
        self.canvas = canvas
        """The design canvas, so 套用模板 can re-crop the imported pictures.

        Optional for the same reason it is on the board panel: tests and
        scripts build this panel for its controls alone.
        """
        self._updating = False
        self._db = operators.database()

        layout = self.content_layout
        layout.setSpacing(6)

        # Shown only when the current board is not a card. A board with no
        # template on it has no wording and no slot to edit, and the panel used
        # to open with two empty lists -- which looks broken rather than
        # explains itself.
        self.not_a_card = QLabel()
        self.not_a_card.setWordWrap(True)
        warn_surface(self.not_a_card)
        self.apply_button = QPushButton("套用「通行证」模板")
        self.apply_button.setMinimumHeight(28)
        self.apply_button.clicked.connect(self._on_apply_template)
        layout.addWidget(self.not_a_card)
        layout.addWidget(self.apply_button)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._build_operator_group(layout)
        self._build_art_group(layout)
        self._build_text_group(layout)
        self._build_cutout_group(layout)
        layout.addStretch(1)

        self.refresh()

    def _on_apply_template(self) -> None:
        """Put the 通行证 template on this board, then show what it brought.

        Done here rather than sending the user off to 板框设置: they opened this
        panel wanting to edit a card, and the reason it was empty is the one
        thing that has to change. Walking them to another panel and back is a
        step that exists only because of how the code is arranged.

        The pictures already on the board are re-framed here too -- the card
        arrived after them, and without this they would keep the full-size
        default 76.2 mm and have to be right-clicked into an opening by hand.
        """
        board = self._board
        board.outline = badge_card.badge_spec()
        board.outline_ready = True
        board.texts = badge_card.badge_texts()
        board.cutouts = badge_card.badge_cutouts()
        if self.canvas is not None:
            self.canvas.frame_illustrations_for_card()
        self.refresh()
        self.changed.emit()

    # ------------------------------------------------------------------ operator

    def _build_operator_group(self, layout: QVBoxLayout) -> None:
        group = section("干员", compact=True)
        self.operator_group = group
        inner = QVBoxLayout(group)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(4)

        self.operator_search = QLineEdit()
        self.operator_search.setPlaceholderText("输入干员名或编号…")
        self.operator_search.setClearButtonEnabled(True)
        self.operator_search.textChanged.connect(self._on_search)
        inner.addWidget(self.operator_search)

        self.operator_results = QListWidget()
        self.operator_results.setFixedHeight(150)
        self.operator_results.currentItemChanged.connect(self._on_operator_picked)
        inner.addWidget(self.operator_results)

        self.current_operator = QLabel()
        self.current_operator.setWordWrap(True)
        info_surface(self.current_operator)
        inner.addWidget(self.current_operator)

        # The number is editable on its own, not only through the operator
        # lookup: the database is a scrape, and a card for someone whose 编号 is
        # missing -- or simply different from what the wiki says -- has to be
        # buildable. It drives both the text tokens **and the barcode**, so an
        # edit re-renders the strip rather than leaving the printed code
        # disagreeing with the printed number.
        self.number_edit = QLineEdit()
        self.number_edit.setPlaceholderText("情报编号（可直接改）")
        self.number_edit.setClearButtonEnabled(True)
        self.number_edit.textEdited.connect(self._on_number_edited)
        inner.addWidget(row("编号", self.number_edit))

        clear = QPushButton("清除选择")
        clear.setFixedHeight(24)
        clear.clicked.connect(self._on_clear_operator)
        inner.addWidget(clear)

        self.insert_art_button = QPushButton("放入干员二维码与条形码")
        self.insert_art_button.setMinimumHeight(28)
        self.insert_art_button.setToolTip(
            "按当前干员生成透明底的二维码（45°）和情报编号条形码（90°），"
            "分别放进各自的定位块，并删掉模板里的占位文字与定位块"
        )
        self.insert_art_button.clicked.connect(self._on_insert_card_art)
        inner.addWidget(self.insert_art_button)

        self.art_status = QLabel()
        self.art_status.setWordWrap(True)
        muted(self.art_status)
        inner.addWidget(self.art_status)

        layout.addWidget(group)

    def _on_search(self, _text: str = "") -> None:
        self._fill_results(self.operator_search.text())

    def _fill_results(self, query: str) -> None:
        """List matching operators, remembering who each row stands for.

        What is remembered is the **page name**, not the display name: it is
        the roster's unique key, and 暮落 / 暮落(集成战略) would otherwise be
        two rows that resolve to the same person. The row's own text still
        carries the name, profession and number; parsing that back apart would
        break on a name containing the separator.
        """
        self._updating = True
        try:
            self.operator_results.clear()
            for item in self._db.search(query):
                entry = QListWidgetItem(item.display)
                entry.setData(Qt.UserRole, item.page)
                self.operator_results.addItem(entry)
        finally:
            self._updating = False

    def _on_operator_picked(self, current, _previous=None) -> None:
        if self._updating or current is None:
            return
        page = current.data(Qt.UserRole)
        if not page:
            return
        self.apply_operator(page)

    def apply_operator(self, name: str) -> bool:
        """Store the operator on the document and re-render every token.

        ``name`` may be either the page name (what the search list supplies) or
        a bare display name (what a hand-built board or an older draft has).
        The page name is tried first because a display name is not an identity:
        暮落 and 暮落(集成战略) share one, and resolving by name gave the second
        person the first person's QR symbol.

        Returns whether it was found, so a caller -- a test, or the panel
        restoring its own state -- can tell "picked" from "no such operator"
        without inspecting the document afterwards.
        """
        person = self._db.find_page(name) or self._db.find(name)
        if person is None:
            return False
        had_art = any(
            item.import_id == badge_content.ART_IMPORT_ID
            for item in self._board.placements
        )
        self._board.operator_name = person.name
        self._board.operator_number = person.number
        self._board.operator_page = person.page
        self._refresh_operator_label()
        # Every row that holds a token re-renders. Without this the choice
        # looks like it did nothing: the label still shows `{代号}` because the
        # list was built before the operator existed.
        self._refresh_all_text_labels()
        # Once the artwork is on the card it follows the operator, so picking
        # someone else does not leave the previous person's QR code sitting
        # there. Before that first insert nothing is added: choosing an operator
        # to edit their wording should not silently draw four objects.
        if had_art:
            self._insert_card_art(announce=False)
        self.changed.emit()
        return True

    def _on_number_edited(self, text: str) -> None:
        """Take a hand-typed 编号 into the document, barcode included.

        The barcode is rebuilt when the strip is already on the card, for the
        same reason picking a different operator rebuilds it: otherwise the
        printed code and the printed number on the same card would disagree,
        and only a scanner would notice.
        """
        if self._updating:
            return
        value = text.strip()
        if value == self._board.operator_number:
            return
        had_art = any(
            item.import_id == badge_content.ART_IMPORT_ID
            for item in self._board.placements
        )
        self._board.operator_number = value
        self._refresh_operator_label()
        self._refresh_all_text_labels()
        if had_art:
            self._insert_card_art(announce=False)
        self.changed.emit()

    def _on_clear_operator(self) -> None:
        if not self._board.operator_name:
            return
        self._board.operator_name = ""
        self._board.operator_number = ""
        self._board.operator_page = ""
        self._refresh_operator_label()
        self._refresh_all_text_labels()
        self.changed.emit()

    def _on_insert_card_art(self) -> None:
        """Drop the chosen operator's artwork onto the card.

        Re-inserting replaces the previous set rather than stacking a second
        copy: the user pressing the button twice wants the codes to follow the
        operator they just picked, not four codes on one card. Removal is by
        :data:`badge_content.ART_IMPORT_ID`, so pictures the user imported
        themselves are left alone.
        """
        self._insert_card_art(announce=True)

    def _insert_card_art(self, *, announce: bool) -> bool:
        board = self._board
        if not board.operator_name:
            if announce:
                self.art_status.setText("先在上面选一位干员。")
            return False
        try:
            # The board's own 编号 is what the strip encodes: it may have been
            # typed by hand, or cleared to deliberately print no barcode.
            art = badge_content.badge_art(
                board=board, number=board.operator_number
            )
        except badge_content.MissingArt as exc:
            self.art_status.setText(str(exc))
            return False

        board.placements = [
            item for item in board.placements if item.import_id != badge_content.ART_IMPORT_ID
        ]
        board.placements.extend(art.placements)
        # The lines that *name* the artwork come off the card: keeping them next
        # to the real thing prints the word "干员二维码" beside the symbol.
        board.texts = [
            item for item in board.texts if item.label not in badge_content.ART_ELEMENT_LABELS
        ]
        self._updating = True
        try:
            self.text_list.clear()
            for item in board.texts:
                self.text_list.addItem(QListWidgetItem(""))
            for index in range(len(board.texts)):
                self._refresh_text_label(index)
            self.text_list.setCurrentRow(0 if board.texts else -1)
        finally:
            self._updating = False
        self._report_art(art)
        self.changed.emit()
        return True

    def _report_art(self, art: badge_content.BadgeArt) -> None:
        """Say what landed on the card.

        The per-element table above already lists every slot; this line reports
        only what the *automatic* rule decided, so the reason a slot is empty
        ("no 情报编号", "no 职业 in the roster") is visible rather than looking
        like a failure.
        """
        notes: list[str] = []
        if art.barcode is None and self._board.operator_number.strip():
            notes.append("条形码未放")
        if art.profession is None:
            notes.append("资料库里没有职业")
        # Automatic means the operator's own faction, and 罗德岛 is left empty
        # on purpose -- the crest is already on the silkscreen above it.
        if art.faction is None and not badge_content.resolved_choice(
            self._board, "faction"
        ):
            faction = str(getattr(self._current_person(), "faction", "") or "")
            notes.append(
                "阵营 logo 未放（罗德岛的标志已在丝印层）"
                if faction == badge_content.RHODES_FACTION
                else "资料库里没有阵营 logo"
            )
        self.art_status.setText("；".join(notes) if notes else "")

    def _current_person(self):
        board = self._board
        return self._db.find_page(board.operator_page) or self._db.find(
            board.operator_name
        )

    # -------------------------------------------------------------- card art

    def _build_art_group(self, layout: QVBoxLayout) -> None:
        """One searchable row per element the card puts on automatically.

        The automatic rule is a good default and a bad cage: the roster is a
        scrape, and the user may want a different profession icon, a different
        faction, or a plain card with no crest. Each row therefore offers
        **自动 / 不放 / a chosen asset**, searchable, and re-inserting applies
        it immediately.

        The 罗德岛 crest is listed here as well even though it is not an
        artwork *slot*: the user asked to be able to remove it like the rest.
        """
        group = section("卡面素材", compact=True)
        self.art_group = group
        inner = QVBoxLayout(group)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(4)

        self.art_search = QLineEdit()
        self.art_search.setPlaceholderText("搜索素材（职业 / 阵营 / 干员名）…")
        self.art_search.setClearButtonEnabled(True)
        self.art_search.textChanged.connect(self._on_art_search)
        inner.addWidget(self.art_search)

        self.art_list = QListWidget()
        self.art_list.setFixedHeight(150)
        self.art_list.currentRowChanged.connect(self._on_art_row)
        inner.addWidget(self.art_list)

        self.art_pick = combo(["自动", "不放"])
        self.art_pick.currentIndexChanged.connect(self._on_art_pick)
        inner.addWidget(row("这一项", self.art_pick))

        self.art_apply = QPushButton("重放到卡面")
        self.art_apply.setMinimumHeight(26)
        self.art_apply.clicked.connect(self._on_reapply_art)
        inner.addWidget(self.art_apply)

        layout.addWidget(group)

    def _art_rows(self) -> list[tuple[str, str]]:
        """``(kind, label)`` for every element, in listing order."""
        return [
            ("qr", "二维码"),
            ("barcode", "条形码"),
            ("profession", "职业图标"),
            ("faction", "阵营 logo"),
            (badge_content.EMBLEM_KIND, "罗德岛标志（丝印）"),
        ]

    def _on_art_search(self, _text: str = "") -> None:
        self._fill_art_list()

    def _fill_art_list(self) -> None:
        """List the rows, narrowed by the search box.

        Matching is on the row's own label and its current choice, so typing a
        profession or faction name finds the element using it.
        """
        query = self.art_search.text().strip().casefold()
        self._updating = True
        try:
            self.art_list.clear()
            for kind, label in self._art_rows():
                choice = self._describe_choice(kind)
                text = f"{label}　·　{choice}"
                if query and query not in text.casefold():
                    continue
                entry = QListWidgetItem(text)
                entry.setData(Qt.UserRole, kind)
                self.art_list.addItem(entry)
        finally:
            self._updating = False
        if self.art_list.count() and self.art_list.currentRow() < 0:
            self.art_list.setCurrentRow(0)
        self._on_art_row(self.art_list.currentRow())

    def _art_kind_at(self, index: int) -> str:
        if not 0 <= index < self.art_list.count():
            return ""
        return self.art_list.item(index).data(Qt.UserRole) or ""

    def _describe_choice(self, kind: str) -> str:
        choice = badge_content.resolved_choice(self._board, kind)
        if choice == badge_content.CHOICE_OFF:
            return "不放"
        if choice:
            return choice
        return "自动"

    def _art_options(self, kind: str) -> list[str]:
        """The pickable values for one row: 自动, 不放, then every asset."""
        options = ["自动", "不放"]
        if kind == "profession":
            options += sorted(
                {
                    person.profession
                    for person in self._db.operators
                    if person.profession
                }
            )
        elif kind == "faction":
            options += sorted(
                {
                    person.faction
                    for person in self._db.operators
                    if person.faction
                }
            )
        return options

    def _on_art_row(self, index: int) -> None:
        kind = self._art_kind_at(index)
        enabled = bool(kind)
        self.art_pick.setEnabled(enabled)
        self.art_apply.setEnabled(enabled)
        if not enabled:
            return
        self._updating = True
        try:
            self.art_pick.clear()
            self.art_pick.addItems(self._art_options(kind))
            current = self._describe_choice(kind)
            found = self.art_pick.findText(current)
            self.art_pick.setCurrentIndex(found if found >= 0 else 0)
        finally:
            self._updating = False

    def _on_art_pick(self, _index: int = 0) -> None:
        """Record the pick, then put the whole set back on the card."""
        if self._updating:
            return
        kind = self._art_kind_at(self.art_list.currentRow())
        if not kind:
            return
        value = self.art_pick.currentText()
        if value == "自动":
            self._board.art_choices.pop(kind, None)
        elif value == "不放":
            self._board.art_choices[kind] = badge_content.CHOICE_OFF
        else:
            self._board.art_choices[kind] = value
        self._reapply_art()

    def _on_reapply_art(self) -> None:
        self._reapply_art()

    def _reapply_art(self) -> None:
        """Put the current choices on the card.

        ``badge_art`` is the single implementation of "what belongs on the
        card", so a change is one rebuild rather than a per-element edit here.
        The panel never inserts on its own: it always calls the same path the
        button does, so the two cannot drift apart.
        """
        if not self._board.operator_name:
            self.art_status.setText("先在上面选一位干员。")
            return
        # ``_insert_card_art`` reports the result itself.
        self._insert_card_art(announce=False)
        # ⚠️ **Only the current row's label is refreshed, not the whole list.**
        # Rebuilding the list re-selects the row, which re-runs ``_on_art_row``
        # and resets the combo to the value read back from the document -- so a
        # second pick in the same session was silently reverted to the first.
        # Measured: choosing 不放 then 术师 left the choice at 不放.
        self._refresh_art_row_label()

    def _refresh_art_row_label(self) -> None:
        """Rewrite the current row's text so it shows the chosen value."""
        index = self.art_list.currentRow()
        kind = self._art_kind_at(index)
        if not kind:
            return
        label = dict(self._art_rows())[kind]
        self._updating = True
        try:
            self.art_list.item(index).setText(
                f"{label}　·　{self._describe_choice(kind)}"
            )
        finally:
            self._updating = False

    def _refresh_all_text_labels(self) -> None:
        for index in range(len(self._board.texts)):
            self._refresh_text_label(index)

    def _refresh_operator_label(self) -> None:
        board = self._board
        # The box shows the document's value, and is only written when it
        # differs -- typing into it must not be undone by the next refresh.
        if self.number_edit.text() != board.operator_number:
            self.number_edit.setText(board.operator_number)
        if not board.operator_name:
            self.current_operator.setText("尚未选择干员　·　代号与编号保持为占位符")
            return
        person = self._db.find_page(board.operator_page) or self._db.find(
            board.operator_name
        )
        parts = [f"当前：{board.operator_name}"]
        if board.operator_number:
            parts.append(board.operator_number)
        if person is not None:
            if person.profession:
                parts.append(person.profession)
            if person.faction:
                parts.append(person.faction)
        self.current_operator.setText("　".join(parts))

    # -------------------------------------------------------------------- text

    def _build_text_group(self, layout: QVBoxLayout) -> None:
        group = section("文字", compact=True)
        self.text_group = group
        inner = QVBoxLayout(group)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(4)

        self.text_list = QListWidget()
        self.text_list.setFixedHeight(170)
        self.text_list.currentRowChanged.connect(self._on_text_row)
        inner.addWidget(self.text_list)

        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("这一行显示的文字")
        self.text_edit.textEdited.connect(self._on_text_changed)
        inner.addWidget(row("内容", self.text_edit))

        # Three decimals, not two: the template's own positions are quoted to
        # 0.001 mm (``-13.633``), and a two-decimal field rounds that to
        # ``-13.63`` on the first write -- so merely opening the panel and
        # nudging an unrelated field would shift every line, invisibly at
        # first and cumulatively over a session.
        self.text_size = spin(1.0, 0.2, 40.0, 0.1, " mm", decimals=3)
        self.text_size.valueChanged.connect(self._on_text_geometry)
        inner.addWidget(row("字号", self.text_size))

        self.text_x = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=3)
        self.text_y = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=3)
        self.text_x.valueChanged.connect(self._on_text_geometry)
        self.text_y.valueChanged.connect(self._on_text_geometry)
        inner.addWidget(row("锚点 X", self.text_x))
        inner.addWidget(row("锚点 Y", self.text_y))

        self.text_rotation = spin(0.0, -180.0, 180.0, 90.0, "°", decimals=1)
        self.text_rotation.valueChanged.connect(self._on_text_geometry)
        inner.addWidget(row("旋转", self.text_rotation))

        self.text_layer = combo([item.name for item in LAYERS])
        self.text_layer.currentIndexChanged.connect(self._on_text_geometry)
        inner.addWidget(row("图层", self.text_layer))

        reset = QPushButton("恢复模板原文")
        reset.setFixedHeight(26)
        reset.clicked.connect(self._on_reset_text)
        inner.addWidget(reset)

        layout.addWidget(group)

    def _on_text_row(self, index: int) -> None:
        """Show the chosen line's values without echoing them back as edits."""
        self._updating = True
        try:
            item = self._text_at(index)
            enabled = item is not None
            for widget in (
                self.text_edit,
                self.text_size,
                self.text_x,
                self.text_y,
                self.text_rotation,
                self.text_layer,
            ):
                widget.setEnabled(enabled)
            if item is None:
                self.text_edit.clear()
                return
            self.text_edit.setText(item.text)
            self.text_size.setValue(item.size_mm)
            self.text_x.setValue(item.x_mm)
            self.text_y.setValue(item.y_mm)
            self.text_rotation.setValue(item.rotation_deg)
            index_ = self.text_layer.findText(item.layer)
            if index_ >= 0:
                self.text_layer.setCurrentIndex(index_)
        finally:
            self._updating = False

    def _text_at(self, index: int):
        if 0 <= index < len(self._board.texts):
            return self._board.texts[index]
        return None

    def _on_text_changed(self, text: str) -> None:
        if self._updating:
            return
        item = self._text_at(self.text_list.currentRow())
        if item is None:
            return
        item.text = text
        self._refresh_text_label(self.text_list.currentRow())
        self.changed.emit()

    def _on_text_geometry(self) -> None:
        if self._updating:
            return
        item = self._text_at(self.text_list.currentRow())
        if item is None:
            return
        item.size_mm = self.text_size.value()
        item.x_mm = self.text_x.value()
        item.y_mm = self.text_y.value()
        item.rotation_deg = self.text_rotation.value()
        item.layer = self.text_layer.currentText()
        self.changed.emit()

    def _on_reset_text(self) -> None:
        """Put the selected line back to the template's own wording.

        Only that line, and only its wording: the position is something the
        user may have moved on purpose, and resetting it too would undo work
        they did not ask to lose.
        """
        index = self.text_list.currentRow()
        item = self._text_at(index)
        source = self._template_text(index)
        if item is None or source is None:
            return
        item.text = source.text
        self._updating = True
        try:
            self.text_edit.setText(item.text)
        finally:
            self._updating = False
        self._refresh_text_label(index)
        self.changed.emit()

    @staticmethod
    def _template_text(index: int):
        items = badge_card.BADGE_TEXTS
        return items[index] if 0 <= index < len(items) else None

    def _refresh_text_label(self, index: int) -> None:
        item = self._text_at(index)
        entry = self.text_list.item(index)
        if item is None or entry is None:
            return
        label = item.label or f"文字 {index + 1}"
        shown = item.resolved_text(
            name=self._board.operator_name, number=self._board.operator_number
        )
        entry.setText(f"{label}　·　{shown}")

    # ----------------------------------------------------------------- cutouts

    def _build_cutout_group(self, layout: QVBoxLayout) -> None:
        group = section("挖槽", compact=True)
        self.cutout_group = group
        inner = QVBoxLayout(group)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(4)

        self.cutout_list = QListWidget()
        self.cutout_list.setFixedHeight(52)
        self.cutout_list.currentRowChanged.connect(self._on_cutout_row)
        inner.addWidget(self.cutout_list)

        self.cutout_width = spin(20.0, 0.5, 60.0, 0.5, " mm", decimals=3)
        self.cutout_height = spin(3.0, 0.5, 60.0, 0.5, " mm", decimals=3)
        self.cutout_x = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=3)
        self.cutout_y = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=3)
        for widget in (
            self.cutout_width,
            self.cutout_height,
            self.cutout_x,
            self.cutout_y,
        ):
            widget.valueChanged.connect(self._on_cutout_changed)
        inner.addWidget(row("宽", self.cutout_width))
        inner.addWidget(row("高", self.cutout_height))
        inner.addWidget(row("中心 X", self.cutout_x))
        inner.addWidget(row("中心 Y", self.cutout_y))

        self.cutout_error = QLabel()
        self.cutout_error.setWordWrap(True)
        warn_surface(self.cutout_error)
        inner.addWidget(self.cutout_error)

        layout.addWidget(group)

    def _on_cutout_row(self, index: int) -> None:
        self._updating = True
        try:
            item = self._cutout_at(index)
            enabled = item is not None
            for widget in (
                self.cutout_width,
                self.cutout_height,
                self.cutout_x,
                self.cutout_y,
            ):
                widget.setEnabled(enabled)
            if item is None:
                return
            self.cutout_height.setEnabled(item.shape != "circle")
            self.cutout_width.setValue(item.width_mm)
            self.cutout_height.setValue(item.height_mm)
            self.cutout_x.setValue(item.x_mm)
            self.cutout_y.setValue(item.y_mm)
        finally:
            self._updating = False
        self._refresh_cutout_error()

    def _cutout_at(self, index: int):
        if 0 <= index < len(self._board.cutouts):
            return self._board.cutouts[index]
        return None

    def _on_cutout_changed(self) -> None:
        if self._updating:
            return
        item = self._cutout_at(self.cutout_list.currentRow())
        if item is None:
            return
        item.width_mm = self.cutout_width.value()
        item.height_mm = (
            item.width_mm if item.shape == "circle" else self.cutout_height.value()
        )
        item.x_mm = self.cutout_x.value()
        item.y_mm = self.cutout_y.value()
        self._refresh_cutout_error()
        self.changed.emit()

    def _refresh_cutout_error(self) -> None:
        item = self._cutout_at(self.cutout_list.currentRow())
        problems = item.validate() if item is not None else []
        self.cutout_error.setText("；".join(problems))
        self.cutout_error.setVisible(bool(problems))

    # ------------------------------------------------------------------ binding

    def _set_editors_visible(self, visible: bool) -> None:
        for group in (
            self.operator_group,
            self.art_group,
            self.text_group,
            self.cutout_group,
        ):
            group.setVisible(visible)

    def set_board(self, board: ArtBoard) -> None:
        """Point the panel at another board, as a tab switch requires."""
        self._board = board
        self.refresh()

    def refresh(self) -> None:
        """Rebuild both lists from the document."""
        is_card = _is_badge_board(self._board)
        self.not_a_card.setVisible(not is_card)
        self.apply_button.setVisible(not is_card)
        if not is_card:
            self.not_a_card.setText("当前板子还不是【通行证】模板卡。")
        # The editors are hidden rather than left empty on a non-card: an empty
        # list plus a disabled search box is what "broken" looks like, whereas
        # one line of explanation plus a button is what "not yet" looks like.
        self._set_editors_visible(is_card)
        self._updating = True
        try:
            self.text_list.clear()
            for item in self._board.texts:
                self.text_list.addItem(QListWidgetItem(""))
            for index in range(len(self._board.texts)):
                self._refresh_text_label(index)

            self.cutout_list.clear()
            for item in self._board.cutouts:
                self.cutout_list.addItem(QListWidgetItem(item.label))

            if self._board.texts:
                self.text_list.setCurrentRow(0)
            if self._board.cutouts:
                self.cutout_list.setCurrentRow(0)
        finally:
            self._updating = False

        self._refresh_operator_label()
        # Restore the search box to the current operator so a re-pointed panel
        # is showing the board it is actually editing. ``operator_search`` is a
        # QLineEdit: the old code called ``findText``/``setCurrentIndex``, which
        # belong to QComboBox and made every refresh raise AttributeError.
        if self._board.operator_name and not self.operator_search.text():
            self.operator_search.setText(self._board.operator_name)
        self._fill_results(self.operator_search.text())
        # The choices belong to the document, so a re-pointed panel shows the
        # new board's own set rather than the previous tab's.
        self._fill_art_list()
        if self._db:
            note_text(self.status_label, "")
        else:
            self.status_label.setText(
                "找不到干员资料库，搜索不可用；模板文字与挖槽仍可编辑。\n"
                f"查找位置：{operators.db_path()}"
            )
            warn_surface(self.status_label)

        self._on_text_row(self.text_list.currentRow())
        self._on_cutout_row(self.cutout_list.currentRow())
        self._refresh_art_status()

    def _refresh_art_status(self) -> None:
        """Say whether this board already carries the operator's artwork.

        Without this the panel opens on a finished card with a blank status
        line, which reads as "nothing was inserted" even though the codes are
        sitting on the board.
        """
        on_card = [
            item
            for item in self._board.placements
            if item.import_id == badge_content.ART_IMPORT_ID
        ]
        if not on_card:
            self.art_status.setText("")
            return
        kinds = {item.layer for item in on_card}
        self.art_status.setText(f"卡面上已有 {len(on_card)} 个素材对象　·　{len(kinds)} 个图层")


_template_content_panel: "TemplateContentPanel | None" = None


def show_template_content(main_window: QWidget, board: ArtBoard) -> TemplateContentPanel:
    """Show the template-content panel, reusing or re-targeting the cached one."""
    global _template_content_panel
    canvas = getattr(main_window, "design_canvas", None)
    _template_content_panel, created = reusable_panel(
        _template_content_panel,
        main_window,
        lambda window: TemplateContentPanel(window, board, canvas),
    )
    if not created:
        # A tab switch re-points the panel rather than opening a second one:
        # two panels editing two boards would both look like "the" template
        # panel, and the one on screen is the one the user believes they are
        # typing into.
        _template_content_panel.set_board(board)
    # Refresh the canvas reference on both paths, so a panel pointed at another
    # board re-crops that board's pictures rather than the previous one's.
    _template_content_panel.canvas = canvas
    if created:
        _template_content_panel.resize(440, 680)
        _template_content_panel.place_top_right()
    _template_content_panel.show()
    _template_content_panel.raise_()
    return _template_content_panel


class LayerPanel(QWidget):
    """The layer list: what is shown, and which layer edits go to.

    A docked pane rather than a floating panel. It is on screen whenever no
    object is selected, and being permanent is the point: which layer the next
    import lands on is a question the user has constantly, and answering it
    from a panel that has to be opened first makes it a chore.

    Follows the editor's arrangement -- a stack with the current layer marked,
    a visibility control per row -- but only lists the seven layers this tool
    actually produces. The inner layers, paste masks, assembly and 3D shells
    that EasyEDA Pro declares are not offered: a row that cannot affect this
    design would be a control that looks usable but is not.

    Colours are read from the layer table the editor itself writes, so a shape
    drawn here is the colour the user will see when it is opened there.
    """

    changed = Signal()
    """Emitted when visibility or the current layer changes."""

    def __init__(self, parent: QWidget, board: ArtBoard) -> None:
        super().__init__(parent)
        self._board = board
        self._updating = False
        self._swatches: dict[str, LayerSwatch] = {}
        self._rows: dict[str, QWidget] = {}
        self._radios: dict[str, QRadioButton] = {}
        self._checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("图层")
        themed(title, lambda: f"color: {TEXT}; font-weight: 600; font-size: 13px;")
        layout.addWidget(title)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(6)
        for text, stretch in (("当前", 0), ("", 0), ("层名", 1), ("显示", 0)):
            label = QLabel(text)
            muted(label)
            header_layout.addWidget(label, stretch)
        layout.addWidget(header)

        for item in LAYERS:
            self._add_layer_row(layout, item)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        info_surface(self.status_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.refresh()

    # ----------------------------------------------------------------- rows

    def _add_layer_row(self, layout: QVBoxLayout, item) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 1, 2, 1)
        row_layout.setSpacing(6)

        radio = QRadioButton()
        radio.setToolTip(f"把「{item.name}」设为当前层")
        radio.clicked.connect(lambda _=False, key=item.key: self._set_current(key))
        row_layout.addWidget(radio)
        self._radios[item.key] = radio

        swatch = LayerSwatch(item.color)
        self._swatches[item.key] = swatch
        row_layout.addWidget(swatch)

        name = QLabel(item.short)
        name.setToolTip(item.name)
        row_layout.addWidget(name, 1)

        visible = QCheckBox()
        visible.setChecked(self._board.layers.is_visible(item.key))
        visible.setToolTip(f"显示 / 隐藏{item.name}")
        visible.stateChanged.connect(
            lambda _state, key=item.key: self._set_visible(key)
        )
        row_layout.addWidget(visible)
        self._checks[item.key] = visible

        self._rows[item.key] = row
        layout.addWidget(row)

    def set_board(self, board: ArtBoard) -> None:
        """Point the panel at another document, as reopening a project does."""
        self._board = board
        self._updating = False
        self.refresh()

    def refresh(self) -> None:
        """Re-read the stack, without echoing changes back to it."""
        self._updating = True
        try:
            for key, radio in self._radios.items():
                radio.setChecked(self._board.layers.current == key)
                self._checks[key].setChecked(self._board.layers.is_visible(key))
                self._swatches[key].set_dimmed(
                    not self._board.layers.is_visible(key)
                )
        finally:
            self._updating = False

        current = self._board.layers.current_layer
        hidden = [
            item.name
            for item in LAYERS
            if not self._board.layers.is_visible(item.key)
        ]
        text = f"当前层：{current.name}"
        if hidden:
            text += "\n已隐藏 {} 层：{}".format(len(hidden), "、".join(hidden))
        self.status_label.setText(text)

    # ------------------------------------------------------------- handlers

    def _set_current(self, key: str) -> None:
        if self._updating:
            return
        self._board.layers.set_current(key)
        self.refresh()
        self.changed.emit()

    def _set_visible(self, key: str) -> None:
        if self._updating:
            return
        self._board.layers.set_visible(key, self._checks[key].isChecked())
        # A hidden layer cannot stay current: the next import would land
        # somewhere invisible and look like it failed.
        if not self._board.layers.is_visible(key) and self._board.layers.current == key:
            for candidate in LAYER_KEYS:
                if self._board.layers.is_visible(candidate):
                    self._board.layers.set_current(candidate)
                    break
        self.refresh()
        self.changed.emit()


class LayerSwatch(QWidget):
    """A small colour chip showing which colour a layer is drawn in."""

    SIZE = 13

    def __init__(self, color: str) -> None:
        super().__init__()
        self._color = QColor(color)
        self._dimmed = False
        self.setFixedSize(self.SIZE, self.SIZE)

    def set_dimmed(self, value: bool) -> None:
        if value != self._dimmed:
            self._dimmed = value
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colour = QColor(self._color)
        if self._dimmed:
            colour.setAlpha(70)
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(colour)
        painter.drawRoundedRect(
            QRectF(0.5, 0.5, self.SIZE - 1, self.SIZE - 1), 2, 2
        )


def show_layers(main_window: QWidget, *, focus: bool = True) -> LayerPanel | None:
    """Reveal the layer pane.

    The layer list is no longer a floating panel -- it is docked to the right
    and shows itself whenever nothing is selected -- so this exists for the
    toolbar and menu entry, which have to do *something*. It clears the
    selection, which is what brings the layer list back, and focuses the pane.
    """
    pane = getattr(main_window, "layer_panel", None)
    if pane is None:
        QMessageBox.information(
            main_window, "暂不可用", "主窗口尚未准备好设计画布。"
        )
        return None

    canvas = getattr(main_window, "design_canvas", None)
    if canvas is not None and canvas.selected is not None:
        canvas.selected = None
        canvas.selection_changed.emit(None)
        canvas.update()
    pane.refresh()
    pane.setVisible(True)
    if focus:
        pane.setFocus()
    return pane


_layer_panel: LayerPanel | None = None


class GroupManagerPanel(FloatingPanel):
    """组合管理：which elements are grouped, what they are called, and how to undo it.

    The user asked for three things by name -- choose which elements go in,
    leave a group, and manage/rename them -- and they are all the same
    question asked from different directions, so they live in one panel rather
    than three menu items with no place to see the result.

    ⚠️ **The member list is what makes this honest.** A group is invisible on
    the canvas -- the artwork looks the same whether or not it belongs to one
    -- so without a place that lists the members, "组合成功了吗" has no answer
    the user can check. Every row therefore spells out what is in it.
    """

    changed = Signal()
    """Emitted after an edit, so the canvas repaints and the change is undoable."""

    def __init__(self, parent: QWidget, canvas) -> None:
        super().__init__(parent, "组合管理")
        self.resize(460, 560)
        self._canvas = canvas
        self._updating = False
        # ⚠️ Set **before** any widget is built. ``currentRowChanged`` fires
        # while the list boxes are being populated, which reaches
        # ``_refresh_buttons`` and reads this -- and a missing attribute there
        # is an exception raised from inside a Qt signal, which aborts the
        # caller's `show` rather than surfacing as a failed operation.
        self._pending: list = []

        layout = self.content_layout
        layout.setSpacing(6)

        self.group_list = QListWidget()

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # ---------------------------------------------------- existing groups
        groups_box = section("1 · 已有组合", compact=True)
        groups_layout = QVBoxLayout(groups_box)
        groups_layout.setContentsMargins(0, 0, 0, 0)
        groups_layout.setSpacing(4)
        self.group_list = QListWidget()
        self.group_list.setMinimumHeight(120)
        self.group_list.currentRowChanged.connect(self._on_group_selected)
        groups_layout.addWidget(self.group_list)

        name_row = QWidget()
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(6)
        name_layout.addWidget(QLabel("名称"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("给这个组合起个名字")
        self.name_edit.editingFinished.connect(self._on_rename)
        name_layout.addWidget(self.name_edit, 1)
        groups_layout.addWidget(name_row)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        self.select_button = QPushButton("选中")
        self.select_button.clicked.connect(self._on_select_group)
        self.ungroup_button = QPushButton("解散组合")
        self.ungroup_button.clicked.connect(self._on_ungroup)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.ungroup_button)
        button_layout.addStretch(1)
        groups_layout.addWidget(button_row)
        layout.addWidget(groups_box)

        # ------------------------------------------------------- new group
        create_box = section("2 · 组合所选元素", compact=True)
        create_layout = QVBoxLayout(create_box)
        create_layout.setContentsMargins(0, 0, 0, 0)
        create_layout.setSpacing(4)

        self.selection_list = QListWidget()
        self.selection_list.setMinimumHeight(140)
        self.selection_list.currentRowChanged.connect(self._on_selection_row)
        create_layout.addWidget(self.selection_list)

        self.add_button = QPushButton("＋ 加入画布上选中的元素")
        self.add_button.clicked.connect(self._on_add_selection)
        create_layout.addWidget(self.add_button)

        self.new_name_edit = QLineEdit()
        self.new_name_edit.setPlaceholderText("新组合的名称（可留空）")
        create_layout.addWidget(self.new_name_edit)

        self.make_button = QPushButton("组合所选")
        self.make_button.setMinimumHeight(28)
        self.make_button.clicked.connect(self._on_make_group)
        create_layout.addWidget(self.make_button)

        self.remove_button = QPushButton("从列表移除选中项")
        self.remove_button.clicked.connect(self._on_remove_rows)
        create_layout.addWidget(self.remove_button)
        layout.addWidget(create_box)

        layout.addStretch(1)
        # The entries the panel offers are exactly the ones the canvas can
        # perform; both ask the same question, so the checkboxes here and the
        # menu's greyed-out state cannot disagree.
        self.refresh()

    # ------------------------------------------------------------- model

    @property
    def board(self):
        return self._canvas.board

    def _group_row_text(self, group) -> str:
        """One line describing a group, short enough to read without scrolling.

        ⚠️ The row used to list every member's full file name, which overran the
        panel and produced a horizontal scrollbar -- so the user saw
        「图片 pcbart-shot…（板画·白）、」 and had to scroll to find out what
        the group was. The row now names the group and *counts* what is in it;
        the members themselves are the pending list's job, one section down,
        where there is room to spell them out.
        """
        members = self.board.group_all_members(group)
        counts: dict[str, int] = {}
        for item in members:
            label = "文字" if _is_text_element(item) else "图片"
            counts[label] = counts.get(label, 0) + 1
        summary = "、".join(f"{count} 个{label}" for label, count in counts.items())
        tones = sorted(
            {
                {"white": "白", "black": "黑"}.get(item.tone, "")
                for item in members
                if not _is_text_element(item)
            }
            - {""}
        )
        if tones:
            summary += f"（板画 {''.join(tones)}）"
        return f"{self.board.group_label(group)}：{summary}"

    def refresh(self) -> None:
        """Rebuild both lists from the document and the current selection."""
        self._updating = True
        try:
            current = self._current_group()
            self.group_list.clear()
            for group in self.board.groups:
                self.group_list.addItem(QListWidgetItem(self._group_row_text(group)))
            if current is not None and current in self.board.groups:
                self.group_list.setCurrentRow(self.board.groups.index(current))
            elif self.board.groups:
                self.group_list.setCurrentRow(0)

            # The pending list keeps whatever the user assembled, minus the
            # objects that are no longer on the board at all.
            live = {id(item) for item in (*self.board.placements, *self.board.texts)}
            self._pending = [
                item for item in getattr(self, "_pending", []) if id(item) in live
            ]
            self._refresh_pending()
            self._refresh_buttons()
        finally:
            self._updating = False

    def _refresh_pending(self) -> None:
        self.selection_list.clear()
        for item in self._pending:
            group = self.board.group_of(item)
            suffix = f"　[已在「{self.board.group_label(group)}」]" if group else ""
            self.selection_list.addItem(
                QListWidgetItem(f"{_element_kind_name(item)}{suffix}")
            )

    def _refresh_buttons(self) -> None:
        row = self.group_list.currentRow()
        has_group = 0 <= row < len(self.board.groups)
        self.ungroup_button.setEnabled(has_group)
        self.select_button.setEnabled(has_group)
        self.name_edit.setEnabled(has_group)
        self.make_button.setEnabled(len(self._pending) >= 2)
        self.remove_button.setEnabled(bool(self.selection_list.selectedItems()))
        self.add_button.setEnabled(self._canvas.selected is not None)

    def _current_group(self):
        row = self.group_list.currentRow()
        if 0 <= row < len(self.board.groups):
            return self.board.groups[row]
        return None

    # ------------------------------------------------------------ actions

    def _on_group_selected(self, _row: int) -> None:
        group = self._current_group()
        self.name_edit.blockSignals(True)
        self.name_edit.setText(self.board.group_label(group) if group else "")
        self.name_edit.blockSignals(False)
        self.name_edit.setPlaceholderText(
            "给这个组合起个名字" if group else "先在上面选一个组合"
        )
        self._refresh_buttons()

    def _on_rename(self) -> None:
        if self._updating:
            return
        group = self._current_group()
        if group is None:
            return
        if not self.board.rename_group(group, self.name_edit.text()):
            return
        self._apply(f"已重命名为「{self.board.group_label(group)}」")

    def _on_select_group(self) -> None:
        group = self._current_group()
        if group is None or not self._canvas.select_group(group):
            return
        self._set_status(f"已选中「{self.board.group_label(group)}」的全部元素")

    def _on_ungroup(self) -> None:
        group = self._current_group()
        if group is None:
            return
        label = self.board.group_label(group)
        if not self.board.dissolve_group(group):
            return
        self._apply(f"已解散「{label}」，元素都还在")

    def _on_add_selection(self) -> None:
        """Put whatever is selected on the canvas into the pending list.

        The user asked to choose the elements freely, and this is that choice:
        select on the canvas -- Ctrl-click for several, Shift-click for a range
        -- then bring the set in here. The panel deliberately does not try to
        guess the set from its own list, because the canvas is where the user
        can see what they are picking.
        """
        chosen = self._canvas.selection_set()
        if not chosen:
            self._set_status("画布上还没有选中任何元素", warn=True)
            return
        pending = list(getattr(self, "_pending", []))
        added = 0
        for item in chosen:
            if item not in pending:
                pending.append(item)
                added += 1
        self._pending = pending
        self._refresh_pending()
        self._refresh_buttons()
        self._set_status(f"已加入 {added} 个元素，共 {len(pending)} 个待组合")

    def _on_selection_row(self, _row: int) -> None:
        self._refresh_buttons()

    def _on_remove_rows(self) -> None:
        selected = {
            self.selection_list.row(item) for item in self.selection_list.selectedItems()
        }
        self._pending = [
            item
            for index, item in enumerate(getattr(self, "_pending", []))
            if index not in selected
        ]
        self._refresh_pending()
        self._refresh_buttons()

    def _on_make_group(self) -> None:
        pending = list(getattr(self, "_pending", []))
        if len(pending) < 2:
            self._set_status("至少需要两个元素才能组合", warn=True)
            return
        group = self.board.make_group(pending, name=self.new_name_edit.text())
        if group is None:
            self._set_status("这些元素无法组成一个组合", warn=True)
            return
        self._pending = []
        self.new_name_edit.clear()
        self._apply(f"已组合 {len(self.board.group_all_members(group))} 个元素")

    def _apply(self, message: str) -> None:
        """Push the change out: repaint, mark the document dirty, refresh."""
        self.refresh()
        self._set_status(message)
        self._canvas.document_changed.emit()
        # One undo step for the whole edit, recorded through the canvas so it
        # lands in the same history as every other document change.
        self._canvas._record()
        self._canvas.update()
        self.changed.emit()

    def _set_status(self, message: str, *, warn: bool = False) -> None:
        self.status.setText(message)
        if warn:
            warn_surface(self.status, size=11)
        else:
            muted(self.status)


def _is_text_element(item) -> bool:
    from pcbart.artboard import TextElement

    return isinstance(item, TextElement)


def _element_kind_name(item) -> str:
    """A short human label for one element, for the manager's lists."""
    from pcbart.artboard import ImagePlacement

    if isinstance(item, ImagePlacement):
        import os

        name = os.path.basename(item.path) or "图片"
        # Elided rather than printed whole: a long source file name is the
        # normal case -- 「立绘 缄默德克萨斯 幽兰秘辛.png」 -- and letting it
        # through pushes the rest of the row out of the panel.
        if len(name) > 22:
            name = name[:11] + "…" + name[-9:]
        tone = {"white": "·白", "black": "·黑"}.get(item.tone, "")
        return f"图片 {name}{tone}"
    if _is_text_element(item):
        text = (item.text or "").strip() or "（空文字）"
        if len(text) > 14:
            text = text[:14] + "…"
        return f"文字 {text}"
    return str(getattr(item, "label", "元素"))


_group_manager_panel = None


def show_group_manager(main_window: QWidget):
    """Open 组合管理 for ``main_window``'s canvas."""
    global _group_manager_panel

    canvas = main_window.design_canvas
    panel, created = reusable_panel(
        _group_manager_panel, main_window, lambda parent: GroupManagerPanel(parent, canvas)
    )
    _group_manager_panel = panel
    if created:
        panel.place_top_right()
    else:
        # Re-pointed on every open: a tab switch replaces the document, and a
        # panel still bound to the previous board would edit the wrong one.
        panel._canvas = canvas
    panel.refresh()
    panel.show()
    panel.raise_()
    return panel



class ImagePropertyPanel(QWidget):
    """Step 2: properties of the selected image.

    Replaces the statistics list while an image is selected, mirroring how
    PCB editors swap the inspector for the selected object's properties.
    """

    changed = Signal()
    closed = Signal()
    fit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placement = None
        self._layer_stack = None
        self._updating = False
        self._run_key: str | None = None
        """Which run of edits the last change belonged to; see
        :meth:`edit_run_key`."""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title = QLabel("图片属性")
        themed(title, lambda: f"color: {TEXT}; font-weight: 600; font-size: 13px;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        close = QPushButton("×")
        close.setFixedSize(24, 24)
        close.setToolTip("取消选择")
        close.clicked.connect(self.closed.emit)
        header_layout.addWidget(close)
        layout.addWidget(header)

        self.name_label = QLabel("未选择")
        self.name_label.setWordWrap(True)
        muted(self.name_label)
        layout.addWidget(self.name_label)

        position = section("位置", compact=True)
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(0, 0, 0, 0)
        position_layout.setSpacing(3)
        self.x_spin = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=2)
        self.y_spin = spin(0.0, -1000.0, 1000.0, 0.5, " mm", decimals=2)
        self.x_spin.valueChanged.connect(self._on_position)
        self.y_spin.valueChanged.connect(self._on_position)
        position_layout.addWidget(row("中心 X", self.x_spin))
        position_layout.addWidget(row("中心 Y", self.y_spin))
        center_button = QPushButton("居中")
        center_button.setFixedHeight(24)
        center_button.clicked.connect(self._on_centre)
        position_layout.addWidget(center_button)
        layout.addWidget(position)

        size = section("尺寸", compact=True)
        size_layout = QVBoxLayout(size)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(3)
        self.width_spin = spin(40.0, 0.1, 1000.0, 0.5, " mm", decimals=2)
        self.height_spin = spin(40.0, 0.1, 1000.0, 0.5, " mm", decimals=2)
        self.width_spin.valueChanged.connect(self._on_size)
        self.height_spin.valueChanged.connect(self._on_size)
        size_layout.addWidget(row("宽度", self.width_spin))
        size_layout.addWidget(row("高度", self.height_spin))
        self.aspect_check = QCheckBox("锁定长宽比")
        self.aspect_check.setChecked(True)
        self.aspect_check.stateChanged.connect(self._on_aspect)
        size_layout.addWidget(self.aspect_check)
        layout.addWidget(size)

        transform = section("变换", compact=True)
        transform_layout = QVBoxLayout(transform)
        transform_layout.setContentsMargins(0, 0, 0, 0)
        transform_layout.setSpacing(3)
        self.rotation_spin = spin(0.0, -180.0, 180.0, 1.0, "°", decimals=1)
        self.rotation_spin.valueChanged.connect(self._on_rotation)
        transform_layout.addWidget(row("旋转", self.rotation_spin))

        flip_row = QWidget()
        flip_layout = QHBoxLayout(flip_row)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        flip_layout.setSpacing(4)
        self.flip_h = QCheckBox("水平翻转")
        self.flip_v = QCheckBox("垂直翻转")
        self.flip_h.stateChanged.connect(self._on_flip)
        self.flip_v.stateChanged.connect(self._on_flip)
        flip_layout.addWidget(self.flip_h)
        flip_layout.addWidget(self.flip_v)
        flip_layout.addStretch(1)
        transform_layout.addWidget(flip_row)
        layout.addWidget(transform)

        layer_group = section("图层", compact=True)
        layer_layout = QVBoxLayout(layer_group)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.setSpacing(3)
        # Built from the same layer table the layer panel uses, so the two
        # cannot offer different sets of layers.
        self.layer_combo = combo([item.name for item in LAYERS])
        self.layer_combo.currentIndexChanged.connect(self._on_layer)
        layer_layout.addWidget(row("放置层", self.layer_combo))
        self.opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_layer)
        layer_layout.addWidget(row("不透明度", self.opacity_slider))
        self.layer_note = QLabel()
        self.layer_note.setWordWrap(True)
        muted(self.layer_note)
        layer_layout.addWidget(self.layer_note)
        layout.addWidget(layer_group)

        fit_button = QPushButton("按板框缩放")
        fit_button.setMinimumHeight(28)
        fit_button.clicked.connect(self._on_fit)
        layout.addWidget(fit_button)

        layout.addStretch(1)

    # ------------------------------------------------------------- binding

    def set_placement(self, placement) -> None:
        """Show the given placement, without echoing changes back."""
        self._placement = placement
        self._updating = True
        try:
            self.name_label.setText(
                Path(placement.path).name if placement.path else "图像"
            )
            self.x_spin.setValue(placement.x_mm)
            self.y_spin.setValue(placement.y_mm)
            self.width_spin.setValue(placement.width_mm)
            self.height_spin.setValue(placement.height_mm)
            self.rotation_spin.setValue(placement.rotation_deg)
            self.aspect_check.setChecked(placement.keep_aspect)
            self.flip_h.setChecked(placement.flip_horizontal)
            self.flip_v.setChecked(placement.flip_vertical)
            self.opacity_slider.setValue(int(round(placement.opacity * 100)))
            index = self.layer_combo.findText(placement.layer)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
        finally:
            self._updating = False
        self._refresh_layer_note()

    def set_layer_stack(self, stack) -> None:
        """Read the document's layer state, so the note can flag a hidden layer."""
        self._layer_stack = stack
        self._refresh_layer_note()

    def refresh_layers(self) -> None:
        """Re-read the layer state, as the layer panel changing it requires."""
        if self._placement is not None:
            self._refresh_layer_note()

    def clear_placement(self) -> None:
        self._placement = None
        self.name_label.setText("未选择")
        self.layer_note.setText("")

    def _emit(self) -> None:
        if not self._updating and self._placement is not None:
            self._run_key = self._field_key()
            self.changed.emit()

    def _field_key(self) -> str:
        """Which field the current edit came from, for undo coalescing.

        One key per control, so dragging the X spinbox merges into a single
        undo step while a following edit to the width starts a new one. The
        placement is part of the key too: the same field on a different object
        must not continue the previous object's run.

        The control is identified by its attribute name on this panel rather
        than by its ``objectName``, which Qt leaves empty unless someone sets
        it -- every field would otherwise share one key and a position edit
        would merge with a size edit.
        """
        sender = self.sender()
        field = "field"
        for name, value in vars(self).items():
            if value is sender:
                field = name
                break
        return f"{id(self._placement)}:{field}"

    def edit_run_key(self) -> str | None:
        """The run key for the edit just reported, or ``None`` for a one-off.

        Read by the window when :attr:`changed` fires; see
        :meth:`pcbart.undo.UndoStack.record`.
        """
        return self._run_key

    # ------------------------------------------------------------ handlers
    #
    # Every handler bails out while the panel is being refilled. ``set_value``
    # emits the same signals a user does, so without this guard showing a
    # placement writes that placement back to itself -- one field at a time,
    # using whatever the *other* fields still held. Setting X first clobbered Y
    # with the previous placement's value, which is how a moved object jumped
    # back to y = 0 the moment its layer changed.

    def _on_position(self) -> None:
        if self._updating or self._placement is None:
            return
        self._placement.x_mm = self.x_spin.value()
        self._placement.y_mm = self.y_spin.value()
        self._emit()

    def _on_size(self) -> None:
        if self._updating or self._placement is None:
            return
        width = self.width_spin.value()
        height = self.height_spin.value()
        if self.aspect_check.isChecked() and self._placement.width_mm > 0:
            ratio = self._placement.height_mm / self._placement.width_mm
            self._updating = True
            try:
                if self.sender() is self.width_spin:
                    height = width * ratio
                    self.height_spin.setValue(height)
                else:
                    width = height / ratio if ratio else height
                    self.width_spin.setValue(width)
            finally:
                self._updating = False
        self._placement.width_mm = max(0.1, width)
        self._placement.height_mm = max(0.1, height)
        self._emit()

    def _on_aspect(self) -> None:
        if self._updating or self._placement is None:
            return
        self._placement.keep_aspect = self.aspect_check.isChecked()
        self._emit()

    def _on_rotation(self) -> None:
        if self._updating or self._placement is None:
            return
        self._placement.rotation_deg = self.rotation_spin.value()
        # The angle is part of the painting's transform, so it travels with the
        # 翻转 flags -- see :meth:`_carry_transform_to_the_pair`.
        self._carry_transform_to_the_pair()
        self._emit()

    def _on_flip(self) -> None:
        if self._updating or self._placement is None:
            return
        self._placement.flip_horizontal = self.flip_h.isChecked()
        self._placement.flip_vertical = self.flip_v.isChecked()

        # A board painting is two placements -- the white tone and the black --
        # and the exporter writes them as one geometry derived from the white.
        # Turning only the selected one would leave the other where it was, so
        # the tones would no longer line up and the exported board would tear
        # along their seam. Rotating has the same problem, which is why the
        # angle travels with them too.
        self._carry_transform_to_the_pair()
        self._emit()

    def _carry_transform_to_the_pair(self) -> None:
        """Apply this placement's 翻转 and 旋转 to its painting's other tone."""
        placement = self._placement
        if placement is None or placement.tone not in ("white", "black"):
            return
        if not placement.import_id:
            return

        # ``window()``, not ``parentWidget()``: this pane is wrapped in a
        # QScrollArea, so its immediate parent is the viewport rather than the
        # main window, and looking one level up finds nothing.
        canvas = getattr(self.window(), "design_canvas", None)
        if canvas is None:
            return
        for other in canvas.board.placements:
            if other is placement or other.tone not in ("white", "black"):
                continue
            if other.import_id != placement.import_id:
                continue
            other.flip_horizontal = placement.flip_horizontal
            other.flip_vertical = placement.flip_vertical
            other.rotation_deg = placement.rotation_deg

    def _on_layer(self) -> None:
        if self._updating or self._placement is None:
            return
        self._placement.layer = self.layer_combo.currentText()
        self._placement.opacity = self.opacity_slider.value() / 100.0
        self._refresh_layer_note()
        self._emit()

    def _refresh_layer_note(self) -> None:
        """Say which colour the current layer paints, and whether it is shown."""
        key = self._placement.layer_key if self._placement is not None else None
        if key is None:
            self.layer_note.setText("")
            return
        item = layer(key)
        text = f"颜色 {item.color}"
        if self._layer_stack is not None and not self._layer_stack.is_visible(key):
            text += "　·　⚠ 该层当前已隐藏"
        self.layer_note.setText(text)

    def _on_centre(self) -> None:
        if self._placement is None:
            return
        self._placement.x_mm = 0.0
        self._placement.y_mm = 0.0
        self.set_placement(self._placement)
        self._emit()

    def _on_fit(self) -> None:
        if self._placement is not None:
            self.fit_requested.emit()


class ProjectTabBar(QTabBar):
    """The row of open boards, between the toolbar and the work area.

    A tab bar rather than anything more elaborate, because "several documents
    at once, click one to switch" already has a shape every user knows -- from
    a browser, an editor, an EDA tool. The navigation the bar needs is the
    part QTabBar does not do on its own: the next and previous shortcuts, and
    a ``+`` that is always visible rather than hidden at the end of the row.

    The bar owns no documents. It reports what was clicked and draws what it
    is told, which is what keeps the actual switching in one place.
    """

    new_requested = Signal()
    close_requested = Signal(int)
    close_others_requested = Signal(int)
    current_changed = Signal(int)
    moved = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectTabBar")
        self.setDocumentMode(True)
        self.setMovable(True)
        # ``False`` even though the tabs *are* closable: this flag is what asks
        # the *style* for a close button, and a stylesheet on the tab bar
        # supplies none -- the tabs come out with no visible way to close them.
        # The ✕ is put in the slot by hand instead (see ``make_tab``).
        self.setTabsClosable(False)
        self.setExpanding(False)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.ElideRight)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        themed(
            self,
            lambda: f"""
                            QTabBar#ProjectTabBar {{
                                background: {WINDOW_BG};
                                border-bottom: 1px solid {BORDER};
                            }}
                            QTabBar#ProjectTabBar::tab {{
                                background: {PANEL_BG};
                                color: {MUTED};
                                border: 1px solid {BORDER};
                                border-bottom: none;
                                border-top-left-radius: 5px;
                                border-top-right-radius: 5px;
                                padding: 5px 8px;
                                margin: 3px 2px 0 0;
                                min-width: 86px;
                                max-width: 210px;
                            }}
                            QTabBar#ProjectTabBar::tab:selected {{
                                background: {PANEL_BG};
                                color: {TEXT};
                                font-weight: 600;
                                border-color: {ACCENT};
                                border-bottom: 2px solid {PANEL_BG};
                            }}
                            QTabBar#ProjectTabBar::tab:hover {{ color: {TEXT}; }}
                            QTabBar#ProjectTabBar::close-button {{
                                subcontrol-position: right;
                                margin: 2px;
                                padding: 1px;
                                border-radius: 3px;
                            }}
                            QTabBar#ProjectTabBar::close-button:hover {{
                                background: rgba(185, 28, 28, 0.14);
                            }}
            """,
        )
        self.tabBarClicked.connect(self._on_clicked)
        self.tabMoved.connect(self.moved.emit)
        self.currentChanged.connect(self.current_changed.emit)
        self.customContextMenuRequested.connect(self._on_context_menu)

        #: Which side the hand-made ✕ sits on. One button is reused per tab:
        #: building a new one on every refresh leaked three widgets a tick,
        #: which over a session grew the process by gigabytes.
        self._close_side = QTabBar.ButtonPosition.RightSide

        #: What the row currently shows, so a refresh that changes nothing
        #: does not touch a single widget. The dirty flag is re-derived on a
        #: 500 ms timer, and rebuilding the row every tick would throw away
        #: hover state and make the tab under the cursor flicker.
        self._shown: list[tuple[str, bool]] = []

    def make_tab(self, title: str, index: int, *, dirty: bool = False) -> None:
        """(Re)build the tab at ``index`` from the title and its dirty state.

        The dirty mark is a dot rather than a word: it has to survive the tab
        being narrowed to ``未命名 12…`` and still read at a glance, because
        the whole point of showing it is that the user is about to switch away
        from something unwritten.
        """
        label = f"● {title}" if dirty else title
        while self.count() <= index:
            super().insertTab(self.count(), "")
        super().setTabText(index, label)
        super().setTabToolTip(
            index,
            f"{title}\n有未保存的修改" if dirty else f"{title}\n单击切换",
        )
        if self.tabButton(index, self._close_side) is None:
            self.setTabButton(
                index, self._close_side, self._new_close_button()
            )

    def _new_close_button(self) -> QToolButton:
        """One tab's ✕.

        The click is resolved to a *tab index at click time* rather than
        captured when the button was made: tabs are reordered and removed, so
        an index remembered at construction would sooner or later point at
        somebody else's board.
        """
        button = QToolButton(self)
        button.setObjectName("TabClose")
        button.setText("✕")
        button.setToolTip("关闭这个板子")
        button.setCursor(Qt.PointingHandCursor)
        button.setAutoRaise(True)
        button.setFixedSize(16, 16)
        button.setFocusPolicy(Qt.NoFocus)
        themed(
            button,
            lambda: f"""
                            QToolButton#TabClose {{
                                background: transparent; border: none; color: {MUTED};
                                font-size: 10px; padding: 0;
                            }}
                            QToolButton#TabClose:hover {{
                                color: {DANGER}; background: rgba(185, 28, 28, 0.12);
                                border-radius: 3px;
                            }}
            """,
        )
        button.clicked.connect(lambda _=False, source=button: self._close_tab(source))
        return button

    def _close_tab(self, button: QToolButton) -> None:
        """Which tab is ``button`` on? Asked now, not remembered."""
        for index in range(self.count()):
            if self.tabButton(index, self._close_side) is button:
                self.close_requested.emit(index)
                return

    def set_titles(self, projects) -> None:
        """Show exactly ``projects``, in order, keeping the current index."""
        wanted_state = [(item.title, item.is_dirty()) for item in projects]
        if wanted_state == self._shown and self.count() == len(wanted_state):
            return
        self._shown = wanted_state

        blocked = self.blockSignals(True)
        try:
            wanted = len(projects)
            while self.count() > wanted:
                super().removeTab(self.count() - 1)
            for index, (title, dirty) in enumerate(wanted_state):
                self.make_tab(title, index, dirty=dirty)
            if wanted:
                self.setCurrentIndex(min(self.currentIndex(), wanted - 1))
        finally:
            self.blockSignals(blocked)

    def _on_clicked(self, index: int) -> None:
        """Middle-click closes, the way it does in every tabbed application.

        Kept here rather than in the window because it is a fact about tab
        bars, not about boards -- and a middle click that lands on the bar
        itself (index ``-1``) must do nothing rather than close whatever
        happens to be selected.
        """
        if index < 0:
            return
        if QApplication.mouseButtons() == Qt.MiddleButton:
            self.close_requested.emit(index)

    def _on_context_menu(self, point: QPoint) -> None:
        index = self.tabAt(point)
        if index < 0:
            menu = QMenu(self)
            new_action = menu.addAction("新建板子")
            chosen = menu.exec(self.mapToGlobal(point))
            if chosen is new_action:
                self.new_requested.emit()
            return
        menu = QMenu(self)
        close_action = menu.addAction("关闭这个板子")
        others_action = menu.addAction("关闭其他板子")
        menu.addSeparator()
        new_action = menu.addAction("新建板子")
        chosen = menu.exec(self.mapToGlobal(point))
        if chosen is close_action:
            self.close_requested.emit(index)
        elif chosen is others_action:
            self.close_others_requested.emit(index)
        elif chosen is new_action:
            self.new_requested.emit()


class MainWindow(QMainWindow):
    """The application window, which is the one place a design can be lost.

    Closing is the moment the draft question matters, and intercepting it
    requires overriding ``closeEvent`` -- which means a subclass rather than a
    plain ``QMainWindow`` built in a function. The tab shortcuts live here too:
    they are window-wide keys, and a keyboard handler on a child widget only
    fires while that widget has focus.
    """

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if confirm_discard_all(self, action="退出"):
            event.accept()
            return
        # Refusing the event leaves the window exactly as it was. Done rather
        # than reopening a new one, because a reopened window would lose the
        # in-flight edit state the user just chose to keep.
        event.ignore()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Ctrl+Tab / Ctrl+Shift+Tab walk the open boards."""
        modifiers = event.modifiers()
        if event.key() == Qt.Key_Tab and modifiers & Qt.ControlModifier:
            step = -1 if modifiers & Qt.ShiftModifier else 1
            tabs = getattr(self, "project_tabs", None)
            if tabs is not None and tabs.count():
                tabs.setCurrentIndex((tabs.currentIndex() + step) % tabs.count())
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------- the open boards
    #
    # One board is on screen at a time; the rest wait in the workspace. These
    # two properties are the seam between the two: everything that used to ask
    # the window "which draft is this" now asks the *active project*, so the
    # panels and the export paths did not each have to learn about tabs.

    @property
    def workspace(self) -> Workspace:
        """Every open board. Created on first use from the canvas's board.

        Lazily rather than in ``build_window`` so a window built by a test
        still gets a workspace the moment anything asks for one, without that
        test having to know the tab bar exists.
        """
        found = getattr(self, "_workspace", None)
        if found is None:
            canvas = getattr(self, "design_canvas", None)
            board = None if canvas is None else canvas.board
            found = Workspace(board)
            if canvas is not None:
                # The canvas started recording before this workspace existed.
                # Handing its stack to the opening project keeps one history
                # per board instead of one the tab switch would abandon.
                found.active.adopt_history(canvas.history)
            self._workspace = found
        return found

    @property
    def active_project(self) -> Project:
        return self.workspace.active

    @property
    def _draft_path(self) -> str | None:
        return self.active_project.draft_path

    @_draft_path.setter
    def _draft_path(self, value: str | None) -> None:
        self.active_project.draft_path = None if value is None else str(value)

    @property
    def _draft_saved_fingerprint(self) -> str | None:
        return self.active_project.saved_fingerprint

    @_draft_saved_fingerprint.setter
    def _draft_saved_fingerprint(self, value: str | None) -> None:
        self.active_project.saved_fingerprint = value


def _relative_time(stamp: float) -> str:
    """A saved-time as people say it: 刚刚 / 3 分钟前 / 昨天 14:30 / 09-20 14:30.

    Absolute times are what a list of drafts actually needs for anything older
    than a day -- "2 天前" leaves the user counting backwards to work out which
    one they were editing.
    """
    import datetime

    when = datetime.datetime.fromtimestamp(stamp)
    now = datetime.datetime.now()
    seconds = (now - when).total_seconds()
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if when.date() == now.date():
        return when.strftime("%H:%M")
    if (now.date() - when.date()).days == 1:
        return f"昨天 {when.strftime('%H:%M')}"
    if when.year == now.year:
        return when.strftime("%m-%d %H:%M")
    return when.strftime("%Y-%m-%d %H:%M")


class RenameEdit(QLineEdit):
    """The rename box that appears on a draft row.

    Escape has to put the old name back rather than commit a half-typed one:
    this widget is one keystroke away from the row that opens the user's work,
    and a stray Escape followed by a click elsewhere must not silently rename
    a draft to whatever was in the box at the time.
    """

    editing_cancelled = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._original = text

    def set_original(self, text: str) -> None:
        self._original = text

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key_Escape:
            self.setText(self._original)
            self.editing_cancelled.emit()
            return
        super().keyPressEvent(event)


class DraftRow(QPushButton):
    """One draft in the sidebar list: its name, when it was saved, and a way
    to rename it in place.

    A button rather than a plain list item because the whole row is the
    target -- opening a draft is the main thing a row does, and a click
    anywhere on it should do that. The rename control is a separate button for
    exactly that reason: a permanently editable name field would swallow the
    click that opens the draft, which is the one interaction this list exists
    for. The pencil swaps the label for a real text box, so renaming is still
    typing in a box rather than answering a dialog.
    """

    rename_committed = Signal(str)
    editing_started = Signal()
    editing_finished = Signal()

    def __init__(
        self,
        name: str,
        subtitle: str,
        current: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DraftRow")
        self.setMinimumHeight(42)
        self.setText("")  # the name and time are drawn by child widgets

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(0)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self.name_label = QLabel(name)
        themed(
            self.name_label,
            lambda: f"color: {TEXT}; font-weight: {'600' if current else '400'};"
            "background: transparent; border: none;",
        )
        self.name_label.setWordWrap(False)
        self.name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top_layout.addWidget(self.name_label, 1)

        self.name_edit = RenameEdit(name)
        self.name_edit.setObjectName("DraftRenameEdit")
        self.name_edit.setToolTip("输入新名字后按回车；按 Esc 取消")
        themed(
            self.name_edit,
            lambda: f"""
                            QLineEdit#DraftRenameEdit {{
                                background: {PANEL_BG};
                                color: {TEXT};
                                border: 1px solid {ACCENT};
                                border-radius: 3px;
                                padding: 1px 3px;
                                font-weight: 600;
                            }}
            """,
        )
        self.name_edit.setVisible(False)
        top_layout.addWidget(self.name_edit, 1)

        self.rename_button = QToolButton()
        self.rename_button.setObjectName("DraftRenameButton")
        self.rename_button.setText("✎")
        self.rename_button.setToolTip("重命名这份草稿")
        self.rename_button.setCursor(Qt.PointingHandCursor)
        self.rename_button.setFixedSize(20, 18)
        self.rename_button.setFocusPolicy(Qt.NoFocus)
        themed(
            self.rename_button,
            lambda: f"""
                            QToolButton#DraftRenameButton {{
                                background: transparent; border: none; color: {MUTED};
                                font-size: 12px;
                            }}
                            QToolButton#DraftRenameButton:hover {{
                                color: {ACCENT}; background: rgba(37, 99, 235, 0.10);
                                border-radius: 3px;
                            }}
            """,
        )
        self.rename_button.clicked.connect(self.edit_name)
        top_layout.addWidget(self.rename_button, 0, Qt.AlignTop)
        layout.addWidget(top)

        self.subtitle_label = QLabel(subtitle)
        themed(
            self.subtitle_label,
            lambda: f"color: {SUCCESS if current else MUTED}; font-size: 10px;"
            "background: transparent; border: none;",
        )
        layout.addWidget(self.subtitle_label)

        themed(
            self,
            lambda: f"""
                            QPushButton#DraftRow {{
                                background: {"#eef4ff" if current else PANEL_BG};
                                border: 1px solid {"#bcd3f5" if current else BORDER};
                                border-radius: 4px;
                                text-align: left;
                            }}
                            QPushButton#DraftRow:hover {{ border-color: {ACCENT}; }}
            """,
        )

        self.name_edit.returnPressed.connect(self._commit)
        self.name_edit.editingFinished.connect(self._commit)
        self.name_edit.editing_cancelled.connect(self._cancel)

    # ------------------------------------------------------------- renaming

    def edit_name(self) -> None:
        """Swap the label for an editable box, with the name selected."""
        self.name_label.setVisible(False)
        self.rename_button.setVisible(False)
        self.name_edit.set_original(self.name_label.text())
        self.name_edit.setText(self.name_label.text())
        self.name_edit.setVisible(True)
        self.name_edit.setFocus(Qt.MouseFocusReason)
        self.name_edit.selectAll()
        self.editing_started.emit()

    def _commit(self) -> None:
        if not self.name_edit.isVisible():
            return
        name = self.name_edit.text().strip()
        original = self.name_label.text()
        self._finish()
        if name and name != original:
            self.rename_committed.emit(name)

    def _cancel(self) -> None:
        self._finish()

    def _finish(self) -> None:
        self.name_edit.setVisible(False)
        self.name_label.setVisible(True)
        self.rename_button.setVisible(True)
        self.editing_finished.emit()


class DraftPanel(QWidget):
    """The left sidebar: the drafts kept inside the application.

    Two jobs, and they belong together. The list is how yesterday's work gets
    reopened -- one click, newest first, no remembering where it was saved --
    and the status line is how the user knows whether what is on the canvas has
    been written down at all. Putting them in one column means "continue" and
    "don't lose this" are answered in the same place.

    The dirty flag is re-derived from the document on a timer rather than
    pushed by every edit path. There are dozens of those -- dragging a picture,
    every slider in two floating panels, layer visibility -- and a single
    missed one would mean the exit prompt silently failed to appear. Comparing
    a fingerprint cannot miss an edit, because it does not rely on anyone
    announcing one.
    """

    save_requested = Signal()
    save_as_requested = Signal()
    open_requested = Signal()
    open_path_requested = Signal(str)
    rename_requested = Signal(str, str)
    delete_requested = Signal(str)
    new_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._path: str | None = None
        self._saved_fingerprint: str | None = None
        self._dirty = False
        self._drafts: list[DraftInfo] = []
        self._editing_path: str | None = None
        #: What the list currently shows, so an unchanged refresh is a no-op.
        self._shown_list: tuple | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("草稿")
        themed(title, lambda: f"color: {TEXT}; font-weight: 600; font-size: 13px;")
        layout.addWidget(title)

        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        self.state_label.setMinimumHeight(44)
        layout.addWidget(self.state_label)

        self.save_button = QPushButton("保存草稿")
        self.save_button.setMinimumHeight(28)
        self.save_button.setToolTip("存进下面的草稿列表")
        self.save_button.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.save_button)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        self.new_button = QPushButton("新建")
        self.new_button.setMinimumHeight(24)
        self.new_button.setToolTip("在新标签里开一块空板，不影响当前设计")
        self.new_button.clicked.connect(self.new_requested.emit)
        row_layout.addWidget(self.new_button)

        # 另存为 rather than 重命名: renaming a draft is now something you do
        # on its row, and this button is for keeping the current one and
        # starting a copy under another name. Two different jobs that used to
        # share a word.
        self.save_as_button = QPushButton("另存为…")
        self.save_as_button.setMinimumHeight(24)
        self.save_as_button.setToolTip("另存成一份新草稿")
        self.save_as_button.clicked.connect(self.save_as_requested.emit)
        row_layout.addWidget(self.save_as_button)
        layout.addWidget(row)

        list_title = QLabel("已保存的草稿")
        list_title.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; margin-top: 4px;"
        )
        layout.addWidget(list_title)

        # The list scrolls on its own so a long history cannot push the save
        # button off the bottom of the column.
        self._list_area = QScrollArea()
        self._list_area.setWidgetResizable(True)
        self._list_area.setFrameShape(QFrame.NoFrame)
        self._list_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_area.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_area.setWidget(self._list_host)
        layout.addWidget(self._list_area, 1)

        self.empty_label = QLabel("还没有草稿")
        self.empty_label.setWordWrap(True)
        muted(self.empty_label)
        layout.addWidget(self.empty_label)

        self.refresh()

    # ---------------------------------------------------------------- state

    def set_document_state(
        self,
        path: str | None,
        saved_fingerprint: str | None,
        dirty: bool,
        drafts: list[DraftInfo] | None = None,
    ) -> None:
        self._path = path
        self._saved_fingerprint = saved_fingerprint
        self._dirty = dirty
        if drafts is not None:
            self._drafts = drafts
        self.refresh()

    def mark_saved(self, path: str, saved_fingerprint: str) -> None:
        self._path = path
        self._saved_fingerprint = saved_fingerprint
        self._dirty = False
        self.refresh()

    def is_dirty(self) -> bool:
        return self._dirty

    def current_path(self) -> str | None:
        return self._path

    def refresh(self) -> None:
        """Redraw from the state the window last pushed in."""
        self._refresh_state()
        self._refresh_list()

    def _refresh_state(self) -> None:
        if self._dirty:
            text = "● 有未保存的修改"
            text += (
                "\n点【保存草稿】写回列表里的那一份"
                if self._path
                else "\n点【保存草稿】存进下面的列表"
            )
            warn_surface(self.state_label, size=11, padding="6px 8px")
        else:
            if self._path:
                text = f"✓ 已保存：{Path(self._path).stem}"
            else:
                text = "新文档，暂无修改"
            themed(
                self.state_label,
                lambda: f"background: {PANEL_BG};"
                f" color: {SUCCESS if self._path else MUTED};"
                f" border: 1px solid {BORDER}; border-radius: 4px;"
                f" padding: 6px 8px; font-size: 11px;",
            )
        self.state_label.setText(text)

    def _begin_row_edit(self, path: str) -> None:
        """Remember that a row is being renamed.

        The list is rebuilt from the store on a timer, and rebuilding it while
        the user is typing would throw away the box they are typing in --
        every half second. The rebuild is held off until the rename is done.
        """
        self._editing_path = path

    def _end_row_edit(self) -> None:
        if self._editing_path is None:
            return
        self._editing_path = None
        # The rename may have changed the row, and the timer's next tick is up
        # to half a second away; redrawing now keeps the list in step with
        # what the user just typed.
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Rebuild the draft rows, newest first, marking the open one.

        Skipped entirely when nothing that shows has changed. The timer runs
        twice a second and the rows are real widgets with child labels and
        signals; rebuilding them on every tick threw the old ones away and
        built new ones for no reason, which is work the user pays for as
        flicker. The saved time is rounded to the minute because that is the
        finest thing the subtitle can say ("刚刚 / 3 分钟前 / 14:30").
        """
        if self._editing_path is not None:
            return

        current = str(self._path) if self._path else None
        signature = (
            current,
            self._dirty,
            tuple(
                (
                    info.name,
                    int(info.saved_at // 60),
                    info.bytes,
                    str(info.path) == current,
                )
                for info in self._drafts
            ),
        )
        if signature == self._shown_list:
            return
        self._shown_list = signature

        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Reparented away and hidden *now*, not just scheduled for
                # deletion: deleteLater only runs when the event loop gets
                # round to it, and until then the old rows are still children
                # of the host -- so every refresh left another set behind and
                # findChildren saw all of them.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        for info in self._drafts:
            is_current = current is not None and str(info.path) == current
            subtitle = _relative_time(info.saved_at)
            if is_current:
                subtitle = "正在编辑" + (
                    "　·　有未保存修改" if self._dirty else f"　·　{subtitle}"
                )
            subtitle += f"　·　{info.size_text}"
            row = DraftRow(info.name, subtitle, is_current, self._list_host)
            path = str(info.path)
            row.setToolTip(f"{path}\n单击打开；✎ 改名")
            row.clicked.connect(
                lambda _=False, target=path: self.open_path_requested.emit(target)
            )
            row.rename_committed.connect(
                lambda name, target=path: self.rename_requested.emit(target, name)
            )
            row.editing_started.connect(
                lambda target=path: self._begin_row_edit(target)
            )
            row.editing_finished.connect(self._end_row_edit)
            row.setContextMenuPolicy(Qt.CustomContextMenu)
            row.customContextMenuRequested.connect(
                lambda _point, target=path, name=info.name: self._row_menu(
                    target, name
                )
            )
            self._list_layout.addWidget(row)

        self._list_layout.addStretch(1)
        has_any = bool(self._drafts)
        self._list_area.setVisible(has_any)
        self.empty_label.setVisible(not has_any)

    def _row_menu(self, path: str, name: str) -> None:
        """Right-click menu for one draft row."""
        menu = QMenu(self)
        open_action = menu.addAction("打开")
        rename_action = menu.addAction("重命名…")
        delete_action = menu.addAction("删除")
        chosen = menu.exec(QCursor.pos())
        if chosen is open_action:
            self.open_path_requested.emit(path)
        elif chosen is rename_action:
            self.rename_requested.emit(path, name)
        elif chosen is delete_action:
            self.delete_requested.emit(path)


# ------------------------------------------------------- the open boards
#
# Switching tabs is the one operation that has to touch every pane, because
# each of them holds its own reference to the document. A pane that is missed
# keeps editing the board that was just replaced -- which looks like the new
# tab ignoring the user's next few clicks, the same failure the draft loader
# used to have. So all of it lives in one function, and the tab bar only
# reports which index was clicked.


def _remember_view(window) -> None:
    """Store the active board's zoom and pan before leaving it."""
    canvas = getattr(window, "design_canvas", None)
    if canvas is None:
        return
    workspace = getattr(window, "_workspace", None)
    if workspace is None:
        return
    canvas.board = workspace.active.board
    workspace.active.view = canvas.view_state()


def _show_project(window, project: Project, *, message: str = "") -> None:
    """Point the whole window at ``project``.

    Every pane reads the document through its own reference, so each one has
    to be re-pointed here. Missing one leaves a panel editing the document
    that is no longer on screen.
    """
    canvas = window.design_canvas
    canvas.board = project.board
    canvas.selected = None
    canvas.selection_changed.emit(None)
    # The board's history comes with the board. Sharing one stack across tabs
    # would let Ctrl+Z in this tab undo an edit made in another one.
    canvas.set_history(project.history)
    canvas.set_view_state(project.view)

    window.board = project.board
    window.layer_panel.set_board(project.board)
    window.image_inspector.set_layer_stack(project.board.layers)
    window.image_inspector.clear_placement()

    # The board-setup panel holds the old board, so it is closed rather than
    # left editing a document that is no longer active.
    _drop_board_panel(window)
    # 模板内容 is re-pointed instead of closed: it is a pane the user works in
    # alongside the canvas, and having it vanish on every tab switch would make
    # comparing two cards' wording impossible. Re-pointing is what keeps it
    # editing the board on screen rather than the one it was opened for.
    content = live_panel("_template_content_panel")
    if content is not None:
        try:
            if content.parentWidget() is window:
                content.set_board(project.board)
        except RuntimeError:
            pass
    preprocess = live_panel("_preprocess_panel")
    if preprocess is not None:
        try:
            if preprocess.parentWidget() is window:
                preprocess._refresh_board_note()
        except RuntimeError:
            pass

    refresh_project_tabs(window)
    refresh_draft_panel(window)
    if message:
        window.statusBar().showMessage(message)


def activate_project(window, index: int) -> Project:
    """Make one open board the one on screen."""
    workspace = window.workspace
    if index == workspace.index:
        refresh_project_tabs(window)
        return workspace.active
    _remember_view(window)
    project = workspace.switch(index)
    _show_project(window, project, message=f"已切换到：{project.title}")
    return project


def refresh_project_tabs(window) -> None:
    """Redraw the tab row from the workspace."""
    tabs = getattr(window, "project_tabs", None)
    if tabs is None or not hasattr(tabs, "set_titles"):
        return
    workspace = window.workspace
    tabs.set_titles(workspace.projects)
    blocked = tabs.blockSignals(True)
    try:
        tabs.setCurrentIndex(workspace.index)
    finally:
        tabs.blockSignals(blocked)


def new_project_tab(window) -> Project:
    """Open a fresh board in a new tab.

    Nothing is discarded and nothing is asked: the board the user was on stays
    open in its own tab, which is the entire reason tabs were added. The old
    behaviour -- clear the canvas after a confirmation -- is what the tabs
    replace.
    """
    _remember_view(window)
    project = window.workspace.new()
    _show_project(window, project, message=f"已新建板子：{project.title}")
    return project


def close_project_tab(window, index: int) -> bool:
    """Close one tab, asking about unwritten changes first.

    Returns whether the tab was closed. Refusing the question leaves the tab
    exactly where it was, including when it is not the active one.
    """
    workspace = window.workspace
    if not 0 <= index < len(workspace):
        return False
    project = workspace[index]
    if project.is_dirty() and not _confirm_discard_project(window, project):
        return False

    # The closing tab's view is not worth remembering, but the active one's is:
    # closing a background tab must not move the view of the board on screen.
    if index == workspace.index:
        canvas = getattr(window, "design_canvas", None)
        if canvas is not None:
            project.view = canvas.view_state()
    else:
        _remember_view(window)

    workspace.close(index)
    _show_project(window, workspace.active)
    window.statusBar().showMessage(f"已关闭：{project.title}")
    return True


def close_other_project_tabs(window, index: int) -> bool:
    """Keep one board open and close the rest, asking about each unwritten one."""
    workspace = window.workspace
    if not 0 <= index < len(workspace):
        return False
    kept = workspace[index]
    # Every question is asked *before* anything closes. Closing some tabs and
    # then stopping on a "no" would leave the user with a window they did not
    # ask for and no way to tell which tabs were already gone.
    for project in [item for item in workspace if item is not kept]:
        if project.is_dirty() and not _confirm_discard_project(window, project):
            return False

    workspace.close_others(index)
    _show_project(window, workspace.active, message=f"只保留：{kept.title}")
    return True


# ------------------------------------------------------------------ drafts
#
# One place owns the draft state: which file it is, and the fingerprint taken
# when it was last written. The panel displays it, the exit prompt consults it,
# and nothing else needs to know how a draft is stored.
#
# With several boards open, "the current draft" means the *active project's*
# draft. The window's ``_draft_path`` property forwards there, so the panel and
# the prompts did not each have to learn about tabs.


def draft_dirty(window) -> bool:
    """Whether saving the active board now would write something different.

    Computed fresh rather than read from the panel's cached flag. The panel's
    copy is refreshed on a timer, and the exit prompt must be right even if the
    user edited something in the last fraction of a second before closing --
    which is exactly when they would be most upset to lose it.
    """
    canvas = getattr(window, "design_canvas", None)
    if canvas is None:
        return False
    return fingerprint(canvas.board) != window.active_project.saved_fingerprint


def draft_store(window) -> DraftStore:
    """The window's draft library, created on first use."""

    store = getattr(window, "_draft_store", None)
    if store is None:
        store = DraftStore()
        window._draft_store = store
    return store


def refresh_draft_panel(window, *, reload_list: bool = True) -> None:
    """Push the current dirty state and the draft list into the sidebar."""
    panel = getattr(window, "draft_panel", None)
    if panel is None:
        return
    store = draft_store(window)
    drafts = store.list() if reload_list else None
    panel.set_document_state(
        getattr(window, "_draft_path", None),
        getattr(window, "_draft_saved_fingerprint", None),
        draft_dirty(window),
        drafts,
    )


def _write_project_draft(window, project: Project, path: str) -> bool:
    """Write one project's document to ``path``, without switching to it.

    Taking the project rather than reading the active one is what lets the
    exit prompt save every open board, and lets "关闭其他板子" write the ones
    in the background. The fingerprint is taken by the project itself, from
    the same serialiser that wrote the file, so "saved" can never mean
    something slightly different from "what is on disk".
    """
    try:
        saved = save_draft(project.board, path)
    except OSError as exc:
        QMessageBox.warning(
            window, "保存失败", f"无法写入草稿文件：\n{exc}"
        )
        return False
    project.draft_path = str(saved)
    project.mark_saved()
    # A board that is still called 未命名 takes the draft's name, so the tab
    # row reads as the user's own names instead of "未命名 4" by the end of a
    # session. A name the user gave the board is left alone.
    if project.is_unnamed():
        project.name_from_file(str(saved))
    return True


def save_draft_to(window, path: str) -> bool:
    """Save the active board under ``path`` and remember it as the saved one."""
    canvas = getattr(window, "design_canvas", None)
    if canvas is None:
        return False
    project = window.active_project
    project.board = canvas.board
    if not _write_project_draft(window, project, path):
        return False
    refresh_draft_panel(window)
    refresh_project_tabs(window)
    window.statusBar().showMessage(f"草稿已保存：{Path(path).name}")
    return True


def _stash_working_state(window, project: Project) -> bool:
    """Write a project's editing session into the draft library.

    Called after a product file is written. The ``.epro2`` and the Gerber set
    are *products*: the first is geometry for the editor, the second is an
    order. Neither carries the editing session -- the source picture, the
    preprocessing parameters behind it, which layers were switched on -- so a
    menu item called 另存为 that left the tab marked ● would be claiming the
    work is not saved when the user just saved it.

    The draft is where the session is kept, and writing it here is a real
    write rather than a cleared flag: the fingerprint is only allowed to move
    forward when something has actually been put on disk.
    """
    path = project.draft_path
    if not path:
        store = draft_store(window)
        path = str(store.path_for(store.unique_name(draft_name_for(project))))
    return _write_project_draft(window, project, path)


def save_all_drafts(window) -> bool:
    """Write every open board that has unwritten changes. False if any failed."""
    canvas = getattr(window, "design_canvas", None)
    if canvas is not None:
        window.active_project.board = canvas.board
    store = draft_store(window)

    ok = True
    for project in window.workspace.projects:
        if not project.is_dirty():
            continue
        path = project.draft_path
        if not path:
            path = str(store.path_for(store.unique_name(draft_name_for(project))))
        ok = _write_project_draft(window, project, path) and ok
    refresh_draft_panel(window)
    refresh_project_tabs(window)
    return ok


def save_draft_as(window) -> bool:
    """Save under a new name, asked for in place rather than in a file dialog.

    The library is the point: asking for a *path* would put the draft outside
    it and back in the world of remembering where things were saved.
    """
    store = draft_store(window)
    current = getattr(window, "_draft_path", None)
    suggested = Path(current).stem if current else store.unique_name()
    name, accepted = QInputDialog.getText(
        window, "另存为新草稿", "新草稿名称：", text=suggested
    )
    if not accepted:
        return False
    name = str(name).strip()
    if not name:
        return False
    target = store.path_for(name)
    # Replacing a draft the user is not currently editing is a destructive act
    # they did not ask for; naming a new draft after an existing one is a
    # plausible slip rather than an instruction to overwrite it.
    if target.exists() and str(target) != str(current):
        answer = QMessageBox.question(
            window,
            "已存在同名草稿",
            f"「{name}」已经有一份草稿了，要覆盖它吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
    return save_draft_to(window, str(target))


def draft_name_for(project: Project) -> str:
    """What to call a board that has never been saved to the draft library.

    The tab title when the user gave the board a name, otherwise the artwork's
    filename -- which is what they recognise it by -- and finally 未命名. The
    exporter's neutral ``ArtBoard`` is deliberately *not* used here: it is a
    reasonable file title and a poor row in a list of the user's drafts.
    """
    from pcbart.export import default_title

    if not project.is_unnamed():
        return project.title
    found = default_title(project.board)
    return UNTITLED if found == "ArtBoard" else found


def save_draft_clicked(window) -> bool:
    """The panel's 保存草稿 button.

    Saves straight into the library, naming the first one after the artwork so
    the list is readable without the user having to invent a name up front.
    """
    canvas = getattr(window, "design_canvas", None)
    if canvas is None:
        return False
    current = getattr(window, "_draft_path", None)
    if current:
        return save_draft_to(window, current)

    store = draft_store(window)
    name = store.unique_name(draft_name_for(window.active_project))
    return save_draft_to(window, str(store.path_for(name)))


def new_draft(window) -> bool:
    """The sidebar's 新建 button: a fresh board in a new tab."""
    new_project_tab(window)
    return True


def rename_draft(window, path: str, name: str) -> bool:
    """Rename a saved draft in the library."""
    store = draft_store(window)
    new_path = store.path_for(name)
    source = Path(path)
    if str(new_path) == str(source):
        return True
    if new_path.exists():
        QMessageBox.warning(
            window, "无法重命名", f"「{name}」已经有一份草稿了。"
        )
        return False
    try:
        source.replace(new_path)
    except OSError as exc:
        QMessageBox.warning(window, "无法重命名", f"重命名失败：\n{exc}")
        return False

    # Any open board following that draft follows the file, name and all: its
    # next save would otherwise recreate the old name behind the user, and the
    # tab would sit there naming a file that no longer exists. The library is
    # where the user names their work, so renaming a draft *is* naming the
    # board that is open on it.
    for project in window.workspace.projects:
        if str(project.draft_path or "") == str(source):
            project.draft_path = str(new_path)
            project.rename_from_project(str(new_path))
    refresh_draft_panel(window)
    refresh_project_tabs(window)
    window.statusBar().showMessage(f"草稿已改名为「{name}」")
    return True


def rename_draft_prompt(window, path: str, current_name: str) -> bool:
    """Ask for a new name, then rename. Used by the row's right-click menu.

    The row's own ✎ box does not come through here: that one is typed in
    place, while this is the menu route for anyone who reaches for the
    context menu first.
    """
    name, accepted = QInputDialog.getText(
        window, "重命名草稿", "草稿名称：", text=current_name
    )
    if not accepted:
        return False
    name = str(name).strip()
    if not name or name == current_name:
        return False
    return rename_draft(window, path, name)


def delete_draft(window, path: str) -> bool:
    """Delete a draft, with a confirmation naming it."""
    target = Path(path)
    # Which *open boards* follow this file. Deleting the draft of a background
    # tab is as destructive as deleting the one on screen, so both are asked
    # about rather than only the active one.
    linked = [
        project
        for project in window.workspace.projects
        if str(project.draft_path or "") == str(target)
    ]
    if linked:
        dirty = [project for project in linked if project.is_dirty()]
        detail = (
            "正在编辑这份草稿，而且有未保存的修改"
            if dirty
            else "正在编辑这份草稿"
        )
        answer = QMessageBox.warning(
            window,
            "这份草稿正在使用",
            f"「{target.stem}」是{detail}。\n删除它会丢掉这份设计，确定要删吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False

    if not draft_store(window).delete(target):
        QMessageBox.warning(window, "无法删除", f"删不掉：{target}")
        return False

    for project in linked:
        # The file is gone, so the document is no longer backed by anything.
        # Keeping the path would make the next save silently resurrect it.
        project.draft_path = None
        project.saved_fingerprint = None
    refresh_draft_panel(window)
    refresh_project_tabs(window)
    window.statusBar().showMessage(f"已删除草稿：{target.stem}")
    return True


def _drop_board_panel(window) -> None:
    """Close the board-setup panel, which holds a reference to the old board."""
    global _board_panel
    panel = live_panel("_board_panel")
    if panel is None:
        _board_panel = None
        return
    try:
        mine = panel.parentWidget() is window
    except RuntimeError:
        mine = False
    if mine:
        panel.hide()
        panel.deleteLater()
        _board_panel = None


def open_draft_file(window, path: str | None = None) -> bool:
    """Open a saved draft, in its own tab.

    Opening used to replace whatever was on the canvas, which is why it had to
    ask about unsaved changes first. With tabs there is nothing to throw away:
    a draft whose file is already open is switched to, and anything else opens
    beside what the user was looking at.
    """
    if path is None:
        # Falling back to the library rather than a file dialog: the list is
        # where drafts live now, and the panel already opens them by path.
        path, _filter = QFileDialog.getOpenFileName(
            window, "打开草稿文件", str(draft_store(window).root), DRAFT_FILTER
        )
        if not path:
            return False

    workspace = window.workspace
    for index, project in enumerate(workspace):
        if str(project.draft_path or "") == str(path):
            activate_project(window, index)
            window.statusBar().showMessage(f"已切换到：{project.title}")
            return True

    try:
        board = load_draft(path)
    except ValueError as exc:
        QMessageBox.warning(window, "无法打开草稿", str(exc))
        return False

    _remember_view(window)
    project = Project(board, title=Path(path).stem, draft_path=str(path))
    workspace.add(project)
    _show_project(window, project, message=f"已打开草稿：{project.title}")
    return True


def _confirm_discard_project(window, project: Project) -> bool:
    """Ask about one named project's unwritten changes.

    The name is in the question because with several boards open "还有未保存
    的修改" is not enough to act on: the user has to know *which* one they are
    about to lose.
    """
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("还有未保存的修改")
    box.setText(f"「{project.title}」有未保存的修改。要保存吗？")
    box.setInformativeText(
        "选择「保存草稿」会写入"
        + (
            f" {Path(project.draft_path).name}"
            if project.draft_path
            else "一个新的草稿文件"
        )
        + "，然后继续。"
    )
    save_button = box.addButton("保存草稿", QMessageBox.AcceptRole)
    discard_button = box.addButton("不保存", QMessageBox.DestructiveRole)
    cancel_button = box.addButton("取消", QMessageBox.RejectRole)
    box.setDefaultButton(save_button)
    box.exec()

    clicked = box.clickedButton()
    if clicked is cancel_button:
        return False
    if clicked is discard_button:
        return True
    _ = save_button

    path = project.draft_path
    if not path:
        store = draft_store(window)
        path = str(store.path_for(store.unique_name(draft_name_for(project))))
    return _write_project_draft(window, project, path)


def confirm_discard_changes(window, *, action: str = "退出") -> bool:
    """Ask about the active board's unsaved work before an action that loses it.

    Kept for the paths that act on the active document only. The exit route
    uses :func:`confirm_discard_all`, which covers every open tab -- quitting
    with four boards open must not silently drop three of them.
    """
    if not draft_dirty(window):
        return True
    return _confirm_discard_project(window, window.active_project)


def confirm_discard_all(window, *, action: str = "退出") -> bool:
    """Ask about every open board with unwritten changes.

    Asked in tab order and saved or discarded one at a time, so a "取消"
    stops the whole thing rather than leaving half the boards written. The
    ones already answered are done, which is what the user's answers said.
    """
    canvas = getattr(window, "design_canvas", None)
    if canvas is not None and hasattr(window, "_workspace"):
        window.active_project.board = canvas.board

    for project in window.workspace.dirty():
        if not _confirm_discard_project(window, project):
            return False
    refresh_project_tabs(window)
    return True


def show_about(main_window: QWidget) -> None:
    """The product summary and the version number.

    The summary names what the tool does; the version answers "which build was
    it" when a user reports a problem.
    """
    from pcbart import __version__

    QMessageBox.information(
        main_window,
        "关于",
        f"PCB 艺术板工具　v{__version__}\n\n"
        "主要功能：\n"
        "· 多板并行编辑（顶部工程栏，可新建 / 切换 / 关闭）\n"
        "· 文件菜单：新建板子、另存为工程（.epro2）、"
        "另存为下单文件（Gerber 压缩包）\n"
        "· 导入图片、线稿转换、降噪、板画风格\n"
        "· 板框设置（矩形 / 圆角矩形 / 圆形 / 异形板 / 立牌）\n"
        "· 通行证模板（干员搜索、二维码与条形码、卡面素材）\n"
        "· 图层显示与编辑（正反铜层 / 丝印层 / 阻焊层 / 板框层）\n"
        "· 画布缩放、平移、图片放置、尺寸调整与翻转旋转\n"
        "· 草稿库（保存在软件内，可重命名 / 删除）\n\n"
        "许可证：GPL-3.0（GNU 通用公共许可证第 3 版）。\n"
        "完整条款见安装目录下的 LICENSE.txt；\n"
        "源代码：https://github.com/XiKaaaSaMa/EzPCBArtTool",
    )


def show_shortcuts(main_window: QWidget) -> None:
    """【帮助 → 快捷键】: the whole cheat sheet, on demand.

    The lines come from ``shortcut_lines()``, which is assembled from the same
    tuples the tests check the real bindings against -- so this page cannot
    teach a key the window does not have.
    """
    QMessageBox.information(main_window, "快捷键", "\n".join(shortcut_lines()))


#: How the user answers :func:`ask_print_kind`.
PRINT_NORMAL = "normal"
PRINT_COLOR = "color"


def ask_print_kind(main_window: QWidget, board, parent: QWidget | None = None):
    """Ask whether this board is a **普通打板** or a **彩色丝印打板**.

    Returns :data:`PRINT_NORMAL`, :data:`PRINT_COLOR`, or ``None`` if the user
    cancelled.

    ⚠️ **This is not a cosmetic choice -- the two produce different boards.**
    A picture on the canvas (an unprocessed import, or the card's own 二维码 /
    条形码 / 职业图标 / 阵营 logo) reaches a board only through the EDA's
    colour-silkscreen export. Measured: on a card with two pictures, a plain
    Gerber export left both silkscreen layers with **zero** draw commands.

    So the answer decides how the artwork is written:

    * 普通打板 traces every picture into filled regions. The Gerber set then
      carries the artwork, and no colour job is involved.
    * 彩色丝印打板 keeps the pictures, and the EDA writes the colour files.

    Only asked when there is something to decide: a board with no pictures
    exports straight through, so the common case stays one click.
    """
    from pcbart.export import picture_placements

    pictures = picture_placements(board)
    if not pictures:
        return PRINT_NORMAL

    box = QMessageBox(parent or main_window)
    box.setWindowTitle("导出方式")
    box.setIcon(QMessageBox.Question)
    box.setText(f"这张板上有 {len(pictures)} 张图片，要按哪种方式导出？")
    box.setInformativeText(
        "普通打板：图片会被描成填充图形，随普通 Gerber 一起生产，不开彩印。\n"
        "彩色丝印打板：保留原图配色，由嘉立创 EDA 出彩色丝印文件。\n\n"
        "⚠️ 两者产出不同的板子：彩色丝印的图片不写入普通 Gerber，"
        "选错会让这些图案在成品上直接消失。"
    )
    normal = box.addButton("普通打板", QMessageBox.AcceptRole)
    color = box.addButton("彩色丝印打板", QMessageBox.AcceptRole)
    box.addButton("取消", QMessageBox.RejectRole)
    box.setDefaultButton(color)
    box.exec()
    clicked = box.clickedButton()
    if clicked is normal:
        return PRINT_NORMAL
    if clicked is color:
        return PRINT_COLOR
    return None


def _confirm_plain_gerber_for_pictures(
    main_window: QWidget, count: int
) -> bool:
    """Warn that a plain Gerber will trace the board's pictures. ``True`` to go on.

    Its own function so tests can answer it without a dialog; see
    :func:`ask_print_kind` for why the distinction matters at all.
    """
    box = QMessageBox(main_window)
    box.setWindowTitle("这是普通 Gerber")
    box.setIcon(QMessageBox.Information)
    box.setText(
        f"这张板上有 {count} 张图片，它们会按普通 Gerber 描成填充图形。"
    )
    box.setInformativeText(
        "要出彩色丝印下单文件，请改用【另存为工程】导出 .epro2，"
        "再在嘉立创 EDA 里用「一键生成」出彩印包 —— "
        "彩色数据只能由 EDA 自己编码。\n\n"
        "继续导出的是普通打板文件，图片会以单色图形生产。"
    )
    box.addButton("继续导出普通 Gerber", QMessageBox.AcceptRole)
    cancel = box.addButton("取消", QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.exec()
    return box.clickedButton() is not cancel


def export_project(main_window: QWidget, parent: QWidget | None = None) -> str | None:
    """【文件 → 另存为工程】: ask where to save, write the board, report it.

    Returns the written path, or ``None`` when the user cancelled or the export
    was refused. Every refusal is shown as a message: a silent no-op would be
    indistinguishable from a broken button, which is the failure mode this
    whole editor keeps being audited for.
    """
    canvas = getattr(main_window, "design_canvas", None)
    if canvas is None:
        QMessageBox.information(main_window, "暂不可用", "主窗口尚未准备好设计画布。")
        return None

    from pcbart.export import default_title, export_design

    project = main_window.active_project
    project.board = canvas.board

    # Which kind of order this is decides how the pictures are written, and
    # getting it wrong makes them vanish from the finished board -- so it is
    # asked before the file dialog, not after the fact.
    kind = ask_print_kind(main_window, canvas.board, parent)
    if kind is None:
        return None

    # Reopening the dialog at the last target is what makes the second save of
    # a board cheap; the name follows the board unless the user set one.
    if project.project_path:
        suggested = project.project_path
    else:
        suggested = f"{default_title(canvas.board)}.epro2"
    path, _filter = QFileDialog.getSaveFileName(
        parent or main_window,
        "另存为工程",
        suggested,
        "EasyEDA Pro 工程 (*.epro2);;所有文件 (*)",
    )
    if not path:
        return None
    if not path.lower().endswith(".epro2"):
        path += ".epro2"

    try:
        report = export_design(
            canvas.board, path, color_print=(kind == PRINT_COLOR)
        )
    except ValueError as exc:
        QMessageBox.warning(parent or main_window, "无法导出", str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        import traceback

        traceback.print_exc()
        QMessageBox.warning(parent or main_window, "导出失败", f"写入文件时出错：\n{exc}")
        return None

    # The product file is remembered, and the tab takes its name: after
    # "另存为工程" the user has named the *board*, and the tab row should say
    # so even if the tab had picked up another name earlier.
    written = str(report["output"])
    project.project_path = written
    project.rename_from_project(written)
    stashed = _stash_working_state(main_window, project)
    refresh_project_tabs(main_window)
    refresh_draft_panel(main_window)

    # ⚠️ **The dialog says only that the save worked.** The user asked for that
    # in as many words: 「只提示工程保存完成」. It used to append the whole writer
    # report, which on a card with one misplaced illustration grew to a page of
    # near-identical lines and buried the sentence the user actually needed.
    #
    # The report is not thrown away -- it goes to the status bar, where it is
    # readable and dismissible instead of blocking. It still has to be said
    # somewhere: its entries include 「图片已移出立绘区，本层没有内容可印」, i.e. a
    # picture that will not appear on the board at all. Dropping that would turn
    # a warning into silent data loss, which is the failure this project keeps
    # designing against.
    note = (
        f"已保存工程，板框 {report['width_mm']:.1f} × {report['height_mm']:.1f} mm"
        f"，{report['vertices']} 个顶点。"
    )
    detail = str(report["note"]) if report["artwork_placements"] else ""
    QMessageBox.information(
        parent or main_window,
        "工程保存完成",
        f"已保存：{Path(written).name}",
    )
    summary = note
    if detail:
        summary += f"　·　{detail}"
    if stashed:
        summary += "　·　当前编辑状态已存进草稿库"
    main_window.statusBar().showMessage(summary)
    return written


def export_gerber_project(main_window: QWidget) -> str | None:
    """【文件 → 另存为下单文件】: one Gerber zip the user can upload as-is.

    A *file name* is asked for rather than a folder, and only the zip is left
    behind. This is the "send it to the fab" action, and a folder of seven
    loose layers is a chore at exactly that moment. The full set still lives
    inside the zip, so uploading it is the same as uploading the folder's own
    archive.
    """
    canvas = getattr(main_window, "design_canvas", None)
    if canvas is None:
        QMessageBox.information(main_window, "暂不可用", "主窗口尚未准备好设计画布。")
        return None

    from pcbart.export import default_title, picture_placements
    from pcbart.gerber import DEFAULT_PREFIX, export_gerber_archive
    from pcbart.paths import default_output_dir

    project = main_window.active_project
    project.board = canvas.board

    # ⚠️ **A Gerber has no picture primitive.** This export always traces, so
    # the artwork does print -- but if the user meant to order colour, this
    # file is not that order, and saying so now is cheaper than a board that
    # comes back without the colour they designed.
    if picture_placements(canvas.board) and not _confirm_plain_gerber_for_pictures(
        main_window, len(picture_placements(canvas.board))
    ):
        return None

    # Default to a folder named after the design, beside the draft if there is
    # one -- so the two halves of "my work" end up next to each other instead
    # of in whatever directory the shortcut happened to start in.
    draft = project.draft_path
    start = str(Path(draft).parent) if draft else str(default_output_dir())
    suggested = project.order_path or str(
        Path(start) / f"{default_title(canvas.board)}.zip"
    )
    if not Path(suggested).is_absolute():
        suggested = str(Path(start) / Path(suggested).name)
    path, _filter = QFileDialog.getSaveFileName(
        main_window,
        "另存为下单文件（Gerber 压缩包）",
        suggested,
        "Gerber 压缩包 (*.zip);;所有文件 (*)",
    )
    if not path:
        return None
    if not path.lower().endswith(".zip"):
        path += ".zip"

    title = default_title(canvas.board)
    try:
        report = export_gerber_archive(
            canvas.board, path, prefix=DEFAULT_PREFIX, title=title
        )
    except ValueError as exc:
        QMessageBox.warning(main_window, "无法导出", str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        import traceback

        traceback.print_exc()
        QMessageBox.warning(
            main_window, "导出失败", f"写入 Gerber 文件时出错：\n{exc}"
        )
        return None

    written = str(report["archive"])
    project.order_path = written
    project.name_from_file(written)
    stashed = _stash_working_state(main_window, project)
    refresh_project_tabs(main_window)
    refresh_draft_panel(main_window)

    names = "\n".join(f"· {entry['name']}" for entry in report["files"])
    QMessageBox.information(
        main_window,
        "下单文件已生成",
        f"{report['note']}\n\n包含：\n{names}\n\n文件：{written}"
        + ("\n\n当前编辑状态已同时存进草稿库。" if stashed else ""),
    )
    main_window.statusBar().showMessage(
        f"下单文件已生成，含 {len(report['files'])} 个 Gerber 文件"
    )
    return written


#: Denoise kernel for a softly shaded picture, as a fraction of its short side.
#:
#: Tuned against the drawing that prompted the auto-detect: 0.65% of its 1378 px
#: short side is 9 px, which is where the fragmented rings stop (74 fragments at
#: 5 px, 31 at 9) while the fine hair strands are still intact (15 px began
#: erasing them). Expressed as a fraction rather than a number because the
#: slider's unit is pixels of the user's own picture.
SOFT_SHADING_KERNEL_SHARE = 0.0065


#: Keyboard shortcuts for the 编辑 menu, following the conventions an EDA tool
#: uses. Only operations this tool actually has appear -- there is no redo stack
#: and no multi-selection, so ``Ctrl+Y`` and ``Ctrl+A`` are absent rather than
#: bound to something that would do nothing.
EDIT_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("撤销", "Ctrl+Z", "undo"),
    ("-", "", ""),
    ("复制", "Ctrl+C", "copy_selection"),
    ("剪切", "Ctrl+X", "cut_selection"),
    ("粘贴", "Ctrl+V", "paste"),
    ("复制一份", "Ctrl+D", "duplicate_selection"),
    ("删除", "Del", "delete_selection"),
    ("旋转 90°", "R", "rotate_quarter_turn"),
    ("水平翻转", "", "flip_horizontal"),
    ("垂直翻转", "", "flip_vertical"),
    ("居中", "", "centre_selection"),
    ("按板框缩放", "", "fit_to_board"),
    ("-", "", ""),
    ("组合所选…", "Ctrl+G", "group_selection"),
    ("取消组合", "Ctrl+Shift+G", "ungroup_selection"),
    ("组合管理…", "", "manage_groups"),
)

#: Entries that need something selected, and so are greyed out otherwise.
EDIT_NEEDS_SELECTION = frozenset(
    {
        "复制",
        "剪切",
        "复制一份",
        "删除",
        "旋转 90°",
        "水平翻转",
        "垂直翻转",
        "居中",
    }
)


def _group_selected(window: QWidget, canvas) -> None:
    """组合所选…: ask for a name, then group.

    The name is asked for up front rather than left to 组合管理 afterwards,
    because a group is what the user will see in the layer list and in the
    editor -- naming it at the moment it is created is when they know what it
    is. Cancelling the prompt cancels the whole operation: a group appearing
    after the user pressed 取消 would be the tool acting on a withdrawn
    instruction.
    """
    from PySide6.QtWidgets import QInputDialog

    chosen = canvas.selection_set()
    if len(chosen) < 2:
        QMessageBox.information(
            window,
            "无法组合",
            "请先在画布上选中至少两个元素。\n"
            "Ctrl+单击＝逐个加选，Shift+单击＝选中一段范围。",
        )
        return

    default = f"组合 {len(canvas.board.groups) + 1}"
    name, accepted = QInputDialog.getText(
        window, "组合所选", "给这个组合起个名字：", text=default
    )
    if not accepted:
        return
    if canvas.group_selection(name):
        show_group_manager(window).refresh()


def _wire_edit_actions(
    window: QMainWindow, actions_by_name: dict, canvas, on_fit_requested
) -> None:
    """Connect the 编辑 menu to the canvas's editing operations.

    The menu and the canvas's right-click menu call the same methods, so an
    operation cannot behave one way from the keyboard and another from the
    mouse. Shortcuts ride on the menu actions, which is what makes them work
    wherever focus is rather than only while the canvas has it.
    """
    operations = {
        "undo": canvas.undo,
        "copy_selection": canvas.copy_selection,
        "cut_selection": canvas.cut_selection,
        "paste": canvas.paste,
        "duplicate_selection": canvas.duplicate_selection,
        "delete_selection": canvas.delete_selection,
        "rotate_quarter_turn": lambda: canvas.rotate_selection(90.0),
        "flip_horizontal": lambda: canvas.flip_selection(horizontal=True),
        "flip_vertical": lambda: canvas.flip_selection(horizontal=False),
        "centre_selection": canvas.centre_selection,
        "fit_to_board": on_fit_requested,
        "group_selection": lambda: _group_selected(window, canvas),
        "ungroup_selection": canvas.ungroup_selection,
        "manage_groups": lambda: show_group_manager(window),
    }

    actions: dict[str, QAction] = {}
    for label, shortcut, operation in EDIT_ACTIONS:
        if label == "-":
            # A separator in the menu, skipped here: it carries no action.
            continue
        action = actions_by_name[label]
        action.triggered.connect(operations[operation])
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            # Without this the shortcut only fires while the menu bar has
            # focus, which is never where the user is.
            action.setShortcutContext(Qt.WindowShortcut)
        actions[label] = action

    # The canvas's own right-click entries are the same operations, so they are
    # wired through the menu actions' enabled state below rather than being
    # allowed to drift.
    def sync_enabled(_selection=None) -> None:
        has_selection = canvas.selected is not None
        for label in EDIT_NEEDS_SELECTION:
            actions[label].setEnabled(has_selection)
        actions["粘贴"].setEnabled(canvas.clipboard_size > 0)
        # 组合 needs *two* objects, so it cannot ride on EDIT_NEEDS_SELECTION --
        # a single object satisfies that set and would leave the entry looking
        # available for an operation that would then do nothing.
        actions["组合所选…"].setEnabled(canvas.can_group())
        # 取消组合 only means something when something selected is actually in
        # a group. Offering it otherwise invites a click that reports nothing.
        actions["取消组合"].setEnabled(
            any(canvas.board.group_of(item) for item in canvas.selection_set())
        )
        actions["组合管理…"].setEnabled(bool(canvas.board.groups))
        # Undo is available on its own terms: it is about the document's
        # history, not about what is selected. A greyed-out entry is how the
        # user knows they have reached the beginning.
        actions["撤销"].setEnabled(canvas.can_undo())

    canvas.selection_changed.connect(sync_enabled)
    # Copying changes neither the selection nor the document, but it is what
    # makes 粘贴 possible, so it has to refresh the enabled state too.
    canvas.clipboard_changed.connect(sync_enabled)
    # Every edit changes whether there is anything to undo, and undo itself
    # changes it again.
    canvas.history_changed.connect(sync_enabled)
    # A paste or a delete changes what is on the board without moving the
    # selection, so the layer/draft panes are told separately.
    canvas.document_changed.connect(lambda: refresh_workspace_state(window))
    canvas.fit_to_board_requested.connect(on_fit_requested)
    canvas.reset_view_requested.connect(canvas.fit_view)
    sync_enabled()


# The board templates this build knows about. Registered once when the module
# loads so the board panel's 套用模板 entry has something to list.
badge_card.register()


def _is_badge_board(board: ArtBoard) -> bool:
    """Whether ``board`` is a 通行证 card, so 模板内容 has something to edit.

    Asked of the outline rather than of a flag, for the same reason the canvas
    asks it that way: a board that stops being a card stops offering the entry
    without anything having to remember to clear a flag.
    """
    return (
        board.outline.shape == OutlineShape.TEMPLATE
        and badge_card.is_badge_template(board.outline.template or "")
    )


def build_window() -> QMainWindow:
    window = MainWindow()
    window.setWindowTitle("PCB 艺术板工具")
    window.resize(1420, 900)

    menu = window.menuBar()
    # Only entries that do something are listed. 显示网格 / 显示掩膜叠加 /
    # 显示孔位 / 打开工程 / 运行工艺校验 describe capabilities that do not
    # exist yet, and a menu item that silently does nothing is worse than an
    # absent one -- the user cannot tell it apart from a bug.
    actions_by_name: dict[str, QAction] = {}

    def add_items(sub: QMenu, items: list[str]) -> None:
        for item in items:
            if item == "-":
                sub.addSeparator()
                continue
            action = QAction(item, window)
            actions_by_name[item] = action
            sub.addAction(action)

    # 文件 leads with what is done *to a board* -- start one, save it, send it
    # -- and puts the in-app draft library underneath, because the library is
    # the working state rather than a product. 另存为工程 / 另存为下单文件 are
    # named for their targets: one goes to the EDA to keep editing, the other
    # goes to the fab as a Gerber zip.
    add_items(
        menu.addMenu("文件"),
        [
            "新建板子",
            "-",
            "另存为工程…",
            "另存为下单文件…",
            "保存草稿",
            "打开草稿…",
            "-",
            "打开图片…",
            "图像预处理…",
            "板框设置…",
            "模板内容…",
            "-",
            "退出",
        ],
    )
    add_items(
        menu.addMenu("编辑"),
        [
            "撤销",
            "-",
            "复制",
            "剪切",
            "粘贴",
            "复制一份",
            "删除",
            "-",
            "旋转 90°",
            "水平翻转",
            "垂直翻转",
            "居中",
            "按板框缩放",
            "-",
            "组合所选…",
            "取消组合",
            "组合管理…",
            "-",
            "适应窗口",
            "放大",
            "缩小",
        ],
    )
    # 界面设置 sits under 视图 with the other "how it looks" commands rather
    # than in 编辑, which is for editing the *board*. A theme is a property of
    # the application, and 视图 is where a user looks for it.
    add_items(menu.addMenu("视图"), ["图层…", "-", "界面设置…"])
    add_items(menu.addMenu("帮助"), ["快捷键", "关于"])

    new_project_action = actions_by_name["新建板子"]
    export_action = actions_by_name["另存为工程…"]
    gerber_action = actions_by_name["另存为下单文件…"]
    save_draft_action = actions_by_name["保存草稿"]
    open_draft_action = actions_by_name["打开草稿…"]
    quit_action = actions_by_name["退出"]
    about_action = actions_by_name["关于"]
    shortcuts_action = actions_by_name["快捷键"]
    open_image_action = actions_by_name["打开图片…"]
    preprocess_action = actions_by_name["图像预处理…"]
    board_action = actions_by_name["板框设置…"]
    content_action = actions_by_name["模板内容…"]
    layers_action = actions_by_name["图层…"]
    fit_action = actions_by_name["适应窗口"]
    zoom_in_action = actions_by_name["放大"]
    zoom_out_action = actions_by_name["缩小"]
    settings_action = actions_by_name["界面设置…"]

    toolbar = QToolBar("主工具栏")
    toolbar.setMovable(False)
    toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    # The toolbar carries no stylesheet of its own any more: its look comes
    # from the application-wide Material sheet (tools/qt_theme.py), which is
    # what lets it follow a theme switch. A local sheet here would keep the old
    # colours.
    toolbar_by_name: dict[str, QAction] = {}
    # 重新生成 and 校验 are gone: nothing regenerates yet (export writes
    # straight from the document) and there is no validator to run. They were
    # the two most obviously dead buttons on the bar.
    #
    # The icons are Iconoir, drawn from pcbart.ui_icons rather than taken from
    # the platform style. The native ones are a fixed light-theme set: on a
    # dark panel they read as muddy grey shapes, and they cannot be recoloured.
    for text, icon_name in [
        ("打开图片", "open_image"),
        ("图像预处理", "preprocess"),
        ("板框设置", "board_setup"),
        ("图层", "layers"),
        ("适应窗口", "fit_window"),
    ]:
        action = QAction(qt_theme_icon(icon_name, TEXT, size=18), text, window)
        toolbar_by_name[text] = action
        toolbar.addAction(action)
    toolbar.addSeparator()
    export_toolbar_action = QAction(
        qt_theme_icon("save_project", TEXT, size=18), "另存为工程", window
    )
    toolbar.addAction(export_toolbar_action)
    gerber_toolbar_action = QAction(
        qt_theme_icon("export_gerber", TEXT, size=18), "另存为下单文件", window
    )
    toolbar.addAction(gerber_toolbar_action)
    window.addToolBar(toolbar)
    window.export_action = export_action
    window.gerber_action = gerber_action
    window.new_project_action = new_project_action

    canvas = DesignCanvas()
    window.design_canvas = canvas
    window.board = canvas.board

    # The right pane answers two questions in one place: "which layer will this
    # go on" when nothing is selected, and "how big is this picture" when
    # something is. Driving it from the selection keeps the switch tied to what
    # the user is doing rather than to two panes taking turns being visible.
    layer_pane = LayerPanel(window, canvas.board)
    window.layer_panel = layer_pane
    inspector = ImagePropertyPanel()
    inspector.set_layer_stack(canvas.board.layers)
    window.image_inspector = inspector

    def on_selection(placement) -> None:
        # The scroll areas are toggled, not the panes inside them: hiding a
        # scroll area's widget leaves the empty frame behind.
        #
        # ⚠️ Only *artwork* opens the image inspector. A text line or a cutout
        # is not a picture -- it has no file, no aspect ratio and no opacity --
        # and feeding one to the inspector crashed on the first attribute it
        # read. They are edited in 模板内容 instead, which is also where their
        # own panel lives, so the dock falls back to the layer list.
        if placement is None or not isinstance(placement, ImagePlacement):
            inspector_area.setVisible(False)
            layer_pane.refresh()
            layer_area.setVisible(True)
        else:
            layer_area.setVisible(False)
            inspector.set_placement(placement)
            inspector_area.setVisible(True)

    def on_inspector_changed() -> None:
        canvas.update()
        # The property pane edits the same placements the canvas draws, so its
        # changes belong in the same undo history. Keyed by field so a spinbox
        # drag or a typed number is one step rather than one per increment.
        canvas.record_external_change(inspector.edit_run_key())

    def on_inspector_closed() -> None:
        canvas.selected = None
        canvas.selection_changed.emit(None)
        canvas.update()

    def on_fit_requested() -> None:
        board = canvas.board
        selected = inspector._placement
        fitted = board.fit_image_to_board(margin_mm=1.0)
        if fitted is board:
            # Nothing to do: the artwork already fits.
            return
        # Applied into the live board rather than swapped for the returned copy.
        # The undo history, the layer panel and the board-setup panel all hold
        # the board object, so replacing it would leave every one of them
        # editing a document no longer on screen.
        apply_board_dict(board, board_to_dict(fitted))
        # The placements keep their identity through that apply, so the
        # inspector and the selection stay valid and nothing has to be
        # re-pointed.
        if selected is not None:
            inspector.set_placement(selected)
        canvas.record_external_change()
        canvas.placement_changed.emit()
        canvas.update()

    def on_placement_changed() -> None:
        """Keep the inspector showing the canvas's numbers after a drag.

        The board panel refreshes itself: it listens to the canvas directly.
        """
        selected = canvas.selected
        if inspector.isVisible() and selected is not None:
            inspector.set_placement(selected)

    def on_layers_changed() -> None:
        canvas.update()
        # Hiding the layer of the current selection must not leave the property
        # pane editing something that is no longer on screen.
        if (
            canvas.selected is not None
            and not canvas.board.layers.selectable(canvas.selected.layer_key)
        ):
            canvas.selected = None
            canvas.selection_changed.emit(None)
        elif inspector.isVisible():
            inspector.refresh_layers()

    inspector.changed.connect(on_inspector_changed)
    inspector.closed.connect(on_inspector_closed)
    inspector.fit_requested.connect(on_fit_requested)
    canvas.selection_changed.connect(on_selection)
    canvas.placement_changed.connect(on_placement_changed)
    layer_pane.changed.connect(on_layers_changed)

    # Each pane scrolls inside the dock. The property pane is around 500 px of
    # content, so a short window would otherwise squeeze its controls into each
    # other -- the same reason the floating panels scroll.
    def scrollable(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        area.setWidget(widget)
        return area

    layer_area = scrollable(layer_pane)
    inspector_area = scrollable(inspector)
    window.layer_area = layer_area
    window.inspector_area = inspector_area

    right_side = QWidget()
    right_layout = QVBoxLayout(right_side)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)
    right_layout.addWidget(layer_area, 1)
    right_layout.addWidget(inspector_area, 1)
    # Both live in the same slot; exactly one is visible at a time, and the
    # layer list is the one that shows when nothing is selected.
    inspector_area.setVisible(False)

    # The left column is the draft pane. It is not a parameter column: every
    # number in it is computed from the document, and it holds the one action
    # the user reaches for while working -- saving what they have so far.
    draft_pane = DraftPanel(window)
    window.draft_panel = draft_pane
    draft_area = scrollable(draft_pane)
    window.draft_area = draft_area
    draft_area.setMinimumWidth(200)
    draft_area.setMaximumWidth(280)

    splitter = QSplitter(Qt.Horizontal)
    splitter.setHandleWidth(1)
    splitter.addWidget(draft_area)
    splitter.addWidget(canvas)
    splitter.addWidget(right_side)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 0)
    splitter.setSizes([SIDE_WIDTH, 800, SIDE_WIDTH])

    # The open boards sit directly under the toolbar, above everything the
    # boards are shown in -- the same place a browser puts its tabs, and the
    # reason a switch is one click away from anywhere in the window. It is not
    # a QToolBar: toolbars are user-movable, and a tab row that could be
    # dragged off the top of the window would be a surprise.
    tabs = ProjectTabBar()
    # ``Preferred`` so a short row of tabs takes only the width it needs, and
    # the trailing stretch below absorbs the rest. The minimum is what stops a
    # long row from claiming the whole window and hiding the ＋.
    tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    tabs.setMinimumWidth(120)
    new_tab_button = QToolButton()
    new_tab_button.setObjectName("NewProjectTab")
    new_tab_button.setText("＋")
    new_tab_button.setToolTip("新建板子（在新标签里打开，不影响当前板子）")
    new_tab_button.setCursor(Qt.PointingHandCursor)
    new_tab_button.setFixedSize(26, 24)
    new_tab_button.setFocusPolicy(Qt.NoFocus)
    themed(
        new_tab_button,
        lambda: f"""
                    QToolButton#NewProjectTab {{
                        background: {PANEL_BG};
                        color: {TEXT};
                        border: 1px solid {BORDER};
                        border-radius: 4px;
                        font-size: 14px;
                        padding-bottom: 2px;
                    }}
                    QToolButton#NewProjectTab:hover {{
                        border-color: {ACCENT}; color: {ACCENT};
                    }}
        """,
    )
    tab_row = QWidget()
    tab_row.setObjectName("ProjectTabRow")
    tab_layout = QHBoxLayout(tab_row)
    tab_layout.setContentsMargins(8, 0, 8, 0)
    tab_layout.setSpacing(4)
    tab_layout.addWidget(tabs)
    tab_layout.addWidget(new_tab_button)
    # The stretch comes *after* the ＋, so the two stay together at the left
    # and the empty band sits out of the way on the right. Putting it between
    # them would push the ＋ to the far edge, where it reads as belonging to
    # the window rather than to the tabs.
    tab_layout.addStretch(1)
    themed(
        tab_row,
        lambda: f"""
                    QWidget#ProjectTabRow {{
                        background: {WINDOW_BG};
                        border-bottom: 1px solid {BORDER};
                    }}
        """,
    )

    central = QWidget()
    central_layout = QVBoxLayout(central)
    central_layout.setContentsMargins(0, 0, 0, 0)
    central_layout.setSpacing(0)
    central_layout.addWidget(tab_row)
    central_layout.addWidget(splitter, 1)
    window.setCentralWidget(central)
    window.project_tabs = tabs
    window.new_tab_button = new_tab_button

    # The panes parent to the main window itself, so they float above the
    # whole layout and stay inside the application frame.
    for action in (preprocess_action, toolbar_by_name["图像预处理"]):
        action.triggered.connect(lambda: show_preprocess(window))
    for action in (board_action, toolbar_by_name["板框设置"]):
        action.triggered.connect(lambda: show_board_setup(window))
    # 模板内容 edits the current board's text and cutouts, so it is enabled
    # only while a template is applied -- an empty list of text would be a
    # panel of disabled controls.
    def _open_template_content() -> None:
        open_template_content(window)

    content_action.triggered.connect(_open_template_content)
    # ⚠️ **Always enabled.** The first version disabled this until the board was
    # already a 通行证 card, and the result was a grey menu entry that reads as
    # "this has not been built" -- the user reported exactly that. A control
    # that cannot act yet is better off opening and saying why than sitting
    # there dead, so the panel explains the situation and offers the one action
    # that fixes it.
    content_action.setEnabled(True)
    for action in (layers_action, toolbar_by_name["图层"]):
        action.triggered.connect(lambda: show_layers(window))
    toolbar_by_name["适应窗口"].triggered.connect(canvas.fit_view)
    fit_action.triggered.connect(canvas.fit_view)
    zoom_in_action.triggered.connect(lambda: canvas.zoom_by(1.25))
    zoom_out_action.triggered.connect(lambda: canvas.zoom_by(1 / 1.25))

    # The 编辑 menu and the canvas right-click menu drive the same operations,
    # so both go through the canvas rather than one of them reaching into the
    # board itself. Shortcuts live on the menu actions, which is what makes
    # them work wherever focus happens to be -- the canvas is not the only
    # place a user expects Ctrl+C to work.
    _wire_edit_actions(window, actions_by_name, canvas, on_fit_requested)

    for action in (export_action, export_toolbar_action):
        action.triggered.connect(lambda: export_project(window))
    for action in (gerber_action, gerber_toolbar_action):
        action.triggered.connect(lambda: export_gerber_project(window))

    # The tab row reports clicks; the workspace owns which board that means.
    # Keeping the two apart is what lets the switching be tested without a
    # window and the tab bar be built without knowing what a board is.
    tabs.new_requested.connect(lambda: new_project_tab(window))
    new_tab_button.clicked.connect(lambda: new_project_tab(window))
    new_project_action.triggered.connect(lambda: new_project_tab(window))
    tabs.close_requested.connect(lambda index: close_project_tab(window, index))
    tabs.close_others_requested.connect(
        lambda index: close_other_project_tabs(window, index)
    )
    tabs.current_changed.connect(lambda index: activate_project(window, index))
    tabs.moved.connect(lambda source, target: _move_project_tab(window, source, target))

    # The draft panel and the menu drive the same three operations; the panel
    # is what the user actually uses, the menu entries exist so a keyboard or
    # menu-driven user is not left without them.
    draft_pane.save_requested.connect(lambda: save_draft_clicked(window))
    draft_pane.save_as_requested.connect(lambda: save_draft_as(window))
    draft_pane.open_requested.connect(lambda: open_draft_file(window))
    draft_pane.open_path_requested.connect(
        lambda path: open_draft_file(window, path)
    )
    draft_pane.new_requested.connect(lambda: new_draft(window))
    draft_pane.delete_requested.connect(lambda path: delete_draft(window, path))
    draft_pane.rename_requested.connect(
        lambda path, name: rename_draft(window, path, name)
    )
    save_draft_action.triggered.connect(lambda: save_draft_clicked(window))
    open_draft_action.triggered.connect(lambda: open_draft_file(window))

    # File-level shortcuts, the ones the cheat sheet teaches. They are installed
    # here rather than in the menu table because the entries carry no shortcut
    # yet -- and a cheat sheet naming a key nothing listens to would be the
    # placeholder problem in miniature. ``WindowShortcut`` is what makes them
    # fire wherever focus happens to be, the same rule the 编辑 menu follows.
    for action, shortcut in (
        (new_project_action, "Ctrl+N"),
        (save_draft_action, "Ctrl+S"),
        (export_action, "Ctrl+E"),
        (gerber_action, "Ctrl+Shift+E"),
        (open_draft_action, "Ctrl+O"),
    ):
        action.setShortcut(QKeySequence(shortcut))
        action.setShortcutContext(Qt.WindowShortcut)
    # Ctrl+Shift+S has to live on its own shortcut: one key sequence per
    # action, and the draft panel's button is the shared implementation anyway.
    save_as_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), window)
    save_as_shortcut.setContext(Qt.WindowShortcut)
    save_as_shortcut.activated.connect(lambda: save_draft_as(window))

    quit_action.triggered.connect(window.close)
    open_image_action.triggered.connect(lambda: show_preprocess(window).drop_zone.browse())
    toolbar_by_name["打开图片"].triggered.connect(
        lambda: show_preprocess(window).drop_zone.browse()
    )
    about_action.triggered.connect(lambda: show_about(window))
    shortcuts_action.triggered.connect(lambda: show_shortcuts(window))
    settings_action.triggered.connect(lambda: show_settings(window))

    # The settings panel searches up the parent chain for this, so it is bound
    # per window rather than being a module function: the panel must switch the
    # window it belongs to, and a second window would otherwise recolour the
    # wrong one.
    window.apply_colour_scheme = lambda scheme: apply_colour_scheme(window, scheme)

    status = window.statusBar()
    themed(status, lambda: f"background: {PANEL_BG}; color: {MUTED};")
    status.showMessage("就绪")

    # The draft starts unnamed and empty, so the first save has nothing to
    # write yet. The fingerprint is taken now, which makes "no edits yet"
    # mean "identical to a fresh document" rather than "never saved".
    window._draft_path = None
    window._draft_saved_fingerprint = fingerprint(canvas.board)
    refresh_project_tabs(window)

    # The dirty flag is re-derived on a timer instead of being pushed by every
    # edit path. There are dozens of those, spread across the canvas and three
    # panels, and one missed signal would mean the exit prompt never appears --
    # the failure mode being a silently lost design, not a cosmetic glitch.
    # A timer cannot miss an edit because it does not depend on being told.
    draft_timer = QTimer(window)
    draft_timer.setInterval(500)
    draft_timer.timeout.connect(lambda: refresh_workspace_state(window))
    draft_timer.start()
    window._draft_timer = draft_timer
    refresh_workspace_state(window)

    # The theme sheet is applied to the *window* rather than to the
    # application, because an application-level sheet forces Qt back to its
    # built-in style and throws the custom proxy away -- see
    # ``tools.qt_theme.apply_scheme``. A window built after startup therefore
    # has to be given the sheet here; one built before it is covered by
    # ``apply_scheme`` walking the top-level widgets.
    style_window(window, current_scheme())

    return window


def apply_colour_scheme(window, scheme):
    """Recolour the running window in place. Returns the same window.

    **Nothing is rebuilt, closed or restarted.** The window keeps every open
    board, every panel that is open stays open, and the selection, scroll
    position and zoom are exactly where they were -- the user is looking at
    these while they drag a colour around, and a switch that recreated any of
    them would move the thing they are looking at.

    Three things have to happen for the colours to actually change:

    1. **The module palette is re-bound** (:func:`set_palette`), so anything
       built from here on is already correct.
    2. **The application style is replaced** (``qt_apply_scheme``): Qt's
       stylesheet, palette and proxy style are all global, so this repaints
       every standard control at once.
    3. **Every registered per-widget builder is re-run** (:func:`restyle`).
       This is the step the earlier rebuild existed to avoid: a widget's
       stylesheet is a string, so a colour interpolated into it at construction
       time is frozen. Widgets now register *how* to build their stylesheet
       (see :func:`themed`) and this walks the tree asking each one again.

    ⚠️ **Icons need their own pass.** A :class:`QIcon` has the colour baked
    into its pixmaps, so re-running stylesheets does not touch it; the toolbar
    and menu icons are rebuilt here from the new palette.
    """
    from PySide6.QtWidgets import QApplication

    set_palette(scheme)
    app = QApplication.instance()
    if app is not None:
        qt_apply_scheme(app, scheme, font_point_size=app.font().pointSizeF())

    # Merged rather than replaced: the file also holds the external redraw-model
    # path, which this path does not own and must not erase.
    save_settings_keeping(settings_for(scheme))

    # Per-widget stylesheets first, then the icons: both read the palette
    # rebound above, and doing them in this order means a widget that shows an
    # icon and has its own stylesheet is fully correct after one pass.
    restyle(window)
    _restyle_icons(window)

    # A floating panel may itself carry themed styles and icons; it is a child
    # of the window, so `restyle(window)` already covered it, but its own
    # colour-scheme bookkeeping has to be told which scheme is now in force.
    #
    panel = live_panel("_settings_panel")
    if panel is not None:
        try:
            if panel.parentWidget() is window:
                panel.set_scheme(scheme)
        except RuntimeError:
            # The panel died with an earlier window; nothing to update.
            pass

    window.update()
    return window


def _restyle_icons(window) -> int:
    """Rebuild every themed icon in the window from the current palette.

    Icons are rasterised pixmaps, so unlike a stylesheet they cannot simply be
    asked to repaint -- the new colour has to be drawn into new pixmaps. Each
    entry below is the same mapping the toolbar and menu were built with, so
    this is a re-application of that mapping rather than a second definition.

    Returns how many icons were replaced, which the tests assert on: an icon
    pass that silently found nothing would leave a dark theme with pale icons
    and no test failure.
    """
    mapping = {
        "打开图片": ("open_image", "text"),
        "图像预处理": ("preprocess", "text"),
        "板框设置": ("board_setup", "text"),
        "图层": ("layers", "text"),
        "适应窗口": ("fit_window", "text"),
        "另存为工程": ("save_project", "text"),
        "另存为下单文件": ("export_gerber", "text"),
    }
    palette = {
        "text": TEXT,
        "muted": MUTED,
        "accent": ACCENT,
        "danger": DANGER,
    }

    changed = 0
    toolbar = window.findChild(QToolBar)
    if toolbar is not None:
        for action in toolbar.actions():
            entry = mapping.get(action.text())
            if entry is None:
                continue
            name, tone_name = entry
            action.setIcon(qt_theme_icon(name, palette[tone_name], size=18))
            changed += 1
    return changed


def refresh_workspace_state(window) -> None:
    """One timer tick: the active board's dirty dot, and the list beside it.

    Both are re-derived from the documents rather than pushed by every edit
    path, for the same reason the dirty flag always was: a missed signal would
    mean a silently lost design, and the two tabs that need a dot are exactly
    the ones the user is not looking at.
    """
    canvas = getattr(window, "design_canvas", None)
    if canvas is not None and hasattr(window, "_workspace"):
        window.active_project.board = canvas.board
    refresh_draft_panel(window)
    refresh_project_tabs(window)


def _move_project_tab(window, source: int, target: int) -> None:
    """A tab was dragged. The document order changes, nothing else does."""
    workspace = window.workspace
    try:
        workspace.move(source, target)
    except IndexError:
        return
    refresh_project_tabs(window)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screenshot",
        help="render the window to this PNG and exit (layout check)",
    )
    parser.add_argument(
        "--open-preprocess",
        action="store_true",
        help="open the floating preprocessing panel on startup",
    )
    parser.add_argument(
        "--open-layers",
        action="store_true",
        help="open the floating layer panel on startup",
    )
    parser.add_argument(
        "--open-board",
        action="store_true",
        help="open the floating board-setup panel on startup",
    )
    parser.add_argument(
        "--image",
        help=(
            "load this image at startup and import it, so screenshots show the "
            "picture on the canvas"
        ),
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="with --image, stop after loading so the pre-import state is shown",
    )
    parser.add_argument(
        "--board-shape",
        choices=list(SHAPE_CHOICES),
        help="preselect a board shape for the board panel",
    )
    parser.add_argument(
        "--board-base",
        choices=list(IRREGULAR_BASES),
        help="preselect the irregular base (implies 异形板)",
    )
    parser.add_argument(
        "--standee",
        action="store_true",
        help="with --open-board, tick 立牌 so the screenshot shows the tab",
    )
    parser.add_argument(
        "--standee-width",
        type=float,
        help="with --standee, tab width in mm",
    )
    parser.add_argument(
        "--standee-height",
        type=float,
        help="with --standee, tab height in mm",
    )
    parser.add_argument(
        "--standee-offset",
        type=int,
        help="with --standee, 水平位置 slider value, -100 to 100",
    )
    parser.add_argument(
        "--gerber-to",
        metavar="DIR",
        help=(
            "write the Gerber set for the current design into DIR and exit; "
            "with --image and --board-shape this is a headless conversion, "
            "and it is how a packaged build's export path gets verified"
        ),
    )
    parser.add_argument(
        "--draft-to",
        metavar="FILE",
        help="save a draft to FILE and exit (headless check of the draft path)",
    )
    parser.add_argument("--width", type=int, default=1420)
    parser.add_argument("--height", type=int, default=900)
    # Named --art-style, not --style. Qt reads a bare ``-style`` from the
    # command line as its own widget-style override, so ``--style 板画`` was
    # consumed by Qt before argparse ever saw it and the application printed
    # "invalid style override" on every screenshot run.
    parser.add_argument(
        "--art-style",
        choices=[name for name, _ in STYLE_OPTIONS],
        help="preselect an artwork style, to review that state",
    )
    parser.add_argument(
        "--board-art-side",
        choices=[key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS],
        help="with --art-style 板画, preselect 顶面/底面/两面",
    )
    parser.add_argument("--flip-vertical", action="store_true",
                        help="with --art-style 板画, tick 上下翻转")
    parser.add_argument("--flip-horizontal", action="store_true",
                        help="with --art-style 板画, tick 左右翻转")
    parser.add_argument("--tone", type=int,
                        help="with --art-style 板画, set 明暗分界 in hundredths (0-100)")
    parser.add_argument("--outline-mm", type=int,
                        help="with --art-style 板画, set 轮廓宽度 in hundredths of a mm")
    parser.add_argument(
        "--side",
        choices=["top", "bottom"],
        default="top",
        help="preselect the board side",
    )
    parser.add_argument(
        "--collapsed",
        action="store_true",
        help="start with the preprocessing panel minimised to its title bar",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei UI", 9)

    # ⚠️ **The theme has to be applied here, at startup.** It used to be
    # applied only when the user *changed* a colour, so a freshly launched
    # application ran on Windows' native controls -- a blue check box, a blue
    # slider, square toolbar buttons. The saved scheme was read, but never
    # handed to Qt, so the MD3 styling (which is where the check box and slider
    # take their shapes and colours from) only appeared after visiting the
    # settings panel once.
    #
    # ``apply_colour_scheme`` is deliberately not used here: it works by
    # re-running every widget's style builder, and at this point no widgets
    # exist yet. Startup wants the one-shot version, which is ``qt_apply_scheme``.
    app.setFont(font)
    qt_apply_scheme(app, current_scheme(), font_point_size=font.pointSizeF())

    window = build_window()
    window.resize(args.width, args.height)

    panel: PreprocessPanel | None = None
    if (
        args.open_preprocess
        or args.art_style
        or args.side != "top"
        or args.collapsed
        or args.image
    ):
        panel = show_preprocess(window)
        if args.art_style:
            names = [name for name, _ in STYLE_OPTIONS]
            panel.style_combo.setCurrentIndex(names.index(args.art_style))
        if args.side == "bottom":
            panel.side_control.buttons[1].setChecked(True)  # type: ignore[attr-defined]
            panel._refresh_layers()
        if args.board_art_side:
            side_keys = [key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS]
            panel.board_art_side_combo.setCurrentIndex(
                side_keys.index(args.board_art_side)
            )
        if args.flip_vertical:
            panel.flip_vertical_check.setChecked(True)
        if args.flip_horizontal:
            panel.flip_horizontal_check.setChecked(True)
        if args.tone is not None:
            panel.tone_slider.setValue(args.tone)
        if args.outline_mm is not None:
            panel.outline_mm_slider.setValue(args.outline_mm)
        if args.collapsed:
            panel.collapse()
        panel.place_top_right()

    if args.image and panel is not None:
        panel.load_image(args.image)
        # Screenshots that pass --image are reviewing the canvas, so the import
        # step is pressed for them. --no-import shows the new pre-import state.
        if not args.no_import:
            panel.import_to_canvas()

    if args.open_board or args.board_shape or args.standee:
        board_panel = show_board_setup(window)
        if board_panel is not None and args.board_shape:
            board_panel.shape_combo.setCurrentIndex(
                SHAPE_CHOICES.index(args.board_shape)
            )
        if board_panel is not None and args.board_base:
            board_panel.base_combo.setCurrentIndex(
                IRREGULAR_BASES.index(args.board_base)
            )
        if board_panel is not None and args.standee:
            board_panel.standee_check.setChecked(True)
            if args.standee_width is not None:
                board_panel.standee_width_spin.setValue(args.standee_width)
            if args.standee_height is not None:
                board_panel.standee_height_spin.setValue(args.standee_height)
            if args.standee_offset is not None:
                board_panel.standee_x_slider.setValue(args.standee_offset)
        if board_panel is not None:
            board_panel.place_top_right()
            board_panel.raise_()

    if args.open_layers:
        # The layer list is docked, so revealing it means clearing the
        # selection rather than opening a floating panel.
        show_layers(window)

    if args.gerber_to or args.draft_to:
        # Headless checks of the two export paths. They exist so a *packaged*
        # build can be verified without a person clicking through the UI --
        # "it starts" is not evidence that the writer inside it works.
        code = 0
        if args.draft_to:
            path = save_draft(window.design_canvas.board, args.draft_to)
            print(f"wrote draft {path}")
        if args.gerber_to:
            from pcbart.export import default_title
            from pcbart.gerber import DEFAULT_PREFIX, export_gerber

            report = export_gerber(
                window.design_canvas.board,
                args.gerber_to,
                prefix=DEFAULT_PREFIX,
                title=default_title(window.design_canvas.board),
            )
            for entry in report["files"]:
                print(f"wrote {entry['name']} ({entry['bytes']} bytes)")
            print(f"archive {Path(str(report['archive'])).name}")
        return code

    if args.screenshot:
        # Render with the real platform plugin so the system fonts are present:
        # Qt's offscreen plugin ships an empty font database, which turns every
        # character into a tofu box. WA_DontShowOnScreen lays the window out and
        # paints it without ever putting it on the user's desktop.
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        app.processEvents()
        window.grab().save(args.screenshot)
        print(f"wrote {args.screenshot}")
        return 0

    window.show()
    for floating in (panel, _board_panel):
        if floating is not None and floating.parentWidget() is window:
            floating.place_top_right()
            floating.raise_()

    # Check for a newer release, off the UI thread and only after the window is
    # up. Delayed rather than immediate so a slow check cannot delay the first
    # paint, and so the update panel does not fight the startup panels for the
    # same corner of the screen. Skipped entirely for screenshot runs, which
    # must be deterministic and must not reach the network.
    if _auto_check_enabled_setting():
        QTimer.singleShot(
            5000, lambda: start_update_check(window, manual=False)
        )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
