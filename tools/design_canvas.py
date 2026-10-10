"""The design canvas: board outline plus the placed image, with editing.

Replaces the earlier stand-in canvas, which drew a fixed circle. Geometry now
comes from :class:`pcbart.artboard.ArtBoard`, so the board shape and the image
placement are whatever the user configured rather than decoration.

The surface is black and the X/Y axes cross it at the origin, the way a CAD
editor presents a sheet. The artwork is painted in its layer's colour, and
those colours are chosen by EasyEDA to read on a dark board view -- silkscreen
yellow and copper red all but disappear against white. The panels around the
canvas keep the application's light palette; only the board view is dark.

Editing follows the editor conventions the user asked for:

* click the image to select it, which shows a slate panel behind it and four
  triangular handles on the edges;
* drag inside the image to move it (with a short press-and-hold, so a passing
  click does not shift it);
* drag a handle to resize from that edge.

The keyboard and right-click entries follow the same conventions an EDA tool
uses -- copy, paste, cut, delete, duplicate, nudge with the arrow keys, rotate
with ``R``. Only operations this tool actually has are bound rather than
pretending, and ``Ctrl+Z`` is real: the canvas keeps a document history (see
:mod:`pcbart.undo`) and every edit route -- drag, keyboard, context menu, the
Edit menu, the property pane -- records into it.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from pcbart.artboard import ArtBoard, ImagePlacement, drawn_size_mm
from pcbart.design_layers import layer
from pcbart.image_size import UnreadablePicture, picture_size
from pcbart.undo import UndoStack

#: Resolved font families, keyed by the file the exporter would use.
#:
#: ``QFontDatabase.addApplicationFont`` loads the file once and returns an id;
#: resolving it on every repaint would re-read the file. The value is the family
#: Qt assigned, which is what a QFont has to be built from.
_LOADED_FONT_FAMILIES: dict[str, str] = {}


def template_font(family: str, text: str) -> QFont:
    """A ``QFont`` that draws ``text`` with the face the exporter will use.

    **The canvas and the exporters have to agree on the font, not just the
    anchor.** The page used to draw every template line with Qt's default face
    while the writers rendered the family the template asks for. When those
    differ, the *same* anchor produces different ink: measured on the card,
    ``阵营logo`` came out 0.93 mm wider and 0.76 mm taller on screen than in the
    file, and ``©HYPERGRYPH`` 2.01 mm wider. That is what "the preview looks
    offset" was -- nothing was moving, the on-screen glyphs were a different
    size from the exported ones.

    ``pcbart.text_render.font_path`` is reused rather than reimplemented: it
    already knows which file each family maps to, and -- importantly -- that a
    family which cannot draw the line's characters must be passed over. The
    anchor metrics then match exactly, because Qt's ``setPixelSize`` and PIL's
    ``truetype(size)`` use the same em box.
    """
    font = QFont()
    if not family and not text:
        return font

    from pcbart.text_render import font_path

    path = font_path(family, text)
    if path is None:
        return font

    key = str(path)
    resolved = _LOADED_FONT_FAMILIES.get(key)
    if resolved is None:
        font_id = QFontDatabase.addApplicationFont(key)
        families = (
            QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        )
        resolved = families[0] if families else ""
        _LOADED_FONT_FAMILIES[key] = resolved
    if resolved:
        # ⚠️ **The family name is not enough on its own.** Qt resolves a family
        # name against every face registered in the process, and a name as
        # generic as ``Franklin Gothic`` can be served by a *different* file
        # than the one ``font_path`` picked -- measured at 7.997 mm from Qt
        # against 6.781 mm from FRAMDCN.TTF for the same string, which is
        # exactly the kind of mismatch between screen and file this whole
        # exercise is about. ``setFamilies`` lists the file's own family first
        # and Qt then pins the face to that registration.
        font.setFamilies([resolved])
    return font

HANDLE_SIZE = 11.0
HANDLE_HIT = 13.0

#: How far a pasted or duplicated copy lands from its original, in mm. Enough
#: to see that there are now two, small enough that the copy is still where the
#: user was looking.
DUPLICATE_OFFSET_MM = 3.0

#: Arrow-key nudge, and the finer step when Shift is held. The coarse step is
#: millimetre-friendly because the document is in millimetres; the fine step is
#: for lining a picture up against a board edge.
NUDGE_MM = 1.0
NUDGE_FINE_MM = 0.1

#: A keyboard rotation step. Quarter turns are what a picture placed by eye
#: needs; arbitrary angles are the property panel's job.
ROTATE_STEP_DEG = 90.0

#: Run keys that tell the history to merge consecutive edits into one step. A
#: drag emits a motion event per frame and a held arrow key repeats on a timer;
#: without merging, one Ctrl+Z would walk back a pixel at a time.
_DRAG_RUN = "drag"
_NUDGE_RUN = "nudge"
_FRAME_RUN = "frame"
"""Panning a picture inside its window is one gesture, like a drag."""

# Painting colours. The drawing surface is black, the way an EDA tool shows a
# board: the artwork is painted in its *layer's* colour, and the silkscreen
# yellow and the board painting's pale tones are washed out against a light
# background. The panels around the canvas keep the application's light palette
# -- this is the board view, not the whole window.
CANVAS_BG = "#000000"
"""The drawing surface, black at the user's request."""

GRID_MINOR = "#1d232a"
GRID_MAJOR = "#2b343d"
AXIS = QColor(196, 208, 220, 190)
"""The coordinate axes and the arrows at their positive ends.

Brighter than any grid line, because the one thing the grid cannot say is which
line is zero. Neutral rather than tinted: red, green, blue and magenta are all
taken by layers, and an axis in one of them would read as a layer of the board.

The axes are the one piece of chrome painted **over** the artwork rather than
under it. An imported picture is centred on the origin, so underneath it the
origin marker and the middle of both axes are hidden by any opaque image --
which is exactly the case the axes exist for. They are drawn semi-transparent
so they read as an overlay on a ruler, not as ink on the board.
"""

_AXIS_HALO = QColor(0, 0, 0, 210)
"""Outline behind every axis stroke and label.

The axes cross whatever the artwork happens to be, so no single colour can be
legible: light grey vanishes on white paper, dark grey vanishes on black ink.
An outline of the opposite shade underneath keeps each stroke readable on both.
"""

RULER_BG = "#0c1013"
BORDER = "#2b333c"
MUTED = "#8b96a3"
"""Hints and ruler text: light enough to read on black."""

PLACEHOLDER_FILL = "#e8ecf0"
"""Stand-in block for a placement whose picture cannot be decoded."""

SELECTION_FILL = QColor(74, 84, 97, 150)
"""The plate behind the selected object. A dark slate lift off the black --
the point is to show the object's extent, not to brighten the canvas."""

SELECTION_EDGE = QColor(214, 223, 233)

FRAME_EDGE = "#ffb300"
"""The framing outline: a distinct amber, so "this is being framed" cannot be
mistaken for "this is selected"."""

FRAME_GHOST_OPACITY = 0.28
"""How strongly the part of a picture outside the framing window is shown.

While the user is choosing which part of an illustration goes in the region,
the rest of the picture has to stay visible enough to judge a drag against --
but faint enough that the window itself still reads as the print. Draws below
1.0 on purpose: at full strength the crop would be indistinguishable from the
part that is actually kept."""

CUTOUT_EDGE = "#ff5252"
"""The 挖槽 outline. Red because a hole is the one thing on the canvas that has
to be impossible to mistake for artwork, and it matches how the editor draws a
cut path."""

CUTOUT_FILL = QColor(0, 0, 0, 150)
"""Inside the cutout, darker than the canvas: the board is gone there.

A hole cannot be shown by an outline alone on a black canvas -- the board
around it is already near-black -- so the opening is painted as a genuinely
darker patch, which is what "nothing behind the card" looks like."""

TOKEN_HIGHLIGHT = QColor(255, 179, 0, 70)
"""Wash behind a text line that still holds an unresolved ``{代号}`` token.

Only while no operator is chosen. Once one is, the tokens are gone and the wash
would be reporting a condition that no longer exists."""


def _clamp_window(window: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Keep a window inside the picture, scaling it down *as a whole* if needed.

    A window that ran off the edge would show blank space where the picture
    ends; a window wider than the picture has nothing to slide against.

    ⚠️ **The two axes are clamped together, not independently.** Clamping each
    one on its own squashes the rectangle: zooming out until the height hit the
    picture's edge while the width kept growing took a 精0 crop from aspect
    0.690 to 1.000 -- and nothing ever brought it back, because every later
    zoom scaled the *square* window. The visible symptom was the one the user
    reported: the figure drawn into a tall opening from a square window shrinks
    to a band with blank space above and below, and no amount of zooming out
    fills it again.

    Scaling both axes by one factor keeps the window's shape, which is the
    property the region crop depends on.
    """
    x0, y0, x1, y1 = window
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    # One factor for both axes, so the shape survives the clamp.
    shrink = min(1.0, 1.0 / max(width, height, 1e-9))
    width *= shrink
    height *= shrink
    x0 = min(max(0.0, min(x0, x1)), 1.0 - width)
    y0 = min(max(0.0, min(y0, y1)), 1.0 - height)
    return (x0, y0, x0 + width, y0 + height)


def _zoom_window(
    window: tuple[float, float, float, float], factor: float
) -> tuple[float, float, float, float]:
    """Scale a window about its own centre."""
    x0, y0, x1, y1 = window
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = abs(x1 - x0) / 2.0 * factor
    half_h = abs(y1 - y0) / 2.0 * factor
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _window_of_aspect(
    window: tuple[float, float, float, float] | None,
    aspect: float | None,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """A window inside a picture, reshaped to ``aspect`` about its centre.

    ``aspect`` is the *window's* width divided by its height, which is what the
    region needs; the picture's own pixel aspect is taken out first, because a
    window is a fraction of the picture and a fraction of a non-square picture
    is not the shape it looks like.
    """
    width_px, height_px = image_size
    if not width_px or not height_px:
        return window or (0.0, 0.0, 1.0, 1.0)

    if window is None:
        window = (0.0, 0.0, 1.0, 1.0)
    x0, y0, x1, y1 = window
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width = abs(x1 - x0)
    height = abs(y1 - y0)

    if aspect is not None and aspect > 0:
        # The largest window of the wanted shape that fits inside the current
        # one, so the picture is cropped rather than sampled outside itself.
        # Target shape in fractions: a fraction of a non-square picture is not
        # the shape it looks like, hence the pixel aspect.
        want = aspect * height_px / width_px
        if width / max(height, 1e-9) > want:
            width = height * want          # too wide: take from the sides
        else:
            height = width / want          # too tall: take from the ends

    return _clamp_window((cx - width / 2.0, cy - height / 2.0,
                          cx + width / 2.0, cy + height / 2.0))


def _full_image_rect(
    local: QRectF, source: QRectF, width_px: int, height_px: int
) -> QRectF | None:
    """Where the *whole* picture would land when ``source`` fills ``local``.

    The framing view draws the window normally and the rest of the picture as
    a faint ghost, so the user can see what a drag is moving against. The ghost
    has to sit in the same frame as the window: extrapolating from the window's
    own rect keeps the two aligned by construction, instead of computing the
    picture's position a second way and hoping the two agree.

    ``None`` when the window already covers the whole picture -- there is no
    outside to draw, and a second full-strength copy would just be a repaint.
    """
    if width_px <= 0 or height_px <= 0:
        return None
    if (
        source.left() <= 0
        and source.top() <= 0
        and source.right() >= width_px
        and source.bottom() >= height_px
    ):
        return None

    full_w = local.width() * width_px / max(1.0, source.width())
    full_h = local.height() * height_px / max(1.0, source.height())
    return QRectF(
        local.left() - source.left() / width_px * full_w,
        local.top() - source.top() / height_px * full_h,
        full_w,
        full_h,
    )


class DesignCanvas(QWidget):
    """Editable view of the board and the artwork on it."""

    selection_changed = Signal(object)
    """Emitted with the placement when it is selected, or ``None`` on deselect."""

    placement_changed = Signal()
    """Emitted after a move or resize so the inspector can refresh."""

    document_changed = Signal()
    """Emitted when artwork is added or removed -- a paste, a delete, a
    duplicate.

    Separate from :attr:`placement_changed` on purpose: that one says "the
    selected object's numbers moved", this one says "the set of objects changed
    and anything listing them is now out of date". A delete has no placement
    left to report, so it cannot use the first.
    """

    clipboard_changed = Signal()
    """Emitted when the clipboard gains or loses content.

    Needed on its own because copying changes no selection and no document --
    but it is what decides whether 粘贴 is available, and a menu that stays
    greyed out after a successful copy looks like the copy failed.
    """

    history_changed = Signal()
    """Emitted when undo becomes available or stops being available.

    The 撤销 menu entry is greyed out when there is nothing to go back to, and
    that state changes on every edit and on every undo -- neither of which is
    visible to the action that shows it.
    """

    fit_to_board_requested = Signal()
    """Ask the window to run 按板框缩放, which owns the board-level logic."""

    reset_view_requested = Signal()
    """Ask the window to run 适应窗口."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.board = ArtBoard()
        self.selected: ImagePlacement | None = None
        """The selected placement, or ``None``. Kept as the object itself
        rather than a flag, because with several objects on different layers
        "selected" has to name which one."""

        self.extra_selection: list = []
        """The *other* selected objects, for grouping them together.

        ⚠️ **A separate list rather than turning ``selected`` into a list.**
        ``selected`` is read in three dozen places -- the property panel, the
        handles, every move and resize, the Delete key -- and all of them mean
        "the one object being edited". Making it a collection would have
        rewritten all of them for a feature that only matters while assembling
        a group.

        So the multi-selection is additive state: :attr:`selected` is still the
        primary object the handles belong to, and this holds whatever else the
        user Ctrl-clicked. Every existing code path keeps working untouched,
        and the three operations that care about the set -- 组合, 取消组合 and
        组合管理 -- read :meth:`selection_set`.

        It holds placements *and* text lines, because the user asked to choose
        which elements go in and the card's wording is an element like any
        other.
        """

        self._selection_anchor = None
        """The object a Shift-click extends from, so a range can be picked."""

        self._press_centres: dict[int, tuple[float, float]] = {}
        """Where each dragged object was when the drag began.

        A dict keyed by ``id`` rather than one ``_press_centre``, because a drag
        moves the whole multi-selection and a group at once. Each target is
        placed at *its own* start plus the total pointer delta, so the objects
        keep their relative offsets and the drag cannot accumulate rounding
        error the way a per-event nudge would.
        """

        self._pixmaps: dict[str, QPixmap] = {}
        """Decoded pictures, keyed by path.

        Several placements can share one source file -- the same drawing on
        copper and on silkscreen -- so decoding per path avoids a second read
        and keeps the two in sync.
        """

        self._tinted_cache: dict[str, QPixmap] = {}
        """Layer-tinted copies, keyed by pixmap and colour.

        Tinting is a full-size image operation and the layers barely change, so
        the result is kept rather than recomputed on every repaint.
        """

        self._mode = ""
        self._press_pos = QPointF()
        self._press_mm = (0.0, 0.0)
        self._press_size = (0.0, 0.0)
        self._press_centre = (0.0, 0.0)
        self._handle = ""
        self._press_window = None
        """The window at the start of a framing pan, so the drag is absolute
        rather than accumulating rounding error frame by frame."""
        self._hold_timer = None
        self._pending_move = False

        # View transform. The zoom is applied on top of the fitted scale, so
        # "fit" stays meaningful and can be returned to in one step.
        self._zoom = 1.0
        self._pan_mm = [0.0, 0.0]
        """View centre offset in millimetres. Moving the view must never move
        the geometry, so panning changes only this."""

        self._space_held = False
        self._pan_anchor = None

        self._clipboard: list[ImagePlacement] = []

        self.framing = None
        """The placement whose *contents* are being framed, or ``None``.

        Framing is a different gesture from placing, and the two are easy to
        confuse on screen, so it is an explicit mode rather than something
        inferred from where the pointer is: with a region to fill, dragging
        **pans the picture inside its window** and the object itself stays put.
        Outside this mode the same drag moves the object, which is what every
        other placement does.
        """

        self.framing_region = None
        """The region being filled, cached for the overlay and the aspect lock."""
        """Copied artwork, in placement order, or empty.

        Holds *copies* rather than references: the user may delete or edit the
        original after copying it, and what gets pasted has to be what was on
        the clipboard at the moment of the copy.
        """

        self.history = UndoStack(self.board)
        """This board's edit history. Swapped for the incoming board's own stack
        when the window switches tabs, so ``Ctrl+Z`` never reaches across into
        a document the user is not looking at.
        """

        # Visible geometry, recomputed on every paint.
        self._plot = QRectF()
        self._px_per_mm = 1.0
        self._fit_px_per_mm = 1.0
        self._fit_taken = False
        """Whether a scale has been fitted at all yet. A canvas that has never
        been fitted takes the fit on its first paint, so it opens with the
        content framed; every later fit is asked for explicitly."""
        self._frame_outline: list[tuple[float, float]] = []
        """The board outline for the frame being painted, computed once."""

        self._decoration_cache: dict = {}
        """Mapped screen geometry for the card's fixed decorations, per view.

        Keyed by layer, and dropped wholesale when :attr:`_view_key` changes --
        moving an object reuses it, zooming or panning drops it. See
        :meth:`_decoration_screens`.
        """

        # View arithmetic, refreshed by :meth:`_refresh_view_cache`.
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._scale = 1.0
        self._view_key = None
        """The transform those five numbers describe, for cache invalidation."""
        self._decoration_view = None
        """Which :attr:`_view_key` the decoration cache was built for."""

        # Sizing comes last: it can deliver a resize event, and that event
        # takes the fit, which needs the view state above to exist already.
        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------ geometry

    def set_board(self, board) -> None:
        self.board = board
        self.fit_view()
        self.update()

    def set_placement(self, placement, *, append: bool = True) -> None:
        """Adopt artwork created elsewhere (for example by the upload).

        Appends by default: importing a second picture should add it, not
        replace what is already on the board. Pass ``append=False`` to replace
        the first placement instead.
        """
        if append:
            self.board.add_placement(placement)
        else:
            self.board.image = placement
        self._ensure_pixmap(placement.path)
        self.update()

    def _ensure_pixmap(self, path: str) -> QPixmap | None:
        """Decode a picture once and remember it.

        The source is preferred over the processed line art for the pixmap
        cache because the processed result is tinted at paint time; caching the
        tinted version would bake one layer's colour into the image.
        """
        if not path:
            return None
        if path in self._pixmaps:
            return self._pixmaps[path]
        pixmap = QPixmap(path)
        self._pixmaps[path] = pixmap
        return pixmap if not pixmap.isNull() else None

    def _pixmap_for(self, placement) -> QPixmap | None:
        """What to draw for a placement: the line art if it exists.

        A processed placement is shown as flat black-and-white and tinted in
        its layer's colour, the way the editor shows a layer. Falling back to
        the source picture keeps an unprocessed import visible.
        """
        path = placement.processed_path or placement.path
        pixmap = self._ensure_pixmap(path)
        if pixmap is None or pixmap.isNull():
            return None
        return pixmap

    # ------------------------------------------------------------ transform

    def _recompute_plot(self) -> None:
        """Refresh the viewport rect and the scale the view is drawn at.

        Deliberately *not* a re-fit. The fitted scale used to be recomputed on
        every paint from whatever was on the board, which made dragging a
        handle fight the cursor: each step resized the picture, which changed
        what "fitted" meant, which rescaled the view under the pointer, so the
        edge never caught up with the mouse. The fit is a view decision now --
        only :meth:`fit_view`, :meth:`refit_view` (a resized window, a board
        the setup panel changed) and a document switch move it.

        The user's zoom and pan are applied on top of the fitted scale, so
        "fit" stays a meaningful thing to return to.
        """
        self._plot = self._plot_rect()
        if not self._fit_taken:
            self._fit_px_per_mm = self._fit_scale(self._plot)
            self._fit_taken = True
        self._px_per_mm = self._fit_px_per_mm * self._zoom
        self._refresh_view_cache()

    def _plot_rect(self) -> QRectF:
        """The drawing area, inside the rulers and their margin."""
        ruler = 22
        margin = 26
        return QRectF(
            ruler + margin,
            ruler + margin,
            max(10.0, self.width() - ruler - 2 * margin),
            max(10.0, self.height() - ruler - 2 * margin),
        )

    def _framing_points(self) -> list[tuple[float, float]]:
        """The geometry the view frames: the board, or the artwork without one.

        A board drives the fit once it exists. Before that the artwork does,
        because an imported picture on a boardless canvas has to be visible and
        editable on its own -- there is nothing else to frame.
        """
        return self.board.outline_polygon() or self._image_bounds_mm()

    def _fit_scale(self, plot: QRectF) -> float:
        """Pixels per millimetre that put the framed geometry inside ``plot``."""
        points = self._framing_points()
        if not points:
            # Nothing to frame yet: the canvas is empty because the picture is
            # waiting to be imported. Falling through to 1 px/mm would draw a
            # grid spanning hundreds of millimetres, which reads as a bug
            # rather than as an empty canvas. A blank board's scale is chosen
            # so a typical board's worth of the grid fills the viewport.
            default_span_mm = 100.0
            return min(plot.width(), plot.height()) / default_span_mm

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # A zero span (a single point, or an axis-aligned degenerate shape) has
        # no scale of its own; the other axis still does.
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        return min(plot.width() / span_x, plot.height() / span_y)

    def _image_bounds_mm(self) -> list[tuple[float, float]]:
        """Corners of every visible placement, rotation included."""
        points: list[tuple[float, float]] = []
        for placement in self.board.visible_placements():
            points.extend(placement.corners())
        return points

    # ---------------------------------------------------------- view control

    MIN_ZOOM = 0.05
    MAX_ZOOM = 40.0

    def fit_view(self) -> None:
        """Reset zoom and pan, so the board fills the viewport again."""
        self._zoom = 1.0
        self._pan_mm[0] = 0.0
        self._pan_mm[1] = 0.0
        self.refit_view()

    def refit_view(self) -> None:
        """Re-frame the content, keeping the user's zoom and pan.

        Called when the *window onto* the document changes rather than the
        document itself: the widget was resized, or the board -- the thing the
        view frames -- was edited in the setup panel. Artwork edits do not call
        it, which is what keeps a handle drag anchored to the cursor.
        """
        self._plot = self._plot_rect()
        self._fit_px_per_mm = self._fit_scale(self._plot)
        self._fit_taken = True
        self._px_per_mm = self._fit_px_per_mm * self._zoom
        self._refresh_view_cache()
        self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom about a point, keeping whatever is under it in place.

        Zooming about the cursor is what every CAD tool does; zooming about
        the centre instead makes the user chase the feature they were looking
        at. The anchor is honoured by shifting the pan so the millimetre
        coordinate under it does not move.
        """
        before_zoom = self._zoom
        after = max(self.MIN_ZOOM, min(self.MAX_ZOOM, before_zoom * factor))
        if abs(after - before_zoom) < 1e-9:
            return

        if anchor is not None:
            before_mm = self.to_mm(anchor)
            self._zoom = after
            self._px_per_mm = self._fit_px_per_mm * self._zoom
            self._refresh_view_cache()
            after_mm = self.to_mm(anchor)
            self._pan_mm[0] += before_mm[0] - after_mm[0]
            self._pan_mm[1] += before_mm[1] - after_mm[1]
        else:
            self._zoom = after
            self._px_per_mm = self._fit_px_per_mm * self._zoom
        self._refresh_view_cache()
        self.update()

    def pan_by_pixels(self, dx: float, dy: float) -> None:
        """Move the view by a screen-space delta.

        The geometry is untouched; only the window onto it moves.
        """
        scale = max(self._px_per_mm, 1e-9)
        self._pan_mm[0] -= dx / scale
        self._pan_mm[1] += dy / scale
        self._refresh_view_cache()
        self.update()

    @property
    def view_zoom(self) -> float:
        return self._zoom

    def view_state(self) -> tuple[float, float, float]:
        """The view as one value: zoom and pan, with no geometry in it.

        Saved per project when the user switches tabs. Zoom and pan are the
        part of "where I was" that is easy to lose and annoying to rebuild --
        reopening a board fitted to the window is technically correct and
        still costs the user the place they had found.
        """
        return (float(self._zoom), float(self._pan_mm[0]), float(self._pan_mm[1]))

    def set_view_state(self, state: tuple[float, float, float] | None) -> None:
        """Restore a :meth:`view_state`. ``None`` fits the board instead."""
        if state is None:
            self.fit_view()
            return
        zoom, pan_x, pan_y = state
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(zoom)))
        self._pan_mm[0] = float(pan_x)
        self._pan_mm[1] = float(pan_y)
        # The saved state is zoom and pan *on top of* a fit, and this board is
        # not the one the scale was computed from, so the fit is taken again
        # before the stored zoom is applied to it.
        self.refit_view()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        # A resize can arrive while the constructor is still running, before
        # there is any view state to refit; the first paint takes the fit then.
        if not hasattr(self, "_fit_taken"):
            return
        # A bigger window shows the same view, so the fit follows the widget.
        # Only the base scale moves with it; the user's zoom and pan stay.
        self.refit_view()

    def _refresh_view_cache(self) -> None:
        """Cache the view arithmetic that ``to_screen`` needs per point.

        ``to_screen`` is the single hottest function in the whole program: one
        repaint of the card maps about fifty thousand vertices, and every call
        was doing ``self._plot.center()`` -- constructing a ``QPointF`` and
        reading it back twice -- plus attribute lookups for the pan and scale.
        Capturing the three numbers once per paint is the same arithmetic with
        none of the Qt objects, and measured at roughly a third of the cost.

        ⚠️ **Every path that changes the plot, the pan or the zoom has to call
        this immediately** -- not only the repaint. A repaint is *scheduled*,
        not synchronous, so between ``zoom_by()`` and the next frame the two
        halves of the canvas would disagree: ``to_screen`` would map through
        the stale numbers while the drag maths still divided by the live
        ``_px_per_mm``, and a 15 mm drag would come out as 1 mm. That is
        exactly the bug the guard test
        ``test_every_view_change_refreshes_the_mapping`` pins.

        Every mutator is covered: :meth:`_recompute_plot` (which every paint
        calls), :meth:`refit_view` (and so :meth:`fit_view` and
        :meth:`set_view_state`), :meth:`zoom_by` and :meth:`pan_by_pixels`.
        """
        plot = self._plot
        self._origin_x = plot.x() + plot.width() / 2.0
        self._origin_y = plot.y() + plot.height() / 2.0
        self._pan_x = self._pan_mm[0]
        self._pan_y = self._pan_mm[1]
        self._scale = self._px_per_mm
        self._view_key = (
            round(self._origin_x, 3),
            round(self._origin_y, 3),
            round(self._pan_x, 6),
            round(self._pan_y, 6),
            round(self._scale, 6),
        )

    def to_screen(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(
            self._origin_x + (x_mm - self._pan_x) * self._scale,
            self._origin_y - (y_mm - self._pan_y) * self._scale,
        )

    def to_mm(self, point: QPointF) -> tuple[float, float]:
        return (
            (point.x() - self._origin_x) / self._scale + self._pan_x,
            (self._origin_y - point.y()) / self._scale + self._pan_y,
        )

    def image_polygon(self, placement=None) -> QPolygonF:
        """A placement's rotated rectangle, in screen coordinates.

        Defaults to the selected placement, which is what the handles and the
        selection outline are drawn from.
        """
        target = placement if placement is not None else self.selected
        if target is None:
            return QPolygonF()
        return QPolygonF([self.to_screen(x, y) for x, y in target.corners()])

    def _image_bounds_px(self) -> QRectF | None:
        """Screen rectangle covering the selected placement, rotation included."""
        target = self.selected
        if target is None:
            return None
        top_left = self.to_screen(*target.corners()[0])
        bottom_right = self.to_screen(*target.corners()[2])
        return QRectF(top_left, bottom_right).normalized()

    def _image_target_rect(self, placement=None) -> QRectF:
        """Where a placement's pixmap is drawn, before rotation.

        The placement defines its own aspect ratio, and the artwork is fitted
        to it rather than stretched to fill it. A placement whose ratio differs
        from the source -- which happens when the user resizes one side with the
        ratio unlocked -- therefore keeps the artwork's proportions and leaves
        the remainder transparent, instead of distorting the picture.
        """
        target = placement if placement is not None else self.selected
        if target is None:
            return QRectF()
        pixmap = self._pixmap_for(target)
        if pixmap is None:
            return QRectF()

        # The fitted size comes from the shared definition rather than being
        # recomputed here: the exporters letterbox the artwork the same way,
        # and a second copy of this rule is how the canvas and the exported
        # board come to disagree about the picture's shape and position.
        # The ratio is the *window's*, not the whole picture's: once the user
        # frames part of an image, the part is what gets fitted into the box,
        # and measuring the whole picture would letterbox it differently from
        # the export.
        source_ratio = target.source_aspect_for(pixmap.width(), pixmap.height())
        draw_w_mm, draw_h_mm = drawn_size_mm(
            target.width_mm, target.height_mm, source_ratio
        )
        centre = self.to_screen(target.x_mm, target.y_mm)
        draw_w = draw_w_mm * self._px_per_mm
        draw_h = draw_h_mm * self._px_per_mm

        return QRectF(
            centre.x() - draw_w / 2.0,
            centre.y() - draw_h / 2.0,
            draw_w,
            draw_h,
        )

    # ------------------------------------------------------------- handles

    def handle_centres(self) -> dict[str, QPointF]:
        """Screen positions of the four edge handles.

        Only for artwork. A text line is sized by its font size rather than by
        its box, so an edge handle would have nothing honest to drag -- see the
        content panel, where the size is a field.
        """
        if not isinstance(self.selected, ImagePlacement):
            return {}
        bounds = self._image_bounds_px()
        if bounds is None:
            return {}
        return {
            "top": QPointF(bounds.center().x(), bounds.top()),
            "bottom": QPointF(bounds.center().x(), bounds.bottom()),
            "left": QPointF(bounds.left(), bounds.center().y()),
            "right": QPointF(bounds.right(), bounds.center().y()),
        }

    def _handle_at(self, point: QPointF) -> str:
        if self.selected is None:
            return ""
        for name, centre in self.handle_centres().items():
            if (point - centre).manhattanLength() <= HANDLE_HIT * 1.5:
                return name
        return ""

    def _contains(self, point: QPointF) -> bool:
        polygon = self.image_polygon()
        return bool(polygon.count()) and polygon.containsPoint(
            point, Qt.OddEvenFill
        )

    # ------------------------------------------------------------- editing

    def linked(self, placement) -> list[ImagePlacement]:
        """The placements that act as one object with ``placement``.

        Two things can make objects act as one, and both are handled here so
        every operation that moves or copies artwork asks one question.

        * **A 板画's two tones.** The white is on silkscreen and the black on
          copper, and the exporter writes them as a single geometry derived
          from the white one. Handling the tones separately tears the exported
          board along their seam, which is why the property panel already
          carries 翻转/旋转 across the pair. Copying, deleting and moving have
          exactly the same need, so they ask here.
        * **An explicit group** the user made -- see
          :class:`pcbart.artboard.PlacementGroup`.

        ⚠️ **The explicit group wins when there is one.** A painting's two tones
        belong to the same group by construction, so the two answers agree
        there; and for a group the user made out of unrelated objects there is
        no tone pairing to fall back on. Asking the other way round would mean
        a user who grouped two silkscreen pictures still only ever moved one.

        Line art is *not* paired: a picture spread over three layers really is
        three objects, each with its own size and position, and the user picks
        and edits them individually -- unless they group them, because then
        that is exactly what they asked for.
        """
        if placement is None:
            return []

        group = self.board.group_of(placement)
        if group is not None:
            members = self.board.group_all_members(group)
            if len(members) >= 2:
                # Text lines can be dragged but not resized or rotated the way
                # a placement is, so callers that resize filter for
                # ``ImagePlacement`` themselves. Returning everything here is
                # what makes a group move as one.
                return [
                    item
                    for item in members
                    if isinstance(item, ImagePlacement)
                ]

        if placement.tone not in ("white", "black") or not placement.import_id:
            return [placement]
        return [
            item
            for item in self.board.placements
            if item.import_id == placement.import_id
            and item.tone in ("white", "black")
        ]

    def copy_selection(self) -> bool:
        """Put the selection (and a painting's other tone) on the clipboard."""
        group = self.linked(self.selected)
        if not group:
            return False
        self._clipboard = [replace(item) for item in group]
        self.clipboard_changed.emit()
        return True

    def cut_selection(self) -> bool:
        if not self.copy_selection():
            return False
        return self.delete_selection()

    def paste(self) -> ImagePlacement | None:
        """Add a copy of the clipboard, offset so it is visibly a new object."""
        if not self._clipboard:
            return None
        return self._insert_clones(self._clipboard)

    def duplicate_selection(self) -> ImagePlacement | None:
        """Copy the selection in place, without touching the clipboard.

        Deliberately not implemented as copy-then-paste: that would replace
        whatever the user had copied, so duplicating would quietly destroy a
        clipboard they were still using.
        """
        group = self.linked(self.selected)
        if not group:
            return None
        return self._insert_clones(group)

    def _insert_clones(self, source: list[ImagePlacement]) -> ImagePlacement | None:
        """Add offset copies of ``source`` and select the first one."""
        if not source:
            return None

        # One fresh import id for the whole group, so a copied painting's two
        # tones still belong together -- and so the copy is independent of the
        # original rather than joining its painting on export.
        new_import_id = uuid.uuid4().hex[:8]

        clones: list[ImagePlacement] = []
        for item in source:
            clone = replace(item)
            clone.x_mm += DUPLICATE_OFFSET_MM
            clone.y_mm -= DUPLICATE_OFFSET_MM
            clone.import_id = new_import_id
            self._ensure_pixmap(clone.path)
            if clone.processed_path:
                self._ensure_pixmap(clone.processed_path)
            self.board.add_placement(clone)
            clones.append(clone)

        # Pasting repeatedly cascades rather than stacking every copy on one
        # spot, so the second paste is as visible as the first.
        if source is self._clipboard:
            self._clipboard = [replace(item) for item in clones]

        first = clones[0]
        self.selected = first
        self.selection_changed.emit(first)
        self.document_changed.emit()
        self._record()
        self.update()
        return first

    def delete_selection(self) -> bool:
        """Remove the selection, and a painting's other tone with it."""
        group = self.linked(self.selected)
        if not group:
            return False
        for item in group:
            self.board.remove_placement(item)
        self.selected = None
        self.selection_changed.emit(None)
        self.document_changed.emit()
        self._record()
        self.update()
        return True

    def nudge_selection(self, dx_mm: float, dy_mm: float) -> bool:
        """Move the selection by millimetres, as the arrow keys do."""
        group = self.linked(self.selected)
        if not group:
            return False
        for item in group:
            item.x_mm += dx_mm
            item.y_mm += dy_mm
        self.placement_changed.emit()
        # A held arrow key must come back in one Ctrl+Z, not one per repeat.
        self._record(_NUDGE_RUN)
        self.update()
        return True

    def rotate_selection(self, degrees: float) -> bool:
        """Turn the selection, keeping a painting's two tones together."""
        group = self.linked(self.selected)
        if not group:
            return False
        for item in group:
            item.rotation_deg = (item.rotation_deg + degrees + 180.0) % 360.0 - 180.0
        self.placement_changed.emit()
        self._record()
        self.update()
        return True

    def flip_selection(self, *, horizontal: bool) -> bool:
        group = self.linked(self.selected)
        if not group:
            return False
        for item in group:
            if horizontal:
                item.flip_horizontal = not item.flip_horizontal
            else:
                item.flip_vertical = not item.flip_vertical
        self.placement_changed.emit()
        self._record()
        self.update()
        return True

    def centre_selection(self) -> bool:
        group = self.linked(self.selected)
        if not group:
            return False
        for item in group:
            item.x_mm = 0.0
            item.y_mm = 0.0
        self.placement_changed.emit()
        self._record()
        self.update()
        return True

    # --------------------------------------------------------------- groups

    def can_group(self) -> bool:
        """Whether the current selection is enough to form a group.

        Two *distinct* objects. Checked rather than assumed so the menu entry
        can be greyed out instead of failing silently when the user has one
        thing selected -- a disabled item explains itself; an item that does
        nothing does not.
        """
        return len(self.selection_set()) >= 2

    def group_selection(self, name: str = ""):
        """Put everything selected into one named group.

        Returns the new group, or ``None`` when there was nothing to group.
        Text lines are accepted alongside placements: the card's wording is an
        element the user can see and pick, and refusing to group it would mean
        the card's own captions could never belong to a named element.
        """
        chosen = self.selection_set()
        if len(chosen) < 2:
            return None
        group = self.board.make_group(chosen, name=name)
        if group is None:
            return None
        self.document_changed.emit()
        self._record()
        self.update()
        return group

    def ungroup_selection(self) -> bool:
        """Take everything selected out of its group, dissolving empty ones."""
        chosen = self.selection_set()
        if not chosen:
            return False
        if not self.board.ungroup(chosen):
            return False
        self.document_changed.emit()
        self._record()
        self.update()
        return True

    def select_group(self, group) -> bool:
        """Select every member of ``group``."""
        members = self.board.group_members(group)
        if not members:
            return False
        self.selected = members[0]
        self.extra_selection = list(members[1:])
        self._selection_anchor = members[0]
        self.selection_changed.emit(members[0])
        self.update()
        return True

    @property
    def clipboard_size(self) -> int:
        """How many objects a paste would add; 0 disables the menu entry."""
        return len(self._clipboard)

    # -------------------------------------------------------------- framing

    def _region_choices(self):
        """The badge regions this board offers, or an empty list.

        Asked of the board's outline rather than of a flag on the canvas, so a
        board that stops being a badge card stops offering the entries without
        anything having to remember to clear a flag.
        """
        from pcbart.badge_card import BADGE_REGIONS, is_badge_template

        if self.board.outline.shape != "template":
            return []
        if not is_badge_template(self.board.outline.template or ""):
            return []
        return list(BADGE_REGIONS)

    def card_region_for(self, placement):
        """Which opening this card prescribes for ``placement``.

        The stage decides -- 精0's short window, 精2's full-height one -- because
        the two are different layouts rather than a preference, and the user's
        instruction is that choosing the template must not leave them picking
        长区/短区 by hand. A picture that already arrives cropped to one of the
        two ratios (the figure library is not, but hand-made art may be) names
        its own opening and that wins, since the shape is evidence and the stage
        is only a default.

        Returns ``None`` when the board is not a card.
        """
        from pcbart.badge_card import (
            BADGE_REGIONS,
            is_badge_template,
            region_for_aspect,
            stage_of,
        )

        if not is_badge_template(self.board.outline.template or ""):
            return None

        width_px, height_px = self._pixel_size(placement)
        region = None
        if width_px and height_px:
            region = region_for_aspect(width_px / float(height_px))
        if region is None:
            wanted = (
                "large" if stage_of(self.board.outline.template or "") == "精2" else "small"
            )
            region = next(
                (item for item in BADGE_REGIONS if item.key == wanted),
                BADGE_REGIONS[0],
            )
        return region

    def framed_region(self, placement):
        """The opening ``placement`` is **actually in right now**, or ``None``.

        ⚠️ **This is not the same question as :meth:`card_region_for`, and
        confusing the two is a reported bug.** ``card_region_for`` answers "which
        opening does this card prescribe?" -- the stage's default. This one
        answers "which opening was this picture put into?", which is the one the
        user chose when they right-clicked 放入短区 or 放入长区.

        The difference matters because 长区 is a taller opening on the same
        card: a 精0 card prescribes 短区, but the user is allowed to put a
        picture in 长区 anyway, and the crop has to keep that opening's shape.
        Reshaping it back to the prescribed one left a tall box holding a
        short-shaped crop -- reported as 「取景框从短区切换到长区的时候不会改变」.
        """
        from pcbart.badge_card import BADGE_REGIONS

        key = getattr(placement, "region", "")
        if key:
            chosen = next((item for item in BADGE_REGIONS if item.key == key), None)
            if chosen is not None:
                return chosen
        return self.card_region_for(placement)

    def card_illustrations(self):
        """The pictures the *user* brought, as opposed to the card's own art.

        A card carries generated elements too -- the 二维码, the 条形码, the 职业
        图标 and (on the short card) the 阵营 logo -- and those are already
        sized and placed by :mod:`pcbart.badge_content`. Framing them would
        shrink a 3.2 mm code into an illustration opening, so they are excluded
        by the same identity ``badge_content`` uses to find and replace its own
        work.
        """
        from pcbart.badge_content import ART_IMPORT_ID

        return [
            item
            for item in self.board.placements
            if getattr(item, "import_id", "") != ART_IMPORT_ID
        ]

    def _is_framing_session(self, placement) -> bool:
        """Whether ``placement`` is the one currently being framed by hand.

        ⚠️ **Only an explicit session counts, and that is a deliberate change.**
        Dragging used to pan the crop of *any* picture sitting in an opening,
        because the object's position was locked to that opening. The user has
        since asked for the opposite: the picture itself moves freely on the
        canvas (so a figure can be arranged around the card's top cutout, or
        pushed partly off the board), and the opening only decides which part is
        kept. So a plain drag moves the picture again, and the crop-pan gesture
        lives where the user asks for it -- 右键 → 放入短区/长区 enters this
        session and says so on the canvas.
        """
        return self.framing is not None and placement is self.framing

    def _framing_target(self):
        """The placement whose *crop* the canvas is currently presenting.

        The explicitly framed one when there is one; otherwise the selected
        picture that sits in an opening. Used by the ghost, the on-canvas hint
        and Ctrl+wheel, so all three agree about which crop is being adjusted.
        """
        if self.framing is not None:
            return self.framing
        selected = self.selected
        if selected is not None and getattr(selected, "region", ""):
            return selected
        return None

    def _region_path(self, placement) -> QPainterPath | None:
        """The illustration opening as a screen-space path, or ``None``.

        The picture is clipped to this, so the canvas shows exactly what the
        writers print. ⚠️ **Without it the canvas and the export disagree the
        moment the picture is moved**: ``build_copper`` clips ink to the opening,
        so ink drawn outside it on screen is ink the user thinks they will get.
        """
        from pcbart.badge_card import BADGE_REGIONS

        key = getattr(placement, "region", "")
        if not key:
            return None
        region = next((item for item in BADGE_REGIONS if item.key == key), None)
        if region is None:
            return None
        x0, y0, x1, y1 = region.bounds
        path = QPainterPath()
        path.addPolygon(
            QPolygonF(
                [
                    self.to_screen(x, y)
                    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                ]
            )
        )
        path.closeSubpath()
        return path

    def _region_bounds(self, placement):
        """The opening's millimetre bounds, or ``None``."""
        from pcbart.badge_card import BADGE_REGIONS

        key = getattr(placement, "region", "")
        if not key:
            return None
        region = next((item for item in BADGE_REGIONS if item.key == key), None)
        return None if region is None else region.bounds

    @staticmethod
    def _warning_colour(colour: QColor) -> QColor:
        """The layer's colour, darkened, for a picture that has left its opening.

        The user asked for exactly this: 顶层 → 暗红, 顶层丝印 → 深黄, and so on
        down the layer table. Darkening the layer's own colour rather than
        picking a fixed warning colour keeps the reading "this is the same
        object, in the same layer, and something is wrong with it" -- a red
        warning on a silk-layer picture would point at the wrong layer.
        """
        return QColor(colour).darker(200)

    def reset_framing(self, placement) -> bool:
        """Re-fit the crop to its opening, discarding the user's pan and zoom.

        The escape hatch for "I dragged it somewhere and cannot find the
        picture again". Deliberately separate from
        :meth:`frame_into_card_region`, which *preserves* an existing window --
        re-importing or re-applying a card must not throw away a crop the user
        chose, so resetting has to be asked for.
        """
        if placement is None or not getattr(placement, "region", ""):
            return False
        # ⚠️ The opening the picture is *already in*, not the one the card
        # prescribes: 重置取景 means "put this crop back the way it was for the
        # opening I chose", so a picture the user deliberately moved to 长区 on a
        # 精0 card must not snap back into 短区.
        region = self.framed_region(placement)
        if region is None:
            return False
        # Clearing the window first is what makes this a *reset*: the fit below
        # starts from the whole picture instead of from the current crop.
        self._set_window(placement, None)
        self._apply_region(placement, region, run_key=_FRAME_RUN)
        return True

    def frame_into_card_region(
        self, placement, *, reframe: bool = False, run_key: str | None = None
    ) -> bool:
        """Put ``placement`` into the opening this card prescribes, cropped.

        ⚠️ **This is the part that was missing, and its absence is why the user
        saw 「没有自动裁切我放的图片」.** The 立绘 region was applied only when
        somebody called :meth:`start_framing` -- and the only callers were
        scripts. The panel never called it, so a picture imported onto a card
        landed at its default 76.2 mm with no ``region`` and no
        ``source_window``: nothing cropped it, and both writers treated it as an
        ordinary picture. The 彩色丝印 evidence that suggested otherwise came
        from a preview script that framed the placement itself.

        ⚠️ **The early-return rule is what this method exists to get right, and
        it was wrong twice.** Returning whenever ``region`` was set left a
        picture framed for the short window still declaring ``"small"`` after
        the card was switched to 精2 -- the box and the window kept the old
        opening's shape while the region key named a different one, so the
        export stretched the picture into an opening it was never framed for.
        Moving the rule the other way, to always re-frame, threw away the window
        the user had just panned and zoomed by hand.

        ``reframe`` is the switch between those two: ``True`` when the layout
        itself changed (every picture is redone for the new opening), ``False``
        for a caller that only wants pictures picked up when they have *never*
        been framed -- a plain import, or re-applying the card already in force.
        A placement the user deliberately put in the other opening therefore
        keeps it; the template decides the default, it does not overrule a
        direct choice.

        ⚠️ **It applies the crop without entering framing mode.** The picture is
        placed in the opening and the part outside it is genuinely cut, but a
        drag still moves the object and no amber outline appears -- the user did
        not ask to frame anything, they asked to import a picture. Entering the
        mode here left every automatic import in 「取景中」, which is a state
        they then had to find their way out of.
        """
        if placement is None:
            return False
        region = self.card_region_for(placement)
        if region is None:
            return False
        if not reframe and (placement.region or placement.source_window is not None):
            return False

        self._apply_region(placement, region, run_key=run_key)
        return True

    def frame_illustrations_for_card(self, *, reframe: bool = True) -> int:
        """Put every picture the user imported into this card's opening.

        Called when a 通行证 template is applied or switched, so the layout
        choice determines the crop: 精0 sends the figure to the short window and
        精2 to the full-height one, with nobody right-clicking 放入长区.

        ``reframe`` false leaves a picture alone when it already sits in the
        opening this card prescribes *or* in an opening the user picked by hand;
        only pictures that were never framed at all are picked up. That is the
        re-apply path, where the card has not changed and a deliberate override
        should stand.

        The card's generated elements are skipped -- see
        :meth:`card_illustrations` -- and so is a board that is not a card.
        Returns how many placements were (re)framed.

        Edits from one template switch coalesce into a single undo step, but the
        key carries the template name so that two *different* switches stay two
        steps. ⚠️ A bare ``_FRAME_RUN`` does not work here: switching 精0 → 精2
        and back inside the coalescing window (which is what clicking through
        the combo does) would merge into the earlier step, and one Ctrl+Z would
        jump past the layout the user actually came from.
        """
        from pcbart.badge_card import is_badge_template

        template = self.board.outline.template or ""
        if not is_badge_template(template):
            return 0

        framed = 0
        for placement in self.card_illustrations():
            if self.frame_into_card_region(
                placement,
                reframe=reframe,
                run_key=f"{_FRAME_RUN}:{template}",
            ):
                framed += 1
        return framed

    def start_framing(self, placement, region) -> None:
        """Frame ``placement``'s contents inside ``region``, and enter framing.

        The placement takes the region's shape and position, and its window is
        fitted to the region's aspect so the picture is never stretched. The
        user then pans and zooms inside; the part of the picture outside the
        window is what the export cuts away.

        This is the *interactive* entry: it leaves the canvas in framing mode,
        where a drag moves the picture inside its window rather than the object.
        A caller that only wants the crop applied -- an automatic import -- uses
        :meth:`frame_into_card_region`, which does the same geometry without
        taking over the drag gesture.
        """
        if placement is None or region is None:
            return
        self.framing = placement
        self.framing_region = region
        self._apply_region(placement, region)

    def _apply_region(self, placement, region, *, run_key: str | None = None) -> None:
        """Give ``placement`` the region's box and a window of its shape.

        Split out from :meth:`start_framing` so the geometry can be applied
        without entering the framing gesture. ⚠️ **The two really are separate
        concerns, and conflating them made an automatic import look broken**: it
        framed the picture correctly and then left the canvas in 「取景中」, with
        the amber outline, the ghost of the cropped-away part and drag meaning
        "pan the picture" -- a mode the user never asked for and had to escape
        by right-clicking.

        ``run_key`` merges the recording with the neighbouring edits of the same
        kind. Framing every picture on a card is one user decision -- picking
        the template -- so without it a card with three pictures would take
        three Ctrl+Z presses to undo one choice.
        """
        if placement is None or region is None:
            return
        self.selected = placement
        # The geometry changes *before* the panels are told, not after: the
        # property pane reads the placement when it hears the signal, so
        # emitting first left it showing the size and position from before the
        # region was applied -- a window that says 76.20 mm while the object on
        # the canvas is 38.86 mm.
        self._fit_window_to_region(placement, region)
        self.selection_changed.emit(placement)
        self.placement_changed.emit()
        self._record(run_key)
        self.update()

    def stop_framing(self) -> None:
        self.framing = None
        self.framing_region = None
        self.update()

    def _image_path_for(self, placement) -> str:
        """Which file defines this placement's pixels.

        The processed result when there is one -- 线稿 and 板画 write a new
        picture and that is what gets printed and framed -- otherwise the
        source. One definition, because the canvas, the framing math and both
        writers have to be looking at the same file for a window expressed as a
        fraction of it to mean the same thing everywhere.
        """
        return placement.processed_path or placement.path

    def _pixel_size(self, placement) -> tuple[int, int]:
        """The picture's size in pixels, or ``(0, 0)`` when it cannot be read.

        Painting may tolerate a missing picture -- the placement still needs an
        outline on screen -- so this stays forgiving. Framing may **not**: see
        :meth:`_fit_window_to_region`, which reads the size directly and raises
        rather than silently skipping the crop.
        """
        try:
            return picture_size(self._image_path_for(placement))
        except UnreadablePicture:
            return (0, 0)

    def _set_window(self, placement, window) -> None:
        """Write one framing window to everything that shares the picture.

        A 板画 is two placements -- the white tone on silk and the black on
        copper -- and the exporter derives one from the other. Giving the two
        different windows crops them at different offsets, so the finished
        board tears along their seam. :meth:`linked` already answers "what
        moves with this", and framing is one more thing that moves with it.

        ⚠️ **Every crop edit funnels through here, so the shape is enforced
        here too.** Doing it in each caller instead is how the squash got in:
        ``window_zoom`` and ``_pan_framed`` each clamped their own way, and the
        one that passed no aspect flattened the window against the picture's
        edge. A single gate means a future crop gesture cannot reintroduce it.

        ``None`` is passed by :meth:`reset_framing` to mean "no crop yet", and
        is written through unchanged -- there is no shape to enforce.

        ⚠️ **Callers pass a raw window and let this method reshape it.** An
        earlier version reshaped in each gesture as well, which made the gate
        decorative: deleting it broke nothing, because every caller still
        happened to constrain its own window. A single enforcement point is only
        real if it is the *only* one.
        """
        if window is not None:
            window = self._constrain_window(placement, window)
        for item in self.linked(placement):
            item.source_window = window

    def _fit_window_to_region(self, placement, region) -> None:
        """Give the placement the region's box and a window of the same shape.

        The box is the region, exactly: the artwork is going to be printed
        there, so its outline on the canvas has to be the region and not
        something the user has to line up by hand.

        ⚠️ **Setting the window is the crop, not an optional extra.** The
        window is what both writers cut the picture with; a placement whose box
        is the region but whose window is still ``None`` exports the *whole
        picture* scaled down into that box. That is the reported defect --
        「放置短区是强制把原图缩小，而非裁切」 -- and it happened whenever the
        pixel size came back ``(0, 0)`` and the code below returned early.

        So a size that cannot be read now raises. Placing a picture into a
        region without a window is never the user's intent: they asked for a
        crop, and silently printing something else is worse than refusing.
        """
        for item in self.linked(placement):
            item.x_mm = region.centre_x_mm
            item.y_mm = region.centre_y_mm
            item.width_mm = region.width_mm
            item.height_mm = region.height_mm
            item.rotation_deg = 0.0
            item.keep_aspect = False
            # Remember which opening this was framed into. The export clips to
            # it, so a later resize or move cannot push ink outside the card's
            # own window.
            item.region = region.key

        width_px, height_px = picture_size(self._image_path_for(placement))
        self._set_window(
            placement,
            _window_of_aspect(
                placement.source_window, region.aspect, (width_px, height_px)
            ),
        )

    def window_zoom(self, factor: float) -> bool:
        """Zoom the *crop* in or out about the window's centre.

        Zooming *in* means showing less of the picture, so the window shrinks --
        which is the direction users expect from "zoom the picture in".

        Acts on :meth:`_framing_target`, so it works for a picture sitting in a
        card opening even when no explicit framing gesture armed it. That
        matters: this is the only way the interface can zoom a crop, and before
        it was reachable the automatic import had no adjustable zoom at all.

        ⚠️ **The window is re-shaped by :meth:`_set_window` on the way in** --
        the single place every crop edit funnels through. Doing it here as well
        would leave two enforcement points, which is the arrangement that let
        the squash in: ``window_zoom`` reshaped and ``_pan_framed`` did not, so
        the shape depended on which gesture ran last.
        """
        placement = self._framing_target()
        if placement is None or placement.source_window is None:
            return False
        size = self._pixel_size(placement)
        if not size[0] or not size[1]:
            return False
        self._set_window(
            placement, _clamp_window(_zoom_window(placement.source_window, 1.0 / factor))
        )
        self._record(_FRAME_RUN)
        self.update()
        return True

    def _constrain_window(self, placement, window, size=None):
        """``window`` forced back into the shape its opening prescribes.

        Called by :meth:`_set_window` for every crop edit (zoom, pan, and
        anything added later). It is not cosmetic: the export fits the cropped
        pixels into the placement box *preserving their aspect* (see
        ``drawn_size_mm``), so a window of the wrong shape leaves blank bands
        inside the opening. Measured on the reported drawing: a 精0 crop
        squashed from aspect 0.690 to 1.000 drew the figure as a narrow band
        with blank space above and below -- exactly the screenshot the user sent.

        ⚠️ **It also repairs documents already saved with a squashed window**,
        because a window of the wrong shape is never something the user can
        choose: every path that creates one uses the region's aspect. Anything
        else in a card opening is the old bug's fingerprint.

        ⚠️ **It uses :meth:`framed_region`, not :meth:`card_region_for`.** The
        opening the picture was put into is the one whose shape the crop must
        keep -- asking the card for its *prescribed* opening instead reshaped a
        长区 crop back to 短区's, so switching openings appeared to do nothing.
        """
        if size is None:
            size = self._pixel_size(placement)
        if not size[0] or not size[1]:
            return window
        region = self.framed_region(placement)
        if region is None:
            return window
        # ``_window_of_aspect`` fits the largest window of the wanted shape
        # inside the one it is given, so a squashed window comes back correct
        # and centred inside itself rather than jumping.
        return _window_of_aspect(window, region.aspect, size)

    # --------------------------------------------------------------- history

    def _record(self, run_key: str | None = None) -> None:
        """Push the current document state onto the history.

        Every mutating method calls this, and the property pane's edits route
        through :meth:`record_external_change`. Missing a route is the failure
        mode to watch for: the edit succeeds but ``Ctrl+Z`` skips over it,
        which is worse than no undo at all because the user believes it is
        covered.
        """
        self.history.record(run_key)
        self.history_changed.emit()

    def record_external_change(self, run_key: str | None = None) -> None:
        """Note an edit made through a panel rather than through the canvas.

        The property pane writes to the same placements the canvas draws, so
        its changes belong in the same history. It cannot call :meth:`_record`
        without knowing about this widget, so the window connects its ``changed``
        signal here instead.
        """
        self._record(run_key)

    def set_history(self, history) -> None:
        """Adopt another document's history, as a tab switch requires."""
        self.history = history
        self.history_changed.emit()

    def can_undo(self) -> bool:
        return self.history.can_undo

    def undo(self) -> bool:
        """Step the document back one edit.

        The selection is kept when the object it names still exists -- the
        restored placements reuse their objects, so the common case needs no
        re-pointing. When the undone edit *deleted* something, undoing brings
        it back as a different object and the stale selection is dropped rather
        than left pointing at a placement no longer on the board.
        """
        if not self.history.undo():
            return False

        # Undoing a delete brings placements back; undo can also restore a
        # placement whose picture was never drawn since a restart. Warming the
        # cache here is what keeps the restored object visible on the first
        # repaint instead of as an empty rectangle.
        for placement in self.board.placements:
            self._ensure_pixmap(placement.path)
            if placement.processed_path:
                self._ensure_pixmap(placement.processed_path)

        # The selection survives an undo only while the object it names is
        # still in the document. Which list to look in depends on the kind --
        # text lives beside the placements, not among them -- so the check asks
        # each list rather than assuming.
        selected = self.selected
        if selected is not None and not (
            selected in self.board.placements
            or selected in self.board.texts
            or selected in self.board.cutouts
        ):
            self.selected = None
            self.selection_changed.emit(None)

        # Both sets of listeners have to hear about this: the object set and
        # the object's numbers can each have changed, and it depends entirely
        # on what was undone.
        self.document_changed.emit()
        self.placement_changed.emit()
        self.history_changed.emit()
        self.update()
        return True

    # ------------------------------------------------------------ menu

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """The right-click menu, for the object under the pointer.

        Right-clicking an object selects it first. A menu that acted on
        something other than what is under the pointer would be a trap: the
        user's last click is what the entries appear to be about, and they have
        no way to tell that it is not.
        """
        x_mm, y_mm = self.to_mm(event.pos())
        hit = self.board.placement_at(x_mm, y_mm)
        if hit is not None and hit is not self.selected:
            self.selected = hit
            self.selection_changed.emit(hit)
            self.update()

        menu = QMenu(self)
        if hit is not None:
            self._fill_object_menu(menu)
        else:
            self._fill_canvas_menu(menu)

        menu.exec(event.globalPos())
        event.accept()

    def _fill_object_menu(self, menu: QMenu) -> None:
        for text, slot in (
            ("复制", self.copy_selection),
            ("剪切", self.cut_selection),
            ("复制一份", self.duplicate_selection),
            ("删除", self.delete_selection),
        ):
            action = menu.addAction(text)
            action.triggered.connect(slot)

        menu.addSeparator()
        rotate = menu.addAction("旋转 90°")
        rotate.triggered.connect(lambda: self.rotate_selection(ROTATE_STEP_DEG))
        flip_h = menu.addAction("水平翻转")
        flip_h.triggered.connect(lambda: self.flip_selection(horizontal=True))
        flip_v = menu.addAction("垂直翻转")
        flip_v.triggered.connect(lambda: self.flip_selection(horizontal=False))

        # On a badge card, filling a region is the operation this object is
        # most likely to need, so it is offered right here rather than hidden
        # behind a panel. It only appears when the board really is that card:
        # on any other board the entries would promise something that has no
        # region to fill.
        regions = self._region_choices()
        if regions:
            menu.addSeparator()
            for region in regions:
                entry = menu.addAction(f"放入{region.label}")
                entry.triggered.connect(
                    lambda _checked=False, r=region: self.start_framing(self.selected, r)
                )
            # Offered for anything already sitting in an opening, not only
            # while an explicit framing gesture is running: the automatic
            # import does not enter that gesture, so keying these on ``framing``
            # left the user without any way to zoom the crop or to undo a pan
            # they had taken too far.
            target = self._framing_target()
            if target is not None and target.source_window is not None:
                zoom_in = menu.addAction("放大取景")
                zoom_in.triggered.connect(lambda: self.window_zoom(1.25))
                zoom_out = menu.addAction("缩小取景")
                zoom_out.triggered.connect(lambda: self.window_zoom(1 / 1.25))
                reset = menu.addAction("重置取景")
                reset.triggered.connect(
                    lambda _checked=False, item=target: self.reset_framing(item)
                )
            if self.framing is not None:
                done = menu.addAction("结束取景")
                done.triggered.connect(self.stop_framing)

        menu.addSeparator()
        centre = menu.addAction("居中")
        centre.triggered.connect(self.centre_selection)
        fit = menu.addAction("按板框缩放")
        fit.triggered.connect(self.fit_to_board_requested.emit)

        menu.addSeparator()
        self._add_paste_entry(menu)

    def _fill_canvas_menu(self, menu: QMenu) -> None:
        self._add_paste_entry(menu)
        menu.addSeparator()
        fit = menu.addAction("按板框缩放")
        fit.triggered.connect(self.fit_to_board_requested.emit)
        reset = menu.addAction("适应窗口")
        reset.triggered.connect(self.reset_view_requested.emit)

    def _add_paste_entry(self, menu: QMenu) -> None:
        paste = menu.addAction("粘贴")
        paste.setEnabled(self.clipboard_size > 0)
        paste.triggered.connect(self.paste)

    # --------------------------------------------------------------- input

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        position = event.position()

        # Panning is on the middle button and on space-drag, matching the
        # editor. It takes priority over everything: with the view moving under
        # the cursor, a drag that also moved an object would be unpredictable.
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_held
        ):
            self._mode = "pan"
            self._pan_anchor = position
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            return

        # A handle takes priority: it sits on the edge, inside the image.
        handle = self._handle_at(position)
        if handle and self.selected is not None:
            target = self.selected
            self._mode = "resize"
            self._handle = handle
            self._press_pos = position
            self._press_size = (target.width_mm, target.height_mm)
            self._press_centre = (target.x_mm, target.y_mm)
            event.accept()
            return

        x_mm, y_mm = self.to_mm(position)
        modifiers = event.modifiers()
        additive = bool(modifiers & Qt.ControlModifier)
        extending = bool(modifiers & Qt.ShiftModifier)
        # Text is checked before the artwork. The lines are thin and are painted
        # on top of the picture, so the topmost thing under the pointer is the
        # text -- taking the artwork first would make the words unpickable
        # wherever they cross a picture, which is most of the card.
        text_hit = self.board.text_line(x_mm, y_mm)
        if text_hit is not None:
            if additive:
                self.toggle_selection(text_hit)
                event.accept()
                return
            if extending:
                self.extend_selection_to(text_hit)
                event.accept()
                return
            self.select(text_hit)
            self._arm_drag(
                position, x_mm, y_mm, (text_hit.x_mm, text_hit.y_mm)
            )
            event.accept()
            return

        hit = self.board.placement_at(x_mm, y_mm)
        if hit is not None:
            if additive:
                # Ctrl-click builds up a set, so the click must not also arm a
                # drag -- otherwise the pointer's own jitter would move the
                # object the user was only trying to add.
                self.toggle_selection(hit)
                event.accept()
                return
            if extending:
                self.extend_selection_to(hit)
                event.accept()
                return
            if hit is not self.selected:
                self.select(hit)
            # With a region being filled *by hand*, a drag pans the picture
            # inside its window; the object stays where the region put it. This
            # is the opposite of what a drag does everywhere else, which is why
            # the mode is explicit and says so on the canvas -- see
            # :meth:`_is_framing_session` for why only an explicit session
            # behaves this way.
            if self._is_framing_session(hit):
                self._mode = "frame"
                self._press_pos = position
                self._press_window = hit.source_window
                self._pending_move = True
                event.accept()
                return
            # Movement is armed but not started: a click that merely selects
            # must not nudge the artwork. The drag begins on the next move.
            self._arm_drag(position, x_mm, y_mm, (hit.x_mm, hit.y_mm))
            event.accept()
            return

        if self.selected is not None:
            self.select(None)

    def _drag_targets(self) -> list:
        """Everything a drag currently moves.

        The union of the multi-selection and any group those objects belong to.
        A group's other members join even when they were not individually
        clicked -- and text members too, which is why this asks
        :meth:`ArtBoard.group_all_members` rather than the placement-only
        :meth:`ArtBoard.group_members`. Moving one half of a 板画 away from the
        other is the failure the group exists to prevent, and a drag is the
        easiest way to cause it.
        """
        targets = self.selection_set()
        if self.selected is None:
            return targets
        group = self.board.group_of(self.selected)
        if group is None:
            return targets
        for item in self.board.group_all_members(group):
            if item not in targets:
                targets.append(item)
        return targets

    def _arm_drag(self, position, x_mm: float, y_mm: float, centre) -> None:
        """Record the press state for the objects this drag will move.

        Called on press rather than computed on the first move, for the reason
        the old single-target version did the same: the drag has to know where
        everything started, and by the time the pointer has moved the objects
        are already somewhere else.
        """
        targets = self._drag_targets()
        # The clicked object is always a target, even if ``selected`` was not
        # set to it -- a Ctrl-click exits before arming, so by here it is.
        self._mode = "move"
        self._press_pos = position
        self._press_mm = (x_mm, y_mm)
        self._press_centre = centre
        self._press_centres = {
            id(item): (item.x_mm, item.y_mm) for item in targets
        }
        self._pending_move = True

    def select(self, item) -> None:
        """Make ``item`` the selection, whatever kind of thing it is.

        One entry point so the four places that clear or set the selection --
        a click, Escape, a layer being hidden, closing the property pane -- all
        stay in step, including emitting the signal the panels listen to.

        A plain selection **clears the extra ones**. That is what makes an
        ordinary click a reliable way out of a half-built multi-selection:
        without it, Ctrl-clicking three things and then clicking a fourth
        somewhere else would leave the first three still marked.
        """
        self.extra_selection = []
        self._selection_anchor = item
        if item is self.selected:
            self.selection_changed.emit(item)
            self.update()
            return
        self.selected = item
        self.selection_changed.emit(item)
        self.update()

    def selection_set(self) -> list:
        """Everything currently selected, primary first, in document order.

        This is the list the group commands act on. Deduplicated by identity
        rather than by ``==``: two placements with identical geometry are still
        two objects, and a dataclass comparison would collapse them into one
        and then build a group with fewer members than the user picked.
        """
        chosen = [self.selected, *self.extra_selection]
        seen: set[int] = set()
        ordered = []
        for item in chosen:
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            ordered.append(item)
        # Document order, so a group's member list does not depend on the order
        # the user happened to click in.
        document = list(self.board.placements) + list(self.board.texts)
        rank = {id(item): index for index, item in enumerate(document)}
        return sorted(ordered, key=lambda item: rank.get(id(item), len(document)))

    def toggle_selection(self, item) -> None:
        """Add or remove ``item`` from the multi-selection (Ctrl-click).

        Removing the primary promotes one of the others, so there is always a
        primary while anything is selected -- the handles and the property
        panel have nothing to work with otherwise.
        """
        if item is None:
            return
        if item is self.selected:
            remaining = [entry for entry in self.extra_selection if entry is not item]
            self.selected = remaining[0] if remaining else None
            self.extra_selection = remaining[1:] if remaining else []
            self._selection_anchor = self.selected
            self.selection_changed.emit(self.selected)
            self.update()
            return
        others = [entry for entry in self.extra_selection if entry is not item]
        if len(others) == len(self.extra_selection):
            others.append(item)
        self.extra_selection = others
        self._selection_anchor = item
        self.selection_changed.emit(self.selected)
        self.update()

    def extend_selection_to(self, item) -> None:
        """Select everything between the anchor and ``item`` (Shift-click).

        Ranges over the *document* order -- placements then text -- rather than
        the order objects were clicked, so the same two clicks always pick the
        same set. Grouped objects contribute their whole group, because
        selecting one member of a group and not the others is the state the
        group exists to prevent.
        """
        if item is None:
            return
        document = list(self.board.placements) + list(self.board.texts)
        rank = {id(entry): index for index, entry in enumerate(document)}
        # ⚠️ Falls back to the current selection when no anchor was recorded.
        # ``.selected`` is assigned directly in several places -- the fixture,
        # the preprocessing panel after an import, undo -- and none of those go
        # through :meth:`select`, so the anchor can legitimately be unset while
        # something *is* selected. Treating that as "no range" would make
        # Shift-click silently behave like a plain click, which is the kind of
        # no-op the user cannot tell apart from a broken key.
        anchor = self._selection_anchor or self.selected
        if anchor is None or id(anchor) not in rank or id(item) not in rank:
            self.select(item)
            return
        low, high = sorted((rank[id(anchor)], rank[id(item)]))
        span = document[low : high + 1]

        expanded = []
        for entry in span:
            group = self.board.group_of(entry) if isinstance(entry, ImagePlacement) else None
            expanded.extend(self.board.group_members(group) if group else [entry])
        self.selected = item
        self.extra_selection = [entry for entry in expanded if entry is not item]
        self.selection_changed.emit(item)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        position = event.position()
        if self._mode == "pan" and self._pan_anchor is not None:
            delta = position - self._pan_anchor
            self._pan_anchor = position
            self.pan_by_pixels(delta.x(), delta.y())
            event.accept()
            return

        if self._mode == "move" and event.buttons() & Qt.LeftButton:
            self._pending_move = False
            # Work in screen pixels, not millimetres: to_mm() measures from the
            # viewport centre, so subtracting two of its readings while the
            # artwork itself is moving cancels the motion out.
            #
            # ⚠️ Absolute, not incremental. Each target is placed at *its own*
            # press position plus the total delta, so the drag cannot drift --
            # nudging by a per-event delta accumulates every rounding error the
            # pointer produces, and a slow drag then lands short of the cursor.
            targets = self._drag_targets()
            if not targets:
                return
            delta = position - self._press_pos
            dx = delta.x() / self._px_per_mm
            dy = -delta.y() / self._px_per_mm
            for item in targets:
                start = self._press_centres.get(id(item))
                if start is None:
                    continue
                item.x_mm = start[0] + dx
                item.y_mm = start[1] + dy
            self.update()
            event.accept()
            return

        if self._mode == "frame" and event.buttons() & Qt.LeftButton:
            self._pending_move = False
            self._pan_framed(position)
            event.accept()
            return

        if self._mode == "resize" and event.buttons() & Qt.LeftButton:
            self._apply_resize(position)
            event.accept()
            return

        self._update_cursor(position)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode == "pan":
            self._mode = ""
            self._pan_anchor = None
            self._update_cursor(event.position())
            return
        if self._mode == "frame" and not self._pending_move:
            # One history step for the whole pan, like a drag.
            self._record(_FRAME_RUN)
            self.placement_changed.emit()
        elif self._mode in ("move", "resize") and not self._pending_move:
            # One history step for the whole drag, recorded on release rather
            # than per motion event: a drag emits dozens of those, and each
            # would otherwise become its own undo step.
            self._record(_DRAG_RUN)
            self.placement_changed.emit()
            # Text and cutouts live outside ``placements``, and the undo
            # snapshot covers them, so a drag needs the document signal too or
            # the panels that show the template's content would keep the old
            # numbers.
            if not isinstance(self.selected, ImagePlacement):
                self.document_changed.emit()
        self._mode = ""
        self._handle = ""
        self._press_window = None
        self._pending_move = False
        self._update_cursor(event.position())

    def _pan_framed(self, position: QPointF) -> None:
        """Slide the picture under the window by the drag distance.

        The picture follows the pointer: dragging right shows a part of it that
        is further left, so the window moves the other way. The conversion goes
        through the drawn rectangle, because that is the size the user sees the
        picture at.

        Acts on :meth:`_framing_target`, not on :attr:`framing`: a picture the
        automatic import placed in an opening is panned the same way even
        though no explicit framing gesture was ever started.
        """
        placement = self._framing_target()
        if placement is None or placement.source_window is None:
            return
        drawn = self._image_target_rect(placement)
        if drawn.width() <= 0 or drawn.height() <= 0:
            return

        delta = position - self._press_pos
        x0, y0, x1, y1 = self._press_window or placement.source_window
        width = x1 - x0
        height = y1 - y0
        # The raw move; ``_set_window`` reshapes it. Deliberately not constrained
        # here -- see there for why a second enforcement point is a liability.
        self._set_window(
            placement,
            _clamp_window(
                (
                    x0 - delta.x() / drawn.width() * width,
                    y0 - delta.y() / drawn.height() * height,
                    x1 - delta.x() / drawn.width() * width,
                    y1 - delta.y() / drawn.height() * height,
                )
            ),
        )
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom the view, anchored at the pointer -- or the crop, with Ctrl.

        The canvas is the one place a wheel *should* do something: this is a
        view, not a value. The rule against wheel input applies to editable
        controls, whose value the user cannot see changing.

        ⚠️ **Ctrl+wheel is the crop's only mouse path.** ``window_zoom`` existed
        but nothing in the interface called it -- only scripts and tests did --
        so a picture sitting in a card opening could be panned into place but
        never zoomed. Plain wheel keeps zooming the *view*, deliberately: the
        user has stated that view zoom lives on the wheel, and moving it would
        break the gesture for everything that is not a framed picture.
        """
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        if event.modifiers() & Qt.ControlModifier:
            target = self._framing_target()
            if target is not None and self.window_zoom(1.15 ** steps):
                event.accept()
                return
        self.zoom_by(1.15 ** steps, event.position())
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Ctrl+Z is handled here as well as on the 编辑 menu. The menu action
        # carries the shortcut so it works wherever focus is; this branch keeps
        # the canvas working on its own, which is also how the tests reach it.
        if (
            event.key() == Qt.Key_Z
            and event.modifiers() & Qt.ControlModifier
            and not event.modifiers() & Qt.ShiftModifier
        ):
            if self.undo():
                event.accept()
                return

        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return

        if event.key() == Qt.Key_Escape and self.selected is not None:
            self.selected = None
            self.selection_changed.emit(None)
            self.update()
            event.accept()
            return

        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selection():
                event.accept()
                return

        if event.key() == Qt.Key_R and not event.modifiers() & Qt.ControlModifier:
            if self.rotate_selection(ROTATE_STEP_DEG):
                event.accept()
                return

        step = self._nudge_step(event.modifiers())
        if step and event.key() in (
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down,
        ):
            dx = {Qt.Key_Left: -step, Qt.Key_Right: step}.get(event.key(), 0.0)
            dy = {Qt.Key_Down: -step, Qt.Key_Up: step}.get(event.key(), 0.0)
            if self.nudge_selection(dx, dy):
                event.accept()
                return

        super().keyPressEvent(event)

    @staticmethod
    def _nudge_step(modifiers) -> float:
        """How far an arrow key moves the selection, or 0 to leave it alone.

        Shift is the finer step rather than the coarser one: the base step is
        already a whole millimetre, and the case that needs help is landing a
        picture exactly on a board edge, not crossing the board faster.
        """
        if modifiers & Qt.ShiftModifier:
            return NUDGE_FINE_MM
        if modifiers & (Qt.ControlModifier | Qt.AltModifier):
            return 0.0
        return NUDGE_MM

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor(event.position() if hasattr(event, "position") else QPointF())
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _update_cursor(self, position: QPointF) -> None:
        if self._space_held:
            self.setCursor(Qt.OpenHandCursor)
            return
        if self._handle_at(position):
            name = self._handle_at(position)
            self.setCursor(
                Qt.SizeVerCursor if name in ("top", "bottom") else Qt.SizeHorCursor
            )
        elif self._contains(position):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _apply_resize(self, position: QPointF) -> None:
        """Resize from one edge, keeping the opposite edge pinned."""
        image = self.selected
        if image is None:
            return

        x_mm, y_mm = self.to_mm(position)
        width, height = self._press_size
        centre_x, centre_y = self._press_centre

        if self._handle == "right":
            left = centre_x - width / 2.0
            new_width = max(0.1, x_mm - left)
            new_height = height
            new_centre_x = left + new_width / 2.0
            new_centre_y = centre_y
        elif self._handle == "left":
            right = centre_x + width / 2.0
            new_width = max(0.1, right - x_mm)
            new_height = height
            new_centre_x = right - new_width / 2.0
            new_centre_y = centre_y
        elif self._handle == "top":
            bottom = centre_y - height / 2.0
            new_height = max(0.1, y_mm - bottom)
            new_centre_y = bottom + new_height / 2.0
            new_centre_x = centre_x
            new_width = width
        else:  # bottom
            top = centre_y + height / 2.0
            new_height = max(0.1, top - y_mm)
            new_centre_y = top - new_height / 2.0
            new_centre_x = centre_x
            new_width = width

        if image.keep_aspect and width > 0 and height > 0:
            ratio = height / width
            if self._handle in ("left", "right"):
                new_height = new_width * ratio
                # Keep the vertical centre, so a horizontal drag does not
                # drift the placement up or down.
                new_centre_y = centre_y
            else:
                new_width = new_height / ratio if ratio else new_width
                new_centre_x = centre_x

        image.width_mm = new_width
        image.height_mm = new_height
        image.x_mm = new_centre_x
        image.y_mm = new_centre_y
        self.update()

    # --------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(CANVAS_BG))
        # Asked for once per frame rather than per drawing helper: with a
        # traced irregular board this is the one computed value worth not
        # computing twice. The view no longer frames from it -- the fit is a
        # stored scale -- but the board still has to be drawn.
        self._frame_outline = self.board.outline_polygon()
        self._recompute_plot()

        # The board is drawn on top of the artwork. It is a boundary rather
        # than a layer, and the setup panel previews the board through this
        # canvas: painted underneath an opaque picture, a board that sits
        # inside the artwork -- a circle inscribed in a square image, say --
        # would be completely hidden and the preview would show nothing.
        #
        # The selection is split around the artwork on purpose. Its fill is a
        # *backdrop* -- the object's extent has to be visible -- while its
        # frame and handles go on top. Painting the fill after the artwork
        # washed a grey over it, which made selecting an object change its
        # apparent colour and hid the very layer colour the canvas is supposed
        # to communicate.
        #
        # So: grid, then the axes over the artwork, then the geometry that is
        # not part of the drawing -- the selection frame, the handles and the
        # board outline.
        self._paint_grid(painter)
        if self.selected is not None:
            self._paint_selection_backdrop(painter)
        # Decorations, artwork and wording go down in **one** layer-ordered
        # pass. Painting them as three separate passes stacked the categories
        # rather than the layers: every line beat every picture, and the fixed
        # decorations always sat under both, whatever layer each was on.
        self._paint_board_content(painter)
        self._paint_cutouts(painter)
        self._paint_axes(painter)
        if self.selected is not None:
            self._paint_selection_frame(painter)
        self._paint_board(painter)
        self._paint_rulers(painter)

    #: Closest two grid lines may come on screen, in pixels. The step doubles
    #: until it clears this, so the grid thins out instead of turning into a
    #: solid wash when the user zooms out.
    GRID_MIN_PX = 14.0

    #: Every second grid line is a major one, and carries a tick and a number
    #: on the axes below. The rulers use the same rule.
    GRID_MAJOR_EVERY = 2

    def _grid_step(self) -> float:
        """Millimetres between adjacent grid lines at the current zoom.

        Shared by the grid, the axes' ticks and the rulers so that all three
        stay on the same lines. Computing it separately in each of them is how
        they come apart as the zoom changes -- and a tick that misses its grid
        line is worse than no tick.
        """
        step = 5.0
        while step * self._px_per_mm < self.GRID_MIN_PX:
            step *= 2
        return step

    def _paint_grid(self, painter: QPainter) -> None:
        plot = self._plot
        if plot.width() <= 10 or plot.height() <= 10:
            return
        step = self._grid_step()

        painter.save()
        painter.setClipRect(plot)
        span = max(plot.width(), plot.height()) / max(self._px_per_mm, 1e-6)
        count = int(span / step) + 2
        for index in range(-count, count + 1):
            mm = index * step
            major = index % self.GRID_MAJOR_EVERY == 0
            painter.setPen(QPen(QColor(GRID_MAJOR if major else GRID_MINOR), 1))
            x = self.to_screen(mm, 0).x()
            y = self.to_screen(0, mm).y()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.restore()

    def _paint_axes(self, painter: QPainter) -> None:
        """The X and Y axes through the board's origin.

        The grid says how big a thing is; the axes say *where zero is*, which
        no grid line can: a line every ten millimetres looks like every other
        line, and after a pan there is nothing on screen to say which one is
        the origin. The board is centred on the origin, so this is also the
        crosshair a designer expects to find in the middle of the sheet.

        Painted semi-transparent **over** the artwork. Underneath, the picture
        -- which is centred on the origin -- covers the origin marker and the
        middle of both axes, so the one case the axes exist for is the one
        where they would be invisible. The transparency keeps them an overlay
        on a ruler rather than ink on the board.

        Drawn twice: once in a dark halo, once in the light colour. A single
        colour cannot work here, because the artwork under the axes may be
        black ink or white paper -- or a photograph's worth of everything in
        between -- and light grey text on white paper is as unreadable as dark
        text on black. The halo gives every stroke a dark edge, so each one
        stays legible whichever it crosses.
        """
        plot = self._plot
        if plot.width() <= 10 or plot.height() <= 10:
            return
        origin = self.to_screen(0.0, 0.0)
        # Each axis is drawn only if it is on screen; panning far enough away
        # leaves the canvas as a plain grid rather than a line pinned to an
        # edge, which would read as a border rather than as an axis.
        on_x = plot.top() <= origin.y() <= plot.bottom()
        on_y = plot.left() <= origin.x() <= plot.right()
        if not (on_x or on_y):
            return

        step = self._grid_step()
        span = max(plot.width(), plot.height()) / max(self._px_per_mm, 1e-6)
        count = int(span / step) + 2

        def draw(colour: QColor, *, outlined: bool) -> None:
            painter.setPen(QPen(colour, 1.2))
            if on_x:
                painter.drawLine(
                    QPointF(plot.left(), origin.y()),
                    QPointF(plot.right(), origin.y()),
                )
            if on_y:
                painter.drawLine(
                    QPointF(origin.x(), plot.top()),
                    QPointF(origin.x(), plot.bottom()),
                )

            # Ticks on the major grid crossings, with the millimetre reading
            # beside each. The rulers carry the same numbers at the window's
            # edge; these are for the middle of the canvas, where the eye is.
            tick = 3.0
            for index in range(-count, count + 1):
                if index % self.GRID_MAJOR_EVERY or index == 0:
                    continue
                mm = index * step
                if on_x:
                    x = self.to_screen(mm, 0).x()
                    if plot.left() <= x <= plot.right():
                        painter.drawLine(
                            QPointF(x, origin.y() - tick),
                            QPointF(x, origin.y() + tick),
                        )
                if on_y:
                    y = self.to_screen(0, mm).y()
                    if plot.top() <= y <= plot.bottom():
                        painter.drawLine(
                            QPointF(origin.x() - tick, y),
                            QPointF(origin.x() + tick, y),
                        )

            # The origin itself: the crossing point marked off, so "0, 0" is a
            # place on the board rather than an inference from the numbers.
            if on_x and on_y:
                # Hollow on purpose: a filled marker would cover the very
                # crossing it is there to point at.
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(origin, 2.8, 2.8)

            # Positive direction, named at the end of each axis. Without the
            # arrows the axes are directionless lines, and which way Y grows is
            # exactly the question a millimetre grid raises.
            arrow = QPen(colour, 1.2)
            arrow.setJoinStyle(Qt.MiterJoin)
            painter.setPen(arrow)
            painter.setBrush(colour)
            if on_x:
                tip_x = plot.right() - 7
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(tip_x, origin.y()),
                            QPointF(tip_x - 8, origin.y() - 4),
                            QPointF(tip_x - 8, origin.y() + 4),
                        ]
                    )
                )
            if on_y:
                tip_y = plot.top() + 7
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(origin.x(), tip_y),
                            QPointF(origin.x() - 4, tip_y + 8),
                            QPointF(origin.x() + 4, tip_y + 8),
                        ]
                    )
                )

            # Text last, in both passes, so the halo of a label is never drawn
            # over the light glyph of another one.
            labels = []
            for index in range(-count, count + 1):
                if index % self.GRID_MAJOR_EVERY or index == 0:
                    continue
                mm = index * step
                if on_x:
                    x = self.to_screen(mm, 0).x()
                    if plot.left() <= x <= plot.right():
                        labels.append(
                            (QRectF(x - 22, origin.y() + 4, 44, 13),
                             Qt.AlignHCenter | Qt.AlignTop, f"{mm:.0f}")
                        )
                if on_y:
                    y = self.to_screen(0, mm).y()
                    if plot.top() <= y <= plot.bottom():
                        labels.append(
                            (QRectF(origin.x() - 34, y - 7, 30, 14),
                             Qt.AlignRight | Qt.AlignVCenter, f"{mm:.0f}")
                        )
            if on_x:
                labels.append(
                    (QRectF(plot.right() - 31, origin.y() - 18, 16, 14),
                     Qt.AlignRight | Qt.AlignVCenter, "X")
                )
            if on_y:
                labels.append(
                    (QRectF(origin.x() + 6, plot.top() + 13, 16, 14),
                     Qt.AlignLeft | Qt.AlignVCenter, "Y")
                )
            for rect, alignment, text in labels:
                self._axis_label(
                    painter, rect, alignment, text, colour=colour, outlined=outlined
                )

        painter.save()
        painter.setClipRect(plot)
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        # Two passes: a dark outline, then the light stroke on top of it. A
        # single colour cannot work here, because the artwork underneath may
        # be white paper, black ink, or anything in between.
        draw(_AXIS_HALO, outlined=True)
        draw(AXIS, outlined=False)
        painter.restore()

    @staticmethod
    def _axis_label(
        painter: QPainter,
        rect: QRectF,
        alignment,
        text: str,
        *,
        colour: QColor,
        outlined: bool,
    ) -> None:
        """One axis label, outlined when it is the halo pass.

        ``QPainterPath.addText`` is what makes the outline possible: drawing the
        text twice offset by a pixel would leave the halo lopsided, while
        stroking the same path keeps the halo centred on every glyph.

        This method is used by *both* passes, so ``outlined`` means "draw the
        halo". The halo's glyphs are filled as well as stroked, which is what
        the light pass on top of them needs -- left hollow, the light glyphs
        would land inside dark outlines of themselves and stay unreadable.
        """
        if not outlined:
            painter.setPen(colour)
            painter.drawText(rect, alignment, text)
            return

        path = QPainterPath()
        # Qt's text origin is the baseline, so the advance width and the font
        # metrics give the position the alignment implies.
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text)
        x = rect.x()
        if alignment & Qt.AlignHCenter:
            x = rect.center().x() - width / 2.0
        elif alignment & Qt.AlignRight:
            x = rect.right() - width
        y = rect.center().y() + metrics.ascent() / 2.0 - 1.0
        path.addText(QPointF(x, y), painter.font(), text)
        painter.setBrush(colour)
        painter.setPen(QPen(colour, 2.0))
        painter.drawPath(path)

    def _paint_board(self, painter: QPainter) -> None:
        if not self.board.layers.is_visible("outline"):
            # The outline follows its layer's visibility like anything else.
            # Drawing it regardless would make the layer panel's outline row
            # appear broken.
            return

        points = self._frame_outline
        if len(points) < 2:
            # No board yet: say so instead of leaving the canvas looking
            # finished, since the outline is now a step the user opens
            # deliberately rather than something importing a picture performs.
            if self.board.image is not None:
                painter.save()
                painter.setPen(QColor(MUTED))
                painter.drawText(
                    self._plot.adjusted(10, 8, 0, 0),
                    Qt.AlignLeft | Qt.AlignTop,
                    "尚未设置板框　·　工具栏「板框设置」",
                )
                painter.restore()
            return
        polygon = QPolygonF([self.to_screen(x, y) for x, y in points])

        painter.save()
        painter.setBrush(Qt.NoBrush)
        # The outline's own EasyEDA colour, not a canvas constant: the layer
        # panel shows that colour, so the two have to be the same or the panel
        # misreads.
        painter.setPen(QPen(QColor(layer("outline").color), 2))
        painter.drawPolygon(polygon)
        painter.restore()

    def _paint_empty_hint(self, painter: QPainter) -> None:
        """The message for a canvas with no artwork showing.

        A template card carries its own content across the middle of the plot,
        and the "no picture yet" hint has nowhere to go that is not on top of
        it -- the card fills the canvas at fit zoom, so there is no empty band
        above or below either. The status bar already says the same thing
        ("尚未载入图片"), so on a card the hint is dropped rather than drawn
        over the wording the user is trying to check.

        The "layers are hidden" case is kept: that condition is not in the
        status bar, and a board that looks empty for a reason the user cannot
        see is exactly the case worth speaking up about.
        """
        hidden = bool(self.board.placements)
        if not hidden and self.board.texts:
            return
        message = (
            "所有图层都已隐藏\n请在「图层」面板中打开图层"
            if hidden
            else "尚未放置图片\n请先点击「导入图片」"
        )
        painter.save()
        # Centred, and on a patch of bare canvas: the axes run through the
        # centre of the plot, so a message on a transparent background has the
        # X axis ruled through one line and the Y axis through both. The patch
        # is the background colour, so it reads as a gap in the grid rather
        # than as a card pasted onto the canvas.
        box = self._plot.adjusted(0, 40, 0, 40)
        text_box = painter.boundingRect(box, Qt.AlignCenter, message)
        painter.fillRect(text_box.adjusted(-8, -5, 8, 5), QColor(CANVAS_BG))
        painter.setPen(QColor(MUTED))
        painter.drawText(box, Qt.AlignCenter, message)
        painter.restore()

    def _draw_state(self, placement):
        """Everything one placement needs to be painted, or ``None``.

        ``(pixmap, target, local, source, draw_pixmap, opacity)``. Shared by
        the clipped draw and the framing ghost so the two cannot disagree about
        which part of the picture is shown or how strongly.
        """
        pixmap = self._pixmap_for(placement)
        if pixmap is None:
            return None
        design_layer = layer(placement.layer_key)
        target = self._image_target_rect(placement)
        # The layer's own transparency comes first, the object's opacity
        # second. Solder masks are recorded as semi-transparent in the layer
        # table, and without that a stacked board shows only the top layer: an
        # opaque mask rectangle covers the copper and silkscreen underneath it
        # completely.
        opacity = max(0.05, min(1.0, placement.opacity)) * max(
            0.05, min(1.0, design_layer.transparency)
        )
        local = QRectF(
            -target.width() / 2.0,
            -target.height() / 2.0,
            target.width(),
            target.height(),
        )
        # Only the framed part of the picture is drawn into the box. The window
        # is stored on the placement, so the canvas shows exactly what the
        # exporters will write -- the same reason the transform lives there.
        left, top, right, bottom = placement.source_rect(
            pixmap.width(), pixmap.height()
        )
        source = QRectF(
            left, top, max(1, right - left), max(1, bottom - top)
        )
        draw_pixmap = pixmap
        if placement.processed_path:
            draw_pixmap = self._tinted(pixmap, QColor(design_layer.color))
        elif self._is_generated_ink(placement):
            # Black-on-transparent generated artwork (the operator codes): its
            # alpha is the ink, so it needs the layer colour to be visible on
            # the dark canvas.
            draw_pixmap = self._tinted_by_alpha(
                pixmap, QColor(design_layer.color)
            )
        return pixmap, target, local, source, draw_pixmap, opacity

    def _paint_framing_ghosts(self, painter: QPainter) -> None:
        """The part of the framed picture that the crop cuts away.

        A drag while framing pans the picture *inside* its window, and with
        everything outside invisible there is nothing on screen to judge the
        pan against. Faint rather than full strength, so the window still reads
        as what will actually be printed.

        Every tone of the picture gets a ghost, not just the one that was
        clicked: a 板画 is two placements sharing one crop, and dimming one half
        would leave the other at full strength outside the window.

        ⚠️ **Only during an explicit framing session -- not for every picture
        that sits in an opening.** Tying it to selection was tried and reverted:
        the automatic import leaves the picture selected, so the card came up
        with the whole illustration ghosted and an amber border on it, which is
        the 「留在取景中」 state the user asked to be rid of. Right-clicking
        放入短区/长区 starts a session for anyone who wants the offcuts visible
        while judging a pan.
        """
        framing = self.framing
        if framing is None:
            return
        visible = self.board.visible_placements()
        for item in self.linked(framing):
            if not any(item is placement for placement in visible):
                continue
            state = self._draw_state(item)
            if state is None:
                continue
            pixmap, target, local, source, draw_pixmap, opacity = state
            ghost = _full_image_rect(
                local, source, pixmap.width(), pixmap.height()
            )
            if ghost is None:
                continue
            painter.save()
            self._apply_image_transform(painter, item, target)
            painter.setOpacity(opacity * FRAME_GHOST_OPACITY)
            painter.drawPixmap(
                ghost,
                draw_pixmap,
                QRectF(0.0, 0.0, pixmap.width(), pixmap.height()),
            )
            painter.restore()

    @staticmethod
    def _apply_image_transform(
        painter: QPainter, placement, target: QRectF
    ) -> None:
        """Put the painter in the picture's own frame: mirror, rotate, place.

        旋转 and 翻转 were stored, shown in the property panel and then ignored
        by the canvas, so the artwork never actually turned over. The transform
        mirrors the placement's own definition (see ``ImagePlacement.matrix``):
        mirror in the picture's frame first, then rotate, then place. Qt's Y
        axis points down while the design frame points up, so the angle is
        negated to keep "positive rotation" turning the same way the selection
        outline does.

        Shared by the clipped draw and the framing ghost: a second copy of
        this rule is how the ghost and the picture it belongs to would come
        apart.
        """
        painter.translate(target.center())
        painter.rotate(-placement.rotation_deg)
        painter.scale(
            -1.0 if placement.flip_horizontal else 1.0,
            -1.0 if placement.flip_vertical else 1.0,
        )

    def _paint_single(self, painter: QPainter, placement) -> None:
        """One placement: the whole picture, with what will not print dimmed.

        A processed placement is drawn in its layer's colour, the way the
        editor paints a layer. The image itself is strictly black and white,
        so recolouring it gives the layer's ink without the artwork's own
        colours leaking through.

        ⚠️ **The picture outside the illustration opening is drawn *dim*, not
        omitted.** It used to be clipped away entirely, which made an import
        look as though the rest of the drawing had been deleted -- reported as
        「其他部分直接被删除了」. The opening decides what *prints*; it does not
        decide what the user is allowed to see while arranging the picture.

        Dimming rather than hiding keeps both readings honest at once:

        * the user can see and drag the whole figure, including the part that
          must be moved into the opening;
        * the bright part is still exactly what will be manufactured, so the
          canvas never promises ink the board will not carry.

        ⚠️ **When the picture has left its opening entirely it is drawn in the
        layer's own darkened colour** (顶层 → 暗红, 顶层丝印 → 深黄, …). That is
        the warning the user asked for: the picture contributes nothing to the
        board.
        """
        design_layer = layer(placement.layer_key)
        polygon = self.image_polygon(placement)
        state = self._draw_state(placement)
        region_bounds = self._region_bounds(placement)
        region_path = self._region_path(placement)
        outside = (
            region_bounds is not None
            and not placement.overlaps_rect(region_bounds)
        )
        ink = (
            QColor(self._warning_colour(QColor(design_layer.color)))
            if outside
            else QColor(design_layer.color)
        )

        painter.save()
        if state is not None:
            _pixmap, target, local, source, draw_pixmap, opacity = state
            if outside:
                draw_pixmap = self._tinted(draw_pixmap, ink)

            if region_path is None:
                # Not on a card: nothing to dim, the picture is its own clip.
                painter.setClipPath(self._path_from(polygon))
                painter.setOpacity(opacity)
                self._apply_image_transform(painter, placement, target)
                painter.drawPixmap(local, draw_pixmap, source)
            else:
                # 1. The part that prints, clipped to the opening.
                painter.save()
                painter.setClipPath(region_path)
                painter.setOpacity(opacity)
                self._apply_image_transform(painter, placement, target)
                painter.drawPixmap(local, draw_pixmap, source)
                painter.restore()

                # 2. The rest of the picture, dimmed.
                #
                # ⚠️ **This has to draw the *whole* picture, not the crop.**
                # ``local`` and ``source`` describe the framing window scaled to
                # fill the placement box, so drawing them again through the
                # complement of the opening paints nothing -- the window *is*
                # the box, so there is no offcut inside it. That is why a
                # first attempt here looked correct in code and rendered
                # identically to not drawing at all.
                #
                # ``_full_image_rect`` is the same extrapolation the framing
                # ghost uses: it places the whole picture where it would land if
                # the window covered all of it. Drawing that through the
                # complement puts the offcut around the opening.
                #
                # ⚠️ **Skipped while a framing session is running**, because
                # ``_paint_framing_ghosts`` already draws the whole picture for
                # that placement -- once, below every object. Drawing it here
                # too would put the same translucent pixels down twice and make
                # the offcut noticeably darker during a session than outside
                # one, which reads as the artwork changing when the user starts
                # framing.
                if self._is_framing_session(placement):
                    pass
                else:
                    ghost = _full_image_rect(
                        local, source, _pixmap.width(), _pixmap.height()
                    )
                    if ghost is not None:
                        offcut = self._path_from(polygon).subtracted(region_path)
                        if not offcut.isEmpty():
                            painter.save()
                            painter.setClipPath(offcut)
                            painter.setOpacity(opacity * FRAME_GHOST_OPACITY)
                            self._apply_image_transform(painter, placement, target)
                            painter.drawPixmap(
                                ghost,
                                draw_pixmap,
                                QRectF(0.0, 0.0, _pixmap.width(), _pixmap.height()),
                            )
                            painter.restore()
        else:
            # No preview available: show a placeholder rather than nothing, so
            # the placement is still visible and editable.
            painter.setBrush(QColor(PLACEHOLDER_FILL))
            painter.setPen(QPen(QColor(BORDER), 1))
            painter.drawPolygon(polygon)
        painter.restore()

        # The placement's outline takes the layer's colour, so which layer an
        # object is on is readable without opening the property panel. The
        # outline itself is **not** clipped: it is the object's own edge, and
        # the user needs it to see where the picture went.
        painter.save()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(
            QPen(
                ink if outside else QColor(design_layer.color),
                2 if outside else 1,
                Qt.DashLine,
            )
        )
        painter.drawPolygon(polygon)
        painter.restore()

        if outside:
            painter.save()
            painter.setPen(QPen(ink))
            painter.drawText(
                self.to_screen(
                    placement.x_mm - placement.width_mm / 2.0,
                    placement.y_mm + placement.height_mm / 2.0,
                )
                + QPointF(4.0, -6.0),
                "已移出立绘区，不会印出",
            )
            painter.restore()

        # While a framing session is running, say plainly that this box is the
        # opening being filled and that a drag pans the picture inside it rather
        # than moving the object. Without it the gesture looks like the object
        # refusing to move.
        #
        # ⚠️ **Only during the explicit session.** A picture moved freely around
        # the card keeps an ordinary drag, so marking every framed picture would
        # describe the wrong gesture and put the 「取景中」 look on imports the
        # user did not ask to frame.
        if self.framing is placement:
            painter.save()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(FRAME_EDGE), 2, Qt.SolidLine))
            painter.drawPolygon(polygon)
            painter.setPen(QPen(QColor(FRAME_EDGE)))
            painter.drawText(
                self.to_screen(
                    placement.x_mm - placement.width_mm / 2.0,
                    placement.y_mm + placement.height_mm / 2.0,
                )
                + QPointF(4.0, -6.0),
                "取景中：拖动可移动图片　·　Ctrl+滚轮缩放",
            )
            painter.restore()

    def _paint_board_content(self, painter: QPainter) -> None:
        """Everything printed on the board, bottom layer first.

        The three sources -- the template's fixed decorations, the imported
        artwork and the editable wording -- are one visual stack, so they are
        merged and sorted by :func:`paint_rank` rather than drawn in three
        passes. Sorting each source internally is not enough: it fixes the
        order *within* a category while leaving the categories themselves
        stacked by kind.

        The framing ghost is deliberately left out of this ordering: it is a
        full-picture overlay with no layer of its own, and it goes down before
        every object so it cannot wash over any of them.
        """
        from pcbart import badge_art

        from pcbart.design_layers import paint_rank

        # ⚠️ **Decorations are grouped into one draw call per layer, not one per
        # item.** The card carries 324 of them, and drawing each inside its own
        # ``save`` / ``setClipPath`` / ``restore`` cost 4935 clip installs per
        # repaint -- the clip path and the layer's opacity are the same for
        # every decoration on a layer, so installing them once per layer is the
        # same picture for a fraction of the work. Measured: the board-content
        # phase went from ~138 ms to ~26 ms per frame.
        entries: list[tuple[int, int, str, object]] = []
        order = 0
        for placement in self.board.visible_placements():
            entries.append(
                (paint_rank(placement.layer_key), order, "image", placement)
            )
            order += 1
        for text in self.board.visible_texts():
            entries.append((paint_rank(text.layer_key), order, "text", text))
            order += 1

        # ``(rank, layer key, layer id)`` -> the decorations on it, in source
        # order. A dict keyed this way keeps the grouping and the painting
        # order in one pass.
        decorations: dict[tuple[int, str, int], list] = {}
        for item in badge_art.decorations_for(self.board):
            layer_id = int(item.get("layer", 0))
            key = badge_art.layer_key_for_id(layer_id)
            if key and not self.board.layers.is_visible(key):
                continue
            decorations.setdefault(
                (paint_rank(key), key, layer_id), []
            ).append(item)

        if not entries and not decorations:
            return
        if not self.board.visible_placements():
            # Nothing of the user's is showing. On a plain board that is the
            # "import a picture" hint; on the card the wording alone is content
            # and is drawn as normal.
            self._paint_empty_hint(painter)

        # Everything goes down in layer order. The decorations on a layer are
        # one step in that order, drawn before any image or text on the same
        # layer -- within a layer they are the background, and the source file
        # stores them that way.
        merged: list[tuple[int, int, str, object]] = list(entries)
        for index, ((rank, _key, _layer_id), items) in enumerate(
            sorted(decorations.items(), key=lambda pair: pair[0][0])
        ):
            merged.append((rank, -len(decorations) + index, "decorations", items))
        merged.sort(key=lambda entry: (entry[0], entry[1]))

        # The fixed decorations are clipped to the board: a primitive that runs
        # past the cut line (the source file has several) must not paint on the
        # surrounding sheet. The clip is installed for the decoration draws
        # only -- the user's own artwork is not subject to it.
        outline = self.board.outline_polygon()
        clip = QPainterPath()
        if outline:
            clip.addPolygon(
                QPolygonF([self.to_screen(x, y) for x, y in outline])
            )
            clip.closeSubpath()

        self._paint_framing_ghosts(painter)
        for _rank, _order, kind, item in merged:
            if kind == "decorations":
                painter.save()
                if not clip.isEmpty():
                    painter.setClipPath(clip, Qt.IntersectClip)
                self._paint_decoration_group(painter, item)
                painter.restore()
            elif kind == "image":
                self._paint_single(painter, item)
            else:
                self._paint_text(painter, item)

    def _paint_decoration_group(self, painter: QPainter, items) -> None:
        """Every decoration on one layer, with the layer's paint state set once.

        ⚠️ **One ``to_screen`` result per vertex, reused on every repaint.**
        Mapping the card's ~50 000 decoration vertices is the single largest
        cost in a drag frame, and it was being redone from scratch each frame
        even though the view has not moved. The mapped screen polygons are
        cached against the view transform; a drag moves an object, not the
        view, so the second and later frames reuse them.
        """
        from pcbart import badge_art

        layer_id = int(items[0].get("layer", 0))
        key = badge_art.layer_key_for_id(layer_id)
        design_layer = layer(key) if key else None
        colour = QColor(design_layer.color if design_layer else "#c0c0c0")
        painter.setOpacity(
            max(0.05, min(1.0, design_layer.transparency))
            if design_layer
            else 1.0
        )
        cached = self._decoration_screens(layer_id, items)
        for item, screen in zip(items, cached):
            if item.get("kind") == "fill":
                path = QPainterPath()
                for polygon in screen:
                    path.addPolygon(polygon)
                    path.closeSubpath()
                path.setFillRule(Qt.OddEvenFill)
                painter.fillPath(path, colour)
                continue

            width = max(1.0, float(item.get("width_mm", 0.254)) * self._px_per_mm)
            pen = QPen(colour, width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for points in screen:
                if points.count() >= 2:
                    painter.drawPolyline(points)
        painter.setOpacity(1.0)

    def _decoration_screens(self, layer_id: int, items) -> list:
        """Screen-space geometry for a layer's decorations, cached per view.

        The cache key is the transform itself -- plot origin, pan, scale -- so
        it is automatically invalidated by any view change and by nothing else.
        That is the point: a drag changes the artwork, not the view, and the
        mapped card is exactly what a drag should *not* recompute.
        """
        key = (layer_id, len(items))
        # ⚠️ **Clear before looking up, not after a miss.** Clearing on every
        # miss throws away the entries the *other* layers just stored, so each
        # layer missed again on the next frame and the cache never hit -- the
        # first version of this measured no faster than no cache at all. The
        # transform is the same for every layer, so one comparison is enough to
        # know whether the whole cache is stale.
        if self._decoration_view != self._view_key:
            self._decoration_cache.clear()
            self._decoration_view = self._view_key

        hit = self._decoration_cache.get(key)
        if hit is not None:
            return hit

        mapped = []
        for item in items:
            if item.get("kind") == "fill":
                mapped.append(
                    [
                        QPolygonF([self.to_screen(x, y) for x, y in ring])
                        for ring in item.get("rings") or []
                    ]
                )
            else:
                mapped.append(
                    [
                        QPolygonF([self.to_screen(x, y) for x, y in line])
                        for line in item.get("paths") or []
                    ]
                )
        self._decoration_cache[key] = mapped
        return mapped

    def _paint_cutouts(self, painter: QPainter) -> None:
        """Every hole in the board, as a red outline over a darkened opening.

        Drawn even on a hidden outline layer: a cutout is a property of the
        board rather than a drawing on it, and a slot that vanished with the
        outline layer would look like it had been deleted.
        """
        for item in self.board.cutouts:
            ring = [self.to_screen(x, y) for x, y in item.ring()]
            polygon = QPolygonF(ring)
            painter.save()
            painter.setBrush(self._cutout_brush(painter, polygon))
            painter.setPen(QPen(QColor(CUTOUT_EDGE), 1.6, Qt.DashLine))
            painter.drawPolygon(polygon)
            painter.restore()

    @staticmethod
    def _cutout_brush(painter: QPainter, polygon: QPolygonF) -> QBrush:
        """The fill for a hole: darker than whatever is behind it.

        Composited against the canvas rather than set to a flat dark, so the
        hole reads as an opening on black and does not turn into a pale slab if
        the canvas colour is ever changed again.
        """
        brush = QBrush(CUTOUT_FILL)
        brush.setStyle(Qt.SolidPattern)
        return brush

    def _paint_text(self, painter: QPainter, item) -> None:
        """One template line, in the layer colour of the layer it is on.

        Drawn with the anchor at the stored point and its own rotation, so what
        the user sees lines up with where the exporters will put it. The font is
        the canvas's own: the families the template names (``Agency FB``,
        ``Bahnschrift``) are not on every machine, and substituting a family
        still puts the line in the right place at the right size -- whereas
        skipping a line whose font is missing would silently drop card content.
        """
        design_layer = layer(item.layer_key)
        origin = self.to_screen(item.x_mm, item.y_mm)
        text = item.resolved_text(
            name=self.board.operator_name, number=self.board.operator_number
        )
        if not text:
            return
        height_px = max(1.0, item.size_mm * self._px_per_mm)

        painter.save()
        painter.translate(origin)
        # Qt's Y axis runs down while the design frame runs up, so the angle is
        # negated for the same reason the artwork's rotation is.
        painter.rotate(-item.rotation_deg)
        # The template's own face, so the on-screen glyphs are the size the
        # exported ones will be -- see :func:`template_font`.
        font = template_font(item.font, text)
        font.setPixelSize(max(1, int(round(height_px))))
        painter.setFont(font)

        # ⚠️ **The template's LEFT_BOTTOM anchor is the line box's bottom, not
        # the baseline.** Qt draws from the baseline, so the pen has to be
        # lifted by one descender; the exporters apply the same shift in
        # ``pcbart.text_render``. Applying it in only one of the two is what
        # made the preview and the board disagree.
        from pcbart.text_render import line_box_descent_mm

        descent_px = line_box_descent_mm(item.size_mm, item.font, text) * self._px_per_mm

        if self.board.outline.shape == "template" and not self.board.operator_name:
            if "{代号}" in item.text or "{编号}" in item.text:
                metrics = painter.fontMetrics()
                width = metrics.horizontalAdvance(text)
                painter.fillRect(
                    QRectF(
                        0.0,
                        -metrics.ascent() - descent_px,
                        width,
                        metrics.height(),
                    ),
                    TOKEN_HIGHLIGHT,
                )

        # Painted twice -- a dark outline under a light fill -- for the same
        # reason the axis labels are: template text sits on top of the artwork
        # and crosses whatever colours the picture happens to have, and one
        # flat colour cannot be read on both white paper and black ink.
        path = QPainterPath()
        path.addText(0.0, -descent_px, font, text)
        painter.setPen(QPen(_AXIS_HALO, max(1.0, height_px * 0.10)))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(design_layer.color))
        painter.drawPath(path)
        painter.restore()

        if item is self.selected:
            self._paint_text_selection(painter, item)

    def _paint_text_selection(self, painter: QPainter, item) -> None:
        """The selected line's frame, matching the artwork's selection style."""
        painter.save()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(SELECTION_EDGE, 1.5, Qt.DashLine))
        painter.drawPolygon(
            QPolygonF([self.to_screen(x, y) for x, y in item.corners()])
        )
        painter.restore()

    def _tinted(self, pixmap: QPixmap, color: QColor) -> QPixmap:
        """The line art recoloured to ``color``, with the paper knocked out.

        The line art is black ink on white paper, fully opaque. Filling it with
        a colour outright would paint the whole rectangle -- the white paper
        included -- which is what a naive tint does, because a greyscale PNG
        has alpha 255 everywhere. So the ink is turned into an alpha channel:
        dark becomes opaque, white becomes transparent, and the layer's colour
        shows only through the ink. That is what the editor displays.

        Built with numpy rather than with ``CompositionMode_DestinationIn``:
        that composition applies the mask unevenly on this data (the paper
        stayed opaque), and the explicit version is easier to reason about.
        The buffer is copied into the QImage, which does not take ownership.

        Cached by pixmap and colour: this is a full-size image operation and
        panning would otherwise recompute it on every frame.
        """
        key = f"{id(pixmap)}:{color.name()}"
        cached = self._tinted_cache.get(key)
        if cached is not None:
            return cached

        image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        width, height = image.width(), image.height()
        if width <= 0 or height <= 0:
            return pixmap
        stride = image.bytesPerLine()
        grey = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
        grey = grey.reshape(height, stride)[:, :width]

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = color.red()
        rgba[:, :, 1] = color.green()
        rgba[:, :, 2] = color.blue()
        # Ink is dark, so the alpha is the inverse of the grey level. Scaling by
        # the layer's transparency keeps the mask layers see-through, as the
        # editor shows them.
        rgba[:, :, 3] = ((255 - grey) * color.alphaF()).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        tinted = QPixmap.fromImage(
            QImage(
                rgba.data,
                width,
                height,
                rgba.strides[0],
                QImage.Format_RGBA8888,
            ).copy()
        )
        self._tinted_cache[key] = tinted
        return tinted

    def _tinted_by_alpha(self, pixmap: QPixmap, color: QColor) -> QPixmap:
        """Recolour an image whose own alpha channel is its ink mask.

        The badge's QR symbol is **black on transparent** and deliberately
        skips the line-art pipeline (tracing it would turn a readable symbol
        into an unreadable one). Drawing it raw put black modules on the black
        canvas, so the symbol was invisible -- an empty selection diamond with
        no code in it. Their alpha is the ink, so the layer colour replaces the
        black instead of being derived from how dark each pixel is.

        :meth:`_tinted` cannot be used for this: its mask is "how dark is the
        pixel", and a transparent pixel is black, so the whole quiet zone would
        come out opaque.
        """
        key = f"alpha:{id(pixmap)}:{color.name()}"
        cached = self._tinted_cache.get(key)
        if cached is not None:
            return cached

        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        if width <= 0 or height <= 0:
            return pixmap
        stride = image.bytesPerLine()
        raw = np.frombuffer(
            image.constBits(), dtype=np.uint8, count=stride * height
        ).reshape(height, stride)
        pixels = raw[:, : width * 4].reshape(height, width, 4)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = color.red()
        rgba[:, :, 1] = color.green()
        rgba[:, :, 2] = color.blue()
        rgba[:, :, 3] = (pixels[:, :, 3] * color.alphaF()).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        tinted = QPixmap.fromImage(
            QImage(
                rgba.data,
                width,
                height,
                rgba.strides[0],
                QImage.Format_RGBA8888,
            ).copy()
        )
        self._tinted_cache[key] = tinted
        return tinted

    def _is_generated_ink(self, placement) -> bool:
        """Whether ``placement`` holds a generated code whose alpha is its ink.

        **Three of the card's four generated elements qualify.** All of them
        are drawn as ``alpha = the shape``, with white RGB and no
        ``processed_path`` -- the line-art pipeline would clean a QR symbol into
        an unreadable pattern -- so the canvas has to recolour them by alpha to
        make them visible on the dark board:

        * the QR symbol and the barcode -- black ink on transparent;
        * the profession plate -- a filled square with the glyph punched out;
        * the faction logo -- a filled triangle with the crest punched out
          (measured: 255 in the plate, 0 through the castle and the lettering).

        The four share the badge's ``import_id`` because one button inserts
        them, which is why the ``kind`` is what the test keys on.

        ⚠️ **``kind`` is empty on every other placement**, including the ones
        the property panel creates and the ones an older draft saved before the
        field existed -- those fall through to the plain ``drawPixmap`` and keep
        their own colours.
        """
        if not getattr(placement, "import_id", ""):
            return False
        from pcbart.badge_content import ART_IMPORT_ID

        if placement.import_id != ART_IMPORT_ID:
            return False
        return getattr(placement, "kind", "") in (
            "qr",
            "barcode",
            "profession",
            "faction",
        )

    def _paint_selection_backdrop(self, painter: QPainter) -> None:
        """A light grey panel *behind* the selected artwork.

        Behind, not over: it exists so the extent of pale artwork is obvious,
        and drawing it on top instead recoloured the object every time it was
        selected.
        """
        polygon = self.image_polygon()
        if not polygon.count():
            return
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(SELECTION_FILL))
        painter.drawPath(self._path_from(polygon))
        painter.restore()

    def _paint_selection_frame(self, painter: QPainter) -> None:
        """The selection outline and its four edge handles."""
        polygon = self.image_polygon()
        if not polygon.count():
            return
        painter.save()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(SELECTION_EDGE, 1.5))
        painter.drawPath(self._path_from(polygon))
        painter.restore()

        painter.save()
        painter.setPen(QPen(SELECTION_EDGE, 1.5))
        painter.setBrush(QColor(CANVAS_BG))
        for centre in self.handle_centres().values():
            painter.drawPolygon(self._triangle(centre))
        painter.restore()

    @staticmethod
    def _path_from(polygon: QPolygonF) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()
        return path

    @staticmethod
    def _triangle(centre: QPointF, size: float = HANDLE_SIZE) -> QPolygonF:
        half = size / 2.0
        return QPolygonF(
            [
                QPointF(centre.x() - half, centre.y() - half),
                QPointF(centre.x() + half, centre.y() - half),
                QPointF(centre.x(), centre.y() + half),
            ]
        )

    def _paint_rulers(self, painter: QPainter) -> None:
        ruler = 22
        painter.fillRect(QRectF(0, 0, self.width(), ruler), QColor(RULER_BG))
        painter.fillRect(QRectF(0, 0, ruler, self.height()), QColor(RULER_BG))
        painter.setPen(QColor(BORDER))
        painter.drawLine(0, ruler, self.width(), ruler)
        painter.drawLine(ruler, 0, ruler, self.height())

        step = self._grid_step()

        painter.setPen(QColor(MUTED))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        plot = self._plot
        span = max(plot.width(), plot.height()) / max(self._px_per_mm, 1e-6)
        count = int(span / step) + 2
        for index in range(-count, count + 1):
            if index % self.GRID_MAJOR_EVERY:
                continue
            mm = index * step
            x = self.to_screen(mm, 0).x()
            y = self.to_screen(0, mm).y()
            if plot.left() <= x <= plot.right():
                painter.drawLine(QPointF(x, ruler - 4), QPointF(x, ruler))
                painter.drawText(QPointF(x + 2, ruler - 6), f"{mm:.0f}")
            if plot.top() <= y <= plot.bottom():
                painter.drawLine(QPointF(ruler - 4, y), QPointF(ruler, y))
                painter.drawText(QPointF(3, y + 8), f"{mm:.0f}")

        painter.setPen(QColor(MUTED))
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QRectF(plot.left(), plot.bottom() - 18, plot.width(), 16),
            Qt.AlignRight,
            "单位 mm",
        )
