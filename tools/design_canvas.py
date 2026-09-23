"""The design canvas: board outline plus the placed image, with editing.

Replaces the earlier stand-in canvas, which drew a fixed circle. Geometry now
comes from :class:`pcbart.artboard.ArtBoard`, so the board shape and the image
placement are whatever the user configured rather than decoration.

Editing follows the editor conventions the user asked for:

* click the image to select it, which shows a light grey panel behind it and
  four triangular handles on the edges;
* drag inside the image to move it (with a short press-and-hold, so a passing
  click does not shift it);
* drag a handle to resize from that edge.

The keyboard and right-click entries follow the same conventions an EDA tool
uses -- copy, paste, cut, delete, duplicate, nudge with the arrow keys, rotate
with ``R``. Only operations this tool actually has are bound: there is no
undo stack, no multi-selection and no z-order to move things through, so those
keys do nothing rather than pretending.
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

# Painting colours, matching the palette used by the surrounding application.
CANVAS_BG = "#ffffff"
GRID_MINOR = "#e8ecf0"
GRID_MAJOR = "#d3dae1"
RULER_BG = "#f7f8fa"
BORDER = "#ccd2da"
MUTED = "#6b7480"
OUTLINE = "#c026d3"

SELECTION_FILL = QColor(200, 206, 214, 150)
SELECTION_EDGE = QColor(60, 72, 88)


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

        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._mode = ""
        self._press_pos = QPointF()
        self._press_mm = (0.0, 0.0)
        self._press_size = (0.0, 0.0)
        self._press_centre = (0.0, 0.0)
        self._handle = ""
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
        """Copied artwork, in placement order, or empty.

        Holds *copies* rather than references: the user may delete or edit the
        original after copying it, and what gets pasted has to be what was on
        the clipboard at the moment of the copy.
        """

        # Visible geometry, recomputed on every paint.
        self._plot = QRectF()
        self._px_per_mm = 1.0
        self._fit_px_per_mm = 1.0
        self._frame_outline: list[tuple[float, float]] = []
        """The board outline for the frame being painted, computed once."""

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
        """Fit the viewport, with room for the rulers.

        The board drives the fit once it exists. Before that the artwork does,
        because an imported picture on a boardless canvas has to be visible
        and editable on its own -- there is nothing else to frame.

        The fitted scale is only the base; the user's zoom and pan are applied
        on top of it, so "fit" stays a meaningful thing to return to.
        """
        ruler = 22
        margin = 26
        plot = QRectF(
            ruler + margin,
            ruler + margin,
            max(10.0, self.width() - ruler - 2 * margin),
            max(10.0, self.height() - ruler - 2 * margin),
        )
        self._plot = plot

        points = self._frame_outline
        if not points:
            points = self._image_bounds_mm()
        if not points:
            # Nothing to frame yet: the canvas is empty because the picture is
            # waiting to be imported. Falling through to 1 px/mm would draw a
            # grid spanning hundreds of millimetres, which reads as a bug
            # rather than as an empty canvas. A blank board's scale is chosen
            # so a typical board's worth of the grid fills the viewport.
            default_span_mm = 100.0
            self._fit_px_per_mm = min(
                plot.width(), plot.height()
            ) / default_span_mm
        else:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            # A zero span (a single point, or an axis-aligned degenerate
            # shape) has no scale of its own; the other axis still does.
            span_x = max(max(xs) - min(xs), 1e-6)
            span_y = max(max(ys) - min(ys), 1e-6)
            self._fit_px_per_mm = min(
                plot.width() / span_x, plot.height() / span_y
            )

        self._px_per_mm = self._fit_px_per_mm * self._zoom

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
            after_mm = self.to_mm(anchor)
            self._pan_mm[0] += before_mm[0] - after_mm[0]
            self._pan_mm[1] += before_mm[1] - after_mm[1]
        else:
            self._zoom = after
            self._px_per_mm = self._fit_px_per_mm * self._zoom
        self.update()

    def pan_by_pixels(self, dx: float, dy: float) -> None:
        """Move the view by a screen-space delta.

        The geometry is untouched; only the window onto it moves.
        """
        scale = max(self._px_per_mm, 1e-9)
        self._pan_mm[0] -= dx / scale
        self._pan_mm[1] += dy / scale
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
        self.update()

    def to_screen(self, x_mm: float, y_mm: float) -> QPointF:
        centre = self._plot.center()
        return QPointF(
            centre.x() + (x_mm - self._pan_mm[0]) * self._px_per_mm,
            centre.y() - (y_mm - self._pan_mm[1]) * self._px_per_mm,
        )

    def to_mm(self, point: QPointF) -> tuple[float, float]:
        centre = self._plot.center()
        return (
            (point.x() - centre.x()) / self._px_per_mm + self._pan_mm[0],
            (centre.y() - point.y()) / self._px_per_mm + self._pan_mm[1],
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
        source_ratio = pixmap.width() / max(1, pixmap.height())
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
        """Screen positions of the four edge handles."""
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

        A board painting is **two** placements -- the white tone on silkscreen
        and the black on copper -- and the exporter writes them as a single
        geometry derived from the white one. Handling the tones separately
        tears the exported board along their seam, which is why the property
        panel already carries 翻转/旋转 across the pair. Copying, deleting and
        moving have exactly the same need, so they ask here.

        Line art is *not* paired: a picture spread over three layers really is
        three objects, each with its own size and position, and the user picks
        and edits them individually.
        """
        if placement is None:
            return []
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
        self.update()
        return True

    @property
    def clipboard_size(self) -> int:
        """How many objects a paste would add; 0 disables the menu entry."""
        return len(self._clipboard)

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
        hit = self.board.placement_at(x_mm, y_mm)
        if hit is not None:
            if hit is not self.selected:
                self.selected = hit
                self.selection_changed.emit(hit)
                self.update()
            # Movement is armed but not started: a click that merely selects
            # must not nudge the artwork. The drag begins on the next move.
            self._mode = "move"
            self._press_pos = position
            self._press_mm = (x_mm, y_mm)
            self._press_centre = (hit.x_mm, hit.y_mm)
            self._pending_move = True
            event.accept()
            return

        if self.selected is not None:
            self.selected = None
            self.selection_changed.emit(None)
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
            target = self.selected
            if target is None:
                return
            delta = position - self._press_pos
            target.x_mm = self._press_centre[0] + delta.x() / self._px_per_mm
            target.y_mm = self._press_centre[1] - delta.y() / self._px_per_mm
            self.update()
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
        if self._mode in ("move", "resize") and not self._pending_move:
            self.placement_changed.emit()
        self._mode = ""
        self._handle = ""
        self._pending_move = False
        self._update_cursor(event.position())

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom the view, anchored at the pointer.

        The canvas is the one place a wheel *should* do something: this is a
        view, not a value. The rule against wheel input applies to editable
        controls, whose value the user cannot see changing.
        """
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        self.zoom_by(1.15 ** steps, event.position())
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
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
        # The outline is asked for once per frame and shared. Both the framing
        # and the drawing need it, and with a traced irregular board it is the
        # one computed value worth not computing twice.
        self._frame_outline = self.board.outline_polygon()
        self._recompute_plot()

        # The board is drawn last, on top of the artwork. It is a boundary
        # rather than a layer, and the setup panel previews the board through
        # this canvas: painted underneath an opaque picture, a board that sits
        # inside the artwork -- a circle inscribed in a square image, say --
        # would be completely hidden and the preview would show nothing.
        #
        # The selection is split around the artwork on purpose. Its fill is a
        # *backdrop* -- pale artwork needs something behind it to read against
        # -- while its frame and handles go on top. Painting the fill after the
        # artwork washed a 60% grey over it, which made selecting an object
        # change its apparent colour and hid the very layer colour the canvas
        # is supposed to communicate.
        self._paint_grid(painter)
        if self.selected is not None:
            self._paint_selection_backdrop(painter)
        self._paint_image(painter)
        if self.selected is not None:
            self._paint_selection_frame(painter)
        self._paint_board(painter)
        self._paint_rulers(painter)

    def _paint_grid(self, painter: QPainter) -> None:
        plot = self._plot
        if plot.width() <= 10 or plot.height() <= 10:
            return
        step = 5.0
        while step * self._px_per_mm < 14:
            step *= 2

        painter.save()
        painter.setClipRect(plot)
        span = max(plot.width(), plot.height()) / max(self._px_per_mm, 1e-6)
        count = int(span / step) + 2
        for index in range(-count, count + 1):
            mm = index * step
            major = index % 2 == 0
            painter.setPen(QPen(QColor(GRID_MAJOR if major else GRID_MINOR), 1))
            x = self.to_screen(mm, 0).x()
            y = self.to_screen(0, mm).y()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.restore()

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

    def _paint_image(self, painter: QPainter) -> None:
        """Draw every placement on a visible layer, tinted by its layer."""
        visible = self.board.visible_placements()
        if not visible:
            message = (
                "所有图层都已隐藏\n请在「图层」面板中打开图层"
                if self.board.placements
                else "尚未放置图片\n请先点击「导入图片」"
            )
            painter.save()
            painter.setPen(QColor(MUTED))
            painter.drawText(self._plot, Qt.AlignCenter, message)
            painter.restore()
            return

        for placement in visible:
            self._paint_single(painter, placement)

    def _paint_single(self, painter: QPainter, placement) -> None:
        """One placement: the picture, clipped to its rectangle.

        A processed placement is drawn in its layer's colour, the way the
        editor paints a layer. The image itself is strictly black and white,
        so recolouring it gives the layer's ink without the artwork's own
        colours leaking through.
        """
        design_layer = layer(placement.layer_key)
        polygon = self.image_polygon(placement)
        pixmap = self._pixmap_for(placement)

        painter.save()
        painter.setClipPath(self._path_from(polygon))
        if pixmap is not None:
            target = self._image_target_rect(placement)
            # The layer's own transparency comes first, the object's opacity
            # second. Solder masks are recorded as semi-transparent in the
            # layer table, and without that a stacked board shows only the top
            # layer: an opaque mask rectangle covers the copper and silkscreen
            # underneath it completely.
            painter.setOpacity(
                max(0.05, min(1.0, placement.opacity))
                * max(0.05, min(1.0, design_layer.transparency))
            )

            # 旋转 and 翻转 were stored, shown in the property panel and then
            # ignored here, so the artwork never actually turned over. The
            # transform mirrors the placement's own definition (see
            # ``ImagePlacement.matrix``): mirror in the picture's frame first,
            # then rotate, then place. Qt's Y axis points down while the design
            # frame points up, so the angle is negated to keep "positive
            # rotation" turning the same way the selection outline does.
            painter.translate(target.center())
            painter.rotate(-placement.rotation_deg)
            painter.scale(
                -1.0 if placement.flip_horizontal else 1.0,
                -1.0 if placement.flip_vertical else 1.0,
            )
            local = QRectF(
                -target.width() / 2.0,
                -target.height() / 2.0,
                target.width(),
                target.height(),
            )
            source = QRectF(0, 0, pixmap.width(), pixmap.height())
            if placement.processed_path:
                painter.drawPixmap(
                    local,
                    self._tinted(pixmap, QColor(design_layer.color)),
                    source,
                )
            else:
                painter.drawPixmap(local, pixmap, source)
            painter.setOpacity(1.0)
        else:
            # No preview available: show a placeholder rather than nothing, so
            # the placement is still visible and editable.
            painter.setBrush(QColor("#e8ecf0"))
            painter.setPen(QPen(QColor(BORDER), 1))
            painter.drawPolygon(polygon)
        painter.restore()

        # The placement's outline takes the layer's colour, so which layer an
        # object is on is readable without opening the property panel.
        painter.save()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(design_layer.color), 1, Qt.DashLine))
        painter.drawPolygon(polygon)
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

        step = 5.0
        while step * self._px_per_mm < 14:
            step *= 2

        painter.setPen(QColor(MUTED))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        plot = self._plot
        span = max(plot.width(), plot.height()) / max(self._px_per_mm, 1e-6)
        count = int(span / step) + 2
        for index in range(-count, count + 1):
            if index % 2:
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
