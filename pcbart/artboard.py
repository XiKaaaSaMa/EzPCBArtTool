"""The design document: a board outline plus the artwork placed on it.

This is the format-independent intermediate representation the roadmap calls
for. Geometry lives here, expressed in millimetres, and nothing in this module
knows about Qt, ``.epro2`` or any other output format -- adapters translate
this into whatever the target needs.

The two halves of the editing workflow map onto the two dataclasses:

* :class:`BoardOutlineSpec` is what the board-setup panel fills in. It answers
  "what shape is the board".
* :class:`ImagePlacement` is what the canvas edits. It answers "where is the
  artwork on that board".

Keeping them separate matters because the artwork can be placed before any
board exists at all -- importing a picture puts it on the canvas at a real
size, and the outline is added afterwards as its own decision. A board that
has not been configured therefore reports no outline geometry rather than a
default rectangle the user never chose.

An irregular outline will combine the artwork's contour with a preset shape
rather than replacing it, which is why it is not one of the preset shapes
below and why it will get its own panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .compose import (
    IRREGULAR_BASES,
    IrregularBase,
    base_polygon,
    combine_with_base,
    exterior_ring,
    trace_artwork_ring,
)
from .design_layers import LayerStack
from .geometry import ensure_cw, rounded_rect_ring
from .standee import (
    DEFAULT_TAB_HEIGHT_MM,
    DEFAULT_TAB_WIDTH_MM,
    attach_tab,
    contact_span,
    travel_mm,
)

__all__ = [
    "DEFAULT_IMAGE_LONG_EDGE_MM",
    "DEFAULT_TAB_HEIGHT_MM",
    "DEFAULT_TAB_WIDTH_MM",
    "OUTLINE_SHAPES",
    "SHAPE_CHOICES",
    "IrregularBase",
    "IRREGULAR_BASES",
    "OutlineShape",
    "BoardOutlineSpec",
    "ImagePlacement",
    "drawn_size_mm",
    "placement_for_pixels",
    "ArtBoard",
    "template_names",
    "register_template",
    "build_template",
]


OUTLINE_SHAPES = (
    "rect",
    "rounded",
    "circle",
)
"""The plain board shapes: each is described by its own dimensions alone.

Every shape here is described by its own dimensions alone, which is what lets
the setup panel offer shape first and size second: the size controls follow
from the shape rather than the other way round.
"""

SHAPE_CHOICES = OUTLINE_SHAPES + ("irregular",)
"""What the shape picker offers, in order.

``irregular`` is a choice without being a *preset*: it does not describe a
shape on its own, it describes combining the artwork's outline with one. It
therefore lives outside :data:`OUTLINE_SHAPES`, whose members are the shapes a
size can be typed straight into.
"""

DEFAULT_IMAGE_LONG_EDGE_MM = 76.2
"""Longest edge an imported image is scaled to, in millimetres.

Three inches, which is the size EasyEDA Pro gives an imported picture. Matching
the editor means a board drawn here carries the same geometry the user already
sees there, instead of needing a mental conversion.
"""


class OutlineShape:
    """Names for :data:`OUTLINE_SHAPES`, to avoid bare strings in callers."""

    RECT = "rect"
    ROUNDED = "rounded"
    CIRCLE = "circle"
    IRREGULAR = "irregular"
    """Combines the artwork's outline with a shape chosen in its own panel.

    Present in :data:`SHAPE_CHOICES` but not in :data:`OUTLINE_SHAPES`: it is a
    choice the user makes, not a shape whose size can be typed in directly.
    """

    LABELS = {
        RECT: "矩形",
        ROUNDED: "圆角矩形",
        CIRCLE: "圆形",
        IRREGULAR: "异形板",
    }

    #: Shapes a circle's worth of vertices can be described by, as opposed to
    #: two independent side lengths. Drives which size controls a panel shows.
    SQUARE_ONLY = (CIRCLE,)

    #: Shapes whose size follows the artwork rather than the other way round.
    ARTWORK_DRIVEN = (IRREGULAR,)


# --------------------------------------------------------------------- templates


_TEMPLATES: dict[str, Any] = {}
"""Registered board templates, keyed by name.

The template feature is not implemented yet: the user will supply template
files later. This registry is the interface they will plug into, so the dialog
and the model already handle a template being present without any rework --
:func:`template_names` simply returns an empty list until one is added.
"""


def template_names() -> list[str]:
    """Names of the registered templates, sorted.

    Empty until templates are supplied. Callers should treat an empty list as
    "no template option to show" rather than as an error.
    """
    return sorted(_TEMPLATES)


def register_template(name: str, spec: "BoardOutlineSpec") -> None:
    """Register a board template under ``name``.

    This is the intended entry point for the templates the user will supply.
    Re-registering a name replaces it, so a template file can be reloaded
    without restarting.
    """
    if not name:
        raise ValueError("模板名称不能为空")
    _TEMPLATES[name] = spec


def build_template(name: str) -> "BoardOutlineSpec":
    """A fresh copy of the named template.

    A copy, not the stored instance: the dialog edits what it is given, and a
    shared instance would let one session's edits leak into the next. The
    returned spec records which template it came from, so the caller can show
    it and so a later save knows the provenance.
    """
    if name not in _TEMPLATES:
        raise KeyError(f"未注册的模板: {name}")
    return replace(_TEMPLATES[name], template=name)


# ----------------------------------------------------------------- outline


@dataclass
class BoardOutlineSpec:
    """What shape the board is, in millimetres."""

    shape: str = OutlineShape.RECT
    width_mm: float = 60.0
    height_mm: float = 60.0
    corner_radius_mm: float = 3.0
    """Corner radius, used only by :attr:`OutlineShape.ROUNDED`."""

    margin_mm: float = 2.0
    """Extra board around the artwork when tracing an irregular outline.

    A board exactly the size of the artwork leaves no room for a cut path, and
    any tracing error eats into the subject.
    """

    simplify_mm: float = 0.35
    """Contour simplification tolerance for irregular outlines."""

    keep_aspect: bool = True
    """Whether width and height stay locked together.

    Provided here rather than only in the dialog so a caller editing the spec
    programmatically gets the same result as one using the UI.
    """

    template: str | None = None
    """Name of the template this spec came from, if any. Reserved."""

    irregular_base: str = IrregularBase.TRACE
    """How the artwork's outline combines with a shape, for an irregular board.

    ``trace``       the outline is the whole board -- a free-standing cut-out
    ``rect``        a rectangle forms the lower part, the outline the upper
    ``rounded``     the same, with the rectangle's bottom corners rounded
    """

    base_top_ratio: float = 0.35
    """Where the preset part of a combined board stops, as a fraction of its
    height measured from the bottom.

    Below this line the board is solidly the preset shape; above it, only the
    artwork's silhouette remains. This is the "how much base does the standee
    need" control.
    """

    standee: bool = False
    """Whether to hang an insertion tab under the board.

    Off by default, and independent of the shape: the tab is something the
    board *has* rather than something it *is*, so switching between a rectangle
    and an irregular outline keeps it.
    """

    standee_width_mm: float = DEFAULT_TAB_WIDTH_MM
    """Tab width in millimetres, measured along the board's bottom edge."""

    standee_height_mm: float = DEFAULT_TAB_HEIGHT_MM
    """How far the tab protrudes below that edge."""

    standee_offset_mm: float = 0.0
    """How far the tab's centre is moved sideways, in millimetres.

    Measured from the centre of the part of the bottom the tab can attach to,
    not from the board's bounding box: a traced outline is often lopsided, and
    an offset from the box centre would put the middle of the range somewhere
    the board is not. Zero means "centred on the flat the tab sits on", which
    is also the default and what the panel's slider calls 0.
    """

    @property
    def diameter_mm(self) -> float:
        """Diameter of a circular board.

        A circle has one degree of freedom, not two, so it is stored as a
        square bounding box and read through this property. Callers that offer
        a single ``直径`` field get to stay in one number while the geometry
        keeps the single representation the canvas and the exporters want.
        """
        return self.width_mm

    @diameter_mm.setter
    def diameter_mm(self, value: float) -> None:
        size = max(float(value), 0.1)
        self.width_mm = size
        self.height_mm = size

    def squared(self) -> "BoardOutlineSpec":
        """A copy whose width and height are equal, keeping the shorter side.

        Used when switching to a shape with one degree of freedom: taking the
        shorter side means the new board never grows past what the user
        already sized.
        """
        side = min(self.width_mm, self.height_mm)
        return replace(self, width_mm=side, height_mm=side)

    def enclosing_size(self, width_mm: float, height_mm: float) -> tuple[float, float]:
        """The smallest board of this shape that holds a ``width_mm`` by
        ``height_mm`` rectangle.

        A rectangle board takes the artwork's own bounds, so the picture sits
        edge to edge. A circle cannot: the smallest circle around a rectangle
        is the one through its corners, whose diameter is the rectangle's
        diagonal. Sizing a circle to the shorter side instead -- which is
        tempting because it keeps the number small -- would leave the corners
        of the artwork hanging off the board, which is not a board.
        """
        width = max(0.1, float(width_mm))
        height = max(0.1, float(height_mm))
        if self.shape in OutlineShape.SQUARE_ONLY:
            diameter = math.hypot(width, height)
            return diameter, diameter
        return width, height

    def fitted_around(self, width_mm: float, height_mm: float) -> "BoardOutlineSpec":
        """A copy sized to enclose a ``width_mm`` by ``height_mm`` rectangle.

        The rule is the same for every shape -- the board holds the artwork --
        which is what makes it safe to apply on a first board and on a change
        of shape alike.
        """
        width, height = self.enclosing_size(width_mm, height_mm)
        return replace(self, width_mm=width, height_mm=height)

    def after_shape_change(
        self,
        new_shape: str,
        artwork_size: tuple[float, float] | None = None,
    ) -> "BoardOutlineSpec":
        """The spec to use after the user picks ``new_shape``.

        Shape is chosen before size, so this runs before the user has had a
        chance to set dimensions and only has to land on something sensible:

        * Between the two two-dimensional shapes the size carries over
          untouched -- a 60 × 30 rectangle is still a 60 × 30 board when its
          corners are rounded.
        * Involving a circle the board is re-derived, because a circle stores
          a single diameter and cannot inherit two sides. The reference is the
          artwork, which is what the board has to hold; using the *old board*
          as the reference instead would grow it a little on every switch and
          never shrink back.
        * With no artwork there is nothing to hold, so the previous board is
          the reference -- the shape changes but the board still covers the
          area the user had set.
        """
        changed = replace(self, shape=new_shape)
        if not (self.shape in OutlineShape.SQUARE_ONLY or new_shape in OutlineShape.SQUARE_ONLY):
            return changed

        reference = artwork_size if artwork_size is not None else (
            self.width_mm,
            self.height_mm,
        )
        return changed.fitted_around(*reference)

    def with_size(self, width_mm: float, height_mm: float) -> "BoardOutlineSpec":
        """Resize, honouring :attr:`keep_aspect`.

        When the aspect ratio is locked, the height follows the width so a
        caller can drive the size from a single control.
        """
        width = max(float(width_mm), 0.1)
        if not self.keep_aspect or self.width_mm <= 0 or self.height_mm <= 0:
            height = max(float(height_mm), 0.1)
        else:
            height = width * (self.height_mm / self.width_mm)
        return replace(self, width_mm=width, height_mm=height)

    def polygon(self) -> list[tuple[float, float]]:
        """Outline vertices centred on the origin, in millimetres.

        Only the parametric shapes are produced here. ``irregular`` needs the
        artwork, so it is built by :meth:`ArtBoard.outline_polygon` instead.
        """
        if self.shape == OutlineShape.CIRCLE:
            # A valid circle stores equal sides (see validate); the minimum is
            # used so an inconsistent spec still draws a circle rather than an
            # ellipse that would contradict the panel's single diameter field.
            radius = min(self.width_mm, self.height_mm) / 2.0
            steps = 96
            return [
                (
                    radius * math.cos(2 * math.pi * i / steps),
                    radius * math.sin(2 * math.pi * i / steps),
                )
                for i in range(steps)
            ]

        if self.shape == OutlineShape.ROUNDED:
            return _rounded_rect(
                self.width_mm,
                self.height_mm,
                min(self.corner_radius_mm, min(self.width_mm, self.height_mm) / 2.0),
            )

        half_w = self.width_mm / 2.0
        half_h = self.height_mm / 2.0
        # Clockwise in a Y-up frame, matching what board outlines expect.
        return [
            (-half_w, half_h),
            (half_w, half_h),
            (half_w, -half_h),
            (-half_w, -half_h),
        ]

    def validate(self) -> list[str]:
        """Human-readable problems with this spec; empty when it is usable."""
        problems: list[str] = []
        known = OUTLINE_SHAPES + (OutlineShape.IRREGULAR,)
        if self.shape not in known:
            problems.append(f"未知的板框外形: {self.shape}")
        if self.width_mm <= 0 or self.height_mm <= 0:
            problems.append("板框尺寸必须为正数")
        if min(self.width_mm, self.height_mm) < 3.0:
            problems.append("板框最小边不得小于 3.0 mm（嘉立创工艺下限）")
        if self.shape == OutlineShape.CIRCLE and not math.isclose(
            self.width_mm, self.height_mm, rel_tol=0.0, abs_tol=1e-6
        ):
            problems.append("圆形板框只有一个尺寸，请用直径设置（宽高必须相等）")
        if self.shape == OutlineShape.IRREGULAR:
            if self.irregular_base not in IRREGULAR_BASES:
                problems.append(f"未知的异形基础形状: {self.irregular_base}")
            if not 0.0 <= self.base_top_ratio <= 1.0:
                problems.append("矩形上边界必须在 0 到 100% 之间")
        if self.shape == OutlineShape.ROUNDED and self.corner_radius_mm < 0:
            problems.append("圆角半径不能为负")
        if self.shape == OutlineShape.ROUNDED:
            limit = min(self.width_mm, self.height_mm) / 2.0
            if self.corner_radius_mm > limit:
                problems.append(f"圆角半径不得超过 {limit:.2f} mm")
        if self.margin_mm < 0:
            problems.append("外边距不能为负")
        if self.simplify_mm < 0:
            problems.append("简化容差不能为负")
        if self.standee:
            if self.standee_width_mm <= 0:
                problems.append("立牌插舌宽度必须为正数")
            if self.standee_height_mm <= 0:
                problems.append("立牌插舌高度必须为正数")
        return problems


def _rounded_rect(
    width: float, height: float, radius: float, segments: int = 12
) -> list[tuple[float, float]]:
    """A rounded rectangle centred on the origin.

    Delegates to :func:`pcbart.geometry.rounded_rect_ring` so there is one
    implementation. This wrapper exists because the outline is clockwise in a
    Y-up frame while the shared helper returns the counter-clockwise ring the
    complex-polygon writer wants.
    """
    return ensure_cw(rounded_rect_ring(width, height, radius, segments))


# -------------------------------------------------------------- image placement


@dataclass
class ImagePlacement:
    """Where the artwork sits on the board, in millimetres.

    ``x_mm``/``y_mm`` are the placement's centre. Centre is used rather than a
    corner because every resize keeps the centre fixed, which is what dragging
    a side handle should do.
    """

    path: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    width_mm: float = 40.0
    height_mm: float = 40.0
    rotation_deg: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    keep_aspect: bool = True
    opacity: float = 1.0
    layer: str = "顶层丝印层"
    """Display name of the layer this artwork sits on.

    The name, not the key, because it is what the property panel shows and
    what gets written to the board file. :attr:`layer_key` translates it.
    """

    processed_path: str = ""
    """Cached black-and-white result of the line-art pipeline, if it has run.

    The canvas draws *this* when it is present, tinted in the layer's colour
    the way the editor shows a layer, and falls back to the source picture
    when it is not. Storing a path rather than an array keeps the document
    small and lets the processed image be reused across restarts.
    """

    tone: str = ""
    """Which tone of a board painting this object holds.

    ``""`` for line art, which is one image placed on one or more layers.
    ``"white"`` or ``"black"`` for the two halves of a 板画, where the artwork
    is *two different images* that belong together.

    Recorded explicitly rather than inferred from the layer, because the writer
    has to know the pairing. Tracing each mask on its own sends it through its
    own simplification, so the two stop sharing an edge and leave a hairline of
    bare board wherever they meet -- measured at up to 0.17 mm on a real
    drawing. Knowing they are a pair is what lets the writer derive one tone
    from the other and tile the board exactly.
    """

    import_id: str = ""
    """Which import this object came from.

    Importing the same file twice produces two independent elements -- one per
    import -- and each has to keep its own size, position, layer and line art.
    Matching by file path instead, which is what the panel used to do, made the
    second import adopt the first one's objects and silently move them: a
    picture imported once as silk and once as colour silk ended up showing only
    the second presentation.

    Empty for objects created outside the import flow, which is what keeps
    hand-built documents (tests, scripts, the board panel) out of the grouping.
    """

    @property
    def layer_key(self) -> str:
        from .design_layers import LayerStack

        return LayerStack().layer_for_name(self.layer)

    @property
    def source_aspect(self) -> float:
        """Width divided by height. Guards against a zero height."""
        return self.width_mm / self.height_mm if self.height_mm else 1.0

    def contains_point(self, x_mm: float, y_mm: float) -> bool:
        """Whether a board-space point falls inside the rotated rectangle.

        Used for picking. Doing it by transforming the point into the
        placement's own frame avoids depending on a geometry library for what
        is four comparisons.
        """
        angle = math.radians(-self.rotation_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = x_mm - self.x_mm
        dy = y_mm - self.y_mm
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        return (
            abs(local_x) <= self.width_mm / 2.0
            and abs(local_y) <= self.height_mm / 2.0
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds before rotation, as ``(left, bottom, right, top)``."""
        half_w, half_h = self.width_mm / 2.0, self.height_mm / 2.0
        return (
            self.x_mm - half_w,
            self.y_mm - half_h,
            self.x_mm + half_w,
            self.y_mm + half_h,
        )

    def corners(self) -> list[tuple[float, float]]:
        """The four corners after rotation, clockwise from top-left.

        Returned in a Y-up frame, so "top" is the larger Y.
        """
        half_w, half_h = self.width_mm / 2.0, self.height_mm / 2.0
        local = [
            (-half_w, half_h),
            (half_w, half_h),
            (half_w, -half_h),
            (-half_w, -half_h),
        ]
        angle = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        result = []
        for lx, ly in local:
            result.append(
                (
                    self.x_mm + lx * cos_a - ly * sin_a,
                    self.y_mm + lx * sin_a + ly * cos_a,
                )
            )
        return result

    def resize_to(
        self,
        width_mm: float,
        height_mm: float,
        anchor: tuple[float, float] | None = None,
    ) -> "ImagePlacement":
        """Return a resized copy, honouring :attr:`keep_aspect`.

        ``anchor`` is the point that must stay put -- the opposite edge or
        corner when dragging a handle. When omitted the centre is kept, which
        is what a dialog changing the size numerically wants.
        """
        width = max(float(width_mm), 0.1)
        if self.keep_aspect and self.width_mm > 0 and self.height_mm > 0:
            height = width * (self.height_mm / self.width_mm)
        else:
            height = max(float(height_mm), 0.1)

        if anchor is None:
            return replace(self, width_mm=width, height_mm=height)

        # Keep the anchor fixed by shifting the centre by half the growth.
        dx = (self.width_mm - width) / 2.0
        dy = (self.height_mm - height) / 2.0
        # Which way the anchor lies relative to the centre.
        sx = 1.0 if anchor[0] < self.x_mm else -1.0
        sy = 1.0 if anchor[1] < self.y_mm else -1.0
        return replace(
            self,
            width_mm=width,
            height_mm=height,
            x_mm=self.x_mm - dx * sx,
            y_mm=self.y_mm - dy * sy,
        )

    def move_to(self, x_mm: float, y_mm: float) -> "ImagePlacement":
        return replace(self, x_mm=float(x_mm), y_mm=float(y_mm))

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.width_mm <= 0 or self.height_mm <= 0:
            problems.append("图片尺寸必须为正数")
        if not 0.0 <= self.opacity <= 1.0:
            problems.append("不透明度必须在 0 到 1 之间")
        return problems

    # ------------------------------------------------------------ transform

    def matrix(self) -> tuple[float, float, float, float, float, float]:
        """The affine map from the placement's own frame to board millimetres.

        Returns ``(a, b, c, d, e, f)`` for::

            x' = a*x + c*y + e
            y' = b*x + d*y + f

        The local frame has its origin at the placement's centre, X to the
        right and Y up, and the artwork already at its drawn size. The order is
        **mirror, then rotate, then translate**: a flip is something you do to
        the picture, and a rotation turns the flipped picture.

        This is the single definition of what 翻转 and 旋转 mean for a
        placement. The canvas, the ``.epro2`` writer and the Gerber writer all
        go through it, because the alternative is three implementations that
        agree until one of them is changed -- which is exactly how these two
        controls came to be stored, shown in the property panel, and then
        ignored by everything that draws or exports the artwork.
        """
        radians = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        scale_x = -1.0 if self.flip_horizontal else 1.0
        scale_y = -1.0 if self.flip_vertical else 1.0
        return (
            cos_a * scale_x,
            sin_a * scale_x,
            -sin_a * scale_y,
            cos_a * scale_y,
            float(self.x_mm),
            float(self.y_mm),
        )

    def local_to_board(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """One point from the placement's own frame to board millimetres."""
        a, b, c, d, e, f = self.matrix()
        return (a * x_mm + c * y_mm + e, b * x_mm + d * y_mm + f)

    def board_ring(
        self, ring: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """A whole ring from the placement's own frame to board millimetres."""
        a, b, c, d, e, f = self.matrix()
        return [
            (a * x + c * y + e, b * x + d * y + f) for x, y in ring
        ]


def drawn_size_mm(
    box_width_mm: float, box_height_mm: float, source_aspect: float
) -> tuple[float, float]:
    """The size the artwork is actually drawn at inside its placement box.

    A placement has two independent sides, and unlocking the aspect ratio lets
    them disagree with the picture. The artwork is *fitted* into the box rather
    than stretched to fill it -- distorting a drawing because a box changed
    shape is never what the user meant -- so the drawn size is the largest
    rectangle of the source's own aspect that fits.

    Shared by the canvas and the exporters. They used to disagree: the canvas
    letterboxed and the exporter stretched, so a picture whose box aspect had
    been changed exported at a different shape from the one on screen.
    """
    box_width = max(0.01, float(box_width_mm))
    box_height = max(0.01, float(box_height_mm))
    aspect = float(source_aspect) if source_aspect and source_aspect > 0 else 1.0

    box_aspect = box_width / box_height
    if aspect > box_aspect:
        # Wider than the box: the width is what limits it.
        return box_width, box_width / aspect
    return box_height * aspect, box_height


def placement_for_pixels(
    path: str,
    width_px: int,
    height_px: int,
    long_edge_mm: float = DEFAULT_IMAGE_LONG_EDGE_MM,
) -> ImagePlacement:
    """A placement that scales an imported image to ``long_edge_mm``.

    Centred on the origin, with the aspect ratio taken from the pixel size, so
    the picture lands on the canvas at a real-world size the moment it is
    imported -- before any board exists to derive one from.

    A degenerate pixel size falls back to the placement defaults rather than
    dividing by zero; the caller has already rejected empty images, so this is
    only a guard.
    """
    if width_px <= 0 or height_px <= 0:
        return ImagePlacement(path=path)

    if width_px >= height_px:
        width_mm = long_edge_mm
        height_mm = long_edge_mm * height_px / width_px
    else:
        height_mm = long_edge_mm
        width_mm = long_edge_mm * width_px / height_px

    return ImagePlacement(path=path, width_mm=width_mm, height_mm=height_mm)


# --------------------------------------------------------------------- board


@dataclass
class ArtBoard:
    """The whole document: a board outline plus the artwork on it.

    The artwork is a *list* of placements rather than one picture, because the
    editor works in layers: the same drawing can sit on the top silkscreen and
    on the top copper as two objects with their own sizes and positions, and
    each can be shown or hidden on its own. List order is draw order, so later
    entries appear on top.
    """

    outline: BoardOutlineSpec = field(default_factory=BoardOutlineSpec)
    placements: list[ImagePlacement] = field(default_factory=list)
    outline_ready: bool = False
    """Whether a board outline has been configured.

    False means the artwork has been imported but no board has been chosen
    yet, so :meth:`outline_polygon` returns nothing and the canvas shows the
    picture alone instead of presenting a default rectangle as the user's
    choice.
    """

    layers: LayerStack = field(default_factory=LayerStack)
    """Which layers are on screen and which one is current.

    View state that travels with the document so the canvas, the layer panel
    and the selection rules all read one source.
    """

    def __init__(
        self,
        outline: BoardOutlineSpec | None = None,
        placements: list[ImagePlacement] | None = None,
        outline_ready: bool = False,
        layers: LayerStack | None = None,
        image: ImagePlacement | None = None,
    ) -> None:
        # ``image`` is accepted as a convenience for callers that work with a
        # single picture, which is most of them. It seeds the list rather than
        # living alongside it, so there is still only one place the artwork is
        # stored.
        self.outline = outline if outline is not None else BoardOutlineSpec()
        self.placements = list(placements) if placements else []
        if image is not None:
            self.placements.append(image)
        self.outline_ready = outline_ready
        self.layers = layers if layers is not None else LayerStack()
        self._silhouette_cache = None
        self._irregular_cache = None
        self._irregular_error = None
        self._parametric_cache = None
        self._standee_cache = None
        self._standee_error = None

    @property
    def image(self) -> ImagePlacement | None:
        """The first artwork placement, or ``None``.

        The board outline follows one picture -- the one an irregular outline
        is traced from -- and that is this one. Callers that want everything
        should read :attr:`placements`.
        """
        return self.placements[0] if self.placements else None

    @image.setter
    def image(self, value: ImagePlacement | None) -> None:
        if value is None:
            self.placements.clear()
        elif self.placements:
            self.placements[0] = value
        else:
            self.placements.append(value)

    # ------------------------------------------------------------ placements

    def add_placement(self, placement: ImagePlacement) -> ImagePlacement:
        """Append artwork and return it, so the caller can select it."""
        self.placements.append(placement)
        return placement

    def remove_placement(self, placement: ImagePlacement) -> bool:
        """Remove one placement. Returns whether it was there."""
        if placement in self.placements:
            self.placements.remove(placement)
            return True
        return False

    def placement_at(self, x_mm: float, y_mm: float) -> ImagePlacement | None:
        """The topmost *selectable* placement containing this point.

        Topmost means last in list order, matching what the canvas paints on
        top. Placements on hidden layers are skipped: selecting something the
        user cannot see would make the next drag move an invisible object.
        """
        for placement in reversed(self.placements):
            if not self.layers.selectable(placement.layer_key):
                continue
            if placement.contains_point(x_mm, y_mm):
                return placement
        return None

    def visible_placements(self) -> list[ImagePlacement]:
        """Placements whose layer is currently shown, in draw order."""
        return [
            item
            for item in self.placements
            if self.layers.is_visible(item.layer_key)
        ]

    def validate(self) -> list[str]:
        problems = list(self.outline.validate())
        for placement in self.placements:
            problems.extend(placement.validate())
        return problems

    def outline_polygon(self) -> list[tuple[float, float]]:
        """Board outline vertices in millimetres.

        Empty until the user configures a board: an unconfigured document has
        no outline, and inventing one would put geometry on the canvas that
        nobody asked for.

        This is the finished shape -- the standee tab, when it is switched on,
        is already part of it, because that is what gets drawn, measured and
        exported.
        """
        if not self.outline_ready:
            return []
        # A copy: the ring behind this comes from a cache (a shape's own
        # polygon, or the traced composition), and a caller that edited the
        # list in place would be editing what every later frame draws.
        return list(self._standee_outline(self._base_outline()))

    def _base_outline(self) -> list[tuple[float, float]]:
        """The board without its standee tab.

        Kept separate because two things need the board on its own: the tab is
        attached to it, and the slider's range is measured along it. Doing that
        from the finished outline would measure the *tab's* bottom edge instead
        of the board's.

        Parametric shapes are cached by value here, which is what lets the
        attached-tab cache below recognise them by identity: a shape's own
        polygon is otherwise rebuilt from scratch on every call, and this is on
        the paint path.
        """
        if self.outline.shape == OutlineShape.IRREGULAR:
            return self._irregular_outline()

        spec = self.outline
        key = (
            spec.shape,
            spec.width_mm,
            spec.height_mm,
            spec.corner_radius_mm,
        )
        if self._parametric_cache is not None and self._parametric_cache[0] == key:
            return self._parametric_cache[1]
        ring = spec.polygon()
        self._parametric_cache = (key, ring)
        return ring

    def _standee_key(self) -> tuple:
        """What the attached tab depends on."""
        spec = self.outline
        return (
            spec.standee,
            spec.standee_width_mm,
            spec.standee_height_mm,
            spec.standee_offset_mm,
        )

    def _standee_outline(self, ring) -> list[tuple[float, float]]:
        """``ring`` with the standee tab unioned on, or ``ring`` unchanged.

        Cached against the board ring *itself* (the cached list object, not a
        copy of its points): the union is real geometry work and this runs in
        ``paintEvent``, but the ring only changes when the board does, and both
        outline paths hand back the same object until then.

        Never raises. An exception escaping ``paintEvent`` takes the whole
        window down, so a tab that cannot be built leaves the board as it was
        and records why in :attr:`standee_error` for the panel to show.
        """
        spec = self.outline
        if not spec.standee or len(ring) < 3:
            self._standee_error = None
            return ring

        key = self._standee_key()
        cache = self._standee_cache
        if cache is not None and cache[0] is ring and cache[1] == key:
            self._standee_error = cache[3]
            return cache[2]

        try:
            result, attached = attach_tab(
                ring,
                spec.standee_width_mm,
                spec.standee_height_mm,
                spec.standee_offset_mm,
            )
        except Exception as exc:  # noqa: BLE001 - paint must not raise
            result, attached = ring, False
            error: str | None = str(exc)
        else:
            error = None if attached else "插舌所在位置的下边缘太窄，未能与板框相连"

        self._standee_error = error
        self._standee_cache = (ring, key, result, error)
        return result

    @property
    def standee_error(self) -> str | None:
        """Why the standee tab is not part of the outline, or ``None``.

        The panel shows this, and the exporter refuses on it: a board that was
        asked for a tab and silently went out without one is exactly the kind
        of file that opens fine and is wrong.
        """
        return self._standee_error

    def standee_travel_mm(self) -> float:
        """How far the tab can slide sideways and still be attached.

        Measured on the board without its tab, and in millimetres rather than
        as a percentage, so the panel can tell the user the real range the way
        the irregular board's slider does.

        Zero covers two different situations -- a tab that exactly fills the
        bottom edge, and a tab *wider* than it -- which the panel tells apart
        through :meth:`standee_contact_mm` rather than by guessing.
        """
        spec = self.outline
        if not spec.standee or not self.outline_ready:
            return 0.0
        ring = self._base_outline()
        if len(ring) < 3:
            return 0.0
        try:
            return travel_mm(
                ring, spec.standee_width_mm, spec.standee_height_mm
            )
        except Exception:  # noqa: BLE001 - a broken ring means no travel
            return 0.0

    def standee_contact_mm(self) -> float:
        """How wide the part of the bottom edge the tab attaches to is.

        This is the real usable width -- a circle's bottom contributes only the
        narrow arc inside the weld band, not its diameter -- so the panel can
        tell a tab that fills the edge from one that overhangs it.
        """
        spec = self.outline
        if not spec.standee or not self.outline_ready:
            return 0.0
        ring = self._base_outline()
        if len(ring) < 3:
            return 0.0
        try:
            span = contact_span(ring, spec.standee_height_mm)
        except Exception:  # noqa: BLE001 - a broken ring has no usable edge
            return 0.0
        if span is None:
            return 0.0
        return max(0.0, float(span[1]) - float(span[0]))

    def irregular_source(self) -> str | None:
        """Path of the artwork an irregular outline is traced from."""
        if self.image is None or not self.image.path:
            return None
        return self.image.path

    def _silhouette_key(self) -> tuple:
        """What the traced silhouette depends on.

        Deliberately **without** the artwork's position. A drag changes where
        the picture sits, not what its outline looks like, so including x/y
        here forced a full re-trace on every frame of a drag -- 31 ms each, on
        a 33 ms budget. The trace is cached against the picture and its drawn
        size, and the position is applied afterwards as a translation.
        """
        image = self.image
        return (
            None if image is None else image.path,
            None if image is None else image.width_mm,
            None if image is None else image.height_mm,
            self.outline.simplify_mm,
        )

    def _base_key(self) -> tuple:
        """What the preset part of a combined board depends on."""
        spec = self.outline
        return (
            spec.irregular_base,
            spec.width_mm,
            spec.height_mm,
            spec.corner_radius_mm,
            spec.base_top_ratio,
        )

    def _traced_silhouette(self) -> list[tuple[float, float]]:
        """The artwork's silhouette, centred on the artwork, cached.

        This is the expensive step -- a full preprocessing pass and contour
        extraction -- and the only one a drag does not invalidate.
        """
        image = self.image
        assert image is not None
        key = self._silhouette_key()
        if self._silhouette_cache is not None and self._silhouette_cache[0] == key:
            return self._silhouette_cache[1]

        ring = trace_artwork_ring(
            image.path,
            width_mm=image.width_mm,
            height_mm=image.height_mm,
            simplify_mm=self.outline.simplify_mm,
        )
        self._silhouette_cache = (key, ring)
        return ring

    def _irregular_outline(self) -> list[tuple[float, float]]:
        """Combine the artwork's silhouette with the chosen preset shape.

        Never raises. The canvas calls this while painting, and an exception
        escaping ``paintEvent`` tears down the whole window instead of showing
        an error -- a mask with no foreground pixels was enough to do it.
        Failures fall back to the preset base and are recorded in
        :attr:`_irregular_error` for the panel to display.

        Two-stage cache: the trace is reused whenever only the position or the
        base changed, which covers dragging and slider tweaks. Only a change to
        the picture, its size, or the simplification tolerance re-traces.
        """
        image = self.image
        if image is None or not image.path:
            # Nothing to trace. A combined board can still show its base, since
            # that part does not depend on the artwork; a pure trace cannot.
            self._irregular_error = None
            if self.outline.irregular_base == IrregularBase.TRACE:
                return []
            return self._irregular_base_only()

        spec = self.outline
        # Position and base settings are cheap to re-apply, so they are kept
        # out of the expensive cache and applied every call instead.
        cheap_key = (image.x_mm, image.y_mm) + self._base_key()
        if (
            self._irregular_cache is not None
            and self._irregular_cache[0] == cheap_key
            and self._irregular_cache[2] == self._silhouette_key()
        ):
            return self._irregular_cache[1]

        try:
            ring = self._traced_silhouette()
            result = self._compose_from(ring, image, spec)
        except Exception as exc:  # noqa: BLE001 - paint must not raise
            self._irregular_error = str(exc)
            result = self._irregular_base_only()
        else:
            self._irregular_error = None

        self._irregular_cache = (cheap_key, result, self._silhouette_key())
        return result

    def _compose_from(self, ring, image, spec) -> list[tuple[float, float]]:
        """Place an already-traced silhouette and combine it with the base.

        Split out from the tracing so a drag can skip the expensive half: the
        ring is reused and only the translation and union are recomputed, which
        is 0.13 ms against 31 ms for the trace.

        The order still matters: the silhouette is moved to where the picture
        is drawn *before* the union, because the union happens in board
        coordinates. Shifting the finished union would drag the base along with
        the artwork.
        """
        if not ring:
            return self._irregular_base_only()

        # trace_artwork_ring centres the silhouette on the artwork's own centre;
        # move it to where the placement actually sits on the board.
        placed = [(x + image.x_mm, y + image.y_mm) for x, y in ring]

        if spec.irregular_base == IrregularBase.TRACE:
            return placed
        if spec.irregular_base not in IrregularBase.COMBINING:
            return placed

        return combine_with_base(
            placed,
            spec.irregular_base,
            width_mm=spec.width_mm,
            height_mm=spec.height_mm,
            corner_radius_mm=spec.corner_radius_mm,
            top_ratio=spec.base_top_ratio,
        )

    def _irregular_base_only(self) -> list[tuple[float, float]]:
        """Just the preset part, for when there is no artwork to trace."""
        spec = self.outline
        if spec.irregular_base not in IrregularBase.COMBINING:
            return []
        polygon = base_polygon(
            spec.irregular_base,
            spec.width_mm,
            spec.height_mm,
            spec.corner_radius_mm,
            spec.base_top_ratio,
        )
        return exterior_ring(polygon)

    def size_mm(self) -> tuple[float, float]:
        """Board bounding-box size in millimetres."""
        points = self.outline_polygon()
        if not points:
            return (self.outline.width_mm, self.outline.height_mm)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (max(xs) - min(xs), max(ys) - min(ys))

    def fit_image_to_board(self, margin_mm: float = 0.0) -> "ArtBoard":
        """Scale the artwork down so it fits inside the board.

        Returns a new board; the original is untouched, which keeps the caller
        free to undo by simply not using the result. Every placement is scaled
        by the same factor, so a drawing that spans several layers keeps its
        relative geometry instead of each layer ending up a different size.
        """
        if not self.placements:
            return self

        board_w, board_h = self.outline.width_mm, self.outline.height_mm
        usable_w = max(0.1, board_w - 2 * margin_mm)
        usable_h = max(0.1, board_h - 2 * margin_mm)

        scales = [
            min(usable_w / item.width_mm, usable_h / item.height_mm)
            for item in self.placements
            if item.width_mm > 0 and item.height_mm > 0
        ]
        if not scales:
            return self
        scale = min(scales)
        if scale >= 1.0:
            return self

        # ``replace`` on the board would copy the list by reference, so the
        # resized placements go into a fresh list.
        resized = [
            replace(
                item,
                width_mm=item.width_mm * scale,
                height_mm=item.height_mm * scale,
            )
            for item in self.placements
        ]
        return replace(self, placements=resized)
