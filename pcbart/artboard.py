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
from .geometry import circle_ring, densify_ring, ensure_cw, rounded_rect_ring
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
    "PlacementGroup",
    "TextElement",
    "CutoutElement",
    "NAME_TOKEN",
    "NUMBER_TOKEN",
    "ENGLISH_TOKEN",
    "TOKENS",
    "substitute_tokens",
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
    TEMPLATE = "template"
    """A board whose outline comes from a template, not from a preset.

    The 通行证 card is this: a rectangle with cut corners and a notch on one
    edge, which no preset can express. The ring lives in ``template_outline``
    and the preset size fields are ignored for it.
    """
    """Combines the artwork's outline with a shape chosen in its own panel.

    Present in :data:`SHAPE_CHOICES` but not in :data:`OUTLINE_SHAPES`: it is a
    choice the user makes, not a shape whose size can be typed in directly.
    """

    LABELS = {
        RECT: "矩形",
        ROUNDED: "圆角矩形",
        CIRCLE: "圆形",
        IRREGULAR: "异形板",
        TEMPLATE: "模板",
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
    """Name of the template this spec came from, if any."""

    template_outline: tuple[tuple[float, float], ...] = ()
    """The template's outline ring, centred on the origin, in millimetres.

    A tuple of points rather than a list so the spec stays hashable and the
    outline caches can key on it. Empty unless ``shape`` is
    :attr:`OutlineShape.TEMPLATE`.
    """

    template_arcs: tuple[float | None, ...] = ()
    """One signed arc sweep per :attr:`template_outline` segment.

    Empty for a straight-edged template.  The card's four rounded corners use
    this: the exported path keeps native ``ARC`` tokens, while the canvas and
    geometry checks work from :meth:`polygon` after the curves are sampled.
    """

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
        if self.shape == OutlineShape.TEMPLATE:
            # Already centred and in millimetres; nothing to derive.  When the
            # template names arc sweeps, sample them so the canvas, validation
            # and Gerber all see the curved corner rather than its chord.
            points = [(float(x), float(y)) for x, y in self.template_outline]
            if self.template_arcs:
                return densify_ring(points, self.template_arcs)
            return points
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

    def polygon_raw(self) -> list[tuple[float, float]]:
        """The template's corner vertices, without sampling its arcs."""
        if self.shape == OutlineShape.TEMPLATE:
            return [(float(x), float(y)) for x, y in self.template_outline]
        return self.polygon()

    def outline_path(self) -> list[Any]:
        """The outline as an EasyEDA path token stream, in millimetres.

        Straight templates use the same polyline path as before.  A template
        with arc metadata keeps native ``ARC`` segments so the editor gets a
        round corner that stays editable as an arc rather than a sampled
        polygon.  The stream starts on a straight segment because the grammar
        expects the opening ``L`` run before an arc token.
        """
        from .primitives import Mode

        vertices = self.polygon_raw()
        if len(vertices) < 3:
            return []

        arcs = self.template_arcs if self.shape == OutlineShape.TEMPLATE else ()
        if not arcs:
            # ⚠️ **The loop closes explicitly, and forgetting that is what made
            # a rectangular board export with one edge missing.** A path is a
            # start point plus one segment per following point, so a four-corner
            # ring written as four points draws only three edges -- measured on
            # a 60 x 60 board, the file carried 顶点 左上→右上→右下→左下 and the
            # whole left side was gone. On a circle the same omission removes one
            # 3.75-degree segment out of 96 and nobody would ever see it, which is
            # why it went unnoticed until the user made a square board.
            #
            # The arc branch below has always closed (`range(1, len + 1)` with a
            # modulo), and ``test_the_template_outline_samples_its_corner_arcs``
            # already asserts the invariant that the first and last decoded
            # points coincide. This branch simply never did it, so the same
            # outline was closed or not depending on whether it had curves.
            path: list[Any] = [vertices[0][0], vertices[0][1], Mode.L]
            path.extend([value for point in vertices[1:] for value in point])
            path.extend(vertices[0])
            return path

        start = next(
            (
                index
                for index in range(len(vertices))
                if (arcs[index] if index < len(arcs) else None) is None
            ),
            0,
        )
        path = [vertices[start][0], vertices[start][1], Mode.L]
        for step in range(1, len(vertices) + 1):
            segment = (start + step - 1) % len(vertices)
            end = vertices[(segment + 1) % len(vertices)]
            sweep = arcs[segment] if segment < len(arcs) else None
            if sweep is None:
                path.extend(end)
            else:
                path.extend([Mode.ARC, float(sweep), end[0], end[1]])
        return path

    def validate(self) -> list[str]:
        """Human-readable problems with this spec; empty when it is usable."""
        problems: list[str] = []
        # ``TEMPLATE`` belongs here for the same reason ``IRREGULAR`` does: it
        # is a shape the model supports that is not one of the presets a size
        # can be typed into. Omitting it made every 通行证 card fail validation
        # with "未知的板框外形: template", so a card could be designed and
        # never exported -- the two exporters both validate before writing.
        known = OUTLINE_SHAPES + (OutlineShape.IRREGULAR, OutlineShape.TEMPLATE)
        if self.shape not in known:
            problems.append(f"未知的板框外形: {self.shape}")
        if self.shape == OutlineShape.TEMPLATE and len(self.template_outline) < 3:
            problems.append("模板板框缺少轮廓点，请重新套用模板")
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


def _obround_ring(
    width: float, height: float, segments: int = 32
) -> list[tuple[float, float]]:
    """A stadium-shaped ring centred on the origin, counter-clockwise, Y up.

    The long lanyard opening is a routed slot: its two ends are half-circles,
    not square corners.  Building it from the short side's radius is what keeps
    the straight runs and both caps sharing one tangent.
    """
    width = max(float(width), 0.01)
    height = max(float(height), 0.01)
    radius = min(width, height) / 2.0
    half_w = width / 2.0
    half_h = height / 2.0
    points: list[tuple[float, float]] = []
    if width >= height:
        left_x = -half_w + radius
        right_x = half_w - radius
        points.append((left_x, -radius))
        points.append((right_x, -radius))
        for index in range(segments + 1):
            angle = -math.pi / 2.0 + math.pi * index / segments
            points.append((right_x + radius * math.cos(angle), radius * math.sin(angle)))
        for index in range(segments + 1):
            angle = math.pi / 2.0 + math.pi * index / segments
            points.append((left_x + radius * math.cos(angle), radius * math.sin(angle)))
    else:
        bottom_y = -half_h + radius
        top_y = half_h - radius
        points.append((-radius, bottom_y))
        points.append((-radius, top_y))
        for index in range(segments + 1):
            angle = math.pi + math.pi * index / segments
            points.append((radius * math.cos(angle), top_y + radius * math.sin(angle)))
        for index in range(segments + 1):
            angle = math.pi * index / segments
            points.append((radius * math.cos(angle), bottom_y + radius * math.sin(angle)))
    return points


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

    source_window: tuple[float, float, float, float] | None = None
    """Which part of the picture to draw, as ``(x0, y0, x1, y1)``.

    Fractions of the source picture, 0-1, with Y measured **from the top** like
    the image itself rather than like the board. ``None`` means the whole
    picture, which is what every placement meant before this existed.

    This is how the 通行证 template lets the user frame the artwork: the region
    on the card is a window onto the picture, and the part outside it is what
    gets cut off. Storing the window on the placement means the canvas, the
    ``.epro2`` writer and the Gerber writer all read one definition -- the same
    reason 旋转/翻转 live here.
    """

    region: str = ""
    """Key of the 通行证 region this artwork was framed into, or empty.

    ⚠️ **This is an export-time clip, not a label.** The region on the card is
    where the artwork is printed, and the user asked for the picture outside it
    to be cut -- but the region was only a canvas guide: resize or move the
    framed picture afterwards (both are allowed, and both are normal editing)
    and the export wrote ink outside the card's own opening.

    Recording *which* region makes the boundary enforceable at write time, so
    what gets manufactured is always what the card's layout allows. Empty for
    everything that was never framed -- a plain board, or an import on a board
    that is not the card -- and those keep the old behaviour exactly.
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

    kind: str = ""
    """What sort of generated artwork this is, for the ones the card inserts.

    ``"qr"`` / ``"barcode"`` / ``"profession"`` / ``"faction"``, or empty for a
    picture the user imported. It exists because the four share one
    ``import_id`` -- one button inserts them -- but they do **not** share a
    representation: the two codes are black-on-transparent and the canvas has
    to recolour them by alpha, while the faction logo is a white line drawing
    that must be drawn as-is. Without this the canvas could only tell the four
    apart by their layer or size.
    """

    element_id: str = ""
    """Stable identity for this object, so a group can name its members.

    ⚠️ **Position in the list cannot serve as an identity.** Undo restores the
    document by overwriting placements *in place* and reusing the objects, and
    deleting or reordering shifts every index after it -- so a group storing
    indices would silently re-point at different artwork after the first undo.
    An id that travels with the object survives all of it.

    Minted by :meth:`ArtBoard.element_id` rather than in ``__post_init__``: a
    dataclass that generated one on construction would give the same placement
    a fresh identity every time it was copied or rebuilt, which is exactly what
    this field exists to prevent.
    """

    @property
    def layer_key(self) -> str:
        from .design_layers import LayerStack

        return LayerStack().layer_for_name(self.layer)

    @property
    def source_aspect(self) -> float:
        """Width divided by height. Guards against a zero height."""
        return self.width_mm / self.height_mm if self.height_mm else 1.0

    def source_rect(self, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """The part of a ``width_px`` x ``height_px`` picture to use, in pixels.

        One implementation for the canvas and both writers: a window that the
        three of them rounded differently would show the user one framing and
        export another.
        """
        if width_px <= 0 or height_px <= 0:
            return (0, 0, 0, 0)
        window = self.source_window
        if window is None:
            return (0, 0, width_px, height_px)

        x0, y0, x1, y1 = window
        left = max(0, min(width_px, int(round(min(x0, x1) * width_px))))
        right = max(0, min(width_px, int(round(max(x0, x1) * width_px))))
        top = max(0, min(height_px, int(round(min(y0, y1) * height_px))))
        bottom = max(0, min(height_px, int(round(max(y0, y1) * height_px))))
        # A collapsed window would divide by zero downstream; fall back to the
        # whole picture rather than producing a zero-width mask.
        if right - left < 1 or bottom - top < 1:
            return (0, 0, width_px, height_px)
        return (left, top, right, bottom)

    def source_aspect_for(self, width_px: int, height_px: int) -> float:
        """Aspect of the part of the picture being used, not of the whole."""
        left, top, right, bottom = self.source_rect(width_px, height_px)
        width = right - left
        height = bottom - top
        return width / height if height else 1.0

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

    def overlaps_rect(self, bounds: tuple[float, float, float, float]) -> bool:
        """Whether this placement's box still touches an axis-aligned rectangle.

        Used for one question the canvas asks on every frame: *has this picture
        been moved completely out of its illustration opening?* The picture is
        allowed to go anywhere -- the user asked for free movement so that a
        figure can be arranged around the card's top cutout -- but a picture that
        has left the opening entirely prints nothing, and silently showing an
        empty opening is how the user loses their artwork without noticing.

        ⚠️ **An exact test, not a bounding-box overlap.** A rotated picture's
        bounding box stays large, so comparing boxes would report "still
        overlapping" for a 45-degree picture whose corners no longer reach the
        opening -- and that is precisely the case where the warning matters.
        The separating-axis test below is exact for two convex shapes and costs
        four projections.
        """
        x0, y0, x1, y1 = bounds
        low_x, high_x = min(x0, x1), max(x0, x1)
        low_y, high_y = min(y0, y1), max(y0, y1)
        opening = [
            (low_x, low_y),
            (high_x, low_y),
            (high_x, high_y),
            (low_x, high_y),
        ]
        points = self.corners()

        # The opening's two axes, then the picture's own two. Two convex shapes
        # are disjoint exactly when some axis separates their projections.
        edge = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        side = (points[3][0] - points[0][0], points[3][1] - points[0][1])
        axes = ((1.0, 0.0), (0.0, 1.0), edge, side)
        for ax, ay in axes:
            length = math.hypot(ax, ay)
            if length < 1e-9:
                continue
            ax, ay = ax / length, ay / length
            mine = [x * ax + y * ay for x, y in points]
            theirs = [x * ax + y * ay for x, y in opening]
            if max(mine) < min(theirs) or max(theirs) < min(mine):
                return False
        return True

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

    def board_arc_ring(self, ring):
        """A ring of lines and arcs, over the same matrix as ``board_ring``.

        ⚠️ **A mirror reverses every arc's sweep.** ``sweeps[i]`` is signed
        relative to the direction of travel, so reflecting the ring turns a
        counter-clockwise bulge into a clockwise one. Leaving the signs alone
        on a flipped placement mirrored the vertices but not the curves, which
        reads as every arc bending the wrong way -- the same class of defect as
        the flip that used to move the selection box instead of the picture
        (see :mod:`pcbart.placements` history in the docs).

        The determinant's sign is exactly that question: the matrix is
        mirror-then-rotate, no scaling, so it is a reflection precisely when
        one of the two flips is on -- in which case ``a*d - b*c`` is negative.
        A pure rotation keeps the sweeps, and so does a flip in both axes
        (that is a half turn, not a reflection).
        """
        from .geometry import ArcRing

        a, b, c, d, e, f = self.matrix()
        mirrored = (a * d - b * c) < 0
        return ArcRing(
            points=[
                (a * x + c * y + e, b * x + d * y + f) for x, y in ring.points
            ],
            sweeps=[-s if mirrored else s for s in ring.sweeps],
        )


# ----------------------------------------------------- template text and cutouts

NAME_TOKEN = "{代号}"
NUMBER_TOKEN = "{编号}"
ENGLISH_TOKEN = "{英文名}"

#: The substitutions the operator picker performs on template text.
#:
#: A token rather than a "this text is the name" flag, because the card's own
#: wording is part of the design: the vertical strip reads ``ARKNIGHTS-`` and
#: then the operator's number, and a flag field would have had to store that
#: prefix somewhere else, where editing it would be a second, differently-behaved
#: thing. With a token the user rewrites the whole line in one box, and
#: :func:`substitute_tokens` leaves a line with no token alone rather than
#: guessing which part of it was meant to be the number.
#:
#: ``{英文名}`` exists beside ``{代号}`` because the 干员代号 line is set in the
#: template's own ``Agency FB Negreta`` -- a Latin-only face -- so the Chinese
#: name draws as empty boxes there. Having two tokens rather than making
#: ``{代号}`` mean English keeps both available: the user can put the Chinese
#: name on some other line, or switch this one back, without any code change.
TOKENS = (NAME_TOKEN, ENGLISH_TOKEN, NUMBER_TOKEN)


def substitute_tokens(text: str, *, name: str = "", number: str = "") -> str:
    """Replace :data:`TOKENS` in ``text`` with the current operator's values.

    An empty value leaves the token visible. Showing ``{代号}`` on the card is
    the honest outcome: it says no operator has been chosen yet, where blanking
    it would look exactly like a card whose name was deleted.

    ``{英文名}`` is derived from ``name`` through
    :func:`pcbart.operator_names.english_name`, so a caller only has to pass
    the name it already has.

    ⚠️ **One pass, not four replacements.** Doing ``replace(NAME).replace(...)``
    re-scans the *substituted* text, so a name that happens to contain
    ``{编号}`` -- or any of the 463 operator names containing a brace -- would
    be rewritten into the number. Scanning once for every token and emitting
    each match straight to the output is the only version that cannot reparse
    its own work.
    """
    if not text:
        return ""
    from .operator_names import english_name

    values = {
        NAME_TOKEN: name,
        ENGLISH_TOKEN: english_name(name) if name else "",
        NUMBER_TOKEN: number,
    }
    result: list[str] = []
    index = 0
    while index < len(text):
        for token, value in values.items():
            if text.startswith(token, index):
                # An empty value leaves the token in place rather than running
                # it through as "" -- see the docstring.
                result.append(value if value else token)
                index += len(token)
                break
        else:
            result.append(text[index])
            index += 1
    return "".join(result)


@dataclass
class TextElement:
    """A line of text that is part of the board, not part of a picture.

    Template cards are mostly text -- the operator's name, the number strip,
    the fixed wording -- and the user has to be able to change all of it. This
    is stored as geometry rather than as a rendered image because the whole
    point is that it stays editable, and because a ten-millimetre glyph
    exported as a picture would be a bitmap of the letter that the board house
    has to trace.

    ``x_mm``/``y_mm`` are the anchor, not the centre: template text is anchored
    where it was drawn (``LEFT_BOTTOM`` throughout, measured from the source
    files), and an anchor survives a change of wording while a centre would
    make the line jump sideways every time a character was typed.
    """

    text: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    size_mm: float = 1.0
    """Cap height in millimetres, matching the font size the source file uses."""

    element_id: str = ""
    """Stable identity, for the same reason :attr:`ImagePlacement.element_id`
    exists: a group names its members by id, and a text line is an element the
    user can put in one. See that field."""

    rotation_deg: float = 0.0
    layer: str = "顶层丝印层"
    font: str = ""
    """Family name as the source file spells it.

    Kept for the editor's own use and for writing the file back; the canvas
    falls back to its own font, because the exact face is not installed
    everywhere and a substituted font still shows the line in the right place.
    """
    anchor: str = "left_bottom"
    label: str = ""
    """What this line is for, shown in the content panel's list.

    Never used as the text itself: a label the user can see is one they will
    try to edit, so the placeholder wording and the label are separate.
    """

    @property
    def layer_key(self) -> str:
        from .design_layers import LayerStack

        return LayerStack().layer_for_name(self.layer)

    def resolved_text(self, *, name: str = "", number: str = "") -> str:
        return substitute_tokens(self.text, name=name, number=number)

    def contains_point(self, x_mm: float, y_mm: float) -> bool:
        """Whether a board point is close enough to count as clicking this line.

        Text has no stored outline -- its extent depends on the font -- so the
        hit box is derived from the line's own length. A rough box is right
        here: the alternative is measuring glyphs, and a click that lands on
        the line but one millimetre outside its box would read as the canvas
        ignoring it.
        """
        width, height, bbox_y = self.box_mm()
        return (
            self.x_mm <= x_mm <= self.x_mm + width
            and bbox_y <= y_mm <= bbox_y + height
        )

    def box_mm(self) -> tuple[float, float, float]:
        """``(width, height, bottom_y)`` of the line's hit box.

        Measured from the anchor. A quarter-width letter is a reasonable
        average for the mixed CJK/Latin wording these cards carry, and it is
        only ever used for picking.
        """
        height = max(self.size_mm, 0.1)
        width = max(len(self.text), 1) * height * 0.62
        return (width, height, self.y_mm)

    def corners(self) -> list[tuple[float, float]]:
        """The hit box's four corners, rotated about the anchor.

        Rotated because a quarter of the template's lines are turned 90 degrees
        (the vertical strips), and a box that ignored the rotation would sit
        flat across the card and pick up clicks that belong to the artwork.
        """
        width, height, _ = self.box_mm()
        angle = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        points = []
        for lx, ly in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)):
            points.append(
                (
                    self.x_mm + lx * cos_a - ly * sin_a,
                    self.y_mm + lx * sin_a + ly * cos_a,
                )
            )
        return points

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.size_mm <= 0:
            problems.append("文字大小必须为正数")
        return problems


@dataclass
class PlacementGroup:
    """Several placements that the user treats as one element.

    The immediate use is a 板画, which is **two** placements -- the white tone
    on silkscreen and the black on copper -- that are one picture to whoever
    drew it. Before this they were two rows in the layer list, two things to
    click, and two things to move, and keeping them registered was entirely on
    the user.

    The members are :attr:`ImagePlacement.element_id` values rather than
    indices or object references -- see that field for why.

    ``name`` is what the user calls it, shown in the layer list and written to
    the board file as EasyEDA Pro's own ``groupName``. Empty is allowed and
    means "not named yet"; the panel falls back to a generated label rather
    than inventing a name to store.
    """

    id: str = ""
    name: str = ""
    members: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("组合缺少标识")
        if len(self.members) < 2:
            # Not an error the user has to fix -- the board prunes these rather
            # than exporting them -- but it is a state the document should not
            # report as healthy, because a one-member group means the exit
            # paths (delete, ungroup, undo) left something behind.
            problems.append("组合至少需要两个元素")
        return problems


@dataclass
class CutoutElement:
    """A hole cut through the board, as the template draws it.

    The 通行证 card has three: a long lanyard opening across the top and a
    round hanging hole in each top corner.  A cutout is a distinct kind of
    thing from an outline -- a second, closed ring *inside* the board rather
    than part of its boundary -- and the exporters have to treat it that way,
    or the opening is simply not cut.

    ``shape`` is ``"rect"``, ``"circle"`` or ``"obround"``.  The last two are
    what the source card uses for the lanyard opening and the corner holes; a
    rectangle would be read by a board house as a routed slot of the wrong
    shape, and a polygon approximation would leave visible facets at this size.
    """

    key: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    width_mm: float = 20.0
    height_mm: float = 3.0
    corner_radius_mm: float = 0.0
    """Corner rounding for the rectangular shape."""
    shape: str = "rect"
    label: str = "挖槽"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        half_w, half_h = self.width_mm / 2.0, self.height_mm / 2.0
        return (
            self.x_mm - half_w,
            self.y_mm - half_h,
            self.x_mm + half_w,
            self.y_mm + half_h,
        )

    def contains_point(self, x_mm: float, y_mm: float) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 <= x_mm <= x1 and y0 <= y_mm <= y1

    def ring(self) -> list[tuple[float, float]]:
        """The opening as a closed ring, counter-clockwise, Y up.

        Wound counter-clockwise while the board's own outline is clockwise: the
        two have to disagree for a cutting program to read the inner one as a
        hole instead of a second board.
        """
        # Built about the origin first, then moved -- and *only* then.
        # ``bounds`` already returns absolute coordinates, so offsetting a ring
        # built from it would add the centre twice and put the slot at roughly
        # double its Y. That is not a hypothetical: it put a 24 mm slot 43 mm
        # above the card's top edge, which the outline layer then reported as a
        # 143 mm-tall board.
        if self.shape == "circle":
            ring = circle_ring(0.0, 0.0, max(self.width_mm, self.height_mm) / 2.0, 64)
        elif self.shape == "obround":
            ring = _obround_ring(self.width_mm, self.height_mm)
        elif self.corner_radius_mm > 0:
            ring = rounded_rect_ring(
                self.width_mm, self.height_mm, self.corner_radius_mm
            )
        else:
            half_w, half_h = self.width_mm / 2.0, self.height_mm / 2.0
            ring = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]
        return [(self.x_mm + x, self.y_mm + y) for x, y in ring]

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.shape not in ("rect", "circle", "obround"):
            problems.append(f"{self.label}的形状无法识别")
        if self.width_mm <= 0 or self.height_mm <= 0:
            problems.append(f"{self.label}的尺寸必须为正数")
        if self.shape == "circle" and not math.isclose(
            self.width_mm, self.height_mm, rel_tol=0.0, abs_tol=0.02
        ):
            problems.append(f"{self.label}是圆孔，宽高必须相等")
        if self.corner_radius_mm < 0:
            problems.append(f"{self.label}的圆角不能为负")
        elif self.corner_radius_mm > min(self.width_mm, self.height_mm) / 2.0:
            problems.append(
                f"{self.label}的圆角超过了短边的一半，无法成形"
            )
        return problems


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

    texts: list[TextElement] = field(default_factory=list)
    """Text that belongs to the board itself.

    A template card is mostly wording, and the user asked for all of it to be
    visible and editable rather than baked into the outline. Empty for a board
    that is not a template.
    """

    cutouts: list[CutoutElement] = field(default_factory=list)
    """Openings cut through the board, as distinct from its outline."""

    groups: list[PlacementGroup] = field(default_factory=list)
    """Sets of placements the user edits as one element.

    板画 is the case this exists for: its two tones are one picture, and moving
    or deleting them apart leaves a board whose halves no longer line up. The
    groups are also written to the board file as EasyEDA Pro's own ``GROUP``
    records, so the editor shows them as one element there too.

    Kept on the document rather than on each placement, for the reason this
    project keeps making the same choice: a name duplicated across members can
    be renamed on one of them and then the two disagree, and there is no way to
    tell which is right.
    """

    operator_name: str = ""
    operator_number: str = ""
    operator_page: str = ""
    """Which operator the card is being made for.

    Held on the document rather than substituted into the text once, so the
    tokens stay in ``texts`` and switching operators does not lose the
    template's own wording. Empty until the user picks one.

    ``operator_page`` is the **identity**; the name is only what the card
    prints. 暮落 and 暮落(集成战略) have the same name and different wiki pages,
    and resolving the artwork by name gave the second one the first one's QR
    code. Older drafts predate the field and fall back to a name lookup.
    """

    art_choices: dict = field(default_factory=dict)
    """Per-element overrides for the artwork the card inserts automatically.

    Keyed by kind -- ``"qr"`` / ``"barcode"`` / ``"profession"`` /
    ``"faction"`` / ``"emblem"`` -- and valued:

    * absent, or ``""`` -- **automatic**, i.e. what the operator's own record
      says;
    * ``"none"`` -- the user removed that element from this card;
    * anything else -- the asset to use instead, by the name the panel lists
      (a profession, a faction, or ``"on"``/``"off"`` for the crest).

    A mapping rather than one field per element: the panel edits them
    uniformly, and an element added later needs no new document field. It
    round-trips through the draft as plain JSON, so old drafts simply have no
    entry and behave exactly as they did.
    """

    def __init__(
        self,
        outline: BoardOutlineSpec | None = None,
        placements: list[ImagePlacement] | None = None,
        outline_ready: bool = False,
        layers: LayerStack | None = None,
        image: ImagePlacement | None = None,
        texts: list[TextElement] | None = None,
        cutouts: list[CutoutElement] | None = None,
        groups: list[PlacementGroup] | None = None,
        operator_name: str = "",
        operator_number: str = "",
        operator_page: str = "",
        art_choices: dict | None = None,
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
        self.texts = list(texts) if texts else []
        self.cutouts = list(cutouts) if cutouts else []
        self.groups = list(groups) if groups else []
        # Minted on demand rather than in ``ImagePlacement.__post_init__``, so
        # a placement built by a test or a script only gains an identity once
        # something actually needs to refer to it.
        self._element_seq = 0
        self.operator_name = operator_name
        self.operator_number = operator_number
        self.operator_page = operator_page
        self.art_choices = dict(art_choices) if art_choices else {}
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

    # ------------------------------------------------------------- groups

    def element_id(self, element) -> str:
        """This object's stable identity, minting one if it has none.

        Takes a placement *or* a text line: both carry ``element_id`` and both
        can be put in a group, and one helper means the canvas and the model
        cannot disagree about which id belongs to what.

        Not part of the dataclasses' construction: a uuid generated in
        ``__post_init__`` would be regenerated by every ``replace()``, every
        draft load and every undo restore, so a group's member list would go
        stale the first time the user did anything.
        """
        if not getattr(element, "element_id", ""):
            self._element_seq += 1
            element.element_id = f"e{self._element_seq}"
        return element.element_id

    def element_by_id(self, element_id: str):
        """The placement or text line carrying ``element_id``, or ``None``."""
        for item in (*self.placements, *self.texts):
            if item.element_id == element_id:
                return item
        return None

    def placement_by_element_id(self, element_id: str) -> ImagePlacement | None:
        """The placement carrying ``element_id``, or ``None``."""
        for placement in self.placements:
            if placement.element_id == element_id:
                return placement
        return None

    def group_of(self, element) -> PlacementGroup | None:
        """The group ``element`` belongs to, or ``None``.

        Membership is looked up by id rather than by index the whole way
        through, so deleting an object cannot make a group claim a neighbour it
        never contained.
        """
        if element is None or not getattr(element, "element_id", ""):
            return None
        return self.group_by_id(element.element_id)

    def group_by_id(self, element_id: str) -> PlacementGroup | None:
        """The group whose member list holds ``element_id``."""
        for group in self.groups:
            if element_id in group.members:
                return group
        return None

    def group_members(self, group: PlacementGroup | None) -> list:
        """The group's live members, in document order.

        ⚠️ **Placements only.** The writers emit ``GROUP`` records over traced
        geometry, and a text line is written as its own ``STRING`` record with
        no group reference to set -- so a group containing text cannot yet be
        expressed in the board file, and reporting text here would make the
        exporters promise something they cannot write. Text membership is kept
        on the document and shown in the manager; :meth:`group_all_members`
        returns it for the UI.

        Order comes from :attr:`placements`, not from the member list: a member
        list records *which* objects are together, and document order is what
        the canvas, the layer panel and the exporters all read.
        """
        if group is None:
            return []
        wanted = set(group.members)
        return [
            item
            for item in self.placements
            if item.element_id and item.element_id in wanted
        ]

    def group_all_members(self, group: PlacementGroup | None) -> list:
        """Every live member of ``group`` -- placements and text, in order.

        The canvas and the group manager use this one, because the user picked
        the text too and a selection that quietly dropped it would make
        dragging a group leave its captions behind.
        """
        if group is None:
            return []
        wanted = set(group.members)
        ordered = [
            item
            for item in (*self.placements, *self.texts)
            if item.element_id and item.element_id in wanted
        ]
        return ordered

    def group_label(self, group: PlacementGroup) -> str:
        """What to call a group on screen.

        A group the user has not named still needs a label -- an unnamed row in
        the manager would be unclickable -- but the fallback is generated here
        and never stored, so it cannot go stale against the document.
        """
        if group.name.strip():
            return group.name.strip()
        index = self.groups.index(group) + 1 if group in self.groups else 1
        return f"组合 {index}"

    def make_group(
        self, elements: list, name: str = ""
    ) -> PlacementGroup | None:
        """Put ``elements`` into one new group.

        Accepts placements and text lines. Members already in another group are
        **moved** into the new one rather than being refused: selecting three
        objects and asking for a group is unambiguous about the outcome, and an
        object can only be in one group -- that is what the editor's single
        ``groupId`` field can express.

        Returns ``None`` when there is nothing to do: fewer than two objects
        that are actually on the board, or fewer than two *distinct* ones.
        Forming a one-member group would leave a row that can never be used.
        """
        known = {id(item): item for item in (*self.placements, *self.texts)}
        unique: list = []
        seen: set[int] = set()
        for element in elements:
            key = id(element)
            if key not in known or key in seen:
                continue
            seen.add(key)
            unique.append(known[key])
        if len(unique) < 2:
            return None

        self.ungroup(unique, drop_empty=True)
        from .epru import new_id

        group = PlacementGroup(
            id=new_id(),
            name=name.strip(),
            members=[self.element_id(item) for item in unique],
        )
        self.groups.append(group)
        return group

    def ungroup(
        self, elements: list, *, drop_empty: bool = True
    ) -> list[PlacementGroup]:
        """Take ``elements`` out of whatever group holds them.

        ``drop_empty`` removes a group left with fewer than two members, which
        is what "解散组合" means. Passing ``False`` keeps the shell so a caller
        that is about to re-add a different set can reuse the name.

        Returns the groups that were touched, so the caller can report what
        changed rather than describing the intent.
        """
        wanted = {
            item.element_id
            for item in elements
            if getattr(item, "element_id", "")
        }
        affected: list[PlacementGroup] = []
        for group in self.groups:
            remaining = [mid for mid in group.members if mid not in wanted]
            if len(remaining) == len(group.members):
                continue
            group.members = remaining
            affected.append(group)
        if drop_empty:
            self.groups = [group for group in self.groups if len(group.members) >= 2]
        return affected

    def dissolve_group(self, group: PlacementGroup) -> bool:
        """Break one group apart, keeping every placement."""
        if group not in self.groups:
            return False
        self.groups.remove(group)
        return True

    def rename_group(self, group: PlacementGroup, name: str) -> bool:
        """Give a group a name. Returns whether it changed."""
        cleaned = name.strip()
        if group not in self.groups or group.name == cleaned:
            return False
        group.name = cleaned
        return True

    def prune_groups(self) -> list[PlacementGroup]:
        """Drop member ids that no longer name a placement, and empty groups.

        Called before the document is written and before the layer list is
        built. A stale id is invisible on the canvas -- nothing draws for it --
        but it makes the group's size wrong everywhere it is reported, and the
        board file would carry a ``GROUP`` record whose members do not all
        exist.
        """
        live = {
            item.element_id
            for item in (*self.placements, *self.texts)
            if item.element_id
        }
        removed: list[PlacementGroup] = []
        for group in self.groups:
            group.members = [mid for mid in group.members if mid in live]
        for group in list(self.groups):
            if len(group.members) < 2:
                self.groups.remove(group)
                removed.append(group)
        return removed

    def visible_placements(self) -> list[ImagePlacement]:
        """Placements whose layer is currently shown, in draw order.

        Sorted by the layer stack -- copper over silkscreen over mask -- rather
        than by the order the objects happen to sit in :attr:`placements`.
        Insertion order is not a stacking decision: importing a silkscreen
        picture after a copper one used to paint it on top, so which of two
        overlapping objects won depended on which was imported last.
        """
        return self.in_layer_order(
            item
            for item in self.placements
            if self.layers.is_visible(item.layer_key)
        )

    @staticmethod
    def in_layer_order(items) -> list:
        """Sort drawables bottom-to-top by the layer stack, stably.

        Stable, so objects on one layer keep the order they were created in --
        within a layer that *is* the stacking decision the user made.
        """
        from .design_layers import paint_rank

        return sorted(items, key=lambda item: paint_rank(item.layer_key))

    def validate(self) -> list[str]:
        problems = list(self.outline.validate())
        for placement in self.placements:
            problems.extend(placement.validate())
        for element in self.texts:
            problems.extend(element.validate())
        for cutout in self.cutouts:
            problems.extend(cutout.validate())
        for group in self.groups:
            problems.extend(group.validate())
        return problems

    # -------------------------------------------------- text and cutouts

    def visible_texts(self) -> list[TextElement]:
        """Text lines whose layer is showing, in draw order.

        Layer-ordered for the same reason as :meth:`visible_placements`: a line
        does not go on top by being later in the list.
        """
        return self.in_layer_order(
            item for item in self.texts if self.layers.is_visible(item.layer_key)
        )

    def text_line(self, x_mm: float, y_mm: float) -> TextElement | None:
        """The topmost text line under a point, or ``None``.

        Checked before the artwork when picking: the lines are thin and sit on
        top of the picture, so a click that lands on one is a click on the
        words rather than on whatever is behind them.
        """
        for item in reversed(self.texts):
            if not self.layers.selectable(item.layer_key):
                continue
            if item.contains_point(x_mm, y_mm):
                return item
        return None

    def cutout(self, key: str) -> CutoutElement | None:
        """One cutout by its template key, for the panel to bind to."""
        for item in self.cutouts:
            if item.key == key:
                return item
        return None

    def cutout_at(self, x_mm: float, y_mm: float) -> CutoutElement | None:
        """The cutout under a point, if any."""
        for item in reversed(self.cutouts):
            if item.contains_point(x_mm, y_mm):
                return item
        return None

    def cutout_rings(self) -> list[list[tuple[float, float]]]:
        """Every opening as a closed ring, for the canvas and both exporters.

        One accessor rather than three readers of ``self.cutouts``: a hole that
        the canvas draws but the Gerber writer forgets is a board that comes
        back solid, and that is the failure this method exists to prevent.
        """
        return [item.ring() for item in self.cutouts]

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

    def outline_path(self) -> list[Any]:
        """The outline for the writer, keeping native arcs when it can.

        A standee tab is a separate polygon welded onto the board, so it has no
        arc metadata to preserve; that path falls back to the sampled ring.
        """
        if not self.outline_ready:
            return []
        if self.outline.standee:
            from .primitives import Point, polyline_path

            return polyline_path([Point(x, y) for x, y in self.outline_polygon()])
        return self.outline.outline_path()

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
            spec.template,
            spec.template_outline,
            spec.template_arcs,
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
            # The template ring is part of the outline, so it belongs in the
            # key: omitting it would keep the cached ring of the template that
            # was applied first.
            spec.template_outline,
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
