"""Polygon helpers for EasyEDA Pro complex polygons.

EasyEDA Pro represents an area as a *complex polygon*: a list of rings where the
first ring is the outer contour and the rest are holes. Rings wound one way are
solid, rings wound the other way are cut out, so getting the winding wrong turns
a hole into a blob.

Winding convention
------------------
The specification phrases the rule as "outer clockwise, holes counter-clockwise"
but does not say in which frame. Measuring real projects settles it. An ``IMAGE``
record from a board written by EasyEDA Pro 3.2.149, decoded and converted to
design coordinates, has::

    ring 0 (outer)  design area -1873511.9   -> clockwise
    ring 1 (hole)   design area +1357522.5   -> counter-clockwise

So the rule is stated in the **design frame**, Y up. Because the file itself
stores Y downward, the same rings look reversed in the raw file text -- which is
why the raw bytes appear to contradict the spec.

Every function in this module therefore works in millimetres, Y up, and
:func:`normalize_complex_polygon` produces clockwise outer rings and
counter-clockwise holes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .primitives import Mode

__all__ = [
    "Ring",
    "ArcRing",
    "signed_area",
    "ensure_ccw",
    "ensure_cw",
    "is_ccw",
    "normalize_complex_polygon",
    "normalize_arc_polygon",
    "reverse_arc_ring",
    "signed_area_arc",
    "is_ccw_arc",
    "arc_ring_to_path",
    "arc_polygon_to_path",
    "rect_ring",
    "circle_ring",
    "rounded_rect_ring",
    "valid_polygon",
    "safe_union",
    "ring_to_path",
    "complex_polygon_to_path",
    "path_to_rings",
    "sample_arc",
    "sample_cubic",
    "densify_path",
    "densify_ring",
]

Ring = Sequence[tuple[float, float]]


@dataclass
class ArcRing:
    """A closed ring whose edges are **lines and circular arcs**.

    ``sweeps[i]`` is the curve *arriving at* ``points[i]``; index 0 is the
    closing segment back to the first point. ``0.0`` means a straight edge.

    Why this exists beside :class:`Ring`: a chord-only polygon can only
    approximate a curve by adding vertices, so the pixel grid's staircase
    survives simplification -- either that, or the tolerance is raised until
    real corners are rounded off. Carrying the arc lets the editor draw the
    curve the trace actually describes. See :mod:`pcbart.vtrace`.
    """

    points: list[tuple[float, float]]
    sweeps: list[float]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.sweeps):
            raise ValueError(
                f"弧环的点数与扫向数不一致：{len(self.points)} 点 / "
                f"{len(self.sweeps)} 扫向"
            )

    @property
    def plain(self) -> list[tuple[float, float]]:
        """The vertices alone, for measuring and for winding."""
        return list(self.points)


def signed_area(ring: Ring) -> float:
    """Shoelace area. Positive means counter-clockwise in a Y-up frame."""
    total = 0.0
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def is_ccw(ring: Ring) -> bool:
    return signed_area(ring) > 0


def ensure_ccw(ring: Ring) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in ring]
    return points if is_ccw(points) else points[::-1]


def ensure_cw(ring: Ring) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in ring]
    return points if not is_ccw(points) else points[::-1]


def normalize_complex_polygon(
    outer: Ring,
    holes: Iterable[Ring] = (),
    islands: Iterable[Ring] = (),
) -> list[list[tuple[float, float]]]:
    """Build a complex polygon with the winding EasyEDA Pro expects.

    The rule, measured against the editor rather than read off the
    specification: **the hole must be wound opposite to the outer ring**, and
    the outer ring may go either way. A five-square experiment settled it --
    squares whose hole opposed the outer ring came out hollow, squares where
    the two matched came out solid, and the outer winding made no difference
    (see ``docs/M3_ENCODING.md``).

    This function emits outer clockwise and holes counter-clockwise, which is
    one of the two combinations that work, so nothing here needs changing.
    Additional solid regions -- separate islands inside a hole -- take the
    outer ring's winding.

    No Y flip happens on the way out: the file stores Y up-positive, the same
    direction as the design frame, so the winding above is the winding that
    reaches the editor. An earlier version of this docstring claimed the file
    stored Y downward and flipped it; that was written before the axis was
    measured and is wrong.
    """
    result = [ensure_cw(outer)]
    result.extend(ensure_cw(island) for island in islands if len(island) >= 3)
    result.extend(ensure_ccw(hole) for hole in holes if len(hole) >= 3)
    return result


def reverse_arc_ring(ring: ArcRing) -> ArcRing:
    """The same closed curve travelled the other way, arcs included.

    ⚠️ **Reversing a ring reverses every sweep.** The sweep is signed relative
    to the direction of travel, so walking the same arc backwards turns a
    counter-clockwise 90 degrees into a clockwise 90. A version that reversed
    the points and kept the sweeps produced a ring that read as its own
    complement -- every arc bulging to the wrong side of its chord.

    ⚠️ **And the sweeps have to shift by one, not stay put.** ``sweeps[i]``
    belongs to the segment *arriving at* ``points[i]``, so on the reversed ring
    the segment arriving at new vertex ``j`` is the reverse of the old segment
    arriving at ``points[(n-j) % n]`` -- not the one at ``n-1-j``. Off by one
    there pairs every arc with the wrong chord: the half-disc came back with
    its arc and its straight edge swapped, which draws a different shape.
    """
    count = len(ring.points)
    if count < 2:
        return ArcRing(list(ring.points), list(ring.sweeps))
    points = list(reversed(ring.points))
    sweeps = [-ring.sweeps[(count - index) % count] for index in range(count)]
    return ArcRing(points, sweeps)


def signed_area_arc(ring: ArcRing) -> float:
    """Enclosed area of a ring with arcs, **signed**, in the design frame.

    ⚠️ **A two-vertex circle has zero vertices-area but a real area.** The
    points alone span no polygon at all: a circle written as two points joined
    by two 180-degree sweeps has a chord area of exactly 0, and a half-disc's
    chords cancel to 0 too. Deciding the winding from the vertices therefore
    answered "clockwise" for both directions, and the normaliser left the ring
    as it found it -- which is a hole that fills in, or a shape that becomes
    its own complement.

    The true area adds each curved segment's **circular segment** on top of the
    chord contribution: ``(r^2/2) * (theta - sin theta)`` for a segment
    sweeping ``theta`` radians. For a two-arc circle that is ``2 * (r^2/2) * pi
    = pi * r^2``, and for a half-disc ``(r^2/2) * pi`` -- both exact.
    """
    import math

    points = ring.points
    count = len(points)
    total = 0.0
    for index in range(count):
        x1, y1 = points[index - 1]
        x2, y2 = points[index]
        # The chord's own contribution, exactly as for a plain polygon.
        total += x1 * y2 - x2 * y1
        sweep = float(ring.sweeps[index])
        if abs(sweep) < 1e-9:
            continue
        chord = math.hypot(x2 - x1, y2 - y1)
        half = math.radians(sweep) / 2.0
        sin_half = math.sin(half)
        if chord <= 1e-12 or abs(sin_half) <= 1e-12:
            continue
        radius = chord / (2.0 * sin_half)
        theta = math.radians(sweep)
        total += radius * radius * (theta - math.sin(theta))
    return total / 2.0


def is_ccw_arc(ring: ArcRing) -> bool:
    """Whether an arc ring is wound counter-clockwise."""
    return signed_area_arc(ring) > 0


def normalize_arc_polygon(
    outer: ArcRing,
    holes: Iterable[ArcRing] = (),
    islands: Iterable[ArcRing] = (),
) -> list[ArcRing]:
    """Arc rings wound the way :func:`normalize_complex_polygon` winds plains.

    Outer clockwise, holes counter-clockwise, islands like the outer ring --
    the same rule, except that a ring being turned around has to have its
    sweeps negated as well. See :func:`reverse_arc_ring`.

    ⚠️ **An arc ring needs only two vertices, where a chord ring needs three.**
    Two points joined by two 180-degree sweeps is a circle, and two points
    joined by one sweep and one chord is a half-disc -- both perfectly good
    areas that the plain-polygon ``>= 3`` rule would silently drop.

    The winding test is :func:`is_ccw_arc`, not :func:`is_ccw`: on rings like
    those two the vertex area is zero and the plain test answers arbitrarily.
    """
    result = [
        outer if not is_ccw_arc(outer) else reverse_arc_ring(outer)
    ]
    for island in islands:
        if len(island.points) >= 2:
            result.append(
                island if not is_ccw_arc(island) else reverse_arc_ring(island)
            )
    for hole in holes:
        if len(hole.points) >= 2:
            result.append(
                hole if is_ccw_arc(hole) else reverse_arc_ring(hole)
            )
    return result


def normalize_outline_rings(
    outer: Ring,
    holes: Iterable[Ring] = (),
) -> list[list[tuple[float, float]]]:
    """Rings for a board outline that contains cutouts.

    Each ring becomes its own closed curve on the board-outline layer, so the
    editor sees the shape by containment rather than by winding. The windings
    are still set opposite to each other so a winding-aware reader agrees.
    """
    result = [ensure_ccw(outer)]
    result.extend(ensure_cw(hole) for hole in holes if len(hole) >= 3)
    return result


def ring_to_path(ring: Ring) -> list[float | str]:
    """Convert a ring to a single-polygon ``L`` token stream, closed.

    The first vertex is repeated at the end. All 47 ``L``-form rings in the
    captured project are explicitly closed, and omitting it is not safe to
    assume away: a first attempt at the M3 experiment left the rings open and
    the editor showed a board with no copper on it at all.

    The ``R`` and ``CIRCLE`` forms are inherently closed and do not repeat a
    vertex; this function only produces ``L`` streams.

    For rings whose vertices are joined by arcs, use
    :func:`arc_ring_to_path` instead: a chord-only stream turns every curve
    into a chain of short straight segments, which is what makes a traced
    picture look ragged along its edges.
    """
    points = list(ring)
    if not points:
        return []
    out: list[float | str] = [points[0][0], points[0][1], Mode.L]
    for x, y in points[1:]:
        out.extend([x, y])
    if len(points) > 1 and points[-1] != points[0]:
        out.extend([points[0][0], points[0][1]])
    return out


def arc_polygon_to_path(rings: Sequence[ArcRing]) -> list[list[float | str]]:
    """A complex polygon of arc rings, in the shape a ``FILL`` body expects.

    The counterpart of :func:`complex_polygon_to_path` for geometry that
    carries curves: one token stream per ring, first ring the area and the rest
    its holes.
    """
    return [arc_ring_to_path(ring.points, ring.sweeps) for ring in rings]


def arc_ring_to_path(ring: Ring, sweeps: Sequence[float]) -> list[float | str]:
    """A closed ring whose segments are **arcs**, in the editor's own notation.

    The form is the one the template files use::

        [x0, y0, "L", x1, y1, "ARC", sweep, x2, y2, "L", x3, y3, ...]

    A point list where ``"ARC", <sweep>, x, y`` means "from the current point,
    sweep ``sweep`` degrees to ``(x, y)``". Measured from the source template's
    own ``FILL`` record: one ring mixes ``L`` and ``ARC`` segments, and the
    ARC's coordinates are an **absolute endpoint**, not an offset.

    ⚠️ **This is why traced artwork looked ragged.** ``findContours`` walks the
    pixel boundary, so a soft edge arrives as a staircase, and with chords
    alone the only way to smooth it is to move the vertices -- which either
    keeps the staircase (tolerance too small) or rounds off real corners (too
    large). Arcs let the editor draw the curve the fit describes.

    ``sweeps`` has one entry per vertex, the first being the closing segment
    (from the last point back to the first). A zero sweep is written as a
    straight ``L``.
    """
    points = list(ring)
    if not points or len(sweeps) != len(points):
        return ring_to_path(ring)

    out: list[float | str] = [points[0][0], points[0][1], Mode.L]
    for index in range(1, len(points)):
        x, y = points[index]
        sweep = float(sweeps[index])
        if abs(sweep) < 1e-6:
            out.extend([x, y])
        else:
            out.extend([Mode.ARC, sweep, x, y])
    closing = float(sweeps[0])
    if abs(closing) >= 1e-6:
        out.extend([Mode.ARC, closing, points[0][0], points[0][1]])
    elif points[-1] != points[0]:
        out.extend([points[0][0], points[0][1]])
    return out


def complex_polygon_to_path(rings: Sequence[Ring]) -> list[list[float | str]]:
    """Convert rings into the list-of-polygons form a ``FILL``/``POUR`` expects.

    Each ring is closed on its own, which is what makes the first one an area
    and the rest holes.
    """
    return [ring_to_path(ring) for ring in rings]


def path_to_rings(path: object) -> list[list[tuple[float, float]]]:
    """Best-effort decode of a path payload into rings of points."""
    if not isinstance(path, list):
        return []

    # Distinguish "list of polygons" from "one flat polygon".
    if path and isinstance(path[0], list):
        polygons = path
    else:
        polygons = [path]

    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        rings.append(_decode_single(polygon))
    return [ring for ring in rings if ring]


def _rect_ring(
    x: float,
    y: float,
    width: float,
    height: float,
    rotation_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """A standalone ``R`` rectangle in a Y-up frame.

    ``(x, y)`` is the top-left anchor and the rectangle grows downward, so the
    local corners use ``-height`` for the bottom edge.  A non-zero rotation is
    applied about that anchor, matching the file's own path grammar.
    """
    import math

    angle = math.radians(float(rotation_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    local = ((0.0, 0.0), (width, 0.0), (width, -height), (0.0, -height))
    return [
        (
            x + u * cos_a - v * sin_a,
            y + u * sin_a + v * cos_a,
        )
        for u, v in local
    ]


def sample_arc(
    start: tuple[float, float],
    end: tuple[float, float],
    sweep_deg: float,
    segments: int = 16,
) -> list[tuple[float, float]]:
    """Points along a circular arc, including its two endpoints.

    ``sweep_deg`` is signed and measured counter-clockwise in the Y-up frame
    the model works in.  EasyEDA stores the angle in the file's Y-down frame,
    so callers converting a record pass the negated value; that sign is what
    decides which side of the chord the centre sits on.
    """
    import math

    if segments < 2:
        segments = 2
    sweep = math.radians(float(sweep_deg))
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    chord = math.hypot(dx, dy)
    if chord <= 1e-12 or abs(sweep) <= 1e-12:
        return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]

    half = sweep / 2.0
    sin_half = math.sin(half)
    if abs(sin_half) <= 1e-12:
        return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
    radius = abs(chord / (2.0 * sin_half))
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    nx = -dy / chord
    ny = dx / chord
    height = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))
    sign = 1.0 if sweep > 0 else -1.0
    centre_x = mid_x + sign * height * nx
    centre_y = mid_y + sign * height * ny

    start_angle = math.atan2(start[1] - centre_y, start[0] - centre_x)
    out = []
    for index in range(segments + 1):
        angle = start_angle + sweep * index / segments
        out.append(
            (
                centre_x + radius * math.cos(angle),
                centre_y + radius * math.sin(angle),
            )
        )
    return out


def sample_cubic(
    start: tuple[float, float],
    control1: tuple[float, float],
    control2: tuple[float, float],
    end: tuple[float, float],
    segments: int = 16,
) -> list[tuple[float, float]]:
    """Points along a cubic Bezier, including both endpoints.

    EasyEDA stores these as ``x y C x1 y1 x2 y2 x3 y3``: the current point is
    the start, the next three pairs are the two controls and the endpoint.  A
    decoder that treats the controls as ordinary vertices turns a logo or a
    glyph into a fan of chords, and filling that fan is what produced the
    ellipse blobs on the card's side text.
    """
    if segments < 2:
        segments = 2
    out: list[tuple[float, float]] = []
    for index in range(segments + 1):
        t = index / segments
        u = 1.0 - t
        x = (
            u * u * u * start[0]
            + 3.0 * u * u * t * control1[0]
            + 3.0 * u * t * t * control2[0]
            + t * t * t * end[0]
        )
        y = (
            u * u * u * start[1]
            + 3.0 * u * u * t * control1[1]
            + 3.0 * u * t * t * control2[1]
            + t * t * t * end[1]
        )
        out.append((x, y))
    return out


def densify_path(node: object, segments: int = 16) -> list[list[tuple[float, float]]]:
    """Decode a path token stream into polylines, sampling arcs.

    A plain ``L`` stream loses every corner radius because an ``ARC`` token
    stores only its endpoint in the numeric run.  Sampling the curve here keeps
    a canvas preview and a rebuilt file from turning a rounded card corner into
    a diagonal cut.
    """
    if not isinstance(node, list) or not node:
        return []
    if isinstance(node[0], list):
        rings: list[list[tuple[float, float]]] = []
        for item in node:
            rings.extend(densify_path(item, segments))
        return rings

    head = node[0]
    if head == Mode.CIRCLE and len(node) >= 4:
        import math

        cx, cy, radius = float(node[1]), float(node[2]), float(node[3])
        steps = max(16, segments * 2)
        return [
            [
                (
                    cx + radius * math.cos(2 * math.pi * i / steps),
                    cy + radius * math.sin(2 * math.pi * i / steps),
                )
                for i in range(steps)
            ]
        ]
    if head == Mode.R and len(node) >= 5:
        x, y, width, height = (float(value) for value in node[1:5])
        rotation = float(node[5]) if len(node) >= 6 else 0.0
        return [_rect_ring(x, y, width, height, rotation)]

    points: list[tuple[float, float]] = []
    pending: list[float] = []
    index = 0

    def flush() -> None:
        for offset in range(0, len(pending) - 1, 2):
            points.append((pending[offset], pending[offset + 1]))

    while index < len(node):
        item = node[index]
        if isinstance(item, str):
            flush()
            pending.clear()
            index += 1
            if item in (Mode.ARC, Mode.CARC) and index + 2 < len(node):
                angle = -float(node[index])
                end = (float(node[index + 1]), float(node[index + 2]))
                if points:
                    points.extend(sample_arc(points[-1], end, angle, segments)[1:])
                else:
                    points.append(end)
                index += 3
            elif item == Mode.C and index + 5 < len(node):
                control1 = (float(node[index]), float(node[index + 1]))
                control2 = (float(node[index + 2]), float(node[index + 3]))
                end = (float(node[index + 4]), float(node[index + 5]))
                if points:
                    points.extend(
                        sample_cubic(points[-1], control1, control2, end, segments)[1:]
                    )
                else:
                    points.append(end)
                index += 6
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            pending.append(float(item))
        index += 1
    flush()
    return [points] if points else []


def densify_ring(
    points: Sequence[tuple[float, float]],
    arcs: Sequence[float | None] = (),
    segments: int = 16,
) -> list[tuple[float, float]]:
    """Densify a ring given its vertices and one arc sweep per segment.

    ``arcs[i]`` describes the segment from ``points[i]`` to
    ``points[(i + 1) % len(points)]``; ``None`` keeps a straight edge.
    """
    vertices = [(float(x), float(y)) for x, y in points]
    if len(vertices) < 2:
        return vertices
    out = [vertices[0]]
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        sweep = arcs[index] if index < len(arcs) else None
        if sweep is None:
            out.append(end)
        else:
            out.extend(sample_arc(start, end, -float(sweep), segments)[1:])
    if out and out[-1] == out[0]:
        out.pop()
    return out


def _decode_single(node: list) -> list[tuple[float, float]]:
    if not node:
        return []
    head = node[0]

    if head == Mode.CIRCLE and len(node) >= 4:
        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return _circle_points(cx, cy, r)

    if head == Mode.R and len(node) >= 5:
        x, y, w, h = float(node[1]), float(node[2]), float(node[3]), float(node[4])
        rotation = float(node[5]) if len(node) >= 6 else 0.0
        return _rect_ring(x, y, w, h, rotation)

    points: list[tuple[float, float]] = []
    pending: list[float] = []
    index = 0
    while index < len(node):
        item = node[index]
        if isinstance(item, str):
            points.extend(_pairs(pending))
            pending = []
            index += 1
            if item in (Mode.ARC, Mode.CARC):
                # angle, then endX endY
                if index + 2 < len(node):
                    points.append((float(node[index + 1]), float(node[index + 2])))
                index += 3
            continue
        if isinstance(item, (int, float)):
            pending.append(float(item))
        index += 1
    points.extend(_pairs(pending))
    return points


def _pairs(values: Sequence[float]) -> list[tuple[float, float]]:
    return [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]


def _circle_points(cx: float, cy: float, r: float, segments: int = 64) -> list[tuple[float, float]]:
    import math

    return [
        (cx + r * math.cos(2 * math.pi * i / segments), cy + r * math.sin(2 * math.pi * i / segments))
        for i in range(segments)
    ]


def rect_ring(x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    """A counter-clockwise rectangle ring."""
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def circle_ring(
    cx: float,
    cy: float,
    r: float,
    segments: int = 64,
) -> list[tuple[float, float]]:
    """A counter-clockwise circle approximated by ``segments`` vertices."""
    return _circle_points(cx, cy, r, segments)


def rounded_rect_ring(
    width: float,
    height: float,
    radius: float,
    segments: int = 12,
) -> list[tuple[float, float]]:
    """A rounded rectangle centred on the origin, wound counter-clockwise.

    Each corner is an arc of ``segments`` steps centred on the inset point, and
    consecutive arcs are joined by the straight edges between them. Both of
    those are load-bearing: emitting only the arcs, or winding them the wrong
    way, produces a self-intersecting polygon whose *bounds* are still the
    requested size while its area is far too small. That mistake shipped once
    -- a 60 × 40 board with a 5 mm radius came out 40% short on area -- because
    the size checks passed and nothing measured the geometry.
    """
    import math

    width = max(float(width), 0.1)
    height = max(float(height), 0.1)
    half_w, half_h = width / 2.0, height / 2.0
    radius = min(max(float(radius), 0.0), min(width, height) / 2.0)

    if radius <= 0:
        return [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

    # Counter-clockwise, starting at the top-right corner: each arc then runs
    # straight into the edge that follows it (top, left, bottom, right).
    corners = [
        (half_w - radius, half_h - radius, 0.0),
        (-half_w + radius, half_h - radius, 90.0),
        (-half_w + radius, -half_h + radius, 180.0),
        (half_w - radius, -half_h + radius, 270.0),
    ]
    points: list[tuple[float, float]] = []
    for cx, cy, start_deg in corners:
        for step in range(segments + 1):
            angle = math.radians(start_deg + 90.0 * step / segments)
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
    return points


# ------------------------------------------------------------ shapely safety


def valid_polygon(ring: Ring):
    """A Shapely polygon for ``ring`` that is safe to use in set operations.

    Traced contours routinely arrive self-intersecting or with repeated
    vertices: a bitmap boundary can touch itself at a single pixel, and
    simplification can fold a thin feature onto itself. Handing such a polygon
    to ``unary_union`` raises ``TopologyException: side location conflict`` and
    the whole outline fails -- which is how an irregular board stopped tracing
    entirely.

    ``buffer(0)`` is Shapely's standard repair. It can return a MultiPolygon
    when the ring pinches into separate pieces, so the caller is handed
    whatever comes out rather than assuming a single Polygon.

    Returns ``None`` when nothing usable remains.
    """
    from shapely.geometry import Polygon

    if len(ring) < 3:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def safe_union(shapes):
    """Union of several geometries, repairing each one first.

    Doing this in one place keeps every caller from having to remember the
    repair, and makes the failure mode explicit: invalid input is fixed rather
    than allowed to raise from deep inside Shapely with a message that names
    a coordinate and nothing else.

    GEOS can still raise ``TopologyException: side location conflict`` on
    *valid* input whose edges nearly coincide -- a traced silhouette running
    within a hair of the base rectangle's edge does exactly that. That is a
    numerical robustness failure, not a modelling one, so the union is retried
    on a snapped grid, and a final pairwise pass keeps whatever can be merged
    so a board still comes out instead of the whole outline failing.
    """
    from shapely.ops import unary_union

    usable = []
    for shape in shapes:
        if shape is None:
            continue
        candidate = shape
        if not candidate.is_valid:
            try:
                candidate = candidate.buffer(0)
            except Exception:  # noqa: BLE001 - an unrepairable piece is dropped
                continue
        if not candidate.is_empty and candidate.area > 0:
            usable.append(candidate)
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    try:
        return unary_union(usable)
    except Exception:  # noqa: BLE001 - retry on a snapped grid
        pass

    snapped = _snap_shapes(usable)
    if snapped is not None:
        try:
            return unary_union(snapped)
        except Exception:  # noqa: BLE001 - fall through to the pairwise pass
            pass

    merged = None
    for shape in usable:
        if merged is None:
            merged = shape
            continue
        try:
            merged = unary_union([merged, shape])
        except Exception:  # noqa: BLE001 - skip the piece that will not merge
            continue
    if merged is None:
        return None
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda part: part.area)
    return merged


def _snap_shapes(shapes):
    """The shapes snapped to a fine grid, or ``None`` if unsupported.

    Snapping moves a coordinate by at most half a grid step -- 0.05 micrometre
    at the precision used here, thousands of times below the 0.0254 mm file
    unit -- and is enough to stop nearly-coincident edges from confusing GEOS.
    """
    try:
        from shapely import set_precision

        return [set_precision(shape, 1e-4) for shape in shapes]
    except Exception:  # noqa: BLE001 - snapping is best-effort
        return None
