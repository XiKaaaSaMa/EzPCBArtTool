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

from typing import Iterable, Sequence

from .primitives import Mode

__all__ = [
    "Ring",
    "signed_area",
    "ensure_ccw",
    "ensure_cw",
    "is_ccw",
    "normalize_complex_polygon",
    "rect_ring",
    "circle_ring",
    "rounded_rect_ring",
    "valid_polygon",
    "safe_union",
    "ring_to_path",
    "complex_polygon_to_path",
    "path_to_rings",
]

Ring = Sequence[tuple[float, float]]


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


def _decode_single(node: list) -> list[tuple[float, float]]:
    if not node:
        return []
    head = node[0]

    if head == Mode.CIRCLE and len(node) >= 4:
        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return _circle_points(cx, cy, r)

    if head == Mode.R and len(node) >= 5:
        x, y, w, h = float(node[1]), float(node[2]), float(node[3]), float(node[4])
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

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
