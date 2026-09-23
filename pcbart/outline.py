"""Turn a binary mask into board-outline rings.

The chain here is::

    mask -> contours (with hole hierarchy) -> simplify -> scale to mm
         -> process compensation -> validated rings

Two decisions matter more than the rest:

**Holes.** A board outline can contain interior cutouts -- the counter of a
letter "P", the middle of a ring. OpenCV returns those as a parent/child
hierarchy; dropping them silently fills the cutout and ships the wrong board.

**Point count.** A traced bitmap contour has one vertex per boundary pixel, so a
400 px wide shape arrives with well over a thousand points. Written straight to
the file that is megabytes of JSON, and the editor becomes sluggish to open and
edit. Simplification is therefore not optional; :func:`simplify_ring` exposes
the tolerance directly so callers can trade fidelity against file size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "OutlineParams",
    "OutlineResult",
    "mask_to_rings_px",
    "simplify_ring",
    "scale_rings",
    "fit_rings_to_box",
    "extract_outline_rings",
    "detect_circle",
    "snap_to_axes",
    "snap_coordinates",
]


@dataclass
class OutlineParams:
    """Controls for contour extraction, simplification and scaling."""

    simplify_tolerance_px: float = 1.5
    """Douglas-Peucker tolerance in pixels. 0 keeps every boundary pixel."""

    axis_snap_deg: float = 5.0
    """Snap outline edges within this many degrees of an axis onto it exactly.

    A traced bitmap is a staircase, and Douglas-Peucker approximates each
    staircase with one diagonal line, so a feature that should be perfectly
    vertical comes out a fraction of a degree off. Over a 70 mm edge that is a
    visible slant, and it also breaks the assumption that board edges are
    axis-aligned. Snapping removes it.

    Set to 0 to disable. Values above roughly 10 degrees start bending
    genuinely diagonal edges.
    """

    coord_snap_px: float = 0.75
    """Merge vertices whose x (or y) differ by less than this, in pixels.

    Bitmap tracing puts vertices a fraction of a pixel apart along what is
    conceptually one straight edge. Collapsing them restores shared
    coordinates and removes the small residual wobble."""

    min_ring_area_px: float = 64.0
    """Whole rings smaller than this are discarded."""

    target_width_mm: float | None = None
    """Scale so the outline's bounding box is this wide."""

    target_height_mm: float | None = None
    """Scale so the outline's bounding box is this tall."""

    pixels_per_mm: float | None = None
    """Direct scale, overriding the target dimensions."""

    preserve_aspect: bool = True
    """Keep the source aspect ratio when scaling."""

    offset_mm: float = 0.0
    """Grow (positive) or shrink (negative) the outline.

    Positive values can enforce a minimum feature width; negative values inset
    the board from the artwork. Applied with Shapely's buffer.
    """

    center: bool = True
    """Move the result so its bounding-box centre is at the origin."""

    origin: tuple[float, float] = (0.0, 0.0)
    """Final translation applied after centring."""

    detect_circles: bool = True
    """Emit an exact ``CIRCLE`` for rings that are demonstrably circular.

    A circle traced from a bitmap becomes a polygon, and even 60 vertices read
    as a faceted disc in the editor. EasyEDA Pro has a native ``CIRCLE`` path
    form, so a ring that fits one closely is written as one instead.
    """

    circle_tolerance: float = 0.03
    """Relative radial deviation allowed when recognising a circle.

    A perfect circle has zero deviation; 0.03 means the radius may vary by up
    to 3 percent. Rounded squares and slots exceed this comfortably, so they
    stay as polygons.
    """


@dataclass
class OutlineResult:
    outer: list[tuple[float, float]]
    scale_mm_per_px: float = 0.0
    source_size_px: tuple[int, int] = (0, 0)
    points_before: int = 0
    points_after: int = 0
    outer_circle: tuple[float, float, float] | None = None
    """``(cx, cy, r)`` in millimetres when the outer ring is circular."""

    @property
    def ring_count(self) -> int:
        return 1

    @property
    def uses_native_circles(self) -> bool:
        return self.outer_circle is not None

    def summary(self) -> str:
        circles = ", outer as CIRCLE" if self.outer_circle is not None else ""
        return (
            f"outer {len(self.outer)} points; "
            f"{self.points_before} -> {self.points_after} points; "
            f"scale {self.scale_mm_per_px:.6f} mm/px{circles}"
        )


def detect_circle(
    ring: list[tuple[float, float]],
    tolerance: float = 0.03,
) -> tuple[float, float, float] | None:
    """Return ``(cx, cy, r)`` if ``ring`` is circular within ``tolerance``.

    The test is on radial spread: every vertex must sit within
    ``tolerance * r`` of the mean radius about the centroid. That accepts a
    traced circle (which is uniformly faceted) and rejects a square, a slot or
    a rounded rectangle, whose corners or flat sides deviate far more.
    """
    if len(ring) < 8:
        return None

    n = len(ring)
    # Centroid of the polygon, not the vertex average: a traced circle has
    # uneven vertex spacing and the vertex average drifts.
    area2 = 0.0
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-9:
        return None
    cx /= 3.0 * area2
    cy /= 3.0 * area2

    radii = [math.hypot(x - cx, y - cy) for x, y in ring]
    if not radii:
        return None
    mean_r = sum(radii) / len(radii)
    if mean_r <= 0:
        return None

    deviation = max(abs(r - mean_r) for r in radii) / mean_r
    if deviation <= tolerance:
        return (cx, cy, mean_r)
    return None


def mask_to_rings_px(
    mask: np.ndarray,
    min_ring_area_px: float = 64.0,
) -> list[list[tuple[float, float]]]:
    """Trace a mask into pixel-space outer rings, largest first.

    Only outermost contours are returned. A board outline is one closed edge,
    and an enclosed ring nested inside it is drawn by the editor rather than
    cut, so tracing one would produce geometry that claims a hole the board
    does not have.
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    outers = [_contour_to_points(c) for c in contours]
    outers = [r for r in outers if abs(_polygon_area(r)) >= min_ring_area_px]
    if not outers:
        raise ValueError("掩膜中没有找到有效轮廓")
    outers.sort(key=lambda r: abs(_polygon_area(r)), reverse=True)
    return outers


def _contour_to_points(contour) -> list[tuple[float, float]]:
    return [(float(p[0][0]), float(p[0][1])) for p in contour]


def _polygon_area(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def simplify_ring(
    ring: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Douglas-Peucker simplification; ``tolerance`` is in the ring's own units."""
    if tolerance <= 0 or len(ring) <= 4:
        return list(ring)

    import cv2

    contour = np.array(ring, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, tolerance, True)
    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return points if len(points) >= 3 else list(ring)


def scale_rings(
    rings: list[list[tuple[float, float]]],
    scale_mm_per_px: float,
) -> list[list[tuple[float, float]]]:
    """Scale pixel rings into millimetres, flipping Y to up-positive.

    Images have Y growing downward; board coordinates grow upward. Flipping
    here keeps the artwork the right way up in the editor.
    """
    return [
        [(x * scale_mm_per_px, -y * scale_mm_per_px) for x, y in ring]
        for ring in rings
    ]


def snap_to_axes(
    ring: list[tuple[float, float]],
    angle_tolerance_deg: float = 5.0,
) -> list[tuple[float, float]]:
    """Force edges that are nearly horizontal or vertical to be exactly so.

    Douglas-Peucker turns a staircase into a single diagonal, leaving a
    long "vertical" edge a fraction of a degree off-axis. Over tens of
    millimetres that is plainly visible, and it also means a board edge that
    should align with another does not.

    Each edge within the tolerance is rotated about its own midpoint, so
    neighbouring edges stay connected to within half the correction. Runs of
    consecutive axis-aligned edges are then unified onto a shared coordinate to
    remove the residual mismatch.
    """
    if angle_tolerance_deg <= 0 or len(ring) < 3:
        return list(ring)

    points = list(ring)
    n = len(points)
    tolerance = math.radians(angle_tolerance_deg)

    adjusted: list[tuple[float, float]] = []
    vertical: list[bool] = []
    horizontal: list[bool] = []

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            adjusted.append((x2, y2))
            vertical.append(False)
            horizontal.append(False)
            continue

        angle = math.atan2(dy, dx)
        # Distance from the edge direction to the nearest axis.
        to_horizontal = min(abs(angle), abs(abs(angle) - math.pi))
        to_vertical = abs(abs(angle) - math.pi / 2)

        if to_vertical <= tolerance:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half = length / 2.0
            adjusted.append((mx, my + half))
            vertical.append(True)
            horizontal.append(False)
        elif to_horizontal <= tolerance:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half = length / 2.0
            adjusted.append((mx + half, my))
            vertical.append(False)
            horizontal.append(True)
        else:
            adjusted.append((x2, y2))
            vertical.append(False)
            horizontal.append(False)

    # Rebuild the ring: the end of each edge and the start of the next must be
    # the same point, so average the two after snapping.
    result: list[tuple[float, float]] = []
    for i in range(n):
        prev_end = adjusted[(i - 1) % n]
        this_start = points[i]
        if vertical[(i - 1) % n]:
            # Previous edge is vertical: its x is authoritative.
            result.append((prev_end[0], this_start[1]))
        elif horizontal[(i - 1) % n]:
            result.append((this_start[0], prev_end[1]))
        else:
            result.append(this_start)

        if vertical[i]:
            result[-1] = (adjusted[i][0], result[-1][1])
        elif horizontal[i]:
            result[-1] = (result[-1][0], adjusted[i][1])

    return result


def snap_coordinates(
    ring: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Collapse vertices sharing nearly-equal coordinates.

    Tracing one straight bitmap edge yields many vertices whose x values differ
    by a fraction of a pixel. Rewriting each to a shared value removes the
    wobble and lets later stages recognise the edge as one straight run.
    """
    if tolerance <= 0 or len(ring) < 3:
        return list(ring)

    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]

    def cluster(values: list[float]) -> dict[float, float]:
        ordered = sorted(set(values))
        mapping: dict[float, float] = {}
        group: list[float] = [ordered[0]]
        for value in ordered[1:]:
            if value - group[-1] <= tolerance:
                group.append(value)
            else:
                representative = sum(group) / len(group)
                for member in group:
                    mapping[member] = representative
                group = [value]
        representative = sum(group) / len(group)
        for member in group:
            mapping[member] = representative
        return mapping

    x_map = cluster(xs)
    y_map = cluster(ys)
    return [(x_map[x], y_map[y]) for x, y in ring]


def simplify_ring_axis_aware(
    ring: list[tuple[float, float]],
    tolerance: float,
    angle_tolerance_deg: float = 5.0,
    coord_tolerance: float = 0.75,
) -> list[tuple[float, float]]:
    """Simplify, then straighten whatever came out nearly axis-aligned."""
    simplified = simplify_ring(ring, tolerance)
    snapped = snap_to_axes(simplified, angle_tolerance_deg)
    return snap_coordinates(snapped, coord_tolerance)


def fit_rings_to_box(
    rings: list[list[tuple[float, float]]],
    target_width_mm: float | None,
    target_height_mm: float | None,
) -> float:
    """Compute the scale that fits all rings into the requested box.

    With both targets given the smaller scale wins, so the artwork always fits
    even when its aspect ratio differs from the requested one.
    """
    if target_width_mm is None and target_height_mm is None:
        raise ValueError("需要指定 target_width_mm 或 target_height_mm")

    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        raise ValueError("轮廓包围盒无效")

    candidates: list[float] = []
    if target_width_mm:
        candidates.append(target_width_mm / width)
    if target_height_mm:
        candidates.append(target_height_mm / height)
    return min(candidates)


def extract_outline_rings(
    mask: np.ndarray,
    params: OutlineParams | None = None,
) -> OutlineResult:
    """Full mask-to-rings pipeline, returning millimetre coordinates.

    The outline is validated and process-corrected with Shapely: small holes are
    dropped, the optional offset is applied, and the result is guaranteed to be
    a set of simple, non-self-intersecting rings.
    """
    from .geometry import safe_union, valid_polygon

    params = params or OutlineParams()
    height_px, width_px = mask.shape[:2]

    outer_rings_px = mask_to_rings_px(
        mask,
        min_ring_area_px=params.min_ring_area_px,
    )

    points_before = sum(len(r) for r in outer_rings_px)

    def prepare(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Simplify, then straighten, unless the ring is actually a circle.

        Axis snapping must not touch a circle: it would pull every vertex within
        tolerance of an axis onto that axis, turning the disc into a rounded
        square and inflating its radius. So circularity is checked first and
        such rings are left alone.
        """
        simplified = simplify_ring(ring, params.simplify_tolerance_px)
        if params.detect_circles and detect_circle(simplified, params.circle_tolerance):
            return simplified
        return snap_coordinates(
            snap_to_axes(simplified, params.axis_snap_deg), params.coord_snap_px
        )

    outer_simplified = [prepare(r) for r in outer_rings_px]

    if params.pixels_per_mm:
        scale = 1.0 / params.pixels_per_mm
    elif params.target_width_mm or params.target_height_mm:
        scale = fit_rings_to_box(
            outer_simplified, params.target_width_mm, params.target_height_mm
        )
    else:
        # Fall back to a physical size that keeps the artwork printable:
        # assume the source is a 100 mm wide design.
        scale = 100.0 / width_px if width_px else 1.0

    outer_mm = scale_rings(outer_simplified, scale)

    # Build with Shapely so buffer/validation happen on real geometry.
    #
    # Every ring is repaired before the union. A traced contour can touch
    # itself at a single pixel, and handing that to ``unary_union`` raises
    # ``TopologyException: side location conflict`` -- which took out irregular
    # board tracing completely with a message naming only a coordinate.
    main = max(outer_mm, key=lambda ring: abs(_polygon_area(ring)))
    others = [r for r in outer_mm if r is not main]

    candidates = [valid_polygon(main)]
    candidates.extend(valid_polygon(island) for island in others)
    combined = safe_union(candidates)
    if combined is None:
        raise ValueError("轮廓处理失败：没有可用的外框几何")

    if params.offset_mm:
        grown = combined.buffer(params.offset_mm, join_style=1, mitre_limit=5.0)
        if grown.is_empty:
            raise ValueError("工艺补偿后轮廓为空，请减小补偿值")
        combined = grown

    # Offsetting changes the ring geometry, so rebuild the ring from the
    # result rather than carrying the pre-offset vertices forward.
    outer_rings_final, _ = _rings_from_geometry(combined)
    if not outer_rings_final:
        raise ValueError("轮廓处理失败：没有剩余外框")

    main_ring = max(outer_rings_final, key=lambda r: abs(_polygon_area(r)))

    if params.center:
        min_x = min(p[0] for p in main_ring)
        max_x = max(p[0] for p in main_ring)
        min_y = min(p[1] for p in main_ring)
        max_y = max(p[1] for p in main_ring)
        shift_x = -(min_x + max_x) / 2.0 + params.origin[0]
        shift_y = -(min_y + max_y) / 2.0 + params.origin[1]
        main_ring = [(x + shift_x, y + shift_y) for x, y in main_ring]

    # Recognise the circle after the ring geometry is final, so the fit is made
    # on the same numbers that would otherwise be written as a polygon.
    outer_circle: tuple[float, float, float] | None = None
    if params.detect_circles:
        # detect_circle runs on the already-shifted ring, so its centre is
        # final; adding the shift again would translate the circle twice.
        outer_circle = detect_circle(main_ring, params.circle_tolerance)

    # A ring written as a native CIRCLE costs one path form, not one vertex per
    # sample, so it counts as a single point here.
    points_after = 1 if outer_circle is not None else len(main_ring)

    return OutlineResult(
        outer=main_ring,
        scale_mm_per_px=scale,
        source_size_px=(width_px, height_px),
        points_before=points_before,
        points_after=points_after,
        outer_circle=outer_circle,
    )


def _rings_from_geometry(geometry) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Split a Shapely (Multi)Polygon into outer rings and holes.

    Outer rings are wound clockwise and holes counter-clockwise, matching the
    convention EasyEDA Pro uses in the design frame.
    """
    outers: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []

    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        outers.append(_orient(list(polygon.exterior.coords)[:-1], clockwise=True))
        for interior in polygon.interiors:
            holes.append(_orient(list(interior.coords)[:-1], clockwise=False))

    outers.sort(key=lambda r: abs(_polygon_area(r)), reverse=True)
    return outers, holes


def _orient(ring: list[tuple[float, float]], clockwise: bool) -> list[tuple[float, float]]:
    area = _polygon_area(ring)
    is_clockwise = area < 0
    if is_clockwise == clockwise:
        return ring
    return list(reversed(ring))
