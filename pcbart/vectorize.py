"""Turn a line-art mask into copper geometry.

This is M3's first half: the picture becomes something the fab house can
actually etch.

Two styles are planned, and the seam between them is :data:`CopperStyle`:

``FILL``  (implemented)
    Every inked region becomes a complex polygon -- an outer ring plus the
    holes inside it -- written as one ``FILL``. What you see in the drawing is
    what ends up on the board, block for block.

``TRACE`` (reserved)
    The mask is thinned to centrelines and each centreline becomes a ``LINE``
    with a locally-measured width. Better for fine line work, but it *changes*
    the shape: a solid black area collapses to a single stroke. The entry point
    exists so the writing half does not have to be reworked when it lands.

Everything here works in millimetres in the design frame and returns plain
point lists, so it is testable without Qt and without the file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import normalize_complex_polygon

__all__ = [
    "CopperStyle",
    "VectorizeParams",
    "ComplexPolygon",
    "mask_to_complex_polygons",
    "vectorize_mask",
]


class CopperStyle:
    """How a mask is turned into copper."""

    FILL = "fill"
    """Region -> complex polygon -> ``FILL``. Implemented."""

    TRACE = "trace"
    """Centreline -> ``LINE`` with a measured width. Reserved for M3b."""

    LABELS = {
        FILL: "填充（还原色块）",
        TRACE: "走线（按线宽）",
    }


#: One shape: the outer ring first, then the holes cut out of it.
ComplexPolygon = list[list[tuple[float, float]]]


@dataclass
class VectorizeParams:
    """Settings for turning a mask into copper geometry."""

    style: str = CopperStyle.FILL

    simplify_mm: float = 0.08
    """Douglas-Peucker tolerance, in millimetres.

    Smaller than the outline tracer's default because copper carries far more
    detail than a board edge: at 0.35 mm a small glyph turns into a blob. The
    cost is vertex count, and therefore file size.
    """

    min_island_area_mm2: float = 0.02
    """Separate blobs smaller than this are dropped.

    Scanning noise leaves specks, and every speck would become its own copper
    island. The default is a fraction of a square millimetre -- big enough to
    remove dust, small enough to keep a full stop in a caption.
    """

    min_hole_area_mm2: float = 0.02
    """Holes smaller than this are filled in rather than cut out."""

    coord_snap_mm: float = 0.002
    """Merge vertices closer than this along an axis.

    Bitmap tracing leaves runs of nearly identical coordinates along what is
    conceptually one straight edge; collapsing them restores shared
    coordinates.
    """

    ink_is_dark: bool = True
    """Which end of the grey range is ink.

    The line-art pipeline emits 0 for ink and 255 for paper, so the default
    treats **dark** pixels as the shape. Getting this backwards is not subtle
    but it is easy to do: the first run produced a board whose copper was the
    paper, leaving the drawing as holes.
    """

    threshold: int = 128
    """Grey level splitting ink from paper, for non-binary inputs."""

    @property
    def is_implemented(self) -> bool:
        return self.style == CopperStyle.FILL


def vectorize_mask(
    mask: np.ndarray,
    px_per_mm: float,
    params: VectorizeParams | None = None,
) -> list[ComplexPolygon]:
    """Copper geometry for ``mask``, as complex polygons in millimetres.

    Dispatches on the style so the two implementations can share everything
    around them -- scaling, simplification, unit handling. ``TRACE`` raises
    rather than silently returning fill geometry: emitting a shape the user did
    not ask for is worse than saying it is not built yet.
    """
    params = params or VectorizeParams()

    if params.style == CopperStyle.FILL:
        return mask_to_complex_polygons(mask, px_per_mm, params)

    if params.style == CopperStyle.TRACE:
        raise NotImplementedError(
            "走线风格（骨架化 → 中心线 + 线宽）尚未实现，"
            "目前只支持填充风格。"
        )

    raise ValueError(f"未知的铜层风格: {params.style}")


def _as_grey_plane(source: np.ndarray) -> np.ndarray:
    """Normalise any ink mask to a 0/255 grey plane.

    Masks reach this module in more than one convention: the line-art pipeline
    emits 0/255, a caller tracing a boolean array passes True/False, and another
    passes 0/1. Treating the last two as grey levels is not a small mistake --
    with ``ink_is_dark`` every value from 0 to 1 is below the threshold, so the
    **entire rectangle becomes ink** and a QR symbol exports as a solid slab.
    Booleans and 0/1 arrays are therefore scaled to 0/255 before thresholding,
    which leaves a real 0/255 mask untouched.
    """
    import cv2

    plane = np.asarray(source)
    if plane.ndim == 3:
        plane = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)
    if plane.dtype == bool:
        return plane.astype(np.uint8) * 255
    if plane.size and plane.dtype != np.uint8:
        plane = plane.astype(np.uint8, copy=False)
    if plane.size and int(plane.max()) <= 1:
        return plane * 255
    return plane


def mask_to_complex_polygons(
    mask: np.ndarray,
    px_per_mm: float,
    params: VectorizeParams | None = None,
) -> list[ComplexPolygon]:
    """Trace every inked region into an outer ring plus its own holes.

    ``extract_outline_rings`` cannot be reused here: it keeps one outer ring
    and treats the rest as islands, which is right for a board edge and wrong
    for artwork, where every glyph, every dot and every stroke is its own
    region with its own holes -- the counter of an "o" belongs to that "o", not
    to whatever happens to be the largest shape on the page.

    Returns rings already wound the way the file needs them: outer clockwise,
    holes counter-clockwise. A hole wound the same way as its outer ring is
    ignored by the editor, which is what makes a letter's counter print solid.
    """
    import cv2

    params = params or VectorizeParams()
    if px_per_mm <= 0:
        raise ValueError("px_per_mm 必须为正数")

    source = _as_grey_plane(mask)
    # Ink is dark in the pipeline's output, so ink becomes the foreground.
    if params.ink_is_dark:
        binary = (source <= params.threshold).astype(np.uint8) * 255
    else:
        binary = (source > params.threshold).astype(np.uint8) * 255
    if not binary.any():
        return []

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or not len(contours):
        return []

    # RETR_CCOMP gives a flat two-level tree: each entry's parent is either -1
    # (an outer boundary) or the index of the outer boundary it sits in.
    hierarchy = hierarchy[0]
    to_mm = 1.0 / px_per_mm

    # Centre the result on the origin, matching the board outline and the
    # canvas. Without this the geometry sits with its origin at the image's
    # top-left corner while the board outline is built around its own centre,
    # so the copper lands half a picture diagonally away from the board --
    # drawn correctly, but in the wrong place.
    height_px, width_px = binary.shape[:2]
    offset_x = -width_px / 2.0
    offset_y = -height_px / 2.0

    def to_ring(contour) -> list[tuple[float, float]]:
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        simplified = _simplify(points, params.simplify_mm * px_per_mm)
        # Pixel Y grows downward; the design frame grows upward. The flip is
        # applied about the image's vertical centre, hence the offset.
        return [((x + offset_x) * to_mm, -(y + offset_y) * to_mm) for x, y in simplified]

    def ring_area(ring: list[tuple[float, float]]) -> float:
        total = 0.0
        for index, (x1, y1) in enumerate(ring):
            x2, y2 = ring[(index + 1) % len(ring)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    holes_by_parent: dict[int, list[list[tuple[float, float]]]] = {}
    outers: list[tuple[int, list[tuple[float, float]]]] = []

    for index, contour in enumerate(contours):
        ring = to_ring(contour)
        if len(ring) < 3:
            continue
        parent = int(hierarchy[index][3])
        if parent == -1:
            outers.append((index, ring))
        else:
            holes_by_parent.setdefault(parent, []).append(ring)

    polygons: list[ComplexPolygon] = []
    for index, outer in outers:
        if ring_area(outer) < params.min_island_area_mm2:
            continue
        holes = [
            hole
            for hole in holes_by_parent.get(index, [])
            if ring_area(hole) >= params.min_hole_area_mm2
        ]
        polygons.append(normalize_complex_polygon(outer, holes))

    # Biggest first, so a truncated file still carries the main artwork.
    polygons.sort(key=lambda rings: ring_area(rings[0]), reverse=True)
    return polygons


def _simplify(
    points: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    """Douglas-Peucker with a tolerance in pixels, guarded against collapse."""
    import cv2

    if len(points) < 4 or tolerance <= 0:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, float(tolerance), True)
    result = [(float(p[0][0]), float(p[0][1])) for p in approx]
    # A tolerance wide enough to erase a small ring returns almost nothing;
    # keeping the original is better than emitting a degenerate polygon.
    return result if len(result) >= 3 else points
