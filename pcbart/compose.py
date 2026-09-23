"""Building an irregular board from the artwork plus a preset shape.

An irregular board is not "the artwork's outline instead of a shape". It is the
artwork's outline *combined with* one: a standee keeps a solid rectangular base
so it can stand, while everything above that base follows the character's
silhouette -- ears, hair, tails, whatever sticks out.

That gives one composition rule with three settings:

``trace``     the whole board is the artwork's outline (a free-standing cut-out)
``rect``      a rectangle forms the lower part, the outline keeps the upper part
``rounded``   the same, with the rectangle's bottom corners rounded

Everything here works in millimetres in the design frame: X to the right, Y up,
origin at the board's centre. The pixel-to-millimetre flip happens inside
:mod:`pcbart.outline`, which is the single place that knows images grow
downward and boards grow upward.

Geometry is composed with Shapely rather than by hand (a hand-written union was
already wrong once in this project). The result is returned as plain point
lists, so callers stay free of the dependency.
"""

from __future__ import annotations

import numpy as np

from .geometry import safe_union
from .outline import OutlineParams

__all__ = [
    "IRREGULAR_BASES",
    "IrregularBase",
    "base_polygon",
    "exterior_ring",
    "trace_artwork_ring",
    "combine_with_base",
]


class IrregularBase:
    """How an irregular board combines the artwork with a preset shape."""

    TRACE = "trace"
    RECT = "rect"
    ROUNDED = "rounded"

    LABELS = {
        TRACE: "原图裁切",
        RECT: "矩形结合",
        ROUNDED: "圆角矩形结合",
    }

    #: Bases that contribute a shape of their own. ``TRACE`` contributes
    #: nothing: the outline is the whole board.
    COMBINING = (RECT, ROUNDED)


IRREGULAR_BASES = (IrregularBase.TRACE, IrregularBase.RECT, IrregularBase.ROUNDED)


#: Half-extent of the cutting half-plane, in millimetres. Large enough that no
#: realistic board reaches it; it exists so the base can be cut by a straight
#: edge without writing the intersection by hand.
_CLIP_HALF = 10_000.0


def trace_artwork_ring(
    path: str,
    width_mm: float,
    height_mm: float,
    simplify_mm: float = 0.35,
) -> list[tuple[float, float]]:
    """The artwork's silhouette, in millimetres, centred on the artwork.

    Traced from the **subject** of the original picture, not from its line art.
    That distinction is the whole job: the line art marks strokes, so tracing it
    followed every hair line and produced spikes wherever strokes nearly
    touched -- the outline described the drawing rather than the character.
    :func:`pcbart.image_prep.subject_mask` fills what the strokes enclose, which
    is the footprint a board outline actually needs.

    ``width_mm`` and ``height_mm`` are the artwork's own placement size, so the
    outline lands exactly where the picture is drawn -- the scale comes from
    the placement rather than from fitting the silhouette's bounding box, which
    would creep inwards wherever the artwork has transparent margins.

    Returns an empty list when the artwork has no usable silhouette, which the
    caller can treat as "no outline yet" rather than as a failure.
    """
    import cv2

    from .image_prep import imread_unicode, subject_mask
    from .outline import extract_outline_rings

    width_mm = max(0.1, float(width_mm))
    height_mm = max(0.1, float(height_mm))

    raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    alpha: np.ndarray | None = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        image = raw[:, :, :3]
    else:
        image = raw

    mask = subject_mask(image, alpha, fill=True)
    if mask is None or not mask.any():
        return []

    source_h_px, source_w_px = mask.shape[:2]
    source_w_px = max(1, int(source_w_px))
    source_h_px = max(1, int(source_h_px))
    px_per_mm = source_w_px / width_mm

    # Tolerance is a pixel quantity, so it has to be converted from the
    # millimetre value the UI offers; a fixed pixel tolerance would smooth
    # small images away and leave large ones jagged.
    tolerance_px = max(0.5, simplify_mm * px_per_mm)

    try:
        rings = extract_outline_rings(
            mask,
            OutlineParams(

                pixels_per_mm=px_per_mm,
                simplify_tolerance_px=tolerance_px,
                # Circles are recognised after scaling and would become an
                # exact CIRCLE for the board; an artwork outline is rarely one
                # and the union needs a polygon either way.
                detect_circles=False,
                # No centring: the caller positions the ring, and centring
                # would silently move the artwork off its drawn position.
                center=False,
            ),
        )
    except ValueError:
        return []

    # extract_outline_rings returns millimetres measured from the image's
    # top-left corner, already flipped to Y-up. Shift so the ring is centred on
    # the artwork's own centre, matching how the placement is drawn. The height
    # used is the one the trace actually produced, so an artwork whose pixels
    # are not exactly the placement's aspect ratio still lines up.
    drawn_height_mm = source_h_px / px_per_mm
    offset_x = -width_mm / 2.0
    offset_y = drawn_height_mm / 2.0
    return [(x + offset_x, y + offset_y) for x, y in rings.outer]


def exterior_ring(polygon) -> list[tuple[float, float]]:
    """A polygon's outer ring as plain points, empty when there isn't one.

    A polygon with no area has no exterior to speak of: at ``top_ratio=0`` the
    base is cut away entirely and Shapely returns a zero-length ``LineString``
    rather than a Polygon. Calling ``.exterior`` on that raises, and this is
    reached from the canvas's paint pass, so it is handled here once instead of
    at every call site.
    """
    if polygon is None or polygon.is_empty:
        return []
    exterior = getattr(polygon, "exterior", None)
    if exterior is None:
        return []
    return [(float(x), float(y)) for x, y in list(exterior.coords)[:-1]]


def base_polygon(
    base: str,
    width_mm: float,
    height_mm: float,
    corner_radius_mm: float,
    top_ratio: float,
):
    """The preset part of a combined board, as a Shapely polygon.

    The shape is the preset's full footprint cut off at ``top_ratio``: below
    that line the board is the solid base, above it the artwork's silhouette
    is what remains. Cutting rather than shrinking keeps the base's rounded
    corners on the outside of the board where they belong.
    """
    from shapely.geometry import Polygon, box

    width_mm = max(0.1, float(width_mm))
    height_mm = max(0.1, float(height_mm))
    ratio = min(max(float(top_ratio), 0.0), 1.0)

    if base == IrregularBase.ROUNDED:
        from .geometry import rounded_rect_ring

        radius = min(
            max(0.0, float(corner_radius_mm)), min(width_mm, height_mm) / 2.0
        )
        shape = Polygon(rounded_rect_ring(width_mm, height_mm, radius))
    else:
        shape = box(
            -width_mm / 2.0,
            -height_mm / 2.0,
            width_mm / 2.0,
            height_mm / 2.0,
        )

    # At ratio 0 there is no base at all. Intersecting would leave a zero-area
    # sliver anyway, so return an empty polygon directly.
    if ratio <= 0.0:
        return Polygon()
    if ratio >= 1.0:
        return shape

    cut_y = -height_mm / 2.0 + ratio * height_mm
    return shape.intersection(
        box(-_CLIP_HALF, -_CLIP_HALF, _CLIP_HALF, cut_y)
    )


def combine_with_base(
    ring: list[tuple[float, float]],
    base: str,
    width_mm: float,
    height_mm: float,
    corner_radius_mm: float = 3.0,
    top_ratio: float = 0.35,
) -> list[tuple[float, float]]:
    """Union the artwork's silhouette with the preset base, as one ring.

    The artwork's outline and the base overlap wherever the artwork reaches
    down into the base, so the union closes into a single shape: a solid lower
    part with the silhouette rising out of it.

    Returns an empty list when there is no silhouette and no base -- the caller
    treats that as "no board yet" rather than as an error.

    This is where the reported ``TopologyException: side location conflict``
    came from. The silhouette is a *traced* ring: a bitmap boundary can touch
    itself at one pixel, and simplification folds thin features onto
    themselves. ``buffer(0)`` -- the usual repair -- raises that exception on
    some of those inputs rather than fixing them, and an unvalidated polygon
    handed to ``unary_union`` raises it too. Both are repaired through
    :mod:`pcbart.geometry` now, and a silhouette that cannot be repaired falls
    back to the base rather than failing the board.
    """
    base_shape = base_polygon(
        base, width_mm, height_mm, corner_radius_mm, top_ratio
    )

    if len(ring) < 3:
        return exterior_ring(base_shape)

    silhouette = _repaired_polygon(ring)
    if silhouette is None:
        # Nothing usable came out of the trace; the base is still a board.
        return exterior_ring(base_shape)

    merged = safe_union([base_shape, silhouette])
    if merged is None:
        return exterior_ring(base_shape)

    if merged.geom_type == "MultiPolygon":
        # The artwork floats clear of the base, so the two never joined. The
        # larger piece is reported as the board, the same way a stray island
        # is dropped elsewhere in this codebase; drawing two disjoint boards
        # would be a worse answer than one honest one.
        merged = max(merged.geoms, key=lambda part: part.area)
    if merged.is_empty:
        return []

    return exterior_ring(merged)


def _repaired_polygon(ring: list[tuple[float, float]]):
    """A valid polygon for a traced ring, or ``None``.

    Repairs in stages, because the usual single ``buffer(0)`` can itself raise
    on a pathological ring: first drop points that repeat a neighbour, which
    removes the commonest cause of self-touching, then try the polygon and
    ``buffer(0)``, and finally a tiny negative buffer that pulls a folded edge
    apart instead of resolving the crossing.
    """
    from shapely.geometry import Polygon

    cleaned = _dedupe_consecutive(ring)
    if len(cleaned) < 3:
        return None

    candidates = []
    try:
        candidates.append(Polygon(cleaned))
    except Exception:  # noqa: BLE001 - a broken ring is not an error here
        pass

    if candidates and not candidates[0].is_valid:
        for repair in (0.0, -1e-9):
            try:
                candidates.append(candidates[0].buffer(repair))
            except Exception:  # noqa: BLE001 - try the next repair
                continue
    elif not candidates:
        return None

    for candidate in candidates:
        try:
            if candidate.is_valid and not candidate.is_empty and candidate.area > 0:
                return candidate
        except Exception:  # noqa: BLE001 - skip an unusable candidate
            continue
    return None


def _dedupe_consecutive(
    ring: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop points that repeat the previous one.

    A traced contour frequently contains runs of identical vertices where the
    boundary stalled on a pixel; they carry no shape information and are a
    common source of degenerate edges.
    """
    result: list[tuple[float, float]] = []
    for point in ring:
        x, y = float(point[0]), float(point[1])
        if result and result[-1][0] == x and result[-1][1] == y:
            continue
        result.append((x, y))
    # A ring that ends where it started has a duplicate seam vertex.
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result
