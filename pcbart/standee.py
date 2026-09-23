"""The standee tab: a small rectangle joined to the bottom edge of the board.

A standee is a board that stands upright in a base, and the tab is the piece
that plugs into that base. It hangs below the board, and its upper edge lies on
the board's own bottom edge so the two read as one piece of material.

That shared edge is the whole problem here. Two shapes that meet along a *line*
union into one polygon; two shapes that meet at a single *point* do not, and
Shapely returns a ``MultiPolygon`` -- which for a board outline would mean a tab
that is not attached, or a tab that silently disappears when only one ring can
be written. A circular board is exactly that case: its lowest point is a point,
so a tab whose top edge sits on it touches at one vertex and nowhere else.

The fix is a small **weld**: the tab reaches a little way up *into* the board.
The exposed part is still exactly the requested height -- everything above the
board's bottom edge is inside the board's own footprint and vanishes into the
union -- but the joint now has real area, so it survives the union and the tab
cannot fall off. On a board whose bottom is a flat edge, the welded and the
unwelded tab produce the identical outline.

Everything is in millimetres in the design frame: X to the right, Y up. The
board's own bottom edge is found from the outline rather than assumed to be at
``-height/2``, because a traced erratic outline is not centred on the origin.
"""

from __future__ import annotations

from .compose import exterior_ring
from .geometry import safe_union, valid_polygon

__all__ = [
    "DEFAULT_TAB_WIDTH_MM",
    "DEFAULT_TAB_HEIGHT_MM",
    "attach_tab",
    "contact_span",
    "ring_bounds",
    "travel_mm",
    "weld_mm",
]


DEFAULT_TAB_WIDTH_MM = 16.0
"""Default tab width, in millimetres, measured along the board's bottom edge."""

DEFAULT_TAB_HEIGHT_MM = 3.0
"""Default tab height, in millimetres, measured downwards from that edge."""

_WELD_MIN_MM = 0.1
_WELD_MAX_MM = 0.5
"""How far the tab reaches up into the board.

Bounded on both sides: deep enough to connect to a curved bottom (a circle's
lowest region is narrow, so a hairline weld would miss it), and shallow enough
that it stays inside the smallest board this tool allows.
"""


def weld_mm(tab_height_mm: float) -> float:
    """The weld depth for a tab of this height.

    Measured in the same units as the tab, so a tall tab is welded as deeply as
    it protrudes and a stubby one still gets a joint that holds.
    """
    return min(_WELD_MAX_MM, max(_WELD_MIN_MM, float(tab_height_mm)))


def ring_bounds(ring) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of a ring, in millimetres."""
    xs = [float(point[0]) for point in ring]
    ys = [float(point[1]) for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


def contact_span(ring, tab_height_mm: float) -> tuple[float, float] | None:
    """Where the board's material sits along the strip the tab attaches to.

    Returns the left and right extent of the board inside the weld band just
    above its lowest point, or ``None`` when there is no usable geometry. This
    is what both the tab's position and the slider's range are derived from:
    the tab can only slide along the material that is actually there, so a
    narrow or curved bottom reports a narrow range instead of a range whose
    ends would leave the tab floating in mid-air.
    """
    if len(ring) < 3:
        return None

    from shapely.geometry import box

    polygon = _board_polygon(ring)
    if polygon is None:
        return None

    min_x, min_y, max_x, _max_y = polygon.bounds
    band = box(
        min_x - 1.0,
        min_y,
        max_x + 1.0,
        min_y + weld_mm(tab_height_mm),
    )
    try:
        contact = polygon.intersection(band)
    except Exception:  # noqa: BLE001 - a broken ring is reported as no contact
        return None
    if contact.is_empty:
        return None
    left, _bottom, right, _top = contact.bounds
    if right <= left:
        return None
    return float(left), float(right)


def travel_mm(ring, tab_width_mm: float, tab_height_mm: float) -> float:
    """How far the tab's centre may slide sideways, in millimetres.

    Half the contact span less half the tab, so the whole tab stays over solid
    board at either end of the slider. A bottom narrower than the tab reports
    zero: the tab is still placed, but there is nowhere to move it to, and the
    panel says so rather than offering a slider that cannot do anything.
    """
    span = contact_span(ring, tab_height_mm)
    if span is None:
        return 0.0
    left, right = span
    return max(0.0, (right - left - max(0.0, float(tab_width_mm))) / 2.0)


def attach_tab(
    ring,
    width_mm: float,
    height_mm: float,
    offset_mm: float = 0.0,
):
    """Union a standee tab onto ``ring``.

    ``offset_mm`` slides the tab sideways from the centre of the contact span.
    Returns ``(ring, attached)``: the outline to draw, and whether the tab is
    really part of it. A tab that cannot reach the board at all -- slid past
    the end of a narrow bottom, or a bottom narrower than the tab itself -- is
    left off and reported, because a board that quietly loses a feature the
    user switched on is worse than one that says it could not.
    """
    from shapely.geometry import box

    board = _board_polygon(ring)
    if board is None:
        return list(ring), False

    span = contact_span(ring, height_mm)
    centre_x = 0.0 if span is None else (span[0] + span[1]) / 2.0
    _min_x, min_y, _max_x, _max_y = ring_bounds(ring)

    half = max(0.05, float(width_mm)) / 2.0
    height = max(0.05, float(height_mm))
    tab = box(
        centre_x + float(offset_mm) - half,
        min_y - height,
        centre_x + float(offset_mm) + half,
        min_y + weld_mm(height),
    )

    merged = safe_union([board, tab])
    if merged is None or merged.is_empty:
        return list(ring), False

    if merged.geom_type == "MultiPolygon":
        # More than one piece means the tab never reached the board: the weld
        # band is designed so that any real attachment overlaps by area, and an
        # overlap unions into a single polygon. Picking the largest piece here
        # would silently trade one failure for a worse one -- a tab slid off
        # the end leaves the *tab* as a piece of its own, and adopting it would
        # export a standee with no board attached to it.
        return list(ring), False

    result = exterior_ring(merged)
    if len(result) < 3:
        return list(ring), False
    return result, True


def _board_polygon(ring):
    """The outline as a Shapely polygon, reduced to its largest piece."""
    polygon = valid_polygon(ring)
    if polygon is None:
        return None
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda part: part.area)
    if polygon.is_empty:
        return None
    return polygon
