"""PCB primitive payloads, plus conversion between raw units and millimetres.

EasyEDA Pro stores everything in one flat unit (1 unit == 1 mil). File Y grows
upward, the same direction as the design frame, so Y passes through unchanged;
the conversion is a pure scale, including for Y values buried inside path
arrays.

The path grammar is EasyEDA Pro's own, a small SVG-``path``-like notation::

    x y L x y x y ...              polyline: every pair is a vertex
    x y ARC angle endX endY        two-point arc
    x y CARC angle endX endY       centre arc
    x1 y1 C x2 y2 x3 y3 x4 y4      cubic Bezier
    R x y w h rot isCCW round      rectangle (standalone)
    CIRCLE cx cy r [isCCW]         circle (standalone)

These tokens are deliberately asymmetric, so conversion walks the stream rather
than treating every number as a coordinate.

Field roles were captured from a real EasyEDA Pro 3.2.149 project; see
``docs/M0_FINDINGS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .units import unit_to_mm, mm_to_unit, y_from_file, y_to_file

__all__ = [
    "Point",
    "PathSegment",
    "Mode",
    "board_outline_rect",
    "line",
    "poly",
    "fill",
    "circle_path",
    "rect_path",
    "record_to_mm",
    "body_to_unit",
    "shape_to_mm",
    "shape_to_unit",
    "special_pads_to_mm",
    "special_pads_to_unit",
    "outline_bbox_mm",
]

# Fields whose values are lengths along X.
X_FIELDS = ("startX", "endX", "centerX", "x", "originX")
# Fields whose values are lengths along Y (negated on the way out).
Y_FIELDS = ("startY", "endY", "centerY", "y", "originY")
# Fields that are lengths but not axis-dependent.
SIZE_FIELDS = (
    "width",
    "height",
    "holeDiameter",
    "viaDiameter",
    "innerDiameter",
    "fontSize",
    "strokeWidth",
    "thickness",
    "size",
    "radius",
    "r",
    "round",
    "drill",
)

# Path holders, and how deeply their polygons are nested.
PATH_FIELDS = ("path",)

# Drill and pad shape descriptors. EasyEDA Pro writes these as objects with
# named lengths, for example::
#
#     "hole":       {"holeType": "ROUND", "width": 39.3701, "height": 39.3701}
#     "defaultPad": {"padType": "ELLIPSE", "width": 39.3701, "height": 39.3701}
#
# The format specification describes the same fields positionally
# (``["ROUND", width, height]``), but no project written by EasyEDA Pro 3.2.x
# uses that form, and the editor silently discards a pad encoded that way --
# the file still opens, it just has no hole in it. Always emit the object form.
SHAPE_FIELDS = ("hole", "defaultPad")
SHAPE_LENGTH_KEYS = ("width", "height", "radius")


class Mode:
    """Path grammar tokens."""

    L = "L"
    ARC = "ARC"
    CARC = "CARC"
    C = "C"
    R = "R"
    CIRCLE = "CIRCLE"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class PathSegment:
    """A decoded path, kept in the same token order to allow round trips."""

    raw: list[Any]
    mode: str
    points: list[Point] = field(default_factory=list)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def convert_path_to_mm(path: Any) -> Any:
    """Convert a path payload to millimetres (scaling and flipping Y)."""

    def walk(node: Any, depth: int) -> Any:
        if isinstance(node, list):
            # A list of polygons (nested) or a flat single-polygon token stream.
            if node and isinstance(node[0], list):
                return [walk(item, depth + 1) for item in node]
            if node and isinstance(node[0], str) and node[0] in (Mode.R, Mode.CIRCLE):
                return _convert_standalone(node)
            if any(isinstance(item, str) for item in node):
                return _convert_stream(node)
            # Unknown shape: scale every number conservatively as a coordinate pair.
            return [unit_to_mm(v) if _is_number(v) else v for v in node]
        return node

    return walk(path, 0)


def convert_path_to_unit(path: Any) -> Any:
    """Convert a path from millimetres back to raw units."""

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            if node and isinstance(node[0], list):
                return [walk(item) for item in node]
            if node and isinstance(node[0], str) and node[0] in (Mode.R, Mode.CIRCLE):
                return _convert_standalone_out(node)
            if any(isinstance(item, str) for item in node):
                return _convert_stream_out(node)
            return [mm_to_unit(v) if _is_number(v) else v for v in node]
        return node

    return walk(path)


def _convert_standalone(node: list[Any]) -> list[Any]:
    mode = node[0]
    out: list[Any] = [mode]
    if mode == Mode.CIRCLE:
        # CIRCLE cx cy r [isCCW]
        out.append(unit_to_mm(node[1]))
        out.append(unit_to_mm(y_from_file(node[2])))
        out.append(unit_to_mm(node[3]))
        out.extend(node[4:])
        return out
    # R x y w h rot isCCW round
    out.append(unit_to_mm(node[1]))
    out.append(unit_to_mm(y_from_file(node[2])))
    out.append(unit_to_mm(node[3]))
    out.append(unit_to_mm(node[4]))
    out.append(node[5])
    out.extend(node[6:7])
    if len(node) > 7:
        out.append(unit_to_mm(node[7]))
    return out


def _convert_standalone_out(node: list[Any]) -> list[Any]:
    mode = node[0]
    out: list[Any] = [mode]
    if mode == Mode.CIRCLE:
        out.append(mm_to_unit(node[1]))
        out.append(mm_to_unit(y_to_file(node[2])))
        out.append(mm_to_unit(node[3]))
        out.extend(node[4:])
        return out
    out.append(mm_to_unit(node[1]))
    out.append(mm_to_unit(y_to_file(node[2])))
    out.append(mm_to_unit(node[3]))
    out.append(mm_to_unit(node[4]))
    out.append(node[5])
    out.extend(node[6:7])
    if len(node) > 7:
        out.append(mm_to_unit(node[7]))
    return out


def _convert_stream(node: list[Any]) -> list[Any]:
    """Convert a mixed token stream (``L`` / ``ARC`` / ``CARC`` / ``C`` runs)."""
    out: list[Any] = []
    i = 0
    pending: list[float] = []

    def flush_coords(coords: list[float]) -> None:
        for index, value in enumerate(coords):
            out.append(
                unit_to_mm(y_from_file(value)) if index % 2 == 1 else unit_to_mm(value)
            )

    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            if pending:
                flush_coords(pending)
                pending = []
            out.append(item)
            i += 1
            if item in (Mode.ARC, Mode.CARC):
                # angle, then endX endY
                out.append(node[i])
                out.append(unit_to_mm(node[i + 1]))
                out.append(unit_to_mm(y_from_file(node[i + 2])))
                i += 3
            continue
        pending.append(float(item))
        i += 1

    if pending:
        flush_coords(pending)
    return out


def _convert_stream_out(node: list[Any]) -> list[Any]:
    out: list[Any] = []
    i = 0
    pending: list[float] = []

    def flush_coords(coords: list[float]) -> None:
        for index, value in enumerate(coords):
            out.append(
                mm_to_unit(y_to_file(value)) if index % 2 == 1 else mm_to_unit(value)
            )

    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            if pending:
                flush_coords(pending)
                pending = []
            out.append(item)
            i += 1
            if item in (Mode.ARC, Mode.CARC):
                out.append(node[i])
                out.append(mm_to_unit(node[i + 1]))
                out.append(mm_to_unit(y_to_file(node[i + 2])))
                i += 3
            continue
        pending.append(float(item))
        i += 1

    if pending:
        flush_coords(pending)
    return out


def record_to_mm(record: Any) -> dict[str, Any] | None:
    """Return a primitive payload with units converted to millimetres."""
    body = getattr(record, "body", None)
    if not isinstance(body, dict):
        return None

    out: dict[str, Any] = {}
    for key, value in body.items():
        if key in X_FIELDS and _is_number(value):
            out[key] = unit_to_mm(value)
        elif key in Y_FIELDS and _is_number(value):
            out[key] = unit_to_mm(y_from_file(value))
        elif key in SIZE_FIELDS and _is_number(value):
            out[key] = unit_to_mm(value)
        elif key in PATH_FIELDS:
            out[key] = convert_path_to_mm(value)
        elif key in SHAPE_FIELDS:
            out[key] = shape_to_mm(value)
        elif key == "specialPad":
            out[key] = special_pads_to_mm(value)
        elif key == "coords" and isinstance(value, list):
            out[key] = [
                unit_to_mm(y_from_file(v)) if index % 2 else unit_to_mm(v)
                for index, v in enumerate(value)
                if _is_number(v)
            ]
        else:
            out[key] = value
    return out


def shape_to_mm(value: Any) -> Any:
    """Convert a drill or pad shape descriptor's lengths to millimetres.

    Handles both the object form EasyEDA Pro writes and the positional array
    form the specification documents, so files from either source can be read.
    """
    if isinstance(value, dict):
        out = dict(value)
        for key in SHAPE_LENGTH_KEYS:
            if _is_number(out.get(key)):
                out[key] = unit_to_mm(out[key])
        return out
    if isinstance(value, list) and len(value) >= 3:
        out = list(value)
        for index in range(1, len(out)):
            if _is_number(out[index]):
                out[index] = unit_to_mm(out[index])
        return out
    return value


def shape_to_unit(value: Any, decimals: int = 4) -> Any:
    """Inverse of :func:`shape_to_mm`."""
    if isinstance(value, dict):
        out = dict(value)
        for key in SHAPE_LENGTH_KEYS:
            if _is_number(out.get(key)):
                out[key] = mm_to_unit(float(out[key]), decimals)
        return out
    if isinstance(value, list) and len(value) >= 3:
        out = list(value)
        for index in range(1, len(out)):
            if _is_number(out[index]):
                out[index] = mm_to_unit(float(out[index]), decimals)
        return out
    return value


def special_pads_to_mm(value: Any) -> Any:
    """Convert each ``{"startLayer", "endLayer", "pad"}`` entry's pad."""
    if not isinstance(value, list):
        return value
    out = []
    for entry in value:
        if isinstance(entry, dict) and "pad" in entry:
            copy = dict(entry)
            copy["pad"] = shape_to_mm(copy["pad"])
            out.append(copy)
        else:
            out.append(entry)
    return out


def special_pads_to_unit(value: Any, decimals: int = 4) -> Any:
    """Inverse of :func:`special_pads_to_mm`."""
    if not isinstance(value, list):
        return value
    out = []
    for entry in value:
        if isinstance(entry, dict) and "pad" in entry:
            copy = dict(entry)
            copy["pad"] = shape_to_unit(copy["pad"], decimals)
            out.append(copy)
        else:
            out.append(entry)
    return out


def body_to_unit(body_mm: dict[str, Any], decimals: int = 4) -> dict[str, Any]:
    """Convert a millimetre primitive payload into raw ``.epru`` units.

    The inverse of :func:`record_to_mm`, and the single place where the
    design-frame to file-frame conversion happens on the way out.
    """
    out: dict[str, Any] = {}
    for key, value in body_mm.items():
        if key in X_FIELDS and _is_number(value):
            out[key] = mm_to_unit(value, decimals)
        elif key in Y_FIELDS and _is_number(value):
            out[key] = mm_to_unit(y_to_file(value), decimals)
        elif key in SIZE_FIELDS and _is_number(value):
            out[key] = mm_to_unit(value, decimals)
        elif key in PATH_FIELDS:
            out[key] = convert_path_to_unit(value)
        elif key in SHAPE_FIELDS:
            out[key] = shape_to_unit(value, decimals)
        elif key == "specialPad":
            out[key] = special_pads_to_unit(value, decimals)
        elif key == "coords" and isinstance(value, list):
            out[key] = [
                mm_to_unit(y_to_file(v), decimals) if index % 2 else mm_to_unit(v, decimals)
                for index, v in enumerate(value)
                if _is_number(v)
            ]
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------ builders


def circle_path(cx_mm: float, cy_mm: float, r_mm: float) -> list[Any]:
    """A single-polygon circle, in millimetres."""
    return [Mode.CIRCLE, cx_mm, cy_mm, r_mm]


def rect_path(x_mm: float, y_mm: float, w_mm: float, h_mm: float, round_mm: float = 0.0) -> list[Any]:
    """A standalone rectangle polygon, in millimetres."""
    return [Mode.R, x_mm, y_mm, w_mm, h_mm, 0, 0, round_mm]


def polyline_path(points: Sequence[Point]) -> list[Any]:
    """A polyline token stream from vertices, in millimetres."""
    if not points:
        return []
    out: list[Any] = [points[0].x, points[0].y, Mode.L]
    for point in points[1:]:
        out.extend([point.x, point.y])
    return out


def line(
    layer_id: int,
    start: Point,
    end: Point,
    width_mm: float,
    net: str = "",
    locked: bool = False,
) -> dict[str, Any]:
    return {
        "partitionId": "",
        "groupId": 0,
        "netName": net,
        "layerId": layer_id,
        "startX": start.x,
        "startY": start.y,
        "endX": end.x,
        "endY": end.y,
        "width": width_mm,
        "locked": locked,
        "zIndex": -1,
    }


def poly(
    layer_id: int,
    path_mm: list[Any],
    width_mm: float,
    net: str = "",
    locked: bool = False,
    poly_type: str = "NORMAL",
) -> dict[str, Any]:
    """A ``POLY`` payload. ``path_mm`` is a single polygon (flat token stream).

    The board outline is expressed this way: one ``POLY`` per edge chain,
    drawn on layer 11.
    """
    return {
        "partitionId": "",
        "groupId": 0,
        "netName": net,
        "layerId": layer_id,
        "width": width_mm,
        "path": path_mm,
        "locked": locked,
        "zIndex": -1,
        "polyType": poly_type,
    }


def fill(
    layer_id: int,
    complex_polygon_mm: list[Any],
    width_mm: float,
    net: str = "",
    style: str = "SOLID",
    locked: bool = False,
) -> dict[str, Any]:
    """A ``FILL`` payload.

    ``complex_polygon_mm`` is a complex polygon: a list whose first entry is the
    outer contour and whose remaining entries are holes. EasyEDA Pro expects
    the outer contour wound the opposite way from the holes, so pass the result
    of :mod:`pcbart.geometry` rather than hand-built lists.
    """
    return {
        "partitionId": "",
        "groupId": 0,
        "netName": net,
        "layerId": layer_id,
        "width": width_mm,
        "fillStyle": style,
        "path": complex_polygon_mm,
        "locked": locked,
        "zIndex": -1,
        "isBridgingCopper": False,
        "networkList": [],
        "refs": None,
    }


def board_outline_rect(
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    line_width_mm: float = 0.125,
) -> dict[str, Any]:
    """A closed rectangular board outline as a single ``POLY`` on layer 11.

    The square corners are two short arcs in EasyEDA Pro's own output; a plain
    polyline is simpler and is what a hand-drawn outline amounts to, so it is
    the form used to validate the round trip.
    """
    points = [
        Point(x_mm, y_mm),
        Point(x_mm + width_mm, y_mm),
        Point(x_mm + width_mm, y_mm + height_mm),
        Point(x_mm, y_mm + height_mm),
        Point(x_mm, y_mm),
    ]
    return poly(11, polyline_path(points), line_width_mm)


def outline_bbox_mm(records: Sequence[Any]) -> tuple[float, float, float, float] | None:
    """Bounding box of outline records, in millimetres.

    Coordinates are converted to a Y-up frame, so the result is
    ``(min_x, min_y, max_x, max_y)`` in ordinary design-tool orientation.
    """
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        body = record_to_mm(record)
        if not body:
            continue
        for key in ("startX", "endX"):
            if _is_number(body.get(key)):
                xs.append(float(body[key]))
        for key in ("startY", "endY"):
            if _is_number(body.get(key)):
                ys.append(float(body[key]))
        path = body.get("path")
        if path is not None:
            for x, y in _iter_path_points(path):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _iter_path_points(node: Any):
    """Yield ``(x, y)`` pairs from any path nesting, best effort."""
    if not isinstance(node, list):
        return
    if node and isinstance(node[0], list):
        for item in node:
            yield from _iter_path_points(item)
        return
    if node and node[0] == Mode.CIRCLE:
        cx, cy, r = node[1], node[2], node[3]
        yield (cx - r, cy - r)
        yield (cx + r, cy + r)
        return
    if node and node[0] == Mode.R:
        x, y, w, h = node[1], node[2], node[3], node[4]
        yield (x, y)
        yield (x + w, y + h)
        return

    i = 0
    pending: list[float] = []
    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            for index in range(0, len(pending) - 1, 2):
                yield (pending[index], pending[index + 1])
            pending = []
            i += 1
            if item in (Mode.ARC, Mode.CARC):
                if i + 2 < len(node):
                    yield (node[i + 1], node[i + 2])
                i += 3
            continue
        if _is_number(item):
            pending.append(float(item))
        i += 1
    for index in range(0, len(pending) - 1, 2):
        yield (pending[index], pending[index + 1])
