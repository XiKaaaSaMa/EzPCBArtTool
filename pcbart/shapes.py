"""Closed-outline shapes for board geometry.

All coordinates are millimetres in a Y-up frame. Shapes are returned as plain
vertex lists so they can be fed straight into a polyline outline; the Y flip
into EasyEDA Pro's screen frame happens later, at the encoding boundary.

Every builder supports ``origin`` (translation) and ``rotation_deg``, and
returns vertices in counter-clockwise order for a convex shape.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "regular_polygon_points",
    "regular_star_points",
    "star_circumradius",
    "star_inner_radius",
    "circle_points_mm",
    "polygon_edge_lengths",
    "polygon_perimeter",
    "bounding_box",
    "translate_points",
    "rotate_points",
    "center_on_origin",
]

Point = tuple[float, float]


def star_inner_radius(points: int, circumradius: float) -> float:
    """Inner radius of the classic star with the given outer radius.

    For a five-pointed star the shape is fixed: the inner (concave) vertices
    are the intersection points of the pentagram's five strokes. Solving for
    one of those intersections puts it at::

        r = R * cos(2*pi/n) / cos(pi/n)

    which for ``n = 5`` is ``R / phi^2`` (about ``0.381966 R``). This is what
    makes the result a 五角星 rather than an arbitrary zig-zag ring.
    """
    if points < 3:
        raise ValueError("a star needs at least 3 points")
    return circumradius * math.cos(2.0 * math.pi / points) / math.cos(math.pi / points)


def star_circumradius(points: int, edge_length: float) -> float:
    """Outer radius of the classic star whose ten edges are all ``edge_length``.

    With the inner radius fixed by :func:`star_inner_radius`, the edge joining
    an outer vertex to an adjacent inner one (they sit ``pi/n`` apart) has
    length::

        L = R * tan(pi / n)

    so the outer radius follows by inverting it::

        R = L / tan(pi / n)

    For ``n = 5`` and ``L = 30 mm`` this gives ``R = 41.2911 mm``.
    """
    if points < 3:
        raise ValueError("a star needs at least 3 points")
    if edge_length <= 0:
        raise ValueError("edge length must be positive")
    return edge_length / math.tan(math.pi / points)


def regular_star_points(
    points: int = 5,
    edge_length: float = 30.0,
    origin: Point = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> list[Point]:
    """Vertices of the classic star where every one of the ``2*points`` edges is equal.

    Vertices alternate outer, inner, outer, inner... around the centre, each
    step advancing ``pi/points``. With ``rotation_deg=0`` an odd-pointed star
    points straight up, because the first outer vertex is placed at 90 degrees.

    The result is not explicitly closed: the closing edge from the last vertex
    back to the first is implied, matching how the outline is stored.
    """
    radius_outer = star_circumradius(points, edge_length)
    radius_inner = star_inner_radius(points, radius_outer)

    vertices: list[Point] = []
    step = 2.0 * math.pi / points
    base = math.pi / 2.0 + math.radians(rotation_deg)

    for index in range(points):
        outer_angle = base + index * step
        inner_angle = outer_angle + step / 2.0
        vertices.append(
            (
                radius_outer * math.cos(outer_angle),
                radius_outer * math.sin(outer_angle),
            )
        )
        vertices.append(
            (
                radius_inner * math.cos(inner_angle),
                radius_inner * math.sin(inner_angle),
            )
        )

    return translate_points(vertices, origin)


def regular_polygon_points(
    sides: int = 6,
    edge_length: float = 30.0,
    origin: Point = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> list[Point]:
    """Vertices of a regular polygon with a given edge length."""
    if sides < 3:
        raise ValueError("a polygon needs at least 3 sides")
    if edge_length <= 0:
        raise ValueError("edge length must be positive")

    radius = edge_length / (2.0 * math.sin(math.pi / sides))
    step = 2.0 * math.pi / sides
    base = math.pi / 2.0 + math.radians(rotation_deg)

    vertices = [
        (radius * math.cos(base + i * step), radius * math.sin(base + i * step))
        for i in range(sides)
    ]
    return translate_points(vertices, origin)


def circle_points_mm(
    cx: float,
    cy: float,
    radius: float,
    segments: int = 64,
) -> list[Point]:
    """A counter-clockwise circle as a vertex ring, in millimetres.

    Used when a circle has to be handled as an ordinary ring (validation,
    area calculations) rather than as EasyEDA Pro's native ``CIRCLE`` path.
    """
    if radius <= 0:
        raise ValueError("圆的半径必须为正数")
    if segments < 8:
        segments = 8
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / segments),
            cy + radius * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]


def polygon_edge_lengths(points: Sequence[Point], closed: bool = True) -> list[float]:
    """Length of each edge, including the closing edge when ``closed``."""
    if len(points) < 2:
        return []
    count = len(points) if closed else len(points) - 1
    lengths: list[float] = []
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    return lengths


def polygon_perimeter(points: Sequence[Point], closed: bool = True) -> float:
    return sum(polygon_edge_lengths(points, closed))


def bounding_box(points: Sequence[Point]) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)``."""
    if not points:
        raise ValueError("no points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def translate_points(points: Sequence[Point], offset: Point) -> list[Point]:
    dx, dy = offset
    return [(x + dx, y + dy) for x, y in points]


def rotate_points(
    points: Sequence[Point],
    angle_deg: float,
    around: Point = (0.0, 0.0),
) -> list[Point]:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = around
    result: list[Point] = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        result.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return result


def center_on_origin(points: Sequence[Point]) -> list[Point]:
    """Translate so the bounding-box centre lands on the origin."""
    min_x, min_y, max_x, max_y = bounding_box(points)
    return translate_points(points, (-(min_x + max_x) / 2.0, -(min_y + max_y) / 2.0))
