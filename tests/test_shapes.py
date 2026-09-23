"""Shape-builder tests, focused on the star geometry.

A five-pointed star board outline is a decagon: five outer tips alternating
with five inner notches. For the classic star the inner vertices are the
pentagram's intersection points, which fixes the inner radius at
``R * cos(2*pi/5) / cos(pi/5) = R / phi^2`` and makes every edge ``R * tan(36)``
long. These tests pin that relationship down, because getting it wrong
produces a shape that still *looks* like a star but has the wrong edge length.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.geometry import is_ccw  # noqa: E402
from pcbart.shapes import (  # noqa: E402
    bounding_box,
    center_on_origin,
    polygon_edge_lengths,
    polygon_perimeter,
    regular_polygon_points,
    regular_star_points,
    star_circumradius,
    star_inner_radius,
)

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def test_inner_radius_is_outer_over_phi_squared():
    """The pentagram intersection sits at R/phi^2, about 0.381966 R."""
    radius = 41.2915
    assert star_inner_radius(5, radius) == pytest.approx(radius / PHI**2, rel=1e-12)
    assert star_inner_radius(5, 1.0) == pytest.approx(0.381966, abs=1e-6)


def test_circumradius_from_edge_length():
    """R = edge / tan(pi/n); for a 30 mm edge that is 41.2911 mm."""
    assert star_circumradius(5, 30.0) == pytest.approx(41.2911, abs=1e-3)


def test_edge_length_relation_holds():
    """Every edge must equal R * tan(pi/n)."""
    radius = star_circumradius(5, 30.0)
    inner = star_inner_radius(5, radius)
    half_step = math.pi / 5
    edge = math.sqrt(radius**2 + inner**2 - 2 * radius * inner * math.cos(half_step))
    assert edge == pytest.approx(30.0, rel=1e-12)
    assert edge == pytest.approx(radius * math.tan(math.pi / 5), rel=1e-12)


def test_star_has_ten_equal_edges():
    points = regular_star_points(points=5, edge_length=30.0)
    lengths = polygon_edge_lengths(points, closed=True)

    assert len(points) == 10
    assert len(lengths) == 10
    for length in lengths:
        assert length == pytest.approx(30.0, rel=1e-12)
    assert max(lengths) - min(lengths) < 1e-9


def test_star_perimeter_is_ten_edges():
    points = regular_star_points(points=5, edge_length=30.0)
    assert polygon_perimeter(points) == pytest.approx(300.0, rel=1e-12)


def test_star_vertices_alternate_between_two_radii():
    points = regular_star_points(points=5, edge_length=30.0)
    radii = [math.hypot(x, y) for x, y in points]

    outer = star_circumradius(5, 30.0)
    inner = star_inner_radius(5, outer)
    for index, radius in enumerate(radii):
        expected = outer if index % 2 == 0 else inner
        assert radius == pytest.approx(expected, rel=1e-12)


def test_star_angles_step_by_pi_over_five():
    """Consecutive vertices are 36 degrees apart for a five-pointed star."""
    points = regular_star_points(points=5, edge_length=30.0)
    angles = [math.atan2(y, x) for x, y in points]

    for index in range(len(angles)):
        delta = angles[(index + 1) % len(angles)] - angles[index]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        assert abs(delta) == pytest.approx(math.pi / 5, rel=1e-9)


def test_star_points_up_by_default():
    points = regular_star_points(points=5, edge_length=30.0)
    min_x, min_y, max_x, max_y = bounding_box(points)
    # The topmost vertex sits on the Y axis for an upward-pointing star.
    tip = max(points, key=lambda p: p[1])
    assert tip[0] == pytest.approx(0.0, abs=1e-9)
    assert tip[1] == pytest.approx(max_y, abs=1e-12)


def test_star_winding_is_counter_clockwise():
    points = regular_star_points(points=5, edge_length=30.0)
    assert is_ccw(points)


def test_star_is_a_valid_simple_polygon():
    """A board outline must be a valid, non-self-intersecting ring.

    Shapely is the authority here rather than a hand-rolled orientation test:
    a naive sign test reports a crossing for edges that are merely *collinear*
    without overlapping, which is what the five-pointed star's horizontal
    edges do.
    """
    shapely = pytest.importorskip("shapely.geometry")
    from shapely.validation import explain_validity

    points = regular_star_points(points=5, edge_length=30.0)
    polygon = shapely.Polygon(points)

    assert polygon.is_valid, explain_validity(polygon)
    assert polygon.exterior.is_simple
    assert polygon.area > 0


def test_star_has_no_non_adjacent_edge_intersections():
    shapely = pytest.importorskip("shapely.geometry")

    points = regular_star_points(points=5, edge_length=30.0)
    edges = [
        shapely.LineString([points[i], points[(i + 1) % len(points)]])
        for i in range(len(points))
    ]

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            adjacent = (j == i + 1) or (i == 0 and j == len(edges) - 1)
            if adjacent:
                continue
            assert edges[i].intersection(edges[j]).is_empty, f"edges {i} and {j} intersect"


def test_star_bounding_box_matches_tip_geometry():
    """Half-width is R*sin(72deg); the top tip reaches R before centring."""
    points = regular_star_points(points=5, edge_length=30.0)
    radius = star_circumradius(5, 30.0)
    min_x, min_y, max_x, max_y = bounding_box(points)

    assert max_x == pytest.approx(radius * math.sin(2 * math.pi / 5), rel=1e-9)
    assert max_y == pytest.approx(radius, rel=1e-9)


def test_centering_shifts_the_shape_but_keeps_edges():
    points = regular_star_points(points=5, edge_length=30.0)
    centered = center_on_origin(points)

    lengths = polygon_edge_lengths(centered, closed=True)
    assert all(length == pytest.approx(30.0, rel=1e-12) for length in lengths)

    min_x, min_y, max_x, max_y = bounding_box(centered)
    assert (min_x + max_x) == pytest.approx(0.0, abs=1e-9)
    assert (min_y + max_y) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("points,edge", [(3, 20.0), (6, 15.0), (10, 5.0)])
def test_other_star_sizes_are_equilateral(points: int, edge: float):
    vertices = regular_star_points(points=points, edge_length=edge)
    lengths = polygon_edge_lengths(vertices, closed=True)
    assert len(lengths) == 2 * points
    for length in lengths:
        assert length == pytest.approx(edge, rel=1e-12)


@pytest.mark.parametrize("sides,edge", [(3, 10.0), (5, 30.0), (8, 7.5)])
def test_regular_polygons_are_equilateral(sides: int, edge: float):
    vertices = regular_polygon_points(sides=sides, edge_length=edge)
    assert len(vertices) == sides
    for length in polygon_edge_lengths(vertices, closed=True):
        assert length == pytest.approx(edge, rel=1e-12)


def test_degenerate_arguments_are_rejected():
    with pytest.raises(ValueError):
        regular_star_points(points=2, edge_length=10)
    with pytest.raises(ValueError):
        regular_star_points(points=5, edge_length=0)
    with pytest.raises(ValueError):
        regular_polygon_points(sides=2, edge_length=10)


def test_rotation_keeps_edges_intact():
    points = regular_star_points(points=5, edge_length=30.0, rotation_deg=17.0)
    for length in polygon_edge_lengths(points, closed=True):
        assert length == pytest.approx(30.0, rel=1e-12)
