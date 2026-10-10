"""Polygon winding and path-encoding tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.geometry import (  # noqa: E402
    complex_polygon_to_path,
    densify_path,
    ensure_ccw,
    ensure_cw,
    is_ccw,
    normalize_complex_polygon,
    path_to_rings,
    rect_ring,
    signed_area,
)
from pcbart.primitives import (  # noqa: E402
    Mode,
    convert_path_to_mm,
    convert_path_to_unit,
    polyline_path,
    Point,
)
from pcbart.units import unit_to_mm  # noqa: E402


def test_signed_area_of_ccw_square_is_positive():
    assert signed_area(rect_ring(0, 0, 10, 10)) > 0


def test_ensure_cw_flips_a_ccw_ring():
    ring = rect_ring(0, 0, 10, 10)
    assert not is_ccw(ensure_cw(ring))


def test_ensure_ccw_flips_a_cw_ring():
    ring = list(reversed(rect_ring(0, 0, 10, 10)))
    assert is_ccw(ensure_ccw(ring))


def test_normalize_sets_outer_cw_and_holes_ccw():
    """EasyEDA Pro's rule, stated in the design frame (Y up).

    The format spec says "outer clockwise, holes counter-clockwise" without
    naming a frame. A real project settles it: decoding an IMAGE record and
    negating Y gives an outer ring with negative (clockwise) area and a hole
    with positive (counter-clockwise) area.
    """
    outer = rect_ring(0, 0, 10, 10)
    hole = list(reversed(rect_ring(2, 2, 4, 4)))
    rings = normalize_complex_polygon(outer, [hole])

    assert not is_ccw(rings[0]), "outer ring must be clockwise"
    assert is_ccw(rings[1]), "hole ring must be counter-clockwise"
    assert len(rings) == 2


def test_normalize_flips_wrong_input_winding():
    """Input winding is irrelevant; the output is always normalised."""
    outer = list(reversed(rect_ring(0, 0, 10, 10)))  # supplied counter-clockwise
    hole = rect_ring(2, 2, 4, 4)  # supplied clockwise
    rings = normalize_complex_polygon(outer, [hole])

    assert not is_ccw(rings[0])
    assert is_ccw(rings[1])


def test_normalize_places_islands_with_the_outer_ring():
    """A solid island inside a hole winds like the outer contour."""
    outer = rect_ring(0, 0, 20, 20)
    hole = list(reversed(rect_ring(4, 4, 12, 12)))
    island = rect_ring(8, 8, 4, 4)
    rings = normalize_complex_polygon(outer, [hole], islands=[island])

    assert not is_ccw(rings[0]), "outer clockwise"
    assert not is_ccw(rings[1]), "island also clockwise"
    assert is_ccw(rings[2]), "hole counter-clockwise"


def test_normalize_drops_degenerate_holes():
    rings = normalize_complex_polygon(rect_ring(0, 0, 10, 10), [[(0, 0), (1, 1)]])
    assert len(rings) == 1


def test_complex_polygon_to_path_shape():
    rings = normalize_complex_polygon(rect_ring(0, 0, 4, 4))
    paths = complex_polygon_to_path(rings)
    assert len(paths) == 1
    assert paths[0][2] == Mode.L


def test_path_round_trip_through_rings():
    ring = rect_ring(0, 0, 10, 20)
    from pcbart.geometry import ring_to_path

    decoded = path_to_rings(ring_to_path(ring))
    # The encoder closes the ring, so the decode comes back one vertex longer
    # with the first point repeated at the end.
    assert decoded[0][:-1] == pytest.approx(list(ring))
    assert decoded[0][-1] == pytest.approx(decoded[0][0])


def test_ring_to_path_closes_the_ring():
    """Every ``L`` ring in the captured project repeats its first vertex.

    An open ring is not a safe assumption: the first M3 experiment left them
    open and the editor showed a board with no copper at all.
    """
    from pcbart.geometry import ring_to_path

    path = ring_to_path([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    numbers = [t for t in path if isinstance(t, (int, float))]

    assert len(numbers) == 8, "四点流：首点起、末点回到首点"
    assert (numbers[0], numbers[1]) == (numbers[-2], numbers[-1])


def test_ring_to_path_does_not_double_close():
    """A ring that already repeats its first vertex stays as it is."""
    from pcbart.geometry import ring_to_path

    closed = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]
    numbers = [t for t in ring_to_path(closed) if isinstance(t, (int, float))]

    assert len(numbers) == 8


def test_path_to_rings_handles_circle():
    rings = path_to_rings([Mode.CIRCLE, 10.0, 5.0, 2.0])
    assert len(rings) == 1
    assert len(rings[0]) > 8


def test_path_to_rings_handles_rectangle():
    rings = path_to_rings([Mode.R, 1.0, 2.0, 4.0, 6.0, 0, 0, 0])
    assert rings[0] == [(1.0, 2.0), (5.0, 2.0), (5.0, -4.0), (1.0, -4.0)]


# ------------------------------------------------------------------ unit paths


def test_polyline_path_encodes_L_mode():
    path = polyline_path([Point(0, 0), Point(10, 0), Point(10, 5)])
    assert path == [0, 0, "L", 10, 0, 10, 5]


def test_path_conversion_scales_but_keeps_y_sign():
    """The file's Y axis already points up, so only the scale changes.

    Negating here was a bug: it mirrored every generated board top-to-bottom.
    The evidence for Y-up is recorded on ``pcbart.units.FILE_Y_IS_UP``.
    """
    raw = [0, 0, "L", 3937.0079, -1968.5039]
    mm = convert_path_to_mm(raw)
    assert mm[0] == pytest.approx(0.0)
    assert mm[1] == pytest.approx(0.0)
    assert mm[3] == pytest.approx(100.0, abs=1e-3)
    assert mm[4] == pytest.approx(-50.0, abs=1e-3)


def test_path_conversion_round_trips():
    raw = [28.956, -74.1664, "L", 28.956, 23.6424]
    assert convert_path_to_unit(convert_path_to_mm(raw)) == pytest.approx(raw)


def test_arc_conversion_keeps_angle_unscaled():
    """ARC carries an angle between the start point and the end point.

    The angle must not be scaled or sign-flipped: it is a rotation, not a
    length.
    """
    raw = [1140.0044, -930.4824, "ARC", 93.3779, 1181.1899, -971.9999]
    mm = convert_path_to_mm(raw)
    assert mm[2] == "ARC"
    assert mm[3] == pytest.approx(93.3779)
    assert mm[4] == pytest.approx(unit_to_mm(1181.1899), abs=1e-6)
    assert mm[5] == pytest.approx(unit_to_mm(-971.9999), abs=1e-6)


def test_arc_round_trips():
    raw = [1140.0044, -930.4824, "ARC", 93.3779, 1181.1899, -971.9999]
    assert convert_path_to_unit(convert_path_to_mm(raw)) == pytest.approx(raw)


def test_circle_conversion_scales_center_and_radius():
    raw = [Mode.CIRCLE, 2955, 690, 10]
    mm = convert_path_to_mm(raw)
    assert mm[1] == pytest.approx(unit_to_mm(2955), abs=1e-6)
    assert mm[2] == pytest.approx(unit_to_mm(690), abs=1e-6)
    assert mm[3] == pytest.approx(unit_to_mm(10), abs=1e-6)


def test_nested_complex_polygon_conversion():
    """A list of polygons must be converted element-wise, not flattened."""
    raw = [[0, 0, "L", 100, 0, 100, 100], [10, 10, "L", 20, 10, 20, 20]]
    mm = convert_path_to_mm(raw)
    assert isinstance(mm[0], list) and isinstance(mm[1], list)
    assert mm[0][3] == pytest.approx(unit_to_mm(100), abs=1e-9)


def test_a_rectangle_grows_downward_from_its_top_edge():
    """``R x y w h`` anchors at the top edge, not the bottom.

    Decoding it as ``y + h`` lifted every panel by its own height; the
    通行证 artwork field landed above the card and the whole template looked
    mis-positioned in the editor.
    """
    ring = densify_path(["R", 0.0, 10.0, 4.0, 2.0])[0]
    ys = [y for _, y in ring]
    assert min(ys) == pytest.approx(8.0)
    assert max(ys) == pytest.approx(10.0)


def test_a_rotated_rectangle_follows_its_angle():
    ring = densify_path(["R", 0.0, 10.0, 4.0, 2.0, 90.0])[0]
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(2.0)
    assert min(ys) == pytest.approx(10.0)
    assert max(ys) == pytest.approx(14.0)


def test_a_cubic_curve_is_sampled_between_its_endpoints():
    """The two control points are not vertices of a polyline.

    Treating them as vertices turned the card's vector logos and side text
    into fans of chords, which then filled as solid ellipses.
    """
    path = [0.0, 0.0, "C", 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    points = densify_path(path, segments=8)[0]
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (10.0, 0.0)
    assert len(points) == 9
    # A cubic from (0,0) to (10,0) with both controls above the chord has a
    # midpoint above zero; a polyline through the controls would not.
    assert max(y for _, y in points) > 7.0
