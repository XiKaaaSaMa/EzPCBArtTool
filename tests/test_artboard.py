"""Tests for the design document: board outline plus image placement.

Geometry only -- this module has no Qt and no file format, so everything here
is deterministic and fast.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    DEFAULT_IMAGE_LONG_EDGE_MM,
    OUTLINE_SHAPES,
    SHAPE_CHOICES,
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    IrregularBase,
    OutlineShape,
    build_template,
    placement_for_pixels,
    register_template,
    template_names,
)


# ------------------------------------------------------------------ outline


def test_rect_outline_is_four_corners_centred_on_origin():
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=40)
    points = spec.polygon()

    assert len(points) == 4
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) == pytest.approx(-30)
    assert max(xs) == pytest.approx(30)
    assert min(ys) == pytest.approx(-20)
    assert max(ys) == pytest.approx(20)


def test_rect_outline_bounds_match_the_requested_size():
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=50, height_mm=30)
    board = ArtBoard(outline=spec)
    width, height = board.size_mm()

    assert width == pytest.approx(50, abs=1e-6)
    assert height == pytest.approx(30, abs=1e-6)


def test_circle_outline_is_round_not_a_box():
    spec = BoardOutlineSpec(shape=OutlineShape.CIRCLE, width_mm=40, height_mm=40)
    points = spec.polygon()

    assert len(points) > 32, "a circle needs many vertices"
    radii = [math.hypot(x, y) for x, y in points]
    assert min(radii) == pytest.approx(20, abs=1e-6)
    assert max(radii) == pytest.approx(20, abs=1e-6)


def test_circle_stores_a_single_dimension():
    """A circle has one degree of freedom, so width and height always agree.

    Two independent sides would let the panel show a single diameter while the
    geometry came out as something else.
    """
    spec = BoardOutlineSpec(shape=OutlineShape.CIRCLE, width_mm=40, height_mm=40)
    assert spec.diameter_mm == pytest.approx(40)

    spec.diameter_mm = 25
    assert spec.width_mm == pytest.approx(25)
    assert spec.height_mm == pytest.approx(25)
    assert spec.validate() == []


def test_circle_with_mismatched_sides_is_rejected():
    spec = BoardOutlineSpec(shape=OutlineShape.CIRCLE, width_mm=40, height_mm=20)
    assert any("直径" in p for p in spec.validate())


def test_squared_keeps_the_shorter_side():
    """Switching to a one-dimension shape must not grow past what was set."""
    spec = BoardOutlineSpec(width_mm=60, height_mm=30)
    squared = spec.squared()

    assert squared.width_mm == pytest.approx(30)
    assert squared.height_mm == pytest.approx(30)
    assert (spec.width_mm, spec.height_mm) == pytest.approx((60, 30)), "原对象不变"


def test_rounded_rect_has_rounded_corners():
    spec = BoardOutlineSpec(
        shape=OutlineShape.ROUNDED, width_mm=60, height_mm=40, corner_radius_mm=5
    )
    points = spec.polygon()

    assert len(points) > 20, "corners are expanded into arcs"
    # The extreme corner of a rounded rectangle must be pulled in.
    assert not any(abs(x) == pytest.approx(30) and abs(y) == pytest.approx(20)
                   for x, y in points)
    # But the edge midpoints still reach the full size.
    assert max(p[0] for p in points) == pytest.approx(30, abs=1e-6)
    assert max(p[1] for p in points) == pytest.approx(20, abs=1e-6)


def test_zero_radius_rounded_rect_equals_a_rectangle():
    rounded = BoardOutlineSpec(
        shape=OutlineShape.ROUNDED, width_mm=50, height_mm=30, corner_radius_mm=0
    )
    assert len(rounded.polygon()) == 4


@pytest.mark.parametrize("size", [(60, 40, 5), (76.2, 76.2, 3), (40, 20, 10)])
def test_rounded_rect_is_a_valid_polygon_of_the_right_area(size):
    """Bounds alone do not prove the outline is a rounded rectangle.

    A ring whose corner arcs are emitted in the wrong order still spans the
    right width and height, so every size assertion passes, while the polygon
    crosses itself and its area comes out far too small -- a 60 × 40 board with
    a 5 mm radius lost 40% of its area that way. Validity and area are what
    catch it.
    """
    shapely = pytest.importorskip("shapely.geometry")
    width, height, radius = size
    spec = BoardOutlineSpec(
        shape=OutlineShape.ROUNDED,
        width_mm=width,
        height_mm=height,
        corner_radius_mm=radius,
    )

    polygon = shapely.Polygon(spec.polygon())
    expected = width * height - (4 - math.pi) * radius * radius

    assert polygon.is_valid, "自交的圆角矩形"
    assert polygon.is_simple
    assert polygon.area == pytest.approx(expected, rel=0.01)


# --------------------------------------------------------- enclosing artwork


def test_a_rectangle_board_takes_the_artwork_bounds():
    spec = BoardOutlineSpec(shape=OutlineShape.RECT)
    width, height = spec.enclosing_size(76.2, 40)

    assert (width, height) == pytest.approx((76.2, 40))


def test_a_circle_board_grows_to_the_artwork_diagonal():
    """The smallest circle around a rectangle passes through its corners.

    Sizing it to the shorter side instead would leave the artwork's corners
    hanging off the board.
    """
    spec = BoardOutlineSpec(shape=OutlineShape.CIRCLE)
    width, height = spec.enclosing_size(60, 30)

    assert width == pytest.approx(math.hypot(60, 30))
    assert height == pytest.approx(math.hypot(60, 30))
    assert spec.fitted_around(60, 30).validate() == []


def test_fitted_around_keeps_the_shape_and_other_fields():
    spec = BoardOutlineSpec(
        shape=OutlineShape.ROUNDED, corner_radius_mm=7, keep_aspect=False
    )
    fitted = spec.fitted_around(50, 20)

    assert fitted.shape == OutlineShape.ROUNDED
    assert fitted.corner_radius_mm == pytest.approx(7)
    assert fitted.keep_aspect is False
    assert (fitted.width_mm, fitted.height_mm) == pytest.approx((50, 20))


def test_fitted_around_survives_a_degenerate_artwork_size():
    spec = BoardOutlineSpec(shape=OutlineShape.RECT)
    width, height = spec.enclosing_size(0, -5)
    assert width > 0 and height > 0


# --------------------------------------------------------- changing shape


def test_between_two_dimensional_shapes_the_size_is_kept():
    """Rounding the corners of a 60 × 30 board must not resize it."""
    spec = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=60, height_mm=30, corner_radius_mm=4
    )
    changed = spec.after_shape_change(OutlineShape.ROUNDED)

    assert changed.shape == OutlineShape.ROUNDED
    assert (changed.width_mm, changed.height_mm) == pytest.approx((60, 30))
    assert changed.corner_radius_mm == pytest.approx(4)


def test_to_a_circle_the_board_is_derived_from_the_artwork():
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=30)
    changed = spec.after_shape_change(OutlineShape.CIRCLE, artwork_size=(60, 30))

    assert changed.diameter_mm == pytest.approx(math.hypot(60, 30))


def test_from_a_circle_the_board_returns_to_the_artwork_bounds():
    spec = BoardOutlineSpec(shape=OutlineShape.CIRCLE, width_mm=100, height_mm=100)
    changed = spec.after_shape_change(OutlineShape.RECT, artwork_size=(60, 30))

    assert (changed.width_mm, changed.height_mm) == pytest.approx((60, 30))


def test_without_artwork_the_previous_board_is_the_reference():
    """Nothing to enclose, so the board keeps the area it already had."""
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=30)
    changed = spec.after_shape_change(OutlineShape.CIRCLE)

    assert changed.diameter_mm == pytest.approx(math.hypot(60, 30))


def test_the_shape_change_is_not_cumulative():
    """Deriving from the artwork, not the old board, keeps it stable."""
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=30)
    first = spec.after_shape_change(OutlineShape.CIRCLE, artwork_size=(60, 30))

    for _ in range(4):
        spec = first.after_shape_change(OutlineShape.RECT, artwork_size=(60, 30))
        first = spec.after_shape_change(OutlineShape.CIRCLE, artwork_size=(60, 30))

    assert first.diameter_mm == pytest.approx(math.hypot(60, 30))


# ------------------------------------------------------------------ validation


def test_validate_accepts_a_sane_spec():
    assert BoardOutlineSpec().validate() == []


def test_validate_rejects_a_board_smaller_than_the_process_allows():
    spec = BoardOutlineSpec(width_mm=2.0, height_mm=2.0)
    problems = spec.validate()
    assert any("3.0 mm" in p for p in problems)


def test_validate_rejects_a_negative_radius():
    spec = BoardOutlineSpec(shape=OutlineShape.ROUNDED, corner_radius_mm=-1)
    assert any("圆角" in p for p in spec.validate())


def test_validate_rejects_a_radius_larger_than_the_board():
    spec = BoardOutlineSpec(
        shape=OutlineShape.ROUNDED, width_mm=20, height_mm=20, corner_radius_mm=50
    )
    assert any("不得超过" in p for p in spec.validate())


def test_validate_rejects_an_unknown_shape():
    assert any("未知" in p for p in BoardOutlineSpec(shape="hexagon").validate())


def test_all_declared_preset_shapes_are_accepted_by_validate():
    """Every preset is a standalone shape, artwork not required."""
    assert OutlineShape.IRREGULAR not in OUTLINE_SHAPES, "异形不是预设外形"
    for shape in OUTLINE_SHAPES:
        assert BoardOutlineSpec(shape=shape).validate() == [], shape


# ------------------------------------------------------------- aspect locking


def test_keep_aspect_scales_height_with_width():
    spec = BoardOutlineSpec(width_mm=60, height_mm=30, keep_aspect=True)
    resized = spec.with_size(120, 999)

    assert resized.width_mm == pytest.approx(120)
    assert resized.height_mm == pytest.approx(60), "height must follow the ratio"


def test_keep_aspect_off_honours_both_values():
    spec = BoardOutlineSpec(width_mm=60, height_mm=30, keep_aspect=False)
    resized = spec.with_size(120, 10)

    assert resized.width_mm == pytest.approx(120)
    assert resized.height_mm == pytest.approx(10)


def test_size_is_never_zero_or_negative():
    spec = BoardOutlineSpec(keep_aspect=False)
    assert spec.with_size(0, -5).width_mm > 0
    assert spec.with_size(0, -5).height_mm > 0


# ------------------------------------------------------------- image placement


def test_placement_bounds_are_centred():
    image = ImagePlacement(x_mm=10, y_mm=20, width_mm=40, height_mm=30)
    left, bottom, right, top = image.bounds()

    assert (left, bottom, right, top) == pytest.approx((-10, 5, 30, 35))


def test_placement_corners_are_clockwise_from_top_left():
    image = ImagePlacement(x_mm=0, y_mm=0, width_mm=40, height_mm=20)
    corners = image.corners()

    assert len(corners) == 4
    assert corners[0] == pytest.approx((-20, 10)), "top-left, Y-up"
    assert corners[1] == pytest.approx((20, 10))
    assert corners[2] == pytest.approx((20, -10))
    assert corners[3] == pytest.approx((-20, -10))


def test_rotation_moves_the_corners():
    image = ImagePlacement(width_mm=40, height_mm=20, rotation_deg=90)
    corners = image.corners()
    # A quarter turn swaps the extents.
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(20, abs=1e-6)
    assert max(ys) - min(ys) == pytest.approx(40, abs=1e-6)


def test_resize_without_anchor_keeps_the_centre():
    image = ImagePlacement(x_mm=5, y_mm=-3, width_mm=40, height_mm=20, keep_aspect=False)
    resized = image.resize_to(80, 40)

    assert (resized.x_mm, resized.y_mm) == pytest.approx((5, -3))
    assert (resized.width_mm, resized.height_mm) == pytest.approx((80, 40))


def test_resize_with_aspect_locked_keeps_the_ratio():
    image = ImagePlacement(width_mm=40, height_mm=20, keep_aspect=True)
    resized = image.resize_to(80, 999)

    assert resized.width_mm == pytest.approx(80)
    assert resized.height_mm == pytest.approx(40)


def test_resize_anchored_to_a_corner_keeps_that_corner_fixed():
    """Dragging a handle must pin the opposite corner, like every editor."""
    image = ImagePlacement(x_mm=0, y_mm=0, width_mm=40, height_mm=40, keep_aspect=False)
    bottom_right = (20.0, -20.0)

    resized = image.resize_to(80, 80, anchor=bottom_right)

    # The bottom-right corner must not have moved.
    assert resized.corners()[2] == pytest.approx(bottom_right)


def test_resize_anchored_to_the_opposite_corner():
    image = ImagePlacement(x_mm=0, y_mm=0, width_mm=40, height_mm=40, keep_aspect=False)
    top_left = (-20.0, 20.0)

    resized = image.resize_to(80, 80, anchor=top_left)

    assert resized.corners()[0] == pytest.approx(top_left)


def test_move_to_sets_the_centre():
    image = ImagePlacement()
    moved = image.move_to(12.5, -7.5)
    assert (moved.x_mm, moved.y_mm) == pytest.approx((12.5, -7.5))


def test_validate_rejects_bad_opacity():
    assert any("不透明度" in p for p in ImagePlacement(opacity=1.5).validate())


# -------------------------------------------------------------------- board


def test_board_without_an_image_still_reports_a_size():
    board = ArtBoard(outline=BoardOutlineSpec(width_mm=50, height_mm=30))
    assert board.size_mm() == pytest.approx((50, 30))


def test_board_tracks_whether_the_outline_was_configured():
    """A default board must not masquerade as a user choice."""
    assert ArtBoard().outline_ready is False
    configured = ArtBoard(
        outline=BoardOutlineSpec(width_mm=40, height_mm=40), outline_ready=True
    )
    assert configured.outline_ready is True


def test_fit_image_shrinks_an_oversized_placement():
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=40, height_mm=40),
        image=ImagePlacement(width_mm=100, height_mm=50, keep_aspect=True),
    )
    fitted = board.fit_image_to_board()

    assert fitted.image is not None
    assert fitted.image.width_mm <= 40 + 1e-6
    assert fitted.image.height_mm <= 40 + 1e-6
    # Aspect ratio preserved.
    assert fitted.image.width_mm / fitted.image.height_mm == pytest.approx(2.0)


def test_fit_image_leaves_a_small_placement_alone():
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=100, height_mm=100),
        image=ImagePlacement(width_mm=20, height_mm=20),
    )
    assert board.fit_image_to_board() is board


def test_fit_image_respects_the_margin():
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=40, height_mm=40),
        image=ImagePlacement(width_mm=100, height_mm=100),
    )
    fitted = board.fit_image_to_board(margin_mm=5)
    assert fitted.image is not None
    assert fitted.image.width_mm == pytest.approx(30)


def test_fit_image_without_a_placement_is_a_no_op():
    board = ArtBoard()
    assert board.fit_image_to_board() is board


def test_board_validate_combines_outline_and_image_problems():
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=2, height_mm=2),
        image=ImagePlacement(opacity=3.0),
    )
    problems = board.validate()
    assert any("3.0 mm" in p for p in problems)
    assert any("不透明度" in p for p in problems)


def test_a_trace_only_irregular_board_needs_artwork():
    """With nothing to trace there is no board, and no rectangle invented.

    A standee base could still be drawn from the preset alone, but 原图裁切
    has no preset part at all -- it *is* the artwork's outline.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=IrregularBase.TRACE,
            width_mm=50,
            height_mm=30,
        ),
        outline_ready=True,
    )
    assert board.outline_polygon() == []
    assert board.irregular_source() is None


def test_a_combining_base_is_drawn_before_any_artwork_arrives():
    """The base does not depend on the artwork, so it can be shown alone."""
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=IrregularBase.RECT,
            width_mm=50,
            height_mm=30,
            base_top_ratio=0.5,
        ),
        outline_ready=True,
    )
    points = board.outline_polygon()

    assert len(points) == 4
    ys = [p[1] for p in points]
    assert min(ys) == pytest.approx(-15), "底座下沿应在板框底部"
    assert max(ys) == pytest.approx(0), "上边界在 50% 处"


def test_irregular_source_reports_the_image_path():
    board = ArtBoard(
        image=ImagePlacement(path="art.png"),
        outline=BoardOutlineSpec(shape=OutlineShape.IRREGULAR),
        outline_ready=True,
    )
    assert board.irregular_source() == "art.png"


def test_an_unconfigured_board_has_no_outline():
    """An imported picture is not a board; no shape may be invented for it."""
    board = ArtBoard(image=ImagePlacement(path="art.png", width_mm=76.2))

    assert board.outline_polygon() == []
    assert board.size_mm() == pytest.approx((60.0, 60.0)), "回落到规格默认值"


def test_the_outline_appears_once_it_is_configured():
    board = ArtBoard(
        image=ImagePlacement(path="art.png"),
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
    )
    assert board.outline_polygon() == []
    board.outline_ready = True
    assert len(board.outline_polygon()) == 4


# ------------------------------------------------------- imported image size


def test_placement_for_pixels_scales_the_long_edge():
    """EasyEDA Pro imports a picture at three inches on its longest side."""
    placement = placement_for_pixels("art.png", 1000, 500)

    assert placement.width_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM)
    assert placement.height_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM / 2)
    assert (placement.x_mm, placement.y_mm) == pytest.approx((0, 0))
    assert placement.path == "art.png"


def test_placement_for_pixels_handles_a_tall_image():
    placement = placement_for_pixels("art.png", 400, 1200)

    assert placement.height_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM)
    assert placement.width_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM / 3)


def test_placement_for_pixels_honours_a_custom_long_edge():
    placement = placement_for_pixels("art.png", 200, 100, long_edge_mm=50)
    assert placement.width_mm == pytest.approx(50)
    assert placement.height_mm == pytest.approx(25)


def test_placement_for_pixels_survives_a_degenerate_size():
    """A zero pixel size must not divide by zero."""
    placement = placement_for_pixels("art.png", 0, 0)
    assert placement.width_mm > 0
    assert placement.height_mm > 0


# ------------------------------------------------------------------ templates


def test_no_templates_are_registered_by_default():
    """The feature is pending user-supplied files; the interface is ready."""
    assert template_names() == []


def test_registering_a_template_makes_it_available():
    spec = BoardOutlineSpec(width_mm=42, height_mm=24, shape=OutlineShape.ROUNDED)
    register_template("test-card", spec)
    try:
        assert "test-card" in template_names()
        loaded = build_template("test-card")
        assert loaded.width_mm == pytest.approx(42)
        assert loaded.shape == OutlineShape.ROUNDED
        assert loaded.template == "test-card"
    finally:
        from pcbart.artboard import _TEMPLATES

        _TEMPLATES.pop("test-card", None)


def test_template_build_returns_a_copy():
    """Editing a loaded template must not corrupt the stored one."""
    spec = BoardOutlineSpec(width_mm=42, height_mm=24)
    register_template("copy-test", spec)
    try:
        first = build_template("copy-test")
        first.width_mm = 999
        second = build_template("copy-test")
        assert second.width_mm == pytest.approx(42)
    finally:
        from pcbart.artboard import _TEMPLATES

        _TEMPLATES.pop("copy-test", None)


def test_unknown_template_is_rejected():
    with pytest.raises(KeyError):
        build_template("does-not-exist")


def test_empty_template_name_is_rejected():
    with pytest.raises(ValueError):
        register_template("", BoardOutlineSpec())
