"""Tests for the standee tab: the piece that plugs into a display base.

The tab is small, which is exactly why it needs real geometry assertions rather
than a size check. It is unioned onto the board, and the interesting failures
all leave every dimension looking right:

* a tab whose top edge merely *touches* the board unions into two separate
  pieces, so the exported board comes out as a tab with no board on it, or as a
  board with no tab;
* a weld that reaches too far up would be invisible in the outline but would
  change the piece's area, so the area is asserted rather than only the bounds;
* a tab slid past the end of the bottom edge would be silently dropped.

Point 2 is why the rectangle case asserts *area*: the tab adds exactly
``width × height`` of material and nothing more, no matter how deeply it is
welded into the board.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    DEFAULT_TAB_HEIGHT_MM,
    DEFAULT_TAB_WIDTH_MM,
    ArtBoard,
    BoardOutlineSpec,
    OutlineShape,
)
from pcbart.standee import attach_tab, contact_span, travel_mm  # noqa: E402

shapely = pytest.importorskip("shapely.geometry")
from shapely.geometry import Polygon  # noqa: E402


def board_with_tab(**kwargs):
    """A 50 × 30 rectangle with a standee tab, ready to measure."""
    spec = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=50.0, height_mm=30.0, standee=True
    )
    for name, value in kwargs.items():
        setattr(spec, name, value)
    return ArtBoard(outline=spec, outline_ready=True)


def shape_of(board):
    return Polygon(board.outline_polygon())


# ------------------------------------------------------------- the defaults


def test_the_tab_defaults_to_the_requested_size():
    """16 × 3 mm, and off until the user asks for it."""
    spec = BoardOutlineSpec()

    assert (spec.standee_width_mm, spec.standee_height_mm) == pytest.approx(
        (DEFAULT_TAB_WIDTH_MM, DEFAULT_TAB_HEIGHT_MM)
    )
    assert (DEFAULT_TAB_WIDTH_MM, DEFAULT_TAB_HEIGHT_MM) == pytest.approx((16.0, 3.0))
    assert spec.standee is False, "立牌是选项，不能默认打开"


def test_switching_the_tab_on_leaves_the_board_otherwise_unchanged():
    plain = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )
    tabbed = board_with_tab()

    assert plain.standee_error is None
    assert shape_of(plain).area == pytest.approx(1500.0)
    assert shape_of(tabbed).area == pytest.approx(1500.0 + 16.0 * 3.0)


def test_a_plain_board_has_no_tab_geometry():
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )

    assert board.size_mm() == pytest.approx((50.0, 30.0))
    assert board.standee_travel_mm() == 0.0


# ------------------------------------------------------------- the geometry


def test_the_tab_hangs_the_requested_distance_below_the_board():
    board = board_with_tab()
    ring = board.outline_polygon()

    ys = [y for _x, y in ring]
    assert min(ys) == pytest.approx(-15.0 - DEFAULT_TAB_HEIGHT_MM)
    assert max(ys) == pytest.approx(15.0), "插舌不得改变板框上边缘"


def test_the_tab_is_the_requested_width_and_sits_in_the_middle():
    board = board_with_tab()
    ring = board.outline_polygon()

    below = [(x, y) for x, y in ring if y < -15.0 - 1e-9]
    widths = [x for x, _y in below]
    assert (max(widths) - min(widths)) == pytest.approx(DEFAULT_TAB_WIDTH_MM)
    assert sum(widths) / len(widths) == pytest.approx(0.0, abs=1e-6)
    # Both of the tab's upper corners lie on the board's bottom edge, which is
    # what "插舌的上边与原板框下边缘重合" asks for.
    assert min(widths) == pytest.approx(-DEFAULT_TAB_WIDTH_MM / 2)
    assert max(widths) == pytest.approx(DEFAULT_TAB_WIDTH_MM / 2)


def test_the_weld_adds_no_material_inside_the_board():
    """Area is the check that catches a weld reaching into the board.

    A weld deep enough to matter would still land on the right bounds and the
    right protruding height -- the extra material is *inside* the board, where
    the outline cannot show it. For a rectangle the tab adds exactly its own
    area, so any surplus is a weld gone wrong.
    """
    plain = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )
    tabbed = board_with_tab()

    expected = shape_of(plain).area + DEFAULT_TAB_WIDTH_MM * DEFAULT_TAB_HEIGHT_MM
    assert shape_of(tabbed).area == pytest.approx(expected)


def test_the_tab_and_the_board_are_one_piece():
    """A tab that only touches the board is not attached to it.

    Two polygons that meet at a single point stay two polygons after a union,
    which would export as a board and a tab that are not joined. The weld exists
    to make the joint overlap by area, and the result must therefore be a single
    polygon.
    """
    shape = shape_of(board_with_tab())

    assert shape.geom_type == "Polygon"
    assert shape.is_valid


@pytest.mark.parametrize(
    "shape, size",
    [
        (OutlineShape.RECT, (50.0, 30.0)),
        (OutlineShape.ROUNDED, (50.0, 30.0)),
        (OutlineShape.CIRCLE, (40.0, 40.0)),
    ],
)
def test_every_shape_can_carry_a_tab(shape, size):
    """Including a circle, whose lowest point is a single point.

    That case is the reason the weld exists: the tab's top edge lies on the
    circle's bottom, so without reaching up into the board the union would
    produce two pieces and the tab would not be a standee tab at all.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=shape, width_mm=size[0], height_mm=size[1], standee=True
        ),
        outline_ready=True,
    )

    ring = board.outline_polygon()
    polygon = Polygon(ring)

    assert polygon.is_valid
    assert polygon.geom_type == "Polygon", f"{shape} 的插舌没有真正连成一体"
    ys = [y for _x, y in ring]
    assert min(ys) == pytest.approx(-size[1] / 2 - DEFAULT_TAB_HEIGHT_MM)


def test_a_rounded_board_keeps_its_tab_on_the_flat_bottom():
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.ROUNDED,
            width_mm=50,
            height_mm=30,
            corner_radius_mm=5,
            standee=True,
        ),
        outline_ready=True,
    )
    ring = board.outline_polygon()
    below = [x for x, y in ring if y < -15.0 - 1e-9]

    # The flat bottom runs from -20 to +20; a corner radius of 5 does not reach
    # the middle of the edge, so the tab is a clean rectangle down there.
    assert (max(below) - min(below)) == pytest.approx(DEFAULT_TAB_WIDTH_MM)


# --------------------------------------------------------------- the slider


def test_the_slider_moves_the_tab_along_the_bottom_edge():
    centred = board_with_tab(standee_offset_mm=0.0)
    moved = board_with_tab(standee_offset_mm=10.0)

    centred_below = [x for x, y in centred.outline_polygon() if y < -15.0 - 1e-9]
    moved_below = [x for x, y in moved.outline_polygon() if y < -15.0 - 1e-9]

    assert sum(moved_below) / len(moved_below) == pytest.approx(10.0)
    assert sum(centred_below) / len(centred_below) == pytest.approx(0.0)
    # The tab keeps its size and stays a rectangle wherever it goes.
    assert (max(moved_below) - min(moved_below)) == pytest.approx(
        DEFAULT_TAB_WIDTH_MM
    )


def test_the_travel_is_the_real_room_left_on_the_bottom_edge():
    """Half the edge less half the tab, measured on the board itself.

    A fixed percentage of the board's width would let the tab slide off the end
    of a short bottom edge -- and off the end of a circle's, which is far
    narrower than its diameter.
    """
    board = board_with_tab()

    assert board.standee_travel_mm() == pytest.approx(
        (50.0 - DEFAULT_TAB_WIDTH_MM) / 2
    )


def test_a_wider_tab_has_less_room_to_move():
    narrow = board_with_tab(standee_width_mm=6.0)
    wide = board_with_tab(standee_width_mm=40.0)

    assert wide.standee_travel_mm() < narrow.standee_travel_mm()
    assert wide.standee_travel_mm() == pytest.approx((50.0 - 40.0) / 2)


def test_the_travel_follows_the_board_not_its_bounding_box():
    """A circle's usable bottom is a small arc, not its 40mm full width.

    Deriving the range from the bounding box would offer a range the tab
    physically cannot use, and the slider would then detach the tab at its
    ends.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.CIRCLE,
            width_mm=40,
            height_mm=40,
            standee=True,
            standee_width_mm=4.0,
        ),
        outline_ready=True,
    )

    assert 0.0 < board.standee_travel_mm() < 5.0


# ------------------------------------------------------------- failure modes


def test_a_tab_that_cannot_reach_the_board_is_left_off_and_reported():
    """A tab slid clear of the board must not become the whole board.

    The union of a board and a detached tab is a MultiPolygon. Adopting the
    larger piece, which is what the irregular composer does for artwork, would
    hand back the bare board here; adopting the piece that carries the tab would
    hand back a *tab*, and the export would contain no board at all. Neither is
    acceptable, so the tab is refused and the reason recorded.
    """
    board = board_with_tab(standee_offset_mm=200.0)

    ring = board.outline_polygon()

    assert board.standee_error, "插舌没接上必须说明原因"
    assert Polygon(ring).area == pytest.approx(1500.0), "应回退到原板框"
    assert max(y for _x, y in ring) == pytest.approx(15.0)
    assert min(y for _x, y in ring) == pytest.approx(-15.0), "插舌不应留在几何里"


def test_a_board_that_has_not_been_configured_still_has_no_outline():
    """The tab is part of a board, so it cannot conjure one up on its own."""
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30, standee=True),
        outline_ready=False,
    )

    assert board.outline_polygon() == []
    assert board.standee_travel_mm() == 0.0


@pytest.mark.parametrize("width, height", [(0.0, 3.0), (-1.0, 3.0), (16.0, 0.0)])
def test_a_degenerate_tab_is_rejected_before_it_is_drawn(width, height):
    spec = BoardOutlineSpec(
        width_mm=50,
        height_mm=30,
        standee=True,
        standee_width_mm=width,
        standee_height_mm=height,
    )

    problems = spec.validate()

    assert any("立牌" in problem for problem in problems)


def test_a_switched_off_tab_is_not_validated():
    """An unused tab's numbers must not block a board that does not use them."""
    spec = BoardOutlineSpec(
        width_mm=50, height_mm=30, standee=False, standee_width_mm=0.0
    )

    assert spec.validate() == []


# ------------------------------------------------------- the shared edges


def test_the_tab_is_attached_to_the_welded_band_not_the_bounding_box():
    """The join is measured on the board, which a traced outline can offset.

    ``contact_span`` reports where material actually is along the strip the tab
    attaches to; a traced board is not centred on the origin, so the bounding
    box's middle can easily be empty space.
    """
    square = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    assert contact_span(square, 3.0) == pytest.approx((0.0, 20.0))
    assert travel_mm(square, 6.0, 3.0) == pytest.approx((20.0 - 6.0) / 2)


def test_the_tab_lands_on_an_offset_board():
    """A board whose bottom edge is not at ``-height/2`` still gets its tab."""
    offset_square = [(10.0, 5.0), (30.0, 5.0), (30.0, 25.0), (10.0, 25.0)]

    ring, attached = attach_tab(offset_square, 6.0, 3.0)

    assert attached
    assert min(y for _x, y in ring) == pytest.approx(2.0)
    below = [x for x, y in ring if y < 5.0 - 1e-9]
    assert sum(below) / len(below) == pytest.approx(20.0)


def test_a_bottom_narrower_than_the_tab_reports_no_travel():
    hairlines = [(0.0, 0.0), (1.0, 0.0), (1.0, 10.0), (0.0, 10.0)]

    assert travel_mm(hairlines, 16.0, 3.0) == 0.0


def test_the_board_size_and_the_piece_of_material_are_two_different_numbers():
    """The panel quotes the board; the canvas frames the whole piece.

    Both numbers are correct and they describe different things: a 50 × 30
    board with a 3 mm tab is 50 × 33 mm of material. The status line says so by
    naming the tab separately, and ``size_mm`` reports what the canvas and the
    exporter actually measure.
    """
    board = board_with_tab()

    assert (board.outline.width_mm, board.outline.height_mm) == pytest.approx(
        (50.0, 30.0)
    )
    assert board.size_mm() == pytest.approx((50.0, 30.0 + DEFAULT_TAB_HEIGHT_MM))
    assert math.isclose(board.outline.height_mm, 30.0)
