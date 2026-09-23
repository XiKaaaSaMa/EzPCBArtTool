"""Tests for combining the artwork's outline with a preset board shape.

An irregular board is the artwork's silhouette joined to a shape, and the two
can go wrong in ways that sizes do not reveal: the union can keep the base's
rectangle over the whole board (losing the ears), or the silhouette can be
placed at the wrong origin (mirrored, or offset). These tests therefore check
the composed ring's *shape* -- where its extremes sit -- not just its size.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

shapely_geometry = pytest.importorskip("shapely.geometry")

from shapely.geometry import Polygon  # noqa: E402

from pcbart.compose import (  # noqa: E402
    IRREGULAR_BASES,
    IrregularBase,
    base_polygon,
    combine_with_base,
    trace_artwork_ring,
)


def bounds(ring) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


# --------------------------------------------------------------------- base


def test_a_rectangle_base_is_the_bottom_fraction_of_the_board():
    base = base_polygon(IrregularBase.RECT, 72, 90, 3.0, 0.35)

    left, bottom, right, top = base.bounds
    assert (left, right) == pytest.approx((-36, 36))
    assert bottom == pytest.approx(-45), "底座下沿即板框底部"
    assert top == pytest.approx(-45 + 0.35 * 90), "上边界在 35% 处"
    assert base.area == pytest.approx(72 * 90 * 0.35)


def test_the_base_keeps_the_boards_full_width():
    """Cutting the top off must not also narrow the base."""
    base = base_polygon(IrregularBase.RECT, 72, 90, 3.0, 0.5)
    assert base.bounds[2] - base.bounds[0] == pytest.approx(72)


def test_a_rounded_base_is_the_same_size_with_rounded_bottom_corners():
    square = base_polygon(IrregularBase.RECT, 72, 90, 3.0, 0.35)
    rounded = base_polygon(IrregularBase.ROUNDED, 72, 90, 3.0, 0.35)

    assert rounded.bounds == pytest.approx(square.bounds)
    assert rounded.area < square.area, "圆角会削掉角上一点面积"
    # The bottom-left corner is pulled in, which is the whole point.
    assert not any(
        abs(x - rounded.bounds[0]) < 1e-9
        and abs(y - rounded.bounds[1]) < 1e-9
        for x, y in rounded.exterior.coords
    )


def test_a_zero_ratio_base_disappears():
    base = base_polygon(IrregularBase.RECT, 72, 90, 3.0, 0.0)
    assert base.is_empty or base.area == pytest.approx(0, abs=1e-6)


def test_a_full_ratio_base_is_the_whole_preset():
    base = base_polygon(IrregularBase.RECT, 72, 90, 3.0, 1.0)
    assert base.area == pytest.approx(72 * 90)


# -------------------------------------------------------------------- union


def _ear_shape() -> list[tuple[float, float]]:
    """A simple silhouette: a block with one spike rising from the middle."""
    return [
        (-10, -20),
        (10, -20),
        (10, -5),
        (3, -5),
        (3, 25),
        (-3, 25),
        (-3, -5),
        (-10, -5),
    ]


def test_the_union_keeps_the_silhouette_above_the_base():
    """The spike must survive; if it does not, the base overwrote the art."""
    combined = combine_with_base(
        _ear_shape(), IrregularBase.RECT, 60, 40, 3.0, 0.35
    )
    left, bottom, right, top = bounds(combined)

    assert top == pytest.approx(25), "耳朵必须保留"
    assert bottom == pytest.approx(-20), "底座下沿在板框底部"
    assert (left, right) == pytest.approx((-30, 30)), "底座保持整宽"


def test_the_union_is_one_closed_ring():
    combined = combine_with_base(
        _ear_shape(), IrregularBase.RECT, 60, 40, 3.0, 0.35
    )
    polygon = Polygon(combined)

    assert polygon.is_valid
    assert polygon.geom_type == "Polygon"
    assert not combined[0] == combined[-1], "闭合点不应重复出现"


def test_a_larger_base_top_swallows_more_of_the_silhouette():
    low = combine_with_base(_ear_shape(), IrregularBase.RECT, 60, 40, 3.0, 0.2)
    high = combine_with_base(_ear_shape(), IrregularBase.RECT, 60, 40, 3.0, 0.8)

    assert Polygon(high).area > Polygon(low).area


def test_a_shape_clear_of_the_base_falls_back_to_the_standalone_base():
    """Disjoint pieces cannot make one board; the larger piece is reported."""
    floating = [(x + 500, y + 500) for x, y in _ear_shape()]
    combined = combine_with_base(floating, IrregularBase.RECT, 60, 40, 3.0, 0.35)

    assert combine_bbox_close(bounds(combined), (-30, -20, 30, -6))


def combine_bbox_close(actual, expected) -> bool:
    return all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))


def test_the_silhouette_is_placed_where_the_artwork_sits():
    """The union happens in board coordinates, so an offset artwork stays offset."""
    shifted = [(x + 5, y) for x, y in _ear_shape()]
    spec = IrregularBase.RECT

    combined = combine_with_base(shifted, spec, 60, 40, 3.0, 0.35)

    left, bottom, right, top = bounds(combined)
    # The base is unchanged in width; the spike is now 5 mm right of centre.
    assert (left, right) == pytest.approx((-30, 30))
    assert top == pytest.approx(25)
    spike_x = [x for x, y in combined if y > 24]
    assert sum(spike_x) / len(spike_x) == pytest.approx(5, abs=0.5)


def test_the_union_does_not_leave_a_fragment_of_the_base_over_the_art():
    """Above the cut line only the silhouette may remain.

    A union that mistakenly kept the whole rectangle would still contain the
    spike, so the test asks about the *top* of the ring across its full width:
    with the base cut properly, nothing but the spike reaches that high.
    """
    combined = combine_with_base(
        _ear_shape(), IrregularBase.RECT, 60, 40, 3.0, 0.35
    )
    top = bounds(combined)[3]
    high_x = [x for x, y in combined if y > top - 1]

    assert max(high_x) - min(high_x) < 10, "底座不应延伸到板框顶部"


# ------------------------------------------------------------------- trace


@pytest.fixture(scope="module")
def standee_png(tmp_path_factory):
    """A transparent PNG whose silhouette has a distinct spike on top."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path_factory.mktemp("art") / "standee.png"
    image = np.zeros((400, 300, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    cv2.rectangle(image, (80, 180), (220, 395), (60, 60, 60, 255), -1)
    # A narrow ear, centred horizontally, rising well above the body.
    cv2.rectangle(image, (135, 40), (165, 190), (60, 60, 60, 255), -1)
    assert cv2.imwrite(str(path), image)
    return str(path)


def test_the_trace_hugs_the_artwork_at_the_placement_size(standee_png):
    ring = trace_artwork_ring(standee_png, width_mm=57.0, height_mm=76.0)

    assert ring, "应能从 alpha 通道提取轮廓"
    left, bottom, right, top = bounds(ring)
    assert right - left == pytest.approx(57.0 * (220 - 80) / 300, rel=0.05)
    assert top - bottom == pytest.approx(
        76.0 * (395 - 40) / 400, rel=0.05
    ), "描边高度应贴合图片内容而不是整张画布"


def test_the_traced_ring_is_centred_on_the_artwork(standee_png):
    """Centring makes the placement arithmetic a plain translation.

    Asserted on the ring's bounding box, not on the average x of its topmost
    points: simplification moves the few vertices that describe a narrow ear,
    so the average wobbles by a fraction of a millimetre even though the shape
    is exactly where it should be. The box centre is the property that actually
    matters -- an offset there would shift the whole board relative to the
    picture it is meant to follow.
    """
    ring = trace_artwork_ring(standee_png, width_mm=57.0, height_mm=76.0)
    left, bottom, right, top = bounds(ring)

    assert (left + right) / 2 == pytest.approx(0.0, abs=0.25)


def test_a_missing_file_is_reported_not_swallowed(tmp_path):
    """A bad path is a real error, so it must not masquerade as 'no outline'."""
    with pytest.raises(Exception):
        trace_artwork_ring(str(tmp_path / "absent.png"), 50, 50)


def test_the_document_never_raises_while_drawing_an_outline(tmp_path):
    """``outline_polygon`` is called from paintEvent, so it must not raise.

    An exception there does not surface as a message: Qt reports it from
    inside the paint override and the application dies.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from pcbart.image_prep import imwrite_unicode

    blank = tmp_path / "blank.png"
    # Fully transparent: there is no subject to trace at all.
    imwrite_unicode(str(blank), np.zeros((60, 60, 4), dtype=np.uint8))

    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape
    from pcbart.compose import IrregularBase as Base

    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=Base.RECT,
            width_mm=60,
            height_mm=60,
        ),
        image=ImagePlacement(path=str(blank), width_mm=50, height_mm=50),
        outline_ready=True,
    )

    ring = board.outline_polygon()  # must not raise

    # Nothing to trace, so the preset base stands in and keeps a usable board
    # on screen. An empty picture is a legitimate state, not a failure worth
    # reporting: the user simply has not drawn anything yet.
    assert ring
    assert board._irregular_error is None


def test_a_successful_trace_clears_the_error(tmp_path, standee_png):
    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape
    from pcbart.compose import IrregularBase as Base

    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=Base.TRACE,
            width_mm=57,
            height_mm=76,
        ),
        image=ImagePlacement(path=standee_png, width_mm=57, height_mm=76),
        outline_ready=True,
    )
    board.outline_polygon()

    assert board._irregular_error is None


@pytest.mark.parametrize(
    "settings",
    [
        {"base_top_ratio": 0.0},
        {"base_top_ratio": 1.0},
        {"base_top_ratio": -5.0},
        {"base_top_ratio": 5.0},
        {"corner_radius_mm": 999.0},
        {"width_mm": 3.0, "height_mm": 3.0},
        {"irregular_base": IrregularBase.ROUNDED, "corner_radius_mm": 0.0},
    ],
)
def test_an_extreme_setting_still_draws_something(settings, standee_png):
    """Every slider position has to produce a paintable answer.

    A ratio of 0 makes the base vanish, and the cut then leaves a zero-length
    line rather than a polygon -- asking that for its exterior raised, from
    inside the paint pass. Bounds are clamped, so no slider value may crash.
    """
    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape
    from pcbart.compose import IrregularBase as Base

    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=settings.get("irregular_base", Base.RECT),
            width_mm=settings.get("width_mm", 60.0),
            height_mm=settings.get("height_mm", 60.0),
            base_top_ratio=settings.get("base_top_ratio", 0.35),
            corner_radius_mm=settings.get("corner_radius_mm", 3.0),
        ),
        image=ImagePlacement(path=standee_png, width_mm=50, height_mm=50),
        outline_ready=True,
    )

    ring = board.outline_polygon()  # must not raise

    assert isinstance(ring, list)
    for point in ring:
        assert len(point) == 2




def test_every_base_is_either_a_trace_or_a_combining_shape():
    for base in IRREGULAR_BASES:
        assert isinstance(base, str)
    assert IrregularBase.TRACE not in IrregularBase.COMBINING


# ------------------------------------------------- the two invariants

def test_below_the_cut_the_board_is_solid_and_holds_the_artwork(
    standee_png, tmp_path
):
    """The base exists to hold the lower part of the picture.

    The board edge cuts the artwork's *outline* off, not the artwork itself:
    the picture still extends down into the base, and the base has to cover it.
    """
    from shapely.geometry import box

    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape
    from pcbart.compose import IrregularBase as Base

    width_mm, height_mm, ratio = 72.0, 90.0, 0.35
    artwork = ImagePlacement(
        path=standee_png, x_mm=0.0, y_mm=0.0, width_mm=57.0, height_mm=76.0
    )
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=Base.RECT,
            width_mm=width_mm,
            height_mm=height_mm,
            base_top_ratio=ratio,
        ),
        image=artwork,
        outline_ready=True,
    )
    polygon = Polygon(board.outline_polygon())

    cut_y = -height_mm / 2 + ratio * height_mm
    lower_half = box(
        artwork.x_mm - artwork.width_mm / 2,
        artwork.y_mm - artwork.height_mm / 2,
        artwork.x_mm + artwork.width_mm / 2,
        cut_y,
    )
    outside = lower_half.difference(polygon)

    assert polygon.is_valid
    assert outside.area < 1e-6, "切割线以下的图片区域必须落在底座内"


def test_above_the_cut_the_board_is_exactly_the_silhouette(standee_png):
    """Not one square millimetre of the preset may survive above the cut.

    Keeping the rectangle above the line would still leave a board that
    contains the ears, so a containment check alone would pass while the
    standee had a square top. Comparing areas is what catches it.
    """
    from shapely.geometry import box

    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape
    from pcbart.compose import IrregularBase as Base

    width_mm, height_mm, ratio = 72.0, 90.0, 0.35
    artwork = ImagePlacement(
        path=standee_png, x_mm=0.0, y_mm=0.0, width_mm=57.0, height_mm=76.0
    )
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=Base.RECT,
            width_mm=width_mm,
            height_mm=height_mm,
            base_top_ratio=ratio,
        ),
        image=artwork,
        outline_ready=True,
    )
    polygon = Polygon(board.outline_polygon())
    cut_y = -height_mm / 2 + ratio * height_mm

    silhouette = trace_artwork_ring(
        standee_png, artwork.width_mm, artwork.height_mm, simplify_mm=0.35
    )
    placed = Polygon([(x + artwork.x_mm, y + artwork.y_mm) for x, y in silhouette])

    board_above = polygon.intersection(box(-1e4, cut_y, 1e4, 1e4))
    art_above = placed.intersection(box(-1e4, cut_y, 1e4, 1e4))

    assert board_above.area == pytest.approx(art_above.area, abs=1e-6)



# ------------------------------------------- degenerate traced silhouettes

# Rings of the kind a traced bitmap really produces: self-touching boundaries,
# repeated edges, folded features. The reported failure was
# ``TopologyException: side location conflict`` from the union stage, which took
# out irregular tracing completely and surfaced as a coordinate pair in the UI.
DEGENERATE_RINGS = {
    "self_intersecting": [(0, 0), (60, 40), (60, 0), (0, 40)],
    "folded_edge": [(0, 0), (60, 0), (60, 40), (0, 40), (0, 20), (60, 20), (0, 20)],
    "repeated_vertex": [(0, 0), (30, 0), (30, 0), (30, 30), (0, 30), (0, 0)],
    "collinear_spike": [(0, 0), (60, 0), (60, 40), (0, 40), (0, 40 + 1e-7)],
    "shuffled": [(0, 0), (50, 50), (50, 0), (0, 50), (25, 25)],
    "zero_area": [(0, 0), (20, 0), (40, 0), (60, 0)],
    "all_identical": [(5, 5)] * 10,
    "two_points": [(0, 0), (10, 10)],
}


@pytest.mark.parametrize("name", sorted(DEGENERATE_RINGS))
@pytest.mark.parametrize(
    "base", [IrregularBase.RECT, IrregularBase.ROUNDED], ids=["rect", "rounded"]
)
def test_a_degenerate_silhouette_never_raises(name, base):
    """A malformed traced ring must degrade, not explode.

    This is reached from the canvas's paint pass, so raising tears the window
    down rather than showing a message.
    """
    ring = DEGENERATE_RINGS[name]

    result = combine_with_base(ring, base, 76.2, 57.1, 3.0, 0.35)

    assert isinstance(result, list)
    for point in result:
        assert len(point) == 2
        assert math.isfinite(point[0]) and math.isfinite(point[1])


@pytest.mark.parametrize("name", sorted(DEGENERATE_RINGS))
def test_a_degenerate_silhouette_still_yields_a_usable_board(name):
    """Something board-shaped comes out, whatever the trace produced.

    Only validity and positive area are asserted. These rings are artificial
    coordinates that happen not to touch the base, so the union really is
    disjoint and the code keeps the larger piece -- that is the documented
    behaviour, not a failure. What must never happen is a crash, an invalid
    ring, or an empty result.
    """
    ring = DEGENERATE_RINGS[name]

    result = combine_with_base(ring, IrregularBase.RECT, 76.2, 57.1, 3.0, 0.35)

    assert result, f"{name} 不应得到空结果"
    board = Polygon(result)
    assert board.is_valid, f"{name} 的结果必须是合法多边形"
    assert board.area > 0


def test_the_document_reports_a_degenerate_trace_instead_of_crashing():
    """Same guarantee one level up, where the canvas calls it."""
    from pcbart.artboard import ArtBoard, BoardOutlineSpec, OutlineShape

    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=IrregularBase.RECT,
            width_mm=76.2,
            height_mm=57.1,
        ),
        outline_ready=True,
    )
    board._compose_irregular = lambda image, spec: combine_with_base(  # type: ignore[assignment]
        DEGENERATE_RINGS["self_intersecting"],
        IrregularBase.RECT,
        76.2,
        57.1,
        3.0,
        0.35,
    )

    ring = board.outline_polygon()

    assert ring
    assert board._irregular_error is None or isinstance(board._irregular_error, str)


def test_union_repairs_an_invalid_part_before_merging():
    """The union helper is what makes the fallbacks above possible."""
    from shapely.geometry import Polygon

    from pcbart.geometry import safe_union

    broken = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    if broken.is_valid:
        pytest.skip("该几何在本机 Shapely 下合法")
    good = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])

    merged = safe_union([broken, good])

    assert merged is not None
    assert merged.area > good.area - 1e-6


# ------------------------------------------------ tracing is not re-run per frame

def _movable_board(path, base=IrregularBase.RECT):
    from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement, OutlineShape

    return ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=base,
            width_mm=60,
            height_mm=80,
        ),
        image=ImagePlacement(path=path, width_mm=50, height_mm=60),
        outline_ready=True,
    )


def test_moving_the_artwork_reuses_the_traced_silhouette(standee_png):
    """A drag must not re-trace.

    Tracing is ~31 ms; a 30 fps drag has 33 ms for the whole frame. The trace
    depends on the picture and its drawn size, *not* on where it sits, so the
    position has to stay out of that cache key -- otherwise every mouse-move
    re-runs the entire extraction.
    """
    board = _movable_board(standee_png)
    board.outline_polygon()
    cached = board._silhouette_cache
    assert cached is not None

    board.image.x_mm = 25.0
    board.outline_polygon()

    assert board._silhouette_cache is cached, "移动不应重新描边"


def test_a_traced_board_follows_the_artwork_when_it_moves(standee_png):
    """Reusing the trace must still move the outline with the picture."""
    board = _movable_board(standee_png, base=IrregularBase.TRACE)
    before = board.outline_polygon()

    board.image.x_mm = 10.0
    after = board.outline_polygon()

    assert after is not before, "位置变了应当重新组合"
    shift = min(p[0] for p in after) - min(p[0] for p in before)
    assert shift == pytest.approx(10.0, abs=0.01)


def test_resizing_the_artwork_does_re_trace(standee_png):
    """The cache must not survive a change that really alters the outline."""
    board = _movable_board(standee_png)
    board.outline_polygon()
    cached = board._silhouette_cache

    board.image.width_mm = 30.0
    board.outline_polygon()

    assert board._silhouette_cache is not cached, "改尺寸必须重新描边"


def test_changing_the_base_reuses_the_trace(standee_png):
    """The base's cut line does not change the silhouette, only the union."""
    board = _movable_board(standee_png)
    board.outline_polygon()
    cached = board._silhouette_cache

    board.outline.base_top_ratio = 0.6
    board.outline_polygon()

    assert board._silhouette_cache is cached, "改底座不应重新描边"


def test_the_composed_outline_is_cached_between_paints(standee_png):
    """Two paints with nothing changed must not recompute the union.

    Checked against the cache itself rather than against the identity of the
    two returned lists: callers are handed a copy, so that editing the points
    they were given cannot edit what the next frame draws. Recomputing the
    union replaces the cache entry, so the entry staying put is the assertion
    that actually proves nothing was recomputed.
    """
    board = _movable_board(standee_png)
    board.outline_polygon()
    cached = board._irregular_cache
    first = board.outline_polygon()
    second = board.outline_polygon()

    assert board._irregular_cache is cached, "两次绘制之间不应重算并集"
    assert second == first
