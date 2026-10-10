"""Tests for the badge (通行证) template's artwork regions.

The numbers in :mod:`pcbart.badge_template` are measurements of a real
EasyEDA Pro file, so the tests here do two different jobs:

* pin the *rules* -- which artwork goes in which region, and how it is allowed
  to sit -- with cases that are easy to reason about;
* pin the *measurements* against the source file itself, so that if someone
  edits the constants, or the template is re-exported with a region that moved,
  the disagreement is reported instead of silently placing artwork in the wrong
  spot.

The source file lives outside the repository (3.7 MB) in the materials library.
When it is not there the measurement tests skip, so the suite still runs on a
machine that has not got the templates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_template as bt  # noqa: E402
from pcbart.paths import asset_root  # noqa: E402

#: ⚠️ Looked up under ``PCBART_ASSET_ROOT`` rather than a literal path. The
#: library sits outside the repository and on another drive on some machines,
#: so a hard-coded location would make these measurement tests silently skip
#: for every collaborator -- and "skipped" looks exactly like "not applicable"
#: in a test report, which is how a real regression would slip through.
TEMPLATE_CANDIDATES = (
    asset_root()
    / "通行证模板源文件"
    / "ProPrj_鸿雪_2026-09-25.epro2",
)


def template_path() -> Path | None:
    for candidate in TEMPLATE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# ------------------------------------------------------------------- regions


def test_regions_sit_inside_the_card():
    """Neither region may hang off the 50 × 100 board."""
    for region in (bt.SMALL_REGION, bt.LARGE_REGION):
        x0, y0, x1, y1 = region.bounds
        assert x0 >= -bt.CARD_WIDTH_MM / 2 - 1e-9
        assert x1 <= bt.CARD_WIDTH_MM / 2 + 1e-9
        assert y0 >= -bt.CARD_HEIGHT_MM / 2 - 1e-9
        assert y1 <= bt.CARD_HEIGHT_MM / 2 + 1e-9


def test_the_two_regions_have_the_measured_sizes():
    """Straight from the template's two filled rectangles."""
    assert bt.SMALL_REGION.width_mm == pytest.approx(38.861, abs=0.001)
    assert bt.SMALL_REGION.height_mm == pytest.approx(56.324, abs=0.001)
    assert bt.LARGE_REGION.width_mm == pytest.approx(42.526, abs=0.001)
    assert bt.LARGE_REGION.height_mm == pytest.approx(99.626, abs=0.001)


def test_the_large_region_is_nearly_the_whole_card():
    """It is the tall one, and the reason 精二 artwork needs its own rule."""
    assert bt.LARGE_REGION.height_mm > 9 * bt.LARGE_REGION.width_mm / 4
    assert bt.LARGE_REGION.height_mm > bt.SMALL_REGION.height_mm


def test_region_bounds_agree_with_centre_and_size():
    region = bt.LARGE_REGION
    x0, y0, x1, y1 = region.bounds
    assert (x0 + x1) / 2 == pytest.approx(region.centre_x_mm)
    assert (y0 + y1) / 2 == pytest.approx(region.centre_y_mm)
    assert x1 - x0 == pytest.approx(region.width_mm)
    assert y1 - y0 == pytest.approx(region.height_mm)


def test_each_region_has_its_own_key_and_label():
    keys = [region.key for region in bt.REGIONS.values()]
    assert len(keys) == len(set(keys))
    for region in bt.REGIONS.values():
        assert region.label


# ---------------------------------------------------------------------- rules


def test_elite_0_and_elite_1_use_the_short_region():
    for kind in (bt.ArtworkKind.ELITE0, bt.ArtworkKind.ELITE1):
        plan = bt.plan_artwork(kind)
        assert plan.region is bt.SMALL_REGION
        assert plan.mode is bt.PlacementMode.WIDTH_FIT


def test_elite_2_uses_the_tall_region_and_containment():
    plan = bt.plan_artwork(bt.ArtworkKind.ELITE2)
    assert plan.region is bt.LARGE_REGION
    assert plan.mode is bt.PlacementMode.CONTAINED


def test_a_skin_can_go_in_either_region_and_the_rule_follows():
    """小区域皮肤 and 大区域皮肤 are two different choices, not one derived."""
    small = bt.plan_artwork(bt.ArtworkKind.SKIN, skin_region=bt.SMALL_REGION)
    large = bt.plan_artwork(bt.ArtworkKind.SKIN, skin_region=bt.LARGE_REGION)
    assert small.region is bt.SMALL_REGION and small.mode is bt.PlacementMode.WIDTH_FIT
    assert large.region is bt.LARGE_REGION and large.mode is bt.PlacementMode.CONTAINED


def test_a_skin_rejects_a_region_that_is_not_one():
    with pytest.raises(TypeError):
        bt.plan_artwork(bt.ArtworkKind.SKIN, skin_region="large")


def test_an_unknown_kind_is_rejected_rather_than_defaulted():
    """Silently falling back would place the wrong picture without saying so."""
    with pytest.raises(TypeError):
        bt.plan_artwork("elite2")


def test_only_amiya_is_offered_elite_1():
    plain = bt.kinds_for_operator()
    amiya = bt.kinds_for_operator(amiya=True)
    assert bt.ArtworkKind.ELITE1 not in plain
    assert bt.ArtworkKind.ELITE1 in amiya
    assert len(amiya) == len(plain) + 1
    # The other three are always offered.
    for kind in (bt.ArtworkKind.ELITE0, bt.ArtworkKind.ELITE2, bt.ArtworkKind.SKIN):
        assert kind in plain and kind in amiya


# ------------------------------------------------------------------- geometry


def test_rotated_bounds_of_an_unrotated_box_is_the_box():
    bounds = bt.rotated_bounds((10.0, 20.0), (4.0, 6.0))
    assert bounds == pytest.approx((8.0, 17.0, 12.0, 23.0))


def test_a_quarter_turn_swaps_the_bounds():
    bounds = bt.rotated_bounds((0.0, 0.0), (4.0, 6.0), 90.0)
    assert bounds == pytest.approx((-3.0, -2.0, 3.0, 2.0))


def test_bounds_of_a_rotated_box_are_larger_than_the_box():
    """A rotated rectangle needs more room, which is the whole difficulty.

    Not every side grows -- turning a obviously-wide box can shorten its
    horizontal extent -- but the box's *area* always does, and that is what
    decides whether it still fits.
    """
    plain = bt.rotated_bounds((0.0, 0.0), (10.0, 2.0))
    turned = bt.rotated_bounds((0.0, 0.0), (10.0, 2.0), 45.0)
    plain_area = (plain[2] - plain[0]) * (plain[3] - plain[1])
    turned_area = (turned[2] - turned[0]) * (turned[3] - turned[1])
    assert turned_area > plain_area
    # The narrow side is the one that grows, and it is what stops it fitting.
    assert (turned[3] - turned[1]) > (plain[3] - plain[1])


def test_a_square_box_is_smallest_at_45_degrees():
    """The other direction of the same fact, so neither is mistaken for a rule."""
    straight = bt.rotated_bounds((0.0, 0.0), (10.0, 10.0))
    turned = bt.rotated_bounds((0.0, 0.0), (10.0, 10.0), 45.0)
    straight_w = straight[2] - straight[0]
    turned_w = turned[2] - turned[0]
    assert turned_w > straight_w
    assert turned_w == pytest.approx(10.0 * 2 ** 0.5)


def test_negative_sizes_are_rejected():
    with pytest.raises(ValueError):
        bt.rotated_bounds((0.0, 0.0), (-1.0, 2.0))


def test_fits_inside_a_centred_box():
    region = bt.SMALL_REGION
    assert bt.fits_inside(region, region.centre, (10.0, 10.0))
    assert not bt.fits_inside(region, region.centre, (region.width_mm + 1.0, 10.0))


def test_overflow_reports_zero_when_it_fits():
    region = bt.LARGE_REGION
    assert bt.overflow_mm(region, region.centre, (20.0, 20.0)) == (0.0, 0.0)


def test_overflow_counts_the_overhang_on_the_bad_side():
    """A centred picture 4 mm too wide overhangs 2 mm on each side."""
    region = bt.SMALL_REGION
    over_x, over_y = bt.overflow_mm(region, region.centre, (region.width_mm + 4.0, 10.0))
    assert over_x == pytest.approx(2.0)
    assert over_y == 0.0


def test_overflow_of_an_off_centre_picture_reads_the_far_edge():
    """Pushed to one side, the whole overhang is on that side."""
    region = bt.SMALL_REGION
    x1 = region.bounds[2]
    over_x, _ = bt.overflow_mm(
        region, (x1, region.centre_y_mm), (10.0, 10.0)
    )
    assert over_x == pytest.approx(5.0)


# ------------------------------------------------------------------- contain


def test_a_picture_that_already_fits_is_left_alone():
    """Calling contain on a good placement must not nudge it."""
    region = bt.LARGE_REGION
    fit = bt.contain(region, region.centre, (20.0, 40.0))
    assert fit.scale == 1.0
    assert fit.dx_mm == 0.0 and fit.dy_mm == 0.0
    assert not fit.changed


def test_too_wide_is_shrunk_to_the_region_width():
    region = bt.LARGE_REGION
    fit = bt.contain(region, region.centre, (region.width_mm * 2, 10.0))
    assert fit.scale == pytest.approx(0.5)
    assert fit.dx_mm == 0.0 and fit.dy_mm == 0.0


def test_too_tall_is_shrunk_to_the_region_height():
    region = bt.LARGE_REGION
    fit = bt.contain(region, region.centre, (10.0, region.height_mm * 4))
    assert fit.scale == pytest.approx(0.25)


def test_a_picture_is_only_ever_shrunk_never_grown():
    """Growing it would move an edge the user had already placed."""
    region = bt.LARGE_REGION
    for size in ((1.0, 1.0), (10.0, 10.0), (region.width_mm, region.height_mm)):
        assert bt.contain(region, region.centre, size).scale <= 1.0


def test_an_off_centre_picture_is_slid_back_in_without_resizing():
    region = bt.LARGE_REGION
    x0, y0, _, _ = region.bounds
    fit = bt.contain(region, (x0 - 20.0, region.centre_y_mm), (10.0, 10.0))
    assert fit.scale == 1.0
    assert fit.dx_mm > 0.0
    assert fit.dy_mm == 0.0


def test_contain_always_produces_something_that_fits():
    """The property that matters, over deliberately awkward inputs."""
    region = bt.LARGE_REGION
    cases = [
        ((0.0, 0.0), (1.0, 1.0), 0.0),
        ((0.0, 0.0), (500.0, 20.0), 0.0),
        ((0.0, 0.0), (20.0, 500.0), 0.0),
        ((50.0, 50.0), (30.0, 30.0), 0.0),
        ((-80.0, -80.0), (30.0, 30.0), 0.0),
        ((0.0, 0.0), (80.0, 30.0), 30.0),
        ((0.0, 0.0), (30.0, 80.0), 137.0),
        ((12.0, -40.0), (90.0, 90.0), 45.0),
        (region.centre, (region.width_mm, region.height_mm), 0.0),
    ]
    for centre, size, rotation in cases:
        fit = bt.contain(region, centre, size, rotation)
        new_centre = (centre[0] + fit.dx_mm, centre[1] + fit.dy_mm)
        new_size = (size[0] * fit.scale, size[1] * fit.scale)
        assert bt.fits_inside(region, new_centre, new_size, rotation), (
            f"contain failed for {centre} {size} rot={rotation}"
        )


def test_contain_is_idempotent():
    """Running it twice must not walk the picture across the region."""
    region = bt.LARGE_REGION
    fit = bt.contain(region, (60.0, 60.0), (90.0, 40.0))
    centre = (60.0 + fit.dx_mm, 60.0 + fit.dy_mm)
    size = (90.0 * fit.scale, 40.0 * fit.scale)
    second = bt.contain(region, centre, size)
    assert not second.changed


def test_a_touching_edge_counts_as_inside():
    """Exact fits are the common case -- a width-fit picture is exactly as wide."""
    region = bt.SMALL_REGION
    assert bt.fits_inside(region, region.centre, (region.width_mm, region.height_mm))


def test_contain_rejects_a_region_with_no_size():
    broken = bt.ArtworkRegion("x", "x", 0.0, 10.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        bt.contain(broken, (0.0, 0.0), (1.0, 1.0))


def test_apply_contain_changes_the_placement_and_reports_the_fit():
    from pcbart.artboard import ImagePlacement

    region = bt.LARGE_REGION
    placement = ImagePlacement(
        path="", x_mm=60.0, y_mm=60.0, width_mm=90.0, height_mm=40.0
    )
    fit = bt.apply_contain(region, placement)
    assert fit.changed
    assert bt.fits_inside(
        region,
        (placement.x_mm, placement.y_mm),
        (placement.width_mm, placement.height_mm),
        placement.rotation_deg,
    )


def test_apply_contain_keeps_a_rotated_placement_inside():
    from pcbart.artboard import ImagePlacement

    region = bt.SMALL_REGION
    placement = ImagePlacement(
        path="",
        x_mm=0.0,
        y_mm=0.0,
        width_mm=80.0,
        height_mm=20.0,
        rotation_deg=35.0,
    )
    bt.apply_contain(region, placement)
    assert bt.fits_inside(
        region,
        (placement.x_mm, placement.y_mm),
        (placement.width_mm, placement.height_mm),
        placement.rotation_deg,
    )


# ---------------------------------------------------------------- placement


def test_a_width_fit_placement_matches_the_region_width():
    plan = bt.plan_artwork(bt.ArtworkKind.ELITE0)
    placement = bt.artwork_placement(plan, "x.png", aspect=0.5)
    assert placement.width_mm == pytest.approx(bt.SMALL_REGION.width_mm)
    assert placement.height_mm == pytest.approx(bt.SMALL_REGION.width_mm / 0.5)
    assert (placement.x_mm, placement.y_mm) == bt.SMALL_REGION.centre


def test_a_contained_placement_fits_the_region_out_of_the_box():
    for aspect in (0.3, 0.5, 1.0, 2.0, 5.0):
        plan = bt.plan_artwork(bt.ArtworkKind.ELITE2)
        placement = bt.artwork_placement(plan, "x.png", aspect=aspect)
        assert bt.fits_inside(
            bt.LARGE_REGION,
            (placement.x_mm, placement.y_mm),
            (placement.width_mm, placement.height_mm),
        ), f"aspect {aspect} did not fit"


def test_a_contained_placement_is_as_large_as_it_can_be():
    """It should touch at least one pair of edges, or it is leaving room unused."""
    for aspect in (0.3, 0.7, 1.0, 2.2):
        plan = bt.plan_artwork(bt.ArtworkKind.ELITE2)
        placement = bt.artwork_placement(plan, "x.png", aspect=aspect)
        touches_width = placement.width_mm == pytest.approx(
            bt.LARGE_REGION.width_mm, abs=1e-6
        )
        touches_height = placement.height_mm == pytest.approx(
            bt.LARGE_REGION.height_mm, abs=1e-6
        )
        assert touches_width or touches_height


def test_a_width_fit_placement_reports_the_overhang_when_it_is_not_cropped():
    """The half-body pictures are pre-cropped; an uncropped one overhangs.

    Matching the width is what this mode promises, so the overhang is reported
    rather than quietly fixed by changing the width.
    """
    plan = bt.plan_artwork(bt.ArtworkKind.ELITE0)
    placement = bt.artwork_placement(plan, "x.png", aspect=0.5)
    _, over_y = bt.overflow_mm(
        bt.SMALL_REGION,
        (placement.x_mm, placement.y_mm),
        (placement.width_mm, placement.height_mm),
    )
    assert over_y > 0.0


def test_a_placement_carries_the_layer_and_the_paths_through():
    plan = bt.plan_artwork(bt.ArtworkKind.ELITE2)
    placement = bt.artwork_placement(
        plan,
        "source.png",
        aspect=1.0,
        processed_path="cache.png",
        import_id="abc123",
    )
    assert placement.layer == bt.ARTWORK_LAYER
    assert placement.path == "source.png"
    assert placement.processed_path == "cache.png"
    assert placement.import_id == "abc123"


def test_the_artwork_layer_is_the_top_silkscreen():
    """The two worked examples in the file put their artwork there."""
    assert bt.ARTWORK_LAYER == "顶层丝印层"


def test_an_empty_path_is_allowed_for_geometry_only_callers():
    plan = bt.plan_artwork(bt.ArtworkKind.ELITE0)
    placement = bt.artwork_placement(plan, "", aspect=0.5)
    assert placement.path == ""
    assert placement.width_mm > 0


# ----------------------------------------------- against the real template file


@pytest.mark.skipif(template_path() is None, reason="模板源文件不在本机素材库")
def test_the_constants_match_the_template_file():
    """Measure the real file and require the constants to agree.

    This is the test that notices a template whose regions moved: it fails with
    the measured numbers instead of letting artwork land on top of the card's
    printed frame.
    """
    from tools.extract_badge_regions import extract_regions

    regions = extract_regions(template_path(), min_mm=30.0)
    assert len(regions) == 2, "模板里应当有且只有两个立绘填充区"

    measured = sorted(
        (box.width, box.height, box.centre[0] - card.centre[0], box.centre[1] - card.centre[1])
        for box, card in regions
    )
    expected = sorted(
        (r.width_mm, r.height_mm, r.centre_x_mm, r.centre_y_mm)
        for r in (bt.SMALL_REGION, bt.LARGE_REGION)
    )
    for got, want in zip(measured, expected):
        for a, b in zip(got, want):
            assert a == pytest.approx(b, abs=0.01)


@pytest.mark.skipif(template_path() is None, reason="模板源文件不在本机素材库")
def test_the_template_has_four_boards_two_of_them_templates():
    """The file is four cards: two with an artwork region and two examples."""
    from tools.extract_badge_regions import extract_boards

    boards = extract_boards(template_path())
    assert len(boards) == 4
    for board in boards:
        assert board.width == pytest.approx(bt.CARD_WIDTH_MM, abs=0.2)
        assert board.height == pytest.approx(bt.CARD_HEIGHT_MM, abs=0.2)


# ------------------------------------------------------- text anchor check

def test_the_text_table_matches_the_template_files():
    """Every line's position, size, rotation, layer and font, re-measured.

    This is the check whose absence let a preview look offset: the table is
    hand-typed from measurements, so a typo there moves a line on every card.
    ``tools/verify_text_anchors.py`` re-reads both source projects and compares
    field by field; running it here keeps it from rotting.
    """
    from tools import verify_text_anchors as checker

    directory = checker.TEMPLATE_DIR
    if not directory.is_dir():
        pytest.skip("通行证模板源文件未安装")

    small = checker.read_texts(directory / checker.SMALL_CARD)
    large = checker.read_texts(directory / checker.LARGE_CARD)
    assert not checker.compare(
        small, "小卡", tolerance_mm=checker.POSITION_TOLERANCE_MM
    )
    assert not checker.compare(
        large, "大卡", tolerance_mm=checker.TEMPLATE_DRIFT_MM
    )


def test_the_two_templates_drift_only_by_a_harmless_amount():
    """Where the authored files disagree, the gap has to stay invisible.

    The table follows the small card. A future re-export that moves a line far
    enough to matter would otherwise leave the table describing one file while
    the other is used.
    """
    from tools import verify_text_anchors as checker

    directory = checker.TEMPLATE_DIR
    if not directory.is_dir():
        pytest.skip("通行证模板源文件未安装")

    small = checker.read_texts(directory / checker.SMALL_CARD)
    large = checker.read_texts(directory / checker.LARGE_CARD)
    for row in small:
        other = next(
            (
                item
                for item in large
                if item.text == row.text
                and item.layer == row.layer
                and abs(item.rotation_deg - row.rotation_deg) < 0.01
            ),
            None,
        )
        if other is None:
            assert row.text in {"阵营logo"}, row.text
            continue
        assert abs(other.x_mm - row.x_mm) <= checker.TEMPLATE_DRIFT_MM, row.text
        assert abs(other.y_mm - row.y_mm) <= checker.TEMPLATE_DRIFT_MM, row.text


def test_the_text_anchor_is_the_line_box_bottom_not_the_baseline():
    """The template's LEFT_BOTTOM sits one descender below the glyphs.

    This is the bug the user reported as "the horizontal text sits too low".
    Rendering from the baseline put every line about a millimetre down: against
    the project's own rendered card, STAFF's ink bottom came out 0.02 mm *below*
    its anchor where the template puts it 1.07 mm *above* it.

    The source project settles the question itself: the ``50.000mm`` dimension
    label is anchored at exactly the y of the line it labels, yet in the drawn
    card the digits sit above that line -- one descender of clearance.
    """
    from pcbart.text_render import line_box_descent_mm, render_text_mask

    # A line with a clear, measurable box: the footer's own wording.
    for label, text, size, family in (
        ("副标题", "Operator of Rhodes Island", 0.8, "default2"),
        ("职业阵营行", r"SUPPORTER \ RHODES ISLAND", 0.8, "default2"),
        ("STAFF", "STAFF", 1.6, "Dubai Regular"),
    ):
        rendered = render_text_mask(text, size_mm=size, family=family)
        assert rendered is not None, label
        bottom = rendered.centre_mm[1] - rendered.size_mm[1] / 2
        # The ink bottom has to sit *above* the anchor by roughly the descent.
        # Before the fix it was at or below zero for every one of these.
        assert bottom > 0.0, f"{label}: 墨迹底 {bottom:+.3f} 落在锚点下方"
        descent = line_box_descent_mm(size, family, text)
        assert descent > 0.0, label
        # And the clearance must be a fraction of a descender, not a whole line.
        assert bottom <= descent + size * 0.35, (
            f"{label}: 墨迹底 {bottom:.3f} 高出 descender {descent:.3f} 太多"
        )


def test_the_canvas_and_the_exporter_shift_text_by_the_same_amount():
    """One definition of the descender, or the preview drifts from the board.

    The canvas draws with Qt and the writers render with PIL. If only one of
    them lifted the pen, the two would disagree by exactly the descender --
    which is the same failure mode as the font substitution fixed alongside it.
    """
    from pcbart.text_render import line_box_descent_mm
    from pcbart import badge_card

    for item in badge_card.badge_texts():
        text = item.resolved_text(name="阿米娅", number="R001")
        descent = line_box_descent_mm(item.size_mm, item.font, text)
        assert 0.0 < descent < item.size_mm, (
            f"{item.label}: descender {descent:.3f} 不在 (0, {item.size_mm}) 内"
        )
