"""Tests for the colour-dodge line-art conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.lineart import (  # noqa: E402
    MAX_RADIUS,
    MIN_RADIUS,
    RADIUS_RATIO,
    LineArtParams,
    auto_radius,
    desaturate,
    dodge_response,
    line_art,
    line_art_preview,
    local_max,
)


# ------------------------------------------------------------------ radius


def test_radius_follows_image_size():
    """A fixed radius would give thin lines on big images and fat on small ones.

    1000 px short side at the calibrated ratio is 10.5, so 10 after rounding.
    """
    assert auto_radius((1000, 1000)) == round(1000 * RADIUS_RATIO)
    assert auto_radius((2000, 2000)) > auto_radius((1000, 1000))


def test_radius_uses_the_short_side():
    """A wide panorama must not get a radius scaled from its long side."""
    assert auto_radius((400, 4000)) == auto_radius((400, 400))


def test_radius_is_clamped_at_both_ends():
    assert auto_radius((1, 1)) == MIN_RADIUS
    assert auto_radius((100000, 100000)) == MAX_RADIUS


def test_strength_scales_the_radius():
    assert auto_radius((1000, 1000), strength=2.0) > auto_radius((1000, 1000), 1.0)


# -------------------------------------------------------------- primitives


def test_desaturate_matches_photoshop_formula():
    """Photoshop 去色 is (max + min) / 2 per pixel, not a luma weighting."""
    pixel = np.array([[[0, 128, 255]]], np.uint8)
    assert desaturate(pixel)[0, 0] == pytest.approx((0 + 255) / 2)


def test_desaturate_passes_greyscale_through():
    grey = np.array([[0, 128, 255]], np.uint8)
    assert np.array_equal(desaturate(grey), grey.astype(np.float32))


def test_local_max_of_a_uniform_image_is_that_value():
    flat = np.full((40, 40), 0.3, np.float32)
    maximum = local_max(flat, radius=5)
    assert maximum == pytest.approx(0.3)


def test_local_max_finds_a_bright_spot_in_a_dark_field():
    image = np.zeros((41, 41), np.float32)
    image[20, 20] = 1.0
    maximum = local_max(image, radius=3)
    # The spot raises the maximum over its whole neighbourhood.
    assert maximum[20, 20] == pytest.approx(1.0)
    assert maximum[20, 22] == pytest.approx(1.0)
    # Far away the maximum is still zero.
    assert maximum[0, 0] == pytest.approx(0.0)


# ------------------------------------------------------------- the response


def test_dodge_response_alone_whitens_a_large_dark_area():
    """The defect the multiply term exists to fix.

    Inside a uniform black region every pixel equals its own local maximum, so
    the plain dodge ratio is 1 regardless of how dark the region is.
    """
    grey = np.full((120, 120), 0.2, np.float32)
    dodge_only = np.clip(grey / np.maximum(local_max(grey, radius=5), 1e-6), 0, 1)
    assert dodge_only[60, 60] == pytest.approx(1.0)


def test_response_keeps_a_large_dark_area_dark():
    """Multiplying by the original grey restores absolute darkness."""
    grey = np.full((120, 120), 0.2, np.float32)
    response = dodge_response(grey, radius=5)
    assert response[60, 60] == pytest.approx(0.2, abs=1e-6)


def test_response_is_bounded():
    rng = np.random.default_rng(3)
    grey = rng.random((80, 80)).astype(np.float32)
    response = dodge_response(grey, radius=4)
    assert response.min() >= 0.0
    assert response.max() <= 1.0


# ------------------------------------------------------------------ output


def test_line_art_returns_two_tone_by_default():
    rng = np.random.default_rng(11)
    image = (rng.random((160, 160, 3)) * 255).astype(np.uint8)
    result = line_art(image)

    assert result.dtype == np.uint8
    assert result.ndim == 2
    values = np.unique(result)
    assert set(values.tolist()) <= {0, 255}, "output must be pure black and white"


def test_line_art_keeps_a_dark_collar_solid():
    """A regression for the reported defect: a big black area must stay black."""
    image = np.full((400, 400, 3), 255, np.uint8)
    # A dark block far larger than any plausible filter radius.
    image[120:300, 120:300] = 40
    result = line_art(image)

    centre = result[210, 210]
    assert centre == 0, "the interior of a large dark shape must be inked"


def test_line_art_of_a_blank_image_is_blank():
    blank = np.full((120, 120), 255, np.uint8)
    result = line_art(blank)
    assert (result == 255).all()


def test_line_art_draws_a_line_where_tone_changes():
    image = np.full((200, 200), 255, np.uint8)
    image[:, 100:] = 0
    result = line_art(image)
    # Somewhere near the boundary there must be ink.
    assert (result[:, 95:105] < 128).any()


def test_alpha_is_composited_onto_white():
    """Transparent areas are paper, so they must not become ink."""
    image = np.zeros((150, 150, 4), np.uint8)
    image[:, :, 0:3] = 30
    image[50:100, 50:100, 3] = 255  # only this square is a real, dark subject

    result = line_art(image)
    assert (result[0:20, 0:20] == 255).all(), "transparent背景 must be paper"
    assert (result[60:90, 60:90] < 128).any(), "the solid subject must be inked"


def test_line_art_accepts_greyscale_input():
    grey = np.full((120, 120), 200, np.uint8)
    grey[40:80, 40:80] = 60
    result = line_art(grey)
    assert result.shape == (120, 120)


# ------------------------------------------------------------------ params


def test_params_clamp_to_a_usable_window():
    assert LineArtParams(strength=99).normalised().strength == 4.0
    assert LineArtParams(strength=-5).normalised().strength == 0.2
    assert LineArtParams(threshold=-1).normalised().threshold == 0.02
    assert LineArtParams(threshold=5).normalised().threshold == 0.98


def test_lower_threshold_puts_more_of_the_image_in_ink():
    image = np.full((300, 300), 255, np.uint8)
    image[100:200, 100:200] = 60

    light = line_art(image, params=LineArtParams(threshold=0.75))
    heavy = line_art(image, params=LineArtParams(threshold=0.25))

    assert float((heavy < 128).mean()) >= float((light < 128).mean())


# ------------------------------------------------------------ silhouette

from pcbart.lineart import silhouette_outline  # noqa: E402


def test_silhouette_outline_follows_alpha_not_tone():
    """The outline must come from transparency, independent of colour.

    A light subject on a light background produces no dodge response at its
    boundary, so tone cannot supply this line.
    """
    from pcbart.lineart import silhouette_outline

    alpha = np.zeros((120, 120), np.uint8)
    alpha[30:90, 30:90] = 255  # a square subject

    outline = silhouette_outline(alpha, width=3)

    assert outline[30, 60] > 0, "top edge must be inked"
    assert outline[89, 60] > 0, "bottom edge must be inked"
    assert outline[60, 30] > 0, "left edge must be inked"
    assert outline[60, 89] > 0, "right edge must be inked"
    assert outline[60, 60] == 0, "the interior must stay clear"


def test_silhouette_outline_stays_inside_the_subject():
    """Growing outwards would make a board bigger than the artwork."""
    from pcbart.lineart import silhouette_outline

    alpha = np.zeros((100, 100), np.uint8)
    alpha[40:60, 40:60] = 255

    outline = silhouette_outline(alpha, width=9)

    outside = (alpha == 0) & (outline > 0)
    assert not outside.any(), "no ink may fall outside the subject"


def test_silhouette_outline_of_empty_alpha_is_empty():
    from pcbart.lineart import silhouette_outline

    assert silhouette_outline(np.zeros((50, 50), np.uint8), 5).sum() == 0


def test_line_art_adds_the_outline_for_transparent_input():
    """End to end: a pale subject on a pale background keeps its boundary."""
    image = np.zeros((200, 200, 4), np.uint8)
    image[:, :, :3] = 250          # very light subject
    image[40:160, 40:160, 3] = 255  # opaque square

    without = line_art(image, params=LineArtParams(silhouette=False))
    with_outline = line_art(image, params=LineArtParams(silhouette=True))

    # The boundary row must gain ink.
    assert (with_outline[40, 40:160] < 128).any()
    assert not (without[40, 40:160] < 128).any()
    assert (with_outline < 128).sum() > (without < 128).sum()


def test_silhouette_can_be_switched_off():
    image = np.zeros((160, 160, 4), np.uint8)
    image[:, :, :3] = 240
    image[30:130, 30:130, 3] = 255

    off = line_art(image, params=LineArtParams(silhouette=False, threshold=0.46))
    reference_off = line_art(image, params=LineArtParams(silhouette=False))
    assert np.array_equal(off, reference_off)


def test_the_outline_is_found_without_an_alpha_channel():
    """A JPEG or flattened PNG has no alpha, and must still get its boundary.

    This was a real gap: the option did nothing at all for such pictures, so an
    irregular board traced from a JPEG came out with no outer edge. The subject
    is now inferred by comparing each pixel with the colour of the picture's
    border.
    """
    image = np.full((200, 200, 3), 250, np.uint8)   # light background, no alpha
    image[40:160, 50:150] = 60                       # dark subject

    outline = silhouette_outline(None, width=3, image=image)

    assert outline[40, 100] > 0, "top edge must be inked"
    assert outline[159, 100] > 0, "bottom edge must be inked"
    assert outline[100, 50] > 0, "left edge must be inked"
    assert outline[100, 149] > 0, "right edge must be inked"
    assert outline[100, 100] == 0, "the interior must stay clear"


def test_tone_based_outline_ignores_the_picture_border():
    """The border is background, so nothing is inked along the canvas edge."""
    image = np.full((200, 200, 3), 250, np.uint8)
    image[40:160, 50:150] = 60

    outline = silhouette_outline(None, width=3, image=image)

    assert outline[0, :].sum() == 0, "图片外缘是背景，不应描边"
    assert outline[-1, :].sum() == 0
    assert outline[:, 0].sum() == 0
    assert outline[:, -1].sum() == 0


def test_a_fully_opaque_alpha_is_not_treated_as_a_cut_out():
    """Four channels does not mean the picture has a background.

    An opaque alpha describes nothing. Treating it as a mask handed back the
    whole rectangle as the subject, which produced a rectangular board that
    looked like the outline had worked.
    """
    image = np.full((200, 200, 3), 250, np.uint8)
    image[40:160, 50:150] = 60
    alpha = np.full((200, 200), 255, np.uint8)

    outline = silhouette_outline(alpha, width=3, image=image)

    assert outline[0, :].sum() == 0, "不透明 alpha 不得给出画布边界"
    assert outline[40, 100] > 0, "应退回按颜色推断主体"
    assert outline[100, 100] == 0


def test_real_transparency_still_wins_over_tone():
    """When alpha genuinely cuts out a subject, it is the exact answer."""
    image = np.full((200, 200, 3), 250, np.uint8)
    image[40:160, 50:150] = 60
    alpha = np.zeros((200, 200), np.uint8)
    alpha[10:80, 10:80] = 255       # a cut-out that does not match the dark area

    outline = silhouette_outline(alpha, width=3, image=image)

    assert outline[10, 40] > 0, "应按透明通道而不是深浅描边"


def test_tone_based_outline_keeps_only_the_subject():
    """Specks elsewhere in the frame must not join the board outline."""
    image = np.full((200, 200, 3), 250, np.uint8)
    image[40:160, 50:150] = 60          # the subject
    image[5:10, 5:10] = 0               # a speck in the corner

    outline = silhouette_outline(None, width=3, image=image)

    assert outline[7, 7] == 0, "杂点不应成为板框的一部分"
    assert outline[40, 100] > 0


def test_a_uniform_image_has_no_outline():
    """Nothing stands out from the background, so nothing is invented."""
    image = np.full((120, 120, 3), 200, np.uint8)
    assert silhouette_outline(None, width=3, image=image).sum() == 0


def test_a_tiny_image_does_not_crash():
    image = np.full((2, 2, 3), 128, np.uint8)
    assert silhouette_outline(None, width=3, image=image).size >= 0


def test_silhouette_width_is_forced_odd_and_clamped():
    assert LineArtParams(silhouette_width=8).normalised().silhouette_width == 7
    assert LineArtParams(silhouette_width=0).normalised().silhouette_width == 1
    assert LineArtParams(silhouette_width=999).normalised().silhouette_width == 31


def test_wider_silhouette_adds_more_ink():
    image = np.zeros((200, 200, 4), np.uint8)
    image[:, :, :3] = 250
    image[50:150, 50:150, 3] = 255

    thin = line_art(image, params=LineArtParams(silhouette_width=1))
    thick = line_art(image, params=LineArtParams(silhouette_width=9))

    assert (thick < 128).sum() > (thin < 128).sum()


# ----------------------------------------------------------------- preview


def test_preview_keeps_single_pixel_strokes_visible():
    """Decimating a two-tone image is lossy, so the preview must not be too
    aggressive: a thin stroke has to survive as *something* rather than vanish.

    Averaging keeps a 1 px line visible at reduced size; nearest-neighbour
    drops it. This pins the choice.
    """
    image = np.full((2400, 1600, 3), 255, np.uint8)
    image[600:1800, 400:1200] = 30           # a large shape
    image[100, 100:1500] = 30                # a 1 px horizontal stroke

    preview = line_art_preview(image, max_side=400)

    # Some ink must survive along the row the thin stroke occupied.
    row = int(100 * preview.shape[0] / image.shape[0])
    band = preview[max(0, row - 2) : row + 3, :]
    assert (band < 220).any(), "the thin stroke must remain visible in preview"


def test_preview_downscales_a_large_image():
    big = np.full((2400, 1600, 3), 255, np.uint8)
    big[600:1800, 400:1200] = 30
    preview = line_art_preview(big, max_side=400)
    assert min(preview.shape[:2]) == 400


def test_preview_leaves_a_small_image_at_full_size():
    small = np.full((200, 200), 255, np.uint8)
    preview = line_art_preview(small, max_side=400)
    assert preview.shape == (200, 200)


def test_preview_and_full_output_agree_on_a_dark_shape():
    """The radius rule is scale-invariant, so a preview must match the result."""
    image = np.full((900, 600, 3), 255, np.uint8)
    image[200:700, 150:450] = 35

    full = line_art(image)
    preview = line_art_preview(image, max_side=300)

    full_ink = float((full < 128).mean())
    preview_ink = float((preview < 128).mean())
    assert abs(full_ink - preview_ink) < 0.05


# ------------------------------------------- filling from outside the picture


def test_a_rule_across_the_top_does_not_swallow_the_picture():
    """The fill must start *outside* the picture, not at its top-left corner.

    This was a real bug with a very misleading symptom. ``_fill_enclosed_holes``
    flooded from pixel ``(0, 0)``. A drawing with a horizontal rule near the top
    traps that corner, so everything below the rule was unreachable and counted
    as "enclosed" -- the subject became the whole canvas and the board outline
    came out as a plain rectangle. The right edge looked like a straight line
    because it *was* one.
    """
    from pcbart.image_prep import _fill_enclosed_holes

    mask = np.zeros((100, 100), np.uint8)
    mask[20, :] = 1          # a rule spanning the full width
    mask[40:60, 40:60] = 1   # the actual subject

    filled = _fill_enclosed_holes(mask)

    assert filled[5, 5] == 0, "横线以上的背景被误判为封闭"
    assert filled[80, 5] == 0, "横线以下的背景被误判为封闭"
    assert filled[50, 50] == 1, "主体应当保留"
    assert filled.mean() < 0.5, "整幅图不应变成主体"


def test_filling_still_works_when_the_subject_is_closed():
    """Flooding from outside must not stop the useful case from working."""
    from pcbart.image_prep import _fill_enclosed_holes

    mask = np.zeros((60, 60), np.uint8)
    mask[10, 10:50] = 1
    mask[49, 10:50] = 1
    mask[10:50, 10] = 1
    mask[10:50, 49] = 1

    filled = _fill_enclosed_holes(mask)

    assert filled[30, 30] == 1, "闭合轮廓内部应被填充"
    assert filled[2, 2] == 0, "外部背景不应被填充"


def test_the_subject_mask_of_a_framed_drawing_is_not_the_whole_canvas():
    """End to end on the shape of picture that exposed it."""
    from pcbart.image_prep import subject_mask

    image = np.full((300, 240, 3), 250, np.uint8)
    image[20, :] = 200          # a pale rule across the top
    image[24, :] = 200
    image[100:280, 60:180] = 40  # the subject

    mask = subject_mask(image, None, fill=True)

    assert mask is not None
    assert mask.mean() < 0.8, "整幅图被当成了主体"
    ys, xs = np.where(mask > 0)
    assert ys.min() > 40, "轮廓应贴合主体而不是从画框开始"
    assert xs.min() > 20


# ------------------------------------------------- separate things in a picture


def _poster() -> np.ndarray:
    """A character, three letters and a mug beside it, on white.

    Every piece has its own closed outline. This is the shape of picture that
    exposed the connecting-line defect: a drawing holding several *separate*
    things, which the outline pass used to weld into one blob.
    """
    image = np.full((520, 520, 3), 255, np.uint8)
    cv2.ellipse(image, (240, 320), (120, 110), 0, 0, 360, (0, 0, 0), 5)
    cv2.circle(image, (240, 320), 30, (0, 0, 0), 3)
    for index, letter in enumerate("ABC"):
        cv2.putText(
            image,
            letter,
            (120 + index * 90, 90),
            cv2.FONT_HERSHEY_DUPLEX,
            2.2,
            (0, 0, 0),
            5,
        )
    cv2.circle(image, (470, 300), 30, (0, 0, 0), 4)
    return image


def test_every_separate_thing_in_a_picture_keeps_its_own_outline():
    """Not just the biggest piece: the mug and the letters are subjects too.

    Keeping only the largest component threw them away; welding them together
    first is what drew lines across the empty background between them. Both
    failures look like "the outline is wrong", so the count is asserted rather
    than just the coverage.
    """
    from pcbart.image_prep import subject_mask

    mask = subject_mask(_poster(), None, fill=True)
    assert mask is not None

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    assert count - 1 == 5, (
        f"应保留 5 个独立元素（主体 + 三个字母 + 杯子），实际 {count - 1}"
    )

    # 每个元素都要落在它该在的位置，而不是被并成一块
    centres = sorted(tuple(np.round(centroids[i]).astype(int)) for i in range(1, count))
    assert any(abs(y - 67) < 15 for _x, y in centres), "文字那一行丢了"
    assert any(x > 430 for x, _y in centres), "旁边的杯子丢了"


def test_the_outline_does_not_cross_the_empty_space_between_elements():
    """The reported defect, measured directly.

    Bridging everything into one blob meant the traced boundary ran through
    the background between the character and the mug. Sampling the gaps is what
    catches that; the element count above would still be right if the pieces
    were joined by a one-pixel thread.
    """
    from pcbart.image_prep import subject_mask

    mask = np.asarray(subject_mask(_poster(), None, fill=True))

    # 主体右缘约在 x=360，杯子左缘约在 x=440；中间这条竖带是背景。
    band = mask[:, 390:425]
    assert band.mean() < 0.05, (
        f"主体与杯子之间被填上了（该带前景 {100 * band.mean():.1f}%）"
    )


def _dashed_ring(size: int = 420, radius: int = 110) -> np.ndarray:
    """A circle whose outline is broken into visibly separate strokes.

    This is the failure mode of hand-drawn line art: not one wide gap, but
    many strokes that stop just short of each other. The fill leaks out
    through every one of them unless they are bridged.
    """
    image = np.full((size, size, 3), 255, np.uint8)
    segments = 8
    gap_fraction = 0.04
    step = 2 * np.pi / segments
    for index in range(segments):
        first = index * step + step * gap_fraction
        last = (index + 1) * step - step * gap_fraction
        points = np.array(
            [
                [
                    int(size / 2 + radius * np.cos(t)),
                    int(size / 2 + radius * np.sin(t)),
                ]
                for t in np.linspace(first, last, 40)
            ],
            np.int32,
        )
        cv2.polylines(image, [points], False, (0, 0, 0), 3)
    return image


def test_hand_drawn_outlines_still_get_bridged():
    """The other half of the bargain: a drawing with real gaps must not regress.

    The adaptive kernel exists to tell these two cases apart. If it ever picks
    the narrow one here, the fill leaks out through the stroke breaks and the
    subject collapses to just its strokes.
    """
    from pcbart.image_prep import _adaptive_gap, _base_subject_mask, subject_mask

    image = _dashed_ring()
    base = _base_subject_mask(image, None).astype(np.uint8)
    short = min(base.shape)
    wide = max(3, int(short * 0.045)) | 1
    assert _adaptive_gap(base) == wide, "有断笔的线稿应当选用宽核"

    mask = subject_mask(image, None, fill=True)
    assert mask is not None
    assert mask.mean() > 0.15, "断笔没被桥接，填充从缺口漏了出去"


def test_a_picture_whose_outlines_already_close_uses_the_narrow_kernel():
    """The counterpart to the test above, so the cut between them is pinned."""
    from pcbart.image_prep import _adaptive_gap, _base_subject_mask

    base = _base_subject_mask(_poster(), None).astype(np.uint8)
    short = min(base.shape)
    narrow = max(3, int(short * 0.006)) | 1
    assert _adaptive_gap(base) == narrow


def test_line_art_does_not_draw_the_gaps_closed():
    """End to end on the defect the user reported: "连接件" between elements.

    The silhouette pass runs off ``subject_mask``, so once that welded the
    picture's separate pieces together the outline was traced across the empty
    background -- a line the source never had. Asserting on the mask alone
    would miss it, so this checks the picture that actually gets drawn.
    """
    from pcbart.image_prep import subject_mask

    image = _poster()
    mask = np.asarray(subject_mask(image, None, fill=False)).astype(np.uint8)

    # 把掩膜画成轮廓（与 silhouette_outline 同样的做法）
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    outline = (edge & mask).astype(np.uint8)

    # 主体与杯子之间的竖带里，除了各自的边，不该有横向的连接线
    band = outline[:, 390:425]
    assert band.mean() < 0.05, (
        f"主体和杯子之间被画上了连接线（该带 {100 * band.mean():.1f}% 有墨）"
    )


def test_smoothing_does_not_weld_the_pieces_back_together():
    """The step that undid the rest of the work, caught directly.

    Smoothing runs after the pieces have been separated, and the closing half
    of open-then-close bridges anything closer than the kernel. Applied to the
    combined mask it re-merged the character and the mug -- 289k and 35k pixels
    became one 343k blob -- and the outline then crossed the space between
    them again. Per-piece smoothing cannot merge anything, so the count is the
    assertion.
    """
    from pcbart.image_prep import _outer_components, _smooth_each_component

    mask = np.zeros((240, 400), np.uint8)
    cv2.rectangle(mask, (20, 60), (160, 200), 1, -1)     # 主体
    cv2.rectangle(mask, (190, 90), (260, 170), 1, -1)    # 旁边的小元素，相距 30px

    before = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[0] - 1
    assert before == 2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    smoothed = _smooth_each_component(_outer_components(mask), kernel)

    after = cv2.connectedComponentsWithStats(smoothed.astype(np.uint8), 8)[0] - 1
    assert after == 2, f"平滑后两个元素被焊在一起了（{before} -> {after}）"
    assert smoothed[130, 175] == 0, "两个元素之间的背景被填上了"
