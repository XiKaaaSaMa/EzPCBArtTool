"""Tests for the working-resolution budget, and for what it must not break.

A large import used to freeze the window for tens of seconds. The fix reduces
the picture before any processing step runs, which is only safe if two things
hold, and these tests are those two things:

* **small pictures are left completely alone** -- reducing one, or upscaling
  one, would change the filter radius the algorithm derives from the short side
  and produce thicker strokes than the drawing has;
* **pixel-measured settings survive the reduction** -- "去杂点 5 px" and
  "轮廓宽度 5 px" mean five pixels *of the picture the user loaded*, so after a
  reduction the same number must become a proportionally smaller count. Applied
  unchanged it would filter a different amount of ink than the panel says.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pcbart.work_scale import (  # noqa: E402
    MAX_WORK_LONG_SIDE,
    MAX_WORK_PIXELS,
    needs_reduction,
    prepare_working_image,
    scaled_param,
    work_scale,
    working_size,
)


# ------------------------------------------------------------- the budget


def test_small_pictures_are_not_touched():
    """The common case: a drawing around 1000 px goes straight through."""
    for size in ((960, 1619), (800, 800), (1000, 1000), (2000, 1000)):
        assert not needs_reduction(*size), size
        assert work_scale(*size) == 1.0
        assert working_size(*size) == size


def test_a_phone_photo_is_reduced():
    """4000 px is what a phone camera produces, and what froze the window."""
    assert needs_reduction(4000, 3000)
    assert work_scale(4000, 3000) < 1.0


def test_the_long_side_is_bounded():
    width, height = working_size(12000, 6000)

    assert max(width, height) <= MAX_WORK_LONG_SIDE


def test_a_long_panorama_is_bounded_too():
    """The long-side rule alone can be defeated by a wide picture.

    A 2000 x 6000 strip passes a long-side test at 12 megapixels, so there is
    a pixel ceiling as well.
    """
    assert needs_reduction(2000, 6000)
    assert work_scale(2000, 6000) < 1.0

    width, height = working_size(2000, 6000)
    assert width * height <= MAX_WORK_PIXELS


def test_the_aspect_ratio_is_preserved():
    for size in ((8000, 4000), (4000, 8000), (6000, 6000), (12000, 3000)):
        width, height = working_size(*size)
        assert width / height == pytest.approx(size[0] / size[1], rel=0.01), size


def test_degenerate_sizes_do_not_crash():
    for size in ((0, 0), (1, 1), (0, 100)):
        assert needs_reduction(*size) is False
        assert work_scale(*size) == 1.0


def test_the_scale_never_exceeds_one():
    """Upscaling would add no information and would change the radius rule."""
    for size in ((10, 10), (100, 100), (5000, 5000)):
        assert work_scale(*size) <= 1.0


# -------------------------------------------------------- carrying settings


def test_a_setting_is_unchanged_when_nothing_was_reduced():
    """Callers use this unconditionally, so scale 1.0 must be a no-op."""
    assert scaled_param(5, 1.0) == 5
    assert scaled_param(5, 1.5) == 5


def test_a_pixel_setting_scales_with_the_picture():
    """5 px of an 8000 px picture is 1.25 px of a 2000 px one."""
    scale = work_scale(8000, 8000)

    assert scaled_param(5, scale) == pytest.approx(5 * scale)


def test_a_setting_never_scales_away_to_nothing():
    """A filter of zero pixels is not a filter; the floor keeps it at one."""
    assert scaled_param(5, 0.01, minimum=1.0) == 1.0


def test_a_setting_that_may_legitimately_be_zero_stays_zero():
    """去杂点 defaults to off, and 0 must survive the reduction as off."""
    assert scaled_param(0, 0.25, minimum=0.0) == 0.0


# -------------------------------------------------------- the image itself


def test_preparing_a_small_image_returns_it_unchanged():
    image = np.zeros((400, 600, 3), np.uint8)
    alpha = np.zeros((400, 600), np.uint8)

    out, out_alpha, scale = prepare_working_image(image, alpha)

    assert out is image, "小图不应被复制或重采样"
    assert out_alpha is alpha
    assert scale == 1.0


def test_preparing_a_large_image_reduces_both_planes():
    image = np.full((8000, 4000, 3), 128, np.uint8)
    alpha = np.full((8000, 4000), 255, np.uint8)

    out, out_alpha, scale = prepare_working_image(image, alpha)

    assert scale < 1.0
    assert out.shape[1] == out_alpha.shape[1]
    assert out.shape[0] == out_alpha.shape[0]
    assert max(out.shape[:2]) <= MAX_WORK_LONG_SIDE


def test_the_reduced_image_keeps_its_content():
    """A picture reduced for speed must still be the same picture."""
    image = np.full((8000, 8000, 3), 255, np.uint8)
    image[2000:6000, 2000:6000] = 0

    out, _alpha, _scale = prepare_working_image(image)

    # The dark square covered the middle half, and still does.
    middle = out[out.shape[0] // 2, out.shape[1] // 2]
    corner = out[2, 2]
    assert int(middle[0]) < 32, "中间的深色块应还在"
    assert int(corner[0]) > 224, "四角应仍是白底"


# ------------------------------------------- what the reduction must not change


def test_a_large_import_produces_the_same_mask_as_a_full_resolution_one():
    """The board outline is the thing this had to not break.

    Two pictures, one already at the working size and one four times bigger but
    otherwise identical. After processing, their subject masks must agree --
    the mask is what the board is cut from, so a difference here is a different
    board.
    """
    small = np.full((1500, 1500), 255, np.uint8)
    small[400:1100, 400:1100] = 0

    big = np.full((6000, 6000), 255, np.uint8)
    big[1600:4400, 1600:4400] = 0

    from pcbart.image_prep import subject_mask
    from pcbart.work_scale import prepare_working_image

    reference = subject_mask(small, None, fill=True)
    reduced, _alpha, _scale = prepare_working_image(big, None)
    large = subject_mask(reduced, None, fill=True)

    # Same footprint, so the same fraction of the picture is subject.
    assert large.shape[:2] != reference.shape[:2], "这张图确实被降采样了"
    assert large.mean() == pytest.approx(reference.mean(), abs=0.01)


def test_a_large_import_agrees_with_a_full_trace_at_display_size():
    """The preview must show the shape the import will commit.

    Traced at the working resolution, both, so what this pins is that the
    reduction is applied consistently -- one path reducing and the other not
    would make the previewed shape differ from the imported one.
    """
    drawing = np.full((2400, 1600), 255, np.uint8)
    drawing[400:2000, 300:1300] = 0
    drawing[800:1000, 700:900] = 255  # a hole, so the shape is not a blob

    from pcbart.lineart import LineArtParams, line_art
    from pcbart.work_scale import prepare_working_image

    # What the import path does today.
    import_bgr = np.repeat(drawing[:, :, None], 3, axis=2)
    working, alpha, scale = prepare_working_image(import_bgr, None)
    committed = line_art(working, alpha, LineArtParams())

    # And what the full-resolution trace would have produced, reduced for
    # comparison the way the preview window reduces it.
    import cv2

    full = line_art(import_bgr, None, LineArtParams())
    target = (committed.shape[1], committed.shape[0])
    full_reduced = cv2.resize(full, target, interpolation=cv2.INTER_AREA)

    a = committed < 128
    b = full_reduced < 128
    iou = (a & b).sum() / max(1, (a | b).sum())

    assert iou > 0.90, f"降采样后的描线与全分辨率差太多 (IoU={iou:.3f})"


# ------------------------------------------------- the frozen-window fixes


def test_the_frame_scan_does_not_walk_whole_columns():
    """A pixel-by-pixel walk over a filled mask was 42 s of a 44 s import.

    The scan only asks "is this line thinner than a few pixels", so it must
    stop walking once the answer is clear. Measured on a column of solid ink
    thousands of pixels long, the old code walked the whole run at every
    sampled position.
    """
    import time

    from pcbart.image_prep import _line_thickness

    filled = np.ones((6000, 400), np.uint8)

    start = time.perf_counter()
    thickness = _line_thickness(filled, 3000, axis=0)
    elapsed = time.perf_counter() - start

    assert thickness > 3, "实心列的厚度应判定为厚"
    assert elapsed < 1.0, f"厚度检测不该走完整列（{elapsed:.2f}s）"


def test_a_large_mask_is_computed_at_a_bounded_resolution():
    """The mask comes back at the size it went in.

    Callers compare it against images they already hold, and
    ``silhouette_outline`` silently skips the outline when the shapes disagree
    -- so returning a smaller mask would drop the subject's boundary rather
    than failing loudly.
    """
    from pcbart.image_prep import subject_mask

    large = np.full((5000, 3000, 3), 255, np.uint8)
    large[1000:4000, 700:2300] = 0

    mask = subject_mask(large, None, fill=True)

    assert mask is not None
    assert mask.shape[:2] == (5000, 3000), "掩膜必须与输入同尺寸"
    assert mask.dtype == np.uint8
    assert mask.mean() == pytest.approx(0.36, abs=0.05), "主体仍占画面中间那块"
