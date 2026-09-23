"""Tests for the denoising steps.

These assert the *effect* of each step -- fewer specks, holes closed -- rather
than merely that the function returns an array. A denoiser that does nothing
would pass a shape check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.denoise import (  # noqa: E402
    MAX_KERNEL,
    MIN_KERNEL,
    blur_sigma_for_kernel,
    clean_binary,
    denoise_image,
    gaussian_prefilter,
    remove_pinholes,
    remove_specks,
)
from pcbart.lineart import LineArtParams, line_art  # noqa: E402


# ------------------------------------------------------------- the prefilter


def test_prefilter_reduces_high_frequency_noise():
    """The point of the prefilter: noise amplitude drops."""
    rng = np.random.default_rng(5)
    clean = np.full((200, 200), 200, np.uint8)
    noisy = np.clip(
        clean.astype(np.float32) + rng.normal(0, 25, clean.shape), 0, 255
    ).astype(np.uint8)

    before = float(np.std(noisy.astype(np.float32) - clean.astype(np.float32)))
    filtered = gaussian_prefilter(noisy, kernel_size=7)
    after = float(np.std(filtered.astype(np.float32) - clean.astype(np.float32)))

    assert after < before * 0.6, f"noise std should drop sharply: {before} -> {after}"


def test_prefilter_preserves_a_large_edge():
    """Blurring must not erase the structure we are trying to trace."""
    image = np.zeros((200, 200), np.uint8)
    image[:, 100:] = 255

    filtered = gaussian_prefilter(image, kernel_size=5)

    assert filtered[100, 20] < 40, "the dark side must stay dark"
    assert filtered[100, 180] > 215, "the light side must stay light"
    # The transition stays in roughly the same place.
    middle = filtered[100, 90:110]
    assert middle.min() < 128 < middle.max()


def test_kernel_size_is_forced_odd_and_clamped():
    flat = np.full((50, 50), 128, np.uint8)
    # An even kernel must not silently shift the image.
    even = gaussian_prefilter(flat, kernel_size=6)
    assert even.shape == flat.shape
    assert abs(float(even.mean()) - 128.0) < 1.0

    tiny = gaussian_prefilter(flat, kernel_size=1)
    assert tiny.shape == flat.shape

    huge = gaussian_prefilter(flat, kernel_size=999)
    assert huge.shape == flat.shape


def test_prefilter_preserves_uniform_colour():
    """A blur must not change a flat region, or it would shift the threshold."""
    flat = np.full((60, 60), 137, np.uint8)
    filtered = gaussian_prefilter(flat, kernel_size=9)
    assert abs(float(filtered.mean()) - 137.0) < 0.5


def test_prefilter_accepts_colour_and_alpha():
    image = np.full((40, 40, 3), 100, np.uint8)
    assert gaussian_prefilter(image).shape == image.shape

    rgba = np.full((40, 40, 4), 100, np.uint8)
    assert gaussian_prefilter(rgba).shape == rgba.shape


def test_sigma_default_grows_with_the_kernel():
    assert blur_sigma_for_kernel(3) < blur_sigma_for_kernel(15)
    # The original's pairing of kernel 5 with sigma 1.1 should be reproduced.
    assert blur_sigma_for_kernel(5) == pytest.approx(1.1, abs=0.15)


# ---------------------------------------------------------------- specks


def test_remove_specks_drops_isolated_pixels():
    binary = np.full((100, 100), 255, np.uint8)
    binary[50, 50] = 0          # a lone pixel
    binary[10:13, 10:13] = 0    # a small 3x3 blob
    binary[70:90, 70:74] = 0    # a real stroke

    cleaned = remove_specks(binary, min_area=10)

    assert cleaned[50, 50] == 255, "the lone pixel must go"
    assert (cleaned[70:90, 70:74] == 0).all(), "the stroke must stay"


def test_remove_specks_keeps_a_thin_but_long_line():
    """Strokes are long, not small; area is the wrong filter for them."""
    binary = np.full((100, 100), 255, np.uint8)
    binary[50, 10:90] = 0  # 1 px tall, 80 px long = area 80

    cleaned = remove_specks(binary, min_area=30)

    assert (cleaned[50, 10:90] == 0).all()


def test_remove_specks_can_flip_polarity():
    binary = np.zeros((60, 60), np.uint8)
    binary[30, 30] = 255
    binary[10:40, 10:14] = 255

    cleaned = remove_specks(binary, min_area=10, foreground_is_dark=False)

    assert cleaned[30, 30] == 0
    assert (cleaned[10:40, 10:14] == 255).all()


def test_remove_specks_with_zero_area_is_a_no_op():
    binary = np.full((40, 40), 255, np.uint8)
    binary[20, 20] = 0
    assert np.array_equal(remove_specks(binary, min_area=0), binary)


# ---------------------------------------------------------------- pinholes


def test_remove_pinholes_fills_holes_inside_ink():
    binary = np.full((100, 100), 255, np.uint8)
    binary[30:70, 30:70] = 0      # a solid black square
    binary[45:48, 45:48] = 255    # a 3x3 pinhole inside it

    filled = remove_pinholes(binary, max_area=20)

    assert (filled[45:48, 45:48] == 0).all(), "the pinhole must be closed"


def test_remove_pinholes_leaves_open_background_alone():
    """Background connected to the border is not a hole."""
    binary = np.full((100, 100), 255, np.uint8)
    binary[30:70, 30:70] = 0
    binary[0:10, 0:10] = 255  # touches the corner

    filled = remove_pinholes(binary, max_area=500)

    assert (filled[0:10, 0:10] == 255).all()


def test_remove_pinholes_keeps_holes_above_the_threshold():
    binary = np.full((120, 120), 255, np.uint8)
    binary[20:100, 20:100] = 0
    binary[50:70, 50:70] = 255  # a 20x20 hole, area 400

    filled = remove_pinholes(binary, max_area=50)

    assert (filled[50:70, 50:70] == 255).all(), "a large hole is a feature, not noise"


def test_remove_pinholes_handles_an_all_black_image():
    binary = np.zeros((50, 50), np.uint8)
    assert np.array_equal(remove_pinholes(binary, max_area=10), binary)


# ------------------------------------------------------ integration in lineart


def _noisy_line_source(seed: int = 3) -> np.ndarray:
    """A white field with one black bar, plus speckle noise."""
    image = np.full((400, 400, 3), 245, np.uint8)
    image[180:220, 40:360] = 20
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 90, image.shape[:2])
    noisy = image.copy()
    speckle = noise > 70
    noisy[speckle] = 15
    return noisy


def test_denoise_step_reduces_specks_before_tracing():
    """The prefilter's measurable effect, applied as its own step.

    Denoising is deliberately separate from line extraction, so this drives the
    two explicitly: blur the source, then trace it. Folding the blur into
    line_art() made the two impossible to tune or judge apart.
    """
    noisy = _noisy_line_source()

    plain = line_art(noisy)
    denoised = line_art(denoise_image(noisy, enabled=True, kernel_size=5))

    def small_components(image: np.ndarray) -> int:
        count, _labels, stats, _c = cv2.connectedComponentsWithStats(
            (image < 128).astype(np.uint8), 8
        )
        return sum(
            1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < 6
        )

    assert small_components(denoised) < small_components(plain) * 0.5


def test_clean_binary_removes_residual_noise_after_tracing():
    """Cleanup is the third, separate step, applied to the traced result."""
    noisy = _noisy_line_source()

    traced = line_art(denoise_image(noisy, enabled=True, kernel_size=5))
    cleaned = clean_binary(traced, min_blob_area=12)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (cleaned < 128).astype(np.uint8), 8
    )
    tiny = sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < 6)
    assert tiny == 0, "no blobs below 6 px should survive cleanup"


def test_pipeline_stays_binary_through_every_step():
    noisy = _noisy_line_source()
    blurred = denoise_image(noisy, enabled=True, kernel_size=5)
    traced = line_art(blurred)
    cleaned = clean_binary(traced, min_blob_area=8, max_hole_area=6)

    for image in (traced, cleaned):
        assert set(np.unique(image).tolist()) <= {0, 255}


def test_denoise_image_disabled_returns_the_input_untouched():
    """Callers should not have to branch around the step to skip it."""
    noisy = _noisy_line_source()
    assert np.array_equal(denoise_image(noisy, enabled=False), noisy)


def test_clean_binary_with_zero_areas_is_a_no_op():
    """Cleanup must be opt-in; it is off by default for a reason."""
    noisy = _noisy_line_source()
    traced = line_art(noisy)
    assert np.array_equal(clean_binary(traced), traced)


# ------------------------------------------------ the two steps stay separate


def test_line_art_ignores_the_deprecated_denoise_flags():
    """The separation, stated as a test.

    These fields used to change the traced output. They no longer do anything,
    which is what makes it possible to judge line extraction on its own.
    """
    image = _noisy_line_source()

    plain = line_art(image)
    legacy = line_art(
        image,
        params=LineArtParams(denoise=True, denoise_kernel=21, despeckle_area=40),
    )

    assert np.array_equal(plain, legacy)


def test_normalising_params_clears_the_deprecated_fields():
    params = LineArtParams(denoise=True, denoise_kernel=17, despeckle_area=99).normalised()
    assert params.denoise is False
    assert params.despeckle_area == 0


def test_changing_the_denoise_step_does_not_change_line_art_itself():
    """A change on one side of the split must not leak into the other."""
    image = _noisy_line_source()

    first = line_art(denoise_image(image, enabled=True, kernel_size=3))
    second = line_art(denoise_image(image, enabled=True, kernel_size=9))

    # The blurred inputs differ, so the traced results should too -- but that
    # difference comes from this step alone, not from a hidden coupling.
    assert not np.array_equal(first, second)
    # And tracing the same input twice is deterministic.
    assert np.array_equal(first, line_art(denoise_image(image, True, 3)))

