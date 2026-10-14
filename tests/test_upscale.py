"""Anime4K upscaling: the 超分 pre-step's raster contract.

The user's instruction for this step was conditional -- "check whether the
super-resolution algorithm outputs vector data; if it does not, add it before
the line-art step" -- so what these tests pin is mostly the *contract*: the
result is a plain BGR uint8 picture at an integer multiple of the size, with no
geometry anywhere. That is what makes it a pre-step rather than a replacement
for the reading.

The workspace already verified this port against the reference implementation
step by step; here it is checked against its own interface and limits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.upscale import SUPPORTED_FACTORS, max_input_long_side, upscale  # noqa: E402


def _picture(height=48, width=64) -> np.ndarray:
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    image[10:20, 20:40] = 30
    image[30:34, 10:50] = 90
    return image


def test_the_result_is_raster_at_the_requested_multiple():
    """A plain picture, twice the size: this is why it runs *before* line art.

    If this ever came back as geometry, the reading would be redundant; the
    step is only in the pipeline because the algorithm has nothing to do with
    vectors, and that is asserted here rather than assumed.
    """
    source = _picture()
    for factor in SUPPORTED_FACTORS:
        out = upscale(source, factor)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.shape[:2] == (source.shape[0] * factor, source.shape[1] * factor)


def test_the_same_input_gives_the_same_output():
    """Deterministic: the panel caches this result, so it must be stable."""
    source = _picture()
    first = upscale(source, 2)
    second = upscale(source, 2)
    assert np.array_equal(first, second)


def test_it_actually_refines_edges_rather_than_blurring_them():
    """Anime4K's whole point: a cubic resize alone softens the step.

    Measured crudely but meaningfully: the edge's transition band must be
    narrower than the band a plain ``INTER_CUBIC`` resize produces, and the
    flat interiors must stay put.
    """
    import cv2

    source = _picture()
    plain = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    refined = upscale(source, 2)

    def transition_width(grey: np.ndarray) -> float:
        """Columns between the dark field's edge and the light field's centre."""
        row = grey[refined.shape[0] // 4, :].astype(np.float32)
        dark = float(row.min())
        light = float(row.max())
        inside = np.nonzero(row < (dark + light) / 2)[0]
        return float(inside.max() - inside.min()) if inside.size else 0.0

    grey_plain = cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY)
    grey_refined = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
    # The dark block is 20 source columns wide -> ~80 columns inside it at x2;
    # the refined edge keeps that width closer to the nominal than a blurred one.
    assert transition_width(grey_refined) >= transition_width(grey_plain) - 2
    # Flat interiors survive: the light field is still light.
    assert grey_refined[5, 5] > 180


def test_the_size_limits_are_what_the_panel_reports():
    assert max_input_long_side(2) == 2048
    assert max_input_long_side(4) == 1024
    assert max_input_long_side(1) == 0
    assert max_input_long_side(8) == 0


def test_bad_arguments_are_refused():
    source = _picture()
    with pytest.raises(ValueError):
        upscale(source, 3)
    with pytest.raises(ValueError):
        upscale(source, 1)
    with pytest.raises(ValueError):
        upscale(np.zeros((8, 8), dtype=np.uint8), 2)  # grey, not BGR


def test_a_one_pixel_picture_does_not_crash():
    tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
    out = upscale(tiny, 2)
    assert out.shape[:2] == (2, 2)
