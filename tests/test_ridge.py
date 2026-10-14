"""The ridge reading's filter, against scikit-image and on its own terms.

The 脊线线稿 reading is ``skimage.filters.meijering`` reimplemented on numpy +
OpenCV (``pcbart.ridge``) so the tool does not have to carry scipy, networkx,
imageio and tifffile -- about 150 MB unpacked -- for one reading. That trade is
only acceptable while the reimplementation *is* the same filter, so the central
test here pins the response against scikit-image itself whenever scikit-image is
installed, and the rest pin the properties that must hold even where it is not.

⚠️ **Response, not mask.** The thresholding step is a percentile over the
picture, and the response is normalised per scale -- so on synthetic images
with flat plateaus a percentile lands inside a tie and flips wholesale on a
rounding-sized difference. That is a property of percentile thresholding, not
of this port (two builds of scikit-image disagree there too), so the
comparison is on the response values, with the tolerance the port achieves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.ridge import MAX_SCALES, meijering_response  # noqa: E402


def _noise(shape=(120, 160), seed=7):
    rng = np.random.default_rng(seed)
    return (rng.random(shape).astype(np.float32) * 0.9 + 0.05)


def _lines(shape=(200, 260)):
    """A picture with obvious dark strands and a bright one."""
    image = np.ones(shape, dtype=np.float32)
    image[40:48, 20:240] = 0.15
    image[90:96, 20:240] = 0.4
    image[150:156, 20:240] = 0.95
    return image


def test_the_response_matches_scikit_image():
    """The port must be the *same filter*, not a lookalike.

    Measured on both inputs the largest disagreement is ~3e-7 (float32
    rounding); the tolerance below leaves room for a different BLAS but not for
    a different formula. If this fails, the port drifted -- check the three
    details listed in ``pcbart.ridge``'s docstring before touching the
    tolerance.
    """
    skimage_filters = pytest.importorskip(
        "skimage.filters", reason="需要 scikit-image 作对照"
    )
    sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
    for picture in (_noise(), _lines()):
        reference = skimage_filters.meijering(picture, sigmas=sigmas, black_ridges=True)
        mine = meijering_response(picture, sigmas, black_ridges=True)
        difference = float(np.abs(reference - mine).max())
        assert difference < 1e-5, f"与 scikit-image 的最大差 {difference}"


def test_a_flat_picture_has_no_response():
    for level in (0.0, 0.5, 1.0):
        flat = np.full((64, 64), level, dtype=np.float32)
        response = meijering_response(flat, [1.0, 2.0])
        assert response.max() == 0.0, f"纯色 {level} 不该有脊线响应"


def test_a_dark_strand_responds_and_a_bright_one_does_not():
    """``black_ridges`` selects dark strands -- the polarity the reading uses.

    Two pictures, each with one strand: dark-on-light and bright-on-dark. The
    filter must respond to the strand only under its own polarity (measured
    0.311 against 0.000 in both directions), and the flat paper must stay at
    zero either way.
    """
    dark_on_light = np.ones((200, 260), dtype=np.float32)
    dark_on_light[60:70, 20:240] = 0.1
    bright_on_dark = np.full((200, 260), 0.1, dtype=np.float32)
    bright_on_dark[60:70, 20:240] = 1.0

    def response_at(picture, black):
        return float(meijering_response(picture, [2.0], black_ridges=black)[65, 120])

    assert response_at(dark_on_light, True) > 0.2
    assert response_at(dark_on_light, False) < 1e-6
    assert response_at(bright_on_dark, True) < 1e-6
    assert response_at(bright_on_dark, False) > 0.2
    # Paper: flat either polarity, either picture.
    for picture in (dark_on_light, bright_on_dark):
        for black in (True, False):
            flat = meijering_response(picture, [2.0], black_ridges=black)[:20, :20]
            assert flat.max() == 0.0


def test_the_response_is_normalised_to_one():
    """Each scale is normalised to its own maximum, so the answer is 0..1.

    The panel's percentile threshold reads this directly: a response that
    drifted outside 0..1 would move the meaning of "90th percentile" with it.
    """
    response = meijering_response(_lines(), [1.0, 2.0, 3.0])
    assert 0.0 <= response.min()
    assert response.max() == pytest.approx(1.0)


def test_the_scale_list_is_capped():
    """A slider cannot make the filter run an unbounded number of Hessians.

    ``MAX_SCALES`` bounds the cost; the scales past the cap are ignored rather
    than raising, because a caller mid-drag is allowed to ask for too many.
    """
    picture = _lines()
    capped = meijering_response(picture, [1.0] * (MAX_SCALES + 3))
    assert capped.max() > 0
    with_more = meijering_response(picture, [1.0] * (MAX_SCALES + 30))
    assert np.array_equal(capped, with_more)


def test_a_single_scale_and_a_bad_scale_are_both_survivable():
    picture = _lines()
    assert meijering_response(picture, [2.0]).max() > 0
    assert meijering_response(picture, [0.0, -1.0]).max() == 0.0
    assert meijering_response(picture, []).max() == 0.0


def test_a_colour_picture_is_refused():
    with pytest.raises(ValueError):
        meijering_response(np.zeros((8, 8, 3), dtype=np.float32), [1.0])
