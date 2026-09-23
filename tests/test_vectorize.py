"""Tests for turning a line-art mask into copper geometry.

Two things here are load-bearing and were both learned the hard way against
the editor: the **polarity** (ink, not paper, becomes copper) and the
**winding** (a hole must run opposite to its outer ring, or it prints solid).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.geometry import signed_area  # noqa: E402
from pcbart.vectorize import (  # noqa: E402
    CopperStyle,
    VectorizeParams,
    mask_to_complex_polygons,
    vectorize_mask,
)


def make_mask(shape=(200, 200)) -> np.ndarray:
    """Line art in the pipeline's convention: 0 is ink, 255 is paper."""
    mask = np.full(shape, 255, np.uint8)
    cv2.rectangle(mask, (40, 40), (160, 160), 0, -1)
    cv2.rectangle(mask, (80, 80), (120, 120), 255, -1)  # a hole
    return mask


# ---------------------------------------------------------------- polarity


def test_ink_becomes_copper_not_paper():
    """The mask is 0-for-ink; getting this backwards fills the whole page.

    The first real run did exactly that: copper covered the paper and the
    drawing appeared as holes in it.
    """
    mask = make_mask()

    polygons = mask_to_complex_polygons(mask, px_per_mm=10.0)

    assert len(polygons) == 1, "图中只有一个墨块"
    outer = polygons[0][0]
    area = abs(signed_area(outer))
    # The square is 120x120 px at 10 px/mm, so 12x12 mm = 144 mm^2.
    assert area == pytest.approx(144.0, rel=0.05)


def test_an_inverted_mask_can_be_traced_too():
    """A caller with white-on-black input flips one flag, not the image."""
    mask = 255 - make_mask()

    polygons = mask_to_complex_polygons(
        mask, px_per_mm=10.0, params=VectorizeParams(ink_is_dark=False)
    )

    assert len(polygons) >= 1


def test_an_all_paper_mask_yields_nothing():
    assert mask_to_complex_polygons(np.full((50, 50), 255, np.uint8), 10.0) == []


def test_a_colour_input_is_accepted():
    """Previews are sometimes three-channel; the tracer takes them anyway."""
    grey = make_mask()
    colour = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    assert mask_to_complex_polygons(colour, 10.0) == mask_to_complex_polygons(
        grey, 10.0
    )


# ----------------------------------------------------------------- winding


def test_a_hole_is_wound_opposite_to_its_outer_ring():
    """Measured against the editor: matching windings print solid.

    Squares 2 and 3 of the M3 experiment (hole opposing the outer ring) came
    out hollow; squares 4 and 5 (both the same) came out solid. See
    ``docs/M3_ENCODING.md``.
    """
    polygons = mask_to_complex_polygons(make_mask(), px_per_mm=10.0)
    rings = polygons[0]

    assert len(rings) == 2, "外环 + 一个孔"
    outer, hole = signed_area(rings[0]), signed_area(rings[1])
    assert (outer < 0) != (hole < 0), "孔必须与外环绕向相反"
    assert outer < 0, "本实现固定外环顺时针"


def test_holes_are_not_dropped():
    """A dropped hole fills the letter's counter with copper."""
    polygons = mask_to_complex_polygons(make_mask(), px_per_mm=10.0)
    assert len(polygons[0]) == 2


# -------------------------------------------------------------- separation


def test_each_blob_is_its_own_polygon():
    """Unlike a board outline, artwork keeps every shape separate.

    ``extract_outline_rings`` returns one outer ring plus "the rest", which is
    right for a board edge and wrong here: the counter of an "o" belongs to
    that "o", not to whichever shape happens to be largest.
    """
    mask = np.full((200, 300), 255, np.uint8)
    cv2.rectangle(mask, (20, 20), (100, 100), 0, -1)
    cv2.circle(mask, (220, 150), 40, 0, -1)

    polygons = mask_to_complex_polygons(mask, px_per_mm=10.0)

    assert len(polygons) == 2


def test_the_biggest_shape_comes_first():
    mask = np.full((200, 300), 255, np.uint8)
    cv2.circle(mask, (60, 100), 30, 0, -1)      # smaller
    cv2.rectangle(mask, (150, 40), (280, 170), 0, -1)  # bigger

    polygons = mask_to_complex_polygons(mask, px_per_mm=10.0)

    assert abs(signed_area(polygons[0][0])) > abs(signed_area(polygons[1][0]))


# ------------------------------------------------------- size and filtering


def test_the_geometry_is_centred_on_the_origin():
    """The board outline is centred, so the copper has to be too.

    Without this the copper carries the image's top-left origin while the
    board is built around its centre, and the two land half a picture apart.
    """
    polygons = mask_to_complex_polygons(make_mask(), px_per_mm=10.0)

    xs = [p[0] for p in polygons[0][0]]
    ys = [p[1] for p in polygons[0][0]]

    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0, abs=0.01)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(0.0, abs=0.01)


def test_a_pixel_size_maps_to_the_requested_millimetres():
    """The mask's width and the placed width have to agree."""
    mask = make_mask(shape=(200, 400))
    cv2.rectangle(mask, (0, 0), (399, 199), 0, -1)

    polygons = mask_to_complex_polygons(mask, px_per_mm=400 / 20.0)

    xs = [p[0] for p in polygons[0][0]]
    assert max(xs) - min(xs) == pytest.approx(20.0, rel=0.02)


def test_specks_below_the_threshold_are_dropped():
    mask = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(mask, (40, 40), (160, 160), 0, -1)     # the subject
    mask[5:7, 5:7] = 0                                    # a speck

    polygons = mask_to_complex_polygons(
        mask, px_per_mm=10.0, params=VectorizeParams(min_island_area_mm2=1.0)
    )

    assert len(polygons) == 1


def test_tiny_holes_are_filled_rather_than_cut():
    mask = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(mask, (40, 40), (160, 160), 0, -1)
    mask[100:102, 100:102] = 255                          # a pinhole

    polygons = mask_to_complex_polygons(
        mask, px_per_mm=10.0, params=VectorizeParams(min_hole_area_mm2=1.0)
    )

    assert len(polygons[0]) == 1, "过小的孔应被填掉而不是切出来"


def test_zero_pixels_per_mm_is_rejected():
    with pytest.raises(ValueError):
        mask_to_complex_polygons(make_mask(), 0.0)


# ------------------------------------------------------------------- styles


def test_the_fill_style_is_the_implemented_one():
    assert VectorizeParams().style == CopperStyle.FILL
    assert VectorizeParams().is_implemented


def test_the_trace_style_says_it_is_not_built_yet():
    """Better an explicit refusal than silently returning fill geometry.

    Emitting a shape the user did not ask for is worse than saying the feature
    does not exist: the file would look right and be wrong.
    """
    params = VectorizeParams(style=CopperStyle.TRACE)

    with pytest.raises(NotImplementedError):
        vectorize_mask(make_mask(), 10.0, params)


def test_an_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        vectorize_mask(make_mask(), 10.0, VectorizeParams(style="sparkle"))
