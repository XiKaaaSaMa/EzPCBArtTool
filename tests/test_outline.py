"""Mask-to-outline tests: tracing, simplification and scaling maths."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")
shapely = pytest.importorskip("shapely.geometry")

from pcbart.image_prep import PrepParams, preprocess, make_test_image  # noqa: E402
from pcbart.outline import (  # noqa: E402
    OutlineParams,
    extract_outline_rings,
    fit_rings_to_box,
    mask_to_rings_px,
    scale_rings,
    simplify_ring,
)


def polygon_area(ring) -> float:
    total = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


# ------------------------------------------------------------------- tracing


def test_square_mask_traces_to_four_corners():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1

    outers = mask_to_rings_px(mask, min_ring_area_px=1)

    assert len(outers) == 1
    simplified = simplify_ring(outers[0], 1.0)
    assert len(simplified) == 4


def test_ring_mask_traces_only_its_outer_edge():
    """An enclosed void is not a board feature, so it is never traced.

    The editor draws a nested ring on the outline layer rather than cutting it,
    so emitting one would describe a hole the finished board does not have.
    """
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 80, 1, -1)
    cv2.circle(mask, (100, 100), 45, 0, -1)

    outers = mask_to_rings_px(mask, min_ring_area_px=10)

    assert len(outers) == 1
    nearest = min(math.hypot(x - 100.0, y - 100.0) for x, y in outers[0])
    assert nearest > 45.0, "the contour followed the ring's inside edge"


def test_small_components_are_dropped():
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 80, 1, -1)
    mask[10, 10] = 1  # a speck far from the main shape

    outers = mask_to_rings_px(mask, min_ring_area_px=50)

    assert len(outers) == 1


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError, match="没有找到有效轮廓"):
        mask_to_rings_px(np.zeros((50, 50), np.uint8))


# ------------------------------------------------------------ simplification


def test_simplification_reduces_point_count():
    ring = [
        (math.cos(2 * math.pi * i / 500) * 100, math.sin(2 * math.pi * i / 500) * 100)
        for i in range(500)
    ]
    simplified = simplify_ring(ring, 2.0)
    assert len(simplified) < len(ring) / 4
    assert len(simplified) >= 3


def test_zero_tolerance_keeps_every_point():
    ring = [(float(i), float(i % 3)) for i in range(20)]
    assert simplify_ring(ring, 0.0) == ring


def test_simplification_preserves_the_overall_shape():
    ring = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # Insert collinear points that carry no shape information.
    noisy = []
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        noisy.append((x1, y1))
        noisy.append(((x1 + x2) / 2, (y1 + y2) / 2))

    simplified = simplify_ring(noisy, 0.5)
    assert len(simplified) == 4


# ------------------------------------------------------------------- scaling


def test_scale_flips_y_so_the_image_is_not_upside_down():
    """Image Y grows downward; board Y grows upward."""
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
    scaled = scale_rings([ring], 2.0)[0]

    assert scaled[0] == (0.0, 0.0)
    assert scaled[1] == (20.0, 0.0)
    assert scaled[2] == (20.0, -10.0)


def test_fit_to_box_uses_the_tighter_constraint():
    ring = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
    # 100 wide into 50 -> 0.5 ;  50 tall into 200 -> 4.0 ; min wins.
    assert fit_rings_to_box([ring], 50.0, 200.0) == pytest.approx(0.5)


def test_fit_to_single_dimension():
    ring = [(0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)]
    assert fit_rings_to_box([ring], 80.0, None) == pytest.approx(0.8)


def test_fit_requires_a_target():
    with pytest.raises(ValueError):
        fit_rings_to_box([[(0, 0), (1, 0), (1, 1)]], None, None)


# --------------------------------------------------------------- full pipeline


def test_full_pipeline_hits_the_target_width():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask

    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0, center=True)
    )

    xs = [p[0] for p in result.outer]
    assert (max(xs) - min(xs)) == pytest.approx(60.0, abs=1e-6)


def test_full_pipeline_preserves_aspect_ratio():
    image = make_test_image(width=400, height=200, shape="blob")
    mask = preprocess(image, PrepParams()).mask
    result = extract_outline_rings(mask, OutlineParams(target_width_mm=60.0))

    xs = [p[0] for p in result.outer]
    ys = [p[1] for p in result.outer]
    assert (max(xs) - min(xs)) == pytest.approx(60.0, abs=1e-6)
    assert (max(ys) - min(ys)) < 60.0


def test_full_pipeline_returns_a_single_ring():
    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask

    result = extract_outline_rings(mask, OutlineParams(target_width_mm=60.0))

    assert result.ring_count == 1


def test_full_pipeline_centres_on_origin():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask
    result = extract_outline_rings(mask, OutlineParams(target_width_mm=50.0, center=True))

    xs = [p[0] for p in result.outer]
    ys = [p[1] for p in result.outer]
    assert (min(xs) + max(xs)) == pytest.approx(0.0, abs=1e-6)
    assert (min(ys) + max(ys)) == pytest.approx(0.0, abs=1e-6)


def test_origin_offset_is_applied_after_centring():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=50.0, center=True, origin=(10.0, 5.0))
    )

    xs = [p[0] for p in result.outer]
    ys = [p[1] for p in result.outer]
    assert ((min(xs) + max(xs)) / 2) == pytest.approx(10.0, abs=1e-6)
    assert ((min(ys) + max(ys)) / 2) == pytest.approx(5.0, abs=1e-6)


def test_simplification_reduces_point_count_dramatically():
    """A traced bitmap has one vertex per boundary pixel; that must not ship.

    Circle recognition is disabled here so the comparison is about
    simplification itself, not about the CIRCLE shortcut.
    """
    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask

    raw = extract_outline_rings(
        mask,
        OutlineParams(
            target_width_mm=60.0,
            simplify_tolerance_px=0.0,
            detect_circles=False,
        ),
    )
    simplified = extract_outline_rings(
        mask,
        OutlineParams(
            target_width_mm=60.0,
            simplify_tolerance_px=2.0,
            detect_circles=False,
        ),
    )

    assert simplified.points_after < raw.points_after / 5
    assert simplified.points_before == raw.points_before
    assert simplified.points_after > 10, "still a real polygon, just fewer points"


def test_positive_offset_grows_the_outline():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask

    plain = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0, center=False)
    )
    grown = extract_outline_rings(
        mask,
        OutlineParams(target_width_mm=60.0, center=False, offset_mm=1.0),
    )

    assert abs(polygon_area(grown.outer)) > abs(polygon_area(plain.outer))


def test_negative_offset_shrinks_the_outline():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask

    plain = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0, center=False)
    )
    shrunk = extract_outline_rings(
        mask,
        OutlineParams(target_width_mm=60.0, center=False, offset_mm=-1.0),
    )

    assert abs(polygon_area(shrunk.outer)) < abs(polygon_area(plain.outer))


def test_excessive_negative_offset_is_rejected():
    image = make_test_image(shape="blob")
    mask = preprocess(image, PrepParams()).mask
    with pytest.raises(ValueError):
        extract_outline_rings(
            mask,
            OutlineParams(target_width_mm=60.0, offset_mm=-100.0),
        )


# ------------------------------------------------------------- winding rules


def test_outer_ring_is_clockwise_in_the_design_frame():
    """EasyEDA Pro expects the outer contour clockwise."""
    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(mask, OutlineParams(target_width_mm=60.0))

    assert polygon_area(result.outer) < 0, "outer ring must be clockwise"


@pytest.mark.parametrize("shape", ["blob", "star", "ring", "letter"])
def test_all_test_shapes_produce_valid_geometry(shape: str):
    image = make_test_image(shape=shape)
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(mask, OutlineParams(target_width_mm=50.0))

    polygon = shapely.Polygon(result.outer)
    assert polygon.is_valid
    assert polygon.area > 0
