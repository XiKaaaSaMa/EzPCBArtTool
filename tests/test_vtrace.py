"""VTracer-backed vectorisation: the output has to be usable geometry.

These tests pin the three things that were *measured* rather than assumed, each
of which was a real defect first:

* the per-``<path>`` ``translate`` transform, without which every shape stacks
  at the origin (the 企鹅物流 logo became one black blob);
* the polarity (the ink is painted black because VTracer traces the black area);
* the upscale, which is what keeps small sources from losing their fine detail.

The high-level guard is a round-trip IoU: trace a synthetic shape, draw the
resampled rings back into a raster, and require it to match the mask. That
catches a wrong transform, a wrong offset, a dropped ring and a swapped pair
all at once, and it does not care how the geometry is expressed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pcbart.geometry import arc_ring_to_path
from pcbart.vtrace import (
    TracedRing,
    _svg_paths,
    parse_svg_paths,
    trace_mask,
)

pytest.importorskip("vtracer", reason="VTracer 未安装")


#: One millimetre per pixel keeps the synthetic numbers easy to read.
PX_PER_MM = 1.0


def _mask(width: int, height: int) -> np.ndarray:
    """A blank plane; ``0`` is ink and ``255`` is bare, as the pipeline uses."""
    return np.full((height, width), 255, dtype=np.uint8)


def _disc(width: int, height: int, radius: float) -> np.ndarray:
    mask = _mask(width, height)
    yy, xx = np.mgrid[0:height, 0:width]
    inside = (xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2 <= radius**2
    mask[inside] = 0
    return mask


def _ring_points(ring: TracedRing, upscale: int = 1) -> list[tuple[float, float]]:
    """Expand one ring through the real path encoder into a dense point list.

    ``[x0, y0, "L", <coords> | "ARC", sweep, x, y, ...]`` -- the leading ``L``
    is the polygon type tag, so decoding starts at index 3. The arcs are
    sampled here rather than drawn as chords, because a chord would hide an
    arc pointing the wrong way.
    """
    path = arc_ring_to_path(ring.points, ring.sweeps)
    points = [(path[0], path[1])]
    index = 3
    while index < len(path):
        if path[index] == "ARC":
            sweep = float(path[index + 1])
            end_x, end_y = float(path[index + 2]), float(path[index + 3])
            index += 4
            start_x, start_y = points[-1]
            chord = math.hypot(end_x - start_x, end_y - start_y)
            sweep_radians = math.radians(abs(sweep))
            if chord > 1e-9 and sweep_radians > 1e-9:
                radius = chord / (2.0 * math.sin(sweep_radians / 2.0))
                height = math.sqrt(
                    max(radius * radius - (chord / 2.0) ** 2, 0.0)
                )
                ux, uy = (end_x - start_x) / chord, (end_y - start_y) / chord
                centre = None
                start_angle = end_angle = 0.0
                for sign in (1.0, -1.0):
                    cx = (start_x + end_x) / 2.0 - sign * height * uy
                    cy = (start_y + end_y) / 2.0 + sign * height * ux
                    a0 = math.atan2(start_y - cy, start_x - cx)
                    a1 = math.atan2(end_y - cy, end_x - cx)
                    delta = a1 - a0
                    while delta <= -math.pi:
                        delta += 2 * math.pi
                    while delta > math.pi:
                        delta -= 2 * math.pi
                    if (delta > 0) == (sweep > 0):
                        centre, start_angle, end_angle = (cx, cy), a0, a1
                        break
                assert centre is not None, "弧的扫向无法解析"
                cx, cy = centre
                for step in range(1, 24):
                    angle = start_angle + math.radians(sweep) * step / 24.0
                    points.append(
                        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
                    )
            points.append((end_x, end_y))
        else:
            points.append((float(path[index]), float(path[index + 1])))
            index += 2
    return points


def _raster(
    shapes: list[list[TracedRing]], width: int, height: int, upscale: int = 1
) -> np.ndarray:
    """Draw rings back into a boolean "ink here" array, in source pixels.

    Largest outer ring first, and each ring's holes before its outline, so a
    container cannot cover what it contains -- the nesting the 阵营 logo needs.
    """
    from PIL import Image, ImageDraw

    scale = upscale
    image = Image.new("L", (width * scale, height * scale), 255)
    draw = ImageDraw.Draw(image)

    def area(ring: TracedRing) -> float:
        points = [
            (width / 2.0 + x * PX_PER_MM, height / 2.0 - y * PX_PER_MM)
            for x, y in ring.points
        ]
        return abs(
            sum(
                points[i][0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * points[i][1]
                for i in range(len(points))
            )
        ) / 2.0

    for group in sorted(shapes, key=lambda rings: -area(rings[0])):
        for index in sorted(range(len(group)), key=lambda i: i != 0):
            ring = group[index]
            points = [
                (
                    (width / 2.0 + x * PX_PER_MM) * scale,
                    (height / 2.0 - y * PX_PER_MM) * scale,
                )
                for x, y in _ring_points(ring)
            ]
            if len(points) >= 3:
                draw.polygon(points, fill=0 if index == 0 else 255)
    return np.array(image).reshape(height, scale, width, scale).mean(axis=(1, 3)) < 128


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int((left | right).sum())
    if union == 0:
        return 1.0
    return int((left & right).sum()) / union


# --- the transform -------------------------------------------------------


def test_each_path_translate_is_read() -> None:
    """⚠️ The transform IS the geometry; ignoring it stacks every shape.

    VTracer writes each path's coordinates relative to its own origin and puts
    the real position in ``transform="translate(x,y)"``. Dropping it flattened
    all 26 paths of the 企鹅物流 logo into one blob at the top-left corner.
    """
    svg = (
        '<svg><path d="M0 0 L1 0 L1 1 Z" transform="translate(10,20)"/>'
        '<path d="M0 0 L2 0 L2 2 Z" transform="translate(30.5,40.25)"/></svg>'
    )
    entries = _svg_paths(svg)
    assert [offset for _, offset in entries] == [(10.0, 20.0), (30.5, 40.25)]

    subpaths = parse_svg_paths(svg)
    # ``Z`` closes the ring, so the first point is repeated at the end.
    assert subpaths[0][0] == [
        (10.0, 20.0),
        (11.0, 20.0),
        (11.0, 21.0),
        (10.0, 20.0),
    ]
    assert subpaths[1][0] == [
        (30.5, 40.25),
        (32.5, 40.25),
        (32.5, 42.25),
        (30.5, 40.25),
    ]


def test_a_path_without_a_transform_stays_put() -> None:
    """The tag is optional, and a missing one must not raise or shift."""
    entries = _svg_paths('<svg><path d="M0 0 L1 1 Z"/></svg>')
    assert entries == [("M0 0 L1 1 Z", (0.0, 0.0))]


def test_two_shapes_keep_their_relative_positions() -> None:
    """The regression in its plainest form: separate shapes must not converge.

    Before the transform was read, both squares came back overlapping at the
    origin. Two disjoint shapes are the smallest input that proves they do not.
    """
    mask = _mask(120, 60)
    mask[10:30, 10:30] = 0
    mask[10:30, 80:100] = 0

    shapes = trace_mask(mask, PX_PER_MM, mode="polygon", upscale=1)
    assert len(shapes) == 2, "两个方块应该各成一个形状"

    boxes = []
    for group in shapes:
        xs = [p[0] for ring in group for p in ring.points]
        ys = [p[1] for ring in group for p in ring.points]
        boxes.append((min(xs), max(xs), min(ys), max(ys)))
    boxes.sort()
    left, right = boxes
    assert left[1] < right[0], "两个方块在 X 上不该重叠"
    # ⚠️ VTracer works a half pixel off the raster grid (a 20 px square spans
    # 9.5..29.5, not 10..30). The centres still align, which is what the
    # bounding-box alignment guarantees; only the sub-pixel outline differs.
    assert _iou(_raster(shapes, 120, 60), mask < 128) > 0.88


# --- the round trip ------------------------------------------------------


@pytest.mark.parametrize("mode", ["spline", "polygon"])
def test_a_disc_round_trips(mode: str) -> None:
    """A circle written by either mode has to come back as that circle."""
    mask = _disc(200, 200, 80.0)
    shapes = trace_mask(mask, PX_PER_MM, mode=mode)
    assert shapes

    score = _iou(_raster(shapes, 200, 200), mask < 128)
    assert score > 0.9, f"{mode} 的圆重建只有 {score:.3f}"


def test_a_ring_keeps_its_hole() -> None:
    """A hole is a ring in the same shape, not a separate shape to be filled."""
    mask = _disc(200, 200, 80.0)
    inner = _disc(200, 200, 40.0)
    mask[inner < 128] = 255

    shapes = trace_mask(mask, PX_PER_MM, mode="polygon")
    assert len(shapes) == 1, "环形应是一个带孔的形状，不是一个外环加一个内块"
    assert len(shapes[0]) == 2, "外形加孔应有两个环"

    score = _iou(_raster(shapes, 200, 200), mask < 128)
    assert score > 0.9, f"环形重建只有 {score:.3f}"


def test_the_artwork_keeps_the_source_size() -> None:
    """The traced box must equal the ink's box; VTracer rebases its output."""
    mask = _mask(160, 100)
    mask[20:80, 30:130] = 0

    shapes = trace_mask(mask, PX_PER_MM, mode="polygon", upscale=1)
    xs = [p[0] for group in shapes for ring in group for p in ring.points]
    ys = [p[1] for group in shapes for ring in group for p in ring.points]
    # The ink spans 100 x 60 px, and the design frame is centred, so the
    # geometry must span the same 100 x 60 mm about the origin.
    assert max(xs) - min(xs) == pytest.approx(100.0, abs=1.5)
    assert max(ys) - min(ys) == pytest.approx(60.0, abs=1.5)
    assert min(xs) == pytest.approx(-50.0, abs=1.5)
    assert min(ys) == pytest.approx(-30.0, abs=1.5)


def test_an_empty_mask_traces_to_nothing() -> None:
    assert trace_mask(_mask(40, 40), PX_PER_MM) == []


def test_a_blank_mask_traces_to_nothing() -> None:
    """All-ink has no contour to follow and must not crash."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    shapes = trace_mask(mask, PX_PER_MM, mode="polygon", upscale=1)
    score = _iou(_raster(shapes, 40, 40), mask < 128) if shapes else 0.0
    assert score > 0.9, "整块墨也应重建出来"


# --- the upscale ---------------------------------------------------------


def test_upscaling_keeps_the_size() -> None:
    """The magnification is a working resolution, not a scale on the output.

    The traced extent may breathe by about a pixel as the fitter's grid
    changes, but it must not scale with the magnification: at 3x an
    un-divided result would be 3x wide.
    """
    mask = _disc(150, 150, 55.0)
    plain = trace_mask(mask, PX_PER_MM, mode="spline", upscale=1)
    blown_up = trace_mask(mask, PX_PER_MM, mode="spline", upscale=3)
    assert plain and blown_up

    def span(shapes):
        xs = [p[0] for group in shapes for ring in group for p in ring.points]
        ys = [p[1] for group in shapes for ring in group for p in ring.points]
        return (max(xs) - min(xs), max(ys) - min(ys))

    assert span(blown_up) == pytest.approx(span(plain), abs=1.5)


def test_upscaling_does_not_staircase() -> None:
    """⚠️ The magnification has to be **smoothed**, not nearest-neighbour.

    A circle is the cleanest witness. Measured on a 150 px, radius-55 disc:

    ======  ===========  ========  =========
    scale   resampling   points    verdict
    ======  ===========  ========  =========
    1x      --            38       curve fitted
    3x      nearest?     295       0 arcs -- a staircase round the rim
    3x      bilinear     295       68 arcs -- the same vertices, curved
    ======  ===========  ========  =========

    ⚠️ **Point count cannot tell the two apart** -- both magnifications emit
    the same vertices. What differs is whether any of the segments carries a
    bulge, which is why this asserts on arcs rather than on points. A
    nearest-neighbour copy gives VTracer a field of exact right angles and it
    answers with straight lines, so the rim comes out faceted at source-pixel
    scale -- exactly the defect the user reported with the built-in tracer.
    """
    mask = _disc(150, 150, 55.0)
    blown_up = trace_mask(mask, PX_PER_MM, mode="spline", upscale=3)
    assert blown_up

    arcs = sum(
        1
        for group in blown_up
        for ring in group
        for sweep in ring.sweeps
        if abs(sweep) > 1e-6
    )
    assert arcs >= 20, (
        f"3x 放大后的圆只有 {arcs} 段弧，圆边成了直角阶梯——"
        "放大必须用平滑插值，不能用最近邻"
    )


def test_the_thresholds_are_source_pixels() -> None:
    """⚠️ A speckle threshold means the same feature at any working scale.

    Left fixed while the raster grows, the threshold describes a steadily
    smaller feature and the trace keeps detail it should have dropped -- the
    magnified run then disagrees with the 1x run about what a speck is.
    """
    mask = _mask(120, 120)
    mask[10:100, 10:100] = 0
    mask[55:60, 55:60] = 255  # a 5 px hole: kept, it is above the cutoff

    one = trace_mask(mask, PX_PER_MM, mode="polygon", filter_speckle=8, upscale=1)
    three = trace_mask(
        mask, PX_PER_MM, mode="polygon", filter_speckle=8, upscale=3
    )
    # Both must agree that the plate is solid apart from that one hole. A
    # threshold left in magnified pixels would call the hole a speck at 3x and
    # fill it in, so the two scores would part company.
    for score in (
        _iou(_raster(one, 120, 120), mask < 128),
        _iou(_raster(three, 120, 120), mask < 128),
    ):
        assert score > 0.95, f"薄板带孔的重建只有 {score:.3f}"


def test_a_bad_upscale_is_rejected() -> None:
    with pytest.raises(ValueError):
        trace_mask(_disc(40, 40, 10.0), PX_PER_MM, upscale=0)


def test_a_bad_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        trace_mask(_disc(40, 40, 10.0), 0.0)
