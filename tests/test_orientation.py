"""Regression tests for the Y-axis direction and edge straightening.

Both of these were latent bugs that passed every size check: mirroring a board
top-to-bottom and tilting an edge by a fraction of a degree leave all dimensions
correct. These tests assert the properties that sizes cannot catch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.epru import EproArchive  # noqa: E402
from pcbart.generator import generate_outline_board  # noqa: E402
from pcbart.image_prep import PrepParams, preprocess  # noqa: E402
from pcbart.outline import (  # noqa: E402
    OutlineParams,
    detect_circle,
    extract_outline_rings,
    snap_coordinates,
    snap_to_axes,
)
from pcbart.units import FILE_Y_IS_UP, Y_SIGN, y_from_file, y_to_file  # noqa: E402


# --------------------------------------------------------------- Y convention


def test_file_y_is_up():
    """The file's Y axis points up, same as the design frame."""
    assert FILE_Y_IS_UP is True
    assert Y_SIGN == 1.0


def test_y_helpers_are_identity_when_up():
    assert y_from_file(12.5) == 12.5
    assert y_to_file(12.5) == 12.5
    assert y_from_file(-3.0) == -3.0
    assert y_to_file(-3.0) == -3.0


def test_y_helpers_round_trip():
    for value in (0.0, 1.5, -99.25, 1234.5):
        assert y_from_file(y_to_file(value)) == value


def test_asymmetric_feature_keeps_its_orientation(tmp_path: Path):
    """An "F" must stay upright through mask -> outline -> file.

    The two arms sit at the top, so in a Y-up frame they must be at larger Y
    than the bare stem. Mirroring the board would put them at smaller Y and
    this assertion fails.
    """
    canvas = np.full((400, 300), 255, np.uint8)
    cv2.rectangle(canvas, (80, 40), (120, 360), 0, -1)      # stem
    cv2.rectangle(canvas, (80, 40), (250, 90), 0, -1)       # top arm
    cv2.rectangle(canvas, (80, 170), (210, 215), 0, -1)     # middle arm

    mask = preprocess(canvas, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=30.0, center=True)
    )

    ys = [p[1] for p in result.outer]
    y_lo, y_hi = min(ys), max(ys)
    span = y_hi - y_lo

    # The top arm occupies the highest band. Measure the ring's width there.
    def width_near(fraction: float) -> float:
        y_band = y_lo + span * fraction
        xs = [p[0] for p in result.outer if abs(p[1] - y_band) < span * 0.08]
        return (max(xs) - min(xs)) if len(xs) > 1 else 0.0

    top_width = width_near(0.9)
    bottom_width = width_near(0.1)
    assert top_width > bottom_width, (
        "the F's arms must be at high Y; otherwise the board is mirrored"
    )


def _decode_raw_ring(tokens: list) -> list[tuple[float, float]]:
    """Decode an ``L``-mode path into points.

    Index parity is not a reliable way to pick out Y values: the mode token is
    a string mixed into the stream, which shifts every later index.
    """
    points: list[tuple[float, float]] = []
    pending: list[float] = []
    for token in tokens:
        if isinstance(token, str):
            points.extend(
                (pending[i], pending[i + 1]) for i in range(0, len(pending) - 1, 2)
            )
            pending = []
            continue
        pending.append(float(token))
    points.extend((pending[i], pending[i + 1]) for i in range(0, len(pending) - 1, 2))
    return points


def test_generated_file_keeps_high_y_for_the_arms(tmp_path: Path):
    """Same check, read back from the written file rather than in memory."""
    canvas = np.full((400, 300), 255, np.uint8)
    cv2.rectangle(canvas, (80, 40), (120, 360), 0, -1)
    cv2.rectangle(canvas, (80, 40), (250, 90), 0, -1)

    mask = preprocess(canvas, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=30.0, center=True)
    )
    path = generate_outline_board(result.outer, tmp_path / "f.epro2", title="F")

    doc = EproArchive.read(path).document
    record = next(r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11)
    raw_points = _decode_raw_ring(record.body["path"])

    # Raw file Y must keep the same span and ordering as the design ring.
    raw_ys = [p[1] for p in raw_points]
    raw_span = (max(raw_ys) - min(raw_ys)) * 0.0254
    design_span = max(p[1] for p in result.outer) - min(p[1] for p in result.outer)
    assert raw_span == pytest.approx(design_span, rel=1e-3)

    # And the widest part must still be at high Y in both frames.
    def widest_at(points: list[tuple[float, float]], upper: bool) -> float:
        ys = [p[1] for p in points]
        lo, hi = min(ys), max(ys)
        band = lo + (hi - lo) * (0.85 if upper else 0.15)
        xs = [p[0] for p in points if abs(p[1] - band) < (hi - lo) * 0.1]
        return (max(xs) - min(xs)) if len(xs) > 1 else 0.0

    assert widest_at(raw_points, upper=True) > widest_at(raw_points, upper=False)
    assert widest_at(result.outer, upper=True) > widest_at(result.outer, upper=False)


# ------------------------------------------------------------ edge straightening


def _angle_from_vertical(p1, p2) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dx, dy))


def test_snap_to_axes_straightens_a_near_vertical_edge():
    ring = [(0.0, 0.0), (0.3, 50.0), (20.0, 50.0), (20.0, 0.0)]
    snapped = snap_to_axes(ring, 5.0)

    angle = _angle_from_vertical(snapped[0], snapped[1])
    assert angle == pytest.approx(0.0, abs=1e-9), "0.34 deg off should snap to 0"


def test_snap_to_axes_leaves_a_genuine_diagonal_alone():
    ring = [(0.0, 0.0), (20.0, 50.0), (40.0, 0.0), (20.0, -10.0)]
    snapped = snap_to_axes(ring, 5.0)

    # The steep edge is 21.8 degrees off vertical, beyond the 5 degree window.
    angle = _angle_from_vertical(snapped[0], snapped[1])
    original = _angle_from_vertical(ring[0], ring[1])
    assert angle == pytest.approx(original, abs=1e-9)


def test_snap_to_axes_can_be_disabled():
    ring = [(0.0, 0.0), (0.3, 50.0), (20.0, 50.0), (20.0, 0.0)]
    assert snap_to_axes(ring, 0.0) == ring


def test_coord_snap_merges_near_equal_coordinates():
    ring = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.2, 10.0), (0.0, 10.0)]
    snapped = snap_coordinates(ring, 0.5)

    assert snapped[0][0] == snapped[1][0] == snapped[2][0]


def test_coord_snap_can_be_disabled():
    ring = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.2, 10.0), (0.0, 10.0)]
    assert snap_coordinates(ring, 0.0) == ring


def test_letter_p_has_no_tilted_vertical_edges():
    """The decisive test for the reported slant."""
    from pcbart.image_prep import make_test_image

    image = make_test_image(shape="letter")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask,
        OutlineParams(target_width_mm=60.0, detect_circles=False),
    )

    ring = result.outer
    n = len(ring)
    deviations = []
    for i in range(n):
        p1, p2 = ring[i], ring[(i + 1) % n]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < 5.0:
            continue
        angle = _angle_from_vertical(p1, p2)
        if abs(angle) < 10.0:
            deviations.append(abs(angle))

    assert deviations, "expected at least one long vertical edge"
    assert max(deviations) < 1e-6, f"vertical edges still tilted: {deviations}"


def test_axis_snap_does_not_distort_circles():
    """Snapping must be skipped for circles, or a disc becomes a rounded square."""
    from pcbart.image_prep import make_test_image

    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0)
    )

    assert result.outer_circle is not None, "ring should be recognised as a circle"
    cx, cy, radius = result.outer_circle
    assert radius == pytest.approx(30.0, abs=0.5)

    # Every vertex must sit at that radius; a snapped disc would not.
    deviations = [
        abs(math.hypot(x - cx, y - cy) - radius) for x, y in result.outer
    ]
    assert max(deviations) < 0.1


# ------------------------------------------------------------------- circles


def test_detect_circle_accepts_a_circle():
    ring = [
        (50 * math.cos(2 * math.pi * i / 96), 50 * math.sin(2 * math.pi * i / 96))
        for i in range(96)
    ]
    circle = detect_circle(ring, 0.03)
    assert circle is not None
    cx, cy, r = circle
    assert cx == pytest.approx(0.0, abs=1e-6)
    assert cy == pytest.approx(0.0, abs=1e-6)
    assert r == pytest.approx(50.0, abs=1e-6)


def test_detect_circle_rejects_a_square():
    ring = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert detect_circle(ring, 0.03) is None


def test_detect_circle_rejects_a_rounded_rectangle():
    """A stadium shape is close to circular but not close enough."""
    ring = [
        (math.cos(2 * math.pi * i / 64) * 10, math.sin(2 * math.pi * i / 64) * 30)
        for i in range(64)
    ]
    assert detect_circle(ring, 0.03) is None


def test_detect_circle_can_be_disabled():
    from pcbart.image_prep import make_test_image

    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask,
        OutlineParams(target_width_mm=60.0, detect_circles=False),
    )
    assert result.outer_circle is None
    assert len(result.outer) > 10, "should fall back to a polygon"


def test_a_circular_board_writes_one_native_circle(tmp_path: Path):
    """A ring-shaped mask yields one CIRCLE outline, and nothing else.

    The enclosed void is not written: the editor draws a nested ring on the
    outline layer without cutting it, so a board cutout has to be placed in the
    editor rather than guessed at from a mask.
    """
    from pcbart.image_prep import make_test_image

    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(mask, OutlineParams(target_width_mm=60.0))

    path = generate_outline_board(
        result.outer,
        tmp_path / "ring.epro2",
        title="Ring",
        outer_circle=result.outer_circle,
    )
    doc = EproArchive.read(path).document

    outlines = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]
    assert len(outlines) == 1
    assert outlines[0].body["path"][0] == "CIRCLE"
    assert list(doc.iter_type("PAD")) == []


def test_native_circle_centre_is_not_offset(tmp_path: Path):
    """Regression: the centre was shifted twice, moving the circle off-origin."""
    from pcbart.image_prep import make_test_image

    image = make_test_image(shape="ring")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0, center=True)
    )

    path = generate_outline_board(
        result.outer,
        tmp_path / "ring.epro2",
        title="Ring",
        outer_circle=result.outer_circle,
    )
    doc = EproArchive.read(path).document
    record = next(
        r
        for r in doc.iter_type("POLY")
        if r.body.get("layerId") == 11 and r.body["path"][0] == "CIRCLE"
    )
    _, cx, cy, _r = record.body["path"]

    assert cx == pytest.approx(0.0, abs=1e-6)
    assert cy == pytest.approx(0.0, abs=1e-6)
