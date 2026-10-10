"""Arc geometry through the export chain: winding, transforms, and the files.

The 阵营 logo is traced with VTracer and the curves travel to both writers as
``ARC`` segments rather than being flattened into chords. That is four separate
places that can go wrong, and each of them was wrong at least once while this
was built:

* the winding rule for a ring whose vertices alone do not enclose anything;
* reversing a ring, where the sweeps have to shift *and* change sign;
* the placement matrix, where a mirror reverses a curve's direction;
* the Gerber data block, whose word order a reader cares about.

So the tests here work on the real files. A geometry test that only checked the
numbers in memory would have passed through every one of those bugs.
"""

from __future__ import annotations

import math

import pytest

from pcbart.geometry import (
    ArcRing,
    arc_polygon_to_path,
    arc_ring_to_path,
    is_ccw_arc,
    normalize_arc_polygon,
    reverse_arc_ring,
    signed_area_arc,
)

#: A circle as VTracer can write it: two points, two half-turns.
CIRCLE = ArcRing([(0.0, 0.0), (10.0, 0.0)], [180.0, 180.0])
#: A half disc: one arc and one chord.
HALF = ArcRing([(0.0, 0.0), (10.0, 0.0)], [180.0, 0.0])
SQUARE = ArcRing(
    [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    [0.0, 0.0, 0.0, 0.0],
)


# --- the area, which is what the winding rule is derived from ------------


def test_a_two_point_circle_has_the_area_of_a_circle() -> None:
    """⚠️ A circle written as two points has **zero vertex area**.

    That is the trap: ``signed_area`` on the points alone returns 0.0, so a
    winding test built on it answers arbitrarily and the normaliser leaves the
    ring as it found it. The true area has to add each segment's circular
    segment, and the answer for two half-turns is exactly ``pi * r^2``.
    """
    assert signed_area_arc(CIRCLE) == pytest.approx(math.pi * 25.0, rel=1e-9)
    assert signed_area_arc(HALF) == pytest.approx(math.pi * 25.0 / 2.0, rel=1e-9)
    assert signed_area_arc(SQUARE) == pytest.approx(100.0, rel=1e-9)


def test_reversing_a_ring_only_changes_the_sign_of_the_area() -> None:
    """The reversed ring is the same curve travelled the other way.

    ⚠️ If the sweeps do not shift by one *and* flip sign, the arc and the chord
    swap places and the shape is a different one -- not the same shape with the
    opposite orientation.
    """
    for ring in (CIRCLE, HALF, SQUARE):
        reversed_ring = reverse_arc_ring(ring)
        assert signed_area_arc(reversed_ring) == pytest.approx(
            -signed_area_arc(ring), rel=1e-9
        )


def _traced_points(ring: ArcRing) -> list[tuple[float, float]]:
    """The ring's outline sampled through the **real** path encoder."""
    from pcbart.geometry import sample_arc

    path = arc_ring_to_path(ring.points, ring.sweeps)
    cursor = (path[0], path[1])
    points = [cursor]
    index = 3
    while index < len(path):
        if path[index] == "ARC":
            segment = sample_arc(
                cursor, (path[index + 2], path[index + 3]), path[index + 1]
            )
            cursor = (path[index + 2], path[index + 3])
            index += 4
        else:
            segment = [cursor, (path[index], path[index + 1])]
            cursor = (path[index], path[index + 1])
            index += 2
        points.extend(segment[1:])
    return points


def test_reversing_a_half_disc_keeps_the_arc_on_the_same_side() -> None:
    """The half-disc is the case that catches an off-by-one sweep shift.

    The circle and the square are symmetric enough that a wrong shift can
    still cancel out. A half disc is not: if the sweeps stay put instead of
    shifting, the arc and the chord swap ends and the flat side flips over.

    The check is on the **traced outline**, because that is the one thing that
    cannot be fooled: walking the same curve backwards covers the same points
    in the opposite order, so both directions must bulge to the same side.
    """
    forward = _traced_points(HALF)
    backward = _traced_points(reverse_arc_ring(HALF))

    # Both bulge the same way. If the sweeps did not shift, one of these would
    # come out below the chord.
    for name, points in (("正向", forward), ("反向", backward)):
        bulges = [point for point in points if abs(point[1]) > 1.0]
        assert bulges, f"{name}的弧没有鼓出来"
        assert all(point[1] > 0 for point in bulges), f"{name}的弧鼓到了弦的另一侧"

    # And they enclose the same area. ⚠️ Comparing point-by-point does not
    # work: the two directions sample their arcs at different parameters, so
    # the coordinates differ in the last decimal. The enclosed area is the
    # invariant that survives that -- and a wrong sweep shift changes it,
    # because pairing an arc with the wrong chord draws a different region.
    def area(points) -> float:
        return abs(
            sum(
                points[i][0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * points[i][1]
                for i in range(len(points))
            )
        ) / 2.0

    assert area(forward) == pytest.approx(area(backward), rel=1e-3)
    assert area(forward) == pytest.approx(math.pi * 25.0 / 2.0, rel=1e-2)

    # And the sweep itself flips sign with the direction of travel.
    reversed_ring = reverse_arc_ring(HALF)
    path = arc_ring_to_path(reversed_ring.points, reversed_ring.sweeps)
    sweeps = [
        path[index + 1] for index, token in enumerate(path) if token == "ARC"
    ]
    assert sweeps == [-180.0]


# --- the winding the editor needs ---------------------------------------


def test_normalising_winds_the_outer_ring_clockwise() -> None:
    rings = normalize_arc_polygon(
        CIRCLE, holes=[ArcRing([(2.0, 0.0), (8.0, 0.0)], [180.0, 180.0])]
    )
    assert len(rings) == 2
    assert not is_ccw_arc(rings[0]), "外环应当顺时针"
    assert is_ccw_arc(rings[1]), "孔应当与外环相反"


def test_a_two_point_ring_survives_normalising() -> None:
    """The ``>= 3`` rule for chord rings would silently drop these."""
    rings = normalize_arc_polygon(CIRCLE)
    assert len(rings) == 1
    assert len(rings[0].points) == 2


# --- the placement matrix -----------------------------------------------


def _placement(**overrides):
    from pcbart.artboard import ImagePlacement

    defaults = dict(
        path="",
        x_mm=0.0,
        y_mm=0.0,
        width_mm=10.0,
        height_mm=10.0,
        layer="顶层铜",
    )
    defaults.update(overrides)
    return ImagePlacement(**defaults)


def test_a_mirror_reverses_every_arc() -> None:
    """⚠️ Flipping the vertices is not enough; the curves have to flip too.

    A sweep is signed relative to the direction of travel, so a reflection
    turns a counter-clockwise bulge into a clockwise one. Transforming only the
    points leaves arcs bending away from their own chord -- geometry that reads
    as a bad trace rather than as a transform bug.
    """
    flipped = _placement(flip_horizontal=True).board_arc_ring(HALF)
    assert [round(sweep, 6) for sweep in flipped.sweeps] == [-180.0, -0.0]

    plain = _placement().board_arc_ring(HALF)
    assert [round(sweep, 6) for sweep in plain.sweeps] == [180.0, 0.0]


def test_a_half_turn_does_not_reverse_arcs() -> None:
    """Flipping **both** axes is a rotation, not a reflection.

    The determinant's sign is the thing that decides, and two flips make it
    positive again. Keying off "any flip is on" would curve every arc backwards
    on a placement the user had only turned over.
    """
    turned = _placement(flip_horizontal=True, flip_vertical=True).board_arc_ring(HALF)
    assert [round(sweep, 6) for sweep in turned.sweeps] == [180.0, 0.0]


def test_a_rotation_does_not_reverse_arcs() -> None:
    rotated = _placement(rotation_deg=37.0).board_arc_ring(HALF)
    assert [round(sweep, 6) for sweep in rotated.sweeps] == [180.0, 0.0]


def test_a_mirrored_placement_keeps_the_area() -> None:
    """A reflection preserves area; only the sign of the winding changes."""
    placement = _placement(flip_horizontal=True, x_mm=20.0, y_mm=5.0)
    ring = placement.board_arc_ring(CIRCLE)
    assert abs(signed_area_arc(ring)) == pytest.approx(
        abs(signed_area_arc(CIRCLE)), rel=1e-9
    )


# --- the writers ---------------------------------------------------------


def test_the_epro2_path_carries_arc_tokens() -> None:
    path = arc_ring_to_path(CIRCLE.points, CIRCLE.sweeps)
    assert path[2] == "L"
    assert path[3] == "ARC" and path[4] == 180.0
    assert path[7] == "ARC" and path[8] == 180.0


def test_a_gerber_region_writes_arcs_as_g02_g03() -> None:
    """The fab file has to carry the curve, not a chord approximation.

    Flattening the arcs at the last step would put back exactly the faceting
    the whole VTracer switch was made to remove.
    """
    from pcbart.gerber import gerber_text

    # A normalised outer ring is wound clockwise throughout, so a circle comes
    # out as two G02 blocks. The direction word has to follow the sweep's sign,
    # which the counter-clockwise case below pins from the other side.
    text = gerber_text(
        function="Copper,L1,Top", regions=[normalize_arc_polygon(CIRCLE)]
    )
    assert "G75*" in text, "弧模式没有开启"
    assert "G02*" in text, "顺时针的环应当用 G02"
    assert "G03*" not in text, "整个环同向，不该混用 G02/G03"
    assert "I" in text and "J" in text, "弧的中心偏移没有写出来"

    counter_clockwise = gerber_text(
        function="Copper,L1,Top", regions=[[CIRCLE]]
    )
    assert "G03*" in counter_clockwise, "逆时针的环应当用 G03"


def test_a_gerber_arc_block_puts_coordinates_before_i_j() -> None:
    """⚠️ ``X..Y..I..J..D..``, not ``I..J..X..Y..D..``.

    The specification treats a data block as a loose sequence of words, but
    readers are not that loose: gerbonara's coordinate pattern is
    ``(G..)?(X..)?(Y..)?(I..)?(J..)?D..``, and an ``I..J..X..Y..`` block does
    not match it -- the arc is discarded as an unknown statement. Measured on a
    real export, **every one of 4299 arcs** was dropped that way while the file
    still looked plausible in a text editor.
    """
    import re

    from pcbart.gerber import gerber_text

    text = gerber_text(function="Copper,L1,Top", regions=[normalize_arc_polygon(CIRCLE)])
    blocks = [
        line
        for line in text.splitlines()
        if line.startswith(("X", "I", "J")) and "D01" in line
    ]
    assert blocks, "没有写出任何绘制块"
    pattern = re.compile(r"^(?:X-?\d+)?(?:Y-?\d+)?(?:I-?\d+)?(?:J-?\d+)?D0?1\*$")
    for block in blocks:
        assert pattern.match(block), f"数据块顺序不对：{block}"


def test_an_exported_gerber_can_be_read_back() -> None:
    """Independent readers are the only proof; ours agreed with itself.

    ``gerbonara`` is the check, exactly as it is for the board outline. Its
    bounding box is also the guard against a nonsense radius: a near-straight
    fitted arc has a radius of hundreds of metres, and before the sagitta
    filter the card's box came out 194 x 194 mm instead of 50 x 100.
    """
    gerbonara = pytest.importorskip("gerbonara")

    from pcbart.gerber import gerber_text

    # A 14 mm square with a 3 mm rounded-corner arc on each side would be
    # hard to reason about by hand; a **circle** is exact: the two half-turns
    # must come back as a 10 mm box with no bulge beyond the rim.
    rings = normalize_arc_polygon(
        ArcRing([(0.0, 0.0), (10.0, 0.0)], [180.0, 180.0])
    )
    text = gerber_text(function="Copper,L1,Top", regions=[rings])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "test.gtl"
        target.write_text(text, encoding="ascii")
        parsed = gerbonara.GerberFile.open(str(target))
        assert parsed.objects, "读回来是空的——弧被丢弃了"
        box = parsed.bounding_box()
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
        # ⚠️ An arc drawn the wrong way round reaches *beyond* the rim, so the
        # box is the test: both half-turns have to stay on the circle. A
        # full-circle-instead-of-half mistake measured 40 x 40 here against a
        # true 10 x 10, and a bogus radius blows it up by orders of magnitude.
        assert 9.0 <= width <= 11.0, f"宽 {width:.3f} mm 不对"
        assert 9.0 <= height <= 11.0, f"高 {height:.3f} mm 不对"


def test_an_absurd_radius_becomes_a_line() -> None:
    """⚠️ A near-straight cubic fits an arc with a radius of hundreds of metres.

    Left in, those produce I/J offsets no reader can act on and a bounding box
    far larger than the board. Measured on 企鹅物流 the fitted radii reached
    242 metres, and gerbonara then read the card as 194 x 194 mm against a real
    50 x 100.
    """
    from pcbart.vtrace import MAX_ARC_RADIUS_RATIO, drop_shallow_arcs

    # A 1 mm chord through half a degree is a radius of 114.6 mm -- more than
    # twenty times the 1 mm reference below, so it reads as a straight edge.
    absurd = ArcRing([(0.0, 0.0), (1.0, 0.0)], [0.5, 0.0])
    assert drop_shallow_arcs(absurd, reference_mm=1.0).sweeps[0] == 0.0

    # A real curve on the same reference survives: 90 degrees on a 1 mm chord
    # is a radius of 0.71 mm.
    curved = ArcRing([(0.0, 0.0), (1.0, 0.0)], [90.0, 0.0])
    assert drop_shallow_arcs(curved, reference_mm=1.0).sweeps[0] == 90.0
    assert MAX_ARC_RADIUS_RATIO >= 10, "比例太紧会误删真正的曲线"


def test_the_cutoff_is_a_ratio_not_a_length() -> None:
    """⚠️ An absolute bulge threshold measures the segmentation, not the shape.

    The sagitta of a fitted segment shrinks as VTracer splits the curve more
    finely. Measured on the same 企鹅物流 crest, the median sagitta was
    0.000048 mm at 1x and 0.000032 mm at 3x -- for the same underlying curves.
    A fixed 0.01 mm cutoff therefore kept 6% of the arcs at 1x and **0%** at
    3x, which is how the arcs nearly disappeared from the export while every
    geometry test still passed.

    Scaling the reference with the artwork keeps the answer the same at any
    magnification, because both the radius and the reference scale together.
    """
    from pcbart.vtrace import drop_shallow_arcs

    # A 10 mm chord swept 10 degrees is a 57.4 mm radius -- a real curve on a
    # 10 mm mark, and it stays a curve whether the artwork is measured as 10 mm
    # or as 100 mm.
    gentle = ArcRing([(0.0, 0.0), (10.0, 0.0)], [10.0, 0.0])
    assert drop_shallow_arcs(gentle, reference_mm=10.0).sweeps[0] == 10.0
    assert drop_shallow_arcs(gentle, reference_mm=100.0).sweeps[0] == 10.0

    # The same chord swept 0.05 degrees is a 11.5 metre radius: a straight edge
    # at either scale.
    absurd = ArcRing([(0.0, 0.0), (10.0, 0.0)], [0.05, 0.0])
    assert drop_shallow_arcs(absurd, reference_mm=10.0).sweeps[0] == 0.0
    assert drop_shallow_arcs(absurd, reference_mm=100.0).sweeps[0] == 0.0
