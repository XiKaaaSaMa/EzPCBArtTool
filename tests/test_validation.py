"""Outline validation tests: geometry rules and process limits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.shapes import regular_star_points  # noqa: E402
from pcbart.validation import (  # noqa: E402
    MIN_BOARD_SIDE_MM,
    MIN_NECK_WIDTH_MM,
    check_outline,
    check_trace_width,
    min_neck_width_mm,
)


def test_valid_square_passes():
    square = [(0, 0), (50, 0), (50, 30), (0, 30)]
    report = check_outline(square)

    assert report.ok
    assert report.width_mm == pytest.approx(50)
    assert report.height_mm == pytest.approx(30)
    assert report.vertex_count == 4
    assert report.perimeter_mm == pytest.approx(160)


def test_star_outline_is_valid():
    points = regular_star_points(points=5, edge_length=30.0)
    report = check_outline(points, min_side_mm=1.0)

    assert report.ok
    assert report.vertex_count == 10
    assert report.edge_count == 10
    assert report.min_edge_mm == pytest.approx(30.0, rel=1e-9)


def test_board_below_minimum_size_is_an_error():
    tiny = [(0, 0), (2, 0), (2, 2), (0, 2)]
    report = check_outline(tiny)

    assert not report.ok
    assert any(issue.code == "board-too-small" for issue in report.errors)
    assert MIN_BOARD_SIDE_MM == 3.0


def test_self_intersecting_outline_is_rejected():
    """A bow-tie ring crosses itself and cannot become a board."""
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    report = check_outline(bowtie, min_side_mm=1.0)

    assert not report.ok
    assert any(
        issue.code in ("invalid-polygon", "self-intersecting") for issue in report.errors
    )


def test_duplicate_consecutive_vertices_are_a_zero_length_edge():
    ring = [(0, 0), (50, 0), (50, 0), (50, 30), (0, 30)]
    report = check_outline(ring)

    assert not report.ok
    assert any(issue.code == "zero-length-edge" for issue in report.errors)


def test_too_few_vertices_is_rejected():
    report = check_outline([(0, 0), (10, 0)])
    assert not report.ok
    assert any(issue.code == "too-few-vertices" for issue in report.errors)


def test_short_edge_is_a_warning_not_an_error():
    """A 0.1 mm step is below the routable minimum but the shape stays valid.

    The ring has one deliberately tiny edge between (5, 0) and (5.1, 0).
    """
    ring = [(0, 0), (5, 0), (5.1, 0), (10, 0), (10, 10), (0, 10)]
    report = check_outline(ring, min_side_mm=1.0)

    assert report.ok, [str(i) for i in report.errors]
    assert report.min_edge_mm == pytest.approx(0.1, abs=1e-9)
    assert any(issue.code == "short-edge" for issue in report.warnings)
    assert all(issue.severity == "warning" for issue in report.warnings)


def test_repeated_closing_vertex_is_tolerated():
    """A ring that explicitly repeats its first point is normalised, not flagged."""
    ring = [(0, 0), (50, 0), (50, 30), (0, 30), (0, 0)]
    report = check_outline(ring)

    assert report.ok
    assert report.vertex_count == 4


def test_trace_width_below_process_minimum():
    issues = check_trace_width(0.05)
    assert issues
    assert issues[0].code == "trace-too-thin"


def test_trace_width_at_process_minimum_passes():
    assert check_trace_width(0.10) == []
    assert check_trace_width(0.254) == []


def test_report_summary_is_readable():
    points = regular_star_points(points=5, edge_length=30.0)
    summary = check_outline(points, min_side_mm=1.0).summary()

    assert "outline:" in summary
    assert "shortest edge" in summary
    assert "no issues" in summary


# ------------------------------------------------------------ narrowest neck


def test_uniform_bar_reports_its_width():
    """Erosion of a 3 mm bar empties at r = 1.5, so the width is 3 mm."""
    rectangle = [(0, 0), (40, 0), (40, 3), (0, 3)]
    assert min_neck_width_mm(rectangle) == pytest.approx(3.0, abs=0.05)


def test_dumbbell_finds_the_neck_not_the_blocks():
    """The 1 mm bar between two 10 mm blocks is what breaks first."""
    ring = [
        (0, 0),
        (10, 0),
        (10, 4.5),
        (20, 4.5),
        (20, 0),
        (30, 0),
        (30, 10),
        (20, 10),
        (20, 5.5),
        (10, 5.5),
        (10, 10),
        (0, 10),
    ]
    assert min_neck_width_mm(ring) == pytest.approx(1.0, abs=0.1)


def test_thin_sliver_is_flagged():
    sliver = [(0, 0), (50, 0), (50, 0.2), (0, 0.2)]
    report = check_outline(sliver, min_side_mm=0.1)

    assert report.ok, "a thin neck is a handling risk, not invalid geometry"
    assert report.min_neck_mm == pytest.approx(0.2, abs=0.05)
    assert any(issue.code == "thin-neck" for issue in report.warnings)


def test_star_has_no_thin_neck_warning():
    """A star's concave tips taper to zero width, but they are not necks.

    This is the case that makes medial-axis thickness unusable here: such a
    measure would report every star as a sliver. Erosion-based detection ignores
    the tips because eroding them never splits the body.
    """
    points = regular_star_points(points=5, edge_length=30.0)
    report = check_outline(points, min_side_mm=1.0)

    assert report.min_neck_mm is not None
    assert not any(issue.code == "thin-neck" for issue in report.issues)
    assert report.min_neck_mm > MIN_NECK_WIDTH_MM


def test_letter_with_no_neck_is_not_flagged():
    """A letter P has a thick bowl and stem but no narrow waist."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    cv2 = pytest.importorskip("cv2")

    from pcbart.image_prep import PrepParams, preprocess, make_test_image
    from pcbart.outline import OutlineParams, extract_outline_rings

    image = make_test_image(shape="letter")
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask, OutlineParams(target_width_mm=60.0)
    )

    report = check_outline(result.outer, min_side_mm=1.0)
    assert not any(issue.code == "thin-neck" for issue in report.issues)


def test_neck_check_can_be_disabled():
    sliver = [(0, 0), (50, 0), (50, 0.2), (0, 0.2)]
    report = check_outline(sliver, min_side_mm=0.1, min_neck_mm=None)

    assert report.min_neck_mm is None
    assert not any(issue.code == "thin-neck" for issue in report.issues)
