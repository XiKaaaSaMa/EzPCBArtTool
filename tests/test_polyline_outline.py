"""End-to-end tests for arbitrary polyline board outlines.

These cover the star board: an irregular outline that the rectangle path never
exercised, including free-angle vertices, a ten-vertex ring, and picking the
geometry back out of the written file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive  # noqa: E402
from pcbart.generator import generate_polyline_board, outline_record  # noqa: E402
from pcbart.geometry import is_ccw  # noqa: E402
from pcbart.primitives import Mode, record_to_mm  # noqa: E402
from pcbart.shapes import (  # noqa: E402
    bounding_box,
    center_on_origin,
    polygon_edge_lengths,
    regular_star_points,
)
from pcbart.templates import build_minimal_project  # noqa: E402


def read_outline(path: Path) -> list[tuple[float, float]]:
    """Pull the layer-11 polyline vertices back out of a written board."""
    doc = EproArchive.read(path).document
    record = next(
        r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11
    )
    converted = record_to_mm(record)
    tokens = converted["path"]

    points: list[tuple[float, float]] = []
    pending: list[float] = []
    index = 0
    while index < len(tokens):
        item = tokens[index]
        if isinstance(item, str):
            points.extend(
                (pending[i], pending[i + 1]) for i in range(0, len(pending) - 1, 2)
            )
            pending = []
            index += 1
            if item in (Mode.ARC, Mode.CARC):
                index += 3
            continue
        pending.append(float(item))
        index += 1
    points.extend((pending[i], pending[i + 1]) for i in range(0, len(pending) - 1, 2))
    return points


def test_star_file_is_written(tmp_path: Path):
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(points, tmp_path / "star.epro2", title="Star")
    assert path.exists()


def test_star_outline_is_stored_on_layer_11(tmp_path: Path):
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(points, tmp_path / "star.epro2", title="Star")
    doc = EproArchive.read(path).document

    outlines = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]
    assert len(outlines) == 1
    assert outlines[0].body["polyType"] == "NORMAL"


def test_star_edges_survive_the_file_round_trip(tmp_path: Path):
    """The decisive test: ten 30 mm edges after write and read back.

    EasyEDA Pro stores coordinates to four decimals, which is about 1.3e-6 mm,
    so a small tolerance is expected and much tighter ones are not meaningful.
    """
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(
        points, tmp_path / "star.epro2", title="Star", center=True
    )

    recovered = read_outline(path)
    # Drop the repeated closing vertex if the writer added one.
    if recovered and recovered[0] == recovered[-1]:
        recovered = recovered[:-1]

    assert len(recovered) == 10
    lengths = polygon_edge_lengths(recovered, closed=True)
    assert len(lengths) == 10
    for length in lengths:
        assert length == pytest.approx(30.0, abs=1e-5)


def test_star_outline_is_not_degenerate(tmp_path: Path):
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(points, tmp_path / "star.epro2", title="Star")

    recovered = read_outline(path)
    if recovered and recovered[0] == recovered[-1]:
        recovered = recovered[:-1]

    min_x, min_y, max_x, max_y = bounding_box(recovered)
    assert (max_x - min_x) > 70.0
    assert (max_y - min_y) > 70.0
    assert is_ccw(recovered)


def test_star_stays_equilateral_after_centering(tmp_path: Path):
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(
        points, tmp_path / "star.epro2", title="Star", center=True
    )

    recovered = read_outline(path)
    if recovered and recovered[0] == recovered[-1]:
        recovered = recovered[:-1]

    min_x, min_y, max_x, max_y = bounding_box(recovered)
    assert (min_x + max_x) == pytest.approx(0.0, abs=1e-4)
    assert (min_y + max_y) == pytest.approx(0.0, abs=1e-4)


def test_expected_edge_length_is_enforced(tmp_path: Path):
    """A shape that is not equilateral must be rejected before writing."""
    lopsided = regular_star_points(points=5, edge_length=30.0)
    lopsided[3] = (lopsided[3][0] * 1.5, lopsided[3][1])

    with pytest.raises(ValueError, match="not equilateral"):
        generate_polyline_board(
            lopsided,
            tmp_path / "bad.epro2",
            title="Bad",
            expected_edge_mm=30.0,
        )


def test_invalid_outline_is_rejected_before_writing(tmp_path: Path):
    """A self-intersecting outline must fail validation rather than be written."""
    bowtie = [(0, 0), (50, 50), (50, 0), (0, 50)]
    with pytest.raises(ValueError, match="failed validation"):
        generate_polyline_board(bowtie, tmp_path / "bowtie.epro2", title="Bad")


def test_too_few_vertices_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        generate_polyline_board([(0, 0), (10, 0)], tmp_path / "x.epro2", title="X")


def test_polyline_uses_free_angle_corner_mode(tmp_path: Path):
    """The board declares the free-angle polyline tool in its preferences."""
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(
        points, tmp_path / "star.epro2", title="Star", routing_corner="L"
    )
    doc = EproArchive.read(path).document

    preference = next(iter(doc.iter_type("PREFERENCE")), None)
    assert preference is not None
    assert preference.body["routingCorner"] == "L"


def test_outline_record_rejects_degenerate_input():
    doc, _ = build_minimal_project("T")
    with pytest.raises(ValueError):
        outline_record([(0, 0), (1, 1)], doc)


def test_outline_record_path_encodes_L_mode():
    doc, _ = build_minimal_project("T")
    record = outline_record([(0, 0), (10, 0), (10, 10)], doc)
    assert record.body["path"][2] == "L"


def test_generated_star_parses_with_no_unparsed_lines(tmp_path: Path):
    points = regular_star_points(points=5, edge_length=30.0)
    path = generate_polyline_board(points, tmp_path / "star.epro2", title="Star")
    assert EproArchive.read(path).document.unparsed == []
