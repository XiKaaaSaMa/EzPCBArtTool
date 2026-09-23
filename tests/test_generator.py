"""End-to-end tests for board generation.

These cover the M0 deliverable: a ``.epro2`` containing a board outline of a
known size, with the geometry verified after a full write/read cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive  # noqa: E402
from pcbart.generator import BoardSpec, generate_board_with_outline, generate_rect_board  # noqa: E402
from pcbart.primitives import outline_bbox_mm, record_to_mm  # noqa: E402
from pcbart.units import mm_to_unit, unit_to_mm  # noqa: E402


def test_generates_a_file(tmp_path: Path):
    path = generate_rect_board(50, 30, tmp_path / "board.epro2", title="Board")
    assert path.exists()
    assert path.stat().st_size > 0


def test_outline_lands_on_layer_11(tmp_path: Path):
    path = generate_rect_board(50, 30, tmp_path / "board.epro2", title="Board")
    doc = EproArchive.read(path).document

    outlines = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]
    assert len(outlines) == 1


def test_outline_is_closed(tmp_path: Path):
    path = generate_rect_board(50, 30, tmp_path / "board.epro2", title="Board")
    doc = EproArchive.read(path).document
    outline = next(r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11)

    converted = record_to_mm(outline)
    path_tokens = converted["path"]
    first = (path_tokens[0], path_tokens[1])
    tail = [t for t in path_tokens if isinstance(t, (int, float))]
    last = (tail[-2], tail[-1])
    assert first == pytest.approx(last)


def test_outline_dimensions_survive_round_trip(tmp_path: Path):
    """The whole point of M0: 50x30 mm in must be 50x30 mm out."""
    path = generate_rect_board(50, 30, tmp_path / "board.epro2", title="Board")
    doc = EproArchive.read(path).document
    outline = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]

    bbox = outline_bbox_mm(outline)
    assert bbox is not None
    min_x, min_y, max_x, max_y = bbox
    assert (max_x - min_x) == pytest.approx(50.0, abs=1e-3)
    assert (max_y - min_y) == pytest.approx(30.0, abs=1e-3)


def test_outline_width_is_written_in_units(tmp_path: Path):
    path = generate_rect_board(
        20, 20, tmp_path / "board.epro2", title="Board", line_width_mm=0.125
    )
    doc = EproArchive.read(path).document
    outline = next(r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11)

    assert outline.body["width"] == pytest.approx(mm_to_unit(0.125), abs=1e-4)
    assert unit_to_mm(outline.body["width"]) == pytest.approx(0.125, abs=1e-5)


def test_generated_file_has_no_unparsed_lines(tmp_path: Path):
    path = generate_rect_board(10, 10, tmp_path / "board.epro2", title="Board")
    assert EproArchive.read(path).document.unparsed == []


def test_generated_file_has_expected_sections(tmp_path: Path):
    path = generate_rect_board(10, 10, tmp_path / "board.epro2", title="Board")
    doc = EproArchive.read(path).document
    doc_types = [d["docType"] for d in doc.documents()]
    assert "BOARD" in doc_types
    assert "PCB" in doc_types
    assert "BLOB" in doc_types


def test_outline_is_filed_under_the_pcb_document(tmp_path: Path):
    """Geometry must live inside the PCB section, not merely before BLOB.

    CONFIG sits between PCB and BLOB, so an outline appended before BLOB would
    be attributed to CONFIG and the editor would not treat it as board data.
    """
    path = generate_rect_board(10, 10, tmp_path / "board.epro2", title="Board")
    doc = EproArchive.read(path).document
    sections = doc.sections()

    assert any(r.type == "POLY" for r in sections["PCB"])
    assert not any(r.type == "POLY" for r in sections["CONFIG"])
    assert not any(r.type == "POLY" for r in sections["BLOB"])


def test_metadata_title_is_preserved(tmp_path: Path):
    path = generate_rect_board(10, 10, tmp_path / "board.epro2", title="MyArtBoard")
    assert EproArchive.read(path).meta["title"] == "MyArtBoard"


def test_origin_offset_is_applied(tmp_path: Path):
    spec = BoardSpec(width_mm=10, height_mm=5, origin_x_mm=20, origin_y_mm=7, title="Offset")
    path = generate_board_with_outline(spec, tmp_path / "offset.epro2")
    doc = EproArchive.read(path).document
    outline = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]

    bbox = outline_bbox_mm(outline)
    assert bbox is not None
    assert bbox[0] == pytest.approx(20.0, abs=1e-3)
    assert bbox[1] == pytest.approx(7.0, abs=1e-3)


@pytest.mark.parametrize(
    "width,height",
    [(3, 3), (10, 10), (100, 50), (0.5, 200)],
)
def test_various_board_sizes(tmp_path: Path, width: float, height: float):
    path = generate_rect_board(width, height, tmp_path / "b.epro2", title="B")
    doc = EproArchive.read(path).document
    outline = [r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11]
    bbox = outline_bbox_mm(outline)
    assert bbox is not None
    assert (bbox[2] - bbox[0]) == pytest.approx(width, abs=1e-3)
    assert (bbox[3] - bbox[1]) == pytest.approx(height, abs=1e-3)


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-5, 10)])
def test_invalid_dimensions_are_rejected(tmp_path: Path, width: float, height: float):
    with pytest.raises(ValueError):
        generate_rect_board(width, height, tmp_path / "bad.epro2", title="Bad")
