"""Log-format tests: record splitting, round trips, and section handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import (  # noqa: E402
    EproArchive,
    EproDocument,
    Record,
    new_id,
    split_record_line,
)
from pcbart.layers import DEFAULT_LAYERS  # noqa: E402
from pcbart.templates import build_minimal_project  # noqa: E402


def test_split_simple_record():
    line = '{"type":"POLY","ticket":351,"id":"b5a1"}||{"layerId":11,"width":5}|'
    split = split_record_line(line)
    assert split is not None
    header, body = split
    assert json.loads(header)["type"] == "POLY"
    assert json.loads(body.rstrip("|"))["layerId"] == 11


def test_split_empty_body():
    """Deleted records carry an empty payload; the terminator must survive."""
    line = '{"type":"RULE_SELECTOR","ticket":1,"id":"x"}|||'
    split = split_record_line(line)
    assert split is not None
    header, body = split
    assert json.loads(header)["type"] == "RULE_SELECTOR"
    assert body.rstrip("|") == ""


def test_split_handles_json_array_id():
    """LAYER records carry a serialised JSON array as their id."""
    line = '{"type":"LAYER","ticket":1803,"id":"[\\"LAYER\\",1]"}||{"layerId":1}|'
    split = split_record_line(line)
    assert split is not None
    header, _ = split
    parsed = json.loads(header)
    assert json.loads(parsed["id"]) == ["LAYER", 1]


def test_empty_body_is_deleted():
    record = Record(header={"type": "X", "ticket": 1}, body=None)
    assert record.is_deleted
    assert Record(header={"type": "X", "ticket": 1}, body="").is_deleted
    assert not Record(header={"type": "X", "ticket": 1}, body={}).is_deleted


def test_parse_skips_blank_lines_and_reports_bad_ones():
    text = '\n{"type":"META","ticket":1,"id":"META"}||{"a":1}|\n\nnot json\n'
    doc = EproDocument.loads(text)
    assert len(doc.records) == 1
    assert doc.unparsed == ["not json"]


def test_dumps_is_newline_terminated():
    doc = EproDocument([Record({"type": "META", "ticket": 1, "id": "META"}, {"a": 1})])
    assert doc.dumps().endswith("\n")


def test_round_trip_preserves_records():
    original = '{"type":"META","ticket":1,"id":"META"}||{"name":"x"}|'
    doc = EproDocument.loads(original)
    reparsed = EproDocument.loads(doc.dumps())
    assert reparsed.records[0].body == {"name": "x"}
    assert reparsed.records[0].type == "META"


def test_ticket_increases_monotonically():
    doc = EproDocument.loads('{"type":"META","ticket":7,"id":"META"}||{"a":1}|')
    record = doc.append("LINE", {"width": 1}, record_id=new_id())
    assert record.ticket == 8


def test_append_to_section_places_record_inside_that_section():
    """A record belongs to the DOCHEAD preceding it, so it must land in PCB.

    Inserting before BLOB is not enough: CONFIG sits between PCB and BLOB, so
    a naively appended primitive would be filed under CONFIG and the editor
    would not see it as board geometry.
    """
    doc, _ = build_minimal_project("T")
    doc.append_to_section(doc.new_record("POLY", {"layerId": 11}))

    sections = doc.sections()
    assert any(r.type == "POLY" for r in sections["PCB"])
    assert not any(r.type == "POLY" for r in sections["CONFIG"])


def test_append_to_section_inserts_before_the_next_document_head():
    doc, _ = build_minimal_project("T")
    doc.append_to_section(doc.new_record("POLY", {"layerId": 11}), doc_type="PCB")

    index = next(i for i, r in enumerate(doc.records) if r.type == "POLY")
    following = doc.records[index + 1]
    assert following.type == "DOCHEAD"
    assert following.body["docType"] == "CONFIG"


def test_append_to_section_rejects_unknown_section():
    doc, _ = build_minimal_project("T")
    with pytest.raises(ValueError):
        doc.append_to_section(doc.new_record("POLY", {}), doc_type="NOPE")


def test_sections_group_by_document():
    doc, _ = build_minimal_project("T")
    sections = doc.sections()
    assert "PCB" in sections
    assert "BOARD" in sections
    pcb_types = {r.type for r in sections["PCB"]}
    assert "LAYER" in pcb_types
    assert "CANVAS" in pcb_types


def test_minimal_project_declares_every_layer():
    doc, _ = build_minimal_project("T")
    layer_ids = {r.body["layerId"] for r in doc.iter_type("LAYER")}
    assert layer_ids == {layer.id for layer in DEFAULT_LAYERS}
    assert 11 in layer_ids


def test_layer_record_id_matches_editor_convention():
    doc, _ = build_minimal_project("T")
    outline = next(r for r in doc.iter_type("LAYER") if r.body["layerId"] == 11)
    assert json.loads(outline.id) == ["LAYER", 11]
    assert outline.body["layerType"] == "OUTLINE"


def test_document_heads_are_ordered():
    doc, _ = build_minimal_project("T")
    order = [d["docType"] for d in doc.documents()]
    assert order.index("BOARD") < order.index("PCB") < order.index("BLOB")


def test_archive_round_trip(tmp_path: Path):
    doc, _ = build_minimal_project("RoundTrip")
    doc.append_to_section(
        doc.new_record("POLY", {"layerId": 11, "width": 4.9213, "path": [0, 0, "L", 10, 0]})
    )

    path = tmp_path / "RoundTrip.epro2"
    EproArchive(doc, {"title": "RoundTrip"}).write(path)
    assert path.exists()

    loaded = EproArchive.read(path)
    assert loaded.meta["title"] == "RoundTrip"
    assert any(r.type == "POLY" for r in loaded.document.records)
    assert loaded.document.unparsed == []


def test_archive_contains_epru_and_project_json(tmp_path: Path):
    import zipfile

    doc, _ = build_minimal_project("Files")
    path = tmp_path / "Files.epro2"
    EproArchive(doc, {"title": "Files"}).write(path)

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    assert "project2.json" in names
    assert "Files.epru" in names


def test_partial_metadata_is_completed(tmp_path: Path):
    """Passing only a title must still produce a complete project2.json.

    EasyEDA Pro reads several fields from this file; an archive written with
    just {"title": ...} is short but not valid, so the defaults are merged in.
    """
    import json
    import zipfile

    doc, _ = build_minimal_project("Partial")
    path = tmp_path / "Partial.epro2"
    EproArchive(doc, {"title": "Partial"}).write(path)

    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("project2.json").decode("utf-8"))

    assert meta["title"] == "Partial"
    for key in ("cbb_project", "editorVersion", "introduction", "description", "tags"):
        assert key in meta, f"project2.json missing {key}"


def test_archive_accepts_no_metadata_at_all(tmp_path: Path):
    import json
    import zipfile

    doc, _ = build_minimal_project("NoMeta")
    path = tmp_path / "NoMeta.epro2"
    EproArchive(doc).write(path)

    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("project2.json").decode("utf-8"))
    assert meta["title"] == "NoMeta"
    assert "cbb_project" in meta
