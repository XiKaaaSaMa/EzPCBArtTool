"""Regression tests against records extracted from a real EasyEDA Pro project.

``tests/fixtures/real_sample.epru`` holds the document heads, layer table,
board outline and dimension records from a project written by EasyEDA Pro
3.2.149. The dimension record is what makes the format self-calibrating, so
these tests pin down both the parser and the unit factor using real data rather
than values we chose ourselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproDocument  # noqa: E402
from pcbart.primitives import outline_bbox_mm, record_to_mm  # noqa: E402
from pcbart.units import EPRU_UNIT_MM, unit_to_mm  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real_sample.epru"


@pytest.fixture(scope="module")
def document() -> EproDocument:
    return EproDocument.read(FIXTURE)


def test_fixture_parses_cleanly(document: EproDocument):
    assert document.unparsed == []
    assert len(document.records) > 150


def test_fixture_declares_board_and_pcb(document: EproDocument):
    doc_types = [d["docType"] for d in document.documents()]
    assert "BOARD" in doc_types
    assert "PCB" in doc_types


def test_layer_table_is_complete(document: EproDocument):
    layers = document.find("LAYER")
    assert len(layers) >= 60
    ids = {r.body["layerId"] for r in layers}
    assert 11 in ids
    outline = next(r for r in layers if r.body["layerId"] == 11)
    assert outline.body["layerType"] == "OUTLINE"


def test_dimension_record_self_calibrates_the_unit(document: EproDocument):
    """The core M0 result: raw span vs. the length EasyEDA annotated itself."""
    dimensions = [
        r for r in document.iter_type("DIMENSION") if r.body.get("type") == "LENGTH"
    ]
    assert dimensions, "fixture should contain a LENGTH dimension"

    record = dimensions[0]
    coords = record.body["coords"]
    declared = float(record.body["text"]["text"].rstrip("mm"))

    xs = coords[0::2]
    ys = coords[1::2]
    span = max(max(xs) - min(xs), max(ys) - min(ys))

    # The annotation is rounded to 3 decimals (99.873 mm), so the implied
    # factor is only accurate to roughly 1e-6 relative.
    implied = declared / span
    assert implied == pytest.approx(EPRU_UNIT_MM, rel=1e-6)
    assert implied == pytest.approx(0.0254, rel=1e-6)


def test_unit_factor_rules_out_the_alternatives(document: EproDocument):
    """Explicitly reject the two readings the spec text could suggest."""
    record = next(r for r in document.iter_type("DIMENSION") if r.body.get("type") == "LENGTH")
    coords = record.body["coords"]
    declared = float(record.body["text"]["text"].rstrip("mm"))
    xs, ys = coords[0::2], coords[1::2]
    implied = declared / max(max(xs) - min(xs), max(ys) - min(ys))

    assert abs(implied - 0.254) > 0.01, "would mean the spec's 0.01 inch reading"
    assert abs(implied - 0.01) > 0.001, "would mean a 0.01 mm reading"
    assert abs(implied - 0.0254) < 1e-6


def test_dimension_text_is_99_873_mm(document: EproDocument):
    record = next(r for r in document.iter_type("DIMENSION") if "text" in r.body)
    assert record.body["text"]["text"] == "99.873mm"


def test_board_outline_is_on_layer_11(document: EproDocument):
    outlines = [
        r for r in document.iter_type("POLY") if r.body.get("layerId") == 11
    ]
    assert len(outlines) == 10
    for record in outlines:
        assert record.body["polyType"] == "NORMAL"


def test_board_outline_measures_49_873_by_99_873_mm(document: EproDocument):
    """Independent check: the outline's own size must match the annotation."""
    outlines = [r for r in document.iter_type("POLY") if r.body.get("layerId") == 11]
    bbox = outline_bbox_mm(outlines)
    assert bbox is not None

    min_x, min_y, max_x, max_y = bbox
    assert (max_x - min_x) == pytest.approx(49.873, abs=1e-3)
    assert (max_y - min_y) == pytest.approx(99.873, abs=1e-3)


def test_outline_width_is_0_125_mm(document: EproDocument):
    outlines = [r for r in document.iter_type("POLY") if r.body.get("layerId") == 11]
    for record in outlines:
        assert record.body["width"] == pytest.approx(4.9213)
        assert unit_to_mm(record.body["width"]) == pytest.approx(0.125, abs=1e-5)


def test_outline_path_conversion_preserves_y_order(document: EproDocument):
    """Y is not negated: the file's Y axis already points up.

    A raw outline path here runs 2919 -> -930, and that ordering must survive
    conversion. Negating it was the bug that mirrored generated boards; the
    evidence is recorded on ``pcbart.units.FILE_Y_IS_UP``.
    """
    outlines = [r for r in document.iter_type("POLY") if r.body.get("layerId") == 11]
    first = outlines[0]
    converted = record_to_mm(first)
    raw_path = first.body["path"]

    assert raw_path[1] > 0 > raw_path[4]
    assert converted["path"][1] == pytest.approx(unit_to_mm(raw_path[1]), abs=1e-6)
    assert converted["path"][4] == pytest.approx(unit_to_mm(raw_path[4]), abs=1e-6)
    assert converted["path"][1] > 0 > converted["path"][4]


def test_arc_angle_survives_conversion(document: EproDocument):
    """ARC tokens carry an angle that must not be scaled or flipped."""
    arcs = [
        r
        for r in document.iter_type("POLY")
        if r.body.get("layerId") == 11 and "ARC" in r.body.get("path", [])
    ]
    assert arcs
    converted = record_to_mm(arcs[0])
    path = converted["path"]
    assert "ARC" in path
    angle = path[path.index("ARC") + 1]
    assert angle == pytest.approx(93.3779)


def test_fill_width_is_a_stroke_not_a_track(document: EproDocument):
    """FILL.width is a stroke width (can be ~0); it must not be read as a trace."""
    fills = list(document.iter_type("FILL"))
    assert fills
    widths = {r.body["width"] for r in fills}
    assert widths.issubset({0.2, 1})


def test_via_records_are_physical_sizes(document: EproDocument):
    """The project's vias are 4.01 mm mounting holes, not 40 mm craters.

    Under a 0.254 mm/unit reading these would come out at 40.1 mm, which would
    mean the board could not physically contain them.
    """
    vias = [r for r in document.iter_type("VIA") if "viaDiameter" in r.body]
    assert vias
    for record in vias:
        assert unit_to_mm(record.body["viaDiameter"]) == pytest.approx(4.01, abs=1e-4)
        assert unit_to_mm(record.body["holeDiameter"]) == pytest.approx(3.56, abs=1e-4)


def test_preference_holds_the_editor_defaults(document: EproDocument):
    """Independent anchor: EasyEDA's default 10 mil track and 0.61 mm via."""
    record = next(iter(document.iter_type("PREFERENCE")), None)
    assert record is not None
    body = record.body

    assert unit_to_mm(body["lastTrackWidth"]) == pytest.approx(0.254, abs=1e-6)
    assert unit_to_mm(body["lastViaDiameter"]) == pytest.approx(0.61, abs=1e-5)
    assert unit_to_mm(body["lastViaInnerDiameter"]) == pytest.approx(0.305, abs=1e-5)
