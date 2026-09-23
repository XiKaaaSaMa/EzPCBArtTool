#!/usr/bin/env python
"""M0 calibration: derive the ``.epru`` unit factor and Y-axis direction.

Feed it a real ``.epro2`` (or an already-extracted ``.epru``) produced by
EasyEDA Pro and it will report the evidence needed to lock down the format:

* the unit factor, cross-checked against any dimension annotation the project
  contains (EasyEDA writes the measured length as text, which makes the
  project self-calibrating)
* the Y-axis direction
* the record layout of board outlines

Usage::

    python tools/m0_calibrate.py path/to/project.epro2
    python tools/m0_calibrate.py path/to/project.epru --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402
from pcbart.primitives import record_to_mm  # noqa: E402

CANDIDATE_FACTORS: dict[str, float] = {
    "0.01 inch (0.254 mm)": 0.254,
    "1 mil (0.0254 mm)": 0.0254,
    "0.01 mm": 0.01,
    "1 thou rounded (0.025 mm)": 0.025,
}

DIMENSION_TEXT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*mm\s*$", re.IGNORECASE)


@dataclass
class DimensionCheck:
    """A length dimension whose text tells us the true length."""

    raw_span: float
    declared_mm: float
    implied_factor: float
    layer_id: int | None = None
    dimension_type: str | None = None


@dataclass
class CalibrationReport:
    source: str
    record_count: int
    unparsed_lines: int
    document_types: list[str] = field(default_factory=list)
    type_counts: dict[str, int] = field(default_factory=dict)
    dimensions: list[DimensionCheck] = field(default_factory=list)
    implied_factor_median: float | None = None
    best_match: str | None = None
    best_match_error_pct: float | None = None
    y_axis_evidence: list[str] = field(default_factory=list)
    outline_records: list[dict] = field(default_factory=list)
    outline_bbox_mm: tuple[float, float, float, float] | None = None
    narrowest_line_widths_mm: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def load_document(path: Path) -> tuple[EproDocument, str]:
    if path.suffix.lower() == ".epro2":
        archive = EproArchive.read(path)
        return archive.document, f"epro2 (title={archive.meta.get('title')!r})"
    return EproDocument.read(path), "epru"


def find_dimensions(document: EproDocument) -> list[DimensionCheck]:
    """Cross-check every length dimension against its annotated text.

    A ``DIMENSION`` record carries both the raw coordinates it spans and a text
    string describing the measured distance in millimetres. Dividing the two
    yields the unit factor directly, with no assumptions.
    """
    checks: list[DimensionCheck] = []

    for record in document.iter_type("DIMENSION"):
        body = record.body
        if not isinstance(body, dict):
            continue
        if str(body.get("type", "")).upper() != "LENGTH":
            continue
        coords = body.get("coords")
        text_block = body.get("text")
        if not isinstance(coords, list) or not isinstance(text_block, dict):
            continue
        match = DIMENSION_TEXT.match(str(text_block.get("text", "")))
        if not match:
            continue
        declared = float(match.group(1))

        xs = [float(v) for i, v in enumerate(coords) if i % 2 == 0 and isinstance(v, (int, float))]
        ys = [float(v) for i, v in enumerate(coords) if i % 2 == 1 and isinstance(v, (int, float))]
        span_x = max(xs) - min(xs) if xs else 0.0
        span_y = max(ys) - min(ys) if ys else 0.0
        span = max(span_x, span_y)
        if span <= 0 or declared <= 0:
            continue

        checks.append(
            DimensionCheck(
                raw_span=span,
                declared_mm=declared,
                implied_factor=declared / span,
                layer_id=body.get("layerId"),
                dimension_type=body.get("type"),
            )
        )
    return checks


def analyse_y_axis(document: EproDocument) -> list[str]:
    """Collect structural evidence about which way Y grows in the file."""
    evidence: list[str] = []

    for record in document.iter_type("DIMENSION"):
        body = record.body
        if not isinstance(body, dict) or str(body.get("type", "")).upper() != "LENGTH":
            continue
        text_block = body.get("text")
        coords = body.get("coords")
        if not isinstance(text_block, dict) or not isinstance(coords, list) or len(coords) < 4:
            continue
        match = DIMENSION_TEXT.match(str(text_block.get("text", "")))
        if not match:
            continue
        # The label is placed at the midpoint of the measured run; its sign
        # relative to the endpoints reveals the screen direction.
        label_y = text_block.get("y")
        ys = [float(v) for i, v in enumerate(coords) if i % 2 == 1 and isinstance(v, (int, float))]
        if isinstance(label_y, (int, float)) and ys:
            mid = (max(ys) + min(ys)) / 2
            evidence.append(
                f"dimension label y={label_y} midpoint of measured span y={mid:.4f} "
                f"(file units; label {'below' if label_y > mid else 'above'} its span)"
            )
        break

    evidence.append(
        "EasyEDA Pro screen coordinates grow downward; the KiCad importer negates Y "
        "when reading (`ScalePos` returns -ScaleSize(y)), so file Y is screen-down."
    )
    return evidence


TRACK_TYPES = ("LINE", "ARC", "POLY")


def narrowest_widths(document: EproDocument, limit: int = 8) -> list[float]:
    """Distinct track widths in mm, smallest first.

    Sanity check: real boards use track and outline widths in the 0.1-1 mm
    range, so a factor that is off by 10x or 2.54x shows up immediately.

    Only ``LINE``/``ARC``/``POLY`` are considered. ``FILL`` and ``POUR`` also
    carry a ``width`` field, but for those it is an outline-stroke width that
    is legitimately near zero and would otherwise trip the check.
    """
    from pcbart.units import unit_to_mm

    widths: set[float] = set()
    for record in document.records:
        if record.type not in TRACK_TYPES:
            continue
        body = record.body
        if not isinstance(body, dict):
            continue
        value = body.get("width")
        if isinstance(value, (int, float)) and value > 0:
            widths.add(round(unit_to_mm(float(value)), 4))
    return sorted(widths)[:limit]


def collect_outline(document: EproDocument, limit: int = 12) -> list[dict]:
    """Board-outline primitives (layer 11) converted to millimetres."""
    out: list[dict] = []
    doc_type = None

    for record in document.records:
        if record.type == "DOCHEAD" and isinstance(record.body, dict):
            doc_type = record.body.get("docType")
            continue
        if doc_type != "PCB":
            continue
        body = record.body
        if not isinstance(body, dict) or body.get("layerId") != 11:
            continue
        if record.type not in ("POLY", "LINE", "ARC", "FILL"):
            continue
        converted = record_to_mm(record)
        if converted is None:
            continue
        entry = {"record_type": record.type, "id": record.id}
        for key in ("startX", "startY", "endX", "endY", "width"):
            if key in converted:
                entry[key] = round(float(converted[key]), 4)
        if "path" in converted:
            entry["path_preview"] = json.dumps(converted["path"])[:180]
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def outline_bbox(document: EproDocument) -> tuple[float, float, float, float] | None:
    from pcbart.primitives import outline_bbox_mm

    records = [
        r
        for r in document.records
        if isinstance(r.body, dict)
        and r.body.get("layerId") == 11
        and r.type in ("POLY", "LINE", "ARC")
    ]
    return outline_bbox_mm(records)


def calibrate(path: Path) -> CalibrationReport:
    document, source = load_document(path)

    report = CalibrationReport(
        source=f"{path.name} [{source}]",
        record_count=len(document.records),
        unparsed_lines=len(document.unparsed),
        document_types=[
            str(d.get("docType")) for d in document.documents()
        ],
    )

    counts: dict[str, int] = {}
    for record in document.records:
        counts[record.type] = counts.get(record.type, 0) + 1
    report.type_counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    report.dimensions = find_dimensions(document)
    if report.dimensions:
        factors = [d.implied_factor for d in report.dimensions]
        report.implied_factor_median = statistics.median(factors)

        best_name = None
        best_error = None
        for name, candidate in CANDIDATE_FACTORS.items():
            error = abs(report.implied_factor_median - candidate) / candidate * 100
            if best_error is None or error < best_error:
                best_name, best_error = name, error
        report.best_match = best_name
        report.best_match_error_pct = round(best_error, 4)

    report.y_axis_evidence = analyse_y_axis(document)
    report.outline_records = collect_outline(document)
    report.outline_bbox_mm = outline_bbox(document)
    report.narrowest_line_widths_mm = narrowest_widths(document)
    return report


def print_report(report: CalibrationReport) -> None:
    print(f"source              : {report.source}")
    print(f"records             : {report.record_count}")
    print(f"unparsed lines      : {report.unparsed_lines}")
    print(f"documents           : {', '.join(report.document_types)}")
    print()
    print("=== top record types ===")
    for name, count in list(report.type_counts.items())[:14]:
        print(f"  {name:<18} {count}")
    print()

    print("=== unit calibration ===")
    if report.dimensions:
        for check in report.dimensions:
            print(
                f"  raw span={check.raw_span:.4f}  declared={check.declared_mm:.4f}mm"
                f"  =>  {check.implied_factor:.8f} mm/unit"
            )
        print(f"  median implied factor : {report.implied_factor_median:.8f} mm/unit")
        print(f"  closest candidate     : {report.best_match}")
        print(f"  deviation             : {report.best_match_error_pct:.4f} %")
        print()
        print(f"  => set EPRU_UNIT_MM = {report.implied_factor_median:.8f}")
    else:
        print("  no LENGTH dimension found; cannot self-calibrate from this file")
    print()

    print("=== unit sanity check (narrowest primitive widths) ===")
    print(f"  {report.narrowest_line_widths_mm} mm")
    if report.narrowest_line_widths_mm:
        smallest = report.narrowest_line_widths_mm[0]
        if smallest < 0.05:
            print("  WARNING: implies widths far below any fabrication limit -> factor too small")
        elif smallest > 5:
            print("  WARNING: implies implausibly wide primitives -> factor too large")
        else:
            print("  plausible for a real board")
    print()

    print("=== Y axis ===")
    for line in report.y_axis_evidence:
        print(f"  - {line}")
    print()

    print("=== board outline (layer 11) ===")
    if report.outline_bbox_mm:
        min_x, min_y, max_x, max_y = report.outline_bbox_mm
        print(f"  bbox (Y-up): x {min_x:.4f} .. {max_x:.4f}  ({max_x - min_x:.4f} mm wide)")
        print(f"               y {min_y:.4f} .. {max_y:.4f}  ({max_y - min_y:.4f} mm tall)")
    for entry in report.outline_records:
        print(f"  {entry['record_type']:<6} id={entry.get('id')}")
        for key in ("startX", "startY", "endX", "endY", "width"):
            if key in entry:
                print(f"     {key:<7} {entry[key]}")
        if "path_preview" in entry:
            print(f"     path    {entry['path_preview']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help=".epro2 or .epru file")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr)
        return 2

    report = calibrate(args.path)
    print_report(report)

    if args.json:
        args.json.write_text(report.to_json(), encoding="utf-8")
        print()
        print(f"JSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
