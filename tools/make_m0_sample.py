#!/usr/bin/env python
"""Generate the M0 validation files.

Three boards, each isolating one question:

``m0_rect_50x30.epro2``
    A 50 x 30 mm rectangular outline. Opening this in EasyEDA Pro answers
    "does the editor accept our file, and are the dimensions right?".

``m0_rect_100x50.epro2``
    A 100 x 50 mm outline, deliberately different in aspect ratio. If the unit
    factor were wrong by 10x or 2.54x, comparing the two files makes it
    obvious.

``m0_layers.epro2``
    A 20 x 20 mm outline plus one filled rectangle on the top copper layer, to
    check that ``FILL`` primitives and the layer table land correctly.

Usage::

    python tools/make_m0_sample.py --outdir out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument, Record, new_id  # noqa: E402
from pcbart.geometry import complex_polygon_to_path, normalize_complex_polygon, rect_ring  # noqa: E402
from pcbart.generator import BoardSpec, generate_board_with_outline  # noqa: E402
from pcbart.paths import default_output_dir  # noqa: E402
from pcbart.primitives import convert_path_to_unit  # noqa: E402
from pcbart.templates import build_minimal_project  # noqa: E402
from pcbart.units import mm_to_unit  # noqa: E402


def make_rect(width: float, height: float, outdir: Path, name: str) -> Path:
    spec = BoardSpec(
        width_mm=width,
        height_mm=height,
        line_width_mm=0.125,
        title=name,
    )
    return generate_board_with_outline(spec, outdir / f"{name}.epro2")


def make_layers_sample(outdir: Path) -> Path:
    """An outline plus a copper fill, to exercise the FILL primitive."""
    name = "m0_layers"
    document, _uuids = build_minimal_project(title=name)

    outline = BoardSpec(
        width_mm=20,
        height_mm=20,
        line_width_mm=0.125,
        title=name,
    )
    from pcbart.primitives import board_outline_rect

    outline_body = board_outline_rect(
        x_mm=0.0,
        y_mm=0.0,
        width_mm=20.0,
        height_mm=20.0,
        line_width_mm=0.125,
    )
    outline_body["path"] = convert_path_to_unit(outline_body["path"])
    outline_body["width"] = mm_to_unit(0.125)
    document.append_to_section(
        Record({"type": "POLY", "ticket": document.next_ticket(), "id": new_id()}, outline_body),
        doc_type="PCB",
    )

    # A 10 x 6 mm copper pad on the top layer, inset from the outline.
    rings = normalize_complex_polygon(rect_ring(5.0, 7.0, 10.0, 6.0))
    path_mm = complex_polygon_to_path(rings)
    fill_body = {
        "partitionId": "",
        "groupId": 0,
        "netName": "",
        "layerId": 1,
        "width": mm_to_unit(0.1),
        "fillStyle": "SOLID",
        "path": convert_path_to_unit(path_mm),
        "locked": False,
        "zIndex": -1,
        "isBridgingCopper": False,
        "networkList": [],
        "refs": None,
    }
    document.append_to_section(
        Record({"type": "FILL", "ticket": document.next_ticket(), "id": new_id()}, fill_body),
        doc_type="PCB",
    )

    return EproArchive(document, {"title": name}).write(outdir / f"{name}.epro2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=default_output_dir(), help="output directory")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    made = [
        make_rect(50, 30, args.outdir, "m0_rect_50x30"),
        make_rect(100, 50, args.outdir, "m0_rect_100x50"),
        make_layers_sample(args.outdir),
    ]

    print(f"wrote {len(made)} files to {args.outdir.resolve()}")
    for path in made:
        print(f"  {path.name}  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
