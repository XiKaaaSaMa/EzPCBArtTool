#!/usr/bin/env python
"""Print the value-bearing records of a fixture, in millimetres."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproDocument  # noqa: E402
from pcbart.units import unit_to_mm  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/real_sample.epru")
doc = EproDocument.read(path)

print("VIAs")
for record in doc.iter_type("VIA"):
    body = record.body
    print(
        "  diameter {} units = {:.4f} mm | hole {} units = {:.4f} mm | type {}".format(
            body.get("viaDiameter"),
            unit_to_mm(body.get("viaDiameter", 0)),
            body.get("holeDiameter"),
            unit_to_mm(body.get("holeDiameter", 0)),
            body.get("viaType"),
        )
    )

print("DIMENSIONs")
for record in doc.iter_type("DIMENSION"):
    body = record.body
    text = (body.get("text") or {}).get("text")
    coords = body.get("coords") or []
    xs = coords[0::2]
    ys = coords[1::2]
    span = 0.0
    if xs and ys:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    print(f"  text={text!r} span={span} units -> {unit_to_mm(span):.4f} mm")

print("PREFERENCE")
for record in doc.iter_type("PREFERENCE"):
    body = record.body
    for key in ("lastTrackWidth", "lastViaDiameter", "lastViaInnerDiameter"):
        if key in body:
            print(f"  {key} = {body[key]} units = {unit_to_mm(body[key]):.4f} mm")
