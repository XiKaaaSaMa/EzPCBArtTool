#!/usr/bin/env python
"""Print the raw and converted geometry of every outline primitive in a file."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402
from pcbart.primitives import record_to_mm  # noqa: E402

path = Path(sys.argv[1])
if path.suffix.lower() == ".epro2":
    document = EproArchive.read(path).document
else:
    document = EproDocument.read(path)

found = 0
for record in document.records:
    body = record.body
    if not isinstance(body, dict) or body.get("layerId") != 11:
        continue
    if record.type not in ("POLY", "LINE", "ARC", "FILL"):
        continue
    found += 1
    print(f"{record.type} id={record.id}")
    print(f"  raw path   = {body.get('path')}")
    print(f"  raw width  = {body.get('width')}")
    converted = record_to_mm(record)
    print(f"  mm path    = {converted.get('path')}")
    print(f"  mm width   = {converted.get('width')}")

if not found:
    print("no outline primitives found on layer 11")

print()
print("all recorded types:", sorted({r.type for r in document.records}))
