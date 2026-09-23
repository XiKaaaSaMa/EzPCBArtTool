#!/usr/bin/env python
"""Extract a small regression fixture from a real EasyEDA Pro project.

The full sample project is several megabytes of font and image data. For tests
we only need the records that pin down the format: the document heads, the
layer table, the board outline, and (critically) the length dimension that
self-calibrates the unit factor.

Usage::

    python tools/extract_fixture.py <project.epro2> tests/fixtures/real_sample.epru
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402

# Record types worth keeping: they are small and carry the facts under test.
# PREFERENCE matters because it stores the editor's default track and via
# sizes, which are useful independent anchors for the unit factor.
KEEP_IN_PCB = {
    "META",
    "CANVAS",
    "LAYER",
    "LAYER_PHYS",
    "DIMENSION",
    "LINE",
    "ARC",
    "POLY",
    "FILL",
    "VIA",
    "PREFERENCE",
}
KEEP_ALWAYS = {"DOCHEAD", "META"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help=".epro2 or .epru to read")
    parser.add_argument("dest", type=Path, help="fixture file to write")
    args = parser.parse_args()

    if args.source.suffix.lower() == ".epro2":
        document = EproArchive.read(args.source).document
    else:
        document = EproDocument.read(args.source)

    kept = EproDocument()
    current_doc = None

    for record in document.records:
        if record.type == "DOCHEAD":
            current_doc = (record.body or {}).get("docType")
            # Keep only the documents that matter for format tests.
            if current_doc in ("BOARD", "PCB"):
                kept.records.append(record)
            continue

        if current_doc == "PCB" and record.type in KEEP_IN_PCB:
            kept.records.append(record)
        elif current_doc == "BOARD" and record.type in KEEP_ALWAYS:
            kept.records.append(record)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    kept.write(args.dest)

    total = args.dest.stat().st_size
    print(f"extracted {len(kept.records)} of {len(document.records)} records")
    print(f"wrote {args.dest} ({total} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
