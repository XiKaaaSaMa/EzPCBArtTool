#!/usr/bin/env python
"""Determine the polygon winding convention EasyEDA Pro uses in real files.

The published spec says a complex polygon's first ring is the outer contour
wound clockwise and later rings are holes wound counter-clockwise. But the file
stores Y downward while design coordinates are Y up, so "clockwise" can mean
opposite things depending on the frame. This script measures what real FILL and
POUR records actually contain, in both frames, so the convention can be settled
from data instead of from prose.

Usage::

    python tools/analyze_winding.py <file.epro2|file.epru>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402
from pcbart.primitives import Mode  # noqa: E402

Point = tuple[float, float]


def signed_area(points: list[Point]) -> float:
    """Shoelace. Positive = counter-clockwise in the given frame."""
    total = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def decode_ring(node) -> list[Point]:
    """Best-effort decode of one polygon token stream into points."""
    if not isinstance(node, list) or not node:
        return []
    if node[0] == Mode.CIRCLE and len(node) >= 4:
        import math

        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return [
            (cx + r * math.cos(2 * math.pi * i / 32), cy + r * math.sin(2 * math.pi * i / 32))
            for i in range(32)
        ]
    if node[0] == Mode.R and len(node) >= 5:
        x, y, w, h = (float(v) for v in node[1:5])
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    points: list[Point] = []
    pending: list[float] = []
    i = 0
    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            points.extend((pending[j], pending[j + 1]) for j in range(0, len(pending) - 1, 2))
            pending = []
            i += 1
            if item in (Mode.ARC, Mode.CARC):
                if i + 2 < len(node):
                    points.append((float(node[i + 1]), float(node[i + 2])))
                i += 3
            continue
        if isinstance(item, (int, float)):
            pending.append(float(item))
        i += 1
    points.extend((pending[j], pending[j + 1]) for j in range(0, len(pending) - 1, 2))
    return points


def analyze(path: Path) -> None:
    if path.suffix.lower() == ".epro2":
        document = EproArchive.read(path).document
    else:
        document = EproDocument.read(path)

    print(f"file: {path.name}")
    print()

    for record_type in ("FILL", "POLY", "POUR", "IMAGE"):
        records = [
            r
            for r in document.iter_type(record_type)
            if isinstance(r.body, dict) and isinstance(r.body.get("path"), list)
        ]
        if not records:
            continue

        multi = 0
        single = 0
        file_ccw = 0
        file_cw = 0
        design_ccw = 0
        design_cw = 0

        examples: list[str] = []

        for record in records:
            path_data = record.body["path"]
            # A complex polygon is a list whose first element is itself a list.
            if path_data and isinstance(path_data[0], list):
                rings = path_data
                multi += 1
            else:
                rings = [path_data]
                single += 1

            for ring_index, ring_node in enumerate(rings):
                points = decode_ring(ring_node)
                if len(points) < 3:
                    continue
                area_file = signed_area(points)
                # File Y is screen-down; negate Y for the design frame.
                area_design = signed_area([(x, -y) for x, y in points])

                if area_file > 0:
                    file_ccw += 1
                elif area_file < 0:
                    file_cw += 1
                if area_design > 0:
                    design_ccw += 1
                elif area_design < 0:
                    design_cw += 1

                if len(examples) < 4:
                    examples.append(
                        f"      ring {ring_index}: {len(points)} pts  "
                        f"file_area={area_file:+.3f} ({'CCW' if area_file > 0 else 'CW'})  "
                        f"design_area={area_design:+.3f} "
                        f"({'CCW' if area_design > 0 else 'CW'})"
                    )

        print(f"  {record_type}: {len(records)} records ({single} single-ring, {multi} multi-ring)")
        if file_ccw or file_cw:
            print(f"    file frame   : CCW={file_ccw}  CW={file_cw}")
            print(f"    design frame : CCW={design_ccw}  CW={design_cw}")
            for line in examples:
                print(line)
        print()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_winding.py <file.epro2|file.epru>", file=sys.stderr)
        return 2
    analyze(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
