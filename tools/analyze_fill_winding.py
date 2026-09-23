#!/usr/bin/env python
"""Decide the polygon winding rule from FILL records, using nesting depth.

An earlier attempt assumed ``path[0]`` was the outer ring and reported mixed
results. That assumption is wrong: a path holds a flat list of rings whose role
is given by nesting, not by position. This version determines each ring's depth
by point-in-polygon containment, which is what the renderer itself must do, and
then reports the winding of depth-0 (solid) versus depth-1 (hole) rings.

Reported in **raw file coordinates**, with no Y flip, so the result is a rule
that can be applied directly to the numbers in the file.

Usage::

    python tools/analyze_fill_winding.py <file.epro2|file.epru>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402
from pcbart.primitives import Mode  # noqa: E402

Point = tuple[float, float]


def decode_ring(node) -> list[Point]:
    if not isinstance(node, list) or not node:
        return []
    if node[0] == Mode.CIRCLE and len(node) >= 4:
        import math

        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return [
            (cx + r * math.cos(2 * math.pi * i / 24), cy + r * math.sin(2 * math.pi * i / 24))
            for i in range(24)
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


def area(ring: list[Point]) -> float:
    total = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def inside(point: Point, ring: list[Point]) -> bool:
    """Ray casting, counting crossings of a ray toward +X."""
    x, y = point
    hits = 0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                hits += 1
    return hits % 2 == 1


def interior_point(ring: list[Point]) -> Point | None:
    """A point just inside the ring, found by scanning a horizontal line."""
    ys = sorted({p[1] for p in ring})
    if len(ys) < 2:
        return None
    y = (ys[0] + ys[-1]) / 2
    xs = sorted(
        x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
        if (y1 > y) != (y2 > y)
    )
    if len(xs) < 2:
        return None
    return ((xs[0] + xs[1]) / 2, y)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    doc = (
        EproArchive.read(path).document
        if path.suffix.lower() == ".epro2"
        else EproDocument.read(path)
    )

    tally: dict[tuple[int, str], int] = {}
    examples: list[str] = []

    for record in doc.iter_type("FILL"):
        body = record.body
        path_data = body.get("path")
        if not isinstance(path_data, list) or not path_data:
            continue

        raw_rings = path_data if isinstance(path_data[0], list) else [path_data]
        rings = [decode_ring(r) for r in raw_rings]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue

        samples = []
        for ring in rings:
            pt = interior_point(ring)
            if pt is None:
                continue
            depth = sum(1 for other in rings if other is not ring and inside(pt, other))
            a = area(ring)
            winding = "CCW" if a > 0 else "CW"
            tally[(depth, winding)] = tally.get((depth, winding), 0) + 1
            samples.append(f"d{depth}={winding}({abs(a):.0f})")

        if len(examples) < 6:
            examples.append(
                f"  FILL {record.id}: {len(rings)} rings -> " + ", ".join(samples)
            )

    print(f"file: {path.name}")
    print()
    print("raw-coordinate winding by nesting depth (no Y flip):")
    for depth in sorted({d for d, _ in tally}):
        ccw = tally.get((depth, "CCW"), 0)
        cw = tally.get((depth, "CW"), 0)
        role = {0: "solid (outer)", 1: "hole", 2: "island"}.get(depth, f"depth {depth}")
        verdict = "CCW" if ccw > cw else "CW" if cw > ccw else "mixed"
        print(f"  depth {depth} [{role:14}] CCW={ccw:4}  CW={cw:4}   => {verdict}")
    print()
    print("examples:")
    for line in examples:
        print(line)
    print()
    print("Rule implied for new geometry, written in raw file coordinates:")
    for depth in (0, 1, 2):
        ccw = tally.get((depth, "CCW"), 0)
        cw = tally.get((depth, "CW"), 0)
        if ccw or cw:
            want = "counter-clockwise" if ccw >= cw else "clockwise"
            print(f"  depth {depth}: write {want}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
