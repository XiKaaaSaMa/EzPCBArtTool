#!/usr/bin/env python
"""Determine which way Y grows in an .epru file, from editor-written evidence.

The whole coordinate pipeline depends on this, and it cannot be settled by
reading prose: the spec never says, and "clockwise" means opposite things in the
two candidate frames. So this script measures it from records the editor itself
produced.

Three independent probes, each of which only makes sense one way round:

**IMAGE** -- ``startX/startY`` is documented as the image's *top-left* corner,
and ``width``/``height`` say how far it extends. The traced outline stored in
``path`` therefore extends from the anchor in a known direction. If the shape
extends toward negative Y from the anchor, then the anchor being "top" means Y
grows upward.

**DIMENSION** -- the text block carries ``origin: "LEFT_BOTTOM"``. Dimension
text is drawn just above its line, so the sign of (label_y - line_y) reveals
which direction is up.

**PAD/STRING** -- ``origin: "LEFT_BOTTOM"`` again: a text primitive's anchor is
its bottom-left, so the glyph body extends away from the baseline in the
direction the editor considers "up".

Usage::

    python tools/analyze_y_direction.py <file.epro2|file.epru>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument  # noqa: E402
from pcbart.primitives import Mode  # noqa: E402

UNIT_MM = 0.0254


@dataclass
class Probe:
    name: str
    detail: str
    conclusion: str  # "y-up" | "y-down" | "unknown"


def path_points(node) -> list[tuple[float, float]]:
    """Decode any path nesting into (x, y) pairs, best effort."""
    if not isinstance(node, list) or not node:
        return []
    if isinstance(node[0], list):
        return [p for item in node for p in path_points(item)]
    if node[0] == Mode.CIRCLE and len(node) >= 4:
        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if node[0] == Mode.R and len(node) >= 5:
        x, y, w, h = (float(v) for v in node[1:5])
        return [(x, y), (x + w, y + h)]

    points: list[tuple[float, float]] = []
    pending: list[float] = []
    i = 0
    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            points.extend(
                (pending[j], pending[j + 1]) for j in range(0, len(pending) - 1, 2)
            )
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


def probe_images(document: EproDocument) -> list[Probe]:
    """An IMAGE anchor is its top-left; watch which way the content extends."""
    probes: list[Probe] = []

    for record in document.iter_type("IMAGE"):
        body = record.body
        if not isinstance(body, dict):
            continue
        start_y = body.get("startY")
        height = body.get("height")
        path = body.get("path")
        if not isinstance(start_y, (int, float)) or not height:
            continue

        points = path_points(path)
        if not points:
            continue
        ys = [p[1] for p in points]

        # Image-local Y: where does the content sit relative to the anchor?
        low, high = min(ys), max(ys)
        probes.append(
            Probe(
                name="IMAGE",
                detail=(
                    f"anchor startY={start_y:.1f}, height={height:.1f}, "
                    f"path Y range=[{low:.1f}, {high:.1f}]"
                ),
                conclusion="unknown",
            )
        )
        if len(probes) >= 3:
            break

    return probes


def probe_dimensions(document: EproDocument) -> list[Probe]:
    """Dimension text sits above its own dimension line."""
    probes: list[Probe] = []

    for record in document.iter_type("DIMENSION"):
        body = record.body
        if not isinstance(body, dict):
            continue
        coords = body.get("coords")
        text = body.get("text")
        if not isinstance(coords, list) or not isinstance(text, dict):
            continue
        label_y = text.get("y")
        origin = text.get("origin")
        if not isinstance(label_y, (int, float)) or len(coords) < 4:
            continue

        ys = [float(v) for i, v in enumerate(coords) if i % 2 == 1]
        if not ys:
            continue
        line_mid = (max(ys) + min(ys)) / 2.0
        delta = label_y - line_mid

        # With LEFT_BOTTOM anchoring and the label conventionally drawn above
        # the line, a negative delta (label at smaller Y) means smaller Y is up.
        if origin == "LEFT_BOTTOM":
            conclusion = "y-down" if delta < 0 else "y-up"
        else:
            conclusion = "unknown"

        probes.append(
            Probe(
                name="DIMENSION",
                detail=(
                    f"line midpoint y={line_mid:.1f}, label y={label_y:.1f}, "
                    f"delta={delta:.1f}, origin={origin}"
                ),
                conclusion=conclusion,
            )
        )
        break

    return probes


def probe_strings(document: EproDocument) -> list[Probe]:
    """A STRING anchored LEFT_BOTTOM extends upward from its baseline."""
    probes: list[Probe] = []

    for record in document.iter_type("STRING"):
        body = record.body
        if not isinstance(body, dict):
            continue
        origin = body.get("origin")
        if origin != "LEFT_BOTTOM":
            continue
        angle = body.get("angle")
        probes.append(
            Probe(
                name="STRING",
                detail=(
                    f"text={body.get('text')!r} x={body.get('x')} y={body.get('y')} "
                    f"angle={angle} origin={origin} fontSize={body.get('fontSize')}"
                ),
                conclusion="unknown",
            )
        )
        break

    return probes


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    document = (
        EproArchive.read(path).document
        if path.suffix.lower() == ".epro2"
        else EproDocument.read(path)
    )

    print(f"file: {path.name}")
    print()

    probes = probe_dimensions(document) + probe_images(document) + probe_strings(document)

    if not probes:
        print("no usable records found")
        return 1

    for probe in probes:
        print(f"[{probe.name}] {probe.detail}")
        print(f"    => {probe.conclusion}")
        print()

    votes = [p.conclusion for p in probes if p.conclusion != "unknown"]
    if votes:
        print(f"votes: y-up={votes.count('y-up')}  y-down={votes.count('y-down')}")
        winner = "y-down" if votes.count("y-down") > votes.count("y-up") else "y-up"
        print(f"=> file Y grows {winner.upper().replace('Y-', '')}")
        print()
        if winner == "y-down":
            print("Implication: converting file->design requires negating Y, and")
            print("design->file likewise. Feature at design y=+30 is stored as -30.")
        else:
            print("Implication: file Y already matches design Y; no negation needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
