"""Read the artwork regions out of a badge template ``.epro2``.

The template's "put the picture here" rectangles are ordinary ``FILL`` records
on the top layer -- there is nothing in the file that labels them. So they are
found by their *size*, and this tool exists to make that measurement repeatable
rather than a number someone typed once and nobody can re-check.

Run it when the template changes (a new blank board, a moved region) and copy
the printed constants into :mod:`pcbart.badge_template`.

Usage::

    python tools/extract_badge_regions.py <file.epro2> [--min-mm 30]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive  # noqa: E402

MIL_MM = 0.0254
"""One ``.epru`` coordinate unit in millimetres; see :mod:`pcbart.units`."""

TOP_LAYER = 1
OUTLINE_LAYER = 11


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in millimetres, Y up."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


def _fill_rect(path) -> Box | None:
    """The box of a ``['R', x, y, w, h, 0, 0]`` fill.

    ``R`` anchors at the *top-left* and grows downward, while the document's Y
    axis grows upward, so the stored ``y`` is the top edge and ``y - h`` the
    bottom.
    """
    if not isinstance(path, list):
        return None
    for segment in path:
        if isinstance(segment, list) and segment and segment[0] == "R" and len(segment) >= 5:
            x, y, w, h = (float(v) * MIL_MM for v in segment[1:5])
            return Box(x, y - h, x + w, y)
    return None


def _outline_boxes(document) -> list[Box]:
    """One box per board, from the axis-aligned edges on the outline layer.

    **Only axis-aligned segments count.** The outline layer also carries long
    diagonal guide lines -- they run from a card's corners toward a common point
    near the origin, so taking every segment's bounding box merged all four
    cards into two nonsense rectangles (measured: a 191 × 312 "board" and a
    47 × 0 one). A rectangular card's edges are horizontal or vertical; the
    diagonals are not part of its outline, whatever they are for.
    """
    segments: list[Box] = []
    for record in document.iter_type("POLY"):
        if record.body.get("layerId") != OUTLINE_LAYER:
            continue
        path = record.body.get("path")
        if not isinstance(path, list):
            continue
        numbers = [float(v) * MIL_MM for v in path if isinstance(v, (int, float))]
        points = list(zip(numbers[0::2], numbers[1::2]))
        for start, end in zip(points, points[1:]):
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            # A card edge runs along one axis for its whole length; a corner cut
            # is diagonal and the guide lines are diagonal too, so both drop out.
            if dx > 0.5 and dy > 0.5:
                continue
            if max(dx, dy) < MIN_EDGE_MM:
                continue
            segments.append(_box_of([start, end]))
    return _merge_touching(segments)


MIN_EDGE_MM = 20.0
"""Shortest run that still reads as a card edge rather than a corner cut."""


def _box_of(points: Iterable[tuple[float, float]]) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


EDGE_GAP_MM = 3.0
"""Largest break still counted as one side.

The corner notches interrupt a side by about 1.9 mm, while neighbouring cards
sit 15.8 mm apart -- so anything in that wide gap separates the two cases.
"""


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Distance between two intervals; negative when they overlap."""
    return max(b0 - a1, a0 - b1)


def _merge_touching(boxes: list[Box], tolerance: float = EDGE_GAP_MM) -> list[Box]:
    """Group edges that belong to the same card.

    A card's outline is stored as one record per side (often split again around
    the corner cuts), and a single edge is a zero-thickness box -- so overlap in
    *both* axes is the wrong test and left every edge in a box of its own. Edges
    join when the gap between them is small in both axes, which also bridges the
    ~1.9 mm notch where a side is interrupted by a cut corner.
    """
    remaining = list(boxes)
    merged: list[Box] = []
    while remaining:
        current = remaining.pop()
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if (
                    _gap(current.x0, current.x1, other.x0, other.x1) <= tolerance
                    and _gap(current.y0, current.y1, other.y0, other.y1) <= tolerance
                ):
                    current = Box(
                        min(current.x0, other.x0),
                        min(current.y0, other.y0),
                        max(current.x1, other.x1),
                        max(current.y1, other.y1),
                    )
                    remaining.remove(other)
                    changed = True
        merged.append(current)
    return sorted(merged, key=lambda b: (-b.centre[1], b.centre[0]))




def extract_boards(path: Path) -> list[Box]:
    """The card outlines in a template file, in millimetres.

    Exposed separately from :func:`extract_regions` so a test can check the
    board count and size without going through the printing of a report.
    """
    return _outline_boxes(EproArchive.read(path).document)


def extract_regions(path: Path, min_mm: float) -> list[tuple[Box, Box]]:
    """The artwork regions, each paired with the card that contains it."""
    document = EproArchive.read(path).document
    cards = _outline_boxes(document)
    regions: list[tuple[Box, Box]] = []
    for record in document.iter_type("FILL"):
        if record.body.get("layerId") != TOP_LAYER:
            continue
        box = _fill_rect(record.body.get("path"))
        if box is None or box.width < min_mm or box.height < min_mm:
            continue
        owner = min(
            cards,
            key=lambda card: abs(card.centre[0] - box.centre[0])
            + abs(card.centre[1] - box.centre[1]),
            default=None,
        )
        if owner is not None:
            regions.append((box, owner))
    return regions


def extract(path: Path, min_mm: float) -> int:
    cards = extract_boards(path)
    regions = extract_regions(path, min_mm)

    print(f"文件：{path}")
    print(f"识别到 {len(cards)} 块板（板框层），{len(regions)} 个立绘填充区（顶层）")
    print()
    for index, card in enumerate(cards):
        print(
            f"  板 {index + 1}: {card.width:.3f} × {card.height:.3f} mm  "
            f"中心 ({card.centre[0]:.3f}, {card.centre[1]:.3f})"
        )
    print()

    if not regions:
        print("⚠ 没有找到够大的顶层填充矩形；试小一点的 --min-mm。")
        return 1

    print("相对各自板子中心的立绘区（可直接抄进 pcbart/badge_template.py）：")
    for box, card in sorted(regions, key=lambda item: item[0].width):
        cx, cy = box.centre
        ccx, ccy = card.centre
        print(
            f"  ArtworkRegion(\n"
            f"      width_mm={box.width:.3f},\n"
            f"      height_mm={box.height:.3f},\n"
            f"      centre_x_mm={cx - ccx:+.3f},\n"
            f"      centre_y_mm={cy - ccy:+.3f},\n"
            f"  )"
        )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument(
        "--min-mm",
        type=float,
        default=30.0,
        help="忽略小于这个尺寸的顶层填充（默认 30，用来滤掉装饰小矩形）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return extract(args.file, args.min_mm)


if __name__ == "__main__":
    sys.exit(main())
