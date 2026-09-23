#!/usr/bin/env python
"""Build the M3 encoding experiments: copper fills, holes and traces.

Nothing here is guesswork about EasyEDA Pro's *intent* -- the payload shapes
come from the 75 ``FILL`` and 6 ``LINE`` records captured out of a real project
(``tests/fixtures/real_sample.epru``). What that capture cannot answer is the
part it never contained: **not one of those 75 fills has an interior ring.**
Every hole in a drawing -- a letter's counter, the gap between two fingers --
depends on that untested case, so it gets an experiment rather than an
assumption.

``out/m3_1_hole_winding.epro2``
    Five identical squares side by side, each with a square hole in the middle.
    Only the *winding* of the outer and inner rings differs, which is the one
    variable that decides whether the hole is cut. The first square has no hole
    at all and is the control: if it ever shows a hole, something else is wrong.

``out/m3_2_line_layers.epro2``
    Traces of several widths on copper and on silkscreen. Every real ``LINE``
    sample sat on layer 2, so this checks that the same form works on the
    layers this tool actually writes.

Usage::

    python tools/make_m3_experiments.py --outdir out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive, EproDocument, Record, new_id  # noqa: E402
from pcbart.geometry import ensure_ccw, ensure_cw  # noqa: E402
from pcbart.primitives import convert_path_to_unit, fill, line  # noqa: E402
from pcbart.templates import build_minimal_project  # noqa: E402
from pcbart.units import mm_to_unit  # noqa: E402

# Layer ids, from the table captured off a real project.
LAYER_TOP_COPPER = 1
LAYER_TOP_SILK = 3
LAYER_OUTLINE = 11

FILL_WIDTH_RAW = 0.2
"""The ``width`` value real fills carry, written verbatim.

All 75 captured fills hold ``0.2`` (one holds ``1``). It is **not** converted
from millimetres: writing ``mm_to_unit(0.2)`` = 7.874 produced a fill the
editor ignored, and the number the editor itself writes is the only value with
evidence behind it. Whatever unit this field uses, matching the editor beats
guessing at it.
"""

OUTLINE_WIDTH_MM = 0.125


def _square(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    """Four corners of a square, counter-clockwise."""
    half = size / 2.0
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _ring_stream(points_mm, clockwise: bool) -> list:
    """One ring as the flat ``[x, y, "L", x, y, ...]`` token stream, closed.

    The ring **must repeat its first vertex at the end**. All 47 ``L``-form
    rings in the captured project are explicitly closed, and leaving it open is
    how the first attempt at this file produced a board with no copper on it at
    all. The earlier note claiming "the editor closes the ring itself" was
    wrong: it came from reading the ``R`` and ``CIRCLE`` forms, which are
    inherently closed and therefore do not repeat a vertex.
    """
    ring = ensure_cw(points_mm) if clockwise else ensure_ccw(points_mm)
    stream: list = [ring[0][0], ring[0][1], "L"]
    for x, y in ring[1:]:
        stream.extend([x, y])
    # Close the loop.
    stream.extend([ring[0][0], ring[0][1]])
    return stream


def _fill_record(document: EproDocument, layer_id: int, rings_mm: list[list]) -> Record:
    """A ``FILL`` record from rings the caller has already wound.

    The rings arrive as plain point lists in the winding under test, so this
    only converts them to the file's units -- re-winding here would defeat the
    whole point of the experiment.
    """
    path_mm = [_ring_stream(ring, clockwise=_is_cw(ring)) for ring in rings_mm]
    body = fill(layer_id, path_mm, FILL_WIDTH_RAW)
    body["path"] = convert_path_to_unit(path_mm)
    # Written raw, not converted: see FILL_WIDTH_RAW.
    body["width"] = FILL_WIDTH_RAW
    return Record(
        header={"type": "FILL", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def _is_cw(ring) -> bool:
    """Whether a ring is already wound clockwise in the design frame."""
    from pcbart.geometry import signed_area

    return signed_area(ring) < 0


def _outline_record(document: EproDocument, points_mm) -> Record:
    from pcbart.generator import outline_record

    return outline_record(points_mm, document, line_width_mm=OUTLINE_WIDTH_MM)


def build_hole_winding(outdir: Path) -> Path:
    """Five squares: a control, then the four winding combinations.

    Deliberately large and widely spaced. The first version packed five 12 mm
    squares into an 80 mm board, and at the editor's default zoom the hole in
    each one was a few pixels across -- not something a person can judge by
    eye, which defeats the purpose of an experiment whose whole answer is
    "is this one hollow".
    """
    title = "M3-1 孔径绕向对照"
    document, _uuids = build_minimal_project(title=title)

    # One square per row, stacked vertically: reading them off in order is the
    # entire task, so nothing should be side by side competing for space.
    outer_size, hole_size = 40.0, 24.0
    spacing = 52.0
    count = 5
    half_height = spacing * (count - 1) / 2.0 + outer_size
    board_outline = [
        (-32.0, -half_height),
        (32.0, -half_height),
        (32.0, half_height),
        (-32.0, half_height),
    ]
    document.append_to_section(_outline_record(document, board_outline), doc_type="PCB")

    # Top to bottom. ``hole_cw=None`` is the control, drawn with no inner ring:
    # if that one shows a hole, something other than the winding is at work.
    variants = [
        (False, None, "1 对照：实心无孔"),
        (True, False, "2 外CW + 孔CCW"),
        (False, True, "3 外CCW + 孔CW"),
        (True, True, "4 外CW + 孔CW"),
        (False, False, "5 外CCW + 孔CCW"),
    ]

    for index, (outer_cw, hole_cw, _label) in enumerate(variants):
        # File Y grows upward, so the first entry ends up at the top.
        y = half_height - outer_size / 2.0 - index * spacing
        outer = _square(0.0, y, outer_size)
        outer = ensure_cw(outer) if outer_cw else ensure_ccw(outer)
        rings_mm = [outer]
        if hole_cw is not None:
            hole = _square(0.0, y, hole_size)
            rings_mm.append(ensure_cw(hole) if hole_cw else ensure_ccw(hole))
        document.append_to_section(
            _fill_record(document, LAYER_TOP_COPPER, rings_mm), doc_type="PCB"
        )

    path = outdir / "m3_1_hole_winding.epro2"
    EproArchive(document=document, meta={"title": title}).write(path)
    return path


def build_line_layers(outdir: Path) -> Path:
    """Traces on copper and silkscreen, at a few widths.

    The two layers are grouped rather than interleaved. Side by side, four
    copper traces and four silkscreen traces 2 mm apart read as one set of
    eight, which makes "did the silkscreen ones appear?" hard to answer -- and
    that question is the reason the file exists.
    """
    title = "M3-2 走线与图层"
    document, _uuids = build_minimal_project(title=title)

    board_outline = [
        (-40.0, -26.0),
        (40.0, -26.0),
        (40.0, 26.0),
        (-40.0, 26.0),
    ]
    document.append_to_section(_outline_record(document, board_outline), doc_type="PCB")

    def add_line(layer_id: int, x0, y0, x1, y1, width_mm: float) -> None:
        from pcbart.primitives import Point

        body = line(layer_id, Point(x0, y0), Point(x1, y1), width_mm)
        body["startX"] = mm_to_unit(x0)
        body["startY"] = mm_to_unit(y0)
        body["endX"] = mm_to_unit(x1)
        body["endY"] = mm_to_unit(y1)
        body["width"] = mm_to_unit(width_mm)
        document.append_to_section(
            Record(
                header={"type": "LINE", "ticket": document.next_ticket(), "id": new_id()},
                body=body,
            ),
            doc_type="PCB",
        )

    widths = [0.15, 0.25, 0.4, 0.8]
    # 顶层铜：上半部分，从下往上逐条变粗
    for index, width in enumerate(widths):
        y = 2.0 + index * 7.0
        add_line(LAYER_TOP_COPPER, -34.0, y, 34.0, y, width)
    # 顶层丝印：下半部分，同样四种宽度
    for index, width in enumerate(widths):
        y = -5.0 - index * 7.0
        add_line(LAYER_TOP_SILK, -34.0, y, 34.0, y, width)

    path = outdir / "m3_2_line_layers.epro2"
    EproArchive(document=document, meta={"title": title}).write(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for builder in (build_hole_winding, build_line_layers):
        path = builder(outdir)
        size = path.stat().st_size
        print(f"wrote {path}  ({size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
