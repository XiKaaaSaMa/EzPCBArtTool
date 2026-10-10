"""Every board outline must come back to where it started.

⚠️ **This is the defect a square board exposed**: the exported ``.epro2`` was
missing the whole left edge. ``BoardOutlineSpec.outline_path`` had two branches
and only one of them closed the ring -- the arc branch looped ``range(1, n+1)``
with a modulo, while the straight branch wrote ``vertices[0]`` plus
``vertices[1:]`` and stopped.

A path is a start point plus one segment per following point, so a four-corner
ring written as four points draws **three** edges. On the badge card's 96-point
circle the same omission drops one segment of 3.75 degrees, which is invisible;
on a rectangle it is 25% of the board. That asymmetry is why it survived until
somebody made a square board.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import export  # noqa: E402
from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    OutlineShape,
)

#: One board per shape family, including the ones with no curves -- the branch
#: that was wrong is exactly the no-curves one.
SHAPES = {
    "rect": BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60.0, height_mm=60.0),
    "square": BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=50.0, height_mm=50.0),
    "rounded": BoardOutlineSpec(
        shape=OutlineShape.ROUNDED,
        width_mm=60.0,
        height_mm=40.0,
        corner_radius_mm=5.0,
    ),
    "circle": BoardOutlineSpec(shape=OutlineShape.CIRCLE, width_mm=60.0, height_mm=60.0),
}


def path_points(path) -> list[tuple[float, float]]:
    """The ``(x, y)`` pairs in an EasyEDA path token stream.

    Numbers appear as coordinate pairs and as arc sweeps, so the pairs are read
    positionally rather than by filtering -- the token stream is
    ``x, y, "L", x, y, x, y, ...`` with ``"ARC", sweep, x, y`` for curves.
    """
    points: list[tuple[float, float]] = []
    index = 0
    while index < len(path):
        token = path[index]
        if token == "ARC":
            points.append((float(path[index + 2]), float(path[index + 3])))
            index += 4
        elif isinstance(token, str):
            index += 1
        else:
            points.append((float(path[index]), float(path[index + 1])))
            index += 2
    return points


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_outline_path_returns_to_its_first_point(name):
    """The invariant the arc branch always held and the straight one did not.

    Asserted on ``outline_path`` rather than on the written file so the failure
    names the cause: a broken path is a broken path whatever writes it.
    """
    spec = SHAPES[name]
    points = path_points(spec.outline_path())

    assert len(points) >= 4, f"{name}: 至少要有起点和三个角"
    assert points[0] == pytest.approx(points[-1], abs=1e-9), (
        f"{name}: 板框没有回到起点，最后一个顶点到起点的这条边不会画出来"
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_written_epro2_outline_closes(name, tmp_path):
    """And the same thing through the real exporter, which is where the user
    saw it -- ``outline_record_from_path`` does not close anything itself."""
    board = ArtBoard(outline=SHAPES[name], outline_ready=True)
    target = tmp_path / "board.epro2"
    export.export_design(board, str(target), title="闭合")

    with zipfile.ZipFile(target) as archive:
        payload = [
            archive.read(entry).decode("utf-8")
            for entry in archive.namelist()
            if entry.endswith(".epru")
        ][0]

    outlines = [
        json.loads(line.split("||", 1)[1].rstrip("|"))
        for line in payload.splitlines()
        if '"type":"POLY"' in line and '"layerId":11' in line
    ]
    assert outlines, f"{name}: 导出里没有板框 POLY"

    for outline in outlines:
        points = path_points(outline["path"])
        assert points[0] == pytest.approx(points[-1], abs=1e-3), (
            f"{name}: 写出的板框路径没有闭合，会缺一条边"
        )


def test_a_rectangle_exports_four_edges_not_three(tmp_path):
    """The report, stated the way the user met it.

    A 60 x 60 board written as four corners draws three edges; the left side is
    the one that goes missing because it is the edge from the last vertex back
    to the first. Counting *segments* rather than points is what makes this
    assertion mean "four edges".
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60.0, height_mm=60.0),
        outline_ready=True,
    )
    target = tmp_path / "board.epro2"
    export.export_design(board, str(target), title="方形板")

    with zipfile.ZipFile(target) as archive:
        payload = [
            archive.read(entry).decode("utf-8")
            for entry in archive.namelist()
            if entry.endswith(".epru")
        ][0]
    outline = next(
        json.loads(line.split("||", 1)[1].rstrip("|"))
        for line in payload.splitlines()
        if '"type":"POLY"' in line and '"layerId":11' in line
    )

    points = path_points(outline["path"])
    # A closed ring of 4 distinct corners is 4 segments, i.e. 5 points.
    assert len(points) == 5, f"矩形板框应当是 4 条边（5 个点），实际 {len(points)} 个点"

    # And the four edges really are the four sides: each consecutive pair
    # differs in exactly one axis, and the ring covers both x extents.
    xs = sorted({round(point[0], 4) for point in points})
    ys = sorted({round(point[1], 4) for point in points})
    assert len(xs) == 2 and len(ys) == 2, "矩形应当只有两个 x 和两个 y"
    for start, end in zip(points, points[1:]):
        assert (start[0] == end[0]) != (start[1] == end[1]), (
            f"{start} -> {end} 不是一条水平或竖直的边"
        )


def test_the_gerber_outline_closes_too(tmp_path):
    """⚠️ Gerber was already correct, and this pins that it stays so.

    The two writers take different routes -- Gerber passes ``close=True`` to a
    stroke builder that appends the first point itself -- so a fix applied only
    to the ``.epro2`` path must not be mistaken for the whole job. A router
    needs a closed contour, and Gerber has no implicit closure at all.
    """
    from pcbart import gerber

    board = ArtBoard(
        outline=BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60.0, height_mm=60.0),
        outline_ready=True,
    )
    report = gerber.export_gerber(board, tmp_path / "g", title="t")
    text = (tmp_path / "g" / report["files"][0]["name"]).read_text(encoding="ascii")

    import re

    coords = [
        (int(m.group(1)) / 1e6, int(m.group(2)) / 1e6)
        for m in re.finditer(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)
    ]
    assert len(coords) == 5, f"Gerber 板框应当是 4 条边（5 个点），实际 {len(coords)}"
    assert coords[0] == pytest.approx(coords[-1]), "Gerber 板框没有闭合"
