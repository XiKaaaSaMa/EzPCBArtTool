"""Tests for the Gerber export.

The bar here is different from the ``.epro2`` tests. An ``.epro2`` is opened by
one program, and the worst case of a mistake is that the user sees something
wrong on screen. A Gerber set goes to a fab: a wrong file means a wrong board,
in metal, paid for. So these check the format itself -- delimiters, polarity,
units, closed contours -- and not only the numbers.

The other half of the job is *which* files come out. 嘉立创's uploader reads the
set it is handed, so a layer that is empty must produce no file at all, and the
one layer that must always be there is the outline: a board without an edge cut
cannot be routed.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
)
from pcbart.gerber import (  # noqa: E402
    DEFAULT_PREFIX,
    GERBER_LAYERS,
    OUTLINE_LAYER,
    export_gerber,
    export_gerber_archive,
    gerber_text,
)

cv2 = pytest.importorskip("cv2")


def artwork(tmp_path: Path, *, hole: bool = False) -> str:
    """A black rectangle on white, optionally with a filled white hole.

    Ink is dark in this project's convention, so the dark rectangle is the
    shape and the white circle inside it becomes a hole -- which is the case
    Gerber needs clear polarity for.
    """
    pixels = np.full((200, 300), 255, np.uint8)
    cv2.rectangle(pixels, (40, 40), (260, 160), 0, -1)
    if hole:
        cv2.circle(pixels, (150, 100), 25, 255, -1)
    path = tmp_path / ("holed.png" if hole else "solid.png")
    cv2.imwrite(str(path), pixels)
    return str(path)


def board_with(path: str, **kwargs) -> ArtBoard:
    spec = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=40)
    for name, value in kwargs.items():
        setattr(spec, name, value)
    board = ArtBoard(outline=spec, outline_ready=True)
    board.placements.append(
        ImagePlacement(path=path, width_mm=60, height_mm=40, layer="顶层")
    )
    return board


# ------------------------------------------------------------- file set


def test_the_outline_and_the_used_layers_are_written(tmp_path):
    report = export_gerber(
        board_with(artwork(tmp_path)), tmp_path / "out", title="demo"
    )
    names = {entry["name"] for entry in report["files"]}

    assert f"{DEFAULT_PREFIX}{OUTLINE_LAYER.file_name}" in names
    assert f"{DEFAULT_PREFIX}TopLayer.GTL" in names
    # The copper layer is the only one with content, so nothing else is made.
    assert len(names) == 2


def test_the_default_prefix_is_the_one_the_user_asked_for():
    assert DEFAULT_PREFIX == "Gerber_"


def test_an_empty_layer_produces_no_file(tmp_path):
    """A stray empty silkscreen file invites the question the set answers:
    which layers does this board actually have?"""
    report = export_gerber(board_with(artwork(tmp_path)), tmp_path / "out")

    names = {entry["name"] for entry in report["files"]}
    assert not any(
        name.endswith((".GTO", ".GBO", ".GTS", ".GBS", ".GBL")) for name in names
    )
    assert set(report["skipped"]) >= {"底层铜", "顶层丝印"}


def test_the_outline_is_written_even_with_no_artwork(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )

    report = export_gerber(board, tmp_path / "out")

    assert [entry["name"] for entry in report["files"]] == [
        f"{DEFAULT_PREFIX}{OUTLINE_LAYER.file_name}"
    ]


def test_every_file_lands_in_one_folder_with_the_prefix(tmp_path):
    directory = tmp_path / "gerber"
    report = export_gerber(board_with(artwork(tmp_path)), directory, title="demo")

    for entry in report["files"]:
        path = Path(str(entry["path"]))
        assert path.parent == directory
        assert path.name.startswith(DEFAULT_PREFIX)
        assert path.exists()


def test_a_zip_is_produced_for_uploading(tmp_path):
    """嘉立创 takes a zip, and asking the user to zip seven files is a chore."""
    report = export_gerber(
        board_with(artwork(tmp_path)), tmp_path / "out", title="我的板"
    )

    archive = Path(str(report["archive"]))
    assert archive.exists()
    with zipfile.ZipFile(archive) as bundle:
        packed = set(bundle.namelist())
    expected = {str(entry["name"]) for entry in report["files"]}
    assert packed == expected


def test_the_single_file_export_leaves_only_the_zip(tmp_path):
    """The order file is one file. Choosing a name for it must not also drop
    seven loose layers beside it -- that is litter, not a feature."""
    board = board_with(artwork(tmp_path))
    destination = tmp_path / "下单" / "我的板.zip"

    report = export_gerber_archive(board, destination, title="我的板")

    assert Path(str(report["archive"])) == destination
    assert destination.exists()
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "我的板.zip"
    ]


def test_the_single_file_export_contains_the_whole_layer_set(tmp_path):
    """What is inside is the same set the folder export produces, so uploading
    it is equivalent to uploading the folder's zip."""
    board = board_with(artwork(tmp_path, hole=True))
    report = export_gerber_archive(board, tmp_path / "下单.zip", title="demo")

    with zipfile.ZipFile(str(report["archive"])) as bundle:
        packed = set(bundle.namelist())
    assert f"{DEFAULT_PREFIX}{OUTLINE_LAYER.file_name}" in packed
    assert f"{DEFAULT_PREFIX}TopLayer.GTL" in packed
    assert packed == {str(entry["name"]) for entry in report["files"]}


def test_the_single_file_export_refuses_a_board_without_an_outline(tmp_path):
    """The refusal must happen before anything is written, not after a
    half-made zip has already landed where the user asked for it."""
    destination = tmp_path / "下单.zip"
    board = ArtBoard(outline=BoardOutlineSpec(width_mm=50, height_mm=30))

    with pytest.raises(ValueError, match="板框"):
        export_gerber_archive(board, destination)

    assert not destination.exists()


def test_the_single_file_export_reports_what_went_in(tmp_path):
    report = export_gerber_archive(
        board_with(artwork(tmp_path)), tmp_path / "下单.zip", title="我的板"
    )

    assert "顶层铜" in report["layers"]
    assert "板框" in report["layers"]
    assert report["bytes"] > 0
    assert "嘉立创" in str(report["note"])


def test_the_report_lists_what_was_written(tmp_path):
    report = export_gerber(
        board_with(artwork(tmp_path)), tmp_path / "out", title="demo"
    )

    assert "顶层铜" in report["layers"]
    assert "板框" in report["layers"]
    assert "Gerber" in str(report["note"])


# ------------------------------------------------------------- refusals


def test_a_board_without_an_outline_is_refused(tmp_path):
    """Every other problem here has a fallback; this one does not."""
    board = ArtBoard(outline=BoardOutlineSpec(width_mm=50, height_mm=30))

    with pytest.raises(ValueError, match="板框"):
        export_gerber(board, tmp_path / "out")


def test_a_standee_tab_that_did_not_attach_is_refused(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(
            width_mm=50, height_mm=30, standee=True, standee_offset_mm=500.0
        ),
        outline_ready=True,
    )
    board.outline_polygon()

    with pytest.raises(ValueError, match="立牌插舌"):
        export_gerber(board, tmp_path / "out")


# ---------------------------------------------------------- the format


def test_coordinates_are_millimetres_in_the_declared_format():
    text = gerber_text(
        function="Copper,L1,Top", strokes=[[(0.0, 0.0), (1.0, 2.0)]]
    )

    assert "%MOMM*%" in text, "单位必须声明为毫米"
    assert "%FSLAX46Y46*%" in text, "坐标格式必须声明"
    # 1 mm at 4.6 means 1 000 000 units.
    assert "X1000000Y2000000D01*" in text


def test_the_board_outline_is_stroked_not_filled():
    """An edge cut is the line the router follows, not the copper-free area."""
    text = gerber_text(
        function="Profile,NP",
        strokes=[[(0, 0), (10, 0), (10, 10), (0, 10)]],
    )

    assert "G36*" not in text, "板框不应做成填充区域"
    # Four sides of the rectangle, the last one closing it back onto the
    # start point. An open contour would leave the router nowhere to go.
    assert text.count("D01*") == 4
    assert text.rstrip().endswith("M02*"), "文件要有结束命令"


def test_every_statement_is_terminated():
    """Gerber is not line-based: a line without ``*`` swallows the next one.

    Extended commands are wrapped in percent signs, so theirs is the ``*%``
    pair rather than a bare ``*``.
    """
    text = gerber_text(function="Copper,L1,Top", strokes=[[(0, 0), (5, 5)]])

    for line in text.splitlines():
        assert line.endswith("*") or line.endswith("*%"), f"语句没有结束符: {line}"


def test_the_file_is_pure_ascii():
    """Fab-shop parsers are the oldest readers this project writes for."""
    text = gerber_text(
        function="Legend,Top",
        regions=[[[(0, 0), (10, 0), (10, 10), (0, 10)]]],
    )

    text.encode("ascii")  # raises if any non-ASCII slipped in


def test_a_filled_shape_becomes_a_region():
    text = gerber_text(
        function="Copper,L1,Top",
        regions=[[[(0, 0), (10, 0), (10, 10), (0, 10)]]],
    )

    assert "G36*" in text and "G37*" in text, "填充要写成 G36/G37 区域"
    assert "%LPD*%" in text


def test_a_hole_is_cleared_with_clear_polarity():
    """The only way Gerber expresses a hole in a filled shape."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    text = gerber_text(function="Copper,L1,Top", regions=[[outer, hole]])

    assert "%LPC*%" in text, "孔必须用 LPC 清除极性"
    # Dark polarity has to come back, or everything drawn afterwards vanishes.
    assert text.index("%LPC*%") < text.index("%LPD*%", text.index("%LPC*%"))
    assert text.count("G36*") == 2, "外环和孔各是一个区域"


def test_a_real_holed_artwork_exports_with_clear_polarity(tmp_path):
    export_gerber(board_with(artwork(tmp_path, hole=True)), tmp_path / "out")
    text = (tmp_path / "out" / f"{DEFAULT_PREFIX}TopLayer.GTL").read_text(
        encoding="ascii"
    )

    assert "%LPC*%" in text
    assert "G36*" in text and "G37*" in text


def test_the_outline_of_a_standee_includes_the_tab(tmp_path):
    """The tab is part of the board, so it is part of the edge cut."""
    board = board_with(
        artwork(tmp_path), standee=True, standee_width_mm=16, standee_height_mm=3
    )
    export_gerber(board, tmp_path / "out")

    text = (
        tmp_path / "out" / f"{DEFAULT_PREFIX}{OUTLINE_LAYER.file_name}"
    ).read_text(encoding="ascii")
    ys = [
        int(match.group(1)) for match in re.finditer(r"Y(-?\d+)D0[12]\*", text)
    ]

    # The outline was 40 mm tall, and the tab hangs 3 mm below it.
    assert max(ys) - min(ys) == pytest.approx(int(43.0 * 1_000_000))


def test_solder_mask_layers_are_written_when_the_artwork_is_on_one(tmp_path):
    """A Gerber mask layer *is* the set of openings, so unlike the .epro2
    writer this one has somewhere honest to put the geometry."""
    board = board_with(artwork(tmp_path))
    board.placements[0].layer = "顶层阻焊层"

    report = export_gerber(board, tmp_path / "out")

    names = {entry["name"] for entry in report["files"]}
    assert f"{DEFAULT_PREFIX}TopSolderMaskLayer.GTS" in names


def test_the_file_function_attribute_names_the_layer(tmp_path):
    """X2 attributes are how a CAM tool identifies a layer handed to it under
    an ambiguous file name."""
    export_gerber(board_with(artwork(tmp_path)), tmp_path / "out")
    text = (tmp_path / "out" / f"{DEFAULT_PREFIX}TopLayer.GTL").read_text(
        encoding="ascii"
    )

    assert "%TF.FileFunction,Copper,L1,Top*%" in text


def test_every_declared_layer_has_a_unique_file_name():
    names = [spec.file_name for spec in GERBER_LAYERS] + [OUTLINE_LAYER.file_name]
    assert len(names) == len(set(names))
