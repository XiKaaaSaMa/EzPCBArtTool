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


# ------------------------------------------------- the 通行证 card's content


def badge_board():
    """A finished 通行证: the card's fixed artwork, wording and all four codes."""
    from pcbart import badge_card, badge_content

    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    board.placements = list(
        badge_content.badge_art(board=board, number="PL03").placements
    )
    return board


def test_the_card_writes_every_layer_including_the_solder_masks(tmp_path):
    """The fab package has to carry the card's own artwork, not just the picture.

    ⚠️ **This is the bug this test exists for.** The Gerber path walked the
    placements and the wording but never the template's fixed decorations, so a
    card exported for manufacture came out with **empty solder-mask layers** --
    every opening that lets the illustration and both codes show through was
    missing -- and with no border, no wordmark and no crest. The board still
    validated and still uploaded, so only the finished boards would have shown
    it.
    """
    from pcbart.badge_art import decoration_geometry

    board = badge_board()
    regions, strokes = decoration_geometry(board)
    assert regions and strokes, "模板装饰没有变成可写出的几何"

    report = export_gerber(board, tmp_path / "out", title="卡")
    names = {entry["name"] for entry in report["files"]}

    assert report["skipped"] == [], f"仍有空层：{report['skipped']}"
    for suffix in (".GTL", ".GBL", ".GTO", ".GBO", ".GTS", ".GBS", ".GKO"):
        assert any(name.endswith(suffix) for name in names), f"缺少 {suffix}"


def test_the_solder_mask_holds_the_cards_openings(tmp_path):
    """The mask layers carry real geometry, and it lands where the mask is cut.

    Measured, not counted: the layer's own bounding box has to match the
    openings the card's decorations describe.
    """
    import re

    from pcbart.badge_art import decoration_geometry
    from pcbart.design_layers import layer

    board = badge_board()
    export_gerber(board, tmp_path / "out", title="卡")

    regions, stroke_paths = decoration_geometry(board)
    # Both spellings reach the file: openings are drawn as regions, and a rule
    # or a line on the mask comes through as a stroke. The box has to cover
    # both, or a stroke that ran outside it would read as "in range".
    expected: dict[int, tuple[float, float, float, float]] = {}
    for layer_id, rings in regions:
        for ring in rings:
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            box = (min(xs), min(ys), max(xs), max(ys))
            if layer_id not in expected:
                expected[layer_id] = box
                continue
            old = expected[layer_id]
            expected[layer_id] = (
                min(old[0], box[0]),
                min(old[1], box[1]),
                max(old[2], box[2]),
                max(old[3], box[3]),
            )
    for layer_id, pts, _width, _closed in stroke_paths:
        xs = [point[0] for point in pts]
        ys = [point[1] for point in pts]
        if layer_id not in expected:
            expected[layer_id] = (min(xs), min(ys), max(xs), max(ys))
            continue
        old = expected[layer_id]
        expected[layer_id] = (
            min(old[0], min(xs)),
            min(old[1], min(ys)),
            max(old[2], max(xs)),
            max(old[3], max(ys)),
        )

    for spec, suffix in (("top_mask", ".GTS"), ("bottom_mask", ".GBS")):
        text = next(
            (tmp_path / "out").glob(f"*{suffix}")
        ).read_text(encoding="ascii")
        coords = [
            (int(m.group(1)) / 1e6, int(m.group(2)) / 1e6)
            for m in re.finditer(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)
        ]
        assert coords, f"{suffix} 里没有任何坐标"
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]
        layer_id = layer(spec).easyeda_id
        # The file carries the openings; every coordinate must fall inside the
        # box the decorations declare for that layer.
        want = expected[layer_id]
        assert min(xs) >= want[0] - 1e-3 and max(xs) <= want[2] + 1e-3, (
            f"{suffix} 的 X 超出开窗范围：{min(xs)}..{max(xs)} vs {want}"
        )
        assert min(ys) >= want[1] - 1e-3 and max(ys) <= want[3] + 1e-3, (
            f"{suffix} 的 Y 超出开窗范围：{min(ys)}..{max(ys)} vs {want}"
        )


def test_the_board_outline_stays_the_card_size(tmp_path):
    """⚠️ The decorations must not disturb the outline it is written beside.

    A local loop variable once shadowed the outline's own ``points``, so the
    outline file came out holding the last decoration path -- a 45 x 76 mm
    fragment instead of the card. Nothing raised; the board simply would have
    been routed wrong. Read back from the written file so the assertion is
    about the artefact, not about a variable.
    """
    import re

    from pcbart.badge_card import BADGE_OUTLINE

    board = badge_board()
    export_gerber(board, tmp_path / "out", title="卡")
    text = (tmp_path / "out" / f"{DEFAULT_PREFIX}BoardOutlineLayer.GKO").read_text(
        encoding="ascii"
    )
    coords = [
        (int(m.group(1)) / 1e6, int(m.group(2)) / 1e6)
        for m in re.finditer(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)
    ]
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    # These are the *centreline* coordinates, which are the board's own size:
    # the cut follows the centre of a 0.125 mm cutter, so the kerf extends half
    # of that beyond each edge. Asserting 50.125 here would be asserting the
    # kerf, not the card.
    assert width == pytest.approx(50.0, abs=0.01), f"板框宽 {width}"
    assert height == pytest.approx(100.0, abs=0.01), f"板框高 {height}"
    # And the outline really is the card's own ring, not a decoration subset.
    assert len(coords) > len(BADGE_OUTLINE) * 2


# ---------------------------------------------------------- drilled openings


def test_the_card_writes_a_drill_file_for_its_round_holes(tmp_path):
    """The two hanging holes reach the fabricator as drill hits, not only as
    outline contours.

    ⚠️ **Why both.** The board outline layer walks every opening as a closed
    path the router follows, which is right for the lanyard slot and works for a
    round hole too. But a round hole is also literally a drill hit, and that is
    what 嘉立创's own export writes: the sample order package carries
    ``Drill_NPTH_Through.DRL`` with one tool and two coordinates. Omitting it
    left the fab with a contour to interpret rather than a hole to drill.
    """
    import re

    from pcbart.gerber import DRILL_FILE_NAME

    board = badge_board()
    report = export_gerber(board, tmp_path / "out", title="卡")
    names = {entry["name"] for entry in report["files"]}
    assert f"{DEFAULT_PREFIX}{DRILL_FILE_NAME}" in names
    assert report["holes"] == 2, "两张吊孔"

    text = (tmp_path / "out" / f"{DEFAULT_PREFIX}{DRILL_FILE_NAME}").read_text(
        encoding="ascii"
    )
    # One tool, declared once, used twice -- the shape the sample package has.
    assert "M48" in text and text.rstrip().endswith("M30")
    assert "METRIC,LZ,0000.00000" in text
    assert text.count("C3.56000") == 1, "同一孔径只声明一次"
    hits = re.findall(r"X(-?[\d.]+)Y(-?[\d.]+)", text)
    assert len(hits) == 2, f"应当有两个钻孔坐标，实际 {hits}"


def test_the_drill_file_is_ascii_and_only_lists_round_openings(tmp_path):
    """A slot is routed, not drilled, and a Chinese byte breaks the uploader.

    The board carries three openings: two round hanging holes and one obround
    lanyard slot. Listing the slot as a drill hit would tell the fabricator to
    plunge a bit and drag it, which is a different operation from the routed
    contour the outline layer already describes.
    """
    from pcbart.gerber import DRILL_FILE_NAME, drill_holes

    board = badge_board()
    round_only = [item for item in board.cutouts if item.shape == "circle"]
    assert len(round_only) == 2 and len(board.cutouts) == 3

    holes = drill_holes(board)
    assert len(holes) == len(round_only)
    assert all(diameter == pytest.approx(3.56) for _, _, diameter in holes)

    export_gerber(board, tmp_path / "out", title="卡")
    raw = (tmp_path / "out" / f"{DEFAULT_PREFIX}{DRILL_FILE_NAME}").read_bytes()
    raw.decode("ascii")  # raises if a non-ASCII byte reached the file


def test_a_board_without_round_holes_writes_no_drill_file(tmp_path):
    """No empty files: an uploadable set must not carry a drill job with no
    holes in it."""
    from pcbart.gerber import DRILL_FILE_NAME

    from pcbart.artboard import ArtBoard

    from pcbart.artboard import BoardOutlineSpec, OutlineShape

    board = ArtBoard(outline_ready=True)
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=50.0, height_mm=30.0
    )
    report = export_gerber(board, tmp_path / "out", title="空")
    names = {entry["name"] for entry in report["files"]}
    assert f"{DEFAULT_PREFIX}{DRILL_FILE_NAME}" not in names
    assert report["holes"] == 0


def test_a_fragile_outline_warns_in_the_report_the_user_reads(tmp_path):
    """⚠️ A warning the user never sees is not a warning.

    ``check_outline`` computes a thin-neck warning and both writers used to
    discard it, so a board that snaps in the post exported with a note that said
    everything was fine. The report is the only place the user looks.
    """
    from pcbart.artboard import ArtBoard
    from pcbart.gerber import export_gerber

    from pcbart.artboard import BoardOutlineSpec, OutlineShape

    board = ArtBoard(outline_ready=True)
    # A template outline is how an arbitrary ring reaches the spec, so the
    # fixture uses the real path rather than reaching into the geometry. Two
    # 20 x 20 mm blocks joined by a 0.4 mm bridge: the bridge is below the
    # 0.5 mm floor MIN_NECK_WIDTH_MM sets, and above the 0.2 mm routed-slot
    # minimum, so the board is legal to make and still worth warning about.
    ring = (
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 9.8),
        (30.0, 9.8),
        (30.0, 0.0),
        (50.0, 0.0),
        (50.0, 20.0),
        (30.0, 20.0),
        (30.0, 10.2),
        (20.0, 10.2),
        (20.0, 20.0),
        (0.0, 20.0),
    )
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.TEMPLATE, template_outline=ring
    )

    report = export_gerber(board, tmp_path / "out", title="细颈")
    assert report["warnings"], "细颈必须出现在报告的 warnings 里"
    assert any("最窄处" in text for text in report["warnings"])
    assert "⚠️" in report["note"], "警示要出现在用户实际读的那句 note 里"
