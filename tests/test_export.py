"""Tests for exporting the design document to a ``.epro2``.

The point of these is the round trip: what the user sees on the canvas is what
lands in the file. A board that draws correctly but writes the wrong size is
exactly the failure M1's acceptance criterion is about, and it is not visible
from either end alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    IrregularBase,
    OutlineShape,
)
from pcbart.epru import EproArchive  # noqa: E402
from pcbart.export import default_title, export_design  # noqa: E402
from pcbart.primitives import outline_bbox_mm  # noqa: E402


def written_outline_size(path: Path) -> tuple[float, float]:
    """The outline's size as the file actually records it, in millimetres."""
    document = EproArchive.read(path).document
    outlines = [
        record
        for record in document.iter_type("POLY")
        if record.body.get("layerId") == 11
    ]
    bbox = outline_bbox_mm(outlines)
    assert bbox is not None, "文件里没有板框"
    min_x, min_y, max_x, max_y = bbox
    return (max_x - min_x, max_y - min_y)


# --------------------------------------------------------------- the basics


def test_a_rectangle_board_exports_with_its_measured_size(tmp_path):
    """M1's acceptance criterion: 50 x 30 opens as 50 x 30."""
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.RECT, width_mm=50, height_mm=30
        ),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")

    assert Path(report["output"]).exists()
    assert written_outline_size(Path(report["output"])) == pytest.approx((50, 30))
    assert report["width_mm"] == pytest.approx(50)
    assert report["height_mm"] == pytest.approx(30)


def test_the_report_matches_the_file(tmp_path):
    """The numbers shown to the user describe the file that was written."""
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.ROUNDED,
            width_mm=64.5,
            height_mm=22.25,
            corner_radius_mm=4,
        ),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")
    measured = written_outline_size(Path(report["output"]))

    assert measured == pytest.approx((report["width_mm"], report["height_mm"]))


def test_a_circular_board_exports_as_its_diameter(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.CIRCLE, width_mm=40, height_mm=40
        ),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")

    assert written_outline_size(Path(report["output"])) == pytest.approx((40, 40))


def test_the_outline_lands_on_the_board_outline_layer(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )

    report = export_design(board, tmp_path / "board.epro2")
    document = EproArchive.read(report["output"]).document

    on_layer_11 = [
        r for r in document.iter_type("POLY") if r.body.get("layerId") == 11
    ]
    assert len(on_layer_11) == 1


def test_a_standee_tab_is_written_into_the_board_outline(tmp_path):
    """The tab is part of the board, not a second piece of geometry.

    A file with the tab as its own POLY would open as a loose rectangle next to
    the board; the tab only means anything as part of the same closed outline
    the base clamps onto. The measured size is asserted as well as the ring
    count, because a badly attached tab can leave both pieces present.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30, standee=True),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "standee.epro2")
    document = EproArchive.read(report["output"]).document

    on_layer_11 = [
        r for r in document.iter_type("POLY") if r.body.get("layerId") == 11
    ]
    assert len(on_layer_11) == 1
    assert written_outline_size(Path(report["output"])) == pytest.approx((50, 33.0))
    assert report["height_mm"] == pytest.approx(33.0), "报告的高度应包含插舌"
    assert board.outline.height_mm == pytest.approx(30.0), "板框本身没有被改大"


def test_a_standee_tab_that_did_not_attach_is_refused(tmp_path):
    """Refusing beats writing a board the user did not ask for.

    The tab is switched on in the panel, so a file that opens without it looks
    fine and is missing the piece the display base holds.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(
            width_mm=50,
            height_mm=30,
            standee=True,
            standee_offset_mm=200.0,
        ),
        outline_ready=True,
    )

    # Reading the outline is what determines whether the tab attached.
    board.outline_polygon()
    assert board.standee_error

    with pytest.raises(ValueError, match="立牌插舌"):
        export_design(board, tmp_path / "standee.epro2")


def test_the_title_from_the_dialog_wins(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )

    report = export_design(board, tmp_path / "board.epro2", title="立牌")

    assert EproArchive.read(report["output"]).meta["title"] == "立牌"
    assert report["title"] == "立牌"


def test_the_title_defaults_to_the_artwork_name(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        image=ImagePlacement(path=str(tmp_path / "由比滨线稿.png")),
        outline_ready=True,
    )

    assert default_title(board) == "由比滨线稿"

    report = export_design(board, tmp_path / "board.epro2")
    assert EproArchive.read(report["output"]).meta["title"] == "由比滨线稿"


def test_a_board_without_artwork_still_gets_a_title(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30), outline_ready=True
    )
    assert default_title(board) == "ArtBoard"


# --------------------------------------------------------------- refusals


def test_exporting_without_a_board_is_refused_with_a_readable_reason(tmp_path):
    """An empty file would look like success; a message does not."""
    board = ArtBoard()

    with pytest.raises(ValueError) as caught:
        export_design(board, tmp_path / "board.epro2")

    assert "板框" in str(caught.value)
    assert not (tmp_path / "board.epro2").exists()


def test_exporting_an_invalid_spec_is_refused(tmp_path):
    """A circle with two different sides draws one thing and means another."""
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.CIRCLE, width_mm=40, height_mm=20
        ),
        outline_ready=True,
    )

    with pytest.raises(ValueError) as caught:
        export_design(board, tmp_path / "board.epro2")

    assert "直径" in str(caught.value)


def test_an_irregular_board_without_a_trace_is_refused(tmp_path):
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.IRREGULAR,
            irregular_base=IrregularBase.TRACE,
            width_mm=50,
            height_mm=30,
        ),
        outline_ready=True,
    )

    with pytest.raises(ValueError):
        export_design(board, tmp_path / "board.epro2")


# --------------------------------------------------- what is not included


def test_the_report_says_when_there_is_no_copper_to_write(tmp_path):
    """The UI must be able to tell the truth about what the file contains.

    Claiming the picture was exported when it was not is precisely the
    interface-lies-about-behaviour problem this project keeps having to fix.
    """
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        image=ImagePlacement(path="art.png", width_mm=20, height_mm=20),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")

    assert report["artwork_exported"] is False
    assert report["artwork_placements"] == 1
    assert report["copper_shapes"] == 0
    # The message has to say *why* nothing was written, not just that nothing was.
    assert "铜层" in report["note"]


def test_silkscreen_artwork_is_written_not_skipped(tmp_path):
    """Silk prints ink, so a shape on it is material in exactly the copper sense.

    Skipping silk used to look harmless because line art is usually placed on
    copper. It is not harmless: a board painting keeps its white tone *and its
    contour* on silkscreen, so writing copper alone exported the dark field and
    silently dropped the outline and every light area.
    """
    import numpy as np

    from pcbart.image_prep import imwrite_unicode

    art = tmp_path / "art.png"
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[15:45, 15:45] = 255
    imwrite_unicode(str(art), mask)

    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        image=ImagePlacement(
            path=str(art),
            width_mm=20,
            height_mm=20,
            layer="顶层丝印层",
            processed_path=str(art),
        ),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")

    assert report["copper_shapes"] >= 1
    document = EproArchive.read(report["output"]).document
    fills = list(document.iter_type("FILL"))
    assert fills, "丝印图案应当写成 FILL"
    assert all(r.body.get("layerId") == 3 for r in fills)


def test_solder_mask_is_reported_as_not_writable(tmp_path):
    """A mask opening is the absence of mask, so it is not a shape."""
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        image=ImagePlacement(
            path="art.png", width_mm=20, height_mm=20, layer="顶层阻焊层"
        ),
        outline_ready=True,
    )

    report = export_design(board, tmp_path / "board.epro2")

    assert report["copper_shapes"] == 0
    assert "阻焊" in report["note"]


def test_copper_is_written_for_a_placement_on_a_copper_layer(tmp_path):
    """End to end: a line art on copper becomes FILL records in the file."""
    import numpy as np

    from pcbart.image_prep import imwrite_unicode
    from pcbart.vectorize import VectorizeParams

    art = tmp_path / "art.png"
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[40:60, 40:60] = 0          # a hole, so the winding is exercised
    imwrite_unicode(str(art), mask)

    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        image=ImagePlacement(
            path=str(art),
            width_mm=20.0,
            height_mm=20.0,
            layer="顶层",
            processed_path=str(art),
        ),
        outline_ready=True,
    )

    report = export_design(
        board, tmp_path / "board.epro2", copper_params=VectorizeParams()
    )

    assert report["artwork_exported"] is True
    assert report["copper_shapes"] >= 1

    document = EproArchive.read(report["output"]).document
    fills = [r for r in document.iter_type("FILL")]
    assert fills, "文件里应当有 FILL 记录"
    assert all(r.body.get("layerId") == 1 for r in fills)
    # The hole must survive as a second ring of the same shape.
    assert any(len(r.body.get("path") or []) > 1 for r in fills)


def board_painting(tmp_path, *, layers=("顶层丝印层", "顶层")):
    """A two-tone painting on disk, as the preprocessing panel would leave it."""
    import numpy as np

    from pcbart.image_prep import imwrite_unicode

    # White tone: a ring. Black tone: its complement, written in the canvas's
    # ink convention (0 where the tone is set).
    white = np.full((80, 80), 255, np.uint8)
    white[20:60, 20:60] = 0
    white[30:50, 30:50] = 255
    black = 255 - white

    white_path = tmp_path / "white.png"
    black_path = tmp_path / "black.png"
    imwrite_unicode(str(white_path), white)
    imwrite_unicode(str(black_path), black)

    placements = [
        ImagePlacement(
            path=str(white_path),
            processed_path=str(white_path),
            width_mm=20.0,
            height_mm=20.0,
            layer=layers[0],
            tone="white",
            import_id="imp1",
        ),
        ImagePlacement(
            path=str(black_path),
            processed_path=str(black_path),
            width_mm=20.0,
            height_mm=20.0,
            layer=layers[1],
            tone="black",
            import_id="imp1",
        ),
    ]
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=50, height_mm=30),
        outline_ready=True,
    )
    for placement in placements:
        board.add_placement(placement)
    return board


def test_a_board_painting_writes_both_tones(tmp_path):
    report = export_design(board_painting(tmp_path), tmp_path / "paint.epro2")

    assert report["copper_shapes"] >= 2
    assert report["copper_layers"] == ["顶层丝印层", "顶层"]


def test_the_two_tones_tile_without_gaps(tmp_path):
    """The black tone must be derived from the white one, not traced separately.

    Tracing each mask on its own sends them through separate simplification, so
    they stop sharing an edge and leave a hairline of bare board between them --
    measured at 88 mm² uncovered on a real painting. A gap that small is
    invisible in a thumbnail and obvious on the finished board.
    """
    shapely = pytest.importorskip("shapely.geometry")
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    report = export_design(board_painting(tmp_path), tmp_path / "paint.epro2")
    document = EproArchive.read(report["output"]).document

    by_layer: dict[int, list] = {}
    for record in document.iter_type("FILL"):
        rings = []
        raw = record.body["path"]
        groups = raw if raw and isinstance(raw[0], list) else [raw]
        for group in groups:
            nums = [v for v in group if isinstance(v, (int, float))]
            pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            rings.append([(x * 0.0254, y * 0.0254) for x, y in pairs])
        polygon = Polygon(rings[0], rings[1:])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        by_layer.setdefault(record.body["layerId"], []).append(polygon)

    silk = unary_union(by_layer[3])
    copper = unary_union(by_layer[1])

    assert silk.area > 0 and copper.area > 0
    assert silk.intersection(copper).area == pytest.approx(0.0, abs=1e-6)

    frame = silk.union(copper).envelope
    assert frame.difference(silk.union(copper)).area == pytest.approx(0.0, abs=1e-6)


def test_a_painting_only_writes_the_faces_it_was_placed_on(tmp_path):
    report = export_design(
        board_painting(
            tmp_path, layers=("底层丝印层", "底层")
        ),
        tmp_path / "paint.epro2",
    )

    assert report["copper_layers"] == ["底层丝印层", "底层"]


def test_every_shape_exported_is_a_valid_polygon(tmp_path):
    """Validity, not just size: a self-intersecting ring measures correctly.

    The rounded rectangle was self-intersecting for a while and every dimension
    check passed, so this asserts the shape of what lands in the file.
    """
    shapely = pytest.importorskip("shapely.geometry")
    from shapely.geometry import Polygon

    from pcbart.artboard import ArtBoard as _ArtBoard

    for shape in (OutlineShape.RECT, OutlineShape.ROUNDED, OutlineShape.CIRCLE):
        board = _ArtBoard(
            outline=BoardOutlineSpec(
                shape=shape,
                width_mm=60,
                height_mm=60 if shape == OutlineShape.CIRCLE else 40,
                corner_radius_mm=5,
            ),
            outline_ready=True,
        )
        report = export_design(board, tmp_path / f"{shape}.epro2")
        document = EproArchive.read(report["output"]).document
        outline = next(
            r for r in document.iter_type("POLY") if r.body.get("layerId") == 11
        )
        # The file stores the ring in its own units and Y direction; going back
        # through the canvas's own outline gives the same points the user drew.
        points = [
            (round(x, 6), round(y, 6)) for x, y in board.outline_polygon()
        ]
        assert Polygon(points).is_valid, f"{shape} 的轮廓自交"
