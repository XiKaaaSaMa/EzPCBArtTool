"""Tests for putting an operator's QR symbol onto the 通行证 card.

The card reserves four operator-specific spots, each marked by a filled block
the user drew. Only the QR symbol is placed so far. What these tests protect:

* the symbol lands on the **block**, not on the text that names the block --
  positioning from the text shipped once and was 3.3 mm out;
* it is the size that makes its 45-degree corners meet the block's corners;
* its background is transparent, so the card's own artwork shows through;
* the block and the placeholder wording come *off* the card once it is there.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_art, badge_content, operators  # noqa: E402


@pytest.fixture(scope="module")
def database():
    if not operators.database():
        pytest.skip("干员素材库未安装")
    return operators.database()


@pytest.fixture(scope="module")
def art(database):
    return badge_content.badge_art(database.find("阿米娅"))


# ------------------------------------------------------------------ the block

def test_the_symbol_lands_on_the_block_the_user_drew(art):
    """The block is the instruction; the text naming it is only a label."""
    region = badge_content.art_region("qr")
    assert (art.qr.x_mm, art.qr.y_mm) == pytest.approx(region.centre)


def test_the_block_is_measured_where_the_source_file_has_it():
    """Guards against a typo in the table, which nothing else would catch."""
    region = badge_content.art_region("qr")
    assert region.centre == pytest.approx((-21.7279, -26.7969))
    assert (region.width_mm, region.height_mm) == pytest.approx((3.1997, 3.1997))
    assert region.rotation_deg == 45.0
    assert region.layer == "bottom_silk"


def test_the_symbol_is_drawn_on_the_bottom_silkscreen(art):
    assert art.qr.layer == badge_content.QR_LAYER == "底层丝印层"


def test_the_code_area_reaches_the_block_corners(art):
    """The *code*, not the bitmap, is what has to meet the drawn block.

    The user's report was that the symbol left a fat empty margin inside the
    diamond. It did: the block's 3.1997 mm side was being spent on the whole
    bitmap, quiet zone included, so the three finder patterns stopped 4 modules
    short of every corner. The block measures the 21-module code area instead,
    and the transparent quiet zone falls outside it.
    """
    from pcbart import badge_codes

    region = badge_content.art_region("qr")
    assert art.qr.rotation_deg == region.rotation_deg == 45.0
    code = badge_codes.qr_for_operator("阿米娅", code_mm=region.width_mm)
    assert code.version == 1
    assert code.layout.total_modules == 29
    # 21 code modules + 4 quiet modules per side; the *code* area is the block.
    assert code.layout.border_modules == 4
    assert code.layout.module_mm * 21 == pytest.approx(region.width_mm, abs=1e-6)
    # Turning it 45 degrees puts the code area's diagonal on the block's.
    assert code.layout.module_mm * 21 * math.sqrt(2) == pytest.approx(
        region.width_mm * math.sqrt(2), abs=1e-6
    )


def test_the_bitmap_is_bigger_than_the_block_by_its_quiet_zone(art):
    """The quiet zone still exists, outside the block, where a scanner needs it."""
    region = badge_content.art_region("qr")
    assert art.qr.width_mm > region.width_mm
    quiet = art.qr.width_mm - region.width_mm
    # 8 modules total (4 per side) at the code area's own module size.
    assert quiet == pytest.approx(8 * region.width_mm / 21, abs=1e-6)


def test_the_block_measures_the_code_area_not_the_bitmap(art):
    """The distinction that caused the fat margin, pinned end to end.

    ``badge_art`` has to call the encoder with ``code_mm`` on the region size.
    Passing the same number as ``area_mm`` -- what it used to do -- spends the
    block on the bitmap and buries the finder patterns 4 modules inside it, so
    the placement ends up smaller than the block while still *looking* right in
    the numbers.
    """
    from pcbart import badge_codes

    region = badge_content.art_region("qr")
    code_modules = 29 - 2 * 4
    placed_code_mm = art.qr.width_mm * code_modules / 29
    assert placed_code_mm == pytest.approx(region.width_mm, abs=1e-6)
    # And the old, wrong reading is a different size by more than any rounding,
    # so a regression cannot hide inside the tolerance above.
    wrong = badge_codes.qr_for_operator("阿米娅", area_mm=region.width_mm)
    wrong_code_mm = wrong.module_mm * code_modules
    assert wrong_code_mm == pytest.approx(2.3168, abs=0.001)
    assert region.width_mm - wrong_code_mm > 0.8


def test_the_cached_bitmap_is_dense_enough_to_survive_tracing(art):
    """32 px/module is an export requirement, not a display preference.

    The bitmap is what the exporter traces back into geometry. At 12 px/module
    the trace put 11 of 841 module centres on the wrong side of an edge;
    measured 0 of 841 at 32. Lowering this to save a few hundred kilobytes
    silently fills bare QR cells in the exported board.
    """
    from PIL import Image

    region = badge_content.art_region("qr")
    total_modules = 29
    code_modules = total_modules - 2 * 4
    with Image.open(art.qr.path) as image:
        assert image.width == badge_content.QR_MODULE_PX * total_modules
    assert badge_content.QR_MODULE_PX >= 32
    # And it has to stay in step with the region, not drift to a new size.
    assert art.qr.width_mm * code_modules / total_modules == pytest.approx(
        region.width_mm, abs=1e-6
    )


def test_the_code_module_is_now_above_the_print_floor(art):
    """Enlarging the symbol also fixed the module that was under 0.15 mm.

    The old 3.2 mm bitmap made a module 0.1103 mm, below JLCPCB's quoted
    silkscreen floor; the user had to print a sample to prove it scanned. The
    code area measures 3.1997 mm across 21 modules now, so the module is
    0.1524 mm and no longer below the floor the warning names.
    """
    from pcbart import badge_codes

    region = badge_content.art_region("qr")
    code = badge_codes.qr_for_operator("阿米娅", code_mm=region.width_mm)
    assert code.version == 1
    assert code.layout.total_modules == 29
    assert code.module_mm == pytest.approx(0.15237, abs=0.00001)
    assert code.module_mm >= badge_codes.SCAN_MODULE_WARN_MM
    assert code.scan_warning == ""


# ------------------------------------------------------------- transparency

def test_the_symbol_has_no_background(art):
    """A white field would punch a square hole in the card's own artwork."""
    with Image.open(art.qr.path) as image:
        assert image.mode == "RGBA"
        alpha = np.array(image.getchannel("A"))
    assert set(np.unique(alpha).tolist()) == {0, 255}, "只应有全透明与完全不透明"
    assert (alpha == 0).any(), "背景必须是透明的"
    assert (alpha == 255).any(), "模块必须是不透明的"


def test_the_transparent_symbol_still_decodes(art):
    """The quiet zone is transparent, not absent: a scanner must still read it."""
    with Image.open(art.qr.path) as image:
        rgba = np.array(image.convert("RGBA"))
    alpha = rgba[..., 3:4] / 255.0
    flat = (rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    # Scale to a fixed *output* size rather than by a fixed factor: the cached
    # symbol is drawn at several pixels per module, so a factor that used to give
    # a readable 1200 px image now gives a 5600 px one, which is past what the
    # detector will look at.
    target = 900
    scale = max(1, target // max(flat.shape[:2]))
    big = cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    big = cv2.copyMakeBorder(big, 48, 48, 48, 48, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    read, _points, _ = cv2.QRCodeDetector().detectAndDecode(big)
    assert read == "i8c.cn/xE9At"


def test_the_placement_does_not_go_through_the_line_art_pipeline(art):
    """A traced QR code is a QR code that no longer decodes."""
    assert art.qr.processed_path == ""
    assert art.barcode.processed_path == ""


# ---------------------------------------------------------------- the barcode

def test_the_barcode_lands_on_its_own_block(art):
    """Not on the text that names the block -- that mistake was 3.5 mm out."""
    region = badge_content.art_region("barcode")
    assert (art.barcode.x_mm, art.barcode.y_mm) == pytest.approx(region.centre)


def test_the_barcode_block_is_measured_where_the_source_file_has_it():
    region = badge_content.art_region("barcode")
    assert region.centre == pytest.approx((-22.482, -28.839))
    assert (region.width_mm, region.height_mm) == pytest.approx((4.941, 39.245))
    assert region.layer == "top_silk"


def test_the_barcode_bars_fill_the_frame_edge_to_edge(art):
    """**The user's requirement: the proportions match the drawn frame exactly.**

    The frame is 39.245 mm along the bars, and the frame can only ever mark the
    bars -- the ten-module quiet zone on each end is empty space with nothing to
    draw. So the bars span the frame exactly and the quiet zone falls outside
    it. Sizing the whole file to the frame, which is what the old code did,
    left the first and last bar one quiet zone inside each edge.
    """
    from pcbart import badge_codes

    region = badge_content.art_region("barcode")
    frame_mm = max(region.width_mm, region.height_mm)
    code = badge_codes.barcode_for("R001", code_length_mm=frame_mm)
    assert code.layout.code_length_mm == pytest.approx(frame_mm, abs=1e-9)
    # The placement is the *file*, bar height plus the quiet zone along it.
    assert art.barcode.width_mm == pytest.approx(code.layout.length_mm)
    assert art.barcode.height_mm == pytest.approx(code.layout.height_mm)
    assert art.barcode.width_mm > frame_mm


def test_the_barcode_module_is_wide_enough_to_print(art):
    """A wider module than the old reading is the point of the change.

    The old whole-file-in-the-frame reading made a module 0.3964 mm; the bars
    now span the same frame across 79 modules instead of 99.
    """
    from pcbart import badge_codes

    region = badge_content.art_region("barcode")
    code = badge_codes.barcode_for(
        "R001", code_length_mm=max(region.width_mm, region.height_mm)
    )
    assert code.layout.code_modules == 79
    assert code.module_mm == pytest.approx(39.245 / 79, abs=1e-9)
    assert code.scan_warning == ""


def test_the_barcode_is_turned_to_fill_the_tall_frame(art):
    """The strip is long across and short down, so it needs a quarter turn."""
    region = badge_content.art_region("barcode")
    assert art.barcode.rotation_deg == badge_content.BARCODE_ROTATION_DEG == 270.0
    # After the turn, the long side runs along the frame's long side.
    corners = art.barcode.corners()
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    assert (max(ys) - min(ys)) > (max(xs) - min(xs)), "转完应该是竖长的"


def test_the_barcode_reads_the_same_way_as_the_number_beside_it(art):
    """``ARKNIGHTS-编号`` is stored at 270 degrees; the barcode matches it.

    Both run top-to-bottom. The number is text and its direction is measured by
    rendering it; the barcode's is its own rotation. If they disagree the two
    codes on the card point opposite ways.
    """
    from pcbart import badge_card

    vertical = next(
        item
        for item in badge_card.badge_texts()
        if item.label == "编号（竖排）"
    )
    assert vertical.rotation_deg == art.barcode.rotation_deg == 270.0


def test_the_barcode_has_no_background(art):
    with Image.open(art.barcode.path) as image:
        assert image.mode == "RGBA"
        alpha = np.array(image.getchannel("A"))
    assert set(np.unique(alpha).tolist()) == {0, 255}
    assert (alpha == 0).any() and (alpha == 255).any()


def test_the_barcode_bitmap_has_one_pixel_density(art):
    """Both axes at the same pixels per module, or the canvas squashes it.

    The canvas fits the picture into its box by the bitmap's own pixel aspect.
    A strip drawn at 32 px per module across and 600 dpi down had a pixel aspect
    of 27.08 against a physical 9.95, which rendered it 2.7x too short.
    """
    with Image.open(art.barcode.path) as image:
        width_px, height_px = image.size
    total_modules = 79 + 2 * 10  # 79 bar modules + the quiet zone
    assert width_px == badge_content.BARCODE_MODULE_PX * total_modules
    # The same pixels per module on the other axis, which is what keeps the
    # canvas from fitting a squashed picture into the frame.
    module_px = width_px / total_modules
    modules_tall = art.barcode.height_mm / art.barcode.width_mm * total_modules
    assert height_px == pytest.approx(module_px * modules_tall, rel=0.01)


def test_an_operator_with_no_number_simply_gets_no_barcode(database):
    """24 operators have no 情报编号; that is not an error.

    The QR symbol and the profession plate are unaffected -- they do not come
    from the number column -- so the card is complete apart from the strip.
    """
    from pcbart.artboard import ArtBoard

    person = next((item for item in database.operators if not item.number), None)
    if person is None:
        pytest.skip("这个索引里每位干员都有编号")
    board = ArtBoard()
    board.operator_name = person.name
    board.operator_page = person.page
    art = badge_content.badge_art(board=board)
    assert art.qr is not None
    assert art.barcode is None
    # The icon is a separate column, so this operator still gets one.
    assert art.profession is not None
    assert len(art.placements) == 2


# ------------------------------------------------------- the block comes off

def test_the_five_positioning_blocks_are_hidden():
    labels = set(badge_art.marker_labels().values())
    assert labels == {
        "立绘区",
        "职业区",
        "阵营logo区",
        "条形码区",
        "干员二维码区",
    }

def test_hiding_removes_them_from_the_drawn_decorations():
    assert len(badge_art.visible_decorations()) < len(badge_art.decorations())
    hidden = set(badge_art.marker_labels())
    for index, item in enumerate(badge_art.decorations()):
        if index in hidden:
            assert item not in badge_art.visible_decorations()


def test_hiding_also_removes_them_from_the_written_records():
    """A block hidden on the canvas but written to the file comes back."""
    assert len(badge_art.visible_records()) < len(badge_art.records())


def test_the_placement_lines_are_the_ones_the_panel_removes():
    from pcbart.badge_card import BADGE_TEXTS

    labels = {item.label for item in BADGE_TEXTS}
    for label in badge_content.ART_ELEMENT_LABELS:
        if label == "干员二维码":
            # A board saved before the line was renamed carries this spelling.
            assert label not in labels
            continue
        assert label in labels, label


# -------------------------------------------------------------------- basics

def test_two_operators_do_not_share_a_cached_symbol(database):
    first = badge_content.badge_art(database.find("阿米娅"))
    second = badge_content.badge_art(database.find("鸿雪"))
    assert first.qr.path != second.qr.path


def test_operators_sharing_a_display_name_get_separate_cache_files(database):
    """暮落 and 暮落(集成战略) are different people with different codes."""
    plain = database.find("暮落")
    alternate = next(
        (item for item in database.operators if item.page == "暮落(集成战略)"), None
    )
    if plain is None or alternate is None:
        pytest.skip("这个索引里没有暮落的两笔记录")
    first = badge_content.badge_art(plain)
    second = badge_content.badge_art(alternate)
    assert first.qr.path != second.qr.path
    assert Path(first.qr.path).exists() and Path(second.qr.path).exists()


def test_a_card_resolves_the_operator_by_page_not_by_name(database):
    """The board stores the page name, because the display name is not unique.

    A card for 暮落(集成战略) carries ``operator_name == "暮落"`` too, so
    resolving by name handed that person the *other* 暮落's QR symbol. The page
    name is the roster's key and is what the document has to keep.
    """
    from pcbart.artboard import ArtBoard

    alternate = database.find_page("暮落(集成战略)")
    if alternate is None:
        pytest.skip("这个索引里没有暮落的两笔记录")
    board = ArtBoard()
    board.operator_name = alternate.name
    board.operator_page = alternate.page
    art = badge_content.badge_art(board=board)
    expected = badge_content.badge_codes.qr_for_operator("暮落(集成战略)")
    decoded = badge_content.badge_codes.qr_for_operator("暮落")
    assert expected.payload != decoded.payload
    assert Path(art.qr.path).name == "暮落(集成战略).png"


def test_a_board_without_a_page_name_still_resolves_by_name(database):
    """Older drafts predate the page field, so the name remains a fallback."""
    from pcbart.artboard import ArtBoard

    board = ArtBoard()
    board.operator_name = "阿米娅"
    art = badge_content.badge_art(board=board)
    assert Path(art.qr.path).name == "阿米娅.png"


def test_every_item_this_module_adds_shares_one_import_id(art):
    """That id is how the panel replaces the placement instead of stacking."""
    assert {item.import_id for item in art.placements} == {badge_content.ART_IMPORT_ID}


def test_a_board_with_no_operator_produces_nothing():
    from pcbart.artboard import ArtBoard

    plan = badge_content.badge_art(board=ArtBoard())
    assert not plan
    assert plan.placements == ()


def test_an_unknown_operator_is_reported_not_guessed():
    from pcbart.artboard import ArtBoard

    board = ArtBoard()
    board.operator_name = "查无此人"
    with pytest.raises(badge_content.MissingArt):
        badge_content.badge_art(board=board)


def test_a_typed_number_drives_the_barcode_not_the_roster(database):
    """An edited 编号 has to reach the strip, not just the wording.

    The card's number is printed twice -- as text and as Code 128 -- and the
    two agreeing is the whole point. Building the strip from the roster while
    the text shows the typed value would put a number on the card that
    contradicts its own barcode.
    """
    from pcbart.artboard import ArtBoard
    from pcbart import badge_codes

    board = ArtBoard()
    board.operator_name = "阿米娅"
    board.operator_page = "阿米娅"
    board.operator_number = "CUSTOM-7"
    art = badge_content.badge_art(board=board, number=board.operator_number)
    assert art.barcode is not None

    expected = badge_codes.barcode_for(
        "CUSTOM-7",
        code_length_mm=max(
            badge_content.art_region("barcode").width_mm,
            badge_content.art_region("barcode").height_mm,
        ),
    )
    rendered = badge_codes.barcode_for(
        "CUSTOM-7",
        code_length_mm=max(
            badge_content.art_region("barcode").width_mm,
            badge_content.art_region("barcode").height_mm,
        ),
    )
    assert rendered.layout.runs == expected.layout.runs
    # And it is genuinely not the roster's strip.
    roster = badge_codes.barcode_for("R001")
    assert rendered.layout.runs != roster.layout.runs


def test_an_edited_number_does_not_reuse_the_cached_strip(database):
    """The cache is keyed by page name, so an edit must change the key.

    ``_cache_png`` returns an existing file untouched. Keyed on the page alone,
    an edited number would be handed back the *original* strip: the edit would
    look like it worked everywhere except on the board.
    """
    from pcbart.artboard import ArtBoard

    board = ArtBoard()
    board.operator_name = "阿米娅"
    board.operator_page = "阿米娅"
    board.operator_number = "R001"
    original = badge_content.badge_art(board=board, number="R001").barcode

    edited = badge_content.badge_art(board=board, number="HELLO-1").barcode
    assert Path(edited.path) != Path(original.path)
    assert Path(edited.path).exists()


def test_clearing_the_number_removes_the_barcode(database):
    """An empty number is a choice, not an error: the card prints no strip."""
    from pcbart.artboard import ArtBoard

    board = ArtBoard()
    board.operator_name = "阿米娅"
    board.operator_page = "阿米娅"
    art = badge_content.badge_art(board=board, number="")
    assert art.qr is not None
    assert art.barcode is None


# --------------------------------------------------------- profession icon

def test_the_profession_plate_fills_the_block(art):
    """The user's requirement: the plate is as big as the drawn block."""
    region = badge_content.art_region("profession")
    assert art.profession is not None
    assert (art.profession.x_mm, art.profession.y_mm) == pytest.approx(
        region.centre
    )
    # Sized to the region's own two sides, not to its shorter one: the block is
    # 7.600 x 7.601 mm, a rounding artefact of the source file.
    assert art.profession.width_mm == pytest.approx(region.width_mm)
    assert art.profession.height_mm == pytest.approx(region.height_mm)
    assert art.profession.rotation_deg == region.rotation_deg == 0.0
    assert art.profession.layer == badge_content.PROFESSION_LAYER == "顶层丝印层"


def test_the_plate_is_ink_and_the_glyph_is_a_hole(art):
    """背景部分丝印，图标部分镂空.

    The plate is opaque everywhere the glyph is not, and transparent where it
    is -- so the board's own colour reads as the symbol. The corners are the
    plate, which is what distinguishes this from an icon-shaped sticker.
    """
    import numpy as np
    from PIL import Image

    with Image.open(art.profession.path) as image:
        assert image.mode == "RGBA"
        alpha = np.array(image.getchannel("A"))
    assert set(np.unique(alpha).tolist()) == {0, 255}, "只应有丝印与镂空两态"
    assert (alpha == 255).any() and (alpha == 0).any()
    # The plate is the majority and it reaches the corners.
    assert (alpha == 255).mean() > 0.7
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert alpha[corner] == 255, f"角 {corner} 应该是丝印底"


def test_the_export_writes_the_plate_with_its_hole(art):
    """A plate exported as a solid block is a featureless square on the board.

    ``_mask_and_dims`` takes alpha as the shape, so the transparency built by
    the rasteriser is what makes the tracer emit one filled polygon with the
    glyph as its hole.
    """
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    from pcbart import export
    from pcbart.artboard import ArtBoard
    from pcbart.design_layers import LAYERS

    def signed(ring):
        return sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
        )

    def as_polygon(rings):
        ordered = [list(ring) for ring in rings]
        if signed(ordered[0]) < 0:
            ordered[0] = list(reversed(ordered[0]))
        holes = [
            list(reversed(ring)) if signed(ring) > 0 else ring
            for ring in ordered[1:]
        ]
        polygon = Polygon(ordered[0], holes)
        return polygon if polygon.is_valid else make_valid(polygon)

    board = ArtBoard()
    board.operator_name = "阿米娅"
    board.operator_page = "阿米娅"
    built = badge_content.badge_art(board=board)
    board.placements = list(built.placements)

    records, _report = export.build_copper(board)
    layer_id = next(
        entry.easyeda_id
        for entry in LAYERS
        if entry.key == built.profession.layer_key
    )
    plate = built.profession
    shapes = [
        as_polygon(rings)
        for record_layer, rings, _width in records
        if record_layer == layer_id
        and all(
            abs(x - plate.x_mm) <= plate.width_mm
            and abs(y - plate.y_mm) <= plate.height_mm
            for x, y in rings[0]
        )
    ]
    assert shapes, "导出里找不到职业图标几何"
    union = unary_union(shapes)
    full = plate.width_mm * plate.height_mm
    # Real holes, not a solid square: roughly a tenth of the plate is cut out.
    assert union.area < full * 0.97, "整个块是实心的，镂空丢了"
    assert union.area > full * 0.7, "镂空太多，多半把丝印底也切掉了"
    # Centre is inside the glyph, so there is no ink there; the corners are plate.
    assert not union.contains(Point(plate.x_mm, plate.y_mm))
    assert union.contains(Point(plate.x_mm - 3.3, plate.y_mm + 3.3))


def test_every_profession_resolves_to_its_own_icon(database):
    """Eight professions, eight files -- and the roster must not share one."""
    from pcbart.paths import badge_asset_dir

    library = badge_asset_dir()
    seen: dict[str, str] = {}
    for person in database.operators:
        if not person.class_icon_black:
            continue
        assert (library / person.class_icon_black).is_file(), person.name
        seen.setdefault(person.profession, person.class_icon_black)
    assert len(seen) == 8, sorted(seen)
    assert len(set(seen.values())) == 8, "不同职业共用了同一个图标文件"


def test_an_operator_without_a_profession_gets_no_icon(database):
    """博士 / 文月 / 魏彦吾 have a blank 职业 column.

    Inventing a class for them would put a wrong icon on the card, so the slot
    is simply left empty.
    """
    from pcbart.artboard import ArtBoard

    person = next(
        (item for item in database.operators if not item.class_icon_black), None
    )
    if person is None:
        pytest.skip("这个索引里每位干员都有职业图标")
    board = ArtBoard()
    board.operator_name = person.name
    board.operator_page = person.page
    built = badge_content.badge_art(board=board)
    assert built.profession is None
    assert built.qr is not None


# ----------------------------------------------------------- faction logo

def test_the_faction_logo_lands_in_its_block_on_the_top_copper(art):
    region = badge_content.art_region("faction_logo")
    assert art.faction is not None
    assert (art.faction.x_mm, art.faction.y_mm) == pytest.approx(region.centre)
    assert art.faction.rotation_deg == region.rotation_deg == 0.0
    assert art.faction.layer == badge_content.FACTION_LAYER == "顶层铜"


def test_the_block_measures_the_logos_ink_not_its_canvas(art):
    """The library files are square canvases with the mark floating inside.

    Sized by the canvas, "width = the block's width" would describe the empty
    padding rather than the logo: measured over the 48 files, 73-84% of each
    510x510 canvas is transparent, so most marks would have shrunk to a
    fraction of their slot. The placement is the ink.
    """
    from PIL import Image

    from pcbart import badge_content as content

    region = content.art_region("faction_logo")
    with Image.open(art.faction.path) as image:
        assert image.mode == "RGBA"
        box = image.getbbox()
    assert box is not None
    # The cached file is the cropped ink, so its own bounds touch every edge.
    assert box == (0, 0, image.width, image.height)
    # And the mark reaches the block's full width unless height limited it.
    if art.faction.width_mm == pytest.approx(region.width_mm):
        ratio = image.width / image.height
        assert art.faction.height_mm == pytest.approx(
            region.width_mm / ratio, abs=1e-6
        )


def test_every_faction_logo_fits_its_block_without_overlap():
    """The user's rule, checked over the whole library.

    "只要墨迹不和别的丝印冲突，中心和罗德岛标识重合即可" -- so every one of the
    48 marks has to sit inside the block (never spilling over the card) and
    must not run into the **surrounding** silkscreen.

    ⚠️ **The 罗德岛 crest is not "surrounding silkscreen".** It sits on the top
    silkscreen directly over this copper block -- that is the design: the crest
    is fixed card content, the faction logo is a separate element underneath it
    on another layer. So the crest is excluded here, and what is left must be
    empty.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from PIL import Image

    from pcbart import badge_art, badge_content as content
    from pcbart.paths import badge_asset_dir

    directory = badge_asset_dir() / "18-阵营logo"
    if not directory.is_dir():
        pytest.skip("阵营 logo 素材库未安装")

    region = content.art_region("faction_logo")
    block = Polygon(
        [
            (region.centre_x_mm - region.width_mm / 2, region.centre_y_mm - region.height_mm / 2),
            (region.centre_x_mm + region.width_mm / 2, region.centre_y_mm - region.height_mm / 2),
            (region.centre_x_mm + region.width_mm / 2, region.centre_y_mm + region.height_mm / 2),
            (region.centre_x_mm - region.width_mm / 2, region.centre_y_mm + region.height_mm / 2),
        ]
    )

    # Everything on the top silkscreen *except* the 罗德岛 crest, which belongs
    # over this block by design.
    silk = []
    for item in badge_art.visible_decorations():
        if badge_art.layer_key_for_id(int(item.get("layer", 0))) != "top_silk":
            continue
        if badge_art.is_rhodes_emblem(item):
            continue
        for ring in item.get("rings") or ():
            if len(ring) < 3:
                continue
            piece = Polygon(ring)
            if not piece.is_valid:
                piece = piece.buffer(0)
            if not piece.is_empty:
                silk.append(piece)
    silkscreen = unary_union(silk)

    # Apart from the crest the block is empty, so nothing else can collide.
    assert block.intersection(silkscreen).area == pytest.approx(0.0, abs=1e-6)

    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as image:
            alpha = image.convert("RGBA").getchannel("A")
        box = alpha.getbbox()
        if box is None:
            pytest.fail(f"{path.name} 是空图")
        ink_w = box[2] - box[0]
        ink_h = box[3] - box[1]
        aspect = ink_h / ink_w

        width_mm = region.width_mm
        height_mm = width_mm * aspect
        if height_mm > region.height_mm:
            height_mm = region.height_mm
            width_mm = height_mm / aspect

        mark = Polygon(
            [
                (region.centre_x_mm - width_mm / 2, region.centre_y_mm - height_mm / 2),
                (region.centre_x_mm + width_mm / 2, region.centre_y_mm - height_mm / 2),
                (region.centre_x_mm + width_mm / 2, region.centre_y_mm + height_mm / 2),
                (region.centre_x_mm - width_mm / 2, region.centre_y_mm + height_mm / 2),
            ]
        )
        assert mark.difference(block).area == pytest.approx(0.0, abs=1e-6), (
            f"{path.name} 超出了定位区"
        )
        assert mark.intersection(silkscreen).area == pytest.approx(0.0, abs=1e-6), (
            f"{path.name} 撞到了丝印"
        )
