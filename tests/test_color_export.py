"""Colour pictures in the export: the ``OBJ`` path.

A card ordered as 彩色丝印 used to come back in **one colour**. Every placement
was traced into ``FILL`` polygons, which is right for line art -- that really is
two-tone geometry -- but it discards the pixels.

The file-format specification (``easyeda/easyeda-format-skill``) is specific
about which record carries a picture:

* ``IMAGE`` -- a *complex polygon*: ``path`` is a vertex list, so it has shape
  but no pixels and can only print in one colour;
* ``OBJ`` -- a *binary embedded object*: ``path`` is a
  ``data:image/png;base64,...`` URL, i.e. the picture itself.

So a colour placement leaves as an ``OBJ``, and the EDA's
``getGerberFile(..., colorSilkscreen=True)`` turns it into the
``Fabrication_Colorful*`` files.

These tests read the written file back and decode the embedded picture. A test
that only counted records would pass on an OBJ whose payload was the wrong
image, and the whole point is which pixels reached the fab.
"""

from __future__ import annotations

import base64
import io
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import ArtBoard, ImagePlacement, drawn_size_mm  # noqa: E402
from pcbart.badge_card import badge_spec  # noqa: E402
from pcbart.color_image import color_objects, is_color_artwork  # noqa: E402
from pcbart.epru import EproArchive  # noqa: E402
from pcbart.export import export_design  # noqa: E402

MIL = 0.0254


def colour_png(tmp_path: Path, name: str = "colour.png") -> Path:
    """A picture with real colours: red top-left, green bottom-right, white elsewhere.

    Grey or two-tone would pass a "colour survived" check by accident, which is
    the exact defect under test.
    """
    from PIL import Image

    image = Image.new("RGB", (400, 400), (255, 255, 255))
    pixels = image.load()
    for x in range(400):
        for y in range(400):
            if x < 200 and y < 200:
                pixels[x, y] = (220, 30, 30)
            elif x >= 200 and y >= 200:
                pixels[x, y] = (30, 160, 60)
    path = tmp_path / name
    image.save(path)
    return path


def embedded_colours(blob_body) -> set[tuple[int, int, int]]:
    """The distinct colours in a ``BLOB`` record's picture."""
    from PIL import Image

    raw = base64.b64decode(blob_body["content"].split("base64,", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return {rgb for _count, rgb in image.getcolors(maxcolors=1 << 20)}


def blob_records(path: Path) -> list:
    return list(EproArchive.read(path).document.iter_type("BLOB"))


def obj_records(path: Path) -> list:
    return list(EproArchive.read(path).document.iter_type("OBJ"))


def card_with(placement) -> ArtBoard:
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.placements.append(placement)
    return board


# --------------------------------------------------------- what leaves as OBJ


def test_a_colour_picture_leaves_as_an_embedded_object(tmp_path):
    """The defect: 彩色丝印 exported as single-colour silkscreen."""
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        x_mm=1.854,
        y_mm=21.825,
        width_mm=38.861,
        height_mm=56.324,
        layer="顶层丝印层",
    )
    report = export_design(card_with(placement), tmp_path / "card.epro2")

    written = Path(str(report["output"]))
    objects = obj_records(written)
    blobs = blob_records(written)
    assert objects, "彩色图没有写成 OBJ —— 彩印会退化成纯色丝印"
    assert blobs, "OBJ 指向的 BLOB 不存在 —— EDA 里只会显示一个空方块"

    body = objects[0].body
    # ⚠️ The editor reads only ``blob:<id>``. An inline Data URL here imported
    # as an empty rectangle, which is exactly the failure this pins.
    assert body["path"].startswith("blob:"), f"path 必须是 blob 引用，实测 {body['path'][:40]}"
    assert body["path"] == f"blob:{blobs[0].header['id']}", "OBJ 的引用与 BLOB 的 id 不一致"
    assert blobs[0].body["content"].startswith("data:image/png;base64,")
    assert body["layerId"] == 3, "彩色图应当落在顶层丝印"
    assert report["color_pictures"] == 1


def test_the_embedded_colours_are_the_users_colours(tmp_path):
    """Read the pixels back; counting records would not catch a wrong payload."""
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=38.861,
        height_mm=56.324,
        layer="顶层丝印层",
    )
    report = export_design(card_with(placement), tmp_path / "card.epro2")
    blob = blob_records(Path(str(report["output"])))[0]
    colours = embedded_colours(blob.body)
    assert (220, 30, 30) in colours, "红色丢了"
    assert (30, 160, 60) in colours, "绿色丢了"


def test_the_framing_window_is_applied_to_the_colour_picture(tmp_path):
    """What the user framed is what gets printed."""
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=38.861,
        height_mm=56.324,
        layer="顶层丝印层",
        source_window=(0.0, 0.0, 0.5, 0.5),
    )
    blob = color_objects(card_with(placement))[0]["blob"]["body"]
    assert embedded_colours(blob) == {(220, 30, 30)}, "取景没生效"


# ------------------------------------------------------- what keeps tracing


def test_a_traced_picture_is_not_also_embedded(tmp_path):
    """线稿 and 板画 stay traced, on purpose.

    A placement with ``processed_path`` went through the line-art or board-art
    pipeline by the user's choice, so it *is* two-tone geometry. Embedding it as
    a colour picture as well would print it twice: once in colour, once as a
    flat silhouette underneath.
    """
    source = colour_png(tmp_path)
    placement = ImagePlacement(
        path=str(source),
        processed_path=str(source),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
    )
    report = export_design(card_with(placement), tmp_path / "card.epro2")
    assert not obj_records(Path(str(report["output"])))
    assert report["color_pictures"] == 0


def test_copper_keeps_the_traced_path(tmp_path):
    """⚠️ An embedded object cannot stay on copper, so copper is traced.

    The API's own enum ``TPCB_LayersOfObject`` allows only ``3`` (top silk),
    ``4`` (bottom silk) and ``13`` (document). Requesting copper is *accepted
    and then silently moved*: ``create(1, ...)`` returned ``layer: 1`` and
    reading the object back gave ``layer: 3`` -- the picture lands on the
    silkscreen, which is exactly the layer the 阵营 logo must not use.

    The logo is white line art that belongs on **顶层铜**, so it keeps the
    traced path: that is the only way to put it on the layer the user asked for.
    """
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层",
    )
    assert not is_color_artwork(placement)


def test_an_unreadable_picture_falls_back_to_tracing(tmp_path):
    """A missing file must not turn into a missing picture.

    Documents written on another machine name files that are not here. Routing
    those away from the tracer would drop the artwork entirely, so the file is
    checked before it is treated as colour.
    """
    placement = ImagePlacement(
        path=str(tmp_path / "not-here.png"),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
    )
    assert not is_color_artwork(placement)
    assert color_objects(card_with(placement)) == []


# -------------------------------------------------------------- the geometry


@pytest.mark.parametrize("rotation", [0, 90, 180, 270, 37])
@pytest.mark.parametrize("flip_h", [False, True])
@pytest.mark.parametrize("flip_v", [False, True])
def test_the_colour_object_sits_where_the_canvas_draws_it(
    tmp_path, rotation, flip_h, flip_v
):
    """The four corners agree with the model's own transform.

    ``ImagePlacement`` stores its **centre** and rotates about it; ``OBJ`` is
    placed by its **top-left** and rotates about that. A conversion that gets
    the corner wrong is invisible until the picture is turned, so every
    rotation and flip is checked -- against ``board_ring``, the model's own
    implementation, rather than a second copy of the arithmetic.
    """
    source = colour_png(tmp_path)
    placement = ImagePlacement(
        path=str(source),
        x_mm=1.854,
        y_mm=21.825,
        width_mm=38.861,
        height_mm=56.324,
        layer="顶层丝印层",
        rotation_deg=rotation,
        flip_horizontal=flip_h,
        flip_vertical=flip_v,
    )
    body = color_objects(card_with(placement))[0]["object"]["body"]

    left, top, right, bottom = placement.source_rect(400, 400)
    width_mm, height_mm = drawn_size_mm(
        placement.width_mm, placement.height_mm, (right - left) / (bottom - top)
    )

    angle = math.radians(body["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    start_x, start_y = body["startX"] * MIL, body["startY"] * MIL
    width, height = body["width"] * MIL, body["height"] * MIL
    corners = [(0.0, 0.0), (width, 0.0), (width, -height), (0.0, -height)]
    if body["mirror"]:
        corners = [(width - x, y) for x, y in corners]
    got = sorted(
        (
            round(start_x + x * cos_a - y * sin_a, 3),
            round(start_y + x * sin_a + y * cos_a, 3),
        )
        for x, y in corners
    )
    want = sorted(
        (round(x, 3), round(y, 3))
        for x, y in placement.board_ring(
            [
                (-width_mm / 2, height_mm / 2),
                (width_mm / 2, height_mm / 2),
                (width_mm / 2, -height_mm / 2),
                (-width_mm / 2, -height_mm / 2),
            ]
        )
    )
    drift = max(max(abs(a - c), abs(b - d)) for (a, b), (c, d) in zip(got, want))
    assert drift <= 0.002, (
        f"旋转 {rotation}° 翻转 {flip_h}/{flip_v} 时图元偏了 {drift:.4f} mm"
    )


def test_a_picture_outside_the_card_region_is_still_embedded(tmp_path):
    """The region clip is a *board* rule, not a reason to drop the picture.

    ``build_copper`` clips traced ink to the 通行证 opening. A colour picture is
    a whole image placed in that opening, and the EDA needs all of it to render
    the part inside -- so it must not be clipped away by the same rule.
    """
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=120.0,
        height_mm=120.0,
        layer="顶层丝印层",
        region="small",
    )
    body = color_objects(card_with(placement))[0]["object"]["body"]
    assert body["width"] * MIL == pytest.approx(120.0, abs=0.01)


# -------------------------------------------------- which format gets pictures


def test_gerber_always_traces_instead_of_placing_pictures(tmp_path):
    """⚠️ A Gerber file has no picture primitive — so there, everything is traced.

    This is the boundary the full suite caught: routing pictures into the
    shared geometry pass silently emptied eight Gerber tests, because the
    Gerber writer has nowhere to put an ``OBJ``. The choice belongs to the
    *format*, not to the artwork, so ``build_copper`` takes it as an argument
    and only the ``.epro2`` writer asks for pictures.
    """
    from pcbart.export import build_copper

    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        processed_path="",  # still has its pixels: a picture by the rule above
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
    )
    board = card_with(placement)

    traced, report = build_copper(board)
    assert traced, "Gerber 路径必须描边，否则图形会整个丢掉"
    assert report["color_artwork"] == []

    as_picture, report = build_copper(board, pictures=True)
    assert not as_picture, "要图片时不该再描边"
    assert report["color_artwork"] == [placement]


def test_the_generated_card_elements_are_placed_as_pictures(tmp_path):
    """二维码 / 条形码 / 职业图标 / 阵营 logo keep their own artwork.

    All four are transparent-background pictures from the asset library — black
    ink for the codes and the profession icon, white line art for the faction
    logo. Traced, each arrived in the editor as a stack of **填充区域** rather
    than as the picture the library shipped.
    """
    from pcbart import badge_content
    from pcbart.badge_card import badge_texts
    from pcbart.color_image import color_objects

    from pcbart import operators

    if not operators.database():
        pytest.skip("干员素材库未安装")

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.texts = badge_texts()
    board.operator_name = "能天使"
    board.operator_page = "能天使"
    board.operator_number = "PL03"
    art = badge_content.badge_art(board=board, number="PL03")
    board.placements = list(art.placements)
    if not board.placements:
        pytest.skip("这位干员没有可放的卡面素材")

    # ⚠️ 阵营 logo 是**例外**：它必须落在顶层铜，而内嵌对象只能待在丝印层
    # （TPCB_LayersOfObject 只允许 3/4/13，请求铜层会被静默改成丝印层），
    # 所以它只能走描边 —— 那是把它放上铜层的唯一办法。
    placed: list[str] = []
    for placement in board.placements:
        picture = is_color_artwork(placement)
        if placement.kind == "faction":
            assert not picture, "阵营 logo 必须走铜层矢量，不能当图片放丝印"
            assert placement.layer_key == "top_copper"
        else:
            assert picture, f"{placement.kind} 仍被当成几何"
            placed.append(placement.kind)

    objects = color_objects(board)
    assert len(objects) == len(placed)
    assert all(item["object"]["body"]["path"].startswith("blob:") for item in objects)


# ------------------------------------------------ plain order vs colour order


def test_a_plain_order_traces_every_picture(tmp_path):
    """⚠️ The failure mode this guards: pictures vanishing from a normal order.

    A picture lives in the ``.epro2`` as an ``OBJ`` and reaches a board only
    through the EDA's colour-silkscreen export. Measured on a card carrying two
    pictures: the plain Gerber export left the top and bottom silkscreen files
    at 810 and 813 bytes with **zero** draw commands.

    So ``color_print=False`` has to trace. The two orders are different
    products, not two names for one file.
    """
    from pcbart.export import build_copper

    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
    )
    board = card_with(placement)

    traced, report = build_copper(board, pictures=False)
    assert traced, "普通打板必须描边，否则图不会出现在 Gerber 里"
    assert report["color_artwork"] == []

    as_picture, report = build_copper(board, pictures=True)
    assert not as_picture
    assert report["color_artwork"] == [placement]


def test_the_plain_epro2_export_has_no_picture_objects(tmp_path):
    """The written file follows the same choice, not just the geometry pass."""
    placement = ImagePlacement(
        path=str(colour_png(tmp_path)),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
    )
    report = export_design(
        card_with(placement), tmp_path / "plain.epro2", color_print=False
    )
    written = Path(str(report["output"]))
    assert not obj_records(written), "普通打板不该写 OBJ"
    assert not blob_records(written)


def test_the_colour_choice_is_only_asked_when_there_is_a_picture(tmp_path):
    """A board with nothing to decide exports without a question.

    The confirmation is a real decision -- it changes how the artwork is
    written -- so it must not become a dialog the user clicks through on every
    save. A board with no pictures has one correct answer, and gets it silently.
    """
    from pcbart.export import export_needs_color_choice

    plain = ArtBoard(outline=badge_spec(), outline_ready=True)
    assert not export_needs_color_choice(plain)

    with_picture = card_with(
        ImagePlacement(
            path=str(colour_png(tmp_path)),
            width_mm=20.0,
            height_mm=20.0,
            layer="顶层丝印层",
        )
    )
    assert export_needs_color_choice(with_picture)


def test_a_traced_picture_does_not_trigger_the_question(tmp_path):
    """线稿 and 板画 are already geometry, so there is nothing to decide."""
    from pcbart.export import export_needs_color_choice

    source = colour_png(tmp_path)
    board = card_with(
        ImagePlacement(
            path=str(source),
            processed_path=str(source),
            width_mm=20.0,
            height_mm=20.0,
            layer="顶层丝印层",
        )
    )
    assert not export_needs_color_choice(board)


def test_the_faction_logo_stays_on_top_copper(tmp_path):
    """⚠️ The one element that must **not** become a picture.

    The user's requirement is explicit: 阵营 logo goes on **顶层铜** and that is
    not negotiable. An embedded object cannot stay there -- the editor's own
    ``TPCB_LayersOfObject`` allows only layers 3, 4 and 13, and asking for
    copper returns ``layer: 1`` on create while reading the object back gives
    ``layer: 3``. So a picture for this logo would be printed on the silkscreen.

    This pins the layer *and* the path, because either one alone would pass
    while the other was wrong: right layer with an object still gets moved, and
    a traced shape on silk would be the wrong layer.
    """
    from pcbart import badge_content
    from pcbart import operators
    from pcbart.badge_card import badge_texts

    if not operators.database():
        pytest.skip("干员素材库未安装")

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.texts = badge_texts()
    board.operator_name = board.operator_page = "能天使"
    board.operator_number = "PL03"
    art = badge_content.badge_art(board=board, number="PL03")
    board.placements = list(art.placements)
    if art.faction is None:
        pytest.skip("这位干员没有阵营 logo")

    assert art.faction.layer_key == "top_copper", "阵营 logo 必须在顶层铜"
    assert not is_color_artwork(art.faction), "阵营 logo 不能当图片放丝印层"

    # And copper is genuinely outside the layers an object may use, which is
    # *why* it has to be traced.
    from pcbart.color_image import _OBJECT_LAYERS

    assert "top_copper" not in _OBJECT_LAYERS


# ------------------------------------------------- line art stays a picture


def test_line_art_is_embedded_as_a_picture(tmp_path):
    """线稿 leaves the board as a picture, not as traced regions.

    「按理来说图片应该能保留更多细节，应当保留黑白图片」. A traced silhouette
    is simplified geometry: thin strokes drop, curves become chords, and the
    editor receives a "填充区域" the user cannot treat like a picture. The
    pixels are what the import preview showed and approved.

    The placement's ``kind`` is what marks its file as line art; a placement
    whose ``processed_path`` came from somewhere else (板画's tones) still
    traces, because those two halves have to be traced together.
    """
    from pcbart.color_image import is_color_artwork

    source = colour_png(tmp_path)
    line_art = ImagePlacement(
        path=str(source),
        processed_path=str(source),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
        kind="lineart",
    )
    assert is_color_artwork(line_art)

    report = export_design(card_with(line_art), tmp_path / "card.epro2")
    assert report["color_pictures"] == 1
    assert obj_records(Path(str(report["output"])))


def test_board_art_still_traces(tmp_path):
    """板画's two tones are geometry by construction and must keep tracing.

    A tone placement carries ``processed_path`` too, so the rule cannot be
    "anything processed is embedded" -- that would turn a board painting into
    two pictures and break the seam they are built to share.
    """
    from pcbart.color_image import is_color_artwork

    source = colour_png(tmp_path)
    white = ImagePlacement(
        path=str(source),
        processed_path=str(source),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
        tone="white",
    )
    black = ImagePlacement(
        path=str(source),
        processed_path=str(source),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层铜",
        tone="black",
    )
    assert not is_color_artwork(white)
    assert not is_color_artwork(black)


def test_the_embedded_line_art_carries_the_processed_pixels(tmp_path):
    """The embedded bytes come from the framed file, not from the source.

    A window is a fraction of a picture, so it only means something against the
    file it was measured on. 线稿's output has its own resolution; reading the
    source instead would crop a different part than the canvas showed.
    """
    import base64
    import io

    from PIL import Image

    source = tmp_path / "colour.png"
    Image.new("RGB", (400, 400), (200, 40, 40)).save(source)

    processed = tmp_path / "lineart.png"
    # Same size on purpose: the assertion is about *which file* was read, and
    # an unusual size would also make it a resolution test.
    Image.new("L", (400, 400), 255).save(processed)

    placement = ImagePlacement(
        path=str(source),
        processed_path=str(processed),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
        kind="lineart",
    )
    report = export_design(card_with(placement), tmp_path / "card.epro2")
    blobs = blob_records(Path(str(report["output"])))
    assert len(blobs) == 1
    raw = base64.b64decode(blobs[0].body["content"].split("base64,", 1)[1])
    with Image.open(io.BytesIO(raw)) as embedded:
        # The line art was white (paper); the source was red. Reading the
        # source would come back red, so this fails on the wrong file.
        assert embedded.convert("RGB").getpixel((10, 10)) != (200, 40, 40)


def test_the_line_art_window_crops_the_processed_file(tmp_path):
    """The crop is measured on the processed file, and really cuts."""
    import base64
    import io

    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (400, 400), (10, 200, 10)).save(source)
    processed = tmp_path / "lineart.png"
    Image.new("L", (400, 400), 255).save(processed)

    placement = ImagePlacement(
        path=str(source),
        processed_path=str(processed),
        width_mm=20.0,
        height_mm=20.0,
        layer="顶层丝印层",
        kind="lineart",
        source_window=(0.25, 0.0, 0.75, 1.0),
    )
    report = export_design(card_with(placement), tmp_path / "card.epro2")
    blobs = blob_records(Path(str(report["output"])))
    raw = base64.b64decode(blobs[0].body["content"].split("base64,", 1)[1])
    with Image.open(io.BytesIO(raw)) as embedded:
        assert embedded.size == (200, 400), "窗口没有裁切处理后的图"
