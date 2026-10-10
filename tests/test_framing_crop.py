"""Framing a picture into a card region must **crop** it, not shrink it.

The reported defect: 「放置短区是强制把原图缩小，而非裁切」. The region box was
applied -- so the placement looked framed -- but the *window* was skipped
whenever the pixel size came back ``(0, 0)``, and the window is the only thing
that says "this part of the picture". With no window the export places the whole
picture inside the opening and scales it down.

So these tests assert the two halves separately:

* the window exists, has the region's aspect, and covers **less** than the whole
  picture when the shapes differ (that is what makes it a crop);
* the placement's box is the region exactly.

A test that only checked the box would have passed on the broken code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import ArtBoard, ImagePlacement  # noqa: E402
from pcbart.badge_card import BADGE_REGIONS  # noqa: E402
from pcbart.image_size import UnreadablePicture, picture_size  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def canvas(app):
    from tools.design_canvas import DesignCanvas

    widget = DesignCanvas()
    widget.resize(800, 600)
    widget.board = ArtBoard()
    yield widget
    widget.deleteLater()


def _picture(tmp_path: Path, width: int, height: int, name: str = "art.png") -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (width, height), (120, 130, 140)).save(path)
    return path


def _placement(path: Path, width_mm=76.2, height_mm=76.2) -> ImagePlacement:
    return ImagePlacement(
        path=str(path),
        width_mm=width_mm,
        height_mm=height_mm,
        layer="顶层丝印层",
    )


# ------------------------------------------------------------- the size read


def test_picture_size_reads_the_file(tmp_path):
    assert picture_size(str(_picture(tmp_path, 640, 480))) == (640, 480)


def test_picture_size_raises_on_a_missing_file(tmp_path):
    with pytest.raises(UnreadablePicture):
        picture_size(str(tmp_path / "nope.png"))


def test_picture_size_raises_on_a_file_that_is_not_a_picture(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture at all")
    with pytest.raises(UnreadablePicture):
        picture_size(str(broken))


def test_picture_size_raises_on_an_empty_path():
    with pytest.raises(UnreadablePicture):
        picture_size("")


# ---------------------------------------------------------------- the crop


@pytest.mark.parametrize("size", [(1024, 1024), (2000, 800), (800, 2000)])
def test_placing_into_a_region_sets_a_cropping_window(tmp_path, canvas, size):
    """The window is set, its shape is the region's, and it is a real crop.

    ⚠️ The whole point is that the window covers **less** than the picture when
    the shapes differ. Without that, "in the region" is a shrink.
    """
    region = next(item for item in BADGE_REGIONS if item.key == "small")
    path = _picture(tmp_path, *size)
    placement = _placement(path)
    canvas.board = ArtBoard()
    canvas.board.add_placement(placement)

    canvas._fit_window_to_region(placement, region)

    window = placement.source_window
    assert window is not None, "放区之后必须有取景窗口，否则导出的是整张图"
    x0, y0, x1, y1 = window
    width_px, height_px = size
    # The window has the region's shape once the picture's own pixel aspect is
    # taken out -- the same conversion the canvas and the writers share.
    assert abs((abs(x1 - x0) * width_px) / (abs(y1 - y0) * height_px) - region.aspect) < 1e-6

    covered = abs(x1 - x0) * abs(y1 - y0)
    if abs(size[0] / size[1] - region.aspect) > 0.02:
        assert covered < 1.0, "形状不同时必须裁掉一部分，而不是整张缩进去"

    # And the box took the region exactly.
    assert placement.x_mm == pytest.approx(region.centre_x_mm)
    assert placement.y_mm == pytest.approx(region.centre_y_mm)
    assert placement.width_mm == pytest.approx(region.width_mm)
    assert placement.height_mm == pytest.approx(region.height_mm)
    assert placement.region == region.key


def test_a_square_into_the_short_region_crops_the_sides(tmp_path, canvas):
    """A 1:1 picture into a 0.69:1 opening keeps the middle and loses the sides.

    The narrow opening is the shape's own constraint: the largest window of that
    shape inside the square is the full height and 69% of the width.
    """
    region = next(item for item in BADGE_REGIONS if item.key == "small")
    path = _picture(tmp_path, 1024, 1024)
    placement = _placement(path)
    canvas.board = ArtBoard()
    canvas.board.add_placement(placement)

    canvas._fit_window_to_region(placement, region)

    x0, y0, x1, y1 = placement.source_window
    assert y0 == pytest.approx(0.0) and y1 == pytest.approx(1.0)
    assert x1 - x0 == pytest.approx(region.aspect, rel=1e-6)
    # Centred, so the same amount comes off each side.
    assert x0 == pytest.approx(1.0 - x1)


def test_a_wide_picture_into_the_short_region_crops_the_sides(tmp_path, canvas):
    """A 2.5:1 picture into a 0.69:1 opening keeps full height and loses the sides.

    The opening is much narrower than the picture, so the largest window of the
    opening's shape inside the picture is a vertical slice: all of the height,
    part of the width.
    """
    region = next(item for item in BADGE_REGIONS if item.key == "small")
    path = _picture(tmp_path, 2000, 800)
    placement = _placement(path)
    canvas.board = ArtBoard()
    canvas.board.add_placement(placement)

    canvas._fit_window_to_region(placement, region)

    x0, y0, x1, y1 = placement.source_window
    assert y0 == pytest.approx(0.0) and y1 == pytest.approx(1.0)
    assert x1 - x0 < 1.0
    assert x0 == pytest.approx(1.0 - x1)


def test_a_tall_picture_into_the_short_region_crops_top_and_bottom(tmp_path, canvas):
    """A 0.4:1 picture is narrower than the opening, so the *ends* are what go."""
    region = next(item for item in BADGE_REGIONS if item.key == "small")
    path = _picture(tmp_path, 800, 2000)
    placement = _placement(path)
    canvas.board = ArtBoard()
    canvas.board.add_placement(placement)

    canvas._fit_window_to_region(placement, region)

    x0, y0, x1, y1 = placement.source_window
    assert x0 == pytest.approx(0.0) and x1 == pytest.approx(1.0)
    assert y1 - y0 < 1.0
    assert y0 == pytest.approx(1.0 - y1)


def test_an_unreadable_picture_is_refused_rather_than_shrunk(tmp_path, canvas):
    """⚠️ The bug's own mechanism: a size of ``(0, 0)`` used to skip the crop.

    Skipping meant ``source_window`` stayed ``None``, so the export placed the
    whole picture inside the opening at the opening's size -- a shrink. Placing
    it is now refused, because printing something the user did not ask for is
    worse than saying it cannot be done.
    """
    region = next(item for item in BADGE_REGIONS if item.key == "small")
    missing = tmp_path / "gone.png"
    placement = _placement(missing)
    canvas.board = ArtBoard()
    canvas.board.add_placement(placement)

    with pytest.raises(UnreadablePicture):
        canvas._fit_window_to_region(placement, region)


def test_the_wide_and_tall_regions_use_their_own_shape(tmp_path, canvas):
    """精0 and 精2 are different openings, and the window follows the one asked for."""
    path = _picture(tmp_path, 1200, 1200)
    windows: dict[str, tuple[float, float, float, float]] = {}
    for region in BADGE_REGIONS:
        placement = _placement(path)
        canvas.board = ArtBoard()
        canvas.board.add_placement(placement)
        canvas._fit_window_to_region(placement, region)
        windows[region.key] = placement.source_window

    assert set(windows) == {"small", "large"}
    assert windows["small"] != windows["large"]

    # The tall opening is *narrower* in shape (42.526/99.626 = 0.427 against
    # 38.861/56.324 = 0.690), so it keeps a thinner slice of the picture.
    small = windows["small"][2] - windows["small"][0]
    large = windows["large"][2] - windows["large"][0]
    assert large < small

    # Each is still the shape its own region asks for, in real units.
    for region in BADGE_REGIONS:
        x0, y0, x1, y1 = windows[region.key]
        actual = (abs(x1 - x0) * 1200) / (abs(y1 - y0) * 1200)
        assert actual == pytest.approx(region.aspect, rel=1e-6)


# ------------------------------------------- what the published file holds


def test_the_canvas_paints_black_on_white_line_art_as_ink(tmp_path, app):
    """⚠️ The published line-art file must stay **black-on-white**.

    This is the defect the transparent form caused: ``design_canvas._tinted``
    reads the file as *grey* and derives the ink from it -- dark is ink, so the
    alpha is ``255 - grey``. Ink-on-transparent stores the paper at RGB 0, which
    reads as grey 0 = maximum ink, and the whole illustration came out as one
    solid block of the layer colour. The preview was unaffected because it shows
    the pixels directly, which is why only the canvas broke.

    So the conversion to transparency belongs at export, not here, and this
    test pins the file format the canvas depends on.
    """
    import numpy as np
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from pcbart.design_layers import layer
    from pcbart.image_prep import ink_only
    from tools.design_canvas import DesignCanvas

    # A drawing: black ink on a white page.
    page = np.full((64, 64), 255, np.uint8)
    page[16:48, 28:36] = 0
    path = tmp_path / "lineart.png"
    from pcbart.image_prep import imwrite_unicode

    imwrite_unicode(path, page)

    canvas = DesignCanvas()
    canvas.resize(400, 400)
    canvas.board = ArtBoard()
    placement = ImagePlacement(
        path=str(path),
        processed_path=str(path),
        width_mm=20.0,
        height_mm=20.0,
        layer=layer("top_silk").name,
        kind="lineart",
    )
    canvas.board.add_placement(placement)

    pixmap = QPixmap(str(path))
    assert not pixmap.isNull()
    tinted = canvas._tinted(pixmap, QApplication.instance() and __import__(
        "PySide6.QtGui", fromlist=["QColor"]
    ).QColor(255, 204, 0))
    image = tinted.toImage()
    ink = 0
    paper = 0
    for y in range(0, image.height(), 4):
        for x in range(0, image.width(), 4):
            if image.pixelColor(x, y).alpha() > 128:
                ink += 1
            else:
                paper += 1
    canvas.deleteLater()

    # The strokes are 8 of 64 columns, so the ink is a small share. The broken
    # version -- paper at RGB 0 -- made *everything* ink.
    share = ink / max(1, ink + paper)
    assert 0.05 < share < 0.35, (
        f"墨迹占比 {share:.2f}，应当接近线条本身的面积；"
        "整张全为墨迹说明文件被当成了透明底"
    )

    # And the same picture in the export form really is transparent, so the two
    # files differ in exactly the way the two consumers need.
    exported = ink_only(np.asarray(page))
    assert exported.shape == (64, 64, 4)
    assert exported[0, 0, 3] == 0, "纸必须透明"
    assert exported[32, 32, 3] == 255, "墨必须实心"


def test_publishing_line_art_keeps_a_white_page(tmp_path):
    """The file 线稿 writes is black-on-white, which is what the canvas reads.

    Checked on the published bytes rather than on the array passed in, because
    the writer is where the wrong conversion was applied.
    """
    import numpy as np
    from PIL import Image

    from pcbart.image_prep import imwrite_unicode

    mask = np.full((32, 32), 255, np.uint8)
    mask[8:24, 12:20] = 0
    target = tmp_path / "published.png"
    imwrite_unicode(target, mask)

    with Image.open(target) as stored:
        assert stored.mode in ("L", "LA", "P", "RGB", "RGBA")
        assert "A" not in stored.getbands(), (
            "发布的线稿不能带透明通道：画布按灰度推墨迹，透明纸会被当成满版墨"
        )
        grey = stored.convert("L")
        assert grey.getpixel((0, 0)) == 255, "纸应当是白的"
        assert grey.getpixel((16, 16)) == 0, "线应当是黑的"


# --------------------------------------------- the real import path's file


def test_importing_line_art_publishes_a_white_page(tmp_path):
    """⚠️ Drives the **panel's own import**, which is where the defect lived.

    A test that only calls the image writer cannot catch this: the wrong
    conversion was applied by ``_publish_line_art``, so the file on disk came
    out with transparent paper, the canvas read that as full-coverage ink, and
    the illustration appeared as one solid block of the layer colour.

    So this imports through the real panel and then reads the published file
    back. The white page is the contract ``design_canvas._tinted`` depends on.
    """
    pytest.importorskip("PySide6")
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from gui_mockup import STYLE_OPTIONS, build_window, show_preprocess
    from pcbart import badge_card
    from pcbart.image_prep import imwrite_unicode

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1200, 800)
    window.show()
    app = QApplication.instance() or QApplication([])
    app.processEvents()

    try:
        board = window.design_canvas.board
        board.outline = badge_card.badge_spec()
        board.outline_ready = True
        board.texts = badge_card.badge_texts()
        board.cutouts = badge_card.badge_cutouts()

        # A drawing with a clear page and a clear stroke.
        source = tmp_path / "drawing.png"
        page = np.full((600, 400, 3), 250, np.uint8)
        page[150:450, 170:230] = 10
        imwrite_unicode(source, page)

        panel = show_preprocess(window)
        app.processEvents()
        assert panel.load_image(str(source)) is True
        index = [key for key, _hint in STYLE_OPTIONS].index("线稿")
        panel.style_combo.setCurrentIndex(index)
        app.processEvents()
        assert panel.import_to_canvas() is True
        app.processEvents()

        published = [
            item.processed_path
            for item in board.placements
            if item.processed_path
        ]
        assert published, "线稿导入后应当写出处理结果"

        for path in published:
            with Image.open(path) as stored:
                assert "A" not in stored.getbands(), (
                    f"发布的线稿不能带透明通道（{path}）："
                    "画布按灰度推墨迹，透明纸会被读成满版墨，整张变成纯色块"
                )
                grey = stored.convert("L")
                assert grey.getpixel((5, 5)) > 200, "纸应当是白的"
    finally:
        from conftest import release_window

        release_window(window)
