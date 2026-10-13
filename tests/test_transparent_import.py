"""Transparent PNGs are composited onto white at the import boundary.

A game-asset illustration keeps *colour* underneath its transparent pixels --
on the library picture measured here, the transparent 73.6% averages RGB
165-179 with recognisable fragments of the weapon still in it. Anything that
reads pixels without reading alpha (the colour preview, a raw import, the
two-colour detection, the board-art grayscale) turned that residue into a
visible gold background.

The fix composites once, at the boundary that splits colour from alpha, with
the straight-alpha formula ``out = src*a + bg*(1-a)``. Compositing twice would
wash the translucent rim out -- a 50% edge pixel ends at 25% of its colour --
so these tests pin both ends: the boundary does composite, and the pipeline
does not composite a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import build_window, show_preprocess  # noqa: E402
from pcbart.boardart import board_art  # noqa: E402
from pcbart.image_prep import flatten_alpha, imwrite_unicode  # noqa: E402
from pcbart.lineart import line_art  # noqa: E402

#: What the game assets keep under their transparent pixels (measured mean of
#: the library illustration's transparent area, per channel: 165-179).
RESIDUE = 77

#: The opaque colour in the test pictures, as stored in the array. The file
#: round-trips these three numbers unchanged, so no channel reordering enters
#: the picture: what the array says is what the compositing sees.
OPAQUE = (50, 100, 200)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app, tmp_path):
    win = build_window()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    yield win
    from conftest import release_window

    release_window(win)


def _residue_rgba() -> np.ndarray:
    """A transparent picture whose hidden pixels carry the residue colour.

    Same construction the assets use: every pixel has colour, and alpha is the
    only thing keeping the background out of sight. Includes one 50% region so
    the compositing arithmetic has something to get wrong.
    """
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., :3] = RESIDUE
    rgba[8:56, 8:56, :3] = OPAQUE
    rgba[8:56, 8:56, 3] = 255
    rgba[20:30, 20:30, 3] = 128
    return rgba


def _residue_png(path: Path) -> str:
    imwrite_unicode(str(path), _residue_rgba())
    return str(path)


# ---------------------------------------------------------- the formula


def test_flatten_alpha_is_the_straight_alpha_formula():
    rgba = _residue_rgba()
    flat = flatten_alpha(rgba[:, :, :3], rgba[:, :, 3])

    assert (flat[0:8, :, :] == 255).all(), "全透明区应落在白底上"
    assert (flat[40:50, 40:50] == OPAQUE).all(), "不透明区不应被改动"
    expected = np.round(np.array(OPAQUE) * (128 / 255) + 255 * (1 - 128 / 255))
    assert (np.abs(flat[25, 25].astype(int) - expected) <= 1).all(), (
        "半透明区应是一半原色、一半白底"
    )


def test_flatten_alpha_accepts_a_black_background():
    rgba = _residue_rgba()
    flat = flatten_alpha(rgba[:, :, :3], rgba[:, :, 3], background=(0, 0, 0))
    assert (flat[0:8, :, :] == 0).all()


def test_no_alpha_passes_through():
    image = np.full((4, 4, 3), 9, np.uint8)
    assert flatten_alpha(image, None) is image


def test_compositing_twice_is_not_the_same_as_once():
    """Why the operation may run at exactly one boundary."""
    rgba = _residue_rgba()
    alpha = rgba[:, :, 3]
    once = flatten_alpha(rgba[:, :, :3], alpha)
    twice = flatten_alpha(once, alpha)
    assert not np.array_equal(once, twice), "重复合成必须是有损的，否则本组测试失去意义"
    assert (twice[25, 25] > once[25, 25]).all(), "重复合成会把半透明边洗白"


def test_the_pipeline_composites_a_raw_picture_and_not_a_flat_one():
    """Both entry styles must answer the same -- the composite happens once.

    ``line_art`` composites a picture that still carries its alpha, and trusts
    an explicit ``alpha`` to mean the import boundary already did it. The
    board-art tone split makes the arithmetic visible: a 50%-alpha *black*
    region sits at grey 0.50 after one composite (below the 0.62 split, so it
    is dark) and at 0.75 after two (above it, so it flips to light). A
    pipeline that re-composites the already-flat plane therefore cannot give
    the same answer for the two entry styles.
    """
    half_black = np.zeros((64, 64, 4), np.uint8)
    half_black[..., :3] = RESIDUE
    half_black[8:56, 8:56, :3] = 0  # black...
    half_black[8:56, 8:56, 3] = 128  # ...at half transparency
    flat = flatten_alpha(half_black[:, :, :3], half_black[:, :, 3])

    from_rgba = board_art(half_black)
    from_flat = board_art(flat, alpha=half_black[:, :, 3])

    assert from_rgba.black[30:34, 30:34].all(), (
        "50% 黑合成一次后灰阶 0.50，应判成暗部"
    )
    assert (from_rgba.black == from_flat.black).all(), "合成必须恰好发生一次"
    assert (from_rgba.white == from_flat.white).all()

    # The same contract on the line-art side, where the ink is the observable.
    rgba = _residue_rgba()
    flat = flatten_alpha(rgba[:, :, :3], rgba[:, :, 3])
    assert (line_art(rgba) == line_art(flat, alpha=rgba[:, :, 3])).all()


# ------------------------------------------------- the import boundary


def test_the_panel_composites_transparency_onto_white(window, tmp_path):
    panel = show_preprocess(window)
    assert panel.load_image(_residue_png(tmp_path / "残留.png")) is True

    bgr = panel._source_bgr
    assert (bgr[0:8, :, :] == 255).all(), "透明区应被合成成纯白"
    assert (bgr[40:50, 40:50] == OPAQUE).all(), "不透明区不应被改动"
    assert "已合成白底" in panel.source_note.text(), (
        "合成过就要说出来，否则用户不知道白底从哪来"
    )


def test_an_opaque_alpha_channel_is_not_treated_as_transparency(window, tmp_path):
    """A four-channel file whose alpha is all 255 is opaque, not transparent."""

    path = tmp_path / "不透明.png"
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., :3] = 60
    rgba[..., 3] = 255
    imwrite_unicode(str(path), rgba)

    panel = show_preprocess(window)
    assert panel.load_image(str(path)) is True
    assert (panel._source_bgr == 60).all()
    assert "已合成白底" not in panel.source_note.text(), "全 255 的 alpha 不该被提示"


def test_a_gray_alpha_png_is_split_and_composited(window, tmp_path):
    """GrayA (two channels) is the other form ``IMREAD_UNCHANGED`` hands back.

    Read as a plain "three-ish channel" picture it would have been stored as a
    two-channel colour plane -- grey values interleaved with alpha -- and every
    downstream consumer would have seen nonsense.
    """
    path = tmp_path / "grayalpha.png"
    # OpenCV cannot *write* a two-channel PNG (its encoder takes 1/3/4), so the
    # fixture file comes from PIL's "LA" mode. Reading it back with
    # IMREAD_UNCHANGED is what hands the panel a (H, W, 2) array.
    from PIL import Image

    gray = np.full((32, 32, 2), 90, np.uint8)
    gray[..., 1] = 0  # transparent background, grey value still in the file
    gray[8:24, 8:24, 1] = 255  # opaque window...
    gray[8:24, 8:24, 0] = 30  # ...painted dark
    Image.fromarray(gray, mode="LA").save(str(path))

    panel = show_preprocess(window)
    assert panel.load_image(str(path)) is True

    bgr = panel._source_bgr
    assert bgr.shape == (32, 32, 3), "灰度 + alpha 应先展开成三通道再合成"
    assert (bgr[0:4, 0:4, :] == 255).all(), "透明区应合成成纯白"
    assert (bgr[16, 16] == 30).all(), "不透明区应保留灰色"


def test_the_colour_preview_shows_no_residue(window, tmp_path):
    """The upload box is where the residue used to be visible.

    彩色丝印 is an identity pass, so its preview is exactly the colour plane
    the panel holds -- the shortest path from the file to the screen, and the
    one that showed the gold background.
    """
    panel = show_preprocess(window)
    assert panel.load_image(_residue_png(tmp_path / "残留.png")) is True
    color_index = [
        name for name, _hint in gui_mockup.STYLE_OPTIONS
    ].index(gui_mockup.COLOR_STYLE)
    panel.style_combo.setCurrentIndex(color_index)
    window.window().repaint()

    pixmap = panel.drop_zone._pixmap
    assert pixmap is not None
    corner = pixmap.toImage().pixelColor(2, 2)
    assert (corner.red(), corner.green(), corner.blue()) == (255, 255, 255), (
        f"彩色丝印预览的透明区应是白底，实际 {corner.name()}"
    )


def test_a_library_illustration_loses_its_gold_background(window):
    """The reported picture: 73.6% transparent, residue mean RGB 165-179."""
    from pcbart.paths import asset_root

    path = asset_root() / "明日方舟" / "06-立绘-精英0" / "立绘 惊蛰 1.png"
    if not path.exists():
        pytest.skip("素材库不在本机，跳过真实素材回归")

    panel = show_preprocess(window)
    assert panel.load_image(str(path)) is True

    raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    transparent = raw[:, :, 3] < 8
    before = float(raw[:, :, :3][transparent].mean())
    after = float(panel._source_bgr[transparent].mean())

    assert before < 230, f"前提：素材透明区本来就带颜色（实测均值 {before:.0f}）"
    assert after == pytest.approx(255.0, abs=0.5), (
        f"透明区应合成成纯白（修前 {before:.0f}，修后 {after:.0f}）"
    )
