"""Two-colour pictures skip the processing pipeline.

A drawing that is already flat black on flat white does not need tracing, and
tracing it is not neutral: the line-art pass thickens every stroke and adds a
silhouette ring the artwork never had, and on unusual letterforms it can lose
parts of the picture outright. So the panel detects such a picture and turns
处理图像 off by itself -- as a *suggestion*, never a lock, because a user who
wants the traced version of a two-colour drawing must still be able to say so.
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
from gui_mockup import build_window, chrome_height, show_preprocess  # noqa: E402
from pcbart.image_prep import (  # noqa: E402
    imread_unicode,
    imwrite_unicode,
    is_two_tone,
    to_pure_two_tone,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _drawing(path: Path) -> str:
    """Flat black shapes on flat white, with a hint of edge softness.

    The softness matters: a real file from a camera, a screenshot tool or a
    JPEG encoder never has hard edges, and a detector that only accepted
    bit-exact two-value images would reject every file a user actually has.
    """
    image = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(image, (40, 40), (260, 120), (0, 0, 0), -1)
    cv2.circle(image, (150, 200), 60, (0, 0, 0), -1)
    soft = cv2.GaussianBlur(image, (3, 3), 0)
    imwrite_unicode(str(path), soft)
    return str(path)


def _photo(path: Path) -> str:
    """A picture with real mid-tones and colour."""
    image = np.zeros((300, 300, 3), np.uint8)
    for y in range(300):
        image[y, :] = (40 + y // 2, 90, 180 - y // 3)
    cv2.circle(image, (150, 150), 80, (200, 60, 40), -1)
    imwrite_unicode(str(path), image)
    return str(path)


def _fresh_panel(app, tmp_path):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    panel = show_preprocess(window)
    return window, panel


# ------------------------------------------------------------------ detection


def test_flat_two_colour_pictures_are_detected(tmp_path):
    assert is_two_tone(cv2.imread(_drawing(tmp_path / "d.png"))) is True


def test_photographs_are_not_detected(tmp_path):
    assert is_two_tone(cv2.imread(_photo(tmp_path / "p.png"))) is False


def test_a_gradient_is_not_two_colour():
    """Otsu splits any histogram in two, so a ramp must be rejected on shape."""
    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :], (60, 1))
    assert is_two_tone(cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)) is False


def test_a_solid_block_is_not_two_colour():
    assert is_two_tone(np.full((60, 60, 3), 128, np.uint8)) is False


def test_a_blank_page_is_not_two_colour():
    assert is_two_tone(np.full((60, 60, 3), 255, np.uint8)) is False


def test_a_grey_area_is_not_flattened_as_two_colour():
    """A shape filled with a real mid-grey is content, not edge ringing.

    Edge ringing sits on the outline and disappears under an erosion; a filled
    grey area does not. Without that distinction this picture would be read as
    two-colour and the grey would be silently snapped to black.
    """
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (100, 100), (0, 0, 0), -1)
    cv2.circle(image, (140, 140), 40, (128, 128, 128), -1)
    assert is_two_tone(image) is False


def test_a_coloured_shape_is_not_two_colour():
    """Desaturating first would hide this one, so colour is checked directly."""
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (180, 180), (40, 40, 200), -1)
    assert is_two_tone(image) is False


# ------------------------------------------------------------------ conversion


def test_conversion_leaves_exactly_two_values(tmp_path):
    out = to_pure_two_tone(cv2.imread(_drawing(tmp_path / "d.png")))
    assert set(np.unique(out)).issubset({0, 255}), "转换后必须只有黑白两色"


def test_conversion_splits_between_the_two_colours_not_at_128():
    """The threshold is the midpoint of what is present, not a fixed 128.

    A picture whose dark tone is #202020 and whose light tone is #e0e0e0 has a
    midpoint of 128 only by coincidence; move either tone and a fixed
    threshold starts shifting edge pixels to the wrong side.
    """
    image = np.full((60, 60, 3), 230, np.uint8)
    image[10:50, 10:50] = 30
    # A pixel at 150 sits above the true midpoint (130) and must go white.
    image[5, 5] = 150
    out = to_pure_two_tone(image)
    assert tuple(out[5, 5]) == (255, 255, 255), "高于中点的像素应当判为白"


def test_conversion_keeps_transparency():
    """A cut-out must stay cut out; the alpha is not the detector's business."""
    image = np.zeros((40, 40, 4), np.uint8)
    image[:, :, :3] = 255
    image[10:30, 10:30, :3] = 0
    image[0, 0, 3] = 0
    out = to_pure_two_tone(image)
    assert out.shape[2] == 4
    assert out[0, 0, 3] == 0


# ---------------------------------------------------------------------- panel


def test_loading_a_two_colour_picture_switches_processing_off(app, tmp_path):
    window, panel = _fresh_panel(app, tmp_path)
    try:
        assert panel.load_image(_drawing(tmp_path / "d.png")) is True
        assert panel.process_check.isChecked() is False
    finally:
        window.hide()


def test_loading_a_photo_leaves_processing_on(app, tmp_path):
    window, panel = _fresh_panel(app, tmp_path)
    try:
        assert panel.load_image(_photo(tmp_path / "p.png")) is True
        assert panel.process_check.isChecked() is True
    finally:
        window.hide()


def test_the_style_controls_are_hidden_while_processing_is_off(app, tmp_path):
    """A panel of dead sliders invites the user to look for a way to wake them."""
    window, panel = _fresh_panel(app, tmp_path)
    try:
        panel.load_image(_drawing(tmp_path / "d.png"))
        app.processEvents()
        for widget in (
            panel.style_group,
            panel.layer_group,
            panel.tuning_section,
            panel.denoise_section,
            panel.preview_section,
        ):
            assert not widget.isVisible(), "不处理时不该显示处理相关的控件"
    finally:
        window.hide()


def test_the_switch_is_a_suggestion_and_can_be_turned_back_on(app, tmp_path):
    """The user must be able to ask for tracing anyway.

    A mode the tool forces and the user cannot leave would be a worse problem
    than the one this feature solves.
    """
    window, panel = _fresh_panel(app, tmp_path)
    try:
        panel.load_image(_drawing(tmp_path / "d.png"))
        app.processEvents()
        assert panel.process_check.isChecked() is False

        panel.process_check.setChecked(True)
        app.processEvents()
        assert panel.style_group.isVisible(), "勾选后处理选项应当重新显示"
    finally:
        window.hide()


def test_importing_with_processing_off_snaps_the_picture_to_two_values(
    app, tmp_path
):
    """What reaches the board must be exactly two values.

    The source's edge ringing would otherwise be vectorised as extra mid-tones,
    which the silkscreen cannot print.
    """
    window, panel = _fresh_panel(app, tmp_path)
    try:
        panel.load_image(_drawing(tmp_path / "d.png"))
        assert panel.import_to_canvas() is True

        placement = window.design_canvas.board.placements[0]
        assert placement.processed_path, "二值化结果应当落盘，供画布与导出使用"
        # imread_unicode, not cv2.imread: the temporary path is not guaranteed
        # to be ASCII, and cv2's own reader fails silently on those.
        written = imread_unicode(placement.processed_path, cv2.IMREAD_UNCHANGED)
        assert set(np.unique(written)).issubset({0, 255})
    finally:
        window.hide()


def test_a_colour_picture_with_processing_off_is_left_alone(app, tmp_path):
    """Turning the switch off on a photo means "do not touch it".

    Converting it to greyscale here would be a silent edit to the picture the
    user just asked to be left alone.
    """
    window, panel = _fresh_panel(app, tmp_path)
    try:
        source = _photo(tmp_path / "p.png")
        panel.load_image(source)
        panel.process_check.setChecked(False)
        app.processEvents()
        assert panel.import_to_canvas() is True

        placement = window.design_canvas.board.placements[0]
        assert not placement.processed_path, "不处理时不该生成替代图片"
        assert Path(placement.path) == Path(source)
    finally:
        window.hide()


def test_a_raw_import_is_not_repainted_by_a_style_change(app, tmp_path):
    """The committed-style guard has to recognise the unprocessed case.

    A raw import has no style, so a later style change must not treat it as
    one and start writing conversions into it.
    """
    window, panel = _fresh_panel(app, tmp_path)
    try:
        panel.load_image(_drawing(tmp_path / "d.png"))
        panel.import_to_canvas()
        placement = window.design_canvas.board.placements[0]
        before = placement.processed_path

        # Switch the style: it describes the *next* import and must not touch
        # what is already on the board.
        names = [name for name, _ in gui_mockup.STYLE_OPTIONS]
        panel.style_combo.setCurrentIndex(names.index(gui_mockup.BOARD_ART_STYLE))
        app.processEvents()

        assert placement.processed_path == before
        assert len(window.design_canvas.board.placements) == 1
    finally:
        window.hide()


# ------------------------------------------------------------ panel placement


def test_floating_panels_land_inside_a_window_that_was_never_shown(app, tmp_path):
    """Regression: the panel opened off-screen and was invisible.

    Before the window is shown, ``QToolBar.height()`` answers with Qt's default
    widget size instead of the laid-out one, so the chrome total came to 1470 on
    a 900 px window and every floating panel was placed below the bottom edge.
    The panel was open, populated and correct -- and the user could not see it.
    Only the pre-``show()`` screenshot path exposed it, which is why the guard
    belongs in the geometry rather than in the screenshot tool.
    """
    window = build_window()
    try:
        window.resize(1420, 900)
        panel = show_preprocess(window)
        panel.place_top_right()

        assert chrome_height(window) < window.height(), "顶栏高度算得比窗口还高"
        assert panel.y() < window.height() - 100, (
            f"面板被放到窗口外：y={panel.y()}，窗口高 {window.height()}"
        )
    finally:
        window.hide()


def test_an_unshown_toolbar_reports_its_size_hint_not_the_default(app, tmp_path):
    """The measured cause, pinned so a future refactor cannot undo it."""
    from PySide6.QtWidgets import QToolBar

    window = build_window()
    try:
        bar = window.findChildren(QToolBar)[0]
        # It has never been shown, so its own height() is the meaningless
        # default; the size hint is the honest number.
        assert bar.height() != bar.sizeHint().height(), "前提变了，这条测试需要重看"
        assert chrome_height(window) < window.height()
    finally:
        window.hide()
