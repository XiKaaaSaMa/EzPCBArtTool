"""Tests for "the processed picture lands on the layer chosen in preprocessing".

The requirement: except for 彩色丝印, the canvas shows the *processed* image,
on the layer picked in the preprocessing panel, tinted in that layer's colour
the way the editor does it. These tests drive the preprocessing panel itself
and then ask the document which layers the picture ended up on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    COLOR_STYLE,
    STYLE_OPTIONS,
    build_window,
    layer_options,
    show_preprocess,
)
from pcbart.design_layers import layer  # noqa: E402
from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def colour_art(tmp_path):
    """A colour picture: the point is that the canvas must *not* show it."""
    path = tmp_path / "colour.png"
    image = np.zeros((300, 300, 4), dtype=np.uint8)
    cv2.circle(image, (150, 150), 110, (0, 0, 220, 255), -1)
    imwrite_unicode(str(path), image)
    assert path.exists()
    return path


@pytest.fixture
def loaded(app, colour_art):
    """A panel with a picture loaded for preview, *not* yet on the canvas."""
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    widget = show_preprocess(window)
    widget.parent_window = window
    assert widget.load_image(str(colour_art)) is True
    app.processEvents()
    yield widget
    widget.hide()
    window.hide()


@pytest.fixture
def panel(loaded):
    """A panel whose picture has been imported, which most tests here want."""
    assert loaded.import_to_canvas() is True
    return loaded


def board(panel):
    return panel.parent_window.design_canvas.board


def layer_keys(panel) -> list[str]:
    """Layer keys of the elements made by the most recent import.

    Scoped to the import rather than the whole board: importing the same file
    again now adds an independent element instead of rearranging the previous
    one, so a board-wide list would mix imports together.
    """
    return [
        item.layer_key
        for item in board(panel).placements
        if item.import_id == panel._import_id
    ]


def select_layer_choice(panel, index: int) -> None:
    """Choose a presentation *and import it*.

    The presentation controls describe the next import, so selecting a layer
    does nothing on its own; 导入图片 is what puts the picture there.
    """
    panel.layer_combo.setCurrentIndex(index)
    panel.import_to_canvas()


# ------------------------------------------------ uploading vs importing


def test_uploading_does_not_touch_the_canvas(loaded):
    """Upload reads the file for preview; the canvas stays empty until asked.

    This is the whole point of the two-step flow: parameters are settled
    before any geometry exists, so the board never carries a first guess the
    user then has to correct.
    """
    assert board(loaded).placements == []
    assert loaded._source_path is not None, "图片本身应已载入用于预览"


def test_tuning_before_importing_does_not_import(loaded):
    """Sliders must not commit anything: only the button does."""
    loaded.layer_combo.setCurrentIndex(1)
    loaded.strength_slider.setValue(180)
    loaded.threshold_slider.setValue(60)

    assert board(loaded).placements == []


def test_changing_style_before_importing_does_not_import(loaded):
    """A presentation choice rearranges an imported picture, never creates one."""
    names = [name for name, _ in STYLE_OPTIONS]
    loaded.style_combo.setCurrentIndex(names.index(COLOR_STYLE))

    assert board(loaded).placements == []


def test_the_import_button_is_disabled_until_a_picture_is_loaded(app):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    widget = show_preprocess(window)
    try:
        assert not widget.import_button.isEnabled()
    finally:
        window.hide()


def test_the_import_button_becomes_available_after_upload(loaded):
    assert loaded.import_button.isEnabled()


def test_pressing_import_puts_the_picture_on_the_canvas(loaded):
    assert loaded.import_button.isEnabled()

    loaded.import_button.click()

    assert layer_keys(loaded) == ["top_copper"]


def test_importing_twice_creates_two_independent_elements(loaded):
    """Each import is its own element, even from the same file.

    The point is that the same picture can go on the board more than once with
    different settings -- once as 线稿 on copper and once as 彩色丝印 on silk,
    say. Pressing the button again must not take over the previous element.
    """
    loaded.import_to_canvas()
    first = list(board(loaded).placements)
    first_ids = {item.import_id for item in first}

    loaded.import_to_canvas()
    everything = board(loaded).placements

    assert len(everything) == 2
    assert len({item.import_id for item in everything}) == 2, "两次导入应各有身份"
    # The first import's objects survive untouched, id included.
    survivors = [item for item in everything if item.import_id in first_ids]
    assert len(survivors) == 1
    assert survivors[0] is first[0]


def test_a_second_import_can_go_to_a_different_layer(loaded):
    """The real use case: same picture, two presentations, side by side."""
    loaded.import_to_canvas()
    loaded.style_combo.setCurrentIndex(
        [name for name, _ in STYLE_OPTIONS].index(COLOR_STYLE)
    )
    loaded.import_to_canvas()

    kinds = sorted(item.layer_key for item in board(loaded).placements)
    assert kinds == ["top_copper", "top_silk"], "两次导入应各自落在自己的层上"


def test_each_import_keeps_its_own_line_art_state(loaded):
    """A colour-silk import must not strip the line art from an earlier one."""
    loaded.import_to_canvas()
    line_art_import = board(loaded).placements[0]
    assert line_art_import.processed_path

    loaded.style_combo.setCurrentIndex(
        [name for name, _ in STYLE_OPTIONS].index(COLOR_STYLE)
    )
    loaded.import_to_canvas()

    assert line_art_import.processed_path, "先导入的线稿被后一次导入清掉了"
    colour_import = [
        item for item in board(loaded).placements if item is not line_art_import
    ]
    assert colour_import and colour_import[0].processed_path == ""


# ------------------------------------------------------- where it lands


def test_importing_puts_the_picture_on_the_selected_layer(panel):
    """The default choice is copper, so that is where a fresh import goes."""
    assert layer_keys(panel) == ["top_copper"]


def test_choosing_silkscreen_moves_the_picture_there(panel):
    select_layer_choice(panel, 1)  # 顶层丝印层

    assert layer_keys(panel) == ["top_silk"]


def test_choosing_the_bottom_side_imports_to_the_bottom(panel):
    """Clicked, not poked: this checks the wiring, not an internal method."""
    panel.side_control.buttons[1].click()  # type: ignore[attr-defined]
    panel.import_to_canvas()

    assert layer_keys(panel) == ["bottom_copper"]


def test_a_combined_choice_places_one_object_per_layer(panel):
    """顶层 + 顶层丝印层 is two layers, so it is two objects."""
    select_layer_choice(panel, 3)

    assert sorted(layer_keys(panel)) == ["top_copper", "top_silk"]


def test_all_three_layers_can_be_selected(panel):
    select_layer_choice(panel, 5)

    assert sorted(layer_keys(panel)) == ["top_copper", "top_mask", "top_silk"]


def test_changing_the_choice_does_not_leave_the_picture_behind(panel):
    """One import produces one element per layer named, and no leftovers.

    Going from a combined choice to a single layer must not strand the object
    that the combined choice created *for that import*.
    """
    select_layer_choice(panel, 3)
    assert sorted(layer_keys(panel)) == ["top_copper", "top_silk"]

    before = len(board(panel).placements)
    select_layer_choice(panel, 1)

    # The second import is a new element on silk; the first import's two
    # elements stay exactly where they were.
    assert len(board(panel).placements) == before + 1
    assert layer_keys(panel) == ["top_silk"], "新导入只应落在选中的层"


def test_moving_between_layers_keeps_the_geometry(panel):
    """Re-importing onto another layer keeps the geometry already chosen.

    The user has positioned and sized the picture; doing that again on a
    different layer should start from those numbers, not from the default.
    """
    placement = board(panel).placements[0]
    placement.x_mm, placement.y_mm = 12.5, -4.0
    placement.width_mm = 33.0
    placement.height_mm = 22.0

    select_layer_choice(panel, 1)

    moved = [
        p for p in board(panel).placements if p.import_id == panel._import_id
    ][0]
    assert (moved.x_mm, moved.y_mm) == pytest.approx((12.5, -4.0))
    assert (moved.width_mm, moved.height_mm) == pytest.approx((33.0, 22.0))


def test_the_layer_panel_current_layer_follows(panel):
    """Two places choose a layer; they must not disagree."""
    select_layer_choice(panel, 1)
    assert board(panel).layers.current == "top_silk"


# --------------------------------------------------- processed vs original


def test_line_art_is_what_the_canvas_shows(panel):
    """Every placement carries the processed image, not just the selected one."""
    select_layer_choice(panel, 3)  # two layers at once

    placements = board(panel).placements
    assert placements
    for placement in placements:
        assert placement.processed_path, (
            f"{placement.layer} 没有线稿，画布会退回显示彩色原图"
        )
        assert Path(placement.processed_path).exists()


def test_the_processed_file_is_a_line_art_not_the_colour_source(panel):
    placement = board(panel).placements[0]
    # imread_unicode: cv2.imread cannot open this project's non-ASCII path and
    # returns None rather than raising, which would look like a missing file.
    processed = imread_unicode(placement.processed_path, cv2.IMREAD_UNCHANGED)

    # A line art is strictly black and white; the source is a red disc.
    assert processed is not None
    unique = np.unique(processed)
    assert set(unique.tolist()) <= {0, 255}, f"线稿必须是二值图，实际含 {unique[:8]}"


def test_colour_silkscreen_keeps_the_original_picture(loaded):
    """彩色丝印 is the one style that preserves the artwork's own colours."""
    names = [name for name, _ in STYLE_OPTIONS]
    loaded.style_combo.setCurrentIndex(names.index(COLOR_STYLE))
    loaded.import_to_canvas()

    placement = [
        item for item in board(loaded).placements
        if item.import_id == loaded._import_id
    ][-1]
    assert placement.layer_key == "top_silk", "彩色只能印在丝印层"
    assert placement.processed_path == "", "彩色丝印不应挂线稿，否则会显示成单色"


def test_leaving_colour_silkscreen_restores_the_line_art(panel):
    names = [name for name, _ in STYLE_OPTIONS]
    panel.style_combo.setCurrentIndex(names.index(COLOR_STYLE))
    panel.style_combo.setCurrentIndex(0)

    placement = board(panel).placements[0]
    assert placement.processed_path, "切回线稿后应重新挂上线稿"


def test_colour_silkscreen_locks_the_layer(panel):
    names = [name for name, _ in STYLE_OPTIONS]
    panel.style_combo.setCurrentIndex(names.index(COLOR_STYLE))

    assert not panel.layer_combo.isEnabled(), "彩色丝印下呈现方式应锁定"

    panel.import_to_canvas()
    assert board(panel).placements[-1].layer_key == "top_silk"


# ------------------------------------------------------- the layer choices


def test_every_presentation_choice_names_real_layers():
    for side in ("顶层", "底层"):
        for label, keys, hint in layer_options(side):
            assert keys, f"{label} 没有指定图层"
            for key in keys:
                layer(key)  # raises if unknown
            assert hint


def test_the_side_picks_the_matching_layers():
    top = {key for _label, keys, _hint in layer_options("顶层") for key in keys}
    bottom = {
        key for _label, keys, _hint in layer_options("底层") for key in keys
    }

    assert top == {"top_copper", "top_silk", "top_mask"}
    assert bottom == {"bottom_copper", "bottom_silk", "bottom_mask"}


# ------------------------------------------------------- the source note


def test_the_source_note_does_not_accumulate_text(loaded):
    """Refreshing the label must not append to its own previous output.

    It used to recover the description by splitting its own text on
    "　·　板框", but the unconfigured branch writes "　·　尚未设置板框", which does
    not contain that separator -- so every refresh added another copy and the
    label read "尚未设置板框 · 尚未设置板框 · …".
    """
    for _ in range(4):
        loaded._refresh_board_note()

    text = loaded.source_note.text()
    assert text.count("尚未设置板框") == 1, text
    assert loaded._source_description in text


def test_the_source_note_switches_to_the_board_size_once_set(loaded):
    from pcbart.artboard import BoardOutlineSpec

    loaded.import_to_canvas()
    artboard = board(loaded)
    artboard.outline = BoardOutlineSpec(width_mm=50, height_mm=30)
    artboard.outline_ready = True

    loaded._refresh_board_note()

    text = loaded.source_note.text()
    assert "板框 50.0 × 30.0 mm" in text
    assert "尚未设置板框" not in text
