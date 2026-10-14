"""The board-painting controls, driven through Qt rather than inspected.

The panel's job is to make the document say what the controls say, so these
checks move a control and then ask the *document* what happened: which layers
hold objects, and what image each of them points at. Reading the widgets back
would only prove they echo themselves.

Qt is imported through ``importorskip`` so the pure-Python part of the suite
still runs without PySide6. The window is never shown on screen.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    BOARD_ART_SIDE_OPTIONS,
    BOARD_ART_STYLE,
    COLOR_STYLE,
    STYLE_OPTIONS,
    build_window,
    show_preprocess,
)
from pcbart.design_layers import layer  # noqa: E402
from pcbart.image_prep import imread_unicode  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def painting_panel(app, tmp_path):
    """A preprocessing panel with a painting loaded and 板画 selected.

    The source is synthesised rather than read from the repository so the test
    does not depend on a user file, and it carries real transparency so the
    silhouette path is exercised rather than the tone-inference one.

    Three bands, not two: light, *mid* and dark. The middle one is what makes
    the tone-threshold checks meaningful -- with only light and dark in the
    picture, every threshold between them produces the same split, so a broken
    threshold control would pass. The bands are sized so that neither the
    threshold nor the flip can be confused with the other.
    """
    import cv2

    rgba = np.zeros((160, 120, 4), np.uint8)
    rgba[20:140, 20:100, :3] = 235  # light
    rgba[20:140, 20:100, 3] = 255
    rgba[60:80, 20:100, :3] = 150  # mid: swings with the threshold
    rgba[100:120, 20:100, :3] = 30  # dark
    source = tmp_path / "painting.png"
    cv2.imwrite(str(source), rgba)

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_preprocess(window)
    panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    panel.load_image(str(source))
    app.processEvents()
    yield panel
    from conftest import release_window

    # The *window* is what has to be freed: the panel is its child, and freeing
    # the panel alone leaves the window and its whole widget tree alive.
    release_window(window)


def imported(panel):
    """The placements belonging to the panel's current import."""
    board = panel.parentWidget().design_canvas.board
    return [p for p in board.placements if p.import_id == panel._import_id]


# ------------------------------------------------------- control visibility


def test_board_art_offers_its_own_side_choice(painting_panel):
    """板画 replaces the layer list with 顶面/底面/两面.

    The line-art list names physical layer combinations, which cannot express a
    painting: it is always the same two layers, and it may go on both faces.
    """
    assert painting_panel.board_art_side_combo.isVisible()
    assert painting_panel.board_art_side_row.isVisible()
    assert not painting_panel.presentation_row.isVisible()
    assert not painting_panel.side_control.isVisible()


def test_the_side_choice_has_exactly_three_entries(painting_panel):
    combo = painting_panel.board_art_side_combo
    assert combo.count() == 3
    assert [combo.itemText(i) for i in range(3)] == [
        label for _key, label, _hint in BOARD_ART_SIDE_OPTIONS
    ]


def test_the_line_art_sections_are_hidden_while_painting(painting_panel):
    """Knobs that do nothing to a painting must not be on screen.

    A control that silently has no effect is the placeholder problem this
    panel has already been burned by.
    """
    assert not painting_panel.tuning_section.isVisible()
    assert not painting_panel.prestep_section.isVisible()
    assert not painting_panel.cleanup_section.isVisible()
    assert painting_panel.board_art_section.isVisible()


def test_the_board_art_sections_are_hidden_for_line_art(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index("线稿")
    )
    assert painting_panel.tuning_section.isVisible()
    assert painting_panel.prestep_section.isVisible()
    assert painting_panel.cleanup_section.isVisible()
    assert not painting_panel.board_art_section.isVisible()
    assert painting_panel.presentation_row.isVisible()
    assert painting_panel.side_control.isVisible()


def test_the_preview_switches_when_the_style_changes(painting_panel):
    """Picking a style must show that style, with no further clicks.

    The preview used to wait for some other control to be touched, so the
    upload box kept showing the previous style's picture and the new style
    looked broken.
    """
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index("线稿")
    )
    painting_panel._refresh_preview()
    line_art = painting_panel.drop_zone._caption

    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    painting_caption = painting_panel.drop_zone._caption

    assert line_art != painting_caption
    assert "线稿" in line_art
    assert "板画" in painting_caption


def test_the_preview_labels_each_style_differently(painting_panel):
    labels = {}
    for style in ("线稿", BOARD_ART_STYLE, COLOR_STYLE):
        painting_panel.style_combo.setCurrentIndex(
            [name for name, _hint in STYLE_OPTIONS].index(style)
        )
        labels[style] = painting_panel.drop_zone._caption

    assert len(set(labels.values())) == 3, labels


def test_colour_style_previews_the_original_picture(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(COLOR_STYLE)
    )
    assert "彩色丝印" in painting_panel.drop_zone._caption


def test_the_preview_image_itself_changes_with_the_style(painting_panel):
    """Not just the caption: the pixels in the box have to be the new style.

    Compared by *content*, not by size. Both styles render at the same
    resolution, so a size check would pass even if the box kept showing the
    previous style's picture -- which is exactly the bug being guarded.
    """
    def signature():
        image = painting_panel.drop_zone._pixmap.toImage()
        pixels = bytes(image.constBits())
        return (image.width(), image.height(), hashlib.sha1(pixels).hexdigest())

    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index("线稿")
    )
    line_art = signature()

    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    painting = signature()

    assert line_art[:2] == painting[:2], "both render at the same size"
    assert line_art[2] != painting[2], "but the pixels must differ"


# -------------------------------------------------------- sections by style


def test_colour_style_hides_every_processing_section(painting_panel):
    """彩色丝印 processes nothing, so only 上传/风格/呈现方式 remain.

    Every later section is a knob or a preview for a conversion that this style
    does not perform.
    """
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(COLOR_STYLE)
    )

    assert not painting_panel.tuning_section.isVisible()
    assert not painting_panel.prestep_section.isVisible()
    assert not painting_panel.cleanup_section.isVisible()
    assert not painting_panel.board_art_section.isVisible()
    assert not painting_panel.preview_section.isVisible()


def test_line_art_shows_the_processing_sections(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index("线稿")
    )
    assert painting_panel.tuning_section.isVisible()
    assert painting_panel.prestep_section.isVisible()
    assert painting_panel.cleanup_section.isVisible()
    assert painting_panel.preview_section.isVisible()


def test_the_preview_section_is_reachable_for_every_processing_style(painting_panel):
    """The big preview must not be trapped inside one style's controls.

    It used to live inside 线稿调试, so choosing 板画 hid the only way to open
    it -- the style that changes the picture most had no way to inspect it.
    """
    for style in ("线稿", BOARD_ART_STYLE):
        painting_panel.style_combo.setCurrentIndex(
            [name for name, _hint in STYLE_OPTIONS].index(style)
        )
        assert painting_panel.preview_section.isVisible(), style


def test_the_big_preview_opens_while_painting(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    panel = painting_panel.open_preview_window()

    assert panel.isVisible()
    assert panel._image_size is not None


def test_the_big_preview_shows_the_composite_not_one_mask(painting_panel):
    """Handing the window a raw mask would draw a tone instead of the artwork."""
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    painting_panel.open_preview_window()
    painting_panel.refresh_preview_full()

    art = painting_panel._full_processed
    shown = painting_panel._preview_panel.view._image
    assert shown is not None and not shown.isNull()
    assert (shown.width(), shown.height()) == (art.white.shape[1], art.white.shape[0])


# ---------------------------------------------------------------- invert


def test_inverting_swaps_the_tones_in_the_preview(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(BOARD_ART_STYLE)
    )
    painting_panel.invert_check.setChecked(False)
    painting_panel._refresh_preview()
    plain = painting_panel._inline_processed

    painting_panel.invert_check.setChecked(True)
    painting_panel._refresh_preview()
    inverted = painting_panel._inline_processed

    assert np.array_equal(inverted.white, plain.black)
    assert np.array_equal(inverted.black, plain.white)


def test_unchecking_invert_restores_the_original(painting_panel):
    painting_panel.invert_check.setChecked(True)
    painting_panel._refresh_preview()
    painting_panel.invert_check.setChecked(False)
    painting_panel._refresh_preview()

    plain = painting_panel._inline_processed
    assert plain.white.any() and plain.black.any()


def test_inverting_reaches_the_imported_files(painting_panel):
    """The preview and the board must agree about which tone is which."""
    painting_panel.invert_check.setChecked(True)
    painting_panel.import_to_canvas()
    paths = {p.layer_key: p.processed_path for p in imported(painting_panel)}

    # The white file carries the white tone, which after inverting is the big
    # one: this picture is mostly background, and inverting puts the background
    # in white ink.
    white_ink = (imread_unicode(paths["top_silk"], 0) < 128).mean()
    black_ink = (imread_unicode(paths["top_copper"], 0) < 128).mean()

    assert white_ink > black_ink
    assert white_ink + black_ink == pytest.approx(1.0, abs=0.02)


def test_resetting_clears_the_invert(painting_panel):
    painting_panel.invert_check.setChecked(True)
    painting_panel._reset_tuning()
    assert not painting_panel.invert_check.isChecked()


# ------------------------------------------------------------- the preview


def test_the_preview_shows_the_two_tone_composite(painting_panel):
    """The thumbnail must show what the board will look like, not one mask."""
    display = painting_panel._inline_processed
    assert hasattr(display, "white"), "板画 must resolve to two tones"
    assert display.is_complete_partition


def test_lowering_the_tone_threshold_makes_more_white(painting_panel):
    """The mid band is 150/255 = 0.59, so 0.40 keeps it white and 0.70 does not."""
    painting_panel.tone_slider.setValue(40)
    painting_panel._refresh_preview()
    low = painting_panel._inline_processed.white.mean()

    painting_panel.tone_slider.setValue(70)
    painting_panel._refresh_preview()
    high = painting_panel._inline_processed.white.mean()

    assert low > high, "a lower threshold must keep more of the picture white"


def test_the_flip_checkboxes_change_the_preview(painting_panel):
    painting_panel.flip_vertical_check.setChecked(False)
    painting_panel.flip_horizontal_check.setChecked(False)
    painting_panel._refresh_preview()
    plain = painting_panel._inline_processed.white.copy()

    painting_panel.flip_horizontal_check.setChecked(True)
    painting_panel._refresh_preview()
    mirrored = painting_panel._inline_processed.white

    assert np.array_equal(mirrored, plain[:, ::-1])


def test_a_flip_reaches_the_full_resolution_conversion(painting_panel):
    """The preview is a scaled conversion; the import is a full-size one.

    Flipping one and not the other would mean the picture the user approved is
    not the picture that lands on the board.
    """
    painting_panel.flip_vertical_check.setChecked(False)
    full = painting_panel._process(
        painting_panel._source_bgr, painting_panel._source_alpha
    )
    painting_panel.flip_vertical_check.setChecked(True)
    flipped = painting_panel._process(
        painting_panel._source_bgr, painting_panel._source_alpha
    )

    assert np.array_equal(flipped.white, full.white[::-1])


def test_resetting_restores_the_board_art_defaults(painting_panel):
    painting_panel.tone_slider.setValue(80)
    painting_panel.flip_horizontal_check.setChecked(True)
    painting_panel.flip_vertical_check.setChecked(True)
    painting_panel.background_check.setChecked(False)

    painting_panel._reset_tuning()

    assert painting_panel.tone_slider.value() == 62
    assert not painting_panel.flip_horizontal_check.isChecked()
    assert not painting_panel.flip_vertical_check.isChecked()
    assert painting_panel.background_check.isChecked()


# ----------------------------------------------------------- what gets made


def test_importing_places_white_on_silk_and_black_on_copper(painting_panel):
    painting_panel.import_to_canvas()
    placed = {p.layer_key for p in imported(painting_panel)}

    assert placed == {"top_silk", "top_copper"}


def test_each_tone_points_at_its_own_file(painting_panel):
    """Two tones cannot share one file: the second would overwrite the first
    and both layers would draw the same image."""
    painting_panel.import_to_canvas()
    paths = {p.layer_key: p.processed_path for p in imported(painting_panel)}

    assert paths["top_silk"] != paths["top_copper"]

    silk_ink = (imread_unicode(paths["top_silk"], 0) < 128).mean()
    copper_ink = (imread_unicode(paths["top_copper"], 0) < 128).mean()
    # Silk carries the white tone, which covers less of this picture; copper
    # carries its complement.
    assert silk_ink < copper_ink
    assert silk_ink + copper_ink == pytest.approx(1.0, abs=0.02)


def test_the_silk_file_is_published_in_the_canvas_ink_convention(painting_panel):
    """The canvas tints *ink*, so the file must be 0 where the tone is set.

    Publishing the mask's own polarity would tint the empty area instead and
    draw every shape inside out -- a defect that still looks like a plausible
    picture, so it is asserted rather than eyeballed.
    """
    painting_panel.import_to_canvas()
    art = painting_panel._full_processed
    paths = {p.layer_key: p.processed_path for p in imported(painting_panel)}

    white_file = imread_unicode(paths["top_silk"], 0)
    assert np.array_equal(white_file == 0, art.white)

    black_file = imread_unicode(paths["top_copper"], 0)
    assert np.array_equal(black_file == 0, art.black)


def test_both_faces_get_both_tones(painting_panel):
    keys = [key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS]
    painting_panel.board_art_side_combo.setCurrentIndex(keys.index("both"))
    painting_panel.import_to_canvas()
    placed = {p.layer_key for p in imported(painting_panel)}

    assert placed == {"top_silk", "top_copper", "bottom_silk", "bottom_copper"}


def test_bottom_moves_both_tones_to_the_bottom(painting_panel):
    keys = [key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS]
    painting_panel.board_art_side_combo.setCurrentIndex(keys.index("bottom"))
    painting_panel.import_to_canvas()
    placed = {p.layer_key for p in imported(painting_panel)}

    assert placed == {"bottom_silk", "bottom_copper"}


def test_changing_the_side_does_not_move_an_import_already_placed(painting_panel):
    """板面 describes the *next* import, like 呈现方式 does.

    Acting on it immediately moved elements the user had already positioned --
    a bug this project has already shipped once.
    """
    painting_panel.import_to_canvas()
    keys = [key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS]
    painting_panel.board_art_side_combo.setCurrentIndex(keys.index("both"))

    assert {p.layer_key for p in imported(painting_panel)} == {
        "top_silk",
        "top_copper",
    }


def test_reimporting_after_a_side_change_moves_the_picture(painting_panel):
    """The side change is deferred, not ignored: 导入图片 is what acts on it."""
    painting_panel.import_to_canvas()
    keys = [key for key, _label, _hint in BOARD_ART_SIDE_OPTIONS]
    painting_panel.board_art_side_combo.setCurrentIndex(keys.index("bottom"))
    painting_panel.import_to_canvas()

    assert {p.layer_key for p in imported(painting_panel)} == {
        "bottom_silk",
        "bottom_copper",
    }


# --------------------------------------------------------- other styles


def test_colour_style_still_locks_to_silk(painting_panel):
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index(COLOR_STYLE)
    )
    painting_panel.import_to_canvas()

    assert {p.layer_key for p in imported(painting_panel)} == {"top_silk"}


def test_switching_away_from_board_art_leaves_the_board_usable(painting_panel):
    painting_panel.import_to_canvas()
    painting_panel.style_combo.setCurrentIndex(
        [name for name, _hint in STYLE_OPTIONS].index("线稿")
    )
    painting_panel.import_to_canvas()

    board = painting_panel.parentWidget().design_canvas.board
    assert board.placements, "switching styles must not empty the board"


def test_every_object_lands_on_a_real_layer(painting_panel):
    painting_panel.import_to_canvas()
    for placement in imported(painting_panel):
        assert layer(placement.layer_key).kind in {"copper", "silk"}


# ------------------------------------- switching style must not corrupt state


def test_switching_style_after_import_does_not_crash(painting_panel):
    """板画 resolves to two masks, 线稿 to one image.

    Publishing treats them differently, and the style combo describes the
    *next* import -- so switching it after an import must not feed the new
    style's result to objects committed under the old one. Doing so raised
    AttributeError ('numpy.ndarray' object has no attribute 'white') instead of
    repainting.
    """
    panel = painting_panel
    panel.load_image(panel._source_path)
    panel.import_to_canvas()
    committed = panel._committed_style

    from gui_mockup import STYLE_OPTIONS

    for index in range(len(STYLE_OPTIONS)):
        panel.style_combo.setCurrentIndex(index)

    assert panel._committed_style == committed, "切换风格不应改变已落图对象的身份"


def test_tuning_still_reaches_the_canvas_and_the_committed_file(painting_panel, app):
    """The live-update behaviour the guard above had to not break.

    Moving a slider republishes the preview so the canvas tracks it, and the
    full-resolution conversion follows once the drag settles -- otherwise the
    thumbnail the preview wrote would be what the board is exported from.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    from pcbart.image_prep import imread_unicode

    panel = painting_panel
    panel.load_image(panel._source_path)
    panel.import_to_canvas()
    placement = panel.parentWidget().design_canvas.board.image
    full = imread_unicode(placement.processed_path, 1).shape

    panel.threshold_slider.setValue(30)
    app.processEvents()

    loop = QEventLoop()
    QTimer.singleShot(700, loop.quit)
    loop.exec()

    assert imread_unicode(placement.processed_path, 1).shape == full, (
        "调参结束后应写回全分辨率，而不是把预览缩略图留在落盘位置"
    )
