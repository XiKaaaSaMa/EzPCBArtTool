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
    from conftest import release_window

    # The *window* is what has to be freed: the panel is its child, and freeing
    # the panel alone leaves the window and its whole widget tree alive.
    release_window(window)


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


# ------------------------------------------- tuning must stay interactive


def test_the_trace_cache_survives_uploading_a_picture(loaded):
    """⚠️ The cache has to exist *after* a load, and this pins that.

    ``load_image`` used to reset it to ``None`` -- with a comment saying the
    cached conversion belonged to the previous picture, which was true but not
    the point. Nothing recreated it, so the cache was off for the rest of the
    session and every slider tick recomputed the subject mask: measured 900 ms
    per tick on a 1024 px picture instead of 100 ms, which is precisely the
    "拖动滑块非常卡" the cache was added to fix. A dropped cache is invisible in
    the output, so only an assertion like this catches it.
    """
    assert loaded._trace_cache is not None, "载入图片后缓存必须存在"


def test_a_slider_tick_reuses_the_picture_derived_work(loaded):
    """The second tick must not build the subject mask again.

    Counted rather than timed: a timing assertion would be flaky on a busy
    machine, while the call count is the thing that actually decides whether the
    drag is interactive. The mask and the grey conversion are functions of the
    picture alone, so exactly one computation is right however many ticks run.
    """
    import pcbart.lineart as lineart

    calls = {"count": 0}
    original = lineart.subject_masks

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    lineart.subject_masks = counted
    try:
        # ``load_image`` already ran one conversion, so the mask is in hand
        # before the first tick here. That is the intended behaviour: the
        # picture is processed once on upload and the sliders reuse it.
        before = calls["count"]
        for value in (60, 80, 100, 120, 140):
            loaded.strength_slider.setValue(value)
            loaded._refresh_preview()
        assert calls["count"] == before, (
            f"调参不该重算主体掩膜：{before} -> {calls['count']}"
        )
    finally:
        lineart.subject_masks = original


def test_a_new_picture_invalidates_the_trace_cache(loaded, tmp_path):
    """The cache must not answer for the previous file.

    Keyed on the picture's own bytes, so a different picture cannot hit; this
    pins that the panel does not instead reuse a stale *cache object*'s answer
    across an upload.
    """
    other = tmp_path / "other.png"
    image = np.zeros((260, 340, 4), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (300, 220), (30, 200, 30, 255), -1)
    imwrite_unicode(str(other), image)

    assert loaded.load_image(str(other)) is True
    loaded._refresh_preview()

    from pcbart.lineart import line_art, LineArtParams

    expected = line_art(loaded._source_bgr, loaded._source_alpha, LineArtParams())
    assert loaded._inline_processed.shape[:2] == expected.shape[:2]


def test_switching_readings_swaps_which_controls_are_live(loaded):
    """The controls a reading ignores are hidden, not left looking usable.

    ⚠️ This panel has been reported twice for controls that were visible,
    draggable and did nothing. 线条粗细 and 墨量阈值 are read by 纯线条
    (``mode="dodge"``) and by nothing else: 填充色块 takes its split from Otsu
    and 描边+填充 from the contour density, so in those two the sliders would
    move without changing a pixel. Hiding them is not cosmetics -- a slider the
    user can move but which has no effect is worse than an absent one.
    """
    from gui_mockup import (
        READING_DODGE,
        READING_FILL,
        READING_OPTIONS,
        READING_OUTLINE,
    )

    order = [key for key, _label, _hint in READING_OPTIONS]

    loaded.reading_combo.setCurrentIndex(order.index(READING_FILL))
    assert loaded.strength_row.isHidden()
    assert loaded.threshold_row.isHidden()
    assert loaded.outline_width_row.isHidden()
    assert loaded.outline_density_row.isHidden()

    loaded.reading_combo.setCurrentIndex(order.index(READING_OUTLINE))
    assert loaded.strength_row.isHidden()
    assert loaded.threshold_row.isHidden()
    assert not loaded.outline_width_row.isHidden()
    assert not loaded.outline_density_row.isHidden()

    loaded.reading_combo.setCurrentIndex(order.index(READING_DODGE))
    assert not loaded.strength_row.isHidden()
    assert not loaded.threshold_row.isHidden()
    assert loaded.outline_width_row.isHidden()
    assert loaded.outline_detail_row.isHidden()
    assert loaded.outline_density_row.isHidden()


def test_the_dodge_reading_stays_last_so_the_default_cannot_move(loaded):
    """⚠️ ``_reset_tuning`` returns to index 0, so ordering is a promise.

    Inserting a new reading above 填充色块 would silently change what every
    fresh import does, and the reset button would land on something else. The
    keys are checked rather than the labels so a reworded label cannot hide a
    reordered option.
    """
    from gui_mockup import READING_DODGE, READING_FILL, READING_OPTIONS

    order = [key for key, _label, _hint in READING_OPTIONS]

    assert order[0] == READING_FILL
    assert order[-1] == READING_DODGE
    assert len(order) == len(set(order)), "duplicate reading keys"
    assert loaded.reading_combo.currentIndex() == 0


def test_the_ink_threshold_reaches_the_model_in_the_dodge_reading(loaded):
    """The slider has to arrive at ``LineArtParams`` unchanged.

    0.46 is the calibrated value -- the model's own default as well as the
    point the earlier measurements were taken at -- so the panel must not
    rescale it on the way through.
    """
    from gui_mockup import READING_DODGE, READING_OPTIONS

    order = [key for key, _label, _hint in READING_OPTIONS]
    loaded.reading_combo.setCurrentIndex(order.index(READING_DODGE))

    loaded.threshold_slider.setValue(20)
    low = loaded._lineart_params()
    loaded.threshold_slider.setValue(85)
    high = loaded._lineart_params()
    loaded.strength_slider.setValue(250)
    weight = loaded._lineart_params()

    assert low.mode == READING_DODGE
    assert (low.threshold, high.threshold) == (0.20, 0.85)
    assert weight.strength == 2.5


def test_moving_the_ink_threshold_changes_the_dodge_artwork(loaded):
    """The controls are live end to end, not just plumbed.

    ⚠️ This is the counter-test to the dead control the panel shipped with: on
    the ramp below, 纯线条 must return strictly more ink as the threshold
    rises. The direction is the unintuitive one -- **higher threshold, more
    ink** -- because the threshold is a level on the dodge *ratio* and a pixel
    is inked when its response falls at or below it. Both ends are far from the
    measured values so a small rounding difference cannot flip the comparison.
    """
    from pcbart.lineart import line_art

    from gui_mockup import READING_DODGE, READING_OPTIONS

    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    picture = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)

    order = [key for key, _label, _hint in READING_OPTIONS]
    loaded.reading_combo.setCurrentIndex(order.index(READING_DODGE))
    loaded.silhouette_check.setChecked(False)

    def ink(threshold: int) -> int:
        loaded.threshold_slider.setValue(threshold)
        params = loaded._lineart_params()
        traced = line_art(picture, None, params)
        return int((traced < 128).sum())

    low, high = ink(10), ink(90)

    assert low > 0, "the ramp must produce some line at all"
    assert high > low
    # A superset in practice, not merely a different mask: every pixel inked at
    # the low threshold is still inked at the high one.
    loaded.threshold_slider.setValue(10)
    small = line_art(picture, None, loaded._lineart_params()) < 128
    loaded.threshold_slider.setValue(90)
    large = line_art(picture, None, loaded._lineart_params()) < 128
    assert not (small & ~large).any()


def test_the_reset_button_restores_the_default_reading(loaded):
    from gui_mockup import READING_FILL, READING_OPTIONS

    order = [key for key, _label, _hint in READING_OPTIONS]
    loaded.reading_combo.setCurrentIndex(order.index("outline"))
    loaded.outline_density_slider.setValue(200)
    loaded.outline_width_slider.setValue(80)

    loaded._reset_tuning()

    assert loaded.reading_combo.currentIndex() == order.index(READING_FILL)
    assert loaded.outline_density_slider.value() == 60
    assert loaded.outline_width_slider.value() == 20
    assert not loaded.arc_smooth_check.isChecked(), "弧线平滑默认为关"


def test_resetting_from_the_dodge_reading_puts_the_sliders_back(loaded):
    """Reset has to undo the new reading's controls too.

    They are hidden while 填充色块 is selected, so a value left behind would be
    invisible until the user next picks 纯线条 -- exactly the kind of stale
    state the panel's reset button exists to prevent.
    """
    from gui_mockup import READING_DODGE, READING_FILL, READING_OPTIONS

    order = [key for key, _label, _hint in READING_OPTIONS]
    loaded.reading_combo.setCurrentIndex(order.index(READING_DODGE))
    loaded.strength_slider.setValue(300)
    loaded.threshold_slider.setValue(90)

    loaded._reset_tuning()

    assert loaded.strength_slider.value() == 100
    assert loaded.threshold_slider.value() == 46
    assert loaded._reading() == READING_FILL
    assert loaded.strength_row.isHidden()


def test_arc_smoothing_runs_on_the_settled_pass_not_the_preview(loaded):
    """⚠️ The switch must not put 1.8 s back into every slider tick.

    Measured: the rest of the conversion is 36 ms on a 1024 px picture and
    ``clean_strokes_with_arcs`` is **1824 ms** on top of it. Running it per tick
    would reintroduce exactly the 调参卡顿 this panel was just fixed for, so it
    is deliberately gated to ``max_side is None`` -- the settled
    full-resolution conversion -- while the live preview stays at display
    resolution. The user still sees the smoothed result, one beat after they
    stop dragging rather than 20 times during the drag.
    """
    import pcbart.lineart as lineart
    from pcbart.work_scale import prepare_working_image

    calls = {"count": 0}
    original = lineart.clean_strokes_with_arcs

    def counted(mask, **kwargs):
        calls["count"] += 1
        return original(mask, **kwargs)

    lineart.clean_strokes_with_arcs = counted
    try:
        loaded.arc_smooth_check.setChecked(True)
        loaded._refresh_preview()          # the live path: must NOT smooth
        assert calls["count"] == 0, "实时预览不该跑弧线平滑"

        bgr, alpha, scale = prepare_working_image(
            loaded._source_bgr, loaded._source_alpha
        )
        loaded._process(bgr, alpha, param_scale=scale)
        assert calls["count"] == 1, "全分辨率那一遍必须跑弧线平滑"
    finally:
        lineart.clean_strokes_with_arcs = original
        loaded.arc_smooth_check.setChecked(False)
