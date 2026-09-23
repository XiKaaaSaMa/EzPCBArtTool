"""Tests for the board-setup panel, driven through Qt rather than inspected.

The panel's job is to make the document say what the controls say, so the
checks below ask the document, not the widget: switch a shape, then look at
what the board actually became.

Qt is imported through ``importorskip`` because the rest of the suite is pure
Python and has to keep running where PySide6 is missing. The window is never
shown on screen -- ``WA_DontShowOnScreen`` lays it out and paints it while
leaving the desktop alone -- so running the suite does not disturb the user.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    IRREGULAR_BASES,
    OUTLINE_SHAPES,
    SHAPE_CHOICES,
    BoardSetupPanel,
    IrregularBase,
    OutlineShape,
    build_window,
    show_preprocess,
)
from pcbart.artboard import (  # noqa: E402
    DEFAULT_IMAGE_LONG_EDGE_MM,
    ImagePlacement,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def panel(app):
    """A board panel over a fresh window, laid out but never shown."""
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    widget = BoardSetupPanel(window, window.design_canvas.board)
    widget.parent_window = window
    widget.show()
    app.processEvents()
    yield widget
    widget.hide()
    window.hide()


def select_shape(panel, shape: str) -> None:
    panel.shape_combo.setCurrentIndex(SHAPE_CHOICES.index(shape))


def wheel(widget) -> None:
    """One wheel notch delivered to the widget's own centre."""
    centre = widget.rect().center()
    event = QWheelEvent(
        QPointF(centre),
        QPointF(widget.mapToGlobal(centre)),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


# ---------------------------------------------------------- shape before size


def test_opening_the_panel_configures_the_board(panel):
    """The panel is the act of choosing a board, so the canvas gets one."""
    assert panel.parent_window.design_canvas.board.outline_ready is True


def test_shapes_carry_the_size_over_between_two_dimensional_shapes(panel):
    board = panel.parent_window.design_canvas.board
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(80.0)
    panel.height_spin.setValue(40.0)

    select_shape(panel, OutlineShape.ROUNDED)

    assert board.outline.shape == OutlineShape.ROUNDED
    assert (board.outline.width_mm, board.outline.height_mm) == pytest.approx((80, 40))


def test_a_circle_shows_one_dimension_and_stores_a_square(panel):
    board = panel.parent_window.design_canvas.board
    select_shape(panel, OutlineShape.CIRCLE)

    assert panel.diameter_row.isVisibleTo(panel)
    assert not panel.width_row.isVisibleTo(panel)
    assert not panel.height_row.isVisibleTo(panel)
    assert not panel.board_aspect_check.isVisibleTo(panel)
    assert board.outline.width_mm == pytest.approx(board.outline.height_mm)


def test_the_diameter_control_writes_both_sides(panel):
    board = panel.parent_window.design_canvas.board
    select_shape(panel, OutlineShape.CIRCLE)

    panel.diameter_spin.setValue(42.0)

    assert board.outline.diameter_mm == pytest.approx(42.0)
    assert board.outline.height_mm == pytest.approx(42.0)


def test_leaving_a_circle_restores_the_two_dimension_controls(panel):
    select_shape(panel, OutlineShape.CIRCLE)
    select_shape(panel, OutlineShape.ROUNDED)

    assert panel.width_row.isVisibleTo(panel)
    assert panel.height_row.isVisibleTo(panel)
    assert panel.radius_row.isVisibleTo(panel)
    assert not panel.diameter_row.isVisibleTo(panel)


def test_only_the_rounded_shape_offers_a_corner_radius(panel):
    select_shape(panel, OutlineShape.ROUNDED)
    assert panel.radius_row.isVisibleTo(panel)

    select_shape(panel, OutlineShape.RECT)
    assert not panel.radius_row.isVisibleTo(panel)


def test_a_circle_grows_to_hold_the_artwork(panel):
    """Switching to a circle must not leave the picture hanging off the board.

    The smallest circle around a rectangle goes through its corners, so the
    diameter is the diagonal -- not the shorter side, which is the tempting
    and wrong choice.
    """
    board = panel.parent_window.design_canvas.board
    board.image = ImagePlacement(width_mm=60.0, height_mm=30.0)

    select_shape(panel, OutlineShape.CIRCLE)

    expected = math.hypot(60.0, 30.0)
    assert board.outline.diameter_mm == pytest.approx(expected)
    # The painted geometry, not just the number, is a circle of that size.
    xs = [p[0] for p in board.outline_polygon()]
    ys = [p[1] for p in board.outline_polygon()]
    assert (max(xs) - min(xs)) == pytest.approx(expected)
    assert (max(ys) - min(ys)) == pytest.approx(expected)


def test_switching_shape_repeatedly_does_not_creep(panel):
    """Re-deriving from the artwork must be stable, not cumulative.

    Deriving the circle from the *previous board* instead would grow it a
    little on every switch and never shrink back.
    """
    board = panel.parent_window.design_canvas.board
    board.image = ImagePlacement(width_mm=60.0, height_mm=30.0)

    select_shape(panel, OutlineShape.CIRCLE)
    first = board.outline.diameter_mm
    for _ in range(4):
        select_shape(panel, OutlineShape.RECT)
        select_shape(panel, OutlineShape.CIRCLE)

    assert board.outline.diameter_mm == pytest.approx(first)


def test_the_image_size_controls_resize_the_artwork(panel):
    board = panel.parent_window.design_canvas.board
    board.image = ImagePlacement(width_mm=60.0, height_mm=30.0)
    panel._sync_controls()

    panel.image_width_spin.setValue(120.0)

    assert board.image.width_mm == pytest.approx(120.0)
    assert board.image.height_mm == pytest.approx(60.0), "长宽比锁定"


def test_dragging_on_the_canvas_updates_the_panels_fields(panel):
    """The canvas and the panel edit the same picture, so both must agree.

    A drag happens entirely inside the canvas; without a refresh the panel
    would keep showing the size the artwork had before the drag.
    """
    window = panel.parent_window
    board = window.design_canvas.board
    board.image = ImagePlacement(width_mm=60.0, height_mm=30.0)
    panel._sync_controls()
    assert panel.image_width_spin.value() == pytest.approx(60.0)

    # Simulate the result of a drag, then announce it the way the canvas does.
    board.image.width_mm = 90.0
    board.image.height_mm = 45.0
    window.design_canvas.placement_changed.emit()

    assert panel.image_width_spin.value() == pytest.approx(90.0)
    assert panel.image_height_spin.value() == pytest.approx(45.0)


# ---------------------------------------------------------------- irregular


def select_base(panel, base: str) -> None:
    panel.base_combo.setCurrentIndex(IRREGULAR_BASES.index(base))


@pytest.fixture
def irregular_panel(panel, tmp_path):
    """A panel showing 异形板 over a standee-like artwork."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    source = tmp_path / "standee.png"
    image = np.zeros((400, 300, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    cv2.rectangle(image, (80, 180), (220, 395), (60, 60, 60, 255), -1)
    cv2.rectangle(image, (135, 40), (165, 190), (60, 60, 60, 255), -1)
    assert cv2.imwrite(str(source), image)

    from pcbart.artboard import placement_for_pixels

    window = panel.parent_window
    window.design_canvas.set_placement(placement_for_pixels(str(source), 300, 400))
    panel._sync_controls()
    select_shape(panel, OutlineShape.IRREGULAR)
    return panel


def test_irregular_is_offered_as_a_shape(panel):
    model = panel.shape_combo.model()
    labels = [model.item(i).text() for i in range(panel.shape_combo.count())]

    assert OutlineShape.LABELS[OutlineShape.IRREGULAR] in labels


def test_the_irregular_settings_appear_only_for_that_shape(panel):
    select_shape(panel, OutlineShape.RECT)
    assert not panel.irregular_group.isVisibleTo(panel)

    select_shape(panel, OutlineShape.IRREGULAR)
    assert panel.irregular_group.isVisibleTo(panel)


def test_the_three_base_choices_are_offered_in_order(irregular_panel):
    labels = [irregular_panel.base_combo.itemText(i)
              for i in range(irregular_panel.base_combo.count())]

    assert labels == [
        IrregularBase.LABELS[IrregularBase.TRACE],
        IrregularBase.LABELS[IrregularBase.RECT],
        IrregularBase.LABELS[IrregularBase.ROUNDED],
    ]


def test_a_traced_board_hides_the_artwork_size_controls(irregular_panel):
    """Under 原图裁切 the board size *is* the artwork size.

    Showing both would put two sets of numbers on screen describing one thing.
    """
    select_base(irregular_panel, IrregularBase.TRACE)

    assert not irregular_panel.image_group.isVisibleTo(irregular_panel)


def test_a_combining_base_shows_the_artwork_size_controls(irregular_panel):
    select_base(irregular_panel, IrregularBase.TRACE)
    select_base(irregular_panel, IrregularBase.RECT)

    assert irregular_panel.image_group.isVisibleTo(irregular_panel)


def test_the_base_sliders_only_apply_to_a_combining_base(irregular_panel):
    select_base(irregular_panel, IrregularBase.TRACE)
    assert not irregular_panel.base_top_row.isVisibleTo(irregular_panel)
    assert not irregular_panel.image_x_row.isVisibleTo(irregular_panel)

    select_base(irregular_panel, IrregularBase.RECT)
    assert irregular_panel.base_top_row.isVisibleTo(irregular_panel)
    assert irregular_panel.image_x_row.isVisibleTo(irregular_panel)


def test_the_base_top_slider_moves_the_cut_line(irregular_panel):
    select_base(irregular_panel, IrregularBase.RECT)

    irregular_panel.base_top_slider.setValue(60)

    assert irregular_panel._board.outline.base_top_ratio == pytest.approx(0.60)
    # And the drawn geometry changes with it, not just the number.
    ys = [p[1] for p in irregular_panel._board.outline_polygon()]
    height = irregular_panel._board.outline.height_mm
    assert max(ys) >= -height / 2 + 0.35 * height


def test_the_horizontal_slider_moves_the_artwork_inside_the_base(irregular_panel):
    select_base(irregular_panel, IrregularBase.RECT)
    board = irregular_panel._board
    board.image.x_mm = 0.0

    irregular_panel.image_x_slider.setValue(100)
    right = board.image.x_mm

    irregular_panel.image_x_slider.setValue(-100)
    left = board.image.x_mm

    assert right > 0 > left
    assert right == pytest.approx(-left), "两侧行程应对称"
    # The travel is the real free space, so the artwork stays inside the base.
    assert right <= (board.outline.width_mm - board.image.width_mm) / 2 + 1e-6


def test_a_traced_board_grows_to_the_artwork(irregular_panel):
    """原图裁切 has no preset, so the board follows the picture exactly."""
    select_base(irregular_panel, IrregularBase.TRACE)
    board = irregular_panel._board

    assert board.outline.width_mm == pytest.approx(board.image.width_mm)
    assert board.outline.height_mm == pytest.approx(board.image.height_mm)


def test_the_rounded_base_exposes_the_corner_radius(irregular_panel):
    select_base(irregular_panel, IrregularBase.ROUNDED)
    assert irregular_panel.radius_row.isVisibleTo(irregular_panel)

    select_base(irregular_panel, IrregularBase.RECT)
    assert not irregular_panel.radius_row.isVisibleTo(irregular_panel)


def test_an_irregular_board_without_artwork_draws_no_trace(panel):
    """Nothing to trace means no board, not an invented one."""
    select_shape(panel, OutlineShape.IRREGULAR)
    select_base(panel, IrregularBase.TRACE)

    assert panel._board.outline_polygon() == []
    assert "图片" in panel.status_label.text(), "应提示需要先导入图片"


def test_the_irregular_outline_reaches_the_canvas(irregular_panel):
    """The composed shape the panel describes is what the canvas paints."""
    select_base(irregular_panel, IrregularBase.RECT)
    ring = irregular_panel._board.outline_polygon()

    assert len(ring) > 8
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    board = irregular_panel._board
    # The base spans the full board width, and the ear rises above the cut.
    assert (min(xs), max(xs)) == pytest.approx(
        (-board.outline.width_mm / 2, board.outline.width_mm / 2)
    )
    assert max(ys) > -board.outline.height_mm / 2 + 0.35 * board.outline.height_mm


# ----------------------------------------------------------------- 立牌


def tick_standee(panel) -> None:
    panel.standee_check.setChecked(True)


def tab_centre_x(panel) -> float:
    """Where the tab sits, read from the geometry rather than the model.

    The model's offset is what the slider writes; this is what the board
    actually became, which is the number the user sees on the canvas.
    """
    ring = panel._board.outline_polygon()
    bottom = min(y for _x, y in ring)
    below = [x for x, y in ring if y < bottom + 1e-9]
    return sum(below) / len(below)


def test_the_standee_option_is_off_until_it_is_ticked(panel):
    select_shape(panel, OutlineShape.RECT)

    assert not panel._board.outline.standee
    assert not panel.standee_group.isVisibleTo(panel), "未勾选时不应显示立牌设置"


def test_ticking_standee_reveals_its_settings(panel):
    select_shape(panel, OutlineShape.RECT)

    tick_standee(panel)

    assert panel._board.outline.standee is True
    assert panel.standee_group.isVisibleTo(panel)


def test_the_tab_starts_at_the_default_size(panel):
    """16 × 3 mm, and hanging off the bottom without changing the board."""
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)

    tick_standee(panel)

    board = panel._board
    assert panel.standee_width_spin.value() == pytest.approx(16.0)
    assert panel.standee_height_spin.value() == pytest.approx(3.0)
    assert board.outline.width_mm == pytest.approx(50.0), "插舌不应改变板框宽度"
    ys = [y for _x, y in board.outline_polygon()]
    assert min(ys) == pytest.approx(-15.0 - 3.0)
    assert max(ys) == pytest.approx(15.0)


def test_typing_a_tab_size_resizes_the_tab(panel):
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)

    panel.standee_width_spin.setValue(8.0)
    panel.standee_height_spin.setValue(2.0)

    ring = panel._board.outline_polygon()
    bottom = min(y for _x, y in ring)
    below = [x for x, y in ring if y < bottom + 1e-9]

    assert (max(below) - min(below)) == pytest.approx(8.0)
    assert bottom == pytest.approx(-15.0 - 2.0)


def test_the_position_slider_slides_the_tab_along_the_board(panel):
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)

    panel.standee_x_slider.setValue(100)
    right = tab_centre_x(panel)
    panel.standee_x_slider.setValue(-100)
    left = tab_centre_x(panel)

    assert right > 0 > left
    assert right == pytest.approx(-left), "两侧行程应对称"
    # And the tab is still attached at both ends of the slider.
    assert panel._board.standee_error is None


def test_the_slider_range_is_the_real_room_on_the_bottom_edge(panel):
    """±(board width − tab width) / 2, and it says so in millimetres."""
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)

    travel = panel._board.standee_travel_mm()

    assert travel == pytest.approx((50.0 - 16.0) / 2)
    panel.standee_x_slider.setValue(100)
    assert panel._board.outline.standee_offset_mm == pytest.approx(travel)
    assert f"±{travel:.1f} mm" in panel.standee_note.text()


def test_a_tab_as_wide_as_the_bottom_edge_offers_no_slider(panel):
    """With no room to slide, the slider is disabled rather than inert.

    This is the same rule the irregular board's slider follows -- a control
    that looks usable but cannot do anything is worse than no control.
    """
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(20.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)
    contact = panel._board.standee_contact_mm()
    panel.standee_width_spin.setValue(contact)

    assert panel._board.standee_travel_mm() == pytest.approx(0.0)
    assert not panel.standee_x_slider.isEnabled()
    assert "没有可移动的余量" in panel.standee_note.text()


def test_a_tab_wider_than_the_bottom_edge_still_attaches(panel):
    """A tab wider than the flat it hangs from is a real, wanted case.

    The default 16 mm tab is wider than the usable bottom of any circle this
    tool can draw, and that is not a fault: the tab is cut to fit the slot in
    the display base, so its corners reaching past a curved bottom is exactly
    what it is supposed to look like. The note states the fact -- there is
    nowhere to slide to -- without telling the user to shrink the one dimension
    the base dictates.
    """
    select_shape(panel, OutlineShape.CIRCLE)
    tick_standee(panel)

    contact = panel._board.standee_contact_mm()
    width = panel._board.outline.standee_width_mm

    assert width == pytest.approx(16.0)
    assert 0.0 < contact < width, "圆板下边缘可用宽度应远小于默认插舌宽度"
    assert panel._board.standee_error is None, "插舌仍应连在板框上"
    assert not panel.standee_x_slider.isEnabled()
    assert "比下边缘宽" in panel.standee_note.text()
    assert "请缩小宽度" not in panel.standee_note.text()


def test_the_tab_stays_attached_when_the_shape_changes(panel):
    """The tab belongs to the board, not to one of its shapes.

    A circle's lowest point is a single point, so this is also the case that
    would split the board and the tab into two separate pieces.
    """
    select_shape(panel, OutlineShape.RECT)
    tick_standee(panel)

    select_shape(panel, OutlineShape.CIRCLE)

    assert panel._board.outline.standee is True
    assert panel.standee_group.isVisibleTo(panel)
    assert panel._board.standee_error is None
    assert tab_centre_x(panel) == pytest.approx(0.0, abs=1e-6)


def test_a_smaller_board_pulls_the_tab_back_inside_the_edge(panel):
    """Shrinking the board must not leave the slider pointing off the end.

    The offset is stored in millimetres, so a board resized around it can leave
    the number outside the new range; the slider and the geometry would then
    describe two different tabs.
    """
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)
    panel.standee_x_slider.setValue(100)

    panel.width_spin.setValue(18.0)

    travel = panel._board.standee_travel_mm()
    assert travel == pytest.approx((18.0 - 16.0) / 2), "板变窄后余量应只剩 1mm"
    assert abs(panel._board.outline.standee_offset_mm) <= travel + 1e-9
    assert panel.standee_x_slider.value() == 100, "滑块位置应保持在其量程端点上"
    assert panel._board.standee_error is None


def test_switching_the_tab_off_and_on_keeps_its_settings(panel):
    """Unticking 立牌 hides the settings; it does not throw them away.

    The checkbox is a switch, not a reset button, and a user who turns the tab
    off to compare against a plain board expects the tab they tuned to come
    back the way they left it.
    """
    select_shape(panel, OutlineShape.RECT)
    panel.board_aspect_check.setChecked(False)
    panel.width_spin.setValue(50.0)
    panel.height_spin.setValue(30.0)
    tick_standee(panel)
    panel.standee_width_spin.setValue(9.0)
    panel.standee_height_spin.setValue(2.0)
    panel.standee_x_slider.setValue(60)
    before = panel._board.outline.standee_offset_mm

    panel.standee_check.setChecked(False)
    assert panel._board.outline_polygon() and panel._board.size_mm() == pytest.approx(
        (50.0, 30.0)
    ), "关掉立牌后应回到原板框"

    panel.standee_check.setChecked(True)

    assert panel._board.outline.standee_width_mm == pytest.approx(9.0)
    assert panel._board.outline.standee_height_mm == pytest.approx(2.0)
    assert panel._board.outline.standee_offset_mm == pytest.approx(before)


def test_the_status_line_names_the_tab_separately_from_the_board(panel):
    select_shape(panel, OutlineShape.RECT)
    tick_standee(panel)

    text = panel.status_label.text()

    assert "立牌插舌" in text
    assert "16.0 × 3.0 mm" in text


# ------------------------------------------------------- importing a picture


def test_importing_a_picture_puts_it_on_the_canvas_without_a_board(app, tmp_path):
    """The whole point of the rework: artwork first, board on request.

    Checked end to end through the real import path, because the old
    behaviour -- an outline appearing the moment a picture loaded -- is
    exactly the thing being removed. Pressing 导入图片 is what puts the
    picture there now, so both halves of the flow are asserted.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from pcbart.image_prep import imwrite_unicode

    source = tmp_path / "art.png"
    pixels = np.zeros((200, 400, 3), dtype=np.uint8)
    pixels[40:160, 60:340] = 255
    imwrite_unicode(str(source), pixels)

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    try:
        panel = show_preprocess(window)
        assert panel.load_image(str(source)) is True

        board = window.design_canvas.board
        assert board.placements == [], "上传阶段不应写入画布"

        assert panel.import_to_canvas() is True

        assert board.image is not None, "点导入后图片应落到画布上"
        assert board.outline_polygon() == [], "导入图片不得生成板框"
        assert board.outline_ready is False
        # EasyEDA Pro's three-inch import, on the longest edge.
        assert board.image.width_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM)
        assert board.image.height_mm == pytest.approx(DEFAULT_IMAGE_LONG_EDGE_MM / 2)
    finally:
        window.hide()


def test_the_template_entry_is_visible_but_not_selectable(panel):
    """Template files have not been supplied; the option must not pretend."""
    model = panel.shape_combo.model()
    item = model.item(panel.shape_combo.count() - 1)

    assert item is not None and item.text() == BoardSetupPanel.TEMPLATE_OPTION
    assert not item.isEnabled()
    assert not panel.template_combo.isEnabled()


@pytest.mark.parametrize(
    "attribute",
    [
        "width_spin",
        "height_spin",
        "diameter_spin",
        "radius_spin",
        "image_width_spin",
        "image_height_spin",
    ],
)
def test_the_wheel_never_edits_a_value(panel, attribute):
    """Wheel-over-a-field must not change it; that rule is a user request.

    Checked by delivering a real wheel event rather than by reading the code,
    because a control can look protected and still accept the event.
    """
    widget = getattr(panel, attribute)
    before = widget.value()

    wheel(widget)

    assert widget.value() == before
