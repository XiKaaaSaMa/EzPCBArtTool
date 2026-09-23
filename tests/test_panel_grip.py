"""Tests for the resize grip on every floating panel.

The panels have always been resizable from all eight edges -- but only from a
6 px border, and the sole hint was the cursor changing. Users reported not
finding it, so the visible corner grip is a discoverability fix, and these
tests hold the two halves of that: the grip is *there* everywhere it should be,
and dragging it does what it looks like it does.

Tested through real mouse events rather than by calling the resize methods,
because the failure this guards against is the wiring: a grip that paints
correctly and does not move the panel would pass any check that only looked at
its appearance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import (  # noqa: E402
    FloatingPanel,
    PreviewPanel,
    ResizeGrip,
    build_window,
    show_board_setup,
    show_preprocess,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app):
    win = build_window()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    yield win
    win.hide()


@pytest.fixture
def panels(window, app):
    """Every floating panel the application can open."""
    made = [
        show_preprocess(window),
        show_board_setup(window),
        PreviewPanel(window),
    ]
    for panel in made:
        panel.show()
        panel.place_top_right()
    app.processEvents()
    yield made
    for panel in made:
        panel.hide()


def drag_grip(app, panel, dx: int, dy: int) -> None:
    """Drag the corner grip by ``dx``/``dy``, as a person would."""
    grip = panel.resize_grip
    start = grip.mapToGlobal(grip.rect().center())
    events = (
        (QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
        (QEvent.MouseMove, start + QPoint(dx, dy), Qt.NoButton, Qt.LeftButton),
        (QEvent.MouseButtonRelease, start + QPoint(dx, dy), Qt.LeftButton, Qt.NoButton),
    )
    for kind, pos, button, buttons in events:
        event = QMouseEvent(
            kind, QPointF(grip.rect().center()), QPointF(pos), button, buttons, Qt.NoModifier
        )
        app.sendEvent(grip, event)
    app.processEvents()


# ------------------------------------------------------------- it is there


def test_every_floating_panel_has_a_grip(panels):
    """All of them, because "some panels resize and others do not" is exactly
    the kind of inconsistency that makes a user stop trying."""
    for panel in panels:
        grip = panel.resize_grip
        assert isinstance(grip, ResizeGrip), type(panel).__name__
        assert grip.isVisibleTo(panel), f"{type(panel).__name__} 没有显示缩放手柄"


def test_the_grip_sits_in_the_bottom_right_corner(panels):
    for panel in panels:
        grip = panel.resize_grip
        centre = grip.geometry().center()
        assert panel.rect().contains(centre), f"{type(panel).__name__} 手柄跑到面板外"
        # Bottom-right means past the middle on both axes.
        assert centre.x() > panel.width() / 2
        assert centre.y() > panel.height() / 2


def test_the_grip_shows_the_diagonal_resize_cursor(panels):
    for panel in panels:
        assert panel.resize_grip.cursor().shape() == Qt.SizeFDiagCursor


def test_the_grip_reports_a_tooltip(panels):
    for panel in panels:
        assert "拖动" in panel.resize_grip.toolTip()


def test_the_grip_is_not_covered_by_the_widget_under_it(panels, app):
    """It has to be the widget actually hit at its own centre.

    The preprocessing panel puts a full-width button in its footer; a grip
    merely painted on top of that would swallow clicks meant for the button,
    and one painted underneath would never be draggable.
    """
    for panel in panels:
        grip = panel.resize_grip
        hit = panel.childAt(grip.geometry().center())
        assert hit is grip, f"{type(panel).__name__} 的手柄被 {hit} 挡住"


def test_the_grip_does_not_overlap_any_button(panels):
    """It must not sit on a control the user is trying to click."""
    from PySide6.QtCore import QRect

    for panel in panels:
        grip_rect = panel.resize_grip.geometry()
        for button in panel.findChildren(QPushButton):
            if not button.isVisibleTo(panel):
                continue
            top_left = button.mapTo(panel, button.rect().topLeft())
            button_rect = QRect(top_left, button.size())
            assert not button_rect.intersects(grip_rect), (
                f"{type(panel).__name__} 的手柄压住了「{button.text()}」"
            )


# ------------------------------------------------------------ it resizes


def test_dragging_the_grip_resizes_the_panel(window, app, panels):
    """The whole point: it does what it looks like it does."""
    panel = panels[0]
    panel.move(40, 40)  # room to grow in every direction
    app.processEvents()
    before = (panel.width(), panel.height())

    drag_grip(app, panel, 70, 45)

    assert panel.width() == pytest.approx(before[0] + 70, abs=3)
    assert panel.height() == pytest.approx(before[1] + 45, abs=3)


def test_the_grip_and_the_corner_agree(window, app, panels):
    """One resize implementation, so the visible handle cannot drift from the
    invisible 6 px corner it sits on. The same drag has to land on the same
    geometry either way."""
    by_grip = panels[0]
    by_grip.move(40, 40)
    app.processEvents()
    grip_before = (by_grip.width(), by_grip.height())
    drag_grip(app, by_grip, 60, 40)
    grip_delta = (by_grip.width() - grip_before[0], by_grip.height() - grip_before[1])

    by_edge = panels[1]
    by_edge.move(40, 40)
    app.processEvents()
    edge_before = (by_edge.width(), by_edge.height())
    corner = QPoint(by_edge.width() - 2, by_edge.height() - 2)
    by_edge.begin_resize("se", by_edge.mapToGlobal(corner))
    by_edge.apply_resize(by_edge.mapToGlobal(corner) + QPoint(60, 40))
    by_edge.end_resize()
    app.processEvents()
    edge_delta = (by_edge.width() - edge_before[0], by_edge.height() - edge_before[1])

    assert grip_delta == pytest.approx(edge_delta, abs=3)


def test_the_minimum_size_still_holds(window, app, panels):
    """Dragging inward past the floor must not crush the panel."""
    panel = panels[0]
    panel.move(40, 40)
    app.processEvents()

    drag_grip(app, panel, -900, -900)

    assert panel.width() == FloatingPanel.MIN_WIDTH
    assert panel.height() == FloatingPanel.MIN_HEIGHT


def test_the_panel_cannot_be_dragged_out_of_the_window(window, app, panels):
    """The panel is an in-app panel; growing it past the window is not allowed."""
    panel = panels[0]
    panel.move(window.width() - panel.width() - 12, window.height() - panel.height() - 12)
    app.processEvents()

    drag_grip(app, panel, 400, 400)

    assert panel.x() + panel.width() <= window.width()
    assert panel.y() + panel.height() <= window.height()


def test_the_grip_can_shrink_as_well_as_grow(window, app, panels):
    """A double-headed arrow promises both directions."""
    panel = panels[0]
    panel.move(40, 40)
    app.processEvents()
    drag_grip(app, panel, 80, 60)
    grown = (panel.width(), panel.height())

    drag_grip(app, panel, -40, -30)

    assert panel.width() < grown[0]
    assert panel.height() < grown[1]


# -------------------------------------------------------------- its states


def test_the_grip_is_hidden_when_collapsed(panels, app):
    """Collapsed to the title bar there is no corner to drag."""
    for panel in panels:
        panel.collapse()
        app.processEvents()
        assert not panel.resize_grip.isVisibleTo(panel), type(panel).__name__
        panel.expand()
        app.processEvents()
        assert panel.resize_grip.isVisibleTo(panel), type(panel).__name__


def test_the_grip_is_hidden_when_maximised(panels, app):
    """A maximised panel fills the window and its edges are disabled, so the
    grip would be a handle that does nothing."""
    for panel in panels:
        panel.maximise()
        app.processEvents()
        assert not panel.resize_grip.isVisibleTo(panel), type(panel).__name__
        panel.restore()
        app.processEvents()
        assert panel.resize_grip.isVisibleTo(panel), type(panel).__name__


def test_the_grip_follows_the_corner_when_the_panel_resizes(panels, app):
    panel = panels[0]
    panel.move(40, 40)
    panel.resize(500, 600)
    app.processEvents()

    centre = panel.resize_grip.geometry().center()

    assert centre.x() == pytest.approx(panel.width() - FloatingPanel.GRIP_SIZE // 2, abs=4)
    assert centre.y() == pytest.approx(panel.height() - FloatingPanel.GRIP_SIZE // 2, abs=4)


def test_the_grip_sits_below_the_content(window, app):
    """The strip is reserved by the layout, so the content stops above it.

    Without the reservation the grip would sit over the bottom 16 px of the
    content, which on a short panel means over the last visible control.
    """
    panel = show_preprocess(window)
    panel.show()
    app.processEvents()

    scroll_bottom = panel._scroll.mapTo(panel, panel._scroll.rect().bottomLeft()).y()
    footer_top = panel._footer.mapTo(panel, panel._footer.rect().topLeft()).y()
    grip_top = panel.resize_grip.geometry().top()

    lowest_content = max(scroll_bottom, footer_top)
    assert grip_top >= lowest_content, "手柄不应压在内容上"


# -------------------------------------------------------- all eight handles


def test_every_edge_and_corner_has_its_own_handle(panels):
    """Eight of them: four edges plus four corners.

    The previous version had a single corner grip, and the right and top edges
    were not reachable at all -- the scroll bar sits on the right border and
    the title bar on the top one, so presses there never reached the panel.
    """
    for panel in panels:
        zones = {grip.zone for grip in panel.grips()}
        assert zones == set(ResizeGrip.ZONES), type(panel).__name__


def test_every_handle_is_hit_at_its_own_position(panels, app):
    """Each handle must be the widget actually hit where it is drawn.

    Anything else on top -- the scroll bar on the right border, the title bar
    on the top one -- means that edge cannot be dragged however good it looks.
    """
    for panel in panels:
        for grip in panel.grips():
            hit = panel.childAt(grip.geometry().center())
            assert hit is grip, (
                f"{type(panel).__name__} 的 {grip.zone} 手柄被 {hit} 挡住"
            )


@pytest.mark.parametrize(
    "zone, axis, sign",
    [
        ("e", "w", 1),
        ("w", "w", -1),
        ("s", "h", 1),
        ("n", "h", -1),
    ],
)
def test_an_edge_handle_changes_only_its_own_axis(window, app, panels, zone, axis, sign):
    """A straight edge promises one direction, so it must not move the other."""
    panel = panels[0]
    panel.move(80, 80)
    panel.resize(420, 560)
    app.processEvents()
    before = (panel.width(), panel.height())

    drag_zone(app, panel, zone, 60 * sign, 0 if axis == "w" else 60 * sign)

    width, height = panel.width(), panel.height()
    if axis == "w":
        assert width == pytest.approx(before[0] + 60, abs=3), "宽度应改变"
        assert height == pytest.approx(before[1], abs=3), "高度不该变"
    else:
        assert height == pytest.approx(before[1] + 60, abs=3), "高度应改变"
        assert width == pytest.approx(before[0], abs=3), "宽度不该变"


@pytest.mark.parametrize("zone", ["ne", "nw", "se", "sw"])
def test_a_corner_handle_changes_both_axes(window, app, panels, zone):
    panel = panels[0]
    panel.move(80, 80)
    panel.resize(420, 560)
    app.processEvents()

    dx = 55 if "e" in zone else -55
    dy = 45 if "s" in zone else -45
    before = (panel.width(), panel.height(), panel.x(), panel.y())

    drag_zone(app, panel, zone, dx, dy)

    assert panel.width() == pytest.approx(before[0] + abs(dx), abs=4)
    assert panel.height() == pytest.approx(before[1] + abs(dy), abs=4)
    # A west or north corner drags the origin too; the opposite edge must stay.
    if "w" in zone:
        assert panel.x() != before[2]
    else:
        assert panel.x() == before[2]
    if "n" in zone:
        assert panel.y() != before[3]
    else:
        assert panel.y() == before[3]


def test_each_handle_shows_the_cursor_for_its_direction(panels):
    """The cursor is half the hint; a vertical edge must not offer diagonal."""
    expected = {
        "n": Qt.SizeVerCursor,
        "s": Qt.SizeVerCursor,
        "e": Qt.SizeHorCursor,
        "w": Qt.SizeHorCursor,
        "ne": Qt.SizeBDiagCursor,
        "sw": Qt.SizeBDiagCursor,
        "nw": Qt.SizeFDiagCursor,
        "se": Qt.SizeFDiagCursor,
    }
    for panel in panels:
        for grip in panel.grips():
            assert grip.cursor().shape() == expected[grip.zone], grip.zone


def test_the_corner_arrows_mirror_so_they_point_outwards(panels):
    """The glyph has to point the way that corner grows.

    An identical arrow at all four corners points the wrong way on half of
    them, which is worse than no arrow: it tells the user to drag inwards.
    """
    panel = panels[0]
    images = {}
    for zone in ("nw", "ne", "sw", "se"):
        images[zone] = panel._grips[zone].grab().toImage()

    # East and west corners are horizontal mirrors of each other; north and
    # south are vertical mirrors. Comparing pixels is what catches a glyph that
    # was simply copied to all four.
    assert images["nw"] != images["ne"], "左右角箭头应互为镜像"
    assert images["nw"] != images["sw"], "上下角箭头应互为镜像"


def test_the_border_strip_keeps_the_handles_off_the_content(window, app):
    """No handle may sit on a control: it would swallow the click.

    The preprocessing panel's 导入图片 button is at its bottom-right, which is
    exactly where a floating grip would land.
    """
    from PySide6.QtCore import QRect

    for panel in panels_of(window):
        panel.show()
        app.processEvents()
        for grip in panel.grips():
            grip_rect = grip.geometry()
            for button in panel.findChildren(QPushButton):
                if not button.isVisibleTo(panel):
                    continue
                top_left = button.mapTo(panel, button.rect().topLeft())
                assert not QRect(top_left, button.size()).intersects(grip_rect), (
                    f"{type(panel).__name__} 的 {grip.zone} 手柄压住了"
                    f"「{button.text()}」"
                )


def panels_of(window):
    """Every floating panel, freshly opened."""
    made = [show_preprocess(window), show_board_setup(window), PreviewPanel(window)]
    for panel in made:
        panel.show()
        panel.place_top_right()
    return made


def drag_zone(app, panel, zone, dx: int, dy: int) -> None:
    """Drag one named handle, as a person would."""
    grip = panel._grips[zone]
    start = grip.mapToGlobal(grip.rect().center())
    events = (
        (QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
        (QEvent.MouseMove, start + QPoint(dx, dy), Qt.NoButton, Qt.LeftButton),
        (QEvent.MouseButtonRelease, start + QPoint(dx, dy), Qt.LeftButton, Qt.NoButton),
    )
    for kind, pos, button, buttons in events:
        event = QMouseEvent(
            kind, QPointF(grip.rect().center()), QPointF(pos), button, buttons, Qt.NoModifier
        )
        app.sendEvent(grip, event)
    app.processEvents()
