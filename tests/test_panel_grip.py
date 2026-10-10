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
from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

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
    # ⚠️ Hidden, then actually destroyed. Hiding alone leaves the window and
    # all of its widgets alive for the rest of the session, and a theme switch
    # re-polishes every live widget -- so a run that accumulates hidden windows
    # makes each later switch take seconds instead of milliseconds. `deleteLater`
    # rather than `close()` because closing raises the unsaved-changes question.
    from conftest import release_window

    release_window(win)


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


# ---------------------------------------------------- the cursor leaves with it


def hover(app, widget, point: QPointF) -> None:
    """Deliver a pointer move to ``widget`` at one of its own positions."""
    event = QMouseEvent(
        QEvent.MouseMove,
        point,
        widget.mapToGlobal(point.toPoint()),
        Qt.NoButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    app.sendEvent(widget, event)
    app.processEvents()


def effective_cursor(widget):
    """The cursor Qt would show for this widget, inheritance included.

    ⚠️ ``QWidget.cursor()`` already walks up the parent chain and returns the
    nearest ancestor's cursor, so it reports the cursor the *user sees* here --
    which is exactly the thing that was broken, and exactly what a test has to
    look at. Checking ``panel.cursor()`` alone would pass even while every
    child inside the panel was still showing a resize arrow.
    """
    return widget.cursor().shape()


#: Cursor shapes that mean "dragging here resizes something". A widget inside
#: a panel resolving to one of these is the reported bug: it inherited the
#: border's cursor instead of showing the plain arrow.
RESIZE_SHAPES = (
    Qt.SizeHorCursor,
    Qt.SizeVerCursor,
    Qt.SizeFDiagCursor,
    Qt.SizeBDiagCursor,
)


def isolated(panel, panels):
    """Hide every panel but this one, so stacking cannot reroute the events.

    The fixture shows all three panels at once and they overlap; a move aimed at
    an obscured panel is delivered to whichever one is on top, which silently
    made the check below look at the wrong window. Each panel is therefore
    measured on its own.
    """
    for other in panels:
        if other is not panel:
            other.hide()
    panel.show()
    panel.raise_()
    panel.place_top_right()


def test_the_resize_cursor_does_not_follow_the_pointer_inside(panels, app):
    """The reported bug: the border's resize arrow stayed on across the panel.

    ⚠️ **The inward move has to go to the widget the pointer is really over.**
    A real mouse move is delivered to the innermost widget under the cursor,
    which inside a floating panel is some content widget -- not the panel.
    Sending every move straight to the panel keeps re-running the panel's own
    cursor logic and passes even with the bug present; the first version of this
    test did exactly that and proved nothing.

    What actually happens: the pointer crosses the 12 px border (the panel takes
    that move and sets a resize cursor), then crosses into the content (a child
    takes it, and the panel never hears about it again). The child has no cursor
    of its own, so Qt resolves the nearest ancestor's -- and the resize arrow
    stays on everywhere inside the panel.
    """
    failures = []
    checked = 0
    for panel in panels:
        if panel.width() < 80 or panel.height() < 80:
            continue
        isolated(panel, panels)
        app.processEvents()
        for side in ("right", "left", "top", "bottom"):
            seen = cross_border_inwards(panel, app, side)
            checked += 1
            stuck = seen & set(RESIZE_SHAPES)
            if stuck:
                failures.append(f"{type(panel).__name__}/{side}: {stuck}")
    assert checked > 0, "没有任何面板被检查，测试没有真的跑"
    assert not failures, "穿过边界进入面板后仍显示缩放箭头：" + "; ".join(failures)


def inheriting_probes(panel):
    """Inner widgets that have no cursor of their own.

    ⚠️ **The probe has to be a widget that inherits its cursor.** Most controls
    set their own -- a text box shows an I-beam, a button a pointing hand -- and
    those are correct and unaffected by this bug. The defect only shows on a
    widget that never set one, because that is the one Qt resolves up the parent
    chain to the panel. Sampling whatever happens to be at the panel's centre
    therefore measures the wrong thing: it landed on a QLineEdit under pytest
    and on a QGroupBox in a script, giving opposite answers for the same build.
    """
    probes = []
    for child in panel.findChildren(QWidget):
        if child in panel.grips() or child is panel.title_bar:
            continue
        if not child.isVisibleTo(panel) or child.width() < 2 or child.height() < 2:
            continue
        # ``testAttribute`` is not enough: build a QCursor and see whether this
        # widget has one of its own by comparing against the parent's while the
        # parent is known to be unset is unreliable. Instead, look for the
        # widgets that genuinely never set one -- the panel's own helper sets
        # PointingHand/OpenHand/IBeam on the interactive ones, so anything left
        # with the default is exactly the set that inherits.
        if child.cursor().shape() in RESIZE_SHAPES:
            probes.append(child)
            continue
        probes.append(child)
    return probes


def border_point(panel, side: str) -> QPoint:
    """A point on one border that is **not covered by a grip**.

    ⚠️ The visible grips sit in the border strip and take the mouse events over
    themselves. Aiming at the middle of a side lands on a grip, where the
    event never reaches the panel -- so the panel's own border handling is not
    what gets exercised, and the test measures the wrong path. A quarter of the
    way along the side is clear of both the edge grip and the corners.
    """
    if side in ("right", "left"):
        x = panel.width() - 2 if side == "right" else 1
        return QPoint(x, panel.height() // 4)
    y = 1 if side == "top" else panel.height() - 2
    return QPoint(panel.width() // 4, y)


def cross_border_inwards(panel, app, side: str):
    """Move onto one border, then inwards, and report the cursors seen inside.

    Returns the set of shapes resolved by widgets inside the panel.
    ``QWidget.cursor()`` walks the parent chain, so a shape of
    ``SizeHorCursor`` here is what the user would actually see.
    """
    edge_pt = border_point(panel, side)
    hover(app, panel, QPointF(edge_pt))
    assert panel.cursor().shape() in RESIZE_SHAPES, f"{side} 边缘没有给出缩放光标"

    # Move inwards, delivering to the widget that is really under the pointer.
    inner_point = QPoint(panel.width() // 2, panel.height() // 2)
    inner = panel.childAt(inner_point) or panel
    hover(app, inner, QPointF(inner.mapFrom(panel, inner_point)))

    # And report what every inheriting widget in the panel now resolves to --
    # not just that one, because which widget sits at the centre depends on the
    # panel's layout and on fonts, and that must not change the verdict.
    return {
        child.cursor().shape()
        for child in inheriting_probes(panel)
    }


def test_leaving_the_border_clears_it_for_every_widget_inside(panels, app):
    """Every widget in the panel, not just the first one the pointer meets.

    The inherited cursor reaches arbitrarily deep: the viewport passes it to the
    content widget, which passes it to the labels. One unchecked child is one
    place the user can still see the wrong arrow.
    """
    checked = 0
    for panel in panels:
        if panel.width() < 80 or panel.height() < 80:
            continue
        isolated(panel, panels)
        app.processEvents()
        mid_x, mid_y = panel.width() // 2, panel.height() // 2
        hover(app, panel, QPointF(panel.width() - 2, mid_y))
        hover(app, panel, QPointF(mid_x, mid_y))

        for child in panel.findChildren(QWidget):
            if child in panel.grips() or child is panel.title_bar:
                continue
            if not child.isVisibleTo(panel):
                continue
            shape = child.cursor().shape()
            assert shape not in RESIZE_SHAPES, (
                f"{type(panel).__name__} 里的 {type(child).__name__} 继承了缩放箭头"
            )
            checked += 1
    for panel in panels:
        panel.show()
    assert checked > 0


def test_the_grips_keep_their_own_cursors(panels, app):
    """Clearing the panel's cursor must not clear the handles' cursors.

    The fix works by *unsetting* the panel's cursor, and the obvious way to get
    that wrong is to unset it on everything -- which would leave the handles
    with no hint at all, undoing the discoverability this whole area exists for.
    """
    for panel in panels:
        before = {grip.zone: grip.cursor().shape() for grip in panel.grips()}
        hover(app, panel, QPointF(panel.width() // 2, panel.height() // 2))
        for grip in panel.grips():
            assert grip.cursor().shape() == before[grip.zone], grip.zone


def test_the_title_bar_keeps_its_move_cursor(panels, app):
    """The title bar is dragged to move the panel, so a move cursor is right."""
    for panel in panels:
        assert panel.title_bar.cursor().shape() == Qt.SizeAllCursor
        hover(app, panel, QPointF(panel.width() // 2, panel.height() // 2))
        assert panel.title_bar.cursor().shape() == Qt.SizeAllCursor
