"""The canvas side of grouping: picking several, and moving them as one.

The model and the file format are covered in ``test_groups``; this is the part
the user actually touches -- Ctrl-click to choose, Shift-click for a range, and
a drag that carries the whole group rather than one member.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from pcbart.artboard import ArtBoard, ImagePlacement, TextElement  # noqa: E402
from pcbart.design_layers import layer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def canvas(app, tmp_path):
    from tools.design_canvas import DesignCanvas

    widget = DesignCanvas()
    widget.resize(800, 600)
    widget._test_drafts = tmp_path
    board = ArtBoard()
    for index, (x, y) in enumerate(((0.0, 0.0), (30.0, 0.0), (0.0, 30.0), (30.0, 30.0))):
        board.add_placement(
            ImagePlacement(
                path="",
                x_mm=x,
                y_mm=y,
                width_mm=12.0,
                height_mm=12.0,
                layer=layer("top_silk").name,
            )
        )
    widget.board = board
    widget.selected = board.placements[0]
    widget.refit_view()
    yield widget
    widget.deleteLater()


def click(canvas, item, *, ctrl: bool = False, shift: bool = False):
    """A real press+release on ``item``, so the handler path is exercised.

    ⚠️ Sent to the canvas and converted through ``to_screen``, not by calling
    the handler with fabricated millimetres: the click has to go through the
    same coordinate conversion the user's pointer does, or a sign error in it
    would pass here and move the wrong object on screen.
    """
    point = canvas.to_screen(item.x_mm, item.y_mm)
    modifiers = Qt.NoModifier
    if ctrl:
        modifiers |= Qt.ControlModifier
    if shift:
        modifiers |= Qt.ShiftModifier
    for kind in (QMouseEvent.MouseButtonPress, QMouseEvent.MouseButtonRelease):
        event = QMouseEvent(
            kind,
            QPointF(point),
            Qt.LeftButton,
            Qt.LeftButton,
            modifiers,
        )
        if kind == QMouseEvent.MouseButtonPress:
            canvas.mousePressEvent(event)
        else:
            canvas.mouseReleaseEvent(event)


# --------------------------------------------------------------- picking


def test_ctrl_click_builds_a_multi_selection(canvas):
    """The user asked to "自由选择组合的元素", and this is the pick."""
    first, second, third = canvas.board.placements[:3]

    click(canvas, first)
    click(canvas, second, ctrl=True)
    click(canvas, third, ctrl=True)

    chosen = canvas.selection_set()
    assert chosen[0] is first, "第一个点击的是主选"
    # Compared by identity, not as a set: ImagePlacement is a dataclass with
    # `eq=True`, so it is unhashable -- and a membership test on it would also
    # compare by value, which is the mistake the canvas itself avoids.
    assert len(chosen) == 3
    assert all(
        any(item is expected for expected in (first, second, third))
        for item in chosen
    )


def test_ctrl_click_on_the_primary_removes_it(canvas):
    first, second = canvas.board.placements[:2]
    click(canvas, first)
    click(canvas, second, ctrl=True)

    click(canvas, first, ctrl=True)

    assert canvas.selected is second, "主选被移除后应当有一个顶上"
    assert canvas.selection_set() == [second]


def test_a_plain_click_clears_the_extra_selection(canvas):
    """Otherwise a half-built selection would follow the user around: picking
    three things and then clicking a fourth would leave the first three marked."""
    first, second, third = canvas.board.placements[:3]
    click(canvas, first)
    click(canvas, second, ctrl=True)

    click(canvas, third)

    assert canvas.selection_set() == [third]


def test_shift_click_selects_the_range_between(canvas):
    first = canvas.board.placements[0]
    third = canvas.board.placements[2]

    click(canvas, first)
    click(canvas, third, shift=True)

    chosen = canvas.selection_set()
    expected = canvas.board.placements[:3]
    assert len(chosen) == len(expected)
    assert all(
        any(item is want for want in expected) for item in chosen
    )


def test_the_selection_is_in_document_order(canvas):
    """A group's member list must not depend on which order the user clicked,
    or the same two clicks would produce different files."""
    first, second, third = canvas.board.placements[:3]
    click(canvas, third)
    click(canvas, first, ctrl=True)
    click(canvas, second, ctrl=True)

    assert canvas.selection_set() == [first, second, third]


def test_selection_set_does_not_collapse_identical_objects(canvas):
    """Two placements with identical geometry are still two objects. Comparing
    with ``==`` -- which a dataclass provides -- would merge them and build a
    group with fewer members than the user picked."""
    import copy

    twin = copy.copy(canvas.board.placements[0])
    canvas.board.add_placement(twin)
    canvas.selected = canvas.board.placements[0]
    canvas.extra_selection = [twin]

    assert len(canvas.selection_set()) == 2


# -------------------------------------------------------------- dragging


def test_a_drag_moves_every_selected_object(canvas):
    first, second = canvas.board.placements[:2]
    click(canvas, first)
    click(canvas, second, ctrl=True)

    start = canvas.to_screen(first.x_mm, first.y_mm)
    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonPress,
            QPointF(start),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    before = [(item.x_mm, item.y_mm) for item in (first, second)]
    canvas.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.MouseMove,
            QPointF(start.x() + 40, start.y() + 20),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    canvas.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonRelease,
            QPointF(start.x() + 40, start.y() + 20),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )

    moved = [(item.x_mm, item.y_mm) for item in (first, second)]
    assert moved != before
    dx = moved[0][0] - before[0][0]
    dy = moved[0][1] - before[0][1]
    assert (moved[1][0] - before[1][0]) == pytest.approx(dx)
    assert (moved[1][1] - before[1][1]) == pytest.approx(dy), (
        "两个元素必须一起走、位移相同"
    )


def press(canvas, item):
    """Begin a drag on ``item``; returns the screen point it started from."""
    start = canvas.to_screen(item.x_mm, item.y_mm)
    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonPress,
            QPointF(start),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    return start


def move_to(canvas, point, *, release: bool = False):
    for kind, buttons in (
        [
            (QMouseEvent.MouseMove, Qt.LeftButton),
            (QMouseEvent.MouseButtonRelease, Qt.LeftButton),
        ]
        if release
        else [(QMouseEvent.MouseMove, Qt.LeftButton)]
    ):
        canvas.mouseMoveEvent(
            QMouseEvent(
                kind,
                QPointF(point),
                Qt.NoButton if kind == QMouseEvent.MouseMove else Qt.LeftButton,
                buttons,
                Qt.NoModifier,
            )
        )


def test_a_drag_is_absolute_so_it_cannot_drift(canvas):
    """⚠️ Each object is placed at *its own* press position plus the total delta.

    Nudging by a per-event delta accumulates every rounding error the pointer
    produces, so a thousand small moves land somewhere the cursor is not. The
    single-target version of this code was absolute for that reason; making it
    multi-target must not quietly change it.
    """
    first, second = canvas.board.placements[:2]
    canvas.board.make_group([first, second], name="甲")
    canvas.selected = first
    canvas._selection_anchor = first

    start = press(canvas, first)
    origin = first.x_mm
    step = 2
    for index in range(1, 21):
        move_to(canvas, start + QPointF(step * index, 0.0))

    expected = first.x_mm
    # One jump of the same total distance must give the same answer as twenty
    # small ones -- that equality *is* "no accumulation".
    first.x_mm = origin
    move_to(canvas, start + QPointF(step * 20, 0.0), release=True)
    assert first.x_mm == pytest.approx(expected, abs=1e-9), (
        f"分 20 次走和一次走完必须落在同一处：{expected:.9f} vs {first.x_mm:.9f}"
    )


def test_dragging_a_group_member_carries_the_other(canvas):
    """Moving one half of a 板画 away from the other is the failure the group
    exists to prevent, and a drag is the easiest way to cause it."""
    first, second = canvas.board.placements[:2]
    others = canvas.board.placements[2:]
    canvas.board.make_group([first, second], name="甲")
    canvas.selected = first
    canvas.extra_selection = []
    canvas._selection_anchor = first

    targets = canvas._drag_targets()

    assert first in targets and second in targets
    assert not any(item in targets for item in others), "别的元素不该被带上"


def test_a_group_containing_text_drags_its_wording_too(canvas):
    """The user picked the text, so it moves with the group."""
    image = canvas.board.placements[0]
    text = TextElement(text="标题", x_mm=40.0, y_mm=40.0)
    canvas.board.texts = [text]
    assert canvas.board.make_group([image, text], name="图和字") is not None
    canvas.selected = image

    targets = canvas._drag_targets()

    assert image in targets and text in targets




# ---------------------------------------------------------------- the menu


def test_the_edit_menu_offers_the_group_commands():
    """A capability the user asked for by name has to be reachable."""
    from gui_mockup import EDIT_ACTIONS

    labels = [row[0] for row in EDIT_ACTIONS]
    for wanted in ("组合所选…", "取消组合", "组合管理…"):
        assert wanted in labels, f"编辑菜单缺少「{wanted}」"


def test_the_group_entries_are_greyed_out_when_they_would_do_nothing(app, tmp_path):
    """⚠️ 组合 needs *two* objects, so it cannot ride on the generic
    needs-a-selection set -- one object satisfies that and would leave the entry
    looking available for an operation that then does nothing."""
    import gui_mockup

    window = gui_mockup.build_window()
    from PySide6.QtCore import Qt as _Qt

    window.setAttribute(_Qt.WA_DontShowOnScreen, True)
    try:
        window.resize(1200, 800)
        window.show()
        app.processEvents()
        canvas = window.design_canvas
        board = canvas.board
        board.placements.clear()

        from pcbart.design_layers import layer as _layer

        first = ImagePlacement(
            path="", width_mm=10.0, height_mm=10.0, layer=_layer("top_silk").name
        )
        second = ImagePlacement(
            path="", x_mm=20.0, width_mm=10.0, height_mm=10.0,
            layer=_layer("top_silk").name,
        )
        board.placements.extend([first, second])

        actions = _menu_actions(window)
        canvas.select(None)
        assert not actions["组合所选…"].isEnabled(), "没有选中时不能组合"
        assert not actions["取消组合"].isEnabled()

        canvas.select(first)
        assert not actions["组合所选…"].isEnabled(), "只选一个时不能组合"
        assert not actions["取消组合"].isEnabled(), "不在组合里时不该能取消组合"

        canvas.extra_selection = [second]
        canvas.can_group()
        canvas.selection_changed.emit(first)
        assert actions["组合所选…"].isEnabled(), "选两个后应当可以组合"

        board.make_group([first, second], name="甲")
        canvas.selection_changed.emit(first)
        assert actions["取消组合"].isEnabled(), "在组合里时应当可以取消组合"
        assert actions["组合管理…"].isEnabled(), "有组合时管理入口可用"
    finally:
        from conftest import release_window

        release_window(window)


def _menu_actions(window) -> dict:
    """The 编辑 menu's actions, keyed by label.

    Read from the menu itself rather than from a list of names, so a label
    that was never added to the visible menu is caught -- an action that exists
    only in the wiring table is not a command the user can reach.
    """
    from PySide6.QtGui import QAction

    actions: dict[str, QAction] = {}
    for menu in window.menuBar().findChildren(type(window.menuBar().addMenu("temp"))):
        if menu.title() != "编辑":
            continue
        for action in menu.actions():
            if action.text():
                actions[action.text()] = action
    return actions
