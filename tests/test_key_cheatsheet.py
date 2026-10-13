"""The keyboard cheat sheet 【帮助 → 快捷键】, and the bindings it teaches.

The page names key sequences and mouse gestures. A row naming a key nothing
listens to would be worse than no cheat sheet at all -- it teaches a gesture
the tool does not have -- so these tests tie the sheet's contents to the
window's real bindings rather than to the strings that happen to be typed
into it. Adding a row means adding the binding first; that rule is checked
mechanically below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QAction,
    QKeyEvent,
    QMouseEvent,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import (  # noqa: E402
    KEY_CHIPS,
    MOUSE_CHIPS,
    build_window,
    new_project_tab,
)
from pcbart.draft import DraftStore  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app, tmp_path):
    win = build_window()
    win._draft_store = DraftStore(tmp_path / "library")
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    yield win
    from conftest import release_window

    release_window(win)


def chip_sequences(entry: str) -> list[str]:
    """Every key sequence a chip names: ``"Ctrl+C / V / X"`` is three.

    Alternatives written as a bare letter inherit the first one's modifiers;
    one that spells out its own sequence is used as it stands.
    """
    parts = [part.strip() for part in entry.split("/")]
    first = parts[0]
    prefix = first.rsplit("+", 1)[0] + "+" if "+" in first else ""
    return [first] + [
        part if "+" in part else prefix + part for part in parts[1:]
    ]


def installed_bindings(window) -> set[str]:
    """Every key sequence the window answers to, from actions and shortcuts.

    ``Ctrl+Tab`` is in neither collection: it is a ``keyPressEvent`` on the
    window, so the caller presses it instead.
    """
    bound = set()
    for action in window.findChildren(QAction):
        sequence = action.shortcut().toString()
        if sequence:
            bound.add(sequence)
    for shortcut in window.findChildren(QShortcut):
        bound.add(shortcut.key().toString())
    return bound


def action_named(window, text: str) -> QAction:
    return next(
        action for action in window.findChildren(QAction) if action.text() == text
    )


def mouse_event(kind, *, button=Qt.LeftButton, buttons=None):
    """The six-argument form: the shorter overload is deprecated in Qt 6."""
    buttons = button if buttons is None else buttons
    position = QPointF(60.0, 60.0)
    return QMouseEvent(
        kind, position, position, button, buttons, Qt.NoModifier
    )


# --------------------------------------------- the sheet only teaches keys
#                                          the window really answers to


def test_every_key_chip_names_a_binding_the_window_installs(window):
    """The mechanical version of "教的键都真实可用"."""
    bound = installed_bindings(window)

    # Ctrl+Tab is handled by the window's own keyPressEvent, so the only proof
    # it exists is that it works: open a second board, press the key, look.
    new_project_tab(window)
    window.project_tabs.setCurrentIndex(0)
    window.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Tab, Qt.ControlModifier)
    )
    assert window.project_tabs.currentIndex() == 1, "Ctrl+Tab 没有切换板子"
    bound.add("Ctrl+Tab")

    missing = [
        (entry, sequence)
        for entry, _label in KEY_CHIPS
        for sequence in chip_sequences(entry)
        if sequence not in bound
    ]
    assert not missing, f"快捷键页教了没有绑定的键：{missing}"


def test_the_help_menu_opens_the_sheet(window, monkeypatch):
    """【帮助 → 快捷键】 shows every row of the sheet.

    The page is built from the same tuples the binding check reads, so this
    also pins the menu entry to the real content: a row added to KEY_CHIPS
    that never reaches the dialog fails here.
    """
    messages: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append(str(args[-1])),
    )

    action_named(window, "快捷键").trigger()

    assert messages
    text = messages[0]
    for key, label in KEY_CHIPS + MOUSE_CHIPS:
        assert f"{key}　{label}" in text, f"快捷键页缺少：{key} {label}"
    assert "快捷键" in text
    assert "鼠标手势" in text


def test_the_sheet_no_longer_sits_on_the_main_window(window):
    """It moved into 帮助, so the window keeps its vertical space back.

    The bar used to be the widget above the tab row; the central layout now
    holds exactly the tab row and the splitter.
    """
    layout = window.centralWidget().layout()
    widgets = [
        layout.itemAt(index).widget() for index in range(layout.count())
    ]
    assert len(widgets) == 2, "主窗口顶层只剩工程栏与工作区"
    assert widgets[0] is not None and widgets[0].isAncestorOf(window.project_tabs)


def test_the_shortcuts_fire_wherever_focus_is(window):
    """``WindowShortcut``, not the menu-focused default.

    The user reads the sheet and presses the key with focus wherever they last
    clicked -- which is never the menu bar.
    """
    for action in window.findChildren(QAction):
        if not action.shortcut().toString():
            continue
        assert action.shortcutContext() == Qt.WindowShortcut, (
            f"「{action.text()}」的快捷键只在有菜单焦点时生效"
        )
    for shortcut in window.findChildren(QShortcut):
        assert shortcut.context() == Qt.WindowShortcut


def test_ctrl_n_opens_a_board_and_ctrl_s_writes_the_draft(window, app):
    """The two chips that are cheap to check end to end.

    The key binding itself is the previous test's business; here the action
    behind it has to do what the chip says.
    """
    boards = window.project_tabs.count()
    action_named(window, "新建板子").trigger()
    app.processEvents()
    assert window.project_tabs.count() == boards + 1, "Ctrl+N 没有新建板子"

    assert window._draft_store.list() == []
    action_named(window, "保存草稿").trigger()
    app.processEvents()
    assert len(window._draft_store.list()) == 1, "Ctrl+S 没有写进草稿库"


# ------------------------------------------ the gestures the second row
#                                         teaches, on the canvas itself


def test_double_clicking_the_canvas_refits_the_view(window, app):
    """「双击画布 · 整板回到视野」— measured by the effect.

    Zoom and pan away, double-click, and the view is back to the fit.
    """
    canvas = window.design_canvas
    canvas.fit_view()
    canvas.zoom_by(3.0)
    canvas.pan_by_pixels(120, 40)
    assert canvas.view_state() != (1.0, 0.0, 0.0)

    canvas.mouseDoubleClickEvent(mouse_event(QEvent.MouseButtonDblClick))
    app.processEvents()

    zoom, pan_x, pan_y = canvas.view_state()
    assert zoom == pytest.approx(1.0), "双击没有回到适应窗口"
    assert pan_x == pytest.approx(0.0)
    assert pan_y == pytest.approx(0.0)


def test_a_middle_double_click_keeps_meaning_pan(window, app):
    """Only the selecting button refits; the middle button stays the pan."""
    canvas = window.design_canvas
    canvas.fit_view()
    canvas.zoom_by(2.0)

    canvas.mouseDoubleClickEvent(
        mouse_event(QEvent.MouseButtonDblClick, button=Qt.MiddleButton)
    )
    app.processEvents()

    assert canvas.view_zoom == pytest.approx(2.0)


def test_the_other_mouse_chips_are_real_gestures(window, app):
    """滚轮缩放 / 中键·空格拖动平移.

    Ctrl+单击 / Shift+单击（加选 / 选一段）needs artwork on the board and is
    covered where that machinery is tested, in ``test_canvas_groups``.
    """
    canvas = window.design_canvas
    canvas.fit_view()

    centre = QPointF(80.0, 80.0)
    canvas.wheelEvent(
        QWheelEvent(
            centre,
            centre,
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
    )
    assert canvas.view_zoom > 1.0, "滚轮没有缩放视图"

    canvas.mousePressEvent(
        mouse_event(QEvent.MouseButtonPress, button=Qt.MiddleButton)
    )
    assert canvas._mode == "pan", "中键按下没有进入平移"
    canvas.mouseReleaseEvent(
        mouse_event(QEvent.MouseButtonRelease, button=Qt.MiddleButton)
    )
    assert canvas._mode != "pan"

    canvas.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
    )
    canvas.mousePressEvent(mouse_event(QEvent.MouseButtonPress))
    assert canvas._mode == "pan", "空格+左键没有进入平移"
    canvas.mouseReleaseEvent(mouse_event(QEvent.MouseButtonRelease))
    canvas.keyReleaseEvent(
        QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier)
    )
