"""Tests for the project tab row: several boards open at once.

Driven through the real window, because the thing that can actually go wrong
here is a pane keeping its reference to the board that was just switched away
from. A model test cannot catch that -- it only shows up as a panel editing a
document nobody is looking at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QToolButton  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import (  # noqa: E402
    build_window,
    close_project_tab,
    new_project_tab,
    refresh_project_tabs,
)
from pcbart.artboard import BoardOutlineSpec, OutlineShape  # noqa: E402
from pcbart.draft import DraftStore  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def no_modal_can_block(monkeypatch):
    """Every dialog is answered, and an unexpected one fails the test.

    ``QMessageBox.exec`` blocks until someone clicks it. A test that reaches
    one without saying so does not fail -- it hangs, and on this machine it
    puts a real dialog on the user's desktop. Answering "取消" here makes the
    safe outcome the default, so a half-written test cannot trap the run.
    """
    opened: list[str] = []

    def answer(box, *args, **kwargs):
        opened.append(box.windowTitle())
        buttons = {button.text(): button for button in box.buttons()}
        monkeypatch.setattr(
            box, "clickedButton", lambda: buttons.get("取消") or None
        )
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)
    return opened


@pytest.fixture
def window(app, tmp_path):
    win = build_window()
    win._draft_store = DraftStore(tmp_path / "library")
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    refresh_project_tabs(win)
    yield win
    # Closing would open the unsaved-changes question; hiding avoids the modal.
    win.hide()


def size_board(window, width: float) -> None:
    board = window.design_canvas.board
    board.outline_ready = True
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=width, height_mm=30.0
    )


def save_board(window, name: str) -> None:
    """Write the active board's draft, so it counts as saved from here on."""
    store = window._draft_store
    assert gui_mockup.save_draft_to(window, str(store.path_for(name))) is True


# ------------------------------------------------------------- the tab row


def test_the_fixtures_point_the_library_at_a_scratch_folder(window, tmp_path):
    """The library has to be redirected in *every* GUI test file.

    Saving a product file also stashes the editing session into the draft
    library, and the library defaults to the user's real one. A test that
    builds a real window without this override writes into the developer's
    own sidebar -- which is what happened, leaving a dozen empty rows behind.
    """
    from pcbart.draft import DraftStore

    assert window._draft_store.root == tmp_path / "library"
    assert window._draft_store.root != DraftStore().root


def test_the_tab_row_starts_with_one_board(window):
    assert window.project_tabs.count() == 1
    assert window.project_tabs.currentIndex() == 0
    assert len(window.workspace) == 1


def test_the_tab_row_sits_between_the_banner_and_the_work_area(window):
    """The user asked for one row of tabs above the canvas: switching boards
    has to be reachable without hunting through a menu."""
    layout = window.centralWidget().layout()
    widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
    tabs = window.project_tabs
    # The bar's host row is the widget between the banner and the splitter,
    # and the bar has to be in the central layout rather than floating.
    hosts = [item for item in widgets if item is not None and item.isAncestorOf(tabs)]
    assert len(hosts) == 1, "工程栏应在主布局里"
    assert widgets.index(hosts[0]) == 1, "工程栏应紧跟在提示条下面、画布上面"


def test_the_plus_button_is_visible_and_opens_a_board(window, app):
    """A tab strip whose only way to add a tab is a menu is a strip nobody
    finds. The ＋ is a real button next to the tabs."""
    button = window.new_tab_button
    assert isinstance(button, QToolButton)
    assert button.isVisibleTo(window)

    button.click()
    app.processEvents()

    assert len(window.workspace) == 2
    assert window.project_tabs.count() == 2


def test_each_board_keeps_its_own_geometry(window):
    size_board(window, 40.0)
    first = window.active_project

    new_project_tab(window)
    size_board(window, 90.0)

    assert window.design_canvas.board.outline.width_mm == pytest.approx(90.0)
    window.project_tabs.setCurrentIndex(0)
    assert window.design_canvas.board is first.board
    assert window.design_canvas.board.outline.width_mm == pytest.approx(40.0)


def test_switching_repoints_every_pane(window):
    """A pane left holding the old board looks like the new tab ignoring the
    user's next few clicks, so all of them are checked."""
    size_board(window, 40.0)
    new_project_tab(window)
    size_board(window, 90.0)
    window.layer_panel._set_current("bottom_silk")

    window.project_tabs.setCurrentIndex(0)

    board = window.design_canvas.board
    assert window.board is board
    assert window.layer_panel._board is board
    assert window.image_inspector is not None


def test_the_view_comes_back_where_it_was_left(window):
    """Reopening a board fitted to the window is technically right and still
    costs the user the place they had found."""
    first = window.active_project
    window.design_canvas.zoom_by(4.0)
    before = window.design_canvas.view_zoom

    new_project_tab(window)
    assert window.design_canvas.view_zoom == pytest.approx(1.0)

    window.project_tabs.setCurrentIndex(0)
    assert window.design_canvas.board is first.board
    assert window.design_canvas.view_zoom == pytest.approx(before)


def tab_close_buttons(window):
    """The hand-made ✕ widgets on the tab row."""
    from PySide6.QtWidgets import QToolButton

    return [
        button
        for button in window.project_tabs.findChildren(QToolButton)
        if button.objectName() == "TabClose"
    ]


def test_every_tab_has_a_visible_close_button(window):
    """A stylesheet on the tab bar removes the one the platform style would
    draw, so the tabs have to carry their own -- otherwise there is no visible
    way to close a board at all."""
    new_project_tab(window)
    new_project_tab(window)

    assert window.project_tabs.count() == 3
    assert len(tab_close_buttons(window)) == 3
    assert all(button.isVisibleTo(window) for button in tab_close_buttons(window))


def test_a_close_button_closes_its_own_tab_after_the_row_changes(window, app):
    """The click resolves to a tab index at click time. An index captured when
    the button was made would point at somebody else's board as soon as the
    row was reordered."""
    new_project_tab(window)
    new_project_tab(window)
    second = window.workspace[1]
    app.processEvents()

    # Move that tab to the end, so a remembered index would now be wrong.
    window.project_tabs.moveTab(1, 2)
    app.processEvents()
    assert window.workspace[2] is second

    window.project_tabs.tabButton(
        2, window.project_tabs._close_side  # noqa: SLF001
    ).click()
    app.processEvents()

    assert second not in window.workspace.projects
    assert len(window.workspace) == 2


def test_the_timer_does_not_rebuild_the_tab_row(window, app):
    """The dirty dot is re-derived twice a second. Rebuilding the row each
    tick threw the old tab widgets away and made new ones -- which stacked up
    until the process had grown by gigabytes."""
    new_project_tab(window)
    new_project_tab(window)
    app.processEvents()

    before = tab_close_buttons(window)
    for _ in range(50):
        gui_mockup.refresh_workspace_state(window)
    app.processEvents()

    assert window.project_tabs.count() == 3
    assert tab_close_buttons(window) == before, "按钮被反复重建了"


def test_an_unchanged_refresh_leaves_the_tab_row_alone(window):
    """Hover state lives on the widgets; rebuilding them every 500 ms makes
    the tab under the cursor flicker.

    Written as a sentinel: the title is changed by hand, so a refresh that
    rebuilt the row would write the real title back over it. It surviving is
    the proof that nothing was touched.
    """
    window.project_tabs.setTabText(0, "手写的标题")

    gui_mockup.refresh_project_tabs(window)

    assert window.project_tabs.tabText(0) == "手写的标题", "没有变化时应跳过重建"

    # But a real change must still get through.
    window.active_project.title = "换了个名字"
    gui_mockup.refresh_project_tabs(window)
    assert window.project_tabs.tabText(0) == "换了个名字"


def test_the_timer_does_not_rebuild_the_draft_rows(window, app):
    """Same leak, one panel over: the rows are real widgets with child labels
    and signals."""
    store = window._draft_store
    for name in ("甲", "乙", "丙"):
        store.save(window.design_canvas.board, name)
    gui_mockup.refresh_workspace_state(window)
    app.processEvents()

    before = window.draft_panel.findChildren(gui_mockup.DraftRow)
    for _ in range(50):
        gui_mockup.refresh_workspace_state(window)
    app.processEvents()

    after = window.draft_panel.findChildren(gui_mockup.DraftRow)
    assert len(after) == len(before) == 3
    # And the rows are the same objects, not freshly built copies.
    assert all(item in before for item in after)


def test_ctrl_tab_walks_the_open_boards(window, app):
    from PySide6.QtGui import QKeyEvent

    new_project_tab(window)
    window.project_tabs.setCurrentIndex(0)

    event = QKeyEvent(
        QKeyEvent.KeyPress, Qt.Key_Tab, Qt.ControlModifier
    )
    window.keyPressEvent(event)

    assert window.project_tabs.currentIndex() == 1
    assert window.design_canvas.board is window.workspace[1].board


# ---------------------------------------------------------- closing tabs


def test_closing_a_board_leaves_the_others_open(window):
    size_board(window, 40.0)
    save_board(window, "第一块")
    new_project_tab(window)
    size_board(window, 90.0)
    save_board(window, "第二块")
    keep = window.active_project

    assert close_project_tab(window, 0) is True

    assert len(window.workspace) == 1
    assert window.active_project is keep
    assert window.design_canvas.board.outline.width_mm == pytest.approx(90.0)
    assert window.project_tabs.count() == 1


def test_closing_the_last_board_leaves_a_fresh_one(window):
    """The window always has a document: every pane reads the active project,
    so "nothing is open" would have to be handled in all of them."""
    size_board(window, 40.0)
    save_board(window, "唯一的")

    assert close_project_tab(window, 0) is True

    assert len(window.workspace) == 1
    assert window.design_canvas.board.outline_ready is False
    assert window.draft_panel.current_path() is None


def test_closing_a_dirty_board_asks_first(window, monkeypatch):
    """The safe answer is the default, so answering nothing keeps the tab."""
    size_board(window, 40.0)

    assert close_project_tab(window, 0) is False
    assert window.design_canvas.board.outline_ready is True, "取消后不应关闭"


def test_the_close_question_names_the_board(window, monkeypatch):
    """With four boards open, "还有未保存的修改" is not enough to act on."""
    size_board(window, 40.0)
    asked: list[str] = []

    def answer(box, *args, **kwargs):
        asked.append(box.text())
        buttons = {button.text(): button for button in box.buttons()}
        monkeypatch.setattr(box, "clickedButton", lambda: buttons["取消"])
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)

    close_project_tab(window, 0)

    assert asked and "未命名" in asked[0], "问题要点出是哪一块板子"


def test_closing_a_background_dirty_board_still_asks(
    window, monkeypatch
):
    """The tab being closed is not always the tab on screen; asking only
    about the active one would drop the other board without a word."""
    size_board(window, 40.0)
    first = window.active_project
    new_project_tab(window)
    assert window.active_project is not first

    asked: list[str] = []
    monkeypatch.setattr(
        gui_mockup,
        "_confirm_discard_project",
        lambda win, project: asked.append(project.title) or True,
    )

    assert close_project_tab(window, 0) is True
    assert asked == [first.title]


def test_closing_the_others_keeps_only_the_clicked_board(window):
    size_board(window, 40.0)
    save_board(window, "第一块")
    new_project_tab(window)
    size_board(window, 60.0)
    save_board(window, "要留的")
    keep = window.active_project
    new_project_tab(window)
    size_board(window, 90.0)
    save_board(window, "第三块")

    assert gui_mockup.close_other_project_tabs(window, 1) is True

    assert len(window.workspace) == 1
    assert window.active_project is keep
    assert window.design_canvas.board.outline.width_mm == pytest.approx(60.0)


def test_closing_the_others_refuses_before_dropping_anything(window, monkeypatch):
    """A "no" part-way through must not leave a window the user did not ask
    for with some tabs already gone."""
    size_board(window, 40.0)
    new_project_tab(window)
    size_board(window, 60.0)
    new_project_tab(window)
    size_board(window, 90.0)

    monkeypatch.setattr(
        gui_mockup, "_confirm_discard_project", lambda win, project: False
    )

    assert gui_mockup.close_other_project_tabs(window, 2) is False
    assert len(window.workspace) == 3, "拒绝之后一个都不该关"


# --------------------------------------------------------- dirty marking


def test_an_edited_board_is_marked_in_the_tab_row(window):
    """The dot has to be on the tab the user is *not* looking at, because
    that is the one about to be forgotten."""
    new_project_tab(window)
    size_board(window, 90.0)

    gui_mockup.refresh_workspace_state(window)

    assert window.project_tabs.tabText(1).startswith("●")
    assert not window.project_tabs.tabText(0).startswith("●")


def test_the_dot_clears_when_the_board_is_written_down(window):
    size_board(window, 40.0)
    gui_mockup.refresh_workspace_state(window)
    assert window.project_tabs.tabText(0).startswith("●")

    gui_mockup.save_draft_clicked(window)
    gui_mockup.refresh_workspace_state(window)

    assert not window.project_tabs.tabText(0).startswith("●")
    assert window.active_project.draft_path


def test_an_unnamed_board_keeps_its_name_when_saved(window):
    """A board nobody has named stays 未命名 in the tab; the draft is the file
    behind it, not a new identity for it."""
    size_board(window, 40.0)

    gui_mockup.save_draft_clicked(window)

    assert window.active_project.is_unnamed() is True


# ------------------------------------------------------- quitting with many


def test_quitting_asks_about_every_unwritten_board(window, monkeypatch):
    """Four boards open with three of them dirty: silently dropping the three
    the user was not looking at is the worst thing this feature could do."""
    size_board(window, 40.0)
    new_project_tab(window)
    size_board(window, 60.0)
    new_project_tab(window)
    size_board(window, 90.0)

    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box, *a, **k: 0)
    monkeypatch.setattr(
        gui_mockup,
        "_confirm_discard_project",
        lambda win, project: asked.append(project.title) or True,
    )

    assert gui_mockup.confirm_discard_all(window) is True
    assert len(asked) == 3


def test_a_cancel_while_quitting_keeps_the_window_open(window, monkeypatch):
    size_board(window, 40.0)
    new_project_tab(window)
    size_board(window, 60.0)

    answers = iter([True, False])
    monkeypatch.setattr(
        gui_mockup,
        "_confirm_discard_project",
        lambda win, project: next(answers),
    )

    assert gui_mockup.confirm_discard_all(window) is False


def test_a_clean_window_quitting_asks_nothing(window, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(
        gui_mockup,
        "_confirm_discard_project",
        lambda win, project: asked.append(project.title) or True,
    )

    assert gui_mockup.confirm_discard_all(window) is True
    assert asked == []


def test_saving_all_writes_every_dirty_board(window):
    size_board(window, 40.0)
    new_project_tab(window)
    size_board(window, 60.0)
    new_project_tab(window)
    size_board(window, 90.0)

    assert gui_mockup.save_all_drafts(window) is True

    assert window.workspace.dirty() == []
    for index, project in enumerate(window.workspace):
        assert project.draft_path, f"第 {index} 块板子应写到自己的草稿"
        assert Path(project.draft_path).exists()
