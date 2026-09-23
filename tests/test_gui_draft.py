"""Tests for the draft sidebar and the unsaved-changes question at exit.

Driven through the real window rather than by inspecting the panel, because
what matters is the behaviour the user experiences: edit something, close the
window, and be asked about it. The one thing a test here must never do is let a
modal dialog block the run -- the question is answered by clicking the button a
person would click, through the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    build_window,
    draft_dirty,
    open_draft_file,
    refresh_draft_panel,
    rename_draft as gui_mockup_rename,
    save_draft_to,
)
from pcbart.artboard import (  # noqa: E402
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def no_modal_can_block(monkeypatch):
    """A dialog that opens without a test expecting it does not fail -- it
    hangs, and puts a real window on the user's desktop. Answering 取消 makes
    the safe outcome the default so a half-written test cannot trap the run.
    """
    def answer(box, *args, **kwargs):
        buttons = {button.text(): button for button in box.buttons()}
        monkeypatch.setattr(
            box, "clickedButton", lambda: buttons.get("取消") or None
        )
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)


@pytest.fixture
def window(app):
    win = build_window()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    yield win
    # Closing here would raise the unsaved-changes dialog. Hidden instead, so
    # the fixture never blocks on a modal window.
    win._draft_saved_fingerprint = None
    win.hide()


def edit_board(window) -> None:
    """One edit, of the kind a user makes constantly."""
    window.design_canvas.board.outline_ready = True
    window.design_canvas.board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=50.0, height_mm=30.0
    )


# --------------------------------------------------------------- the pane


def test_the_left_pane_starts_as_a_clean_new_document(window):
    assert window.draft_panel.current_path() is None
    assert draft_dirty(window) is False
    assert "新文档" in window.draft_panel.state_label.text()


def test_the_pane_reports_an_unsaved_edit(window):
    edit_board(window)

    refresh_draft_panel(window)

    assert draft_dirty(window) is True
    assert "未保存" in window.draft_panel.state_label.text()


def test_the_pane_offers_a_manual_save(window):
    """The user asked for a save button in this pane, not only a menu item."""
    assert window.draft_panel.save_button.isEnabled()
    assert window.draft_panel.save_button.text() == "保存草稿"


def test_saving_names_the_file_and_clears_the_warning(window, tmp_path):
    edit_board(window)
    refresh_draft_panel(window)

    assert save_draft_to(window, str(tmp_path / "我的草稿.pcbdraft")) is True

    assert draft_dirty(window) is False
    # The name now lives on the draft's row in the list rather than in a
    # separate label, so that is where "which draft is this" is answered.
    assert window.draft_panel.current_path() == str(tmp_path / "我的草稿.pcbdraft")
    assert "已保存" in window.draft_panel.state_label.text()
    assert "我的草稿" in window.draft_panel.state_label.text()


def test_further_edits_after_saving_are_still_noticed(window, tmp_path):
    """The bug this guards against: the panel says "saved" forever because the
    dirty flag was cleared once and never reconsidered."""
    edit_board(window)
    save_draft_to(window, str(tmp_path / "d.pcbdraft"))
    assert draft_dirty(window) is False

    window.design_canvas.board.placements.append(
        ImagePlacement(path="new.png", width_mm=10, height_mm=10)
    )
    refresh_draft_panel(window)

    assert draft_dirty(window) is True
    assert "未保存" in window.draft_panel.state_label.text()


def test_the_status_survives_a_drag_on_the_canvas(window, tmp_path):
    """Canvas edits do not go through any panel, which is exactly why the
    dirty check compares the document instead of trusting a signal."""
    edit_board(window)
    save_draft_to(window, str(tmp_path / "d.pcbdraft"))

    window.design_canvas.board.outline.width_mm = 77.0
    refresh_draft_panel(window)

    assert draft_dirty(window) is True


def test_saving_twice_writes_to_the_same_file(window, tmp_path):
    """Save, keep working, save again -- without being asked for a name
    again."""
    edit_board(window)
    first = tmp_path / "d.pcbdraft"
    save_draft_to(window, str(first))

    window.design_canvas.board.outline.width_mm = 61.0
    refresh_draft_panel(window)
    assert window.draft_panel.current_path() == str(first)


# ------------------------------------------------------------ the question


def test_closing_a_clean_window_does_not_ask(window, monkeypatch):
    """A question about nothing trains the user to click through the one that
    matters."""
    asked = []

    def record(*args, **kwargs):
        asked.append(args)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "exec", record)

    refresh_draft_panel(window)
    window.close()

    assert asked == [], "没有修改时不应弹询问"


def test_closing_with_unsaved_work_asks(window, app, monkeypatch):
    """Answered the way a person answers it: by clicking 不保存."""
    seen = {}

    def answer(box, *args, **kwargs):
        seen["text"] = box.text()
        buttons = {b.text(): b for b in box.buttons()}
        seen["buttons"] = set(buttons)
        # Answering the way a click does: report which button was chosen.
        monkeypatch.setattr(box, "clickedButton", lambda: buttons["不保存"])
        return QMessageBox.DestructiveRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)

    edit_board(window)
    refresh_draft_panel(window)
    window.close()
    app.processEvents()

    assert "未保存" in seen["text"]
    assert {"保存草稿", "不保存", "取消"} <= seen["buttons"]


def test_cancelling_the_question_keeps_the_window_open(window, monkeypatch):
    """取消 must mean the close did not happen -- the alternative is losing
    the work the user just chose to keep."""
    def answer(box, *args, **kwargs):
        buttons = {b.text(): b for b in box.buttons()}
        monkeypatch.setattr(box, "clickedButton", lambda: buttons["取消"])
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)

    edit_board(window)
    refresh_draft_panel(window)
    window.close()

    assert window.isVisible() is True, "取消后窗口应仍然打开"
    assert draft_dirty(window) is True, "取消不应丢弃修改"


def test_saving_from_the_question_writes_the_file(window, tmp_path, monkeypatch):
    """The 保存草稿 button in the dialog saves and then lets the close happen."""
    edit_board(window)
    save_draft_to(window, str(tmp_path / "d.pcbdraft"))
    window.design_canvas.board.outline.width_mm = 88.0
    refresh_draft_panel(window)

    def answer(box, *args, **kwargs):
        buttons = {b.text(): b for b in box.buttons()}
        monkeypatch.setattr(box, "clickedButton", lambda: buttons["保存草稿"])
        return QMessageBox.AcceptRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)
    window.close()

    from pcbart.draft import load_draft

    assert load_draft(tmp_path / "d.pcbdraft").outline.width_mm == pytest.approx(88.0)
    assert draft_dirty(window) is False


# ------------------------------------------------------- opening a draft


def test_opening_a_draft_replaces_the_document(window, tmp_path):
    source = build_window()
    source.setAttribute(Qt.WA_DontShowOnScreen, True)
    try:
        source.design_canvas.board.outline_ready = True
        source.design_canvas.board.outline = BoardOutlineSpec(
            shape=OutlineShape.ROUNDED, width_mm=77.0, height_mm=44.0
        )
        path = tmp_path / "saved.pcbdraft"
        save_draft_to(source, str(path))
    finally:
        source.hide()

    assert open_draft_file(window, str(path)) is True

    assert window.design_canvas.board.outline.width_mm == pytest.approx(77.0)
    assert window.design_canvas.board.outline.shape == OutlineShape.ROUNDED
    assert window.draft_panel.current_path() == str(path)
    assert draft_dirty(window) is False


def test_a_loaded_draft_reaches_the_other_panes(window, tmp_path):
    """Every pane holds its own reference; a missed one edits the document
    that was just replaced."""
    from pcbart.draft import save_draft

    board = build_window().design_canvas.board
    board.outline_ready = True
    board.outline = BoardOutlineSpec(width_mm=64.0, height_mm=32.0)
    board.layers.set_visible("top_silk", False)
    path = tmp_path / "saved.pcbdraft"
    save_draft(board, path)

    assert open_draft_file(window, str(path)) is True

    assert window.board is window.design_canvas.board, "窗口持有的板框引用应更新"
    assert window.layer_panel._board is window.board
    assert window.design_canvas.board.layers.is_visible("top_silk") is False


def test_opening_a_broken_draft_leaves_the_document_alone(
    window, tmp_path, monkeypatch
):
    """A file that cannot be read must not clear the board the user is on."""
    import gui_mockup

    edit_board(window)
    before = window.design_canvas.board.outline.width_mm
    broken = tmp_path / "broken.pcbdraft"
    broken.write_text("not json at all", encoding="utf-8")

    # Both dialogs would block the run: the unsaved-changes question (the
    # document *is* dirty here) and the failure warning. Answer the first by
    # allowing the action, and record the second.
    monkeypatch.setattr(
        gui_mockup, "confirm_discard_changes", lambda *a, **k: True
    )
    warnings = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )

    assert open_draft_file(window, str(broken)) is False

    assert window.design_canvas.board.outline.width_mm == pytest.approx(before)
    assert warnings, "应该告诉用户文件读不了"


# ------------------------------------------------- headless export paths
#
# The packaged build cannot be driven by clicking through its UI from a test,
# so these two flags are how its writers get verified. They are also the
# cleanest proof that the draft and Gerber paths work without any dialog.


def test_the_headless_flags_write_a_draft_and_a_gerber_set(tmp_path, monkeypatch):
    """The flags are how a *packaged* build's writers get verified: the frozen
    exe can be run with them, so "it starts" stops being the only evidence.

    ``main`` makes its own ``QApplication``, and Qt allows exactly one per
    process. This runs in a subprocess, which is both the only way to satisfy
    that and closer to how the packaged exe is actually exercised.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    draft = tmp_path / "草稿.pcbdraft"
    gerber = tmp_path / "gerber"

    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "gui_mockup.py"),
            # A board has to exist before there is anything to write, and the
            # standee exercises the piece of geometry this release added.
            "--board-shape",
            "rect",
            "--standee",
            "--draft-to",
            str(draft),
            "--gerber-to",
            str(gerber),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert draft.exists(), "草稿应被写出"
    outline = gerber / "Gerber_BoardOutlineLayer.GKO"
    assert outline.exists(), f"板框 Gerber 应被写出；stdout={result.stdout}"

    from pcbart.draft import load_draft

    restored = load_draft(draft)
    assert restored.outline_ready is True
    assert restored.outline.standee is True, "立牌状态应随草稿保存"


# -------------------------------------------------------- the draft library


@pytest.fixture
def library_window(window, tmp_path):
    """A window whose draft library is a scratch folder."""
    from pcbart.draft import DraftStore

    window._draft_store = DraftStore(tmp_path / "library")
    refresh_draft_panel(window)
    return window


def rows_of(window):
    import gui_mockup

    return window.draft_panel.findChildren(gui_mockup.DraftRow)


def test_the_sidebar_lists_the_saved_drafts(library_window, tmp_path):
    """The whole point: reopening yesterday's work is one click in the panel."""
    edit_board(library_window)
    save_draft_to(
        library_window, str(library_window._draft_store.path_for("第一个"))
    )

    refresh_draft_panel(library_window)

    assert [row.name_label.text() for row in rows_of(library_window)] == ["第一个"]


def test_each_draft_appears_exactly_once(library_window):
    """Rebuilding the list must replace the rows, not stack another set.

    The rows are deleted with ``deleteLater``, which only runs when the event
    loop gets to it -- so without reparenting them immediately they stayed
    children of the host and every refresh added a duplicate set.
    """
    edit_board(library_window)
    store = library_window._draft_store
    save_draft_to(library_window, str(store.path_for("甲")))
    save_draft_to(library_window, str(store.path_for("乙")))

    for _ in range(4):
        refresh_draft_panel(library_window)

    names = sorted(row.name_label.text() for row in rows_of(library_window))
    assert names == ["乙", "甲"], names


def test_clicking_a_row_opens_that_draft_in_its_own_tab(library_window):
    """One click on the row, not a file dialog -- and the board that was on
    screen stays open, which is what tabs bought."""
    store = library_window._draft_store
    edit_board(library_window)
    library_window.design_canvas.board.outline.width_mm = 77.0
    save_draft_to(library_window, str(store.path_for("要打开的")))
    was = library_window.active_project

    library_window.active_project.board = library_window.design_canvas.board
    library_window.workspace.new(title="另一个板子")
    gui_mockup_refresh = refresh_draft_panel
    gui_mockup_refresh(library_window)

    row = rows_of(library_window)[0]
    row.click()

    assert library_window.design_canvas.board.outline.width_mm == pytest.approx(77.0)
    assert library_window.draft_panel.current_path() == str(store.path_for("要打开的"))
    assert len(library_window.workspace) == 2, "原来的板子应留在自己的标签里"
    assert was in library_window.workspace.projects


def test_clicking_a_row_for_a_board_that_is_already_open_switches_to_it(
    library_window,
):
    """It must not read the file again: that would throw away the unsaved
    edits in the tab the user is coming back to."""
    store = library_window._draft_store
    edit_board(library_window)
    library_window.design_canvas.board.outline.width_mm = 77.0
    save_draft_to(library_window, str(store.path_for("甲")))
    opened = library_window.active_project

    library_window.workspace.new(title="另一个板子")
    refresh_draft_panel(library_window)
    assert library_window.active_project is not opened

    # An edit after the save, still unwritten.
    opened.board.outline.width_mm = 88.0
    row = next(
        item for item in rows_of(library_window) if item.name_label.text() == "甲"
    )
    row.click()

    assert library_window.active_project is opened
    assert library_window.design_canvas.board.outline.width_mm == pytest.approx(88.0)
    assert len(library_window.workspace) == 2, "不应重复打开同一个草稿"


# ------------------------------------------------- renaming inside the row
#
# The rename box lives on the row itself, so it is typing in a box rather than
# answering a dialog. The two things that can go wrong with it are both about
# the list rebuilding on its own: the box being replaced mid-keystroke, and a
# commit arriving after the row it belonged to is gone.


def row_for(window, name):
    return next(
        item for item in rows_of(window) if item.name_label.text() == name
    )


def test_the_row_offers_a_rename_box(library_window):
    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    library_window._draft_store.save(library_window.design_canvas.board, "甲")
    refresh_draft_panel(library_window)

    row = row_for(library_window, "甲")
    assert row.rename_button.isVisibleTo(row)
    assert row.name_edit.isVisible() is False

    row.edit_name()

    assert row.name_edit.isVisible() is True
    assert row.name_edit.text() == "甲"
    assert row.name_label.isVisible() is False, "编辑时不应同时显示两个名字"


def test_typing_a_new_name_renames_the_draft(library_window, app):
    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    store = library_window._draft_store
    store.save(library_window.design_canvas.board, "旧名字")
    refresh_draft_panel(library_window)

    row = row_for(library_window, "旧名字")
    row.edit_name()
    row.name_edit.setText("新名字")
    row.name_edit.returnPressed.emit()
    app.processEvents()

    names = [info.name for info in store.list()]
    assert names == ["新名字"]
    assert not (store.root / "旧名字.pcbdraft").exists()


def test_escape_puts_the_old_name_back(library_window, app):
    """Esc must cancel, not commit whatever was half-typed when the user
    reached for it."""
    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    store = library_window._draft_store
    store.save(library_window.design_canvas.board, "原名")
    refresh_draft_panel(library_window)

    row = row_for(library_window, "原名")
    row.edit_name()
    row.name_edit.setText("半途而废")
    row.name_edit.editing_cancelled.emit()
    app.processEvents()

    assert [info.name for info in store.list()] == ["原名"]


def test_the_list_timer_does_not_replace_the_box_while_typing(
    library_window, app
):
    """The sidebar is redrawn from the store on a timer. Rebuilding it while
    the user is typing would throw the box away every half second."""
    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    library_window._draft_store.save(library_window.design_canvas.board, "甲")
    refresh_draft_panel(library_window)

    row = row_for(library_window, "甲")
    row.edit_name()
    row.name_edit.setText("正在输入")

    # What the timer does, twice, while the user is still typing.
    refresh_draft_panel(library_window)
    refresh_draft_panel(library_window)

    still_there = row_for(library_window, "甲")
    assert still_there is row, "输入框被刷掉了"
    assert row.name_edit.isVisible() is True
    assert row.name_edit.text() == "正在输入"


def test_renaming_a_draft_that_is_open_renames_its_tab(library_window, app):
    """The document follows the file. Otherwise the next save would recreate
    the old name behind the user, and the tab would name a file that is gone."""
    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    store = library_window._draft_store
    save_draft_to(library_window, str(store.path_for("甲")))
    assert library_window.active_project.title == "甲"

    assert gui_mockup_rename(library_window, str(store.path_for("甲")), "乙") is True

    assert library_window.active_project.draft_path == str(store.path_for("乙"))
    assert library_window.active_project.title == "乙"


def test_renaming_to_a_taken_name_is_refused(library_window, monkeypatch):
    """Two rows with the same name would open whichever file happened to
    sort first, which is not a thing the user can reason about."""
    import gui_mockup

    library_window.active_project.board = library_window.design_canvas.board
    edit_board(library_window)
    store = library_window._draft_store
    store.save(library_window.design_canvas.board, "甲")
    store.save(library_window.design_canvas.board, "乙")

    warned: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warned.append(str(a[-1]))),
    )

    assert gui_mockup_rename(library_window, str(store.path_for("甲")), "乙") is False

    assert warned, "应该告诉用户名字被占了"
    assert sorted(info.name for info in store.list()) == ["乙", "甲"]


def test_a_draft_survives_a_restart(library_window, tmp_path, app):
    """A *new* window must see the library the old one wrote.

    This is what "next time I can open it directly" means, and it is the one
    thing an in-memory list would quietly fail at.
    """
    import gui_mockup
    from pcbart.draft import DraftStore

    edit_board(library_window)
    library_window.design_canvas.board.outline.width_mm = 64.0
    library_window._draft_store.save(
        library_window.design_canvas.board, "跨重启"
    )

    reopened = build_window()
    reopened.setAttribute(Qt.WA_DontShowOnScreen, True)
    try:
        reopened._draft_store = DraftStore(library_window._draft_store.root)
        refresh_draft_panel(reopened)
        assert [
            row.name_label.text() for row in rows_of(reopened)
        ] == ["跨重启"]

        gui_mockup.open_draft_file(
            reopened, str(reopened._draft_store.path_for("跨重启"))
        )
        assert reopened.design_canvas.board.outline.width_mm == pytest.approx(64.0)
    finally:
        reopened.hide()


def test_the_open_draft_is_marked_in_the_list(library_window):
    edit_board(library_window)
    store = library_window._draft_store
    save_draft_to(library_window, str(store.path_for("甲")))
    save_draft_to(library_window, str(store.path_for("乙")))
    open_draft_file(library_window, str(store.path_for("甲")))
    refresh_draft_panel(library_window)

    marked = {
        row.name_label.text(): row.subtitle_label.text()
        for row in rows_of(library_window)
    }

    assert "正在编辑" in marked["甲"]
    assert "正在编辑" not in marked["乙"]


def test_an_empty_library_says_so_instead_of_showing_a_blank_box(
    library_window, monkeypatch
):
    """A panel with an empty area reads as broken; it has to say what to do."""
    import gui_mockup

    monkeypatch.setattr(gui_mockup.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    for info in library_window._draft_store.list():
        library_window._draft_store.delete(info.path)
    refresh_draft_panel(library_window)

    assert rows_of(library_window) == []
    assert library_window.draft_panel.empty_label.isVisibleTo(
        library_window.draft_panel
    )


def test_the_library_can_be_a_scratch_folder(library_window, tmp_path):
    """The tests must not write into the user's real library, and this pins
    that the store honours the folder it is given."""
    assert library_window._draft_store.root == tmp_path / "library"
