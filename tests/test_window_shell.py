"""Tests for the window shell: which controls exist, and which actually work.

The whole reason these exist is that the interface spent a long time showing
controls that did nothing -- menu entries with no connection, panels full of
hard-coded numbers. Asserting *behaviour* rather than presence is the only way
to catch that: a dead button looks exactly like a live one until it is pressed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import build_window, show_preprocess  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app, tmp_path):
    """A real window whose *draft library* is a scratch folder.

    The library has to be redirected even here. Saving a product file now also
    stashes the editing session, so a test that exports from a real window
    without this would write into the user's own draft library -- every run
    leaving another pair of rows in their sidebar.
    """
    from pcbart.draft import DraftStore

    win = build_window()
    win._draft_store = DraftStore(tmp_path / "library")
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
def artwork_png(tmp_path):
    """A picture to import, synthesised here rather than read from ``out/``.

    These tests used to load ``out/check_letter.png``, a leftover from a manual
    render run. ``out/`` is git-ignored, so on any machine that had not run
    that render the fixture was missing: ``load_image`` returned False, the
    canvas stayed empty, and four tests hung instead of failing -- a missing
    asset looked exactly like a frozen application.
    """
    import cv2

    from pcbart.image_prep import imwrite_unicode, make_test_image

    path = tmp_path / "art.png"
    # Three channels, like the photos the panel actually receives: the source
    # path branches on ``ndim``, so a grayscale-only fixture would leave the
    # colour branch untested.
    image = cv2.cvtColor(make_test_image(shape="letter"), cv2.COLOR_GRAY2BGR)
    imwrite_unicode(str(path), image)
    assert path.exists()
    return str(path)


def all_actions(window) -> dict[str, object]:
    found: dict[str, object] = {}
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if not action.isSeparator():
                found[action.text()] = action
    for bar in window.findChildren(QToolBar):
        for action in bar.actions():
            if not action.isSeparator():
                found.setdefault(action.text(), action)
    return found


# ------------------------------------------------- the removed placeholders


def test_the_window_does_not_call_itself_a_layout_preview(window):
    """It used to say "其余为布局预览"; almost everything in it is real now."""
    assert "预览" not in window.windowTitle()


def test_the_placeholder_statistics_list_is_gone(window):
    """Its numbers were hard-coded and none of its controls were connected."""
    assert not hasattr(window, "stats_view")
    assert not hasattr(gui_mockup, "stats_panel")


def test_the_left_parameter_column_is_gone(window):
    """155 lines of static mock-up that duplicated real controls elsewhere."""
    assert not hasattr(gui_mockup, "parameter_panel")

    splitter = window.centralWidget().findChild(type(window.findChild))
    # Three columns now: the draft pane, the canvas, and the right dock. The
    # left one is not the old mock-up coming back -- that had no signals and
    # hard-coded numbers, while this one is driven by the document.
    from PySide6.QtWidgets import QSplitter

    splitters = window.findChildren(QSplitter)
    assert len(splitters) == 1
    assert splitters[0].count() == 3


def test_the_left_column_is_the_draft_pane(window):
    """The left column answers "is my work saved", not "what are the numbers"."""
    draft = window.draft_panel
    assert window.draft_area.widget() is draft
    assert draft.current_path() is None, "新文档还没有草稿文件"

    # It holds real actions, not displayed constants: a save button that works
    # and a status line derived from the document.
    assert draft.save_button.text() == "保存草稿"
    assert "新文档" in draft.state_label.text()


def test_controls_for_unbuilt_features_are_not_offered(window):
    """A menu entry that silently does nothing reads as a bug."""
    labels = set(all_actions(window))
    for absent in ("显示网格", "显示掩膜叠加", "显示孔位", "打开工程…", "运行工艺校验"):
        assert absent not in labels, f"{absent} 描述的功能还不存在"


def test_the_removed_placeholder_buttons_are_gone(window):
    labels = set(all_actions(window))
    assert "重新生成" not in labels, "没有可重新生成的步骤"
    assert "校验" not in labels, "还没有校验器可运行"
    assert "运行独立校验器" not in labels


# --------------------------------------------------------- what does work


def test_the_two_saves_are_named_for_their_targets(window):
    """One product file for the editor, one for the fab. The old "导出" was
    ambiguous about which of the two it wrote, and they are not the same
    thing: an .epro2 keeps being edited, a Gerber zip gets ordered."""
    labels = all_actions(window)
    assert "另存为工程…" in labels
    assert "另存为下单文件…" in labels
    assert "另存为工程" in labels, "工具栏也要有"
    assert "另存为下单文件" in labels, "工具栏也要有"


def test_export_reports_a_problem_instead_of_failing_silently(window, monkeypatch):
    """Exporting an empty document must say why, not do nothing."""
    shown: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(str(args[-1])),
    )
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "warning",
        lambda *args, **kwargs: shown.append(str(args[-1])),
    )
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(Path("out") / "_never.epro2"), ""),
    )

    gui_mockup.export_project(window)

    assert shown, "没有任何提示"
    assert any("板框" in message for message in shown)


def test_saving_as_a_project_names_the_tab_after_the_file(
    window, tmp_path, monkeypatch
):
    """After 另存为工程 the board has a name, and the tab row should say it."""
    from pcbart.artboard import BoardOutlineSpec

    window.design_canvas.board.outline = BoardOutlineSpec(width_mm=50, height_mm=30)
    window.design_canvas.board.outline_ready = True

    target = tmp_path / "猫娘板.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    monkeypatch.setattr(gui_mockup.QMessageBox, "information", lambda *a, **k: None)

    gui_mockup.export_project(window)

    assert window.active_project.title == "猫娘板"
    assert window.active_project.project_path == str(target)
    assert "猫娘板" in window.project_tabs.tabText(0)


def test_export_writes_a_file_and_says_what_is_in_it(window, tmp_path, monkeypatch):
    """The save must report what it wrote -- in the status bar, not a dialog.

    ⚠️ **The report moved out of the dialog at the user's request**
    (「只提示工程保存完成」): a card with one misplaced illustration appended the
    writer's whole log to the message box, which ran to a page of near-identical
    lines. The dialog now says only that the file was saved, and the detail goes
    to the status bar.

    The *intent* of this test is unchanged and is what matters: the user is
    still told what went into the file. Asserting it against the status bar
    keeps that true without re-introducing the wall of text.
    """
    from pcbart.artboard import BoardOutlineSpec

    window.design_canvas.board.outline = BoardOutlineSpec(width_mm=50, height_mm=30)
    window.design_canvas.board.outline_ready = True

    target = tmp_path / "out.epro2"
    messages: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append(str(args[-1])),
    )

    result = gui_mockup.export_project(window)

    assert result == str(target)
    assert target.exists()

    # The dialog stays short: just the fact that the save happened.
    assert messages and "已保存" in messages[-1], (
        f"对话应当只说保存完成，实际 {messages!r}"
    )
    assert "50.0" not in messages[-1], (
        f"板框尺寸不该再塞进对话（用户要求只提示保存完成），实际 {messages[-1]!r}"
    )

    # And the fact is still reported somewhere the user can read.
    status = window.statusBar().currentMessage()
    assert "50.0" in status, f"状态栏应当写明写出的板框尺寸，实际 {status!r}"


def test_opening_a_picture_offers_the_file_dialog(window, monkeypatch):
    """打开图片 has to do something; it was a dead menu entry."""
    from gui_mockup import show_preprocess as open_preprocess

    monkeypatch.setattr(open_preprocess, "__call__", open_preprocess, raising=False)
    called: list[bool] = []
    panel = open_preprocess(window)

    monkeypatch.setattr(panel.drop_zone, "browse", lambda: called.append(True))
    all_actions(window)["打开图片"].trigger()

    assert called, "打开图片没有打开文件对话框"


def test_quit_is_connected_to_something(window):
    """退出 must be wired. Asserted through the bound signal rather than by
    patching ``close``: the lambda captured the bound method when it was
    connected, so replacing the attribute afterwards changes nothing.
    """
    action = all_actions(window)["退出"]
    # 2triggered(bool) is the form Qt reports for the overload actually used;
    # plain "triggered(bool)" silently reports zero receivers and would make
    # every wired action look dead.
    assert action.receivers("2triggered(bool)") > 0


def test_about_says_what_the_tool_can_do(window, monkeypatch):
    """The box names the features, which build this is, and the licence.

    Asserting features by their own words rather than a whole sentence on
    purpose: the wording is edited freely (the 2026-10-13 copy pass rewrote
    it), but the box has to keep saying what the tool does, which build this
    is, and where the source lives.
    """
    from pcbart import __version__

    messages: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append(str(args[-1])),
    )

    all_actions(window)["关于"].trigger()

    assert messages
    text = messages[0]
    assert "主要功能" in text
    assert __version__ in text
    assert "GPL-3.0" in text
    assert "github.com" in text


def test_zoom_actions_reach_the_canvas(window):
    before = window.design_canvas.view_zoom

    all_actions(window)["放大"].trigger()

    assert window.design_canvas.view_zoom > before


# ------------------------------------------------------------- right dock


def test_the_layer_list_is_docked_not_floating(window):
    """It is the resting state of the right pane, so it lives in the layout."""
    pane = window.layer_panel
    assert not pane.isWindow(), "图层列表不应是独立窗口"
    assert pane.parent() is not None
    assert window.layer_area is not None


def test_the_right_pane_shows_layers_when_nothing_is_selected(window):
    assert window.layer_area.isVisibleTo(window)
    assert not window.inspector_area.isVisibleTo(window)


def test_selecting_a_picture_swaps_the_pane_to_its_properties(window, artwork_png):
    panel = show_preprocess(window)
    panel.load_image(artwork_png)
    assert panel.import_to_canvas() is True
    window.design_canvas.fit_view()

    assert window.design_canvas.selected is not None
    assert window.inspector_area.isVisibleTo(window)
    assert not window.layer_area.isVisibleTo(window)


def test_deselecting_brings_the_layers_back(window, artwork_png):
    panel = show_preprocess(window)
    panel.load_image(artwork_png)
    panel.import_to_canvas()

    window.design_canvas.selected = None
    window.design_canvas.selection_changed.emit(None)

    assert window.layer_area.isVisibleTo(window)
    assert not window.inspector_area.isVisibleTo(window)


def test_the_property_pane_edits_the_size(window, artwork_png):
    """图片尺寸修改 is the point of the pane while something is selected."""
    panel = show_preprocess(window)
    panel.load_image(artwork_png)
    panel.import_to_canvas()

    inspector = window.image_inspector
    inspector.width_spin.setValue(30.0)

    assert window.design_canvas.board.placements[0].width_mm == pytest.approx(30.0)


def test_the_layer_pane_still_controls_the_stack(window):
    window.layer_panel._set_current("bottom_silk")
    assert window.design_canvas.board.layers.current == "bottom_silk"


def test_the_layer_toolbar_entry_reveals_the_pane(window, artwork_png):
    """The toolbar 图层 entry cannot open a panel that is already docked, so it
    clears the selection -- which is what brings the list back."""
    panel = show_preprocess(window)
    panel.load_image(artwork_png)
    panel.import_to_canvas()
    assert window.design_canvas.selected is not None

    all_actions(window)["图层"].trigger()

    assert window.layer_area.isVisibleTo(window)
