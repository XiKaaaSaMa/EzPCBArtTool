"""Tests for the 文件 menu: 新建板子 and the two 另存为 targets.

The three entries exist because there was previously no way to start a new
board without quitting the program, and no way to produce an order file at
all. Both are checked by *running them* and looking at what lands on disk,
because a menu entry that does nothing is exactly the failure the interface
has been audited for.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import build_window  # noqa: E402
from pcbart.artboard import (  # noqa: E402
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
)
from pcbart.draft import DraftStore  # noqa: E402
from pcbart.gerber import DEFAULT_PREFIX, OUTLINE_LAYER  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def no_modal_can_block(monkeypatch):
    """No dialog may block the run.

    ``QMessageBox.exec`` waits for a click, and one that opens without a test
    expecting it does not fail -- it hangs, with a real window on the user's
    desktop. The static helpers are no safer: ``QMessageBox.warning`` is only
    a shortcut that calls ``exec`` internally. All of them are intercepted
    here, and a test that wants a specific answer overrides this.
    """
    def answer(box, *args, **kwargs):
        buttons = {button.text(): button for button in box.buttons()}
        monkeypatch.setattr(
            box, "clickedButton", lambda: buttons.get("取消") or None
        )
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(answer))


@pytest.fixture
def window(app, tmp_path):
    win = build_window()
    win._draft_store = DraftStore(tmp_path / "library")
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    yield win
    win.hide()


def artwork(tmp_path: Path) -> str:
    """A dark rectangle on white: the copper layer's only content."""
    import cv2

    pixels = np.full((160, 240), 255, np.uint8)
    cv2.rectangle(pixels, (30, 30), (210, 130), 0, -1)
    path = tmp_path / "art.png"
    cv2.imwrite(str(path), pixels)
    return str(path)


def ready_board(window, path: str | None = None) -> None:
    board = window.design_canvas.board
    board.outline_ready = True
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=60, height_mm=40
    )
    if path is not None:
        board.placements.append(
            ImagePlacement(path=path, width_mm=60, height_mm=40, layer="顶层")
        )


def menu(window):
    found = {}
    for top in window.menuBar().actions():
        submenu = top.menu()
        if submenu is None:
            continue
        for action in submenu.actions():
            if not action.isSeparator():
                found[action.text()] = action
    return found


# --------------------------------------------------------------- 新建板子


def test_the_file_menu_offers_a_new_board(window):
    """The whole point: starting a second board used to mean quitting."""
    labels = menu(window)
    assert "新建板子" in labels


def test_new_board_opens_a_tab_and_keeps_the_current_one(window, app):
    ready_board(window)
    first = window.active_project

    menu(window)["新建板子"].trigger()
    app.processEvents()

    assert len(window.workspace) == 2
    assert first in window.workspace.projects
    assert window.design_canvas.board is not first.board
    assert window.design_canvas.board.outline_ready is False, "新板子应是空的"


def test_new_board_does_not_ask_about_unsaved_work(window, monkeypatch):
    """Nothing is discarded, so there is nothing to ask about. The question
    that used to be here is what made a second board impossible."""
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box, *a, **k: asked.append("x"))
    ready_board(window)

    menu(window)["新建板子"].trigger()

    assert asked == []
    assert len(window.workspace) == 2


# ------------------------------------------------------------ 另存为工程


def test_save_as_project_writes_the_file_and_names_it(window, tmp_path, monkeypatch):
    ready_board(window)
    target = tmp_path / "工程" / "猫娘板.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    shown: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(str(args[-1])),
    )

    assert gui_mockup.export_project(window) == str(target)

    assert target.exists()
    assert window.active_project.project_path == str(target)
    assert window.active_project.title == "猫娘板"
    assert shown and "猫娘板" in shown[0]


def test_save_as_project_adds_the_extension_when_it_is_missing(
    window, tmp_path, monkeypatch
):
    ready_board(window)
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "无后缀"), ""),
    )
    monkeypatch.setattr(gui_mockup.QMessageBox, "information", lambda *a, **k: None)

    written = gui_mockup.export_project(window)

    assert written is not None and written.endswith(".epro2")
    assert Path(written).exists()


def test_the_second_save_starts_at_the_first_target(window, tmp_path, monkeypatch):
    """A board with a known target should not send the user back to a blank
    dialog every time."""
    ready_board(window)
    first = tmp_path / "甲.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(first), ""),
    )
    monkeypatch.setattr(gui_mockup.QMessageBox, "information", lambda *a, **k: None)
    gui_mockup.export_project(window)

    suggested: list[str] = []

    def capture(_parent, _title, start, *args, **kwargs):
        suggested.append(str(start))
        return ("", "")

    monkeypatch.setattr(gui_mockup.QFileDialog, "getSaveFileName", capture)
    gui_mockup.export_project(window)

    assert suggested == [str(first)]


# ------------------------------------------------------- 另存为下单文件


def test_save_as_order_file_writes_one_uploadable_zip(
    window, tmp_path, monkeypatch
):
    ready_board(window, artwork(tmp_path))
    target = tmp_path / "下单" / "猫娘板.zip"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    shown: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(str(args[-1])),
    )

    assert gui_mockup.export_gerber_project(window) == str(target)

    assert target.exists()
    assert [item.name for item in target.parent.iterdir()] == ["猫娘板.zip"], (
        "下单只应留下一个压缩包，不该散落一堆层文件"
    )
    with zipfile.ZipFile(target) as bundle:
        packed = set(bundle.namelist())
    assert f"{DEFAULT_PREFIX}{OUTLINE_LAYER.file_name}" in packed
    assert f"{DEFAULT_PREFIX}TopLayer.GTL" in packed
    assert window.active_project.order_path == str(target)
    assert shown and "嘉立创" in shown[0]


def test_save_as_order_file_reports_a_problem_instead_of_writing_junk(
    window, tmp_path, monkeypatch
):
    """No outline means no board to cut; the refusal has to be visible and
    must not leave a half-made zip at the target."""
    target = tmp_path / "下单.zip"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    warned: list[str] = []
    monkeypatch.setattr(
        gui_mockup.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warned.append(str(args[-1]))),
    )

    assert gui_mockup.export_gerber_project(window) is None

    assert warned and "板框" in warned[0]
    assert not target.exists()


def test_the_order_file_is_remembered_separately_from_the_project(
    window, tmp_path, monkeypatch
):
    """The two products are different files for different readers; saving one
    must not make the other dialog jump to the wrong folder."""
    ready_board(window, artwork(tmp_path))
    project_target = tmp_path / "工程" / "甲.epro2"
    order_target = tmp_path / "下单" / "甲.zip"

    monkeypatch.setattr(
        gui_mockup.QMessageBox, "information", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_target), ""),
    )
    gui_mockup.export_project(window)
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(order_target), ""),
    )
    gui_mockup.export_gerber_project(window)

    project = window.active_project
    assert project.project_path == str(project_target)
    assert project.order_path == str(order_target)
    assert Path(project.project_path).exists()
    assert Path(project.order_path).exists()


def test_each_board_remembers_its_own_targets(window, tmp_path, monkeypatch):
    ready_board(window)
    monkeypatch.setattr(
        gui_mockup.QMessageBox, "information", lambda *a, **k: None
    )
    first = tmp_path / "甲.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(first), ""),
    )
    gui_mockup.export_project(window)
    first_project = window.active_project

    gui_mockup.new_project_tab(window)
    ready_board(window)
    second = tmp_path / "乙.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(second), ""),
    )
    gui_mockup.export_project(window)

    assert first_project.project_path == str(first)
    assert window.active_project.project_path == str(second)
    assert window.project_tabs.tabText(0) == "甲"
    assert window.project_tabs.tabText(1) == "乙"


def test_saving_a_product_also_stores_the_working_session(
    window, tmp_path, monkeypatch
):
    """The .epro2 and the Gerber set are products; neither carries the source
    picture or the parameters behind it. A 另存为 that left the tab marked ●
    would claim the work is unsaved right after the user saved it."""
    ready_board(window, artwork(tmp_path))
    target = tmp_path / "甲.epro2"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    monkeypatch.setattr(gui_mockup.QMessageBox, "information", lambda *a, **k: None)

    gui_mockup.export_project(window)

    project = window.active_project
    assert project.draft_path, "应该同时落了一份草稿"
    assert Path(project.draft_path).exists()
    assert project.is_dirty() is False
    # And the dot in the tab row goes with it.
    assert not window.project_tabs.tabText(0).startswith("●")


def test_the_order_file_save_also_stores_the_working_session(
    window, tmp_path, monkeypatch
):
    ready_board(window, artwork(tmp_path))
    target = tmp_path / "下单" / "甲.zip"
    monkeypatch.setattr(
        gui_mockup.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    monkeypatch.setattr(gui_mockup.QMessageBox, "information", lambda *a, **k: None)

    gui_mockup.export_gerber_project(window)

    assert window.active_project.draft_path
    assert window.active_project.is_dirty() is False
