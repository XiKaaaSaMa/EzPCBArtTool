#!/usr/bin/env python
"""Render a set of realistic interface states to PNGs, for reviewing layout.

A single screenshot of an empty window answers nothing: the states that break
are the crowded ones -- nine tabs open, a long draft name, a floating panel
over a tab row. This drives the real window through those states and writes
one image per state, so the review is against what the user will actually see
rather than against a description of it.

The states are chosen to be the awkward cases, not the happy path:

    01-single       one board, the resting state
    02-three-tabs   three boards, one of them with unwritten changes
    03-ten-tabs     ten tabs, which is where the row has to scroll
    04-long-names   names long enough to elide
    05-dirty-other  a background tab dirty, the active one clean
    06-panel-over   the preprocessing panel over the tab row
    07-drafts       a full draft library with a rename box open
    08-renaming     a draft row mid-rename
    09-artwork      artwork on the canvas, tab row in its normal state
    10-narrow       a narrow window, where the tab row is squeezed

Usage::

    python tools/simulate_ui.py                 # all ten, into out/sim/
    python tools/simulate_ui.py --only 03-ten-tabs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
)
from pcbart.draft import DraftStore  # noqa: E402
from pcbart.projects import Project  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gui_mockup  # noqa: E402
from gui_mockup import (  # noqa: E402
    PANEL_BG,
    TEXT,
    WINDOW_BG,
    build_window,
    refresh_project_tabs,
    refresh_workspace_state,
    show_preprocess,
)

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "out" / "sim"
SCRATCH_DRAFTS = SCRATCH / "drafts"


def make_board(width: float = 60.0, height: float = 40.0) -> ArtBoard:
    """A board with a real outline, so the canvas has something to frame."""
    return ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.ROUNDED, width_mm=width, height_mm=height
        ),
        outline_ready=True,
    )


def add_artwork(board: ArtBoard) -> None:
    """One picture on the board, using the sample the other tools produce."""
    sample = ROOT / "out" / "check_letter.png"
    if not sample.exists():
        return
    board.placements.append(
        ImagePlacement(
            path=str(sample), width_mm=40.0, height_mm=25.0, layer="顶层"
        )
    )


def open_tabs(window, titles: list[str], *, dirty: set[int] | None = None) -> None:
    """Give the window ``titles`` as its open boards."""
    dirty = dirty or set()
    workspace = window.workspace
    # Rebuild from scratch rather than adding to whatever is there.
    workspace._projects = []  # noqa: SLF001 - this tool drives the model directly
    for index, title in enumerate(titles):
        project = Project(make_board(50.0 + index * 8.0), title=title)
        if index not in dirty:
            project.saved_fingerprint = project.__dict__["saved_fingerprint"]
        workspace._projects.append(project)  # noqa: SLF001
    workspace._index = 0  # noqa: SLF001
    if dirty:
        # Marking a project dirty means its fingerprint no longer matches the
        # document, which is what the paint code actually tests.
        for index in dirty:
            workspace[index].saved_fingerprint = "stale"
    window.design_canvas.board = workspace.active.board
    window.design_canvas.fit_view()
    refresh_workspace_state(window)


def fill_drafts(window, names: list[str]) -> DraftStore:
    """Put drafts in a scratch library and show them in the sidebar.

    A scratch library, never the real one: this script exists to render
    pictures, and a rendering run that quietly added a dozen rows to the user's
    own sidebar would be a nasty surprise.
    """
    store = DraftStore(SCRATCH_DRAFTS)
    for existing in store.list():
        existing.path.unlink()
    for name in names:
        store.save(make_board(), name)
    window._draft_store = store
    refresh_workspace_state(window)
    return store


def seed_artwork(window) -> None:
    add_artwork(window.design_canvas.board)
    window.design_canvas.fit_view()


# ------------------------------------------------------------------ states


def state_single(window, app) -> None:
    """The resting state: one board, nothing on it."""


def state_three_tabs(window, app) -> None:
    open_tabs(window, ["猫娘板", "底座", "第三块"], dirty={0})


def state_ten_tabs(window, app) -> None:
    open_tabs(
        window,
        [f"第 {index} 块" for index in range(1, 11)],
        dirty={1, 4, 8},
    )


def state_long_names(window, app) -> None:
    open_tabs(
        window,
        [
            "由比滨结衣透明背景立牌",
            "初音未来 Q 版 60x40",
            "这是一个特别特别长的板子名字用来测试截断",
            "短",
        ],
        dirty={2},
    )


def state_dirty_other(window, app) -> None:
    open_tabs(window, ["干净的", "有改动", "另一个"], dirty={1})
    window.project_tabs.setCurrentIndex(0)


def state_panel_over(window, app) -> None:
    open_tabs(window, ["猫娘板", "底座"], dirty={1})
    panel = show_preprocess(window)
    panel.place_top_right()
    panel.raise_()


def state_drafts(window, app) -> None:
    fill_drafts(window, ["猫娘板", "底座试做", "初音 v2", "废弃的想法"])
    open_tabs(window, ["猫娘板"])


def state_renaming(window, app) -> None:
    fill_drafts(window, ["猫娘板", "底座试做", "初音 v2"])
    rows = [
        item
        for item in window.draft_panel.findChildren(gui_mockup.DraftRow)
        if item.name_label.text() == "底座试做"
    ]
    if rows:
        rows[0].edit_name()
        rows[0].name_edit.setText("底座试做 v3")


def state_artwork(window, app) -> None:
    open_tabs(window, ["猫娘板", "底座"], dirty={0})
    seed_artwork(window)


def state_narrow(window, app) -> None:
    window.resize(900, 620)
    open_tabs(
        window,
        ["猫娘板", "底座", "初音 v2", "第四个", "第五个", "第六个"],
        dirty={2},
    )


STATES = {
    "01-single": state_single,
    "02-three-tabs": state_three_tabs,
    "03-ten-tabs": state_ten_tabs,
    "04-long-names": state_long_names,
    "05-dirty-other": state_dirty_other,
    "06-panel-over": state_panel_over,
    "07-drafts": state_drafts,
    "08-renaming": state_renaming,
    "09-artwork": state_artwork,
    "10-narrow": state_narrow,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(SCRATCH))
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(STATES),
        help="render only this state (repeatable)",
    )
    parser.add_argument("--width", type=int, default=1420)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(
        f"QWidget {{ color: {TEXT}; }} QMainWindow {{ background: {WINDOW_BG}; }}"
    )

    wanted = args.only or sorted(STATES)
    for name in wanted:
        window = build_window()
        window.resize(args.width, args.height)
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        app.processEvents()

        STATES[name](window, app)
        app.processEvents()

        target = outdir / f"{name}.png"
        window.grab().save(str(target))
        width = window.project_tabs.count()
        print(f"{name}: {width} 个标签 -> {target}")
        window.hide()
        window.deleteLater()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
