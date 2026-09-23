"""Render clean screenshots for the website tutorial.

The states are chosen for the tutorial's narration, not for layout review --
that is what ``simulate_ui.py`` is for. Two properties matter here and neither
is optional:

* The draft library is a scratch folder. A public screenshot must not show
  whatever rows happen to be in the developer's own sidebar.
* The artwork is synthesised. The tutorial goes on a public site, so it uses
  the tool's own test picture rather than any user material.

Usage: python tools/make_tutorial_shots.py --outdir out/tutorial_shots

The filenames match the ones referenced by the website article
``pcb-art-guide.md`` (copied to ``public/images/pcb-art/``), so re-running
this refreshes that tutorial's pictures.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import gui_mockup as g  # noqa: E402

SCRATCH = ROOT / "out" / "_tutorial_shots"


def artwork(work: Path) -> str:
    import cv2

    from pcbart.image_prep import imwrite_unicode, make_test_image

    work.mkdir(parents=True, exist_ok=True)
    path = work / "sample.png"
    imwrite_unicode(
        str(path), cv2.cvtColor(make_test_image(shape="letter"), cv2.COLOR_GRAY2BGR)
    )
    return str(path)


def setup(window, app, image: str):
    """Import the sample and hand back the panel, so states start from one place."""
    panel = g.show_preprocess(window)
    panel.load_image(image)
    panel.import_to_canvas()
    window.design_canvas.fit_view()
    app.processEvents()
    return panel


def state_main(window, app, image):
    """The resting state: empty board, nothing imported, no panels open."""
    window.design_canvas.fit_view()
    app.processEvents()


def state_lineart(window, app, image):
    panel = setup(window, app, image)
    panel.style_combo.setCurrentIndex(0)
    panel.place_top_right()
    panel.raise_()


def state_boardart(window, app, image):
    panel = setup(window, app, image)
    names = [name for name, _ in g.STYLE_OPTIONS]
    panel.style_combo.setCurrentIndex(names.index(g.BOARD_ART_STYLE))
    panel.place_top_right()
    panel.raise_()


def state_board(window, app, image):
    setup(window, app, image)
    panel = g.show_board_setup(window)
    panel.shape_combo.setCurrentIndex(g.SHAPE_CHOICES.index("rounded"))
    panel.raise_()


def state_standee(window, app, image):
    setup(window, app, image)
    panel = g.show_board_setup(window)
    panel.shape_combo.setCurrentIndex(g.SHAPE_CHOICES.index("rect"))
    panel.standee_check.setChecked(True)
    panel.raise_()


def state_selected(window, app, image):
    """Selection live, so the size handles and the right-hand pane both show.

    The preprocessing panel is closed first: it covers exactly the corner the
    handles live in, and a screenshot of the selection feature that does not
    show the selection teaches nothing.
    """
    panel = setup(window, app, image)
    panel.hide()
    canvas = window.design_canvas
    if canvas.board.placements:
        canvas.selected = canvas.board.placements[0]
        canvas.selection_changed.emit(canvas.selected)
        canvas.update()
    app.processEvents()


def state_drafts(window, app, image):
    panel = setup(window, app, image)
    panel.hide()
    g.save_draft(
        window.design_canvas.board, str(SCRATCH / "library" / "示例板.pcbdraft")
    )
    g.refresh_workspace_state(window)
    app.processEvents()


STATES = {
    "01-interface": state_main,
    "02-lineart": state_lineart,
    "03-boardart": state_boardart,
    "04-selected": state_selected,
    "05-board": state_board,
    "06-standee": state_standee,
    "07-drafts": state_drafts,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "out" / "tutorial_shots"))
    parser.add_argument("--only", action="append", choices=sorted(STATES))
    parser.add_argument("--width", type=int, default=1420)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    image = artwork(SCRATCH)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(
        f"QWidget {{ color: {g.TEXT}; }} QMainWindow {{ background: {g.WINDOW_BG}; }}"
    )

    from pcbart.draft import DraftStore

    for name in args.only or sorted(STATES):
        window = g.build_window()
        window._draft_store = DraftStore(SCRATCH / "library")
        # build_window() fills the sidebar from the *default* library before we
        # get a chance to redirect it, and every later refresh compares against
        # what is already on screen. Without this the real draft rows would sit
        # in the screenshot no matter what store we point at.
        g.refresh_draft_panel(window)
        window.resize(args.width, args.height)
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        app.processEvents()

        STATES[name](window, app, image)
        app.processEvents()

        target = outdir / f"{name}.png"
        window.grab().save(str(target))
        print(f"{name} -> {target}")
        window.hide()
        window.deleteLater()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
