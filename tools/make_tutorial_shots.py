"""Render clean screenshots for the website tutorial.

The states are chosen for the tutorial's narration, not for layout review --
that is what ``simulate_ui.py`` is for. Two properties matter here and neither
is optional:

* The draft library is a scratch folder. A public screenshot must not show
  whatever rows happen to be in the developer's own sidebar.
* The artwork is passed in, not discovered. The default is the tool's own
  synthesised test picture, so a run with no arguments never reaches for user
  material; ``--image`` points it at a real one when the tutorial wants to
  show what the styles actually do to a character illustration.

Usage: python tools/make_tutorial_shots.py --outdir out/tutorial_shots
       python tools/make_tutorial_shots.py --image "P:/PS抠图/由比滨透明背景.png"

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


def setup(window, app, image: str, style: str | None = None, reading: str | None = None):
    """Load, configure, then import -- in that order, and the order matters.

    The style describes the *next* import, so setting it after the import
    changes the panel's preview but leaves the canvas holding the previous
    style. That is correct behaviour, not a quirk to work around: get the
    sequence wrong here and the screenshot shows one style in the preview box
    and another on the canvas. The 读法 is the same kind of control and is set
    the same way; ``reading`` takes a key from ``g.READING_OPTIONS``.
    """
    panel = g.show_preprocess(window)
    panel.load_image(image)
    if style is not None:
        names = [name for name, _ in g.STYLE_OPTIONS]
        panel.style_combo.setCurrentIndex(names.index(style))
    if reading is not None:
        order = [key for key, _label, _hint in g.READING_OPTIONS]
        panel.reading_combo.setCurrentIndex(order.index(reading))
    panel.import_to_canvas()
    window.design_canvas.fit_view()
    app.processEvents()
    return panel


def state_main(window, app, image):
    """The resting state: empty board, nothing imported, no panels open."""
    window.design_canvas.fit_view()
    app.processEvents()


def state_lineart(window, app, image):
    panel = setup(window, app, image, style="线稿")
    panel.place_top_right()
    panel.raise_()


def state_boardart(window, app, image):
    panel = setup(window, app, image, style=g.BOARD_ART_STYLE)
    panel.place_top_right()
    panel.raise_()


def state_board(window, app, image):
    setup(window, app, image)
    panel = g.show_board_setup(window)
    panel.shape_combo.setCurrentIndex(g.SHAPE_CHOICES.index("rounded"))
    panel.raise_()


def state_irregular(window, app, image):
    """The cut-out outline -- the shape that makes an illustration a standee.

    Given its own screenshot because it is the reason this tool exists, and it
    is the one state that cannot be judged from any other picture: the board
    edge here is the artwork's own silhouette, not a preset.
    """
    panel = setup(window, app, image)
    panel.hide()
    board_panel = g.show_board_setup(window)
    board_panel.shape_combo.setCurrentIndex(g.SHAPE_CHOICES.index("irregular"))
    board_panel.base_combo.setCurrentIndex(
        g.IRREGULAR_BASES.index("trace")
    )
    board_panel.raise_()
    app.processEvents()
    window.design_canvas.fit_view()
    app.processEvents()


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
    # Save into whichever library this window was given, so the sidebar keeps
    # exactly one row instead of inheriting names from an earlier state's run.
    g.save_draft(
        window.design_canvas.board,
        str(window._draft_store.root / "示例板.pcbdraft"),
    )
    g.refresh_workspace_state(window)
    app.processEvents()


def state_xdog(window, app, image):
    """The XDoG reading on the canvas: the finest of the five.

    Shown on the canvas rather than in a panel because the difference between
    the readings *is* the drawing they produce -- a dropdown screenshot would
    show five names and nothing about which one keeps a hair strand.
    """
    panel = setup(window, app, image, style="线稿", reading=g.READING_XDOG)
    panel.place_top_right()
    panel.raise_()


def state_ridge(window, app, image):
    """The ridge reading: bolder, continuous strokes."""
    panel = setup(window, app, image, style="线稿", reading=g.READING_RIDGE)
    panel.place_top_right()
    panel.raise_()


def state_presteps(window, app, image):
    """The 图像预处理 section, which is where 超分 and 滤镜 live.

    The panel is scrolled so 5-7 are in view (the window grab alone would show
    only the upload box): the point of the shot is *where* the new controls
    are, and they sit below the fold in every other screenshot.
    """
    panel = g.show_preprocess(window)
    panel.load_image(image)
    app.processEvents()
    panel.resize(panel.width(), 760)
    panel.place_top_right()
    panel.raise_()
    app.processEvents()
    scroll = panel._scroll.verticalScrollBar()
    scroll.setValue(int(scroll.maximum() * 0.62))
    app.processEvents()


STATES = {
    "01-interface": state_main,
    "02-lineart": state_lineart,
    "03-boardart": state_boardart,
    "04-selected": state_selected,
    "05-board": state_board,
    "06-standee": state_standee,
    "07-drafts": state_drafts,
    "08-irregular": state_irregular,
    "09-xdog": state_xdog,
    "10-ridge": state_ridge,
    "11-presteps": state_presteps,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "out" / "tutorial_shots"))
    parser.add_argument(
        "--image",
        help=(
            "artwork to import; defaults to the tool's synthesised test picture "
            "so an argument-free run never touches user material"
        ),
    )
    parser.add_argument(
        "--variant",
        default="",
        help=(
            "name for a second artwork's set, appended to each filename as "
            "-<variant> so it can sit beside the first (e.g. --variant feiyu). "
            "Deliberately not a filename suffix: a leading dash would be read "
            "as another option."
        ),
    )
    parser.add_argument("--only", action="append", choices=sorted(STATES))
    parser.add_argument("--width", type=int, default=1420)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    image = args.image or artwork(SCRATCH)
    if not Path(image).exists():
        print(f"图片不存在：{image}", file=sys.stderr)
        return 2

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(
        f"QWidget {{ color: {g.TEXT}; }} QMainWindow {{ background: {g.WINDOW_BG}; }}"
    )

    from pcbart.draft import DraftStore

    wanted = args.only or sorted(STATES)
    for name in wanted:
        window = g.build_window()
        # A library per state, not one shared folder: 07-drafts saves a row,
        # and a shared folder would leave that row sitting in every later
        # screenshot as though the user had already saved something.
        window._draft_store = DraftStore(SCRATCH / "library" / name)
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

        target = outdir / (
            f"{name}-{args.variant}.png" if args.variant else f"{name}.png"
        )
        # Only overwrite the states this run actually renders. Wiping the whole
        # folder first would make `--only 03-boardart` delete the other seven
        # pictures, which is a surprising way to lose the set you just made.
        window.grab().save(str(target))
        print(f"{name} -> {target}")
        window.hide()
        window.deleteLater()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
