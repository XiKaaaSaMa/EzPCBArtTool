"""Render screenshots of a finished 通行证 card with every asset in place.

Two example operators are driven through the real window: the template is
applied, the operator is chosen, and the artwork is inserted through the same
panel method the button calls. Nothing here reaches past the UI to draw the card
itself -- a screenshot assembled from the model would prove the model works and
say nothing about whether the panel wires it up.

**The draft library is redirected.** The window saves unwritten work when it
closes, and driving it against the real profile put a dozen ``未命名`` rows into
the developer's sidebar during this project's own debugging. ``PCBART_DRAFTS_DIR``
points at a scratch folder instead.

Usage: python tools/make_badge_card_shots.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SCRATCH = ROOT / "out" / "_badge_card_shots"
DEFAULT_OPERATORS = ("阿米娅", "山")


def _prepare_environment(scratch: Path) -> None:
    """Keep the run out of the user's real draft library and asset overrides."""
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def build_card(window, app, name: str):
    """Apply the template, pick the operator, insert the artwork -- via the UI."""
    from PySide6.QtCore import Qt

    from pcbart import badge_card

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    import gui_mockup as g

    panel = g.show_template_content(window, board)
    panel.refresh()
    assert panel.apply_operator(name), f"数据库里没有「{name}」"
    assert panel._insert_card_art(announce=False), "素材没有放进卡面"
    app.processEvents()
    return board, panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=SCRATCH)
    parser.add_argument(
        "--operators",
        default=",".join(DEFAULT_OPERATORS),
        help="comma-separated names; one window is built per name",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args(argv)

    _prepare_environment(args.outdir)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    names = [item.strip() for item in args.operators.split(",") if item.strip()]
    written: list[Path] = []

    # One window for the whole run: building a second window after closing the
    # first hangs, because the panel cache and the undo stack both hold
    # references to the closed one. Re-pointing a single window at a fresh board
    # is what the tab bar does, so it is the path the app already exercises.
    window = g.build_window()
    window.resize(args.width, args.height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    for index, name in enumerate(names):
        if index:
            g.new_project_tab(window)
            app.processEvents()
        board, panel = build_card(window, app, name)
        g.show_layers(window, focus=False)
        app.processEvents()

        canvas = window.design_canvas
        canvas.fit_view()
        app.processEvents()

        full = args.outdir / f"card-{name}.png"
        window.grab().save(str(full))
        written.append(full)

        # The canvas alone, so the card is legible rather than a few hundred
        # pixels wide inside a window shot. Rendered from the widget rather than
        # grabbed from the window: a grab of a sub-rectangle depends on where the
        # widget happens to sit, which changes with the dock and panel layout.
        close = args.outdir / f"card-{name}-canvas.png"
        image = QImage(canvas.size(), QImage.Format_RGB32)
        canvas.render(image)
        image.save(str(close))
        written.append(close)

        placements = [
            item for item in board.placements if item.import_id == "badge-art"
        ]
        print(f"{name}: {len(placements)} 个素材对象, 文字 {len(board.texts)} 行")
        panel.close()
        app.processEvents()

    for path in written:
        print(f"wrote {path}")
    # Qt's teardown of a window that owns a cached floating panel hangs: the
    # panel keeps a reference to the window and the destroyed-signal chain never
    # settles. The screenshots are already on disk, so the process exits here
    # instead of waiting for a clean shutdown that this script does not need.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
