"""Render a zoomed-in screenshot of one operator's card artwork, for checking.

The full-window shot is what the user sees, but a 3.2 mm QR symbol inside a
100 mm card is about twenty pixels across on screen -- too small to judge
whether it landed on the right spot and the right way up. This renders the
card at a fixed scale and crops the middle, so the four operator assets can be
looked at directly.

Usage: python tools/make_badge_art_zoom.py --operator 阿米娅 --out x.png
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operator", default="阿米娅")
    parser.add_argument("--out", type=Path, default=SCRATCH / "zoom.png")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--zoom", type=float, default=3.5)
    args = parser.parse_args(argv)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(SCRATCH / "drafts")

    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    import gui_mockup as g
    from pcbart import badge_card

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(args.width, args.height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    panel = g.show_template_content(window, board)
    panel.refresh()
    assert panel.apply_operator(args.operator), args.operator
    assert panel._insert_card_art(announce=False)
    app.processEvents()

    canvas = window.design_canvas
    canvas.fit_view()
    app.processEvents()
    # Zoom about the point the assets sit in rather than the canvas centre: the
    # codes are within a few millimetres of each other near the bottom left, and
    # a centred view of the whole card shows the illustration field instead.
    from pcbart import badge_content

    art = badge_content.badge_art(board=board)
    if art.placements:
        focus_x = sum(item.x_mm for item in art.placements) / len(art.placements)
        focus_y = sum(item.y_mm for item in art.placements) / len(art.placements)
        # Anchor on the widget centre: the drawing area is inset by the rulers
        # and the margin, so ``rect().center()`` is not the point the board is
        # centred on.
        anchor = canvas.rect().center()
        canvas.zoom_by(args.zoom, anchor)
        point = canvas.to_screen(focus_x, focus_y)
        centre = canvas._plot.center()
        # The delta is "where the point is" minus "where it should be": the pan
        # moves the view, so the point travels the opposite way.
        canvas.pan_by_pixels(centre.x() - point.x(), centre.y() - point.y())
    app.processEvents()

    image = QImage(canvas.size(), QImage.Format_RGB32)
    canvas.render(image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(args.out))
    print(f"wrote {args.out} ({image.width()}x{image.height()})")
    for item in board.placements:
        if item.import_id == "badge-art":
            print(
                f"  {item.layer}: ({item.x_mm:.3f}, {item.y_mm:.3f}) "
                f"{item.width_mm:.3f}x{item.height_mm:.3f} rot={item.rotation_deg}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
