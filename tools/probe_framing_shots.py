"""Render the canvas after a PLAIN import (no manual framing), all three styles.

⚠️ The numbers can be right while the picture is still wrong, so this exists to
look at it. Each style gets the same figure, imported through the real panel
with nothing else done to it -- which is exactly the flow the user reported
failing.

Output: ``out/_framing_auto/<style>.png``.

Usage::

    python tools/probe_framing_shots.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["PCBART_DRAFTS_DIR"] = str(Path(tempfile.gettempdir()) / "pcbart-fsa-drafts")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools import gui_mockup as G  # noqa: E402

from pcbart.paths import asset_root  # noqa: E402

#: Resolved under ``PCBART_ASSET_ROOT``; the library is gigabytes and lives
#: outside the repository, so this may not be a literal path.
FIGURE = (
    asset_root()
    / "明日方舟"
    / "09-立绘-皮肤"
    / "立绘 缄默德克萨斯 幽兰秘辛.png"
)
OUT = ROOT / "out" / "_framing_auto"


def run(app, label: str, style: str, index: int) -> None:
    """One style, imported through the real panel, then a canvas screenshot.

    ⚠️ The window is **never closed**. ``window.close()`` runs the app's own
    unsaved-work check, and this script leaves the document dirty by design --
    so closing it pops the 保存草稿 dialog once per style and then hangs on it.
    That is the application behaving correctly; a screenshot script must not
    put itself in the way of it. The process exits hard at the end instead.
    """
    window = G.build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    from pcbart import badge_card

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    panel = G.show_preprocess(window)
    app.processEvents()
    panel.load_image(str(FIGURE))
    panel.style_combo.setCurrentIndex(index)
    app.processEvents()
    panel.import_to_canvas()          # the whole flow: no manual framing
    app.processEvents()
    panel.close()
    app.processEvents()

    canvas = window.design_canvas
    canvas.fit_view()
    app.processEvents()
    target = OUT / f"{label}.png"
    canvas.grab().save(str(target))

    print(f"  {label:6s} -> {target.name}")
    for item in board.placements:
        window_box = item.source_window
        print(
            f"      {item.layer_key:12s} region={item.region or '（无）':6s}"
            f" 窗口={tuple(round(v, 3) for v in window_box) if window_box else '（无）'}"
        )
    # Not ``window.close()``: see this function's docstring.
    app.processEvents()


def main() -> int:
    from tools.gui_mockup import BOARD_ART_STYLE, COLOR_STYLE, STYLE_OPTIONS

    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    keys = [key for key, _hint in STYLE_OPTIONS]
    print(f"素材 {FIGURE.name}")
    run(app, "彩色丝印", COLOR_STYLE, keys.index(COLOR_STYLE))
    run(app, "线稿", "线稿", keys.index("线稿"))
    run(app, "板画", BOARD_ART_STYLE, keys.index(BOARD_ART_STYLE))
    # Hard exit, the same recipe ``make_badge_card_shots`` uses: Qt's teardown
    # of a window that owns a cached floating panel does not settle. Every
    # screenshot is already on disk by here, so there is nothing to wait for.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
