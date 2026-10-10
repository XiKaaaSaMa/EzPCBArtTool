"""Profile one canvas repaint on a finished card, to find what a drag costs.

A drag emits a move event per frame and each one repaints the whole canvas, so
whatever is slow here is what "拖动有点卡" is made of. This builds the card the
way the interface does and times each paint phase separately, then repeats the
whole thing with cProfile.

The numbers are wall-clock on the machine it runs on; the *ranking* is the
point, not the absolute milliseconds.

Usage::

    python tools/diagnose_drag_perf.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from pcbart.paths import asset_root  # noqa: E402

LIBRARY = asset_root() / "明日方舟"


def _prepare_environment(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def build_card(window, app):
    """A finished card, through the same panel calls the interface makes."""
    from pcbart import badge_card

    import gui_mockup as g

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    content = g.show_template_content(window, board)
    content.refresh()
    content.apply_operator("能天使")
    content._insert_card_art(announce=False)
    content.close()

    panel = g.show_preprocess(window)
    figure = LIBRARY / "06-立绘-精英0" / "立绘 能天使 1.png"
    panel.load_image(str(figure))
    panel.import_to_canvas()
    panel.close()
    app.processEvents()
    return board


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_perf")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)

    _prepare_environment(args.outdir)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import gui_mockup as g

    window = g.build_window()
    window.resize(1680, 1020)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    board = build_card(window, app)
    canvas = window.design_canvas
    canvas.fit_view()
    app.processEvents()

    # Move the selection the way a drag does, so the numbers include the extra
    # work a selected object brings (backdrop, frame, handles).
    placement = board.placements[-1]
    canvas.selected = placement
    app.processEvents()

    image = QImage(canvas.size(), QImage.Format_RGB32)

    def one_frame(index: int) -> None:
        placement.x_mm = 0.01 * index
        canvas.render(image)

    # Warm up: first paint pays for caches that a real drag does not.
    for index in range(5):
        one_frame(index)
    app.processEvents()

    start = time.perf_counter()
    for index in range(args.frames):
        one_frame(index)
    elapsed = time.perf_counter() - start
    per_frame = elapsed / args.frames * 1000.0
    print(f"整帧：{per_frame:.2f} ms/帧　（{args.frames} 帧共 {elapsed:.3f}s）")
    print(f"理论上限：{1000.0 / per_frame:.0f} fps")

    # Break the frame into its phases, by calling the paint helpers directly.
    from PySide6.QtGui import QPainter

    phases = [
        ("网格 _paint_grid", canvas._paint_grid),
        ("选中底板 _paint_selection_backdrop", canvas._paint_selection_backdrop),
        ("板面内容 _paint_board_content", canvas._paint_board_content),
        ("挖槽 _paint_cutouts", canvas._paint_cutouts),
        ("坐标轴 _paint_axes", canvas._paint_axes),
        ("选中框 _paint_selection_frame", canvas._paint_selection_frame),
        ("板框 _paint_board", canvas._paint_board),
        ("标尺 _paint_rulers", canvas._paint_rulers),
    ]
    canvas._recompute_plot()
    print()
    print("各阶段（每阶段单独跑 10 次取平均）：")
    for label, func in phases:
        runs = 10
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        begin = time.perf_counter()
        for _ in range(runs):
            func(painter)
        cost = (time.perf_counter() - begin) / runs * 1000.0
        painter.end()
        print(f"  {label:36s} {cost:7.3f} ms")

    # And inside the board content, which is where the card's artwork lives.
    print()
    print("_paint_board_content 内部（10 次平均）：")
    inner = [
        ("decorations_for", lambda: __import__("pcbart.badge_art", fromlist=["x"]).decorations_for(board)),
        ("visible_placements", board.visible_placements),
        ("visible_texts", board.visible_texts),
    ]
    for label, func in inner:
        runs = 20
        begin = time.perf_counter()
        for _ in range(runs):
            func()
        cost = (time.perf_counter() - begin) / runs * 1000.0
        print(f"  {label:36s} {cost:7.3f} ms")

    if args.profile:
        print()
        profiler = cProfile.Profile()
        profiler.enable()
        for index in range(args.frames):
            one_frame(index)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(24)
        print(stream.getvalue())

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
