"""Frame real illustrations on the 通行证 card and render what it looks like.

Framing is the one gesture where the picture outside its window has to stay
visible. A drag there pans the picture *inside* the window, and with everything
outside clipped away there is nothing on screen to judge the pan against. This
tool drives that path with real illustrations picked out of the full library
(``PCB艺术板工具素材库\\明日方舟\\`` -- the card's own library carries no figures),
saves a screenshot before and after a pan, and records the window each one
ended up with.

The window is written by the real canvas, not by this script: the screenshots
are what the panel and the canvas actually paint.

``PCBART_DRAFTS_DIR`` is redirected before any window exists, because closing a
driven window writes a draft and that must not land in the user's library.

Usage::

    python tools/preview_badge_figure.py ^
        --figure "…\\06-立绘-精英0\\立绘 能天使 1.png" ^
        --figure "…\\08-立绘-精英2\\立绘 银灰 2.png"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _prepare_environment(scratch: Path) -> None:
    """Keep the run out of the user's real draft library."""
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def _build_window(app, width: int, height: int):
    """The real window, shown off-screen so nothing pops up on the desktop."""
    from PySide6.QtCore import Qt

    import gui_mockup as g

    window = g.build_window()
    window.resize(width, height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    return window


def _import_figure(window, figure: Path, style: str):
    """Apply the card and import the figure the way the interface does.

    The picture goes through the real 图像预处理 panel -- load, style, then
    导入图片 -- rather than being placed by hand. The panel is what chooses the
    layer, computes the conversion and writes the processed file the canvas
    draws from, so bypassing it would leave the preview showing something the
    panel would never produce.
    """
    from pcbart import badge_card

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    import gui_mockup as g

    panel = g.show_preprocess(window)
    if not panel.load_image(str(figure)):
        raise SystemExit(f"读不出这张图：{figure}")
    index = next(
        (
            position
            for position, (name, _hint) in enumerate(g.STYLE_OPTIONS)
            if name == style
        ),
        None,
    )
    if index is None:
        raise SystemExit(f"没有这个风格：{style}")
    panel.style_combo.setCurrentIndex(index)
    if not panel.import_to_canvas():
        raise SystemExit(f"导入失败：{figure}")

    placements = window.design_canvas.board.placements
    if not placements:
        raise SystemExit("导入后画布上没有任何对象")
    placement = placements[-1]
    window.design_canvas.fit_view()
    panel.close()
    return placement


def _canvas_shot(canvas, path: Path) -> None:
    """The canvas alone, so the card is legible rather than a slice of a window."""
    from PySide6.QtGui import QImage

    image = QImage(canvas.size(), QImage.Format_RGB32)
    canvas.render(image)
    assert image.save(str(path)), f"截图写入失败：{path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--figure",
        action="append",
        required=True,
        help="illustration file; may be given more than once",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="small / large, one per figure in order; defaults to small then large",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_badge_figure")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument(
        "--style",
        default="彩色丝印",
        help="风格 to import with: 线稿 / 板画 / 彩色丝印",
    )
    parser.add_argument(
        "--pan",
        default="-110,-70",
        help="the framing pan to demonstrate, in screen pixels: dx,dy",
    )
    args = parser.parse_args(argv)

    pan_x, pan_y = (float(part) for part in args.pan.split(","))

    _prepare_environment(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card

    app = QApplication.instance() or QApplication([])
    report: list[dict] = []

    # One window for the whole run, switching tabs between figures: a second
    # window built after the first is closed hangs, because the floating-panel
    # cache and the undo stack still point at the closed one.
    window = _build_window(app, args.width, args.height)
    import gui_mockup as g

    for index, text in enumerate(args.figure):
        figure = Path(text)
        if not figure.exists():
            raise SystemExit(f"找不到立绘：{figure}")
        key = (
            args.region[index]
            if index < len(args.region)
            else ("small" if index % 2 == 0 else "large")
        )
        region = next(
            (item for item in badge_card.BADGE_REGIONS if item.key == key), None
        )
        if region is None:
            raise SystemExit(f"没有这个区域：{key}")
        if index:
            g.new_project_tab(window)
            app.processEvents()

        canvas = window.design_canvas
        placement = _import_figure(window, figure, args.style)

        canvas.selected = placement
        canvas.start_framing(placement, region)
        app.processEvents()
        before = args.outdir / f"figure-{key}-before.png"
        _canvas_shot(canvas, before)

        canvas._press_pos = QPointF(0.0, 0.0)
        canvas._press_window = placement.source_window
        canvas._pan_framed(QPointF(pan_x, pan_y))
        app.processEvents()
        after = args.outdir / f"figure-{key}-after.png"
        _canvas_shot(canvas, after)

        entry = {
            "figure": str(figure),
            "region": key,
            "region_label": region.label,
            "window": [round(value, 6) for value in placement.source_window or ()],
            "placement": {
                "x_mm": placement.x_mm,
                "y_mm": placement.y_mm,
                "width_mm": placement.width_mm,
                "height_mm": placement.height_mm,
                "layer": placement.layer,
            },
            "before": str(before),
            "after": str(after),
        }
        report.append(entry)
        print(
            f"{figure.name} → {region.label}　窗口 {entry['window']}　"
            f"截图 {after.name}"
        )

    manifest = args.outdir / "figures.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {manifest}")
    # Qt's teardown of a window that owns a cached floating panel does not
    # settle; the screenshots are already written, so the process ends here.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
