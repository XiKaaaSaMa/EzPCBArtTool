"""Measure where a framed illustration actually lands in each export.

The canvas clips an illustration to its region; whether the *exported* geometry
respects the same region is a separate question, and one that has been reported
wrong. This builds a card, frames a real illustration into a region, exports
both formats and prints the ink's bounding box against the region's own bounds.

Usage::

    python tools/diagnose_figure_region.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pcbart.paths import asset_root

#: ⚠️ Resolved through ``PCBART_ASSET_ROOT`` rather than written out. The
#: library lives outside the repository and on a different drive on some
#: machines, so a literal path here would make the tool unusable for anyone
#: else and would bake one developer's directory layout into the source.
LIBRARY = asset_root() / "明日方舟"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_region_check")
    parser.add_argument("--region", default="small", choices=["small", "large"])
    parser.add_argument("--style", default="彩色丝印")
    parser.add_argument("--zoom", type=float, default=1.8)
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card, export, gerber

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(1400, 900)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

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
    index = next(
        i for i, (name, _hint) in enumerate(g.STYLE_OPTIONS) if name == args.style
    )
    panel.style_combo.setCurrentIndex(index)
    panel.import_to_canvas()
    panel.close()

    placement = board.placements[-1]
    region = next(
        item for item in badge_card.BADGE_REGIONS if item.key == args.region
    )
    canvas = window.design_canvas
    canvas.selected = placement
    canvas.start_framing(placement, region)
    if args.zoom > 1.0:
        canvas.window_zoom(args.zoom)
    app.processEvents()

    window_box = placement.source_window
    print("== 取景状态 ==")
    print(f"  区域      {region.label}")
    print(f"  区域范围  x {region.bounds[0]:.3f} .. {region.bounds[2]:.3f}"
          f"    y {region.bounds[1]:.3f} .. {region.bounds[3]:.3f}")
    print(f"  放置尺寸  {placement.width_mm:.3f} × {placement.height_mm:.3f} mm"
          f"   中心 ({placement.x_mm:.3f}, {placement.y_mm:.3f})")
    print(f"  取景窗口  {[round(v, 4) for v in window_box or ()]}")

    from pcbart.artboard import drawn_size_mm

    width_px, height_px = canvas._pixel_size(placement)
    left, top, right, bottom = placement.source_rect(width_px, height_px)
    used = (right - left, bottom - top)
    drawn = drawn_size_mm(
        placement.width_mm, placement.height_mm, used[0] / used[1]
    )
    print(f"  源图      {width_px} × {height_px} px   取用 {used[0]} × {used[1]} px")
    print(f"  绘制尺寸  {drawn[0]:.3f} × {drawn[1]:.3f} mm（画布上实际画的）")

    # ------------------------------------------------------------ .epro2
    out = export.export_design(board, args.outdir / "card.epro2")
    print()
    print("== .epro2 ==")
    print(f"  copper_shapes {out.get('copper_shapes')}  layers {out.get('copper_layers')}")

    # ------------------------------------------------------------ Gerber
    report = gerber.export_gerber(
        board, args.outdir / "gerber", title="check", archive=False
    )
    print()
    print("== Gerber ==")
    for label in ("顶层铜", "顶层丝印", "底层丝印"):
        entry = next(
            (item for item in report["files"] if item["label"] == label), None
        )
        if entry is None:
            continue
        text = Path(str(entry["path"])).read_text(encoding="ascii")
        coords = [
            (int(m.group(1)) / 1e6, int(m.group(2)) / 1e6)
            for m in re.finditer(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)
        ]
        if not coords:
            print(f"  {label:8s} （空）")
            continue
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]
        print(
            f"  {label:8s} x {min(xs):8.3f} .. {max(xs):8.3f}"
            f"    y {min(ys):8.3f} .. {max(ys):8.3f}"
        )

    # ------------------------------------------------- the figure's own ink
    # The layers above carry the template's fixed artwork as well, so their
    # bounding box is the whole card and says nothing about the illustration.
    # Measure the placement's own geometry, with the card's decorations left
    # out of the board so nothing else can contribute.
    from pcbart.artboard import ArtBoard
    from pcbart.design_layers import layer as design_layer
    from pcbart.export import build_copper

    bare = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    bare.placements.append(placement)
    records, _report = build_copper(bare)
    wanted = design_layer(placement.layer_key).easyeda_id
    points = [
        point for layer_id, rings, _width in records if layer_id == wanted
        for ring in rings for point in ring
    ]
    print()
    print("== 立绘本身的墨迹（隔离测量）==")
    if not points:
        print("  （没有几何）")
    else:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        print(f"  墨迹范围  x {min(xs):8.3f} .. {max(xs):8.3f}"
              f"    y {min(ys):8.3f} .. {max(ys):8.3f}")
        print(f"  墨迹尺寸  {max(xs) - min(xs):.3f} × {max(ys) - min(ys):.3f} mm")
        out_x = max(
            placement.x_mm - placement.width_mm / 2 - min(xs),
            max(xs) - (placement.x_mm + placement.width_mm / 2),
        )
        out_y = max(
            placement.y_mm - placement.height_mm / 2 - min(ys),
            max(ys) - (placement.y_mm + placement.height_mm / 2),
        )
        print(f"  超出区域  x 方向 {out_x:+.3f} mm    y 方向 {out_y:+.3f} mm"
              f"    （正数=超出）")

    # What the region itself allows, for comparison.
    print()
    print("== 区域允许的范围 ==")
    print(
        f"  立绘区    x {placement.x_mm - placement.width_mm / 2:.3f}"
        f" .. {placement.x_mm + placement.width_mm / 2:.3f}"
        f"    y {placement.y_mm - placement.height_mm / 2:.3f}"
        f" .. {placement.y_mm + placement.height_mm / 2:.3f}"
    )

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
