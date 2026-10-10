"""Check that a finished 通行证 carries every element it is supposed to.

The card is more than an outline: the template supplies fixed artwork (border,
wordmark, crest, hatching, mask openings), the operator supplies four variable
elements (二维码 / 条形码 / 职业图标 / 阵营 logo), and the panel supplies the
wording. A missing piece does not raise -- it just is not on the board -- so
this walks the design and prints what is there and what is not, from the
document rather than from a rendered image.

Run after touching the extractor, the content builder or the exporters::

    python tools/audit_badge_elements.py
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
from pcbart.paths import asset_root  # noqa: E402

LIBRARY = asset_root() / "明日方舟"


def _prepare_environment(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operator", default="能天使")
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_audit")
    args = parser.parse_args(argv)

    _prepare_environment(args.outdir)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card, badge_content, badge_art
    from pcbart.design_layers import LAYERS, LAYER_PAINT_ORDER

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
    assert content.apply_operator(args.operator), args.operator
    assert content._insert_card_art(announce=False), "卡面素材没有放上去"
    content.close()

    # The illustration is the user's own import, not part of the card's
    # library, so it is loaded from the full library the way the panel does.
    panel = g.show_preprocess(window)
    figure = LIBRARY / "06-立绘-精英0" / f"立绘 {args.operator} 1.png"
    if figure.exists():
        panel.load_image(str(figure))
        panel.import_to_canvas()
    panel.close()
    app.processEvents()

    print(f"== 通行证元素核对（{args.operator}）==")
    print()

    # ------------------------------------------------------- fixed artwork
    decorations = badge_art.decorations_for(board)
    print(f"模板固定装饰：{len(decorations)} 项")
    by_layer: dict[str, int] = {}
    for item in decorations:
        key = badge_art.layer_key_for_id(int(item.get("layer", 0)))
        by_layer[key] = by_layer.get(key, 0) + 1
    for key in LAYER_PAINT_ORDER:
        spec = next((item for item in LAYERS if item.key == key), None)
        if spec is None:
            continue
        print(f"    {spec.name:12s} {by_layer.get(key, 0):4d} 项")

    # The one fixed element that must be there and is easy to lose.
    emblem = sum(1 for item in decorations if badge_art.is_rhodes_emblem(item))
    print(f"    其中罗德岛标志：{emblem} 项（应当 > 0）")

    print()
    print("卡面文字：")
    for item in board.texts:
        text = item.resolved_text(
            name=board.operator_name, number=board.operator_number
        )
        print(f"    {item.label:22s} {text or '（空）':32s} {item.layer}")
    print(f"    共 {len(board.texts)} 行")

    print()
    print("挖槽：")
    for item in board.cutouts:
        print(f"    {item.label:12s} {item.shape:9s} {item.width_mm} × {item.height_mm} mm")

    # --------------------------------------------------- variable elements
    art = badge_content.badge_art(board=board, number=board.operator_number)
    print()
    print("干员素材四要素：")
    checks = (
        ("二维码", art.qr),
        ("条形码", art.barcode),
        ("职业图标", art.profession),
        ("阵营 logo", art.faction),
    )
    for label, placement in checks:
        if placement is None:
            print(f"    {label:10s} 缺")
            continue
        source = Path(placement.processed_path or placement.path)
        print(
            f"    {label:10s} 有　{placement.width_mm:.3f} × "
            f"{placement.height_mm:.3f} mm　{placement.layer}"
            f"　{source.name if source.name else '（无文件）'}"
        )

    print()
    print("立绘：")
    figure_placements = [
        item for item in board.placements if item.import_id not in ("badge-art",)
        and item.kind == ""
    ]
    for placement in figure_placements:
        print(
            f"    {Path(placement.path).name}　{placement.width_mm:.3f} × "
            f"{placement.height_mm:.3f} mm　取景 "
            f"{[round(v, 4) for v in (placement.source_window or ())] or '整图'}"
        )
    if not figure_placements:
        print("    （未导入）")

    # ------------------------------------------------------------- exports
    from pcbart.gerber import export_gerber, GERBER_LAYERS

    report = export_gerber(board, args.outdir / "gerber", title="audit", archive=False)
    names = {Path(str(entry["path"])).suffix.upper() for entry in report["files"]}
    print()
    print("Gerber 各层：")
    for spec in GERBER_LAYERS:
        suffix = Path(spec.file_name).suffix.upper()
        mark = "有" if suffix in names else "**缺**"
        print(f"    {spec.label:12s} {suffix:5s} {mark}")
    print(f"    板框 GKO      {'有' if '.GKO' in names else '**缺**'}")
    if report.get("skipped"):
        print(f"    ⚠️ 空层：{report['skipped']}")

    manifest = {
        "operator": board.operator_name,
        "decorations": len(decorations),
        "by_layer": by_layer,
        "emblem": emblem,
        "texts": len(board.texts),
        "cutouts": len(board.cutouts),
        "elements": {
            label: placement is not None for label, placement in checks
        },
        "figures": len(figure_placements),
        "gerber_layers": sorted(names),
    }
    path = args.outdir / "audit.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"wrote {path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
