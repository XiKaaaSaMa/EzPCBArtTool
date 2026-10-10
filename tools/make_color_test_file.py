"""Build complete 通行证 cards with a colour illustration, for the EDA.

Produces the files to hand to EasyEDA Pro when checking the colour-silkscreen
path. Everything goes through the real interface -- the template panel, the
preprocessing panel, the canvas's own framing -- so the file is what the
application actually makes, not a fixture assembled around it.

⚠️ **The framing step is part of the flow, not optional.** An earlier version of
this script stopped after importing the illustration, and the file it produced
carried the picture at its 76.20 mm import size instead of the 38.861 x 56.324
region -- the crop the user asked for was never applied.

Usage::

    # The two built-in cases (能天使 精0 / 银灰 精2)
    python tools/make_color_test_file.py

    # Any number of illustrations, by library name
    python tools/make_color_test_file.py --figure "立绘 陈 2.png" --figure "立绘 艾雅法拉 2.png"
    python tools/make_color_test_file.py --stage 精0 --figure "立绘 银灰 1.png" --outdir out/2026.10.8test

``--figure`` takes a file name from one of the illustration folders; the stage
(精0 / 精2) picks the folder and the card size. The framing region follows the
stage, so a 精2 illustration is placed in the long region even when the card
carries a small-format picture.

⚠️ **``--zoom`` defaults to 1.0, which is what the application itself does.**
It used to default to 1.8, and that silently cropped the head off a 精0
illustration -- 能天使 came out as a torso with no face. The card's opening
already selects which part of the picture is shown; zooming further is a
framing choice the *user* makes, not something a test export should assume on
their behalf.
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

from pcbart.paths import asset_root

#: ⚠️ Resolved through ``PCBART_ASSET_ROOT`` rather than written out. The
#: library lives outside the repository and on a different drive on some
#: machines, so a literal path here would make the tool unusable for anyone
#: else and would bake one developer's directory layout into the source.
LIBRARY = asset_root() / "明日方舟"

#: ``(label, stage, operator page, folder, file name, region key)``.
CASES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("精0", "精0", "能天使", "06-立绘-精英0", "立绘 能天使 1.png", "small"),
    ("精2", "精2", "银灰", "08-立绘-精英2", "立绘 银灰 2.png", "large"),
)

#: Which folder and card region each stage uses. The stage decides the card
#: format -- a 精0 illustration belongs on a short card, a 精2 one on a tall
#: card -- so ``--figure`` only has to name the picture.
STAGES: dict[str, tuple[str, str]] = {
    "精0": ("06-立绘-精英0", "small"),
    "精2": ("08-立绘-精英2", "large"),
}


def _figure_stem(file_name: str) -> str:
    """``立绘 银灰 2.png`` -> ``银灰``, the operator page to look up."""
    stem = Path(file_name).stem
    if stem.startswith("立绘 "):
        stem = stem[len("立绘 ") :]
    return stem.rsplit(" ", 1)[0].strip()


def custom_cases(figures: list[str], stage: str):
    """Turn ``--figure`` names into the same tuple shape as :data:`CASES`.

    The operator is resolved from the file name rather than typed separately:
    naming one thing twice is a way for the two to disagree, and the card would
    then carry someone else's QR code with no visible sign of the mismatch.
    """
    folder, region = STAGES[stage]
    built = []
    for name in figures:
        page = _figure_stem(name)
        built.append((stage, stage, page, folder, name, region))
    return tuple(built)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "彩印测试")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="取景窗口放大倍率（1.0 = 产品自身的取景，不额外放大）",
    )
    parser.add_argument("--only", default="", help="只做一个：精0 / 精2")
    parser.add_argument(
        "--stage",
        default="精2",
        choices=sorted(STAGES),
        help="--figure 用的立绘档位（决定文件夹与卡片尺寸）",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="文件名",
        help="立绘文件名，可重复；给了它就不用内置的两个用例",
    )
    args = parser.parse_args(argv)

    if args.figure:
        cases = custom_cases(args.figure, args.stage)
    else:
        cases = CASES

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card, export

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(1400, 900)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    written: list[Path] = []
    for index, (label, stage, page, folder, name, region_key) in enumerate(cases):
        if args.only and args.only != label:
            continue
        figure = LIBRARY / folder / name
        if not figure.exists():
            print(f"跳过 {label} {name}：找不到 {figure}", file=sys.stderr)
            continue
        if index:
            g.new_project_tab(window)
            app.processEvents()

        board = window.design_canvas.board
        board.outline = badge_card.badge_spec(stage)
        board.outline_ready = True
        board.texts = badge_card.badge_texts()
        board.cutouts = badge_card.badge_cutouts()

        content = g.show_template_content(window, board)
        content.refresh()
        if not content.apply_operator(page):
            raise SystemExit(f"资料库里没有「{page}」")
        content._insert_card_art(announce=False)
        content.close()
        app.processEvents()

        panel = g.show_preprocess(window)
        if not panel.load_image(str(figure)):
            raise SystemExit(f"读不出这张图：{figure}")
        style_index = next(
            position
            for position, (style, _hint) in enumerate(g.STYLE_OPTIONS)
            if style == g.COLOR_STYLE
        )
        panel.style_combo.setCurrentIndex(style_index)
        if not panel.import_to_canvas():
            raise SystemExit(f"导入失败：{figure}")
        panel.close()

        # ---- the step that was missing: put the illustration in its region
        canvas = window.design_canvas
        placement = board.placements[-1]
        region = next(
            item for item in badge_card.BADGE_REGIONS if item.key == region_key
        )
        canvas.selected = placement
        canvas.start_framing(placement, region)
        if args.zoom > 1.0:
            canvas.window_zoom(args.zoom)
        app.processEvents()

        # Named after the illustration, not the stage: a batch shares one
        # stage, and two cards from it must not write the same file.
        target = args.outdir / f"{Path(name).stem}.epro2"
        report = export.export_design(board, target)
        written.append(target)

        window_box = placement.source_window or ()
        print(
            f"{label}  {page}：立绘 {placement.width_mm:.3f} × "
            f"{placement.height_mm:.3f} mm　取景 "
            f"{[round(v, 4) for v in window_box]}　彩图 "
            f"{report['color_pictures']} 张　→ {target.name}"
        )

    if not written:
        print("什么都没生成。", file=sys.stderr)
        return 1
    print()
    for path in written:
        print(f"wrote {path}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
