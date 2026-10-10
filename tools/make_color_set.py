"""Export the whole illustration set as one project file, plus a contact sheet.

What it produces
----------------
Running this in an empty output folder gives, in one go:

* one ``.epro2`` per illustration, each a complete 通行证 (template artwork,
  wording, cutouts, the four operator elements, and the framed illustration);
* **one merged ``.epro2``** holding every card side by side, so the set can be
  compared without opening eight files;
* a contact sheet PNG rendered by the application itself, for a quick look.

Everything goes through the real window -- the template panel, the preprocessing
panel, the canvas's own framing -- so the files are what the application makes,
not a fixture assembled around it. See :mod:`pcbart.merge` for how the merge
works and why a uniform translation is the correct transform.

``--style`` picks the conversion, and each one lands where it belongs:

* **彩色丝印** keeps the picture's own colours; the EDA writes the colour job.
* **线稿** becomes a black-and-white **picture** on the silkscreen. The layer is
  chosen for the caller (``--presentation 顶层丝印层``) because the EDA's
  embedded-object API accepts silk only -- asking for copper is accepted and
  then silently moved to silk, so a line-art card left on the default 顶层
  copper presentation would come out as traced fill regions instead of a
  picture.
* **板画** stays two traced tones, on silk and copper together.

Usage::

    python tools/make_color_set.py --outdir out/2026.10.8test
    python tools/make_color_set.py --outdir out/2026.10.8test --style 线稿
    python tools/make_color_set.py --outdir out/2026.10.8test --style 线稿 \
        --figure "精0|立绘 能天使 1.png"
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

from make_color_test_file import LIBRARY, STAGES  # noqa: E402

#: The default set: 精0 and 精2, four operators each, chosen to span the range
#: the library actually contains -- dark and light, near-grey and saturated,
#: human and machine, plain and densely detailed.
DEFAULT_FIGURES: tuple[tuple[str, str], ...] = (
    ("精0", "立绘 能天使 1.png"),
    ("精0", "立绘 陈 1.png"),
    ("精0", "立绘 艾雅法拉 1.png"),
    ("精0", "立绘 银灰 1.png"),
    ("精2", "立绘 银灰 2.png"),
    ("精2", "立绘 陈 2.png"),
    ("精2", "立绘 艾雅法拉 2.png"),
    ("精2", "立绘 Miss.Christine 2.png"),
)


def _parse_figures(entries: list[str], default_stage: str):
    """``["精2|立绘 陈 2.png", "立绘 甲 1.png"]`` -> ``[(stage, name), ...]``."""
    out = []
    for entry in entries:
        stage, sep, name = entry.partition("|")
        if not sep:
            stage, name = default_stage, entry
        if stage not in STAGES:
            raise SystemExit(f"未知档位「{stage}」，只能是 {'/'.join(sorted(STAGES))}")
        out.append((stage, name))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, required=True, help="产物目录")
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="[档位|]文件名",
        help="立绘文件名；可重复。不给则导出内置的八张",
    )
    parser.add_argument("--stage", default="精0", choices=sorted(STAGES))
    parser.add_argument(
        "--style",
        default="彩色丝印",
        help="图像处理风格：彩色丝印 / 线稿 / 板画",
    )
    parser.add_argument(
        "--presentation",
        default=None,
        help="呈现方式（下拉里的文字）。线稿默认「顶层丝印层」，"
        "因为图片对象在 EDA 里只能落在丝印层",
    )
    parser.add_argument("--columns", type=int, default=4, help="合并文件的网格列数")
    parser.add_argument("--gap-mm", type=float, default=8.0, help="卡片间距")
    parser.add_argument("--cell-width", type=int, default=460, help="总览图每格宽度")
    args = parser.parse_args(argv)

    figures = (
        _parse_figures(args.figure, args.stage) if args.figure else DEFAULT_FIGURES
    )

    # ⚠️ **线稿 has to go on a silkscreen layer to leave as a picture.** The
    # EDA's embedded-object API accepts silk only -- a request for copper is
    # accepted and then silently moved to silk -- so a line-art card on the
    # default 顶层 (copper) presentation comes back as traced fill regions
    # instead of the black-and-white picture the user asked for. Choosing the
    # layer here rather than asking the user to remember it keeps the two
    # matching.
    presentation = args.presentation
    if presentation is None:
        presentation = "顶层丝印层" if args.style == "线稿" else None

    label = {"彩色丝印": "彩印", "线稿": "线稿", "板画": "板画"}.get(
        args.style, args.style
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    # The window saves unwritten work when it closes; pointing the draft library
    # at the output folder keeps this run out of the user's real sidebar.
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card, export, merge

    import gui_mockup as g

    from make_color_contact_sheet import _tile

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(1400, 900)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    written: list[Path] = []
    captured: list[tuple[str, Path]] = []
    shots = args.outdir / "_canvas"
    shots.mkdir(parents=True, exist_ok=True)

    for index, (stage, name) in enumerate(figures):
        folder, region_key = STAGES[stage]
        figure = LIBRARY / folder / name
        if not figure.exists():
            print(f"跳过 {stage} {name}：找不到 {figure}", file=sys.stderr)
            continue
        page = Path(name).stem.removeprefix("立绘 ").rsplit(" ", 1)[0].strip()

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
            if style == args.style
        )
        panel.style_combo.setCurrentIndex(style_index)
        app.processEvents()
        if presentation:
            index = panel.layer_combo.findText(presentation)
            if index < 0:
                raise SystemExit(
                    f"呈现方式里没有「{presentation}」："
                    + " / ".join(
                        panel.layer_combo.itemText(i)
                        for i in range(panel.layer_combo.count())
                    )
                )
            panel.layer_combo.setCurrentIndex(index)
            app.processEvents()
        if not panel.import_to_canvas():
            raise SystemExit(f"导入失败：{figure}")
        panel.close()

        canvas = window.design_canvas
        placement = board.placements[-1]
        region = next(
            item for item in badge_card.BADGE_REGIONS if item.key == region_key
        )
        canvas.selected = placement
        canvas.start_framing(placement, region)
        # ⚠️ Leave the framing gesture. ``start_framing`` is the interactive
        # entry: it also sets ``canvas.framing``, which dims everything outside
        # the window and turns dragging into a crop. A finished card is not in
        # that mode, and a shot taken here would show the crop overlay. The
        # application's own automatic import uses ``frame_into_card_region``,
        # which applies the same geometry without entering the mode; this script
        # needs the explicit region, so it clears the mode by hand.
        canvas.framing = None
        canvas.selected = None
        canvas.fit_view()
        app.processEvents()

        image = QImage(canvas.size(), QImage.Format_RGB32)
        canvas.render(image)
        shot = shots / f"card-{index:02d}-{page}-{stage}.png"
        image.save(str(shot))
        captured.append((f"{page} {stage}", shot))

        target = args.outdir / f"{Path(name).stem}_{label}.epro2"
        report = export.export_design(board, target)
        written.append(target)
        print(
            f"{stage}  {page}：{placement.layer}  {placement.width_mm:.3f} × "
            f"{placement.height_mm:.3f} mm　彩图 {report['color_pictures']} 张"
            f"　填充 {report['copper_shapes']} 块　→ {target.name}"
        )

    if not written:
        print("什么都没导出。", file=sys.stderr)
        return 1

    # ---- everything into one file, and one picture of the whole set
    merged_path = args.outdir / f"立绘{label}测试_全部{len(written)}张.epro2"
    summary = merge.merge_projects(
        written,
        merged_path,
        title=merged_path.stem,
        columns=args.columns,
        gap_mm=args.gap_mm,
    )

    sheet = args.outdir / f"立绘{label}测试_全部{len(written)}张_总览.png"
    _tile(captured, sheet, columns=args.columns, cell_width=args.cell_width)

    print()
    print(f"单卡：{len(written)} 个 .epro2")
    print(
        f"合并：{merged_path.name}　{summary['boards']} 张卡　"
        f"网格 {summary['columns']}×{summary['rows']}　"
        f"整版 {summary['panel_mm'][0]:.1f} × {summary['panel_mm'][1]:.1f} mm"
    )
    print(f"总览：{sheet.name}")
    sys.stdout.flush()
    # Qt's teardown of a window that owns a cached floating panel hangs; the
    # files are already on disk, so exit rather than wait for a clean shutdown.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
