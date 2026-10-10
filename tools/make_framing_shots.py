"""Render the 立绘取景 comparison on a *finished* 通行证, with all its facts.

Two states are captured from the same session, so the only difference between
them is whether the framing ghost is painted:

* 取景中 -- the window at full strength, the rest of the picture faint. This is
  what a drag is judged against, and the change this round.
* 结束取景 -- window only. This is what the tool drew before the change, and
  what the export writes.

The card is built the whole way through: the operator is chosen, the four card
elements (二维码 / 条形码 / 职业图标 / 阵营 logo) are inserted, the wordmark
lines are rendered, and only then is the illustration imported and framed. So
the screenshots show what a finished card looks like, not a blank template.

Every measurement the picture depends on is printed on the tile as well --
region size, the crop window in source fractions, the dimmed margin, the
placement's own numbers, layer and style -- because "looks right" is not
something a reader can check from a screenshot alone.

Output goes to ``out/_badge_framing_shots/``: one full-window shot per state
per figure, plus a 2x2 contact sheet.

``PCBART_DRAFTS_DIR`` is redirected before any window exists: driving the real
window writes a draft on close, and that must not reach the user's library.

Usage::

    python tools/make_framing_shots.py
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
from pcbart.paths import asset_root  # noqa: E402

LIBRARY = asset_root() / "明日方舟"

#: ``(label, operator page, library folder, file name, region key)``. 精0 goes
#: in the short window and 精2 in the tall one, which is what each was drawn for.
FIGURES: tuple[tuple[str, str, str, str, str], ...] = (
    ("精0", "能天使", "06-立绘-精英0", "立绘 能天使 1.png", "small"),
    ("精2", "银灰", "08-立绘-精英2", "立绘 银灰 2.png", "large"),
)


def _prepare_environment(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def _shot(source, path: Path):
    """One PNG of ``source``: the whole window, or just the canvas."""
    from PySide6.QtGui import QImage

    if hasattr(source, "render") and hasattr(source, "_plot"):
        image = QImage(source.size(), QImage.Format_RGB32)
        source.render(image)
    else:
        image = source.grab().toImage()
    if not image.save(str(path)):
        raise SystemExit(f"截图写入失败：{path}")
    return image


def _caption(image, title: str, details: list[str]):
    """A caption block above ``image``: the title, then the measured facts."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 19)
            small = ImageFont.truetype(candidate, 16)
            break
    else:  # pragma: no cover - only on a machine without either font
        font = small = None

    lines = [(title, font, (230, 236, 244))]
    lines += [(text, small, (156, 168, 182)) for text in details]

    line_height = 28
    height = 14 + line_height * len(lines)
    bar = Image.new("RGB", (image.width, image.height + height), (22, 26, 31))
    bar.paste(image, (0, height))
    draw = ImageDraw.Draw(bar)
    y = 8
    for text, face, colour in lines:
        draw.text((14, y), text, fill=colour, font=face)
        y += line_height
    return bar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "out" / "_badge_framing_shots"
    )
    parser.add_argument("--style", default="彩色丝印")
    parser.add_argument("--width", type=int, default=1680)
    parser.add_argument("--height", type=int, default=1020)
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.72,
        help=(
            "视图倍率。小于 1 让画面比卡片宽一圈，框外的暗影铺在卡外；"
            "大于 1 会把卡填满视口，反而看不见暗影"
        ),
    )
    parser.add_argument(
        "--window-zoom",
        type=float,
        default=1.8,
        help=(
            "取景窗口的放大倍率，等于用户在取景区里滚轮放大。不动它的话窗口"
            "常常占满整张图（精0 短区占 69%），框外只剩两三毫米、暗影看不出来"
        ),
    )
    parser.add_argument(
        "--canvas-only",
        action="store_true",
        help="只截画布，不截整个窗口（默认截整个窗口，含右侧属性栏）",
    )
    args = parser.parse_args(argv)

    _prepare_environment(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card, badge_content

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(args.width, args.height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    tiles: list[tuple[str, list[str], object]] = []
    for index, (grade, page, folder, name, region_key) in enumerate(FIGURES):
        figure = LIBRARY / folder / name
        if not figure.exists():
            raise SystemExit(f"找不到立绘：{figure}")
        region = next(
            item for item in badge_card.BADGE_REGIONS if item.key == region_key
        )
        if index:
            g.new_project_tab(window)
            app.processEvents()

        canvas = window.design_canvas
        board = canvas.board
        board.outline = badge_card.badge_spec()
        board.outline_ready = True
        board.texts = badge_card.badge_texts()
        board.cutouts = badge_card.badge_cutouts()

        # The card's own content first: operator, then the four elements. The
        # illustration is not one of them -- the card library carries none, so
        # it is imported from the full library just below.
        content = g.show_template_content(window, board)
        content.refresh()
        if not content.apply_operator(page):
            raise SystemExit(f"资料库里没有「{page}」")
        if not content._insert_card_art(announce=False):
            raise SystemExit(f"卡面素材没有放上去：{page}")
        content.close()
        app.processEvents()

        panel = g.show_preprocess(window)
        if not panel.load_image(str(figure)):
            raise SystemExit(f"读不出这张图：{figure}")
        style_index = next(
            (
                position
                for position, (style, _hint) in enumerate(g.STYLE_OPTIONS)
                if style == args.style
            ),
            None,
        )
        if style_index is None:
            raise SystemExit(f"没有这个风格：{args.style}")
        panel.style_combo.setCurrentIndex(style_index)
        if not panel.import_to_canvas():
            raise SystemExit(f"导入失败：{figure}")
        panel.close()

        placement = board.placements[-1]
        canvas.selected = placement
        canvas.start_framing(placement, region)
        # Push in, the way a user framing a portrait would: it is what leaves a
        # visible margin of picture outside the window to be dimmed.
        if args.window_zoom > 1.0:
            canvas.window_zoom(args.window_zoom)
        canvas.fit_view()
        canvas.zoom_by(args.zoom, canvas.rect().center())
        app.processEvents()

        width_px, height_px = canvas._pixel_size(placement)
        left, top, right, bottom = placement.source_window
        window_px = (right - left) * width_px
        spill = placement.width_mm * (width_px / window_px - 1.0) / 2.0
        art = badge_content.badge_art(board=board, number=board.operator_number)

        common = [
            f"干员 {board.operator_name}　编号 {board.operator_number or '（无）'}"
            f"　卡面素材 {len(art.placements)} 个"
            f"（二维码/条形码/职业/阵营按库里有的放）",
            f"{region.label}　区域中心 ({region.centre_x_mm:.3f}, "
            f"{region.centre_y_mm:.3f}) mm",
            f"取景窗口 x {left:.4f}–{right:.4f}　y {top:.4f}–{bottom:.4f}"
            f"（源图 {width_px} × {height_px} px 的占比）",
            f"框外每侧约 {spill:.1f} mm 暗影　窗口占源图宽 "
            f"{(right - left) * 100:.1f}%　风格 {args.style}",
            f"图片位置 ({placement.x_mm:.3f}, {placement.y_mm:.3f}) mm　"
            f"尺寸 {placement.width_mm:.3f} × {placement.height_mm:.3f} mm　"
            f"图层 {placement.layer}",
            f"源图 {name}",
        ]

        framing = args.outdir / f"{region_key}-framing.png"
        _shot(canvas if args.canvas_only else window, framing)
        tiles.append(
            (
                f"{grade} · {page}　取景中（框外 28% 暗影，本轮改动）",
                common,
                framing,
            )
        )

        # The same picture, the same window, with framing stopped: this is what
        # the canvas drew before the change.
        canvas.framing = None
        app.processEvents()
        clipped = args.outdir / f"{region_key}-clipped.png"
        _shot(canvas if args.canvas_only else window, clipped)
        tiles.append(
            (f"{grade} · {page}　结束取景（框外不画，改动前 / 导出）", common, clipped)
        )

        print(
            f"{grade} · {page} → {region.label}　窗口 "
            f"[{left:.4f}, {top:.4f}, {right:.4f}, {bottom:.4f}]　"
            f"框外每侧 ≈ {spill:.1f} mm"
        )

    from PIL import Image

    # The caption goes on the per-state files too, not only on the sheet: these
    # are the images a reader looks at on their own, and a measurement they
    # cannot see next to the picture is a measurement they have to take on
    # trust.
    images = []
    for title, details, path in tiles:
        captioned = _caption(Image.open(path).convert("RGB"), title, details)
        captioned.save(path)
        images.append(captioned)
    columns = 2
    rows = (len(images) + columns - 1) // columns
    cell_w = max(image.width for image in images) + 10
    cell_h = max(image.height for image in images) + 10
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (12, 14, 18))
    for index, image in enumerate(images):
        x = (index % columns) * cell_w + 5
        y = (index // columns) * cell_h + 5
        sheet.paste(image, (x, y))
    sheet_path = args.outdir / "取景对比.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}")
    for title, _details, path in tiles:
        print(f"  {title}: {path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
