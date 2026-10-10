"""One PNG showing every card side by side, rendered by the application itself.

Why this exists
---------------
A merged ``.epro2`` holds all the cards, but looking at it still means opening
the editor. This renders the same cards through the real window -- template,
operator, artwork, illustration framing -- and tiles the canvas shots into one
picture, so the whole set can be checked at a glance before anything is opened
in EasyEDA Pro.

⚠️ **This renders the model, not the file.** It answers "do the eight cards look
right", not "did the writer put every record in the file". The file itself is
checked structurally (record counts, embedded pictures, coordinate deltas); see
``tools/preview_color_export.py`` for reading the pixels back out of it.

Usage::

    python tools/make_color_contact_sheet.py --outdir out/2026.10.8test
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

from make_color_test_file import (  # noqa: E402
    CASES,
    LIBRARY,
    STAGES,
    custom_cases,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "out")
    parser.add_argument("--stage", default="精0", choices=sorted(STAGES))
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="[档位|]文件名",
        help="立绘文件名；要混档位就写成「精2|立绘 陈 2.png」，可重复",
    )
    parser.add_argument(
        "--stages",
        default="精0,精2",
        help="内置用例要跑哪些档位，逗号分隔（默认两个都跑）",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=520)
    args = parser.parse_args(argv)

    if args.figure:
        # One run may mix stages -- a 精0 card is a different layout from a 精2
        # one, and a contact sheet that only ever shows one of them cannot show
        # that the two card formats both carry their elements.
        cases = []
        for entry in args.figure:
            stage, _, name = entry.rpartition("|")
            if not stage:
                stage, name = args.stage, entry
            cases.extend(custom_cases([name], stage))
        cases = tuple(cases)
    else:
        wanted = {item.strip() for item in args.stages.split(",") if item.strip()}
        cases = tuple(case for case in CASES if case[1] in wanted)

    shots = args.outdir / "_canvas"
    shots.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(1400, 900)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    captured: list[tuple[str, Path]] = []
    for index, (label, stage, page, folder, name, region_key) in enumerate(cases):
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

        canvas = window.design_canvas
        placement = board.placements[-1]
        region = next(
            item for item in badge_card.BADGE_REGIONS if item.key == region_key
        )
        canvas.selected = placement
        canvas.start_framing(placement, region)
        # Leave the framing gesture: the shot is a finished card, not a
        # work-in-progress with the amber crop overlay and the dimmed outside.
        canvas.framing = None
        canvas.selected = None
        canvas.fit_view()
        app.processEvents()

        image = QImage(canvas.size(), QImage.Format_RGB32)
        canvas.render(image)
        target = shots / f"card-{index:02d}-{page}-{stage}.png"
        image.save(str(target))
        captured.append((f"{page} {stage}", target))
        print(f"{page} {stage} -> {target.name}")

    if not captured:
        print("没有可导出的卡片。", file=sys.stderr)
        return 1

    sheet = args.outdir / "立绘彩印测试_全部卡片总览.png"
    _tile(captured, sheet, columns=args.columns, cell_width=args.cell_width)
    print(f"\n总览：{sheet}")
    for _, path in captured:
        print(f"  单张：{path}")

    sys.stdout.flush()
    os._exit(0)


def _tile(
    items: list[tuple[str, Path]],
    target: Path,
    *,
    columns: int,
    cell_width: int,
) -> None:
    """Lay the card shots out in a grid with their labels, and write ``target``."""
    from PIL import Image, ImageDraw

    crops: list[tuple[str, Image.Image]] = []
    for label, path in items:
        with Image.open(path) as image:
            frame = image.convert("RGB")
        # Trim the surrounding background the canvas leaves around the card, so
        # eight shots fit without most of the sheet being empty margin. The
        # tolerance is loose on purpose: the canvas grid lines are a near-match
        # for the background and would otherwise defeat the trim.
        box = _content_box(frame, tolerance=28)
        frame = frame.crop(box)
        scale = cell_width / float(frame.width)
        frame = frame.resize(
            (cell_width, max(1, int(frame.height * scale))), Image.LANCZOS
        )
        crops.append((label, frame))

    label_h = 26
    pad = 12
    cols = max(1, columns)
    rows = (len(crops) + cols - 1) // cols
    cell_h = max(image.height for _label, image in crops) + label_h
    sheet_w = cols * cell_width + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)

    for index, (label, image) in enumerate(crops):
        column = index % cols
        row = index // cols
        x = pad + column * (cell_width + pad)
        y = pad + row * (cell_h + pad)
        draw.rectangle(
            [x - 2, y - 2, x + cell_width + 1, y + image.height + 1],
            outline=(210, 214, 220),
        )
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 6), label, fill=(32, 36, 44))

    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)


def _content_box(frame, tolerance: int = 28):
    """The bounding box of the picture, ignoring a near-uniform border.

    Scanned from the outside in, so an interior line that happens to match the
    background does not cut the card in half.
    """
    width, height = frame.size
    pixels = frame.load()

    def row_is_empty(y: int) -> bool:
        first = pixels[0, y]
        return all(
            abs(pixels[x, y][0] - first[0]) <= tolerance
            and abs(pixels[x, y][1] - first[1]) <= tolerance
            and abs(pixels[x, y][2] - first[2]) <= tolerance
            and sum(pixels[x, y]) > 690  # only light backgrounds get trimmed
            for x in range(0, width, max(1, width // 200))
        )

    def col_is_empty(x: int) -> bool:
        first = pixels[x, 0]
        return all(
            abs(pixels[x, y][0] - first[0]) <= tolerance
            and abs(pixels[x, y][1] - first[1]) <= tolerance
            and abs(pixels[x, y][2] - first[2]) <= tolerance
            and sum(pixels[x, y]) > 690
            for y in range(0, height, max(1, height // 200))
        )

    top = 0
    while top < height - 1 and row_is_empty(top):
        top += 1
    bottom = height - 1
    while bottom > top and row_is_empty(bottom):
        bottom -= 1
    left = 0
    while left < width - 1 and col_is_empty(left):
        left += 1
    right = width - 1
    while right > left and col_is_empty(right):
        right -= 1
    return (left, top, right + 1, bottom + 1)


if __name__ == "__main__":
    raise SystemExit(main())
