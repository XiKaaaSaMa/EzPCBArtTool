"""Export a finished 通行证 as a fab package and draw every layer it wrote.

The point is to see what a *manufacturer* would receive, not what the canvas
shows. Each Gerber file is parsed back with ``gerbonara`` -- an independent
reader -- and rendered to a PNG in the layer's own colour, then the seven
layers are composited in the order 嘉立创 builds them (copper over silkscreen
over mask), so a mask opening reads as a hole in the mask rather than as a
patch of colour.

Output goes to ``out/_badge_gerber/``: one PNG per layer, the composite, and a
manifest listing the file sizes and the measured extent of each layer.

The card is built through the real interface -- operator, four card elements,
wording -- and the illustration is imported through the preprocessing panel, so
nothing here is drawn by hand.

Usage::

    python tools/preview_badge_gerber.py
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

#: ``(operator page, folder, file name, region key)``.
FIGURE = ("能天使", "06-立绘-精英0", "立绘 能天使 1.png", "small")


def _prepare_environment(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch / "drafts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "out" / "_badge_gerber"
    )
    parser.add_argument("--style", default="彩色丝印")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--px-per-mm", type=float, default=12.0, help="渲染分辨率"
    )
    args = parser.parse_args(argv)

    _prepare_environment(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pcbart import badge_card

    import gui_mockup as g

    app = QApplication.instance() or QApplication([])
    window = g.build_window()
    window.resize(args.width, args.height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    canvas = window.design_canvas
    board = canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    page, folder, name, region_key = FIGURE
    content = g.show_template_content(window, board)
    content.refresh()
    if not content.apply_operator(page):
        raise SystemExit(f"资料库里没有「{page}」")
    if not content._insert_card_art(announce=False):
        raise SystemExit("卡面素材没有放上去")
    content.close()

    panel = g.show_preprocess(window)
    figure = LIBRARY / folder / name
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
    region = next(
        item for item in badge_card.BADGE_REGIONS if item.key == region_key
    )
    canvas.selected = placement
    canvas.start_framing(placement, region)
    canvas.window_zoom(1.8)

    from pcbart.gerber import export_gerber

    package = args.outdir / "gerber"
    if package.exists():
        import shutil

        shutil.rmtree(package)
    report = export_gerber(board, package, title="通行证", archive=True)

    # Read every file back with a third-party parser and render it. What the
    # writer believes it emitted and what a reader finds are two different
    # claims, and only the second one reaches the fab.
    from PIL import Image, ImageDraw, ImageFont

    from gerbonara.rs274x import GerberFile

    colours = {
        ".GTL": (255, 40, 40),
        ".GBL": (60, 90, 255),
        ".GTO": (255, 214, 0),
        ".GBO": (110, 230, 90),
        ".GTS": (210, 110, 255),
        ".GBS": (255, 120, 220),
        ".GKO": (250, 250, 250),
    }
    draw_order = (".GBS", ".GBO", ".GBL", ".GTS", ".GTO", ".GTL", ".GKO")

    # ⚠️ **One frame for every layer.** Each Gerber's bounding box covers only
    # what that layer happens to contain, so placing them at the same pixel
    # origin -- or worse, resizing each to a common size -- slides the layers
    # apart. The board outline is the reference: it spans the whole card, and
    # every other layer is drawn into the same millimetre window.
    outline_extent = GerberFile.open(
        str(next(p for p in package.glob("*.GKO")))
    ).bounding_box()
    (frame_x0, frame_y0), (frame_x1, frame_y1) = outline_extent
    scale = args.px_per_mm
    margin = 20
    frame_w = int((frame_x1 - frame_x0) * scale) + margin * 2
    frame_h = int((frame_y1 - frame_y0) * scale) + margin * 2

    def render(path: Path, colour):
        """Raster of one layer, in the shared millimetre frame."""
        parsed = GerberFile.open(str(path))
        canvas_img = Image.new("L", (frame_w, frame_h), 0)
        drawer = ImageDraw.Draw(canvas_img)

        def to_px(x, y):
            return (
                margin + (x - frame_x0) * scale,
                frame_h - margin - (y - frame_y0) * scale,
            )

        for obj in parsed.objects:
            kind = type(obj).__name__
            if kind == "Region":
                # ``outline`` is the boundary as a flat list of points. Only
                # the outer ring is filled: the clear-polarity holes are
                # separate objects, and filling each ring would close them.
                points = [to_px(*pt) for pt in obj.outline or []]
                if len(points) >= 3:
                    drawer.polygon(points, fill=255)
                continue
            if kind == "Line":
                # ``as_primitive()`` is the parser's flattened segment, which
                # carries its own aperture width -- the raw object only knows
                # its aperture. Using it means an arc is drawn exactly as the
                # reader resolved it, not as a second implementation would.
                primitive = obj.as_primitive()
                width_px = max(1, int(round(float(primitive.width) * scale)))
                drawer.line(
                    [
                        to_px(primitive.x1, primitive.y1),
                        to_px(primitive.x2, primitive.y2),
                    ],
                    fill=255,
                    width=width_px,
                )
        (x0, y0), (x1, y1) = parsed.bounding_box()
        return canvas_img, (x0, y0, x1, y1)

    rendered: dict[str, object] = {}
    extents: dict[str, list[float]] = {}
    # By extension, not by the ``Gerber_`` prefix: the archive sits in the same
    # folder under that prefix too, and it is a zip, not a layer.
    for path in sorted(package.glob("Gerber_*")):
        if path.suffix.upper() not in colours:
            continue
        extension = path.suffix.upper()
        image, box = render(path, colours.get(extension, (255, 255, 255)))
        rendered[extension] = image
        extents[extension] = [round(value, 4) for value in box]
        image.save(args.outdir / f"layer-{extension.lstrip('.')}.png")

    # Every layer is already in the same frame, so compositing is a straight
    # paint in the production order -- copper over silkscreen over mask.
    size = (frame_w, frame_h)
    composite = Image.new("RGB", size, (8, 10, 13))
    for extension in draw_order:
        image = rendered.get(extension)
        if image is None:
            continue
        mask = image if extension not in (".GTS", ".GBS") else image
        if extension in (".GTS", ".GBS"):
            # A mask layer is the *set of openings*: show it as "no mask here",
            # which is how the etcher reads it. Painting it as another colour
            # would suggest material where the mask is cut away.
            composite.paste(Image.new("RGB", size, (12, 16, 21)), (0, 0), mask)
            continue
        composite.paste(Image.new("RGB", size, colours[extension]), (0, 0), mask)

    composite_path = args.outdir / "合图.png"
    composite.save(composite_path)

    font = None
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            # Sized to the tile width: four columns leave about 250 px each,
            # and a caption that overruns its neighbour is worse than a small
            # one.
            font = ImageFont.truetype(candidate, 16)
            break

    labels = {
        ".GTL": "顶层铜",
        ".GBL": "底层铜",
        ".GTO": "顶层丝印",
        ".GBO": "底层丝印",
        ".GTS": "顶阻焊开窗",
        ".GBS": "底阻焊开窗",
        ".GKO": "板框",
    }
    # A grid of every file the package holds, plus the composite. Each tile is
    # captioned with its own measured extent, so a layer that came out empty or
    # in the wrong place is visible without opening the files.
    tiles: list[tuple[str, object]] = []
    for extension in (".GTL", ".GBL", ".GTO", ".GBO", ".GTS", ".GBS", ".GKO"):
        image = rendered.get(extension)
        if image is None:
            continue
        # A mask file is the *set of openings*, so on its own tile it is drawn
        # in its own colour -- that is what makes "there is geometry here" and
        # "here is where the mask is cut" readable. Only in the composite does
        # it become an absence.
        flat = Image.new("RGB", size, (8, 10, 13))
        flat.paste(Image.new("RGB", size, colours[extension]), (0, 0), image)
        box = extents.get(extension)
        caption = labels[extension]
        if box:
            caption += f"　{box[2] - box[0]:.1f}×{box[3] - box[1]:.1f}"
        tiles.append((caption, flat))
    tiles.append(("合图（暗=阻焊挖掉）", composite))

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    pad = 8
    head = 30
    tile_w, tile_h = size[0] // 2, size[1] // 2
    sheet = Image.new(
        "RGB",
        (columns * (tile_w + pad) + pad, rows * (tile_h + head + pad) + pad),
        (16, 19, 23),
    )
    drawer = ImageDraw.Draw(sheet)
    for index, (caption, tile) in enumerate(tiles):
        x = pad + (index % columns) * (tile_w + pad)
        y = pad + (index // columns) * (tile_h + head + pad)
        drawer.text((x, y), caption, fill=(210, 220, 230), font=font)
        sheet.paste(tile.resize((tile_w, tile_h)), (x, y + head))
    sheet_path = args.outdir / "分层预览.png"
    sheet.save(sheet_path)

    manifest = {
        "board": {
            "operator": board.operator_name,
            "number": board.operator_number,
            "placements": len(board.placements),
            "window": list(placement.source_window or ()),
        },
        "files": [
            {
                "name": Path(str(entry["path"])).name,
                "bytes": Path(str(entry["path"])).stat().st_size,
            }
            for entry in report["files"]
        ],
        "empty": report.get("skipped") or [],
        "note": report.get("note"),
        "archive": report.get("archive"),
        "extents_mm": extents,
    }
    manifest_path = args.outdir / "gerber.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for entry in manifest["files"]:
        print(f"  {entry['name']:42s} {entry['bytes']:9d} bytes")
    print(f"空层：{manifest['empty'] or '（无）'}")
    print(f"wrote {sheet_path}")
    print(f"wrote {manifest_path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
