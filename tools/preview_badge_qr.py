"""Preview the 干员二维码 placement on the 通行证 card, for one or more operators.

This renders the card the way the application does -- the real template, the
real panel, the real button -- and saves two images per operator: the whole
window and a close-up of the symbol, because a 3.2 mm QR code inside a 100 mm
card is only about twenty pixels across in a window shot.

``--random 2`` draws two operators out of the roster instead of naming them, so
a check is not quietly made against the two people whose code was already
looked at.

Usage::

    python tools/preview_badge_qr.py --random 2
    python tools/preview_badge_qr.py --operator 阿米娅
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _scratch(outdir: Path) -> Path:
    scratch = outdir / "drafts"
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch)
    return scratch


def _pick_random(count: int, seed: int | None) -> list[str]:
    from pcbart import operators

    database = operators.database()
    if not database:
        raise SystemExit("找不到干员资料库")
    rows = [item for item in database.operators if item.page]
    if count > len(rows):
        raise SystemExit(f"只找到 {len(rows)} 位干员")
    rng = random.Random(seed)
    return [item.page for item in rng.sample(rows, count)]


def _closeup(canvas, placement, scale: float, path: Path, app) -> None:
    """Zoom the canvas onto one placement and save that view."""
    canvas.fit_view()
    app.processEvents()
    canvas.zoom_by(scale, canvas.rect().center())
    point = canvas.to_screen(placement.x_mm, placement.y_mm)
    centre = canvas._plot.center()
    canvas.pan_by_pixels(centre.x() - point.x(), centre.y() - point.y())
    app.processEvents()
    from PySide6.QtGui import QImage

    image = QImage(canvas.size(), QImage.Format_RGB32)
    canvas.render(image)
    image.save(str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_qr_preview")
    parser.add_argument("--zoom", type=float, default=9.0)
    parser.add_argument(
        "--barcode-zoom",
        type=float,
        default=2.0,
        help="fit the 49 mm strip in the viewport, unlike the 3.2 mm QR square",
    )
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=950)
    args = parser.parse_args(argv)

    pages = list(args.operator)
    if args.random:
        pages += _pick_random(args.random, args.seed)
    if not pages:
        raise SystemExit("用 --operator 指定干员，或用 --random 2 随机抽两位")

    args.outdir.mkdir(parents=True, exist_ok=True)
    _scratch(args.outdir)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    import gui_mockup as g
    from pcbart import badge_card, badge_codes, badge_content, operators

    app = QApplication.instance() or QApplication([])
    database = operators.database()
    window = g.build_window()
    window.resize(args.width, args.height)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    report: list[dict] = []
    for index, page in enumerate(pages):
        if index:
            g.new_project_tab(window)
            app.processEvents()
        person = database.find_page(page) or database.find(page)
        if person is None:
            print(f"跳过「{page}」：资料库里没有")
            continue

        board = window.design_canvas.board
        board.outline = badge_card.badge_spec()
        board.outline_ready = True
        board.texts = badge_card.badge_texts()
        board.cutouts = badge_card.badge_cutouts()

        panel = g.show_template_content(window, board)
        panel.refresh()
        assert panel.apply_operator(person.page), person.page
        assert panel._insert_card_art(announce=False), "二维码没有放进卡面"
        app.processEvents()

        canvas = window.design_canvas
        stem = person.page
        full = args.outdir / f"card-{stem}.png"
        window.grab().save(str(full))
        art = badge_content.badge_art(board=board)
        close = args.outdir / f"qr-{stem}.png"
        _closeup(canvas, art.qr, args.zoom, close, app)
        barcode_shot = None
        if art.barcode is not None:
            barcode_shot = args.outdir / f"barcode-{stem}.png"
            # The strip is tall and thin, so it needs a smaller zoom than the
            # QR square to fit in the same viewport.
            _closeup(canvas, art.barcode, args.barcode_zoom, barcode_shot, app)

        # Rebuild the code at the *placement's* own bitmap area, so the numbers
        # reported here are the ones on the card rather than the default size.
        code = badge_codes.qr_for_operator(person.page, area_mm=art.qr.width_mm)
        region = badge_content.art_region("qr")
        # ``total_modules`` includes the border; the code area is the rest.
        code_modules = code.layout.total_modules - 2 * code.layout.border_modules
        code_mm = code.module_mm * code_modules
        entry = {
            "page": person.page,
            "name": person.name,
            "number": person.number,
            "payload": code.payload,
            "version": code.version,
            "total_modules": code.layout.total_modules,
            "module_mm": round(code.module_mm, 6),
            "code_mm": round(code_mm, 6),
            "block_mm": region.width_mm,
            "bitmap_mm": round(code.layout.area_mm, 6),
            "centre": [art.qr.x_mm, art.qr.y_mm],
            "rotation_deg": art.qr.rotation_deg,
            "layer": art.qr.layer,
            "png": art.qr.path,
            "full_shot": str(full),
            "closeup": str(close),
        }
        if art.barcode is not None:
            bar_region = badge_content.art_region("barcode")
            entry["barcode"] = {
                "payload": person.number,
                "total_modules": 79 + 20,
                "code_modules": 79,
                "module_mm": round(
                    art.barcode.width_mm / (79 + 20), 6
                ),
                "code_mm": round(max(bar_region.width_mm, bar_region.height_mm), 6),
                "bar_height_mm": round(min(bar_region.width_mm, bar_region.height_mm), 6),
                "file_mm": round(art.barcode.width_mm, 6),
                "centre": [art.barcode.x_mm, art.barcode.y_mm],
                "rotation_deg": art.barcode.rotation_deg,
                "layer": art.barcode.layer,
                "png": art.barcode.path,
                "closeup": str(barcode_shot),
            }
        report.append(entry)
        print(
            f"{person.name}（{person.page}）: {code.payload} · v{code.version} · "
            f"码区 {code_mm:.4f} mm（定位块 {region.width_mm:.4f}）· "
            f"含静区 {code.layout.area_mm:.4f} mm · "
            f"{code.module_mm:.4f} mm/模块"
        )
        if art.barcode is not None:
            bar_module = art.barcode.width_mm / (79 + 20)
            print(
                f"    条形码 {person.number} · 条区 "
                f"{max(badge_content.art_region('barcode').width_mm, badge_content.art_region('barcode').height_mm):.4f} mm"
                f"（定位框）· 含静区 {art.barcode.width_mm:.4f} mm · "
                f"{bar_module:.4f} mm/模块 · 旋转 {art.barcode.rotation_deg:.0f}°"
            )
        else:
            print("    条形码 无情报编号，只出二维码")
        panel.close()
        app.processEvents()

    manifest = args.outdir / "preview.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {manifest}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
