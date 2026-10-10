"""Render 1:1 printable sheets for the badge QR codes.

Two modes:

* ``--short-url`` — the original size sweep: the same short link printed at
  several physical sizes, to find where a phone actually locks on.
* ``--factions`` — a faction-major sample sheet: one row of operators per
  faction, each drawn at the card's real size plus two enlarged copies.
  This is the sheet that checks whether the code still reads when the payload
  changes, not just when the size does.

**"The card's real size" means the whole file, quiet zone included**, which is
what has to be drawn to reproduce the module size the board gets. The card's
drawn block measures the *code area* (3.1997 mm); the transparent quiet zone
extends 4 modules past it on each side, so the file is 4.41863 mm. Sizing the
sample sheet by the block instead would print every symbol 27% denser than the
card and understate whether it scans.

Both are SVG so each square is exactly its requested size on paper. The sheet
must be printed at 100% / actual size; "fit to page" silently rescales the
squares and the whole experiment becomes meaningless.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_codes, badge_content, operators  # noqa: E402

PREVIEW_DIR = Path(__file__).resolve().parent.parent / "out" / "_badge_preview"
DEFAULT_OUT = PREVIEW_DIR / "qr-short-test.svg"

#: Factions for the sample sheet, most populous first. Six per page is what
#: fits A4 landscape at a readable label size; more than that and either the
#: type or the gap between squares goes below what a phone camera needs.
DEFAULT_FACTIONS = (
    "罗德岛",
    "维多利亚",
    "哥伦比亚",
    "萨尔贡",
    "炎",
    "拉特兰",
)

def _card_file_mm() -> float:
    """The QR file size the card actually writes, quiet zone included.

    Read from the card's own region table rather than typed here: the block
    measures the code area, and the file is that plus 4 quiet modules per side.
    A sheet printed at a stale size would answer the wrong question about
    whether the real card scans.
    """
    block_mm = badge_content.art_region("qr").width_mm
    return badge_codes.qr_for_operator(
        "阿米娅", code_mm=block_mm
    ).layout.area_mm


#: The card's real QR file, and the two enlarged copies drawn beside it.
CARD_SIZE_MM = _card_file_mm()
REAL_SIZE_MM = 5.6
BIG_SIZE_MM = 7.0
CARD_LABEL = f"{CARD_SIZE_MM:.2f}"
PAGE_W_MM = 297.0
PAGE_H_MM = 210.0

#: Vertical pitch between two operators inside a faction column. The tallest
#: thing in a row is the enlarged square plus its caption, so anything under
#: about 12 mm would collide; this leaves the rest as breathing room.
OPERATOR_PITCH_MM = 24.0

#: Named picks per faction, so the sheet shows the operators someone would
#: recognise on sight rather than whatever sorts last by name. The point of the
#: exercise is to compare images across payloads, and an unrecognised name
#: makes a mismatch impossible to interpret. Factions without an entry fall
#: back to automatic sampling.
PICKS = {
    "罗德岛": ("阿米娅", "凯尔希", "史尔特尔", "鸿雪", "温蒂", "泥岩"),
    "维多利亚": ("风笛", "澄闪", "琴柳", "傀影", "号角", "苇草"),
    "哥伦比亚": ("山", "星极", "蛇屠箱", "霍尔海雅", "冰酿", "刻刀"),
    "萨尔贡": ("森蚺", "异客", "狮蝎", "慑砂", "蜜蜡", "特米米"),
    "炎": ("赤刃明霄陈", "左乐", "惊蛰", "乌有", "清流", "九色鹿"),
    "拉特兰": ("莫斯提马", "空弦", "菲亚梅塔", "送葬人", "塑心", "安比尔"),
}


def _fmt(value: float) -> str:
    """Millimetres with 0.01 mm precision and no trailing noise.

    SVG user units are the printed millimetres here, so every coordinate is a
    string that has to be stable: rounding at format time (rather than when
    accumulating positions) is what keeps the module grid from drifting.
    """
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _qr_group(qr, x_mm: float, y_mm: float, label: str) -> list[str]:
    """One QR square plus its caption, drawn once and re-placed by transform."""
    layout = qr.layout
    module = layout.module_mm
    rows = [f'<g transform="translate({_fmt(x_mm)} {_fmt(y_mm)})">']
    rows.append(
        f'<rect x="0" y="0" width="{_fmt(layout.area_mm)}" '
        f'height="{_fmt(layout.area_mm)}" fill="#fff"/>'
    )
    for row, line in enumerate(layout.matrix):
        for col, ink in enumerate(line):
            if not ink:
                continue
            rows.append(
                f'<rect x="{_fmt(col * module)}" y="{_fmt(row * module)}" '
                f'width="{_fmt(module)}" height="{_fmt(module)}" fill="#000"/>'
            )
    rows.append(
        f'<text x="0" y="{_fmt(layout.area_mm + 2.0)}" font-size="1.4" '
        f'font-family="sans-serif">{html.escape(label)}</text>'
    )
    rows.append(
        f'<text x="0" y="{_fmt(layout.area_mm + 3.5)}" font-size="1.0" '
        f'font-family="sans-serif" fill="#444">'
        f"v{layout.version} / {layout.module_mm:.3f} mm</text>"
    )
    rows.append("</g>")
    return rows


def _sample_operators(per_faction: int = 6) -> tuple[list[tuple[str, list]], list[str]]:
    """Pick the roster the sheet draws, faction by faction.

    A faction with a :data:`PICKS` entry uses that order; the rest fall back to
    ``(rarity, name)``. Either way a faction with fewer rows than
    ``per_faction`` contributes what it has instead of being dropped or padded
    with duplicates -- a short row is information about the roster, not a
    layout failure.

    Returns the picked rows plus the factions that had no operators at all.
    """
    database = operators.database()
    rows: list[tuple[str, list]] = []
    missing: list[str] = []
    for faction in DEFAULT_FACTIONS:
        members = [item for item in database.operators if item.faction == faction]
        if not members:
            missing.append(faction)
            continue

        # One entry per operator name: the database lists alternate forms
        # (阿米娅 / 阿米娅(医疗)) separately and they would look like duplicates.
        by_name: dict[str, object] = {}
        for item in members:
            by_name.setdefault(item.name.split("(")[0], item)

        chosen = []
        for name in PICKS.get(faction, ()):
            item = by_name.pop(name, None)
            if item is not None:
                chosen.append(item)
        for item in sorted(members, key=lambda row: (row.rarity, row.name), reverse=True):
            if len(chosen) >= per_faction:
                break
            base = item.name.split("(")[0]
            if base in by_name:
                chosen.append(by_name.pop(base))
        rows.append((faction, chosen[:per_faction]))
    return rows, missing


@dataclass(frozen=True)
class SheetGeometry:
    """Where the sample sheet's grid lands, in millimetres."""

    columns: int
    rows: int
    left: float
    top: float
    column_pitch: float
    row_pitch: float

    @property
    def width(self) -> float:
        return PAGE_W_MM - 4.0

    @property
    def height(self) -> float:
        return PAGE_H_MM - 18.0


def _geometry(count: int, per_faction: int) -> SheetGeometry:
    columns = max(1, int(np.clip(count, 1, 7)))
    per_faction = max(1, per_faction)
    height = 5.6 + per_faction * OPERATOR_PITCH_MM
    column_pitch = min(51.0, (PAGE_W_MM - 16.0) / columns)
    row_pitch = height + 1.2
    top = 20.0
    if top + len(range(0, count, columns)) * row_pitch > PAGE_H_MM:
        # Too many factions for one page: tighten rather than overflow.
        row_pitch = (PAGE_H_MM - top - 4.0) / len(range(0, count, columns))
        height = row_pitch - 1.2
    return SheetGeometry(
        columns=columns,
        rows=per_faction,
        left=8.0,
        top=top,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )


def _write_faction_sheet(
    rows: list[tuple[str, list]],
    out: Path,
    *,
    per_faction: int,
    png: bool = False,
) -> None:
    geometry = _geometry(len(rows), per_faction)
    sample = badge_codes.qr_for_operator(rows[0][1][0].page, area_mm=BIG_SIZE_MM)
    big_module = sample.module_mm
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W_MM:g}mm" '
        f'height="{PAGE_H_MM:g}mm" viewBox="0 0 {PAGE_W_MM:g} {PAGE_H_MM:g}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="8" y="11" font-size="4.2" font-family="sans-serif">'
        "通行证二维码 · 多阵营实尺打印测试</text>",
        '<text x="8" y="16" font-size="1.9" font-family="sans-serif" fill="#444">'
        f"请按 100% / 实际大小打印，不要「适应页面」。每格左起：卡上真实 "
        f"{CARD_LABEL} mm、{REAL_SIZE_MM:g} mm、放大 {BIG_SIZE_MM:g} mm"
        "（同码同内容，只改尺寸）。</text>",
    ]

    for index, (faction, members) in enumerate(rows):
        column = index % geometry.columns
        row = index // geometry.columns
        x = geometry.left + column * geometry.column_pitch
        y = geometry.top + row * geometry.row_pitch
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(y - 1.4)}" '
            f'x2="{_fmt(x + geometry.column_pitch - 4.0)}" y2="{_fmt(y - 1.4)}" '
            'stroke="#bbb" stroke-width="0.2"/>'
        )
        parts.append(
            f'<text x="{_fmt(x)}" y="{_fmt(y + 3.0)}" font-size="3.0" '
            f'font-weight="bold" font-family="sans-serif">{html.escape(faction)}</text>'
        )
        parts.append(
            f'<text x="{_fmt(x)}" y="{_fmt(y + 5.0)}" font-size="1.5" '
            f'font-family="sans-serif" fill="#666">{len(members)} 位干员</text>'
        )
        for slot, operator in enumerate(members):
            top = y + 6.0 + slot * OPERATOR_PITCH_MM
            name = operator.name
            parts.append(
                f'<text x="{_fmt(x)}" y="{_fmt(top + 4.2)}" font-size="2.2" '
                f'font-family="sans-serif">{html.escape(name)}</text>'
            )
            parts.append(
                f'<text x="{_fmt(x)}" y="{_fmt(top + 7.4)}" font-size="1.4" '
                f'font-family="sans-serif" fill="#555">'
                f"{html.escape(operator.number or '—')}</text>"
            )
            short = badge_codes.short_url_for(operator.page)
            parts.append(
                f'<text x="{_fmt(x)}" y="{_fmt(top + 10.4)}" font-size="1.2" '
                f'font-family="sans-serif" fill="#777">'
                f"{html.escape(short or '（无短链，回退长链接）')}</text>"
            )
            base = badge_codes.qr_for_operator(operator.page)
            parts.append(
                f'<text x="{_fmt(x)}" y="{_fmt(top + 13.6)}" font-size="1.2" '
                f'font-family="sans-serif" fill="#555">'
                f"v{base.version} · {base.bytes} 字节</text>"
            )

            for label, area, dx in (
                (CARD_LABEL, CARD_SIZE_MM, 14.0),
                (f"{REAL_SIZE_MM:g}", REAL_SIZE_MM, 18.4),
                (f"{BIG_SIZE_MM:g}", BIG_SIZE_MM, 23.2),
            ):
                qr = badge_codes.qr_for_operator(operator.page, area_mm=area)
                parts += _qr_group(qr, x + dx, top, f"{label} mm")
            parts.append(
                f'<text x="{_fmt(x + 23.2)}" y="{_fmt(top + 9.4)}" font-size="1.1" '
                f'font-family="sans-serif" fill="#888">放大对照</text>'
            )

    footer_y = PAGE_H_MM - 10.0
    card_module = badge_codes.qr_for_operator(
        "阿米娅", area_mm=CARD_SIZE_MM
    ).module_mm
    parts.append(
        f'<text x="8" y="{_fmt(footer_y)}" font-size="1.5" '
        f'font-family="sans-serif" fill="#555">'
        f"共 {len(rows)} 个阵营 / {sum(len(members) for _, members in rows)} 位干员 · "
        f"{CARD_LABEL} mm 每模块约 {card_module:.3f} mm · "
        "扫码后应跳转到对应干员的 PRTS 页面</text>"
    )
    parts.append(
        f'<text x="8" y="{_fmt(footer_y + 4.0)}" font-size="1.5" '
        f'font-family="sans-serif" fill="#555">'
        f"每格最左那颗就是卡上的真实尺寸（{CARD_LABEL} mm，模块 "
        f"{card_module:.3f} mm）；右边两颗只作对照。扫得出来才算通过。"
        "</text>"
    )
    parts.append("</svg>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"  卡上真实 {CARD_LABEL} mm → {card_module:.4f} mm/模块")

    if png:
        _write_png(out.with_suffix(".png"), rows, geometry)


def _write_png(
    out: Path,
    rows: list[tuple[str, list]],
    geometry: SheetGeometry,
    *,
    dpi: int = 1200,
) -> None:
    """Same sheet as a raster, so it can be previewed and printed without SVG."""
    scale = dpi / 25.4
    width = int(round(PAGE_W_MM * scale))
    height = int(round(PAGE_H_MM * scale))
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    regular = r"C:\Windows\Fonts\msyh.ttc"
    bold = r"C:\Windows\Fonts\msyhbd.ttc"

    def font(pt_mm: float, path: str = regular) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(path, max(8, int(round(pt_mm * scale))))

    def mm(value: float) -> float:
        return value * scale

    def text(x: float, y: float, value: str, pt_mm: float, fill="#111", path=regular):
        draw.text((mm(x), mm(y)), value, fill=fill, font=font(pt_mm, path))

    def paste_qr(qr, x: float, y: float) -> None:
        layout = qr.layout
        size = len(layout.matrix)
        module_px = max(1, int(round(layout.module_mm * scale)))
        array = np.full((size, size), 255, np.uint8)
        for row, line in enumerate(layout.matrix):
            for col, ink in enumerate(line):
                if ink:
                    array[row, col] = 0
        image = Image.fromarray(array, "L")
        image = image.resize((size * module_px,) * 2, Image.NEAREST).convert("RGB")
        page.paste(image, (int(round(mm(x))), int(round(mm(y)))))

    text(8, 4, "通行证二维码 · 多阵营实尺打印测试", 4.2, path=bold)
    text(
        8,
        11,
        f"请按 100% / 实际大小打印，不要「适应页面」。每格左起：卡上真实 "
        f"{CARD_LABEL} mm、{REAL_SIZE_MM:g} mm、放大 {BIG_SIZE_MM:g} mm"
        "（同码同内容，只改尺寸）。",
        1.9,
        fill="#444",
    )

    for index, (faction, members) in enumerate(rows):
        column = index % geometry.columns
        row = index // geometry.columns
        x = geometry.left + column * geometry.column_pitch
        y = geometry.top + row * geometry.row_pitch
        draw.line(
            (mm(x), mm(y - 1.4), mm(x + geometry.column_pitch - 4.0), mm(y - 1.4)),
            fill=(187, 187, 187),
            width=2,
        )
        text(x, y, faction, 3.0, path=bold)
        text(x, y + 3.6, f"{len(members)} 位干员", 1.5, fill="#666")
        for slot, operator in enumerate(members):
            top = y + 6.0 + slot * OPERATOR_PITCH_MM
            text(x, top, operator.name, 2.2)
            text(x, top + 3.6, operator.number or "—", 1.4, fill="#555")
            text(
                x,
                top + 6.4,
                badge_codes.short_url_for(operator.page) or "（无短链，回退长链接）",
                1.2,
                fill="#777",
            )
            qr = badge_codes.qr_for_operator(operator.page)
            text(
                x,
                top + 9.2,
                f"v{qr.version} · {qr.bytes} 字节",
                1.2,
                fill="#555",
            )
            for label, area, dx in (
                (CARD_LABEL, CARD_SIZE_MM, 14.0),
                (f"{REAL_SIZE_MM:g}", REAL_SIZE_MM, 18.4),
                (f"{BIG_SIZE_MM:g}", BIG_SIZE_MM, 23.2),
            ):
                one = badge_codes.qr_for_operator(operator.page, area_mm=area)
                paste_qr(one, x + dx, top)
                text(x + dx, top + area, label + " mm", 1.1, fill="#666")
            text(x + 23.2, top + 9.4, "放大对照", 1.1, fill="#888")

    text(
        8,
        PAGE_H_MM - 11.0,
        f"共 {len(rows)} 个阵营 / {sum(len(members) for _, members in rows)} 位干员 · "
        "扫码后应跳转到对应干员的 PRTS 页面",
        1.5,
        fill="#555",
    )
    text(
        8,
        PAGE_H_MM - 7.0,
        f"每格最左那颗就是卡上的真实尺寸（{CARD_LABEL} mm）；"
        "右边两颗只作对照。扫得出来才算通过。",
        1.5,
        fill="#555",
    )
    page.save(out, dpi=(dpi, dpi))
    print(f"wrote {out}")


def _write_size_sweep(args: argparse.Namespace) -> int:
    short = badge_codes.short_url_payload(args.short_url)
    sizes = [float(part.strip()) for part in args.sizes.split(",") if part.strip()]
    if not sizes:
        raise SystemExit("--sizes 不能为空")
    codes = [
        (f"{size:g} mm", badge_codes.qr_for_short_url(short, area_mm=size))
        for size in sizes
    ]
    codes.append(
        (
            f"{CARD_LABEL} mm + http",
            badge_codes.qr_for_payload("http://" + short, area_mm=sizes[0]),
        )
    )

    width = max(70.0, 8.0 + len(codes) * 13.0)
    height = 45.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" '
        f'viewBox="0 0 {width:g} {height:g}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="5" y="6" font-size="3" font-family="sans-serif">'
        "Badge QR short-link size test - print at 100%</text>",
        '<text x="5" y="10" font-size="1.8" font-family="sans-serif" fill="#444">'
        f"Target: {html.escape(short)} (scheme-less)</text>",
    ]
    for index, (label, qr) in enumerate(codes):
        parts += _qr_group(qr, 5 + index * 13, 14, label)
    parts += [
        '<text x="5" y="38" font-size="1.2" font-family="sans-serif" fill="#444">'
        f"The first square is the card's real size ({CARD_LABEL} mm); later ones are"
        " controls. Print at 100%."
        "</text>",
        "</svg>",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for label, qr in codes:
        print(
            f"{label} -> v{qr.version}, {qr.layout.total_modules} modules, "
            f"{qr.module_mm:.4f} mm/module"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--short-url", help="e.g. http://i8c.cn/RkhQm")
    parser.add_argument(
        "--sizes",
        default=f"{CARD_SIZE_MM:.2f},5.6,6.4,7.2,8.0",
        help="comma-separated mm; the first is the card's real file size",
    )
    parser.add_argument(
        "--factions",
        action="store_true",
        help="render the faction-major operator sheet instead of the size sweep",
    )
    parser.add_argument(
        "--per-faction", type=int, default=6, help="operators per faction (default 6)"
    )
    parser.add_argument("--png", action="store_true", help="also write a 1200 dpi PNG")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.factions:
        out = args.out or (PREVIEW_DIR / "qr-faction-test.svg")
        rows, missing = _sample_operators(max(1, args.per_faction))
        if not rows:
            raise SystemExit("干员数据库为空，先确认素材库索引存在")
        _write_faction_sheet(
            rows, out, per_faction=max(1, args.per_faction), png=args.png
        )
        for faction in missing:
            print(f"  ! 数据库里没有“{faction}”阵营的干员，已跳过")
        for faction, members in rows:
            names = "、".join(item.name for item in members)
            print(f"  {faction}: {names}")
        return 0

    if not args.short_url:
        parser.error("需要 --short-url，或改用 --factions")
    args.out = args.out or DEFAULT_OUT
    return _write_size_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
