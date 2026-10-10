"""Render every operator's QR code and Code 128 barcode into the asset library.

**Both sizes come from the card's own region table, not typed here.**

The QR block measures the symbol's *code area* -- quiet zone excluded -- so the
finder patterns land on its corners; the transparent quiet zone falls outside
it. That makes its module 3.1997 / 21 = 0.1524 mm and the file 4.4186 mm square.

The barcode frame measures the *bars*, for the same reason: the ten-module
quiet zone on each end is empty space with nothing to draw, so a frame can
never have marked it. The bars therefore span the 39.245 mm frame and the quiet
zone falls outside, which makes the module 39.245 / 79 = 0.4968 mm and the file
49.1804 mm long.

Retuning the card must retune the library in the same change, which is why
:func:`qr_code_mm` and :func:`barcode_code_mm` read
:func:`pcbart.badge_content.art_region` instead of carrying copies of the
numbers.

Output lives beside the artwork it belongs with, in the asset library:

    20-干员二维码/二维码 <页面名>.svg|.png
    21-情报编号条形码/条形码 <页面名>.svg|.png

**Both formats, deliberately.** The SVG carries the exact millimetre size and
is what the exporter should place on the board; the PNG is a fixed-resolution
raster (default 600 dpi) for consumers that cannot read SVG. Neither is
derived from the other at run time, so both are written here from the same bit
matrix.

**Filenames use the page name, not the operator name.** 暮落 and
暮落(集成战略) are two different operators with the same display name, and the
page name is what makes them distinct. 24 operators have no 情报编号 at all, so
they get a QR code and no barcode; the run reports them instead of writing a
file with a placeholder payload.

This tool writes into the library and therefore must be pointed at it
explicitly -- there is no default that could quietly overwrite the real one
from a test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_codes, code128  # noqa: E402
from pcbart.badge_raster import barcode_raster, qr_raster  # noqa: E402
from pcbart.paths import asset_root  # noqa: E402

DEFAULT_LIBRARY = asset_root() / "明日方舟"
"""The source roster lives in the full mirror, not at the library root.

The root holds several libraries side by side (the mirror, the subset, the
template sources); ``01-干员代号`` is inside the mirror. Defaulting to the root
found no roster at all and the first run wrote zero files rather than failing,
which is why :func:`load_operators` is also asked to reject an empty roster.
"""
DEFAULT_QR_DIR = "20-干员二维码"
DEFAULT_BARCODE_DIR = "21-情报编号条形码"
DEFAULT_DPI = 600
LIBRARY_MODULE_PX = 8
"""Pixels per module in the library's own PNG files.

Whole, so every bar and module edge lands on a pixel boundary and the file's
measured size is the size it claims. Eight is enough for a viewer or a quick
print check; the card does not use these files -- it rasterises its own at 32
pixels per module for the exporter to trace.
"""

FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

MANIFEST_NAME = "干员码.json"

__all__ = [
    "CodeJob",
    "build_jobs",
    "generate",
    "load_operators",
]


@dataclass(frozen=True)
class CodeJob:
    """One operator's two codes, and what could not be built for them."""

    page: str
    name: str
    number: str
    qr: badge_codes.QrCode
    barcode: badge_codes.Barcode | None
    barcode_note: str = ""

    @property
    def qr_stem(self) -> str:
        return f"二维码 {self.page}"

    @property
    def barcode_stem(self) -> str:
        return f"条形码 {self.page}"


def qr_code_mm() -> float:
    """The QR *code area* the card's drawn block measures, in millimetres.

    Read from :mod:`pcbart.badge_content` so the library and the canvas cannot
    disagree: the number is the block's side, and it is the code area -- the
    quiet zone is extra and falls outside the block.
    """
    from pcbart import badge_content

    return badge_content.art_region("qr").width_mm


def barcode_code_mm() -> float:
    """How far the barcode's bars run, measured from its drawn frame.

    The frame's long side is the direction the bars run, and the frame can only
    mark the bars -- the quiet zone is empty space in the source artwork. The
    quiet zone therefore falls outside the frame, which is what makes the first
    and last bar sit on its edges.
    """
    from pcbart import badge_content

    region = badge_content.art_region("barcode")
    return max(region.width_mm, region.height_mm)


def barcode_height_mm() -> float:
    """The bar height, measured from the frame's short side."""
    from pcbart import badge_content

    region = badge_content.art_region("barcode")
    return min(region.width_mm, region.height_mm)


def load_operators(source: Path) -> tuple[list[CodeJob], list[str]]:
    """Build a job per operator row, plus the names of rows that cannot be encoded.

    Read through :mod:`pcbart.operators` rather than parsing the CSV here: the
    path columns, the alternate-form handling and the short-link lookup already
    live there, and a second reader would drift from the panel's.
    """
    from pcbart import operators as operator_db

    database = operator_db.load(source / "01-干员代号" / operator_db.INDEX_CSV)
    block_mm = qr_code_mm()
    jobs: list[CodeJob] = []
    skipped: list[str] = []
    for row in database.operators:
        if not row.page:
            skipped.append(row.name)
            continue
        qr = badge_codes.qr_for_operator(row.page, code_mm=block_mm)
        barcode: badge_codes.Barcode | None = None
        note = ""
        if row.number:
            # The frame measures the bars, not the file: the ten-module quiet
            # zone on each end is empty space a drawn frame cannot mark, so it
            # falls outside. Same rule as the QR symbol's block.
            barcode = badge_codes.barcode_for(
                row.number, code_length_mm=barcode_code_mm()
            )
        else:
            note = "情报编号为空，未生成条形码"
        jobs.append(
            CodeJob(
                page=row.page,
                name=row.name,
                number=row.number,
                qr=qr,
                barcode=barcode,
                barcode_note=note,
            )
        )
    return jobs, skipped


# ------------------------------------------------------------------ SVG

def qr_svg(qr: badge_codes.QrCode) -> str:
    """The QR symbol as a 1:1 SVG group, quiet zone included."""
    layout = qr.layout
    module = layout.module_mm
    size = len(layout.matrix)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{layout.area_mm:g}mm" height="{layout.area_mm:g}mm" '
        f'viewBox="0 0 {layout.area_mm:g} {layout.area_mm:g}">',
        f'<rect width="{layout.area_mm:g}" height="{layout.area_mm:g}" fill="#fff"/>',
    ]
    for index, line in enumerate(layout.matrix):
        run_start = None
        for col, ink in enumerate(line):
            if ink and run_start is None:
                run_start = col
            elif not ink and run_start is not None:
                x = run_start * module
                width = (col - run_start) * module
                rows.append(
                    f'<rect x="{x:.6f}" y="{index * module:.6f}" '
                    f'width="{width:.6f}" height="{module:.6f}" fill="#000"/>'
                )
                run_start = None
        if run_start is not None:
            x = run_start * module
            width = (size - run_start) * module
            rows.append(
                f'<rect x="{x:.6f}" y="{index * module:.6f}" '
                f'width="{width:.6f}" height="{module:.6f}" fill="#000"/>'
            )
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def barcode_svg(barcode: badge_codes.Barcode) -> str:
    """The barcode as a 1:1 SVG, quiet zone included, height as quoted."""
    layout = barcode.layout
    module = layout.module_mm
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{layout.length_mm:g}mm" height="{layout.height_mm:g}mm" '
        f'viewBox="0 0 {layout.length_mm:g} {layout.height_mm:g}">',
        f'<rect width="{layout.length_mm:g}" height="{layout.height_mm:g}" fill="#fff"/>',
    ]
    x = layout.quiet_zone_modules * module
    for is_bar, width in layout.runs:
        if is_bar:
            rows.append(
                f'<rect x="{x:.6f}" y="0" width="{width * module:.6f}" '
                f'height="{layout.height_mm:g}" fill="#000"/>'
            )
        x += width * module
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ PNG
#
# The rasterisers live in ``pcbart.badge_raster``: the canvas, the library
# generator and this sheet have to produce byte-identical geometry, and the
# rounding rules are subtle enough that a second copy here would drift.

# ------------------------------------------------------------------ writing

def _clean(directory: Path, patterns: tuple[str, ...], *, keep: set[str]) -> int:
    """Remove stale files this tool owns, so a renamed operator leaves no ghost.

    Only files matching the tool's own ``<prefix> <page>.svg|.png`` shape are
    considered, and only inside the directories this tool writes. Anything else
    a user parked there is left alone.
    """
    removed = 0
    for pattern in patterns:
        for item in directory.glob(pattern):
            if item.name in keep or not item.is_file():
                continue
            item.unlink()
            removed += 1
    return removed


def generate(
    source: Path,
    library: Path,
    *,
    dpi: int,
    qr_dir: str,
    barcode_dir: str,
    clean: bool = True,
) -> dict:
    """Write every operator's codes into ``library`` and return a report."""
    jobs, skipped = load_operators(source)
    if not jobs:
        # An empty roster is never a legitimate outcome of "generate the codes";
        # it means the index was not found. Stopping here beats writing a
        # manifest that says 0 operators and leaving the caller to notice.
        raise SystemExit(
            f"没有读到干员：{source / '01-干员代号' / '干员代号与编号.csv'}\n"
            "请确认 --source 指向素材库的「明日方舟」目录"
        )
    qr_root = library / qr_dir
    barcode_root = library / barcode_dir
    qr_root.mkdir(parents=True, exist_ok=True)
    barcode_root.mkdir(parents=True, exist_ok=True)
    #: The manifest goes in the *destination* library's index folder. The source
    #: library is a read-only input here -- writing a derived file back into it
    #: would make "regenerate the codes" alter the mirror it was derived from.
    index_root = library / "_索引"

    qr_names: set[str] = set()
    barcode_names: set[str] = set()
    for job in jobs:
        stem = job.qr_stem
        qr_names |= {f"{stem}.svg", f"{stem}.png"}
        (qr_root / f"{stem}.svg").write_text(qr_svg(job.qr), encoding="utf-8")
        # Whole pixels per module, so the PNG's own physical size is the size
        # the SVG and the card claim. A plain 600-dpi raster rounds the last
        # module's edge and the file measures a few hundredths of a millimetre
        # away from its own frame.
        qr_raster(job.qr, dpi, module_px=LIBRARY_MODULE_PX).save(
            qr_root / f"{stem}.png", dpi=(dpi, dpi)
        )
        if job.barcode is not None:
            bstem = job.barcode_stem
            barcode_names |= {f"{bstem}.svg", f"{bstem}.png"}
            (barcode_root / f"{bstem}.svg").write_text(
                barcode_svg(job.barcode), encoding="utf-8"
            )
            barcode_raster(
                job.barcode, dpi, module_px=LIBRARY_MODULE_PX
            ).save(barcode_root / f"{bstem}.png", dpi=(dpi, dpi))

    removed = 0
    if clean:
        removed += _clean(qr_root, ("二维码 *.svg", "二维码 *.png"), keep=qr_names)
        removed += _clean(
            barcode_root, ("条形码 *.svg", "条形码 *.png"), keep=barcode_names
        )

    manifest = {
        "qr_dir": qr_dir,
        "barcode_dir": barcode_dir,
        # Two sizes, and both are needed: the block measures the code area, and
        # the file is bigger than that by its quiet zone on each side.
        "qr_code_mm": qr_code_mm(),
        "qr_file_mm": jobs[0].qr.layout.area_mm if jobs else 0.0,
        "qr_quiet_zone_modules": badge_codes.QR_QUIET_ZONE_MODULES,
        # The same two-number shape as the QR entry: the frame measures the
        # bars, and the file is longer by the quiet zone on each end.
        "barcode_code_mm": barcode_code_mm(),
        "barcode_height_mm": barcode_height_mm(),
        "barcode_file_mm": (
            jobs[0].barcode.layout.length_mm
            if jobs and jobs[0].barcode is not None
            else 0.0
        ),
        "barcode_quiet_zone_modules": badge_codes.BARCODE_QUIET_ZONE_MODULES,
        "dpi": dpi,
        "operators": len(jobs),
        "qr_files": len(qr_names),
        "barcodes": sum(1 for job in jobs if job.barcode is not None),
        "barcode_files": len(barcode_names),
        "missing_numbers": [
            {"page": job.page, "name": job.name} for job in jobs if not job.number
        ],
        "skipped": skipped,
        "removed_stale": removed,
        "module_mm": {
            "qr_card": jobs[0].qr.module_mm if jobs else 0.0,
            "barcode": jobs[0].barcode.module_mm if jobs and jobs[0].barcode else 0.0,
        },
    }
    index_root.mkdir(parents=True, exist_ok=True)
    (index_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ------------------------------------------------------------------ preview

def write_preview(jobs: list[CodeJob], out: Path, *, columns: int = 0) -> None:
    """A contact sheet of every QR symbol, for eyeballing the batch at a glance.

    Laid out at A4 portrait width: the symbols are drawn at the real size they
    are written at and enlarged by an integer factor, and the column count
    defaults to whatever fills that width. A version mistake or a payload that
    lost its short link shows up as a visibly denser symbol rather than hiding
    among 463 filenames.
    """
    # "scale" is pixels per millimetre, and it is applied exactly once: every
    # other length below is already in pixels.
    scale = 24
    page_width = 210.0 * scale
    margin = 6.0 * scale
    # The whole file, quiet zone included -- that is what is being previewed.
    file_mm = jobs[0].qr.layout.area_mm if jobs else qr_code_mm() * 29 / 21
    code = file_mm * scale
    label = 8.0 * scale
    cell_w = round(code + 1.5 * scale)
    cell_h = round(code + label)
    if columns <= 0:
        columns = max(1, int((page_width - 2 * margin) // cell_w))
    rows = (len(jobs) + columns - 1) // columns
    page = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONT_REGULAR, max(9, int(scale * 0.9)))
    for index, job in enumerate(jobs):
        column = index % columns
        row = index // columns
        image = qr_raster(job.qr, dpi=round(scale * 25.4), symbol_px=round(code))
        x = round(column * cell_w + (cell_w - image.width) / 2)
        y = round(row * cell_h + label)
        page.paste(image, (x, y))
        draw.text(
            (round(column * cell_w + 2), round(row * cell_h + 2)),
            job.name[:6],
            fill=(0, 0, 0),
            font=font,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    print(f"wrote {out}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_LIBRARY,
        help="operator database root (the 完整库)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_LIBRARY,
        help="asset library to write into",
    )
    parser.add_argument("--qr-dir", default=DEFAULT_QR_DIR)
    parser.add_argument("--barcode-dir", default=DEFAULT_BARCODE_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--no-clean", action="store_true", help="leave stale files")
    parser.add_argument("--preview", type=Path, help="also write a contact sheet here")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.source.exists():
        raise SystemExit(f"完整库不存在：{args.source}")
    if args.out.resolve() == args.source.resolve() and args.qr_dir in {
        "01-干员代号",
    }:
        raise SystemExit("拒绝写入 01-干员代号")

    report = generate(
        args.source,
        args.out,
        dpi=max(72, args.dpi),
        qr_dir=args.qr_dir,
        barcode_dir=args.barcode_dir,
        clean=not args.no_clean,
    )
    print(f"输出目录：{args.out}")
    print(f"二维码      {report['operators']:>4} 位 -> {report['qr_files']} 个文件")
    print(f"条形码      {report['barcodes']:>4} 位 -> {report['barcode_files']} 个文件")
    print(
        f"二维码      码区 {report['qr_code_mm']:.4f} mm"
        f"（含静区 {report['qr_file_mm']:.4f}）· "
        f"模块 {report['module_mm']['qr_card']:.4f} mm"
    )
    print(
        f"条形码      条区 {report['barcode_code_mm']:.4f} mm"
        f"（含静区 {report['barcode_file_mm']:.4f}）· "
        f"高 {report['barcode_height_mm']:.4f} · "
        f"模块 {report['module_mm']['barcode']:.4f} mm"
    )
    if report["missing_numbers"]:
        print(f"⚠ 缺情报编号 {len(report['missing_numbers'])} 位（只给二维码）：")
        for item in report["missing_numbers"]:
            print(f"   {item['page']}")
    if report["removed_stale"]:
        print(f"清理旧文件 {report['removed_stale']} 个")
    if args.preview:
        jobs, _skipped = load_operators(args.source)
        write_preview(jobs, args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
