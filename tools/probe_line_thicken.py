"""Probe: hard two-tone threshold, then thicken the strokes.

The model returns a *soft response*, and the previous probe thresholded it
adaptively at a fixed share (6%). That share is a guess about how much of the
picture is line, and on a mechanical subject it is the wrong guess: Lancet-2's
response has a long continuous tail rather than two clusters, so a 6% cut keeps
only the strongest panel edges and drops the rest of the drawing.

This probe answers the user's own suggestion -- "直接转成黑白两色、然后加粗线条"
-- by scanning the two knobs separately and showing every combination:

* **the threshold**, as a quantile of the response (with an absolute floor, for
  the same reason :func:`tools.probe_line_redraw` needs one: a flat field puts
  the quantile inside its own zero population and inks the whole frame);
* **the thickening**, as a dilation radius in **millimetres on the finished
  board**. Thickening is what buys a printable stroke out of a thin one, and it
  is the half that a purely threshold-based reading cannot do.

The order is threshold-then-thicken, not the reverse. Growing the *response*
before thresholding would change which pixels pass; growing the *mask* after
thresholding only makes the chosen strokes wider, which is what was asked for.

Usage::

    python tools/probe_line_thicken.py --source out/_line_redraw_elite0
    python tools/probe_line_thicken.py --source path/to/response.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402

# The threshold rule is shared with the redraw probe rather than re-stated
# here: it carries three clamps that were each measured against a failure, and
# a second copy would drift. See that module's docstring.
from tools.probe_line_redraw import ADAPTIVE_FLOOR, adaptive_level  # noqa: E402

#: Default quantiles to sweep, in percent of the response.
DEFAULT_QUANTILES = (5.0, 8.0, 12.0, 18.0, 25.0)

#: Default thickening radii, in millimetres on the finished board.
DEFAULT_RADII_MM = (0.0, 0.08, 0.15, 0.22)

#: Absolute floor for the adaptive threshold, re-exported for the report.
FLOOR = ADAPTIVE_FLOOR

#: Long side the drawing is placed at, for millimetres <-> pixels.
DRAWN_MM = 76.2


def thicken(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Grow every stroke by ``radius_px`` on all sides.

    Dilation with a disc, which is the operation "加粗" actually names: a
    one-pixel line becomes ``2r+1`` wide and stays centred where it was.

    ⚠️ **A dilate-then-erode pair does not thicken a line at all.** Opening a
    thin curve returns the curve, so the measured width would be identical at
    every setting -- the same trap :func:`pcbart.lineart._widen_lines` records.
    The growth has to be left in place.

    ⚠️ **Dilation also closes the gap between neighbouring strokes.** Once the
    gap is under twice the radius the two merge into one blob, which on a
    technical drawing full of parallel panel lines is the real cost of
    thickening and the thing the contact sheet is for looking at.
    """
    if radius_px < 0.5:
        return mask
    size = int(round(radius_px)) * 2 + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), element) > 0


def _cell(label: str, image: np.ndarray, tile: int) -> np.ndarray:
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = bgr.shape[:2]
    factor = min(tile / width, tile / height)
    resized = cv2.resize(
        bgr,
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    cell = np.full((tile + 32, tile, 3), 236, np.uint8)
    y0 = 32 + (tile - resized.shape[0]) // 2
    x0 = (tile - resized.shape[1]) // 2
    cell[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(cell, (0, 0), (tile, 31), (43, 43, 43), -1)
    cv2.putText(
        cell, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 238, 238), 1,
        cv2.LINE_AA,
    )
    return cell


def contact_sheet(entries, path: Path, tile: int, columns: int) -> None:
    cells = [_cell(label, image, tile) for label, image in entries]
    while len(cells) % columns:
        cells.append(np.full((tile + 32, tile, 3), 236, np.uint8))
    rows = [
        np.hstack(cells[index : index + columns])
        for index in range(0, len(cells), columns)
    ]
    imwrite_unicode(str(path), np.vstack(rows))


def current_reading(path: Path) -> np.ndarray | None:
    """The shipped algorithm's own output for this sample, if it was written."""
    stem = path.name.split("__")[0]
    neighbour = path.with_name(f"{stem}__90-current-algorithm.png")
    if not neighbour.exists():
        return None
    plane = imread_unicode(str(neighbour), cv2.IMREAD_UNCHANGED)
    return plane <= 128


def process(response_path: Path, outdir: Path, args) -> list[tuple]:
    response = imread_unicode(str(response_path), cv2.IMREAD_UNCHANGED)
    if response is None:
        raise ValueError(f"无法读取 {response_path}")
    if response.ndim == 3:
        response = cv2.cvtColor(response, cv2.COLOR_BGR2GRAY)

    height, width = response.shape[:2]
    px_per_mm = max(height, width) / float(args.drawn_mm)
    label = response_path.name.split("__")[0]

    entries = [
        ("model response", response),
    ]
    rows = []
    for quantile in args.quantiles:
        level = adaptive_level(response, quantile)
        base = response <= level
        for radius_mm in args.radii_mm:
            radius_px = radius_mm * px_per_mm
            mask = thicken(base, radius_px)
            coverage = 100.0 * float(mask.mean())
            components, _ = cv2.connectedComponents(
                mask.astype(np.uint8), connectivity=8
            )
            rows.append((quantile, level, radius_mm, radius_px, coverage, components - 1, mask))
            stem = f"{label}__q{quantile:04.1f}_r{radius_mm:.2f}mm"
            imwrite_unicode(
                str(outdir / f"{stem}.png"), np.where(mask, 0, 255).astype(np.uint8)
            )
            entries.append(
                (
                    f"{quantile:.0f}% t={level:.0f} · {radius_mm:.2f}mm",
                    np.where(mask, 0, 255).astype(np.uint8),
                )
            )

    current = current_reading(response_path)
    if current is not None:
        entries.append(("current algorithm", np.where(current, 0, 255).astype(np.uint8)))

    contact_sheet(
        entries,
        outdir / f"{label}__grid.png",
        args.tile,
        columns=len(args.radii_mm) + 1,
    )

    print(f"  {label}: {width}x{height}, {px_per_mm:.2f} px/mm")
    for quantile, level, radius_mm, radius_px, coverage, comps, _mask in rows:
        print(
            f"    分位{quantile:5.1f}% (t={level:5.1f}) + 加粗{radius_mm:.2f}mm"
            f"({radius_px:4.1f}px) → 墨量 {coverage:5.2f}%  {comps:5d} 块"
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=str(ROOT / "out" / "_line_redraw_elite0"),
        help="一个目录（取其中 *__01-raw-response.png）或单个响应图",
    )
    parser.add_argument("--outdir", default=str(ROOT / "out" / "_line_thicken"))
    parser.add_argument(
        "--quantiles", nargs="*", type=float, default=list(DEFAULT_QUANTILES)
    )
    parser.add_argument(
        "--radii-mm", nargs="*", type=float, default=list(DEFAULT_RADII_MM),
        help="加粗半径，单位是成品板上的毫米",
    )
    parser.add_argument("--tile", type=int, default=300)
    parser.add_argument("--drawn-mm", type=float, default=DRAWN_MM)
    args = parser.parse_args(argv)

    source = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        targets = sorted(source.glob("*__01-raw-response.png"))
    else:
        targets = [source]
    if not targets:
        raise SystemExit(f"没有找到响应图：{source}")

    report = [
        "# 硬二值 + 加粗扫描（未接入软件）",
        "",
        f"- 输入：`{source}`",
        f"- 输出：`{outdir}`",
        f"- 阈值分位：{args.quantiles}（阈值 = max({FLOOR:.0f}, 该分位)）",
        f"- 加粗半径：{args.radii_mm} mm（成品长边 {args.drawn_mm} mm）",
        "",
        "**顺序是「先二值、再加粗」**：加粗作用在已选定的笔画上、只让它们变宽；"
        "反过来先加粗响应会改变哪些像素过线，那是另一回事。",
        "",
    ]

    for target in targets:
        print(f"处理 {target.name} …")
        rows = process(target, outdir, args)
        report.append(f"## {target.stem.split('__')[0]}")
        report.append("")
        report.append("| 分位 | 阈值 | 加粗(mm) | 加粗(px) | 墨量 | 色块数 |")
        report.append("|---|---|---|---|---|---|")
        for quantile, level, radius_mm, radius_px, coverage, comps, _m in rows:
            report.append(
                f"| {quantile:.1f}% | {level:.1f} | {radius_mm:.2f} | "
                f"{radius_px:.1f} | {coverage:.2f}% | {comps} |"
            )
        report.append("")

    (outdir / "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n输出目录：{outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
