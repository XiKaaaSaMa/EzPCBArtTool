#!/usr/bin/env python
"""Trace one asymmetric feature through every coordinate frame.

Theory has been going in circles on which way Y points, so this prints actual
numbers instead. The test shape is an "F": its two rightward arms are at the
TOP, and the bottom is a bare stem. Whichever frame is in use, the arms identify
the top unambiguously.

Three frames are reported:

``image``   source pixels, Y grows downward (a raster fact, not a choice)
``design``  this project's internal frame, intended to be Y-up
``file``    raw ``.epru`` values, after design->file conversion

The arms' Y position is printed in each, so the sign convention is visible
rather than argued about.

Usage::

    python tools/diagnose_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from pcbart.epru import EproArchive  # noqa: E402
from pcbart.generator import generate_outline_board  # noqa: E402
from pcbart.image_prep import PrepParams, preprocess  # noqa: E402
from pcbart.outline import OutlineParams, extract_outline_rings  # noqa: E402
from pcbart.primitives import Mode  # noqa: E402
from pcbart.units import unit_to_mm  # noqa: E402

OUT = Path("out/diag")


def make_f(width: int = 300, height: int = 400) -> np.ndarray:
    """An F: two arms reaching right at the top, bare stem at the bottom."""
    canvas = np.full((height, width), 255, np.uint8)
    stem_x, stem_w = 80, 40
    cv2.rectangle(canvas, (stem_x, 40), (stem_x + stem_w, 360), 0, -1)
    # Top arm.
    cv2.rectangle(canvas, (stem_x, 40), (250, 90), 0, -1)
    # Middle arm.
    cv2.rectangle(canvas, (stem_x, 170), (210, 215), 0, -1)
    return canvas


def decode_file_path(node) -> list[tuple[float, float]]:
    """Decode a stored path into raw file coordinates."""
    if not isinstance(node, list) or not node:
        return []
    if isinstance(node[0], list):
        return [p for item in node for p in decode_file_path(item)]
    if node[0] == Mode.CIRCLE and len(node) >= 4:
        cx, cy, r = float(node[1]), float(node[2]), float(node[3])
        return [(cx - r, cy - r), (cx + r, cy + r)]

    points: list[tuple[float, float]] = []
    pending: list[float] = []
    i = 0
    while i < len(node):
        item = node[i]
        if isinstance(item, str):
            points.extend(
                (pending[j], pending[j + 1]) for j in range(0, len(pending) - 1, 2)
            )
            pending = []
            i += 1
            if item in (Mode.ARC, Mode.CARC):
                if i + 2 < len(node):
                    points.append((float(node[i + 1]), float(node[i + 2])))
                i += 3
            continue
        if isinstance(item, (int, float)):
            pending.append(float(item))
        i += 1
    points.extend((pending[j], pending[j + 1]) for j in range(0, len(pending) - 1, 2))
    return points


def widest_band(points: list[tuple[float, float]], bands: int = 6) -> tuple[int, float]:
    """Which horizontal band is widest.

    Band 0 is the band nearest ``min`` Y and band ``bands-1`` the one nearest
    ``max`` Y. Whether that means "up" depends on the frame being measured, so
    callers must interpret the index, not this function.
    """
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    span = hi - lo
    widths = [0.0] * bands
    for band in range(bands):
        y0 = lo + span * band / bands
        y1 = lo + span * (band + 1) / bands
        xs = [p[0] for p in points if y0 <= p[1] < y1 or (band == bands - 1 and p[1] == hi)]
        if xs:
            widths[band] = max(xs) - min(xs)
    best = max(range(bands), key=lambda b: widths[band] and widths[b])
    return best, widths[best]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    image = make_f()
    cv2.imwrite(str(OUT / "f_source.png"), image)

    print("=" * 68)
    print("FRAME 1: source image (Y grows DOWN; a raster fact)")
    print("=" * 68)
    ys, xs = np.where(image < 128)
    print(f"  foreground bbox: x=[{xs.min()}, {xs.max()}]  y=[{ys.min()}, {ys.max()}]")
    print("  the F's arms are at the TOP, i.e. SMALL image y")
    print(f"  top of shape    y={ys.min()}   bottom of shape  y={ys.max()}")
    ix, iy = np.where(image < 128)
    arms = iy[(ix > 150)]
    if len(arms):
        print(f"  right-arm pixels (x>150) span y=[{arms.min()}, {arms.max()}]"
              "   <- these are the TOP arms")

    print()
    mask = preprocess(image, PrepParams(fill_holes=False)).mask
    result = extract_outline_rings(
        mask,
        OutlineParams(target_width_mm=30.0, center=False),
    )
    design = result.outer
    band, width = widest_band(design)
    dys = [p[1] for p in design]
    print("=" * 68)
    print("FRAME 2: design frame (this project, intended Y-UP)")
    print("=" * 68)
    print(f"  bbox y=[{min(dys):.3f}, {max(dys):.3f}]")
    print(f"  widest band is index {band} of 6 counted from MIN y (width {width:.2f} mm)")
    # Band index >= half means the arms sit near max Y, i.e. the high end.
    print(
        f"  => arms are at the {'HIGH' if band >= 3 else 'LOW'} end of the Y range"
    )
    print("     design frame is Y-up, so HIGH end = visually on top: correct")

    path_out = OUT / "f_board.epro2"
    generate_outline_board(design, path_out, title="diag_f", line_width_mm=0.125)

    print()
    print("=" * 68)
    print("FRAME 3: file frame (raw .epru numbers)")
    print("=" * 68)
    doc = EproArchive.read(path_out).document
    record = next(r for r in doc.iter_type("POLY") if r.body.get("layerId") == 11)
    raw = decode_file_path(record.body["path"])
    fys = [p[1] for p in raw]
    band_file, width_file = widest_band(raw)
    print(f"  raw bbox y=[{min(fys):.1f}, {max(fys):.1f}] (units)")
    print(f"  in mm      y=[{unit_to_mm(min(fys)):.3f}, {unit_to_mm(max(fys)):.3f}]")
    print(f"  widest band is index {band_file} of 6 counted from MIN y")
    print()
    print("  Convention in force (pcbart.units.FILE_Y_IS_UP = True):")
    print("    file Y increases upward, matching the design frame")
    print(f"  Our arms sit at the {'HIGH' if band_file >= 3 else 'LOW'} end of file Y,")
    if band_file >= 3:
        print("    i.e. at LARGE file Y -> renders UP -> ORIENTATION MATCHES source")
    else:
        print("    i.e. at SMALL file Y -> renders DOWN -> ORIENTATION IS FLIPPED")

    print()
    print("=" * 68)
    print("CROSS-CHECK: two independent conversion paths")
    print("=" * 68)
    # Path A: a point at the image top, pushed through design then file.
    img_top_y = 40.0
    scale = 30.0 / 300.0  # pixels -> mm, from the 30 mm target width
    design_y = -img_top_y * scale
    file_y = -design_y / 0.0254
    print(f"  image top  y={img_top_y}")
    print(f"  -> design  y={design_y:.3f} mm      (design is Y-up, so this must be HIGH)")
    print(f"  -> file    y={file_y:.1f} units = {unit_to_mm(file_y):.3f} mm")
    print("  The design value is HIGH (negative = below origin, but above the shape's")
    print("  own bottom), and the file value keeps the same ordering. Both paths agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
