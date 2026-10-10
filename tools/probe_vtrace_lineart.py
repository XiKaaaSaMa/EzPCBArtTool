"""Can VTracer lower the line-art step's error rate? A measurement, not a claim.

The line-art step is a **per-pixel** judgement today: one Otsu level over the
subject's luminance, applied to every pixel independently. VTracer offers two
different things, and they are easy to conflate:

* **arc fitting** -- it changes how an outline is *expressed*. Applying it
  after the mask exists cannot change which pixels were classified as ink;
  it can only clean the mask's raggedness if the smoothed geometry is
  rasterised back.
* **colour segmentation** -- it groups pixels into flat regions by colour
  similarity before anything is thresholded. That *can* change the
  classification, because the decision becomes per region instead of per
  pixel.

So four readings are produced and measured side by side:

======  ====================================================================
A       现状：``line_art()`` fill 模式（主体上做全局 Otsu）
B       A 的掩膜再用 spline→弧描一遍、光栅化回来（只清毛刺）
C       VTracer 彩色分割 → 每个区域按自己的颜色定墨（改判据）
D       C 的结果再过一遍 B（既改判据又清毛刺）
======  ====================================================================

Each is compared on the things the user actually reported: whether fine
detail survives, whether the subject's outer boundary is complete, how much
stray speckle it sheds, and whether the answer holds up when the picture is
small.

Usage::

    python tools/probe_vtrace_lineart.py
    python tools/probe_vtrace_lineart.py --image "P:/PS抠图/由比滨透明背景.png"
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pcbart.paths import asset_root  # noqa: E402

_LIBRARY = asset_root() / "明日方舟" / "08-立绘-精英2"

#: ⚠️ The library entries are resolved under ``PCBART_ASSET_ROOT`` because the
#: artwork is gigabytes of artwork that lives outside the repository. The
#: 由比滨 entry is a path into this developer's own PS reference folder, which
#: nobody else has -- the script reports a missing subject and moves on rather
#: than failing, so the remaining subjects still produce the comparison.
DEFAULT_SUBJECTS = (
    ("由比滨-透明背景", r"P:\PS抠图\由比滨透明背景.png"),
    ("立绘-能天使-精英2", str(_LIBRARY / "立绘 能天使 2.png")),
    ("立绘-Miss.Christine-精英2", str(_LIBRARY / "立绘 Miss.Christine 2.png")),
)


# --- rendering VTracer's colour SVG back to a class map ------------------


def _render_colour(svg: str, width: int, height: int):
    """VTracer's colour SVG as an RGB array."""
    import numpy as np
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    if not renderer.isValid():
        return None
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, False)
    renderer.render(painter)
    painter.end()
    raw = np.frombuffer(
        image.constBits(), np.uint8
    ).reshape(height, image.bytesPerLine() // 4, 4)[:, :width, :]
    return raw[:, :, [2, 1, 0]].copy()  # BGR order, to match OpenCV


def _colour_layers(svg: str, width: int, height: int):
    """``[(fill_rgb, mask)]`` for every distinct fill colour VTracer emitted."""
    import numpy as np
    import re

    flat = _render_colour(svg, width, height)
    if flat is None:
        return []
    fills = re.findall(r'<path[^>]*\bfill="(#[0-9a-fA-F]{6})"', svg)
    out = []
    for value in dict.fromkeys(fills):
        rgb = np.array(
            [int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)],
            dtype=np.uint8,
        )
        # The renderer writes BGR; the SVG names RGB.
        bgr = rgb[::-1]
        mask = np.all(flat == bgr, axis=2)
        if mask.any():
            out.append((rgb, mask))
    return out


def _trace_svg(image, colour: bool, precision: int = 6, difference: int = 16):
    """Run VTracer once and hand back the raw SVG text."""
    import numpy as np
    import vtracer

    from pcbart.image_prep import imwrite_unicode

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "in.png"
        target = Path(folder) / "out.svg"
        # VTracer's binding wants a file and reports nothing about failure, so
        # the write is done through the project's own unicode-safe helper.
        if image.ndim == 2:
            plane = image
        else:
            plane = image[:, :, ::-1]  # BGR -> RGB for the PNG writer
        imwrite_unicode(source, np.ascontiguousarray(plane))
        vtracer.convert_image_to_svg_py(
            str(source),
            str(target),
            colormode="color" if colour else "binary",
            hierarchical="cutout" if colour else "stacked",
            mode="spline",
            filter_speckle=2 if colour else 4,
            color_precision=precision if colour else 6,
            layer_difference=difference if colour else 16,
            corner_threshold=60,
            length_threshold=2.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        return target.read_text(encoding="utf-8")


# --- the four readings ---------------------------------------------------


def reading_a_current(image, alpha):
    """现状：fill 模式的 Otsu。"""
    import numpy as np

    from pcbart.lineart import line_art

    if alpha is not None:
        rgba = np.dstack([image, alpha])
    else:
        rgba = image
    return line_art(rgba)


def reading_c_colour_regions(image, alpha):
    """VTracer 彩色分割，逐区域按自己的颜色定墨。"""
    import cv2
    import numpy as np

    from pcbart.image_prep import imread_unicode  # noqa: F401 - parity import

    height, width = image.shape[:2]
    svg = _trace_svg(image, colour=True)
    layers = _colour_layers(svg, width, height)
    if not layers:
        return None

    # ⚠️ The decision is made **per region**, which is the whole point. A
    # region's own mean colour is measured, and the level is Otsu over those
    # region colours weighted by area -- the same statistic the current code
    # computes per pixel, but now it cannot split a region down the middle.
    weights = np.array([mask.sum() for _, mask in layers], dtype=float)
    greys = np.array(
        [
            float(cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2GRAY)[0, 0])
            for rgb, _ in layers
        ]
    )
    order = np.argsort(greys)
    sorted_grey = greys[order]
    sorted_weight = weights[order]
    cumulative = np.cumsum(sorted_weight)
    total = cumulative[-1]
    total_sum = float((sorted_grey * sorted_weight).sum())
    best_level, best_variance = 0.0, -1.0
    running_sum = 0.0
    for index in range(len(sorted_grey) - 1):
        running_sum += sorted_grey[index] * sorted_weight[index]
        weight_low = cumulative[index]
        weight_high = total - weight_low
        if weight_low <= 0 or weight_high <= 0:
            continue
        mean_low = running_sum / weight_low
        mean_high = (total_sum - running_sum) / weight_high
        variance = weight_low * weight_high * (mean_low - mean_high) ** 2
        if variance > best_variance:
            best_variance = variance
            best_level = (sorted_grey[index] + sorted_grey[index + 1]) / 2.0

    ink = np.zeros((height, width), dtype=bool)
    for rgb, mask in layers:
        grey = float(cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2GRAY)[0, 0])
        if grey <= best_level:
            ink |= mask

    traced = np.where(ink, 0, 255).astype(np.uint8)
    # Same outer outline the current pipeline adds, so the comparison is about
    # the classification and not about who remembered to draw an edge.
    from pcbart.lineart import _subject_footprint, outer_outline

    footprint = _subject_footprint(alpha, image)
    outline = outer_outline(footprint, 5)
    if outline.shape[:2] == traced.shape[:2] and outline.size:
        traced = np.where(outline > 0, 0, traced).astype(np.uint8)
    return traced


def reading_b_arc_cleaned(mask, px_per_mm: float = 1.0):
    """A 的掩膜用 spline→弧描一遍再光栅化回来，只清毛刺。

    ⚠️ ``arc_rings_for`` returns **millimetres centred on the origin**, because
    that is what the writers need. Drawing those numbers straight into a pixel
    canvas puts the artwork off the frame -- measured, it left a corner of hair
    in the top-left and blank everywhere else. The mapping back is the inverse
    of ``_ring_to_mm``: ``x_px = w/2 + x_mm * px_per_mm``, ``y_px = h/2 - y_mm
    * px_per_mm``.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    from pcbart.export import arc_rings_for

    height, width = mask.shape[:2]
    polygons = arc_rings_for(mask, px_per_mm)
    if not polygons:
        return mask

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    # Largest area first, holes before their outline -- the nesting the crest
    # needs, and the same order ``compare_vectorizers`` uses.
    def area(ring):
        points = _flatten(ring)
        return abs(
            sum(
                points[i][0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * points[i][1]
                for i in range(len(points))
            )
        ) / 2.0

    for polygon in sorted(polygons, key=lambda rings: -area(rings[0])):
        for index in sorted(range(len(polygon)), key=lambda i: i != 0):
            points = [
                (width / 2.0 + x * px_per_mm, height / 2.0 - y * px_per_mm)
                for x, y in _flatten(polygon[index])
            ]
            if len(points) >= 3:
                draw.polygon(points, fill=0 if index == 0 else 255)
    return np.array(image, dtype=np.uint8)


def _flatten(ring):
    """An ``ArcRing`` through the real path encoder, sampled to points."""
    import math

    from pcbart.geometry import arc_polygon_to_path

    path = arc_polygon_to_path([ring])[0]
    points = [(path[0], path[1])]
    index = 3
    while index < len(path):
        if path[index] == "ARC":
            sweep = path[index + 1]
            end = (path[index + 2], path[index + 3])
            index += 4
            start = points[-1]
            chord = math.hypot(end[0] - start[0], end[1] - start[1])
            radians = math.radians(abs(sweep))
            if chord > 1e-9 and radians > 1e-9:
                radius = chord / (2 * math.sin(radians / 2))
                height = math.sqrt(max(radius**2 - (chord / 2) ** 2, 0))
                ux, uy = (end[0] - start[0]) / chord, (end[1] - start[1]) / chord
                for sign in (1, -1):
                    cx = (start[0] + end[0]) / 2 - sign * height * uy
                    cy = (start[1] + end[1]) / 2 + sign * height * ux
                    a0 = math.atan2(start[1] - cy, start[0] - cx)
                    a1 = math.atan2(end[1] - cy, end[0] - cx)
                    delta = a1 - a0
                    while delta <= -math.pi:
                        delta += 2 * math.pi
                    while delta > math.pi:
                        delta -= 2 * math.pi
                    if (delta > 0) == (sweep > 0):
                        break
                for step in range(1, 32):
                    angle = a0 + math.radians(sweep) * step / 32
                    points.append(
                        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
                    )
            points.append(end)
        else:
            points.append((path[index], path[index + 1]))
            index += 2
    return points


# --- measurement ---------------------------------------------------------


def measure(mask, alpha, reference=None):
    """The numbers behind "does it look right", on a 0 = ink plane."""
    import cv2
    import numpy as np

    ink = mask < 128
    height, width = ink.shape
    numbers = {}

    # 1) 主体外边界有没有墨。The documented defect: pale hair against a pale
    # background left the silhouette torn, measured at 10.3% covered.
    if alpha is not None and alpha.size == ink.size:
        subject = alpha > 0
        if subject.any():
            edge = cv2.morphologyEx(
                subject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            )
            edge = edge > 0
            if edge.any():
                # 放宽 1px：轮廓可以画在边界内侧而仍然连成一条。
                near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                numbers["边界覆盖"] = float(near[edge].mean())

    # 2) 碎片：小于短边 0.3% 的连通块，是 Otsu 边界掉下来的渣。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    floor = max(9.0, (min(height, width) * 0.003) ** 2)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    numbers["碎块数"] = int((areas < floor).sum()) if areas.size else 0
    numbers["块总数"] = int(max(0, count - 1))

    # 3) 细结构：墨点到背景的距离，越小越细。细线保不住会抬高这个中位数。
    if ink.any():
        distance = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        numbers["墨宽中位"] = float(np.median(distance[ink]) * 2.0)

    numbers["墨量"] = float(ink.mean())
    if reference is not None and reference.shape == ink.shape:
        numbers["与A一致"] = float((ink == (reference < 128)).mean())
    return numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_lineart_vtrace")
    parser.add_argument("--image", default="", help="只测这一张")
    parser.add_argument("--short-side", type=int, default=1000,
                        help="工作分辨率（长边），0 = 原尺寸")
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from pcbart.image_prep import imread_unicode
    from pcbart.work_scale import working_size

    subjects = DEFAULT_SUBJECTS
    if args.image:
        name = Path(args.image).stem
        subjects = ((name, args.image),)

    font = None
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 18)
            break

    for name, relative in subjects:
        source = Path(relative)
        if not source.exists():
            print(f"跳过「{name}」：找不到 {source}", file=sys.stderr)
            continue

        raw = imread_unicode(str(source), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"跳过「{name}」：读不出来", file=sys.stderr)
            continue
        alpha = None
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            bgr = raw[:, :, :3]
        elif raw.ndim == 3:
            bgr = raw[:, :, :3]
        else:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        # 工作分辨率：和产品里一样，先缩再处理。
        height, width = bgr.shape[:2]
        target_w, target_h = working_size(width, height)
        if (target_w, target_h) != (width, height):
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(
                    alpha, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
        print(f"\n=== {name}  {target_w}x{target_h} ===")

        results = {}
        timings = {}

        start = time.perf_counter()
        a = reading_a_current(bgr, alpha)
        timings["A 现状"] = time.perf_counter() - start
        results["A 现状"] = a

        start = time.perf_counter()
        b = reading_b_arc_cleaned(a)
        timings["B A+弧清毛刺"] = time.perf_counter() - start
        results["B A+弧清毛刺"] = b

        start = time.perf_counter()
        c = reading_c_colour_regions(bgr, alpha)
        timings["C 彩色区域"] = time.perf_counter() - start
        if c is None:
            print("  C 失败：VTracer 没给出彩色区域")
        else:
            results["C 彩色区域"] = c
            start = time.perf_counter()
            d = reading_b_arc_cleaned(c)
            timings["D C+弧清毛刺"] = time.perf_counter() - start
            results["D C+弧清毛刺"] = d

        reference = results["A 现状"]
        rows = []
        for label, mask in results.items():
            numbers = measure(mask, alpha, reference if label != "A 现状" else None)
            rows.append((label, numbers, timings.get(label, 0.0)))

        header = f"{'方案':<16}{'墨量':>8}{'边界覆盖':>10}{'碎块':>7}{'墨宽中位':>10}{'与A一致':>9}{'秒':>7}"
        print(header)
        for label, numbers, seconds in rows:
            print(
                f"{label:<16}"
                f"{numbers['墨量']:>8.4f}"
                f"{numbers.get('边界覆盖', float('nan')):>10.3f}"
                f"{numbers['碎块数']:>7d}"
                f"{numbers.get('墨宽中位', 0):>10.2f}"
                f"{numbers.get('与A一致', 1.0):>9.3f}"
                f"{seconds:>7.2f}"
            )

        # 视觉图：原图 + 各方案，裁人物上半身放大看细节。
        cells = []
        preview = bgr[:, :, ::-1].astype(np.uint8)
        if alpha is not None:
            preview = np.where(
                (alpha > 0)[:, :, None], preview, 255
            ).astype(np.uint8)
        view = Image.fromarray(preview)
        side = 320
        scale = side / max(view.size)
        view = view.resize(
            (max(1, int(view.width * scale)), max(1, int(view.height * scale))),
            Image.LANCZOS,
        )
        for label, mask in results.items():
            plane = Image.fromarray(mask, "L").convert("RGB")
            plane = plane.resize(
                (max(1, int(plane.width * scale)), max(1, int(plane.height * scale))),
                Image.LANCZOS,
            )
            sheet = Image.new("RGB", (plane.width, plane.height + 26), (24, 27, 32))
            sheet.paste(plane, (0, 26))
            ImageDraw.Draw(sheet).text(
                (6, 4), label, fill=(226, 232, 240), font=font
            )
            cells.append(sheet)
        original = Image.new("RGB", (view.width, view.height + 26), (24, 27, 32))
        original.paste(view, (0, 26))
        ImageDraw.Draw(original).text((6, 4), "原图", fill=(255, 214, 102), font=font)
        cells.insert(0, original)
        total = sum(cell.width for cell in cells) + 8 * (len(cells) - 1)
        board = Image.new("RGB", (total, cells[0].height), (16, 19, 23))
        x = 0
        for cell in cells:
            board.paste(cell, (x, 0))
            x += cell.width + 8
        target = args.outdir / f"{name}-线稿对比.png"
        board.save(target)
        print(f"wrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
