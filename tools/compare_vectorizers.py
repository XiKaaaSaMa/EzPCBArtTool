"""Compare the built-in tracer against the two VTracer-based options.

The built-in path is ``findContours`` + Douglas-Peucker, so its output is a
chain of straight segments. Measured on a 256 px icon, its outer ring came back
as 1018 points with the pixel grid's steps still visible -- which is the ragged
edge the user reported.

Two replacements are compared, because they trade off differently:

* **VTracer spline** -- fits curves, then converts the Béziers to the arcs the
  editor can actually draw. Smoothest, at the cost of a lossy conversion.
* **VTracer polygon** -- straight edges only, but far fewer and better placed
  than the built-in tracer's.

Each panel is drawn from the geometry that option would *export*, including the
arcs, so the sheet shows what a board would get rather than what a library says
it produced.

Usage::

    python tools/compare_vectorizers.py
    python tools/compare_vectorizers.py --only 阵营logo
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pcbart.paths import asset_root  # noqa: E402

#: Both resolved under ``PCBART_ASSET_ROOT`` -- the library is several gigabytes
#: and lives outside the repository, so it cannot be a literal path.
ASSETS = asset_root() / "明日方舟-艺术板素材"

ASSET_ROOT = asset_root() / "明日方舟"

#: ``(label, relative path)``. A spread of the artwork this tool handles:
#: opaque plates with knock-outs, white line art on transparency, and the small
#: fine-detail pieces where a fixed pixel threshold hurts most.
SUBJECTS: tuple[tuple[str, str], ...] = (
    ("职业图标-医疗", "15-职业大图/黑/图标 职业 医疗 大图.png"),
    ("职业图标-先锋", "15-职业大图/黑/图标 职业 先锋 大图.png"),
    ("职业图标-狙击", "15-职业大图/黑/图标 职业 狙击 大图.png"),
    ("阵营logo-企鹅物流", "18-阵营logo/Logo 企鹅物流.png"),
    ("阵营logo-莱茵生命", "18-阵营logo/Logo 莱茵生命.png"),
    ("阵营logo-乌萨斯", "18-阵营logo/Logo 乌萨斯.png"),
    ("阵营logo-拉特兰", "18-阵营logo/Logo 拉特兰.png"),
    ("团队logo-使徒", "16-团队logo/团队logo 使徒.png"),
    ("皮肤logo-韵系列", "17-皮肤logo（大）/Skin logo 0011-韵系列.png"),
)

#: Subjects that arrive as a **picture**, not as a mask: they go through the
#: tool's own line-art step first, so the comparison shows what an imported
#: illustration would actually be traced from. ``(label, relative path)``
#: under :data:`ASSET_ROOT`.
ILLUSTRATIONS: tuple[tuple[str, str], ...] = (
    ("立绘-精英0-12F", "06-立绘-精英0/立绘 12F 1.png"),
    ("立绘-精英0-Castle-3", "06-立绘-精英0/立绘 Castle-3 1.png"),
    ("立绘-精英2-Miss.Christine", "08-立绘-精英2/立绘 Miss.Christine 2.png"),
    ("半身像-精英0-12F", "02-半身像-精英0/半身像 12F 1.png"),
)

PANEL_PIXELS = 300
_PX_PER_MM = 4.0


def ink_plane(source: Path):
    """The picture as a **0 = ink, 255 = bare** plane, whichever way it arrives.

    ⚠️ **The card's artwork does not agree on which way round it is.** The
    profession plates are opaque black plates with the glyph knocked out; the
    faction logos are **white line art on transparency**. Treating dark as ink
    for both showed a solid square for one and a blank sheet for the other -- so
    the ink is chosen from the picture: mostly-light opaque pixels mean the
    alpha channel is the drawing.
    """
    import numpy as np
    import numpy as np
    from PIL import Image

    image = Image.open(source).convert("RGBA")
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    opaque = alpha > 128

    if not opaque.any():
        return np.full(image.size[::-1], 255, dtype=np.uint8)
    if pixels[:, :, :3][opaque].mean() > 200:
        return np.where(opaque, 0, 255).astype(np.uint8)

    flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flat.alpha_composite(image)
    grey = np.array(flat.convert("L"))
    return np.where(grey < 128, 0, 255).astype(np.uint8)


def draw_rings(shapes, size: tuple[int, int], sweeps_honoured: bool):
    """Raster of ``[shape][ring]`` geometry, in millimetres.

    ⚠️ **Biggest shape first, and each shape's holes before its outline.** The
    artwork nests -- the 阵营 logo is a ring with the mascot and lettering
    inside it -- and drawing in the traced order let an inner shape be covered
    by the next one's fill, so the mascot disappeared under a black blob.
    Sorting by area is what the nesting actually requires: a container is
    always bigger than what it contains, and the outline of a container is
    drawn after its own hole so the hole stays open for the contents.
    """
    from PIL import Image, ImageDraw

    width, height = size
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    centre_x, centre_y = width / 2.0, height / 2.0

    def area(ring) -> float:
        points = _ring_points(ring, (centre_x, centre_y), sweeps_honoured)
        total = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    # Largest first: a container is painted before whatever sits inside it.
    ordered = sorted(shapes, key=lambda rings: -area(rings[0]))
    for rings in ordered:
        for index in sorted(range(len(rings)), key=lambda i: i != 0):
            ring = rings[index]
            points = _ring_points(ring, (centre_x, centre_y), sweeps_honoured)
            if len(points) >= 3:
                draw.polygon(points, fill=0 if index == 0 else 255)
    return image


def _ring_points(ring, centre: tuple[float, float], sweeps_honoured: bool):
    """One ring's outline in image pixels, flattening arcs when asked.

    ⚠️ **The Y flip here reverses every sweep too.** The rings are in the design
    frame (Y up); this raster is Y-down, so ``py = centre_y - y * scale`` is a
    reflection -- and a reflection reverses an arc's direction. Drawing the mm
    sweep unchanged bent every arc the wrong way: a circle came out as seven
    concave scallops, and the exported numbers were correct the whole time.
    """
    centre_x, centre_y = centre
    points: list[tuple[float, float]] = []
    for position, (x, y) in enumerate(ring.points):
        px = centre_x + x * _PX_PER_MM
        py = centre_y - y * _PX_PER_MM
        sweep = ring.sweeps[position] if sweeps_honoured else 0.0
        if points and abs(sweep) > 1e-6:
            points.extend(_arc_points(points[-1], (px, py), -sweep))
        points.append((px, py))
    return points


def _arc_points(start, end, sweep_deg: float, steps: int = 14):
    """Points along the arc from ``start`` to ``end`` sweeping ``sweep_deg``.

    The centre is the point on the chord's perpendicular bisector at distance
    ``sqrt(r^2 - (c/2)^2)`` from the chord, where ``r = c / (2 sin(sweep/2))``.
    There are two candidates; the right one is whichever makes the travel from
    start to end turn the way ``sweep_deg`` says.

    ⚠️ **Pick the candidate by matching the sign, not by matching the value.**
    An earlier version also demanded ``abs(abs(delta) - sweep) < 1e-6``, which
    floating-point arithmetic on a real shape almost never satisfies; the loop
    then fell through to the *last* candidate -- the wrong side -- and every arc
    bent outwards. Judging only the direction is exact and cannot miss.
    """
    x1, y1 = start
    x2, y2 = end
    chord = math.hypot(x2 - x1, y2 - y1)
    sweep = math.radians(abs(sweep_deg))
    if chord < 1e-9 or sweep < 1e-9:
        return []

    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius = chord / (2.0 * math.sin(sweep / 2.0))
    height = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))
    ux, uy = (x2 - x1) / chord, (y2 - y1) / chord

    best = None
    for sign in (1.0, -1.0):
        cx = mx - sign * height * uy
        cy = my + sign * height * ux
        a0 = math.atan2(y1 - cy, x1 - cx)
        a1 = math.atan2(y2 - cy, x2 - cx)
        delta = a1 - a0
        while delta <= -math.pi:
            delta += 2 * math.pi
        while delta > math.pi:
            delta -= 2 * math.pi
        if (delta > 0) == (sweep_deg > 0):
            best = (cx, cy, a0)
            break
    if best is None:
        return []

    cx, cy, a0 = best
    out = []
    for step in range(1, steps):
        angle = a0 + math.radians(sweep_deg) * step / steps
        out.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return out


class _PlainRing:
    """A ring with no curves, so both label sites can use one draw path."""

    __slots__ = ("points", "sweeps")

    def __init__(self, points):
        self.points = points
        self.sweeps = [0.0] * len(points)


def ours_rings(plane):
    """The built-in tracer's polygons, in the same ring shape as VTracer's."""
    import numpy as np

    from pcbart.vectorize import VectorizeParams, mask_to_complex_polygons

    polygons = mask_to_complex_polygons(
        np.where(plane < 128, 0, 255).astype(np.uint8),
        px_per_mm=_PX_PER_MM,
        params=VectorizeParams(),
    )
    return [[_PlainRing(ring) for ring in polygon] for polygon in polygons]


#: Magnification handed to VTracer, which is what keeps a small source from
#: losing its fine detail (its speckle and length thresholds are fixed pixel
#: counts, so a small source pays for them twice).
#:
#: ⚠️ **3x with the mask resampled smoothly, not 2x nearest.** The upscale and
#: VTracer's curve fit pull against each other and only some combinations work.
#: Measured, share of the output that is arcs:
#:
#: ======  ==========  =========
#: scale   resampling  arc share
#: ======  ==========  =========
#: 1x      --          91-96%
#: 2x      nearest      0%
#: 2x      bilinear     0%
#: 3x      bilinear    73-79%
#: 4x      bilinear    51-59%
#: ======  ==========  =========
#:
#: A nearest-neighbour magnification is the worst of both: every source pixel
#: becomes a flat block, so every block edge is a right angle and the output is
#: nothing but straight lines. ``pcbart.vtrace.trace_mask`` resamples the grey
#: plane for the same reason and scales its thresholds with the magnification.
UPSCALE = 3


def vtracer_rings(plane, mode: str):
    from pcbart.vtrace import trace_mask

    return trace_mask(plane, px_per_mm=_PX_PER_MM, mode=mode, upscale=UPSCALE)


def label(image, text: str):
    """A caption above a panel, so the sheet reads without the script."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 20)
            break
    height = 34
    sheet = Image.new("RGB", (image.width, image.height + height), (24, 27, 32))
    sheet.paste(image, (0, height))
    ImageDraw.Draw(sheet).text((8, 6), text, fill=(226, 232, 240), font=font)
    return sheet


def comparison_sheet(plane, name: str, panel: int, source_note: str = "原图"):
    """The four-panel sheet for one 0/255 ink plane, or ``None`` if empty."""
    import numpy as np
    from PIL import Image

    ink = plane < 128
    if not ink.any():
        return None
    size = (plane.shape[1], plane.shape[0])

    built_in = ours_rings(plane)
    spline = vtracer_rings(plane, "spline")
    polygon = vtracer_rings(plane, "polygon")

    counts = {
        "本工具": (sum(len(r.points) for s in built_in for r in s), 0),
        "spline": (
            sum(len(r.points) for s in spline for r in s),
            sum(1 for s in spline for r in s for v in r.sweeps if abs(v) > 1e-6),
        ),
        "polygon": (sum(len(r.points) for s in polygon for r in s), 0),
    }

    source_view = Image.fromarray(
        np.where(ink, 0, 255).astype(np.uint8), "L"
    ).convert("RGB")

    def view(shapes, honour: bool):
        return draw_rings(shapes, size, honour).convert("RGB")

    panels = [
        label(source_view.resize((panel, panel)), source_note),
        label(
            view(built_in, False).resize((panel, panel)),
            f"本工具（全直线）{counts['本工具'][0]} 点",
        ),
        label(
            view(spline, True).resize((panel, panel)),
            f"VTracer spline→弧（{UPSCALE}x）　"
            f"{counts['spline'][0]} 点 / {counts['spline'][1]} 弧",
        ),
        label(
            view(polygon, False).resize((panel, panel)),
            f"VTracer polygon（{UPSCALE}x）　{counts['polygon'][0]} 点",
        ),
    ]

    total_w = sum(item.width for item in panels) + 8 * (len(panels) - 1)
    sheet = Image.new("RGB", (total_w, panels[0].height), (16, 19, 23))
    x = 0
    for item in panels:
        sheet.paste(item, (x, 0))
        x += item.width + 8
    return sheet, name, counts


def line_art_plane(source: Path):
    """A real illustration's ink plane, through the tool's own line-art step.

    ⚠️ **This is the case that matters.** The logos are already flat two-colour
    art; an imported illustration reaches the tracer only after ``line_art()``
    has decided what is ink, and that mask is what gets vectorised. Comparing
    the tracers on the raw picture would measure a path the tool never runs.
    """
    import cv2
    import numpy as np
    from PIL import Image

    from pcbart.lineart import line_art
    from pcbart.work_scale import working_size

    image = np.array(Image.open(source).convert("RGBA"))
    # ``line_art`` runs on a work-scale copy of anything large -- mirror that
    # here so the sheet shows the mask at the resolution the tracer actually
    # receives, not at the source resolution the export never sees.
    width, height = working_size(image.shape[1], image.shape[0])
    if (width, height) != (image.shape[1], image.shape[0]):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    art = line_art(image)
    ink = np.asarray(art)
    if ink.ndim == 3:
        ink = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
    return np.where(ink > 128, 255, 0).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=ROOT / "out" / "_vector_compare")
    parser.add_argument("--only", default="", help="只做名称包含这段的一个")
    parser.add_argument("--panel", type=int, default=PANEL_PIXELS)
    parser.add_argument(
        "--with-illustrations",
        action="store_true",
        help="额外跑「立绘 → 线稿 → 矢量」的端到端对比",
    )
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(args.outdir / "drafts")

    written: list[Path] = []

    def emit(sheet, name, counts, note: str) -> None:
        if sheet is None:
            print(f"跳过「{name}」：掩膜是空的", file=sys.stderr)
            return
        target = args.outdir / f"{name}-对比.png"
        sheet.save(target)
        written.append(target)
        print(
            f"wrote {target}　{note} 本工具 {counts['本工具'][0]} 点 / "
            f"spline {counts['spline'][0]} 点 {counts['spline'][1]} 弧 / "
            f"polygon {counts['polygon'][0]} 点"
        )

    for name, relative in SUBJECTS:
        if args.only and args.only not in name:
            continue
        source = ASSETS / relative
        if not source.exists():
            print(f"跳过「{name}」：找不到 {source}", file=sys.stderr)
            continue
        result = comparison_sheet(ink_plane(source), name, args.panel)
        if result is not None:
            emit(*result, note="")

    if args.with_illustrations:
        for name, relative in ILLUSTRATIONS:
            if args.only and args.only not in name:
                continue
            source = ASSET_ROOT / relative
            if not source.exists():
                print(f"跳过「{name}」：找不到 {source}", file=sys.stderr)
                continue
            result = comparison_sheet(
                line_art_plane(source), name, args.panel, source_note="线稿（成品输入）"
            )
            if result is not None:
                emit(*result, note="[线稿路径]")

    if not written:
        print("什么都没生成。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
