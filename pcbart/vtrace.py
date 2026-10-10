"""Vectorise a mask with VTracer, in the editor's own curve notation.

Why VTracer
-----------
The built-in tracer is ``findContours`` + Douglas-Peucker, so its output is a
chain of **straight** segments. A curve can then only be approximated by many
short chords, which is what makes a traced picture look ragged -- measured on a
256 px icon, the outer ring came back as 1018 points with the steps of the pixel
grid still in them.

VTracer (Rust, MIT, ``pip install vtracer``) fits curves rather than classifying
points, which is the part that could not be tuned by hand: three different
corner heuristics were tried here and each fixed one shape while breaking
another.

What comes out
--------------
VTracer emits SVG, and SVG curves are **cubic Béziers**, which the editor's
``FILL`` records cannot express -- they take straight segments and ``ARC``
sweeps only. So the Béziers are converted to arcs here, by sampling each one and
fitting arcs through the samples. Sampling alone would only move the staircase
from the pixel grid to the sample spacing; fitting arcs is what actually
smooths.

``mode="polygon"`` is supported too: VTracer's own straight-line output, with
far fewer points than the built-in tracer and no fitting step.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .geometry import ArcRing

__all__ = [
    "TracedRing",
    "trace_mask",
    "fit_arcs_through",
    "parse_svg_paths",
    "VTracerError",
]

#: One traced contour: points in millimetres plus a sweep per vertex.
#:
#: An alias of :class:`pcbart.geometry.ArcRing`, deliberately. The tracer and
#: both writers have to agree on the *type*, not merely on the fields: the
#: writers dispatch on it to decide whether a ring's sweeps need negating under
#: a mirror, and two structurally-identical classes would make that check false
#: for traced geometry only -- the failure would be arcs bending the wrong way
#: on flipped placements, which looks like a bad trace rather than a type slip.
TracedRing = ArcRing


class VTracerError(RuntimeError):
    """Raised when VTracer cannot be run or its output cannot be read."""


#: How far a fitted arc may stray from the curve, in source pixels. A quarter
#: pixel is below what the raster can represent, so the fit is limited by the
#: input rather than by this number.
ARC_TOLERANCE_PX = 0.25

#: Samples per Bézier used when fitting. Enough that the chord error of the
#: sampling is itself well under the tolerance.
_BEZIER_SAMPLES = 24

#: Largest sweep a single fitted arc may carry.
#:
#: ⚠️ **A run that wraps more than this is split, not fitted.** ``_sweep_degrees``
#: picks the direction that passes through the run's middle sample, which for a
#: long run means the *long way round*: measured on the card's own artwork, that
#: produced segments of -229° and 288° whose sum for one closed ring came to
#: -464°, a shape that cannot be drawn. Splitting keeps every arc a gentle bend.
_MAX_ARC_SWEEP = 120.0


def _require_vtracer():
    try:
        import vtracer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise VTracerError(
            "缺少 vtracer。安装：python -m pip install vtracer"
        ) from exc
    return vtracer


#: ⚠️ **The ``translate`` transform is geometry, not decoration.** VTracer
#: writes each ``<path>`` in that path's own coordinates and carries its real
#: position in ``transform="translate(x,y)"``. Ignoring it stacks every shape at
#: the image's top-left corner: on the 企鹅物流 logo all 26 paths piled into one
#: black blob that swallowed the mascot, while VTracer's *own* SVG rendered at
#: IoU 0.969 against the mask. The same SVG, drawn from the parsed rings, must
#: match that number.
_TRANSLATE_RE = re.compile(
    r"""transform\s*=\s*["']\s*translate\(\s*([-\d.eE+]+)\s*[, ]\s*([-\d.eE+]+)\s*\)"""
)


def _svg_paths(text: str) -> list[tuple[str, tuple[float, float]]]:
    """Every ``<path>`` as ``(d, (tx, ty))``, in document order."""
    entries: list[tuple[str, tuple[float, float]]] = []
    for tag in re.findall(r"<path\b[^>]*>", text):
        data = re.search(r'\bd\s*=\s*"([^"]+)"', tag)
        if data is None:
            continue
        shift = _TRANSLATE_RE.search(tag)
        offset = (
            (float(shift.group(1)), float(shift.group(2)))
            if shift is not None
            else (0.0, 0.0)
        )
        entries.append((data.group(1), offset))
    return entries


def _shift_ring(ring: TracedRing, offset: tuple[float, float]) -> TracedRing:
    """``ring`` moved by ``offset`` source pixels; the sweeps are unchanged."""
    dx, dy = offset
    return TracedRing(
        points=[(x + dx, y + dy) for x, y in ring.points],
        sweeps=list(ring.sweeps),
    )


def parse_svg_paths(text: str) -> list[list[list[tuple[float, float]]]]:
    """Every ``<path>`` in ``text``, as a list of subpaths of pixel points.

    The Béziers are flattened here: this is only used to *measure* and to draw
    ``polygon`` mode. The curve-preserving path goes through
    :func:`_parse_svg_segments` instead.

    Only the commands VTracer emits are handled -- ``M``, ``L``, ``C`` and
    ``Z``, in absolute form -- and an unknown command raises rather than being
    skipped, because a silently dropped command is a hole in the artwork.
    """
    out: list[list[list[tuple[float, float]]]] = []
    for d, (dx, dy) in _svg_paths(text):
        subpaths: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        start = (0.0, 0.0)
        cursor = (0.0, 0.0)
        for command, body in _iter_commands(d):
            numbers = [float(v) for v in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", body)]
            if command == "M":
                if current:
                    subpaths.append(current)
                cursor = (numbers[0], numbers[1])
                start = cursor
                current = [cursor]
            elif command == "L":
                cursor = (numbers[0], numbers[1])
                current.append(cursor)
            elif command == "C":
                curve = _cubic_points(cursor, numbers)
                current.extend(curve[1:])
                cursor = curve[-1]
            elif command == "Z":
                if current and current[0] != current[-1]:
                    current.append(current[0])
                if current:
                    subpaths.append(current)
                current = []
                cursor = start
            else:  # pragma: no cover - vtracer does not emit these
                raise VTracerError(f"不支持的 SVG 路径命令：{command}")
        if current:
            subpaths.append(current)
        out.append(
            [
                [(x + dx, y + dy) for x, y in sub]
                for sub in subpaths
                if len(sub) >= 3
            ]
        )
    return out


def _iter_commands(d: str):
    """``(command, arguments)`` for each command in an SVG path string."""
    for match in re.finditer(r"([MLCZ])([^MLCZ]*)", d):
        yield match.group(1), match.group(2)


def _cubic_points(
    start: tuple[float, float], arguments: list[float]
) -> list[tuple[float, float]]:
    """Sample a cubic Bézier into ``_BEZIER_SAMPLES`` + 1 points."""
    x0, y0 = start
    x1, y1, x2, y2, x3, y3 = arguments[:6]
    t = np.linspace(0.0, 1.0, _BEZIER_SAMPLES + 1)
    inv = 1.0 - t
    x = (
        inv**3 * x0
        + 3 * inv**2 * t * x1
        + 3 * inv * t**2 * x2
        + t**3 * x3
    )
    y = (
        inv**3 * y0
        + 3 * inv**2 * t * y1
        + 3 * inv * t**2 * y2
        + t**3 * y3
    )
    return [(float(a), float(b)) for a, b in zip(x, y)]


def _parse_svg_segments(d: str):
    """Subpaths as ``[(kind, points)]``, keeping Béziers as Béziers.

    ``kind`` is ``"C"`` for a cubic (with its four control/end points) or
    ``"L"`` for a straight run, so the converter can sample a curve and leave a
    straight edge straight.
    """
    subpaths: list[list[tuple[str, list[tuple[float, float]]]]] = []
    segments: list[tuple[str, list[tuple[float, float]]]] = []
    cursor = (0.0, 0.0)
    for command, body in _iter_commands(d):
        numbers = [float(v) for v in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", body)]
        if command == "M":
            if segments:
                subpaths.append(segments)
            cursor = (numbers[0], numbers[1])
            segments = []
        elif command == "L":
            end = (numbers[0], numbers[1])
            segments.append(("L", [cursor, end]))
            cursor = end
        elif command == "C":
            x1, y1, x2, y2, x3, y3 = numbers[:6]
            segments.append(
                (
                    "C",
                    [cursor, (x1, y1), (x2, y2), (x3, y3)],
                )
            )
            cursor = (x3, y3)
        elif command == "Z":
            if segments:
                subpaths.append(segments)
            segments = []
    if segments:
        subpaths.append(segments)
    return subpaths


def _circumcircle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Centre and radius of the circle through three points, or ``None``."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / denominator
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / denominator
    centre = np.array([ux, uy])
    return centre, float(np.linalg.norm(centre - a))


def fit_arcs_through(
    points: list[tuple[float, float]], tolerance: float
) -> tuple[list[tuple[float, float]], list[float]]:
    """``(kept_points, sweeps)`` for an **open** chain, approximating it with arcs.

    ⚠️ **Open, and per curve segment.** An earlier version fitted whole rings,
    which merged a straight edge and the corner beside it into one arc: measured
    on an 11-point rounded rectangle, the fit collapsed it to five points and a
    total turn of 333° instead of 360°. Fitting each of VTracer's own curve
    segments on its own keeps the corners where VTracer put them, and the shape
    comes out right by construction rather than by a heuristic that has to
    guess where a corner is.

    ``sweeps[i]`` is the sweep arriving at ``points[i]``; ``sweeps[0]`` is
    ``0.0`` because the first point is where the chain starts.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    count = len(pts)
    if count < 3:
        return list(points), [0.0] * count

    kept: list[np.ndarray] = [pts[0]]
    sweeps: list[float] = [0.0]
    index = 0
    while index < count - 1:
        far = max(_longest_fitting_run(pts, index, tolerance), index + 1)
        start_point, end_point = pts[index], pts[far]
        sweep = 0.0
        if far > index + 1:
            middle = pts[_most_distant_from_chord(pts, index, far)]
            circle = _circumcircle(start_point, middle, end_point)
            if circle is not None:
                sweep = _sweep_degrees(circle[0], start_point, middle, end_point)
        kept.append(end_point)
        sweeps.append(sweep)
        index = far

    return [(float(p[0]), float(p[1])) for p in kept], sweeps


def _ring_from_segments(
    segments: list[tuple[str, list[tuple[float, float]]]],
    tolerance: float,
) -> TracedRing | None:
    """One ring, with **one arc per curve segment** VTracer drew.

    ⚠️ **One segment, one arc -- not a greedy fit across several.** A greedy
    fit that runs freely over a ring cannot tell a curve from a corner: measured
    on an 11-point rounded rectangle it collapsed the shape to five points and a
    total turn of 333° instead of 360°, and on the card's own logos it produced
    rings summing to -516° and -594°, which no closed curve can. VTracer already
    decided where the corners are when it split the path into segments, so each
    ``C`` becomes exactly one arc and every segment boundary stays a vertex.

    A cubic that is not close to circular (a long flat curve, or an S-shape)
    keeps its sampled points as straight segments instead: an arc through it
    would bulge away from the artwork, and a few extra vertices are cheaper than
    a wrong shape.
    """
    if not segments:
        return None

    points: list[tuple[float, float]] = [segments[0][1][0]]
    sweeps: list[float] = [0.0]

    for kind, values in segments:
        if kind == "L":
            end = values[-1]
            if _same(end, points[-1]):
                continue
            points.append(end)
            sweeps.append(0.0)
            continue

        sampled = _cubic_points(values[0], _flatten(values))
        arc = _arc_for_cubic(sampled, tolerance)
        if arc is None:
            # ⚠️ **Not an arc means "a straight edge", not "keep every sample".**
            # A rectangle's sides arrive as cubics too, and keeping their
            # sampled points turned a four-corner shape into 96 vertices --
            # worse than the built-in tracer, from the option that is supposed
            # to be smoother. An arc that does not fit is a line here, so the
            # run collapses to its endpoints.
            for point in (sampled[0], sampled[-1]):
                if _same(point, points[-1]):
                    continue
                points.append(point)
                sweeps.append(0.0)
            continue

        # ⚠️ **The extreme point of the curve has to survive as a vertex.** An
        # arc is defined by its two ends and its bulge, and the bulge is not at
        # an end: measured on a 300 px circle, keeping only the seven segment
        # endpoints left the ring 264.7 px tall -- the bottom of the circle fell
        # inside a segment and vanished. So the sampled point furthest from the
        # chord becomes a vertex of its own, splitting the segment into two
        # arcs wherever the curve turns through its own extremum.
        far = _most_distant_from_chord(
            [np.asarray(point, dtype=float) for point in sampled],
            0,
            len(sampled) - 1,
        )
        pieces: list[list[tuple[float, float]]] = []
        if 0 < far < len(sampled) - 1 and _same(sampled[far], sampled[0]) is False:
            pieces = [sampled[: far + 1], sampled[far:]]
        else:
            pieces = [sampled]

        for piece in pieces:
            piece_arc = _arc_for_cubic(piece, tolerance)
            for point in (piece[0], piece[-1]):
                if _same(point, points[-1]):
                    continue
                points.append(point)
                sweeps.append(piece_arc or 0.0)

    if len(points) < 3:
        return None

    if _same(points[-1], points[0]):
        points.pop()
        closing = sweeps.pop()
        sweeps[0] = closing
    return TracedRing(points=points, sweeps=sweeps)


def _arc_for_cubic(
    sampled: list[tuple[float, float]], tolerance: float
) -> float | None:
    """The sweep of the single arc through a sampled cubic, or ``None``.

    ``None`` means "not an arc": the segment is too far from circular for one
    arc to describe it, so the caller keeps its points as straight edges. The
    check is on the **sampled points**, which lie on the curve, against the
    circle through its ends and midpoint.
    """
    if len(sampled) < 3:
        return None
    start = np.asarray(sampled[0], dtype=float)
    middle = np.asarray(sampled[len(sampled) // 2], dtype=float)
    end = np.asarray(sampled[-1], dtype=float)
    circle = _circumcircle(start, middle, end)
    if circle is None:
        return None
    centre, radius = circle
    if radius <= 1e-9:
        return None
    deviation = max(
        abs(float(np.linalg.norm(np.asarray(p, dtype=float) - centre)) - radius)
        for p in sampled
    )
    # ``tolerance`` is in source pixels and so is the circle's radius here: the
    # samples come straight from the SVG.
    if deviation > max(tolerance * 2.0, radius * 0.08):
        return None
    return _sweep_degrees(centre, start, middle, end)


def _same(a, b) -> bool:
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def _flatten(values) -> list[float]:
    """The six numbers of a cubic, from the four points ``parse_*`` returns."""
    out: list[float] = []
    for point in values[1:]:
        out.extend([float(point[0]), float(point[1])])
    return out


def _longest_fitting_run(
    pts: list[np.ndarray], start: int, tolerance: float
) -> int:
    """Largest ``end`` index whose points fit one circle within ``tolerance``."""
    best = start + 1
    upper = min(len(pts) - 1, start + 200)
    for end in range(start + 2, upper + 1):
        if not _fits_arc(pts, start, end, tolerance):
            break
        middle = pts[_most_distant_from_chord(pts, start, end)]
        circle = _circumcircle(pts[start], middle, pts[end])
        if circle is not None:
            sweep = _sweep_degrees(
                circle[0], pts[start], middle, pts[end]
            )
            if abs(sweep) > _MAX_ARC_SWEEP:
                break
        best = end
    return best


def _fits_arc(
    pts: list[np.ndarray], start: int, end: int, tolerance: float
) -> bool:
    middle = _most_distant_from_chord(pts, start, end)
    circle = _circumcircle(pts[start], pts[middle], pts[end])
    if circle is None:
        # Collinear: straight is always acceptable within tolerance.
        return True
    centre, radius = circle
    for position in range(start + 1, end):
        if abs(float(np.linalg.norm(pts[position] - centre)) - radius) > tolerance:
            return False
    return True


def _most_distant_from_chord(
    pts: list[np.ndarray], start: int, end: int
) -> int:
    if end - start < 2:
        return start
    chord = pts[end] - pts[start]
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return (start + end) // 2
    offsets = np.array([pts[i] - pts[start] for i in range(start + 1, end)])
    cross = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    return start + 1 + int(np.argmax(cross / length))


def _sweep_degrees(
    centre: np.ndarray,
    start: np.ndarray,
    middle: np.ndarray,
    end: np.ndarray,
) -> float:
    """Signed sweep from ``start`` to ``end`` that passes through ``middle``."""
    first = math.atan2(start[1] - centre[1], start[0] - centre[0])
    through = math.atan2(middle[1] - centre[1], middle[0] - centre[0])
    last = math.atan2(end[1] - centre[1], end[0] - centre[0])
    to_end = (last - first) % (2 * math.pi)
    to_through = (through - first) % (2 * math.pi)
    sweep = to_end if to_through <= to_end else to_end - 2 * math.pi
    return math.degrees(sweep)


def trace_mask(
    mask: np.ndarray,
    px_per_mm: float,
    *,
    mode: str = "spline",
    tolerance_px: float = ARC_TOLERANCE_PX,
    filter_speckle: int = 2,
    upscale: int = 3,
) -> list[list[TracedRing]]:
    """Vectorise ``mask`` into rings of millimetres, centred on the origin.

    ``mode`` is VTracer's own: ``"spline"`` fits curves (converted to arcs) and
    ``"polygon"`` emits straight edges. The return shape matches
    :func:`pcbart.vectorize.mask_to_complex_polygons`: one entry per shape, the
    first ring its outer contour and the rest holes.

    ``filter_speckle`` is in **source** pixels. It and VTracer's internal
    ``length_threshold`` are multiplied by ``upscale`` before use, because both
    are fixed pixel counts -- see below. The result is divided back on the way
    out, so the geometry is unchanged in millimetres.

    ⚠️ **Why the upscale exists at all.** VTracer decides "is this a curve or a
    staircase?" at the resolution it is handed, and a small source loses that
    judgement: 团队logo 使徒 (139x249) came back missing the snake's eye and the
    wing hatches, IoU 0.890 against its own mask, where the built-in tracer kept
    them at 0.928.

    ⚠️ **And why it is 3x with a smoothed magnification, not 2x nearest.** The
    upscale and the curve fit pull against each other, and only one combination
    works. Measured on 使徒 and 拉特兰:

    ======  ===========  ==========  ==================
    scale   resampling   arc share   verdict
    ======  ===========  ==========  ==================
    1x      --           91-96%      smooth, but drops fine detail
    2x      nearest       0%         pixel staircase: every corner is 90 deg
    2x      bilinear      0%         same staircase, only blurred first
    3x      bilinear     73-79%      curves *and* the fine detail
    4x      bilinear     51-59%      fitting starts to unravel again
    ======  ===========  ==========  ==================

    A nearest-neighbour upscale is the worst of both: it makes each source pixel
    a flat block, so VTracer's corner detector finds a right angle at every
    block edge and emits nothing but straight lines. Smoothed resampling
    reconstructs an edge *inside* the block, which is what the fitter needs.
    Going past 4x wins nothing and costs resolution in the fit.

    Two things about VTracer's output had to be handled, both measured rather
    than assumed:

    * **It traces the black area.** A 60x40 black rectangle on a 100 px canvas
      came back as exactly that rectangle; the same picture inverted came back
      as the canvas with the rectangle as a hole. The pipeline's ``0`` means
      ink, so the ink is painted black for VTracer.
    * **It moves the path's first point to the origin.** The rectangle's output
      began at ``(0,0)`` where the source began at ``(20,30)``; a circle centred
      on the canvas began at ``(0,0)`` where its top point was ``(200,50)``. No
      single offset explains both, so the output is aligned by **bounding box**
      instead of by a guessed translation -- which is self-calibrating and does
      not depend on VTracer's internal convention staying put.
    """
    import cv2
    import tempfile

    from .image_prep import imwrite_unicode

    if px_per_mm <= 0:
        raise ValueError("px_per_mm 必须为正数")
    vtracer = _require_vtracer()

    plane = np.asarray(mask)
    if plane.ndim == 3:
        plane = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)
    # ``0`` is ink, matching the rest of the pipeline (``ink_is_dark``).
    ink = plane <= 128
    if not ink.any():
        return []
    if upscale < 1:
        raise ValueError("upscale 必须 >= 1")
    if upscale > 1:
        # ⚠️ **Resample the greyscale, not the boolean.** Interpolating the
        # 0/255 plane puts the edge back *between* the magnified pixels; a
        # nearest-neighbour copy instead hands VTracer a field of hard right
        # angles, and it answers with a staircase (0% arcs -- see docstring).
        grey = cv2.resize(
            np.where(ink, 0, 255).astype(np.uint8),
            None,
            fx=float(upscale),
            fy=float(upscale),
            interpolation=cv2.INTER_LINEAR,
        )
        ink = grey <= 128
    height, width = ink.shape[:2]
    working_scale = px_per_mm * upscale

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "mask.png"
        # ⚠️ **Inverted: VTracer traces the black area.** See the docstring.
        imwrite_unicode(source, np.where(ink, 0, 255).astype(np.uint8))
        target = Path(folder) / "mask.svg"
        vtracer.convert_image_to_svg_py(
            str(source),
            str(target),
            colormode="binary",
            hierarchical="cutout",
            mode=mode,
            # ⚠️ Both thresholds are **source** pixels and are scaled with the
            # magnification. Left at their 1x values they describe a much
            # smaller feature on the magnified raster, which is why the
            # unsmoothed 2x run kept every pixel corner.
            filter_speckle=filter_speckle * upscale,
            corner_threshold=60,
            length_threshold=2.0 * upscale,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        svg = target.read_text(encoding="utf-8")

    # ⚠️ **No canvas-ring handling is needed, and adding it broke things.**
    # With the polarity right -- ink painted black -- VTracer returns the ink
    # and nothing else: measured on a rectangle, a circle and the card's own
    # artwork, every output began with the artwork's own contour. An earlier
    # version dropped the first subpath whenever the ink avoided the image
    # border, which threw the artwork away and left a fragment in the corner.
    # The "canvas ring" seen before that was the wrong polarity's output.
    # The SVG's own structure is preserved: each ``<path>`` is one piece of
    # artwork, and each subpath inside it is one ring (the first the outline,
    # the rest holes). Flattening that here is what keeps a glyph's counters
    # attached to the glyph.
    if mode == "spline":
        per_path: list[list[TracedRing]] = []
        for d, offset in _svg_paths(svg):
            rings = [
                _shift_ring(ring, offset)
                for ring in (
                    _ring_from_segments(segments, tolerance_px)
                    for segments in _parse_svg_segments(d)
                )
                if ring is not None and len(ring.points) >= 3
            ]
            if rings:
                per_path.append(rings)
    else:
        per_path = [
            [
                TracedRing(points=list(sub), sweeps=[0.0] * len(sub))
                for sub in subpaths
                if len(sub) >= 3
            ]
            for subpaths in parse_svg_paths(svg)
        ]
        per_path = [rings for rings in per_path if rings]

    if not per_path:
        return []

    # ⚠️ **Align by the box CENTRE, not a corner.** VTracer moves the path's
    # first point to the origin, and that point is wherever it chose to start --
    # on a circle it is neither the top nor the left edge. Aligning corner to
    # corner therefore placed the artwork a bit off in both axes: measured on a
    # 300 px circle, the traced points sat 28.7 to 44.8 mm from the centre where
    # every one of them should have been 37.5. The centres are the one pair of
    # numbers a translation cannot get wrong.
    rows = np.nonzero(ink.any(axis=1))[0]
    columns = np.nonzero(ink.any(axis=0))[0]
    traced_x = [p[0] for rings in per_path for ring in rings for p in ring.points]
    traced_y = [p[1] for rings in per_path for ring in rings for p in ring.points]
    ink_centre = (
        (float(columns[0]) + float(columns[-1])) / 2.0,
        (float(rows[0]) + float(rows[-1])) / 2.0,
    )
    traced_centre = (
        (min(traced_x) + max(traced_x)) / 2.0,
        (min(traced_y) + max(traced_y)) / 2.0,
    )
    offset = (
        ink_centre[0] - traced_centre[0],
        ink_centre[1] - traced_centre[1],
    )

    # The artwork's own size, used to judge whether an arc's radius is absurd.
    # ⚠️ This has to be measured **here**, on the design-frame geometry: the
    # cutoff is a ratio to it, and a ratio to the working raster would move with
    # the magnification instead of with the artwork.
    span_mm = (
        (float(columns[-1] - columns[0]) ** 2 + float(rows[-1] - rows[0]) ** 2)
        ** 0.5
    ) / px_per_mm
    artwork_mm = max(span_mm, 1e-6)

    shapes: list[list[TracedRing]] = []
    for rings in per_path:
        converted = [
            # ``working_scale``, not ``px_per_mm``: the rings are in the
            # upscaled raster's pixels and have to come back down with it.
            _ring_to_mm(ring, width, height, working_scale, mode, offset)
            for ring in rings
        ]
        if mode == "spline":
            converted = [drop_shallow_arcs(ring, artwork_mm) for ring in converted]
        converted = [ring for ring in converted if len(ring.points) >= 3]
        if converted:
            shapes.append(converted)
    return shapes


def _ring_to_mm(
    ring: TracedRing,
    width: int,
    height: int,
    px_per_mm: float,
    mode: str,
    offset: tuple[float, float] = (0.0, 0.0),
) -> TracedRing:
    """One traced ring, converted into the design frame.

    SVG is Y-down with its origin at the top-left; the design frame is Y-up with
    its origin at the centre. ``offset`` is where VTracer put the origin -- the
    ink's bounding box in the source mask -- so adding it puts the shape back
    where it belongs before the frame change.
    """
    points = ring.points
    sweeps = ring.sweeps
    offset_x, offset_y = offset
    converted = [
        (
            (x + offset_x - width / 2.0) / px_per_mm,
            (height / 2.0 - (y + offset_y)) / px_per_mm,
        )
        for x, y in points
    ]
    if mode == "spline":
        # The fit ran in SVG space, where Y is flipped relative to the design
        # frame, so every sweep changes sign with it.
        sweeps = [-value for value in sweeps]
    return TracedRing(points=converted, sweeps=sweeps)


#: How many times the artwork's own size an arc's radius may be before it is
#: written as a straight edge.
#:
#: ⚠️ **The test is a ratio, not a length.** A near-straight fitted cubic is
#: circular with an enormous radius, and those arcs are what break a downstream
#: reader: measured on 企鹅物流 the fitted radii reached **242 metres**, and
#: gerbonara's bounding box for the whole card came out 194 x 194 mm against a
#: real 50 x 100.
#:
#: ⚠️ **An absolute bulge cannot be the test, and measuring showed why.** The
#: sagitta of a segment shrinks as the trace is split more finely: on the same
#: 企鹅物流 crest the median sagitta was 0.000048 mm at 1x and 0.000032 mm at
#: 3x, for the *same* underlying curves. A 0.01 mm cutoff therefore kept 6% of
#: the arcs at 1x but 0% at 3x -- it was measuring the segmentation, not the
#: shape. A ratio of radius to artwork size is invariant to all of that: a
#: 0.5 mm-wide glint on a 15 mm crest has a radius of about its own size, while
#: a straight edge's radius runs to thousands of times it.
#:
#: 20x the artwork's diagonal is generous enough to keep every curve that
#: follows the drawing (the median fitted radius on these crests is about 1x
#: the mark) while cutting the degenerate fits that made the box meaningless.
MAX_ARC_RADIUS_RATIO = 20.0


def drop_shallow_arcs(
    ring: TracedRing, reference_mm: float = 0.0
) -> TracedRing:
    """Turn arcs whose radius dwarfs the artwork into straight edges.

    ``reference_mm`` is the artwork's own size -- its bounding-box diagonal
    works well. A ring whose radius exceeds ``MAX_ARC_RADIUS_RATIO`` times that
    is indistinguishable from a line at this scale, and leaving it in puts a
    nonsense centre offset into the file.

    ⚠️ **Do not go back to an absolute millimetre threshold.** See
    :data:`MAX_ARC_RADIUS_RATIO`: the bulge of a fitted segment is a function of
    how finely VTracer split the curve, so a fixed cutoff silently means
    something different at every magnification.
    """
    points = ring.points
    count = len(points)
    sweeps = list(ring.sweeps)
    limit = (
        max(1.0, float(reference_mm)) * MAX_ARC_RADIUS_RATIO
        if reference_mm > 0
        else float("inf")
    )
    for index in range(count):
        sweep = float(sweeps[index])
        if abs(sweep) < 1e-9:
            continue
        start = points[index - 1]
        end = points[index]
        chord = math.hypot(end[0] - start[0], end[1] - start[1])
        if chord <= 1e-12:
            sweeps[index] = 0.0
            continue
        half = math.sin(math.radians(abs(sweep)) / 2.0)
        if half <= 1e-12:
            sweeps[index] = 0.0
            continue
        radius = abs(chord / (2.0 * half))
        if radius > limit:
            sweeps[index] = 0.0
    return TracedRing(points=list(points), sweeps=sweeps)
