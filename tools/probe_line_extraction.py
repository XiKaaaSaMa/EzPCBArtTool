"""Probe: can the lines be read on their own, before any product change?

This is a *measurement*, not a product refactor. The line-art pipeline the tool
ships asks a per-pixel question -- "is this pixel ink?" -- and answers it with
tone. That question cannot separate a drawn line from the edge of a coloured
region, because in a raster they are the same kind of signal. This probe asks a
different question: **how strong is a boundary here?**

The reading:

1. **Zero-mean difference of Gaussians, on Lab.** ``G_σ - G_1.6σ`` is exactly
   zero across any flat field, whatever its colour or brightness, and peaks on
   every step. Taking the magnitude makes a boundary a boundary whichever side
   is darker, and running it over the Lab channels lets a pure hue change -- no
   luminance step at all -- register as what it is. A full black-to-white step
   reads 1.0, so the result is a contrast and can be thresholded meaningfully.
2. **Strength-seeded hysteresis.** The strongest ``density`` share of the
   subject is the seed; every weaker pixel *connected to a seed* joins it and
   everything else is dropped. This is what joins a broken stroke without
   letting isolated texture in -- the two failures the shipped density
   quantile trades between.
3. **The subject's outer contour, added explicitly.** A pale figure against a
   pale page has no tonal boundary to find, so it is drawn from the subject
   mask instead of from the picture. This is also the shape the board outline
   is cut from.
4. **Fill weight, for the second half of the question.** The Otsu dark mask
   grows inward from its own boundary by ``weight`` times its component's
   inscribed radius. At 0 the output is pure lines; at 1 it is the shipped
   block reading; in between it slides continuously. No second algorithm.

Nothing here is wired into the panel. It writes, for every sample, the original
next to each variant, all in one folder, plus a labelled contact sheet and a
report with timings and ink coverage.

Usage::

    python tools/probe_line_extraction.py
    python tools/probe_line_extraction.py --operators 能天使 临光 --max-side 1400
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from pcbart.lineart import (  # noqa: E402 - after sys.path setup
    DOG_UNIT_STEP,
    OUTLINE_DOG_RATIO,
    desaturate,
    outer_outline,
    subject_masks,
)
from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402
from pcbart.paths import asset_root  # noqa: E402
from pcbart.work_scale import work_scale  # noqa: E402

#: Where the complete library lives, resolved through ``PCBART_ASSET_ROOT`` so
#: the probe also runs on a machine that keeps it on another drive. The 精0 /
#: 精2 directories are the ones the user asked for: they are the same operator
#: at two illustration ages, so the reading can be judged on the same character
#: across a large scale change.
LIBRARY = asset_root() / "明日方舟"
ELITE = {
    0: ("06-立绘-精英0", "1"),
    2: ("08-立绘-精英2", "2"),
}
DEFAULT_OPERATORS = ("能天使", "临光", "乌尔比安", "Miss.Christine")

#: Contrast floor from :mod:`pcbart.lineart`: below this nothing is a boundary
#: however little ink the density has produced, which is what stops a flat
#: field from returning itself as "the strongest 6%".
MIN_CONTRAST = 0.02


@dataclass(frozen=True)
class Variant:
    """One parameter combination to render and compare."""

    key: str
    label: str
    sigma_ratio: float
    """Detector blur radius as a fraction of the picture's short side."""

    density: float
    """Share of the subject seeded as strong boundary."""

    color_weight: float
    """How much a Lab hue change counts against a luminance change of the
    same size. 0 reads luminance alone."""

    fill_weight: float = 0.0
    """How far the Otsu dark mask grows back in, 0..1. See the module doc."""

    vector: bool = False
    """Re-describe the mask with VTracer's spline fit, then rasterise it back."""

    current: bool = False
    """Render the shipped reading instead of the probe's, as the other endpoint."""


#: The variants the user asked to see. A0/A and B differ only in whether hue
#: counts, so the pair isolates that one decision; C shows what a coarser
#: detector and a larger budget do; D is the vector pass on B; E/F are the
#: slider moving from pure lines towards the shipped block reading; T is the
#: shipped reading itself, so both ends of the slider are on the page.
VARIANTS = (
    Variant("A0-fine-lum", "A0 fine lum-only", 0.0012, 0.03, 0.0),
    Variant("A-mid-lum", "A mid lum-only", 0.0016, 0.06, 0.0),
    Variant("B-mid-lab", "B mid lum+chroma", 0.0016, 0.06, 0.7),
    Variant("C-bold-lab", "C bold lum+chroma", 0.0024, 0.09, 0.7),
    Variant("D-mid-lab-vector", "D B + vector arcs", 0.0016, 0.06, 0.7, vector=True),
    Variant("E-fill-40", "E B + fill 40%", 0.0016, 0.06, 0.7, fill_weight=0.40),
    Variant("F-fill-75", "F B + fill 75%", 0.0016, 0.06, 0.7, fill_weight=0.75),
    Variant("T-current-fill", "T shipped: fill only", 0.0, 0.0, 0.0, current=True),
)

#: The weights the slider strip steps through, from pure lines to solid.
SLIDER_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: The settings the slider strip runs at -- variant B.
SLIDER_BASE = Variant("slider", "lines", 0.0016, 0.06, 0.7)


# --- the reading ---------------------------------------------------------


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with an odd kernel derived from ``sigma``.

    Same rule as :func:`pcbart.lineart._gaussian`: OpenCV measures an even
    kernel from a corner rather than its centre, which shifts the response.
    """
    size = max(3, int(2 * round(3.0 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    return cv2.GaussianBlur(image.astype(np.float32), (size, size), float(sigma))


def boundary_response(
    bgr: np.ndarray, sigma: float, color_weight: float
) -> np.ndarray:
    """Local contrast in 0..1, luminance and hue together.

    A magnitude, so a boundary counts whichever side is darker; zero-mean, so
    a flat field reads exactly 0.0 whatever its colour. Normalised so a full
    black-to-white luminance step reads 1.0, which makes the floor and the
    density quantile mean the same thing they mean in
    :func:`pcbart.lineart.outline_response`.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    channels = [1.0, color_weight, color_weight]
    total = np.zeros(lab.shape[:2], np.float32)
    for index, weight in enumerate(channels):
        if weight <= 0:
            continue
        channel = lab[:, :, index]
        if index > 0:
            channel = channel - 128.0  # OpenCV's 8-bit Lab is offset-neutral
        diff = _gaussian(channel, sigma) - _gaussian(
            channel, sigma * OUTLINE_DOG_RATIO
        )
        total += (diff * weight) ** 2
    return np.sqrt(total) / (255.0 * DOG_UNIT_STEP)


def hysteresis(
    response: np.ndarray,
    subject: np.ndarray,
    density: float,
    weak_ratio: float,
    floor: float,
) -> np.ndarray:
    """Strong boundaries seed, weak ones join only if connected to a seed.

    The density quantile alone answers "how strong is strong", but a broken
    stroke spends its budget on the strong halves and drops the faint
    continuations. Hysteresis keeps those: a weak pixel is inked when a path
    of weak pixels reaches a strong one. The two failures the shipped reading
    trades between -- texture in, strokes out -- are exactly the two ends this
    fixes: isolated texture has no seed nearby, and a faint stroke continuation
    is connected to one.
    """
    values = response[subject]
    if values.size == 0:
        return np.zeros(response.shape, bool)
    strong_level = max(floor, float(np.quantile(values, 1.0 - density)))
    strong = response >= strong_level
    if not strong.any():
        return np.zeros(response.shape, bool)
    weak_level = max(floor, strong_level / max(1.01, float(weak_ratio)))
    weak = response >= weak_level
    weak &= subject

    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if count <= 1:
        return strong & subject
    marked = np.zeros(count, bool)
    marked[np.unique(labels[strong & weak])] = True
    marked[0] = False
    return marked[labels]


def _otsu_level(grey8: np.ndarray, subject: np.ndarray) -> float:
    """Otsu's split, measured over the subject only.

    Same reasoning as :func:`pcbart.lineart._tone_split_level`: the empty
    background outside a cut-out is a huge spike at white and it drags the
    level up with it.
    """
    sample = grey8[subject] if subject.any() else grey8
    if sample.size == 0:
        return 128.0
    column = np.ascontiguousarray(sample.reshape(-1, 1))
    level, _ = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level)


def grow_fill(
    subject: np.ndarray, grey8: np.ndarray, weight: float
) -> np.ndarray:
    """The dark mask, grown inward from its own boundary by ``weight``.

    Each connected dark component has an inscribed radius: the largest
    distance from one of its pixels to the nearest non-dark pixel. Thresholding
    that distance at ``weight`` times the radius returns a thinner and thinner
    rim as the weight falls, and the whole component at 1.0. Because the rim
    hugs the component's own outline, which is where the drawn lines already
    sit, the fill appears to grow out of the lines rather than switching on.
    """
    if weight <= 0:
        return np.zeros(subject.shape, bool)
    level = _otsu_level(grey8, subject)
    dark = (grey8 <= level) & subject
    if not dark.any():
        return np.zeros(subject.shape, bool)

    distance = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    count, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    peak = np.zeros(count, np.float32)
    np.maximum.at(peak, labels[dark], distance[dark])
    return dark & (distance <= float(np.clip(weight, 0.0, 1.0)) * peak[labels])


def drop_specks(ink: np.ndarray, min_area: int) -> np.ndarray:
    """Remove ink components below ``min_area`` pixels."""
    if min_area <= 0:
        return ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return ink
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels]


def clean_alpha_streaks(
    alpha: np.ndarray, min_thickness: int, low: int = 7
) -> tuple[np.ndarray, float, float]:
    """Drop the hair-thin streaks some cut-outs carry in their alpha.

    ⚠️ **A measured source defect, not a guess.** 乌尔比安 精0's alpha has 41
    rows that are opaque across more than three quarters of the frame's width,
    three to five pixels thick, running from the figure out to the image edge.
    They are not part of the artwork, but they are part of the *mask*, so
    ``subject_mask`` merges them in and the outer contour then draws a
    rectangle around the whole frame plus horizontal rules straight through
    the figure. The same defect is in the shipped reading -- see the T tile of
    the contact sheet, where those lines are drawn too.

    Two steps, because one is not enough:

    1. **A low alpha threshold.** The streaks are partly transparent, with
       values in the 1-7 band where the figure itself sits at 255. Measured on
       乌尔比安 精0 that band is 2.4% of the frame and dropping it takes the long
       horizontal rows from 41 to zero. An alpha of 7/255 is 3% opacity, which
       nothing legitimate in these files relies on.
    2. **An opening** whose kernel follows the picture's short side with a
       floor of 3 px, for the streaks that survive the threshold. Same
       ratio-with-a-floor rule as the project's despeckling, and for the same
       reason: the defect never gets thinner than a pixel, so a kernel that
       follows the picture down stops being able to open anything.

    Returns ``(alpha, low_share, opened_share)`` so the caller reports what was
    taken out rather than doing it invisibly.
    """
    if alpha is None:
        return alpha, 0.0, 0.0
    height, width = alpha.shape[:2]
    opaque = (alpha > 0).astype(np.uint8)
    total = max(1.0, float(opaque.sum()))

    kept = (alpha > int(low)).astype(np.uint8)
    low_share = float(opaque.sum() - kept.sum()) / total

    size = max(3, int(round(min(height, width) * 0.0022)))
    if size % 2 == 0:
        size -= 1
    size = max(size, int(min_thickness) | 1)
    opened = cv2.morphologyEx(
        kept,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    opened_share = float(opaque.sum() - opened.sum()) / total
    return (opened * 255).astype(np.uint8), low_share, opened_share


#: Magnification before the vector fit, matching :data:`pcbart.export.ARC_TRACE_UPSCALE`.
#: VTracer decides "curve or staircase" at the resolution it is handed, and at
#: 1x a 2000 px drawing still comes back mostly as staircases -- measured, a 1x
#: fit of this artwork produced **4x** the vertices of the product path.
VECTOR_UPSCALE = 3


def vector_svg(ink: np.ndarray, path: Path) -> dict:
    """Write the vectorised reading as an SVG, with the product's own settings.

    The raster variants can only show whether the *classification* is right;
    this is the half of the question a bitmap cannot answer -- whether the
    outline comes back as smooth curves, and what the geometry weighs. The
    settings deliberately mirror :func:`pcbart.export.arc_rings_for`:
    magnification, the thresholds scaled with it, and a spline fit converted to
    arcs. Anything else would measure a file the exporter would never write.

    Returns the measured vector weight, so the report can state it rather than
    leaving "it is a vector" as an unquantified claim.
    """
    try:
        import vtracer

        from pcbart.image_prep import imwrite_unicode
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"    ! SVG skipped: {exc}")
        return {}

    import tempfile
    import time as _time

    from pcbart.vtrace import parse_svg_paths

    started = _time.perf_counter()
    height, width = ink.shape[:2]
    # Smoothed magnification, never nearest: a nearest copy gives the fitter a
    # field of right angles and it answers with straight lines only.
    grey = cv2.resize(
        np.where(ink, 0, 255).astype(np.uint8),
        None,
        fx=float(VECTOR_UPSCALE),
        fy=float(VECTOR_UPSCALE),
        interpolation=cv2.INTER_LINEAR,
    )

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "ink.png"
        # VTracer traces the black area; ``ink`` is our 0 = ink plane already.
        imwrite_unicode(str(source), grey)
        try:
            vtracer.convert_image_to_svg_py(
                str(source),
                str(path),
                colormode="binary",
                hierarchical="cutout",
                mode="spline",
                filter_speckle=2 * VECTOR_UPSCALE,
                corner_threshold=60,
                length_threshold=float(2.0 * VECTOR_UPSCALE),
                max_iterations=10,
                splice_threshold=45,
                path_precision=3,
            )
        except Exception as exc:  # noqa: BLE001 - optional, report and keep going
            print(f"    ! SVG failed: {exc}")
            return {}

    elapsed = _time.perf_counter() - started
    if not path.exists():
        return {}
    # Read back with the project's own parser, which is what the exporter uses
    # to turn this SVG into geometry -- so the count below is what would be
    # written, not an estimate from the file size.
    text = path.read_text(encoding="utf-8")
    try:
        pieces = parse_svg_paths(text)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! SVG written but unreadable: {exc}")
        return {"seconds": elapsed, "bytes": path.stat().st_size}
    rings = sum(len(piece) for piece in pieces)
    vertices = sum(len(ring) for piece in pieces for ring in piece)
    return {
        "seconds": elapsed,
        "bytes": path.stat().st_size,
        "pieces": len(pieces),
        "rings": rings,
        "vertices": vertices,
        "source_px": (width, height),
    }


def widen(ink: np.ndarray, width: int) -> np.ndarray:
    """Grow a thin line mask to roughly ``width`` pixels.

    Same operation as :func:`pcbart.lineart._widen_lines`: dilating by half the
    target width keeps the result centred on the original stroke, where a
    dilate-then-erode pair would be the identity on a thin curve. This runs
    before the vector fit because VTracer needs a stroke with an inside to
    follow -- handed a one-pixel line it breaks it into dots.
    """
    if width <= 2:
        return ink
    size = max(3, int(width) | 1)
    grown = cv2.dilate(
        ink.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    return grown > 0


def arc_smooth(ink: np.ndarray) -> np.ndarray:
    """Re-describe the mask through VTracer's spline fit, back to a raster.

    This is the vector half of the question: it cannot move a pixel's side of
    the threshold, it can only round the staircase and merge neighbouring
    fragments along the way. ``px_per_mm=1`` keeps the probe in pixel units;
    the product pass is the same call the exporter already uses for the faction
    crest, so what is seen here is what geometry would be written.
    """
    from pcbart.lineart import clean_strokes_with_arcs

    try:
        smoothed = clean_strokes_with_arcs(
            np.where(ink, 0, 255).astype(np.uint8), px_per_mm=1.0
        )
    except Exception as exc:  # noqa: BLE001 - optional step, report and keep ink
        print(f"    ! arc smoothing failed: {exc}")
        return ink
    return smoothed <= 128


# --- plumbing ------------------------------------------------------------


def _composite_on_white(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Flatten transparency the way silkscreen sees it -- onto white paper."""
    if alpha is None:
        return bgr
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    white = np.full_like(bgr, 255.0, dtype=np.float32)
    return (bgr.astype(np.float32) * a + white * (1.0 - a)).astype(np.uint8)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _load_pair(path: Path):
    raw = imread_unicode(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"无法读取 {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        return raw[:, :, :3].copy(), raw[:, :, 3].copy()
    return raw, None


def _reduce(image, alpha, max_side: int):
    """Bound the work and hand back the scale that was used."""
    height, width = image.shape[:2]
    scale = work_scale(width, height)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if alpha is not None:
            alpha = cv2.resize(
                alpha,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
    height, width = image.shape[:2]
    if max_side > 0 and max(height, width) > max_side:
        scale2 = max_side / max(height, width)
        image = cv2.resize(
            image,
            (
                max(1, int(round(width * scale2))),
                max(1, int(round(height * scale2))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        if alpha is not None:
            alpha = cv2.resize(
                alpha,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        scale *= scale2
    return image, alpha, scale


def _cell(label: str, image: np.ndarray, tile: int) -> np.ndarray:
    """One labelled tile of the contact sheet."""
    bgr = _as_bgr(image)
    height, width = bgr.shape[:2]
    factor = min(tile / width, tile / height)
    resized = cv2.resize(
        bgr,
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    cell = np.full((tile + 34, tile, 3), 236, np.uint8)
    y0 = 34 + (tile - resized.shape[0]) // 2
    x0 = (tile - resized.shape[1]) // 2
    cell[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(cell, (0, 0), (tile, 33), (43, 43, 43), -1)
    cv2.putText(
        cell,
        label,
        (9, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (238, 238, 238),
        1,
        cv2.LINE_AA,
    )
    return cell


def _contact_sheet(entries, path: Path, tile: int, columns: int = 4) -> None:
    cells = [_cell(label, image, tile) for label, image in entries]
    while len(cells) % columns:
        cells.append(np.full((tile + 34, tile, 3), 236, np.uint8))
    rows = [
        np.hstack(cells[index : index + columns])
        for index in range(0, len(cells), columns)
    ]
    imwrite_unicode(str(path), np.vstack(rows))


def _render(ink: np.ndarray) -> np.ndarray:
    """Ink as it will be shown: black lines on white paper."""
    return np.where(ink, 0, 255).astype(np.uint8)


def _shipped_fill(colour, alpha) -> np.ndarray:
    """The reading the tool ships today, for comparison as the other endpoint.

    ``line_art`` in its default fill mode: Otsu over the subject, every pixel
    sent to black or white, the outer contour on top. It has no notion of a
    line, which is exactly why the pure-line output looks so different.
    """
    from pcbart.lineart import line_art

    rgba = colour if alpha is None else np.dstack([colour, alpha])
    return line_art(rgba) <= 128


def render_reading(
    variant: Variant,
    flat: np.ndarray,
    subject: np.ndarray,
    grey8: np.ndarray,
    masks,
    responses: dict,
    args,
) -> tuple[np.ndarray, float, int, np.ndarray | None]:
    """One variant's ink mask, plus the sigma and the outline width it used.

    Split out of :func:`process_sample` so the slider strip can reuse it
    instead of carrying a second copy of the same steps -- the same reason
    :func:`pcbart.lineart.line_art` keeps one implementation of a reading.
    """
    height, width = flat.shape[:2]
    short_side = min(height, width)

    key = (variant.sigma_ratio, variant.color_weight)
    if key not in responses:
        sigma = max(0.8, short_side * variant.sigma_ratio)
        responses[key] = boundary_response(flat, sigma, variant.color_weight)
    response = responses[key]
    sigma = max(0.8, short_side * variant.sigma_ratio)

    lines = hysteresis(
        np.where(subject, response, -1.0),
        subject,
        variant.density,
        args.weak_ratio,
        args.floor,
    )
    lines = drop_specks(lines, args.min_blob)

    # The outer contour is added from the subject mask, not from the picture:
    # where a pale figure meets a pale page there is no tonal boundary for the
    # detector to find, and this is also the shape the board outline is cut
    # from.
    outline_width = max(3, int(round(sigma * 2.0)) | 1)
    ink = lines | (outer_outline(masks.footprint, outline_width) > 0)

    if variant.fill_weight > 0:
        ink = ink | grow_fill(subject, grey8, variant.fill_weight)
    if variant.vector:
        # Widen first, then fit: the fitter needs a stroke with an inside to
        # follow, and the width is the printed width the placement will be
        # exported at. The widened mask is returned alongside so the SVG is
        # written from the very geometry that was fitted, rather than widening
        # a second time.
        fitted = widen(ink, outline_width)
        return arc_smooth(fitted), sigma, outline_width, fitted
    return ink, sigma, outline_width, None


def slider_strip(
    name: str,
    level: int,
    flat: np.ndarray,
    alpha: np.ndarray | None,
    subject: np.ndarray,
    grey8: np.ndarray,
    masks,
    responses: dict,
    args,
    outdir: Path,
) -> list[float]:
    """The fill-weight slider, one tile per step, in a single wide sheet.

    This is the picture the question is really about: whether "pure lines" and
    "solid blocks" are two ends of one control or two separate algorithms. The
    tiles share every other setting, so the only thing moving is the weight.
    """
    entries = [("00 ORIGINAL", flat)]
    lines_ink, _sigma, _width, _fitted = render_reading(
        SLIDER_BASE, flat, subject, grey8, masks, responses, args
    )
    entries.append(("lines only  w=0.00", _render(lines_ink)))
    coverages = [100.0 * float(lines_ink.mean())]
    for weight in SLIDER_WEIGHTS[1:]:
        step = Variant(
            f"w{weight:.1f}",
            f"w={weight:.2f}",
            SLIDER_BASE.sigma_ratio,
            SLIDER_BASE.density,
            SLIDER_BASE.color_weight,
            fill_weight=weight,
        )
        ink, _sigma, _width, _fitted = render_reading(
            step, flat, subject, grey8, masks, responses, args
        )
        coverages.append(100.0 * float(ink.mean()))
        entries.append((step.label, _render(ink)))
    shipped = _shipped_fill(flat, alpha)
    entries.append(("shipped fill reading", _render(shipped)))
    _contact_sheet(
        entries,
        outdir / f"{name}-精英{level}__slider-strip.png",
        args.tile,
        columns=4,
    )
    shipped_share = 100.0 * float(shipped.mean())
    print(
        "    滑块条总墨量："
        + "  ".join(
            f"w={weight:.1f}:{value:.1f}%"
            for weight, value in zip(SLIDER_WEIGHTS, coverages)
        )
        + f"  现状:{shipped_share:.1f}%"
    )
    return list(zip(SLIDER_WEIGHTS, coverages)), shipped_share


def process_sample(
    name: str,
    level: int,
    path: Path,
    outdir: Path,
    variants,
    args,
):
    t0 = time.perf_counter()
    colour, alpha = _load_pair(path)
    colour, alpha, _scale = _reduce(colour, alpha, args.max_side)
    flat = _composite_on_white(colour, alpha)
    height, width = flat.shape[:2]

    # The streaks are removed before the subject is measured, so the outline,
    # the Otsu sample and the fill all describe the artwork rather than the
    # cut-out's own defects. The share is reported, not hidden.
    cleaned_alpha, low_share, streak_share = clean_alpha_streaks(
        alpha, args.min_thickness, args.alpha_low
    )

    masks = subject_masks(flat, cleaned_alpha)
    if masks.footprint is not None and masks.footprint.any():
        subject = masks.footprint > 0
    else:
        subject = np.ones((height, width), bool)

    grey8 = np.clip(desaturate(flat), 0, 255).astype(np.uint8)
    load_time = time.perf_counter() - t0

    responses: dict = {}
    rows = []
    entries = [("00 ORIGINAL", flat)]

    for variant in variants:
        started = time.perf_counter()
        vector_stats: dict = {}
        if variant.current:
            # Same cleaned alpha as the probe variants, so the comparison is
            # between readings and not between a cleaned and an uncleaned
            # subject.
            ink = _shipped_fill(colour, cleaned_alpha)
            sigma = 0.0
            fitted = None
        else:
            ink, sigma, _width, fitted = render_reading(
                variant, flat, subject, grey8, masks, responses, args
            )

        coverage = 100.0 * float(ink.mean())
        components, _ = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
        elapsed = time.perf_counter() - started

        stem = f"{name}-精英{level}__{variant.key}"
        imwrite_unicode(str(outdir / f"{stem}.png"), _render(ink))
        if variant.vector and fitted is not None:
            # Written from the very mask that went into the fitter, so the SVG
            # and the raster share one widening instead of two.
            vector_stats = vector_svg(fitted, outdir / f"{stem}.svg")
        entries.append((variant.label, _render(ink)))
        rows.append((variant, sigma, elapsed, coverage, components - 1, vector_stats))

    imwrite_unicode(
        str(outdir / f"{name}-精英{level}__00-original.png"), flat
    )
    _contact_sheet(
        entries, outdir / f"{name}-精英{level}__contact-sheet.png", args.tile
    )
    slider_rows, shipped_share = slider_strip(
        name,
        level,
        flat,
        cleaned_alpha,
        subject,
        grey8,
        masks,
        responses,
        args,
        outdir,
    )

    print(
        f"  {name} 精{level}: {width}x{height}, load {load_time:.2f}s, "
        f"subject {100.0 * subject.mean():.0f}%, "
        f"alpha 低阈值 {100.0 * low_share:.2f}% + 开运算共 {100.0 * streak_share:.2f}%"
    )

    # Weight the two vectorisers on the pure-line mask, at the same physical
    # scale the export would use for this placement.
    weights = vector_weight(
        render_reading(SLIDER_BASE, flat, subject, grey8, masks, responses, args)[0],
        width / 76.0,
    )
    for label, shapes, vertices, seconds in weights:
        print(
            f"    矢量规模 {label:24s}: {shapes:5d} 块, "
            f"{vertices:8d} 顶点, {seconds:5.2f}s"
        )

    for variant, sigma, elapsed, coverage, components, vector_stats in rows:
        extra = ""
        if vector_stats:
            extra = (
                f"  矢量 {vector_stats.get('pieces', 0)} 块/"
                f"{vector_stats.get('rings', 0)} 环/"
                f"{vector_stats.get('vertices', 0)} 顶点"
                f"，{vector_stats.get('bytes', 0) / 1e6:.2f} MB，"
                f"{vector_stats.get('seconds', 0):.1f}s"
            )
        print(
            f"    {variant.key:22s} sigma={sigma:5.2f}px "
            f"density={variant.density:.2f} fill={variant.fill_weight:.2f} "
            f"-> {coverage:5.2f}% ink, {components:5d} blobs, {elapsed:5.2f}s"
            f"{extra}"
        )
    return rows, streak_share, weights, slider_rows, shipped_share


def vector_weight(ink: np.ndarray, px_per_mm: float) -> list[tuple[str, int, int, float]]:
    """Weigh both vectorisers on the same mask -- the choice matters enormously.

    The exporter has two, and which one a line drawing should use is not
    obvious until it is measured:

    * ``mask_to_complex_polygons`` -- Douglas-Peucker on the traced contours.
      This is what every line-art placement already goes through.
    * ``trace_mask`` -- VTracer's spline fit converted to arcs, used today for
      the faction crest.

    ⚠️ **The arc path explodes on line art, and this is the measurement that
    says so.** A crest is a handful of simple closed marks, where the fit pays
    for itself. A line drawing is thousands of short strokes with jagged
    boundaries, and the fitter answers each one with many Béziers, each sampled
    into a run of points -- measured on Miss.Christine 精2 the same mask came
    back as **5 583 vertices** through Douglas-Peucker and **961 817** through
    the arc fit, a factor of 172, for a board whose entire template is about
    47 000 vertices.

    Returns ``[(name, shapes, vertices, seconds)]``.
    """
    import time as _time

    from pcbart.vectorize import VectorizeParams, vectorize_mask

    rows = []
    plane = np.where(ink, 0, 255).astype(np.uint8)

    started = _time.perf_counter()
    polygons = vectorize_mask(plane, px_per_mm, VectorizeParams(simplify_mm=0.08))
    vertices = sum(len(ring) for polygon in polygons for ring in polygon)
    rows.append(
        ("Douglas-Peucker 0.08mm", len(polygons), vertices, _time.perf_counter() - started)
    )

    try:
        from pcbart.vtrace import trace_mask

        started = _time.perf_counter()
        pieces = trace_mask(
            plane,
            px_per_mm,
            mode="spline",
            upscale=VECTOR_UPSCALE,
        )
        arcs = sum(1 for piece in pieces for ring in piece for s in ring.sweeps if abs(s) > 1e-9)
        vertices = sum(len(ring.points) for piece in pieces for ring in piece)
        rows.append(
            (
                f"VTracer 弧 ({arcs} 段)",
                len(pieces),
                vertices,
                _time.perf_counter() - started,
            )
        )
    except Exception as exc:  # noqa: BLE001 - optional dependency
        print(f"    ! 弧线测量跳过: {exc}")
    return rows


def self_check() -> list[str]:
    """Guard the one claim the slider depends on, on a synthetic picture.

    At ``weight == 1.0`` the distance threshold has to return the entire dark
    mask, or the end of the slider is not the shipped reading and the "two ends
    of one control" description would be false. Checked here rather than
    eyeballed because a soft edge can leave the outermost ring of a component
    just outside the threshold.
    """
    lines = []
    height, width = 160, 220
    grey = np.full((height, width), 235, np.uint8)
    cv2.rectangle(grey, (30, 40), (100, 130), 70, -1)
    cv2.circle(grey, (160, 80), 45, 180, -1)
    subject = np.ones((height, width), bool)

    level = _otsu_level(grey, subject)
    dark = (grey <= level) & subject
    grown = grow_fill(subject, grey, 1.0)
    missing = int((dark & ~grown).sum())
    lines.append(
        f"- 自检 grow_fill(1.0) 与 Otsu 暗块：漏 {missing} px，"
        f"多 {int((grown & ~dark).sum())} px"
    )
    if missing:
        lines.append("  ⚠️ 滑块满量程不等于现状填色读法，端点对不上")

    empty = grow_fill(subject, np.full_like(grey, 255), 1.0)
    lines.append(f"- 自检 纯白图：填色 {int(empty.sum())} px（应为 0）")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "out" / "_line_probe"),
        help="where the originals and every variant are written",
    )
    parser.add_argument(
        "--operators",
        nargs="*",
        default=list(DEFAULT_OPERATORS),
        help="operator names to test, 精0 and 精2 each",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=[0, 2],
        help="illustration ages to test (0 and/or 2)",
    )
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument(
        "--weak-ratio", type=float, default=2.5,
        help="hysteresis: weak level = strong level / this",
    )
    parser.add_argument("--floor", type=float, default=MIN_CONTRAST)
    parser.add_argument(
        "--min-blob", type=int, default=6,
        help="drop line components smaller than this many pixels",
    )
    parser.add_argument(
        "--min-thickness", type=int, default=3,
        help="opening kernel floor for the alpha streak cleanup, in pixels",
    )
    parser.add_argument(
        "--alpha-low", type=int, default=7,
        help="alpha below this is treated as background before the opening",
    )
    parser.add_argument("--tile", type=int, default=430)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = [
        "# 线条提取探针（未接入软件）",
        "",
        f"- 素材库：`{LIBRARY}`",
        f"- 输出目录：`{outdir}`",
        f"- 工作分辨率上限：{args.max_side} px",
        f"- 迟滞阈值：强 = 主体响应的分位（密度），弱 = 强 / {args.weak_ratio}",
        f"- 对比度下限：{args.floor}，最小色块：{args.min_blob} px",
        "",
        "## 变体参数",
        "",
        "| 变体 | σ/短边 | 密度 | 色差权重 | 填充 | 矢量化 |",
        "|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        report.append(
            f"| {variant.key} | {variant.sigma_ratio:.4f} | {variant.density:.2f} | "
            f"{variant.color_weight:.2f} | {variant.fill_weight:.2f} | "
            f"{'是' if variant.vector else '否'} |"
        )
    report.append("")
    report.append("## 自检")
    report.append("")
    report.extend(self_check())
    report.append("")

    for name in args.operators:
        for level in args.levels:
            folder, suffix = ELITE[level]
            path = LIBRARY / folder / f"立绘 {name} {suffix}.png"
            if not path.exists():
                print(f"  跳过（不存在）：{path}")
                continue
            print(f"处理 {name} 精{level} …")
            rows, streak_share, weights, slider_rows, shipped_share = process_sample(
                name, level, path, outdir, VARIANTS, args
            )
            report.append(f"## {name} · 精英{level}")
            report.append("")
            report.append(f"源文件：`{path}`")
            report.append("")
            report.append(
                f"预清理移除 {100.0 * streak_share:.2f}% 的不透明像素"
                f"（alpha 阈值 >{args.alpha_low}，开运算核下限 {args.min_thickness}px）。"
            )
            report.append("")
            report.append("| 变体 | σ(px) | 墨量 | 色块数 | 用时(s) | 矢量规模 |")
            report.append("|---|---|---|---|---|---|")
            for variant, sigma, elapsed, coverage, components, vector_stats in rows:
                if vector_stats:
                    weight = (
                        f"{vector_stats.get('rings', 0)} 环 / "
                        f"{vector_stats.get('vertices', 0)} 顶点 / "
                        f"{vector_stats.get('bytes', 0) / 1e6:.2f} MB"
                    )
                else:
                    weight = "—"
                report.append(
                    f"| {variant.key} | {sigma:.2f} | {coverage:.2f}% | "
                    f"{components} | {elapsed:.2f} | {weight} |"
                )
            report.append("")
            report.append("### 两种矢量化的规模对比（同一张纯线条掩膜）")
            report.append("")
            report.append("| 矢量化方式 | 图形数 | 顶点数 | 用时(s) |")
            report.append("|---|---|---|---|")
            for label, shapes, vertices, seconds in weights:
                report.append(
                    f"| {label} | {shapes} | {vertices} | {seconds:.2f} |"
                )
            report.append("")
            report.append("### 填色滑块（其余参数取 B）")
            report.append("")
            report.append(
                "| 填色权重 | "
                + " | ".join(f"{weight:.1f}" for weight, _ in slider_rows)
                + f" | 现状 fill |"
            )
            report.append(
                "|---|" + "---|" * (len(slider_rows) + 1)
            )
            report.append(
                "| 墨量 | "
                + " | ".join(f"{value:.2f}%" for _, value in slider_rows)
                + f" | {shipped_share:.2f}% |"
            )
            report.append("")
            report.append(
                "⚠️ 满量程（w=1.0）与现状 fill 并不逐像素相同：现状还额外叠加了"
                "`outer_outline` 外轮廓，且它的 Otsu 采样范围不同。详见记忆 "
                "`pcb-line-extraction-probe`。"
            )
            report.append("")

    (outdir / "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n输出目录：{outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
