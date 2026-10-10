"""Probe: redraw the illustration as clean two-tone line art with a learned model.

The pixel readings in :mod:`pcbart.lineart` ask a *local* question -- "is this
pixel an edge?", "is it darker than its neighbourhood?" -- and a local question
cannot produce a drawing. It can only trace what is already there, so shaded
areas come out as clumps of contour, fine texture comes out as noise, and no
setting makes a lock of hair read as one confident stroke.

The picture the user compared against was not filtered, it was **redrawn**: an
image model looked at the illustration and drew line art of it, inventing clean
strokes and dropping detail it judged irrelevant. That is a different operation
and it needs a different kind of tool.

This probe runs **Anime2Sketch** (Mukosame, MIT), a U-Net trained on pairs of
anime frame / hand-drawn sketch. Because its target during training *was* a
human line drawing, it does the redrawing rather than the tracing: it returns
single clean strokes, keeps characters, clothing and weapons legible, and
leaves flat colour areas empty.

Four stages, each measurable on its own:

1. **Redraw** -- the model, on a padded canvas so any aspect ratio survives the
   U-Net's eight halvings.
2. **Rescale the response, then threshold** -- the model returns a soft grey
   response, so it is enlarged back to the working resolution *before* the
   threshold. Thresholding first and enlarging after would put a staircase on
   every stroke.
3. **Force two tones** -- pure 0/255, because silkscreen cannot print grey.
4. **Drop over-thin strokes** -- the user's own requirement ("去除过细的线条"),
   expressed in **millimetres**, so the filter means the same thing on a board
   as it does on screen. A stroke thinner than the fab floor is not ink, it is
   a defect.

Nothing here is wired into the panel.

Usage::

    python tools/probe_line_redraw.py
    python tools/probe_line_redraw.py --operators 乌尔比安 --levels 0 2
    python tools/probe_line_redraw.py --min-width-mm 0.10 0.18 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402
from pcbart.model_paths import (  # noqa: E402
    check_model_dir,
    default_model_dir,
    resolve_model_dir,
)
from pcbart.paths import asset_root  # noqa: E402
from pcbart.work_scale import work_scale  # noqa: E402

#: Resolved through ``PCBART_ASSET_ROOT``, not written out: the library lives
#: outside the repository and on another drive on some machines.
LIBRARY = asset_root() / "明日方舟"
ELITE = {
    0: ("06-立绘-精英0", "1"),
    2: ("08-立绘-精英2", "2"),
}
DEFAULT_OPERATORS = ("乌尔比安", "能天使", "临光", "Miss.Christine")

#: Where the model and its architecture are looked for by default.
#:
#: ⚠️ **The model is not shipped** -- 183 MB of weights against a 178 MB
#: installer, and a CPU redraw peaks at 4.6 GB of RAM -- so the folder is
#: resolved at run time (see :mod:`pcbart.model_paths`): ``--model-dir``, then
#: ``PCBART_A2S_MODEL``, then the folder chosen in【界面设置】, then this drop-in
#: location. Kept as a module attribute because the probe's own callers and
#: tests read it, but it is a *fallback*, not the answer.
MODEL_DIR = default_model_dir()

#: The U-Net halves the picture eight times, so each side of what it is handed
#: has to divide by 256 or the innermost block sees a ragged tensor. Rather
#: than trusting every size to land on a multiple, the picture is padded to one.
SIZE_MULTIPLE = 256

#: Long side the model is run at, in pixels. It was trained at 512, and asking
#: for much more does not add drawing -- it adds input the network has no
#: calibration for. The response is scaled back up afterwards, so the strokes
#: still land at the working resolution; see :func:`model_response`.
DEFAULT_MODEL_SIZE = 512

#: Absolute floor for an adaptive threshold, in response units.
#:
#: ⚠️ **A quantile alone is not safe, and this is why.** Asking for "the
#: darkest 5% of the response" always returns 5% of the pixels as ink, whether
#: or not the picture has any lines: on a flat field every pixel's response is
#: near 255, so the quantile lands inside that population and the whole frame
#: becomes ink. The floor is what stops it -- the same role
#: :data:`pcbart.lineart.OUTLINE_MIN_CONTRAST` plays for the pixel reading.
#: Measured: nothing in the library's strokes sits above 250, so a floor there
#: would erase real lines; 60 is below every stroke the model produced and
#: above the flat-field noise.
ADAPTIVE_FLOOR = 60.0

DEFAULT_QUANTILES = (4.0, 6.0, 9.0)
"""Response quantiles to try as thresholds, in percent.

**Why the threshold has to adapt at all, measured.** The model's output is a
*soft response*, not a calibrated ink map, and its scale moves with the
picture:

=======================  ======  ======  ======
picture (精0)              1%      5%     10%
=======================  ======  ======  ======
小色 (bright photo)       172.7   248.9  254.1
清流 (pale illustration)   52.9   235.1  253.6
闪击 (mid grey)           132.5   208.4  237.4
傀影 (dark, low contrast) 110.3   211.2  243.9
=======================  ======  ======  ======

A fixed 200 is right for 闪击 and 傀影 and returns almost nothing for 清流 and
小色 -- measured, 清流 at 200 was a few disconnected fragments while the same
picture at its own 5% quantile (235) came back as the figure, the hat and the
weapon. "Keep the darkest N%" is scale-free, which is the same reasoning the
project already uses for the contour reading's density setting.
"""


def load_model(model_dir: Path, device: str = "auto", variant: str = "improved"):
    """The Anime2Sketch generator, with one of its two published weight sets.

    ``variant`` selects which weights, and the two are not interchangeable:

    * ``"default"`` -- the original release.
    * ``"improved"`` -- the author's *artifact-free* retrain, published
      specifically for **dark and low-contrast** pictures. The library's
      illustrations are exactly that (乌尔比安's cloak is nearly black on black),
      and measured side by side the improved weights returned noticeably
      longer unbroken strokes and a legible face where the default left the
      middle of the figure as scattered fragments. The loading path differs
      too: the improved weights replace six deconvolution layers with
      resize-plus-conv blocks, which is why the switch lives in the reference
      implementation rather than here.

    The architecture is imported from the downloaded reference implementation
    rather than re-declared: a hand-copied U-Net whose layer order drifted by
    one would still load a state dict in some orders and silently draw the
    wrong thing, and the only cheap way to rule that out is to use the exact
    code the weights were published with.
    """
    import importlib.util

    import torch

    if variant not in ("default", "improved"):
        raise ValueError(f"未知模型变体：{variant}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_py = model_dir / "model.py"
    weights = model_dir / "weights" / (
        "improved.bin" if variant == "improved" else "netG.pth"
    )
    if not model_py.exists() or not weights.exists():
        raise FileNotFoundError(
            f"缺少模型文件：\n  {model_py}\n  {weights}\n"
            "运行 tools/fetch_anime2sketch.py 获取。"
        )

    spec = importlib.util.spec_from_file_location("a2s_model", model_py)
    module = importlib.util.module_from_spec(spec)
    # ``model.py`` loads its weights from a path relative to the CWD, so the
    # loader is driven from inside the model directory.
    previous = os.getcwd()
    os.chdir(model_dir)
    try:
        spec.loader.exec_module(module)
        net = module.create_model(variant)
    finally:
        os.chdir(previous)
    net.eval()
    return net.to(device), device


def _pad_to_multiple(image: np.ndarray, multiple: int = SIZE_MULTIPLE):
    """Pad BGR (or grey) on the right and bottom with white.

    White rather than black: the model reads a picture the way it would read a
    frame, and a black bar is a hard edge it will dutifully draw a line along.
    The crop after inference removes the padding, so it cannot leave a mark.
    """
    height, width = image.shape[:2]
    target_h = ((height + multiple - 1) // multiple) * multiple
    target_w = ((width + multiple - 1) // multiple) * multiple
    if target_h == height and target_w == width:
        return image, (0, 0, width, height)
    fill = 255 if image.ndim == 2 else (255, 255, 255)
    canvas = np.full((target_h, target_w) + image.shape[2:], fill, dtype=image.dtype)
    canvas[:height, :width] = image
    return canvas, (0, 0, width, height)


def _model_canvas(shape: tuple[int, int], model_size: int) -> tuple[int, int]:
    """The size the network is fed, as a multiple of :data:`SIZE_MULTIPLE`.

    The long side is brought to ``model_size`` and each side is then rounded
    **up** to a multiple of 256, because the U-Net halves the picture eight
    times and a side that does not divide leaves the innermost block with a
    ragged tensor. Rounding up rather than to-nearest avoids the case where a
    side lands one pixel under a multiple and the aspect ratio is changed to
    compensate.
    """
    height, width = shape
    scale = float(model_size) / max(1, max(height, width))
    target_h = int(np.ceil(height * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    target_w = int(np.ceil(width * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    return max(SIZE_MULTIPLE, target_h), max(SIZE_MULTIPLE, target_w)


def run_model(net, device: str, colour: np.ndarray) -> np.ndarray:
    """The model's soft response for one picture, in 0..255 (dark = line).

    Returns a ``float32`` plane at the size it was handed, *before* any
    threshold. That is deliberate: the caller rescales this response rather
    than a binary mask, so upscaling cannot bake a staircase into the stroke.
    """
    import torch

    padded, _crop = _pad_to_multiple(colour)
    # The reference pipeline normalises to [-1, 1] over RGB channels; matching
    # it exactly matters, because a differently scaled input changes which
    # strokes the network considers confident.
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        output = net(tensor.to(device))
    # Output is a single channel in [-1, 1]; -1 is a line, matching the
    # reference's own post-processing.
    plane = output[0, 0].detach().cpu().numpy()
    plane = (plane + 1.0) / 2.0 * 255.0
    return np.clip(plane, 0, 255).astype(np.float32)


def remove_thin_strokes(mask: np.ndarray, min_width_px: float) -> np.ndarray:
    """Drop every stroke thinner than ``min_width_px``.

    The test is a morphological opening with a disc of that diameter: a stroke
    with nowhere a disc of that size fits is removed outright, and a stroke
    that is thick enough survives largely unchanged. That is the honest reading
    of "去掉过细的线条" -- it is a statement about whether ink can carry, which
    is a width, not a count of pixels.

    ⚠️ **An opening is not the same as filtering by area.** A long hairline has
    a large pixel count and a tiny width, so an area filter keeps exactly the
    strokes this is meant to drop. Measured on 乌尔比安 the thin alpha streaks
    are thousands of pixels each and under two pixels wide.
    """
    if min_width_px <= 1:
        return mask
    diameter = max(3, int(round(float(min_width_px))) | 1)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, element, iterations=1
    )
    return opened > 0


def stroke_width_stats(mask: np.ndarray) -> dict:
    """How thick the ink actually is, as percentiles of local stroke width.

    Measured through the distance transform: for any ink pixel the distance to
    the nearest paper is at most half the width of the stroke it sits on, so
    twice the distance transform's local maxima is a stroke's width. Taking the
    maxima rather than every pixel avoids counting the thin ends of a thick
    stroke as thin strokes of their own.
    """
    if not mask.any():
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # A pixel is on the centreline when no 8-neighbour is farther from paper.
    dilated = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    ridge = (distance >= dilated - 1e-6) & (distance > 0)
    widths = 2.0 * distance[ridge]
    if widths.size == 0:
        widths = 2.0 * distance[distance > 0]
    return {
        "median": float(np.median(widths)),
        "p10": float(np.percentile(widths, 10)),
        "p90": float(np.percentile(widths, 90)),
        "max": float(widths.max()),
    }


def clean_alpha_border(image: np.ndarray, alpha: np.ndarray | None):
    """Composite onto white and drop the cut-out's own edge artefacts.

    ⚠️ **The library's alpha carries hair-thin opaque streaks** -- 41 rows
    across more than three quarters of 乌尔比安 精0's width, three to five pixels
    thick, at alpha 1-7 where the figure itself is 255. Left in, the model
    reads them as part of the picture and draws a border around the frame. They
    are removed before the composite so the network never sees them.
    """
    if alpha is None:
        return image, 0.0
    kept = (alpha > 7).astype(np.uint8)
    height, width = kept.shape[:2]
    size = max(3, int(round(min(height, width) * 0.0022))) | 1
    opened = cv2.morphologyEx(
        kept, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    )
    removed = float(kept.sum() - opened.sum()) / max(1.0, float(kept.sum()))
    a = (opened * 255).astype(np.float32) / 255.0
    a = a[:, :, None]
    flat = image.astype(np.float32) * a + 255.0 * (1.0 - a)
    return flat.astype(np.uint8), removed


def _reduce(image: np.ndarray, alpha: np.ndarray | None, max_side: int):
    height, width = image.shape[:2]
    scale = work_scale(width, height)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    if max_side > 0 and max(height, width) > max_side:
        scale2 = max_side / max(height, width)
        size = (max(1, round(width * scale2)), max(1, round(height * scale2)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    return image, alpha


def paper_plateau(response: np.ndarray) -> int | None:
    """The value the background sits at, or ``None`` when there is no plateau.

    ⚠️ **This exists because a quantile alone can flood the frame, and that was
    measured.** The model's response piles up on a single value where there is
    no line -- 88% of 清流's pixels are exactly 254, 90% of 小色's, 78% of 断罪者's.
    A quantile can land *inside* that pile: 清流's 12th percentile is 254, and
    ``response <= 254`` then selects the entire background, giving 99.2% ink and
    a solid black card. The threshold must stay below the pile.

    The plateau is the most common value, and it only counts as one when it sits
    in the upper half of the observed range -- a mostly dark picture has a mode
    too, but it is the artwork, not paper, and capping the threshold just under
    it would ink everything.
    """
    if response.size == 0:
        return None
    counts = np.bincount(response.reshape(-1).astype(np.uint8), minlength=256)
    mode = int(np.argmax(counts))
    highest = int(response.max())
    if mode < highest * 0.5:
        return None
    return mode


def adaptive_level(response: np.ndarray, quantile: float, floor: float = ADAPTIVE_FLOOR) -> float:
    """The threshold for ``quantile`` percent of the response.

    Three clamps, each answering a different failure:

    1. the **quantile** asks for the darkest N%, which is what makes the reading
       follow a picture's own contrast instead of a fixed level;
    2. the **floor** stops a picture with no lines from returning its own noise
       (a flat field puts the quantile inside the pile, and without a floor the
       whole frame inks);
    3. staying **below the paper plateau** stops the opposite failure -- the
       quantile landing inside the background pile itself. See
       :func:`paper_plateau` for the measurement.
    """
    level = max(float(floor), float(np.percentile(response, quantile)))
    plateau = paper_plateau(response)
    if plateau is not None:
        level = min(level, float(plateau) - 1.0)
    return level


def model_response(
    net,
    device: str,
    colour: np.ndarray,
    model_size: int,
) -> np.ndarray:
    """The network's soft response, enlarged back to the working resolution.

    Two things happen at different resolutions on purpose:

    * the **model** runs at its own size (:data:`DEFAULT_MODEL_SIZE`), which is
      where it has calibration;
    * the **threshold** is applied afterwards, in the caller, to a response that
      has been scaled back up.

    Doing it the other way -- threshold at model size, then enlarge the mask --
    puts every stroke's staircase into the final drawing. The response is
    resampled with cubic interpolation for the same reason: it is a *soft*
    field, so interpolating it reconstructs an edge *between* pixels instead of
    magnifying a block.
    """
    height, width = colour.shape[:2]
    model_h, model_w = _model_canvas((height, width), model_size)
    small = cv2.resize(colour, (model_w, model_h), interpolation=cv2.INTER_AREA)
    padded = run_model(net, device, small)[:model_h, :model_w]
    return cv2.resize(
        padded.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )


def _cell(label: str, image: np.ndarray, tile: int) -> np.ndarray:
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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
        cell, label, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 238, 238), 1, cv2.LINE_AA
    )
    return cell


def contact_sheet(entries, path: Path, tile: int, columns: int = 4) -> None:
    cells = [_cell(label, image, tile) for label, image in entries]
    while len(cells) % columns:
        cells.append(np.full((tile + 34, tile, 3), 236, np.uint8))
    rows = [
        np.hstack(cells[index : index + columns])
        for index in range(0, len(cells), columns)
    ]
    imwrite_unicode(str(path), np.vstack(rows))


def _load_pair(path: Path):
    """Colour and alpha, for every channel layout the library actually uses.

    ⚠️ An 8-bit PNG with transparency can come back as **two** channels from
    ``IMREAD_UNCHANGED`` (grey + alpha), not four. Treating only the 4-channel
    case as transparent would send a 2-channel array into ``cvtColor``, which
    raises -- and the shape checks further down would not have caught it
    first.
    """
    raw = imread_unicode(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"无法读取 {path}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    channels = raw.shape[2]
    if channels == 4:
        return raw[:, :, :3].copy(), raw[:, :, 3].copy()
    if channels == 2:
        return cv2.cvtColor(raw[:, :, 0], cv2.COLOR_GRAY2BGR), raw[:, :, 1].copy()
    if channels == 3:
        return raw.copy(), None
    raise ValueError(f"{path}: 不支持的通道数 {channels}")


def current_reading(colour, alpha) -> np.ndarray:
    """What the tool does today, for a same-page comparison."""
    from pcbart.lineart import line_art

    rgba = colour if alpha is None else np.dstack([colour, alpha])
    return line_art(rgba) <= 128


def process(net, device, name, level, path, outdir, args):
    raw = _load_pair(path)
    colour, alpha = _reduce(raw[0], raw[1], args.max_side)
    flat, removed = clean_alpha_border(colour, alpha)
    height, width = flat.shape[:2]

    # px_per_mm uses the same convention as the rest of the project: the
    # illustration is drawn at 76.2 mm on its long side by default.
    px_per_mm = max(height, width) / float(args.drawn_mm)
    # ``level`` is empty for manifest samples, which are already named. The
    # stem has to collapse the separator rather than leave a dangling
    # "精英" in every filename.
    stem = f"{name}-精英{level}" if str(level) else str(name)

    started = time.perf_counter()
    response = model_response(net, device, flat, args.model_size)
    model_time = time.perf_counter() - started

    imwrite_unicode(str(outdir / f"{stem}__00-original.png"), flat)
    # The raw response, shown as the model produced it. Everything after this
    # is a threshold and a width filter, so this tile is where to look to see
    # whether a mistake came from the network or from the cleanup.
    imwrite_unicode(
        str(outdir / f"{stem}__01-raw-response.png"),
        np.clip(response, 0, 255).astype(np.uint8),
    )

    # --- axis 1: the response threshold --------------------------------
    # Adaptive by default: the model's response scale follows the picture's
    # own brightness, so a fixed level that suits one illustration erases
    # another. See DEFAULT_QUANTILES for the measurements.
    threshold_rows = []
    for quantile in args.quantiles:
        level = adaptive_level(response, quantile)
        mask = remove_thin_strokes(
            response <= level, args.base_width_mm * px_per_mm
        )
        stats = stroke_width_stats(mask)
        coverage = 100.0 * float(mask.mean())
        threshold_rows.append((quantile, level, coverage, stats, mask))
        imwrite_unicode(
            str(outdir / f"{stem}__q{quantile:04.1f}.png"),
            np.where(mask, 0, 255).astype(np.uint8),
        )
        print(
            f"    分位 {quantile:4.1f}% → 阈值 {level:5.1f}（上限 {ADAPTIVE_FLOOR:.0f}）: "
            f"墨量 {coverage:5.2f}%  线宽中位 {stats['median']:5.2f}px  "
            f"p10 {stats['p10']:5.2f}px  最粗 {stats['max']:5.1f}px"
        )

    # --- axis 2: the thinnest stroke that survives ----------------------
    base_level = adaptive_level(response, args.base_quantile)
    width_rows = []
    for width_mm in args.min_width_mm:
        min_px = width_mm * px_per_mm
        mask = remove_thin_strokes(response <= base_level, min_px)
        stats = stroke_width_stats(mask)
        coverage = 100.0 * float(mask.mean())
        width_rows.append((width_mm, min_px, coverage, stats, mask))
        imwrite_unicode(
            str(outdir / f"{stem}__w{width_mm:.2f}mm.png"),
            np.where(mask, 0, 255).astype(np.uint8),
        )
        print(
            f"    最细>{width_mm:.2f}mm ({min_px:5.1f}px): 墨量 {coverage:5.2f}%  "
            f"线宽中位 {stats['median']:5.2f}px  p10 {stats['p10']:5.2f}px"
        )

    current = current_reading(colour, alpha)
    imwrite_unicode(
        str(outdir / f"{stem}__90-current-algorithm.png"),
        np.where(current, 0, 255).astype(np.uint8),
    )

    entries = [
        ("00 ORIGINAL", flat),
        ("01 model response", np.clip(response, 0, 255).astype(np.uint8)),
    ]
    entries += [
        (f"q {quantile:.1f}% (t={level:.0f})", np.where(mask, 0, 255).astype(np.uint8))
        for quantile, level, _cov, _st, mask in threshold_rows
    ]
    entries += [
        (f"min {width_mm:.2f} mm", np.where(mask, 0, 255).astype(np.uint8))
        for width_mm, _px, _cov, _st, mask in width_rows
    ]
    entries.append(("current algorithm", np.where(current, 0, 255).astype(np.uint8)))
    contact_sheet(entries, outdir / f"{stem}__contact-sheet.png", args.tile)

    print(
        f"  {stem}: {width}x{height}, {px_per_mm:.2f} px/mm, "
        f"alpha 清理 {100.0 * removed:.2f}%, 模型 {model_time:.2f}s"
    )
    return threshold_rows, width_rows, px_per_mm, model_time, removed


def read_manifest(path: Path) -> list[tuple[str, str, Path]]:
    """``[(label, colour_type, path)]`` from a two-or-three column text file.

    A manifest exists so a batch chosen for *coverage* can say what each sample
    is covering. The operator/level pairing the other entry point uses cannot
    express "the darkest picture in the library" or "the most polychrome one",
    and those are the properties this batch was picked for.

    One sample per line: ``label<TAB>colour type<TAB>path``. Blank lines and
    ``#`` comments are skipped, so the file can be edited by hand.
    """
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split("\t") if part.strip()]
        if len(parts) == 2:
            parts.append("未分类")
        if len(parts) != 3:
            raise SystemExit(f"{path}:{number}: 需要「标签<TAB>色彩类型<TAB>路径」")
        label, colour_type, target = parts
        rows.append((label, colour_type, Path(target)))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default=str(ROOT / "out" / "_line_redraw")
    )
    parser.add_argument("--operators", nargs="*", default=list(DEFAULT_OPERATORS))
    parser.add_argument("--levels", nargs="*", type=int, default=[0, 2])
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument(
        "--model-size", type=int, default=DEFAULT_MODEL_SIZE,
        help="long side the network runs at; it was trained at 512",
    )
    parser.add_argument(
        "--quantiles", nargs="*", type=float, default=list(DEFAULT_QUANTILES),
        help="响应分位（百分比），自适应阈值扫描；阈值 = max(下限, 该分位)",
    )
    parser.add_argument(
        "--base-quantile", type=float, default=6.0,
        help="最细线宽扫描时固定使用的分位",
    )
    parser.add_argument(
        "--base-width-mm", type=float, default=0.10,
        help="width held fixed while the threshold is swept",
    )
    parser.add_argument(
        "--min-width-mm", nargs="*", type=float, default=[0.10, 0.15, 0.25],
        help="stroke widths to keep, in millimetres on the finished board",
    )
    parser.add_argument(
        "--drawn-mm", type=float, default=76.2,
        help="long side the illustration is drawn at, for mm<->px",
    )
    parser.add_argument("--tile", type=int, default=430)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--model", default="improved", choices=("default", "improved"),
        help="improved = 作者针对深色/低对比图重训的无瑕疵版本",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Anime2Sketch 模型文件夹；不给则依次看 PCBART_A2S_MODEL、"
             "界面设置里的选择，都没有才用默认投放位置",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="改用清单里的样本：每行「标签<TAB>色彩类型<TAB>路径」",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolved here rather than at import time: the choice can come from the
    # saved preferences, which may have changed since this module was loaded,
    # and a missing model deserves a sentence saying which file is missing from
    # which folder, instead of a FileNotFoundError from deep inside the loader.
    model_dir = resolve_model_dir(args.model_dir)
    report = check_model_dir(model_dir, args.model)
    if not report["ok"]:
        raise SystemExit(
            f"模型不可用：{model_dir}\n"
            f"缺少 {'；'.join(report['missing'])}\n\n"
            "可以用 --model-dir 指定，或设置 PCBART_A2S_MODEL，"
            "或在软件的【视图 → 界面设置…】里选择，"
            "或运行 python tools\\fetch_anime2sketch.py 下载。"
        )
    net, device = load_model(model_dir, args.device, args.model)
    print(f"模型已加载，设备 {device}，变体 {args.model}，目录 {model_dir}")

    report = [
        "# AI 重绘线稿探针（未接入软件）",
        "",
        f"- 模型：Anime2Sketch（MIT）`{args.model}`，`{model_dir}`",
        f"- 设备：{device}",
        f"- 网络输入长边：{args.model_size} px（补白到 {SIZE_MULTIPLE} 的倍数）",
        f"- 自适应阈值分位：{args.quantiles}（阈值 = max({ADAPTIVE_FLOOR:.0f}, 该分位)）",
        f"- 成品长边：{args.drawn_mm} mm（用于 mm↔px）",
        f"- 输出目录：`{outdir}`",
        "",
        "| 序号 | 处理 |",
        "|---|---|",
        "| 00 | 原图（合成到白底，清掉 alpha 细条） |",
        "| 01 | 模型输出的软响应（灰阶，亮=纸、暗=线） |",
        "| q* | 自适应阈值扫描：取响应最暗的 N%（有绝对下限） |",
        "| w* | 最细线宽扫描：阈值固定 |",
        "| 90 | 现有 `line_art()` fill 读法，同页对照 |",
        "",
        "⚠️ **模型的响应大量饱和在 255**（纸），只有笔画落在 100–250 之间，"
        "而且这个范围**随图片明暗整体漂移**——亮色主体的响应比深色主体亮几十级。"
        "固定阈值必然顾此失彼（实测亮图在阈值 200 下几乎空白），"
        "所以默认用**自适应分位**：取最暗的 N%，再加一个绝对下限防平色图。",
        "",
    ]

    if args.manifest:
        samples = read_manifest(Path(args.manifest))
    else:
        samples = []
        for name in args.operators:
            for level in args.levels:
                folder, suffix = ELITE[level]
                samples.append(
                    (f"{name} 精{level}", "", LIBRARY / folder / f"立绘 {name} {suffix}.png")
                )

    for label, colour_type, path in samples:
        if not path.exists():
            print(f"  跳过（不存在）：{path}")
            continue
        print(f"处理 {label} …")
        threshold_rows, width_rows, px_per_mm, model_time, removed = process(
            net, device, label, "", path, outdir, args
        )
        report.append(f"## {label}" + (f" · {colour_type}" if colour_type else ""))
        report.append("")
        report.append(f"源文件：`{path}`")
        report.append("")
        report.append(
            f"工作分辨率下 {px_per_mm:.2f} px/mm；模型 {model_time:.2f}s；"
            f"alpha 预清理 {100.0 * removed:.2f}%。"
        )
        report.append("")
        report.append(
            f"### 自适应阈值扫描（最细保留固定 {args.base_width_mm:.2f} mm）"
        )
        report.append("")
        report.append("| 分位 | 实际阈值 | 墨量 | 线宽中位 | 线宽 p10 | 最粗 |")
        report.append("|---|---|---|---|---|---|")
        for quantile, level, coverage, stats, _mask in threshold_rows:
            report.append(
                f"| {quantile:.1f}% | {level:.1f} | {coverage:.2f}% | "
                f"{stats['median']:.2f}px | {stats['p10']:.2f}px | "
                f"{stats['max']:.1f}px |"
            )
        report.append("")
        report.append(
            f"### 最细线宽扫描（阈值固定为 {args.base_quantile:.1f}% 分位）"
        )
        report.append("")
        report.append("| 最细保留 | 像素 | 墨量 | 线宽中位 | 线宽 p10 |")
        report.append("|---|---|---|---|---|")
        for width_mm, min_px, coverage, stats, _mask in width_rows:
            report.append(
                f"| {width_mm:.2f} mm | {min_px:.1f} | {coverage:.2f}% | "
                f"{stats['median']:.2f}px | {stats['p10']:.2f}px |"
            )
        report.append("")

    (outdir / "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n输出目录：{outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
