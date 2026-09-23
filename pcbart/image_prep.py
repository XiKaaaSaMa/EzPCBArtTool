"""Image loading and preprocessing for outline extraction.

The goal of this stage is a clean binary mask where every foreground pixel
means "board material". Everything downstream -- contour tracing, simplification,
process compensation -- operates on that mask, so this is where noise, uneven
lighting and polarity have to be dealt with.

The pipeline is deliberately classical (no learned model): it is deterministic,
fast, inspectable and needs no GPU. An AI stage can later replace or feed the
front of it (see ROADMAP M4); the contract is simply "produce a mask".

Every parameter is exposed because the right amount of denoising depends on the
source: a clean vector export needs almost none, a photographed pencil sketch
needs a lot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PrepParams",
    "PrepResult",
    "load_gray",
    "binarize",
    "auto_polarity",
    "preprocess",
    "mask_to_preview",
]


@dataclass
class PrepParams:
    """Tunable knobs for the preprocessing chain.

    Defaults suit a clean line drawing (dark strokes on a light background).
    """

    threshold_mode: str = "otsu"
    """``"otsu"`` for a global threshold, ``"adaptive"`` for uneven lighting."""

    adaptive_block: int = 35
    """Neighbourhood size for adaptive thresholding; must be odd and > 1."""

    adaptive_c: float = 5.0
    """Constant subtracted in adaptive thresholding; higher removes more speckle."""

    invert: bool | None = None
    """Force polarity. ``None`` detects it, ``True`` treats light-on-dark as subject."""

    blur_radius: int = 1
    """Median blur radius applied before thresholding; 0 disables it."""

    morph_open: int = 1
    """Opening kernel size, removes isolated speckles. 0 disables it."""

    morph_close: int = 2
    """Closing kernel size, bridges small gaps. 0 disables it."""

    min_component_area: int = 64
    """Connected components smaller than this (in pixels) are dropped."""

    fill_holes: bool = True
    """Fill interior holes in the mask.

    Turn on when the artwork should become a filled silhouette. Turn off to
    keep genuine cutouts (the middle of an ``O``) as board holes.
    """

    use_alpha: bool = True
    """Use the image's alpha channel as the mask when present."""


@dataclass
class PrepResult:
    """What preprocessing produced, plus the numbers behind the decisions."""

    mask: np.ndarray
    foreground_ratio: float
    inverted: bool
    source_size: tuple[int, int]
    steps: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    def summary(self) -> str:
        return (
            f"mask {self.width}x{self.height}, foreground {self.foreground_ratio * 100:.1f}%, "
            f"inverted={self.inverted}, steps: {', '.join(self.steps) or 'none'}"
        )


def imread_unicode(path: str | Path, flags: int):
    """Read an image from a path that may contain non-ASCII characters.

    ``cv2.imread`` cannot open a path like ``C:\\图片\\作品.png`` on Windows --
    it hands the bytes to the C runtime, which reads them as the ANSI code
    page, and returns ``None``. Reading the file ourselves and decoding from
    memory sidesteps that entirely. The same applies to ``cv2.imwrite``, which
    is why :func:`imwrite_unicode` exists.
    """
    import cv2
    import numpy as np

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取图片: {path}（{exc.strerror or exc}）") from exc
    if buffer.size == 0:
        raise ValueError(f"无法读取图片: {path}（文件为空）")

    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"无法解码图片: {path}（可能不是图片，或格式不受支持）")
    return image


def imwrite_unicode(path: str | Path, image, ext: str | None = None) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    import cv2
    import numpy as np

    path = Path(path)
    if ext is None:
        ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def load_gray(path: str | Path, use_alpha: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image as grayscale plus an optional alpha mask.

    Alpha, when present and not fully opaque, is treated as a ready-made subject
    mask -- it is almost always more accurate than thresholding the RGB.
    """
    import cv2

    image = imread_unicode(path, cv2.IMREAD_UNCHANGED)

    alpha: np.ndarray | None = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if use_alpha and int(alpha.min()) == 255:
            alpha = None  # fully opaque: nothing useful to learn
        image = image[:, :, :3]

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray, alpha


def auto_polarity(binary: np.ndarray) -> bool:
    """Decide whether ``binary`` needs inverting so foreground is the subject.

    The border of a picture is nearly always background. So if most border
    pixels come out as foreground, the threshold was read the wrong way round.

    Returns ``True`` when the image should be inverted.
    """
    height, width = binary.shape[:2]
    border = np.concatenate(
        [
            binary[0, :],
            binary[-1, :],
            binary[:, 0],
            binary[:, -1],
        ]
    )
    foreground_on_border = float(np.mean(border > 0))
    return foreground_on_border > 0.5


def binarize(gray: np.ndarray, params: PrepParams) -> tuple[np.ndarray, bool]:
    """Threshold to a 0/1 mask with the subject as 1.

    Returns the mask and whether polarity was inverted.
    """
    import cv2

    if params.threshold_mode == "adaptive":
        block = params.adaptive_block
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            params.adaptive_c,
        )
    else:
        # THRESH_BINARY_INV makes dark strokes foreground, the usual case.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = (binary > 0).astype(np.uint8)

    if params.invert is None:
        inverted = auto_polarity(mask)
    else:
        inverted = bool(params.invert)

    if inverted:
        mask = 1 - mask

    return mask, inverted


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    import cv2

    if min_area <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are not reachable from outside the picture.

    The flood has to start *outside* the picture, which is why the mask is
    padded. Flooding from the image's own ``(0, 0)`` -- which is what this did
    -- only reaches whatever that one corner happens to connect to: on a
    drawing with a rule across the top, everything below the rule stayed
    unreached and was therefore taken for enclosed. On a real portrait that
    turned the whole canvas into the subject and the board outline into a
    rectangle, which is exactly the bug that was reported.
    """
    import cv2
    import numpy as np

    height, width = mask.shape[:2]
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask

    flooded = padded.copy()
    # ``(0, 0)`` is now guaranteed to be outside the picture, so the fill can
    # travel along the padding and enter the background from every side.
    cv2.floodFill(flooded, None, (0, 0), 1)

    reached = flooded[1:-1, 1:-1]
    holes = ((mask == 0) & (reached == 0)).astype(np.uint8)
    return np.clip(mask + holes, 0, 1).astype(np.uint8)


def subject_mask(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    fill: bool = True,
) -> np.ndarray | None:
    """Which pixels belong to the artwork's *subject*, as a solid footprint.

    This is deliberately not the same thing as the ink mask that
    :func:`preprocess` produces. The ink mask marks **strokes**; this marks the
    filled shape they describe. Tracing an outline from strokes gave an outline
    that followed every hair line and produced spikes wherever strokes nearly
    touched -- the board edge was describing the *drawing*, not the *character*.

    Two sources, in order:

    1. **Alpha**, when it actually separates anything. A fully opaque alpha is
       not a cut-out, it is a colour image stored with four channels; treating
       it as a mask would hand back the whole rectangle.
    2. **Tone against the border.** The picture's own edge is taken to be
       background -- true of a scan on paper, a product shot on a backdrop,
       and a screenshot of an illustration. The background colour is the
       *median* of the border ring, so one dark corner cannot define it.

    With ``fill`` the enclosed background is then treated as subject. That is
    what turns a line drawing into a footprint: the strokes already enclose the
    character, so everything they enclose *is* the character.

    **Everything the picture contains is kept, not just the biggest piece.** A
    poster with a character, a mug, a bottle and a title is four subjects; the
    board should carry four shapes, and only shapes that sit inside a larger
    one are dropped (an eye inside a face is part of the face). Handing back
    only the largest piece threw the rest away, and welding them together first
    is what drew the connecting lines this function used to produce; see
    :func:`_adaptive_gap` for how the two cases are told apart.

    Returns ``uint8`` valued 0/1, or ``None`` when there is nothing usable.
    """
    import cv2
    import numpy as np

    from .work_scale import work_scale

    base = _base_subject_mask(image, alpha)
    if base is None or not base.any():
        return None

    original_height, original_width = base.shape[:2]
    if original_height < 3 or original_width < 3:
        return base

    # **The whole mask is computed at the working resolution, then scaled
    # back.** Every step below either grows with the picture (the closing
    # kernel, the frame-thickness walk) or is simply linear in its pixels, and
    # on a 3840 x 6476 drawing those steps together took 44 s -- of which one,
    # the frame scan, was 42 s on its own. None of them can see detail finer
    # than the 0.35 mm contour simplification the caller applies afterwards, so
    # the extra resolution bought nothing and cost everything.
    #
    # The result is upscaled back to the input's size rather than returned
    # small, because callers compare this mask against images they already
    # hold. `silhouette_outline` skips the outline entirely when the shapes
    # disagree, so returning a smaller mask would silently drop the subject's
    # boundary instead of failing loudly.
    scale = work_scale(original_width, original_height)
    if scale < 1.0:
        base = cv2.resize(
            base,
            (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    height, width = base.shape[:2]

    if fill:
        # A decorative border drawn around the artwork encloses the whole
        # frame, so flood-filling inside it would mark the entire picture as
        # subject -- which is exactly what happened on a real drawing. The
        # border is removed first, before anything can be trapped by it.
        #
        # The 3 px thickness test is a pixel measurement, so it follows the
        # picture down: a 3 px frame in the source is 1 px after the reduction,
        # and testing it against the original number would keep the frame and
        # lose the outline to a flood fill of the whole canvas.
        base = _remove_frame_lines(
            base, max_thickness=max(1, int(round(3 * scale)))
        )

    base = cv2.morphologyEx(
        base, cv2.MORPH_CLOSE, _ellipse(_adaptive_gap(base))
    )

    if fill:
        filled = _fill_enclosed_holes(base)
        # If the fill swallowed the picture, a closed structure was still
        # enclosing it. Falling back to the strokes is more honest than
        # returning a rectangle that looks like a successful outline.
        if filled.mean() > 0.95:
            filled = base
        base = filled

    subject = _outer_components(base)
    if subject is None or not subject.any():
        return None

    # Smooth the staircase the threshold left behind and drop the thin
    # filaments that survive filling -- strands of hair, the gaps between
    # fingers. Without this the board edge grows whiskers wherever two strokes
    # nearly meet, which is what the user saw on the hair and the hand.
    #
    # **Each piece is smoothed on its own, never the whole mask at once.** The
    # kernel is wide enough to reshape a silhouette, so closing the combined
    # mask bridges the background *between* separate things: measured on the
    # poster that exposed this, a 13 px close took the character-plus-mug from
    # two pieces of 289k and 35k pixels to a single 343k blob, and the board
    # outline then ran through the space between them. Smoothing one piece at a
    # time cannot do that -- there is nothing else in the picture to merge with.
    smooth = _odd(max(3, int(min(height, width) * 0.015)))
    subject = _smooth_each_component(subject, _ellipse(smooth))
    subject = _outer_components(subject)
    if subject is None:
        return None

    if subject.shape[:2] != (original_height, original_width):
        subject = cv2.resize(
            subject,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return subject


def _odd(value: int) -> int:
    """The nearest odd number at least as large as ``value``.

    Morphology kernels must be odd, or OpenCV measures them from a corner
    rather than from the centre.
    """
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _remove_frame_lines(
    mask: np.ndarray, max_thickness: int = 3
) -> np.ndarray:
    """Drop a decorative border drawn around the artwork.

    A frame is a straight line running almost the full width or height of the
    picture, close to its edge, and only a pixel or two thick. Removing it is
    necessary because the frame is closed: flood-filling the enclosed
    background through it marks the entire canvas as subject, which is exactly
    what happened on a real drawing.

    Thickness is measured *along the line itself* -- the length of the
    contiguous ink run crossing it at each position. Testing the neighbouring
    rows instead does not work: on a portrait the shoulders make the rows just
    beside the bottom frame line wide, so the frame looked like a solid band
    and was kept.

    ``max_thickness`` follows the resolution: it is a pixel measurement, so a
    reduced picture needs a proportionally smaller threshold.
    """
    import numpy as np

    height, width = mask.shape[:2]
    if height < 20 or width < 20:
        return mask

    result = mask.copy()
    edge_band_y = max(2, int(height * 0.2))
    edge_band_x = max(2, int(width * 0.2))

    row_ratio = mask.mean(axis=1)
    col_ratio = mask.mean(axis=0)

    # The candidate rows and columns are measured in one vectorised pass each.
    # Calling a per-position helper in a Python loop over every candidate is
    # what made this 42 s on a 3840 x 6476 mask: the scan is over a fifth of
    # the picture on each axis, and each call walked its column pixel by pixel.
    rows = [
        int(i)
        for i in _thin_candidates(row_ratio, edge_band_y, height)
        if _line_thickness(mask, int(i), axis=0, max_thickness=max_thickness)
        <= max_thickness
    ]
    cols = [
        int(j)
        for j in _thin_candidates(col_ratio, edge_band_x, width)
        if _line_thickness(mask, int(j), axis=1, max_thickness=max_thickness)
        <= max_thickness
    ]

    # A frame has sides on both axes. A single long line is far more likely to
    # be part of the drawing -- the top of a hat, the edge of a table -- so it
    # is left alone.
    if not (rows and cols):
        return result

    for i in rows:
        result[max(0, i - 1) : i + 2, :] = 0
    for j in cols:
        result[:, max(0, j - 1) : j + 2] = 0
    return result


def _thin_candidates(ratio, edge_band: int, extent: int):
    """Positions that are mostly ink and near an edge, cheaply.

    Split out from :func:`_remove_frame_lines` so the two axes read the same
    and so the pair ``ratio``/``extent`` cannot be mismatched at a call site.
    """
    import numpy as np

    candidates = np.where(ratio > 0.75)[0]
    return [
        int(i)
        for i in candidates
        if (i < edge_band or i > extent - edge_band)
    ]


def _line_thickness(
    mask: np.ndarray, index: int, axis: int, max_thickness: int = 3
) -> float:
    """Median thickness of the ink crossing position ``index``.

    ``axis=0`` measures vertical thickness of the row ``index``; ``axis=1``
    measures horizontal thickness of the column ``index``. Only positions where
    the line is actually present contribute, so a line that is partly covered
    by the subject is still measured on the part that is visible.

    The walk stops at ``max_thickness``. That is the only comparison the caller
    makes, so measuring further is wasted work -- and the wasted work was
    enormous: a column through a filled portrait is thousands of pixels of
    continuous ink, and walking all of it at every sampled position is what
    made the frame scan take 42 s on a 3840 x 6476 mask.
    """
    import numpy as np

    if axis == 0:
        profile = mask
        line = mask[index, :]
    else:
        profile = mask.T
        line = mask[:, index]

    thicknesses: list[int] = []
    limit = int(max_thickness)
    # Sample rather than walk every position: this only needs an order of
    # magnitude, and the loop is in Python.
    step = max(1, profile.shape[1] // 200)
    for position in range(0, profile.shape[1], step):
        if not line[position]:
            continue
        above = index
        while above > 0 and profile[above - 1, position] and index - above < limit:
            above -= 1
        below = index
        while (
            below < profile.shape[0] - 1
            and profile[below + 1, position]
            and below - index < limit
        ):
            below += 1
        thicknesses.append(below - above + 1)
        if len(thicknesses) >= 32:
            # Enough to take a median from; the rest of the row cannot change
            # the answer from "thin" to "thick" once most samples agree.
            break

    if not thicknesses:
        return 999.0
    return float(np.median(thicknesses))


def _base_subject_mask(
    image: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray | None:
    """The raw subject mask, before gap closing and filling."""
    import cv2
    import numpy as np

    if alpha is not None and alpha.size:
        opaque = float((alpha > 0).mean())
        # Strictly between "there is a background" and "there is no
        # background": a fully opaque alpha says nothing about the subject, and
        # a fully transparent one has nothing to trace.
        if 0.0 < opaque < 0.999:
            return (alpha > 0).astype(np.uint8)

    if image is None or image.size == 0:
        return None

    # Greyscale, as 8-bit integers. The float32 conversion this used to do
    # cost 992 ms on an 8000 px phone photo -- a quarter of the whole import --
    # and `cv2.absdiff` plus a comparison produces a byte-for-byte identical
    # mask in 196 ms. Arithmetic is done in float only for the two scalars that
    # come out of it (the background level and the spread), where the cost is
    # independent of picture size.
    if image.ndim == 2:
        grey = image
    elif image.shape[2] == 4:
        # Flatten onto white before desaturating, so a transparent background
        # reads as paper rather than as black.
        a = image[:, :, 3:4].astype(np.uint16)
        colour = image[:, :, :3].astype(np.uint16)
        flattened = ((colour * a + 255 * (255 - a)) // 255).astype(np.uint8)
        grey = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
    else:
        grey = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    height, width = grey.shape[:2]
    if height < 3 or width < 3:
        return None

    band = max(1, min(height, width) // 50)
    border = np.concatenate(
        [
            grey[:band, :].ravel(),
            grey[-band:, :].ravel(),
            grey[:, :band].ravel(),
            grey[:, -band:].ravel(),
        ]
    )
    background = float(np.median(border))

    spread = float(int(grey.max()) - int(grey.min()))
    if spread <= 1.0:
        return None
    # A fraction of the range rather than a fixed number of levels, so a pale
    # scan and a dark photo behave the same.
    difference = cv2.absdiff(grey, np.full_like(grey, int(round(background))))
    return (difference.astype(np.float32) > max(12.0, spread * 0.12)).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Keep only the biggest blob, which is the subject of a standee.

    Specks elsewhere in the frame would otherwise become part of the outline,
    and a board traced from scattered noise is not a board.
    """
    import cv2
    import numpy as np

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    if stats[biggest, cv2.CC_STAT_AREA] <= 0:
        return None
    return (labels == biggest).astype(np.uint8)


#: Closing kernel used when the artwork's outlines already close on their own,
#: as a fraction of the short side. Small on purpose: at this size the kernel
#: only tidies pixel-level gaps, and the separate things in the picture -- a
#: character, a mug, a bottle, a title -- stay separate.
MULTI_ELEMENT_GAP_RATIO = 0.006

#: Closing kernel used when they do not, as a fraction of the short side. Big
#: on purpose: pencil lines at the crown of the head of a real drawing were
#: separated by roughly 4% of the picture, and anything smaller let the fill
#: leak straight out through them.
BRIDGE_GAP_RATIO = 0.045

#: How much of the wide-kernel result the narrow one has to reproduce before
#: the picture is taken to be self-closing. Measured: pictures whose outlines
#: close score 0.94-0.99, hand-drawn line art with real gaps scores 0.58-0.72.
#: The gap between those groups is wide enough that the exact cut does not
#: matter.
SELF_CLOSING_COVERAGE = 0.85


def _ellipse(size: int) -> "np.ndarray":
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _adaptive_gap(mask: np.ndarray) -> int:
    """How wide a closing kernel this picture actually needs.

    A single fixed kernel cannot serve both pictures this tool has to handle,
    and using the wide one everywhere is what produced the defect this
    replaces: on a poster holding several separate things, a kernel sized to
    bridge a drawing's pencil gaps also bridges the *background* between the
    character and the mug beside it. Everything became one blob, the outline
    ran straight across the empty space, and the user saw connecting lines
    that are not in the picture.

    Both cases are self-describing, so nothing has to be asked of the user.
    Close the mask with a narrow kernel and with a wide one, fill both, and
    compare: if the outlines already close, the narrow kernel reproduces
    almost all of the wide result -- 94% or better on the pictures measured,
    because the wide kernel has nothing left to bridge. If they do not close,
    the narrow fill leaks out and the two results diverge sharply (58-72%),
    and the wide kernel is needed.

    Returns an odd kernel width in pixels.
    """
    import cv2
    import numpy as np

    height, width = mask.shape[:2]
    short = min(height, width)

    narrow = _odd(max(3, int(short * MULTI_ELEMENT_GAP_RATIO)))
    wide = _odd(max(3, int(short * BRIDGE_GAP_RATIO)))
    if narrow == wide:
        return wide

    def filled(kernel: int) -> "np.ndarray":
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(kernel))
        return _fill_enclosed_holes(closed) > 0

    narrow_fill = filled(narrow)
    wide_fill = filled(wide)
    wide_area = int(wide_fill.sum())
    if wide_area == 0:
        return wide

    covered = int(np.logical_and(narrow_fill, wide_fill).sum()) / wide_area
    return narrow if covered >= SELF_CLOSING_COVERAGE else wide


def _smooth_each_component(mask: np.ndarray, kernel: "np.ndarray") -> np.ndarray:
    """Morphological open-then-close, applied to every piece separately.

    Housekeeping on the *shape* of each piece: it takes off the single-pixel
    staircase a threshold leaves behind and closes pinholes inside a silhouette.

    Applied to the whole mask instead, the closing half does something else as
    well: it bridges any two pieces closer together than the kernel, which is
    exactly how the separate things in a picture got welded back together after
    being carefully kept apart. There is no threshold that avoids this -- the
    kernel has to be wide enough to reshape a silhouette, and pieces standing a
    silhouette-width apart are what "separate" looks like.
    """
    import cv2
    import numpy as np

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask

    out = np.zeros(mask.shape[:2], np.uint8)
    for index in range(1, count):
        piece = (labels == index).astype(np.uint8)
        piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, kernel)
        piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, kernel)
        out |= piece
    return out


def _outer_components(
    mask: np.ndarray,
    *,
    min_abs_px: int = 16,
    min_fraction_of_largest: float = 0.002,
) -> np.ndarray | None:
    """Every shape in the mask that is not sitting inside another one.

    The counterpart to :func:`_largest_component`. Keeping only the biggest
    dropped a poster's mug, bottle and title; keeping *everything* instead
    would draw an outline around each eye and each letter of a speech bubble,
    because those are separate ink too. What distinguishes them is nesting:
    real content sits beside the subject, decoration sits within it.

    Nesting is decided by filling each piece's own holes and asking whether the
    result lies inside a bigger piece's filled area -- not by comparing
    bounding boxes, which cannot tell a mug beside a character from a mug in
    front of one.

    Specks are still dropped, by absolute area and relative to the largest
    piece, so scanner noise does not become a board.
    """
    import cv2
    import numpy as np

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None

    order = sorted(
        range(1, count),
        key=lambda i: stats[i, cv2.CC_STAT_AREA],
        reverse=True,
    )
    largest = stats[order[0], cv2.CC_STAT_AREA]
    floor = max(min_abs_px, int(largest * min_fraction_of_largest))

    usable = [i for i in order if stats[i, cv2.CC_STAT_AREA] >= floor]
    if not usable:
        return None

    # Filling each piece's own holes is what makes containment meaningful: the
    # eye strokes alone are a thin ring, and a thin ring is not "inside" the
    # face until it is filled to the area it visually occupies.
    footprints = {
        i: _fill_enclosed_holes((labels == i).astype(np.uint8)) > 0
        for i in usable
    }
    areas = {i: int(footprints[i].sum()) for i in usable}

    keep: list[int] = []
    for position, i in enumerate(usable):
        x_i, y_i, w_i, h_i, _ = stats[i]
        enclosed = False
        for j in usable[:position]:          # only larger pieces can contain it
            x_j, y_j, w_j, h_j, _ = stats[j]
            if (
                x_i < x_j
                or y_i < y_j
                or x_i + w_i > x_j + w_j
                or y_i + h_i > y_j + h_j
            ):
                continue                       # bbox rules containment out
            overlap = int(np.logical_and(footprints[i], footprints[j]).sum())
            if overlap / max(areas[i], 1) > 0.95:
                enclosed = True
                break
        if not enclosed:
            keep.append(i)

    if not keep:
        return None

    out = np.zeros(mask.shape[:2], np.uint8)
    for i in keep:
        out |= (labels == i).astype(np.uint8)
    return out


def preprocess(
    image: str | Path | np.ndarray,
    params: PrepParams | None = None,
) -> PrepResult:
    """Run the full chain and return a mask ready for contour extraction.

    Accepts a path or an already-loaded grayscale array.
    """
    import cv2

    params = params or PrepParams()
    steps: list[str] = []
    alpha: np.ndarray | None = None

    if isinstance(image, (str, Path)):
        gray, alpha = load_gray(image, params.use_alpha)
        steps.append("load")
    else:
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if alpha is not None:
        mask = (alpha > 127).astype(np.uint8)
        inverted = False
        steps.append("alpha")
    else:
        if params.blur_radius > 0:
            kernel = params.blur_radius * 2 + 1
            gray = cv2.medianBlur(gray, kernel)
            steps.append(f"median{params.blur_radius}")

        mask, inverted = binarize(gray, params)
        steps.append(f"threshold:{params.threshold_mode}")
        if inverted:
            steps.append("inverted")

    if params.morph_open > 0:
        size = params.morph_open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        steps.append(f"open{size}")

    if params.morph_close > 0:
        size = params.morph_close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        steps.append(f"close{size}")

    if params.min_component_area > 1:
        before = int(mask.sum())
        mask = _remove_small_components(mask, params.min_component_area)
        after = int(mask.sum())
        if after != before:
            steps.append(f"drop<{params.min_component_area}px")

    if params.fill_holes:
        mask = _fill_enclosed_holes(mask)
        steps.append("fill-holes")

    mask = (mask > 0).astype(np.uint8)
    total = mask.size
    ratio = float(mask.sum()) / total if total else 0.0

    if mask.sum() == 0:
        raise ValueError("预处理后没有任何前景像素，请检查阈值或反转设置")

    return PrepResult(
        mask=mask,
        foreground_ratio=ratio,
        inverted=inverted,
        source_size=(int(gray.shape[1]), int(gray.shape[0])),
        steps=steps,
    )


def mask_to_preview(mask: np.ndarray) -> np.ndarray:
    """Render a mask as a black-on-white 8-bit image for saving or display."""
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def mask_to_rgb_preview(mask: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Render a mask as an RGB image ready for display in a GUI canvas.

    Returns BGR-ordered ``uint8`` data, matching what Qt's ``Format_BGR888``
    expects, so the caller can hand it straight to a ``QImage``.
    """
    import cv2

    gray = mask_to_preview(mask)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        rgb = cv2.resize(
            rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
    return rgb


def make_test_image(
    width: int = 400,
    height: int = 400,
    shape: str = "blob",
    with_hole: bool = False,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise a test silhouette, so the pipeline can be tested without assets.

    ``shape`` is one of ``"blob"``, ``"star"``, ``"ring"`` or ``"letter"``.
    """
    import cv2

    canvas = np.full((height, width), 255, np.uint8)
    center = (width // 2, height // 2)

    if shape == "star":
        import math

        points = []
        for i in range(10):
            radius = 150 if i % 2 == 0 else 58
            angle = math.pi / 2 + i * math.pi / 5
            points.append(
                [int(center[0] + radius * math.cos(angle)), int(center[1] - radius * math.sin(angle))]
            )
        cv2.fillPoly(canvas, [np.array(points, np.int32)], 0)
    elif shape == "ring":
        cv2.circle(canvas, center, 150, 0, -1)
        cv2.circle(canvas, center, 90, 255, -1)
    elif shape == "letter":
        cv2.putText(canvas, "P", (90, 310), cv2.FONT_HERSHEY_SIMPLEX, 9.0, 0, 40)
    else:
        # An irregular closed blob, comfortably inside the canvas.
        blob = np.array(
            [
                [center[0] - 140, center[1] - 40],
                [center[0] - 90, center[1] - 140],
                [center[0] + 40, center[1] - 150],
                [center[0] + 145, center[1] - 60],
                [center[0] + 120, center[1] + 90],
                [center[0] + 20, center[1] + 150],
                [center[0] - 120, center[1] + 110],
            ],
            np.int32,
        )
        cv2.fillPoly(canvas, [blob], 0)
        if with_hole:
            cv2.circle(canvas, (center[0] - 20, center[1]), 45, 255, -1)

    if noise > 0:
        rng = np.random.default_rng(seed)
        speckles = rng.random(canvas.shape) < noise
        canvas[speckles] = 0
        noise_field = rng.normal(0, noise * 60, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise_field, 0, 255).astype(np.uint8)

    return canvas
