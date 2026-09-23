"""Photo and illustration to two-tone line art.

The core is the darkroom-style recipe Photoshop users apply by hand::

    去色 -> 复制图层 -> 反相 -> 颜色减淡 -> 最小值滤镜 -> 盖印 -> 色阶

Those steps collapse to a closed form. With ``G`` the greyscale image in 0..1,
a Minimum filter over the inverted copy returns ``1 - local_max(G)``, and
Color Dodge is ``base / (1 - blend)``, so the first four steps compute

    dodge = G / local_max(G)

which is 1 wherever a pixel is the brightest in its neighbourhood and smaller
where it is darker than its surroundings -- the lines.

That form alone has a flaw: it only compares a pixel with its neighbours, so a
large uniformly dark area normalises to *white*. A solid black collar comes out
empty whenever the filter radius is smaller than the collar. Multiplying by the
original grey restores the absolute tone and fixes it::

    response = dodge * G

A pixel now has to be both darker than its surroundings *and* genuinely dark to
become a line, which is what a line actually is. It also thickens faint strands
(thin hair lines) into solid strokes, so they survive silkscreen printing.

The result is a continuous response; :func:`line_art` maps it to pure black and
white with a Levels window.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "LineArtParams",
    "RADIUS_RATIO",
    "MIN_RADIUS",
    "MAX_RADIUS",
    "auto_radius",
    "desaturate",
    "local_max",
    "dodge_response",
    "line_art",
    "line_art_preview",
]

RADIUS_RATIO = 0.0105
"""Minimum-filter radius per pixel of the image's short side.

Radius sets stroke weight, so it has to follow image resolution: a fixed radius
gives hair-thin lines on a large image and fat ones on a small image. Measured
against a hand-made Photoshop reference, the radius that reproduces its stroke
weight is about 1.05% of the short side (10 px on a 960 px image, 14 px on a
1440 px one).
"""

MIN_RADIUS = 2
MAX_RADIUS = 80
"""Clamp for :func:`auto_radius`. Below 2 no line forms; above 80 the filter
costs more than it can possibly help on any realistic artwork."""


@dataclass
class LineArtParams:
    """User-facing controls for :func:`line_art`."""

    strength: float = 1.0
    """Multiplier on the automatic radius, i.e. on stroke weight.

    1.0 is the calibrated default. Larger values give heavier, more printable
    lines; smaller values keep fine detail but risk washing out at print time.
    """

    threshold: float = 0.46
    """Response level that separates ink from paper, 0..1.

    Lower values put more of the image in ink; higher values keep only the
    strongest lines. Calibrated against a hand-made Photoshop reference of the
    same source: at 0.46 the ink coverage and stroke width both match it to
    within 0.06 percentage points.
    """

    denoise: bool = False
    """Deprecated: denoising is no longer part of line extraction.

    Kept so existing callers keep working. It now does nothing; run
    :func:`pcbart.denoise.denoise_image` as a separate step instead. Folding a
    blur into line extraction meant any change to the denoise setting changed
    the artwork, and the two concerns could not be judged separately.
    """

    denoise_kernel: int = 5
    """Deprecated, unused. See :attr:`denoise`."""

    despeckle_area: int = 0
    """Deprecated: cleanup is a separate step too.

    Use :func:`pcbart.denoise.clean_binary` on the line-art result. Removing
    specks inside line extraction hid ink that the line algorithm had
    deliberately produced, which made the two impossible to tune apart.
    """

    silhouette: bool = True
    """Add the subject's outer outline, taken from transparency.

    Needed whenever the artwork has an alpha channel and its outer edge is
    low-contrast: the dodge response cannot see where the subject ends, so a
    light subject on a light background loses its boundary completely. That
    boundary is what a board outline is cut from, so it matters even though it
    looks like a minor visual detail.
    """

    silhouette_width: int = 5
    """Line width for the outer outline, in pixels at the traced resolution.

    Calibrated to match the ~3.3 px stroke of the reference at full resolution.
    Forced odd; clamped to 1..31.
    """

    def normalised(self) -> "LineArtParams":
        """Clamp to usable ranges."""
        width = int(self.silhouette_width)
        if width % 2 == 0:
            width -= 1
        return LineArtParams(
            strength=float(np.clip(self.strength, 0.2, 4.0)),
            threshold=float(np.clip(self.threshold, 0.02, 0.98)),
            silhouette=bool(self.silhouette),
            silhouette_width=int(np.clip(width, 1, 31)),
            denoise=False,
            denoise_kernel=5,
            despeckle_area=0,
        )


def auto_radius(shape: tuple[int, ...], strength: float = 1.0) -> int:
    """Filter radius for an image of this shape, in pixels.

    ``shape`` is ``(height, width)`` or ``(height, width, channels)``.
    """
    short_side = min(int(shape[0]), int(shape[1]))
    radius = round(short_side * RADIUS_RATIO * float(strength))
    return int(np.clip(radius, MIN_RADIUS, MAX_RADIUS))


def _rescaled_params(params: "LineArtParams | None", scale: float):
    """Carry pixel-measured settings across a change of resolution.

    ``silhouette_width`` is a count of pixels of the picture the user loaded.
    Reducing first and applying it unchanged would silently ink a thinner
    outline than asked for -- the same number means a different amount of
    material at a different resolution. The threshold and strength are ratios
    and carry over as they are.
    """
    if scale >= 1.0:
        return params

    from dataclasses import replace

    from .work_scale import scaled_param

    base = params or LineArtParams()
    return replace(
        base,
        silhouette_width=max(1, int(round(scaled_param(base.silhouette_width, scale, minimum=1.0)))),
    )


def desaturate(bgr: np.ndarray) -> np.ndarray:
    """Photoshop's 去色: HSL desaturation, ``(max + min) / 2`` per pixel."""
    if bgr.ndim == 2:
        return bgr.astype(np.float32)
    # Reductions in uint8, one conversion at the end. Doing the reduction in
    # float32 cost 2137 ms on an 8000 px photo -- most of the entire trace --
    # and uint8 cannot overflow here: max and min are each in 0..255, so their
    # sum fits in uint16 before the halving.
    maximum = bgr.max(axis=2)
    minimum = bgr.min(axis=2)
    return (maximum.astype(np.uint16) + minimum) / 2.0


def local_max(grey01: np.ndarray, radius: int) -> np.ndarray:
    """Maximum grey value within a ``(2*radius+1)`` square around each pixel.

    A rectangular kernel is separable, so OpenCV's morphology is fast even at
    large radii, and it matches the square neighbourhood Photoshop's Minimum
    filter uses by default.
    """
    radius = max(1, int(radius))
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    inverted = (1.0 - grey01).astype(np.float32)
    return 1.0 - cv2.erode(inverted, kernel)


def dodge_response(grey01: np.ndarray, radius: int) -> np.ndarray:
    """``(G / local_max(G)) * G`` in 0..1, before the Levels step."""
    maximum = np.maximum(local_max(grey01, radius), 1e-6)
    dodge = np.clip(grey01 / maximum, 0.0, 1.0)
    return np.clip(dodge * grey01, 0.0, 1.0)


def silhouette_outline(
    alpha: np.ndarray | None,
    width: int = 5,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """The subject's outer boundary as an ink mask.

    Why this is needed at all: the dodge response measures "darker than my
    neighbourhood", which cannot express "the subject ends here". Measured on a
    real drawing, the response along the subject silhouette averages 0.98 -- and
    dropping the threshold as far as 0.30 still inks 0.0% of it, because the
    signal is simply not there. For a light-coloured subject on a light
    background, such as hair against white paper, the outer edge therefore
    vanishes entirely.

    **Where the boundary comes from.** Transparency carries it exactly, so it
    is read from ``alpha`` whenever the alpha actually distinguishes a subject
    from a background. Plenty of pictures have no alpha at all -- a JPEG, a
    screenshot, a flattened PNG -- and those used to get no silhouette
    whatsoever, which is the gap this closes. For them the subject is inferred
    from tone against the border colour instead: whatever looks unlike the
    picture's own edges is the subject.

    The line is grown inwards, never outwards, so the traced shape stays inside
    the original silhouette and a board outline cut from it does not exceed the
    artwork.

    Returns ``uint8``, 0 where ink. An unusable input yields an empty mask.
    """
    import numpy as np

    subject = _subject_mask(alpha, image)
    if subject is None or not subject.any():
        return np.zeros(
            subject.shape if subject is not None else (0, 0), np.uint8
        )

    edge = cv2.morphologyEx(subject, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if width > 1:
        size = 2 * (int(width) // 2) + 1
        edge = cv2.dilate(
            edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    # Keep it inside the subject so the outline does not widen the shape.
    return (edge & subject).astype(np.uint8)


def _subject_mask(
    alpha: np.ndarray | None,
    image: np.ndarray | None,
) -> np.ndarray | None:
    """Which pixels belong to the subject rather than to the background.

    Delegates to :func:`pcbart.image_prep.subject_mask`, which is the single
    implementation: the board outline traces the very same footprint, and two
    copies would drift until the outline and the drawn edge disagreed.

    ``fill=False`` here. This function answers "where is the boundary", and the
    boundary of the strokes-as-drawn is what the line art wants to ink; the
    board outline asks the different question "what solid shape do they
    enclose", which is what filling supplies.
    """
    from .image_prep import subject_mask

    if alpha is not None and alpha.size:
        # Alpha alone answers the question *when it actually describes a
        # cut-out*, so a caller holding only the transparency does not have to
        # supply the colour image. A fully opaque alpha describes nothing, and
        # is left to the tone path below rather than being taken as a mask.
        opaque = float((alpha > 0).mean())
        if 0.0 < opaque < 0.999:
            return subject_mask(np.zeros_like(alpha), alpha, fill=False)
    if image is None:
        return None
    return subject_mask(image, alpha, fill=False)


def _composite_on_white(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Flatten transparency onto white, which is what silkscreen prints on."""
    if alpha is None:
        return bgr
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    white = np.full_like(bgr, 255.0, dtype=np.float32)
    return (bgr.astype(np.float32) * a + white * (1.0 - a)).astype(np.uint8)


def line_art(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    params: LineArtParams | None = None,
) -> np.ndarray:
    """Convert an image to two-tone line art.

    Returns a strictly binary ``uint8`` image, 0 for ink and 255 for paper --
    the polarity a board layer wants, where dark means "put material here".

    Output is binary rather than greyscale on purpose. Silkscreen cannot render
    a grey pixel, so a soft response would print as dirt; the response is
    therefore thresholded rather than stretched.

    ``image`` may be greyscale, BGR or BGRA; ``alpha`` can be passed separately
    for images already split into colour and transparency.

    The caller is responsible for having reduced ``image`` to the working
    resolution; use :func:`pcbart.work_scale.prepare_working_image` to do it and
    to carry any pixel-measured parameters across.
    """
    params = (params or LineArtParams()).normalised()

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]
    image = _composite_on_white(image, alpha)

    grey = desaturate(image) / 255.0
    radius = auto_radius(grey.shape, params.strength)
    response = dodge_response(grey, radius)

    traced = np.where(response <= params.threshold, 0, 255).astype(np.uint8)

    if params.silhouette:
        # The outer boundary comes from transparency when there is any, and is
        # otherwise inferred from tone; see :func:`silhouette_outline` for why
        # the dodge response can never supply it.
        #
        # The colour image is passed along because a picture with no usable
        # alpha -- a JPEG, a flattened PNG -- has no other source for its
        # subject boundary, and skipping the outline for those left the board
        # with no outer edge at all.
        outline = silhouette_outline(alpha, params.silhouette_width, image)
        if outline.shape[:2] == traced.shape[:2]:
            traced = np.where(outline > 0, 0, traced).astype(np.uint8)

    return traced


def line_art_preview(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    params: LineArtParams | None = None,
    max_side: int = 900,
) -> np.ndarray:
    """Line art for an interactive preview, decimated *after* tracing.

    Tracing on a downscaled copy is cheap but wrong: resampling removes detail
    before the line algorithm runs. Measured on a 960x1619 drawing, tracing a
    760 px copy lost 23% of the connected strokes and 18% of the skeletal
    length, and only 87% of the full-resolution ink was reproduced -- thin
    strands suffered most. Sampling with ``INTER_NEAREST`` instead was worse
    still (72% coverage), because dropping pixels at random breaks lines.

    The trace itself is bounded to :data:`~pcbart.work_scale.MAX_WORK_LONG_SIDE`
    before it runs, and the result is decimated from there. That upper bound is
    the point where a reduced trace stops differing visibly from a full one:
    measured against a 3840 px trace and both shown at preview size, tracing at
    2000 px agrees to IoU 0.945 and carries 101.7% of the ink. Tracing at 760 px
    -- which is where the earlier "loses 23% of the strokes" finding came from
    -- agrees only to 0.755, so the bound is deliberately well above the display
    size rather than at it.

    The returned image is a **display thumbnail**: resampling can leave grey
    values, because averaging is the only way to keep a single-pixel stroke
    visible at reduced size. That is fine on screen and is never what gets
    printed -- call :func:`line_art` for the strictly two-tone output.
    """
    from .work_scale import prepare_working_image, work_scale

    height, width = image.shape[:2]

    # One reduction up front, bounded by the working budget, and everything
    # after it works on that copy.
    trace_scale = work_scale(width, height)
    if trace_scale < 1.0:
        image, alpha, trace_scale = prepare_working_image(image, alpha)
        params = _rescaled_params(params, trace_scale)

    height, width = image.shape[:2]
    shortest = min(height, width)

    full = line_art(image, alpha, params)
    if shortest <= max_side:
        return full

    scale = max_side / shortest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # INTER_AREA for the display thumbnail. Decimating a strictly two-tone
    # image always loses single-pixel strokes, but averaging keeps them visible
    # as lighter marks whereas nearest-neighbour drops them outright: measured
    # coverage of the full-resolution ink was 87% with averaging against 72%
    # without. The grey this introduces exists only on screen -- the printed
    # output is produced by line_art() and stays strictly black and white.
    return cv2.resize(full, size, interpolation=cv2.INTER_AREA)
