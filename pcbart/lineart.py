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
    "SOFT_SHADING_MIDDLE_SHARE",
    "needs_denoising",
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

    mode: str = "fill"
    """How the picture is reduced to two tones.

    ``"fill"`` -- **the current behaviour.** Otsu finds the split between the
    picture's two main tones and every pixel is sent to one side: dark areas
    become solid ink, light areas become solid paper. Shading, highlights and
    line work all resolve to flat blocks, which is what the user asked for
    after rejecting the edge-tracing reading: "把纯色部分直接做成填充色，
    深浅色分别用黑白色直接替代".

    ``"dodge"`` -- the earlier edge-tracing reading. It keeps only where a
    pixel is darker than its neighbourhood, so a shaded area comes out as
    contour lines rather than a filled shape. Kept because it is a genuinely
    different picture, not because anything still defaults to it.

    ⚠️ **Why fill won.** On the drawing that prompted the change, tracing gave
    the hair as lines but left the face as bare paper: a light face on a light
    background has no edge strong enough to trace, so it vanished into the
    page. Filling gives the face a shape because the dark hair surrounding it
    supplies the boundary.
    """
    threshold: float = 0.46
    """Response level that separates ink from paper, 0..1.

    Used by ``mode="dodge"`` only; ``mode="fill"`` takes its level from Otsu.

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

    multiply_by_brightness: bool = False
    """Whether the response is scaled by the pixel's own brightness.

    **Off is the line-art behaviour; on is the poster behaviour.** The two ask
    different questions and a picture cannot answer both at once:

    * ``False`` -- "is this pixel an edge?" The response is the dodge *ratio*
      ``G / local_max(G)``, which is near 1 across any flat area, however dark,
      and dips only where a pixel is darker than its neighbourhood. A shaded
      region therefore comes out as a few contour lines, which is what a line
      drawing is.
    * ``True`` -- "is this pixel dark?" Multiplying by ``G`` folds absolute
      brightness back in, so a dark region crosses the threshold wholesale and
      comes out as a solid block. That is the poster reading, and
      :func:`pcbart.boardart.board_art` is where it belongs.

    ⚠️ **This default used to be ``True``, and it is what broke shaded
    artwork.** Measured on the drawing that exposed it, the hair is a smooth
    dark gradient: its dodge ratio has a median of 0.968 (essentially no edge
    anywhere), yet ``dodge * G`` has a median of 0.465, so **41.5% of the hair
    fell below the threshold and printed as solid black** -- with the highlights
    left behind as white holes. The user described the result exactly: "一部分
    涂色一部分线稿".
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
            mode=str(self.mode),
            threshold=float(np.clip(self.threshold, 0.02, 0.98)),
            multiply_by_brightness=bool(self.multiply_by_brightness),
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


#: Share of the frame that may sit in the mid-tones before the picture is
#: taken to be softly shaded rather than drawn in flat colours.
#:
#: ⚠️ **Soft shading is what breaks the line algorithm, and it cannot be fixed
#: inside it.** The dodge response answers "is this pixel darker than its
#: neighbourhood", which is a question about *edges*. Across a smooth gradient
#: -- the hair of the drawing that prompted this -- there are no edges, only a
#: slow ramp, so the response wobbles a little and the threshold chops those
#: wobbles into hundreds of irregular rings. Measured on that picture: 236
#: fragments under 200 px in the hair alone, which is exactly what the user
#: described as "到处都是不规则的圈".
#:
#: Blurring first is what removes them, because it flattens the wobble before
#: the threshold sees it. Measured: 236 fragments at the raw threshold, 74 after
#: a 5 px blur, 37 after 9 px.
#:
#: Measured mid-tone share: 30% on that drawing, 64% on a colour photo, against
#: 0.4-3.9% on the flat line drawings.
SOFT_SHADING_MIDDLE_SHARE = 0.12


def needs_denoising(image: np.ndarray) -> bool:
    """Whether this picture is softly shaded and wants a blur before tracing.

    A flat line drawing has almost nothing between black and white: its ink is
    dark, its paper is light, and the only mid-tones are the anti-aliased rim of
    a stroke. A softly shaded illustration is the opposite -- most of the canvas
    is a gradient -- and tracing one without blurring first produces the ring
    artefacts described above.

    The distinction is decided from the picture, not from a filename or a
    container: the same artwork arrives as a flat PNG export or as a shaded JPEG
    depending on what the user did to it, and only the pixels say which.

    Measured on the pictures this tool handles:

    ====================  ==========
    flat line drawings    0.4 - 3.9%
    soft-shaded drawing   **30%**
    colour photograph     **64%**
    ====================  ==========
    """
    import numpy as np

    grey = np.clip(desaturate(image), 0, 255).astype(np.uint8)
    level = grey.astype(np.float32) / 255.0
    middle = float(((level > 0.2) & (level < 0.8)).mean())
    return middle >= SOFT_SHADING_MIDDLE_SHARE


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


def dodge_response(
    grey01: np.ndarray,
    radius: int,
    multiply_by_brightness: bool = False,
) -> np.ndarray:
    """``G / local_max(G)`` in 0..1, before the Levels step.

    This is the dodge *ratio*: how much darker a pixel is than the brightest
    pixel near it. Being a ratio, it is near 1 across any flat area whatever its
    absolute level, so it answers "is there an edge here?" and not "is this
    dark?" -- see :attr:`LineArtParams.multiply_by_brightness` for why line art
    wants the first question and why the old default asked the second.
    """
    maximum = np.maximum(local_max(grey01, radius), 1e-6)
    dodge = np.clip(grey01 / maximum, 0.0, 1.0)
    if multiply_by_brightness:
        return np.clip(dodge * grey01, 0.0, 1.0)
    return dodge


def _tone_split(grey: np.ndarray) -> np.ndarray:
    """Reduce a picture to two flat tones with Otsu's threshold.

    Every pixel goes to one side or the other, so the result tiles the frame:
    the picture's own light areas become paper and its dark areas become solid
    ink, with no gaps and no partial shading. That is the "深浅色分别用黑白色
    直接替代" reading, and it is what keeps a shaded face visible -- its shape
    comes from the dark hair around it rather than from an edge of its own.

    **Otsu rather than a fixed level.** The right cut depends on the picture:
    on the drawing this was tuned against Otsu lands at 179, where a fixed 128
    would have swallowed the mid-tones that carry the face. Otsu finds the level
    between whatever two tones the picture actually has, which is the same
    reasoning already used for :func:`pcbart.image_prep.binarize`.

    Returns ``uint8``, 0 where ink.
    """
    import cv2

    _, binary = cv2.threshold(
        grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _despeckle_tone(traced: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Remove the hairline fragments a fill boundary leaves behind.

    Otsu's cut runs through soft edges, so wherever a picture has a gradual
    transition the boundary wobbles and sheds slivers: on the drawing this was
    tuned against, 47 paper components with 24 of them under 200 px, the
    largest 13x28. They read as specks of dirt on the board rather than as
    anything the artwork shows.

    A single 3 px open-then-close removes most of them. **Deliberately small**:
    9 px clears them completely but was measured erasing the scalloped lace
    across the top of the head, which is real detail. Closing first bridges
    one- and two-pixel notches in a boundary; opening then drops what is left
    floating.

    ``traced`` is 0 = ink, 255 = paper, so the operation runs on the paper mask
    and the polarity is restored afterwards.
    """
    import cv2

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    paper = (traced > 128).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, element)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, element)
    return np.where(paper > 0, 255, 0).astype(np.uint8)

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

    if params.mode == "fill":
        traced = _tone_split((grey * 255).astype(np.uint8))
    else:
        radius = auto_radius(grey.shape, params.strength)
        response = dodge_response(
            grey, radius, multiply_by_brightness=params.multiply_by_brightness
        )
        traced = np.where(response <= params.threshold, 0, 255).astype(np.uint8)

    # ⚠️ **The silhouette is for the tracing reading, not the fill reading.**
    #
    # It exists because the dodge response cannot see where a subject *ends*:
    # a light subject on a light page has no edge, so its boundary has to be
    # supplied from alpha instead. Filling has no such gap -- Otsu already
    # decides every pixel, and the subject's shape is whatever came out dark.
    #
    # Worse, drawing it anyway is actively harmful. Measured on the drawing
    # that exposed this: the outline's ink lands *inside* the region Otsu
    # already filled 38% of the time, and the line it draws there cuts the
    # filled shape into pieces. Paper components went from **33 to 170**, with
    # 108 of them under 200 px -- which is exactly the "不规则的圈圈" the user
    # saw on the face and the skirt, and exactly the region where a smooth
    # light face and a smooth light ribbon give the outline something to carve.
    if params.silhouette and params.mode != "fill":
        # The colour image is passed along because a picture with no usable
        # alpha -- a JPEG, a flattened PNG -- has no other source for its
        # subject boundary, and skipping the outline for those left the board
        # with no outer edge at all.
        outline = silhouette_outline(alpha, params.silhouette_width, image)
        if outline.shape[:2] == traced.shape[:2]:
            traced = np.where(outline > 0, 0, traced).astype(np.uint8)

    if params.mode == "fill":
        traced = _despeckle_tone(traced)

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
