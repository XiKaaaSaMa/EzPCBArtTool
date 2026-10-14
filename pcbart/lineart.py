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
    "SubjectMasks",
    "RADIUS_RATIO",
    "MIN_RADIUS",
    "MAX_RADIUS",
    "READING_REFERENCE_LONG_SIDE",
    "RIDGE_PREVIEW_TRACE_SIDE",
    "auto_radius",
    "desaturate",
    "local_max",
    "dodge_response",
    "xdog_response",
    "line_art",
    "line_art_preview",
    "outer_outline",
    "silhouette_outline",
    "cut_out_mask",
    "subject_masks",
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

    ``"fill"`` -- **the default.** Otsu finds the split between the picture's
    two main tones and every pixel is sent to one side: dark areas become solid
    ink, light areas become solid paper. Shading, highlights and line work all
    resolve to flat blocks, which is what the user asked for after rejecting
    the edge-tracing reading: "把纯色部分直接做成填充色，深浅色分别用黑白色直接
    替代".

    ``"outline"`` -- **the reading for a shaded illustration.** A contour pass
    finds every tonal boundary and the fill pass still supplies the solid
    darks, so the result is a drawing with lines *and* blocks rather than
    blocks alone. Added in answer to the user's own comparison: on library
    illustrations the fill reading loses eyebrows, chain links and fabric
    weaves into one silhouette, and this reading keeps them.

    ``"dodge"`` -- the edge-tracing reading: keep a pixel when it is darker
    than its neighbourhood, as a *binary* threshold on a box-filter ratio. It
    never looks at absolute brightness, which is exactly what a **dark**
    illustration needs -- measured on the dark character drawing that put this
    reading back on the panel, ``fill`` flooded the dark costume and hair into
    one 15.2% black mass with the costume lines, the leg rings and the gun lost
    inside it, while ``dodge`` returned 11.5% as pure line work with all of
    them intact. It is not a universal replacement: on a pale flat drawing the
    same ratio response turns texture noise into a mesh of hairlines (2868 ink
    blobs against 622 on the dark drawing), which is a job for 降噪 rather than
    for a different reading. See :func:`dodge_response`.

    ⚠️ **The old note here was wrong.** It claimed dodge left a pale face as
    bare paper, and that is what kept this reading out of the UI. Measured
    again on the palest library drawing (opaque-region brightness 190.4 of
    255), the face and hair contours come out complete; the earlier observation
    most likely predates compositing the cut-out onto white. Do not delete the
    dodge branch on the strength of the old note.

    ⚠️ **Why fill was once the only reading, and why that changed.** On the
    drawing that prompted the change, tracing left the face as bare paper: a
    light face on a light background has no edge strong enough to trace, so it
    vanished into the page. Filling gives the face a shape because the dark hair
    around it supplies the boundary. That reasoning is still correct for a
    *flat two-tone* drawing -- which is why ``"fill"`` remains the default -- but
    it is too coarse for a soft-shaded illustration, where the mid-tones carry
    the detail the user is looking at.

    ``"xdog"`` -- the XDoG soft threshold of Kazuhito00's sample, ported via the
    线稿工坊 workspace. A difference of Gaussians, sliced with a tanh rather
    than a hard level, which keeps faint strokes as thin strokes instead of
    dropping them; the output is then cut to the mask at
    :attr:`xdog_level`. Measured on 惊蛰 at the defaults it is the *finest* of
    the readings (6.1% ink at level 128): hair strands and the weapon's
    hatching survive, which is why it joined the panel.

    ``"ridge"`` -- multi-scale ridge detection (Meijering, via the workspace's
    scikit-image panel, reimplemented dependency-free in :mod:`pcbart.ridge`).
    It asks "is this pixel the centre of a dark strand?" instead of "is this a
    boundary?", so it draws bold continuous strokes -- measured 10.0% ink on
    the same picture, visibly heavier than XDoG's. The two new readings are not
    replacements for the three above: on flat two-tone line art they overdraw
    (see each reading's note), which is what 填充色块 remains for.
    """
    threshold: float = 0.46
    """Response level that separates ink from paper, 0..1.

    Used by ``mode="dodge"`` only; ``mode="fill"`` takes its level from Otsu
    -- measured over the cut-out when there is one, see :func:`_tone_split`.
    In ``mode="outline"`` it is not used at all: the level comes from Otsu and
    the contour's own response is fixed by the calibrated constants below.

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
    """Add the subject's outer outline.

    Needed because a subject ends where it ends, and *neither* reading can see
    that from tone alone: the dodge response has no edge to find where a light
    subject meets a light page, and the fill reading has nothing dark enough to
    turn into ink there either. Measured on the cut-out drawing this was fixed
    on, only **10.3%** of the subject's boundary carried ink without the
    outline, so the figure came out torn all round its pale hair and skin.

    **The two modes use different outlines, deliberately.** ``dodge`` rims
    every boundary of the mask (:func:`silhouette_outline`); ``fill`` traces
    only the outer contour (:func:`outer_outline`), because in the fill reading
    the artwork's interior is already solid ink and a ring drawn inside it
    slices the blocks apart. That distinction is the whole reason the outline
    could come back to fill mode after being removed from it.

    That boundary is what a board outline is cut from, so it matters even
    though it looks like a minor visual detail.
    """

    silhouette_width: int = 5
    """Line width for the outer outline, in pixels at the traced resolution.

    Calibrated to match the ~3.3 px stroke of the reference at full resolution.
    Forced odd; clamped to 1..31.
    """

    outline_sigma_ratio: float = 0.0016
    """Contour detector's blur radius, as a fraction of the picture's short side.

    Sets the *scale of detail* the contour reading keeps, which is why it is a
    ratio rather than a pixel count -- the same trap :mod:`pcbart.work_scale`
    exists to avoid. At the low end the detector picks up texture as lines; at
    the high end it drops fine features like eyelashes into their surroundings.
    Measured on the library illustrations the user compared against, 0.0016
    (1.6 px at 1000 px, 3.2 px at 2000 px) keeps eyebrows, chain links and
    fabric weave while leaving flat colour fields clean.
    """

    outline_density: float = 0.06
    """Share of the subject to return as contour, 0.01..0.25.

    The counterpart to :attr:`outline_sigma_ratio`: that one says *how fine* a
    boundary may be, this one says *how many* of them to keep. Together they are
    the two halves of "draw the lines, not the texture", and neither alone is
    enough -- see :data:`OUTLINE_DENSITY` for why the second has to be a share
    rather than a contrast level.
    """

    outline_width_mm: float = 0.20
    """How wide the contour prints, in millimetres.

    **Millimetres rather than pixels, and that is the point.** The user's
    complaint was that a picture's strokes came out *ragged* -- thick where the
    tone change was gradual and missing where it was abrupt -- and a pixel
    radius cannot fix that, because the same pixel means a different amount of
    material on a different board. Converting from millimetres means the line is
    the same width on the board whatever the source resolution was.

    Default 0.20 mm: comfortably above JLCPCB's 0.10 mm floor
    (:data:`pcbart.validation.MIN_TRACE_WIDTH_MM`) with margin for the
    etcher's tolerance.

    ⚠️ It only takes effect in ``mode="outline"``, and converting it needs the
    board's scale, which :func:`line_art` does not know. Callers that know it
    (the preprocessing panel, a CLI with a target size) pass ``px_per_mm`` to
    :func:`line_art`; without it the line is sized as a fraction of the picture
    so a script still gets a sensible drawing.
    """

    # ------------------------------------------------------ XDoG (Kazuhito00)
    # The five upstream parameters plus the level that makes the soft response
    # binary. Defaults are the upstream's own (Kazuhito00/XDoG-OpenCV-Sample,
    # MIT), which is also what the 线稿工坊 workspace's panel ships.

    xdog_sigma: float = 1.4
    """XDoG blur radius, in pixels at a 1000 px long side (see the ratio note
    on :func:`xdog_response`). Sets the width of the lines the filter calls a
    line; it is the XDoG control users move first."""

    xdog_k: float = 1.6
    """Ratio of the second blur radius to the first. The classic DoG value;
    larger values widen the halo around every stroke."""

    xdog_epsilon: float = 0.0
    """Response level the soft threshold sits at, the upstream's ``epsilon``
    (its slider is -1..1 and it divides by 255 before use). Higher values keep
    faint edges; lower values let more of the picture cross into ink."""

    xdog_phi: float = 10.0
    """Steepness of the soft threshold. Low values ramp gently (a pencil
    look), high values approach a hard cut."""

    xdog_gamma: float = 0.98
    """Weight of the wider blur when the two are subtracted. The upstream
    default is very close to 1, which is what keeps flat areas at paper
    white."""

    xdog_level: int = 128
    """Level (0..255) that turns the soft response into a mask; ink below it.

    Needed because the upstream panel returns a *soft* image and this tool's
    masks must be strictly two-tone -- silkscreen cannot print a grey pixel.
    Measured on 惊蛰 at the defaults: level 64 -> 4.4% ink, 128 -> 6.1%,
    192 -> 7.9%, so the slider moves the ink share the way the readout says.
    """

    # --------------------------------------------------- ridge (scikit-image)
    # The 线稿工坊 workspace's 脊线 panel: skimage.filters.meijering. The port
    # in pcbart.ridge is verified against scikit-image; these keep the
    # workspace's names and defaults.

    ridge_sigma_min: float = 1.0
    """Smallest scale of the ridge filter, in pixels at a 1000 px long side."""

    ridge_sigma_max: float = 3.0
    """Largest scale. The filter reports the strongest response over the scales
    in between, so this end picks up the widest strands."""

    ridge_steps: int = 5
    """How many scales to sample between the two ends (scikit-image's own
    default is five; more scales cost time linearly)."""

    ridge_black: bool = True
    """Detect dark ridges. Off selects bright ones instead -- the polarity of
    the *drawing*, not of the output: the mask is always ink-on-paper."""

    ridge_percentile: float = 50.0
    """Percentile of the ridge response kept as ink.

    ⚠️ **Ranked over the pixels that respond**, not over the whole picture:
    keep the strongest ``100 - percentile``% of the strands. The workspace's
    panel ranks over every pixel, which is the same answer on a smoothly shaded
    illustration but floods hard-edged artwork -- see the ridge branch in
    :func:`line_art` for the measurement. The default is 50 rather than the
    workspace's 90 for that reason: **50 here reproduces what the workspace's
    90 draws** (both keep the strongest 10% of the ink on 惊蛰, measured 9.99%
    against 10.0%), while staying sane on hard-edged artwork where its 90 turns
    the page black. The panel's readout always shows the share that came out.
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
            outline_sigma_ratio=float(np.clip(self.outline_sigma_ratio, 0.0002, 0.01)),
            outline_density=float(
                np.clip(
                    self.outline_density, OUTLINE_MIN_DENSITY, OUTLINE_MAX_DENSITY
                )
            ),
            outline_width_mm=float(np.clip(self.outline_width_mm, 0.10, 1.0)),
            xdog_sigma=float(np.clip(self.xdog_sigma, 0.1, 40.0)),
            xdog_k=float(np.clip(self.xdog_k, 1.05, 5.0)),
            xdog_epsilon=float(np.clip(self.xdog_epsilon, -1.0, 1.0)),
            xdog_phi=float(np.clip(self.xdog_phi, 0.1, 80.0)),
            xdog_gamma=float(np.clip(self.xdog_gamma, 0.1, 2.0)),
            xdog_level=int(np.clip(int(self.xdog_level), 1, 254)),
            ridge_sigma_min=float(np.clip(self.ridge_sigma_min, 0.1, 20.0)),
            ridge_sigma_max=float(np.clip(self.ridge_sigma_max, 0.1, 40.0)),
            ridge_steps=int(np.clip(int(self.ridge_steps), 1, 8)),
            ridge_black=bool(self.ridge_black),
            ridge_percentile=float(np.clip(self.ridge_percentile, 0.0, 100.0)),
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

    # ⚠️ **Bounded, and this was measured the hard way.** The share being
    # measured is a proportion of the frame, so it survives a rescale -- but
    # computing it at full size does not. On a 36 043 x 19 653 drawing (708
    # megapixels) this took **28.7 seconds** on the UI thread, because loading
    # a picture has to know the answer before it can fill the panel. Reusing
    # the same bound as the two-colour test keeps every whole-picture statistic
    # in this project on the same budget.
    from .image_prep import _probe_copy

    image = _probe_copy(image)
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


#: The long side at which the XDoG and ridge readings' size-class parameters
#: mean what their sliders say.
#:
#: Both algorithms measure their scale in *pixels*, and both were tuned (in the
#: 线稿工坊 workspace these readings come from) at a 1000 px reference long
#: side: sigma 1.4 is "1.4 px on a 1000 px picture". Multiplying by the
#: picture's own long side over this constant keeps the same amount of detail
#: whatever the source resolution is -- the rule :mod:`pcbart.work_scale`
#: exists to enforce for every other resolution-dependent number.
READING_REFERENCE_LONG_SIDE = 1000.0


#: Long side the ridge reading's *live preview* is traced at; see
#: :func:`line_art_preview` for the measurement behind it.
RIDGE_PREVIEW_TRACE_SIDE = 1000


def xdog_response(
    grey8: np.ndarray,
    sigma: float,
    k_sigma: float,
    gamma: float,
    epsilon: float,
    phi: float,
) -> np.ndarray:
    """The Kazuhito00 XDoG soft response, in the upstream's own 0..255 scale.

    Ported from ``Kazuhito00/XDoG-OpenCV-Sample`` (MIT) via the 线稿工坊
    workspace's ``algorithms/xdog_kazuhito.py``, which already checked the port
    line by line. **Low values are ink**: the response goes dark where a pixel
    is darker than its surroundings, and stays at 255 across flat areas.

    The steps are the upstream's, in the upstream's order:

    1. ``dog = G_sigma - gamma * G_{k*sigma}`` on the **uint8** grey plane, so
       the blur quantises the same way the reference implementation did;
    2. divide by the picture's own maximum -- this is the upstream's
       normalisation, and it is why the same settings give a brighter or
       darker drawing on different artwork;
    3. ``e = 1 + tanh(phi * (dog - epsilon))``, clipped at 1.

    The upstream (and the workspace port) returns this soft image as the
    finished panel; this tool's masks must be strictly binary, so the caller
    puts it through :attr:`LineArtParams.xdog_level`. A flat picture -- no
    response anywhere -- returns all-255 rather than dividing by zero, matching
    the workspace port's guard.
    """
    g1 = cv2.GaussianBlur(grey8, (0, 0), float(sigma)).astype(np.float64)
    g2 = cv2.GaussianBlur(grey8, (0, 0), float(sigma) * float(k_sigma)).astype(np.float64)
    dog = g1 - float(gamma) * g2
    maximum = float(dog.max())
    if maximum < 1e-9:
        return np.full(grey8.shape, 255.0, dtype=np.float64)
    dog = dog / maximum
    response = 1.0 + np.tanh(float(phi) * (dog - float(epsilon) / 255.0))
    np.minimum(response, 1.0, out=response)
    return np.clip(response * 255.0, 0.0, 255.0)


#: Ratio between the detector's two blur radii.
#:
#: The classic DoG value. Larger values widen one side of every edge rather
#: than sharpening it -- the halo the XDoG paper documents -- and on line art
#: that reads as a drop shadow beside every stroke.
OUTLINE_DOG_RATIO = 1.6

#: Peak ``|G_σ - G_kσ|`` at a full black-to-white step.
#:
#: Measured, not derived: with :data:`OUTLINE_DOG_RATIO` at 1.6 the response
#: peaks at 0.1074 for a 0→1 step, whatever σ is (the filter is scale
#: invariant). Dividing the response by this makes it a **contrast in 0..1**,
#: which is the only form a threshold can be stated in and reasoned about.
DOG_UNIT_STEP = 0.1074

#: Share of the subject that comes back as contour, at the default setting.
#:
#: ⚠️ **A share rather than a contrast, and that is the whole design.** A
#: contrast threshold cannot serve both kinds of picture: an illustration has
#: real boundaries at a few percent contrast (a fold in cloth, the rim of a
#: lens) *and* scenery at 20-60% (architecture, foliage), so one fixed level
#: either drops the character's own detail or inks the backdrop wholesale --
#: measured on a library illustration, a threshold low enough to keep eyebrows
#: returned **50% of the frame** as "line". Asking for a share instead means
#: "give me the strongest N% of the local contrast", which behaves the same on
#: a sparse sketch and a dense engraving and cannot run away.
#:
#: 6% is where the rendered result stops being texture and starts being
#: drawing: measured on four library illustrations the contour lands between
#: 2% and 6% of the frame on top of the fill pass's 13-35%.
OUTLINE_DENSITY = 0.06

#: Bounds on the share, so neither extreme is reachable by slider or by picture.
#:
#: The floor keeps a picture with almost no contrast from returning nothing at
#: all; the ceiling stops a picture that is *all* texture from being traced
#: into a solid block.
OUTLINE_MIN_DENSITY = 0.01
OUTLINE_MAX_DENSITY = 0.25

#: Contrast below which nothing is a boundary, however little ink the density
#: has produced so far.
#:
#: ⚠️ **The density quantile alone is not enough, and this is why.** Asking for
#: "the strongest 6% of local contrast" is meaningful only when the picture
#: *has* 6% worth of boundaries. A flat field does not: every pixel answers the
#: same near-zero response, so the quantile lands inside that zero population and
#: the comparison inks most of the frame. Measured on a uniform 300 px field of
#: grey 60, the quantile threshold returned **98.7% of the pixels** as contour --
#: a solid slab of silkscreen where the artwork had nothing at all.
#:
#: This is the floor that stops it. The value comes from the same measurement as
#: the density: JPEG ringing and dithering sit around 1%, so anything below that
#: is not a line, and a picture whose strongest boundary is below it gets no
#: contour -- which is the honest answer.
OUTLINE_MIN_CONTRAST = 0.02


def outline_response(grey01: np.ndarray, sigma: float) -> np.ndarray:
    """Local contrast, in 0..1, at every pixel -- the boundary strength.

    A magnitude: it does not matter which side of the boundary is darker.

    ⚠️ **Both corrections here are load-bearing, and each was measured against a
    failure.**

    1. **The response has to be symmetric.** The textbook form is Extended
       Difference of Gaussians (Winnemöller et al., NPAR 2011),
       ``D = (1+p)·G_σ - p·G_kσ``, inked where ``D < ε``. That asks "is this
       pixel on the *dark* side of a step", and it therefore cannot see a pale
       boundary at all: on a light square in a dark field the response stayed at
       **+0.64 across the whole edge** and not one pixel inked. Eyebrows on a
       shaded face, highlights along a fold and light seams on dark cloth all
       vanished -- exactly what the user reported as 「描边混乱」. Taking the
       magnitude is what makes a boundary a boundary whichever way it runs.
    2. **The response has to be zero-mean.** ``G_σ - G_kσ`` integrates to zero,
       so a constant field answers exactly **0.0** -- verified to float
       precision, because both blurs of a constant are that constant. The XDoG
       form does not have this property: it answers the region's own brightness,
       so a light field sits at ``+0.9`` and any attempt to symmetrise it by
       absolute value inks the entire field.

    See :func:`outline_threshold` for how the result becomes a decision.
    """
    sigma = max(0.4, float(sigma))
    inner = _gaussian(grey01, sigma)
    outer = _gaussian(grey01, sigma * OUTLINE_DOG_RATIO)
    return np.abs(inner - outer) / DOG_UNIT_STEP


def outline_threshold(
    response: np.ndarray,
    density: float = OUTLINE_DENSITY,
    subject: np.ndarray | None = None,
) -> float:
    """The contrast above which a response pixel counts as a drawn boundary.

    ``density`` is the share of the subject to return as contour, clamped to
    :data:`OUTLINE_MIN_DENSITY` .. :data:`OUTLINE_MAX_DENSITY`.

    ``subject`` restricts the measurement to the artwork's own footprint, for
    the same reason :func:`_tone_split` measures over the cut-out: background
    outside the drawing is not a line, and letting it spend the budget would
    make the same figure come out sparser or denser depending on what surrounds
    it.
    """
    import numpy as np

    share = float(np.clip(density, OUTLINE_MIN_DENSITY, OUTLINE_MAX_DENSITY))
    sample = response
    if subject is not None and subject.size == response.size and subject.any():
        sample = response[subject > 0]
    if sample.size == 0:
        return float("inf")
    # The floor wins when it is above the quantile, which is exactly the flat
    # picture case: see :data:`OUTLINE_MIN_CONTRAST`.
    return max(
        OUTLINE_MIN_CONTRAST, float(np.quantile(sample, 1.0 - share))
    )


def outline_lines(
    grey01: np.ndarray,
    sigma: float,
    density: float = OUTLINE_DENSITY,
    subject: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Boolean mask of the picture's boundaries, plus the contrast it used."""
    response = outline_response(grey01, sigma)
    level = outline_threshold(response, density, subject)
    return response >= level, level


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a kernel derived from ``sigma``.

    Odd and at least 3, because OpenCV measures an even kernel from a corner
    rather than from its centre -- the same rule :func:`pcbart.image_prep._odd`
    exists for.
    """
    import cv2

    size = max(3, int(2 * round(3.0 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    return cv2.GaussianBlur(image.astype(np.float32), (size, size), float(sigma))


def _outline_width_px(
    params: "LineArtParams", short_side: int, px_per_mm: float | None
) -> int:
    """The contour's printed width, as pixels at the traced resolution.

    ``outline_width_mm`` is a property of the *board*, and :func:`line_art` has
    no board -- so a caller that knows the scale passes ``px_per_mm``. Without
    it the width falls back to a fraction of the picture, which keeps a
    parameterless call (a test, a script) sensible without pretending to know
    something it does not.
    """
    if px_per_mm and px_per_mm > 0:
        width = int(round(float(params.outline_width_mm) * float(px_per_mm)))
    else:
        width = int(round(short_side * 0.002))
    width = max(1, width)
    # Odd, so the dilating kernel has a centre pixel; capped so a caller asking
    # for a 1 mm line on a small preview does not turn the drawing into a slab.
    return min(width | 1, 101)


def _widen_lines(lines: np.ndarray, width: int) -> np.ndarray:
    """Grow a 1 px contour to roughly ``width`` pixels, centred on the original.

    ⚠️ **A dilate-then-erode pair does not do this, and the first version that
    tried it changed nothing at all.** Dilation by a disc of radius ``r`` then
    erosion by a disc of radius ``r`` is the identity on a thin closed contour --
    the opening of a curve is that curve -- so the measured result was the same
    pixel count at every width from 0.10 mm to 1.00 mm. Thickening a *line* is
    not the same operation as smoothing a *region*.

    What does work is dilating by half the target width: a 1 px stroke grown by
    radius ``r`` becomes roughly ``2r+1`` px wide, and because the growth is
    applied to every pixel of the centre line the result stays centred on it
    without any erosion. The one asymmetry is at the two ends of an open
    contour, which for a closed outline does not arise.

    ``width`` is clamped to odd so the disc has a centre.
    """
    import cv2

    if width <= 1:
        return lines
    width = max(3, width | 1)
    grown = cv2.dilate(
        lines.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width)),
    )
    return grown > 0


def _tone_split_level(grey: np.ndarray, subject: np.ndarray | None = None) -> float:
    """The Otsu level that separates this picture's two tones.

    Split out from :func:`_tone_split` so the level can be computed once and
    reused: it is a property of the *picture*, not of any line-art setting, and
    the interactive path recomputes it on every slider tick otherwise. See
    :func:`_tone_split` for why the sample is the subject rather than the whole
    frame.
    """
    import cv2

    sample = grey
    if subject is not None and subject.size == grey.size and subject.any():
        sample = grey[subject > 0]
    if sample.size == 0:
        sample = grey
    # ``cv2.threshold`` wants a 2-D array; the flattened sample is reshaped
    # into a column rather than a row so it stays contiguous for OpenCV.
    sample = np.ascontiguousarray(sample.reshape(-1, 1))
    level, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level)


def _tone_split(
    grey: np.ndarray, subject: np.ndarray | None = None
) -> np.ndarray:
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

    ⚠️ **The level is measured over the subject, not the whole frame, and that
    was measured.** A cut-out picture is mostly *background*: the transparent
    third of the drawing this was fixed on turns into pure white when flattened,
    which is a single enormous spike at 255. Otsu splits the histogram into two
    classes, so a spike that big drags the level towards it -- and the moment
    anything widens the white peak the level runs away with it. Measured on that
    drawing, the level sat at **170** untouched and jumped to **208** after a
    5 px blur, which swallowed the pale pink hair (median 198) whole: the hair
    came out as one solid blob with the highlights punched out as holes. Over
    the subject alone the same three cases read 158 / 164 / 155 -- stable, and
    the hair keeps its shape either way.

    The subject is passed in rather than recomputed because the caller already
    has it for the outline; ``None`` falls back to the whole frame, which is
    what a picture without a distinguishable subject wants.

    Returns ``uint8``, 0 where ink.
    """
    # Strictly greater, which is the comparison ``cv2.THRESH_BINARY`` itself
    # uses. With ``>=`` a pixel sitting exactly on the level -- the whole dark
    # half of a synthetic two-tone test image, where Otsu returns that very
    # value -- flips to paper and the fill reading produces a blank frame.
    level = _tone_split_level(grey, subject)
    return np.where(grey > level, 255, 0).astype(np.uint8)


#: Despeckle kernel, as a fraction of the picture's short side.
#:
#: ⚠️ **A pixel count would not have worked, and this was measured.** The same
#: drawing run at six sizes with a fixed 3 px kernel:
#:
#: ===========  ==========
#: 400 px        7 fragments
#: 1000 px      22
#: 1382 px      33
#: 4000 px      **85**
#:
#: The fragments scale with the picture, so a fixed kernel cleans the small
#: version and leaves the large one dirty -- which is the same trap
#: :mod:`pcbart.work_scale` exists to avoid. As a fraction, one number covers
#: every size.
#:
#: The value is 3 px on a 1382 px drawing: enough to clear the slivers Otsu's
#: boundary sheds, small enough that the scalloped lace across the top of the
#: head survives. **A larger kernel is not better** -- 9 px cleared the
#: fragments completely and erased that lace.
DESPECKLE_KERNEL_RATIO = 0.0022


def clean_strokes_with_arcs(mask: np.ndarray, *, px_per_mm: float = 1.0) -> np.ndarray:
    """Round a mask's ragged edges by re-describing it with fitted arcs.

    Answers the user's report of 「描边混乱」 at the level a mask *can* be
    fixed at. It is deliberately a separate, opt-in step rather than part of
    :func:`line_art`, because of what it can and cannot do:

    * **It cannot change which pixels were classified as ink.** Every pixel's
      side of the threshold is decided before this runs; re-describing the
      boundary as curves and rasterising it back can only smooth the shapes
      that are already there.
    * **It does round off the staircase and merge neighbouring fragments.** The
      mask is traced by VTracer's spline fit and drawn back, so a boundary that
      stepped one pixel at a time comes back as a curve.

    ⚠️ **Measured cost and benefit, so the trade is explicit.** On three real
    library illustrations the fragment count under the speckle floor fell
    467→265 (43%), 345→264 (23%) and 9→10 (no change), with the ink total moving
    by under 3% -- so it merges small pieces rather than erasing them. The price
    is **4-5x the time** (0.5-1.6 s becomes 2.6-8.1 s at 2000 px), which is why
    the panel exposes it as a switch and does not run it on every slider tick.

    ⚠️ ``px_per_mm`` only sets the unit the tracer's fixed pixel thresholds are
    interpreted in, and ``arc_rings_for`` returns millimetre coordinates about
    the origin. Drawing those straight into a pixel canvas puts the artwork off
    the frame, so the mapping back is done explicitly here -- the same mistake
    is recorded in ``tools/probe_vtrace_lineart.py``.

    Returns the input unchanged when the tracer is not installed, so a missing
    optional dependency degrades to "no smoothing" rather than to no artwork.
    """
    import cv2
    import numpy as np

    if mask is None or mask.size == 0:
        return mask

    from .export import arc_rings_for

    height, width = mask.shape[:2]
    # ``arc_rings_for`` is fed the ink plane in the pipeline's own polarity
    # (0 = ink), which is what the exporter gives it when tracing a placement.
    try:
        polygons = arc_rings_for(mask, px_per_mm)
    except Exception:  # noqa: BLE001 - optional step, never fatal
        return mask
    if not polygons:
        return mask

    from .geometry import arc_polygon_to_path

    canvas = np.full((height, width), 255, np.uint8)
    ordered = sorted(polygons, key=lambda rings: -_ring_area(rings[0]))
    for polygon in ordered:
        for index, ring in enumerate(polygon):
            points = _arc_ring_pixels(ring, width, height, px_per_mm)
            if len(points) < 3:
                continue
            array = np.asarray(points, np.int32).reshape(-1, 1, 2)
            # Index 0 is the outer boundary and everything after it is a hole,
            # so the fill alternates: ink, then bare, then ink again for an
            # island inside a hole. That nesting is why the pieces are drawn
            # largest-first rather than in one pass.
            cv2.fillPoly(canvas, [array], 0 if index == 0 else 255)
    return canvas


def _ring_area(ring) -> float:
    """Signed area of a ring, through the same encoder the writers use."""
    from .geometry import arc_polygon_to_path

    path = arc_polygon_to_path([ring])[0]
    points = [(path[0], path[1])]
    index = 3
    while index < len(path):
        if path[index] == "ARC":
            index += 4
        else:
            points.append((path[index], path[index + 1]))
            index += 2
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _arc_ring_pixels(ring, width: int, height: int, px_per_mm: float):
    """An ``ArcRing`` as integer pixel points, arcs sampled along their sweep.

    ⚠️ The input is **millimetres centred on the origin**, because that is what
    the writers need. Drawing those numbers into a pixel canvas puts the whole
    drawing in one corner -- measured, the first attempt left a scrap of hair at
    the top-left and blank everywhere else. The mapping back is the inverse of
    the exporter's ``_ring_to_mm``: ``x_px = w/2 + x_mm * px_per_mm`` and
    ``y_px = h/2 - y_mm * px_per_mm``.
    """
    import math

    from .geometry import arc_polygon_to_path

    path = arc_polygon_to_path([ring])[0]
    points: list[tuple[int, int]] = []
    start = (float(path[0]), float(path[1]))
    points.append(_to_pixel(start, width, height, px_per_mm))
    index = 3
    while index < len(path):
        if path[index] == "ARC":
            sweep = float(path[index + 1])
            end = (float(path[index + 2]), float(path[index + 3]))
            index += 4
            points.extend(
                _arc_samples(
                    start, end, sweep, width, height, px_per_mm
                )
            )
            start = end
        else:
            point = (float(path[index]), float(path[index + 1]))
            points.append(_to_pixel(point, width, height, px_per_mm))
            start = point
            index += 2
    return points


def _to_pixel(point, width: int, height: int, px_per_mm: float) -> tuple[int, int]:
    return (
        int(round(width / 2.0 + point[0] * px_per_mm)),
        int(round(height / 2.0 - point[1] * px_per_mm)),
    )


def _arc_samples(
    start,
    end,
    sweep_degrees: float,
    width: int,
    height: int,
    px_per_mm: float,
    steps: int = 12,
):
    """Sample a circular arc from a chord and a sweep angle."""
    import math

    points = []
    chord = math.hypot(end[0] - start[0], end[1] - start[1])
    radians = math.radians(abs(sweep_degrees))
    if chord <= 1e-9 or radians <= 1e-9:
        return points
    # A sweep of a full turn or more has no chord to reconstruct the circle
    # from, and the writers never emit one; skip rather than divide by zero.
    if radians >= 2 * math.pi - 1e-9:
        return points
    radius = chord / (2 * math.sin(radians / 2))
    if radius <= 0:
        return points
    height_of_arc = math.sqrt(max(radius**2 - (chord / 2) ** 2, 0.0))
    ux = (end[0] - start[0]) / chord
    uy = (end[1] - start[1]) / chord
    centre = None
    for sign in (1, -1):
        cx = (start[0] + end[0]) / 2 - sign * height_of_arc * uy
        cy = (start[1] + end[1]) / 2 + sign * height_of_arc * ux
        a0 = math.atan2(start[1] - cy, start[0] - cx)
        a1 = math.atan2(end[1] - cy, end[0] - cx)
        delta = a1 - a0
        while delta <= -math.pi:
            delta += 2 * math.pi
        while delta > math.pi:
            delta -= 2 * math.pi
        if (delta > 0) == (sweep_degrees > 0):
            centre = (cx, cy, a0)
            break
    if centre is None:
        return points
    cx, cy, a0 = centre
    for step in range(1, steps + 1):
        angle = a0 + math.radians(sweep_degrees) * step / steps
        points.append(
            _to_pixel(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle)),
                width,
                height,
                px_per_mm,
            )
        )
    return points


def _despeckle_tone(traced: np.ndarray) -> np.ndarray:
    """Remove the hairline fragments a fill boundary leaves behind.

    Otsu's cut runs through soft edges, so wherever a picture has a gradual
    transition the boundary wobbles and sheds slivers: on the drawing this was
    tuned against, 47 paper components with 24 of them under 200 px, the
    largest 13x28. They read as specks of dirt on the board rather than as
    anything the artwork shows.

    Open-then-close removes them. Closing first bridges the one- and two-pixel
    notches a soft boundary leaves; opening then drops whatever is still
    floating. The kernel scales with the picture -- see
    :data:`DESPECKLE_KERNEL_RATIO` for why a fixed one was measured to fail.

    ``traced`` is 0 = ink, 255 = paper, so the operation runs on the paper mask
    and the polarity is restored afterwards.
    """
    import cv2
    import numpy as np

    height, width = traced.shape[:2]
    # ⚠️ **Floored at 3, not scaled all the way down.** Measured on the same
    # drawing at 400, 700 and 1000 px: a scaled kernel drops to 1-2 px and
    # leaves 29, 57 and 92 fragments respectively, while a flat 3 px leaves
    # 3, 16 and 27. The fragments never get smaller than one or two pixels, so
    # there is nothing to gain by shrinking the kernel below 3 -- it stops
    # being able to open anything at all.
    size = max(3, int(round(min(height, width) * DESPECKLE_KERNEL_RATIO)))
    if size % 2 == 0:
        size -= 1

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
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


def outer_outline(
    subject: np.ndarray | None,
    width: int = 5,
) -> np.ndarray:
    """The subject's *outermost* boundary as an ink mask.

    This is the fill reading's counterpart to :func:`silhouette_outline`, and
    the difference is the whole point. ``silhouette_outline`` takes the
    morphological gradient of the mask, so it rims **every** boundary the mask
    has -- including the boundaries of any hole inside it. On a picture where
    the subject footprint is a solid cut-out that is harmless, but on one whose
    footprint has interior holes (measured: 12 of them on a white-background
    JPEG, 13 on the shading fixture) it draws a ring around each hole. Those
    rings land *inside* the artwork and slice the filled blocks apart, which is
    the "不规则圈圈" the user reported five times.

    Tracing the external contour of the filled footprint cannot do that: by
    construction there is exactly one closed line per subject, and nothing
    inside it. It is also the same shape the board outline is cut from, so what
    is drawn here and what gets fabricated agree.

    ``subject`` is the filled footprint from
    :func:`pcbart.image_prep.subject_mask`; an empty or ``None`` mask yields an
    empty result rather than inventing an edge.

    Returns ``uint8``, 0 where ink.
    """
    import cv2
    import numpy as np

    if subject is None or subject.size == 0 or not subject.any():
        return np.zeros((0, 0), np.uint8)

    mask = (subject > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.zeros(mask.shape, np.uint8)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    edge = cv2.morphologyEx(
        filled, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    if width > 1:
        size = 2 * (int(width) // 2) + 1
        edge = cv2.dilate(
            edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    # Kept inside the footprint, like silhouette_outline, so the drawn shape
    # never grows past the subject.
    return (edge & filled).astype(np.uint8)


def cut_out_mask(alpha: np.ndarray | None) -> np.ndarray | None:
    """The subject from transparency alone, or ``None`` when there is none.

    Strictly alpha: a fully opaque channel describes no background, and a fully
    clear one has nothing in it. This is the only source that says *where the
    picture is* independently of what it looks like, which is what makes it
    usable as a region to measure tone over -- see :func:`_tone_split`.
    """
    from .image_prep import subject_mask

    if alpha is None or not alpha.size:
        return None
    opaque = float((alpha > 0).mean())
    if not 0.0 < opaque < 0.999:
        return None
    return subject_mask(np.zeros_like(alpha), alpha, fill=True)


def _subject_footprint(
    alpha: np.ndarray | None,
    image: np.ndarray | None,
) -> np.ndarray | None:
    """The shape the board is cut to, for drawing an outline around.

    Alpha when the picture is a cut-out, otherwise inferred from tone against
    the border. Unlike :func:`cut_out_mask` this always answers when it can:
    an outline is worth having on a flattened JPEG too, and there the subject
    can only come from tone.
    """
    from .image_prep import subject_mask

    cut = cut_out_mask(alpha)
    if cut is not None:
        return cut
    if image is None:
        return None
    return subject_mask(image, alpha, fill=True)


@dataclass(frozen=True)
class SubjectMasks:
    """The two subject regions the fill reading needs, computed once.

    They are separate because they answer different questions and only *one*
    of them is safe to measure tone over:

    * :attr:`cut_out` is the region the artist declared by transparency. It is
      the only source that says "the picture is here" independently of what the
      picture looks like, which is what makes it usable as a histogram sample.
      ``None`` when the file carries no usable alpha.
    * :attr:`footprint` is the shape an outline is drawn around. It falls back
      to a tone-vs-border reading, so a flattened JPEG still gets a boundary.
      ⚠️ It must never be used to choose the Otsu sample: it is derived from the
      tones, so measuring the tones inside it is circular.

    ⚠️ **Why this exists as a value rather than two calls.** Both regions come
    from :func:`pcbart.image_prep.subject_mask`, which is the single most
    expensive step in the whole line-art pass: measured on a 2000 px working
    copy of a library illustration, ``subject_mask`` cost **1.70 s** per call
    and the pass called it *twice*. Building them once and handing the result
    to :func:`line_art` halves the work, and a caller that caches the value
    across slider ticks (the masks do not depend on any line-art parameter)
    removes it from the interactive cost entirely.
    """

    footprint: np.ndarray | None = None
    cut_out: np.ndarray | None = None


def subject_masks(
    image: np.ndarray | None,
    alpha: np.ndarray | None,
) -> SubjectMasks:
    """Both subject regions of ``image``, computed with one pass.

    ``image`` and ``alpha`` must already be at the resolution the caller will
    trace at -- the masks are compared against that image's shape, and
    :func:`outer_outline` silently skips a mask that does not match.

    Exactly one expensive pass is made in either branch: a usable alpha *is*
    the footprint, so it is computed once and handed back as both.
    """
    cut = cut_out_mask(alpha)
    if cut is not None:
        return SubjectMasks(footprint=cut, cut_out=cut)
    if image is None:
        return SubjectMasks()
    from .image_prep import subject_mask

    return SubjectMasks(footprint=subject_mask(image, alpha, fill=True), cut_out=None)


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


@dataclass(frozen=True)
class _PictureDerived:
    """Everything in the fill reading that depends on the picture alone.

    None of these values changes when the user moves a line-art slider: the
    grey plane comes from the colour picture, the split level comes from that
    plane's histogram, and the footprint comes from the picture's alpha or its
    border colour. Together they are the *whole* expensive half of the pass --
    measured on a 2000 px working copy of a library illustration, a conversion
    cost 3.8 s of which ``subject_mask`` was 1.70 s and the grey conversion
    another 0.35 s. The interactive path keeps all of it across ticks; see
    :class:`TraceCache`.
    """

    grey8: np.ndarray
    split_level: float
    footprint: np.ndarray | None


class TraceCache:
    """Remembers the picture-derived half of the fill reading between ticks.

    Why this is needed at all: a slider drag calls the conversion on every
    step, and most of it -- compositing, desaturating, building the subject mask
    and reading the Otsu level off it -- is a function of the *picture* only.
    Recomputing it per step is what the user reported as
    "拖动滑块调参数的时候非常卡，连带整个预览和画布都会卡住". Measured on a
    2000 px working copy, keeping it took a tick from 3.8 s to about 60 ms.

    Keyed on the picture that was handed in, so any change to it (a different
    file, a different working resolution, a changed 降噪 setting, a changed
    抠图) makes the entry stale and it is dropped rather than reused. Keeping
    the key on the *data* rather than on a "which file is loaded" flag is
    deliberate: several of those changes arrive without any signal a caller
    could subscribe to, and a cache that quietly answered for the previous
    picture would be worse than no cache at all.

    The key is a 128-bit digest of the bytes. It is fast -- hashing 12 MB costs
    about 8 ms against the 3.4 s the entry saves -- and the cached value carries
    its own copy of the plane it describes, so a collision could only ever
    return a plane that does not match, never mix two.
    """

    __slots__ = ("_key", "_value")

    def __init__(self) -> None:
        self._key: str | None = None
        self._value: _PictureDerived | None = None

    @staticmethod
    def _key_of(image: np.ndarray, alpha: np.ndarray | None) -> str:
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        digest.update(image.tobytes())
        if alpha is not None:
            digest.update(b"|a|")
            digest.update(alpha.tobytes())
        return f"{image.shape}:{digest.hexdigest()}"

    def matches(
        self, image: np.ndarray, alpha: np.ndarray | None
    ) -> "_PictureDerived | None":
        """The cached value for this exact picture, or ``None``."""
        if self._value is None:
            return None
        if self._key != self._key_of(image, alpha):
            return None
        return self._value

    def store(self, image: np.ndarray, alpha: np.ndarray | None, value: "_PictureDerived") -> None:
        self._key = self._key_of(image, alpha)
        self._value = value

    def clear(self) -> None:
        self._key = None
        self._value = None


def line_art(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    params: LineArtParams | None = None,
    *,
    masks: "SubjectMasks | None" = None,
    trace_cache: "TraceCache | None" = None,
    px_per_mm: float | None = None,
) -> np.ndarray:
    """Convert an image to two-tone line art.

    Returns a strictly binary ``uint8`` image, 0 for ink and 255 for paper --
    the polarity a board layer wants, where dark means "put material here".

    Output is binary rather than greyscale on purpose. Silkscreen cannot render
    a grey pixel, so a soft response would print as dirt; the response is
    therefore thresholded rather than stretched.

    ``image`` may be greyscale, BGR or BGRA. A BGRA picture is flattened onto
    white here (its alpha is a boundary this function owns); an ``alpha``
    passed separately promises the caller already flattened the colour --
    see :func:`pcbart.image_prep.flatten_alpha` for why it must happen once.

    ``masks`` short-circuits the subject computation -- see
    :class:`SubjectMasks` and :func:`subject_masks`.

    ``trace_cache`` supplies the picture-derived results that no line-art
    parameter can change, so an interactive caller that is only moving a slider
    does not recompute them. See :class:`TraceCache`; passing ``None`` computes
    everything, which is what a one-shot conversion wants.

    ``px_per_mm`` is how many traced pixels make a millimetre on the finished
    board. Only ``mode="outline"`` needs it, to turn
    :attr:`LineArtParams.outline_width_mm` into a pixel width.

    The caller is responsible for having reduced ``image`` to the working
    resolution; use :func:`pcbart.work_scale.prepare_working_image` to do it and
    to carry any pixel-measured parameters across.
    """
    params = (params or LineArtParams()).normalised()

    if image.ndim == 3 and image.shape[2] == 4:
        # A picture that still carries its alpha has not been through an import
        # boundary yet, so flatten its colour here -- the same straight-alpha
        # composite every read site does. An explicit ``alpha`` argument means
        # the caller already did that (measured: the panel's colour plane is
        # flat before it gets here), and compositing again would wash the
        # translucent rim out.
        from .image_prep import flatten_alpha

        alpha = image[:, :, 3]
        image = flatten_alpha(image[:, :, :3], alpha)
    cached = trace_cache.matches(image, alpha) if trace_cache is not None else None
    if cached is None:
        grey = desaturate(image) / 255.0
        grey8 = np.clip(grey * 255.0, 0, 255).astype(np.uint8)
        if masks is None:
            masks = subject_masks(image, alpha)
        cached = _PictureDerived(
            grey8=grey8,
            split_level=_tone_split_level(grey8, masks.cut_out),
            footprint=masks.footprint,
        )
        if trace_cache is not None:
            trace_cache.store(image, alpha, cached)
    else:
        grey8 = cached.grey8
        grey = grey8.astype(np.float32) / 255.0
        if masks is not None:
            # A caller that hands in both must get the masks it asked for. The
            # split level still comes from the cache because it is measured
            # over the cut-out, which ``masks`` cannot change.
            cached = _PictureDerived(grey8, cached.split_level, masks.footprint)

    if params.mode == "fill":
        traced = np.where(grey8 > cached.split_level, 255, 0).astype(np.uint8)

        # ⚠️ The outline is back, after being switched off here once. It was
        # switched off because it sliced the filled blocks apart, but the cause
        # was never the idea -- it was *which* outline. The old one rimmed every
        # boundary in the mask, interior holes included, and those rings are
        # what cut the artwork up; the outer contour cannot do that. Meanwhile
        # without any outline the picture has no outer edge at all wherever the
        # subject is pale -- measured on the drawing that exposed this, only
        # **10.3%** of the subject's boundary carried ink, so the figure looked
        # torn. With the outer contour it is 70.7%, which is what the user's own
        # Photoshop reference shows.
        if params.silhouette:
            outline = outer_outline(cached.footprint, params.silhouette_width)
            if outline.shape[:2] == traced.shape[:2] and outline.size:
                traced = np.where(outline > 0, 0, traced).astype(np.uint8)
    elif params.mode == "outline":
        # Lines and blocks, rather than blocks alone. The fill pass supplies
        # solid darks; the contour pass supplies everything that is a boundary
        # between two tones but neither dark nor light enough to cross the Otsu
        # level -- eyebrows, the links of a chain, the weave of cloth. The union
        # is what the user's reference drawing looks like.
        traced = np.where(grey8 > cached.split_level, 255, 0).astype(np.uint8)

        short_side = min(grey.shape[:2])
        sigma = max(0.8, short_side * params.outline_sigma_ratio)
        lines, _level = outline_lines(
            grey, sigma, params.outline_density, cached.footprint
        )

        # The contour's own width is a *pixel* quantity from the detector, and
        # it varies with how abrupt the boundary was. Widening to a fixed
        # printed width is what removes that variation, which is the "ragged"
        # the user reported. ``px_per_mm`` is not known here, so the width is
        # expressed in the same units the caller traced in and converted by the
        # one caller that knows the board's scale.
        lines = _widen_lines(lines, _outline_width_px(params, short_side, px_per_mm))

        # The subject limits *where* the contour may ink, not whether the
        # contour is drawn: without it the scenery behind the character comes
        # out as lines too.
        if cached.footprint is not None and cached.footprint.shape[:2] == lines.shape[:2]:
            lines = np.logical_and(lines, cached.footprint > 0)
        if params.silhouette:
            # The outer boundary is not a tonal edge where a pale subject meets
            # a pale page, so the contour pass cannot find it -- the same gap
            # the fill reading had. Drawing it explicitly keeps the board
            # outline and the drawn edge the same shape.
            outline = outer_outline(cached.footprint, params.silhouette_width)
            if outline.shape[:2] == traced.shape[:2] and outline.size:
                lines = np.logical_or(lines, outline > 0)

        traced = np.where(lines, 0, traced).astype(np.uint8)
    elif params.mode == "dodge":
        radius = auto_radius(grey.shape, params.strength)
        response = dodge_response(
            grey, radius, multiply_by_brightness=params.multiply_by_brightness
        )
        traced = np.where(response <= params.threshold, 0, 255).astype(np.uint8)

        # The tracing reading gets its outline here; the fill reading drew its
        # own above, from the *outer contour* rather than from every boundary --
        # see the note there for why the two are not interchangeable.
        #
        # The colour image is passed along because a picture with no usable alpha
        # -- a JPEG, a flattened PNG -- has no other source for its subject
        # boundary, and skipping the outline for those left the board with no
        # outer edge.
        if params.silhouette:
            outline = silhouette_outline(alpha, params.silhouette_width, image)
            if outline.shape[:2] == traced.shape[:2]:
                traced = np.where(outline > 0, 0, traced).astype(np.uint8)
    elif params.mode == "xdog":
        # Kazuhito00's soft threshold, cut to the mask at ``xdog_level``. The
        # sigma is a count of pixels *at the reference long side*, so it is
        # scaled by the picture's own -- see :data:`READING_REFERENCE_LONG_SIDE`
        # for why a fixed pixel count cannot work here.
        long_side = float(max(grey.shape[:2]))
        sigma = max(0.1, params.xdog_sigma * long_side / READING_REFERENCE_LONG_SIDE)
        response = xdog_response(
            grey8,
            sigma,
            params.xdog_k,
            params.xdog_gamma,
            params.xdog_epsilon,
            params.xdog_phi,
        )
        traced = np.where(response < float(params.xdog_level), 0, 255).astype(np.uint8)
        if params.silhouette:
            outline = silhouette_outline(alpha, params.silhouette_width, image)
            if outline.shape[:2] == traced.shape[:2]:
                traced = np.where(outline > 0, 0, traced).astype(np.uint8)
    elif params.mode == "ridge":
        from .ridge import meijering_response

        long_side = float(max(grey.shape[:2]))
        smallest = max(
            0.2, params.ridge_sigma_min * long_side / READING_REFERENCE_LONG_SIDE
        )
        largest = max(
            smallest, params.ridge_sigma_max * long_side / READING_REFERENCE_LONG_SIDE
        )
        sigmas = np.linspace(smallest, largest, max(1, int(params.ridge_steps)))
        response = meijering_response(grey, sigmas, black_ridges=params.ridge_black)

        # ⚠️ **The percentile ranks the pixels that respond, not the whole
        # picture.** The workspace's panel (and the ``skimage`` panel it wraps)
        # takes the percentile over every pixel, which is the same thing on a
        # smoothly shaded illustration -- measured 10.0% ink at the 90th
        # percentile on 惊蛰 -- but breaks on hard-edged artwork: a flat region
        # gives a ridge response of exactly zero (or a float-sized floor), so
        # on a hard-edged drawing 85-95% of the picture sits on one value, a
        # percentile anywhere in the middle lands *inside* that plateau, and the
        # mask comes out as a black rectangle at every setting. Measured on a
        # synthetic hard-edged drawing: 85.4% ink here against 100.0% through
        # scikit-image's own numbers.
        #
        # Ranking the responding pixels keeps the control meaning what its label
        # says -- "keep the strongest (100 - percentile)% of the strands" -- and
        # degrades honestly: no response at all (blank paper) gives no ink
        # rather than a filled page. The panel's readout reports the ink share
        # that actually came out either way.
        responding = response[response > 0]
        percentile = float(params.ridge_percentile)
        if responding.size == 0:
            traced = np.full(grey.shape, 255, dtype=np.uint8)
        else:
            level = float(np.percentile(responding, percentile))
            traced = np.where(response >= level, 0, 255).astype(np.uint8)
        if params.silhouette:
            outline = silhouette_outline(alpha, params.silhouette_width, image)
            if outline.shape[:2] == traced.shape[:2]:
                traced = np.where(outline > 0, 0, traced).astype(np.uint8)
    else:
        # ⚠️ **An unknown mode used to fall through to dodge.** A typo in a
        # caller then produced a plausible drawing from the wrong algorithm,
        # which is the silent-substitution failure this codebase keeps
        # rejecting. Every reading is a named branch now.
        raise ValueError(f"未知的线稿读法: {params.mode}")

    if params.mode == "fill":
        traced = _despeckle_tone(traced)

    # ⚠️ Arc smoothing is *not* applied here even though `line_art` is where the
    # mask comes from. It costs seconds rather than milliseconds -- see
    # :func:`clean_strokes_with_arcs` -- so it belongs to a deliberate switch,
    # not to every conversion, and the panel calls it separately.

    return traced


def line_art_preview(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    params: LineArtParams | None = None,
    max_side: int = 900,
    *,
    masks: "SubjectMasks | None" = None,
    trace_cache: "TraceCache | None" = None,
    px_per_mm: float | None = None,
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
    ridge_cap = (
        RIDGE_PREVIEW_TRACE_SIDE / max(1, max(height, width))
        if (params or LineArtParams()).mode == "ridge"
        else 1.0
    )
    # ⚠️ **The ridge reading is bounded harder, and only for the live preview.**
    # Its cost is a full Hessian per scale: measured **330 ms at 1000 px and
    # 733 ms at 2000 px** against 36-58 ms for everything any other reading
    # does, so at the work budget a slider tick would bring back the 调参卡顿
    # this panel was fixed for. Both of its sigmas are a *fraction of the long
    # side*, so a smaller trace draws the same drawing on a coarser grid rather
    # than a different one; the settled pass (``max_side is None``) still traces
    # at the work budget, which is the arrangement 弧线平滑 already uses.
    trace_scale = min(trace_scale, ridge_cap)
    if trace_scale < 1.0:
        if trace_scale < work_scale(width, height):
            # The ridge cap is stricter than the work budget. Reducing here
            # rather than on top of ``prepare_working_image`` keeps it one
            # resample instead of two.
            size = (
                max(1, int(round(width * trace_scale))),
                max(1, int(round(height * trace_scale))),
            )
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        else:
            image, alpha, trace_scale = prepare_working_image(image, alpha)
        params = _rescaled_params(params, trace_scale)
        # The masks belong to the picture that was handed in. After a reduction
        # they describe a different (larger) grid, so they cannot be reused --
        # ``outer_outline`` would skip them on a shape mismatch and the figure
        # would silently lose its outline.
        masks = None
        trace_cache = None
        # ``px_per_mm`` describes the *preview's* pixels, so it shrinks with the
        # picture. Without this a reduced run would draw a line whose width is
        # right in source pixels and too wide on screen.
        if px_per_mm is not None:
            px_per_mm = px_per_mm * trace_scale

    height, width = image.shape[:2]
    shortest = min(height, width)

    full = line_art(
        image,
        alpha,
        params,
        masks=masks,
        trace_cache=trace_cache,
        px_per_mm=px_per_mm,
    )
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
