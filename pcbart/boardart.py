"""Turn an illustration into a two-tone board painting.

The look is the one the reference pieces share: the character is ringed by a
white contour, its light areas print white, and its dark areas plus the whole
background become solid black. Two tones, hard edges, no gradients.

That is not a stylistic preference so much as the only thing a board can
render. Silkscreen carries opaque white ink or nothing -- it cannot print a grey
pixel, and a soft edge dries into a ragged one. So the tone split here is a
hard threshold, and every mask leaves this module strictly 0/255.

Three decisions carry the style, and each one was measured rather than guessed:

**The tone split.** At :data:`TONE_THRESHOLD` the reference drawing keeps its
hair, face and dress white while the collar, the eyes and the drawn line work
go black -- 15% of the subject. Push the threshold lower and the drawing loses
its hair definition; push it past 0.74 and the hair itself starts turning
black, which is the opposite of "light areas print white".

**The contour.** Drawn from the *silhouette*, grown inwards, never outwards, so
the outline stays inside the artwork. It matters most exactly where the
character's own edge is dark: without it a black collar against a black
background has no boundary at all and the shape dissolves into the field.

**The background.** Black, and written as a real filled region rather than left
blank. Leaving it blank would look identical only on a black-masked board;
filling it means the piece reads the same whichever colour the fab house
actually coats the board in.

Edges are cleaned with morphology, never a blur: a blur would put grey back
into a mask that is supposed to have exactly two values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .lineart import desaturate

__all__ = [
    "BoardArtParams",
    "BoardArt",
    "board_art",
    "board_art_preview",
    "board_art_records",
    "board_art_outline_mm",
    "tone_to_ink",
    "auto_outline_px",
    "TONE_THRESHOLD",
    "OUTLINE_RATIO",
    "PREVIEW_EMPTY",
]

TONE_THRESHOLD = 0.62
"""Grey level, 0..1, that divides "prints white" from "prints black".

Measured on ``由比滨透明背景.png``: the subject's luminance sits at 0.45 for its
10th percentile and 0.79 at the median, so the split has to land between them.
0.62 puts 15% of the subject in black -- the collar, the eyes and the ink work,
which is the drawing's dark structure and nothing more.
"""

OUTLINE_RATIO = 0.006
"""Contour width, as a fraction of the image's short side.

Width has to follow resolution for the same reason the line-art radius does: a
fixed pixel count gives a hairline on a large image and a heavy band on a small
one. 0.006 reproduces the 6 px contour that reads correctly on the 960 px
reference.
"""

MIN_OUTLINE_PX = 2
MAX_OUTLINE_PX = 40

#: Width written into every FILL record, copied from a captured project. The
#: field's unit is not settled by the capture, so the editor's own value is
#: written verbatim rather than converted.
FILL_WIDTH_RAW = 0.2

#: Which sides a board painting may be placed on.
SIDES: tuple[str, ...] = ("top", "bottom", "both")
"""``both`` puts one copy per side, which is what a two-sided pendant wants.

The white tone always takes silkscreen and the black always takes copper, on
whichever side is chosen. That pairing is fixed rather than configurable:
silk is the only layer that prints white, and copper is the only one that gives
the dark field a solid extent -- a solder-mask opening is an absence of mask,
so it cannot fill anything.
"""


def layer_keys(side: str) -> dict[str, list[str]]:
    """Layer keys for the white and black tones on the chosen side.

    Returns ``{"white": [...], "black": [...]}``; ``side="both"`` gives two
    entries per tone, one for each face.
    """
    if side not in SIDES:
        raise ValueError(f"未知的板面: {side}")

    prefixes = ["top", "bottom"] if side == "both" else [side]
    return {
        "white": [f"{prefix}_silk" for prefix in prefixes],
        "black": [f"{prefix}_copper" for prefix in prefixes],
    }

PREVIEW_EMPTY = 96
"""Grey used in a preview where no layer carries anything.

Only reachable with ``black_background=False``, where the background is left as
bare board rather than filled. Showing it mid-grey keeps the preview from
claiming the board is black there when nothing has been drawn.
"""


@dataclass
class BoardArtParams:
    """Controls for :func:`board_art`."""

    tone_threshold: float = TONE_THRESHOLD
    """See :data:`TONE_THRESHOLD`. Lower keeps more of the picture white."""

    outline_ratio: float = OUTLINE_RATIO
    """Contour width per pixel of the short side. See :data:`OUTLINE_RATIO`."""

    outline_px: int = 0
    """Explicit contour width in pixels; 0 means derive it from the ratio."""

    min_island_px: int = 16
    """Islands of either tone smaller than this are absorbed into the other.

    The tone split runs through the picture's gradients, so it leaves specks
    wherever a soft edge crosses the threshold -- photograph grain, the lace
    texture, the anti-aliased rim of a line. Each speck would become its own
    silkscreen blob.
    """

    cleanup_px: int = 3
    """Morphological kernel for the silhouette edge, in pixels.

    Removes the single-pixel staircase an alpha cut-out leaves behind. This is a
    min/max filter, not a blur, so the edge stays a step rather than becoming a
    ramp.
    """

    min_feature_mm: float = 0.10
    """Narrowest feature either tone may keep, in **millimetres**. 0 disables it.

    0.10 mm is a common floor for 1 oz copper. Without a floor the tone split
    leaves sub-pixel specks wherever a soft edge crosses the threshold, and
    those come off the etcher as ragged dirt rather than as the lines they were
    drawn as. Raise it for a coarser process; lower it for a board whose artwork
    is drawn with hairlines.

    The unit is millimetres, not pixels, because the constraint belongs to the
    board and not to the file. The same 2 px strand is unprintable on a 40 mm
    pendant and perfectly printable on a 200 mm sign; a pixel-based limit would
    reject it in both cases and silently throw away detail on the large board.

    Raising this is not free. The filter that removes dirt also removes fine
    line work, so on artwork drawn with hairlines it trades fidelity for
    printability -- which is the trade the process actually forces.
    """

    black_background: bool = True
    """Fill everything outside the subject black.

    Turn off to leave the background as bare board and draw the character only.
    """

    flip_vertical: bool = False
    """Mirror top-to-bottom before anything else runs.

    Applied to the source rather than to the finished masks so the preview, the
    board outline and the written geometry all agree: flipping the output would
    leave the board rectangle claiming one orientation while the artwork inside
    it showed another.
    """

    flip_horizontal: bool = False
    """Mirror left-to-right. See :attr:`flip_vertical`."""

    invert: bool = False
    """Swap the two tones, producing the negative of the painting.

    What was white becomes black and the other way round, contour included --
    this is an inversion of the finished two-tone picture, not a change to
    where the tone split sits. Shifting the split would move the boundary
    between light and dark; inverting keeps every boundary exactly where it was
    and exchanges the two inks, which is the effect the negative of a print has.

    On a board that means silk and copper swap roles: the dark field is printed
    in ink and the character is left as bare metal.
    """

    def normalised(self) -> "BoardArtParams":
        """Clamp to usable ranges."""
        return replace(
            self,
            tone_threshold=float(np.clip(self.tone_threshold, 0.02, 0.98)),
            outline_ratio=float(np.clip(self.outline_ratio, 0.0, 0.1)),
            outline_px=int(np.clip(self.outline_px, 0, MAX_OUTLINE_PX)),
            min_island_px=max(1, int(self.min_island_px)),
            cleanup_px=int(np.clip(self.cleanup_px, 0, 31)),
            min_feature_mm=float(np.clip(self.min_feature_mm, 0.0, 2.0)),
            black_background=bool(self.black_background),
            flip_vertical=bool(self.flip_vertical),
            flip_horizontal=bool(self.flip_horizontal),
            invert=bool(self.invert),
        )

    def resolved_outline_px(self, shape: tuple[int, ...]) -> int:
        """Contour width in pixels for an image of this shape."""
        if self.outline_px:
            return int(self.outline_px)
        return auto_outline_px(shape, self.outline_ratio)


@dataclass
class BoardArt:
    """The two masks a board painting is made of.

    ``white`` and ``black`` are boolean and, with the default black background,
    partition the frame exactly: every pixel is one or the other, never both and
    never neither. That is worth asserting on rather than assuming -- a gap
    between them would come out as a hairline of bare board through the middle
    of a solid area.
    """

    white: np.ndarray
    black: np.ndarray
    subject: np.ndarray
    outline_px: int = 0
    min_feature_px: int = 0
    """Minimum feature width the tone split was filtered to, in pixels.

    Recorded rather than left implicit: it is how much of the drawing's detail
    the process floor removed, and an empty value here means no floor was
    applied because the output size was not known.
    """

    black_background: bool = True
    """Whether the dark field fills the frame or stops at the character.

    Carried here because it decides how the writer bounds the geometry: the
    filled background is the whole image rectangle, the bare one is the
    subject's own footprint.
    """

    @property
    def is_complete_partition(self) -> bool:
        return not bool((self.white & self.black).any()) and bool(
            (self.white | self.black).all()
        )

    def summary(self) -> str:
        total = self.white.size
        return (
            f"白色 {self.white.sum() / total * 100:.1f}%、"
            f"黑色 {self.black.sum() / total * 100:.1f}%、"
            f"轮廓宽 {self.outline_px} px"
        )


def auto_outline_px(shape: tuple[int, ...], ratio: float = OUTLINE_RATIO) -> int:
    """Contour width for an image of this shape, in pixels."""
    short_side = min(int(shape[0]), int(shape[1]))
    width = round(short_side * float(ratio))
    return int(np.clip(width, MIN_OUTLINE_PX, MAX_OUTLINE_PX))


def board_art(
    image: np.ndarray,
    alpha: np.ndarray | None = None,
    params: BoardArtParams | None = None,
    px_per_mm: float | None = None,
) -> BoardArt:
    """Convert an illustration to the board-painting two-tone form.

    ``image`` may be BGR, BGRA or greyscale; ``alpha`` can be passed separately
    for a picture already split into colour and transparency.

    ``px_per_mm`` is the resolution the artwork will be printed at, and is what
    turns :attr:`BoardArtParams.min_feature_mm` into pixels. Without it the
    process floor cannot be applied -- the same picture is a different board at
    every output size -- so the minimum-feature filter is skipped and only the
    speck filter runs.

    Returns the white and black masks at the source resolution.
    """
    import cv2

    params = (params or BoardArtParams()).normalised()

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]

    image, alpha = _flip(image, alpha, params)

    subject = _subject(image, alpha)
    if subject is None or not subject.any():
        raise ValueError("找不到主体：图片全透明，且没有可用于推断主体的底色")

    if params.cleanup_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.cleanup_px,) * 2
        )
        # Close first: it seals the pinholes an alpha cut-out leaves along the
        # rim, and opening afterwards would only re-expose them.
        subject = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, kernel).astype(bool)

    grey = desaturate(image).astype(np.float32) / 255.0

    feature_px = 0
    if px_per_mm and px_per_mm > 0 and params.min_feature_mm > 0:
        # Rounded to the nearest pixel, and left at 0 when the floor is finer
        # than the smallest kernel OpenCV will accept. Clamping up to a fixed
        # minimum here would impose a limit on a board whose resolution never
        # asked for one, which is the opposite of stating the floor in
        # millimetres.
        feature_px = int(round(params.min_feature_mm * px_per_mm))
        if feature_px < 2:
            feature_px = 0

    light = subject & (grey >= params.tone_threshold)
    light, dark = _clean_tone(
        light, subject, params.min_island_px, feature_px
    )

    outline_px = params.resolved_outline_px(subject.shape)
    ring = _contour(subject, outline_px)

    white = light | ring
    black = ~white if params.black_background else subject & ~white

    if params.invert:
        # Swapping the finished masks rather than inverting the grey keeps the
        # tone *boundary* where the preview drew it and only exchanges the two
        # inks. Inverting the grey would move the split to 1-threshold, which
        # is a different picture rather than the same one in negative.
        white, black = black, white

    return BoardArt(
        white=white,
        black=black,
        subject=subject,
        outline_px=outline_px,
        black_background=params.black_background,
        min_feature_px=feature_px,
    )


def board_art_preview(art: BoardArt) -> np.ndarray:
    """The two masks as a BGR image, for looking at before anything is written."""
    height, width = art.white.shape
    out = np.full((height, width, 3), PREVIEW_EMPTY, np.uint8)
    out[art.black] = 0
    out[art.white] = 255
    return out


def board_art_records(
    art: BoardArt,
    width_mm: float,
    side: str = "top",
    vectorize_params=None,
) -> list[tuple[int, list, float]]:
    """``(layer_id, complex_polygon_mm, fill_width)`` triples for the writer.

    The white tone goes on silkscreen and the black on copper, for whichever
    side ``side`` names. ``"both"`` writes each tone on both faces, which is
    what a two-sided pendant needs -- the same geometry, one copy per layer,
    matching how the editor models a picture that appears on two layers.

    ``width_mm`` is the placement width, so the geometry lands at the size the
    user asked for rather than at some default.

    Only the white is traced. The black is then taken as **whatever the white
    does not cover**, by geometric difference rather than by tracing the other
    mask. Tracing both would send each through its own simplification, so the
    two would no longer share an edge: measured on a real drawing, independently
    traced layers left a hairline of bare board up to 0.17 mm wide wherever they
    met. A difference cannot drift -- the layers tile the board by construction.
    """
    from .design_layers import layer
    from .geometry import safe_union, valid_polygon
    from .vectorize import VectorizeParams

    if width_mm <= 0:
        raise ValueError("板画宽度必须为正数")

    keys = layer_keys(side)
    if vectorize_params is None:
        params = VectorizeParams()
    else:
        params = vectorize_params
    # The area filters are forced off whatever the caller asked for. The style
    # stage already dropped everything below its own thresholds, so applying
    # them again here would remove shapes the preview showed -- and the preview
    # is what the user approved. Simplification is still the caller's to choose.
    params = replace(params, min_island_area_mm2=0.0, min_hole_area_mm2=0.0)

    height_px, width_px = art.white.shape[:2]
    px_per_mm = width_px / float(width_mm)
    half_w = width_px / px_per_mm / 2.0
    half_h = height_px / px_per_mm / 2.0

    white_polygons = _vectorize(art.white, px_per_mm, params)
    records = [
        (layer(key).easyeda_id, rings, FILL_WIDTH_RAW)
        for key in keys["white"]
        for rings in white_polygons
    ]

    if art.black_background:
        # The board itself is the extent of the dark field.
        universe = valid_polygon(
            [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        )
    else:
        # Only the character is drawn, so the dark field stops at its edge.
        universe = safe_union(
            _complex_to_polygon(rings)
            for rings in _vectorize(art.subject, px_per_mm, params)
        )
    if universe is None or universe.is_empty:
        return records

    covered = safe_union(_complex_to_polygon(rings) for rings in white_polygons)
    black = universe.difference(covered) if covered is not None else universe
    black_polygons = _geometry_to_complex(black)

    records.extend(
        (layer(key).easyeda_id, rings, FILL_WIDTH_RAW)
        for key in keys["black"]
        for rings in black_polygons
    )
    return records


def _complex_to_polygon(rings):
    """A complex polygon as one Shapely polygon, repaired if it is invalid."""
    from shapely.geometry import Polygon

    if not rings or len(rings[0]) < 3:
        return None
    polygon = Polygon(rings[0], rings[1:])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _geometry_to_complex(geometry) -> list[list[list[tuple[float, float]]]]:
    """A Shapely polygon or multipolygon as complex polygons for ``FILL``.

    Each outer keeps its own holes. The board-outline code can flatten the two
    into separate rings because every ring there is its own curve on the layer,
    but a ``FILL`` carries its holes inside one record, so the pairing has to
    survive the conversion.
    """
    from shapely.geometry import Polygon

    from .geometry import normalize_complex_polygon

    if geometry is None or geometry.is_empty:
        return []

    polygons = (
        [geometry]
        if isinstance(geometry, Polygon)
        else [part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon)]
    )

    result = []
    for polygon in polygons:
        outer = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
        holes = [
            [(float(x), float(y)) for x, y in interior.coords[:-1]]
            for interior in polygon.interiors
        ]
        if len(outer) >= 3:
            result.append(normalize_complex_polygon(outer, holes))
    return result


def board_art_outline_mm(art: BoardArt, width_mm: float) -> list[tuple[float, float]]:
    """The frame as a board outline, in millimetres, clockwise in Y-up.

    ``board_art_records`` centres the artwork on the origin, so the outline is
    the image rectangle about that same origin. Matching the two is not
    cosmetic: an outline built from a different reference would leave the
    artwork hanging off the board.
    """
    height_px, width_px = art.white.shape[:2]
    half_w = width_mm / 2.0
    half_h = half_w * height_px / width_px
    return [
        (-half_w, half_h),
        (half_w, half_h),
        (half_w, -half_h),
        (-half_w, -half_h),
    ]


def _vectorize(mask: np.ndarray, px_per_mm: float, params):
    """Mask to complex polygons, with 255 meaning "this is a shape"."""
    from .vectorize import VectorizeParams, vectorize_mask

    shaped = np.where(mask, 255, 0).astype(np.uint8)
    # The style masks mark material with 255, the opposite of the line-art
    # convention vectorize_mask defaults to, so the polarity has to be stated.
    stated = VectorizeParams(
        style=params.style,
        simplify_mm=params.simplify_mm,
        min_island_area_mm2=params.min_island_area_mm2,
        min_hole_area_mm2=params.min_hole_area_mm2,
        coord_snap_mm=params.coord_snap_mm,
        ink_is_dark=False,
        threshold=params.threshold,
    )
    return vectorize_mask(shaped, px_per_mm, stated)


def _flip(
    image: np.ndarray,
    alpha: np.ndarray | None,
    params: BoardArtParams,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the mirror flags to the source and to its transparency.

    Both have to move together: mirroring the colour while leaving the alpha
    where it was would put the subject in one place and its silhouette in
    another, and the contour -- which is traced from the alpha -- would land
    beside the drawing rather than around it.

    ``ascontiguousarray`` because a reversed NumPy view has a negative stride,
    and OpenCV is not obliged to accept one.
    """
    if params.flip_horizontal:
        image = np.ascontiguousarray(image[:, ::-1])
        if alpha is not None:
            alpha = np.ascontiguousarray(alpha[:, ::-1])
    if params.flip_vertical:
        image = np.ascontiguousarray(image[::-1])
        if alpha is not None:
            alpha = np.ascontiguousarray(alpha[::-1])
    return image, alpha


def tone_to_ink(mask: np.ndarray) -> np.ndarray:
    """One tone as a 0=ink / 255=paper image.

    The canvas tints ink by its layer's colour and leaves paper alone, so each
    board-art tone has to be handed over in that convention. Publishing the
    mask's own polarity instead would tint the *unused* area and draw every
    shape inside out.
    """
    return np.where(mask, 0, 255).astype(np.uint8)


def _subject(image: np.ndarray, alpha: np.ndarray | None) -> np.ndarray | None:
    """Where the character is.

    A clean alpha channel *is* the silhouette and is used as-is, at its own
    edge: :func:`pcbart.image_prep.subject_mask` is built for artwork whose
    outline has to be inferred, and its bridging and smoothing kernels are
    sized as a fraction of the picture (4.5% of the short side). On a
    hand-cut PNG that is far too heavy -- it rounds off exactly the crisp edge
    the style is built on -- so it is only the fallback.
    """
    from .image_prep import subject_mask

    if alpha is not None and alpha.size:
        opaque = float((alpha > 0).mean())
        if 0.0 < opaque < 0.999:
            # >127 rather than >0: the rim of a cut-out carries a ring of
            # semi-transparent pixels, and counting those as subject inflates
            # the silhouette by a pixel all the way round.
            return alpha > 127

    return subject_mask(image, alpha, fill=True)


def _contour(subject: np.ndarray, width_px: int) -> np.ndarray:
    """A band of ``width_px`` hugging the inside of the subject's edge.

    Eroding by a kernel of radius ``r`` shaves ``r`` pixels off, so the kernel
    has to be sized ``2*width+1`` for the band to actually be ``width`` wide.
    The odd size matters too: an even-sized OpenCV kernel is anchored off-centre,
    which gave a band three pixels deep on one side of a shape and two on the
    other -- a visible asymmetry on a straight edge.
    """
    import cv2

    radius = max(1, int(width_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    eroded = cv2.erode(subject.astype(np.uint8), kernel).astype(bool)
    return subject & ~eroded


def _clean_tone(
    light: np.ndarray,
    subject: np.ndarray,
    min_island_px: int,
    min_feature_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the features each tone can actually survive printing.

    Both tones are filtered. Filtering one and taking the other as its
    complement leaves the dark side free to keep one-pixel filaments -- measured
    at 3.3% of the dark layer -- because a thin *gap* in the white is a thin
    *feature* in the black.

    So each tone is first opened down to a core of printable features, and the
    pixels belonging to neither core are then assigned to whichever core is
    nearer. That is a watershed over the two sets, and it does the job a
    morphological filter cannot: it removes thin features on both sides at once
    while still tiling the subject exactly, so no pixel is left in both tones or
    in neither.

    Opening harder than the requested floor is not an option for tidier output
    -- the lace texture in a real drawing survives at 3 px and is erased at 5, so
    a heavier filter does not clean the artwork up, it redraws it.
    """
    import cv2

    if min_feature_px < 2:
        dark = _drop_small(subject & ~light, min_island_px)
        return subject & ~dark, dark

    # An even kernel is deliberately allowed here. OpenCV's size-2 ellipse is a
    # 2x2 block and is exactly what a "no feature thinner than 2 px" rule needs:
    # it erases one-pixel filaments while leaving two-pixel ones standing. A
    # 3x3 ellipse is a plus shape that erases two-pixel lines as well, so
    # forcing odd sizes would make the floor jump from 1 px straight to 3.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(min_feature_px),) * 2
    )

    def core(mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return _drop_small(opened.astype(bool), min_island_px)

    light_core = core(light)
    dark_core = core(subject & ~light)

    # Pixels outside both cores are drawn to the nearer one.
    undecided = subject & ~(light_core | dark_core)
    if undecided.any():
        to_light = cv2.distanceTransform(
            (~light_core).astype(np.uint8), cv2.DIST_L2, 3
        )
        to_dark = cv2.distanceTransform(
            (~dark_core).astype(np.uint8), cv2.DIST_L2, 3
        )
        light_core = light_core | (undecided & (to_light < to_dark))

    light = subject & light_core
    return light, subject & ~light


def _drop_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components smaller than ``min_area_px``."""
    import cv2

    if min_area_px <= 1 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    keep = np.zeros(mask.shape, bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area_px:
            keep |= labels == index
    return keep
