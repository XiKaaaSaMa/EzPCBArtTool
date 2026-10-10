"""Where a 通行证 (badge) template wants its artwork, and how it must sit there.

The template is a 50 × 100 mm card. Two of its areas are reserved for the
operator's picture, and the user marked both by drawing a filled rectangle on
the top layer -- there is nothing in the file that labels them, so they are
identified by size. :mod:`tools.extract_badge_regions` measures them out of a
template file; the numbers below are its output for
``ProPrj_鸿雪_2026-09-25.epro2`` and can be re-derived the same way when the
blank board replaces that file.

Two things the module decides, both taken from the user's brief:

* **Which region** an artwork kind belongs in. 精英0 and 精英1 go in the short
  region, 精英2 in the tall one; a skin can go in either, which is why the
  caller has to say which.
* **How it is placed.** The short region takes a *pre-cropped* picture matched
  to the region's width, which the user then drags. The tall region takes a
  *full illustration* that may be moved and scaled freely, but must end up
  entirely inside the region -- see :func:`contain`.

Which asset supplies each is the caller's business: what matters here is that
the short region is width-matched and the tall one is contained, because the
tall region is nearly the whole card and a picture free to overflow it would
print over the card's own frame.

Everything is in millimetres with the board centred on the origin, matching
:class:`pcbart.artboard.ArtBoard`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .artboard import ImagePlacement

__all__ = [
    "CARD_WIDTH_MM",
    "CARD_HEIGHT_MM",
    "ARTWORK_LAYER",
    "ArtworkRegion",
    "SMALL_REGION",
    "LARGE_REGION",
    "REGIONS",
    "ArtworkKind",
    "PlacementMode",
    "ArtworkPlan",
    "plan_artwork",
    "rotated_bounds",
    "fits_inside",
    "overflow_mm",
    "Fit",
    "contain",
    "apply_contain",
    "artwork_placement",
]

CARD_WIDTH_MM = 50.0
"""Card width. The file's own dimensions are annotated ``50.000mm``."""

CARD_HEIGHT_MM = 100.0
"""Card height, annotated ``100.000mm`` in the file."""

ARTWORK_LAYER = "顶层丝印层"
"""Where the picture goes, following the two worked examples in the file.

Their artwork sits on the top silkscreen, which is also the layer a two-tone
drawing reads best on. The rectangles that mark the regions are on the top
*copper* layer; that is a guide, not the destination.
"""


@dataclass(frozen=True)
class ArtworkRegion:
    """A rectangle on the card reserved for the operator's picture.

    Position is the rectangle's centre, measured from the centre of the card,
    because that is the frame the rest of the application works in.
    """

    key: str
    label: str
    width_mm: float
    height_mm: float
    centre_x_mm: float
    centre_y_mm: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.centre_x_mm, self.centre_y_mm)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` in board millimetres, Y up."""
        half_w = self.width_mm / 2.0
        half_h = self.height_mm / 2.0
        return (
            self.centre_x_mm - half_w,
            self.centre_y_mm - half_h,
            self.centre_x_mm + half_w,
            self.centre_y_mm + half_h,
        )

    @property
    def aspect(self) -> float:
        """Width divided by height."""
        return self.width_mm / self.height_mm


SMALL_REGION = ArtworkRegion(
    key="small",
    label="短区（窄）",
    width_mm=38.861,
    height_mm=56.324,
    centre_x_mm=1.854,
    centre_y_mm=21.825,
)
"""The short region: 精英0, 精英1, and skins that fit the narrower area."""

LARGE_REGION = ArtworkRegion(
    key="large",
    label="长区（立绘）",
    width_mm=42.526,
    height_mm=99.626,
    centre_x_mm=3.646,
    centre_y_mm=0.102,
)
"""The tall region: 精英2 and skins that use the full illustration."""

REGIONS: dict[str, ArtworkRegion] = {
    SMALL_REGION.key: SMALL_REGION,
    LARGE_REGION.key: LARGE_REGION,
}


class ArtworkKind(Enum):
    """Which picture of the operator is being placed."""

    ELITE0 = "elite0"
    ELITE1 = "elite1"
    ELITE2 = "elite2"
    SKIN = "skin"

    @property
    def label(self) -> str:
        return {
            ArtworkKind.ELITE0: "精英0",
            ArtworkKind.ELITE1: "精英1",
            ArtworkKind.ELITE2: "精英2",
            ArtworkKind.SKIN: "皮肤",
        }[self]


class PlacementMode(Enum):
    """How the picture is allowed to sit in its region."""

    WIDTH_FIT = "width_fit"
    """Width is matched to the region; the user drags it into place.

    Used for the short region. The height follows the picture's own aspect, so
    a picture that has not been pre-cropped to the region will stick out --
    callers can check with :func:`overflow_mm` rather than having the width
    quietly changed, because matching the width is the whole point of this
    mode.
    """

    CONTAINED = "contained"
    """Freely movable and scalable, but it must end up inside the region."""


@dataclass(frozen=True)
class ArtworkPlan:
    """What to do for one artwork choice."""

    kind: ArtworkKind
    region: ArtworkRegion
    mode: PlacementMode

    @property
    def describe(self) -> str:
        return f"{self.kind.label} → {self.region.label}（{self.mode.value}）"


def plan_artwork(
    kind: ArtworkKind, *, skin_region: ArtworkRegion = LARGE_REGION
) -> ArtworkPlan:
    """Pick the region and the placement rule for one artwork choice.

    精英1 exists only for 阿米娅; that is the caller's business, but the region
    is the same short one as 精英0 because both are placed the same way.
    A skin is the one kind that can go in either region, so the caller passes
    which -- the brief describes small-region and large-region skins as
    different choices rather than deriving one from the other.
    """
    if not isinstance(kind, ArtworkKind):
        raise TypeError(f"kind 必须是 ArtworkKind，收到 {kind!r}")

    if kind is ArtworkKind.SKIN:
        region = skin_region
        if not isinstance(region, ArtworkRegion):
            raise TypeError(f"skin_region 必须是 ArtworkRegion，收到 {region!r}")
    elif kind is ArtworkKind.ELITE2:
        region = LARGE_REGION
    else:
        region = SMALL_REGION

    mode = (
        PlacementMode.WIDTH_FIT
        if region is SMALL_REGION
        else PlacementMode.CONTAINED
    )
    return ArtworkPlan(kind=kind, region=region, mode=mode)


# --------------------------------------------------------------------- geometry


def rotated_bounds(
    centre: tuple[float, float],
    size: tuple[float, float],
    rotation_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a rectangle turned about its own centre.

    The bounding box is what actually decides whether a rotated picture stays
    inside a rectangle: for a convex region, "no corner is outside" and "the
    bounding box is inside" are the same statement, so this is exact rather
    than a conservative approximation.
    """
    width, height = size
    cx, cy = centre
    if width < 0 or height < 0:
        raise ValueError("尺寸不能为负")

    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = width / 2.0, height / 2.0
    xs: list[float] = []
    ys: list[float] = []
    for lx, ly in (
        (-half_w, half_h),
        (half_w, half_h),
        (half_w, -half_h),
        (-half_w, -half_h),
    ):
        xs.append(cx + lx * cos_a - ly * sin_a)
        ys.append(cy + lx * sin_a + ly * cos_a)
    return (min(xs), min(ys), max(xs), max(ys))


def fits_inside(
    region: ArtworkRegion,
    centre: tuple[float, float],
    size: tuple[float, float],
    rotation_deg: float = 0.0,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Whether the whole picture is inside the region, edges included."""
    x0, y0, x1, y1 = rotated_bounds(centre, size, rotation_deg)
    rx0, ry0, rx1, ry1 = region.bounds
    return (
        x0 >= rx0 - tolerance
        and y0 >= ry0 - tolerance
        and x1 <= rx1 + tolerance
        and y1 <= ry1 + tolerance
    )


def overflow_mm(
    region: ArtworkRegion,
    centre: tuple[float, float],
    size: tuple[float, float],
    rotation_deg: float = 0.0,
) -> tuple[float, float]:
    """How far the picture sticks out of the region, per axis.

    ``(0, 0)`` means it fits. The number is the overhang on the offending
    *side*, not the total: a picture 4 mm too wide and centred sticks out 2 mm
    on the left and 2 mm on the right, and this reports ``2``. That is the
    useful figure for telling the user how much to shrink by.

    Returned rather than a bare yes/no so a message can say *how far* and in
    which direction, which is what the user needs in order to decide whether to
    scale or to move.
    """
    x0, y0, x1, y1 = rotated_bounds(centre, size, rotation_deg)
    rx0, ry0, rx1, ry1 = region.bounds
    return (
        max(0.0, rx0 - x0, x1 - rx1),
        max(0.0, ry0 - y0, y1 - ry1),
    )


@dataclass(frozen=True)
class Fit:
    """The correction that brings a picture inside its region."""

    scale: float
    """Factor to multiply both sides by, never above 1 -- the picture is only
    ever shrunk, because growing it would move an edge the user positioned."""

    dx_mm: float
    dy_mm: float

    @property
    def changed(self) -> bool:
        return self.scale < 1.0 - 1e-9 or abs(self.dx_mm) > 1e-9 or abs(self.dy_mm) > 1e-9


def contain(
    region: ArtworkRegion,
    centre: tuple[float, float],
    size: tuple[float, float],
    rotation_deg: float = 0.0,
) -> Fit:
    """Work out the correction that puts the picture inside ``region``.

    Two steps, in this order, and the order matters:

    1. **Shrink, if it is too big**, by the smaller of the two axis ratios. The
       picture keeps its aspect and its rotation, and because both the
       rectangle and its bounding box scale about the same centre, only the
       ratio changes -- no new geometry has to be derived for a rotated one.
    2. **Then slide it back in.** Doing it the other way round would shove a
       too-large picture into place and then shrink it about a centre that is
       no longer where the user left it, so a picture that only needed
       shrinking would also appear to jump sideways.

    A picture that already fits gets a scale of 1 and no offset, so calling this
    repeatedly is harmless.
    """
    if region.width_mm <= 0 or region.height_mm <= 0:
        raise ValueError(f"填充区尺寸必须为正：{region.key}")

    box_w = rotated_bounds(centre, size, rotation_deg)
    bounds_w = box_w[2] - box_w[0]
    bounds_h = box_w[3] - box_w[1]

    scale = 1.0
    if bounds_w > region.width_mm:
        scale = min(scale, region.width_mm / bounds_w)
    if bounds_h > region.height_mm:
        scale = min(scale, region.height_mm / bounds_h)

    # The centre does not move when the picture is scaled about itself, so the
    # bounds shrink towards it by the same factor.
    cx, cy = centre
    half_w = bounds_w * scale / 2.0
    half_h = bounds_h * scale / 2.0
    x0, x1 = cx - half_w, cx + half_w
    y0, y1 = cy - half_h, cy + half_h

    rx0, ry0, rx1, ry1 = region.bounds
    dx = 0.0
    if x0 < rx0:
        dx = rx0 - x0
    elif x1 > rx1:
        dx = rx1 - x1
    dy = 0.0
    if y0 < ry0:
        dy = ry0 - y0
    elif y1 > ry1:
        dy = ry1 - y1

    return Fit(scale=scale, dx_mm=dx, dy_mm=dy)


def apply_contain(region: ArtworkRegion, placement: ImagePlacement) -> Fit:
    """Apply :func:`contain` to a placement and report what was changed.

    Flips are untouched: they mirror the picture inside its own rectangle and
    do not move an edge, so the bounding box used here is already correct for
    them.
    """
    fit = contain(
        region,
        (placement.x_mm, placement.y_mm),
        (placement.width_mm, placement.height_mm),
        placement.rotation_deg,
    )
    if fit.scale < 1.0:
        placement.width_mm *= fit.scale
        placement.height_mm *= fit.scale
    placement.x_mm += fit.dx_mm
    placement.y_mm += fit.dy_mm
    return fit


# ---------------------------------------------------------------- placement


def _contained_size(region: ArtworkRegion, aspect: float) -> tuple[float, float]:
    """The largest picture of this aspect that fits the region."""
    if aspect <= 0:
        raise ValueError("宽高比必须为正")
    width = region.width_mm
    height = width / aspect
    if height > region.height_mm:
        height = region.height_mm
        width = height * aspect
    return (width, height)


def artwork_placement(
    plan: ArtworkPlan,
    path: str,
    aspect: float,
    *,
    layer: str = ARTWORK_LAYER,
    processed_path: str = "",
    import_id: str = "",
) -> ImagePlacement:
    """Build the placement the plan asks for.

    ``aspect`` is the picture's width divided by its height. Both modes start
    centred in the region, which is the one position that is equally wrong in
    every direction and therefore the least surprising thing to hand the user.

    ``path`` may be empty when the caller only wants the geometry -- the
    previews run before a file exists, and the model does not mind.
    """
    region = plan.region
    if plan.mode is PlacementMode.WIDTH_FIT:
        width = region.width_mm
        height = width / aspect if aspect > 0 else 0.0
    else:
        width, height = _contained_size(region, aspect)
    return ImagePlacement(
        path=path,
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=width,
        height_mm=height,
        layer=layer,
        processed_path=processed_path,
        import_id=import_id,
    )


def kinds_for_operator(*, amiya: bool = False) -> tuple[ArtworkKind, ...]:
    """The artwork choices to offer, in order.

    精英1 is 阿米娅's alone -- no other operator has a separate 精英1
    illustration -- so offering it for everyone would be a control that cannot
    produce anything.
    """
    choices: Iterable[ArtworkKind] = (
        (ArtworkKind.ELITE0, ArtworkKind.ELITE1, ArtworkKind.ELITE2, ArtworkKind.SKIN)
        if amiya
        else (ArtworkKind.ELITE0, ArtworkKind.ELITE2, ArtworkKind.SKIN)
    )
    return tuple(choices)
