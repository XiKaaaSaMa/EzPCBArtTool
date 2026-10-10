"""The 通行证 (badge) card as a board template: its outline and its windows.

This is the "套用模板 - 通行证" entry. Applying it does two things: it makes the
board the card's shape, and it gives the artwork a *region* to be framed in.

The geometry is measured, not drawn by hand. Both the outline and the two
regions were taken out of the source project the user supplied
(``ProPrj_鸿雪_2026-09-25.epro2``) -- the outline from its board-outline layer,
the regions from the two filled rectangles that mark where artwork goes.
:mod:`tools.extract_badge_regions` re-derives them, so a new card can be
measured the same way instead of someone typing coordinates from a screenshot.

Why a template and not a preset shape: the card is a rectangle with cut corners
*and* a notch on one edge. :class:`pcbart.artboard.OutlineShape` has no preset
that can express that, which is what ``TEMPLATE`` was added for.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artboard import (
    ENGLISH_TOKEN,
    NAME_TOKEN,
    NUMBER_TOKEN,
    BoardOutlineSpec,
    CutoutElement,
    OutlineShape,
    TextElement,
)

__all__ = [
    "BADGE_WIDTH_MM",
    "BADGE_HEIGHT_MM",
    "BADGE_OUTLINE",
    "BADGE_REGIONS",
    "BadgeRegion",
    "TemplateText",
    "BADGE_TEXTS",
    "BADGE_CUTOUT",
    "BADGE_CUTOUTS",
    "BADGE_OUTLINE_ARCS",
    "badge_texts",
    "badge_cutouts",
    "badge_spec",
    "region_for_aspect",
    "register",
    "TEMPLATE_NAME",
]

BADGE_WIDTH_MM = 50.0
"""Card width. The template annotates it as 50.000 mm."""

BADGE_HEIGHT_MM = 100.0
"""Card height, annotated 100.000 mm in the template."""

_DRAWN_OUTLINE: tuple[tuple[float, float], ...] = (
    (-24.9365, 48.9189),
    (-23.9236, 49.9365),
    (23.8308, 49.9365),
    (24.9365, 49.0428),
    (24.9365, 40.7090),
    (23.1150, 38.8875),
    (23.1150, -6.1975),
    (24.9225, -8.0605),
    (24.9365, -48.8696),
    (23.8332, -49.9365),
    (-23.9150, -49.9365),
    (-24.9365, -48.8899),
)
"""The ring as drawn in the file: 49.873 x 99.873 mm.

That is the nominal 50 x 100 card *less half a line width on each side* -- the
board-outline layer draws a 0.127 mm centre line, and the drawn path sits inside
it. The template's own width dimension reads 50.000 mm, and the artwork regions
were placed against that nominal card, so the ring is scaled up to meet it
below. Using the drawn size would leave the short region hanging 0.05 mm off
the top edge.
"""

_SCALE_X = BADGE_WIDTH_MM / 49.873
_SCALE_Y = BADGE_HEIGHT_MM / 99.873

BADGE_OUTLINE: tuple[tuple[float, float], ...] = tuple(
    (round(x * _SCALE_X, 4), round(y * _SCALE_Y, 4)) for x, y in _DRAWN_OUTLINE
)
"""The card's outline, centred on the origin, in millimetres, counter-clockwise.

Twelve points is the whole shape: four cut corners of about 1.4 mm, and a
1.8 mm deep notch down the right-hand edge between y = 38.9 and y = -6.2.
"""

BADGE_OUTLINE_ARCS: tuple[float | None, ...] = (
    89.9791,
    None,
    -90.61,
    None,
    None,
    None,
    None,
    None,
    -94.3362,
    None,
    93.3779,
    None,
)
"""One arc sweep per outline segment; ``None`` means a straight edge.

The four corner values are the source file's own ``ARC`` angles.  Without
them the card's rounded corners become diagonal cuts -- the shape still spans
the right size, so only looking at the board shows the loss.
"""


@dataclass(frozen=True)
class BadgeRegion:
    """A window on the card where artwork is framed."""

    key: str
    label: str
    width_mm: float
    height_mm: float
    centre_x_mm: float
    centre_y_mm: float

    @property
    def aspect(self) -> float:
        """Width divided by height, which is the shape artwork is framed to."""
        return self.width_mm / self.height_mm

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        half_w = self.width_mm / 2.0
        half_h = self.height_mm / 2.0
        return (
            self.centre_x_mm - half_w,
            self.centre_y_mm - half_h,
            self.centre_x_mm + half_w,
            self.centre_y_mm + half_h,
        )


BADGE_REGIONS: tuple[BadgeRegion, ...] = (
    BadgeRegion(
        key="small",
        label="短区（38.861 × 56.324）",
        width_mm=38.861,
        height_mm=56.324,
        centre_x_mm=1.854,
        centre_y_mm=21.825,
    ),
    BadgeRegion(
        key="large",
        label="长区（42.526 × 99.626）",
        width_mm=42.526,
        height_mm=99.626,
        centre_x_mm=3.646,
        centre_y_mm=0.102,
    ),
)
"""The two windows, measured from the filled rectangles in the source project.

The short one takes a 精英0/精英1 portrait or a skin that suits it; the tall
one takes a 精英2 illustration or a skin drawn to fill it. Which one a given
picture suits is the user's call at framing time, so both are offered.
"""


def region_for_aspect(aspect: float, tolerance: float = 0.02) -> BadgeRegion | None:
    """The region whose shape matches ``aspect``, if one does.

    Used to pick a region when a picture arrives already cropped to one of the
    two ratios -- the figure library is cropped that way, so the matching
    window can be chosen without asking.
    """
    for region in BADGE_REGIONS:
        if abs(region.aspect - aspect) <= tolerance:
            return region
    return None


# --------------------------------------------------- what the blank card says
#
# The user asked for the blank template's *content* to appear on the canvas and
# be editable, not just its outline -- the wording is most of what a 通行证 is,
# and a card whose text could only be changed in the EDA would send the user
# back there for the one thing this tool is meant to do.
#
# The numbers below are measured from the two files in
# ``PCB艺术板工具素材库\通行证模板源文件\``, the same way the outline and the
# regions were. Two facts about the pair shaped this table:
#
# * **The two files say the same things.** Of the fourteen lines, twelve appear
#   in both, at positions that differ by at most 0.21 mm -- see
#   ``test_the_two_template_files_place_the_same_text``. The exceptions are the
#   阵营logo guide (small card only) and the invisible ``ARKNIGHTS-DWDB``
#   (below). One template therefore covers both files instead of two that would
#   drift apart, and the test fails if a future re-export moves a line far
#   enough to make that untrue.
# * **Three of the lines are placeholders for artwork**, not wording: 条形码,
#   干员二维码 and 阵营logo name the thing that gets drawn in that spot. They
#   are kept as editable text because that is the honest representation of what
#   the file holds today -- when the barcode and QR generators land they will
#   *replace* these lines rather than have to be given somewhere to go.
#
# ⚠️ ``ARKNIGHTS-DWDB`` sits at a font size of 0.003 mm in both files. It cannot
# be seen, cannot be clicked, and listing it would put a line in the content
# panel that does nothing when selected. It is left out and counted here rather
# than silently dropped.


@dataclass(frozen=True)
class TemplateText:
    """One line of the blank card, before it becomes a document element.

    A frozen spec rather than a live :class:`~pcbart.artboard.TextElement`
    because the template is applied more than once: handing out the same
    objects would make an edit on one board change the template every later
    board is copied from.
    """

    text: str
    x_mm: float
    y_mm: float
    size_mm: float
    rotation_deg: float
    layer: str
    font: str
    label: str


#: Layer display names, spelled as :mod:`pcbart.design_layers` spells them so
#: the canvas can look up a colour without a translation table.
_SILK_TOP = "顶层丝印层"
_SILK_BOTTOM = "底层丝印层"
_COPPER_TOP = "顶层铜"


BADGE_TEXTS: tuple[TemplateText, ...] = (
    TemplateText(
        # ``{英文名}`` rather than ``{代号}``: this line is set in the
        # template's ``Agency FB Negreta``, which draws Latin letters and
        # nothing else, so a Chinese name comes out as empty boxes. The English
        # name is what the design wants here -- and it is what makes the
        # template's own typeface usable.
        text=ENGLISH_TOKEN,
        x_mm=-13.633,
        y_mm=-12.928,
        size_mm=6.0,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="Agency FB Negreta",
        label="干员代号",
    ),
    TemplateText(
        text=f"ARKNIGHTS-{NUMBER_TOKEN}",
        x_mm=-13.760,
        y_mm=-15.468,
        size_mm=1.6,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="Dubai Regular",
        label="编号（横排）",
    ),
    TemplateText(
        text=f"ARKNIGHTS-{NUMBER_TOKEN}",
        x_mm=-20.110,
        y_mm=-37.947,
        size_mm=1.0,
        rotation_deg=270.0,
        layer=_SILK_TOP,
        font="Dubai Regular",
        label="编号（竖排）",
    ),
    TemplateText(
        text="STAFF",
        x_mm=-13.633,
        y_mm=-17.881,
        size_mm=1.6,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="Dubai Regular",
        label="STAFF",
    ),
    TemplateText(
        text="Operator of Rhodes Island",
        x_mm=-13.760,
        y_mm=-42.519,
        size_mm=0.8,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="default2",
        label="副标题",
    ),
    TemplateText(
        text="SUPPORTER \\ RHODES ISLAND",
        x_mm=-13.760,
        y_mm=-44.805,
        size_mm=0.8,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="default2",
        label="职业阵营行",
    ),
    TemplateText(
        text="职业",
        x_mm=-12.886,
        y_mm=-35.444,
        size_mm=3.048,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="黑体",
        label="职业（放图标处）",
    ),
    TemplateText(
        text="条形码",
        x_mm=-23.949,
        y_mm=-25.729,
        size_mm=3.048,
        rotation_deg=270.0,
        layer=_SILK_TOP,
        font="黑体",
        label="条形码（待生成）",
    ),
    TemplateText(
        text="阵营logo",
        x_mm=2.533,
        y_mm=-31.135,
        size_mm=5.0,
        rotation_deg=0.0,
        layer=_COPPER_TOP,
        font="黑体",
        label="阵营logo（仅小卡）",
    ),
    TemplateText(
        text="RHODES ISLAND INC.",
        x_mm=-21.888,
        y_mm=-10.515,
        size_mm=1.0,
        rotation_deg=270.0,
        layer=_SILK_BOTTOM,
        font="Bahnschrift SemiBold",
        label="RHODES ISLAND INC.（竖排）",
    ),
    TemplateText(
        text="干员二维码",
        x_mm=-19.245,
        y_mm=-27.285,
        size_mm=1.0,
        rotation_deg=0.0,
        layer=_SILK_BOTTOM,
        font="黑体",
        label="二维码（待生成）",
    ),
    TemplateText(
        text="UID:用户填写",
        x_mm=-10.204,
        y_mm=-50.266,
        size_mm=2.2,
        rotation_deg=0.0,
        layer=_SILK_BOTTOM,
        font="Bahnschrift SemiBold",
        label="UID",
    ),
    TemplateText(
        text="©HYPERGRYPH",
        x_mm=-13.887,
        y_mm=-49.631,
        size_mm=1.2,
        rotation_deg=0.0,
        layer=_SILK_TOP,
        font="Franklin Gothic Medium",
        label="版权",
    ),
)
"""Every visible line on the blank card, in the order the panel lists them."""

#: The card's openings, measured from the two source files.  The two round
#: holes are ``VIA`` records in the source (3.56 mm drill); the source stores
#: the long opening as a rectangle, but the template draws it as a routed
#: stadium and the user asked for the three shapes to be slot regions.
BADGE_CUTOUTS: tuple[tuple[str, float, float, float, float, str, str], ...] = (
    ("hole_left", -21.012, 46.127, 3.56, 3.56, "circle", "左上圆孔"),
    ("hole_right", 20.898, 46.127, 3.56, 3.56, "circle", "右上圆孔"),
    ("lanyard", -0.044, 46.022, 24.130, 2.832, "obround", "挂绳挖槽"),
)
"""``(key, centre_x, centre_y, width, height, shape, label)``."""

#: Backwards-compatible name for the long lanyard opening, used by callers
#: that predate the two corner holes.
BADGE_CUTOUT: tuple[str, float, float, float, float] = (
    "lanyard",
    -0.044,
    46.022,
    24.130,
    2.832,
)


def badge_texts() -> list[TextElement]:
    """Fresh text elements for a newly applied card.

    New objects every call: the template is applied to more than one board, and
    sharing instances would make an edit on the first board rewrite the second.
    """
    return [
        TextElement(
            text=item.text,
            x_mm=item.x_mm,
            y_mm=item.y_mm,
            size_mm=item.size_mm,
            rotation_deg=item.rotation_deg,
            layer=item.layer,
            font=item.font,
            label=item.label,
        )
        for item in BADGE_TEXTS
    ]


def badge_cutouts() -> list[CutoutElement]:
    """Fresh cutout elements for a newly applied card."""
    return [
        CutoutElement(
            key=key,
            x_mm=x_mm,
            y_mm=y_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            shape=shape,
            label=label,
        )
        for key, x_mm, y_mm, width_mm, height_mm, shape, label in BADGE_CUTOUTS
    ]


TEMPLATE_NAME = "通行证"
"""The base name the board panel lists."""

STAGES: tuple[tuple[str, str], ...] = (
    ("精0", "精0／精1／小皮肤：短立绘窗（38.861 × 56.324 mm）"),
    ("精2", "精2／大皮肤：满高立绘窗（42.526 × 99.626 mm）"),
)
"""The two card layouts, as ``(stage, description)``.

⚠️ **They are not the same card with a different picture in it.** The two
source files carry different fixed artwork: the short layout has the 阵营 logo
block and its own hatching (327 decorations), the tall one replaces that with a
full-height illustration window (187). Picking the wrong layout prints the other
card's border and pattern.

The outline and the two openings are the same in both, so only the decoration
set and which opening is offered to the user change.
"""


def template_name(stage: str = "精0") -> str:
    """The template name a given stage registers under.

    ``精0`` keeps the bare :data:`TEMPLATE_NAME` so every existing draft, test
    and saved project still resolves to the short card -- only the new option
    carries a suffix.
    """
    return TEMPLATE_NAME if stage == "精0" else f"{TEMPLATE_NAME} · {stage}"


def stage_of(template: str) -> str:
    """The stage a template name refers to, defaulting to ``精0``.

    Unrecognised names -- an older draft, or one written before the stages
    existed -- are the short card, which is what every card was until now.
    """
    for stage, _description in STAGES:
        if template == template_name(stage):
            return stage
    return "精0"


def is_badge_template(template: str) -> bool:
    """Whether a board's template is any layout of this card.

    ⚠️ **Compare through this, never against :data:`TEMPLATE_NAME`.** Every
    place that asks "is this the card" has to accept both stages, or the 精2
    card is treated as a plain board: no decorations, no regions, no panel.
    """
    return any(template == template_name(stage) for stage, _ in STAGES)


def register() -> None:
    """Put both card layouts in the board panel's template list.

    Called once when the interface loads. Registering is idempotent, so calling
    it again after a reload is harmless.
    """
    from .artboard import register_template

    for stage, _description in STAGES:
        register_template(template_name(stage), badge_spec(stage))


def badge_spec(stage: str = "精0") -> BoardOutlineSpec:
    """A board spec for one card layout, ready to assign to a document.

    The outline is the same either way -- only :func:`stage_of`'s decorations
    differ -- but the template name records which layout was chosen so a saved
    card keeps it.
    """
    return BoardOutlineSpec(
        shape=OutlineShape.TEMPLATE,
        template=template_name(stage),
        template_outline=BADGE_OUTLINE,
        template_arcs=BADGE_OUTLINE_ARCS,
        width_mm=BADGE_WIDTH_MM,
        height_mm=BADGE_HEIGHT_MM,
    )
