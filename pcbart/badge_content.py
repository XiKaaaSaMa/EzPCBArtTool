"""Put an operator's artwork onto the 通行证 card.

**The QR symbol and the barcode so far.** The card reserves four spots for
something that is about a person rather than part of the template -- a QR
symbol, a barcode, a profession icon and a faction logo. This module used to
insert all four at once and every position was wrong, so the remaining two are
left out until each one has been checked against the block the user drew for
it.

**The regions the user drew are the instruction; the text is only its label.**
Each spot is marked in the source project by a filled rectangle, and the text
naming it (``干员二维码`` and friends) is anchored on that rectangle's *edge*.
Positioning from the text looks close and is wrong -- measured 3.3 mm off for
the QR symbol -- so the coordinates in :data:`ART_REGIONS` come from the filled
shapes themselves.

**A region measures the code, never the quiet zone.** Both codes reserve clear
space around themselves that a drawn block cannot mark -- there is nothing to
draw there. The QR block is a 45-degree square of side 3.1997 mm and measures
the symbol's own 21x21 modules, so its finder patterns land on the block's
corners and the transparent 4-module quiet zone falls outside. The barcode
block is 39.245 mm along the bars and measures the 79 bar modules, so its first
and last bar sit on the frame's edges and the 10-module quiet zone falls
outside. Sizing either *file* to its block instead leaves a fat empty margin
inside it -- the defect the user reported on the QR symbol.

**Both symbols are transparent, and go on the silkscreen.** Their backgrounds
are not white but nothing at all, so the template's own artwork shows through
between the marks; a white field would punch a hole in the card. The QR symbol
is on the bottom silkscreen and the barcode on the top, as the template's own
marker blocks specify. The bitmaps are written to the cache directory rather
than into the library so a placed card keeps working if the library is
regenerated.

**Raw, not traced.** Neither placement carries a ``processed_path``: giving
them one would send the symbols through the line-art pipeline, and a code that
has been "cleaned up" no longer decodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import badge_codes
from .artboard import ImagePlacement
from .paths import badge_asset_dir, cache_dir

__all__ = [
    "CACHE_NAME",
    "QR_LAYER",
    "QR_ROTATION_DEG",
    "QR_MIN_FEATURE_AREA_MM2",
    "QR_SIMPLIFY_MM",
    "QR_COORD_SNAP_MM",
    "QR_MODULE_PX",
    "BARCODE_LAYER",
    "BARCODE_ROTATION_DEG",
    "BARCODE_MODULE_PX",
    "PROFESSION_LAYER",
    "CACHE_REVISION",
    "ART_IMPORT_ID",
    "ART_ELEMENT_LABELS",
    "HIDDEN_DECORATION_LABELS",
    "ART_REGIONS",
    "ArtRegion",
    "art_region",
    "marker_size_pairs",
    "vectorize_params_for",
    "BadgeArt",
    "MissingArt",
    "badge_art",
]

CACHE_NAME = "badge_codes"
"""Cache subfolder holding the generated code bitmaps."""

QR_MODULE_PX = 32
"""Pixels per module in the cached symbol.

Whole, not fractional: the exporter re-rasterises the file, and a symbol whose
total pixel count is not a multiple of the module count has module edges that
land between pixels.

**32 is set by the export, not by the screen.** The bitmap is what the exporter
traces back into geometry, and 12 pixels per module put only 78.8 px/mm behind a
0.152 mm module -- enough to look right and not enough to round-trip: measured
**11 of 841 module centres landed on the wrong side of a traced edge**, so bare
cells came out filled. At 32 px/module (210 px/mm) the same check measured
**0 of 841**. Raising it further changes nothing, so this is the knee of the
curve. The file stays small -- it is binary and flat.
"""

BARCODE_MODULE_PX = 32
"""Pixels per module in the cached barcode.

Same reasoning as :data:`QR_MODULE_PX`: the bitmap is what the exporter traces
back into geometry, so the module has to be many whole pixels wide, and a
fractional module moves bar edges between pixels and changes which modules the
exported geometry covers.
"""

BARCODE_LAYER = "顶层丝印层"
"""Where the barcode is printed -- the layer its marker block is drawn on."""

PROFESSION_LAYER = "顶层丝印层"
"""Where the profession plate goes -- the layer its marker block is drawn on.

Same side as the barcode, and for the same reason: the plate *is* ink, and the
top silkscreen is where the template puts the card's marks.
"""

FACTION_LAYER = "顶层铜"
"""Where the faction logo goes -- the layer its own marker block is drawn on.

The logo is a shape in the copper, not an ink mark: the block the user drew for
it sits on the top copper, and that slot has no other copper in it, so the logo
is what fills it.
"""

RHODES_FACTION = "罗德岛"
"""The faction whose crest the card already carries on its silkscreen.

Used to decide the copper slot's *automatic* value: a 罗德岛 card leaves it
empty rather than printing the same crest twice. An explicit choice overrides
this, so a user who wants it in the copper as well can still have it.
"""

EMBLEM_KIND = "emblem"
"""The card's 罗德岛 crest, which the panel may switch off.

Not an artwork *slot*: every card carries it on the **top silkscreen**, fixed
like the border. It is addressed by the same dictionary of choices as the four
slots only because the user asked to be able to remove it the same way.
"""

CHOICE_AUTO = ""
"""Use whatever the operator's own record says."""

CHOICE_OFF = "none"
"""The user removed this element from this card."""

ART_KINDS = ("qr", "barcode", "profession", "faction", EMBLEM_KIND)
"""Every element the panel can override, in the order it lists them."""

BARCODE_ROTATION_DEG = 270.0
"""How the card draws the barcode.

The strip is 4.941 mm across and 39.245 mm along the bars, so it has to be
turned a quarter turn to fill the frame. 270 (not 90) makes it read in the
same direction as the ``ARKNIGHTS-编号`` line beside it: the text is stored at
270 degrees and measured to run top-to-bottom with its glyph tops facing right,
and a barcode turned the same way reads top-to-bottom as well.
"""

CACHE_REVISION = "v8-qr-barcode-profession"
"""Bumped whenever a cached code bitmap's own appearance changes.

The cache is keyed on the operator, so without this a card placed before the
symbol became transparent would keep drawing the old white-backed file forever
and the fix would look like it had not been applied.

``v5`` changed the QR sizing: the *code area* now measures the drawn block and
the quiet zone falls outside it, so the module is 0.1524 mm instead of
0.1103 mm. ``v6`` raised the QR bitmap to 32 pixels per module so the trace
round-trips. ``v7`` adds the barcode and renames the revision so one key covers
both codes.
"""

QR_LAYER = "底层丝印层"
"""Where the QR symbol is printed.

The user asked for the bottom silkscreen: with a transparent background the
symbol is a light-on-dark mark, and the bottom side is the one the card shows.
"""

QR_ROTATION_DEG = 45.0
"""The symbol is turned 45 degrees to sit square in its region's corners."""

QR_MIN_FEATURE_AREA_MM2 = 0.001
"""Smallest island and hole the QR tracer may keep, in square millimetres.

The default in :class:`~pcbart.vectorize.VectorizeParams` is **0.02**, which is
fine for line art and marginal here: a module is now 0.1524 mm across, so it
covers 0.0232 mm² -- barely over the default, and a module clipped by the
tracer's own rounding drops under it. One thousandth of a square millimetre
keeps every module with room to spare and still rejects a stray pixel.

⚠️ **This was 0.1103 mm (0.0122 mm²) before the code area was enlarged to meet
the block**, when the default really did delete every module; the value is kept
because the margin, not the bare pass, is what makes the export reliable.
"""

QR_SIMPLIFY_MM = 0.005
"""Douglas-Peucker tolerance for the QR tracer.

Far below the line-art default of 0.08 mm. A tolerance that is a large fraction
of a module's width straightens each run of modules into a rectangle: measured
with the old 0.1103 mm module, 0.08 mm (72% of a module) inflated the traced
symbol from 2.91 mm² to 6.28 mm² by filling whole areas between modules. Five
microns keeps each module's own corners.
"""

QR_COORD_SNAP_MM = 0.0005
"""Vertex-merging tolerance for the QR tracer.

The default 0.002 mm is comfortable for line art and small next to a module.
Snapping coordinates that far apart can round module edges onto each other, so
separate modules merge and **bare cells get filled in**. Half a micron is below
what the board can resolve and keeps the grid apart.
"""

ART_IMPORT_ID = "badge-art"
"""Groups the placements this module adds, so re-inserting replaces them."""

ART_ELEMENT_LABELS = (
    "二维码（待生成）",
    "条形码（待生成）",
    "职业（放图标处）",
    "阵营logo（仅小卡）",
    "干员二维码",
)
"""Template text lines that *name* artwork rather than say anything.

They exist so the blank file is self-describing; once the real thing is on the
card they are removed, not kept alongside it.

The last entry is **historical**: the line was labelled ``干员二维码`` before it
was renamed, and a board saved in that window carries the older label. Matching
on the label alone would leave that board's placeholder sitting on the symbol.
"""

HIDDEN_DECORATION_LABELS = (
    "立绘区",
    "干员二维码区",
    "条形码区",
    "职业图标区",
    "阵营logo区",
)
"""The filled blocks the user drew to mark where artwork goes.

They are scaffolding: they say where a thing *will* be, and once the thing is
there they are removed from the card. The list is applied when the fixed
decorations are drawn and written, so the source project keeps its blocks and
re-running the extractor stays a faithful copy.
"""


ART_REGION_SIZE_TOLERANCE_MM = 0.05
"""How far a marker rectangle may differ from the recorded region size.

The source file's arcs and coordinates are quoted to 0.001 mm, but a marker
that was redrawn by hand can be a few hundredths out. Five hundredths of a
millimetre is far below the gap to any other rectangle on the card, so it
tolerates a redrawn marker without ever matching the wrong shape.
"""


class MissingArt(RuntimeError):
    """Raised when the card cannot be built from the library on this machine."""


@dataclass(frozen=True)
class BadgeArt:
    """What ended up on the card."""

    placements: tuple[ImagePlacement, ...]
    qr: ImagePlacement | None = None
    barcode: ImagePlacement | None = None
    profession: ImagePlacement | None = None
    faction: ImagePlacement | None = None

    def __bool__(self) -> bool:
        return bool(self.placements)


@dataclass(frozen=True)
class ArtRegion:
    """One filled region the user drew to say "the artwork goes here".

    The source project marks each of the card's operator-specific spots with a
    filled rectangle, and the *text* that names the spot is anchored on the
    region's edge rather than its centre -- ``干员二维码`` sits 3.3 mm up and to
    the right of the symbol's box. Positioning from the text therefore looks
    close and is wrong, which is exactly what the first version of this module
    did. The region is the instruction; the text is only its label.
    """

    key: str
    label: str
    centre_x_mm: float
    centre_y_mm: float
    width_mm: float
    height_mm: float
    layer: str
    rotation_deg: float = 0.0
    """How the card draws the thing in this region.

    The QR region carries the drawn marker's rotation because the block the user
    drew for it is already a diamond: the symbol has to be turned the same 45
    degrees to land in it.
    """

    @property
    def centre(self) -> tuple[float, float]:
        return (self.centre_x_mm, self.centre_y_mm)

    @property
    def marker_sizes(self) -> tuple[tuple[float, float], ...]:
        """``(width, height)`` pairs this marker can appear as.

        A rotated block reads differently in the two copies of the template we
        hold: the extracted decoration stores the ring as *drawn*, so the QR
        marker comes back as its rotated bounding box (4.525 mm square), while
        the raw source record stores the un-rotated rectangle the rotation is
        applied to (3.1997 mm square). Both are legitimate and both have to be
        recognised, or filtering the marker by size silently misses it in one of
        the two lists.
        """
        box = (self.width_mm, self.height_mm)
        if abs(self.rotation_deg) < 1e-9:
            return (box,)
        radians = math.radians(self.rotation_deg)
        cos = abs(math.cos(radians))
        sin = abs(math.sin(radians))
        return (
            box,
            (
                self.width_mm * cos + self.height_mm * sin,
                self.width_mm * sin + self.height_mm * cos,
            ),
        )


def vectorize_params_for(placement, params=None):
    """Vectorize parameters suited to ``placement``'s own feature size.

    The badge's codes carry no ``processed_path`` and are marked with
    :data:`ART_IMPORT_ID`, which is what tells the exporter they are generated
    at a module size far below what the line-art defaults assume. Everything
    else keeps the caller's parameters untouched -- this is a correction for one
    kind of artwork, not a new global default.
    """
    from dataclasses import replace

    from .vectorize import VectorizeParams

    if getattr(placement, "import_id", "") != ART_IMPORT_ID:
        return params or VectorizeParams()
    base = params or VectorizeParams()
    return replace(
        base,
        simplify_mm=min(base.simplify_mm, QR_SIMPLIFY_MM),
        min_island_area_mm2=min(base.min_island_area_mm2, QR_MIN_FEATURE_AREA_MM2),
        min_hole_area_mm2=min(base.min_hole_area_mm2, QR_MIN_FEATURE_AREA_MM2),
        coord_snap_mm=min(base.coord_snap_mm, QR_COORD_SNAP_MM),
    )


def marker_size_pairs() -> tuple[tuple[float, float, str], ...]:
    """Every ``(width, height, label)`` a positioning marker can appear as.

    One entry per marker per orientation it is stored in, so the same helper
    serves the extracted decorations (rings as drawn) and the raw records
    (rectangles before their rotation). Label blocks are actual wording and are
    never included; only the filled blocks are.
    """
    from .badge_card import BADGE_REGIONS

    rows: list[tuple[float, float, str]] = []
    for region in ART_REGIONS:
        rows += [
            (width, height, f"{region.label}区")
            for width, height in region.marker_sizes
        ]
    for region in BADGE_REGIONS:
        rows.append((region.width_mm, region.height_mm, "立绘区"))
    return tuple(rows)


#: Measured out of the source project's own filled regions, matched by size and
#: position. ``tools/extract_badge_regions.py`` re-derives the two illustration
#: windows the same way; these four are the operator-specific spots and are
#: listed here so the panel, the canvas and the exporters agree on where the
#: artwork belongs without any of them reading a screenshot.
ART_REGIONS: tuple[ArtRegion, ...] = (
    ArtRegion(
        key="qr",
        label="干员二维码",
        centre_x_mm=-21.7279,
        centre_y_mm=-26.7969,
        width_mm=3.1997,
        height_mm=3.1997,
        layer="bottom_silk",
        rotation_deg=45.0,
    ),
    ArtRegion(
        key="barcode",
        label="条形码",
        centre_x_mm=-22.482,
        centre_y_mm=-28.839,
        width_mm=4.941,
        height_mm=39.245,
        layer="top_silk",
    ),
    ArtRegion(
        key="profession",
        label="职业",
        centre_x_mm=-9.960,
        centre_y_mm=-34.128,
        width_mm=7.600,
        height_mm=7.601,
        layer="top_silk",
    ),
    ArtRegion(
        key="faction_logo",
        label="阵营logo",
        centre_x_mm=11.470,
        centre_y_mm=-27.439,
        width_mm=24.321,
        height_mm=21.092,
        layer="top_copper",
    ),
)
"""The four regions that take operator-specific artwork, in source order."""

_REGIONS_BY_KEY = {region.key: region for region in ART_REGIONS}


def art_region(key: str) -> ArtRegion:
    """One region by key, so callers do not index the tuple by position."""
    try:
        return _REGIONS_BY_KEY[key]
    except KeyError as exc:  # pragma: no cover - a typo in this module
        raise KeyError(f"没有这个卡面区域：{key}") from exc


def _cache_png(
    code_dir: str,
    page: str,
    *,
    render,
) -> Path:
    """The cached bitmap for one code, generated on first use.

    **Keyed on the page name, not the operator name.** 暮落 and 暮落(集成战略)
    share a display name and have completely different codes; keying on the
    operator name served one operator's symbol for both.

    The path also carries :data:`CACHE_REVISION`. Bumping that string retires
    every cached symbol at once, which is what a change to the symbol's own
    appearance needs: without it a card placed before the background became
    transparent would keep drawing the old white-backed file, and the fix would
    look like it had not been applied.
    """
    safe = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in page)
    target = cache_dir(CACHE_NAME) / CACHE_REVISION / code_dir / f"{safe}.png"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    render().save(target, dpi=(600, 600))
    return target


def resolved_choice(board, kind: str) -> str:
    """What the user asked for on one element of ``board``.

    ``""`` means automatic -- the operator's own record -- and is what a board
    that predates the field, or one the panel has not touched, reports.
    """
    choices = getattr(board, "art_choices", None) or {}
    value = choices.get(kind, "")
    return str(value or "").strip()


def choose_faction(operator, board):
    """The faction whose logo this card shows, or ``""`` for none.

    **Automatic means "use the operator's own faction, except for 罗德岛".**
    The 罗德岛 crest already sits on the card's silkscreen, so putting the same
    mark in the copper slot underneath would print it twice; 罗德岛 members and
    operators with no faction therefore get an empty slot unless the user picks
    one explicitly.

    Any explicit choice -- including picking 罗德岛 -- is honoured.
    """
    choice = resolved_choice(board, "faction")
    if choice == CHOICE_OFF:
        return ""
    if choice:
        return choice
    faction = str(getattr(operator, "faction", "") or "").strip()
    return "" if faction == RHODES_FACTION else faction


def choose_profession(operator, board) -> str:
    """The profession icon's asset path, or ``""`` for none.

    Automatic uses the operator's own record. An explicit choice names a
    profession, and the library file is looked up from whichever operator has
    one -- the eight files are fixed, so any of them can be reached from any
    card.
    """
    choice = resolved_choice(board, "profession")
    if choice == CHOICE_OFF:
        return ""
    if choice:
        return profession_icon_for(choice)
    return str(getattr(operator, "class_icon_black", "") or "").strip()


@lru_cache(maxsize=8)
def profession_icon_for(profession: str) -> str:
    """The library-relative icon path for a profession name, or ``""``.

    Read from the roster rather than built from a filename pattern: the
    pattern lives in the fetch tool, and a card naming a profession the library
    does not carry has to come back empty rather than invent a path.
    """
    from . import operators as operator_db

    wanted = str(profession or "").strip()
    if not wanted:
        return ""
    for person in operator_db.database().operators:
        if person.profession == wanted and person.class_icon_black:
            return person.class_icon_black
    return ""


@lru_cache(maxsize=8)
def faction_logo_for(faction: str) -> str:
    """The library-relative logo path for a faction name, or ``""``."""
    from . import operators as operator_db

    wanted = str(faction or "").strip()
    if not wanted:
        return ""
    for person in operator_db.database().operators:
        if person.faction == wanted and person.faction_logo:
            return person.faction_logo
    return ""


def badge_art(operator=None, *, board=None, number=None) -> BadgeArt:
    """Build the four card elements for one operator.

    ``operator`` is a resolved :class:`~pcbart.operators.Operator`, or ``None``
    to read the person named on ``board`` out of the database. ``None`` together
    with no name on the board returns an empty plan rather than raising: the
    panel can call this before anyone has been chosen.

    ``number`` overrides the 情报编号 the barcode encodes. ``None`` means "use
    the operator's own", which is what a caller that only knows the person
    wants; an **empty string** means "no barcode", which is how a card whose
    number the user cleared prints no strip. The panel passes the board's own
    value so an edited number is what actually reaches the board -- the QR
    symbol is unaffected, since it encodes the operator's wiki page.

    Every element can also be overridden or removed through
    ``board.art_choices`` -- see :func:`resolved_choice`.
    """
    from . import operators as operator_db

    if operator is None:
        name = getattr(board, "operator_name", "") or ""
        page = getattr(board, "operator_page", "") or ""
        if not name and not page:
            return BadgeArt((), None, None)
        database = operator_db.database()
        # The page name is the identity and the display name is not: 暮落 and
        # 暮落(集成战略) share a name, and resolving by name gave the second
        # person the first person's QR code. A board with no page name (an
        # older draft, or one built by hand) still resolves by name.
        operator = database.find_page(page) or database.find(name)
        if operator is None:
            raise MissingArt(f"数据库里没有「{name or page}」")

    from .badge_raster import (
        barcode_raster_transparent,
        profession_icon_raster,
        qr_raster_transparent,
    )

    library = badge_asset_dir()
    if not library.is_dir():
        raise MissingArt(f"素材库不在：{library}")

    show_qr = resolved_choice(board, "qr") != CHOICE_OFF
    qr = _qr_placement(operator, qr_raster_transparent) if show_qr else None

    barcode = _barcode_placement(
        operator, barcode_raster_transparent, number=number
    )
    if resolved_choice(board, "barcode") == CHOICE_OFF:
        barcode = None

    profession = _profession_placement(
        operator, profession_icon_raster, library, board=board
    )
    faction = _faction_placement(operator, library, board=board)
    placements = tuple(
        item for item in (qr, barcode, profession, faction) if item is not None
    )
    return BadgeArt(
        placements=placements,
        qr=qr,
        barcode=barcode,
        profession=profession,
        faction=faction,
    )


def _qr_placement(operator, render) -> ImagePlacement:
    """The operator's QR symbol, on the block's corners.

    **The region measures the symbol's own modules, quiet zone excluded.** The
    three finder patterns have to land on the drawn diamond's corners, so the
    code area -- not the bitmap -- is the block's side; the transparent
    quiet zone falls outside the block. Sizing the whole bitmap to the block
    instead left the finder patterns 4 modules short of every corner, which
    reads as a fat empty margin around the code.

    The 4-module quiet zone is still in the bitmap: a scanner needs it, it is
    transparent rather than absent, and it sits outside the block.
    """
    region = art_region("qr")
    code_mm = min(region.width_mm, region.height_mm)
    code = badge_codes.qr_for_operator(operator.page, code_mm=code_mm)
    path = _cache_png(
        "qr",
        operator.page,
        render=lambda: render(code.layout, module_px=QR_MODULE_PX),
    )
    return ImagePlacement(
        path=str(path),
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=code.layout.area_mm,
        height_mm=code.layout.area_mm,
        rotation_deg=QR_ROTATION_DEG,
        layer=QR_LAYER,
        import_id=ART_IMPORT_ID,
        kind="qr",
    )


def _faction_placement(operator, library, *, board=None) -> ImagePlacement | None:
    """The chosen faction logo, centred in its block on the top copper.

    **The logo's ink is matched to the block's width.** The user asked for
    "水平边长度与定位区水平边长度相同" and then settled the ambiguity -- the
    library's logos carry a lot of transparent padding (73-84% of a 510x510
    canvas is empty), so sizing the *canvas* to the block would have shrunk the
    mark to a fraction of its slot while the square filled it. The rule is
    therefore applied to the **ink**: the mark's own bounding box is measured
    and scaled so its width equals the block's 24.321 mm, with the height
    following the mark's own aspect.

    **No overflow is allowed, and none is needed.** Under that rule the tallest
    logo (红松骑士团) would want 53.9 mm against a 21.09 mm slot, so anything
    taller than the block is scaled down until it fits -- the width then comes
    out narrower than the block rather than running over the card. Measured over
    all 48 logos: 34 hit the width exactly and 14 are height-limited.

    ⚠️ **The block's own rectangle is the only thing in the area.** The author's
    sample artwork in that slot is filtered out as scaffolding (see
    ``badge_art.SAMPLE_EMBLEM_BOX``), and the surrounding silkscreen does not
    reach into it -- so nothing can collide once the demo comes off.

    The layer is the marker block's own: **top copper**, not silkscreen.

    **Which faction, though, is not always the operator's own.** The 罗德岛
    crest already sits on the silkscreen above this slot, so a 罗德岛 member
    would print the same mark twice; the slot stays empty for them and for
    anyone with no faction, unless the user picks one. See
    :func:`choose_faction`.

    ⚠️ **The tall card carries no faction logo at all.** The 精2 illustration
    window is 42.526 x 99.626 mm and covers the slot completely -- measured:
    the two rectangles overlap over 512.98 mm², which is 100% of the slot. The
    source template agrees: the tall layout has neither the marker block nor
    the ``阵营logo`` wording, so there is no copper slot for the mark to fill
    and no place a mark could go without printing on top of the picture.
    This method returning a placement anyway put a logo across the artwork of
    every 精2 card. The stage comes off the board's template name; a board with
    no template (a hand-built card, or an older draft) is the short layout,
    which is what every card was until the stages existed.
    """
    if board is not None:
        from .badge_card import stage_of, is_badge_template

        template = getattr(getattr(board, "outline", None), "template", "") or ""
        if is_badge_template(template) and stage_of(template) != "精0":
            return None

    faction = choose_faction(operator, board) if board is not None else str(
        getattr(operator, "faction", "") or ""
    ).strip()
    if not faction:
        return None
    relative = faction_logo_for(faction)
    if not relative:
        raise MissingArt(f"素材库里没有「{faction}」的阵营 logo")
    source = library / relative
    if not source.is_file():
        raise MissingArt(f"阵营 logo 不在：{source}")

    region = art_region("faction_logo")

    from PIL import Image

    with Image.open(source) as image:
        alpha = image.convert("RGBA").getchannel("A")
    box = alpha.getbbox()
    if box is None:
        raise MissingArt(f"阵营 logo 是空图：{source}")

    ink_w = box[2] - box[0]
    ink_h = box[3] - box[1]
    aspect = ink_h / ink_w if ink_w else 1.0

    # Width to the block, then never taller than the block.
    width_mm = region.width_mm
    height_mm = width_mm * aspect
    if height_mm > region.height_mm:
        height_mm = region.height_mm
        width_mm = height_mm / aspect if aspect else region.width_mm

    path = _cache_png(
        "faction",
        f"faction-{faction}",
        render=lambda: _cropped_logo(source, box),
    )
    return ImagePlacement(
        path=str(path),
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        rotation_deg=region.rotation_deg,
        layer=FACTION_LAYER,
        import_id=ART_IMPORT_ID,
        kind="faction",
    )


def _cropped_logo(source, box):
    """The logo's ink only, so the placement's size means the mark's size.

    The library files are square canvases with the mark floating in the middle;
    keeping that padding would make "width = the block's width" describe the
    empty canvas rather than the logo. Cropping here, once, is what lets the
    canvas, the ``.epro2`` writer and the Gerber writer all read one size.
    """
    from PIL import Image

    with Image.open(source) as image:
        return image.convert("RGBA").crop(box)


def _profession_placement(
    operator, render, library, *, board=None
) -> ImagePlacement | None:
    """The chosen class icon: a silkscreen plate with the glyph cut out.

    **The plate fills the block exactly, and the glyph is a hole in it.** The
    user asked for "背景部分丝印，图标部分镂空" -- ink everywhere the glyph is
    not, so the bare board shows through as the symbol. The library's ``黑``
    variant is already that picture (opaque dark field, light glyph); this
    turns its light pixels into transparency so the exporter writes the plate
    as the filled shape and the glyph as its hole.

    Sized to the region's **own two sides** rather than to its shorter one: the
    block is 7.600 x 7.601 mm, a rounding artefact of the file, and stretching
    the square icon to both keeps the plate flush with the drawn edges.

    Three operators in the roster have no profession at all (博士, 文月,
    魏彦吾 have a blank 职业 column). No icon is placed for them rather than
    guessing a class -- and the user can pick one anyway, which is the point of
    the panel's override.
    """
    relative = (
        choose_profession(operator, board)
        if board is not None
        else str(getattr(operator, "class_icon_black", "") or "").strip()
    )
    if not relative:
        return None
    source = library / relative
    if not source.is_file():
        raise MissingArt(f"职业图标不在：{source}")

    # The choice is a profession name; the asset path is how it was found. The
    # cache key wants the name, so it does not fragment across a roster edit.
    profession = resolved_choice(board, "profession") if board is not None else ""
    if not profession or profession == CHOICE_OFF:
        profession = str(getattr(operator, "profession", "") or "").strip()

    region = art_region("profession")

    from PIL import Image

    with Image.open(source) as image:
        raster = render(image)

    # Keyed on the **profession**, not the operator: every 术师 shares one icon,
    # so keying on the person would write 400 identical files. The profession
    # name is what the file is chosen by, so the key and the content agree.
    path = _cache_png(
        "profession",
        f"profession-{profession or source.stem}",
        render=lambda: raster,
    )
    return ImagePlacement(
        path=str(path),
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=region.width_mm,
        height_mm=region.height_mm,
        rotation_deg=region.rotation_deg,
        layer=PROFESSION_LAYER,
        import_id=ART_IMPORT_ID,
        kind="profession",
    )


def _barcode_placement(operator, render, *, number=None) -> ImagePlacement | None:
    """The operator's 情报编号 as a Code 128 strip, or ``None`` if they have none.

    ``number`` is the value to encode; ``None`` falls back to the operator's own
    record. An empty value -- whether from ``number=""`` or from an operator who
    simply has no 编号 -- yields no strip at all, which the card tolerates: 24
    operators have no number and still carry their QR symbol.

    **The region measures the bars, not the file.** The frame in the source
    artwork can only mark the bars -- the ten-module quiet zone on each end is
    empty space with nothing to draw -- so the code area is what fills the
    39.245 mm frame and the quiet zone falls outside it. Sizing the whole file
    to the frame would leave a fat empty margin inside it, the same defect the
    QR symbol had.

    **Turned a quarter turn, reading the same way as the number beside it.**
    The frame is 4.941 mm across by 39.245 mm long, so the strip -- which is
    39.245 mm along the bars by 4.941 mm tall -- has to be rotated to fill it.
    The rotation matches the ``ARKNIGHTS-编号`` line's own 270 degrees, so the
    two read in the same direction.

    A missing 情报编号 is not an error: 24 operators have none, and the card
    still carries their QR symbol.
    """
    own_number = str(getattr(operator, "number", "") or "").strip()
    payload = (
        own_number if number is None else str(number).strip()
    )
    if not payload:
        return None

    region = art_region("barcode")
    # The long side is the direction the bars run.
    code_mm = max(region.width_mm, region.height_mm)
    bar_h_mm = min(region.width_mm, region.height_mm)
    code = badge_codes.barcode_for(
        payload, code_length_mm=code_mm, height_mm=bar_h_mm
    )
    # ⚠️ **The cache key has to carry the number once it stops being the
    # operator's own.** `_cache_png` returns an existing file untouched, so a
    # key of just the page name would hand back the *original* strip for an
    # edited number -- the edit would look like it did nothing. Keeping the
    # plain key for the unchanged case leaves every already-written card
    # pointing at the file it was built from.
    key = operator.page if payload == own_number else f"{operator.page}--{payload}"
    path = _cache_png(
        "barcode",
        key,
        render=lambda: render(code, module_px=BARCODE_MODULE_PX),
    )
    return ImagePlacement(
        path=str(path),
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=code.layout.length_mm,
        height_mm=code.layout.height_mm,
        rotation_deg=BARCODE_ROTATION_DEG,
        layer=BARCODE_LAYER,
        import_id=ART_IMPORT_ID,
        kind="barcode",
    )
