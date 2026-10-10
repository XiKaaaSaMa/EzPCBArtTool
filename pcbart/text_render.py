"""Turning a template's text into ink, for the exporters.

The canvas draws text with Qt, but the Gerber writer cannot: a Gerber file has
no notion of a font, only of geometry. So a line that has to reach the board is
rendered to a bitmap and traced into filled shapes, exactly like an imported
picture.

**Why a bitmap and not vector outlines.** The alternative is to pull the glyph
outlines out of the font file and transform them, which would be sharper. It
would also mean carrying a font parser, and -- more importantly -- it would mean
the shapes on the board were whatever *that* font's outlines say, while the
canvas showed Qt's rendering of whatever font Qt resolved. Rendering both from
the same source is the only way the preview and the export are the same
picture, and that is this project's most-repeated defect: two implementations
that agree until one of them is changed.

The raster is deliberately generated at a resolution derived from the *board*
size rather than from a fixed pixel count, so a 1 mm line and a 20 mm line
both come out with comparable edge quality. A fixed size would make small text
illegible and large text needlessly heavy.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from dataclasses import dataclass

__all__ = [
    "FONT_CANDIDATES",
    "RenderedText",
    "font_path",
    "line_box_descent_mm",
    "render_mask",
    "render_text_mask",
    "px_per_mm_for",
]

#: Font files to try, in order, per family the templates name.
#:
#: The families in the source files (``Agency FB``, ``Bahnschrift``,
#: ``Dubai``) are not shipped with every Windows install and are certainly not
#: present on other systems, so every request falls back to a font that is.
#: A card whose lettering is in the wrong typeface is a card the user can still
#: use; a card with no lettering at all is not.
_FALLBACK_CHAIN: tuple[str, ...] = (
    # CJK-capable first: the cards carry Chinese, and a Latin-only fallback
    # renders those glyphs as empty boxes.
    "msyh.ttc",
    "msyh.ttf",
    "simhei.ttf",
    "Deng.ttf",
    "simsun.ttc",
    # Latin-only, for the non-CJK labels on a machine with no CJK font.
    "ARIALUNI.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
)

#: Where to look for those files, in order.
_FONT_DIRS: tuple[Path, ...] = (
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/System/Library/Fonts"),
)

#: Explicit overrides for the families the templates name, so a machine that
#: *does* have them uses them. Values are file names looked up in
#: :data:`_FONT_DIRS`.
FAMILY_FILES: dict[str, str] = {
    "agency fb negreta": "AGENCYR.TTF",
    "agency fb": "AGENCYR.TTF",
    "bahnschrift semibold": "bahnschrift.ttf",
    "bahnschrift": "bahnschrift.ttf",
    "franklin gothic medium": "FRAMDCN.TTF",
    "dubai regular": "DUBAI-REGULAR.TTF",
    "dubai": "DUBAI-REGULAR.TTF",
    "黑体": "simhei.ttf",
}

#: How many pixels represent one millimetre of the board.
#:
#: 40 px/mm is about 1000 dpi: comfortably past the point where tracing picks
#: up raster stair-steps at the feature sizes these cards use, and small enough
#: that a 20 mm line is an 800-pixel bitmap rather than a megapixel one. The
#: clamp keeps a 0.3 mm serial number and a 40 mm headline both usable.
PX_PER_MM_MIN = 12.0
PX_PER_MM_MAX = 96.0


def px_per_mm_for(size_mm: float) -> float:
    """The raster resolution to use for a line whose cap height is ``size_mm``.

    Larger text is rendered at a lower pixels-per-millimetre rate, which keeps
    the bitmap size roughly constant: edge quality depends on the pixel count
    across the glyph, not on the physical size, so spending more pixels on a
    bigger line buys nothing.
    """
    target_px = 160.0
    height = max(float(size_mm), 0.05)
    return max(PX_PER_MM_MIN, min(PX_PER_MM_MAX, target_px / height))


@lru_cache(maxsize=1)
def _scan() -> dict[str, Path]:
    """Every candidate font file that exists, keyed by lower-case file name."""
    found: dict[str, Path] = {}
    for directory in _FONT_DIRS:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for item in entries:
            if item.is_file():
                found.setdefault(item.name.casefold(), item)
    return found


def font_path(family: str = "", text: str = "") -> Path | None:
    """The best available file for ``family`` that can draw ``text``.

    Returns ``None`` only when nothing can, which the caller has to treat as
    "export the text as nothing and say so" rather than as an empty mask -- a
    silently blank line is indistinguishable from a line the user deleted.

    ⚠️ **Coverage is checked, not assumed from the family name.** The template
    asks for ``Agency FB`` for the operator's name, that font *is* installed,
    and it has no CJK glyphs at all -- so rendering 鸿雪 with it produced two
    empty ``.notdef`` boxes. The mask was not blank, so every existing check
    passed; only looking at the card showed the name had become two rectangles.
    A font is therefore only used when it actually draws every character.
    """
    available = _scan()
    order: list[Path] = []
    wanted = FAMILY_FILES.get((family or "").strip().casefold())
    if wanted:
        hit = available.get(wanted.casefold())
        if hit is not None:
            order.append(hit)
    for name in _FALLBACK_CHAIN:
        hit = available.get(name.casefold())
        if hit is not None and hit not in order:
            order.append(hit)
    if not text:
        return order[0] if order else None
    for candidate in order:
        if _covers(candidate, text):
            return candidate
    # Nothing covers it. Returning the first candidate would render boxes,
    # which reads as a font problem in the *file* rather than as a missing
    # glyph everywhere; ``None`` lets the caller report the line by name.
    return None


@lru_cache(maxsize=512)
def _covers(path: Path, text: str) -> bool:
    """Whether ``path`` draws every character of ``text`` rather than a box.

    Measured by rendering: a glyph the font does not have comes out as the same
    ``.notdef`` bitmap every time, so each character's mask is compared against
    a deliberately unmapped code point's. Comparing bitmaps rather than parsed
    character maps keeps this working without a font parser, which is the same
    reason the whole module renders instead of reading outlines.
    """
    if not text:
        return True
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(str(path), 32)
    except OSError:
        return False

    def bitmap(character: str):
        image = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(image).text((4, 4), character, font=font, fill=255)
        return np.asarray(image).tobytes()

    # A private-use code point: almost certainly unmapped, and mapped in a
    # different way by every font, so it cannot collide with a real glyph.
    missing = bitmap("\ue81f")
    for character in set(text):
        if character.isspace():
            continue
        if bitmap(character) == missing:
            return False
    return True


@dataclass(frozen=True)
class RenderedText:
    """A rendered line, positioned relative to the anchor it was drawn at.

    ``centre_mm`` is where the mask's centre sits **relative to the text's
    anchor**, in millimetres, Y up. The caller needs it because the mask is
    cropped to the ink: without this the exporter could only put the mask's
    centre on the anchor, which would shift every line up and right by half its
    own extent -- a displacement that grows with the line and so looks like a
    font problem rather than a placement one.
    """

    mask: np.ndarray
    px_per_mm: float
    centre_mm: tuple[float, float]

    @property
    def size_mm(self) -> tuple[float, float]:
        height_px, width_px = self.mask.shape[:2]
        return (width_px / self.px_per_mm, height_px / self.px_per_mm)


def render_text_mask(
    text: str,
    *,
    size_mm: float,
    family: str = "",
    px_per_mm: float | None = None,
) -> RenderedText | None:
    """``text`` as a black-on-white mask positioned against its anchor.

    The mask has ink as dark pixels, matching what
    :func:`pcbart.vectorize.vectorize_mask` expects (its default is
    ``ink_is_dark=True``). ``None`` when there is no text, no size, or no font.

    ⚠️ **The template's ``LEFT_BOTTOM`` anchor is the bottom of the *line box*,
    not the baseline.** It sits one descender below where the glyphs rest, and
    getting that wrong drops every horizontal line by about a millimetre.

    Measured from the source project's own dimension records, which are the
    cleanest evidence available: the ``50.000mm`` label's anchor y is *exactly*
    the y of the bracket it labels, and in the project's own rendered card the
    numeral bottoms sit about 0.6 mm above that line -- one descender at that
    font size. The same offset shows up on every editable line: comparing the
    card against ``模板``, STAFF's ink bottom sits 1.07 mm above its anchor
    where rendering from the baseline put it 0.02 mm *below*.

    So PIL's ascender-based origin is converted to the line-box bottom
    (``ascent + descent``), and every offset is expressed against that.
    """
    if not text or not text.strip():
        return None
    if size_mm <= 0:
        return None
    # The text goes to the font picker, not just the family: a family that
    # cannot draw these characters has to be passed over for one that can.
    path = font_path(family, text)
    if path is None:
        return None

    from PIL import Image, ImageDraw, ImageFont

    rate = float(px_per_mm if px_per_mm is not None else px_per_mm_for(size_mm))
    pixel_size = max(1, int(round(size_mm * rate)))
    try:
        font = ImageFont.truetype(str(path), pixel_size)
    except OSError:
        return None
    ascent, descent = font.getmetrics()

    # Measured rather than guessed: a CJK line and a Latin one of the same
    # font size are very different widths, and the canvas uses real metrics
    # too. Padding is added because rasterised glyphs can reach a pixel past
    # their reported box.
    probe = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = max(1, right - left + 2)
    height = max(1, bottom - top + 2)

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((-left + 1, -top + 1), text, font=font, fill=0)

    mask = np.asarray(image)
    ink = mask < 128
    if not ink.any():
        # Every glyph was missing from the font. Reported as no mask rather
        # than as an empty one, so the caller can say the line did not export.
        return None
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    top_row = int(np.argmax(rows))
    left_col = int(np.argmax(cols))
    bottom_row = int(len(rows) - np.argmax(rows[::-1]))
    right_col = int(len(cols) - np.argmax(cols[::-1]))
    cropped = mask[top_row:bottom_row, left_col:right_col]

    # Back from the crop to the anchor. PIL's Y grows downward, so a pixel
    # *below* the top of the bitmap is a smaller board Y.
    ink_top_px = top_row - 1 + top  # the +1 undoes the padding translate
    ink_bottom_px = top_row - 1 + bottom
    centre_x_mm = (left + right) / 2.0 / rate
    # ``ascent + descent`` is the line box's bottom, which is what the
    # template's LEFT_BOTTOM anchor marks -- see the docstring. Dropping the
    # ``descent`` term here (i.e. treating the anchor as the baseline) is what
    # put every horizontal line about a millimetre too low.
    centre_y_mm = (
        ascent + descent - (ink_top_px + ink_bottom_px) / 2.0
    ) / rate
    return RenderedText(
        mask=cropped, px_per_mm=rate, centre_mm=(centre_x_mm, centre_y_mm)
    )


def line_box_descent_mm(
    size_mm: float, family: str = "", text: str = ""
) -> float:
    """How far the template's anchor sits below the baseline, in millimetres.

    One definition, used by both writers and the canvas: the canvas draws with
    Qt and the exporters with PIL, and if only one of them applied this shift
    the preview and the board would disagree by exactly the descender -- which
    is the class of bug this whole exercise is about.
    """
    if size_mm <= 0:
        return 0.0
    path = font_path(family, text)
    if path is None:
        return 0.0

    from PIL import ImageFont

    # A fixed reference size: the ratio is what matters, and measuring at the
    # real size would only add rounding.
    reference = 256
    try:
        font = ImageFont.truetype(str(path), reference)
    except OSError:
        return 0.0
    _ascent, descent = font.getmetrics()
    return descent / reference * float(size_mm)


def render_mask(item, *, name: str = "", number: str = "") -> RenderedText | None:
    """One :class:`~pcbart.artboard.TextElement` as a mask, tokens resolved."""
    from .artboard import TextElement

    if not isinstance(item, TextElement):
        return None
    text = item.resolved_text(name=name, number=number)
    return render_text_mask(text, size_mm=item.size_mm, family=item.font)
