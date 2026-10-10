"""Rasterise the badge's QR symbol and barcode at a real physical size.

Both codes have to come out at an exact number of millimetres, which is harder
than it looks: at 600 dpi a QR module is only a few pixels -- 3.6 px at the
card's 0.1524 mm -- so rounding it to a whole number changes the printed size
by a few percent, and rounding a 0.3964 mm barcode strip to 9 px shrinks it by
4%. The symbol is therefore drawn at a density
where a module is several pixels wide and scaled to the requested size in one
step, and the canvas, the exporters and the library generator all use these
functions rather than each having a version with its own rounding.

**Nearest-neighbour, never a smooth filter.** Antialiasing leaves grey on every
module edge; silkscreen cannot print grey, and a decoder handed soft edges loses
the module boundaries. Measured before this was settled: OpenCV read 0 of 60
symbols that had been resampled smoothly and all 60 afterwards. The PNG files in
the library hold exactly two grey levels for the same reason.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from . import badge_codes

__all__ = [
    "PROFESSION_INK_THRESHOLD",
    "qr_raster",
    "qr_raster_transparent",
    "barcode_raster",
    "barcode_raster_transparent",
    "profession_icon_raster",
]

#: Grey level splitting the plate from the glyph on a 职业大图.
#:
#: The artwork is two-tone with almost nothing between: measured on the black
#: variant, 89% of the pixels sit in 0-15 and 10% in 240-255, with roughly a
#: hundred pixels in every other band. Any threshold through the empty middle
#: gives the same result, so this only has to be *somewhere* in it.
PROFESSION_INK_THRESHOLD = 128

#: Density the symbol is laid out at before being scaled to size, in pixels per
#: millimetre. High enough that a module is at least ~2 px wide before scaling,
#: so the rounding error inside the grid is small compared with the module.
_DRAW_DENSITY_PX_PER_MM = 32.0


def _qr_bitmap(layout, module_px: int) -> Image.Image:
    """The QR grid at ``module_px`` pixels per module, quiet zone included.

    The quiet zone is part of ``layout.matrix`` -- the encoder adds its border
    modules -- so drawing the matrix as-is puts the margin on the file exactly
    once. Drawing a second one here would silently shrink the modules.
    """
    size = len(layout.matrix)
    pixels = module_px * size
    image = Image.new("L", (pixels, pixels), 255)
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(layout.matrix):
        col = 0
        while col < size:
            if not line[col]:
                col += 1
                continue
            start = col
            while col < size and line[col]:
                col += 1
            draw.rectangle(
                (
                    start * module_px,
                    row * module_px,
                    col * module_px - 1,
                    (row + 1) * module_px - 1,
                ),
                fill=0,
            )
    return image


def qr_raster(
    code: badge_codes.QrCode,
    dpi: int = 600,
    *,
    symbol_px: int | None = None,
    module_px: int = 0,
) -> Image.Image:
    """One QR symbol as a black-on-white bitmap.

    ``symbol_px`` requests an exact pixel size for callers that draw at a known
    number of pixels per millimetre and need every symbol in a sheet to line up.
    ``module_px`` gives an exact whole number of pixels per module, which is
    what keeps the file's own physical size exact rather than rounded to the
    nearest 600-dpi pixel.
    """
    if module_px > 0:
        layout = code.layout
        size = len(layout.matrix)
        target = module_px * size
        return _qr_bitmap(layout, module_px).resize(
            (target, target), Image.NEAREST
        ).convert("RGB")
    layout = code.layout
    size = len(layout.matrix)
    target = symbol_px if symbol_px is not None else round(layout.area_mm * dpi / 25.4)
    target = max(size, target)
    module_px = max(1, round(layout.module_mm * _DRAW_DENSITY_PX_PER_MM))
    if symbol_px is not None:
        module_px = max(1, round(target / size * 2))
    image = _qr_bitmap(layout, module_px)
    return image.resize((target, target), Image.NEAREST).convert("RGB")


def _barcode_bitmap(layout, module_px: int, height_px: int) -> Image.Image:
    """The bar pattern at ``module_px`` pixels per module, quiet zone included.

    The quiet zone is part of the drawing: it is left white here and the
    transparent variant removes it, so both variants put the bars on the same
    grid instead of each placing them separately.
    """
    image = Image.new("L", (module_px * layout.total_modules, height_px), 255)
    draw = ImageDraw.Draw(image)
    x = layout.quiet_zone_modules * module_px
    for is_bar, width in layout.runs:
        if is_bar:
            draw.rectangle((x, 0, x + width * module_px - 1, height_px - 1), fill=0)
        x += width * module_px
    return image


def barcode_raster(
    code: badge_codes.Barcode,
    dpi: int = 600,
    *,
    module_px: int = 0,
) -> Image.Image:
    """One barcode as a black-on-white bitmap, quiet zone included.

    ``module_px`` asks for an exact whole number of pixels per module instead
    of a 600-dpi raster whose final size is rounded. That rounding is not
    harmless: at 600 dpi the bar span came out 39.2766 mm where the frame is
    39.245, so a library file meant to be measured against the frame was 0.03 mm
    out. Giving a whole number of pixels per module keeps every bar edge on a
    pixel boundary and the span exact.
    """
    layout = code.layout
    if module_px > 0:
        width = module_px * layout.total_modules
        height = max(1, round(layout.height_mm / layout.module_mm * module_px))
        image = _barcode_bitmap(layout, module_px, height)
        return image.resize((width, height), Image.NEAREST).convert("RGB")

    target_width = max(1, round(layout.length_mm / 25.4 * dpi))
    target_height = max(1, round(layout.height_mm / 25.4 * dpi))
    module_px = max(1, round(layout.module_mm * _DRAW_DENSITY_PX_PER_MM))
    height_px = max(1, round(layout.height_mm * _DRAW_DENSITY_PX_PER_MM))
    image = _barcode_bitmap(layout, module_px, height_px)
    return image.resize((target_width, target_height), Image.NEAREST).convert("RGB")


def barcode_raster_transparent(
    code: badge_codes.Barcode,
    dpi: int = 600,
    *,
    module_px: int = 0,
) -> Image.Image:
    """One barcode with a transparent background, like the QR symbol.

    Only the bars are ink; the quiet zone and the spaces between bars are
    transparent rather than white. On the card that is the difference between
    printing a white field over the template's own artwork and letting it show
    through. The quiet zone still *exists* -- it is transparent, not absent.

    ``module_px`` requests an exact whole number of pixels per module, which is
    what a symbol that will be traced back into geometry needs: a fractional
    module leaves bar edges between pixels and the nearest-neighbour resample
    moves them, which changes which modules the exported geometry covers.
    """
    layout = code.layout
    if module_px > 0:
        # **Both axes at the same density.** The canvas fits the picture into
        # the placement box by its own pixel aspect, so a bitmap that is 32 px
        # per module across and 600 dpi down renders squashed: measured 27.08
        # pixel aspect against a 9.95 physical one, which drew the strip 2.7x
        # too short. The height is the same number of pixels per module as the
        # width.
        width = module_px * layout.total_modules
        height = max(1, round(layout.height_mm / layout.module_mm * module_px))
    else:
        px_per_mm = dpi / 25.4
        width = max(1, round(layout.length_mm * px_per_mm))
        height = max(1, round(layout.height_mm * px_per_mm))
        module_px = max(1, round(layout.module_mm * px_per_mm))
    ink = _barcode_bitmap(layout, module_px, height).resize(
        (width, height), Image.NEAREST
    )
    rgba = Image.new("RGBA", ink.size, (0, 0, 0, 0))
    rgba.paste((0, 0, 0, 255), mask=Image.eval(ink, lambda value: 255 - value))
    return rgba


def profession_icon_raster(
    image: Image.Image,
    *,
    size_px: int = 0,
    threshold: int = PROFESSION_INK_THRESHOLD,
) -> Image.Image:
    """A solid silkscreen plate with the profession glyph knocked out of it.

    The card shows the profession as **ink everywhere except the symbol**: a
    filled square of silkscreen with the glyph punched through it, so the
    board's own colour reads as the mark. That is how the template's other
    silkscreen artwork behaves, and it keeps the icon legible without a second
    printing pass.

    The library's ``黑`` variant is already that picture -- an opaque black
    field with a white glyph -- so it is converted rather than redrawn: the
    *dark* pixels become material and the *light* ones become holes.

    ⚠️ **The input's own alpha cannot be used, which is why this exists.** Its
    alpha is ~250 everywhere, so :func:`pcbart.export._mask_and_dims`, which
    treats alpha as the shape, would read the whole square as ink and print a
    featureless block. The flattened transparency built here is what carries
    the knock-out instead: opaque plate, transparent glyph.
    """
    plate = image.convert("L")
    if size_px > 0:
        plate = plate.resize((size_px, size_px), Image.NEAREST)
    # Dark plate -> opaque (silkscreen). Light glyph -> transparent (bare board).
    alpha = plate.point(lambda value: 255 if value < threshold else 0)
    rgba = Image.new("RGBA", plate.size, (0, 0, 0, 0))
    rgba.paste((0, 0, 0, 255), mask=alpha)
    return rgba


def qr_raster_transparent(
    layout,
    dpi: int = 600,
    *,
    module_px: int = 0,
) -> Image.Image:
    """One QR symbol with a transparent background.

    Only the symbol's own modules are ink, and everything outside them -- the
    quiet zone and the spaces between modules -- is transparent rather than
    white. On the card this is the difference between a white field printed over
    the board's existing artwork and a symbol that lets that artwork show
    through; a white field punches a hole in the template wherever it lands.

    The quiet zone still *exists*: it is transparent, not absent, and a scanner
    sees the same four modules of clear space it needs.

    ``module_px`` requests an exact whole number of pixels per module, which is
    what a symbol that will be traced back into geometry needs. Letting the
    total pixel count decide instead leaves a fractional module -- 302 px across
    29 modules is 10.41 each -- and the nearest-neighbour resample then moves
    module edges by a fraction of a module. Measured before this was fixed: 11
    of 841 module centres landed on the wrong side of an edge, which is a symbol
    whose geometry no longer matches the bits that were encoded.
    """
    size = len(layout.matrix)
    if module_px > 0:
        target = module_px * size
    else:
        px_per_mm = dpi / 25.4
        target = max(size, round(layout.area_mm * px_per_mm))
        module_px = max(1, round(layout.module_mm * px_per_mm))
    ink = _qr_bitmap(layout, module_px).resize((target, target), Image.NEAREST)
    rgba = Image.new("RGBA", ink.size, (0, 0, 0, 0))
    rgba.paste((0, 0, 0, 255), mask=Image.eval(ink, lambda value: 255 - value))
    return rgba
