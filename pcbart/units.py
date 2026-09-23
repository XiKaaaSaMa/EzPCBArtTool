"""Unit handling for EasyEDA Pro (``.epru``) documents.

Everything in this application works in millimetres. The on-disk format uses a
smaller unit, and the conversion between the two lives here and nowhere else.
If the calibration ever turns out to be wrong, this is the only module that
needs to change.

Calibration evidence is recorded in ``docs/M0_FINDINGS.md``.
"""

from __future__ import annotations

EPRU_UNIT_MM = 0.0254
"""Length of one ``.epru`` coordinate unit, in millimetres (1 unit == 1 mil).

Verified against a real EasyEDA Pro 3.2.149 project: a length dimension whose
endpoints span 3932.0079 units is annotated by EasyEDA itself as ``99.873mm``,
and ``3932.0079 * 0.0254 == 99.8730``.
"""

EPRU_DECIMALS = 4
"""EasyEDA Pro writes coordinates rounded to this many decimal places."""

MM_PER_INCH = 25.4
MM_PER_MIL = 0.0254
MILS_PER_MM = 1.0 / MM_PER_MIL
MILS_PER_INCH = 1000.0


FILE_Y_IS_UP = True
"""Whether ``.epru`` Y grows upward, like the design frame.

Set to ``True``. Three independent lines of evidence agree, and the earlier
``False`` setting produced vertically mirrored boards:

1. **KiCad's own EasyEDA Pro importer** negates Y while reading
   (``ScalePos`` returns ``-ScaleSize(y)``). KiCad's board frame grows downward,
   so the source must grow upward.
2. **Dimension placement.** In a real project the width dimension sits on the
   ``-Y`` side of the board and the height dimension on the ``-X`` side. With X
   documented as positive-to-the-right, the paired placement puts ``-Y`` below,
   so ``+Y`` is up.
3. **Observed rendering.** With the negation applied, generated boards appeared
   mirrored top-to-bottom in the editor.

Note the published format specification describes ring winding as "outer
clockwise, holes counter-clockwise", which only holds if Y is negated first.
The specification's own prose is therefore written in a Y-down frame while the
data is stored Y-up; this constant records the data's actual convention.
"""

Y_SIGN = 1.0 if FILE_Y_IS_UP else -1.0
"""Multiplier converting a file Y value into the design frame (and back)."""


def y_from_file(value: float) -> float:
    """File Y -> design Y."""
    return Y_SIGN * value


def y_to_file(value: float) -> float:
    """Design Y -> file Y."""
    return Y_SIGN * value


def unit_to_mm(value: float) -> float:
    """Convert a raw ``.epru`` coordinate or length into millimetres."""
    return value * EPRU_UNIT_MM


def mm_to_unit(value: float, decimals: int = EPRU_DECIMALS) -> float:
    """Convert millimetres into a raw ``.epru`` coordinate or length.

    Results are rounded the way EasyEDA Pro rounds them on save, so that a
    generate/read/write cycle is stable rather than accumulating drift.
    """
    return round(value / EPRU_UNIT_MM, decimals)


def mil_to_mm(value: float) -> float:
    return value * MM_PER_MIL


def mm_to_mil(value: float) -> float:
    return value * MILS_PER_MM


def unit_to_mil(value: float) -> float:
    """Raw ``.epru`` units are already mil, so this is an identity."""
    return value


def mil_to_unit(value: float) -> float:
    return value


def format_mm(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}mm"
