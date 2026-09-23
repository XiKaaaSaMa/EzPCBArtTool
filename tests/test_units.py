"""Unit-conversion tests.

The 0.61 mm / 24.0158 unit pair appears in real EasyEDA Pro files (it is the
default via diameter) and is the anchor for the whole format. If the factor
ever moves, these tests are what should stop it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.units import (  # noqa: E402
    EPRU_DECIMALS,
    EPRU_UNIT_MM,
    mm_to_mil,
    mm_to_unit,
    unit_to_mm,
)

# EasyEDA Pro stores coordinates with four decimal places, so any value that
# survives a round trip is only accurate to half a step of that grid.
UNIT_QUANTUM_MM = EPRU_UNIT_MM * 0.5 * 10 ** (-EPRU_DECIMALS)


def test_unit_constant_is_one_mil():
    assert EPRU_UNIT_MM == pytest.approx(0.0254, abs=1e-12)


def test_quantum_is_documented_precision():
    """Half a step of the 4-decimal storage grid, in millimetres."""
    assert UNIT_QUANTUM_MM == pytest.approx(1.27e-6, abs=1e-12)


def test_default_via_diameter_round_trip():
    """EasyEDA's default via is stored as 24.0158 units, i.e. 0.61 mm.

    The stored value is not the exact 4-decimal encoding of 0.61 mm (that
    would be 24.0157); it is 0.00000132 mm high. EasyEDA's own default is
    therefore an approximation, so this asserts the physical value rather than
    bit-exact agreement.
    """
    assert unit_to_mm(24.0158) == pytest.approx(0.61, abs=1e-5)
    assert mm_to_unit(0.61) == pytest.approx(24.0158, abs=1e-4)


def test_dimension_from_real_project():
    """Self-calibration from a real project: 3932.0079 units == 99.873 mm."""
    assert unit_to_mm(3932.0079) == pytest.approx(99.873, abs=1e-3)


def test_default_track_width():
    """A 10-unit track is 0.254 mm, the default 10 mil trace."""
    assert unit_to_mm(10) == pytest.approx(0.254, abs=1e-9)


def test_board_outline_width():
    """Outline strokes are written as 4.9213 units, i.e. 0.125 mm."""
    assert unit_to_mm(4.9213) == pytest.approx(0.125, abs=1e-5)


def test_mm_to_unit_rounds_like_the_editor():
    assert mm_to_unit(1.0) == pytest.approx(39.3701, abs=1e-4)


def test_mil_helpers_agree():
    assert mm_to_mil(1.0) == pytest.approx(39.3701, abs=1e-4)


@pytest.mark.parametrize("value", [0.1, 0.15, 0.254, 1.0, 3.5, 100.0])
def test_round_trip_stability(value: float):
    """mm -> unit -> mm must not drift more than the editor's own rounding."""
    assert unit_to_mm(mm_to_unit(value)) == pytest.approx(value, abs=1e-5)
