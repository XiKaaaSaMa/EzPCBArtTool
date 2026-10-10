"""Tests for the printable badge QR sheets.

The sheet is where the physical question gets answered -- can a phone read the
code at the size the card actually reserves -- so the failures worth catching
are layout ones: a code drawn at the wrong physical size, or a sheet that
covers only some of the factions it claims to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_codes, operators  # noqa: E402
from tools import make_badge_qr_sheet as sheet  # noqa: E402


@pytest.fixture(scope="module")
def roster() -> list[tuple[str, list]]:
    """The sheet's roster, or a skip when the asset library is not installed."""
    if not operators.database():
        pytest.skip("干员素材库未安装")
    rows, _missing = sheet._sample_operators(6)
    if not rows:
        pytest.skip("干员素材库为空")
    return rows


def test_geometry_keeps_the_sheet_inside_a4_landscape():
    geometry = sheet._geometry(6, 6)
    assert geometry.columns == 6
    bottom = geometry.top + geometry.row_pitch
    assert bottom <= sheet.PAGE_H_MM
    right = geometry.left + geometry.columns * geometry.column_pitch
    assert right <= sheet.PAGE_W_MM


def test_geometry_narrows_the_columns_instead_of_running_off_the_page():
    """More factions must not push the last column past the paper edge."""
    geometry = sheet._geometry(7, 6)
    assert geometry.left + geometry.columns * geometry.column_pitch <= sheet.PAGE_W_MM


def test_faction_picks_are_recognisable_and_not_duplicated(roster):
    for faction, members in roster:
        assert 0 < len(members) <= 6, faction
        names = [item.name for item in members]
        assert len(names) == len(set(names)), faction
        assert all(item.faction == faction for item in members), faction


def test_every_operator_on_the_sheet_encodes_a_unique_short_payload(roster):
    payloads = []
    for _faction, members in roster:
        for operator in members:
            qr = badge_codes.qr_for_operator(operator.page)
            assert qr.payload.startswith("i8c.cn/"), operator.name
            assert qr.version == 1, operator.name
            payloads.append(qr.payload)
    assert len(payloads) == len(set(payloads))


def test_the_three_printed_sizes_are_the_sizes_the_svg_claims(roster):
    faction, members = roster[0]
    operator = members[0]
    small = badge_codes.qr_for_operator(operator.page, area_mm=sheet.CARD_SIZE_MM)
    medium = badge_codes.qr_for_operator(operator.page, area_mm=sheet.REAL_SIZE_MM)
    big = badge_codes.qr_for_operator(operator.page, area_mm=sheet.BIG_SIZE_MM)
    assert small.layout.area_mm == pytest.approx(sheet.CARD_SIZE_MM)
    assert medium.layout.area_mm == pytest.approx(sheet.REAL_SIZE_MM)
    assert big.layout.area_mm == pytest.approx(sheet.BIG_SIZE_MM)
    # Same payload at three sizes: only the module size may differ.
    assert small.payload == medium.payload == big.payload
    assert small.module_mm < medium.module_mm < big.module_mm
    assert big.module_mm == pytest.approx(sheet.BIG_SIZE_MM / small.layout.total_modules)


def test_svg_sheet_places_each_square_at_its_nominal_size(roster, tmp_path):
    out = tmp_path / "qr-faction-test.svg"
    sheet._write_faction_sheet(roster, out, per_faction=6)

    text = out.read_text(encoding="utf-8")
    # Each of the three sizes has to be declared as a 1:1 square, otherwise
    # printing at 100% no longer answers the question the sheet exists for.
    for area in (sheet.CARD_SIZE_MM, sheet.REAL_SIZE_MM, sheet.BIG_SIZE_MM):
        # ``_fmt`` is how the sheet writes every length; comparing with a
        # different formatter would fail on trailing digits, not on the size.
        size = sheet._fmt(area)
        assert f'width="{size}" height="{size}"' in text
    faction, members = roster[0]
    assert faction in text
    assert members[0].name in text
    assert text.startswith("<svg")


def test_the_sheet_prints_the_cards_real_file_size(roster):
    """The first square must reproduce what the board actually gets.

    "The card's real size" is the whole file, quiet zone included -- the drawn
    block measures the code area, which is 4 quiet modules smaller per side.
    Sizing the sheet by the block would print every symbol 27% denser than the
    card and answer the wrong question about whether it scans.
    """
    from pcbart import badge_content

    _faction, members = roster[0]
    block_mm = badge_content.art_region("qr").width_mm
    card_code = badge_codes.qr_for_operator(members[0].page, code_mm=block_mm)
    assert sheet.CARD_SIZE_MM == pytest.approx(card_code.layout.area_mm)
    assert sheet.CARD_SIZE_MM > block_mm

    qr = badge_codes.qr_for_operator(members[0].page, area_mm=sheet.CARD_SIZE_MM)
    assert qr.module_mm == pytest.approx(sheet.CARD_SIZE_MM / qr.layout.total_modules)
    # Enlarging the code area fixed the old sub-0.15 mm module, so the sheet no
    # longer has to carry that warning.
    assert qr.module_mm >= badge_codes.SCAN_MODULE_WARN_MM
    assert qr.scan_warning == ""


def test_every_printed_size_is_above_the_print_floor(roster):
    """No square on the sheet should be a knowingly unprintable module."""
    _faction, members = roster[0]
    for area in (sheet.CARD_SIZE_MM, sheet.REAL_SIZE_MM, sheet.BIG_SIZE_MM):
        qr = badge_codes.qr_for_operator(members[0].page, area_mm=area)
        assert qr.module_mm >= badge_codes.SCAN_MODULE_WARN_MM, area
