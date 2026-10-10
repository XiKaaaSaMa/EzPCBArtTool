"""Tests for the badge's QR and Code 128 encoders.

The two codes are small enough that a one-version mistake changes the printed
module size by more than 10%, so the tests check both the bit pattern and the
physical layout, not just "an image came out".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_codes, code128, qr_code  # noqa: E402


# ----------------------------------------------------------------------- QR


def test_qr_layout_includes_the_quiet_zone_in_the_physical_area():
    result = badge_codes.qr_for("阿米娅")
    assert result.payload == "prts.wiki/w/阿米娅"
    assert result.version == 2
    assert result.layout.border_modules == 4
    assert result.layout.total_modules == result.layout.size_modules == 33
    assert result.module_mm == pytest.approx(3.2 / 33)
    assert result.scan_warning, "0.097 mm 模块应该明确警告"


def test_a_short_name_can_reach_version_one_but_still_not_015_mm():
    """Even the best possible PRTS payload is only ~0.11 mm in 3.2 mm."""
    result = badge_codes.qr_for("12F")
    assert result.bytes == len("prts.wiki/w/12F".encode("utf-8")) == 15
    assert result.version == 1
    assert result.layout.total_modules == 29
    assert result.module_mm == pytest.approx(3.2 / 29)
    assert result.module_mm < badge_codes.SCAN_MODULE_WARN_MM
    assert result.scan_warning


def test_the_prts_prefix_cannot_be_shortened():
    """`/w/` is required; the number-only URL is not a redirect."""
    assert badge_codes.prts_url("阿米娅") == "prts.wiki/w/阿米娅"
    with pytest.raises(ValueError):
        badge_codes.prts_url("")
    with pytest.raises(ValueError):
        badge_codes.prts_url("https://prts.wiki/w/阿米娅")


def test_short_url_strips_scheme_and_reaches_version_one():
    assert badge_codes.prts_target_url("阿米娅") == (
        "https://prts.wiki/w/%E9%98%BF%E7%B1%B3%E5%A8%85"
    )
    assert badge_codes.short_url_payload("http://i8c.cn/RkhQm") == "i8c.cn/RkhQm"
    result = badge_codes.qr_for_short_url("http://i8c.cn/RkhQm")
    assert result.payload == "i8c.cn/RkhQm"
    assert result.version == 1
    assert result.layout.total_modules == 29
    assert result.module_mm == pytest.approx(3.2 / 29)
    assert result.scan_warning


def test_the_two_sizing_modes_measure_different_squares():
    """``area_mm`` is the whole file; ``code_mm`` is the code area only.

    Both are needed and confusing them is what put the fat margin inside the
    card's drawn block: the block measures the *code area*, and the quiet zone
    is extra. The two readings have to differ by exactly the quiet zone.
    """
    by_code = badge_codes.qr_for_short_url("i8c.cn/RkhQm", code_mm=3.1997)
    # Passing the file size that mode computes must land on the same grid.
    by_area = badge_codes.qr_for_short_url(
        "i8c.cn/RkhQm", area_mm=by_code.layout.area_mm
    )
    assert by_area.module_mm == pytest.approx(by_code.module_mm, abs=1e-12)
    assert by_area.layout.area_mm == pytest.approx(by_code.layout.area_mm, abs=1e-12)
    # 21 code modules + 4 quiet on each side.
    assert by_code.module_mm * 21 == pytest.approx(3.1997, abs=1e-6)
    assert by_code.layout.area_mm == pytest.approx(4.4186, abs=1e-3)
    with pytest.raises(ValueError):
        badge_codes.qr_for_short_url("i8c.cn/RkhQm", area_mm=3.2, code_mm=3.2)
    with pytest.raises(ValueError):
        qr_code.layout("i8c.cn/RkhQm")


def test_the_library_generator_and_the_card_agree_on_the_size():
    """The batch tool must not carry its own copy of the number.

    Regenerating the library with a stale size would leave 463 files that look
    right and no longer match the card, which is invisible until the board is
    printed.

    ⚠️ **Skips without the operator database, and that is not optional.** The
    comparison asks the card for 阿米娅's own placement, which cannot be built
    when the 8 GB asset library is absent -- so on a collaborator's machine this
    used to *fail* rather than skip, and a red suite on day one is how a real
    regression later gets ignored.
    """
    from pcbart import badge_content, operators
    from tools import build_badge_codes

    if not operators.database():
        pytest.skip("干员素材库未安装")

    block_mm = badge_content.art_region("qr").width_mm
    assert build_badge_codes.qr_code_mm() == pytest.approx(block_mm)
    code = badge_codes.qr_for_operator("阿米娅", code_mm=build_badge_codes.qr_code_mm())
    # What the card actually writes for that operator.
    from pcbart.artboard import ArtBoard

    board = ArtBoard()
    board.operator_name = "阿米娅"
    placement = badge_content.badge_art(board=board).qr
    assert code.layout.area_mm == pytest.approx(placement.width_mm, abs=1e-9)


def test_operator_qr_prefers_the_generated_short_link():
    assert badge_codes.short_url_for("阿米娅") == "http://i8c.cn/xE9At"
    qr = badge_codes.qr_for_operator("缄默德克萨斯")
    assert qr.payload == "i8c.cn/Nt2fr"
    assert qr.version == 1
    fallback = badge_codes.qr_for_operator("不存在的页面")
    assert fallback.payload.startswith("prts.wiki/w/")


def test_generated_short_links_cover_the_roster_and_are_all_v1():
    from pcbart.badge_short_urls import SHORT_URLS

    assert len(SHORT_URLS) == 463
    assert SHORT_URLS["阿米娅"] == "http://i8c.cn/xE9At"
    for short in SHORT_URLS.values():
        qr = badge_codes.qr_for_short_url(short)
        assert qr.version == 1
        assert qr.layout.total_modules == 29


def test_qr_encoding_round_trips_through_opencv():
    cv2 = pytest.importorskip("cv2")
    detector = cv2.QRCodeDetector()
    for payload in ("prts.wiki/w/阿米娅", "i8c.cn/RkhQm"):
        layout = qr_code.layout(payload, area_mm=3.2)
        scale = 12
        size = len(layout.matrix)
        image = np.full(((size + 2) * scale, (size + 2) * scale), 255, np.uint8)
        for row, line in enumerate(layout.matrix):
            for col, ink in enumerate(line):
                if ink:
                    y = (row + 1) * scale
                    x = (col + 1) * scale
                    image[y : y + scale, x : x + scale] = 0
        decoded, _points, _ = detector.detectAndDecode(image)
        assert decoded == payload


# ------------------------------------------------------------------- Code 128


def test_code128b_checksum_and_module_width():
    assert code128.checksum("PL07") == 71
    modules = code128.modules("PL07")
    assert len(modules) == 79
    assert set(modules) == {"0", "1"}


def test_code128_pattern_table_is_structurally_valid():
    assert len(code128.CODE128_PATTERNS) == 107
    assert len(set(code128.CODE128_PATTERNS.values())) == 107
    for value, pattern in code128.CODE128_PATTERNS.items():
        widths = [int(char) for char in pattern]
        assert all(1 <= width <= 4 for width in widths)
        expected = 13 if value == code128.STOP else 11
        assert sum(widths) == expected
        assert len(widths) == (7 if value == code128.STOP else 6)
    assert code128.CODE128_PATTERNS[code128.STOP] == "2331112"


def test_barcode_layout_has_enough_room_for_the_quiet_zones():
    """The frame measures the bars; the quiet zone falls outside them.

    39.245 is what the drawn frame spans, and a frame can only ever mark the
    bars -- the ten-module quiet zone on each end is empty space with nothing
    to draw. Sizing the whole *file* to the frame, which is what this used to
    assert, left a fat empty margin inside it.
    """
    barcode = badge_codes.barcode_for("PL07")
    assert barcode.layout.code_modules == 79
    assert barcode.layout.total_modules == 99  # 79 + 10 quiet modules each side
    assert barcode.layout.code_length_mm == pytest.approx(39.245)
    assert barcode.module_mm == pytest.approx(39.245 / 79)
    assert barcode.layout.length_mm > barcode.layout.code_length_mm
    # 20 quiet modules in total, at the code's own module size.
    assert barcode.layout.length_mm - barcode.layout.code_length_mm == pytest.approx(
        20 * barcode.module_mm
    )
    assert barcode.scan_warning == ""


def test_barcode_sizing_modes_measure_different_spans():
    """``length_mm`` is the whole file; ``code_length_mm`` is the bars only."""
    by_code = badge_codes.barcode_for("PL07", code_length_mm=39.245)
    by_file = badge_codes.barcode_for("PL07", length_mm=by_code.layout.length_mm)
    assert by_file.module_mm == pytest.approx(by_code.module_mm, abs=1e-12)
    assert by_file.layout.code_length_mm == pytest.approx(39.245, abs=1e-12)
    with pytest.raises(ValueError):
        badge_codes.barcode_for("PL07", length_mm=39.245, code_length_mm=39.245)
    with pytest.raises(ValueError):
        code128.layout("PL07", height_mm=4.941)


def test_barcode_rejects_a_non_ascii_operator_number():
    with pytest.raises(ValueError):
        code128.modules("编号")
