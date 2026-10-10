"""The 通行证 card's two machine-readable codes.

The card reserves a 3.2 mm square for the QR symbol and a 4.941 x 39.245 mm
strip for the barcode.  These functions turn an operator row into the exact
bit patterns and report the physical module size; the panel and the exporters
should not each re-derive the QR version or the Code 128 checksum.

The QR payload is the agreed C form::

    prts.wiki/w/<中文页面名>

The ``https://`` prefix is omitted on purpose, and the Chinese characters are
encoded directly as UTF-8.  The PRTS link column in the operator CSV is
percent-encoded and must not be fed to this function; it is longer and would
push the symbol to a larger version.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from . import code128
from . import qr_code

__all__ = [
    "PRTS_PREFIX",
    "QR_AREA_MM",
    "BARCODE_LENGTH_MM",
    "BARCODE_HEIGHT_MM",
    "SCAN_MODULE_WARN_MM",
    "QrCode",
    "Barcode",
    "prts_url",
    "prts_target_url",
    "qr_for",
    "qr_for_payload",
    "qr_for_short_url",
    "qr_for_operator",
    "short_url_for",
    "short_url_payload",
    "barcode_for",
    "module_warning",
]

PRTS_PREFIX = "prts.wiki/w/"
"""Shortest verified working PRTS URL prefix.

``prts.wiki/<page>`` returns 404 and ``prts.wiki/w/<number>`` is an empty
page, so neither can replace this.  The prefix costs 12 bytes; even the
shortest Chinese operator name then needs a version-2 QR symbol.
"""

QR_AREA_MM = 3.2
"""The template's full QR square, quiet zone included."""

QR_QUIET_ZONE_MODULES = 4
"""ISO/IEC 18004 quiet zone per side, in modules."""

BARCODE_LENGTH_MM = 39.245
"""How far the bars run, measured from the frame the template draws.

**This is the code area, not the file.** The drawn frame can only mark the
bars: the ten-module quiet zone on each end is empty space with nothing to
draw, so a hand-drawn block can never have included it. Sizing the whole file
to the frame would leave a fat empty margin inside it -- the same defect the QR
symbol had before its block was redefined the same way.
"""

BARCODE_HEIGHT_MM = 4.941
"""Bar height. The template quotes the frame as 4.941 x 39.245 mm."""

BARCODE_QUIET_ZONE_MODULES = 10
"""Code 128 quiet zone per side, in modules."""

SCAN_MODULE_WARN_MM = 0.15
"""Practical floor for a printed module.

JLCPCB's silkscreen process is generally quoted at 0.15 mm minimum feature.
Below that the ink can bridge or drop out, and a phone camera has less margin
still, so the renderers should show the measured module size and warn instead
of claiming the code will scan.
"""


@dataclass(frozen=True)
class QrCode:
    payload: str
    layout: qr_code.QrLayout

    @property
    def module_mm(self) -> float:
        return self.layout.module_mm

    @property
    def version(self) -> int:
        return self.layout.version

    @property
    def bytes(self) -> int:
        return len(self.payload.encode("utf-8"))

    @property
    def scan_warning(self) -> str:
        return module_warning(self.module_mm, "二维码")


@dataclass(frozen=True)
class Barcode:
    payload: str
    layout: code128.Code128Layout

    @property
    def module_mm(self) -> float:
        return self.layout.module_mm

    @property
    def scan_warning(self) -> str:
        return module_warning(self.module_mm, "条形码")


def prts_url(page_name: str) -> str:
    """Build the scheme-less QR payload from the page-name column."""
    page = str(page_name or "").strip()
    if not page:
        raise ValueError("页面名为空，无法生成 PRTS 链接")
    if page.startswith("http://") or page.startswith("https://"):
        raise ValueError("请传页面名，不要传 PRTS链接 列里的完整 URL")
    return PRTS_PREFIX + page


def prts_target_url(page_name: str) -> str:
    """The full, percent-encoded PRTS URL a shortener should redirect to.

    The shortener's backend mangled raw UTF-8 Chinese characters in testing;
    percent-encoding the path keeps the stored target byte-for-byte correct
    and does not affect the short link's own length.
    """
    page = str(page_name or "").strip()
    if not page:
        raise ValueError("页面名为空，无法生成 PRTS 链接")
    if page.startswith("http://") or page.startswith("https://"):
        raise ValueError("请传页面名，不要传 PRTS链接 列里的完整 URL")
    return "https://prts.wiki/w/" + quote(page, safe="")


def short_url_payload(short_url: str) -> str:
    """Strip the scheme from a short URL for the smallest QR symbol.

    ``i8c.cn/abcd`` is 12 bytes and fits version 1; adding ``http://`` makes
    it 19 bytes and pushes every card back to version 2.
    """
    value = str(short_url or "").strip()
    for prefix in ("https://", "http://"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.rstrip("/")


def qr_for_payload(
    payload: str | bytes,
    *,
    area_mm: float | None = None,
    code_mm: float | None = None,
    level: str = "L",
) -> QrCode:
    """Encode an arbitrary payload at a fixed physical size.

    ``area_mm`` measures the whole bitmap including the quiet zone;
    ``code_mm`` measures the symbol's own modules and lets the quiet zone fall
    outside them. Give one of the two; giving neither uses the badge's own
    3.2 mm area.
    """
    if area_mm is None and code_mm is None:
        area_mm = QR_AREA_MM
    return QrCode(
        payload=payload.decode("utf-8") if isinstance(payload, bytes) else payload,
        layout=qr_code.layout(
            payload,
            area_mm=area_mm,
            code_mm=code_mm,
            level=level,
            border_modules=QR_QUIET_ZONE_MODULES,
        ),
    )


def qr_for(
    page_name: str,
    *,
    area_mm: float | None = None,
    code_mm: float | None = None,
    level: str = "L",
) -> QrCode:
    """Encode the long, scheme-less PRTS URL into the badge's QR square."""
    return qr_for_payload(
        prts_url(page_name),
        area_mm=area_mm,
        code_mm=code_mm,
        level=level,
    )


def qr_for_short_url(
    short_url: str,
    *,
    area_mm: float | None = None,
    code_mm: float | None = None,
    level: str = "L",
) -> QrCode:
    """Encode a generated short URL, dropping its scheme when possible."""
    return qr_for_payload(
        short_url_payload(short_url),
        area_mm=area_mm,
        code_mm=code_mm,
        level=level,
    )


def short_url_for(page_name: str) -> str:
    """Look up the generated short URL for an operator page.

    The mapping is account-specific and generated by
    ``tools/shorten_badge_urls.py``.  An empty result means the caller should
    fall back to the long PRTS URL instead of failing the whole card.
    """
    page = str(page_name or "").strip()
    if not page:
        return ""
    try:
        from .badge_short_urls import SHORT_URLS
    except ImportError:
        return ""
    return str(SHORT_URLS.get(page, "") or "")


def qr_for_operator(
    page_name: str,
    *,
    area_mm: float | None = None,
    code_mm: float | None = None,
    level: str = "L",
    prefer_short: bool = True,
) -> QrCode:
    """Encode the operator's QR, using the short link when one exists.

    Falling back to the long URL keeps the function usable on a machine where
    the generated short-link module has not been built yet.
    """
    if prefer_short:
        short = short_url_for(page_name)
        if short:
            return qr_for_short_url(
                short, area_mm=area_mm, code_mm=code_mm, level=level
            )
    return qr_for(page_name, area_mm=area_mm, code_mm=code_mm, level=level)


def barcode_for(
    number: str,
    *,
    length_mm: float | None = None,
    code_length_mm: float | None = None,
    height_mm: float = BARCODE_HEIGHT_MM,
) -> Barcode:
    """Encode the operator's 情报编号 as Code 128B at a fixed physical size.

    ``code_length_mm`` measures the bars and lets the quiet zone fall outside
    them, which is what the card's drawn frame requires; ``length_mm`` measures
    the whole file and is the older reading. Give one of the two; giving
    neither uses the frame's own 39.245 mm as the code length.
    """
    payload = str(number or "").strip()
    if not payload:
        raise ValueError("情报编号为空，无法生成条形码")
    if length_mm is None and code_length_mm is None:
        code_length_mm = BARCODE_LENGTH_MM
    return Barcode(
        payload=payload,
        layout=code128.layout(
            payload,
            length_mm=length_mm,
            code_length_mm=code_length_mm,
            height_mm=height_mm,
            quiet_zone_modules=BARCODE_QUIET_ZONE_MODULES,
        ),
    )


def module_warning(module_mm: float, label: str) -> str:
    """A human-readable warning when a code's module is below the print floor."""
    if module_mm >= SCAN_MODULE_WARN_MM:
        return ""
    return (
        f"{label}模块只有 {module_mm:.3f} mm，低于建议的 "
        f"{SCAN_MODULE_WARN_MM:.2f} mm；镜头上可能扫不出"
    )
