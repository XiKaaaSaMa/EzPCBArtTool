"""A minimal Code 128B encoder for the badge's operator number.

Code 128B covers ASCII 32-127, which is all an operator number needs.  The
output is a run of black/white modules; the caller decides how wide one module
is in millimetres.  Quiet zones are added by :func:`layout`, because a barcode
printed edge-to-edge against other artwork is the same mistake as a QR symbol
without its margin.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CODE128_PATTERNS",
    "Code128Layout",
    "checksum",
    "layout",
    "modules",
    "runs",
]

#: Module widths for code values 0-106.  Value 106 is the stop pattern, the
#: only one with seven runs instead of six.
CODE128_PATTERNS: dict[int, str] = {
    0: "212222",
    1: "222122",
    2: "222221",
    3: "121223",
    4: "121322",
    5: "131222",
    6: "122213",
    7: "122312",
    8: "132212",
    9: "221213",
    10: "221312",
    11: "231212",
    12: "112232",
    13: "122132",
    14: "122231",
    15: "113222",
    16: "123122",
    17: "123221",
    18: "223211",
    19: "221132",
    20: "221231",
    21: "213212",
    22: "223112",
    23: "312131",
    24: "311222",
    25: "321122",
    26: "321221",
    27: "312212",
    28: "322112",
    29: "322211",
    30: "212123",
    31: "212321",
    32: "232121",
    33: "111323",
    34: "131123",
    35: "131321",
    36: "112313",
    37: "132113",
    38: "132311",
    39: "211313",
    40: "231113",
    41: "231311",
    42: "112133",
    43: "112331",
    44: "132131",
    45: "113123",
    46: "113321",
    47: "133121",
    48: "313121",
    49: "211331",
    50: "231131",
    51: "213113",
    52: "213311",
    53: "213131",
    54: "311123",
    55: "311321",
    56: "331121",
    57: "312113",
    58: "312311",
    59: "332111",
    60: "314111",
    61: "221411",
    62: "431111",
    63: "111224",
    64: "111422",
    65: "121124",
    66: "121421",
    67: "141122",
    68: "141221",
    69: "112214",
    70: "112412",
    71: "122114",
    72: "122411",
    73: "142112",
    74: "142211",
    75: "241211",
    76: "221114",
    77: "413111",
    78: "241112",
    79: "134111",
    80: "111242",
    81: "121142",
    82: "121241",
    83: "114212",
    84: "124112",
    85: "124211",
    86: "411212",
    87: "421112",
    88: "421211",
    89: "212141",
    90: "214121",
    91: "412121",
    92: "111143",
    93: "111341",
    94: "131141",
    95: "114113",
    96: "114311",
    97: "411113",
    98: "411311",
    99: "113141",
    100: "114131",
    101: "311141",
    102: "411131",
    103: "211412",
    104: "211214",
    105: "211232",
    106: "2331112",
}

START_B = 104
STOP = 106

#: Code 128 requires at least ten modules of quiet zone on each side.
QUIET_ZONE_MODULES = 10


def _values(text: str) -> list[int]:
    values = [START_B]
    for char in text:
        code = ord(char) - 32
        if not 0 <= code <= 95:
            raise ValueError("Code 128B supports ASCII characters 32-127 only")
        values.append(code)
    return values


def checksum(text: str) -> int:
    """The modulo-103 check character for a Code 128B payload."""
    values = _values(text)
    return (values[0] + sum(i * value for i, value in enumerate(values[1:], 1))) % 103


def _code_values(text: str) -> list[int]:
    values = _values(text)
    return values + [checksum(text), STOP]


def modules(text: str) -> str:
    """The barcode as a string of ``1`` (bar) and ``0`` (space) modules."""
    bits: list[str] = []
    for value in _code_values(text):
        for index, width in enumerate(CODE128_PATTERNS[value]):
            bits.append(("1" if index % 2 == 0 else "0") * int(width))
    return "".join(bits)


def runs(text: str) -> list[tuple[bool, int]]:
    """The barcode as ``(is_bar, module_count)`` runs.

    Most drawing and export code wants runs, not a string: a run maps to one
    rectangle of a known width, while a module string would have to be scanned
    again by every consumer.
    """
    values = _code_values(text)
    result: list[tuple[bool, int]] = []
    for value in values:
        for index, width in enumerate(CODE128_PATTERNS[value]):
            result.append((index % 2 == 0, int(width)))
    return result


@dataclass(frozen=True)
class Code128Layout:
    """A Code 128B barcode placed at a fixed physical size.

    ``length_mm`` is the **whole file**, quiet zones included. The bars
    themselves span :attr:`code_length_mm`, which is the number a drawn frame
    measures -- see :func:`layout`.
    """

    text: str
    runs: list[tuple[bool, int]]
    total_modules: int
    quiet_zone_modules: int
    length_mm: float
    height_mm: float
    module_mm: float

    @property
    def code_modules(self) -> int:
        return self.total_modules - 2 * self.quiet_zone_modules

    @property
    def code_length_mm(self) -> float:
        """How far the bars themselves run, quiet zones excluded.

        This is what a frame drawn around the barcode measures: the quiet zone
        is empty space and is invisible in the source artwork, so a hand-drawn
        block can only ever mark the bars.
        """
        return self.module_mm * self.code_modules


def layout(
    text: str,
    *,
    length_mm: float | None = None,
    code_length_mm: float | None = None,
    height_mm: float,
    quiet_zone_modules: int = QUIET_ZONE_MODULES,
) -> Code128Layout:
    """Place a Code 128B symbol at a fixed physical size.

    Give **exactly one** of the two lengths, because they measure different
    spans:

    ``length_mm``
        The whole file, quiet zone included. This is the size a reserved
        rectangle measures when the quiet zone is meant to be part of it.
    ``code_length_mm``
        The bars themselves, quiet zone excluded. This is the size that has to
        line up with a *drawn frame*: the quiet zone then falls outside it.
        A frame in the source artwork can only mark the bars, because the
        quiet zone is empty space and has nothing to draw.

    ``height_mm`` is the bar height, which the quiet zone does not affect.
    """
    if (length_mm is None) == (code_length_mm is None):
        raise ValueError("length_mm 和 code_length_mm 必须给一个且只给一个")
    if height_mm <= 0:
        raise ValueError("barcode height must be positive")

    quiet = max(0, int(quiet_zone_modules))
    code_runs = runs(text)
    code_modules = sum(width for _bar, width in code_runs)
    total = code_modules + 2 * quiet
    if code_length_mm is not None:
        code_length = float(code_length_mm)
        if code_length <= 0:
            raise ValueError("barcode code length must be positive")
        module_mm = code_length / code_modules
        length = module_mm * total
    else:
        length = float(length_mm)
        if length <= 0:
            raise ValueError("barcode length must be positive")
        module_mm = length / total

    return Code128Layout(
        text=text,
        runs=code_runs,
        total_modules=total,
        quiet_zone_modules=quiet,
        length_mm=length,
        height_mm=float(height_mm),
        module_mm=module_mm,
    )
