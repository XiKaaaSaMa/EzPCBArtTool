"""A small QR encoder: byte mode, error correction L/M, versions 1-4.

Why this exists instead of OpenCV's encoder: OpenCV picks a conservative
version -- 38 bytes came out as a 33x33 (version 4) symbol where version 3
(29x29) is enough.  On a 3.2 mm badge area every extra version makes the
modules smaller, so the version choice has to be exact.

Only the subset the badge needs is implemented:
  * byte mode, because URLs contain lowercase letters;
  * error-correction levels L and M;
  * versions 1-4, whose single- and two-block layouts are covered.

Version information blocks start at version 7, so this never has to emit one.

A symbol is a list of rows of booleans, ``True`` meaning ink.  The quiet zone
is *not* part of :func:`encode`; :func:`layout` adds it and reports the
module size for a fixed physical area.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EncodeResult",
    "QrLayout",
    "encode",
    "layout",
    "matrix_size",
    "version_for",
]


# --------------------------------------------------------------------- GF(256)

_EXP = [0] * 512
_LOG = [0] * 256
_value = 1
for _index in range(255):
    _EXP[_index] = _value
    _LOG[_value] = _index
    _value <<= 1
    if _value & 0x100:
        _value ^= 0x11D
for _index in range(255, 512):
    _EXP[_index] = _EXP[_index - 255]


def _gf_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def _poly_mul(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            result[i + j] ^= _gf_mul(a, b)
    return result


def _generator(degree: int) -> list[int]:
    poly = [1]
    for i in range(degree):
        poly = _poly_mul(poly, [1, _EXP[i]])
    return poly


def _rs_remainder(data: list[int], degree: int) -> list[int]:
    gen = _generator(degree)
    remainder = list(data) + [0] * degree
    for i in range(len(data)):
        factor = remainder[i]
        if factor == 0:
            continue
        for j, coefficient in enumerate(gen):
            remainder[i + j] ^= _gf_mul(coefficient, factor)
    return remainder[-degree:]


# -------------------------------------------------------------------- capacity

#: Data codewords per version / correction level.
DATA_CODEWORDS = {
    (1, "L"): 19,
    (1, "M"): 16,
    (2, "L"): 34,
    (2, "M"): 28,
    (3, "L"): 55,
    (3, "M"): 44,
    (4, "L"): 80,
    (4, "M"): 64,
}

#: ``[(block_count, data_codewords_per_block), ...]``.
BLOCK_LAYOUT = {
    (1, "L"): [(1, 19)],
    (1, "M"): [(1, 16)],
    (2, "L"): [(1, 34)],
    (2, "M"): [(1, 28)],
    (3, "L"): [(1, 55)],
    (3, "M"): [(1, 44)],
    (4, "L"): [(1, 80)],
    (4, "M"): [(2, 32)],
}

#: Error-correction codewords per block.
EC_CODEWORDS = {
    (1, "L"): 7,
    (1, "M"): 10,
    (2, "L"): 10,
    (2, "M"): 16,
    (3, "L"): 15,
    (3, "M"): 26,
    (4, "L"): 20,
    (4, "M"): 18,
}

_FORMAT_BITS = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}

_ALIGNMENT = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
}

#: Quiet zone required by ISO/IEC 18004, in modules, on each side.
QUIET_ZONE_MODULES = 4


def matrix_size(version: int) -> int:
    """Side length of a QR symbol in modules."""
    return 4 * version + 17


def version_for(payload_bytes: int, level: str = "L") -> int | None:
    """Smallest version that can hold this many bytes, or ``None``."""
    if level not in ("L", "M"):
        raise ValueError("this encoder supports error-correction L and M only")
    # Byte mode: 4-bit mode indicator + 8-bit length for versions 1-9.
    overhead = 2
    for version in (1, 2, 3, 4):
        if payload_bytes + overhead <= DATA_CODEWORDS[(version, level)]:
            return version
    return None


@dataclass(frozen=True)
class EncodeResult:
    matrix: list[list[bool]]
    version: int
    level: str
    mask: int
    byte_length: int

    @property
    def size(self) -> int:
        return len(self.matrix)


@dataclass(frozen=True)
class QrLayout:
    """A QR symbol placed in a fixed area, quiet zone included."""

    matrix: list[list[bool]]
    payload: str
    version: int
    level: str
    border_modules: int
    total_modules: int
    module_mm: float
    area_mm: float

    @property
    def size_modules(self) -> int:
        return self.total_modules


def _bit_stream(payload: bytes) -> list[int]:
    bits: list[int] = []
    bits += [0, 1, 0, 0]  # byte mode
    length = len(payload)
    bits += [(length >> (7 - i)) & 1 for i in range(8)]
    for byte in payload:
        bits += [(byte >> (7 - i)) & 1 for i in range(8)]
    return bits


def _codewords(payload: bytes, version: int, level: str) -> list[int]:
    capacity = DATA_CODEWORDS[(version, level)]
    bits = _bit_stream(payload)
    if len(bits) > capacity * 8:
        raise ValueError("payload does not fit the selected version")

    bits += [0] * min(4, capacity * 8 - len(bits))
    while len(bits) % 8:
        bits.append(0)

    words = [
        int("".join(str(bit) for bit in bits[i : i + 8]), 2)
        for i in range(0, len(bits), 8)
    ]
    pad = (0xEC, 0x11)
    index = 0
    while len(words) < capacity:
        words.append(pad[index % 2])
        index += 1
    return words


def _interleave(words: list[int], version: int, level: str) -> list[int]:
    layout_blocks = BLOCK_LAYOUT[(version, level)]
    ec_length = EC_CODEWORDS[(version, level)]

    blocks: list[tuple[list[int], list[int]]] = []
    cursor = 0
    for count, size in layout_blocks:
        for _ in range(count):
            block = words[cursor : cursor + size]
            cursor += size
            blocks.append((block, _rs_remainder(block, ec_length)))

    result: list[int] = []
    max_data = max(size for _count, size in layout_blocks)
    for i in range(max_data):
        for block, _ec in blocks:
            if i < len(block):
                result.append(block[i])
    for i in range(ec_length):
        for _block, ec in blocks:
            result.append(ec[i])
    return result


def _format_information(level: str, mask: int) -> int:
    data = (_FORMAT_BITS[level] << 3) | mask
    remainder = data << 10
    generator = 0x537
    for i in range(14, 9, -1):
        if remainder & (1 << i):
            remainder ^= generator << (i - 10)
    return ((data << 10) | remainder) ^ 0x5412


def _mask_bit(mask: int, row: int, col: int) -> bool:
    if mask == 0:
        return (row + col) % 2 == 0
    if mask == 1:
        return row % 2 == 0
    if mask == 2:
        return col % 3 == 0
    if mask == 3:
        return (row + col) % 3 == 0
    if mask == 4:
        return (row // 2 + col // 3) % 2 == 0
    if mask == 5:
        return (row * col) % 2 + (row * col) % 3 == 0
    if mask == 6:
        return ((row * col) % 2 + (row * col) % 3) % 2 == 0
    return ((row + col) % 2 + (row * col) % 3) % 2 == 0


def _blank(version: int) -> tuple[list[list[bool]], list[list[bool]]]:
    size = matrix_size(version)
    matrix = [[False] * size for _ in range(size)]
    reserved = [[False] * size for _ in range(size)]

    def fill(row: int, col: int, value: bool) -> None:
        if 0 <= row < size and 0 <= col < size:
            matrix[row][col] = value
            reserved[row][col] = True

    def finder(row: int, col: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                if not (0 <= dr <= 6 and 0 <= dc <= 6):
                    fill(row + dr, col + dc, False)
                    continue
                edge = dr in (0, 6) or dc in (0, 6)
                core = 2 <= dr <= 4 and 2 <= dc <= 4
                fill(row + dr, col + dc, edge or core)

    finder(0, 0)
    finder(0, size - 7)
    finder(size - 7, 0)

    for i in range(8, size - 8):
        value = i % 2 == 0
        fill(6, i, value)
        fill(i, 6, value)

    centres = _ALIGNMENT[version]
    for row in centres:
        for col in centres:
            if (row == 6 and col == 6) or (row == 6 and col == size - 7) or (
                row == size - 7 and col == 6
            ):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    edge = max(abs(dr), abs(dc)) == 2
                    core = dr == 0 and dc == 0
                    fill(row + dr, col + dc, edge or core)

    for i in range(9):
        if not reserved[8][i]:
            fill(8, i, False)
        if not reserved[i][8]:
            fill(i, 8, False)
    for i in range(8):
        fill(8, size - 1 - i, False)
        fill(size - 1 - i, 8, False)
    fill(size - 8, 8, True)
    return matrix, reserved


def _place_data(
    matrix: list[list[bool]], reserved: list[list[bool]], codewords: list[int]
) -> None:
    size = len(matrix)
    bits: list[int] = []
    for word in codewords:
        bits += [(word >> (7 - i)) & 1 for i in range(8)]

    index = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for offset in (0, 1):
                target = col - offset
                if reserved[row][target]:
                    continue
                bit = bits[index] if index < len(bits) else 0
                matrix[row][target] = bool(bit)
                index += 1
        upward = not upward
        col -= 2


def _penalty(matrix: list[list[bool]]) -> int:
    size = len(matrix)
    score = 0

    for line in list(matrix) + [list(col) for col in zip(*matrix)]:
        run = 1
        for i in range(1, size):
            if line[i] == line[i - 1]:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run = 1
        if run >= 5:
            score += 3 + (run - 5)

    for row in range(size - 1):
        for col in range(size - 1):
            block = (
                matrix[row][col],
                matrix[row][col + 1],
                matrix[row + 1][col],
                matrix[row + 1][col + 1],
            )
            if block[0] == block[1] == block[2] == block[3]:
                score += 3

    pattern_a = [True, False, True, True, True, False, True, False, False, False, False]
    pattern_b = [False, False, False, False, True, False, True, True, True, False, True]
    for line in list(matrix) + [list(col) for col in zip(*matrix)]:
        for i in range(size - 10):
            window = line[i : i + 11]
            if window == pattern_a or window == pattern_b:
                score += 40

    dark = sum(1 for row in matrix for value in row if value)
    percent = dark * 100 / (size * size)
    score += int(abs(percent - 50) / 5) * 10
    return score


def encode(
    payload: str | bytes, level: str = "L", force_mask: int | None = None
) -> EncodeResult:
    """Encode payload into the smallest supported QR symbol."""
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    version = version_for(len(data), level)
    if version is None:
        raise ValueError(
            f"payload is too long ({len(data)} bytes); this encoder supports up to version 4"
        )

    words = _codewords(data, version, level)
    final = _interleave(words, version, level)

    best: tuple[int, int, list[list[bool]]] | None = None
    for mask in ([force_mask] if force_mask is not None else range(8)):
        matrix, reserved = _blank(version)
        _place_data(matrix, reserved, final)
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                if reserved[row][col]:
                    continue
                if _mask_bit(mask, row, col):
                    matrix[row][col] = not matrix[row][col]
        info = _format_information(level, mask)
        for i in range(15):
            bit = bool((info >> i) & 1)
            if i < 6:
                matrix[i][8] = bit
            elif i == 6:
                matrix[7][8] = bit
            elif i == 7:
                matrix[8][8] = bit
            elif i == 8:
                matrix[8][7] = bit
            else:
                matrix[8][14 - i] = bit
            if i < 8:
                matrix[8][len(matrix) - 1 - i] = bit
            else:
                matrix[len(matrix) - 15 + i][8] = bit
        score = _penalty(matrix)
        if best is None or score < best[0]:
            best = (score, mask, matrix)

    assert best is not None
    return EncodeResult(
        matrix=best[2], version=version, level=level, mask=best[1], byte_length=len(data)
    )


def layout(
    payload: str | bytes,
    *,
    area_mm: float | None = None,
    code_mm: float | None = None,
    level: str = "L",
    border_modules: int = QUIET_ZONE_MODULES,
) -> QrLayout:
    """Place a QR symbol at a fixed physical size.

    Give **exactly one** of the two sizes, because they measure different
    squares:

    ``area_mm``
        The whole bitmap, quiet zone included.  This is the size a reserved
        rectangle measures.
    ``code_mm``
        The symbol's own modules, quiet zone excluded.  This is the size that
        has to line up with a *drawn boundary*: the quiet zone then falls
        outside that boundary, which is what makes the finder patterns reach
        the corners of a marked block instead of leaving a margin inside it.

    The returned ``module_mm`` is the printed module size in either case,
    which is the number a fabrication check needs.
    """
    if (area_mm is None) == (code_mm is None):
        raise ValueError("area_mm 和 code_mm 必须给一个且只给一个")

    border = max(0, int(border_modules))
    result = encode(payload, level)
    size = result.size
    total = size + 2 * border
    if code_mm is not None:
        code = float(code_mm)
        if code <= 0:
            raise ValueError("QR code area must be positive")
        module_mm = code / size
        area = module_mm * total
    else:
        area = float(area_mm)
        if area <= 0:
            raise ValueError("QR area must be positive")
        module_mm = area / total

    matrix = [[False] * total for _ in range(total)]
    for row in range(size):
        for col in range(size):
            matrix[row + border][col + border] = result.matrix[row][col]
    return QrLayout(
        matrix=matrix,
        payload=payload.decode("utf-8") if isinstance(payload, bytes) else payload,
        version=result.version,
        level=level,
        border_modules=border,
        total_modules=total,
        module_mm=module_mm,
        area_mm=area,
    )
