"""Check that the exported QR geometry still carries the encoded matrix.

The canvas can show a perfect-looking symbol and the exporter can still turn it
into a blob: the tracer's defaults drop any island smaller than 0.02 mm², and
one QR module is 0.0122 mm². That mistake is invisible in a screenshot and total
on the board, so the export is verified by **sampling its own geometry at each
module's centre** and comparing that against the matrix the encoder produced.

Usage: python tools/verify_badge_qr_export.py [--operator 阿米娅]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from shapely.validation import make_valid  # noqa: E402

from pcbart import badge_card, badge_codes, badge_content, export  # noqa: E402
from pcbart.artboard import ArtBoard  # noqa: E402
from pcbart.design_layers import LAYERS  # noqa: E402


def _signed_area(ring) -> float:
    """Twice the signed area of a ring; positive is counter-clockwise."""
    total = 0.0
    for index, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total


def _shapely_polygon(rings):
    """One traced record as a Shapely polygon, with the winding put right.

    ⚠️ **The two conventions are opposite.** EasyEDA expects the outer ring
    clockwise and the holes counter-clockwise (see
    :func:`pcbart.geometry.normalize_complex_polygon`), while Shapely treats the
    first ring as a shell and *ignores* later rings unless the windings say
    "hole". Handing the record over unchanged makes every hole read as a second
    solid, so a QR symbol's bare cells come back as ink and the check reports
    corruption that is not there -- measured 11 false alarms out of 841 cells
    before this was noticed.
    """
    ordered = list(rings)
    if _signed_area(ordered[0]) < 0:
        ordered[0] = list(reversed(ordered[0]))
    holes = [
        list(reversed(ring)) if _signed_area(ring) > 0 else ring
        for ring in ordered[1:]
    ]
    polygon = Polygon(ordered[0], holes)
    return polygon if polygon.is_valid else make_valid(polygon)


def exported_qr_shape(board, placement):
    """The union of every exported ring that falls on the QR symbol."""
    records, _report = export.build_copper(board)
    layer_id = next(
        entry.easyeda_id for entry in LAYERS if entry.key == placement.layer_key
    )
    reach = placement.width_mm * np.sqrt(2) / 2 + 0.4
    shapes = []
    for record_layer, rings, _width in records:
        if record_layer != layer_id:
            continue
        if not all(
            abs(x - placement.x_mm) <= reach and abs(y - placement.y_mm) <= reach
            for x, y in rings[0]
        ):
            continue
        shapes.append(_shapely_polygon(rings))
    if not shapes:
        return None
    return unary_union(shapes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operator", default="阿米娅")
    args = parser.parse_args(argv)

    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    board.operator_name = args.operator
    art = badge_content.badge_art(board=board)
    board.placements = list(art.placements)

    # ``build_copper`` traces the template wording too, and the placeholder line
    # 干员二维码 sits right beside the symbol. Leaving it in made this check
    # report three bare cells as filled when the ink came from the *label*, not
    # from the code. The panel deletes these lines when the artwork is inserted
    # (``badge_content.ART_ELEMENT_LABELS``), so the verification has to as well
    # or it measures a card nobody builds.
    board.texts = [
        item
        for item in board.texts
        if item.label not in badge_content.ART_ELEMENT_LABELS
    ]

    # The placement measures the whole bitmap, quiet zone included, so the
    # matrix is rebuilt with the same overall area and the module size falls out
    # of it. Sizing by the code area instead would build a slightly different
    # grid than the one that was placed.
    matrix = art_matrix = badge_codes.qr_for_operator(
        args.operator, area_mm=art.qr.width_mm
    ).layout.matrix
    size = len(matrix)
    module_mm = art.qr.width_mm / size

    shape = exported_qr_shape(board, art.qr)
    if shape is None:
        print("导出里找不到二维码几何")
        return 1

    # Sample at each module's centre, in the *un-rotated* frame: the geometry is
    # stored rotated, so each sample point is rotated by the same angle. The
    # module grid is centred on the image, matching the tracer's own centring.
    angle = np.radians(art.qr.rotation_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    hits = 0
    mismatches = []
    for row in range(size):
        for col in range(size):
            local_x = (col + 0.5) * module_mm - art.qr.width_mm / 2
            local_y = art.qr.width_mm / 2 - (row + 0.5) * module_mm
            sample_x = art.qr.x_mm + local_x * cos - local_y * sin
            sample_y = art.qr.y_mm + local_x * sin + local_y * cos
            inside = shape.contains(Point(sample_x, sample_y))
            if inside:
                hits += 1
            if bool(matrix[row][col]) != inside:
                mismatches.append((row, col, bool(matrix[row][col]), inside))

    inked = sum(1 for line in matrix for value in line if value)
    print(f"符号 {size}x{size}  模块 {module_mm:.4f} mm  墨模块 {inked}")
    print(f"导出几何覆盖的模块中心 {hits}")
    print(f"与编码矩阵不一致 {len(mismatches)} 处")
    for row, col, want, got in mismatches[:10]:
        print(f"  ({row},{col}) 矩阵={want} 导出={got}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
