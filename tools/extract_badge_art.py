"""Extract the 通行证 template's fixed artwork into a small data file.

The source projects carry a lot more than wording: copper and silkscreen
borders, the vertical RHODES ISLAND strip, the Arknights logo, hatching and
the solder-mask windows.  Those records are not editable content, but a card
that only draws the wording looks empty next to the file the user supplied.

The data is generated rather than hand-copied.  Run this tool after replacing
the template files, review the generated diff, and commit the result:

    python tools/extract_badge_art.py

Only the **short-card** file is used.  It is the layout that carries the fixed
border, the faction-logo area and the lower printed panel; the long-card file
replaces the artwork window with a full-height illustration, so the two cannot
be baked into one fixed decoration set.  The current single 通行证 template
therefore follows the short-card design.  ``STRING`` records are deliberately
not copied here: the panel edits that wording as real text and the exporters
write it as native text, so copying it too would duplicate it.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import pprint
import sys
import zlib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.epru import EproArchive  # noqa: E402
from pcbart.geometry import densify_path  # noqa: E402
from pcbart.primitives import Mode, record_to_mm  # noqa: E402
from pcbart.units import MM_PER_MIL  # noqa: E402
from tools.extract_badge_regions import extract_boards  # noqa: E402

#: Layers that carry the card's visible artwork.  Layer 11 is the board
#: outline and layer 12 is the lanyard slot; both are rebuilt from the model,
#: so copying them here would create a second, competing definition.
ART_LAYERS = (1, 2, 3, 4, 5, 6)

#: Record types that draw something.  ``STRING`` is deliberately absent: the
#: wording is editable content and the exporters write it as native text.
ART_TYPES = ("FILL", "POLY", "LINE", "IMAGE")
"""Record types that draw something.

``REGION`` is deliberately absent even though the source files carry two: those
records are ``regionType=PROHIBIT`` keepouts, a *constraint* rather than
artwork. Drawn as filled shapes they became two 7.24 x 56.13 mm blocks of solid
copper beside the illustration window -- the opposite of what a keepout means.
"""

SOURCE_NAME = "模板-精0、精1、皮肤小.epro2"
OUTPUT = Path(__file__).resolve().parent.parent / "pcbart" / "badge_decorations.py"

#: The two source layouts, keyed the way the board's ``outline.template``
#: names them. The short card is the default and the one :data:`SOURCE_NAME`
#: points at; both are extracted so the user can pick either.
TEMPLATE_VARIANTS: dict[str, str] = {
    "short": "模板-精0、精1、皮肤小.epro2",
    "tall": "模板-精2、皮肤大.epro2",
}
"""``精0/精1/小皮肤`` against ``精2/大皮肤``.

The card is the same size and the same outline in both; what differs is where
the fixed artwork sits, because the tall layout replaces the short one's
illustration window plus 阵营 logo block with a full-height window. Picking the
wrong one prints the other card's borders and hatching.
"""


def _relative(point: tuple[float, float], centre: tuple[float, float]) -> list[float]:
    return [round(point[0] - centre[0], 4), round(point[1] - centre[1], 4)]


def _image_rings(
    body: dict[str, Any], layer: int
) -> list[list[tuple[float, float]]]:
    """A raster-image record's vector paths in the card's own coordinate frame.

    EasyEDA stores an ``IMAGE`` as a path in the source picture's pixel frame
    plus a frame, an angle and a mirror flag.  Three facts about the frame are
    easy to get wrong and all of them showed up in the editor:

    * the picture's own Y axis points **down**, while the document's points up;
    * so ``v`` has to be measured from the path's **larger** Y -- the picture's
      top edge -- and subtracted, not measured from the smaller one;
    * bottom-side images are stored mirrored horizontally, so their local X
      axis runs the other way from a top-side image's.

    The transform below was checked against the front/back EDA screenshots:
    top-side angle 0 extends right/down, top-side angle 270 extends left/down,
    and bottom-side records mirror that horizontal direction.

    ⚠️ **``v`` used to be measured from the path's smaller Y, which drew every
    angle-0 picture upside down.** The 罗德岛 emblem came out with the castle's
    merlons at the bottom and ``RHODES ISLAND`` across the top. Each project
    embeds the author's own rendering of the card, and that is the ground
    truth; reversing ``v`` puts the emblem right with the frame unmoved, so
    nothing else has to change. The error could not show up on the 270-degree
    ``RHODES ISLAND`` strips -- their ``cos`` term is zero -- which is why only
    the horizontal artwork was ever reported.
    """
    rings = densify_path(body.get("path"))
    points = [point for ring in rings for point in ring]
    if not points:
        return []

    start_x = float(body.get("startX", 0.0))
    start_y = float(body.get("startY", 0.0))
    width = float(body.get("width", 0.0))
    height = float(body.get("height", 0.0))
    angle = math.radians(float(body.get("angle", 0.0) or 0.0))
    mirror = bool(body.get("mirror"))
    top_side = int(layer) in (1, 3, 5)

    x0 = min(point[0] for point in points)
    x1 = max(point[0] for point in points)
    y0 = min(point[1] for point in points)
    y1 = max(point[1] for point in points)
    span_x = max(x1 - x0, 1e-9)
    span_y = max(y1 - y0, 1e-9)

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    placed: list[list[tuple[float, float]]] = []
    for ring in rings:
        converted: list[tuple[float, float]] = []
        for x, y in ring:
            u = (x - x0) / span_x * width
            v = (y1 - y) / span_y * height
            if mirror:
                u = width - u
            dx = u * cos_a + v * sin_a
            dy = u * sin_a - v * cos_a
            if not top_side:
                dx = -dx
            converted.append((start_x + dx, start_y + dy))
        placed.append(converted)
    return placed


def _fill_rings(body: dict[str, Any]) -> list[list[tuple[float, float]]]:
    converted = record_to_mm(type("_Record", (), {"body": body})())
    path = converted.get("path") if converted else None
    return densify_path(path)


def _shift_path(node: Any, dx: float, dy: float) -> Any:
    """Translate every absolute coordinate in a path, leaving angles alone."""
    if not isinstance(node, list) or not node:
        return node
    if isinstance(node[0], list):
        return [_shift_path(item, dx, dy) for item in node]

    head = node[0]
    if head == Mode.R and len(node) >= 5:
        out = list(node)
        out[1] = float(out[1]) + dx
        out[2] = float(out[2]) + dy
        return out
    if head == Mode.CIRCLE and len(node) >= 4:
        out = list(node)
        out[1] = float(out[1]) + dx
        out[2] = float(out[2]) + dy
        return out

    out: list[Any] = []
    index = 0
    while index < len(node):
        token = node[index]
        if isinstance(token, str):
            out.append(token)
            index += 1
            if token in (Mode.ARC, Mode.CARC):
                out.append(node[index])  # angle, not a coordinate
                out.append(float(node[index + 1]) + dx)
                out.append(float(node[index + 2]) + dy)
                index += 3
            elif token == Mode.C:
                for _ in range(3):
                    out.append(float(node[index]) + dx)
                    out.append(float(node[index + 1]) + dy)
                    index += 2
            continue
        run: list[float] = []
        while index < len(node) and isinstance(node[index], (int, float)):
            run.append(float(node[index]))
            index += 1
        for offset in range(0, len(run) - 1, 2):
            out.append(run[offset] + dx)
            out.append(run[offset + 1] + dy)
        if len(run) % 2:
            out.append(run[-1])
    return out


def _shift_body(body: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """The raw record body, translated from the source card to the origin.

    ``IMAGE`` keeps its path in the source picture's own frame and is moved by
    its anchor only.  Every other primitive stores absolute coordinates and is
    translated coordinate by coordinate.  Passing these records through
    unchanged except for the translation lets the editor apply the image's own
    angle and mirror rules, which is more reliable than re-deriving them.
    """
    out = dict(body)
    if "path" in out:
        out["path"] = _shift_path(out.get("path"), dx, dy)
    for key in ("startX", "endX"):
        if isinstance(out.get(key), (int, float)):
            out[key] = float(out[key]) + dx
    for key in ("startY", "endY"):
        if isinstance(out.get(key), (int, float)):
            out[key] = float(out[key]) + dy
    return out


def _in_card(record, centre: tuple[float, float]) -> bool:
    """Whether a record's rough bounds belong to the card rather than the frame."""
    body = record.body
    if record.type == "IMAGE":
        x = float(body.get("startX", 0.0)) * MM_PER_MIL - centre[0]
        y = float(body.get("startY", 0.0)) * MM_PER_MIL - centre[1]
        return -35.0 < x < 35.0 and -60.0 < y < 60.0

    converted = record_to_mm(record)
    path = converted.get("path") if converted else None
    if not isinstance(path, list):
        return True
    rings = densify_path(path)
    points = [point for ring in rings for point in ring]
    if not points:
        return True
    xs = [point[0] - centre[0] for point in points]
    ys = [point[1] - centre[1] for point in points]
    return min(xs) < 30.0 and max(xs) > -30.0 and min(ys) < 55.0 and max(ys) > -55.0

def _card_polygon():
    """The card outline as a Shapely polygon, in millimetres."""
    from shapely.geometry import Polygon

    from pcbart.badge_card import BADGE_OUTLINE, BADGE_OUTLINE_ARCS
    from pcbart.geometry import densify_ring

    ring = densify_ring(BADGE_OUTLINE, BADGE_OUTLINE_ARCS)
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _clip_rings(
    rings: list[list[tuple[float, float]]], clip
) -> list[list[list[tuple[float, float]]]]:
    """All subpaths combined with the even-odd rule, then clipped to the card.

    An ``IMAGE`` path is a list of independent contours: one per glyph, letter
    and hole.  Treating the first as the outer ring and every later one as a
    hole cut most of the side text away and left only its oval shells.  XOR-ing
    the ring polygons reproduces the file's own even-odd fill, so glyphs stay
    glyphs and their counters stay open.
    """
    if not rings:
        return []
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    polygons = [
        Polygon([(float(x), float(y)) for x, y in ring])
        for ring in rings
        if len(ring) >= 3
    ]
    if not polygons:
        return []
    combined = polygons[0]
    for polygon in polygons[1:]:
        combined = combined.symmetric_difference(polygon)
    clipped = combined.intersection(clip)

    pieces = []
    if isinstance(clipped, Polygon):
        candidates = [clipped]
    elif isinstance(clipped, MultiPolygon):
        candidates = list(clipped.geoms)
    elif isinstance(clipped, GeometryCollection):
        candidates = [
            item for item in clipped.geoms if isinstance(item, Polygon)
        ]
    else:
        candidates = []
    for polygon in candidates:
        if polygon.is_empty or polygon.area <= 0:
            continue
        outer = [(x, y) for x, y in polygon.exterior.coords]
        holes = [[(x, y) for x, y in interior.coords] for interior in polygon.interiors]
        from pcbart.geometry import normalize_complex_polygon

        pieces.append(normalize_complex_polygon(outer, holes))
    return pieces


def _clip_paths(
    lines: list[list[tuple[float, float]]], clip
) -> list[list[tuple[float, float]]]:
    """Clip open polylines to the card, dropping anything outside it."""
    from shapely.geometry import LineString, MultiLineString

    out: list[list[tuple[float, float]]] = []
    for line in lines:
        if len(line) < 2:
            continue
        clipped = LineString([(float(x), float(y)) for x, y in line]).intersection(clip)
        if clipped.is_empty:
            continue
        if isinstance(clipped, LineString):
            candidates = [clipped]
        elif isinstance(clipped, MultiLineString):
            candidates = list(clipped.geoms)
        else:
            candidates = []
        for item in candidates:
            points = [(x, y) for x, y in item.coords]
            if len(points) >= 2:
                out.append(points)
    return out


def build_items(source: Path) -> list[dict[str, Any]]:
    document = EproArchive.read(source).document
    centre = extract_boards(source)[0].centre
    clip = _card_polygon()
    items: list[dict[str, Any]] = []

    for record in document.records:
        body = record.body
        if not isinstance(body, dict):
            continue
        if record.type not in ART_TYPES:
            continue
        layer = body.get("layerId")
        if layer not in ART_LAYERS:
            continue

        # ⚠️ **``REGION`` records are design constraints, not artwork.** The
        # template carries two of them (top and bottom copper, 7.24 x 56.13 mm,
        # just left of the illustration window) with ``regionType=PROHIBIT``:
        # a keepout that forbids tracks, components, copper and fills. Drawn as
        # a filled shape it came out as a solid block of copper beside the
        # illustration -- which is the *opposite* of what the record means, and
        # it appeared on the card, in the .epro2 and in the Gerber.
        #
        # The rule stays in the source template, which is where a designer
        # needs it; a finished card has nothing for it to constrain.
        if record.type == "REGION":
            continue

        if record.type == "IMAGE":
            rings = [
                [
                    _relative(
                        (point[0] * MM_PER_MIL, point[1] * MM_PER_MIL), centre
                    )
                    for point in ring
                ]
                for ring in _image_rings(body, int(layer))
            ]
            kind = "fill"
            width = 0.2
        elif record.type == "FILL":
            rings = [
                [_relative(point, centre) for point in ring]
                for ring in _fill_rings(body)
            ]
            kind = "fill"
            # FILL records carry the editor's own raw width (0.2); it is not a
            # millimetre measurement and the writer stores it verbatim.
            width = float(body.get("width", 0.2))
        else:
            converted = record_to_mm(record)
            path = converted.get("path") if converted else None
            if record.type == "LINE":
                path = [
                    float(body.get("startX", 0.0)) * MM_PER_MIL,
                    centre[1] - float(body.get("startY", 0.0)) * MM_PER_MIL,
                    "L",
                    float(body.get("endX", 0.0)) * MM_PER_MIL,
                    centre[1] - float(body.get("endY", 0.0)) * MM_PER_MIL,
                ]
            if not isinstance(path, list) or not path:
                continue
            from pcbart.geometry import densify_path

            polylines = [
                [_relative(point, centre) for point in line]
                for line in densify_path(path)
                if len(line) >= 2
            ]
            if not polylines:
                continue
            polylines = _clip_paths(polylines, clip)
            if not polylines:
                continue
            standalone = isinstance(path[0], str) and path[0] in ("CIRCLE", "R")
            items.append(
                {
                    "kind": "stroke",
                    "layer": int(layer),
                    "width_mm": round(
                        float(body.get("width", 10)) * MM_PER_MIL, 4
                    ),
                    "closed": bool(standalone),
                    "paths": polylines,
                }
            )
            continue

        if not rings:
            continue
        for piece in _clip_rings(rings, clip):
            items.append(
                {
                    "kind": kind,
                    "layer": int(layer),
                    "width": round(width, 4),
                    "rings": piece,
                }
            )

    return items


def build_records(source: Path) -> list[dict[str, Any]]:
    """Raw card primitives, translated so the card centre is the origin.

    The flattened ``items`` above feed the canvas.  These records feed the
    exporters, which should not have to reproduce EasyEDA's image transform:
    the editor already knows how to place its own ``IMAGE`` records.
    """
    document = EproArchive.read(source).document
    centre = extract_boards(source)[0].centre
    dx = -centre[0] / MM_PER_MIL
    dy = -centre[1] / MM_PER_MIL
    records: list[dict[str, Any]] = []
    for record in document.records:
        body = record.body
        if not isinstance(body, dict):
            continue
        if record.type not in ART_TYPES:
            continue
        if body.get("layerId") not in ART_LAYERS:
            continue
        # Same reason as in :func:`build_items`: a PROHIBIT region is a design
        # constraint, not something to draw. Writing it back put a solid copper
        # block on the card.
        if record.type == "REGION":
            continue
        if not _in_card(record, centre):
            continue
        records.append(
            {
                "type": record.type,
                "body": _shift_body(body, dx, dy),
            }
        )
    return records


def build_outline_records(source: Path) -> list[dict[str, Any]]:
    """The source card's board-outline POLY records, translated to the origin.

    EasyEDA writes the outline as separate edge chains: one ``POLY`` per straight
    run or arc.  Merging them into one path with several ``ARC`` tokens looked
    valid to this project's reader but the editor did not draw the board.
    """
    document = EproArchive.read(source).document
    centre = extract_boards(source)[0].centre
    dx = -centre[0] / MM_PER_MIL
    dy = -centre[1] / MM_PER_MIL
    records: list[dict[str, Any]] = []
    for record in document.records:
        body = record.body
        if record.type != "POLY" or not isinstance(body, dict):
            continue
        if body.get("layerId") != 11:
            continue
        records.append(
            {
                "type": "POLY",
                "body": _shift_body(body, dx, dy),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="override the template file to read",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        from pcbart.paths import asset_root

        source = asset_root() / "通行证模板源文件" / SOURCE_NAME
    if not source.exists():
        print(f"模板源文件不存在：{source}", file=sys.stderr)
        return 1

    # ⚠️ **Both templates, not just the short one.** The two files carry
    # different fixed artwork -- the short card has the 阵营 logo area and a
    # different illustration window decoration set (327 items against 187) --
    # so a card built from the wrong one draws the other card's layout. The
    # board's `template` key picks which set is used; the outline is identical
    # in both (measured: same 82 vertices, same extent), so it stays shared.
    variants: dict[str, dict[str, Any]] = {}
    for key, file_name in TEMPLATE_VARIANTS.items():
        path = source.parent / file_name
        if not path.exists():
            print(f"模板源文件不存在：{path}", file=sys.stderr)
            return 1
        variants[key] = {
            "source": path.name,
            "items": build_items(path),
            "records": build_records(path),
        }

    short = variants["short"]
    outline_records = build_outline_records(source.parent / short["source"])
    payload = {
        "version": 2,
        "source": short["source"],
        "card_width_mm": 50.0,
        "card_height_mm": 100.0,
        # Kept at the top level so every existing reader keeps working and the
        # short card stays the default.
        "items": short["items"],
        "records": short["records"],
        "outline_records": outline_records,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    chunks = [encoded[index : index + 100] for index in range(0, len(encoded), 100)]
    args.out.write_text(
        '"""Generated by ``tools/extract_badge_art.py`` -- do not edit by hand.\n\n'
        "The 通行证 card's fixed artwork, in board millimetres relative to the\n"
        "card centre, plus the source project's own primitive records.  The\n"
        "payload is compressed because the eight IMAGE paths alone contain\n"
        "tens of thousands of coordinates; keeping it as a Python module means\n"
        "a packaged build cannot miss it through a forgotten data-file rule.\n"
        '"""\n\n'
        "import base64\nimport json\nimport zlib\n\n"
        "_PAYLOAD_B64 = \"\".join(\n"
        + pprint.pformat(chunks, width=100)
        + "\n)\n\n"
        "PAYLOAD = json.loads(\n"
        "    zlib.decompress(base64.b64decode(_PAYLOAD_B64)).decode(\"utf-8\")\n"
        ")\n",
        encoding="utf-8",
    )
    print(f"写入 {args.out}")
    for key, variant in variants.items():
        shapes = variant["items"]
        fill_count = sum(1 for item in shapes if item["kind"] == "fill")
        stroke_count = sum(1 for item in shapes if item["kind"] == "stroke")
        print(
            f"  [{key}] {variant['source']}：{len(shapes)} 个装饰"
            f"（填充 {fill_count}，描边 {stroke_count}）；"
            f"原始记录 {len(variant['records'])} 条"
        )
    print(f"  板框记录 {len(outline_records)} 条（两套共用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
