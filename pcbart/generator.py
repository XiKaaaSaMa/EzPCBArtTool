"""High-level generation of EasyEDA Pro boards.

The entry point for M0 is :func:`generate_rect_board`, which writes a project
containing nothing but a rectangular board outline. That is the smallest file
that still exercises the whole pipeline: document skeleton, unit conversion,
path encoding, and the ZIP container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .epru import EproArchive, EproDocument, Record, new_id
from .primitives import Point, board_outline_rect, convert_path_to_unit, polyline_path, poly
from .shapes import center_on_origin, polygon_edge_lengths, translate_points
from .templates import build_minimal_project

__all__ = [
    "BoardSpec",
    "generate_rect_board",
    "generate_board_with_outline",
    "generate_polyline_board",
    "generate_outline_board",
    "outline_record",
    "generate_from_image",
]


@dataclass
class BoardSpec:
    """板框参数，单位统一为 mm。"""

    width_mm: float
    height_mm: float
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    line_width_mm: float = 0.125
    title: str = "ArtBoard"

    def validate(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("板框宽高必须为正数")
        if self.line_width_mm <= 0:
            raise ValueError("板框线宽必须为正数")


def _outline_record(spec: BoardSpec, document: EproDocument) -> Record:
    """Build the ``POLY`` record for a rectangular outline."""
    body_mm = board_outline_rect(
        x_mm=spec.origin_x_mm,
        y_mm=spec.origin_y_mm,
        width_mm=spec.width_mm,
        height_mm=spec.height_mm,
        line_width_mm=spec.line_width_mm,
    )

    # Convert the millimetre payload into raw units at the boundary.
    from .primitives import convert_path_to_unit
    from .units import mm_to_unit

    body = dict(body_mm)
    body["path"] = convert_path_to_unit(body_mm["path"])
    body["width"] = mm_to_unit(spec.line_width_mm)

    return Record(
        header={"type": "POLY", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def generate_board_with_outline(
    spec: BoardSpec,
    output_path: str | Path,
    editor_version: str = "",
) -> Path:
    """Write a ``.epro2`` containing just a rectangular board outline.

    The outline is a single closed ``POLY`` on layer 11 (Board Outline), which
    is the shape the board-frame tool itself draws.
    """
    spec.validate()

    document, _uuids = build_minimal_project(title=spec.title, editor_version=editor_version)
    outline = _outline_record(spec, document)
    document.append_to_section(outline, doc_type="PCB")

    archive = EproArchive(document=document, meta={"title": spec.title})
    return archive.write(output_path)


def generate_rect_board(
    width_mm: float,
    height_mm: float,
    output_path: str | Path,
    title: str = "ArtBoard",
    line_width_mm: float = 0.125,
) -> Path:
    """Convenience wrapper around :func:`generate_board_with_outline`."""
    spec = BoardSpec(
        width_mm=width_mm,
        height_mm=height_mm,
        line_width_mm=line_width_mm,
        title=title,
    )
    return generate_board_with_outline(spec, output_path)


# ------------------------------------------------------------- arbitrary outlines


def outline_record(
    points_mm: Sequence[tuple[float, float]],
    document: EproDocument,
    line_width_mm: float = 0.125,
    close: bool = True,
) -> Record:
    """Build a ``POLY`` record for an arbitrary closed outline.

    The outline is written as one polyline on layer 11. EasyEDA Pro stores the
    closing edge implicitly, so ``points_mm`` should *not* repeat its first
    vertex; pass ``close=True`` to append it when you want the file to read as
    a visibly closed loop.

    This is the free-angle form of the polyline tool: vertices may sit at any
    angle, with no 45/90-degree constraint applied.
    """
    from .primitives import convert_path_to_unit
    from .units import mm_to_unit

    if len(points_mm) < 3:
        raise ValueError("a closed outline needs at least 3 vertices")

    vertices = [Point(float(x), float(y)) for x, y in points_mm]
    if close and vertices[0] != vertices[-1]:
        vertices.append(vertices[0])

    path_mm = polyline_path(vertices)
    body = poly(11, path_mm, line_width_mm)
    body["path"] = convert_path_to_unit(path_mm)
    body["width"] = mm_to_unit(line_width_mm)

    return Record(
        header={"type": "POLY", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def generate_polyline_board(
    points_mm: Sequence[tuple[float, float]],
    output_path: str | Path,
    title: str = "ArtBoard",
    line_width_mm: float = 0.125,
    origin: tuple[float, float] | None = None,
    center: bool = False,
    close: bool = True,
    expected_edge_mm: float | None = None,
    edge_tolerance_mm: float = 1e-6,
    routing_corner: str = "L",
    editor_version: str = "",
) -> Path:
    """Write a board whose outline is an arbitrary closed polyline.

    Args:
        points_mm: outline vertices, in millimetres, Y-up.
        origin: optional translation applied after centring.
        center: move the shape so its bounding-box centre sits on the origin
            before ``origin`` is applied.
        expected_edge_mm: when given, every edge must have this length. Used to
            assert that a shape really is equilateral before it is written.
        routing_corner: editor drawing-tool corner mode, ``"L"`` for free angle.

    Returns the path written.
    """
    points = [(float(x), float(y)) for x, y in points_mm]
    if len(points) < 3:
        raise ValueError("a closed outline needs at least 3 vertices")
    if line_width_mm <= 0:
        raise ValueError("板框线宽必须为正数")

    if expected_edge_mm is not None:
        lengths = polygon_edge_lengths(points, closed=True)
        worst = max(abs(length - expected_edge_mm) for length in lengths)
        if worst > edge_tolerance_mm:
            raise ValueError(
                f"outline is not equilateral: worst edge deviates by {worst:.6f} mm "
                f"(tolerance {edge_tolerance_mm} mm)"
            )

    if center:
        points = center_on_origin(points)
    if origin is not None:
        points = translate_points(points, origin)

    # Validate before writing: a self-intersecting or degenerate ring cannot
    # become a board, and it is cheaper to fail here than in the editor.
    from .validation import check_outline

    report = check_outline(points)
    if not report.ok:
        details = "; ".join(str(issue) for issue in report.errors)
        raise ValueError(f"outline failed validation: {details}")

    document, _uuids = build_minimal_project(
        title=title,
        editor_version=editor_version,
        routing_corner=routing_corner,
    )
    document.append_to_section(
        outline_record(points, document, line_width_mm=line_width_mm, close=close),
        doc_type="PCB",
    )

    archive = EproArchive(document=document, meta={"title": title})
    return archive.write(output_path)


def circle_outline_record(
    cx_mm: float,
    cy_mm: float,
    radius_mm: float,
    document: EproDocument,
    line_width_mm: float = 0.125,
) -> Record:
    """A board-outline ring written as EasyEDA Pro's native ``CIRCLE`` path.

    Tracing a circle from a bitmap yields a polygon with dozens of vertices, and
    the editor renders that as a faceted disc. The path grammar has a ``CIRCLE``
    form, so a ring that is genuinely circular is written as one: it is exact,
    it stays editable as a circle, and it costs a fraction of the file size.
    """
    from .primitives import Mode, convert_path_to_unit
    from .units import mm_to_unit

    if radius_mm <= 0:
        raise ValueError("圆的半径必须为正数")

    body = poly(11, [0.0, 0.0], line_width_mm)
    body["path"] = convert_path_to_unit([Mode.CIRCLE, cx_mm, cy_mm, radius_mm])
    body["width"] = mm_to_unit(line_width_mm)

    return Record(
        header={"type": "POLY", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def fill_record(
    layer_id: int,
    rings_mm: list,
    width: float,
    document: EproDocument,
) -> Record:
    """A ``FILL`` record for one complex polygon.

    ``rings_mm`` is ``[outer, hole, hole, ...]`` in millimetres, already wound
    the way the editor needs: **the holes opposite to the outer ring**. Squares
    whose hole shared the outer ring's winding came out solid in testing.

    ``width`` is written **verbatim**, not converted. The 75 ``FILL`` records
    captured from a real project all carry ``0.2``; converting 0.2 mm to file
    units produced 7.874, and the field's unit is not something the capture
    settles. Matching the editor's own value beats guessing at a conversion.
    """
    from .geometry import complex_polygon_to_path
    from .primitives import convert_path_to_unit, fill

    path_mm = complex_polygon_to_path(rings_mm)
    body = fill(layer_id, path_mm, width)
    body["path"] = convert_path_to_unit(path_mm)
    body["width"] = width

    return Record(
        header={"type": "FILL", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def generate_outline_board(
    outer_mm: Sequence[tuple[float, float]],
    output_path: str | Path = "board.epro2",
    title: str = "ArtBoard",
    line_width_mm: float = 0.125,
    routing_corner: str = "L",
    editor_version: str = "",
    outer_circle: tuple[float, float, float] | None = None,
    fills: Sequence[tuple[int, list, float]] = (),
) -> Path:
    """Write a board from an outer contour and its artwork.

    Board cutouts are deliberately out of scope: they are the user's to place
    in the editor, where a cutout is drawn interactively and the geometry can
    be judged as it is made. This writer emits the outline and the artwork and
    stops there.

    ``fills`` carries the artwork: ``(layer_id, complex_polygon_mm, width)``
    triples, each written as one ``FILL``. Any layer takes them -- copper for a
    metal field, silkscreen for white ink -- so the parameter is named for the
    primitive rather than for one of its users.

    They arrive already wound the way the editor needs -- outer clockwise, holes
    counter-clockwise -- because that decision belongs to whatever built the
    geometry, not to the writer. A hole *inside a filled shape* is ordinary
    ``FILL`` encoding and is unaffected by the above; only board cutouts are
    gone.
    """
    from .validation import check_outline

    outer = [(float(x), float(y)) for x, y in outer_mm]

    report = check_outline(outer)
    if not report.ok:
        details = "; ".join(str(issue) for issue in report.errors)
        raise ValueError(f"外框未通过校验: {details}")

    document, _uuids = build_minimal_project(
        title=title,
        editor_version=editor_version,
        routing_corner=routing_corner,
    )

    if outer_circle is not None:
        document.append_to_section(
            circle_outline_record(
                outer_circle[0], outer_circle[1], outer_circle[2], document, line_width_mm
            ),
            doc_type="PCB",
        )
    else:
        document.append_to_section(
            outline_record(outer, document, line_width_mm=line_width_mm, close=True),
            doc_type="PCB",
        )

    for layer_id, rings_mm, fill_width in fills:
        document.append_to_section(
            fill_record(layer_id, rings_mm, fill_width, document), doc_type="PCB"
        )

    archive = EproArchive(document=document, meta={"title": title})
    return archive.write(output_path)


def generate_from_image(
    image_path: str | Path,
    output_path: str | Path,
    title: str = "ArtBoard",
    prep_params=None,
    outline_params=None,
    line_width_mm: float = 0.125,
) -> dict[str, object]:
    """Image to board outline in one call.

    Wires :mod:`pcbart.image_prep` into :mod:`pcbart.outline` and then through
    :func:`generate_outline_board`. Returns a report dict describing what was
    produced, which a CLI or GUI can show without re-deriving anything.
    """
    from .image_prep import preprocess
    from .outline import extract_outline_rings

    prep = preprocess(image_path, prep_params)
    rings = extract_outline_rings(prep.mask, outline_params)

    path = generate_outline_board(
        outer_mm=rings.outer,
        output_path=output_path,
        title=title,
        line_width_mm=line_width_mm,
        outer_circle=rings.outer_circle,
    )

    size = _outline_size_mm(rings.outer)
    return {
        "output": str(path),
        "prep": prep.summary(),
        "rings": rings.summary(),
        "ring_count": rings.ring_count,
        "outer_points": len(rings.outer),
        "width_mm": size[0],
        "height_mm": size[1],
        "points_before": rings.points_before,
        "points_after": rings.points_after,
        "outer_is_circle": rings.outer_circle is not None,
    }


def _outline_size_mm(ring: Sequence[tuple[float, float]]) -> tuple[float, float]:
    from .shapes import bounding_box

    min_x, min_y, max_x, max_y = bounding_box(ring)
    return (max_x - min_x, max_y - min_y)


def describe_outline(points_mm: Sequence[tuple[float, float]]) -> dict[str, object]:
    """Summary of an outline: vertex count, edge statistics, size.

    Handy when generating a shape, because it surfaces a mis-built polygon
    (wrong inner radius, flipped winding) before the file is opened in the
    editor.
    """
    from .shapes import bounding_box

    lengths = polygon_edge_lengths(points_mm, closed=True)
    min_x, min_y, max_x, max_y = bounding_box(points_mm)
    return {
        "vertices": len(points_mm),
        "edges": len(lengths),
        "edge_min_mm": min(lengths) if lengths else 0.0,
        "edge_max_mm": max(lengths) if lengths else 0.0,
        "edge_mean_mm": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "perimeter_mm": sum(lengths),
        "width_mm": max_x - min_x,
        "height_mm": max_y - min_y,
        "bbox_mm": (min_x, min_y, max_x, max_y),
    }
