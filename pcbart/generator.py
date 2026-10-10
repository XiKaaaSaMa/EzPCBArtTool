"""High-level generation of EasyEDA Pro boards.

The entry point for M0 is :func:`generate_rect_board`, which writes a project
containing nothing but a rectangular board outline. That is the smallest file
that still exercises the whole pipeline: document skeleton, unit conversion,
path encoding, and the ZIP container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


#: Width written into every ``FILL`` record, copied from the captured project.
#:
#: Duplicated from :mod:`pcbart.export` rather than imported: that module
#: imports this one, so importing back would be a cycle. The value is the
#: editor's own, and both writers have to agree on it or a cutout and the
#: artwork beside it would claim different pen widths.
_FILL_WIDTH_RAW = 0.2


def fill_record(
    layer_id: int,
    rings_mm: list,
    width: float,
    document: EproDocument,
    group_id: str = "",
) -> Record:
    """A ``FILL`` record for one complex polygon.

    ``rings_mm`` is ``[outer, hole, hole, ...]`` in millimetres, already wound
    the way the editor needs: **the holes opposite to the outer ring**. Squares
    whose hole shared the outer ring's winding came out solid in testing.

    ``width`` is written **verbatim**, not converted. The 75 ``FILL`` records
    captured from a real project all carry ``0.2``; converting 0.2 mm to file
    units produced 7.874, and the field's unit is not something the capture
    settles. Matching the editor's own value beats guessing at a conversion.

    ``group_id`` names the ``GROUP`` record this shape belongs to, as a
    **string**; ``""`` is the ungrouped state. ⚠️ The field is not the same type
    in both halves of the format -- PB primitives carry ``"groupId": 0`` when
    ungrouped while symbol records carry ``"groupId": ""`` -- and in a
    real hand-made card it is the **string** form that accompanies a ``GROUP``
    record. Writing ``0`` for an ungrouped shape and a string number when
    grouped matches what the editor itself produced.

    ``rings_mm`` may hold :class:`pcbart.geometry.ArcRing` entries instead of
    plain vertex lists. The ``FILL`` grammar carries curves natively -- the
    template's own records mix ``L`` and ``ARC`` in one ring -- so a traced
    curve does not have to be flattened into chords on the way out. Both kinds
    go through :func:`pcbart.primitives.convert_path_to_unit`, which already
    knows how to convert an ``ARC`` token's operands.
    """
    from .geometry import (
        ArcRing,
        arc_polygon_to_path,
        complex_polygon_to_path,
    )
    from .primitives import convert_path_to_unit, fill

    if rings_mm and isinstance(rings_mm[0], ArcRing):
        path_mm = arc_polygon_to_path(rings_mm)
    else:
        path_mm = complex_polygon_to_path(rings_mm)
    body = fill(layer_id, path_mm, width)
    body["path"] = convert_path_to_unit(path_mm)
    body["width"] = width
    body["groupId"] = group_id

    return Record(
        header={"type": "FILL", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def group_record(group, document: EproDocument, index: int | None = None) -> Record:
    """A ``GROUP`` record: one set of primitives the editor treats as a unit.

    ⚠️ **The whole encoding, because it is not inferable from the name.** A
    group is declared by its own record and joined by the members' ``groupId``
    fields; nothing about a member's geometry says it is grouped. Measured on a
    card the user built in the editor, the record is::

        {"type":"GROUP","ticket":2193,"id":"1"}||{"groupName":"group1"}

    so the id is a **string**, the name lives in the payload under
    ``groupName``, and members point at that id from their own payloads.

    There is no ``firstTicket`` here. The hand-made files do carry one, and in
    some of them it points at a ticket that does not exist -- so it is evidently
    a hint the editor maintains for itself and not something a reader needs. A
    group whose name is empty writes no ``groupName`` key at all rather than an
    empty string, matching the records that were serialised without a name.
    """
    body: dict[str, Any] = {}
    if str(getattr(group, "name", "")).strip():
        body["groupName"] = str(group.name).strip()
    return Record(
        header={
            "type": "GROUP",
            "ticket": document.next_ticket(),
            "id": str(getattr(group, "id", "")),
        },
        body=body,
    )


def outline_record_from_path(
    path_mm: Sequence,
    document: EproDocument,
    line_width_mm: float = 0.125,
) -> Record:
    """Build a board-outline ``POLY`` from an existing token stream.

    Used when the outline carries native ``ARC`` segments.  Going back through
    a vertex list here would silently turn every rounded corner into a chord.
    """
    from .primitives import convert_path_to_unit
    from .units import mm_to_unit

    path = list(path_mm)
    if len(path) < 3:
        raise ValueError("a closed outline needs at least 3 vertices")
    body = poly(11, path, line_width_mm)
    body["path"] = convert_path_to_unit(path)
    body["width"] = mm_to_unit(line_width_mm)
    return Record(
        header={"type": "POLY", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


def decoration_records(
    decorations: Sequence[dict],
    document: EproDocument,
) -> list[Record]:
    """Fixed template artwork as ordinary ``FILL`` and ``POLY`` records.

    The canvas draws the same data, so a decoration visible on screen is a
    decoration in the file.  A closed stroke is closed explicitly: the source
    circles and rectangles are outlines, not filled areas.
    """
    from .primitives import Point, convert_path_to_unit, poly
    from .units import mm_to_unit

    records: list[Record] = []
    for item in decorations:
        if not isinstance(item, dict):
            continue
        layer = int(item.get("layer", 0))
        if item.get("kind") == "fill":
            rings = item.get("rings") or []
            if rings:
                records.append(
                    fill_record(layer, rings, _FILL_WIDTH_RAW, document)
                )
            continue

        for line in item.get("paths") or []:
            points = [Point(float(x), float(y)) for x, y in line]
            if len(points) < 2:
                continue
            if item.get("closed") and points[0] != points[-1]:
                points.append(points[0])
            path = polyline_path(points)
            body = poly(layer, path, float(item.get("width_mm", 0.254)))
            body["path"] = convert_path_to_unit(path)
            body["width"] = mm_to_unit(float(item.get("width_mm", 0.254)))
            records.append(
                Record(
                    header={
                        "type": "POLY",
                        "ticket": document.next_ticket(),
                        "id": new_id(),
                    },
                    body=body,
                )
            )
    return records


def raw_decoration_records(
    decorations: Sequence[dict],
    document: EproDocument,
) -> list[Record]:
    """Copy the template's own primitives back into a new project.

    The bodies already carry EasyEDA's raw units and, for ``IMAGE``, its own
    frame/angle/mirror fields.  Re-encoding them from sampled millimetres put
    every image at a slightly different place from the source project; copying
    them lets the editor apply the transform it wrote in the first place.
    """
    records: list[Record] = []
    for item in decorations:
        if not isinstance(item, dict):
            continue
        record_type = str(item.get("type") or "")
        body = item.get("body")
        if not record_type or not isinstance(body, dict):
            continue
        records.append(
            Record(
                header={
                    "type": record_type,
                    "ticket": document.next_ticket(),
                    "id": new_id(),
                },
                body=dict(body),
            )
        )
    return records


def generate_outline_board(
    outer_mm: Sequence[tuple[float, float]],
    output_path: str | Path = "board.epro2",
    title: str = "ArtBoard",
    line_width_mm: float = 0.125,
    routing_corner: str = "L",
    editor_version: str = "",
    outer_circle: tuple[float, float, float] | None = None,
    outer_path: Sequence | None = None,
    raw_outline: Sequence[dict] = (),
    fills: Sequence[tuple[int, list, float]] = (),
    cutouts: Sequence[Sequence[tuple[float, float]]] = (),
    texts: Sequence = (),
    decorations: Sequence[dict] = (),
    raw_decorations: Sequence[dict] = (),
    color_objects: Sequence[dict] = (),
    groups: Sequence = (),
) -> Path:
    """Write a board from an outer contour, its openings, wording and artwork.

    ``fills`` carries the artwork: ``(layer_id, complex_polygon_mm, width)``
    triples, each written as one ``FILL``. Any layer takes them -- copper for a
    metal field, silkscreen for white ink -- so the parameter is named for the
    primitive rather than for one of its users.

    They arrive already wound the way the editor needs -- outer clockwise, holes
    counter-clockwise -- because that decision belongs to whatever built the
    geometry, not to the writer.

    ``cutouts`` are the openings cut through the board. Each is written as a
    ``FILL`` on the **MULTI layer** -- the layer that means "all layers", i.e.
    material removed everywhere -- which is exactly how the template files the
    user supplied encode their lanyard slot. ⚠️ This is *not* the hole feature
    the project dropped in 2026-09: that was a hole **generator** (drill hits,
    plated barrels, slot channels). This preserves an opening the template
    already had, and a board exported without it comes back solid -- the canvas
    would show a slot the fab never cuts.

    ``texts`` are template wordings, written as the editor's own ``STRING``
    records so they stay editable when the file is opened there.
    """
    from .validation import check_outline, raise_for_errors

    outer = [(float(x), float(y)) for x, y in outer_mm]

    report = check_outline(outer)
    raise_for_errors(report, "外框未通过校验")

    document, _uuids = build_minimal_project(
        title=title,
        editor_version=editor_version,
        routing_corner=routing_corner,
    )

    # ⚠️ Groups first, and this is not a style choice. Members join a group by
    # naming it from their own payload, so a FILL written before its GROUP
    # record refers to something the document has not declared yet. The editor
    # repairs what it can, but the file is then not the shape it was measured
    # from. Appending them here also keeps them ahead of every geometry record
    # below, which is the order the hand-made card uses.
    for group in groups:
        members = list(getattr(group, "members", ()) or ())
        if len(members) < 2:
            # A group with fewer than two live members is not a group. Writing
            # one would put a row in the editor's element list that selects
            # nothing, and the document already prunes these before export --
            # this is the second line of defence for a hand-built board.
            continue
        document.append_to_section(group_record(group, document), doc_type="PCB")

    if raw_outline:
        for record in raw_decoration_records(raw_outline, document):
            document.append_to_section(record, doc_type="PCB")
    elif outer_circle is not None:
        document.append_to_section(
            circle_outline_record(
                outer_circle[0], outer_circle[1], outer_circle[2], document, line_width_mm
            ),
            doc_type="PCB",
        )
    elif outer_path:
        document.append_to_section(
            outline_record_from_path(outer_path, document, line_width_mm=line_width_mm),
            doc_type="PCB",
        )
    else:
        document.append_to_section(
            outline_record(outer, document, line_width_mm=line_width_mm, close=True),
            doc_type="PCB",
        )

    source_decorations = (
        raw_decoration_records(raw_decorations, document)
        if raw_decorations
        else decoration_records(decorations, document)
    )
    for record in source_decorations:
        document.append_to_section(record, doc_type="PCB")

    # Colour pictures: an ``OBJ`` (the placement) **plus** a ``BLOB`` (the
    # bytes it points at). ⚠️ Both records are required. The format reference
    # says ``OBJ.path`` may carry a Data URL, but the editor reads only
    # ``blob:<id>`` -- a file whose OBJ embedded its picture inline imported as
    # an empty rectangle. The pair is written the way the editor writes it.
    for pair in color_objects:
        if not isinstance(pair, dict):
            continue
        blob = pair.get("blob")
        if isinstance(blob, dict) and isinstance(blob.get("body"), dict):
            document.append_to_section(
                Record(
                    header={
                        "type": "BLOB",
                        "ticket": document.next_ticket(),
                        "id": str(blob.get("id") or new_id()),
                    },
                    body=dict(blob["body"]),
                ),
                doc_type="BLOB",
            )
        item = pair.get("object")
        body = item.get("body") if isinstance(item, dict) else None
        if not isinstance(body, dict):
            continue
        document.append_to_section(
            Record(
                header={
                    "type": "OBJ",
                    "ticket": document.next_ticket(),
                    "id": str(item.get("id") or new_id()),
                },
                body=dict(body),
            ),
            doc_type="PCB",
        )

    for record in fills:
        # ⚠️ Not unpacked in the loop header. A ``CopperRecord`` unpacks to
        # exactly ``(layer_id, rings, width)`` for every existing caller, but
        # it also carries the group id as an *attribute* -- so the loop keeps
        # the object and reads the three values off it. Unpacking first would
        # throw the group away before it could be written.
        layer_id, rings_mm, width_value = record
        group_id = str(getattr(record, "group", "") or "")
        document.append_to_section(
            fill_record(
                layer_id,
                rings_mm,
                float(width_value),
                document,
                group_id=group_id,
            ),
            doc_type="PCB",
        )

    multi_id = _layer_id("MULTI")
    for ring in cutouts:
        if len(ring) >= 3:
            document.append_to_section(
                fill_record(multi_id, [list(ring)], _FILL_WIDTH_RAW, document),
                doc_type="PCB",
            )

    for item in texts:
        record = text_record(item, document)
        if record is not None:
            document.append_to_section(record, doc_type="PCB")

    archive = EproArchive(document=document, meta={"title": title})
    return archive.write(output_path)


def _layer_id(layer_type: str) -> int:
    from .layers import layer_by_type

    entry = layer_by_type(layer_type)
    if entry is None:
        raise KeyError(f"没有该层的 EasyEDA 定义: {layer_type}")
    return entry.id


#: Width written into a ``STRING`` record, matching the template files.
TEXT_STROKE_RAW = 7.9921

#: Where the editor's own ``STRING`` records put their z-order.
TEXT_Z_INDEX = -1


def text_record(item, document: EproDocument) -> Record | None:
    """One template line as the editor's own ``STRING`` record.

    Written as text rather than as traced outlines, because the whole point of
    the user's request is that the wording stays editable -- a ten-millimetre
    glyph exported as geometry would reach the EDA as a shape, and changing a
    letter would mean redrawing it.

    The fields and their values are copied from the template files the user
    supplied: same origin (``LEFT_BOTTOM``), same angle convention, same
    ``zIndex`` of ``-1``. Those files are the only evidence available for what
    a hand-authored text record looks like, so matching them beats inventing a
    plausible-looking set of defaults.
    """
    from .units import mm_to_unit

    text = getattr(item, "text", "")
    if not text:
        return None

    from .design_layers import LAYERS, LayerStack

    # ``layer_for_name`` gives the layer's stable *key*; the file needs the
    # EasyEDA type string, so the table is read for it rather than guessed by
    # upper-casing the key (``silk_top`` would become ``SILK_TOP``, which is
    # not a type the layer table has).
    key = LayerStack().layer_for_name(getattr(item, "layer", ""))
    entry = next((candidate for candidate in LAYERS if candidate.key == key), None)
    layer_type = entry.easyeda_type if entry is not None else "TOP_SILK"
    body = {
        "partitionId": "",
        "groupId": 0,
        "layerId": _layer_id(layer_type),
        "x": mm_to_unit(getattr(item, "x_mm", 0.0)),
        "y": mm_to_unit(getattr(item, "y_mm", 0.0)),
        "text": text,
        "fontFamily": getattr(item, "font", "") or "default2",
        # The editor stores the font size in the same units as everything else.
        "fontSize": mm_to_unit(getattr(item, "size_mm", 1.0)),
        "strokeWidth": TEXT_STROKE_RAW,
        "bold": 0,
        "italic": 0,
        "origin": "LEFT_BOTTOM",
        "angle": float(getattr(item, "rotation_deg", 0.0)),
        "reverse": False,
        "expansion": 0,
        "mirror": False,
        "locked": False,
        "zIndex": TEXT_Z_INDEX,
        "specialColor": None,
    }
    return Record(
        header={"type": "STRING", "ticket": document.next_ticket(), "id": new_id()},
        body=body,
    )


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
