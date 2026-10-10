"""Several finished boards in one project file, laid out side by side.

Why this exists
---------------
Exporting eight cards gives eight ``.epro2`` files, and looking at them means
opening eight files in the editor. This module writes the same boards into
**one** project so they can be compared at a glance -- the elements of each card
stay exactly what the single-card export produced, because the records are read
back out of those files rather than rebuilt.

How the merge works
-------------------
Every source file is read, its geometry records are translated by the cell the
board is going into, and they are appended to one fresh project skeleton. The
skeleton (``DOCHEAD`` / ``LAYER`` / ``PRIMITIVE`` / ...) comes from
:func:`pcbart.templates.build_minimal_project`, which is the same base a
single-board export uses, so the merged file has no document-level difference
from an ordinary export.

⚠️ **A uniform translation is the whole transform.** The records store absolute
coordinates -- ``path`` runs for the primitives, ``startX``/``startY`` for
``IMAGE`` and ``OBJ``, ``x``/``y`` for ``STRING`` -- and shifting all of them by
the same delta moves a board without altering its internal geometry. This is the
same operation the badge templates were extracted with
(:mod:`tools.extract_badge_art`), which is why the shift lives here and that
tool imports it: one implementation, already measured against the editor.

⚠️ **``IMAGE.path`` is absolute too, even though it looks local.** A record's
``path`` can extend well past its ``width``/``height`` -- the 罗德岛标志 is
1110x1110 with a 1934-unit-wide path -- and shifting only the anchor would tear
the picture away from its own silhouette. Measured on a real card: the path's
delta and the anchor's delta are the same number.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .epru import EproArchive, EproDocument, Record, new_id
from .geometry import Mode
from .templates import build_minimal_project

__all__ = [
    "CARRY_TYPES",
    "shift_path",
    "shift_body",
    "board_bounds",
    "merge_projects",
]

#: Record types that carry a board's own geometry, and therefore travel with it.
#:
#: The document-level records are **not** here: they describe the project itself
#: (layer table, canvas, display flags) and come once from the fresh skeleton.
#: Copying them per board would write the same layer table eight times.
CARRY_TYPES = (
    "POLY",
    "FILL",
    "LINE",
    "IMAGE",
    "STRING",
    "OBJ",
    "BLOB",
    "REGION",
    "PAD",
    "VIA",
    "GROUP",
)

#: The layer the board outline lives on, in EasyEDA Pro's layer table.
OUTLINE_LAYER = 11

#: Geographic fields holding one coordinate each, by axis.
_X_KEYS = ("startX", "endX", "x", "centerX")
_Y_KEYS = ("startY", "endY", "y", "centerY")


def shift_path(node: Any, dx: float, dy: float) -> Any:
    """Translate every absolute coordinate in a path, leaving angles alone.

    The grammar mixes coordinates with numbers that are *not* coordinates --
    an ``ARC`` token's sweep angle is one -- so the stream is walked rather than
    scaled blindly. ``R`` and ``CIRCLE`` are single-shape paths whose anchor is
    the first pair; the rest are ``L`` runs interleaved with ``ARC`` and ``C``
    segments.
    """
    if not isinstance(node, list) or not node:
        return node
    if isinstance(node[0], list):
        return [shift_path(item, dx, dy) for item in node]

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
            if token in (Mode.ARC, Mode.CARC) and index + 2 < len(node):
                out.append(node[index])  # the sweep angle is not a coordinate
                out.append(float(node[index + 1]) + dx)
                out.append(float(node[index + 2]) + dy)
                index += 3
            elif token == Mode.C and index + 5 < len(node):
                # Two control points and an endpoint, all absolute.
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


def shift_body(body: dict, dx: float, dy: float) -> dict:
    """A record body translated by ``(dx, dy)``, everything else untouched.

    Only fields that hold a coordinate move. Widths, angles, mirror flags, layer
    ids, fonts and text are copied as they are -- re-deriving any of them from
    the moved geometry is how a turned picture ends up facing the wrong way.
    """
    out = dict(body)
    if "path" in out:
        out["path"] = shift_path(out.get("path"), dx, dy)
    for key in _X_KEYS:
        if isinstance(out.get(key), (int, float)) and not isinstance(out.get(key), bool):
            out[key] = float(out[key]) + dx
    for key in _Y_KEYS:
        if isinstance(out.get(key), (int, float)) and not isinstance(out.get(key), bool):
            out[key] = float(out[key]) + dy
    return out


def _path_points(node: Any) -> list[tuple[float, float]]:
    from .geometry import densify_path

    try:
        return [point for ring in densify_path(node) for point in ring]
    except Exception:  # noqa: BLE001 - an undecodable path contributes no bounds
        return []


def _record_points(record: Record) -> list[tuple[float, float]]:
    """Every absolute point a record occupies, for bounding-box purposes."""
    body = record.body
    if not isinstance(body, dict):
        return []
    points = _path_points(body.get("path"))
    if record.type == "IMAGE":
        # ``IMAGE.path`` is a silhouette in the record's own frame and reaches
        # past the placement; the anchor plus the stated size is what the editor
        # actually puts on the board, so that is what the layout measures.
        from .units import EPRU_UNIT_MM

        start_x = body.get("startX")
        start_y = body.get("startY")
        if isinstance(start_x, (int, float)) and isinstance(start_y, (int, float)):
            width = float(body.get("width") or 0.0)
            height = float(body.get("height") or 0.0)
            points = [
                (float(start_x), float(start_y)),
                (float(start_x) + width, float(start_y) - height),
            ]
        return points
    for x_key, y_key in (
        ("startX", "startY"),
        ("endX", "endY"),
        ("x", "y"),
        ("centerX", "centerY"),
    ):
        x, y = body.get(x_key), body.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            points.append((float(x), float(y)))
    return points


def board_bounds(records: Iterable[Record]) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of the board, in file units.

    The **outline** is preferred over every other record, and that is what
    makes the grid even. Decoration is not confined to the board: the mask
    windows and the wordmark strip sit proud of the copper, and one card's
    ``IMAGE`` records reach roughly 20 mm past its own edge. Laying the cards
    out on those extremes would space them by whichever decoration happens to
    stick out furthest, so two cards with different artwork would land on
    different pitches -- visibly ragged, and nothing to do with the boards.

    Falls back to all geometry for a document with no outline record, which is
    what a hand-made file can look like.
    """
    outline = [
        record
        for record in records
        if record.type == "POLY"
        and isinstance(record.body, dict)
        and record.body.get("layerId") == OUTLINE_LAYER
    ]
    boxes = outline or list(records)

    xs: list[float] = []
    ys: list[float] = []
    for record in boxes:
        for x, y in _record_points(record):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _carried(document: EproDocument) -> list[Record]:
    return [r for r in document.records if r.type in CARRY_TYPES]


def _layout(
    boxes: Sequence[tuple[float, float, float, float]],
    columns: int,
    gap: float,
) -> list[tuple[float, float]]:
    """Where each board's lower-left corner goes, in file units.

    Boards are laid out on a grid with their **lower-left corners** aligned, so
    a row of cards shares a baseline no matter how much decoration hangs below
    one of them. Rows run downward because the file's Y axis points up: the
    second row sits at a smaller Y, which is what puts it under the first when
    the editor draws the board.
    """
    if not boxes:
        return []
    count = max(1, int(columns))
    cell_w = max(box[2] - box[0] for box in boxes)
    cell_h = max(box[3] - box[1] for box in boxes)
    step_x = cell_w + gap
    step_y = cell_h + gap
    places: list[tuple[float, float]] = []
    for index, box in enumerate(boxes):
        column = index % count
        row = index // count
        target_x = column * step_x
        target_y = -row * step_y
        places.append((target_x - box[0], target_y - box[1]))
    return places


def merge_projects(
    sources: Sequence[str | Path],
    output_path: str | Path,
    *,
    title: str = "ArtBoard",
    columns: int = 4,
    gap_mm: float = 5.0,
) -> dict:
    """Write every project in ``sources`` into one file, side by side.

    Returns a small report: how many boards went in, the grid used, and the
    bounding box of the result in millimetres. The caller can print it rather
    than re-deriving the numbers from the file.
    """
    from .units import EPRU_UNIT_MM

    gap = float(gap_mm) / EPRU_UNIT_MM

    read: list[list[Record]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for source in sources:
        document = EproArchive.read(source).document
        records = _carried(document)
        if not records:
            continue
        read.append(records)
        boxes.append(board_bounds(records))
    if not read:
        raise ValueError("没有一个源文件含有可写入的图形")

    offsets = _layout(boxes, columns, gap)

    document, _uuids = build_minimal_project(title=title)

    seen_blobs: set[str] = set()
    written = 0
    for index, (records, (dx, dy)) in enumerate(zip(read, offsets)):
        # A group's id is a plain string the editor assigns ("1", "2", ...), so
        # two boards merged into one file both call their first group "1" and
        # their members would join whichever came first. Namespacing by board
        # keeps each board's groups its own; the *displayed* name lives in the
        # payload's ``groupName`` and is left alone.
        group_ids = {
            str(record.id)
            for record in records
            if record.type == "GROUP" and record.id is not None
        }
        for record in records:
            record_type = record.type
            body = record.body
            if not isinstance(body, dict):
                continue

            moved = shift_body(body, dx, dy)

            if record_type == "BLOB":
                # Blob ids are content hashes, so two cards carrying the same
                # picture produce the same id. The first copy already serves
                # both, and ``OBJ.path`` says ``blob:<id>`` -- the pair only
                # works while the two agree, so this id is the one id that must
                # be carried over rather than drawn fresh.
                if str(record.id) in seen_blobs:
                    continue
                seen_blobs.add(str(record.id))
                record_id = str(record.id)
            elif record_type == "GROUP" and record.id is not None:
                record_id = f"{index}:{record.id}"
            else:
                record_id = new_id()

            if group_ids and moved.get("groupId") in group_ids:
                moved = dict(moved)
                moved["groupId"] = f"{index}:{moved['groupId']}"

            # ⚠️ **A ``BLOB`` belongs to its own document section, not to the
            # PCB's.** This is not a formatting nicety: an ``OBJ`` resolves its
            # picture through ``path: "blob:<id>"``, and the editor looks that
            # id up among the ``BLOB`` records of the ``BLOB`` section. A blob
            # written into the PCB section next to its ``OBJ`` is simply not
            # found, and the picture comes back as an empty rectangle --
            # measured: the QR symbols and the illustration vanished from the
            # merged file while every record was present and every id matched.
            # :func:`pcbart.generator.generate_outline_board` writes them the
            # same way for a single board.
            document.append_to_section(
                Record(
                    header={
                        "type": record_type,
                        "ticket": document.next_ticket(),
                        "id": record_id,
                    },
                    body=moved,
                ),
                doc_type="BLOB" if record_type == "BLOB" else "PCB",
            )
        written += 1

    destination = Path(output_path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
    EproArchive.create(title, document=document).write(destination)

    width = max(box[2] - box[0] for box in boxes)
    height = max(box[3] - box[1] for box in boxes)
    rows = math.ceil(len(read) / max(1, int(columns)))
    return {
        "output": str(destination),
        "boards": written,
        "columns": max(1, int(columns)),
        "rows": rows,
        "board_mm": (
            round(width * EPRU_UNIT_MM, 3),
            round(height * EPRU_UNIT_MM, 3),
        ),
        "panel_mm": (
            round(
                (
                    max(1, int(columns)) * width
                    + (max(1, int(columns)) - 1) * gap
                )
                * EPRU_UNIT_MM,
                3,
            ),
            round((rows * height + (rows - 1) * gap) * EPRU_UNIT_MM, 3),
        ),
    }
