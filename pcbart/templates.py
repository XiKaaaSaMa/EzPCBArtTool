"""Boilerplate records that make up a minimal, valid PCB document.

A PCB document is not just its geometry: EasyEDA Pro expects a document head,
metadata, canvas settings, the full layer table, layer physics, and a handful
of editor state records. The shapes here were copied from a project written by
EasyEDA Pro 3.2.149 so a generated board is indistinguishable from a hand-made
one.

Structure of a well-formed ``.epru`` (order matters)::

    BOARD -> SCH -> SCH_PAGE -> PCB -> CONFIG -> PANEL -> BLOB -> FONT

New geometry belongs at the end of the ``PCB`` section, which
:meth:`pcbart.epru.EproDocument.append_to_section` handles. Note that merely
staying ahead of ``BLOB`` is not sufficient: a record is owned by the nearest
preceding ``DOCHEAD``, and ``CONFIG`` and ``PANEL`` both sit between ``PCB``
and ``BLOB``, so an outline appended before ``BLOB`` would be filed under
``CONFIG`` and the editor would not find it.
"""

from __future__ import annotations

from typing import Any

from .epru import EproDocument, Record, new_id, now_ms
from .layers import DEFAULT_LAYERS

__all__ = [
    "build_board_meta",
    "build_schematic_meta",
    "build_schematic_page_meta",
    "build_pcb_meta",
    "build_pcb_document",
    "build_minimal_project",
    "default_canvas",
    "default_preference",
    "PRIMITIVE_DISPLAY",
]


PRIMITIVE_DISPLAY: tuple[tuple[str, bool, bool], ...] = (
    ("ALL", True, False),
    ("COMPONENT", True, True),
    ("PROPERTY", True, True),
    ("COMPONENTSILK", True, True),
    ("TRACK", True, True),
    ("VIA", True, True),
    ("TESTPOINT", True, True),
    ("PAD", True, True),
    ("PADSPAIR", True, False),
    ("NETWORK", True, False),
    ("SUTUREHOLE", True, True),
    ("TEARDROP", True, True),
    ("RATLINE", False, False),
    ("GROUP", True, False),
    ("TEXT", True, True),
    ("IMAGE", True, True),
    ("PICTURE", True, True),
    ("DIMENSION", True, True),
    ("BOARDOUTLINE", True, True),
    ("SLOTREGION", True, True),
    ("FILLREGIEN", True, True),
    ("COPPEROUTLINE", True, True),
    ("COPPERFILLED", True, True),
    ("PROHIBITEDREGION", True, True),
    ("CONSTRAINT", True, True),
    ("PARTITION", False, False),
    ("LINE", True, True),
    ("LOCKED", True, True),
    ("UNLOCKED", True, True),
    ("SHELL", True, True),
    ("BOSS", True, True),
    ("CREASE", True, True),
    ("TOPSHELLCUT", True, True),
    ("SIDESHELLCUT", True, True),
    ("TOPBOTTOMENTITY", True, True),
    ("SIDEENTITY", True, True),
    ("FPC_STIFFENER", True, True),
)
"""Per-primitive ``display``/``pick`` flags, as EasyEDA Pro 3.2.149 writes them.

The editor keeps one ``PRIMITIVE`` record per primitive type. Only emitting
``ALL`` leaves every other type without an explicit state, and a primitive with
no configuration can end up hidden -- a drilled ``PAD`` that exists in the file
but never appears on the board.
"""


def default_canvas() -> dict[str, Any]:
    """Canvas settings for a board drawn in millimetres."""
    return {
        "originX": 0,
        "originY": 0,
        "unit": "mm",
        "gridXSize": 5,
        "gridYSize": 5,
        "snapXSize": 5,
        "snapYSize": 5,
        "altSnapXSize": 1,
        "altSnapYSize": 1,
        "gridType": "OUTLETS",
        "multiGridType": "NONE",
        "multiGridRatio": 5,
        "highlightValue": 0.5,
    }


def default_preference(routing_corner: str = "L") -> dict[str, Any]:
    """A small but complete editor-preference record.

    Values are in raw units (mil). The defaults here are the ones EasyEDA Pro
    ships with for a two-layer board, except for ``routingCorner``.

    ``routingCorner`` is the drawing tool's corner mode:

    ===========  ==========================================
    ``"L45"``    line, 45-degree constrained
    ``"L90"``    line, 90-degree constrained
    ``"R45"``    arc, 45-degree constrained
    ``"R90"``    arc, 90-degree constrained
    ``"L"``      line, free angle  (线条自由角度)
    ``"R"``      arc, free angle
    ===========  ==========================================

    It is a UI preference for *future* drawing, not geometry: primitives store
    absolute coordinates whatever the mode. It is set to free angle here
    because arbitrary-outline work is the normal case for this tool.
    """
    return {
        "startTrackWidthFollowLast": False,
        "lastTrackWidth": 10,
        "startViaSizeFollowLast": False,
        "lastViaInnerDiameter": 12.0078,
        "lastViaDiameter": 24.0158,
        "snap": True,
        "routingMode": "SURROUND",
        "routingCorner": routing_corner,
        "removeLoop": True,
        "rotatingObject": False,
        "trackFollow": False,
        "stretchTrackMinCorner": 1,
        "realTimeUpdateUnusedLayers": False,
        "unusedPadRange": "VIA",
        "pushVia": "OPTIMIZA_OPEN",
        "pathOptimization4BePushed": "SINGLE",
        "currentPadLayer": 1,
        "currentPadShape": "RECT",
        "currentPadHole": "ROUND",
        "currentPadHoleSize": 24.0158,
        "currentPadSize": 39.3701,
        "currentPadRotation": 0,
        "currentPadCornerRadius": 0,
        "trackWidths": [6, 8, 10, 12, 16, 20, 24, 32, 40],
        "viaDiameters": [24.0158, 31.4961, 39.3701],
        "viaHoleDiameters": [12.0078, 15.748, 19.685],
    }


def build_board_meta(title: str = "Board1") -> dict[str, Any]:
    return {"title": title, "zIndex": 1}


def build_schematic_meta(
    board_uuid: str,
    title: str = "Schematic1",
) -> dict[str, Any]:
    return {"title": title, "source": "", "board": board_uuid, "zIndex": None}


def build_schematic_page_meta(
    schematic_uuid: str,
    title: str = "P1",
) -> dict[str, Any]:
    return {"title": title, "schematic": schematic_uuid, "source": "", "zIndex": 1}


def build_pcb_meta(board_uuid: str, title: str = "PCB1") -> dict[str, Any]:
    return {"title": title, "parent": "", "source": "", "board": board_uuid, "zIndex": None}


def _doc_head(
    doc_type: str,
    uuid: str,
    client: str,
    user_uuid: str,
    edit_version: str = "",
) -> Record:
    stamp = now_ms()
    return Record(
        header={"type": "DOCHEAD"},
        body={
            "docType": doc_type,
            "client": client,
            "uuid": uuid,
            "updateTime": stamp,
            "version": str(stamp),
            "editVersion": edit_version,
            "user": {"uuid": user_uuid},
        },
    )


def build_pcb_document(
    project_title: str = "ArtBoard",
    board_uuid: str | None = None,
    schematic_uuid: str | None = None,
    page_uuid: str | None = None,
    pcb_uuid: str | None = None,
    client: str | None = None,
    user_uuid: str | None = None,
    editor_version: str = "",
    routing_corner: str = "L",
) -> tuple[list[Record], dict[str, str]]:
    """Build the document skeleton for a new project.

    Returns the record list plus the generated uuids so callers can reference
    them (for example when pointing metadata at a board).

    The result contains no geometry; add primitives with
    :meth:`pcbart.epru.EproDocument.append_to_section`.
    """
    board_uuid = board_uuid or new_id()
    schematic_uuid = schematic_uuid or new_id()
    page_uuid = page_uuid or new_id()
    pcb_uuid = pcb_uuid or new_id()
    client = client or new_id()
    user_uuid = user_uuid or new_id()

    records: list[Record] = []

    # --- BOARD ---------------------------------------------------------
    records.append(_doc_head("BOARD", board_uuid, client, user_uuid, editor_version))
    records.append(Record({"type": "META", "ticket": 1, "id": "META"}, build_board_meta()))

    # --- SCH -----------------------------------------------------------
    records.append(_doc_head("SCH", schematic_uuid, client, user_uuid, editor_version))
    records.append(
        Record(
            {"type": "META", "ticket": 1, "id": "META"},
            build_schematic_meta(board_uuid),
        )
    )

    # --- SCH_PAGE ------------------------------------------------------
    records.append(_doc_head("SCH_PAGE", page_uuid, client, user_uuid, editor_version))
    records.append(
        Record(
            {"type": "META", "ticket": 1, "id": "META"},
            build_schematic_page_meta(schematic_uuid),
        )
    )

    # --- PCB -----------------------------------------------------------
    records.append(_doc_head("PCB", pcb_uuid, client, user_uuid, editor_version))
    records.append(
        Record({"type": "META", "ticket": 1, "id": "META"}, build_pcb_meta(board_uuid))
    )

    ticket = 1

    def next_ticket() -> int:
        nonlocal ticket
        ticket += 1
        return ticket

    records.append(
        Record(
            {"type": "PRIMITIVE", "ticket": next_ticket(), "id": '["PRIMITIVE","ALL"]'},
            {"display": True, "pick": False},
        )
    )
    for name, display, pick in PRIMITIVE_DISPLAY[1:]:
        records.append(
            Record(
                {
                    "type": "PRIMITIVE",
                    "ticket": next_ticket(),
                    "id": f'["PRIMITIVE","{name}"]',
                },
                {"display": display, "pick": pick},
            )
        )
    records.append(
        Record({"type": "CANVAS", "ticket": next_ticket(), "id": "CANVAS"}, default_canvas())
    )

    for layer in DEFAULT_LAYERS:
        records.append(
            Record(
                {"type": "LAYER", "ticket": next_ticket(), "id": layer.record_id},
                layer.to_body(),
            )
        )

    # Layer physics: copper for the two outer layers, dielectric for the core.
    #
    # ``isKeepIsland`` means "keep copper that is not connected to any net".
    # With it false the editor *deletes* isolated copper when it loads or
    # refills the board -- and every shape this tool writes is isolated:
    # artwork copper has no net to belong to. That silently emptied the copper
    # layers, and the board came up as nothing but its outline.
    #
    # The captured project sets it true on all nine of its LAYER_PHYS records.
    # All three entries here carry copper, so all three get true.
    records.append(
        Record(
            {"type": "LAYER_PHYS", "ticket": next_ticket(), "id": '["LAYER_PHYS",1]'},
            {
                "material": "COPPER",
                "thickness": 1,
                "permittivity": 0,
                "lossTangent": 0,
                "isKeepIsland": True,
                "zIndex": 1,
            },
        )
    )
    records.append(
        Record(
            {"type": "LAYER_PHYS", "ticket": next_ticket(), "id": '["LAYER_PHYS",3]'},
            {
                "material": None,
                "thickness": 0,
                "permittivity": None,
                "lossTangent": None,
                "isKeepIsland": True,
                "zIndex": 1,
            },
        )
    )
    records.append(
        Record(
            {"type": "LAYER_PHYS", "ticket": next_ticket(), "id": '["LAYER_PHYS",2]'},
            {
                "material": "COPPER",
                "thickness": 1,
                "permittivity": 0,
                "lossTangent": 0,
                "isKeepIsland": True,
                "zIndex": 1,
            },
        )
    )

    records.append(
        Record(
            {"type": "ACTIVE_LAYER", "ticket": next_ticket(), "id": "ACTIVE_LAYER"},
            {"layerId": 11},
        )
    )
    records.append(
        Record(
            {"type": "SILK_OPTS", "ticket": next_ticket(), "id": '["SILK_OPTS",3]'},
            {"defaultColor": "#000000", "baseColor": "#FFFFFF"},
        )
    )
    records.append(
        Record(
            {"type": "PREFERENCE", "ticket": next_ticket(), "id": "PREFERENCE"},
            default_preference(routing_corner),
        )
    )
    records.append(
        Record(
            {"type": "NET", "ticket": next_ticket(), "id": '["NET",""]'},
            {
                "netType": None,
                "specialColor": None,
                "retLine": True,
                "differentialName": None,
                "isPositiveNet": False,
                "equalLengthGroupName": None,
            },
        )
    )

    # --- CONFIG --------------------------------------------------------
    records.append(_doc_head("CONFIG", "CONFIG", client, user_uuid, editor_version))
    records.append(
        Record(
            {"type": "META", "ticket": 1, "id": "META"},
            {"defaultSheet": schematic_uuid},
        )
    )

    # --- trailing empty sections --------------------------------------
    records.append(_doc_head("BLOB", "BLOB", client, user_uuid, editor_version))

    uuids = {
        "board": board_uuid,
        "schematic": schematic_uuid,
        "page": page_uuid,
        "pcb": pcb_uuid,
        "client": client,
        "user": user_uuid,
    }
    return records, uuids


def build_minimal_project(
    title: str = "ArtBoard",
    editor_version: str = "",
    routing_corner: str = "L",
) -> tuple[EproDocument, dict[str, str]]:
    """A complete, geometry-free project ready for primitives to be appended."""
    records, uuids = build_pcb_document(
        project_title=title,
        editor_version=editor_version,
        routing_corner=routing_corner,
    )
    document = EproDocument(records)
    return document, uuids
