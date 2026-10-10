"""The 通行证 template's fixed artwork, extracted from the user's file.

The card is more than an outline and a handful of text lines.  Its borders,
the vertical RHODES ISLAND strip, the Arknights logo, hatching and the solder
mask windows all live in the source project as ordinary primitives.  This
module exposes the compact copy produced by :mod:`tools.extract_badge_art` so
the canvas, the ``.epro2`` writer and the Gerber writer all draw the same
thing.

The data is static template content, not a second document model.  It is keyed
off the board's template name, so applying or removing the template switches it
with no draft format change and no chance of a half-copied decoration list.
"""

from __future__ import annotations

from functools import lru_cache

from .badge_decorations import PAYLOAD
from .design_layers import LAYER_PAINT_ORDER

__all__ = [
    "decoration_geometry",
    "decorations",
    "decorations_for",
    "ordered_decorations",
    "visible_decorations",
    "marker_labels",
    "is_rhodes_emblem",
    "emblem_enabled",
    "records",
    "records_for",
    "ordered_records",
    "visible_records",
    "outline_records",
    "outline_records_for",
    "layer_key_for_id",
    "is_badge",
]

#: Bottom-to-top paint order for the card's two sides.
#:
#: ⚠️ **Read from :mod:`pcbart.design_layers`, not written out again here.** The
#: stack -- copper over silkscreen over mask, per side -- is what 嘉立创 builds
#: by default, and it has to be the same answer in the decorations, the
#: artwork, the wording and the layer panel. This module held its own copy once,
#: in the opposite order, which put the mask over the card's copper strip and
#: turned it purple on the canvas.
_LAYER_ORDER = LAYER_PAINT_ORDER

#: EasyEDA stores raw record geometry in mils; the decoration payload is already
#: converted to millimetres. Comparing the two means converting one.
_MM_TO_MIL = 1.0 / 0.0254
_MIL_TO_MM = 0.0254


#: The two source layouts the card ships with. The board's template name ends
#: with the operator's 精英化 stage, so ``通行证 · 精2`` selects the tall set.
_VARIANT_BY_STAGE: dict[str, str] = {
    "精0": "short",
    "精1": "short",
    "精2": "tall",
}


def variant_for(board) -> str:
    """Which layout ``board`` uses: ``"short"`` or ``"tall"``.

    Chosen by the board's own template name, so the canvas, both writers and
    the panel all read the same answer and a card keeps its layout through a
    save and reload. Anything unrecognised -- an older draft, a template
    applied before this existed -- falls back to the short card, which is what
    every card was built from until now.
    """
    from .badge_card import stage_of

    template = getattr(getattr(board, "outline", None), "template", "") or ""
    return _VARIANT_BY_STAGE.get(stage_of(template), "short")


def decorations(board=None) -> tuple[dict, ...]:
    """Every fixed decoration for ``board``, in source order.

    ``board`` may be omitted, which yields the short card -- the default and
    what every caller meant before the two layouts existed.
    """
    if board is None:
        return tuple(PAYLOAD.get("items") or ())
    variant = (PAYLOAD.get("variants") or {}).get(variant_for(board)) or {}
    return tuple(variant.get("items") or PAYLOAD.get("items") or ())


def ordered_decorations(items: tuple[dict, ...] | None = None) -> tuple[dict, ...]:
    """Decorations sorted bottom-to-top by their design layer.

    The source file's record order puts some copper fields after the
    silkscreen wording, which is fine in the editor because layers carry the
    stacking.  The canvas draws one flat list, so it has to impose that order
    itself; otherwise the red artwork field paints over the RHODES ISLAND
    lettering.
    """
    source = decorations() if items is None else items
    rank = {key: index for index, key in enumerate(_LAYER_ORDER)}
    return tuple(
        sorted(
            source,
            key=lambda item: rank.get(
                layer_key_for_id(int(item.get("layer", 0))), len(rank)
            ),
        )
    )


def _labels_by_size() -> tuple[tuple[float, float, str], ...]:
    """``(width, height, label)`` for every block the user drew as a marker.

    Built from the measured region table rather than a second copy of the
    numbers, so a corrected region cannot leave a stale marker behind. The tall
    illustration window is included because the card no longer places
    illustrations at all.
    """
    from .badge_content import marker_size_pairs

    return marker_size_pairs()


def marker_labels(board=None) -> dict[int, str]:
    """Which decoration records are the user's own positioning markers.

    Keyed by record index, because a decoration has no name of its own -- the
    only thing that identifies a marker is the rectangle it draws. Matching on
    the size of the measured region is what the extraction itself does, and a
    3 mm wide rectangle with these dimensions occurs nowhere else on the card.

    ⚠️ **Cached, and it has to stay cached.** This walks every ring of every
    decoration -- 329 items, tens of thousands of points -- and it is reached on
    *every repaint*, which is every frame of a drag. Measured at 4.4 ms per
    call, that was 5% of a 26 ms frame on its own. It is a pure function of the
    static payload, so one call is all it should ever need.

    The returned dict is shared, so callers must treat it as read-only.
    """
    # A board is optional so the helper stays usable on its own; with none it
    # answers for the short card, which is what it meant before the two
    # layouts existed.
    variant = variant_for(board) if board is not None else "short"
    return dict(_marker_labels_cached(variant))


@lru_cache(maxsize=4)
def _marker_labels_cached(variant: str) -> dict[int, str]:
    from .badge_content import ART_REGION_SIZE_TOLERANCE_MM

    lookup = _labels_by_size()
    found: dict[int, str] = {}
    for index, item in enumerate(_decorations_for_variant(variant)):
        if item.get("kind") != "fill":
            continue
        for ring in item.get("rings") or ():
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            for want_w, want_h, label in lookup:
                if (
                    abs(width - want_w) <= ART_REGION_SIZE_TOLERANCE_MM
                    and abs(height - want_h) <= ART_REGION_SIZE_TOLERANCE_MM
                ):
                    found[index] = label
                    break
            if index in found:
                break
    return found


#: Where the 罗德岛 emblem sits, as ``(x0, x1, y0, y1)`` in board millimetres.
#:
#: ⚠️ **This is not a placeholder.** It is the crest that goes on *every* card,
#: on the **top silkscreen**: a solid triangle with the castle and the
#: ``RHODES ISLAND`` lettering knocked out of it (202 rings, ~110 mm², 43% of
#: its own bounding triangle). The author drew it into the template because
#: every card carries it, not as an example of what the copper slot is for --
#: that slot is a separate element on the copper layer, right underneath.
#:
#: It overlaps the faction block's box almost exactly (24.3067 x 21.0260 against
#: 24.321 x 21.092, centres 0.067 mm apart), which is why the two are easy to
#: confuse. They are different layers and different elements: the crest is
#: silkscreen and always present, the faction logo is copper and depends on who
#: the card is for.
SAMPLE_EMBLEM_BOX: tuple[float, float, float, float] = (
    -1.0,
    26.0,
    -39.5,
    -16.5,
)


def is_rhodes_emblem(item: dict) -> bool:
    """Whether a decoration is the 罗德岛 crest that every card carries.

    Identified by sitting **inside the crest's box on the top silkscreen**: it
    is a dense cluster of silkscreen pieces that no other part of the card
    reaches into.
    """
    if layer_key_for_id(int(item.get("layer", 0))) != "top_silk":
        return False
    points = [
        point for ring in (item.get("rings") or ()) for point in ring
    ]
    if not points:
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    centre_x = (min(xs) + max(xs)) / 2.0
    centre_y = (min(ys) + max(ys)) / 2.0
    x0, x1, y0, y1 = SAMPLE_EMBLEM_BOX
    return x0 <= centre_x <= x1 and y0 <= centre_y <= y1


@lru_cache(maxsize=4)
def _visible_decorations_cached(variant: str, emblem: bool) -> tuple[dict, ...]:
    """Decorations with the positioning markers, and optionally the crest.

    The markers say where artwork *will* go, and the artwork is there now, so
    they always come off. The 罗德岛 crest **stays** unless the caller says
    otherwise: it belongs on every card, and ``emblem=False`` is how the panel
    removes it when the user asks for a plain card.

    Cached for the same reason as :func:`marker_labels`: the canvas asks for
    this every frame, and the answer only depends on the static payload and the
    one boolean. ``maxsize=2`` holds both spellings.
    """
    hidden = _marker_labels_cached(variant)
    return tuple(
        item
        for index, item in enumerate(_decorations_for_variant(variant))
        if index not in hidden and (emblem or not is_rhodes_emblem(item))
    )


def visible_decorations(*, emblem: bool = True, board=None) -> tuple[dict, ...]:
    """Decorations for ``board`` with the positioning markers removed.

    The markers say where artwork *will* go, and the artwork is there now, so
    they always come off. The 罗德岛 crest **stays** unless the caller says
    otherwise: it belongs on every card, and ``emblem=False`` is how the panel
    removes it when the user asks for a plain card.

    Cached per layout and per emblem flag; see :func:`marker_labels` for why.
    """
    return _visible_decorations_cached(variant_for(board), emblem)


def _decorations_for_variant(variant: str) -> tuple[dict, ...]:
    """Raw decoration list for a layout key, independent of a board."""
    data = (PAYLOAD.get("variants") or {}).get(variant) or {}
    return tuple(data.get("items") or PAYLOAD.get("items") or ())


def _is_emblem_record(item: dict) -> bool:
    """Whether a raw record is the 罗德岛 crest.

    Raw records carry ``IMAGE`` frames rather than decoded rings, in **mils**,
    with ``startX/startY`` at the frame's **top-left** corner -- so the centre
    is half a size below the start point.
    """
    if item.get("type") != "IMAGE":
        return False
    body = item.get("body") or {}
    if int(body.get("layerId", 0)) != 3:
        return False
    try:
        start_x = float(body.get("startX", 0.0)) * _MIL_TO_MM
        start_y = float(body.get("startY", 0.0)) * _MIL_TO_MM
        width = float(body.get("width", 0.0)) * _MIL_TO_MM
        height = float(body.get("height", 0.0)) * _MIL_TO_MM
    except (TypeError, ValueError):
        return False
    centre_x = start_x + width / 2.0
    centre_y = start_y - height / 2.0
    x0, x1, y0, y1 = SAMPLE_EMBLEM_BOX
    return x0 <= centre_x <= x1 and y0 <= centre_y <= y1


def visible_records(*, emblem: bool = True, board=None) -> tuple[dict, ...]:
    """The source primitives, minus the positioning markers, in source order.

    The writers take raw records rather than the compact draw list, so the
    filter has to exist on both: a marker hidden on the canvas but written into
    the board file would come back as a filled rectangle the user never drew.

    ``emblem=False`` also drops the 罗德岛 crest, matching
    :func:`visible_decorations` -- the two must agree or the canvas and the
    board file would disagree about whether the card carries it.

    Matching is by the record's own ``R`` path. Raw records carry geometry as
    ``["R", x, y, sideA, sideB, angle, ...]`` in **mils**, which is why the sizes
    are converted rather than compared against the millimetre table directly.
    The two sides are compared as an unordered pair: the source writes the short
    side first for the markers and the long side first for other rectangles, and
    which slot holds which is not worth depending on.
    """
    from .badge_content import ART_REGION_SIZE_TOLERANCE_MM

    wanted = [(w * _MM_TO_MIL, h * _MM_TO_MIL) for w, h, _ in _labels_by_size()]
    tolerance = ART_REGION_SIZE_TOLERANCE_MM * _MM_TO_MIL
    keep: list[dict] = []
    for item in records(board):
        if not emblem and _is_emblem_record(item):
            continue
        body = item.get("body") or {}
        path = body.get("path")
        head = path[0] if isinstance(path, (list, tuple)) and path else None
        if not (isinstance(head, (list, tuple)) and head[:1] == ["R"]):
            keep.append(item)
            continue
        try:
            side_a = abs(float(head[3]))
            side_b = abs(float(head[4]))
        except (IndexError, TypeError, ValueError):
            keep.append(item)
            continue
        marker = False
        for want_w, want_h in wanted:
            pair = sorted((side_a, side_b))
            if (
                abs(pair[0] - min(want_w, want_h)) <= tolerance
                and abs(pair[1] - max(want_w, want_h)) <= tolerance
            ):
                marker = True
                break
        if not marker:
            keep.append(item)
    return tuple(keep)


def is_badge(board) -> bool:
    """Whether ``board`` is the 通行证 card, in *either* layout."""
    from .badge_card import is_badge_template

    return (
        getattr(getattr(board, "outline", None), "shape", "") == "template"
        and is_badge_template(getattr(board.outline, "template", "") or "")
    )


def emblem_enabled(board) -> bool:
    """Whether the 罗德岛 crest is on this card.

    **On by default**, for every card, on the top silkscreen -- it is fixed
    card content like the border, not something derived from who the card is
    for. ``board.art_choices["emblem"] == "off"`` removes it at the user's
    request; any other value leaves it on, so a board that predates the field
    keeps the crest it has always had.
    """
    from .badge_content import EMBLEM_KIND, CHOICE_OFF

    choices = getattr(board, "art_choices", None) or {}
    return choices.get(EMBLEM_KIND, "") != CHOICE_OFF


def decorations_for(board) -> tuple[dict, ...]:
    """Decorations to draw or write for ``board``, empty for other boards.

    Positioning markers are filtered out, matching :func:`records_for`; the
    罗德岛 crest follows :func:`emblem_enabled`.
    """
    if not is_badge(board):
        return ()
    return visible_decorations(emblem=emblem_enabled(board), board=board)


def decoration_geometry(
    board,
) -> tuple[list[tuple[int, list[list[tuple[float, float]]]]],
           list[tuple[int, list[tuple[float, float]], float, bool]]]:
    """The card's fixed artwork as geometry a writer can emit directly.

    Returns ``(regions, strokes)``:

    * ``regions`` -- ``(layer_id, [ring, ...])``, the first ring the outer
      contour and the rest holes. Filled areas, ready for a Gerber region.
    * ``strokes`` -- ``(layer_id, points, width_mm, closed)``. The line work:
      the border, the hatching, the printable rules.

    This exists because the ``.epro2`` writer and the Gerber writer need the
    same artwork in *different* encodings -- the first writes the source
    project's own ``FILL``/``LINE`` records back verbatim, the second has to
    emit regions and apertures. Both start from :func:`decorations_for`, so
    neither can carry a decoration the other has dropped.

    ⚠️ **The two were not always both fed.** The Gerber path walked only the
    placements and the wording, so a card exported as a fab package came out
    with no border, no wordmark, no crest and **empty solder-mask layers** --
    every opening that makes the illustration and the codes readable was
    missing. The board still validated and still uploaded; only the finished
    board would have been wrong.
    """
    regions: list[tuple[int, list[list[tuple[float, float]]]]] = []
    strokes: list[tuple[int, list[tuple[float, float]], float, bool]] = []
    for item in decorations_for(board):
        layer_id = int(item.get("layer", 0))
        if item.get("kind") == "fill":
            rings = [
                [(float(x), float(y)) for x, y in ring]
                for ring in (item.get("rings") or [])
                if ring and len(ring) >= 3
            ]
            if rings:
                regions.append((layer_id, rings))
            continue
        width = float(item.get("width_mm", 0.254))
        closed = bool(item.get("closed"))
        for path in item.get("paths") or []:
            points = [(float(x), float(y)) for x, y in path]
            if len(points) >= 2:
                strokes.append((layer_id, points, width, closed))
    return regions, strokes


def records(board=None) -> tuple[dict, ...]:
    """The source project's own primitives for ``board``, at the card centre."""
    if board is None:
        return tuple(PAYLOAD.get("records") or ())
    variant = (PAYLOAD.get("variants") or {}).get(variant_for(board)) or {}
    return tuple(variant.get("records") or PAYLOAD.get("records") or ())


def records_for(board) -> tuple[dict, ...]:
    """Raw records for ``board``, empty for boards that are not the card.

    Positioning markers are filtered out: a block that only said "artwork goes
    here" must not be written into a finished board file. The 罗德岛 crest
    follows :func:`emblem_enabled`.
    """
    if not is_badge(board):
        return ()
    return visible_records(emblem=emblem_enabled(board), board=board)


def outline_records() -> tuple[dict, ...]:
    """The source project's own board-outline edge chains."""
    return tuple(PAYLOAD.get("outline_records") or ())


def outline_records_for(board) -> tuple[dict, ...]:
    """Outline records for ``board``, empty for non-template boards."""
    return outline_records() if is_badge(board) else ()


def ordered_records(items: tuple[dict, ...] | None = None) -> tuple[dict, ...]:
    """Raw records sorted in the same bottom-to-top layer order as the canvas.

    The default source is :func:`visible_records`, so the canvas and the writers
    never disagree about whether a positioning marker is present.
    """
    source = visible_records() if items is None else items
    rank = {key: index for index, key in enumerate(_LAYER_ORDER)}
    return tuple(
        sorted(
            source,
            key=lambda item: rank.get(
                layer_key_for_id(int((item.get("body") or {}).get("layerId", 0))),
                len(rank),
            ),
        )
    )


@lru_cache(maxsize=1)
def _id_to_key() -> dict[int, str]:
    from .design_layers import LAYERS

    return {entry.easyeda_id: entry.key for entry in LAYERS}


def layer_key_for_id(layer_id: int) -> str:
    """Design-layer key for an EasyEDA layer id, or an empty string."""
    return _id_to_key().get(int(layer_id), "")
