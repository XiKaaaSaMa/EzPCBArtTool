"""Export the design document to an EasyEDA Pro project.

The editing model (:mod:`pcbart.artboard`) and the writer
(:mod:`pcbart.generator`) grew up separately: the model knows what the user
drew, the generator knows how to encode a board, and neither knows about the
other. This module is the adapter between them -- the piece that was missing
until now, and the reason the editor could be drawn in but never saved.

**What is exported today is the board outline**, which is what M1's acceptance
criterion asks for. The artwork placements are deliberately *not* written yet.
Encoding a reference picture needs an ``IMAGE`` record, and no example of one
has been measured from a real project; any encoding written from the
specification alone would be a guess. This project has been burned by exactly
that: a ``PAD`` written from the specification's example was silently dropped by
the editor while the file still opened cleanly. Rather than repeat it, artwork
export belongs with M3, and the returned report says so outright so the UI can
be honest about it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

__all__ = [
    "CopperRecord",
    "export_design",
    "default_title",
    "build_copper",
    "picture_placements",
    "export_needs_color_choice",
]


class CopperRecord:
    """One ``(layer_id, rings, width)`` triple that also knows its group.

    ⚠️ **Iterable as exactly three values on purpose.** The triple is the shape
    every consumer already unpacks -- ``build_copper``'s callers, the Gerber
    writer, a dozen tests -- and widening the tuple to carry the group would
    have broken all of them for a field almost none of them care about. The
    group rides alongside as an attribute, so existing unpacking keeps working
    and only the ``.epro2`` writer, which is the one place a ``GROUP`` record
    is emitted, has to read it.
    """

    __slots__ = ("layer_id", "rings", "width", "group")

    def __init__(self, layer_id: int, rings: list, width: float, group: str = ""):
        self.layer_id = layer_id
        self.rings = rings
        self.width = width
        self.group = group

    def __iter__(self):
        # Deliberately a fresh iterator over a tuple rather than ``iter((...))``
        # of the live fields, so unpacking cannot be affected by a later edit.
        return iter((self.layer_id, self.rings, self.width))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        group = f", group={self.group!r}" if self.group else ""
        return (
            f"CopperRecord(layer_id={self.layer_id}, rings={len(self.rings)}, "
            f"width={self.width}{group})"
        )


DEFAULT_LINE_WIDTH_MM = 0.125
"""Board-outline stroke width. Measured at 4.9213 file units in a real project."""


def default_title(board) -> str:
    """A project title for the export dialog to pre-fill.

    The artwork's filename if there is one, because that is what the user
    recognises the design by. Falls back to a neutral name rather than an empty
    field, since the title is written into the file's metadata.
    """
    image = board.image
    if image is not None and image.path:
        stem = Path(image.path).stem.strip()
        if stem:
            return stem
    return "ArtBoard"


def picture_placements(board) -> list:
    """The placements that would leave as pictures rather than as traced ink.

    A picture is not a ``FILL``: it exists in the ``.epro2`` as an ``OBJ`` and
    only reaches a board through the EDA's **colour-silkscreen** export. So the
    list doubles as "what would disappear from a plain Gerber order".
    """
    from .color_image import is_color_artwork

    return [
        item
        for item in getattr(board, "placements", [])
        if is_color_artwork(item)
    ]


def export_needs_color_choice(board) -> bool:
    """Whether the user has to say which kind of order this board is for.

    True when anything on the board would be placed as a picture. Both export
    actions then have to ask, because the two answers produce different boards:

    * **普通打板** traces every picture into filled regions, so the artwork
      prints in the layers' own colours and the Gerber set carries it;
      ``Fabrication_Colorful*`` is not produced at all.
    * **彩色丝印打板** keeps the pictures and the EDA writes the colour job.

    ⚠️ **Measured, not assumed.** On a card carrying two pictures, a plain
    Gerber export left the top and bottom silkscreen files at 810 and 813 bytes
    with **zero** draw commands -- the artwork was simply absent. A board with
    no pictures needs no question, so that case stays a single click.
    """
    return bool(picture_placements(board))


def export_design(
    board,
    output_path: str | Path,
    *,
    title: str = "",
    line_width_mm: float = DEFAULT_LINE_WIDTH_MM,
    routing_corner: str = "L",
    copper_params=None,
    color_print: bool = True,
) -> dict[str, object]:
    """Write ``board`` to ``output_path`` and describe what was written.

    Raises :class:`ValueError` with a message meant for the user when there is
    nothing valid to export. Every refusal here is a state the editor would
    otherwise turn into a file that opens but is wrong, which is the failure
    mode this project keeps running into.

    The returned report is what the caller shows: sizes in millimetres, the
    vertex count, and an explicit note that artwork is not included yet.
    """
    from .generator import describe_outline, generate_outline_board

    if not board.outline_ready:
        raise ValueError("尚未设置板框：请先打开「板框设置」确定板框形状")

    # Validate the spec as well as the geometry. A circle whose width and
    # height disagree draws a circle but describes something else, and that
    # mismatch should stop the export rather than reach the file.
    problems = board.outline.validate()
    if problems:
        raise ValueError("板框设置有问题：" + "；".join(problems))

    points = board.outline_polygon()
    if len(points) < 3:
        raise ValueError(
            "板框几何为空，无法导出。"
            "异形板需要在「板框设置」里选好基础形状并导入图片。"
        )

    # A tab that failed to attach is a feature the user switched on and would
    # not get. Writing the board anyway would produce a file that opens cleanly
    # and is missing the piece the base is supposed to hold -- so refuse, and
    # say which setting to change.
    if board.outline.standee and board.standee_error:
        raise ValueError(
            "立牌插舌没有连到板框上："
            + str(board.standee_error)
            + "。请在「板框设置 → 立牌设置」里调整插舌位置或尺寸。"
        )

    name = title.strip() or default_title(board)

    # A user who types a folder into the save dialog expects it to be made.
    # The writer opens the zip directly and does not create parents, so a name
    # like "工程\甲.epro2" would otherwise fail with a bare FileNotFoundError.
    destination = Path(output_path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    # ``pictures=color_print``: this writer goes to the EDA, which can carry an
    # embedded image. Artwork that still has its pixels -- an unprocessed
    # import, and the card's own 二维码 / 条形码 / 职业图标 / 阵营 logo -- is
    # written as a picture instead of being traced into fill regions.
    #
    # ⚠️ **``color_print=False`` is the plain-board path, and it must trace.**
    # The two orders are different products, not two labels for one file: a
    # picture reaches a board only through the colour-silkscreen export, so a
    # board whose artwork stayed as pictures and was then ordered as a normal
    # PCB would come back with those elements missing entirely.
    copper, copper_report = build_copper(
        board,
        params=copper_params,
        include_text=False,
        pictures=color_print,
    )

    # Template content travels with the board: the wording as the editor's own
    # STRING records so it stays editable there, the fixed decoration as the
    # same FILL/POLY primitives the source project uses, and the openings as
    # MULTI-layer fills -- the same encoding the source templates use for the
    # lanyard slot. Leaving any of them out produces a file that opens cleanly
    # and is quietly missing part of what the canvas showed.
    #
    # Text is deliberately *not* part of ``build_copper`` here. The writer
    # already emits a native STRING record for every line, and tracing the same
    # wording into filled outlines gave the editor two copies: one editable
    # text and one identical but slightly offset region. Gerber still asks for
    # the traced text, because a Gerber file has no STRING primitive.
    texts = [
        replace(
            item,
            text=item.resolved_text(
                name=board.operator_name, number=board.operator_number
            ),
        )
        for item in board.texts
    ]
    from . import badge_art

    decorations = badge_art.ordered_decorations(
        badge_art.decorations_for(board)
    )
    raw_decorations = badge_art.ordered_records(
        badge_art.records_for(board)
    )
    raw_outline = badge_art.outline_records_for(board)
    # Colour pictures travel as ``OBJ`` records -- the only shape that carries
    # pixels -- so the EDA's colour-silkscreen export has something to print.
    # ⚠️ Gated on ``color_print``: for a plain order the pictures were already
    # traced by ``build_copper`` above, and writing them as objects as well
    # would leave the board carrying both a picture *and* its silhouette.
    from .color_image import color_objects

    color_pictures = color_objects(board) if color_print else []
    path = generate_outline_board(
        outer_mm=points,
        outer_path=board.outline_path(),
        raw_outline=raw_outline,
        output_path=output_path,
        title=name,
        line_width_mm=line_width_mm,
        routing_corner=routing_corner,
        fills=copper,
        cutouts=board.cutout_rings(),
        texts=texts,
        raw_decorations=raw_decorations,
        color_objects=color_pictures,
        # Only the groups whose members survived to the geometry: a group whose
        # artwork was never traced has nothing in the file to name, and writing
        # it would leave the editor with an element that selects nothing.
        groups=_exportable_groups(board, copper),
    )

    summary = describe_outline(points)
    note = copper_report["note"]
    if raw_decorations:
        note += f" 已写入模板固定装饰 {len(raw_decorations)} 项。"
    if color_pictures:
        note += f" 已写入彩色图 {len(color_pictures)} 张（保留配色）。"
    # ⚠️ The manufacturability warnings go into the note the user actually
    # reads. They were computed and dropped before this, so a board with a
    # 0.15 mm neck exported silently and snapped in the post -- see
    # :func:`pcbart.validation.warning_sentence`.
    from .validation import check_outline, warning_sentence

    report = check_outline(points)
    warning = warning_sentence(report)
    if warning:
        note += " " + warning
    return {
        "output": str(path),
        "title": name,
        "width_mm": float(summary["width_mm"]),
        "height_mm": float(summary["height_mm"]),
        "vertices": int(summary["vertices"]),
        "perimeter_mm": float(summary["perimeter_mm"]),
        "warnings": [issue.message for issue in report.warnings],
        "artwork_placements": len(board.placements),
        "artwork_exported": bool(copper),
        "copper_shapes": copper_report["shapes"],
        "copper_vertices": copper_report["vertices"],
        "copper_layers": copper_report["layers"],
        "note": note,
        "texts": sum(1 for item in texts if item.text.strip()),
        "cutouts": len(board.cutouts),
        "decorations": len(raw_decorations),
        "color_pictures": len(color_pictures),
    }


#: Width written into every FILL record, copied from the captured project.
FILL_WIDTH_RAW = 0.2


#: Layer kinds this writer emits as filled geometry. Solder mask is
#: deliberately absent here: in an ``.epro2`` a mask opening is the *absence*
#: of mask, so it has no FILL record to write. That is a limit of this format,
#: not of the design -- the Gerber writer includes the mask layers, because a
#: Gerber solder-mask file is *defined* as the set of openings.
WRITABLE_KINDS = ("copper", "silk")


#: Placements traced with **VTracer spline curves, converted to arcs**, instead
#: of the built-in ``findContours`` + Douglas-Peucker chain.
#:
#: ⚠️ **Why the 阵营 logo, and only the 阵营 logo.** The user asked for it by
#: name: the crest is line art whose edges are genuinely curved, and
#: chord-only tracing showed the pixel grid as facets along every one of them.
#: Everything else keeps the built-in tracer for now -- the codes are generated
#: bitmaps where pixel accuracy *is* the specification, and the imported
#: illustrations are the user's own artwork that has not been signed off on a
#: different tracer yet. Widening this set is a visual decision, not a
#: technical one, so it stays a named list rather than a global switch.
ARC_TRACED_KINDS = ("faction",)

#: Magnification handed to VTracer. Its speckle and length thresholds are fixed
#: pixel counts, so a small source pays for them twice; see
#: :func:`pcbart.vtrace.trace_mask`. 3x **with smoothed** resampling is the
#: measured optimum -- a nearest-neighbour magnification gives every source
#: pixel a right angle and the output comes back all straight lines.
ARC_TRACE_UPSCALE = 3


def _mask_and_dims(path: str) -> tuple[np.ndarray, int, int]:
    """Read a placement's picture as ``(ink_plane, width_px, height_px)``.

    ``ink_plane`` follows the convention the rest of the pipeline uses and that
    :class:`~pcbart.vectorize.VectorizeParams` defaults to: **0 where the layer
    gets material, 255 where it does not.**

    Read with ``IMREAD_UNCHANGED`` rather than ``IMREAD_GRAYSCALE`` for one
    reason: **a transparent pixel is not ink.** Flattened to grey, an RGBA
    file's transparent background becomes black, and black means "material
    here" -- so a transparent PNG exports as a filled rectangle. That is not a
    small error: the badge's QR symbol has a transparent background, and writing
    it as a solid block would put a 10 mm² slab of silkscreen over the card's own
    artwork and leave a scanner nothing to read.

    Where the file carries an alpha channel the alpha *is* the shape and the
    colour is ignored: opaque becomes ink (0), transparent becomes bare board
    (255). Where it does not, the rule is dark means ink.
    """
    import numpy as np

    from .image_prep import imread_unicode

    raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"无法读取 {path}")
    height_px, width_px = raw.shape[:2]
    if raw.ndim == 2:
        return raw, width_px, height_px
    if raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        return np.where(alpha > 0, 0, 255).astype(np.uint8), width_px, height_px
    return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), width_px, height_px


def build_copper(
    board,
    params=None,
    kinds: tuple[str, ...] = WRITABLE_KINDS,
    *,
    include_text: bool = True,
    pictures: bool = False,
) -> tuple[list[tuple[int, list, float]], dict]:
    """Filled geometry for every artwork placement, ready for the writer.

    Copper and silkscreen are both written, because both are places where a
    filled shape means "material here": copper is metal, silkscreen is ink.
    Solder mask is not, in an ``.epro2`` -- an opening in it is an absence, not
    a shape -- so by default those placements are counted and reported but not
    emitted. ``kinds`` widens that for callers whose format *can* express an
    opening as a shape; the Gerber writer passes ``("copper", "silk",
    "mask")`` and gets the same traversal with mask included.

    Silkscreen matters for a board painting rather than being a nicety. The
    white tone lives there, so writing copper alone would export the dark field
    and silently drop the contour and the light areas, leaving a board that
    looks nothing like the preview.

    ``pictures`` asks for placements to be left as **pictures** rather than
    traced, and is reported back in ``report["color_artwork"]`` for the writer
    to turn into ``OBJ``/``BLOB`` records. ⚠️ **It is off by default, and it
    must stay off for Gerber**: a Gerber file has no picture primitive at all,
    so asking for one there would drop the artwork instead of placing it. Only
    the ``.epro2`` writer, where the EDA can carry an embedded image, passes
    ``True``.

    Placements in :data:`ARC_TRACED_KINDS` come back as **arc rings** rather
    than vertex lists; both kinds travel in the same ``(layer_id, rings, width)``
    triples, and both writers dispatch on the ring type. See that constant.

    Returns ``(records, report)`` where each record is
    ``(layer_id, complex_polygon_mm, width)``.
    """
    from .artboard import drawn_size_mm
    from .design_layers import layer
    from .image_prep import imread_unicode
    from .text_render import render_mask
    from .vectorize import VectorizeParams, vectorize_mask
    params = params or VectorizeParams()
    records: list[CopperRecord] = []
    shapes = 0
    vertices = 0
    layers_used: list[str] = []
    skipped: list[str] = []
    problems: list[str] = []
    colour_artwork: list = []
    """Colour pictures, left for the ``OBJ`` writer. Reported so the caller can
    tell "nothing to export" from "exported as a picture instead"."""

    # Board paintings first, as matched pairs. They are the one case where two
    # placements are two halves of one picture, and the only case where tracing
    # them separately would be wrong.
    groups, paired_ids = _board_art_groups(board)
    for white, black in groups:
        # A board painting is one element to the user, so both its tones carry
        # the same group id into the file and the editor shows one entry.
        group = board.group_of(white)
        group_id = group.id if group is not None else ""
        for layer_id, rings in _build_board_art_pair(
            white, black, params, problems
        ):
            records.append(
                CopperRecord(layer_id, rings, FILL_WIDTH_RAW, group=group_id)
            )
            shapes += 1
            vertices += sum(len(ring) for ring in rings)
            name = _layer_name(layer_id)
            if name not in layers_used:
                layers_used.append(name)

    # ⚠️ When pictures are wanted, they are **not** traced here. They leave as
    # ``OBJ`` records carrying their own pixels (see :mod:`pcbart.color_image`),
    # so tracing them would both throw the colours away and print the artwork
    # twice: once as a picture and once as a flat silhouette underneath it.
    # Measured before this: a card exported as 彩色丝印 reached the fab as
    # single-colour silkscreen with no colour data at all, and the generated
    # elements arrived as "填充区域" instead of the library's own pictures.
    from .color_image import is_color_artwork

    writable = tuple(kinds)
    for placement in board.placements:
        if id(placement) in paired_ids:
            continue
        if pictures and is_color_artwork(placement):
            colour_artwork.append(placement)
            continue
        design_layer = layer(placement.layer_key)
        if design_layer.kind not in writable:
            skipped.append(design_layer.name)
            continue

        path = placement.processed_path or placement.path
        if not path:
            problems.append(f"{design_layer.name}: 没有图源")
            continue

        try:
            ink, width_px, height_px = _mask_and_dims(path)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            problems.append(f"{design_layer.name}: 无法读取 {path}（{exc}）")
            continue
        # One mask pixel represents this much board, so the geometry lands at
        # the size the user placed on the canvas rather than at some default.
        #
        # ``px_per_mm`` is taken from the *drawn* size, not the placement box:
        # with the aspect ratio unlocked the box and the picture disagree, and
        # measuring against the box would stretch the artwork to fill it. The
        # canvas letterboxes instead (see ``drawn_size_mm``), and the export has
        # to match what the user saw.
        if width_px <= 0 or height_px <= 0 or placement.width_mm <= 0:
            problems.append(f"{design_layer.name}: 尺寸无效")
            continue

        # Only the framed part of the picture is exported. The window is read
        # from the placement so the canvas and both writers crop identically;
        # cropping here rather than in each writer is what keeps the .epro2 and
        # the Gerber from disagreeing about what the user framed.
        left, top, right, bottom = placement.source_rect(width_px, height_px)
        used = ink[top:bottom, left:right]
        used_h, used_w = used.shape[:2]
        if used_w <= 0 or used_h <= 0:
            problems.append(f"{design_layer.name}: 取景区域为空")
            continue

        # ``px_per_mm`` comes from the *window* now. Measuring against the whole
        # picture would scale the artwork by however much was cropped away --
        # frame a quarter of the image and the export would come out four times
        # too small.
        drawn_w_mm, _drawn_h_mm = drawn_size_mm(
            placement.width_mm, placement.height_mm, used_w / used_h
        )
        px_per_mm = used_w / drawn_w_mm

        # The plane is already in the tracer's own convention (0 = ink), so the
        # parameters only need the per-placement feature sizes.
        from .badge_content import vectorize_params_for

        arcs = getattr(placement, "kind", "") in ARC_TRACED_KINDS
        if arcs:
            # One complex polygon per traced piece, curves intact all the way
            # to the file.
            polygons = arc_rings_for(used, px_per_mm)
        else:
            polygons = vectorize_mask(
                used,
                px_per_mm,
                vectorize_params_for(placement, params),
            )

        if not polygons:
            problems.append(f"{design_layer.name}: 线稿里没有可转成图形的色块")
            continue

        # ⚠️ **Decided once per placement, before the per-shape loop.** The
        # picture may be moved anywhere on the canvas -- the user asked for free
        # movement so a figure can clear the card's top cutout -- so "pushed
        # entirely out of the opening" is a state reachable by accident, and the
        # canvas marks such an object in its layer's dark colour. The export has
        # to say so too, or a visible warning becomes a silently missing
        # illustration.
        #
        # ⚠️ **Asking before the loop is what keeps the report readable.** This
        # used to be checked inside the loop, where *every* traced shape of the
        # picture produced the same sentence -- an illustration traced into ~140
        # shapes printed it ~140 times and buried everything else in the dialog.
        # Checking the picture's own box against the opening answers the same
        # question once, and skips the vectorising work for a picture that
        # cannot print anything anyway.
        outside_opening = False
        if placement.region:
            from .badge_card import BADGE_REGIONS

            region = next(
                (item for item in BADGE_REGIONS if item.key == placement.region),
                None,
            )
            if region is not None and not placement.overlaps_rect(region.bounds):
                problems.append(
                    f"{design_layer.name}：图片已移出立绘区，本层没有内容可印"
                )
                outside_opening = True
        if outside_opening:
            continue

        for rings in polygons:
            # 旋转 and 翻转 are stored on the placement and shown in the
            # property panel, so they have to reach the board -- otherwise the
            # file opens with the artwork in a pose the user never chose. The
            # transform comes from the placement itself so this writer, the
            # Gerber writer and the canvas cannot drift apart.
            #
            # ⚠️ Arcs need ``board_arc_ring``, not ``board_ring``: a mirror
            # reverses their sweeps, so transforming only the vertices flips
            # the shape's outline but leaves every curve bending the wrong way.
            if arcs:
                placed = [placement.board_arc_ring(ring) for ring in rings]
            else:
                placed = [placement.board_ring(ring) for ring in rings]

            # ⚠️ A framed illustration is clipped to its 通行证 region here, at
            # write time. The region is where the card's own opening is, and
            # the user's requirement was that the picture outside it is cut --
            # but the region was only a canvas guide, so resizing or moving the
            # framed picture afterwards (both normal edits) exported ink past
            # the opening. Measured: frame it, scale 1.4x, and the ink reached
            # 23.1 mm where the window ends at 21.3 mm.
            #
            # ⚠️ **One record per clipped piece, not one per traced shape.**
            # Clipping can split a shape in two, and each piece carries its own
            # holes. Writing them as a single ``FILL`` would make the second
            # piece a *hole* of the first and punch the artwork out -- see
            # :func:`_clip_to_region`.
            if placement.region:
                if arcs:
                    # Clipping is a boolean operation, which needs the curve
                    # as points. A faction crest is never framed into a card
                    # opening, so this only guards against a hand-built board;
                    # flattening here is honest and local.
                    pieces = _clip_arc_to_region(placed, placement.region)
                else:
                    pieces = _clip_to_region(placed, placement.region)
                if not pieces:
                    # A single shape can fall outside while the picture as a
                    # whole still crosses the opening. Nothing is printed for
                    # this piece, and the picture-wide case was already reported
                    # above -- so this stays quiet rather than repeating.
                    continue
            else:
                pieces = [placed]

            group = board.group_of(placement)
            for piece in pieces:
                records.append(
                    CopperRecord(
                        design_layer.easyeda_id,
                        piece,
                        FILL_WIDTH_RAW,
                        group=group.id if group is not None else "",
                    )
                )
                shapes += 1
                vertices += sum(
                    len(ring.points) if hasattr(ring, "points") else len(ring)
                    for ring in piece
                )

        if design_layer.name not in layers_used:
            layers_used.append(design_layer.name)

    # Template wording, on the same footing as an imported picture: rendered to
    # a mask and traced. Done here rather than in the writers so the `.epro2`
    # and the Gerber cannot disagree about whether the card has lettering --
    # and so a board whose text is missing from the fab files is caught by the
    # same report that counts the artwork.
    #
    # ``include_text=False`` is for the ``.epro2`` writer, which emits native
    # ``STRING`` records instead. Keeping both produced a text and a filled
    # copy at slightly different positions, which the user saw as a ghost.
    if include_text:
        for item in board.texts:
            design_layer = layer(item.layer_key)
            if design_layer.kind not in writable:
                skipped.append(design_layer.name)
                continue
            rendered = render_mask(
                item, name=board.operator_name, number=board.operator_number
            )
            if rendered is None:
                problems.append(f"{design_layer.name}: 文字「{item.text}」没有可导出的字形")
                continue
            polygons = vectorize_mask(rendered.mask, rendered.px_per_mm, params)
            if not polygons:
                problems.append(f"{design_layer.name}: 文字「{item.text}」描不出图形")
                continue
            for rings in polygons:
                placed = [_place_text_ring(ring, item, rendered) for ring in rings]
                records.append((design_layer.easyeda_id, placed, FILL_WIDTH_RAW))
                shapes += 1
                vertices += sum(len(ring) for ring in placed)
            if design_layer.name not in layers_used:
                layers_used.append(design_layer.name)

    if layers_used:
        note = (
            f"已写入 {len(layers_used)} 层（{'、'.join(layers_used)}），"
            f"共 {shapes} 个图形。"
        )
    elif colour_artwork:
        # Every picture on the board is a colour one, so nothing was traced --
        # but the board is far from empty. Saying "no content to write" here
        # would be a lie the user cannot see through.
        note = "图画按原图写入（未描边）：这些是彩色图，保留配色交给 EDA 出彩印。"
    else:
        note = "本次只写入板框，没有可写入的图层内容（图片需在铜层或丝印层上）。"
    if skipped:
        note += f" 阻焊层未写入：{'、'.join(sorted(set(skipped)))}（开窗是挖空，不是图形）。"
    if colour_artwork:
        note += (
            f" 其中 {len(colour_artwork)} 张彩色图按原图写入（保留配色），"
            "由 EDA 的彩色丝印导出出彩印。"
        )
    if problems:
        note += " 跳过：" + "；".join(problems)

    return records, {
        "shapes": shapes,
        "vertices": vertices,
        "layers": layers_used,
        "note": note,
        "color_artwork": colour_artwork,
    }


def _clip_to_region(
    rings: list[list[tuple[float, float]]], region_key: str
) -> list[list[list[tuple[float, float]]]]:
    """A traced shape clipped to the 通行证 opening it was framed into.

    ``rings`` is one complex polygon -- first ring the outer contour, the rest
    its holes, in board millimetres. The result is a **list of complex
    polygons**, each its own outer ring followed by its own holes, because
    clipping can split one shape into several disconnected pieces.

    ⚠️ **Two mistakes used to live here, and together they are why the board
    came out covered by a solid block** (「整个人物都被填充区域覆盖」).

    * **The holes were dropped.** ``shell.difference(hole)`` on a ring returns a
      polygon *with an interior ring* -- not a ``MultiPolygon`` -- so the loop
      below took that single polygon and then read only ``part.exterior``. A
      stroked outline became a solid disc. Measured on a donut mask: 103.7 mm²
      of ink came out as 149.8 mm² of solid. Line work is mostly thin, so this
      hit most of the drawing.
    * **Separate pieces were merged into one shape.** Returning a flat ring list
      let the caller store several pieces as a single ``FILL``, i.e. "the first
      ring is the outer and all the rest are holes". Two blobs therefore punched
      each other out, and one piece could carry more hole area than it had
      shape. On the reported card the largest record had an outer ring of
      463 mm² and 625 mm² of holes.

    Returning complex polygons fixes both: the holes stay attached to the piece
    they belong to, and the caller writes one record per piece.
    """
    from shapely.geometry import Polygon

    from .badge_card import BADGE_REGIONS
    from .geometry import normalize_complex_polygon

    region = next((item for item in BADGE_REGIONS if item.key == region_key), None)
    if region is None:
        # Nothing to clip against: hand the shape back as it came in.
        return [rings]

    x0, y0, x1, y1 = region.bounds
    window = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if len(rings) == 1:
        outer = Polygon(rings[0])
    else:
        outer = Polygon(rings[0], rings[1:])
    if not outer.is_valid:
        outer = outer.buffer(0)
    if outer.is_empty:
        return [rings]

    clipped = outer.intersection(window)
    if clipped.is_empty:
        return []

    pieces = (
        list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
    )
    out: list[list[list[tuple[float, float]]]] = []
    for piece in pieces:
        if piece.geom_type != "Polygon" or piece.is_empty:
            continue
        shell = [(float(x), float(y)) for x, y in piece.exterior.coords]
        if len(shell) < 4:
            continue
        holes = [
            [(float(x), float(y)) for x, y in hole.coords]
            for hole in piece.interiors
        ]
        # The editor reads a "hole" only when it is wound against the outer
        # ring, so the winding is normalised here rather than left to whatever
        # order the boolean operation happened to produce.
        out.append(normalize_complex_polygon(shell, [h for h in holes if len(h) >= 4]))
    return out


def _clip_arc_to_region(rings, region_key: str):
    """An **arc** complex polygon clipped to a region, as plain rings.

    Used only if a curve-traced placement is also framed into a card opening,
    which the faction crest never is -- it sits in its own copper block. The
    boolean intersection needs the curve as points, so this flattens first and
    hands the result to :func:`_clip_to_region`; the alternative would be an
    arc-clipping implementation whose only caller does not exist.
    """
    from .geometry import sample_arc

    flattened: list[list[tuple[float, float]]] = []
    for ring in rings:
        points: list[tuple[float, float]] = []
        count = len(ring.points)
        for index in range(count):
            start = ring.points[index - 1]
            end = ring.points[index]
            sweep = float(ring.sweeps[index])
            if abs(sweep) < 1e-6:
                segment = [start, end]
            else:
                segment = sample_arc(start, end, sweep)
            points.extend(segment if not points else segment[1:])
        if len(points) >= 3:
            flattened.append(points)
    return _clip_to_region(flattened, region_key)


def arc_rings_for(mask, px_per_mm: float):
    """A mask traced with VTracer's spline fit, as arc complex polygons.

    Returns one entry per traced piece, each a ``[outer, hole, ...]`` list of
    :class:`pcbart.geometry.ArcRing`, wound the way the editor needs. The
    caller writes each entry as its own ``FILL``.

    ⚠️ **Every piece has to come back, not just the biggest.** VTracer emits
    one ``<path>`` per piece of artwork, and a crest is many pieces: measured
    on 企鹅物流 the trace returns 26 paths. An earlier version forwarded only
    ``rings[0]``, which exported that card with a 2.76 x 3.68 mm fragment where
    its 24.32 mm crest should be -- the rest of the logo was dropped without a
    word, and the file still opened cleanly.

    An empty list means there was nothing to trace or the tracer is not
    installed; the caller reports that rather than raising from the writer.
    """
    from .vtrace import VTracerError, trace_mask

    try:
        pieces = trace_mask(
            mask,
            px_per_mm,
            mode="spline",
            upscale=ARC_TRACE_UPSCALE,
        )
    except VTracerError:
        return []

    from .geometry import normalize_arc_polygon

    out = []
    for piece in pieces:
        outer, *holes = piece
        out.append(normalize_arc_polygon(outer, holes=holes))
    return out


def _place_text_ring(ring, item, rendered) -> list[tuple[float, float]]:
    """One traced ring from text-mask pixels to board millimetres.

    The ring arrives centred on the origin (``vectorize_mask`` centres its
    output), so it is moved to where the line's ink actually sits and then
    turned about the anchor -- the same rotate-then-translate order the
    placement transform uses, so a 270-degree strip lands where the canvas drew
    it rather than mirrored or offset by its own length.
    """
    import math

    dx, dy = rendered.centre_mm
    angle = math.radians(item.rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    placed: list[tuple[float, float]] = []
    for x, y in ring:
        # Into the line's own frame, with the anchor at the origin.
        lx = x + dx
        ly = y + dy
        placed.append(
            (
                item.x_mm + lx * cos_a - ly * sin_a,
                item.y_mm + lx * sin_a + ly * cos_a,
            )
        )
    return placed


def _layer_name(layer_id: int) -> str:
    """A layer's display name from its EasyEDA id, for the export report."""
    from .design_layers import LAYERS

    for design_layer in LAYERS:
        if design_layer.easyeda_id == layer_id:
            return design_layer.name
    return f"layer {layer_id}"


def _exportable_groups(board, records) -> list:
    """The groups that actually have geometry in ``records``.

    A group is only meaningful in the written file if at least two of its
    members produced a ``FILL``. Two things make that not automatic: a member
    can be dropped as a picture when 彩色丝印 writes it as an ``OBJ`` instead,
    and a member on a layer this writer does not emit -- a solder-mask
    placement, say -- produces no geometry at all. Writing a ``GROUP`` record
    for either case would leave the editor with an element row that selects
    nothing, which is worse than no group.

    Membership is checked through the records' own group ids rather than
    against ``board.groups`` again: the ids in the records are the ones that
    reached the file, so this cannot disagree with what was written.
    """
    counts: dict[str, int] = {}
    for record in records:
        group_id = str(getattr(record, "group", "") or "")
        if group_id:
            counts[group_id] = counts.get(group_id, 0) + 1
    return [
        group
        for group in getattr(board, "groups", []) or []
        if counts.get(str(group.id), 0) >= 2
    ]


def _board_art_groups(board):
    """Placements grouped into board paintings, plus the ids already claimed.

    Two placements form a painting when they carry opposite tones and came from
    the same import. Grouping by import rather than by file is what keeps a
    picture imported twice from being welded into one board: each import is its
    own painting, exactly as it is on the canvas.
    """
    by_import: dict[str, dict[str, object]] = {}
    for placement in board.placements:
        if placement.tone not in ("white", "black"):
            continue
        group = by_import.setdefault(placement.import_id, {})
        group.setdefault(placement.tone, placement)

    groups = []
    claimed: set[int] = set()
    for group in by_import.values():
        white = group.get("white")
        black = group.get("black")
        if white is None or black is None:
            continue
        groups.append((white, black))
        claimed.add(id(white))
        claimed.add(id(black))
    return groups, claimed


def _build_board_art_pair(white, black, params, problems):
    """One painting as ``(layer_id, rings)`` records, tiled exactly.

    Delegates to :func:`pcbart.boardart.board_art_records`, which is the single
    implementation of the pairing: only the white tone is traced and the black
    is taken as whatever it does not cover. Tracing both would send each through
    its own simplification, so they would stop sharing an edge and leave a
    hairline of bare board wherever they met -- measured at 88 mm² uncovered on
    a real painting before this was routed through one implementation.

    The two tones arrive here as *files* rather than as masks, so they are read
    back and the same function the canvas preview used is called again. That is
    deliberate: the exported board has to be the previewed one, and the surest
    way to guarantee it is to run the same code.
    """
    import numpy as np

    from .boardart import BoardArt, board_art_records

    white_mask = _load_tone(white, problems)
    black_mask = _load_tone(black, problems)
    if white_mask is None or black_mask is None:
        return []

    if white_mask.shape != black_mask.shape:
        problems.append(
            f"{white.layer}: 白黑两层尺寸不一致，无法拼合"
            f"（{white_mask.shape} vs {black_mask.shape}）"
        )
        return []

    if white.width_mm <= 0:
        problems.append(f"{white.layer}: 尺寸无效")
        return []

    side = _painting_side(white, black)
    art = BoardArt(
        white=white_mask,
        black=black_mask,
        # The subject only feeds the bare-background branch, which a paired
        # painting never takes: its tones already cover the whole frame.
        subject=np.ones_like(white_mask),
        black_background=True,
    )
    return [
        (layer_id, [white.board_ring(ring) for ring in rings])
        for layer_id, rings, _width in board_art_records(
            art, white.width_mm, side=side, vectorize_params=params
        )
    ]


def _painting_side(white, black) -> str:
    """Which faces a painting's two tones occupy, as a 板画 side key."""
    keys = {white.layer_key, black.layer_key}
    has_top = any(key.startswith("top_") for key in keys)
    has_bottom = any(key.startswith("bottom_") for key in keys)
    if has_top and has_bottom:
        return "both"
    return "bottom" if has_bottom else "top"


def _load_tone(placement, problems):
    """One published tone as a boolean mask, or ``None`` if it cannot be read.

    The files are written in the canvas's ink convention -- 0 where the tone is
    set -- so the comparison is against dark, not light. A file with an alpha
    channel is read through its alpha for the same reason as everywhere else:
    a transparent pixel is not ink.
    """
    path = placement.processed_path or placement.path
    if not path:
        problems.append(f"{placement.layer}: 没有图源")
        return None
    try:
        ink, width_px, height_px = _mask_and_dims(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        problems.append(f"{placement.layer}: 无法读取 {path}（{exc}）")
        return None
    if ink.size == 0:
        problems.append(f"{placement.layer}: 图片为空")
        return None
    # A board painting is two halves of one picture, and the framing applies to
    # the picture, so the window is taken here too -- otherwise the white and
    # black tones would be cropped at different offsets and the exported board
    # would tear along their seam.
    left, top, right, bottom = placement.source_rect(width_px, height_px)
    return ink[top:bottom, left:right]
