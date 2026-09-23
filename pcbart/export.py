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

from pathlib import Path

import cv2

__all__ = ["export_design", "default_title", "build_copper"]


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


def export_design(
    board,
    output_path: str | Path,
    *,
    title: str = "",
    line_width_mm: float = DEFAULT_LINE_WIDTH_MM,
    routing_corner: str = "L",
    copper_params=None,
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

    copper, copper_report = build_copper(board, params=copper_params)

    path = generate_outline_board(
        outer_mm=points,
        output_path=output_path,
        title=name,
        line_width_mm=line_width_mm,
        routing_corner=routing_corner,
        fills=copper,
    )

    summary = describe_outline(points)
    return {
        "output": str(path),
        "title": name,
        "width_mm": float(summary["width_mm"]),
        "height_mm": float(summary["height_mm"]),
        "vertices": int(summary["vertices"]),
        "perimeter_mm": float(summary["perimeter_mm"]),
        "artwork_placements": len(board.placements),
        "artwork_exported": bool(copper),
        "copper_shapes": copper_report["shapes"],
        "copper_vertices": copper_report["vertices"],
        "copper_layers": copper_report["layers"],
        "note": copper_report["note"],
    }


#: Width written into every FILL record, copied from the captured project.
FILL_WIDTH_RAW = 0.2


#: Layer kinds this writer emits as filled geometry. Solder mask is
#: deliberately absent here: in an ``.epro2`` a mask opening is the *absence*
#: of mask, so it has no FILL record to write. That is a limit of this format,
#: not of the design -- the Gerber writer includes the mask layers, because a
#: Gerber solder-mask file is *defined* as the set of openings.
WRITABLE_KINDS = ("copper", "silk")


def build_copper(
    board, params=None, kinds: tuple[str, ...] = WRITABLE_KINDS
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

    Returns ``(records, report)`` where each record is
    ``(layer_id, complex_polygon_mm, width)``.
    """
    from .artboard import drawn_size_mm
    from .design_layers import layer
    from .image_prep import imread_unicode
    from .vectorize import VectorizeParams, vectorize_mask

    params = params or VectorizeParams()
    records: list[tuple[int, list, float]] = []
    shapes = 0
    vertices = 0
    layers_used: list[str] = []
    skipped: list[str] = []
    problems: list[str] = []

    # Board paintings first, as matched pairs. They are the one case where two
    # placements are two halves of one picture, and the only case where tracing
    # them separately would be wrong.
    groups, paired_ids = _board_art_groups(board)
    for white, black in groups:
        for layer_id, rings in _build_board_art_pair(
            white, black, params, problems
        ):
            records.append((layer_id, rings, FILL_WIDTH_RAW))
            shapes += 1
            vertices += sum(len(ring) for ring in rings)
            name = _layer_name(layer_id)
            if name not in layers_used:
                layers_used.append(name)

    writable = tuple(kinds)
    for placement in board.placements:
        if id(placement) in paired_ids:
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
            raw = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
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
        height_px, width_px = raw.shape[:2]
        if width_px <= 0 or height_px <= 0 or placement.width_mm <= 0:
            problems.append(f"{design_layer.name}: 尺寸无效")
            continue
        drawn_w_mm, _drawn_h_mm = drawn_size_mm(
            placement.width_mm, placement.height_mm, width_px / height_px
        )
        px_per_mm = width_px / drawn_w_mm

        polygons = vectorize_mask(raw, px_per_mm, params)
        if not polygons:
            problems.append(f"{design_layer.name}: 线稿里没有可转成图形的色块")
            continue

        for rings in polygons:
            # 旋转 and 翻转 are stored on the placement and shown in the
            # property panel, so they have to reach the board -- otherwise the
            # file opens with the artwork in a pose the user never chose. The
            # transform comes from the placement itself so this writer, the
            # Gerber writer and the canvas cannot drift apart.
            placed = [placement.board_ring(ring) for ring in rings]
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
    else:
        note = "本次只写入板框，没有可写入的图层内容（图片需在铜层或丝印层上）。"
    if skipped:
        note += f" 阻焊层未写入：{'、'.join(sorted(set(skipped)))}（开窗是挖空，不是图形）。"
    if problems:
        note += " 跳过：" + "；".join(problems)

    return records, {
        "shapes": shapes,
        "vertices": vertices,
        "layers": layers_used,
        "note": note,
    }


def _layer_name(layer_id: int) -> str:
    """A layer's display name from its EasyEDA id, for the export report."""
    from .design_layers import LAYERS

    for design_layer in LAYERS:
        if design_layer.easyeda_id == layer_id:
            return design_layer.name
    return f"layer {layer_id}"


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
    set -- so the comparison is against dark, not light.
    """
    from .image_prep import imread_unicode

    path = placement.processed_path or placement.path
    if not path:
        problems.append(f"{placement.layer}: 没有图源")
        return None
    try:
        raw = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        problems.append(f"{placement.layer}: 无法读取 {path}（{exc}）")
        return None
    if raw is None or raw.size == 0:
        problems.append(f"{placement.layer}: 图片为空")
        return None
    return raw < 128
