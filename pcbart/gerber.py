"""Gerber (RS-274X) output, laid out the way 嘉立创 expects it.

The ``.epro2`` export is for continuing to edit the design in EasyEDA Pro. This
module is for *making the board*: the same geometry written as the layer files a
fabricator's CAM tool reads.

Each layer becomes one file, named after the layer and carrying the extension
the industry (and 嘉立创's uploader) recognises:

    Gerber_TopLayer.GTL              顶层铜
    Gerber_BottomLayer.GBL           底层铜
    Gerber_TopSilkscreenLayer.GTO    顶层丝印
    Gerber_BottomSilkscreenLayer.GBO 底层丝印
    Gerber_TopSolderMaskLayer.GTS    顶层阻焊开窗
    Gerber_BottomSolderMaskLayer.GBS 底层阻焊开窗
    Gerber_BoardOutlineLayer.GKO     板框

Filled artwork is written as **regions** (``G36``/``G37``), which is what a
solid area is in Gerber: the outline is walked as a closed path and the inside
is filled. A shape with a hole -- the counter of a letter, the gap between two
fingers -- cannot be expressed by walking one path, so the hole is emitted
under ``%LPC*%`` (clear polarity) and the layer returns to ``%LPD*%``
afterwards. That is the same construction KiCad emits, and it is what CAM
tools have handled for decades.

**Everything in these files is ASCII.** Gerber parsers in fab shops are older
and more literal than anything else this project writes to, and a UTF-8
comment is exactly the kind of thing that turns into a mangled layer name or a
rejected upload. Chinese text stays in the report the UI shows.

The board outline is stroked rather than filled: a fabricator's edge cut is a
centre line the router follows, not an area.
"""

from __future__ import annotations

import datetime
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

__all__ = [
    "DEFAULT_PREFIX",
    "GERBER_LAYERS",
    "OUTLINE_LAYER",
    "export_gerber",
    "export_gerber_archive",
    "gerber_text",
]


DEFAULT_PREFIX = "Gerber_"
"""File-name prefix, as the user asked for it."""

DEFAULT_OUTLINE_WIDTH_MM = 0.125
"""Edge-cut stroke width. Matches what EasyEDA Pro writes for a board outline."""

#: Coordinate format: 4 integer and 6 decimal digits, in millimetres. Six
#: decimals is a nanometre, far finer than any etcher, and it leaves four
#: integer digits for boards up to 9999 mm across.
_COORD_SCALE = 1_000_000
_COORD_FORMAT = "%FSLAX46Y46*%"


class GerberLayer:
    """One output file: what it is called and what it means."""

    def __init__(self, key: str, file_name: str, function: str, label: str) -> None:
        self.key = key
        self.file_name = file_name
        self.function = function
        """The X2 ``FileFunction`` attribute, which is how a CAM tool knows
        which layer it is holding without trusting the file name."""
        self.label = label
        """Chinese name used in the report, not in the file."""


#: The six artwork layers, keyed by the editor's layer key.
GERBER_LAYERS: tuple[GerberLayer, ...] = (
    GerberLayer("top_copper", "TopLayer.GTL", "Copper,L1,Top", "顶层铜"),
    GerberLayer(
        "bottom_copper", "BottomLayer.GBL", "Copper,L2,Bot", "底层铜"
    ),
    GerberLayer(
        "top_silk", "TopSilkscreenLayer.GTO", "Legend,Top", "顶层丝印"
    ),
    GerberLayer(
        "bottom_silk", "BottomSilkscreenLayer.GBO", "Legend,Bot", "底层丝印"
    ),
    GerberLayer(
        "top_mask",
        "TopSolderMaskLayer.GTS",
        "Soldermask,Top",
        "顶层阻焊开窗",
    ),
    GerberLayer(
        "bottom_mask",
        "BottomSolderMaskLayer.GBS",
        "Soldermask,Bot",
        "底层阻焊开窗",
    ),
)

OUTLINE_LAYER = GerberLayer(
    "outline", "BoardOutlineLayer.GKO", "Profile,NP", "板框"
)


def _coordinate(value: float) -> str:
    """One axis of a coordinate, in the file's fixed-point units."""
    return str(int(round(float(value) * _COORD_SCALE)))


def _point(x: float, y: float) -> str:
    return f"X{_coordinate(x)}Y{_coordinate(y)}"


def _is_point(value) -> bool:
    """Whether ``value`` is one ``(x, y)`` pair rather than a list of them."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(number, (int, float)) for number in value)
    )


def _normalise_strokes(strokes, default_width: float, default_closed: bool):
    """Accept both stroke spellings and return ``(points, width, closed)``.

    Plain paths carry the file's own width and closing rule, which is what the
    board outline wants -- every edge-cut contour closes, and there is one
    aperture. The card's fixed decorations need both per path: its hatching is
    a set of straight open rules, while its border rings close, and treating
    the two the same draws a line back over every open one.
    """
    normalised = []
    for item in strokes:
        if not item:
            continue
        if _is_point(item[0]):
            normalised.append((list(item), default_width, default_closed))
        else:
            points = list(item[0])
            width = float(item[1]) if len(item) > 1 else default_width
            closed = bool(item[2]) if len(item) > 2 else default_closed
            normalised.append((points, width, closed))
    return normalised


def gerber_text(
    *,
    function: str,
    regions: list[list[list[tuple[float, float]]]] = (),
    strokes: list = (),
    stroke_width_mm: float = DEFAULT_OUTLINE_WIDTH_MM,
    close_strokes: bool = True,
    created: datetime.datetime | None = None,
) -> str:
    """The text of one Gerber file.

    ``regions`` is a list of complex polygons: each is a list of rings, the
    first the outer contour and the rest holes. ``strokes`` is a list of paths,
    each either a bare path or ``(points, width_mm, closed)``; a bare path uses
    ``stroke_width_mm`` and ``close_strokes``.

    One aperture is declared per distinct width, because a decoration's line
    weight and the board edge's are different jobs -- a single aperture per
    file would silently redraw one of them at the other's width.

    Kept separate from file writing so the format can be asserted as text in
    tests, and so nothing here needs to know where the file goes.
    """
    stamp = created or datetime.datetime.now()
    width = max(0.001, float(stroke_width_mm))
    paths = _normalise_strokes(strokes, width, close_strokes)

    widths: list[float] = []
    for _points, path_width, _closed in paths:
        rounded = round(max(0.001, path_width), 6)
        if rounded not in widths:
            widths.append(rounded)
    if not widths:
        widths = [width]

    lines = [
        "G04 Generated by PCB Art Board Tool*",
        f"G04 Created {stamp.strftime('%Y-%m-%dT%H:%M:%S')}*",
        # X2 attributes. A CAM tool that ignores them loses nothing; one that
        # reads them knows the layer without parsing the file name, which is
        # exactly what 嘉立创's importer does when a name is ambiguous.
        f"%TF.GenerationSoftware,PCBart,PCBArtBoardTool,1*%",
        f"%TF.FileFunction,{function}*%",
        "%TF.FilePolarity,Positive*%",
        _COORD_FORMAT,
        "%MOMM*%",
        "%LPD*%",
        "G01*",
    ]
    # Apertures are numbered from 10 up, the lowest number a Gerber file may
    # use for a draw.
    aperture = {
        value: 10 + index for index, value in enumerate(widths)
    }
    for value in widths:
        lines.append(f"%ADD{aperture[value]}C,{value:.6f}*%")
    lines.append(f"D{aperture[widths[0]]}*")

    for points, path_width, closed in paths:
        if len(points) < 2:
            continue
        lines.append(f"D{aperture[round(max(0.001, path_width), 6)]}*")
        first = points[0]
        lines.append(f"{_point(*first)}D02*")
        lines.append("G01*")
        for point in points[1:]:
            lines.append(f"{_point(*point)}D01*")
        # A closed contour has to come back to where it started or the router
        # has no closed path to follow. An open rule stays open: closing it
        # would draw a segment the card does not have.
        if closed and points[-1] != first:
            lines.append(f"{_point(*first)}D01*")

    for rings in regions:
        rings = [ring for ring in rings if _is_writable_ring(ring)]
        if not rings:
            continue
        lines.extend(_region_lines(rings[0]))
        for hole in rings[1:]:
            # Clear polarity, then straight back to dark. Emitting each hole
            # immediately after the shape it belongs to is what keeps the
            # clearing from eating shapes drawn earlier in the same file.
            lines.append("%LPC*%")
            lines.extend(_region_lines(hole))
            lines.append("%LPD*%")

    lines.append("M02*")
    return "\n".join(lines) + "\n"


def _is_writable_ring(ring) -> bool:
    """Whether a contour has enough geometry to enclose an area.

    ⚠️ **An arc ring needs two vertices where a chord ring needs three.** Two
    points joined by two 180-degree sweeps is a circle, and one sweep plus one
    chord is a half-disc -- both real areas that a plain ``>= 3`` test drops
    without saying so.
    """
    from .geometry import ArcRing

    if isinstance(ring, ArcRing):
        return len(ring.points) >= 2
    return len(ring) >= 3


def _region_lines(ring) -> list[str]:
    """One closed contour as a Gerber region.

    ``ring`` is either a list of ``(x, y)`` vertices or an
    :class:`pcbart.geometry.ArcRing`, whose curved segments are written with
    ``G75`` (multi-quadrant mode) plus ``G02``/``G03``. Chord approximation is
    not used on purpose: the point of tracing the 阵营 logo with VTracer was to
    stop the fab drawing a chain of facets, and flattening the arcs back into
    lines at the last step would put them straight back.
    """
    from .geometry import ArcRing

    if isinstance(ring, ArcRing):
        return _arc_region_lines(ring)

    lines = ["G36*", f"{_point(*ring[0])}D02*", "G01*"]
    lines.extend(f"{_point(*point)}D01*" for point in ring[1:])
    lines.append(f"{_point(*ring[0])}D01*")
    lines.append("G37*")
    return lines


def _arc_region_lines(ring) -> list[str]:
    """A Gerber region whose contour mixes straight runs and circular arcs.

    ``G75`` selects multi-quadrant arc mode, where ``I``/``J`` are the offset
    from the current point to the centre -- so the centre has to be recovered
    from the chord and the sweep. That is the same derivation
    :func:`pcbart.geometry.sample_arc` uses, kept deliberately in step: a
    second one whose sign convention differed would put every arc on the far
    side of its chord in the fab file while the canvas still looked right.
    """
    points = ring.points
    count = len(points)
    if count < 2:
        return []

    # ⚠️ **Segment ``i`` runs from ``points[i-1]`` into ``points[i]``**, and the
    # pen has to start where that segment starts -- at ``points[count-1]``.
    # Starting at ``points[0]`` and walking the segments in index order draws
    # the closing segment first, from the wrong end: measured on a half disc it
    # emitted ``X0Y0`` then an arc to ``X0Y0`` (a full circle of zero extent)
    # and finally the chord, which gerbonara read as a 40 mm shape instead of
    # 10. The whole ring is shifted by one segment, so every arc lands on the
    # previous chord.
    lines = ["G36*", f"{_point(*points[count - 1])}D02*", "G75*"]
    for index in range(count):
        start = points[index - 1]
        end = points[index]
        sweep = float(ring.sweeps[index])
        centre = None if abs(sweep) < 1e-6 else _arc_centre(start, end, sweep)
        if centre is None:
            lines.append("G01*")
            lines.append(f"{_point(*end)}D01*")
            continue
        # Counter-clockwise is G03 and clockwise G02; the signs agree with the
        # model's because both work in a Y-up frame.
        lines.append("G03*" if sweep > 0 else "G02*")
        # ⚠️ **X and Y come before I and J.** A data block is a loose sequence
        # of words in the specification, but readers are not that loose:
        # gerbonara's coordinate regex is ``(G..)?(X..)?(Y..)?(I..)?(J..)?D..``
        # and an ``I..J..X..Y..`` block fails to match, so the whole arc is
        # discarded as an unknown statement. Measured on this export: every one
        # of the 4299 arcs was dropped that way while the file still looked
        # plausible.
        lines.append(
            f"{_point(*end)}"
            f"I{_coordinate(centre[0] - start[0])}"
            f"J{_coordinate(centre[1] - start[1])}D01*"
        )
    lines.append("G37*")
    return lines


def _arc_centre(
    start: tuple[float, float],
    end: tuple[float, float],
    sweep_deg: float,
) -> tuple[float, float] | None:
    """The centre of the circular arc from ``start`` to ``end``.

    ``None`` when the sweep is degenerate and the segment is really a line.
    The centre lies on the chord's perpendicular bisector; the sweep's sign
    chooses which of the two candidates, exactly as
    :func:`pcbart.geometry.sample_arc` does.
    """
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    chord = math.hypot(dx, dy)
    half = math.radians(float(sweep_deg)) / 2.0
    sin_half = math.sin(half)
    if chord <= 1e-12 or abs(sin_half) <= 1e-12:
        return None
    radius = abs(chord / (2.0 * sin_half))
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    nx = -dy / chord
    ny = dx / chord
    height = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))
    sign = 1.0 if sweep_deg > 0 else -1.0
    return (mid_x + sign * height * nx, mid_y + sign * height * ny)


def export_gerber(
    board,
    output_dir: str | Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    title: str = "",
    copper_params=None,
    outline_width_mm: float = DEFAULT_OUTLINE_WIDTH_MM,
    archive: bool = True,
) -> dict[str, object]:
    """Write every layer of ``board`` as Gerber, plus a zip of the set.

    Only layers that actually carry geometry are written. An empty file is not
    harmless: 嘉立创's uploader reads the set it is given, and a stray empty
    silkscreen file invites a question about which layers the board has.
    The outline is always written -- a board without one cannot be cut.

    Returns the report the UI shows: the files, their sizes, and anything that
    was skipped.
    """
    from .design_layers import LAYERS
    from .export import WRITABLE_KINDS, build_copper
    from .validation import check_outline, warning_sentence

    if not board.outline_ready:
        raise ValueError("尚未设置板框：请先打开「板框设置」确定板框形状")

    problems = board.outline.validate()
    if problems:
        raise ValueError("板框设置有问题：" + "；".join(problems))

    points = board.outline_polygon()
    if len(points) < 3:
        raise ValueError(
            "板框几何为空，无法导出。"
            "异形板需要在「板框设置」里选好基础形状并导入图片。"
        )
    if board.outline.standee and board.standee_error:
        raise ValueError(
            "立牌插舌没有连到板框上："
            + str(board.standee_error)
            + "。请在「板框设置 → 立牌设置」里调整插舌位置或尺寸。"
        )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # Solder mask *is* written here, unlike in the .epro2 export. The reason
    # the two differ is not inconsistency: a mask opening is the absence of
    # material, which a FILL record cannot express, while a Gerber solder-mask
    # layer is *defined* as the set of openings. The same geometry that has no
    # honest encoding in one format is exactly right in the other.
    records, artwork_report = build_copper(
        board, params=copper_params, kinds=WRITABLE_KINDS + ("mask",)
    )

    # The card's fixed artwork goes in on the same footing as the imported
    # picture and the wording. It is a *different encoding* of the same design
    # the ``.epro2`` writer replays as the source project's own records -- the
    # border, the wordmark, the crest and, most importantly, the solder-mask
    # openings. Without it the fab package had empty mask layers, so the
    # illustration and both codes would have been printed under the mask.
    from . import badge_art

    decoration_regions, decoration_strokes = badge_art.decoration_geometry(board)

    by_layer: dict[int, list] = {}
    strokes_by_layer: dict[int, list] = {}
    for layer_id, rings, _width in records:
        by_layer.setdefault(layer_id, []).append(rings)
    for layer_id, rings in decoration_regions:
        by_layer.setdefault(layer_id, []).append(rings)
    for layer_id, path_points, width, closed in decoration_strokes:
        strokes_by_layer.setdefault(layer_id, []).append(
            (path_points, width, closed)
        )

    written: list[dict[str, object]] = []
    empty: list[str] = []

    for spec in GERBER_LAYERS:
        key = spec.key
        layer_id = next(
            (item.easyeda_id for item in LAYERS if item.key == key), None
        )
        regions = by_layer.get(layer_id, []) if layer_id is not None else []
        strokes = (
            strokes_by_layer.get(layer_id, []) if layer_id is not None else []
        )
        if not regions and not strokes:
            empty.append(spec.label)
            continue
        path = directory / f"{prefix}{spec.file_name}"
        path.write_text(
            gerber_text(
                function=spec.function,
                regions=regions,
                strokes=strokes,
                # The stroke list is per path and always explicit, so the
                # file-wide default is never consulted here.
                close_strokes=False,
            ),
            encoding="ascii",
        )
        written.append(_entry(spec, path))

    # The board outline plus every opening cut through it. A cutout goes on the
    # same layer as the outline because that is what it is -- a second closed
    # contour the router follows -- and it is a *stroke* rather than a region,
    # so the fab reads it as a cut path instead of as copper to keep. A board
    # exported without it comes back solid while the canvas shows a slot.
    outline_path = directory / f"{prefix}{OUTLINE_LAYER.file_name}"
    outline_path.write_text(
        gerber_text(
            function=OUTLINE_LAYER.function,
            strokes=[points, *board.cutout_rings()],
            stroke_width_mm=outline_width_mm,
        ),
        encoding="ascii",
    )
    written.insert(0, _entry(OUTLINE_LAYER, outline_path))

    # The drilled holes, complementing the outline. A round opening reaches the
    # fabricator twice on purpose: once as a closed contour on the profile layer
    # (the edge-cut path) and once as a drill hit here. That is what the sample
    # package does, and the two agree because they are built from the same
    # ``board.cutouts``.
    holes = drill_holes(board)
    drill_entry = None
    if holes:
        drill_path = directory / f"{prefix}{DRILL_FILE_NAME}"
        drill_path.write_text(drill_text(holes), encoding="ascii")
        drill_entry = {
            "path": drill_path,
            "name": drill_path.name,
            "label": "钻孔（非金属化）",
            "bytes": drill_path.stat().st_size,
        }
        written.append(drill_entry)

    # ⚠️ The manufacturability warnings belong in the report the user reads,
    # not left on the validation object. A board with a 0.15 mm neck used to
    # export with no mention of it at all -- see
    # :func:`pcbart.validation.warning_sentence`.
    outline_report = check_outline(points)

    archive_path = None
    if archive:
        name = (title.strip() or "ArtBoard").replace("/", "_").replace("\\", "_")
        archive_path = directory / f"{prefix}{name}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for entry in written:
                bundle.write(str(entry["path"]), Path(str(entry["path"])).name)

    note = _note(written, empty, archive_path)
    warning = warning_sentence(outline_report)
    if warning:
        note += " " + warning

    return {
        "directory": str(directory),
        "files": written,
        "archive": None if archive_path is None else str(archive_path),
        "skipped": empty,
        "layers": [str(entry["label"]) for entry in written],
        "shapes": int(artwork_report["shapes"]),
        "vertices": int(artwork_report["vertices"]),
        "holes": len(holes),
        "warnings": [issue.message for issue in outline_report.warnings],
        "note": note,
    }


def export_gerber_archive(
    board,
    archive_path: str | Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    title: str = "",
    copper_params=None,
    outline_width_mm: float = DEFAULT_OUTLINE_WIDTH_MM,
) -> dict[str, object]:
    """Write the whole layer set into **one** zip at exactly ``archive_path``.

    The folder export is for a user who wants to look at the layers; this is
    for the one who wants a single file to upload. Keeping the two apart is
    what lets the second one leave no litter: the layers are built in a
    temporary directory and only the zip survives, so choosing
    ``D:\\下单\\猫娘.zip`` puts one file there rather than one file and seven
    companions the user did not ask for.

    The zip itself is written beside its destination and moved into place, for
    the same reason drafts are: a crash or a full disk halfway through must not
    leave something that looks like a finished order file.
    """
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pcbart-gerber-") as scratch:
        report = export_gerber(
            board,
            scratch,
            prefix=prefix,
            title=title,
            copper_params=copper_params,
            outline_width_mm=outline_width_mm,
        )
        built = Path(str(report["archive"]))
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(built, temporary)
        temporary.replace(target)

    files = [
        {
            "name": str(entry["name"]),
            "label": str(entry["label"]),
            "bytes": int(entry["bytes"]),
        }
        for entry in report["files"]
    ]
    labels = "、".join(str(entry["label"]) for entry in files)
    return {
        "archive": str(target),
        "directory": str(target.parent),
        "files": files,
        "skipped": list(report["skipped"]),
        "layers": list(report["layers"]),
        "shapes": int(report["shapes"]),
        "vertices": int(report["vertices"]),
        "bytes": target.stat().st_size,
        "note": (
            f"已打包 {len(files)} 个 Gerber 文件（{labels}）为 "
            f"{target.name}，可直接上传嘉立创。"
        ),
    }


def _entry(spec: GerberLayer, path: Path) -> dict[str, object]:
    return {
        "path": path,
        "name": path.name,
        "label": spec.label,
        "bytes": path.stat().st_size,
    }


def _note(written, empty, archive_path) -> str:
    """One sentence describing what the folder now contains."""
    labels = "、".join(str(entry["label"]) for entry in written)
    text = f"已写入 {len(written)} 个 Gerber 文件：{labels}。"
    if empty:
        text += f" 无内容的层未生成文件：{'、'.join(empty)}。"
    if archive_path is not None:
        text += f" 同时打包了 {Path(archive_path).name}，可直接上传嘉立创。"
    return text


#: Name of the non-plated drill file, matching what 嘉立创's own EasyEDA Pro
#: export emits, so its uploader recognises the job without being asked.
DRILL_FILE_NAME = "Drill_NPTH_Through.DRL"

#: Diameter of every drill this writer emits, in millimetres.
#:
#: The card's openings are all drawn as shapes rather than as a pad with a hole,
#: so every one of them is the same construction: a circular hole of the size
#: the template drew. A round 挖槽 *is* such a hole; the rectangular ones are not
#: and are left to the outline layer, where a routed slot is the correct
#: description. See :func:`drill_holes`.
MIN_DRILL_MM = 0.10

#: Coordinates in the drill file are written with four decimals of a
#: millimetre. The format line has to say so; a reader cannot infer it.
_DRILL_DECIMALS = 4
_DRILL_FORMAT = "METRIC,LZ,0000.00000"


def drill_holes(board) -> list[tuple[float, float, float]]:
    """Every round opening of ``board`` as ``(x_mm, y_mm, diameter_mm)``.

    **Why only the round ones.** A drill file describes holes a drill bit
    makes. A circular opening in the template is exactly that, so writing it
    here is right -- and it is also what the fabricator's own export does: the
    sample order package carries ``Drill_NPTH_Through.DRL`` with one tool,
    ``T01C3.81000``, and two coordinates. A rectangular or obround opening is
    *not* a drilled hole; it is a routed slot, and the board outline layer
    already carries it as a second closed contour. Listing it as a drill would
    tell the fabricator to plunge a bit and drag it, which is a different
    operation from the one the outline describes.

    ⚠️ **Every hole is non-plated.** The card's openings are mechanical --
    lanyard and hanging holes -- with no copper around them, so the file is a
    non-plated one and the header says so. Writing them as plated would put
    barrel plating in a hole that has nothing to connect.
    """
    holes: list[tuple[float, float, float]] = []
    for cutout in getattr(board, "cutouts", []) or []:
        if str(getattr(cutout, "shape", "")) != "circle":
            continue
        diameter = float(
            max(
                getattr(cutout, "width_mm", 0.0),
                getattr(cutout, "height_mm", 0.0),
            )
        )
        if diameter < MIN_DRILL_MM:
            continue
        holes.append(
            (float(getattr(cutout, "x_mm", 0.0)), float(getattr(cutout, "y_mm", 0.0)), diameter)
        )
    return holes


def _drill_coordinate(value: float) -> str:
    """One axis in the drill file's decimal form, sign included.

    ⚠️ **Decimal, not a scaled integer.** Excellon has an integer form and a
    decimal one, and which is in use has to be inferred from the format line;
    the sample package this writer follows declares ``METRIC,LZ,0000.00000``
    and then writes ``X4.191Y-3.429``. Emitting scaled integers under a header
    that promises decimal places is the kind of mismatch a reader does not
    report -- it silently divides by 1000 and drills in the wrong place.

    Trailing zeroes are trimmed, as the sample does. ``-0`` is normalised away
    because a reader is entitled to treat it as a coordinate rather than as
    nothing.
    """
    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        return "0"
    return text


def drill_text(holes: list[tuple[float, float, float]]) -> str:
    """One Excellon file describing every hole, grouped by tool.

    The layout follows the sample package 嘉立创 itself produced (an EasyEDA Pro
    export for a card with the same openings), because the reader is that
    uploader rather than a general Excellon parser: a header between ``M48`` and
    ``%`` declaring one tool per distinct diameter, then a body that selects
    each tool and lists its coordinates, then ``M30``.

    ⚠️ **ASCII only**, for the same reason every other file here is: a UTF-8
    comment is what turns into a rejected upload or a mangled layer name.
    """
    lines = [
        ";TYPE=NON_PLATED",
        ";Layer: NPTH_Through",
        ";Generated by PCB Art Board Tool",
        "M48",
        _DRILL_FORMAT,
    ]

    # One tool per distinct diameter, in a stable order -- sorted rather than in
    # first-seen order so the same board always produces the same file, which is
    # what makes an export diffable and a golden-file test possible.
    diameters = sorted({round(diameter, _DRILL_DECIMALS) for _, _, diameter in holes})
    tools: dict[float, str] = {}
    for index, diameter in enumerate(diameters, start=1):
        number = f"T{index:02d}"
        tools[diameter] = number
        lines.append(f";Hole size {index} = {diameter:.5f} METRIC")
        lines.append(f"{number}C{diameter:.5f}")
    lines.append("%")
    lines.append("G05")
    lines.append("G90")

    for diameter in diameters:
        selected = [
            (x, y) for x, y, size in holes if round(size, _DRILL_DECIMALS) == diameter
        ]
        if not selected:
            continue
        lines.append(tools[diameter])
        for x, y in selected:
            # Signed, with both axes present. A reader that gets one axis only
            # carries the previous value forward, so an omitted zero would move
            # the hole.
            lines.append(f"X{_drill_coordinate(x)}Y{_drill_coordinate(y)}")

    lines.append("M30")
    return "\n".join(lines) + "\n"
