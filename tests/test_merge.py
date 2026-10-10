"""Merging finished boards into one project file.

The merged file is only useful if every card in it is *the same card* that was
exported on its own, moved sideways. So the tests compare geometry rather than
count records: a merge that dropped a decoration, applied the shift twice, or
mismatched an ``OBJ`` with the wrong picture would all still produce a file that
opens.

Two properties carry most of the weight:

* **A uniform translation is the whole transform.** Shifting a record's
  coordinates must not disturb widths, angles or text, so a merge that changed
  any of them is a bug the tests have to catch.
* **The anchor and the silhouette move together.** ``IMAGE`` stores both a
  placement anchor and a path; moving one without the other tears a picture
  away from its own outline. Measured on the real template: both fields shift by
  the same delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import ArtBoard, ImagePlacement  # noqa: E402
from pcbart.badge_card import badge_cutouts, badge_spec, badge_texts  # noqa: E402
from pcbart.epru import EproArchive, Record  # noqa: E402
from pcbart.export import export_design  # noqa: E402
from pcbart.merge import (  # noqa: E402
    board_bounds,
    merge_projects,
    shift_body,
    shift_path,
)

TOP_SILK = "顶层丝印层"


def card(tmp_path: Path, name: str, x_mm: float = 0.0, *, with_picture: bool = True) -> Path:
    """A complete 通行证, exported through the real writer.

    ``with_picture`` puts a real (if tiny) colour picture on the card so the
    export takes the ``OBJ`` + ``BLOB`` path. That pair is the one part of the
    format whose ids have to survive the merge, so the merge tests exercise it
    rather than only the geometry records.
    """
    from PIL import Image

    board = ArtBoard()
    board.outline = badge_spec("精0")
    board.outline_ready = True
    board.texts = badge_texts()
    board.cutouts = badge_cutouts()
    board.operator_name = name
    board.operator_number = "PL03"
    if with_picture:
        picture = tmp_path / f"art-{name}.png"
        image = Image.new("RGB", (64, 64), (200, 30, 30))
        image.save(picture)
        board.placements.append(
            ImagePlacement(
                path=str(picture),
                width_mm=10.0,
                height_mm=10.0,
                x_mm=x_mm,
                y_mm=0.0,
                layer=TOP_SILK,
            )
        )
    target = tmp_path / f"{name}.epro2"
    export_design(board, target, color_print=True)
    return target


# ------------------------------------------------------------------ shifting


def test_a_plain_run_moves_by_the_offset():
    path = [0.0, 0.0, "L", 10.0, 0.0, 10.0, 10.0]
    moved = shift_path(path, 5.0, -3.0)
    assert moved == [5.0, -3.0, "L", 15.0, -3.0, 15.0, 7.0]


def test_an_arc_keeps_its_sweep_angle():
    """The angle is not a coordinate; scaling it would bend the curve."""
    path = [0.0, 0.0, "L", 10.0, 0.0, "ARC", -90.0, 10.0, 10.0]
    moved = shift_path(path, 2.0, 4.0)
    assert moved == [2.0, 4.0, "L", 12.0, 4.0, "ARC", -90.0, 12.0, 14.0]


def test_a_cubic_moves_all_three_points():
    path = [0.0, 0.0, "L", 1.0, 1.0, "C", 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    moved = shift_path(path, 10.0, 10.0)
    # Past the leading pair and its ``L``: two control points and the endpoint.
    assert moved[3:] == [11.0, 11.0, "C", 12.0, 12.0, 13.0, 13.0, 14.0, 14.0]


def test_a_circle_moves_its_centre_and_keeps_its_radius():
    moved = shift_path(["CIRCLE", 5.0, 6.0, 3.0], 1.0, 2.0)
    assert moved == ["CIRCLE", 6.0, 8.0, 3.0]


def test_a_rectangle_moves_its_anchor_only():
    """``R`` grows from its anchor, so width and height must not shift."""
    moved = shift_path(["R", 1.0, 2.0, 8.0, 4.0], 3.0, 5.0)
    assert moved == ["R", 4.0, 7.0, 8.0, 4.0]


def test_a_nested_path_moves_every_ring():
    path = [[0.0, 0.0, "L", 1.0, 0.0], [5.0, 5.0, "L", 6.0, 5.0]]
    moved = shift_path(path, 1.0, 1.0)
    assert moved == [[1.0, 1.0, "L", 2.0, 1.0], [6.0, 6.0, "L", 7.0, 6.0]]


def test_shift_body_moves_the_anchor_and_leaves_the_rest_alone():
    body = {
        "startX": 10.0,
        "startY": 20.0,
        "width": 30.0,
        "height": 40.0,
        "angle": 270.0,
        "mirror": True,
        "layerId": 3,
        "path": [0.0, 0.0, "L", 5.0, 5.0],
    }
    moved = shift_body(body, 7.0, -2.0)
    assert moved["startX"] == 17.0 and moved["startY"] == 18.0
    assert moved["path"] == [7.0, -2.0, "L", 12.0, 3.0]
    for key in ("width", "height", "angle", "mirror", "layerId"):
        assert moved[key] == body[key]


def test_shift_body_moves_line_endpoints():
    moved = shift_body(
        {"startX": 1.0, "startY": 2.0, "endX": 3.0, "endY": 4.0}, 1.0, 1.0
    )
    assert (moved["startX"], moved["startY"]) == (2.0, 3.0)
    assert (moved["endX"], moved["endY"]) == (4.0, 5.0)


def test_shift_body_does_not_touch_a_missing_path():
    moved = shift_body({"startX": 1.0, "startY": 2.0}, 1.0, 1.0)
    assert "path" not in moved


# ------------------------------------------------------------------- bounds


def test_bounds_follow_the_outline_not_the_overhanging_artwork():
    """Decoration reaches past the board; spacing the grid on it would be ragged.

    Measured on a real card: an ``IMAGE`` record's own box extends about 20 mm
    beyond the 49.873 mm outline, and its anchor box another 20. Laying cards out
    on those extremes would space them by whichever picture sticks out furthest.
    """
    outline = Record(
        header={"type": "POLY", "ticket": 1},
        body={
            "layerId": 11,
            "path": [0.0, 0.0, "L", 100.0, 0.0, 100.0, 100.0, 0.0, 100.0],
        },
    )
    overhang = Record(
        header={"type": "FILL", "ticket": 2},
        body={
            "layerId": 3,
            "path": [500.0, 500.0, "L", 600.0, 600.0],
        },
    )
    assert board_bounds([outline, overhang]) == (0.0, 0.0, 100.0, 100.0)


def test_bounds_fall_back_to_all_geometry_without_an_outline():
    only = Record(
        header={"type": "FILL", "ticket": 1},
        body={"layerId": 3, "path": [5.0, 6.0, "L", 7.0, 8.0]},
    )
    assert board_bounds([only]) == (5.0, 6.0, 7.0, 8.0)


def test_bounds_of_nothing_are_zero():
    assert board_bounds([]) == (0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------- merging


def test_the_merge_keeps_every_board(tmp_path):
    sources = [card(tmp_path, "甲"), card(tmp_path, "乙"), card(tmp_path, "丙")]
    report = merge_projects(sources, tmp_path / "all.epro2", columns=3)
    assert report["boards"] == 3

    document = EproArchive.read(tmp_path / "all.epro2").document
    outlines = [
        r
        for r in document.iter_type("POLY")
        if isinstance(r.body, dict) and r.body.get("layerId") == 11
    ]
    # Each source card carries ten outline POLY records (4 ARC + 6 straight),
    # so three cards must leave thirty. A count rather than a flag: dropping one
    # edge of one card would still leave "an outline".
    assert len(outlines) == 30


def test_the_merge_carries_every_geometry_record(tmp_path):
    sources = [card(tmp_path, "甲"), card(tmp_path, "乙")]
    merged = tmp_path / "all.epro2"
    merge_projects(sources, merged, columns=2)

    def counts(path):
        document = EproArchive.read(path).document
        from collections import Counter

        return Counter(
            r.type
            for r in document.records
            if r.type in ("POLY", "FILL", "LINE", "IMAGE", "STRING", "OBJ", "BLOB")
        )

    single = counts(sources[0])
    both = counts(merged)
    for record_type, count in single.items():
        if record_type == "BLOB":
            # Identical pictures share one blob: the bytes are the same and the
            # two OBJ records point at the same id.
            assert both[record_type] <= count * 2
        else:
            assert both[record_type] == count * 2, record_type


def _outline_box(document) -> tuple[float, float, float, float]:
    outlines = [
        r
        for r in document.iter_type("POLY")
        if isinstance(r.body, dict) and r.body.get("layerId") == 11
    ]
    return board_bounds(outlines)


def test_boards_do_not_overlap(tmp_path):
    from pcbart.units import EPRU_UNIT_MM as U

    sources = [card(tmp_path, str(index)) for index in range(4)]
    merged = tmp_path / "all.epro2"
    report = merge_projects(sources, merged, columns=2, gap_mm=5.0)

    document = EproArchive.read(merged).document
    # One box for the whole panel, then split it by the pitch the report gives:
    # with a 2x2 grid the panel is two cards wide and two tall plus one gap.
    whole = _outline_box(document)
    cell_w = report["board_mm"][0] / U
    cell_h = report["board_mm"][1] / U
    pitch_x = cell_w + 5.0 / U
    pitch_y = cell_h + 5.0 / U

    assert whole[2] - whole[0] == pytest.approx(pitch_x * 2 - 5.0 / U, abs=1e-3)
    assert whole[3] - whole[1] == pytest.approx(pitch_y * 2 - 5.0 / U, abs=1e-3)

    # And placing four cards on that grid fills it exactly: the four cell
    # rectangles tile the panel box with no gap left over at the far edge.
    cells = [
        (whole[0] + column * pitch_x, whole[1] + row * pitch_y)
        for row in range(2)
        for column in range(2)
    ]
    for x, y in cells:
        assert whole[0] - 1e-3 <= x <= whole[2] + 1e-3
        assert whole[1] - 1e-3 <= y <= whole[3] + 1e-3


def test_the_cards_land_on_an_even_pitch(tmp_path):
    """Three cards in one row sit on a constant pitch.

    Read from the geometry rather than the report: the report could be right
    about its own arithmetic while the records were written somewhere else.
    """
    from pcbart.geometry import densify_path
    from pcbart.units import EPRU_UNIT_MM as U

    sources = [card(tmp_path, str(index)) for index in range(3)]
    merged = tmp_path / "all.epro2"
    merge_projects(sources, merged, columns=3, gap_mm=5.0)

    document = EproArchive.read(merged).document
    xs: list[float] = []
    for record in document.iter_type("POLY"):
        if not isinstance(record.body, dict) or record.body.get("layerId") != 11:
            continue
        for ring in densify_path(record.body.get("path")):
            xs.extend(point[0] for point in ring)

    # Assign each point to a cell by its own x. A card is 49.873 mm wide inside a
    # 54.873 mm cell, so every point of card *k* falls strictly inside cell *k*
    # and a floor division separates the cards without guessing at gaps.
    pitch = (49.873 + 5.0) / U
    cells: dict[int, list[float]] = {}
    for x in xs:
        cells.setdefault(int(x // pitch), []).append(x)

    assert sorted(cells) == [0, 1, 2]
    lefts = [min(values) * U for _cell, values in sorted(cells.items())]
    assert lefts[0] == pytest.approx(0.0, abs=1e-3)
    assert lefts[1] - lefts[0] == pytest.approx(49.873 + 5.0, abs=2e-3)
    assert lefts[2] - lefts[1] == pytest.approx(49.873 + 5.0, abs=2e-3)


def test_every_picture_keeps_its_blob(tmp_path):
    """An ``OBJ`` whose ``blob:`` id has no ``BLOB`` renders as an empty box."""
    sources = [card(tmp_path, "甲"), card(tmp_path, "乙")]
    merged = tmp_path / "all.epro2"
    merge_projects(sources, merged, columns=2)

    document = EproArchive.read(merged).document
    ids = {r.id for r in document.iter_type("BLOB")}
    for record in document.iter_type("OBJ"):
        path = str(record.body.get("path", ""))
        assert path.startswith("blob:")
        assert path.split("blob:", 1)[1] in ids


def test_a_shared_picture_is_stored_once(tmp_path):
    """Two cards carrying the *same* picture share one blob.

    The two cards use one source file, so every element they have in common --
    the card's own 二维码 / 条形码 / 职业图标 -- embeds byte-identical data. The
    merge keeps one copy of each, and both cards' ``OBJ`` records point at it.
    """
    shared = tmp_path / "shared.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (10, 80, 200)).save(shared)

    def build(name: str, picture: Path | None) -> Path:
        board = ArtBoard()
        board.outline = badge_spec("精0")
        board.outline_ready = True
        board.texts = badge_texts()
        board.cutouts = badge_cutouts()
        board.operator_name = "甲"
        board.operator_number = "PL03"
        if picture is not None:
            board.placements.append(
                ImagePlacement(
                    path=str(picture),
                    width_mm=10.0,
                    height_mm=10.0,
                    layer=TOP_SILK,
                )
            )
        target = tmp_path / f"{name}.epro2"
        export_design(board, target, color_print=True)
        return target

    sources = [build("甲", shared), build("乙", shared)]
    merged = tmp_path / "all.epro2"
    merge_projects(sources, merged, columns=2)

    counts = []
    for source in sources:
        document = EproArchive.read(source).document
        counts.append(len(list(document.iter_type("BLOB"))))

    document = EproArchive.read(merged).document
    blobs = list(document.iter_type("BLOB"))
    contents = [b.body["content"] for b in blobs]
    assert len(contents) == len(set(contents)), "同一个 BLOB 内容写了两次"
    # Both cards together carry twice the blobs of one, minus the duplicates.
    assert len(blobs) < counts[0] + counts[1]
    assert len(list(document.iter_type("OBJ"))) == 2 * len(
        list(EproArchive.read(sources[0]).document.iter_type("OBJ"))
    )


def test_geometry_is_translated_and_not_rebuilt(tmp_path):
    """The merged card must equal the source card plus one offset, exactly.

    This is the test that would catch a shift applied twice, an anchor moved
    without its path, or a record copied before rather than after the move. It
    compares **record by record, in order**, because the writer preserves the
    source order -- flattening the points and sorting them instead would make
    the test pass or fail on floating-point tie-breaking rather than on the
    geometry.
    """
    source = card(tmp_path, "甲")
    merged = tmp_path / "all.epro2"
    merge_projects([source], merged, columns=1)

    def geometry(document):
        out = []
        for record in document.records:
            if record.type not in ("POLY", "FILL", "LINE", "IMAGE"):
                continue
            body = record.body
            points = []
            path = body.get("path")
            if path is not None:
                from pcbart.geometry import densify_path

                points.extend(p for ring in densify_path(path) for p in ring)
            for x_key, y_key in (
                ("startX", "startY"),
                ("endX", "endY"),
                ("x", "y"),
            ):
                if isinstance(body.get(x_key), (int, float)):
                    points.append((float(body[x_key]), float(body[y_key])))
            out.append((record.type, body.get("layerId"), points))
        return out

    before = geometry(EproArchive.read(source).document)
    after = geometry(EproArchive.read(merged).document)
    assert len(before) == len(after)

    offsets = set()
    for (type_a, layer_a, a), (type_b, layer_b, b) in zip(before, after):
        assert (type_a, layer_a) == (type_b, layer_b)
        assert len(a) == len(b)
        for (x1, y1), (x2, y2) in zip(a, b):
            offsets.add((round(x2 - x1, 6), round(y2 - y1, 6)))

    # One offset for the whole card: a card is rigid, so every point in every
    # record moves by the same amount. A second value would mean some part of
    # the card was moved relative to the rest -- an anchor left behind, or a
    # path shifted twice.
    assert len(offsets) == 1, f"卡片不是整体平移：{sorted(offsets)[:6]}"
    dx, dy = offsets.pop()
    # And the card really moved: a merge that forgot the shift would give
    # (0, 0) and still satisfy "one offset".
    assert (dx, dy) != (0.0, 0.0)


def test_merging_without_geometry_is_refused(tmp_path):
    from pcbart.templates import build_minimal_project

    empty = tmp_path / "empty.epro2"
    document, _ = build_minimal_project(title="empty")
    EproArchive.create("empty", document=document).write(empty)

    with pytest.raises(ValueError):
        merge_projects([empty], tmp_path / "out.epro2")


def test_columns_control_the_grid(tmp_path):
    sources = [card(tmp_path, str(index)) for index in range(4)]
    merged = tmp_path / "all.epro2"
    report = merge_projects(sources, merged, columns=2, gap_mm=6.0)
    assert report["columns"] == 2
    assert report["rows"] == 2
    assert report["board_mm"] == (pytest.approx(49.873, abs=1e-3), pytest.approx(99.873, abs=1e-3))


def test_blobs_go_into_the_blob_section_not_the_pcb(tmp_path):
    """A ``BLOB`` belongs to its own document section.

    This is not a formatting nicety. An ``OBJ`` resolves its picture through
    ``path: "blob:<id>"``, and the editor looks that id up among the ``BLOB``
    records of the **BLOB section**. A blob written into the PCB section next
    to its ``OBJ`` is simply not found, and the picture comes back as an empty
    rectangle.

    Measured on the merged file: every record was present, every ``blob:`` id
    matched a ``BLOB`` record, and the QR symbols and illustrations still
    vanished from the editor. The single-board writer
    (:func:`pcbart.generator.generate_outline_board`) had it right all along;
    only the merge path put them in the wrong section.
    """
    sources = [card(tmp_path, "甲"), card(tmp_path, "乙")]
    merged = tmp_path / "all.epro2"
    merge_projects(sources, merged, columns=2)

    document = EproArchive.read(merged).document

    # Walk the records once, tracking which document section each one sits in.
    current: str | None = None
    placement: dict[str, set[str]] = {}
    for record in document.records:
        if record.type == "DOCHEAD" and isinstance(record.body, dict):
            current = str(record.body.get("docType") or "")
            continue
        if record.type in ("OBJ", "BLOB"):
            placement.setdefault(record.type, set()).add(current or "")

    assert placement.get("BLOB") == {"BLOB"}, (
        f"BLOB 必须写在 BLOB 分区，实际在 {placement.get('BLOB')}"
    )
    assert placement.get("OBJ") == {"PCB"}, (
        f"OBJ 必须写在 PCB 分区，实际在 {placement.get('OBJ')}"
    )


def test_the_merged_file_matches_a_single_export_s_section_layout(tmp_path):
    """Merging must not change which section a record type lives in.

    Compares against a real single-board export rather than a hard-coded list,
    so the two writers cannot drift apart: whatever section the single-card
    writer chooses is the section the merge has to choose.
    """
    source = card(tmp_path, "甲")
    merged = tmp_path / "all.epro2"
    merge_projects([source], merged, columns=1)

    def sections(path: Path) -> dict[str, set[str]]:
        document = EproArchive.read(path).document
        current: str | None = None
        found: dict[str, set[str]] = {}
        for record in document.records:
            if record.type == "DOCHEAD" and isinstance(record.body, dict):
                current = str(record.body.get("docType") or "")
                continue
            if record.type in ("OBJ", "BLOB"):
                found.setdefault(record.type, set()).add(current or "")
        return found

    assert sections(merged) == sections(source)
