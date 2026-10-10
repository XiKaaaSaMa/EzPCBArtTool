"""Tests for the 通行证 template's *content*: its wording and its cutout.

The user asked for the blank template's content to appear on the canvas and be
editable -- text, and the lanyard slot. That request touches four things that
have to agree or the card comes out wrong in a way the canvas cannot show:

* the **model** that holds the text and the openings,
* the **canvas** that draws and picks them,
* the **``.epro2`` writer**, which has to keep the wording editable,
* the **Gerber writer**, where a slot that is drawn but not written comes back
  as a solid card.

A fifth thing is checked here rather than anywhere else: that the numbers in
``pcbart.badge_card`` are still the numbers in the files the user supplied.
Those files live outside the repository, so the measurement tests skip when the
library is not on the machine -- the rule tests do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import badge_card  # noqa: E402
from pcbart.artboard import (  # noqa: E402
    ENGLISH_TOKEN,
    NAME_TOKEN,
    NUMBER_TOKEN,
    ArtBoard,
    CutoutElement,
    TextElement,
    substitute_tokens,
)


def template_dir() -> Path | None:
    from pcbart.paths import asset_root

    candidate = asset_root() / "通行证模板源文件"
    return candidate if candidate.exists() else None


def template_files() -> tuple[Path, ...]:
    directory = template_dir()
    if directory is None:
        return ()
    return tuple(sorted(directory.glob("*.epro2")))


# ------------------------------------------------------------------- the model


def test_a_token_is_replaced_only_when_a_value_exists():
    """An unresolved token stays visible.

    Blanking it would look exactly like a card whose name had been deleted,
    where ``{代号}`` says plainly that no operator has been picked yet.
    """
    text = f"{NAME_TOKEN} / {NUMBER_TOKEN}"
    assert substitute_tokens(text, name="鸿雪", number="R145") == "鸿雪 / R145"
    assert substitute_tokens(text) == "{代号} / {编号}"
    assert substitute_tokens(text, name="鸿雪") == "鸿雪 / {编号}"


def test_substitution_does_not_reparse_its_own_output():
    """A name containing a token must not be substituted into.

    ``str.replace`` in a loop would happily turn a ``{编号}`` that arrived as
    somebody's *name* into their number, which is a bug that only shows up for
    one operator and looks like the database being wrong.
    """
    assert substitute_tokens("{代号}", name="{编号}", number="R145") == "{编号}"


def test_a_text_element_resolves_against_a_name_and_number():
    item = TextElement(text=f"ARKNIGHTS-{NUMBER_TOKEN}", size_mm=1.6)
    assert item.resolved_text(name="鸿雪", number="R145") == "ARKNIGHTS-R145"


def test_a_cutout_ring_is_where_the_cutout_says_it_is():
    """The ring's centre is the cutout's centre, not twice it.

    ⚠️ This exact defect shipped into the first draft: ``ring()`` built its
    corners from :attr:`bounds`, which already returns absolute coordinates,
    and then offset them by the centre again. A 2.8 mm slot centred at y = 46
    came out at y = 90 -- above the top of a 100 mm card -- and the board
    outline the Gerber writer produced measured 143 mm tall.
    """
    item = CutoutElement(key="slot", x_mm=1.5, y_mm=-8.0, width_mm=20.0, height_mm=3.0)
    ring = item.ring()
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert (min(xs) + max(xs)) / 2.0 == pytest.approx(1.5)
    assert (min(ys) + max(ys)) / 2.0 == pytest.approx(-8.0)
    assert max(xs) - min(xs) == pytest.approx(20.0)
    assert max(ys) - min(ys) == pytest.approx(3.0)


def test_a_cutout_ring_is_wound_counter_clockwise():
    """Opposite to the board outline, so a cutter reads it as a hole.

    A ring wound the same way as the outline is a second island, which a board
    house can read as a loose piece of board inside the card.
    """
    from pcbart.geometry import signed_area

    item = CutoutElement(key="slot", width_mm=20.0, height_mm=3.0)
    assert signed_area(item.ring()) > 0


def test_a_rounded_cutout_ring_stays_inside_its_own_bounds():
    """The rounded variant still centres on the cutout."""
    item = CutoutElement(
        key="slot", x_mm=3.0, y_mm=4.0, width_mm=20.0, height_mm=6.0,
        corner_radius_mm=2.0,
    )
    ring = item.ring()
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert (min(xs) + max(xs)) / 2.0 == pytest.approx(3.0, abs=1e-6)
    assert (min(ys) + max(ys)) / 2.0 == pytest.approx(4.0, abs=1e-6)
    assert min(xs) >= 3.0 - 10.0 - 1e-6
    assert max(xs) <= 3.0 + 10.0 + 1e-6


def test_a_stadium_cutout_has_two_round_ends():
    """The lanyard opening is a routed slot, not a rectangle with a radius."""
    from pcbart.geometry import signed_area

    item = CutoutElement(
        key="slot", width_mm=24.0, height_mm=3.0, shape="obround"
    )
    ring = item.ring()
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert max(xs) - min(xs) == pytest.approx(24.0, abs=0.01)
    assert max(ys) - min(ys) == pytest.approx(3.0, abs=0.01)
    assert signed_area(ring) > 0
    # The top run is straight: many points share the same Y at the caps'
    # tangent, and no point reaches the corner (min x, max y) of a rectangle.
    assert not any(
        x == pytest.approx(min(xs)) and y == pytest.approx(max(ys))
        for x, y in ring
    )


def test_a_round_cutout_is_round():
    from pcbart.geometry import signed_area

    item = CutoutElement(key="hole", width_mm=3.56, height_mm=3.56, shape="circle")
    ring = item.ring()
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert max(xs) - min(xs) == pytest.approx(3.56, abs=0.01)
    assert max(ys) - min(ys) == pytest.approx(3.56, abs=0.01)
    assert signed_area(ring) > 0


def test_a_cutout_corner_radius_larger_than_its_short_side_is_refused():
    item = CutoutElement(width_mm=20.0, height_mm=4.0, corner_radius_mm=3.0)
    assert any("圆角" in problem for problem in item.validate())


def test_a_zero_sized_cutout_is_refused():
    item = CutoutElement(width_mm=0.0, height_mm=3.0)
    assert item.validate()


def test_a_text_click_box_follows_the_rotation():
    """A vertical line's box is vertical.

    A quarter of the template's lines are turned 90 degrees, and a hit box that
    ignored the rotation would lie across the card and steal clicks from the
    artwork.
    """
    upright = TextElement(text="STAFF", x_mm=0.0, y_mm=0.0, size_mm=2.0)
    turned = TextElement(
        text="STAFF", x_mm=0.0, y_mm=0.0, size_mm=2.0, rotation_deg=90.0
    )
    upright_width = max(x for x, _ in upright.corners()) - min(
        x for x, _ in upright.corners()
    )
    turned_height = max(y for _, y in turned.corners()) - min(
        y for _, y in turned.corners()
    )
    assert turned_height == pytest.approx(upright_width, rel=0.05)


def test_text_is_picked_before_the_artwork():
    """The lines are painted on top, so a click on one belongs to the line.

    Taking the artwork first would make the wording unpickable wherever it
    crosses a picture -- which on a card is most of it.
    """
    from pcbart.artboard import ImagePlacement

    board = ArtBoard(
        texts=[TextElement(text="代号", x_mm=0.0, y_mm=0.0, size_mm=4.0)],
        placements=[ImagePlacement(x_mm=0.0, y_mm=0.0, width_mm=40.0, height_mm=40.0)],
    )
    assert board.text_line(1.0, 1.0) is board.texts[0]
    assert board.placement_at(1.0, 1.0) is board.placements[0]


def test_a_hidden_layer_hides_its_text():
    board = ArtBoard(texts=[TextElement(text="A", layer="顶层丝印层")])
    assert board.visible_texts()
    board.layers.visible["top_silk"] = False
    assert board.visible_texts() == []


# --------------------------------------------------------------- template data


def test_the_template_offers_the_substitutions_it_needs():
    """Every substitution the template uses has to have somewhere to go.

    A template with ``{英文名}`` but no ``{编号}`` would leave the number strip
    showing a literal token however carefully the user picked an operator.
    """
    joined = " ".join(item.text for item in badge_card.BADGE_TEXTS)
    assert ENGLISH_TOKEN in joined
    assert NUMBER_TOKEN in joined


def test_headline_text_is_the_english_name_token():
    """The 代号 line carries ``{英文名}``, not the word 代号.

    Writing the label as the content is the obvious mistake here, and it looks
    right until an operator is chosen and the card still says 代号.

    **English, because of the typeface.** This line is set in the template's
    ``Agency FB Negreta``, which has no CJK glyphs at all -- a Chinese name
    renders as empty boxes. ``{代号}`` still exists for any line the user wants
    the Chinese name on.
    """
    headline = [item for item in badge_card.BADGE_TEXTS if item.size_mm == 6.0]
    assert len(headline) == 1
    assert headline[0].text == ENGLISH_TOKEN
    assert headline[0].font == "Agency FB Negreta"


def test_the_invisible_alignment_text_is_not_offered():
    """``ARKNIGHTS-DWDB`` is 0.003 mm in both files.

    It cannot be seen or clicked, so a row for it in the content panel would be
    a control that does nothing when selected.
    """
    assert not [item for item in badge_card.BADGE_TEXTS if "DWDB" in item.text]
    sizes = [item.size_mm for item in badge_card.BADGE_TEXTS]
    assert min(sizes) >= 0.5


def test_every_template_line_has_a_label():
    """The panel lists lines by their purpose; an unlabelled one is unnamed."""
    assert [item.label for item in badge_card.BADGE_TEXTS if not item.label] == []


def test_fresh_text_and_cutouts_are_new_objects():
    """Applying the template twice must not hand out the same objects.

    The frozen specs are module state: sharing live elements would make an edit
    on the first card rewrite the template every later card is copied from.
    """
    first = badge_card.badge_texts()
    second = badge_card.badge_texts()
    assert first[0] is not second[0]
    first[0].text = "changed"
    assert second[0].text != "changed"
    assert badge_card.badge_texts()[0].text != "changed"

    cuts_a = badge_card.badge_cutouts()
    cuts_b = badge_card.badge_cutouts()
    assert cuts_a[0] is not cuts_b[0]


def test_the_card_has_its_three_openings():
    cutouts = badge_card.badge_cutouts()
    assert [item.key for item in cutouts] == ["hole_left", "hole_right", "lanyard"]
    assert [item.shape for item in cutouts] == ["circle", "circle", "obround"]
    slot = next(item for item in cutouts if item.key == "lanyard")
    assert slot.width_mm == pytest.approx(24.130, abs=0.01)
    assert slot.height_mm == pytest.approx(2.832, abs=0.01)
    for item in cutouts:
        if item.shape == "circle":
            assert item.width_mm == pytest.approx(3.56, abs=0.01)
            assert item.height_mm == pytest.approx(item.width_mm)


def test_the_slot_sits_inside_the_card():
    """A slot outside the board is not a slot, it is a second piece of geometry.

    Measured, not eyeballed: the slot is centred 46 mm up on a card that is
    50 mm from its centre to its top edge.
    """
    card_half_height = badge_card.BADGE_HEIGHT_MM / 2.0
    card_half_width = badge_card.BADGE_WIDTH_MM / 2.0
    for item in badge_card.badge_cutouts():
        x0, y0, x1, y1 = item.bounds
        assert -card_half_width <= x0 and x1 <= card_half_width
        assert -card_half_height <= y0 and y1 <= card_half_height


def test_every_text_line_is_anchored_on_the_card():
    """Each line's *anchor* is on the card, within the source file's tolerance.

    ⚠️ Not "the whole line's box is inside": the anchor is ``LEFT_BOTTOM``, so
    a line's glyphs rise *above* it, and ``UID:用户填写`` is anchored 0.27 mm
    below the bottom edge in the user's own template -- which is correct, since
    its baseline sits just off the card and its letters do not. Asserting a
    strict containment would have failed on the template being faithful.

    The tolerance is half a millimetre: enough for that baseline, far too
    little for a line that has been misplaced.
    """
    half_w = badge_card.BADGE_WIDTH_MM / 2.0
    half_h = badge_card.BADGE_HEIGHT_MM / 2.0
    for item in badge_card.BADGE_TEXTS:
        assert -half_w <= item.x_mm <= half_w, f"{item.label} X 出界"
        assert -half_h - 0.5 <= item.y_mm <= half_h, f"{item.label} Y 出界"


def test_the_template_outline_samples_its_corner_arcs():
    from pcbart.geometry import densify_path, signed_area

    spec = badge_card.badge_spec()
    assert len(spec.polygon_raw()) == 12
    sampled = spec.polygon()
    assert len(sampled) > 40, "画布仍然把圆角画成斜线"
    path = spec.outline_path()
    assert path.count("ARC") == 4
    # The first segment must be a real straight edge.  Ending the opening
    # "L" run with an ARC token made the path jump across a corner and moved
    # that corner to the end; the ARC count alone still looked right.
    decoded = densify_path(path)[0]
    assert decoded[0] == pytest.approx(decoded[-1], abs=1e-6)
    assert signed_area(decoded) == pytest.approx(signed_area(sampled), rel=1e-6)


def test_the_fixed_decoration_data_is_present():
    """The canvas and the writers draw from one generated payload."""
    from pcbart.badge_decorations import PAYLOAD

    assert PAYLOAD["version"] == 2
    assert PAYLOAD["source"].endswith(".epro2")
    # The top-level copy is the short card: every reader that predates the two
    # layouts keeps working and gets the card it always got.
    items = PAYLOAD["items"]
    assert len(items) > 300
    assert {item["kind"] for item in items} == {"fill", "stroke"}
    assert {item["layer"] for item in items} >= {1, 3, 4, 5}


def test_both_card_layouts_are_extracted():
    """⚠️ The 精0 and 精2 cards are not the same layout.

    The short card carries the 阵营 logo block and its own hatching; the tall
    one replaces that with a full-height illustration window. Shipping only one
    meant the 精2 card could not exist at all.
    """
    from pcbart.badge_decorations import PAYLOAD

    variants = PAYLOAD["variants"]
    assert set(variants) == {"short", "tall"}
    for key, variant in variants.items():
        assert variant["source"].endswith(".epro2"), key
        assert len(variant["items"]) > 100, key
        assert variant["records"], key
    # They really are different artwork, not one list copied twice.
    assert len(variants["short"]["items"]) != len(variants["tall"]["items"])


def test_a_tall_card_uses_the_tall_layout():
    """The board picks the layout through its own template name."""
    from pcbart import badge_art, badge_card
    from pcbart.artboard import ArtBoard

    short = ArtBoard(outline=badge_card.badge_spec("精0"), outline_ready=True)
    tall = ArtBoard(outline=badge_card.badge_spec("精2"), outline_ready=True)

    assert badge_art.variant_for(short) == "short"
    assert badge_art.variant_for(tall) == "tall"
    # Both are the card, and each draws its own artwork.
    assert badge_art.is_badge(short) and badge_art.is_badge(tall)
    assert len(badge_art.decorations_for(short)) != len(
        badge_art.decorations_for(tall)
    )


def test_an_old_draft_without_a_stage_is_the_short_card():
    """Documents written before the stages existed must keep their layout."""
    from pcbart import badge_art, badge_card
    from pcbart.artboard import ArtBoard, BoardOutlineSpec, OutlineShape

    legacy = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.TEMPLATE, template="通行证", width_mm=50, height_mm=100
        ),
        outline_ready=True,
    )
    assert badge_card.stage_of("通行证") == "精0"
    assert badge_art.variant_for(legacy) == "short"
    assert badge_art.is_badge(legacy)


def test_the_decoration_payload_keeps_the_artwork_field_and_side_text():
    """The card's own background and lettering are not filtered away.

    An earlier pass treated any large rectangle as a replaceable placeholder
    and removed the artwork field; another treated an IMAGE's subpaths as one
    outer ring plus holes and reduced the vertical RHODES ISLAND lettering to
    oval shells.  Both are visible on the canvas, so they are asserted here.
    """
    from pcbart.badge_decorations import PAYLOAD
    from pcbart.geometry import signed_area

    fills = [item for item in PAYLOAD["items"] if item["kind"] == "fill"]
    artwork = [
        item
        for item in fills
        if item["layer"] == 1
        and any(
            abs((max(x for x, _ in ring) - min(x for x, _ in ring)) - 38.86) < 0.2
            and abs((max(y for _, y in ring) - min(y for _, y in ring)) - 56.32) < 0.2
            for ring in item["rings"]
        )
    ]
    assert artwork, "立绘背景区没有进入固定装饰"

    strip = [
        item
        for item in fills
        if item["layer"] == 3
        and item["rings"]
        and -28 < sum(x for ring in item["rings"] for x, _ in ring) / sum(
            len(ring) for ring in item["rings"]
        ) < -14
        and -10 < sum(y for ring in item["rings"] for _, y in ring) / sum(
            len(ring) for ring in item["rings"]
        ) < 50
    ]
    assert len(strip) >= 10, f"竖排 RHODES ISLAND 只提取到 {len(strip)} 个字形"
    assert any(len(item["rings"]) > 1 for item in strip), "字形的内孔丢了"
    assert any(
        signed_area(ring) > 0
        for item in strip
        for ring in item["rings"][1:]
    ), "字形没有按复杂多边形分组"


def test_the_decorations_are_painted_bottom_side_first():
    """The canvas's flat draw list has to impose the layer stack itself."""
    from pcbart import badge_art

    ordered = badge_art.ordered_decorations()
    keys = [
        badge_art.layer_key_for_id(int(item["layer"])) for item in ordered
    ]
    rank = {key: index for index, key in enumerate(badge_art._LAYER_ORDER)}
    ranks = [rank.get(key, 99) for key in keys]
    assert ranks == sorted(ranks)


def test_the_template_shape_is_accepted_by_validation():
    """A card has to be exportable, and both writers validate before writing.

    ⚠️ ``OutlineShape.TEMPLATE`` was missing from the list of known shapes, so
    every 通行证 board failed validation with "未知的板框外形: template": it
    could be designed on the canvas and never saved to a file.
    """
    spec = badge_card.badge_spec()
    assert spec.validate() == []


def test_a_template_spec_with_no_ring_is_refused():
    """The other half of the same check: template without geometry is empty."""
    from pcbart.artboard import BoardOutlineSpec, OutlineShape

    spec = BoardOutlineSpec(shape=OutlineShape.TEMPLATE, template="通行证")
    assert any("轮廓点" in problem for problem in spec.validate())


# ----------------------------------------------------- against the real files


@pytest.mark.skipif(not template_files(), reason="模板源文件不在本机素材库")
def test_the_generated_decoration_module_matches_the_source():
    """The checked-in payload is reproducible from the user's template."""
    from pcbart.badge_decorations import PAYLOAD
    from tools.extract_badge_art import SOURCE_NAME, build_items

    source = next(
        (path for path in template_files() if path.name == SOURCE_NAME), None
    )
    if source is None:
        pytest.skip(f"模板源文件里没有 {SOURCE_NAME}")
    items = build_items(source)
    assert len(items) == len(PAYLOAD["items"])
    # The generated module round-trips through JSON, so tuples become lists;
    # compare the JSON form rather than the in-memory tuple form.
    import json

    assert json.loads(json.dumps(items)) == PAYLOAD["items"]
    from tools.extract_badge_art import build_records

    records = build_records(source)
    assert json.loads(json.dumps(records)) == PAYLOAD["records"]
    from tools.extract_badge_art import build_outline_records

    outline = build_outline_records(source)
    assert json.loads(json.dumps(outline)) == PAYLOAD["outline_records"]


@pytest.mark.skipif(not template_files(), reason="模板源文件不在本机素材库")
def test_the_two_template_files_place_the_same_text():
    """The shipped table is one card's worth of text, so both files must agree.

    Twelve of the fourteen lines appear in both files at positions within
    0.21 mm. If a future export moves one further than this, the single shared
    table stops being right for both -- and a card built from the other file
    would have its wording in the wrong place. Reported as the measured
    distance so the number to reconcile is visible.
    """
    from pcbart.epru import EproArchive
    from pcbart.units import MM_PER_MIL
    from tools.extract_badge_regions import extract_boards

    measured: list[dict[tuple[str, float], tuple[float, float, float]]] = []
    for path in template_files():
        document = EproArchive.read(path).document
        centre = extract_boards(path)[0].centre
        found: dict[tuple[str, float], tuple[float, float, float]] = {}
        for record in document.iter_type("STRING"):
            body = record.body
            if body.get("text") == "ARKNIGHTS-DWDB":
                continue
            size = body["fontSize"] * MM_PER_MIL
            # Keyed by wording *and* size, because ``ARKNIGHTS-编号`` appears
            # twice -- once horizontal, once down the vertical strip -- at two
            # sizes. Keyed by text alone the second overwrites the first, and
            # the comparison then reports a 22 mm disagreement that is really
            # two different lines being confused with each other.
            found[(body["text"], round(size, 2))] = (
                body["x"] * MM_PER_MIL - centre[0],
                body["y"] * MM_PER_MIL - centre[1],
                size,
            )
        measured.append(found)

    shared = set(measured[0]) & set(measured[1])
    assert len(shared) >= 10, f"两个模板文件共有的文字只有 {len(shared)} 行"
    worst = 0.0
    worst_text = ""
    for key in shared:
        a, b = measured[0][key], measured[1][key]
        distance = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        if distance > worst:
            worst, worst_text = distance, key[0]
    assert worst <= 0.25, f"「{worst_text}」在两个文件里差了 {worst:.3f} mm"


@pytest.mark.skipif(not template_files(), reason="模板源文件不在本机素材库")
def test_the_shipped_table_matches_the_source_files():
    """Every shipped line has a counterpart in the file, at the same place.

    This is the check that catches a typo in the table. It compares position and
    size only -- the *wording* is deliberately not compared, because the two
    placeholders (``{代号}``/``{编号}``) are a change this tool made on purpose
    and would fail a literal text comparison.
    """
    from pcbart.epru import EproArchive
    from pcbart.units import MM_PER_MIL
    from tools.extract_badge_regions import extract_boards

    path = template_files()[0]
    document = EproArchive.read(path).document
    centre = extract_boards(path)[0].centre
    found: list[tuple[float, float, float]] = []
    for record in document.iter_type("STRING"):
        body = record.body
        if body.get("text") == "ARKNIGHTS-DWDB":
            continue
        found.append(
            (
                body["x"] * MM_PER_MIL - centre[0],
                body["y"] * MM_PER_MIL - centre[1],
                body["fontSize"] * MM_PER_MIL,
            )
        )

    for item in badge_card.BADGE_TEXTS:
        nearest = min(
            found,
            key=lambda entry: max(
                abs(entry[0] - item.x_mm), abs(entry[1] - item.y_mm)
            ),
            default=None,
        )
        assert nearest is not None, "模板文件里一行文字都没有"
        assert abs(nearest[0] - item.x_mm) <= 0.25, f"{item.label} 的 X 对不上"
        assert abs(nearest[1] - item.y_mm) <= 0.25, f"{item.label} 的 Y 对不上"
        # The two placeholder lines were reworded, and 编号 appears twice with
        # different sizes, so size is compared rather than text.
        assert nearest[2] == pytest.approx(item.size_mm, abs=0.02), item.label


@pytest.mark.skipif(not template_files(), reason="模板源文件不在本机素材库")
def test_the_shipped_cutouts_match_the_source_files():
    """The three openings' size and position come from the file, not a guess."""
    from pcbart.epru import EproArchive
    from pcbart.units import MM_PER_MIL
    from tools.extract_badge_regions import extract_boards

    path = template_files()[0]
    document = EproArchive.read(path).document
    centre = extract_boards(path)[0].centre
    boxes = []
    for record in document.iter_type("FILL"):
        body = record.body
        if body.get("layerId") != 12:
            continue
        for segment in body.get("path") or []:
            if isinstance(segment, list) and segment and segment[0] == "R":
                x, y, w, h = (float(value) * MM_PER_MIL for value in segment[1:5])
                boxes.append((x + w / 2 - centre[0], y - h / 2 - centre[1], w, h))
    assert len(boxes) == 1, f"MULTI 层上的填充有 {len(boxes)} 个，预期 1 个"
    cx, cy, width, height = boxes[0]
    slot = next(item for item in badge_card.badge_cutouts() if item.key == "lanyard")
    assert slot.width_mm == pytest.approx(width, abs=0.01)
    assert slot.height_mm == pytest.approx(height, abs=0.01)
    assert slot.x_mm == pytest.approx(cx, abs=0.01)
    assert slot.y_mm == pytest.approx(cy, abs=0.01)

    vias = [
        (
            body["centerX"] * MM_PER_MIL - centre[0],
            body["centerY"] * MM_PER_MIL - centre[1],
            body["holeDiameter"] * MM_PER_MIL,
        )
        for record in document.iter_type("VIA")
        if isinstance((body := record.body), dict)
    ]
    assert len(vias) == 2, f"模板源文件里的圆孔有 {len(vias)} 个，预期 2 个"
    round_holes = [
        item for item in badge_card.badge_cutouts() if item.shape == "circle"
    ]
    for hole in round_holes:
        match = min(
            vias,
            key=lambda entry: abs(entry[0] - hole.x_mm) + abs(entry[1] - hole.y_mm),
        )
        assert hole.x_mm == pytest.approx(match[0], abs=0.01)
        assert hole.y_mm == pytest.approx(match[1], abs=0.01)
        assert hole.width_mm == pytest.approx(match[2], abs=0.01)


# ------------------------------------------------------------------- exports


def badge_board(name: str = "鸿雪", number: str = "R145") -> ArtBoard:
    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    board.operator_name = name
    board.operator_number = number
    return board


def test_the_written_project_keeps_the_wording_editable(tmp_path):
    """A ``STRING`` record, not traced geometry.

    The point of the user's request is that the wording can be changed. Text
    exported as outlines would reach the editor as a shape, and correcting a
    letter would mean redrawing it.
    """
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    report = export_design(badge_board(), path, title="card")
    assert report["texts"] == len(badge_card.BADGE_TEXTS)
    assert report["cutouts"] == 3

    document = EproArchive.read(path).document
    strings = [record.body for record in document.iter_type("STRING")]
    assert len(strings) == len(badge_card.BADGE_TEXTS)
    texts = {item["text"] for item in strings}
    # The 代号 line carries the **English** name: the template sets it in
    # ``Agency FB Negreta``, which cannot draw 鸿雪 at all.
    assert "Позёмка" in texts
    assert "ARKNIGHTS-R145" in texts


def test_the_written_project_has_no_unresolved_tokens(tmp_path):
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    export_design(badge_board(), path, title="card")
    document = EproArchive.read(path).document
    for record in document.iter_type("STRING"):
        assert "{代号}" not in record.body["text"]
        assert "{编号}" not in record.body["text"]


def test_exporting_does_not_rewrite_the_documents_own_tokens(tmp_path):
    """The document keeps its placeholders; only the file gets values.

    Substituting into the document would make the choice irreversible -- the
    next operator would have to find the previous one's name inside the string.
    """
    from pcbart.export import export_design

    board = badge_board()
    export_design(board, tmp_path / "card.epro2", title="card")
    joined = " ".join(item.text for item in board.texts)
    assert ENGLISH_TOKEN in joined
    assert NUMBER_TOKEN in joined


def test_the_written_project_keeps_the_cutout(tmp_path):
    """On the MULTI layer, the way the source templates encode it."""
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    export_design(badge_board(), path, title="card")
    document = EproArchive.read(path).document
    multi = [
        record.body
        for record in document.iter_type("FILL")
        if record.body.get("layerId") == 12
    ]
    assert len(multi) == 3
    # The long opening keeps a stadium outline; the two hanging holes are
    # round.  The count alone would pass if all three were rectangles.
    from pcbart.geometry import path_to_rings
    from pcbart.primitives import convert_path_to_mm

    dimensions = []
    for body in multi:
        points = [
            point
            for ring in path_to_rings(convert_path_to_mm(body.get("path")))
            for point in ring
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        dimensions.append((max(xs) - min(xs), max(ys) - min(ys)))
    dimensions.sort()
    assert dimensions[0][0] == pytest.approx(dimensions[0][1], abs=0.1)


def test_the_gerber_outline_carries_the_slot(tmp_path):
    """A drawn-but-unwritten slot comes back as a solid card.

    The slot is a second closed contour on the outline layer, so a router cuts
    it -- not a copper region and not a separate file.
    """
    from pcbart.gerber import export_gerber
    from gerbonara.rs274x import GerberFile

    report = export_gerber(badge_board(), tmp_path, title="card")
    outline = next(
        entry for entry in report["files"] if entry["label"] == "板框"
    )
    parsed = GerberFile.open(str(outline["path"]))
    objects = list(parsed.objects)
    # One board contour plus three openings.  The board and the openings are
    # stroked, so the parser returns one object per segment; the number of
    # starts (D02) is what says how many closed contours the router sees.
    text = Path(str(outline["path"])).read_text(encoding="ascii")
    assert text.count("D02*") == 4
    assert len(objects) >= 4
    (x0, y0), (x1, y1) = parsed.bounding_box()
    assert x1 - x0 == pytest.approx(100.0, abs=0.2) or y1 - y0 == pytest.approx(
        100.0, abs=0.2
    )
    # The card is 100 mm tall; a doubled centre produced a 143 mm file.
    assert max(y1 - y0, x1 - x0) < 101.0


def test_the_gerber_silkscreen_carries_the_text(tmp_path):
    """Text on silk has to reach the fab, or the card prints blank."""
    from pcbart.gerber import export_gerber
    from gerbonara.rs274x import GerberFile

    report = export_gerber(badge_board(), tmp_path, title="card")
    silk = next(
        entry for entry in report["files"] if entry["label"] == "顶层丝印"
    )
    parsed = GerberFile.open(str(silk["path"]))
    assert len(list(parsed.objects)) > 20


def test_a_font_that_cannot_draw_the_text_is_passed_over():
    """⚠️ The defect this pins turned the operator's name into two empty boxes.

    The template asks for ``Agency FB`` for the 代号 line. That font is
    installed on this machine and draws Latin perfectly, but it has no CJK
    glyphs at all -- so ``鸿雪`` came out as two ``.notdef`` rectangles. Every
    numeric check passed: the mask was not blank, the geometry traced fine, the
    file was valid. Only looking at the rendered card showed it.

    So a font is chosen for *coverage*, not for its name.
    """
    from pcbart import text_render

    latin = text_render.font_path("Agency FB Negreta", "ARKNIGHTS-R145")
    cjk = text_render.font_path("Agency FB Negreta", "鸿雪")
    assert cjk is not None, "没有任何字体能画出中文"
    assert cjk != latin, "中文仍然用了画不出它的字体"

    rendered = text_render.render_text_mask(
        "鸿雪", size_mm=6.0, family="Agency FB Negreta"
    )
    assert rendered is not None
    # A pair of notdef boxes is mostly empty space inside two rectangles; real
    # glyphs fill far more of their bounding box. Measuring the ink is what
    # distinguishes "rendered" from "rendered a placeholder".
    import numpy as np

    ink = float((rendered.mask < 128).mean())
    assert ink > 0.2, f"字形墨量只有 {ink:.2f}，像是画成了空方框"


def test_the_family_is_still_used_when_it_can_draw_the_text():
    """The fallback must not throw away a font that works.

    The card's Latin lines are meant to be in the template's own faces; giving
    every line the CJK fallback would make the whole card look different from
    the design for no reason.
    """
    from pcbart import text_render

    chosen = text_render.font_path("Agency FB Negreta", "STAFF")
    assert chosen is not None
    assert chosen.name.casefold() in {"agencyr.ttf", "agencyb.ttf"}


def test_a_text_line_with_no_font_does_not_silently_vanish(tmp_path, monkeypatch):
    """Reported, not dropped.

    A line that exports as nothing looks identical to a line the user deleted,
    so the report has to say which one it was.
    """
    from pcbart import text_render
    from pcbart.export import build_copper

    # The signature carries the text too, because the picker has to check that
    # the font can draw it -- see ``test_a_font_that_cannot_draw_the_text_is_
    # passed_over``.
    monkeypatch.setattr(text_render, "font_path", lambda family="", text="": None)
    board = badge_board()
    _records, report = build_copper(board)
    assert any("没有可导出的字形" in problem for problem in report["note"].split("跳过：")[-1].split("；"))


def test_the_written_outline_keeps_its_rounded_corners(tmp_path):
    """The four corners are native ARC segments, not diagonal chords.

    A sampled polygon would still span 50 x 100 mm, so a size assertion cannot
    catch the loss; the ARC token is what proves the editor gets a round corner.
    """
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    export_design(badge_board(), path, title="card")
    document = EproArchive.read(path).document
    outlines = [
        record.body
        for record in document.iter_type("POLY")
        if record.body.get("layerId") == 11
    ]
    assert len(outlines) == 10
    tokens = [
        token
        for outline in outlines
        for token in outline["path"]
        if isinstance(token, str)
    ]
    assert tokens.count("ARC") == 4


def test_the_written_project_keeps_the_template_decoration(tmp_path):
    """The fixed artwork travels with the card, not only the wording.

    The source template's borders, logos and patterns are ordinary FILL/POLY
    primitives.  A file with the right outline and text but no decoration
    looks empty in the editor, which is the defect this guards.
    """
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    export_design(badge_board(), path, title="card")
    document = EproArchive.read(path).document
    for layer in (1, 3, 4, 5):
        fills = [
            record.body
            for record in document.iter_type("FILL")
            if record.body.get("layerId") == layer
        ]
        assert fills, f"第 {layer} 层没有任何固定装饰"
        assert all(body.get("width") == 0.2 for body in fills), (
            "装饰 FILL 的 width 必须用源文件的原始单位值"
        )
    assert any(
        record.type == "POLY"
        for record in document.iter_type("POLY")
        if record.body.get("layerId") == 3
    ), "顶层丝印的描边装饰没有写进工程"


def test_the_written_project_does_not_trace_text_twice(tmp_path):
    """``.epro2`` gets STRING records only; the traced copy stays in Gerber.

    The user saw the same wording twice, one editable and one a filled region
    slightly out of place.  Both encodings were being written at once.
    """
    from pcbart.epru import EproArchive
    from pcbart.export import build_copper, export_design

    board = badge_board()
    traced, _report = build_copper(board)
    assert traced, "Gerber 仍然需要文字描边"
    without_text, _report = build_copper(board, include_text=False)
    assert without_text == []

    path = tmp_path / "card.epro2"
    export_design(board, path, title="card")
    document = EproArchive.read(path).document
    assert len(list(document.iter_type("STRING"))) == len(badge_card.BADGE_TEXTS)


def test_the_written_project_keeps_native_image_records(tmp_path):
    """The fixed IMAGE records stay IMAGE, with their own transform fields.

    Decoding them into filled outlines and re-encoding them put every logo and
    the vertical RHODES ISLAND strip in the wrong place in the editor.  The
    writer now copies the source records and only translates their anchor.

    **Eight of them, the 罗德岛 crest included.** The crest is not a placeholder:
    every card carries it on the top silkscreen, over the faction slot on the
    copper layer underneath.
    """
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    path = tmp_path / "card.epro2"
    export_design(badge_board(), path, title="card")
    document = EproArchive.read(path).document
    images = list(document.iter_type("IMAGE"))
    assert len(images) == 8
    assert {record.body.get("layerId") for record in images} == {2, 3, 4, 6}
    assert all("path" in record.body for record in images)
    assert all("startX" in record.body for record in images)


def test_the_rhodes_crest_is_on_every_card_unless_removed(tmp_path):
    """The crest is fixed content, not something derived from the operator.

    It sits on the top silkscreen over the faction slot, so a 罗德岛 card would
    otherwise print the same crest twice -- which is why the *copper* slot is
    left empty for them while the silkscreen crest stays.
    """
    from pcbart import badge_art, badge_content
    from pcbart.artboard import ArtBoard
    from pcbart.epru import EproArchive
    from pcbart.export import export_design

    source = next(
        item for item in badge_art.records() if badge_art._is_emblem_record(item)
    )
    assert int(source["body"]["layerId"]) == 3, "罗德岛标志应该在顶层丝印"

    # On by default, for a card with no faction at all.
    board = ArtBoard(
        outline=badge_card.badge_spec(), outline_ready=True
    )
    assert badge_art.emblem_enabled(board)

    path = tmp_path / "card.epro2"
    export_design(board, path, title="card")
    document = EproArchive.read(path).document
    assert any(
        badge_art._is_emblem_record({"type": record.type, "body": record.body})
        for record in document.records
    ), "默认卡面应该有罗德岛标志"

    # And removable, at the user's request.
    board.art_choices = {badge_content.EMBLEM_KIND: badge_content.CHOICE_OFF}
    assert not badge_art.emblem_enabled(board)
    assert not any(
        badge_art.is_rhodes_emblem(item)
        for item in badge_art.decorations_for(board)
    )
    removed = tmp_path / "plain.epro2"
    export_design(board, removed, title="plain")
    plain = EproArchive.read(removed).document
    assert not any(
        badge_art._is_emblem_record({"type": record.type, "body": record.body})
        for record in plain.records
    ), "关掉之后不该再写罗德岛标志"


def test_the_report_counts_the_text_and_the_cutouts(tmp_path):
    from pcbart.export import export_design

    report = export_design(badge_board(), tmp_path / "card.epro2", title="card")
    assert report["texts"] == len(badge_card.BADGE_TEXTS)
    assert report["cutouts"] == 3
    assert report["decorations"] > 100
    assert "固定装饰" in report["note"]


@pytest.mark.skipif(not template_files(), reason="模板源文件不在本机素材库")
def test_an_angle_zero_picture_is_not_drawn_upside_down():
    """The 罗德岛 emblem points the way the author drew it.

    ``v`` is measured from the path's *larger* Y -- the picture's own top edge --
    because the picture's Y axis points down while the document's points up.
    Measuring it from the smaller Y drew every angle-0 picture top-to-bottom
    flipped, so the emblem came out with the castle's merlons at the bottom and
    ``RHODES ISLAND`` across the top.

    The check is on the ink distribution rather than on a coordinate: the
    emblem's words sit in the **bottom** third of its frame and the tapering
    tower fills the middle, so a flipped one puts the densest band at the top.
    """
    import numpy as np

    from pcbart.epru import EproArchive
    from tools.extract_badge_art import _image_rings
    from tools.extract_badge_regions import extract_boards
    from tools.extract_badge_regions import _fill_rect  # noqa: F401 - parity

    source = next(
        (path for path in template_files() if path.name == "模板-精0、精1、皮肤小.epro2"),
        None,
    )
    if source is None:
        pytest.skip("小卡模板不在素材库")

    document = EproArchive.read(source).document
    centre = extract_boards(source)[0].centre
    # IMAGE#7 is the 罗德岛 emblem; it is the last IMAGE in the small template.
    images = list(document.iter_type("IMAGE"))
    emblem = images[-1].body
    assert float(emblem.get("angle", 0) or 0) == 0.0

    rings = _image_rings(emblem, int(emblem["layerId"]))
    points = [point for ring in rings for point in ring]
    ys = [point[1] for point in points]
    top, bottom = max(ys), min(ys)
    span = top - bottom
    # Split the emblem's frame into thirds and count ink in each. The words are
    # a solid bar of glyphs, so the bottom third must carry real ink.
    bottom_third = [
        point for point in points if point[1] < bottom + span / 3.0
    ]
    assert bottom_third, "标志下半部完全没有几何——多半是上下颠倒了"
    # And the outline's widest part (the wings) belongs to the middle, not the
    # very bottom edge; the letters are what reaches the bottom.
    assert len(bottom_third) > len(points) * 0.10


# ------------------------------------------------------- english name line

def test_the_name_line_prints_the_english_name():
    """The 干员代号 line is Latin-only, so it carries the English name.

    Measured: the template's ``Agency FB Negreta`` has no CJK glyphs, so 阿米娅
    came out as two empty boxes. The English name is both what the design
    intends and what makes the template's own typeface usable.
    """
    item = next(t for t in badge_card.badge_texts() if t.label == "干员代号")
    assert item.resolved_text(name="阿米娅", number="R001") == "Amiya"
    assert item.resolved_text(name="暴行", number="RB01") == "Savage"
    assert item.font == "Agency FB Negreta"


def test_the_chinese_name_is_still_available_as_a_token():
    """``{代号}`` keeps working, so a user can put it on another line.

    Two tokens rather than making ``{代号}`` mean English: switching a line back
    is an edit in the panel, not a code change.
    """
    line = TextElement(text=f"{NAME_TOKEN} / {ENGLISH_TOKEN}")
    assert line.resolved_text(name="阿米娅") == "阿米娅 / Amiya"


def test_a_name_with_no_published_english_form_does_not_invent_one():
    """Three cases, three honest answers -- never a made-up transliteration.

    * a Latin display name is already the official one (``Pith``);
    * a Latin run inside a Chinese name is what the game itself spells in Latin
      (``Mechanist(卫戍协议)`` -> ``Mechanist``);
    * nothing Latin at all falls back to the Chinese name (``博士``), because a
      transliteration would put an unlookupable name on the card.
    """
    from pcbart.operator_names import english_name

    assert english_name("Pith") == "Pith"
    assert english_name("Mechanist(卫戍协议)") == "Mechanist"
    assert english_name("领主·Sharp") == "Sharp"
    assert english_name("博士") == "博士"
    assert english_name("预备干员-狙击") == "预备干员-狙击"


def test_every_roster_name_resolves_to_something_printable():
    """No card may end up with an empty 代号 line.

    The line is the card's headline; blanking it is worse than any fallback.
    """
    from pcbart import operators
    from pcbart.operator_names import english_name

    database = operators.database()
    if not database:
        pytest.skip("干员素材库未安装")
    empty = [
        person.name for person in database.operators if not english_name(person.name)
    ]
    assert not empty, empty[:10]


def test_the_gerber_mask_layers_carry_the_cards_openings(tmp_path):
    """Independent read-back of the fab files, with a third-party parser.

    ⚠️ The Gerber path used to write only the placements and the wording, so a
    card's **solder-mask layers came out empty** and its border, wordmark and
    crest were absent entirely. Nothing failed -- the set validated and
    uploaded. This reads the written files back with ``gerbonara`` and measures
    them, so the check does not share code with the writer.
    """
    from pcbart.gerber import export_gerber
    from gerbonara.rs274x import GerberFile

    report = export_gerber(badge_board(), tmp_path, title="card")
    written = [Path(str(entry["path"])) for entry in report["files"]]

    def layer_file(extension: str) -> Path:
        """The written file with this extension.

        Matched on the extension rather than the report's label: the report
        names layers the way the *fab* does, which is not what the design calls
        them, and a test that reads the wrong table proves nothing.
        """
        found = [path for path in written if path.name.endswith(extension)]
        assert found, f"没有写出 {extension}：{[p.name for p in written]}"
        return found[0]

    # Every layer of the card has a file. The two masks are the ones that used
    # to come out empty.
    for extension in (".GTL", ".GBL", ".GTO", ".GBO", ".GTS", ".GBS"):
        layer_file(extension)

    # Each mask holds real geometry.
    for extension, label in ((".GTS", "顶层阻焊开窗"), (".GBS", "底层阻焊开窗")):
        parsed = GerberFile.open(str(layer_file(extension)))
        assert list(parsed.objects), f"{label} 是空的"
        (x0, y0), (x1, y1) = parsed.bounding_box()
        assert x1 > x0 and y1 > y0, f"{label} 没有实际范围"

    # The illustration opening is the tallest thing on the top mask -- the
    # region is 56.333 mm high -- so a file that carries it is at least that
    # tall. One holding only a stray rule would be far shorter.
    (x0, y0), (x1, y1) = GerberFile.open(
        str(layer_file(".GTS"))
    ).bounding_box()
    assert (y1 - y0) > 50.0, (
        f"顶层阻焊开窗只有 {(y1 - y0):.3f} mm 高，立绘窗没进去"
    )


def test_the_gerber_border_is_the_card_size(tmp_path):
    """Read back with ``gerbonara``: the cut path is the card, centred.

    ``gerbonara`` reports the extent the *material* covers, so a 0.125 mm
    cutter adds half its width on each side: 50.125 rather than 50.000. The
    centreline is what the router follows and it is the card's own size -- see
    ``test_the_board_outline_stays_the_card_size``, which reads the raw
    coordinates. Both are asserted, from two different readings, because a
    board that loses 0.05 mm of border per edge is exactly the sort of error
    that only shows up after the boards arrive.
    """
    from pcbart.gerber import export_gerber
    from gerbonara.rs274x import GerberFile

    report = export_gerber(badge_board(), tmp_path, title="card")
    outline = next(
        entry for entry in report["files"] if entry["label"] == "板框"
    )
    parsed = GerberFile.open(str(outline["path"]))
    (x0, y0), (x1, y1) = parsed.bounding_box()
    kerf = 0.125
    assert x1 - x0 == pytest.approx(50.0 + kerf, abs=0.05)
    assert y1 - y0 == pytest.approx(100.0 + kerf, abs=0.05)
    # Centred on the origin, which is where the design frame puts it.
    assert (x0 + x1) / 2 == pytest.approx(0.0, abs=0.05)
    assert (y0 + y1) / 2 == pytest.approx(0.0, abs=0.05)


# ------------------------------------------------------- the keepout regions

def test_the_prohibit_regions_are_not_drawn_as_artwork():
    """⚠️ The template's two PROHIBIT regions must not become copper blocks.

    Both source files carry ``REGION`` records on the top and bottom copper,
    7.24 x 56.13 mm, just left of the illustration window, with
    ``regionType=PROHIBIT`` and ``prohibitType=[TRACK, COMPONENT, COPPER,
    FILL, PLANE]``. They are *keepouts* -- a constraint that says "keep copper
    out of here" -- but the extractor treated every ``REGION`` as a ``FILL``,
    so they were drawn as two solid blocks of copper sitting beside the
    illustration, and written that way into both the .epro2 and the Gerber.

    The rule belongs in the source template, where a designer needs it. On a
    finished card there is nothing left for it to constrain.
    """
    from pcbart import badge_art

    # No record of that kind survives the extraction at all.
    kinds = {item.get("type") for item in badge_art.records()}
    assert "REGION" not in kinds, f"禁止区域仍被当成图元写入：{sorted(kinds)}"

    # And nothing in the drawn decorations has the block's own size.
    for index, item in enumerate(badge_art.decorations()):
        for ring in item.get("rings") or ():
            if not ring:
                continue
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            assert not (
                abs(width - 7.239) < 0.1 and abs(height - 56.133) < 0.1
            ), f"装饰 #{index} 是那块禁止区域"


def test_the_keepout_is_not_in_either_writer(tmp_path):
    """The block must be absent from the canvas list *and* from both exports.

    The two writers read different data -- the .epro2 replays raw records, the
    Gerber reads the drawn decorations -- so a fix on one path alone would
    leave the block on the board through the other.
    """
    from pcbart import badge_art, gerber
    from pcbart.artboard import ArtBoard
    from pcbart.badge_card import badge_spec, badge_texts

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.texts = badge_texts()

    # .epro2 side: the raw records the writer replays.
    assert all(
        item.get("type") != "REGION" for item in badge_art.records_for(board)
    )

    # Gerber side: read the copper file back and measure every filled region.
    report = gerber.export_gerber(board, tmp_path, title="audit", archive=False)
    copper = next(
        entry for entry in report["files"] if entry["label"] == "顶层铜"
    )
    blocks = _gerber_region_extents(Path(str(copper["path"])))
    assert blocks, "顶层铜里没有任何填充区域"
    for width, height in blocks:
        assert not (
            abs(width - 7.239) < 0.05 and abs(height - 56.133) < 0.05
        ), f"禁止区域仍然写进了顶层铜：{width:.3f} × {height:.3f} mm"


def _gerber_region_extents(path: Path) -> list[tuple[float, float]]:
    """``(width, height)`` of every filled region in a Gerber file.

    Parsed from the file's own ``G36``/``G37`` blocks rather than from the
    model, so the assertion is about what a fab would receive.
    """
    import re

    extents: list[tuple[float, float]] = []
    inside = False
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("G36"):
            inside, xs, ys = True, [], []
            continue
        if line.startswith("G37"):
            if xs and ys:
                extents.append((max(xs) - min(xs), max(ys) - min(ys)))
            inside = False
            continue
        if not inside:
            continue
        match = re.search(r"X(-?\d+)Y(-?\d+)D0[12]\*", line)
        if match:
            xs.append(int(match.group(1)) / 1e6)
            ys.append(int(match.group(2)) / 1e6)
    return extents


def test_the_tall_card_carries_no_faction_logo():
    """精2 has no faction slot, so it must not get a faction logo.

    The tall card's illustration window is 42.526 x 99.626 mm and covers the
    slot completely -- measured: the two rectangles overlap over 512.98 mm²,
    exactly 100% of the slot. The source template agrees: the tall layout has
    neither the marker block nor the ``阵营logo`` wording.

    Returning a placement anyway put the logo across the artwork of every 精2
    card, which is a printing defect rather than a cosmetic one: it lands on
    the picture, on the wrong layer, in a spot the layout never reserved.
    """
    from pcbart import badge_content
    from pcbart.artboard import ArtBoard
    from pcbart import operators

    database = operators.database()
    if not database:
        pytest.skip("素材库不在")

    # 银灰 has a faction, so the short card proves the logo is available at all
    # -- without that, the tall-card assertion could pass for the wrong reason.
    person = database.find("银灰")
    assert person is not None and person.faction

    def art(stage: str):
        board = ArtBoard(outline=badge_card.badge_spec(stage), outline_ready=True)
        board.operator_name = person.name
        board.operator_number = person.number
        board.operator_page = person.page
        return badge_content.badge_art(board=board, number=person.number)

    assert art("精0").faction is not None, "小卡应该有阵营 logo"
    assert art("精2").faction is None, "大卡不该有阵营 logo"

    # And the other three elements are unaffected by the rule.
    for stage in ("精0", "精2"):
        built = art(stage)
        assert built.qr is not None
        assert built.barcode is not None
        assert built.profession is not None


def test_the_faction_slot_overlaps_the_tall_illustration_window():
    """The measurement the rule above is based on, kept as a live assertion.

    If someone resizes either region, this fails and points at the reason the
    tall card has no faction logo, rather than letting the overlap creep back.
    """
    from shapely.geometry import box

    from pcbart import badge_card, badge_content

    slot = badge_content.art_region("faction_logo")
    slot_box = box(
        slot.centre_x_mm - slot.width_mm / 2,
        slot.centre_y_mm - slot.height_mm / 2,
        slot.centre_x_mm + slot.width_mm / 2,
        slot.centre_y_mm + slot.height_mm / 2,
    )

    windows = {region.key: region for region in badge_card.BADGE_REGIONS}
    small = windows["small"]
    large = windows["large"]

    small_box = box(
        small.centre_x_mm - small.width_mm / 2,
        small.centre_y_mm - small.height_mm / 2,
        small.centre_x_mm + small.width_mm / 2,
        small.centre_y_mm + small.height_mm / 2,
    )
    large_box = box(
        large.centre_x_mm - large.width_mm / 2,
        large.centre_y_mm - large.height_mm / 2,
        large.centre_x_mm + large.width_mm / 2,
        large.centre_y_mm + large.height_mm / 2,
    )

    assert slot_box.intersection(small_box).area == pytest.approx(0.0, abs=1e-6)
    assert slot_box.intersection(large_box).area == pytest.approx(
        slot_box.area, rel=1e-3
    )
