"""Tests for the 通行证 card and framing artwork into its regions.

The card is the answer to a problem the automatic cropper could not solve. Five
rounds of face detection, saliency, vehicle rules and a hand-tuned table all
left pictures the user had to correct by hand, because "where should the
subject sit" is a judgement the artwork does not contain. The card makes it the
user's call instead: the tool draws the card, marks where artwork goes, and the
user slides the picture until it looks right. The crop happens on export.

What these tests hold:

* the card's geometry is the measured geometry, not a plausible-looking
  redrawing;
* framing puts the picture in the region and locks it to the region's shape, so
  nothing is ever stretched;
* panning moves the picture *inside* its window while the object stays put --
  the opposite of what a drag does everywhere else, which is the one behaviour
  that would be easy to ship backwards;
* the export cuts to the window, and does so at the size the user saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from pcbart import badge_card  # noqa: E402
from pcbart.artboard import ArtBoard, ImagePlacement  # noqa: E402
from pcbart.badge_card import (  # noqa: E402
    BADGE_HEIGHT_MM,
    BADGE_OUTLINE,
    BADGE_REGIONS,
    BADGE_WIDTH_MM,
    badge_spec,
    region_for_aspect,
)
from pcbart.export import build_copper  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def mark_picture(tmp_path: Path, *, quadrant=(0, 0, 400, 400)) -> Path:
    """A white picture with one black patch, to see what a crop kept."""
    data = np.full((800, 800), 255, dtype=np.uint8)
    x0, y0, x1, y1 = quadrant
    data[y0:y1, x0:x1] = 0
    path = tmp_path / "mark.png"
    Image.fromarray(data).save(path)
    return path


# ------------------------------------------------------------------ the card

def test_the_card_is_the_measured_size():
    """50 x 100 mm, from the template's own dimension annotations."""
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    width, height = board.size_mm()
    assert width == pytest.approx(BADGE_WIDTH_MM, abs=0.01)
    assert height == pytest.approx(BADGE_HEIGHT_MM, abs=0.01)


def test_the_outline_is_the_measured_shape():
    """Twelve points: four cut corners and the notch on the right edge."""
    assert len(BADGE_OUTLINE) == 12
    xs = [x for x, _ in BADGE_OUTLINE]
    ys = [y for _, y in BADGE_OUTLINE]
    assert max(xs) - min(xs) == pytest.approx(BADGE_WIDTH_MM, abs=0.01)
    assert max(ys) - min(ys) == pytest.approx(BADGE_HEIGHT_MM, abs=0.01)


def test_the_card_has_the_notch_the_presets_cannot_express():
    """Two points share the far-right column and sit well below the top.

    That pair is the notch: the outline steps in from x = 24.9 to x = 23.1,
    which is why a template shape was added rather than reusing a preset.
    """
    right = max(x for x, _ in BADGE_OUTLINE)
    stepped = [
        x for x, _ in BADGE_OUTLINE if abs(x - right) < 2.0 and x < right - 0.5
    ]
    assert stepped, "轮廓里找不到缺口"


def test_the_outline_is_a_closed_ring():
    """Walking the points in order traces the card and returns to the start.

    The one long step is the left-hand edge, which is the card's full height --
    the outline stores corners, not a point every millimetre. Any *other* long
    step would mean two corners are out of order, which draws a bow tie.
    """
    ring = list(BADGE_OUTLINE)
    gaps = [
        ((ring[(i + 1) % len(ring)][0] - point[0]) ** 2
         + (ring[(i + 1) % len(ring)][1] - point[1]) ** 2) ** 0.5
        for i, point in enumerate(ring)
    ]
    long_steps = [gap for gap in gaps if gap > 55.0]
    assert len(long_steps) == 1, f"意外的长边：{long_steps}"
    # Shorter than the card's height: the corners are cut, so the left edge's
    # flat run ends before the corners do.
    assert long_steps[0] > 0.9 * BADGE_HEIGHT_MM


def test_the_card_registers_under_its_name():
    badge_card.register()
    from pcbart.artboard import build_template, template_names

    assert badge_card.TEMPLATE_NAME in template_names()
    assert build_template(badge_card.TEMPLATE_NAME).shape == "template"


# ---------------------------------------------------------------- the regions

def test_both_regions_sit_inside_the_card():
    """Artwork cannot be framed outside the board it is printed on."""
    for region in BADGE_REGIONS:
        x0, y0, x1, y1 = region.bounds
        assert x0 >= -BADGE_WIDTH_MM / 2 - 1e-6
        assert x1 <= BADGE_WIDTH_MM / 2 + 1e-6
        assert y0 >= -BADGE_HEIGHT_MM / 2 - 1e-6
        assert y1 <= BADGE_HEIGHT_MM / 2 + 1e-6


def test_the_regions_are_the_measured_ones():
    small, large = BADGE_REGIONS
    assert small.width_mm == pytest.approx(38.861, abs=0.001)
    assert small.height_mm == pytest.approx(56.324, abs=0.001)
    assert large.width_mm == pytest.approx(42.526, abs=0.001)
    assert large.height_mm == pytest.approx(99.626, abs=0.001)


def test_the_two_regions_differ_in_shape():
    """Which is why a picture suited to one is not suited to the other."""
    small, large = BADGE_REGIONS
    assert small.aspect > large.aspect * 1.5


def test_a_region_can_be_found_from_an_aspect():
    """The figure library is cropped to these two ratios, so it can be matched."""
    for region in BADGE_REGIONS:
        assert region_for_aspect(region.aspect) is region
    assert region_for_aspect(1.4) is None


# ------------------------------------------------------------------- framing

@pytest.fixture
def canvas(app, tmp_path):
    from tools.design_canvas import DesignCanvas

    widget = DesignCanvas()
    widget.resize(900, 900)
    yield widget
    widget.deleteLater()


def framed(canvas, tmp_path, region_index=0):
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(
        path=str(mark_picture(tmp_path)), width_mm=40.0, height_mm=40.0
    )
    board.placements.append(placement)
    canvas.board = board
    # The canvas is built with a history for the board it started with, so a
    # test that swaps the document has to swap the history with it -- which is
    # what the window does on a tab switch.
    from pcbart.undo import UndoStack

    canvas.set_history(UndoStack(board))
    canvas.start_framing(placement, BADGE_REGIONS[region_index])
    return placement, BADGE_REGIONS[region_index]


def test_framing_moves_the_object_onto_the_region(canvas, tmp_path):
    placement, region = framed(canvas, tmp_path)
    assert placement.x_mm == pytest.approx(region.centre_x_mm)
    assert placement.y_mm == pytest.approx(region.centre_y_mm)
    assert placement.width_mm == pytest.approx(region.width_mm)
    assert placement.height_mm == pytest.approx(region.height_mm)


def test_the_property_pane_is_told_after_the_region_is_applied(canvas, tmp_path):
    """The pane reads the placement when it hears the signal, so the geometry
    has to be in place *before* the signal goes out.

    Emitting first left the panel quoting the size and position from before the
    region was applied -- 76.20 mm on screen while the object was 38.86 mm on
    the canvas -- until some later edit happened to refresh it.
    """
    seen: list[tuple[float, float, float, float]] = []
    canvas.selection_changed.connect(
        lambda item: seen.append(
            (item.x_mm, item.y_mm, item.width_mm, item.height_mm)
        )
    )
    placement, region = framed(canvas, tmp_path)
    assert seen, "选中的信号没有发出去"
    assert seen[-1] == pytest.approx(
        (region.centre_x_mm, region.centre_y_mm, region.width_mm, region.height_mm)
    )


def test_framing_locks_the_picture_to_the_region_shape(canvas, tmp_path):
    """A stretched picture is never what the user meant.

    The window is reshaped to the region's aspect, and it is done as a *crop*:
    the largest window of that shape that fits inside the picture, so no part
    of it samples past the edge.
    """
    placement, region = framed(canvas, tmp_path)
    window = placement.source_window
    assert window is not None
    left, top, right, bottom = window
    assert 0.0 <= left < right <= 1.0
    assert 0.0 <= top < bottom <= 1.0
    assert placement.source_aspect_for(800, 800) == pytest.approx(
        region.aspect, rel=0.01
    )


def test_framing_does_not_stretch_a_square_picture(canvas, tmp_path):
    """The square source keeps its own pixels; the window takes the hit."""
    placement, region = framed(canvas, tmp_path)
    assert placement.width_mm / placement.height_mm == pytest.approx(
        region.aspect, rel=0.001
    )
    window = placement.source_window
    width = (window[2] - window[0]) * 800
    height = (window[3] - window[1]) * 800
    assert width / height == pytest.approx(region.aspect, rel=0.01)


def _painted(canvas, x_mm, y_mm) -> QColor:
    """The canvas pixel at a point given in board millimetres."""
    image = canvas.grab().toImage()
    ratio = canvas.devicePixelRatio()
    point = canvas.to_screen(x_mm, y_mm)
    x = int(round(point.x() * ratio))
    y = int(round(point.y() * ratio))
    x = max(0, min(image.width() - 1, x))
    y = max(0, min(image.height() - 1, y))
    return QColor(image.pixel(x, y))


def plain_picture(tmp_path: Path) -> Path:
    """A uniformly white picture, so "how bright is it" is the whole reading.

    A picture with a dark patch would make the samples below depend on which
    corner of the drawing the window happens to cover, which is a second
    variable this test has no business measuring.
    """
    path = tmp_path / "plain.png"
    Image.fromarray(np.full((800, 800), 255, dtype=np.uint8)).save(path)
    return path


def test_the_picture_outside_the_window_stays_visible_but_dimmed(
    canvas, tmp_path, app
):
    """Framing shows the whole picture: the window at full strength, the rest faint.

    Everything outside used to be clipped away, which left nothing on screen to
    judge a pan against -- the picture slid past a window with no visible
    context, so there was no way to tell how much further it could move.
    """
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(path=str(plain_picture(tmp_path)))
    board.placements.append(placement)
    canvas.board = board
    from pcbart.undo import UndoStack

    canvas.set_history(UndoStack(board))
    region = BADGE_REGIONS[0]
    canvas.start_framing(placement, region)
    # A window over the left part of the picture: the picture continues to the
    # right of the region, which is the part that has to stay visible.
    width_px, height_px = 800, 800
    window_width = region.aspect * height_px / width_px
    placement.source_window = (0.0, 0.0, window_width, 1.0)
    canvas.fit_view()
    app.processEvents()

    inside_y = region.centre_y_mm
    outside_x = region.centre_x_mm + region.width_mm * 0.5 + 2.5
    # The same point, with and without the framing session. Comparing the two
    # renders rather than one absolute reading keeps the assertion independent
    # of whatever template decoration or grid line happens to sit there: the
    # only difference between the two frames is the ghost.
    inside = _painted(canvas, region.centre_x_mm, inside_y)
    with_ghost = _painted(canvas, outside_x, inside_y)
    canvas.framing = None
    app.processEvents()
    without_ghost = _painted(canvas, outside_x, inside_y)

    assert inside.red() > 230, f"窗口内应为原亮度，实测 {inside.name()}"
    assert with_ghost.red() > without_ghost.red() + 20, (
        f"框外应画出暗影，实测 取景={with_ghost.name()} 非取景={without_ghost.name()}"
    )
    assert with_ghost.red() < inside.red() * 0.6, (
        f"框外应当比窗口内明显更暗，实测 框内={inside.name()} 框外={with_ghost.name()}"
    )


def test_the_ghost_never_covers_another_object(canvas, tmp_path, app):
    """The codes and the crest are separate objects and must stay readable.

    The ghost covers a large area -- it is the whole picture, not the window --
    so painting it in the same pass as its own placement would wash a faint
    layer of illustration over whatever the card carries nearby. It goes down
    before every object instead.
    """
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    picture = ImagePlacement(path=str(plain_picture(tmp_path)))
    # A small black patch on the bottom silk, sitting **inside the card but
    # outside the illustration's window** -- so the only thing that can reach it
    # is the ghost, which is the whole picture and therefore wider than the
    # window. It is added first, so a ghost drawn in the same pass as its own
    # placement would land on top of it.
    code = ImagePlacement(
        path=str(mark_picture(tmp_path)), width_mm=4.0, height_mm=4.0
    )
    code.layer = "底层丝印层"
    code.x_mm = -22.0
    code.y_mm = 20.0
    board.placements.extend([code, picture])
    canvas.board = board
    from pcbart.undo import UndoStack

    canvas.set_history(UndoStack(board))
    region = BADGE_REGIONS[1]
    canvas.start_framing(picture, region)
    # Centred, so the ghost reaches out on both sides of the window.
    width = region.aspect
    picture.source_window = ((1 - width) / 2.0, 0.0, (1 + width) / 2.0, 1.0)
    canvas.fit_view()
    app.processEvents()

    point = (code.x_mm - 1.0, code.y_mm + 1.0)  # the patch's own black corner
    with_ghost = _painted(canvas, *point)
    canvas.framing = None
    app.processEvents()
    without_ghost = _painted(canvas, *point)
    assert with_ghost == without_ghost, (
        f"取景暗影盖到了别的对象上：取景={with_ghost.name()} "
        f"非取景={without_ghost.name()}"
    )


def test_both_tones_of_a_painting_are_framed_together(canvas, tmp_path, app):
    """A 板画 is one picture in two placements, and the exporter pairs them.

    The window has to land on both tones: cropping them at different offsets
    makes the finished board tear along their seam.
    """
    from pcbart.undo import UndoStack

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    white = ImagePlacement(
        path=str(plain_picture(tmp_path)), tone="white", import_id="painting"
    )
    black = ImagePlacement(
        path=str(plain_picture(tmp_path)), tone="black", import_id="painting"
    )
    board.placements.extend([white, black])
    canvas.board = board
    canvas.set_history(UndoStack(board))
    canvas.start_framing(white, BADGE_REGIONS[0])
    assert black.source_window == white.source_window

    canvas.window_zoom(1.7)
    assert black.source_window == white.source_window

    canvas._press_pos = QPointF(0.0, 0.0)
    canvas._press_window = white.source_window
    canvas._pan_framed(QPointF(-25.0, -12.0))
    assert black.source_window == white.source_window

    # The box moves with the crop too, or the pair would land apart.
    assert (black.x_mm, black.y_mm, black.width_mm, black.height_mm) == (
        white.x_mm,
        white.y_mm,
        white.width_mm,
        white.height_mm,
    )


def test_framing_does_not_draw_a_ghost_when_the_whole_picture_is_used(
    canvas, tmp_path, app
):
    """With no crop there is nothing outside, so nothing is painted twice."""
    placement, region = framed(canvas, tmp_path)
    placement.source_window = None
    canvas.fit_view()
    app.processEvents()

    centre = _painted(canvas, region.centre_x_mm, region.centre_y_mm)
    assert centre.red() > 40 or centre.green() > 40, (
        f"整图落图时应当照常显示，实测 {centre.name()}"
    )


def test_framing_is_undoable(canvas, tmp_path):
    """Ctrl+Z after framing puts the object back where it was."""
    placement, _region = framed(canvas, tmp_path)
    placed_window = placement.source_window
    assert placed_window is not None
    assert canvas.can_undo()
    canvas.undo()
    # Undone, the object is no longer sitting on the region at its shape.
    assert (placement.width_mm, placement.height_mm) != pytest.approx(
        (BADGE_REGIONS[0].width_mm, BADGE_REGIONS[0].height_mm)
    )


def test_panning_slides_the_picture_inside_its_window(canvas, tmp_path):
    """The object must not move -- that is what makes this framing, not placing."""
    placement, _region = framed(canvas, tmp_path)
    object_before = (placement.x_mm, placement.y_mm)
    canvas.window_zoom(1.6)
    before = placement.source_window

    canvas._press_pos = QPointF(0.0, 0.0)
    canvas._press_window = before
    canvas.framing = placement
    canvas._pan_framed(QPointF(-40.0, 0.0))

    assert placement.x_mm == object_before[0]
    assert placement.y_mm == object_before[1]
    assert placement.source_window != before


def test_dragging_right_shows_an_earlier_part_of_the_picture(canvas, tmp_path):
    """Pictures follow the pointer, so the window travels the other way."""
    placement, _region = framed(canvas, tmp_path)
    canvas.window_zoom(1.6)
    before = placement.source_window
    canvas._press_pos = QPointF(0.0, 0.0)
    canvas._press_window = before
    canvas.framing = placement
    canvas._pan_framed(QPointF(30.0, 0.0))
    after = placement.source_window
    assert after[0] < before[0]


def test_a_pan_never_leaves_the_picture(canvas, tmp_path):
    """Sliding past the edge would show blank paper where the art ends."""
    placement, _region = framed(canvas, tmp_path)
    canvas.window_zoom(2.0)
    canvas._press_pos = QPointF(0.0, 0.0)
    canvas._press_window = placement.source_window
    canvas.framing = placement
    for _ in range(20):
        canvas._press_window = placement.source_window
        canvas._pan_framed(QPointF(500.0, 500.0))
    window = placement.source_window
    assert 0.0 <= window[0] and window[2] <= 1.0
    assert 0.0 <= window[1] and window[3] <= 1.0


def test_zooming_in_shows_less_of_the_picture(canvas, tmp_path):
    placement, _region = framed(canvas, tmp_path)
    before = placement.source_window
    canvas.window_zoom(2.0)
    after = placement.source_window
    assert (after[2] - after[0]) < (before[2] - before[0])
    assert (after[3] - after[1]) < (before[3] - before[1])


def test_a_pan_is_one_undo_step(canvas, tmp_path):
    """A drag emits a motion event per frame; Ctrl+Z must not walk each one.

    Zooming and panning back to back are also one step rather than two, which
    is what a user does in a single sitting and expects to take back in one
    press.
    """
    placement, _region = framed(canvas, tmp_path)
    depth_before = canvas.history.depth
    canvas._press_pos = QPointF(0.0, 0.0)
    canvas.framing = placement
    for step in range(6):
        canvas._press_window = placement.source_window
        canvas._pan_framed(QPointF(-6.0, -2.0))
        # What mouseReleaseEvent does at the end of the gesture.
        canvas._record("frame")
        assert canvas.history.depth <= depth_before + 1, f"第 {step} 步就多记了一条"
    assert canvas.history.depth > depth_before, "整段拖拽一步都没记"


def test_the_regions_are_offered_only_on_the_card(canvas, tmp_path):
    """No other board has a region to fill, so the entries must not appear."""
    canvas.board = ArtBoard()
    assert canvas._region_choices() == []

    canvas.board = ArtBoard(outline=badge_spec(), outline_ready=True)
    assert len(canvas._region_choices()) == len(BADGE_REGIONS)


def test_the_object_menu_offers_the_regions_on_the_card(canvas, tmp_path):
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(path="", width_mm=10.0, height_mm=10.0)
    board.placements.append(placement)
    canvas.board = board
    canvas.selected = placement

    menu = QMenu()
    canvas._fill_object_menu(menu)
    labels = [action.text() for action in menu.actions()]
    assert any("放入" in label for label in labels)


def test_the_object_menu_has_no_region_entries_on_a_plain_board(canvas, tmp_path):
    board = ArtBoard()
    placement = ImagePlacement(path="", width_mm=10.0, height_mm=10.0)
    board.placements.append(placement)
    canvas.board = board
    canvas.selected = placement

    menu = QMenu()
    canvas._fill_object_menu(menu)
    assert not any("放入" in action.text() for action in menu.actions())


# -------------------------------------------------------------------- export

def test_export_keeps_only_the_framed_part(tmp_path):
    """The point of the whole feature: what is outside the window is cut."""
    source = mark_picture(tmp_path, quadrant=(100, 100, 400, 400))
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    # Traced path on purpose -- this asserts how the *vectorised* window is
    # cropped. An unprocessed picture on silk is carried as a colour OBJ now.
    placement = ImagePlacement(
        path=str(source),
        processed_path=str(source),
        x_mm=BADGE_REGIONS[0].centre_x_mm,
        y_mm=BADGE_REGIONS[0].centre_y_mm,
        width_mm=BADGE_REGIONS[0].width_mm,
        height_mm=BADGE_REGIONS[0].height_mm,
        layer="顶层丝印层",
    )
    board.placements.append(placement)

    placement.source_window = (0.0, 0.0, 0.35, 0.35)  # covers the mark
    inside, _ = build_copper(board)
    placement.source_window = (0.6, 0.6, 1.0, 1.0)  # blank corner
    outside, _ = build_copper(board)

    assert inside, "窗口包含方块时应当导出图形"
    assert not outside, "窗口排除方块时不应导出任何图形"


def test_export_size_does_not_follow_the_crop(tmp_path):
    """Framing changes *what* is shown, not how big the artwork is.

    Measuring the scale against the whole picture instead of the window would
    shrink the export by however much was cropped away -- frame a quarter of
    the image and the board would come out four times too small.
    """
    from pcbart.artboard import drawn_size_mm

    source = mark_picture(tmp_path)
    placement = ImagePlacement(path=str(source), width_mm=38.861, height_mm=56.324)
    for window in (None, (0.0, 0.0, 0.5, 0.5), (0.25, 0.1, 0.9, 0.8)):
        placement.source_window = window
        aspect = placement.source_aspect_for(800, 800)
        drawn = drawn_size_mm(placement.width_mm, placement.height_mm, aspect)
        assert drawn[0] <= placement.width_mm + 1e-6
        assert drawn[1] <= placement.height_mm + 1e-6


def test_a_placement_without_a_window_is_unchanged(tmp_path):
    """Every document written before this feature keeps its old behaviour."""
    placement = ImagePlacement(path=str(mark_picture(tmp_path)))
    assert placement.source_window is None
    assert placement.source_rect(800, 800) == (0, 0, 800, 800)


def test_the_window_survives_a_draft_round_trip(tmp_path):
    from pcbart.draft import board_from_dict, board_to_dict

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.placements.append(
        ImagePlacement(path="x.png", source_window=(0.2, 0.1, 0.8, 0.9))
    )
    restored = board_from_dict(board_to_dict(board))
    assert tuple(restored.placements[0].source_window) == (0.2, 0.1, 0.8, 0.9)


def test_the_template_outline_survives_a_draft_round_trip(tmp_path):
    from pcbart.draft import board_from_dict, board_to_dict

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    restored = board_from_dict(board_to_dict(board))
    assert tuple(restored.outline.template_outline) == BADGE_OUTLINE
    assert restored.outline.template == badge_card.TEMPLATE_NAME


def test_a_framed_illustration_cannot_leave_its_region(tmp_path):
    """⚠️ The region is an export-time clip, not just a canvas guide.

    The user's requirement was that the picture outside the card's opening is
    cut. It was only drawn on the canvas, so the two normal edits that follow
    framing -- resizing and moving in the property pane -- pushed ink outside
    it. Measured before the fix: frame, scale 1.4x, and the ink reached
    x = 23.1 mm where the window ends at 21.3 mm.
    """
    from pcbart.artboard import ArtBoard
    from pcbart.badge_card import badge_spec
    from pcbart.design_layers import layer
    from pcbart.export import build_copper

    region = BADGE_REGIONS[0]
    # ``processed_path`` keeps this on the *traced* path on purpose: the region
    # clip is a rule about traced ink, and an unprocessed picture on silk is now
    # carried as a colour OBJ instead (see tests/test_color_export.py).
    source = str(mark_picture(tmp_path))
    placement = ImagePlacement(
        path=source,
        processed_path=source,
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        width_mm=region.width_mm,
        height_mm=region.height_mm,
        layer="顶层丝印层",
        region=region.key,
        source_window=(0.2, 0.2, 0.8, 0.8),
    )

    def ink_bounds():
        board = ArtBoard(outline=badge_spec(), outline_ready=True)
        board.placements.append(placement)
        records, _report = build_copper(board)
        wanted = layer(placement.layer_key).easyeda_id
        points = [
            point
            for layer_id, rings, _width in records
            if layer_id == wanted
            for ring in rings
            for point in ring
        ]
        assert points, "立绘没有导出任何墨迹"
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    x0, y0, x1, y1 = region.bounds

    def assert_inside(what: str):
        lo_x, lo_y, hi_x, hi_y = ink_bounds()
        assert lo_x >= x0 - 1e-6, f"{what}：左边超出到 {lo_x:.3f}（区域 {x0:.3f}）"
        assert hi_x <= x1 + 1e-6, f"{what}：右边超出到 {hi_x:.3f}（区域 {x1:.3f}）"
        assert lo_y >= y0 - 1e-6, f"{what}：下边超出到 {lo_y:.3f}（区域 {y0:.3f}）"
        assert hi_y <= y1 + 1e-6, f"{what}：上边超出到 {hi_y:.3f}（区域 {y1:.3f}）"

    assert_inside("刚取景完")

    # Both edits are normal things to do after framing, and both used to leak.
    placement.width_mm *= 1.4
    placement.height_mm *= 1.4
    assert_inside("放大 1.4 倍之后")

    placement.x_mm += 8.0
    placement.y_mm -= 5.0
    assert_inside("再移动之后")


def test_an_unframed_picture_is_not_clipped(tmp_path):
    """A placement with no region keeps the old behaviour exactly.

    Every board that is not the card, and every import that was never framed,
    has an empty ``region``. Clipping those would silently crop ordinary
    artwork to a card window it was never placed in.
    """
    from pcbart.artboard import ArtBoard
    from pcbart.badge_card import badge_spec
    from pcbart.design_layers import layer
    from pcbart.export import build_copper

    # A picture with something in it: an all-white one has no ink to trace,
    # which would make the measurement below vacuous.
    source = str(mark_picture(tmp_path, quadrant=(0, 0, 800, 800)))
    placement = ImagePlacement(
        path=source,
        processed_path=source,
        width_mm=40.0,
        height_mm=40.0,
        layer="顶层丝印层",
    )
    assert placement.region == ""
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.placements.append(placement)
    records, _report = build_copper(board)
    wanted = layer(placement.layer_key).easyeda_id
    points = [
        point
        for layer_id, rings, _width in records
        if layer_id == wanted
        for ring in rings
        for point in ring
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    # The picture is 40 mm square, so the ink must span about that -- a card
    # region would have cut it to 38.861 x 56.324.
    assert max(xs) - min(xs) == pytest.approx(40.0, abs=0.5)
    assert max(ys) - min(ys) == pytest.approx(40.0, abs=0.5)


def test_framing_records_which_region_was_used(canvas, tmp_path):
    """The canvas has to store the region, or the export has nothing to clip to."""
    placement, region = framed(canvas, tmp_path)
    assert placement.region == region.key


def test_the_region_survives_a_draft_round_trip(tmp_path):
    from pcbart.draft import board_from_dict, board_to_dict

    placement = ImagePlacement(path="x.png", region="large")
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    board.placements.append(placement)
    restored = board_from_dict(board_to_dict(board))
    assert restored.placements[0].region == "large"


# ------------------------------------------ adjusting a crop you did not choose
#
# ⚠️ **The bug these exist for.** The automatic import places a picture in the
# card's opening without starting the framing gesture -- correctly, since the
# user asked to import a picture, not to frame one. But the object stayed
# movable while the export clipped it to the opening, so the only obvious
# gesture, a drag, moved the picture *out of* the window being printed. Measured
# before the fix: one short drag took a 精0 illustration from ``x=1.854`` to
# ``x=7.281`` and the crop window did not change at all. The user's report was
# 「只能获得默认的裁切范围，没办法改」.


def auto_framed(canvas, tmp_path, region_index=0):
    """A picture put in an opening the way an automatic import does it.

    No ``start_framing``: that is the whole point. The geometry is applied and
    the canvas is left with the object merely *selected*, which is the state the
    interface actually produces.
    """
    from pcbart.undo import UndoStack

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(
        path=str(mark_picture(tmp_path)), width_mm=40.0, height_mm=40.0
    )
    board.placements.append(placement)
    canvas.board = board
    canvas.set_history(UndoStack(board))
    canvas.frame_into_card_region(placement)
    canvas.selected = placement
    return placement, BADGE_REGIONS[region_index]


def test_a_plain_drag_moves_a_framed_picture_freely(canvas, tmp_path, app):
    """⚠️ The object moves, and the opening only decides what is kept.

    This reverses the earlier rule, at the user's request: 「我希望立绘能移动的
    范围不要局限于这个框……我们只保留立绘区的图案」. The reason is on the card
    itself -- there is a cutout near the top, so a hard-locked position can leave
    the user unable to keep the whole figure. The opening is a *crop*, not a
    leash.

    ⚠️ **The press handler is what decides this**, so ``_pan_framed`` alone
    cannot be asserted here: that method is only reached once ``mousePressEvent``
    has chosen the framing mode.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    placement, _region = auto_framed(canvas, tmp_path)
    assert canvas.framing is None, "前提检查：自动导入不应当进入取景模式"
    # Without a live view scale ``to_screen`` is degenerate and the press reads
    # as a resize-handle grab instead of the picture.
    canvas.fit_view()
    app.processEvents()

    before_position = (placement.x_mm, placement.y_mm)
    before_window = placement.source_window
    centre = canvas.to_screen(placement.x_mm, placement.y_mm)
    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonPress, centre, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert canvas._mode == "move", (
        f"普通拖动应当移动图片，实际 mode={canvas._mode!r}"
    )
    canvas.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.MouseMove,
            QPointF(centre.x() - 40, centre.y() - 25),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )

    assert (placement.x_mm, placement.y_mm) != before_position, (
        "图片应当跟着指针走"
    )
    assert placement.source_window == before_window, (
        "自由移动不该改取景窗口：立绘区裁的是哪一块，与图片放在哪无关"
    )


def test_a_picture_can_be_moved_right_out_of_its_opening(canvas, tmp_path):
    """No leash: the picture is allowed to leave the opening entirely.

    The user's reason is concrete -- the card has a cutout near the top, so the
    figure sometimes has to sit well above the opening's centre. Clamping the
    position to the opening would make that impossible.

    What happens on export is a separate question, and it is handled: the ink is
    clipped to the opening, and a picture that has left it completely is marked
    and reported rather than silently dropped.
    """
    placement, region = auto_framed(canvas, tmp_path)
    x0, y0, x1, y1 = region.bounds

    # Push it far enough that the box no longer touches the opening at all.
    placement.x_mm = x1 + placement.width_mm + 5.0
    placement.y_mm = (y0 + y1) / 2.0

    assert placement.overlaps_rect(region.bounds) is False
    # And the geometry really is outside: the nearest corner is past the edge.
    assert min(x for x, _y in placement.corners()) > x1


def test_a_mouse_drag_reads_as_framing_only_during_an_explicit_session(
    canvas, tmp_path, app
):
    """The crop-pan gesture lives where the user asks for it.

    ⚠️ Right-click → 放入短区/长区 is what starts a session. Outside one, a drag
    moves the picture; the earlier version panned the crop on *every* framed
    picture, which is exactly the behaviour the user asked to be rid of.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent

    placement, region = auto_framed(canvas, tmp_path)
    canvas.fit_view()
    app.processEvents()

    # Inside a session: the crop pans and the object stays.
    canvas.start_framing(placement, region)
    center = canvas.to_screen(placement.x_mm, placement.y_mm)
    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonPress, center, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert canvas._mode == "frame"
    object_before = (placement.x_mm, placement.y_mm)
    window_before = placement.source_window
    canvas.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.MouseMove,
            QPointF(center.x() - 30, center.y()),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert (placement.x_mm, placement.y_mm) == object_before
    assert placement.source_window != window_before

    # Leaving the session restores the ordinary drag.
    canvas.stop_framing()
    assert canvas._is_framing_session(placement) is False


def test_control_wheel_zooms_the_crop_and_leaves_the_view_alone(canvas, tmp_path, app):
    """Ctrl+wheel had to be added: nothing in the interface called the zoom.

    ``window_zoom`` existed from the start but only scripts and tests reached
    it, so an auto-framed picture could be panned but never zoomed -- half the
    adjustment the automatic crop needs.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QWheelEvent

    placement, _region = auto_framed(canvas, tmp_path)
    before = placement.source_window
    zoom_before = canvas._zoom

    centre = canvas.to_screen(placement.x_mm, placement.y_mm)
    canvas.wheelEvent(
        QWheelEvent(
            centre, centre, QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
            Qt.ControlModifier, Qt.NoScrollPhase, False,
        )
    )

    after = placement.source_window
    assert (after[2] - after[0]) < (before[2] - before[0]), "Ctrl+滚轮应当放大取景"
    assert canvas._zoom == zoom_before, "Ctrl+滚轮不应当同时缩放视图"


def test_a_plain_wheel_still_zooms_the_view(canvas, tmp_path):
    """The view zoom the user asked for stays on the plain wheel.

    取景 shares the wheel only under Ctrl. Moving the view zoom off the plain
    wheel would break the gesture for every board that is not a card.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QWheelEvent

    placement, _region = auto_framed(canvas, tmp_path)
    before = placement.source_window
    zoom_before = canvas._zoom

    centre = canvas.to_screen(placement.x_mm, placement.y_mm)
    canvas.wheelEvent(
        QWheelEvent(
            centre, centre, QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
            Qt.NoModifier, Qt.NoScrollPhase, False,
        )
    )

    assert canvas._zoom != zoom_before, "普通滚轮仍然缩放视图"
    assert placement.source_window == before, "普通滚轮不该动取景"


def test_resetting_the_crop_restores_the_opening_s_fit(canvas, tmp_path):
    """The escape hatch for a pan taken too far.

    ⚠️ It has to be separate from ``frame_into_card_region``, which
    *deliberately* keeps an existing window so a re-import cannot throw away a
    user's crop. Reset is the request to discard it.
    """
    placement, region = auto_framed(canvas, tmp_path)
    default = placement.source_window

    canvas.window_zoom(2.0)
    canvas._press_pos = QPointF(0.0, 0.0)
    canvas._press_window = placement.source_window
    canvas._pan_framed(QPointF(-30.0, -10.0))
    assert placement.source_window != default, "前提检查：取景确实被改过"

    assert canvas.reset_framing(placement) is True
    assert placement.source_window == default, "重置后应当回到默认取景"
    assert placement.width_mm == pytest.approx(region.width_mm, abs=1e-6)
    assert placement.region == region.key


def test_the_context_menu_offers_the_crop_controls_for_an_auto_framed_picture(
    canvas, tmp_path, app
):
    """The controls have to be reachable, not merely implemented.

    A gesture the user cannot discover is the same as a missing feature, and
    ⚠️ the entries used to be keyed on an explicit framing session -- which an
    automatic import never enters, so the menu offered nothing.
    """
    placement, _region = auto_framed(canvas, tmp_path)
    assert canvas.framing is None

    menu = QMenu(canvas)
    canvas._fill_object_menu(menu)
    labels = [action.text() for action in menu.actions() if action.text()]

    assert "重置取景" in labels, f"右键菜单里应当能重置取景，实际 {labels}"
    assert any("放大取景" in label for label in labels), labels
    assert any("缩小取景" in label for label in labels), labels


def test_an_unframed_picture_is_still_moved_by_a_drag(canvas, tmp_path):
    """The other half of the rule: no opening, no crop, ordinary dragging.

    Every ordinary board depends on this, and making *all* drags pan a crop
    would have broken placement for the majority case.
    """
    from pcbart.undo import UndoStack

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(path=str(mark_picture(tmp_path)))
    board.placements.append(placement)
    canvas.board = board
    canvas.set_history(UndoStack(board))

    assert canvas._is_framing_session(placement) is False
    assert canvas._region_bounds(placement) is None, "没有立绘区，就没有裁剪"
    assert canvas._region_path(placement) is None, "没有立绘区，就不裁剪绘制"
    assert placement.region == ""


# ------------------------------------------------- the crop cannot be squashed
#
# ⚠️ **The reported defect these exist for.** 「放入短区 → 缩小图片至完全小于
# 立绘区 → 再放大就无法突破上下界限」. ``_clamp_window`` clamped width and
# height *independently*, so once the height reached the picture's edge the width
# carried on growing: a 精0 crop went from aspect 0.690 to 1.000 and never came
# back, because every later zoom scaled the square window. A square crop drawn
# into the tall opening is letterboxed by ``drawn_size_mm``, which is the blank
# band the user saw above and below the figure. Measured before the fix:
#
#   刚放入短区   w=0.690 h=1.000 aspect=0.6900  want=0.6900  OK
#   缩小 x1      w=0.862 h=1.000 aspect=0.8624  want=0.6900  ***
#   缩小 x2      w=1.000 h=1.000 aspect=1.0000  want=0.6900  ***
#   放大 x7      w=0.210 h=0.210 aspect=1.0000  want=0.6900  ***
#
# The region constants themselves were **re-measured with
# ``tools/extract_badge_regions.py`` against the source template and are
# unchanged** (38.861 × 56.324 at +1.854/+21.825), so the fault was never in the
# region definition.


def _fraction_aspect(canvas, placement):
    """The crop's shape in *picture fraction* space, which is what matters.

    A window is a fraction of the picture, so a fraction of a non-square picture
    is not the shape it looks like -- the pixel aspect has to come out before the
    comparison means anything.
    """
    from pcbart.image_size import picture_size

    width_px, height_px = picture_size(canvas._image_path_for(placement))
    x0, y0, x1, y1 = placement.source_window
    return (x1 - x0) / (y1 - y0), width_px, height_px


def test_zooming_out_cannot_squash_the_crop(canvas, tmp_path):
    """⚠️ The reported repro, at the level of the window's shape.

    Zooming out until the window cannot grow any further must stop, not stretch.
    """
    placement, region = auto_framed(canvas, tmp_path)
    aspect, width_px, height_px = _fraction_aspect(canvas, placement)
    want = region.aspect * height_px / width_px
    assert aspect == pytest.approx(want, rel=0.01), "前提检查：初始形状正确"

    for step in range(1, 9):
        canvas.window_zoom(0.8)
        aspect, _w, _h = _fraction_aspect(canvas, placement)
        assert aspect == pytest.approx(want, rel=0.01), (
            f"第 {step} 次缩小后形状被拉成 {aspect:.4f}，应当是 {want:.4f}"
        )


def test_zooming_back_in_still_reaches_the_ends(canvas, tmp_path):
    """⚠️ The user's exact symptom: 「再放大就无法突破上下界限」.

    Once the window has been pushed to the picture's edge, zooming in has to
    shrink *both* axes so the crop can go back to filling the opening's height.
    With the old per-axis clamp the height stayed pinned at 1.0 forever.
    """
    placement, region = auto_framed(canvas, tmp_path)
    aspect, width_px, height_px = _fraction_aspect(canvas, placement)
    want = region.aspect * height_px / width_px

    for _ in range(8):
        canvas.window_zoom(0.8)
    x0, y0, x1, y1 = placement.source_window
    assert (y1 - y0) == pytest.approx(1.0), "前提检查：确实顶到整幅图的高度"

    for _ in range(6):
        canvas.window_zoom(1.25)

    window = placement.source_window
    aspect, _w, _h = _fraction_aspect(canvas, placement)
    assert aspect == pytest.approx(want, rel=0.01), (
        f"放大后形状应当保持，实际 {aspect:.4f}"
    )
    assert (window[3] - window[1]) < 0.95, (
        f"放大应当能缩小上下范围，实际高度 {window[3] - window[1]:.3f}"
    )


def test_out_and_back_in_returns_to_the_same_crop(canvas, tmp_path):
    """Zooming out then back in is reversible, not a one-way loss.

    The old clamp made it a ratchet: the shape changed on the way out and every
    step afterwards inherited the damage.

    ⚠️ **The walk has to start zoomed in.** The default fit already uses the
    picture's *whole height* (a square picture in a 0.69-aspect opening gives
    ``h = 1.0``), so zooming out from there has nowhere to go and is correctly a
    no-op -- comparing against the default would assert the wrong thing. Zoom in
    first, then walk back out to exactly where it started.
    """
    placement, _region = auto_framed(canvas, tmp_path)

    for _ in range(2):
        canvas.window_zoom(1.25)
    before = placement.source_window

    for _ in range(2):
        canvas.window_zoom(0.8)
    for _ in range(2):
        canvas.window_zoom(1.25)

    after = placement.source_window
    for got, want in zip(after, before):
        assert got == pytest.approx(want, abs=1e-6), (
            f"缩放出去再回来应当回到原窗口：{before} -> {after}"
        )


def test_a_squashed_window_is_repaired_when_the_crop_is_next_edited(canvas, tmp_path):
    """Documents saved by the old clamp repair themselves.

    A window of the wrong shape is never something the user can choose -- every
    path that creates one uses the region's aspect -- so anything else in a card
    opening is the old bug's fingerprint. Any crop edit fixes it, and the
    interface offers 重置取景 for a document that is not edited.
    """
    placement, region = auto_framed(canvas, tmp_path)
    aspect, width_px, height_px = _fraction_aspect(canvas, placement)
    want = region.aspect * height_px / width_px

    # What the old clamp could leave behind.
    placement.source_window = (0.0, 0.0, 1.0, 1.0)
    assert _fraction_aspect(canvas, placement)[0] != pytest.approx(want, rel=0.01)

    canvas.window_zoom(1.0)

    assert _fraction_aspect(canvas, placement)[0] == pytest.approx(want, rel=0.01), (
        "一次取景编辑之后应当回到立绘区的形状"
    )


def test_panning_cannot_squash_the_crop_either(canvas, tmp_path):
    """The other crop gesture goes through the same gate.

    ⚠️ **The fix belongs in one place.** ``_set_window`` is what every crop edit
    funnels through; putting the reshape in each caller instead is exactly how
    the bug got in -- ``window_zoom`` reshaped its window and ``_pan_framed``
    did not. So this test drives the gap the other one misses: it starts from a
    window of the wrong shape (the old bug's fingerprint) and asserts a *pan*
    repairs it, which only holds if the gate is in ``_set_window``.

    A pan that starts from a correct window cannot squash it by itself, so
    without this starting point the assertion would pass even with the gate
    removed.
    """
    placement, region = auto_framed(canvas, tmp_path)
    aspect, width_px, height_px = _fraction_aspect(canvas, placement)
    want = region.aspect * height_px / width_px

    placement.source_window = (0.0, 0.0, 1.0, 1.0)
    assert _fraction_aspect(canvas, placement)[0] != pytest.approx(want, rel=0.01)

    for offset in ((400.0, 0.0), (-900.0, 250.0), (0.0, -800.0)):
        canvas._press_pos = QPointF(0.0, 0.0)
        canvas._press_window = placement.source_window
        canvas._pan_framed(QPointF(*offset))
        aspect, _w, _h = _fraction_aspect(canvas, placement)
        assert aspect == pytest.approx(want, rel=0.01), (
            f"平移后形状被拉成 {aspect:.4f}，应当是 {want:.4f}"
        )


def test_the_crop_fills_the_opening_leaving_no_blank_band(canvas, tmp_path):
    """⚠️ The user's actual screenshot, expressed as geometry.

    ``drawn_size_mm`` fits the cropped pixels into the placement box
    *preserving their aspect*, so a window whose shape differs from its opening
    is letterboxed. Measured on the drawing behind the report:

    ====================  ============  =====================
    窗口                   画出来         上下空白
    ====================  ============  =====================
    区域形状（正确）         38.86×56.32   0 mm
    方形（旧 bug 的残留）     38.86×38.86   **各 8.73 mm**
    ====================  ============  =====================

    That 8.73 mm band above and below the figure is what the user photographed.
    Asserting on the drawn size rather than on the window keeps the test tied to
    the thing they can see.
    """
    from pcbart.artboard import drawn_size_mm
    from pcbart.image_size import picture_size

    placement, region = auto_framed(canvas, tmp_path)
    width_px, height_px = picture_size(canvas._image_path_for(placement))

    def drawn(window):
        x0, y0, x1, y1 = window
        aspect = ((x1 - x0) * width_px) / ((y1 - y0) * height_px)
        return drawn_size_mm(region.width_mm, region.height_mm, aspect)

    _w, height = drawn(placement.source_window)
    assert height == pytest.approx(region.height_mm, abs=0.05), (
        f"默认取景应当铺满立绘区高度，实际只有 {height:.2f}/{region.height_mm:.2f} mm"
    )

    # And it stays full after the gesture that used to break it.
    for _ in range(8):
        canvas.window_zoom(0.8)
    for _ in range(4):
        canvas.window_zoom(1.25)
    _w, height = drawn(placement.source_window)
    assert height == pytest.approx(region.height_mm, abs=0.05), (
        f"缩小再放大之后仍然要铺满立绘区，实际 {height:.2f}/{region.height_mm:.2f} mm"
    )


# ------------------------------------- a picture that has left its opening
#
# The user asked for free movement -- 「我希望立绘能移动的范围不要局限于这个框
# ……我们最上面还有一部分挖孔，如果把边界定死了，用户可能没办法保留完整的图片」
# -- and for a warning when the picture ends up somewhere it prints nothing:
# 「当图片完全离开立绘区时，用对应图层的暗色标示这张图片，例如：顶层-暗红色，
# 顶层丝印-深黄色」.


def test_the_warning_colour_is_the_layer_s_colour_darkened(canvas, tmp_path):
    """⚠️ Per-layer, not one fixed warning colour.

    The point of darkening the layer's own colour is that the reading stays
    "this is the same object on the same layer, and something is wrong". A
    single red warning on a silk-layer picture would point at the wrong layer.
    """
    from PySide6.QtGui import QColor

    from pcbart.design_layers import layer

    expected = {
        "top_copper": "#7f0000",     # 顶层 → 暗红
        "top_silk": "#7f6600",       # 顶层丝印 → 深黄
        "bottom_copper": "#00007f",  # 底层 → 暗蓝
        "bottom_silk": "#33661a",    # 底层丝印 → 暗绿
    }
    for key, want in expected.items():
        base = QColor(layer(key).color)
        got = canvas._warning_colour(base).name()
        assert got == want, f"{layer(key).name}: 警示色应为 {want}，实际 {got}"
        assert QColor(got).value() < base.value(), "警示色必须比图层色暗"


def test_overlap_is_exact_for_a_rotated_picture(canvas, tmp_path):
    """⚠️ A bounding-box test would be wrong here, and this case proves it.

    A rotated picture's bounding box stays wide, so comparing boxes reports
    "still overlapping" for a 45-degree picture whose own edges no longer reach
    the opening -- exactly when the warning matters.

    ⚠️ **The numbers below are the ones that actually disagree.** A first
    attempt offset the picture diagonally by a fraction of its own size, and
    *both* tests called it disjoint -- so the test passed with the exact test
    replaced by a box comparison, i.e. it proved nothing. These coordinates were
    found by searching for a placement where the two answers differ: the box
    overlaps, the shape does not.
    """
    placement, region = auto_framed(canvas, tmp_path)
    x0, y0, x1, y1 = region.bounds

    placement.width_mm = placement.height_mm = 20.0
    placement.rotation_deg = 45.0
    # 8 mm past the top-right corner on both axes: the diamond's bounding box
    # reaches back into the opening, but its nearest edges do not.
    placement.x_mm = x1 + 8.0
    placement.y_mm = y1 + 8.0

    corners = placement.corners()
    box_overlaps = not (
        max(x for x, _y in corners) < x0
        or min(x for x, _y in corners) > x1
        or max(y for _x, y in corners) < y0
        or min(y for _x, y in corners) > y1
    )
    assert box_overlaps, (
        "前提检查：这个用例的包围盒 **必须** 是相交的，否则测不出精确判定"
    )
    assert placement.overlaps_rect(region.bounds) is False, (
        "旋转后只算包围盒会把「已经离开」误判成「还在里面」"
    )

    # Bring it back until a corner does touch, and the exact test agrees.
    placement.x_mm = x1
    placement.y_mm = y1
    assert placement.overlaps_rect(region.bounds) is True


def test_a_picture_entirely_outside_the_opening_is_reported_on_export(tmp_path):
    """⚠️ The warning on the canvas has to be repeated in the export report.

    Free movement makes "pushed right out of the opening" reachable by accident,
    and that state prints nothing. The canvas marks it; an export that said
    nothing would turn a visible warning into a silently missing illustration.

    ⚠️ **And it must be said exactly once.** The check used to run *inside* the
    per-traced-shape loop, so every one of a picture's shapes appended the same
    sentence: a real card with one misplaced illustration produced ~140 copies
    of it, which buried the rest of the report and is what the user photographed
    and reported. Counting the occurrences is the assertion that catches that —
    "is the sentence present" passed both before and after the fix.
    """
    from pcbart.artboard import ArtBoard
    from pcbart.export import build_copper
    from pcbart.image_prep import imwrite_unicode
    from pcbart.vectorize import VectorizeParams

    cv2 = pytest.importorskip("cv2")

    picture = tmp_path / "ink.png"
    ink = np.full((400, 400), 255, np.uint8)
    # Several separate blobs, so the picture traces into more than one shape --
    # with a single shape the per-shape loop could not repeat anything and this
    # test would pass against the old code too.
    for x, y in ((40, 40), (170, 40), (300, 40), (40, 170), (170, 170), (300, 170),
                 (40, 300), (170, 300), (300, 300)):
        cv2.rectangle(ink, (x, y), (x + 60, y + 60), 0, -1)
    imwrite_unicode(str(picture), ink)

    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    region = BADGE_REGIONS[0]
    placement = ImagePlacement(
        path=str(picture),
        processed_path=str(picture),
        width_mm=region.width_mm,
        height_mm=region.height_mm,
        region=region.key,
    )
    # Far enough right that no part of it reaches the opening.
    placement.x_mm = region.bounds[2] + placement.width_mm + 10.0
    placement.y_mm = region.centre_y_mm
    board.placements.append(placement)
    assert placement.overlaps_rect(region.bounds) is False

    _records, report = build_copper(board, params=VectorizeParams())
    note = str(report.get("note", ""))
    assert "移出立绘区" in note, f"导出报告里应当说明图片已移出立绘区，实际：{note!r}"
    assert note.count("移出立绘区") == 1, (
        f"同一张图只该提醒一次，实际 {note.count('移出立绘区')} 次：{note[:400]!r}"
    )


def test_the_canvas_hides_the_ink_that_will_not_be_printed(canvas, tmp_path, app):
    """⚠️ Canvas and export must show the same crop, now that the picture moves.

    With the picture free to sit anywhere, a canvas that drew the whole
    rectangle would promise ink that ``build_copper`` cuts away. So the ink is
    clipped to the opening.

    ⚠️ **The picture has to be larger than the opening for this to measure
    anything.** Two earlier versions of this test passed with the opening clip
    deleted, because in both of them clipping to the *picture's own box* drew
    the same pixels:

    * right after framing, box and opening coincide exactly;
    * moved but at the opening's size, the crop window already excluded the part
      that hung outside, so the second clip had nothing left to remove.

    Oversizing the picture is what puts ink inside the picture's box and
    outside the opening at once -- which is the only state that can tell the two
    clips apart.
    """
    region = BADGE_REGIONS[0]
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    picture = ImagePlacement(
        path=str(plain_picture(tmp_path)),
        width_mm=region.width_mm + 30.0,
        height_mm=region.height_mm + 30.0,
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
    )
    board.placements.append(picture)
    canvas.board = board
    from pcbart.undo import UndoStack

    canvas.set_history(UndoStack(board))
    # A crop over the middle, so the drawn picture really covers the opening.
    canvas.frame_into_card_region(picture)
    # Framing sets the box to the opening; grow it back out, which is the state
    # the user reaches by resizing a picture after it has been placed.
    picture.width_mm = region.width_mm + 30.0
    picture.height_mm = region.height_mm + 30.0
    canvas.fit_view()
    app.processEvents()

    centre_y = region.centre_y_mm
    inside_x = region.centre_x_mm
    outside_x = region.bounds[2] + 6.0

    # The sample outside the opening is inside the picture's own box: without
    # that, "nothing drawn here" would be true for the uninteresting reason.
    assert outside_x < picture.x_mm + picture.width_mm / 2.0, (
        "前提检查：取样点应当在图片框内"
    )
    assert outside_x > region.bounds[2], "前提检查：取样点应当在立绘区外"

    inside = _painted(canvas, inside_x, centre_y)
    outside = _painted(canvas, outside_x, centre_y)

    assert inside.red() > 150, (
        f"立绘区内应当能看到图案，实测 {inside.name()}"
    )
    assert outside.red() < 120, (
        f"立绘区外不该显示图片内容（导出会裁掉），实测 {outside.name()}"
    )


# ------------------------------------------------- the crop must not eat holes
#
# ⚠️ **The reported defect these exist for.** Both of these were wrong in the
# same function, and together they produced the board the user photographed:
# 「整个人物都被填充区域覆盖」.

def _net_area(complex_polygon):
    """Outer ring minus its holes, in mm²."""
    def ring_area(points):
        total = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            total += x0 * y1 - x1 * y0
        return abs(total) / 2.0

    return ring_area(complex_polygon[0]) - sum(
        ring_area(hole) for hole in complex_polygon[1:]
    )


def test_clipping_keeps_the_holes_of_a_ring():
    """⚠️ A stroked ring must stay a ring, not become a solid disc.

    ``shell.difference(hole)`` returns a polygon *with an interior ring*, not a
    ``MultiPolygon``. The old code took that single polygon and then read only
    ``part.exterior`` -- so every enclosed area was filled in.

    This is what turned the drawn outline into a block covering the whole
    illustration opening. Measured on this mask before the fix: 103.7 mm² of ink
    came out as 149.8 mm² of solid.
    """
    import cv2

    from pcbart.export import _clip_to_region
    from pcbart.vectorize import VectorizeParams, mask_to_complex_polygons

    # A donut: outer radius 70 px, inner 40 px, in a frame where 10 px = 1 mm.
    mask = np.full((400, 400), 255, np.uint8)
    cv2.circle(mask, (200, 200), 70, 0, -1)
    cv2.circle(mask, (200, 200), 40, 255, -1)
    px_per_mm = 40.0

    traced = mask_to_complex_polygons(mask, px_per_mm, VectorizeParams())
    assert len(traced) == 1, "前提检查：这个掩膜应当只描出一个复杂多边形"
    assert len(traced[0]) == 2, "前提检查：它应当是一个外环 + 一个孔"

    # The opening is much larger than the donut, so clipping changes nothing.
    out = _clip_to_region(traced[0], BADGE_REGIONS[0].key)
    assert len(out) == 1, f"应当仍是一个复杂多边形，实际 {len(out)} 个"
    assert len(out[0]) == 2, (
        f"孔被吃掉了：裁剪后只剩 {len(out[0])} 个环（期望外环 + 孔）"
    )

    truth = np.pi * ((70 / px_per_mm) ** 2 - (40 / px_per_mm) ** 2)
    got = _net_area(out[0])
    assert got == pytest.approx(truth, rel=0.05), (
        f"净面积应当是环面积 {truth:.1f} mm²，实际 {got:.1f} mm²"
    )


def test_clipping_returns_each_piece_as_its_own_polygon():
    """⚠️ One traced shape can clip into two, and they must stay separate.

    Merging them into a single record means "the first ring is the outer, all
    the rest are holes", so the lower piece punches a hole in the upper one and
    the artwork disappears. On the reported card the largest record had a
    463 mm² outer ring and 625 mm² of holes -- the same defect.

    ⚠️ **The mask has to be ONE traced contour that clips into TWO pieces.**
    An earlier version of this test used two separate blobs, which never reach
    the merge path at all -- each was clipped on its own and the injection that
    merges pieces passed. The tall frame below is what actually splits.
    """
    import cv2

    from pcbart.export import _clip_to_region
    from pcbart.vectorize import VectorizeParams, mask_to_complex_polygons

    # A tall ring, taller than the opening: the opening cuts it into an upper
    # and a lower piece.
    mask = np.full((1200, 1200), 255, np.uint8)
    cv2.rectangle(mask, (500, 60), (700, 1140), 0, 8)
    px_per_mm = 1200 / 120.0

    traced = mask_to_complex_polygons(mask, px_per_mm, VectorizeParams())
    assert len(traced) == 1, "前提检查：应当只描出一个复杂多边形"

    pieces: list = []
    for one in traced:
        pieces.extend(_clip_to_region(one, BADGE_REGIONS[0].key))

    assert len(pieces) == 2, (
        f"被裁成两块时应当各自返回，实际 {len(pieces)} 条"
        f"（合成一条就是「下面那块挖空上面那块」）"
    )
    for piece in pieces:
        assert len(piece) == 1, (
            f"每块都不该把另一块当成自己的孔，实际 {len(piece)} 个环"
        )

    # Each piece is a thin bar, so the total is small -- and crucially it is the
    # *sum*, not one minus the other (which would come out near zero).
    total = sum(_net_area(piece) for piece in pieces)
    assert total > 1.0, f"两块之和不应接近 0，实际 {total:.3f} mm²"


def test_a_clipped_picture_keeps_its_ink_coverage(tmp_path):
    """The end-to-end guard: the written geometry must still look like the picture.

    ⚠️ The two tests above pin the shape of the *return value*; this one catches
    the consequence on a real card. It compares the traced area with the ink
    area of the file the exporter reads -- so a regression that fills the holes,
    or merges pieces, shows up as a number, not as a picture someone has to
    notice.
    """
    import cv2

    from pcbart.artboard import ArtBoard
    from pcbart.export import build_copper
    from pcbart.image_prep import imwrite_unicode
    from pcbart.vectorize import VectorizeParams

    # A picture whose ink is mostly thin line: a big ring plus a small dot.
    # Thin ink is the case that was destroyed, because its enclosed background
    # is large relative to the stroke.
    picture = tmp_path / "ring.png"
    ink = np.full((800, 800), 255, np.uint8)
    cv2.circle(ink, (400, 400), 320, 0, 6)
    cv2.rectangle(ink, (60, 60), (120, 120), 0, -1)
    imwrite_unicode(str(picture), ink)

    region = BADGE_REGIONS[0]
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(
        path=str(picture),
        processed_path=str(picture),
        width_mm=region.width_mm,
        height_mm=region.height_mm,
        x_mm=region.centre_x_mm,
        y_mm=region.centre_y_mm,
        region=region.key,
    )
    board.placements.append(placement)

    # ⚠️ **The truth has to be measured in the size the picture is actually
    # drawn at, not the opening's box.** This is a square picture in a taller
    # opening, so ``drawn_size_mm`` letterboxes it: the ink covers
    # ``region.width_mm`` squared, not width × height. Using the box made the
    # expected value 45% too large and the test fail against correct code.
    from pcbart.artboard import drawn_size_mm

    drawn_w, drawn_h = drawn_size_mm(region.width_mm, region.height_mm, 1.0)
    truth = float((ink < 128).mean()) * drawn_w * drawn_h

    records, _report = build_copper(
        board, params=VectorizeParams(), include_text=False
    )
    assert records, "前提检查：应当写出图形"

    def ring_points(ring):
        """A ring arrives as either ``[(x, y), ...]`` or a ``[x, y, 'L', ...]`` stream."""
        if ring and isinstance(ring[0], (tuple, list)):
            return [(float(p[0]), float(p[1])) for p in ring]
        nums = [v for v in ring if isinstance(v, (int, float))]
        return list(zip(nums[0::2], nums[1::2]))

    # ⚠️ Each record's second element is already ONE complex polygon --
    # ``[outer, hole, ...]`` -- so it goes to ``_net_area`` as-is. Reading it as
    # a list of polygons produced 0 mm², which looked like a rendering failure
    # rather than a mistake in the test.
    got = sum(
        _net_area([ring_points(ring) for ring in complex_polygon])
        for _lid, complex_polygon, _width in records
    )
    assert got == pytest.approx(truth, rel=0.25), (
        f"描出来的墨量应当接近原图 {truth:.1f} mm²，实际 {got:.1f} mm²"
        f"（差太多说明孔被填实或被互相抵扣了）"
    )


# ------------------------------------- switching openings, and visible offcuts


def test_switching_from_the_short_opening_to_the_tall_one_reshapes_the_crop(
    canvas, tmp_path,
):
    """⚠️ The crop has to follow the opening the user chose, not the card's default.

    ⚠️ **This is the reported bug**: 「取景框从短区切换到长区的时候不会改变」.
    The box did change -- 短区 is 38.9 × 56.3 and 长区 is 42.5 × 99.6 -- but the
    *window* was reshaped back to the card's prescribed opening (精0 → 短区)
    every time, so a 长区 box ended up holding a 短区-shaped crop and the
    adjustment looked like it had done nothing.

    A 精0 card is the case that shows it: the card prescribes 短区, yet the user
    is allowed to put a picture in 长区 anyway.
    """
    from pcbart.image_size import picture_size

    placement, _region = auto_framed(canvas, tmp_path)   # 精0 card → 短区
    short = next(r for r in badge_card.BADGE_REGIONS if r.key == "small")
    tall = next(r for r in badge_card.BADGE_REGIONS if r.key == "large")
    assert placement.region == short.key, "前提检查：自动导入落在短区"

    canvas.start_framing(placement, tall)
    canvas.stop_framing()

    x0, y0, x1, y1 = placement.source_window
    width_px, height_px = picture_size(canvas._image_path_for(placement))
    # A window is a fraction of the picture, so the pixel aspect comes out before
    # the shape on the board means anything.
    on_board = ((x1 - x0) * width_px) / ((y1 - y0) * height_px)
    assert on_board == pytest.approx(tall.aspect, rel=0.01), (
        f"长区的取景应当是 {tall.aspect:.4f}，实际 {on_board:.4f}"
        f"（等于短区的 {short.aspect:.4f} 就是没换过来）"
    )
    assert placement.region == tall.key
    assert placement.width_mm == pytest.approx(tall.width_mm, abs=1e-6)
    assert placement.height_mm == pytest.approx(tall.height_mm, abs=1e-6)


def test_resetting_keeps_the_opening_the_user_chose(canvas, tmp_path):
    """重置取景 restores *that* opening's fit -- it does not move the picture back.

    A picture deliberately put in 长区 on a 精0 card must not snap into 短区 when
    the user resets a pan they took too far.
    """
    from pcbart.image_size import picture_size

    placement, _region = auto_framed(canvas, tmp_path)
    tall = next(r for r in badge_card.BADGE_REGIONS if r.key == "large")

    canvas.start_framing(placement, tall)
    canvas.stop_framing()
    default = placement.source_window

    canvas.window_zoom(2.0)
    assert placement.source_window != default, "前提检查：取景确实被改过"

    assert canvas.reset_framing(placement) is True

    assert placement.region == tall.key, "重置不该把图片挪到别的区"
    assert placement.width_mm == pytest.approx(tall.width_mm, abs=1e-6)
    x0, y0, x1, y1 = placement.source_window
    width_px, height_px = picture_size(canvas._image_path_for(placement))
    on_board = ((x1 - x0) * width_px) / ((y1 - y0) * height_px)
    assert on_board == pytest.approx(tall.aspect, rel=0.01)


def test_the_part_outside_the_opening_stays_visible_but_dimmed(canvas, tmp_path, app):
    """⚠️ It must be dimmed, not deleted -- 「其他部分直接被删除了」.

    The opening decides what *prints*; it does not decide what the user may see
    while arranging the picture. Clipping the offcut away entirely made an
    import look as though the rest of the drawing had been thrown out.

    Two readings have to hold at once, and this asserts both: the offcut is
    **darker** than the part that prints, and it is **still drawn**.

    ⚠️ **Four earlier versions of this test measured nothing.** Each failed for a
    geometry reason worth keeping, because they are all easy to repeat:

    1. *Resizing the box does not create an offcut.* ``drawn_size_mm`` fits the
       crop inside the box preserving its aspect, so a taller box only
       letterboxes the picture -- the drawn rectangle stays within the opening.
    2. *There is no room above the opening.* Its top edge is the card's top edge
       (49.99 mm of a 100 mm card), so moving the picture up puts the offcut off
       the viewport.
    3. *Below the opening is not blank.* The card's lower half carries the
       wording, the codes and the crest, so a sample there reads card content and
       cannot tell whether the picture was drawn.
    4. *A single sample row cannot tell "dimmed ink" from "a grid line".* Major
       grid lines are ``#2b343d`` (red 43); a sample that landed on one read 44
       whether or not the offcut existed.

    The clean spot is therefore **beside** the opening, where the card is empty:
    the picture is moved right so its box overhangs the opening's right edge.
    """
    region = BADGE_REGIONS[0]
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    picture = ImagePlacement(
        path=str(plain_picture(tmp_path)),
        width_mm=region.width_mm,
        height_mm=region.height_mm,
    )
    board.placements.append(picture)
    canvas.board = board
    from pcbart.undo import UndoStack

    canvas.set_history(UndoStack(board))
    canvas.frame_into_card_region(picture)

    # Move the picture right. Its box moves with it, so the box and the opening
    # stop coinciding -- that overlap is where the offcut lives.
    shift = 10.0
    picture.x_mm = region.centre_x_mm + shift

    canvas.resize(900, 900)
    canvas.fit_view()
    app.processEvents()
    canvas.repaint()
    app.processEvents()

    box_right = picture.x_mm + picture.width_mm / 2.0
    assert box_right > region.bounds[2], (
        f"前提检查：图片框右沿 {box_right:.2f} 应当超出立绘区右沿 "
        f"{region.bounds[2]:.2f}，否则没有区外部分"
    )

    inside_x = region.centre_x_mm
    # The offcut is "inside the picture's box, outside the opening".
    outside_x = (region.bounds[2] + box_right) / 2.0
    scan_y = region.centre_y_mm

    assert outside_x > region.bounds[2], "前提检查：取样点在立绘区右沿之外"
    assert outside_x < box_right, (
        f"前提检查：取样点 {outside_x:.2f} 应当在图片框之内（右沿 {box_right:.2f}）"
    )

    for label, x_mm in (("区内", inside_x), ("区外", outside_x)):
        screen_x = canvas.to_screen(x_mm, scan_y).x()
        assert 0 <= screen_x <= canvas.width() - 1, (
            f"前提检查：{label}取样点 screen x={screen_x:.1f} 落在画布之外"
            f"（画布宽 {canvas.width()}）"
        )

    def brightest(x_mm, y_mm, half_mm=1.0):
        """The brightest pixel in a small patch.

        ⚠️ **Not a single pixel, and not an average.** A 1 px grid line can make
        any single reading say "content", and an average over mostly-background
        dilutes the answer. Only real painted ink raises the peak.
        """
        best = 0
        for i in range(9):
            for j in range(9):
                best = max(
                    best,
                    _painted(
                        canvas,
                        x_mm - half_mm + 2 * half_mm * i / 8.0,
                        y_mm - half_mm + 2 * half_mm * j / 8.0,
                    ).red(),
                )
        return best

    inside_peak = brightest(inside_x, scan_y)
    outside_peak = brightest(outside_x, scan_y)
    # A control patch beyond the picture on the same side, so anything bright
    # there could only be the canvas itself.
    bare_peak = brightest(box_right + 6.0, scan_y)

    assert inside_peak > 200, f"区内应当是纯白，实测峰值 {inside_peak}"
    # ⚠️ **The threshold clears the grid, not just the background.** The major
    # grid line is red 43; with the offcut branch deleted, a patch on that line
    # still read 44 and a "brighter than the control" test passed. Dimmed white
    # ink is far above that.
    GRID_MAJOR_RED = 60
    assert outside_peak > GRID_MAJOR_RED, (
        f"区外不该被删掉（应当变暗但仍可见）：实测峰值 {outside_peak}，"
        f"网格线本身就有 ~44，说明这里没有图片内容"
    )
    assert outside_peak > bare_peak + 20, (
        f"区外应当明显亮于空白处：实测峰值 {outside_peak}，对照 {bare_peak}"
    )
    assert outside_peak < inside_peak - 20, (
        f"区外应当比区内暗，实测 区内峰值={inside_peak} 区外峰值={outside_peak}"
    )


def test_the_export_still_keeps_only_what_is_inside_the_opening(canvas, tmp_path):
    """⚠️ The dimmed offcut is *visible*, not *printed*.

    Showing the offcut fixes the "my picture was deleted" impression, but it
    must not change what the board carries. Moving the picture so its box
    overhangs the opening is exactly the case that used to write ink outside
    the card's window, so the exported vertices are checked against the
    opening's bounds.
    """
    from pcbart.artboard import ArtBoard
    from pcbart.export import build_copper
    from pcbart.image_prep import imwrite_unicode
    from pcbart.vectorize import VectorizeParams

    cv2 = pytest.importorskip("cv2")

    picture = tmp_path / "ink.png"
    ink = np.full((400, 400), 255, np.uint8)
    cv2.rectangle(ink, (60, 60), (340, 340), 0, -1)
    imwrite_unicode(str(picture), ink)

    region = BADGE_REGIONS[0]
    board = ArtBoard(outline=badge_spec(), outline_ready=True)
    placement = ImagePlacement(
        path=str(picture),
        processed_path=str(picture),
        width_mm=region.width_mm,
        height_mm=region.height_mm,
        region=region.key,
    )
    # Overhang the opening on the right, the way the dimming test does.
    placement.x_mm = region.centre_x_mm + 10.0
    placement.y_mm = region.centre_y_mm
    board.placements.append(placement)

    records, _report = build_copper(
        board, params=VectorizeParams(), include_text=False
    )
    assert records, "前提检查：应当写出图形"

    def ring_points(ring):
        if ring and isinstance(ring[0], (tuple, list)):
            return [(float(p[0]), float(p[1])) for p in ring]
        nums = [v for v in ring if isinstance(v, (int, float))]
        return list(zip(nums[0::2], nums[1::2]))

    x0, y0, x1, y1 = region.bounds
    outside = [
        (x, y)
        for _lid, complex_polygon, _width in records
        for ring in complex_polygon
        for x, y in ring_points(ring)
        if not (x0 - 0.05 <= x <= x1 + 0.05 and y0 - 0.05 <= y <= y1 + 0.05)
    ]
    assert not outside, (
        f"导出不该写出立绘区之外的墨：{len(outside)} 个顶点在区外，"
        f"例如 {outside[:3]}"
    )
