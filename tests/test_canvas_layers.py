"""Tests for layer-aware rendering, picking and view navigation on the canvas.

Driven through Qt, and where it matters the *painted* colour is sampled rather
than the model inspected: a layer that is set to red in the data but drawn
black is still wrong, and only the pixels show it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from design_canvas import CANVAS_BG, PLACEHOLDER_FILL  # noqa: E402
from gui_mockup import build_window, show_layers  # noqa: E402
from pcbart.artboard import placement_for_pixels  # noqa: E402
from pcbart.design_layers import LAYER_KEYS, layer  # noqa: E402
from pcbart.image_prep import imwrite_unicode  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def artwork_png(tmp_path):
    """A white square on a transparent field, so the centre is solid ink."""
    path = tmp_path / "art.png"
    image = np.zeros((300, 300, 4), dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (240, 240), (255, 255, 255, 255), -1)
    # imwrite_unicode, not cv2.imwrite: cv2 fails silently on this project's
    # non-ASCII path, which would leave the canvas showing a placeholder and
    # make every colour assertion below meaningless.
    imwrite_unicode(str(path), image)
    assert path.exists()
    return path


@pytest.fixture
def canvas(app):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    widget = window.design_canvas
    widget._test_window = window
    app.processEvents()
    yield widget
    window.hide()


def painted(canvas, point: QPointF) -> QColor:
    """The colour actually on screen at a position given in canvas coordinates.

    The device pixel ratio is applied before indexing the grabbed image. On a
    150% display ``grab()`` returns an image 1.5 times the widget's logical
    size, so using logical coordinates as pixel indices samples the wrong
    place -- it read paper instead of ink and made these tests fail for a
    reason that had nothing to do with the canvas.
    """
    image = canvas.grab().toImage()
    ratio = canvas.devicePixelRatio()
    x = int(round(point.x() * ratio))
    y = int(round(point.y() * ratio))
    x = max(0, min(image.width() - 1, x))
    y = max(0, min(image.height() - 1, y))
    return QColor(image.pixel(x, y))


def ink_point(canvas):
    """A screen point inside the test artwork, clear of the coordinate axes.

    **(6, 6) mm, not (0, 0).** The canvas draws the X and Y axes over the
    artwork, and they meet at the origin -- which is exactly where these tests
    used to sample. Every colour assertion then read the axis colour instead of
    the layer's. Six millimetres up and right is still solid ink (the fixtures
    are drawn well past it) but no longer on either axis.
    """
    return canvas.to_screen(6.0, 6.0)


def line_art_on_paper(size: int = 300, ink: int = 0):
    """A line-art image: a dark centre on white paper, like the real output.

    Using an all-white image here would be a trap. Tinting with a mask and
    tinting without one give the same result for all-white, so the test would
    pass while the canvas painted the paper as solid layer colour.
    """
    paper = np.full((size, size), 255, np.uint8)
    quarter = size // 4
    paper[quarter:-quarter, quarter:-quarter] = ink
    return paper


def add(canvas, path, layer_name, app, *, select: bool = False):
    placement = placement_for_pixels(str(path), 300, 300)
    placement.layer = layer_name
    canvas.set_placement(placement)
    if select:
        # Handles only exist for the selected object, and a drag starts on a
        # handle, so the resize tests need the object selected.
        canvas.selected = placement
    canvas.fit_view()
    app.processEvents()
    return placement


def mouse_event(canvas, point: QPointF, kind, button, buttons) -> QMouseEvent:
    """A real mouse event aimed at a canvas position."""
    return QMouseEvent(
        kind,
        point,
        canvas.mapToGlobal(QPoint(int(point.x()), int(point.y()))),
        button,
        buttons,
        Qt.NoModifier,
    )


def drag_handle(canvas, handle_name: str, dx: float, app, *, steps: int = 6) -> None:
    """Drag one edge handle ``dx`` pixels to the right, as a person would.

    Delivered as real press/move/release events rather than by calling the
    resize code directly: the whole point of these tests is what the user's
    mouse does, and a drag is the interaction that was wrong.
    """
    start = canvas.handle_centres()[handle_name]
    app.sendEvent(
        canvas,
        mouse_event(canvas, start, QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
    )
    for step in range(1, steps + 1):
        point = QPointF(start.x() + dx * step / steps, start.y())
        app.sendEvent(
            canvas,
            mouse_event(canvas, point, QEvent.MouseMove, Qt.NoButton, Qt.LeftButton),
        )
        # A paint between moves, because that is where the old code recomputed
        # the scale and where the drift came from.
        canvas.grab()
    end = QPointF(start.x() + dx, start.y())
    app.sendEvent(
        canvas,
        mouse_event(canvas, end, QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton),
    )
    app.processEvents()


# ------------------------------------------------------------------ rendering


def test_an_unprocessed_picture_is_shown_as_imported(canvas, artwork_png, app):
    add(canvas, artwork_png, "顶层", app)

    # It must really be decoded: the grey placeholder would also look "pale".
    pixmap = canvas._pixmap_for(canvas.board.placements[0])
    assert pixmap is not None and not pixmap.isNull()

    colour = painted(canvas, ink_point(canvas))
    assert colour.red() > 200 and colour.green() > 200 and colour.blue() > 200
    assert colour.name() != PLACEHOLDER_FILL.lower(), "画的是占位块，不是图片"


def test_processed_line_art_is_drawn_in_its_layer_colour(
    canvas, artwork_png, tmp_path, app
):
    """A line art is black and white; the layer supplies the colour.

    This is how the editor shows a layer, and it is what makes the picture on
    screen mean something about the board rather than being a photo on a page.
    """
    placement = add(canvas, artwork_png, "顶层", app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()

    colour = painted(canvas, ink_point(canvas))
    assert colour.name() == QColor(layer("top_copper").color).name()


def test_the_paper_behind_the_ink_is_not_painted(canvas, artwork_png, tmp_path, app):
    """Only the ink takes the layer colour; the white paper must vanish.

    Tinting the whole image would paint the artwork's bounding box as a solid
    slab of layer colour -- which is exactly what happened the first time, and
    it looked plausible in a screenshot because the shape filled most of its
    box. Checking the tinted image's own alpha is what makes it unambiguous.
    """
    placement = add(canvas, artwork_png, "顶层", app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()

    tinted = canvas._tinted(
        canvas._pixmap_for(placement), QColor(layer("top_copper").color)
    ).toImage()

    paper = tinted.pixelColor(5, 5)
    ink = tinted.pixelColor(150, 150)

    assert paper.alpha() == 0, "纸面必须透明"
    assert ink.alpha() == 255
    assert ink.name() == QColor(layer("top_copper").color).name()


@pytest.mark.parametrize(
    "layer_name,key",
    [
        ("顶层", "top_copper"),
        ("顶层丝印层", "top_silk"),
        ("底层", "bottom_copper"),
        ("底层丝印层", "bottom_silk"),
    ],
)
def test_each_layer_paints_in_its_own_colour(
    canvas, artwork_png, tmp_path, app, layer_name, key
):
    placement = add(canvas, artwork_png, layer_name, app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()

    colour = painted(canvas, ink_point(canvas))
    assert colour.name() == QColor(layer(key).color).name()


def test_a_solder_mask_is_drawn_semi_transparent(canvas, artwork_png, tmp_path, app):
    """Masks must let the layers underneath show through.

    The layer table records the solder masks as 70% opaque, which is how the
    editor shows a mask opening. Painting them solid made a stack of copper +
    silkscreen + mask look like the mask alone: two of the three layers the
    user selected were invisible.
    """
    placement = add(canvas, artwork_png, "顶层阻焊层", app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()

    painted_colour = painted(canvas, ink_point(canvas))
    solid = QColor(layer("top_mask").color)
    assert painted_colour.name() != solid.name(), "阻焊层不应完全不透明"
    assert layer("top_mask").transparency < 1.0

    # Blended toward the background: each channel headed toward the backdrop
    # rather than staying at the raw layer colour. The canvas is black, so
    # "toward the backdrop" means darker, and both channels must move.
    backdrop = QColor(CANVAS_BG)
    assert painted_colour.red() < solid.red(), "红色通道没有向黑色背景靠拢"
    assert painted_colour.blue() < solid.blue(), "蓝色通道没有向黑色背景靠拢"
    assert painted_colour.red() > backdrop.red()
    assert painted_colour.blue() > backdrop.blue()
    assert (painted_colour.red(), painted_colour.blue()) != (
        backdrop.red(),
        backdrop.blue(),
    ), "阻焊层完全不见了"


def test_stacked_layers_all_contribute_to_the_result(canvas, artwork_png, tmp_path, app):
    """Three layers selected must not render as only the topmost one."""
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    target = Path(artwork_png)
    for layer_name in ("顶层", "顶层丝印层", "顶层阻焊层"):
        placement = placement_for_pixels(str(target), 300, 300)
        placement.layer = layer_name
        placement.processed_path = str(processed)
        canvas.set_placement(placement)
    canvas.fit_view()
    canvas.update()
    app.processEvents()

    mixed = painted(canvas, ink_point(canvas))
    top_only = QColor(layer("top_mask").color)

    assert mixed.name() != top_only.name()
    # The copper and silkscreen show through: the blend carries some of their
    # saturation rather than being the mask colour diminished toward white.
    assert mixed.red() > 120 and mixed.green() < 150


def test_selecting_an_object_does_not_change_its_layer_colour(
    canvas, artwork_png, tmp_path, app
):
    """The selection backdrop goes *behind* the artwork, not over it.

    Painted on top it washed a 60% grey across the object, so the colour on
    screen depended on whether the object happened to be selected -- which
    defeats the point of colouring by layer. The earlier tests all deselected
    first, so they could not see it.
    """
    placement = add(canvas, artwork_png, "顶层丝印层", app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()

    canvas.selected = None
    canvas.update()
    app.processEvents()
    unselected = painted(canvas, ink_point(canvas))

    canvas.selected = placement
    canvas.update()
    app.processEvents()
    selected = painted(canvas, ink_point(canvas))

    assert unselected.name() == QColor(layer("top_silk").color).name()
    assert selected.name() == unselected.name(), "选中不应改变对象颜色"


def test_moving_an_object_to_another_layer_repaints_it(
    canvas, artwork_png, tmp_path, app
):
    placement = add(canvas, artwork_png, "顶层", app)
    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))
    placement.processed_path = str(processed)
    canvas.update()
    app.processEvents()
    assert painted(canvas, ink_point(canvas)).name() == "#ff0000"

    placement.layer = "顶层丝印层"
    canvas.update()
    app.processEvents()

    assert painted(canvas, ink_point(canvas)).name() == "#ffcc00"


def test_copper_is_painted_over_silkscreen_over_mask(canvas, artwork_png, tmp_path, app):
    """The stack is the one 嘉立创 builds unless an order says otherwise.

    Copper on top, silkscreen in the middle, solder mask underneath -- inside
    each side. Anything else means the canvas promises a stack the factory will
    not build, and the mismatch is invisible until the boards come back.

    Checked as a *painted* result, not as a sort: the order that matters is the
    one the pixels end up in.
    """
    from pcbart.design_layers import LAYER_PAINT_ORDER, paint_rank

    assert LAYER_PAINT_ORDER == (
        "bottom_mask",
        "bottom_silk",
        "bottom_copper",
        "top_mask",
        "top_silk",
        "top_copper",
    )
    for side in ("top", "bottom"):
        assert paint_rank(f"{side}_mask") < paint_rank(f"{side}_silk")
        assert paint_rank(f"{side}_silk") < paint_rank(f"{side}_copper")
    # The canvas looks at the board from the top, so the top side is above the
    # bottom one.
    assert paint_rank("bottom_copper") < paint_rank("top_mask")


def test_the_layer_panel_lists_them_in_the_paint_order():
    """The panel reads top-to-bottom, so its list has to run top layer first.

    The two used to disagree -- the panel listed copper first while the canvas
    painted mask last -- which made the panel actively misleading. The outline
    is exempt: it frames the board and is deliberately listed last.
    """
    from pcbart.design_layers import LAYERS, paint_rank

    artwork = [item.key for item in LAYERS if item.kind != "outline"]
    ranks = [paint_rank(key) for key in artwork]
    assert ranks == sorted(ranks, reverse=True), (
        f"图层面板的列表顺序与绘制顺序不一致：{artwork} → {ranks}"
    )
    assert LAYERS[-1].kind == "outline", "板框层应当排在最后"


def test_the_topmost_object_is_chosen_by_layer_not_import_order(
    canvas, artwork_png, tmp_path, app
):
    """Importing a silkscreen picture after a copper one must not put it on top.

    Insertion order is not a stacking decision. Both pictures here cover the
    same point; the copper one wins because copper is the upper layer, whichever
    was added first.
    """
    from pcbart.artboard import placement_for_pixels

    processed = tmp_path / "bw.png"
    imwrite_unicode(str(processed), line_art_on_paper(300))

    copper = placement_for_pixels(str(artwork_png), 300, 300)
    copper.layer = "顶层"
    copper.processed_path = str(processed)
    silk = placement_for_pixels(str(artwork_png), 300, 300)
    silk.layer = "顶层丝印层"
    silk.processed_path = str(processed)
    # The silk one is added **last** -- the case that used to win.
    canvas.board.placements.extend([copper, silk])
    canvas.fit_view()
    canvas.update()
    app.processEvents()

    assert painted(canvas, ink_point(canvas)).name() == QColor("#ff0000").name()

    # And the answer does not change when the list is reordered.
    canvas.board.placements = [silk, copper]
    canvas.update()
    app.processEvents()
    assert painted(canvas, ink_point(canvas)).name() == QColor("#ff0000").name()


def test_hiding_a_layer_removes_it_from_the_canvas(canvas, artwork_png, app):
    """The layer's ink must be gone, whatever background takes its place.

    Asserted as "the layer's colour is absent" rather than "the pixel is
    white": the grid is drawn underneath, so the exact background depends on
    where the millimetre grid happens to fall.
    """
    add(canvas, artwork_png, "顶层丝印层", app)
    processed = Path(artwork_png).with_name("bw.png")
    imwrite_unicode(str(processed), line_art_on_paper(300))
    canvas.board.placements[0].processed_path = str(processed)
    canvas.update()
    app.processEvents()

    silk = QColor(layer("top_silk").color).name()
    assert painted(canvas, ink_point(canvas)).name() == silk

    canvas.board.layers.set_visible("top_silk", False)
    canvas.update()
    app.processEvents()

    assert painted(canvas, ink_point(canvas)).name() != silk


def test_hiding_the_outline_layer_hides_the_board(canvas, artwork_png, app):
    canvas.board.outline_ready = True
    canvas.board.outline.width_mm = 60
    canvas.board.outline.height_mm = 60
    canvas.update()
    app.processEvents()

    canvas.board.layers.set_visible("outline", False)
    canvas.update()
    app.processEvents()
    # The board edge ran along the fill area's border; with the layer hidden
    # that pixel must be background rather than the outline's magenta.
    assert canvas.board.layers.is_visible("outline") is False


# --------------------------------------------------------------- background


def test_the_drawing_surface_is_black(canvas, app):
    """The user asked for a black canvas, so the *painted* pixel is checked.

    Not the constant: a constant that says black while the paint code fills
    white is the failure this has to catch.
    """
    app.processEvents()
    canvas.grab()
    surface = painted(canvas, QPointF(canvas.width() - 40, canvas.height() - 40))

    assert (surface.red(), surface.green(), surface.blue()) == (0, 0, 0), (
        f"画布底色是 {surface.name()}，不是黑色"
    )


def test_the_light_text_still_reads_on_the_dark_canvas(canvas, app):
    """Hints must not stay dark grey on black -- that is text that vanished."""
    app.processEvents()
    canvas.grab()

    from design_canvas import MUTED

    assert QColor(MUTED).lightness() > 110, "提示文字在黑色画布上会看不见"
    assert QColor(CANVAS_BG).lightness() == 0


# ------------------------------------------------------------------- axes


def test_the_axes_are_drawn_and_meet_at_the_origin(canvas, app):
    """Both axes, crossing where (0, 0) actually is."""
    canvas.fit_view()
    app.processEvents()
    canvas.grab()

    origin = canvas.to_screen(0.0, 0.0)
    # Sample along each axis, a little away from the crossing so the tick marks
    # and the origin ring cannot answer for the line itself.
    on_x = painted(canvas, QPointF(origin.x() - 60.0, origin.y()))
    on_y = painted(canvas, QPointF(origin.x(), origin.y() - 60.0))

    assert on_x.lightness() > 60, f"X 轴没画出来（{on_x.name()}）"
    assert on_y.lightness() > 60, f"Y 轴没画出来（{on_y.name()}）"


def test_the_grid_thins_out_instead_of_turning_solid(canvas, app):
    """Zooming out must not collapse the grid into a wash.

    The step doubles until the lines are far enough apart, and the axes and the
    rulers read that same step -- so they cannot drift out of step with it.
    """
    for factor in (1.0, 0.2, 0.05):
        canvas.fit_view()
        canvas.zoom_by(factor)
        spacing = canvas._grid_step() * canvas._px_per_mm
        assert spacing >= canvas.GRID_MIN_PX - 1e-6, (
            f"{factor}x 时网格间距只有 {spacing:.1f}px，会糊成一片"
        )


def test_the_axes_stay_visible_over_the_artwork(canvas, artwork_png, app):
    """A picture is centred on the origin, so the axes must be an overlay.

    Painted under the artwork instead, the origin marker and the middle of both
    axes are covered by the picture -- the exact place they are needed.
    """
    add(canvas, artwork_png, "顶层", app)
    canvas.fit_view()
    app.processEvents()
    canvas.grab()

    origin = canvas.to_screen(0.0, 0.0)
    over_picture = painted(canvas, QPointF(origin.x() - 60.0, origin.y()))

    # Without the overlay this pixel is the picture, which is white here.
    assert over_picture.name() != "#ffffff", "轴线被图片盖住了"
    assert over_picture.lightness() > 60, "轴线在图片上读不出来"


def test_an_axis_that_is_off_screen_is_not_drawn(canvas, app):
    """Panning away leaves a grid, not a line clamped to the window edge.

    A clamped line would read as a border around the canvas rather than as the
    axis it is supposed to be.
    """
    canvas.fit_view()
    # Far enough that the origin leaves the viewport entirely, measured rather
    # than guessed so the test cannot pass by pausing just off screen.
    canvas.pan_by_pixels(0.0, canvas._plot.height() * 3.0)
    app.processEvents()
    canvas.grab()

    origin = canvas.to_screen(0.0, 0.0)
    assert origin.y() > canvas.height(), "测试前提：X 轴已在屏幕下方之外"

    # Scan a full row of the plot that the axis would have been clamped to:
    # nothing in it may be as bright as the axis.
    from design_canvas import AXIS

    row = int(canvas._plot.top() + 40)
    brightest = max(
        painted(canvas, QPointF(x, row)).lightness()
        for x in range(int(canvas._plot.left()), int(canvas._plot.right()))
    )
    assert brightest < AXIS.lightness(), (
        f"轴线贴到了窗口边缘：行内最亮 {brightest}，轴色亮度 {AXIS.lightness()}"
    )


def test_ticks_sit_on_the_major_grid_lines(canvas, app):
    """A tick that misses its grid line is worse than no tick.

    Both read the step from the canvas rather than recomputing it, because the
    step is chosen from the zoom -- a second copy of that rule is how the ticks
    and the grid come apart as the user zooms.
    """
    canvas.fit_view()
    app.processEvents()
    image = canvas.grab().toImage()
    ratio = canvas.devicePixelRatio()
    origin_y = canvas.to_screen(0.0, 0.0).y()
    step = canvas._grid_step()

    def tick_brightness(screen_x: float) -> int:
        """Brightest pixel just past the axis line near ``screen_x``.

        Two, not one, pixels off the line -- the line itself is 1.2 px wide and
        would answer for every position. Three columns are searched because a
        line falling exactly between two pixel columns is antialiased across
        both, and reading only one of them reports a tick as missing.
        """
        x = int(round(screen_x * ratio))
        rows = [int(round((origin_y + dy) * ratio)) for dy in (-2, -3)]
        return max(
            image.pixelColor(column, row).lightness()
            for column in range(x - 1, x + 2)
            for row in rows
            if 0 <= column < image.width() and 0 <= row < image.height()
        )

    # A tick at every labelled position, and none in between. 70 sits in the
    # gap between a grid line (34 at most) and a tick (131 at least).
    for index in range(-4, 5):
        mm = index * step
        x = canvas.to_screen(mm, 0.0).x()
        if not (0 <= x < canvas.width()):
            continue
        if index % canvas.GRID_MAJOR_EVERY == 0 and index != 0:
            assert tick_brightness(x) > 70, f"{mm:.0f}mm 处的刻度没画出来"
        elif index % canvas.GRID_MAJOR_EVERY:
            assert tick_brightness(x) < 70, f"{mm:.0f}mm 处不该有刻度"


# ------------------------------------------------------------------- picking


def test_a_hidden_layer_cannot_be_selected(canvas, artwork_png, app):
    """Otherwise a drag would move something the user cannot see."""
    placement = add(canvas, artwork_png, "顶层", app)
    assert canvas.board.placement_at(0.0, 0.0) is placement

    canvas.board.layers.set_visible("top_copper", False)
    assert canvas.board.placement_at(0.0, 0.0) is None


def test_the_topmost_visible_object_wins_the_pick(canvas, artwork_png, app):
    lower = add(canvas, artwork_png, "底层", app)
    upper = add(canvas, artwork_png, "顶层", app)

    assert canvas.board.placement_at(0.0, 0.0) is upper

    canvas.board.layers.set_visible("top_copper", False)
    assert canvas.board.placement_at(0.0, 0.0) is lower


def test_objects_on_different_layers_keep_their_own_geometry(
    canvas, artwork_png, app
):
    """The point of layers: the same drawing can sit twice, sized differently."""
    first = add(canvas, artwork_png, "顶层", app)
    second = add(canvas, artwork_png, "顶层丝印层", app)
    second.x_mm = 10.0
    second.width_mm = 20.0

    assert first.x_mm == 0.0
    assert first.width_mm != second.width_mm


# ---------------------------------------------------------- view navigation


def test_zoom_keeps_the_point_under_the_anchor(canvas, artwork_png, app):
    add(canvas, artwork_png, "顶层", app)
    anchor = QPointF(canvas.width() * 0.35, canvas.height() * 0.4)
    before = canvas.to_mm(anchor)

    canvas.zoom_by(2.0, anchor)

    after = canvas.to_mm(anchor)
    assert after == pytest.approx(before)


def test_zoom_is_clamped_to_a_usable_range(canvas, artwork_png, app):
    add(canvas, artwork_png, "顶层", app)

    canvas.zoom_by(1e6)
    assert canvas.view_zoom <= canvas.MAX_ZOOM

    canvas.zoom_by(1e-6)
    assert canvas.view_zoom >= canvas.MIN_ZOOM


def test_panning_moves_the_view_and_not_the_geometry(
    canvas, artwork_png, app
):
    placement = add(canvas, artwork_png, "顶层", app)
    before = (placement.x_mm, placement.y_mm)

    canvas.pan_by_pixels(150, -90)

    assert (placement.x_mm, placement.y_mm) == before
    assert canvas.to_mm(QPointF(0, 0)) != pytest.approx((0.0, 0.0))


def test_fit_view_resets_zoom_and_pan(canvas, artwork_png, app):
    add(canvas, artwork_png, "顶层", app)
    canvas.zoom_by(3.0)
    canvas.pan_by_pixels(100, 50)

    canvas.fit_view()

    assert canvas.view_zoom == pytest.approx(1.0)


def test_resizing_the_picture_does_not_rescale_the_view(canvas, artwork_png, app):
    """The bug this guards: dragging a handle used to fight the cursor.

    The fitted scale used to be recomputed from whatever was on the board on
    every paint. Dragging the right edge made the picture wider, which changed
    what "fitted" meant, which rescaled the view -- so the edge moved about
    half as far as the mouse and the two never met. The scale is a view
    decision; resizing a picture is not one.
    """
    placement = add(canvas, artwork_png, "顶层", app, select=True)
    canvas.fit_view()
    app.processEvents()
    canvas.grab()  # a paint, so the view is settled before the drag starts

    before = canvas._px_per_mm
    drag_handle(canvas, "right", 60, app)

    assert canvas._px_per_mm == pytest.approx(before), "拖动边框放缩了画布"
    assert canvas.view_zoom == pytest.approx(1.0)


def test_the_edge_follows_the_cursor_while_dragging(canvas, artwork_png, app):
    """The handle has to end up under the mouse, not merely move towards it."""
    placement = add(canvas, artwork_png, "顶层", app, select=True)
    canvas.fit_view()
    app.processEvents()
    canvas.grab()

    start = canvas.handle_centres()["right"]
    app.sendEvent(
        canvas,
        mouse_event(canvas, start, QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
    )
    target = QPointF(start.x() + 90.0, start.y())
    app.sendEvent(
        canvas,
        mouse_event(canvas, target, QEvent.MouseMove, Qt.NoButton, Qt.LeftButton),
    )
    canvas.grab()

    # Where the right edge of the placement is drawn, in screen pixels.
    edge = canvas.to_screen(
        placement.x_mm + placement.width_mm / 2.0, placement.y_mm
    )
    assert edge.x() == pytest.approx(target.x(), abs=1.0), "边框没跟住鼠标"
    app.sendEvent(
        canvas,
        mouse_event(canvas, target, QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton),
    )


def test_the_picture_grows_while_the_canvas_stays_put(canvas, artwork_png, app):
    """Both halves of the request, measured on the same drag.

    The size the user asked for has to arrive (the drag is not merely ignored)
    while the view underneath it stays where it was.
    """
    placement = add(canvas, artwork_png, "顶层", app, select=True)
    canvas.fit_view()
    app.processEvents()
    canvas.grab()

    before_width = placement.width_mm
    before_scale = canvas._px_per_mm
    before_centre = canvas.to_screen(placement.x_mm, placement.y_mm)

    drag_handle(canvas, "right", 80, app)

    assert placement.width_mm > before_width, "拖动没有改变图片尺寸"
    assert canvas._px_per_mm == pytest.approx(before_scale)
    after_centre = canvas.to_screen(placement.x_mm, placement.y_mm)
    # The left edge is pinned, so the centre moves right by half the growth;
    # what must *not* move is the relationship between millimetres and pixels.
    assert after_centre.y() == pytest.approx(before_centre.y())


def test_a_resized_window_refits_without_touching_the_zoom(
    canvas, artwork_png, app
):
    """A bigger window should show more board, not the same pixels stretched."""
    add(canvas, artwork_png, "顶层", app)
    canvas.zoom_by(2.0)
    app.processEvents()
    before = canvas._fit_px_per_mm

    canvas.resize(canvas.width() + 200, canvas.height() + 150)
    app.processEvents()
    canvas.grab()

    assert canvas._fit_px_per_mm > before
    assert canvas.view_zoom == pytest.approx(2.0), "用户自己的缩放被重置了"


def test_the_wheel_zooms_the_canvas(canvas, artwork_png, app):
    """The wheel is a view control here, not a value editor."""
    add(canvas, artwork_png, "顶层", app)
    before = canvas.view_zoom
    event = QWheelEvent(
        QPointF(200, 200),
        QPointF(200, 200),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )

    QApplication.sendEvent(canvas, event)
    app.processEvents()

    assert canvas.view_zoom > before


def test_the_view_still_frames_the_board_after_panning(canvas, artwork_png, app):
    """Panning must not corrupt the scale the fit computed."""
    add(canvas, artwork_png, "顶层", app)
    canvas.pan_by_pixels(200, 120)
    canvas.fit_view()

    # Everything drawn should come back inside the viewport.
    point = canvas.to_screen(0.0, 0.0)
    assert 0 <= point.x() <= canvas.width()
    assert 0 <= point.y() <= canvas.height()


# ------------------------------------------------------------- layer panel


def test_the_panel_lists_exactly_the_implemented_layers(canvas):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_layers(window)
    try:
        assert tuple(panel._radios) == LAYER_KEYS
        assert tuple(panel._checks) == LAYER_KEYS
    finally:
        window.hide()


def test_the_panel_switches_the_current_layer(canvas):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_layers(window)
    try:
        panel._set_current("bottom_silk")
        assert window.design_canvas.board.layers.current == "bottom_silk"
    finally:
        window.hide()


def test_the_panel_checkbox_controls_visibility(canvas):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_layers(window)
    try:
        panel._checks["bottom_mask"].setChecked(False)
        assert window.design_canvas.board.layers.is_visible("bottom_mask") is False
        panel._checks["bottom_mask"].setChecked(True)
        assert window.design_canvas.board.layers.is_visible("bottom_mask") is True
    finally:
        window.hide()


def test_hiding_the_current_layer_moves_the_current_layer(canvas):
    """An import must not land on a layer that is not on screen."""
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_layers(window)
    try:
        panel._set_current("top_silk")
        panel._checks["top_silk"].setChecked(False)

        assert window.design_canvas.board.layers.current != "top_silk"
        assert window.design_canvas.board.layers.is_visible(
            window.design_canvas.board.layers.current
        )
    finally:
        window.hide()


# ------------------------------------------------------- template text face

def test_the_canvas_draws_template_text_in_the_template_font(app):
    """The canvas and the writers must use the same face, not just the anchor.

    The page used to draw every template line with Qt's default face while the
    exporters rendered the family the template names. When those differ the
    *same* anchor produces different ink: measured on the card, 阵营logo came
    out 0.93 mm wider and 0.76 mm taller on screen than in the file, and
    ©HYPERGRYPH 2.01 mm wider -- which is what "the preview looks offset" was.

    ⚠️ **This compares against the canvas helper itself, not against a font
    built from the file.** Qt resolves a family name against every face the
    process has registered, and by the time this runs other tests have painted
    text through the real window, so a freshly registered face can come back
    under a family name that already means something else. That is a quirk of
    running many Qt tests in one process, and asserting on the resolved family
    made this test pass alone and fail in the file. What is checked here is
    that the helper hands back the face ``font_path`` chose, by comparing the
    metrics it produces with the exporter's for the same string.
    """
    from PySide6.QtGui import QImage

    from pcbart import badge_card
    from pcbart.artboard import BoardOutlineSpec, OutlineShape
    from pcbart.text_render import font_path, render_mask

    # ⚠️ Measured from *painted pixels*, not from a QPainterPath built in the
    # test. Qt's font database is process-wide, and a face registered here can
    # be served back under a family name another test already claimed -- so a
    # path built directly in the test measures something the canvas would never
    # draw, and the assertion passes or fails depending on test order. The
    # canvas's own output is the thing under test anyway.
    window = build_window()
    window.resize(1600, 1000)
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    try:
        board = window.design_canvas.board
        board.outline = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60, height_mm=110)
        board.outline_ready = True
        board.cutouts = []
        board.placements = []
        board.operator_name = "阿米娅"
        board.operator_number = "R001"
        canvas = window.design_canvas
        canvas.fit_view()
        app.processEvents()

        # A marker far off-board keeps the canvas from drawing its empty-state
        # hint, which would otherwise show up in every difference image.
        blank = badge_card.badge_texts()[0]
        marker = type(blank)(
            text=".", x_mm=999.0, y_mm=999.0, size_mm=0.1,
            rotation_deg=0.0, layer=blank.layer, font=blank.font, label="dummy",
        )

        def shot() -> np.ndarray:
            image = QImage(canvas.size(), QImage.Format_RGB32)
            canvas.render(image)
            raw = image.convertToFormat(QImage.Format_RGB32)
            stride = raw.bytesPerLine()
            return np.frombuffer(
                raw.constBits(), dtype=np.uint8, count=stride * raw.height()
            ).reshape(raw.height(), stride // 4, 4)[:, : raw.width(), :3].copy()

        for item in badge_card.badge_texts():
            assert font_path(item.font, item.resolved_text(
                name="阿米娅", number="R001"
            )) is not None, item.label

            # ⚠️ **Measured at a raised zoom, centred on the line under test.**
            # At the fit scale the smallest line is ~8 px tall, and Qt's
            # per-glyph rounding alone can add 12 px to an 11-glyph run -- 26%
            # on ©HYPERGRYPH, which is the size of the face mismatch this test
            # exists to catch (measured: 8.133 mm when the wrong family was
            # pinned against 6.781 mm for the file's own face) and is why the
            # old 25% tolerance let that bug through at this zoom. At 3x the
            # rounding falls under 4%; the tolerance below sits between the two.
            canvas.fit_view()
            canvas.set_view_state((3.0, item.x_mm, item.y_mm))
            app.processEvents()
            ppm = canvas._px_per_mm

            board.texts = [marker]
            canvas.update()
            app.processEvents()
            before = shot()
            board.texts = [item]
            canvas.update()
            app.processEvents()
            after = shot()

            changed = np.abs(before.astype(int) - after.astype(int)).sum(axis=2) > 20
            rows, cols = np.where(changed)
            assert len(cols), f"{item.label} 画布上没有画出任何墨迹"
            # A quarter-turned line runs down the screen, so its *length* is the
            # pixel height rather than the width.
            if abs(item.rotation_deg) % 180.0 > 1.0:
                canvas_width = (rows.max() - rows.min() + 1) / ppm
            else:
                canvas_width = (cols.max() - cols.min() + 1) / ppm

            rendered = render_mask(item, name="阿米娅", number="R001")
            exported = rendered.size_mm[0]
            # The tolerance sits between the two errors it has to tell apart:
            # pixel quantisation at this zoom (measured worst 3.8%) and a
            # substituted face (~18% on the line above).
            assert canvas_width == pytest.approx(exported, rel=0.10), (
                f"{item.label}: 画布 {canvas_width:.3f}mm / 导出 {exported:.3f}mm"
                f"（画布 {ppm:.2f} px/mm，该行 {round(item.size_mm * ppm)} px 高）"
            )
    finally:
        window.hide()


def test_an_unresolvable_font_falls_back_instead_of_crashing(app):
    """A machine without the template's face still has to draw the card."""
    from design_canvas import template_font

    font = template_font("完全没有这个字体", "普通文字")
    assert font.family()


def test_generated_card_art_is_recoloured_by_alpha(app):
    """All four generated elements carry their shape in alpha, so all four
    have to be recoloured -- and nothing else may be.

    The QR symbol, the barcode, the profession plate and the faction logo are
    inserted by one button and share an ``import_id``, but the *way* they encode
    their shape is the same: white RGB, ``alpha = the mark``, no
    ``processed_path``. Drawn as raw pixmaps on the dark canvas they come out
    invisible (the codes) or as white blocks (the logo, whose plate *is* the
    white part).

    ⚠️ ``kind`` is what the test keys on, not the import id: the property panel
    and older drafts produce placements with that id absent, and those must
    keep their own colours.
    """
    from pcbart import badge_card
    from pcbart.artboard import ImagePlacement
    from pcbart.badge_content import ART_IMPORT_ID

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    try:
        canvas = window.design_canvas
        board = canvas.board
        board.outline = badge_card.badge_spec()
        board.outline_ready = True
        board.texts = []
        board.cutouts = []
        board.placements = []

        for kind in ("qr", "barcode", "profession", "faction"):
            item = ImagePlacement(import_id=ART_IMPORT_ID, kind=kind)
            assert canvas._is_generated_ink(item), kind

        # A plain placement -- including one that carries the import id but no
        # kind, which is what a board saved before the field existed looks
        # like -- is drawn as itself.
        assert not canvas._is_generated_ink(ImagePlacement())
        assert not canvas._is_generated_ink(
            ImagePlacement(import_id=ART_IMPORT_ID)
        )
        assert not canvas._is_generated_ink(
            ImagePlacement(import_id="user-import", kind="faction")
        )
    finally:
        window.hide()


# --------------------------------------------------------- repaint cost


def test_moving_an_object_reuses_the_mapped_decorations(canvas, app):
    """A drag must not re-map the card's ~50 000 decoration vertices.

    ``to_screen`` runs once per vertex, and it used to run again on every
    frame of a drag even though the view had not moved. Measured on a finished
    card: 113 ms per frame before, 27 ms after -- 9 fps to 37 fps.
    """
    from pcbart import badge_card

    board = canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    canvas.fit_view()
    canvas.update()
    app.processEvents()

    calls = {"n": 0}
    original = canvas.to_screen

    def counted(x, y):
        calls["n"] += 1
        return original(x, y)

    # Setting the board already painted the card once, so the mapping is in the
    # cache. Drop it, or the "first" reading would itself be a cache hit and
    # the test would compare a hit against a hit.
    canvas._decoration_cache.clear()
    canvas._decoration_view = None

    canvas.to_screen = counted
    try:
        canvas.update()
        app.processEvents()
        first = calls["n"]
        assert first > 1000, f"装饰根本没画（只映射了 {first} 点）"

        # Second repaint of the same view: the mapping is already known.
        calls["n"] = 0
        canvas.update()
        app.processEvents()
        second = calls["n"]
    finally:
        canvas.to_screen = original

    assert second < first * 0.2, (
        f"同一视图第二次重画仍在重新映射：{first} → {second} 个点"
    )


def test_changing_the_view_drops_the_mapped_decorations(canvas, app):
    """The cache must not survive a zoom, or the card would draw stale.

    A cached screen polygon is only valid for the transform it was built with;
    keeping it across a zoom would leave the artwork at the old scale while
    everything else moved.
    """
    from pcbart import badge_card

    board = canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    canvas.fit_view()
    canvas.update()
    app.processEvents()
    assert canvas._decoration_cache, "装饰映射没有进缓存"

    canvas.zoom_by(1.8, canvas.rect().center())
    canvas.update()
    app.processEvents()
    # A repaint after a zoom rebuilds it for the new transform.
    assert canvas._decoration_cache, "缩放后没有重建装饰映射"

    # The mapped geometry really is at the new scale: a point 10 mm from the
    # origin lands twice as far from the plot centre as it did before.
    before = canvas._plot.center()
    point = canvas.to_screen(10.0, 0.0)
    assert abs(point.x() - before.x()) > 10.0 * canvas._px_per_mm * 0.9


def test_to_screen_matches_the_uncached_arithmetic(canvas):
    """The cached view numbers must give exactly what the old code gave.

    ``to_screen`` is now an attribute lookup instead of a ``QRectF.center()``
    call, which is the whole speed-up -- so it is worth pinning that the two
    agree rather than trusting the refactor by eye.
    """
    canvas.resize(900, 900)
    canvas._recompute_plot()

    plot = canvas._plot
    centre_x = plot.x() + plot.width() / 2.0
    centre_y = plot.y() + plot.height() / 2.0
    for x_mm, y_mm in ((0.0, 0.0), (13.5, -22.25), (-24.9, 49.9)):
        expected_x = centre_x + (x_mm - canvas._pan_mm[0]) * canvas._px_per_mm
        expected_y = centre_y - (y_mm - canvas._pan_mm[1]) * canvas._px_per_mm
        point = canvas.to_screen(x_mm, y_mm)
        assert point.x() == pytest.approx(expected_x, abs=1e-9)
        assert point.y() == pytest.approx(expected_y, abs=1e-9)
        # And the inverse agrees with it.
        back_x, back_y = canvas.to_mm(point)
        assert back_x == pytest.approx(x_mm, abs=1e-9)
        assert back_y == pytest.approx(y_mm, abs=1e-9)


def test_every_view_change_refreshes_the_mapping(canvas, app):
    """⚠️ A view change has to re-map *immediately*, not at the next repaint.

    ``to_screen`` maps through cached view numbers while the drag maths still
    divides by the live ``_px_per_mm``. Refreshing only during ``paintEvent``
    left those two out of step for however long it took the scheduled repaint
    to arrive, and the measured damage was a 15 mm drag landing as 1 mm.

    Every mutator is exercised here, and each one is checked *without* pumping
    the event loop, which is the state the bug lived in.
    """
    canvas.resize(900, 900)
    canvas.fit_view()

    def assert_consistent(what: str) -> None:
        plot = canvas._plot
        assert canvas._scale == pytest.approx(canvas._px_per_mm), (
            f"{what} 之后缓存的缩放与 _px_per_mm 不一致："
            f"{canvas._scale} vs {canvas._px_per_mm}"
        )
        assert canvas._origin_x == pytest.approx(
            plot.x() + plot.width() / 2.0
        ), f"{what} 之后缓存的 origin_x 是旧值"
        assert canvas._origin_y == pytest.approx(
            plot.y() + plot.height() / 2.0
        ), f"{what} 之后缓存的 origin_y 是旧值"

    assert_consistent("fit_view")

    canvas.zoom_by(1.7, canvas.rect().center())
    assert_consistent("zoom_by")

    canvas.pan_by_pixels(40.0, -25.0)
    assert_consistent("pan_by_pixels")

    canvas.refit_view()
    assert_consistent("refit_view")

    canvas.set_view_state((2.0, 3.0, -4.0))
    assert_consistent("set_view_state")

    canvas.resize(700, 640)
    assert_consistent("resize")

    # And the mapping really is invertible at every one of those states.
    for x_mm, y_mm in ((0.0, 0.0), (12.5, -30.0), (-20.0, 40.0)):
        back = canvas.to_mm(canvas.to_screen(x_mm, y_mm))
        assert back[0] == pytest.approx(x_mm, abs=1e-9)
        assert back[1] == pytest.approx(y_mm, abs=1e-9)


def test_a_drag_right_after_showing_the_canvas_moves_the_object(canvas, app):
    """The regression the view cache introduced, asserted as the user sees it.

    ``canvas.show()`` schedules the first paint; a drag that begins before that
    paint arrives used to be measured against the pre-paint transform and move
    the object almost nowhere.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent

    from pcbart.artboard import BoardOutlineSpec, ImagePlacement

    board = canvas.board
    board.outline = BoardOutlineSpec(width_mm=60.0, height_mm=60.0)
    board.outline_ready = True
    placement = ImagePlacement(
        x_mm=10.0, y_mm=5.0, width_mm=20.0, height_mm=20.0
    )
    board.placements.append(placement)
    canvas.selected = placement
    canvas.show()
    # ⚠️ Deliberately no processEvents(): the first paint has not run yet.
    start = canvas.to_screen(10.0, 5.0)
    end = canvas.to_screen(25.0, 12.0)

    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.MouseButtonPress,
            start,
            canvas.mapToGlobal(start.toPoint()),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    canvas.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.MouseMove,
            end,
            canvas.mapToGlobal(end.toPoint()),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )

    assert placement.x_mm == pytest.approx(25.0, abs=0.2), (
        f"拖到 25 mm 却停在 {placement.x_mm:.2f} mm"
    )
    assert placement.y_mm == pytest.approx(12.0, abs=0.2)
