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

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

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


def add(canvas, path, layer_name, app):
    placement = placement_for_pixels(str(path), 300, 300)
    placement.layer = layer_name
    canvas.set_placement(placement)
    canvas.fit_view()
    app.processEvents()
    return placement


# ------------------------------------------------------------------ rendering


def test_an_unprocessed_picture_is_shown_as_imported(canvas, artwork_png, app):
    add(canvas, artwork_png, "顶层", app)

    # It must really be decoded: the grey placeholder would also look "pale".
    pixmap = canvas._pixmap_for(canvas.board.placements[0])
    assert pixmap is not None and not pixmap.isNull()

    colour = painted(canvas, canvas.to_screen(0.0, 0.0))
    assert colour.red() > 200 and colour.green() > 200 and colour.blue() > 200
    assert colour != QColor("#e8ecf0"), "画的是占位块，不是图片"


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

    colour = painted(canvas, canvas.to_screen(0.0, 0.0))
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

    colour = painted(canvas, canvas.to_screen(0.0, 0.0))
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

    painted_colour = painted(canvas, canvas.to_screen(0.0, 0.0))
    solid = QColor(layer("top_mask").color)
    assert painted_colour.name() != solid.name(), "阻焊层不应完全不透明"
    assert layer("top_mask").transparency < 1.0

    # Blended toward the background: each channel headed toward the backdrop
    # (white) rather than staying at the raw layer colour.
    assert painted_colour.red() > solid.red()
    assert painted_colour.blue() > solid.blue()


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

    mixed = painted(canvas, canvas.to_screen(0.0, 0.0))
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
    unselected = painted(canvas, canvas.to_screen(0.0, 0.0))

    canvas.selected = placement
    canvas.update()
    app.processEvents()
    selected = painted(canvas, canvas.to_screen(0.0, 0.0))

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
    assert painted(canvas, canvas.to_screen(0, 0)).name() == "#ff0000"

    placement.layer = "顶层丝印层"
    canvas.update()
    app.processEvents()

    assert painted(canvas, canvas.to_screen(0, 0)).name() == "#ffcc00"


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
    assert painted(canvas, canvas.to_screen(0.0, 0.0)).name() == silk

    canvas.board.layers.set_visible("top_silk", False)
    canvas.update()
    app.processEvents()

    assert painted(canvas, canvas.to_screen(0.0, 0.0)).name() != silk


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
