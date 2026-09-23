"""The 翻转 and 旋转 controls, from the property panel to the manufactured board.

These two controls existed in the property panel, were written into the draft,
and were then ignored by the canvas *and* by both exporters. The panel really did
let the user tick 水平翻转 and see nothing happen.

So these tests do not check "the flag is stored" -- that was already true and
was exactly the problem. They check that the transform reaches every place the
artwork is drawn or written:

* the canvas, compared against the model rather than against a hand-computed
  pixel, so the two cannot drift;
* the ``.epro2`` writer, whose geometry is read back out of the file;
* the Gerber writer, whose geometry is read back with an independent parser.

An asymmetric glyph is used throughout. A symmetric test image looks identical
flipped or not, which is how a control that does nothing passes a naive test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    drawn_size_mm,
)

cv2 = pytest.importorskip("cv2")


# --------------------------------------------------------- the shared model


def test_the_matrix_is_the_identity_when_nothing_is_transformed():
    placement = ImagePlacement(x_mm=5, y_mm=-3)

    a, b, c, d, e, f = placement.matrix()

    assert (a, b, c, d, e, f) == pytest.approx((1, 0, 0, 1, 5, -3))


def test_horizontal_flip_mirrors_x_only():
    placement = ImagePlacement(width_mm=40, height_mm=20, flip_horizontal=True)

    assert placement.local_to_board(10, 5) == pytest.approx((-10, 5))


def test_vertical_flip_mirrors_y_only():
    placement = ImagePlacement(width_mm=40, height_mm=20, flip_vertical=True)

    assert placement.local_to_board(10, 5) == pytest.approx((10, -5))


def test_both_flips_equal_a_half_turn():
    both = ImagePlacement(
        width_mm=40, height_mm=20, flip_horizontal=True, flip_vertical=True
    )
    half_turn = ImagePlacement(width_mm=40, height_mm=20, rotation_deg=180)

    assert both.local_to_board(7, 3) == pytest.approx(half_turn.local_to_board(7, 3))


def test_the_matrix_agrees_with_the_corner_definition():
    """With no flip, the artwork transform is the box transform exactly.

    ``corners()`` is what the selection box, the handles and hit-testing are
    built from, so the artwork has to sit in the same frame.
    """
    for rotation in (0, 30, 90, -45, 180):
        placement = ImagePlacement(
            x_mm=3, y_mm=-2, width_mm=40, height_mm=20, rotation_deg=rotation
        )
        half_w, half_h = placement.width_mm / 2, placement.height_mm / 2
        local = [
            (-half_w, half_h), (half_w, half_h),
            (half_w, -half_h), (-half_w, -half_h),
        ]
        for corner, point in zip(placement.corners(), local):
            assert corner == pytest.approx(placement.local_to_board(*point)), rotation


def test_a_flip_mirrors_the_artwork_inside_its_box():
    """The box does not move when the picture is flipped.

    A flip is something done to the artwork, not to the placement: the
    selection frame, the handles and the size the user typed all describe the
    same rectangle afterwards. Moving the box instead would make the picture
    jump across the canvas as a side effect of mirroring it.
    """
    plain = ImagePlacement(width_mm=40, height_mm=20, rotation_deg=30)
    flipped = ImagePlacement(
        width_mm=40, height_mm=20, rotation_deg=30, flip_horizontal=True
    )

    assert flipped.corners() == pytest.approx(plain.corners())

    # The artwork's own corners, though, do move -- mirrored within that box.
    half_w, half_h = 20.0, 10.0
    for point in ((-half_w, half_h), (half_w, half_h)):
        assert flipped.local_to_board(*point) != pytest.approx(
            plain.local_to_board(*point)
        )


def test_the_artwork_is_letterboxed_not_stretched():
    """The box may disagree with the picture when the aspect is unlocked; the
    drawing is fitted, never distorted."""
    # A 2:1 picture in a square box keeps its proportions.
    assert drawn_size_mm(40, 40, 2.0) == pytest.approx((40, 20))
    assert drawn_size_mm(40, 40, 0.5) == pytest.approx((20, 40))
    # A box that already matches is filled exactly.
    assert drawn_size_mm(40, 20, 2.0) == pytest.approx((40, 20))


def test_the_drawn_size_never_exceeds_the_box():
    for box_w, box_h in ((40, 40), (10, 50), (50, 10)):
        for aspect in (0.2, 1.0, 3.0):
            w, h = drawn_size_mm(box_w, box_h, aspect)
            assert w <= box_w + 1e-9 and h <= box_h + 1e-9, (box_w, box_h, aspect)


# ------------------------------------------------------------- the canvas


def glyph(tmp_path, size: int = 100) -> str:
    """An asymmetric mark: a square in one corner of the frame.

    Asymmetric on purpose. A centred or symmetric picture looks the same
    mirrored, so it cannot tell a working control from a broken one.
    """
    mask = np.full((size, size), 255, np.uint8)
    mask[15 : size // 2, 15 : size // 2] = 0
    path = tmp_path / "glyph.png"
    cv2.imwrite(str(path), mask)
    return str(path)


def canvas_with(path: str, **kwargs):
    from PySide6.QtWidgets import QApplication

    from tools.design_canvas import DesignCanvas

    application = QApplication.instance() or QApplication([])
    _ = application
    canvas = DesignCanvas()
    canvas.resize(520, 420)
    canvas.show()

    board = canvas.board
    board.outline_ready = True
    board.outline.width_mm = 60
    board.outline.height_mm = 60
    placement = ImagePlacement(
        path=path, processed_path=path, width_mm=40, height_mm=40,
        layer="顶层", **kwargs
    )
    board.placements.append(placement)
    canvas.selected = None
    return canvas, placement


def artwork_centroid(canvas, placement):
    """The centre of the drawn artwork, in screen pixels.

    The placement's own outline is drawn in the same colour as the ink, so a
    border band is excluded -- otherwise the symmetric frame drags whatever
    measurement is taken towards the box centre, which is what made a first
    attempt at this probe report a constant offset on every combination.
    """
    canvas.update()
    canvas.repaint()
    image = canvas.grab().toImage()
    buffer = np.frombuffer(image.bits(), np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )[:, : image.width(), :]
    ink = (buffer[:, :, 2] > 150) & (buffer[:, :, 1] < 100) & (buffer[:, :, 0] < 100)

    # ``grab`` hands back a pixmap at the screen's device pixel ratio, so on a
    # 1.5x display it is half again as large as the widget's logical size. The
    # polygon is in logical coordinates, so both the masking below and the
    # answer have to be scaled -- without this the measurement is off by the
    # ratio and looks like a transform error.
    ratio = image.width() / max(1, canvas.width())

    polygon = canvas.image_polygon(placement)
    xs = [point.x() * ratio for point in polygon]
    ys = [point.y() * ratio for point in polygon]
    margin = 9 * ratio
    ink[: int(min(ys) + margin), :] = False
    ink[int(max(ys) - margin) :, :] = False
    ink[:, : int(min(xs) + margin)] = False
    ink[:, int(max(xs) - margin) :] = False

    found_y, found_x = np.where(ink)
    assert len(found_x), "画布上没有画出手稿"
    return float(found_x.mean()) / ratio, float(found_y.mean()) / ratio


def test_the_canvas_flips_the_artwork(tmp_path):
    """Ticking 水平翻转 has to move the ink, or the control is a lie."""
    path = glyph(tmp_path)
    canvas, placement = canvas_with(path)
    before = artwork_centroid(canvas, placement)

    placement.flip_horizontal = True
    after = artwork_centroid(canvas, placement)

    assert abs(after[0] - before[0]) > 20, "水平翻转没有改变画面"
    assert abs(after[1] - before[1]) < 3, "水平翻转不该改变纵向位置"


def test_the_canvas_flips_vertically(tmp_path):
    path = glyph(tmp_path)
    canvas, placement = canvas_with(path)
    before = artwork_centroid(canvas, placement)

    placement.flip_vertical = True
    after = artwork_centroid(canvas, placement)

    assert abs(after[1] - before[1]) > 20, "垂直翻转没有改变画面"
    assert abs(after[0] - before[0]) < 3, "垂直翻转不该改变横向位置"


def test_the_canvas_rotates_the_artwork(tmp_path):
    path = glyph(tmp_path)
    canvas, placement = canvas_with(path)
    before = artwork_centroid(canvas, placement)

    placement.rotation_deg = 90
    after = artwork_centroid(canvas, placement)

    assert abs(after[0] - before[0]) > 20 or abs(after[1] - before[1]) > 20, (
        "旋转没有改变画面"
    )


def test_the_canvas_matches_the_model_for_every_combination(tmp_path):
    """The canvas must draw exactly what the model says, not merely something
    different.

    Every flip/rotation pair is compared against the model's own transform.
    A mismatch here means the picture on screen is not the picture that will be
    exported, which is the failure this whole area had.
    """
    path = glyph(tmp_path)
    canvas, placement = canvas_with(path)

    source = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(source < 128)
    height_px, width_px = source.shape[:2]
    local = (
        (xs.mean() + 0.5 - width_px / 2.0) / width_px * placement.width_mm,
        (height_px / 2.0 - (ys.mean() + 0.5)) / height_px * placement.height_mm,
    )

    for flip_h in (False, True):
        for flip_v in (False, True):
            for rotation in (0, 90, 180, -90):
                placement.flip_horizontal = flip_h
                placement.flip_vertical = flip_v
                placement.rotation_deg = rotation

                measured = artwork_centroid(canvas, placement)
                board_x, board_y = placement.local_to_board(*local)
                expected = canvas.to_screen(board_x, board_y)

                assert measured == pytest.approx(
                    (expected.x(), expected.y()), abs=6
                ), (flip_h, flip_v, rotation)


# ------------------------------------------------------------ the exports


def exported_geometry(board):
    """One filled layer's geometry as a single Shapely shape."""
    import shapely.geometry as sg
    from shapely.ops import unary_union

    from pcbart.export import build_copper

    records, _report = build_copper(board)
    assert records, "没有写出任何几何"
    return unary_union(
        [sg.Polygon(payload[0], payload[1:]) for _layer, payload, _width in records]
    )


def export_board(path: str, **kwargs) -> ArtBoard:
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=60, height_mm=60), outline_ready=True
    )
    board.placements.append(
        ImagePlacement(
            path=path, processed_path=path, width_mm=40, height_mm=40,
            layer="顶层", **kwargs
        )
    )
    return board


def test_the_export_mirrors_the_geometry(tmp_path):
    """The .epro2 must carry the flip, not just the canvas."""
    path = glyph(tmp_path)

    base = exported_geometry(export_board(path))
    flipped = exported_geometry(export_board(path, flip_horizontal=True))

    assert base.centroid.x != pytest.approx(flipped.centroid.x)
    assert flipped.centroid.x == pytest.approx(-base.centroid.x, abs=0.05)
    assert flipped.centroid.y == pytest.approx(base.centroid.y, abs=0.05)
    # Mirroring moves the ink; it does not resize it.
    assert flipped.area == pytest.approx(base.area, abs=0.5)


def test_the_export_rotates_the_geometry(tmp_path):
    path = glyph(tmp_path)

    base = exported_geometry(export_board(path))
    turned = exported_geometry(export_board(path, rotation_deg=90))

    # A quarter turn counter-clockwise: (x, y) becomes (-y, x).
    assert turned.centroid.x == pytest.approx(-base.centroid.y, abs=0.05)
    assert turned.centroid.y == pytest.approx(base.centroid.x, abs=0.05)
    assert turned.area == pytest.approx(base.area, abs=0.5)


def test_the_exported_geometry_is_still_valid_when_transformed(tmp_path):
    """Rotation cannot introduce a self-intersection.

    A rigid transform of a valid ring stays valid, and this pins that the
    transform is actually rigid -- a mis-signed shear would still produce
    something that looks turned but is not.
    """
    path = glyph(tmp_path)

    for kwargs in (
        {"flip_horizontal": True},
        {"flip_vertical": True},
        {"rotation_deg": 45},
        {"flip_horizontal": True, "rotation_deg": 30},
    ):
        shape = exported_geometry(export_board(path, **kwargs))
        assert shape.is_valid, kwargs


def test_a_rotated_artwork_stays_inside_its_placement(tmp_path):
    """The drawn shape must not grow past the box the user sized.

    A rotation of a fitted rectangle stays within its own bounding circle, and
    the placement box is that rectangle -- so the outline is what stops the
    artwork from creeping over the board edge.
    """
    path = glyph(tmp_path)

    for rotation in (0, 45, 90):
        board = export_board(path, rotation_deg=rotation)
        shape = exported_geometry(board)

        assert shape.bounds[0] >= -20.001, rotation
        assert shape.bounds[2] <= 20.001, rotation
        assert shape.bounds[1] >= -20.001, rotation
        assert shape.bounds[3] <= 20.001, rotation


def test_an_unlocked_box_letterboxes_instead_of_stretching(tmp_path):
    """A 1:1 picture in a 2:1 box is drawn at 1:1, not squashed.

    The canvas and the exporter have to agree here; they used to disagree,
    with the canvas letterboxing and the writer stretching.
    """
    path = glyph(tmp_path)
    board = ArtBoard(
        outline=BoardOutlineSpec(width_mm=60, height_mm=60), outline_ready=True
    )
    board.placements.append(
        ImagePlacement(
            path=path, processed_path=path, width_mm=40, height_mm=20,
            layer="顶层",
        )
    )

    shape = exported_geometry(board)

    assert shape.bounds[2] - shape.bounds[0] <= 20.001
    assert shape.bounds[3] - shape.bounds[1] <= 20.001


# ----------------------------------------------- the Gerber path


def test_the_gerber_carries_the_flip(tmp_path):
    """Read back with an independent parser, not with our own writer.

    The Gerber is what the board is actually made from, so "the canvas looks
    right" is not enough for this one.
    """
    gerbonara = pytest.importorskip("gerbonara")

    path = glyph(tmp_path)
    boxes = {}
    for tag, kwargs in (("base", {}), ("flipped", {"flip_horizontal": True})):
        from pcbart.gerber import export_gerber

        directory = tmp_path / tag
        export_gerber(export_board(path, **kwargs), directory, title=tag)
        copper = gerbonara.GerberFile.open(
            str(directory / "Gerber_TopLayer.GTL")
        )
        (x0, y0), (x1, y1) = copper.bounding_box()
        boxes[tag] = (x0, x1)

    # Mirroring swaps the ink's extent about the placement's centre.
    assert boxes["base"][0] == pytest.approx(-boxes["flipped"][1], abs=0.05)
    assert boxes["base"][1] == pytest.approx(-boxes["flipped"][0], abs=0.05)


def test_the_gerber_carries_the_rotation(tmp_path):
    gerbonara = pytest.importorskip("gerbonara")

    path = glyph(tmp_path)
    from pcbart.gerber import export_gerber

    directory = tmp_path / "rot"
    export_gerber(export_board(path, rotation_deg=90), directory, title="rot")
    copper = gerbonara.GerberFile.open(str(directory / "Gerber_TopLayer.GTL"))
    (x0, y0), (x1, y1) = copper.bounding_box()

    # The glyph lives in one corner, so a quarter turn moves it to another:
    # the bounding box must no longer be anchored bottom-left.
    assert x0 > -20.0 or y0 > -20.0, "旋转没有把图案转过去"


# ------------------------------------------------- a painting is two objects


def test_flipping_one_tone_of_a_painting_moves_its_pair(tmp_path):
    """A board painting is two placements written as one geometry.

    The exporter derives the black tone from the white, so turning only the
    selected one would leave the two half-pictures at different angles and the
    board would tear along their seam.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from tools.gui_mockup import build_window

    application = QApplication.instance() or QApplication([])
    _ = application
    path = glyph(tmp_path)

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    board = window.design_canvas.board
    board.outline_ready = True

    white = ImagePlacement(
        path=path, processed_path=path, width_mm=40, height_mm=40,
        layer="顶层丝印层", tone="white", import_id="imp-1",
    )
    black = ImagePlacement(
        path=path, processed_path=path, width_mm=40, height_mm=40,
        layer="顶层", tone="black", import_id="imp-1",
    )
    board.placements += [white, black]

    inspector = window.image_inspector
    inspector.set_placement(white)
    window.design_canvas.set_placement(white)
    application.processEvents()

    inspector.flip_h.setChecked(True)
    inspector.rotation_spin.setValue(90.0)
    inspector.flip_v.setChecked(True)
    application.processEvents()

    assert black.flip_horizontal is True
    assert black.flip_vertical is True
    assert black.rotation_deg == pytest.approx(90.0)
    window.hide()


def test_another_painting_is_left_alone(tmp_path):
    """Only the same import is a pair; a different one must not be dragged in."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from tools.gui_mockup import build_window

    application = QApplication.instance() or QApplication([])
    _ = application
    path = glyph(tmp_path)

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    board = window.design_canvas.board
    board.outline_ready = True

    first = ImagePlacement(
        path=path, processed_path=path, width_mm=40, height_mm=40,
        layer="顶层丝印层", tone="white", import_id="imp-1",
    )
    second = ImagePlacement(
        path=path, processed_path=path, width_mm=40, height_mm=40,
        layer="底层丝印层", tone="white", import_id="imp-2",
    )
    board.placements += [first, second]

    inspector = window.image_inspector
    inspector.set_placement(first)
    window.design_canvas.set_placement(first)
    application.processEvents()

    inspector.flip_h.setChecked(True)
    application.processEvents()

    assert second.flip_horizontal is False
    assert second.rotation_deg == pytest.approx(0.0)
    window.hide()
