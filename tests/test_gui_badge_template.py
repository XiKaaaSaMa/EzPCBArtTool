"""Tests for applying the 通行证 template and for editing its content on canvas.

Two requests meet here. The user asked for the blank template's content to
appear on the canvas *and be editable*; that means the board panel has to plant
it, the canvas has to draw and pick it, and the document has to remember it
through save, load and undo. A gap in any one of those leaves a feature that
looks finished on screen and loses work.

The canvas tests drag real mouse events rather than calling the model, because
the failure this guards is precisely "the model has it and the canvas does not
know" -- the shape of the 旋转/翻转 defect recorded in PROGRESS, where three
controls stored values that nothing drew.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pcbart import badge_card  # noqa: E402
from pcbart.artboard import ArtBoard, OutlineShape  # noqa: E402
from pcbart.draft import apply_board_dict, board_from_dict, board_to_dict  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app):
    from conftest import release_window
    from tools.gui_mockup import build_window

    win = build_window()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    yield win
    release_window(win)


def badge_board() -> ArtBoard:
    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    return board


# ------------------------------------------------------- applying the template


def test_choosing_the_template_plants_its_content(window):
    """The user's first request: the blank card arrives with everything on it.

    A template that only set the outline would leave the card blank, and the
    wording -- which is most of what a 通行证 is -- would have to be typed in
    again by hand.
    """
    from tools.gui_mockup import BoardSetupPanel

    board = window.design_canvas.board
    panel = BoardSetupPanel(window, board)
    assert board.texts == []

    apply_template(panel, board)
    window.design_canvas.board = board
    assert board.outline.shape == OutlineShape.TEMPLATE
    assert len(board.texts) == len(badge_card.BADGE_TEXTS)
    assert len(board.cutouts) == len(badge_card.BADGE_CUTOUTS)


def apply_template(panel, board):
    """Apply 通行证 the way a user does: pick 套用模板, then the card."""
    from tools.gui_mockup import BoardSetupPanel

    panel.shape_combo.setCurrentIndex(
        panel.shape_combo.findText(BoardSetupPanel.TEMPLATE_OPTION)
    )
    panel.template_combo.setCurrentIndex(panel.template_combo.findText("通行证"))


def test_re_applying_the_template_restores_its_wording(window):
    """Choosing the card again is the way back to a known state.

    The path used here is the user's: shape → preset → shape → 套用模板. Simply
    re-selecting the template in its own combo cannot be tested because there
    is only one card registered, and Qt emits nothing when a one-item combo is
    set to the index it already has.
    """
    from tools.gui_mockup import BoardSetupPanel

    board = window.design_canvas.board
    panel = BoardSetupPanel(window, board)
    apply_template(panel, board)
    board.texts[0].text = "被我改坏了"

    panel.shape_combo.setCurrentIndex(
        panel.shape_combo.findText(OutlineShape.LABELS[OutlineShape.RECT])
    )
    apply_template(panel, board)
    assert board.texts[0].text == badge_card.BADGE_TEXTS[0].text


def test_re_applying_the_template_keeps_the_chosen_operator(window):
    """The operator is a property of *this card*, not of the template.

    Dropping it would blank the name every time the shape was re-selected, and
    the user would have to search for the same person again.
    """
    from tools.gui_mockup import BoardSetupPanel

    board = window.design_canvas.board
    panel = BoardSetupPanel(window, board)
    apply_template(panel, board)
    board.operator_name = "鸿雪"
    board.operator_number = "R145"

    panel.template_combo.setCurrentIndex(0)
    panel.template_combo.setCurrentIndex(panel.template_combo.findText("通行证"))
    assert board.operator_name == "鸿雪"
    assert board.operator_number == "R145"


def test_switching_away_from_the_template_does_not_leave_the_wording(window):
    """A circle is not a card, and its own panel should not list card lines.

    The text is a property of the template, so choosing a preset shape has to
    drop it -- otherwise the content panel would offer to edit wording that the
    board no longer carries.
    """
    from tools.gui_mockup import BoardSetupPanel

    board = window.design_canvas.board
    panel = BoardSetupPanel(window, board)
    apply_template(panel, board)
    assert board.texts and board.cutouts

    panel.shape_combo.setCurrentIndex(
        panel.shape_combo.findText(OutlineShape.LABELS[OutlineShape.RECT])
    )
    # Rect is the first shape; whatever index it lands on, the point is the
    # card's content is gone.
    assert board.texts == []
    assert board.cutouts == []


# ------------------------------------------------------------- on the canvas


def to_screen(canvas, x_mm, y_mm) -> QPointF:
    point = canvas.to_screen(x_mm, y_mm)
    return QPointF(point.x(), point.y())


def click(canvas, app, x_mm, y_mm, *, drag_to=None) -> None:
    """Press (and optionally drag) on the canvas at a board coordinate."""
    start = to_screen(canvas, x_mm, y_mm)
    press = QMouseEvent(
        QMouseEvent.MouseButtonPress, start, canvas.mapToGlobal(start),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    )
    canvas.mousePressEvent(press)
    if drag_to is not None:
        end = to_screen(canvas, *drag_to)
        move = QMouseEvent(
            QMouseEvent.MouseMove, end, canvas.mapToGlobal(end),
            Qt.NoButton, Qt.LeftButton, Qt.NoModifier,
        )
        canvas.mouseMoveEvent(move)
    finish = to_screen(canvas, *(drag_to or (x_mm, y_mm)))
    release = QMouseEvent(
        QMouseEvent.MouseButtonRelease, finish, canvas.mapToGlobal(finish),
        Qt.LeftButton, Qt.NoButton, Qt.NoModifier,
    )
    canvas.mouseReleaseEvent(release)
    app.processEvents()


@pytest.fixture
def card_canvas(window, app):
    from pcbart.undo import UndoStack

    canvas = window.design_canvas
    board = badge_board()
    canvas.board = board
    # The history has to be rebound with the board, exactly as ``_show_project``
    # does on a tab switch: a stack built for the previous document would
    # restore *that* document and leave this one untouched.
    canvas.set_history(UndoStack(board))
    canvas.selected = None
    canvas.fit_view()
    app.processEvents()
    return canvas


def test_clicking_a_line_selects_it(card_canvas, app):
    """The wording is pickable, so it can be dragged or opened for editing."""
    item = card_canvas.board.texts[0]
    # A little way into the line's box, which starts at its anchor.
    click(card_canvas, app, item.x_mm + 1.0, item.y_mm + 0.5)
    assert card_canvas.selected is item


def test_a_line_can_be_dragged(card_canvas, app):
    """Dragging is how the user moves wording without opening the panel."""
    item = card_canvas.board.texts[0]
    before = (item.x_mm, item.y_mm)
    click(
        card_canvas, app, item.x_mm + 1.0, item.y_mm + 0.5,
        drag_to=(item.x_mm + 6.0, item.y_mm + 3.5),
    )
    assert item.x_mm == pytest.approx(before[0] + 5.0, abs=0.2)
    assert item.y_mm == pytest.approx(before[1] + 3.0, abs=0.2)


def test_dragging_a_line_is_one_undo_step(card_canvas, app):
    """One undo returns the whole drag, not one motion event's worth of it.

    A drag emits dozens of move events; recording each would make Ctrl+Z appear
    to do nothing for the first several presses.
    """
    item = card_canvas.board.texts[0]
    before = (item.x_mm, item.y_mm)
    click(
        card_canvas, app, item.x_mm + 1.0, item.y_mm + 0.5,
        drag_to=(item.x_mm + 8.0, item.y_mm + 4.0),
    )
    assert item.x_mm != pytest.approx(before[0])
    assert card_canvas.undo() is True
    assert item.x_mm == pytest.approx(before[0], abs=0.01)
    assert item.y_mm == pytest.approx(before[1], abs=0.01)


def test_a_line_has_no_resize_handles(card_canvas, app):
    """Text is sized by its font size; an edge handle has nothing to drag.

    Offering handles would let the user drag an edge that changes nothing, or
    worse, silently change the size while the panel still shows the old one.
    """
    item = card_canvas.board.texts[0]
    click(card_canvas, app, item.x_mm + 1.0, item.y_mm + 0.5)
    assert card_canvas.handle_centres() == {}


def test_clicking_bare_canvas_clears_the_selection(card_canvas, app):
    item = card_canvas.board.texts[0]
    click(card_canvas, app, item.x_mm + 1.0, item.y_mm + 0.5)
    assert card_canvas.selected is item
    click(card_canvas, app, -24.0, -48.0)
    assert card_canvas.selected is None


def test_the_cutout_is_drawn_where_the_model_says(card_canvas):
    """The canvas asks the cutout for its ring, so the two cannot disagree."""
    slot = card_canvas.board.cutouts[0]
    ring = [card_canvas.to_screen(x, y) for x, y in slot.ring()]
    xs = [point.x() for point in ring]
    ys = [point.y() for point in ring]
    centre = card_canvas.to_screen(slot.x_mm, slot.y_mm)
    assert (min(xs) + max(xs)) / 2.0 == pytest.approx(centre.x(), abs=1.0)
    assert (min(ys) + max(ys)) / 2.0 == pytest.approx(centre.y(), abs=1.0)


def test_a_rotated_line_is_drawn_along_its_own_axis(card_canvas):
    """The 270-degree strips must be vertical, not lying across the card.

    A drawn box that ignored the rotation would still *contain* the text's
    anchor, so every numeric check would pass while the card looked wrong.
    """
    vertical = next(
        item for item in card_canvas.board.texts if item.rotation_deg == 270.0
    )
    corners = [card_canvas.to_screen(x, y) for x, y in vertical.corners()]
    width = max(point.x() for point in corners) - min(point.x() for point in corners)
    height = max(point.y() for point in corners) - min(point.y() for point in corners)
    assert height > width * 2.0, "竖排文字的画布选区没有竖起来"


# ------------------------------------------------------- through saved state


def test_text_and_cutouts_survive_a_draft_round_trip():
    board = badge_board()
    board.operator_name = "鸿雪"
    board.operator_number = "R145"
    board.texts[3].text = "MY OWN WORDING"
    restored = board_from_dict(board_to_dict(board))
    assert [item.text for item in restored.texts] == [
        item.text for item in board.texts
    ]
    assert restored.texts[3].text == "MY OWN WORDING"
    assert restored.cutouts[0].width_mm == pytest.approx(
        board.cutouts[0].width_mm
    )
    assert restored.operator_name == "鸿雪"
    assert restored.operator_number == "R145"


def test_undoing_a_text_edit_restores_the_previous_wording():
    board = badge_board()
    snapshot = board_to_dict(board)
    board.texts[0].text = "CHANGED"
    apply_board_dict(board, snapshot)
    assert board.texts[0].text == badge_card.BADGE_TEXTS[0].text


def test_undo_keeps_the_same_text_objects():
    """A panel or the canvas may hold one of these; swapping breaks them.

    Undo that replaced the list would leave the content panel editing an object
    that is no longer in the document, so the next keystroke would change
    something nothing draws.
    """
    board = badge_board()
    snapshot = board_to_dict(board)
    original = board.texts[0]
    board.texts[0].text = "CHANGED"
    apply_board_dict(board, snapshot)
    assert board.texts[0] is original


def test_dropping_a_cutout_survives_undo():
    board = badge_board()
    snapshot = board_to_dict(board)
    board.cutouts.clear()
    apply_board_dict(board, snapshot)
    assert len(board.cutouts) == len(badge_card.BADGE_CUTOUTS)


def test_the_fingerprint_notices_a_text_edit():
    """The dirty marker is what makes the exit prompt appear.

    A field that saves but does not count would let the user close the window
    and lose the edit with no warning.
    """
    from pcbart.draft import fingerprint

    board = badge_board()
    before = fingerprint(board)
    board.texts[0].text = "CHANGED"
    assert fingerprint(board) != before


def test_the_fingerprint_notices_an_operator_change():
    from pcbart.draft import fingerprint

    board = badge_board()
    before = fingerprint(board)
    board.operator_name = "鸿雪"
    assert fingerprint(board) != before
