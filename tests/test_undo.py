"""Tests for Ctrl+Z.

Undo is held to the standard the editing tests set: the assertion is about what
the document looks like afterwards, not that some bookkeeping moved. A stack
that records steps but restores the wrong state passes any test that only checks
``can_undo``, and fails the user.

Two properties get specific attention because they are the ones a plausible
implementation gets wrong:

* **The live objects change.** Panels hold references to placements and to the
  layer stack, so restoring has to mutate those objects in place. A version that
  rebuilt the board would look right in a document comparison and leave every
  pane editing a document that is no longer on screen.
* **A gesture is one step.** A drag and a held arrow key each emit many changes;
  folding them into one undo step is what makes Ctrl+Z feel like Ctrl+Z rather
  than like walking backwards one pixel at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from pcbart.artboard import ArtBoard, BoardOutlineSpec, ImagePlacement  # noqa: E402
from pcbart.draft import apply_board_dict, board_from_dict, board_to_dict  # noqa: E402
from pcbart.undo import UndoStack  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def make_board(**placement_fields) -> ArtBoard:
    fields = {
        "path": "",
        "x_mm": 10.0,
        "y_mm": 5.0,
        "width_mm": 20.0,
        "height_mm": 20.0,
    }
    fields.update(placement_fields)
    return ArtBoard(
        outline=BoardOutlineSpec(width_mm=60.0, height_mm=60.0),
        placements=[ImagePlacement(**fields)],
        outline_ready=True,
    )


# --------------------------------------------------------------------- stack


def test_a_fresh_stack_has_nothing_to_undo():
    stack = UndoStack(make_board())
    assert not stack.can_undo
    assert stack.depth == 0
    assert stack.undo() is False


def test_undo_restores_the_position_an_edit_changed():
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 42.0
    stack.record()

    assert stack.undo() is True
    assert board.placements[0].x_mm == 10.0


def test_recording_without_a_change_does_not_add_a_step():
    board = make_board()
    stack = UndoStack(board)

    stack.record()
    stack.record()

    assert stack.depth == 0
    assert not stack.can_undo


def test_undo_is_bounded_by_the_limit():
    board = make_board()
    stack = UndoStack(board, limit=3)

    for step in range(6):
        board.placements[0].x_mm = float(step)
        stack.record()

    assert stack.depth == 3
    # Three steps back lands on the oldest state that survived, not on x = 0.
    for _ in range(3):
        stack.undo()
    assert board.placements[0].x_mm == 2.0
    assert not stack.can_undo


def test_an_edit_after_undo_discards_the_redo_tail():
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 20.0
    stack.record()
    board.placements[0].x_mm = 30.0
    stack.record()
    stack.undo()

    board.placements[0].x_mm = 99.0
    stack.record()

    assert stack.depth == 2
    assert board.placements[0].x_mm == 99.0
    stack.undo()
    assert board.placements[0].x_mm == 20.0


# --------------------------------------------------------------- coalescing


def test_a_run_of_edits_sharing_a_key_is_one_step():
    board = make_board()
    stack = UndoStack(board)

    for step in range(1, 6):
        board.placements[0].x_mm = float(step)
        stack.record("inspector:x", now=1000.0)

    assert stack.depth == 1
    stack.undo()
    assert board.placements[0].x_mm == 10.0


def test_a_pause_inside_a_run_starts_a_new_step():
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 20.0
    stack.record("inspector:x", now=1000.0)
    board.placements[0].x_mm = 30.0
    stack.record("inspector:x", now=1000.0 + 5.0)

    assert stack.depth == 2


def test_a_different_key_starts_a_new_step():
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 20.0
    stack.record("inspector:x", now=1000.0)
    board.placements[0].width_mm = 25.0
    stack.record("inspector:width", now=1000.0)

    assert stack.depth == 2


def test_editing_after_an_undo_starts_a_new_run():
    """The run key must not survive an undo, or the new edit merges away.

    Continuing the run would replace the state the user just came back to, so a
    second Ctrl+Z would step straight past it.
    """
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 20.0
    stack.record("inspector:x", now=1000.0)
    stack.undo()

    board.placements[0].x_mm = 55.0
    stack.record("inspector:x", now=1000.0)

    assert stack.depth == 1
    stack.undo()
    assert board.placements[0].x_mm == 10.0


# ------------------------------------------------------------ live objects


def test_restoring_mutates_the_placement_the_panel_is_holding():
    """A panel's reference has to keep pointing at a real, current object.

    Panels hold the placement itself rather than an index, so an undo that
    built a fresh list would leave the inspector editing an object that is no
    longer on the board -- the edit would appear to do nothing.
    """
    board = make_board()
    stack = UndoStack(board)
    held = board.placements[0]

    held.x_mm = 77.0
    stack.record()
    stack.undo()

    assert board.placements[0] is held
    assert held.x_mm == 10.0


def test_restoring_keeps_the_layer_stack_object_identity():
    """The image inspector holds the stack, so undo must not swap it out."""
    board = make_board()
    stack = UndoStack(board)
    held = board.layers

    board.placements[0].x_mm = 42.0
    stack.record()
    stack.undo()

    assert board.layers is held


def test_undo_does_not_switch_layer_visibility_back_on():
    """Layer visibility is view state, not work.

    Undoing a move must not also reveal a layer the user hid in between.
    """
    board = make_board()
    stack = UndoStack(board)

    board.placements[0].x_mm = 42.0
    stack.record()

    board.layers.set_visible("top_copper", False)
    stack.undo()

    assert board.placements[0].x_mm == 10.0
    assert board.layers.is_visible("top_copper") is False


def test_a_deleted_placement_comes_back():
    board = make_board()
    stack = UndoStack(board)

    board.placements.clear()
    stack.record()
    assert not board.placements

    stack.undo()
    assert len(board.placements) == 1
    assert board.placements[0].x_mm == 10.0


def test_outline_changes_are_undone():
    board = make_board()
    stack = UndoStack(board)

    board.outline.width_mm = 120.0
    board.outline.shape = "circle"
    stack.record()

    stack.undo()
    assert board.outline.width_mm == 60.0
    assert board.outline.shape == "rect"


# ------------------------------------------------------- snapshot plumbing


def test_apply_board_dict_writes_into_the_existing_board():
    board = make_board()
    snapshot = board_to_dict(board)

    board.placements[0].x_mm = 99.0
    board.placements[0].width_mm = 3.0
    apply_board_dict(board, snapshot)

    assert board.placements[0].x_mm == 10.0
    assert board.placements[0].width_mm == 20.0


def test_apply_board_dict_can_leave_layers_alone():
    board = make_board()
    snapshot = board_to_dict(board)

    board.layers.set_visible("top_copper", False)
    apply_board_dict(board, snapshot, include_layers=False)

    assert board.layers.is_visible("top_copper") is False


def test_a_snapshot_restores_through_the_undo_stack_round_trip():
    """The snapshot format is the draft format, so it round-trips by design."""
    board = make_board(rotation_deg=90.0, flip_horizontal=True, layer="底层铜层")
    stack = UndoStack(board)

    board.placements[0].rotation_deg = 0.0
    board.placements[0].flip_horizontal = False
    board.placements[0].layer = "顶层丝印层"
    stack.record()
    stack.undo()

    restored = board_from_dict(board_to_dict(board))
    assert restored.placements[0].rotation_deg == 90.0
    assert restored.placements[0].flip_horizontal is True
    assert restored.placements[0].layer == "底层铜层"


# ------------------------------------------------------------- the canvas


@pytest.fixture
def canvas(app):
    from tools.design_canvas import DesignCanvas

    widget = DesignCanvas()
    widget.board = make_board()
    widget.history = UndoStack(widget.board)
    widget.selected = widget.board.placements[0]
    yield widget
    widget.deleteLater()


def test_canvas_nudge_is_undoable(canvas):
    before = canvas.board.placements[0].x_mm

    assert canvas.nudge_selection(5.0, 0.0) is True
    assert canvas.board.placements[0].x_mm == before + 5.0

    assert canvas.undo() is True
    assert canvas.board.placements[0].x_mm == before


def test_canvas_rotate_is_undoable(canvas):
    canvas.rotate_selection(90.0)
    assert canvas.board.placements[0].rotation_deg == 90.0

    canvas.undo()
    assert canvas.board.placements[0].rotation_deg == 0.0


def test_canvas_flip_is_undoable(canvas):
    canvas.flip_selection(horizontal=True)
    assert canvas.board.placements[0].flip_horizontal is True

    canvas.undo()
    assert canvas.board.placements[0].flip_horizontal is False


def test_canvas_centre_is_undoable(canvas):
    canvas.centre_selection()
    assert canvas.board.placements[0].x_mm == 0.0

    canvas.undo()
    assert canvas.board.placements[0].x_mm == 10.0


def test_canvas_duplicate_is_undoable(canvas):
    canvas.duplicate_selection()
    assert len(canvas.board.placements) == 2

    canvas.undo()
    assert len(canvas.board.placements) == 1


def test_canvas_delete_is_undoable(canvas):
    canvas.delete_selection()
    assert not canvas.board.placements

    canvas.undo()
    assert len(canvas.board.placements) == 1


def test_canvas_paste_is_undoable(canvas):
    canvas.copy_selection()
    canvas.paste()
    assert len(canvas.board.placements) == 2

    canvas.undo()
    assert len(canvas.board.placements) == 1


def test_a_whole_drag_is_one_undo_step(canvas):
    """A drag emits many motion events; one Ctrl+Z takes back the whole drag.

    The canvas records on release, so the assertion is on the history depth:
    simulating the events rather than calling the recorder is what makes this a
    test of the canvas rather than of the stack.
    """
    canvas.show()
    start = canvas.to_screen(10.0, 5.0)
    end = canvas.to_screen(25.0, 12.0)

    from PySide6.QtGui import QMouseEvent

    press = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        start,
        canvas.mapToGlobal(start.toPoint()),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    canvas.mousePressEvent(press)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        point = QPointF(
            start.x() + (end.x() - start.x()) * fraction,
            start.y() + (end.y() - start.y()) * fraction,
        )
        move = QMouseEvent(
            QMouseEvent.MouseMove,
            point,
            canvas.mapToGlobal(point.toPoint()),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        canvas.mouseMoveEvent(move)
    release = QMouseEvent(
        QMouseEvent.MouseButtonRelease,
        end,
        canvas.mapToGlobal(end.toPoint()),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    canvas.mouseReleaseEvent(release)

    assert canvas.board.placements[0].x_mm != 10.0
    assert canvas.history.depth == 1
    canvas.undo()
    assert canvas.board.placements[0].x_mm == 10.0


def test_ctrl_z_on_the_canvas_undoes(canvas):
    canvas.nudge_selection(3.0, 0.0)

    event = QKeyEvent(
        QKeyEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier
    )
    canvas.keyPressEvent(event)

    assert canvas.board.placements[0].x_mm == 10.0


def test_ctrl_shift_z_is_not_treated_as_undo(canvas):
    """Redo does not exist, so Ctrl+Shift+Z must not silently undo instead."""
    canvas.nudge_selection(3.0, 0.0)

    event = QKeyEvent(
        QKeyEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier
    )
    canvas.keyPressEvent(event)

    assert canvas.board.placements[0].x_mm == 13.0


def test_undo_drops_a_selection_whose_object_is_gone(canvas):
    """Redoing a delete must not leave the pane editing a vanished object."""
    canvas.delete_selection()
    assert canvas.selected is None

    canvas.undo()
    # The restored placement is a different object from the one that was
    # selected before the delete; selecting nothing is the honest outcome.
    assert canvas.board.image is not None
    assert canvas.selected is None or canvas.selected in canvas.board.placements


def test_history_changed_fires_so_the_menu_can_follow(canvas):
    seen = []
    canvas.history_changed.connect(lambda: seen.append(canvas.can_undo()))

    canvas.nudge_selection(1.0, 0.0)
    canvas.undo()

    assert seen == [True, False]


# ---------------------------------------------------------------- the window


def test_the_edit_menu_offers_undo_and_greys_it_out_until_there_is_one(app):
    from gui_mockup import build_window
    from pcbart.undo import UndoStack as _UndoStack

    window = build_window()
    try:
        canvas = window.design_canvas
        canvas.board = make_board()
        canvas.selected = canvas.board.placements[0]
        canvas.history = _UndoStack(canvas.board)
        canvas.selection_changed.emit(canvas.selected)

        undo_action = next(
            action
            for action in window.menuBar().actions()[1].menu().actions()
            if action.text() == "撤销"
        )
        assert undo_action.shortcut().toString() == "Ctrl+Z"
        assert not undo_action.isEnabled()

        canvas.nudge_selection(2.0, 0.0)
        assert undo_action.isEnabled()

        undo_action.trigger()
        assert not undo_action.isEnabled()
    finally:
        import conftest

        conftest.release_window(window)


def test_switching_tabs_switches_the_history(app):
    """Ctrl+Z in a tab must undo that tab's edit, not the other one's."""
    from gui_mockup import activate_project, build_window
    from pcbart.undo import UndoStack as _UndoStack

    window = build_window()
    try:
        canvas = window.design_canvas
        first_board = make_board()
        canvas.board = first_board
        canvas.selected = first_board.placements[0]
        canvas.history = _UndoStack(first_board)
        window.workspace.active.board = first_board

        # ``new`` leaves the new tab active without re-pointing the panes, so
        # the test walks the tabs the way the user does: back to the first
        # board, then forward to the second.
        second_board = make_board()
        second_project = window.workspace.new(second_board)
        activate_project(window, 0)
        activate_project(window, window.workspace.index_of(second_project))
        assert canvas.board is second_board

        # Edit in the second tab, so the step lands in the second board's own
        # history.
        canvas.selected = canvas.board.placements[0]
        canvas.nudge_selection(30.0, 0.0)
        assert second_board.placements[0].x_mm == 40.0
        assert canvas.can_undo()

        # Back to the first board: it has no steps of its own, and the second
        # board's step must not be reachable from here.
        activate_project(window, 0)
        assert canvas.board is first_board
        assert not canvas.can_undo()
        assert first_board.placements[0].x_mm == 10.0

        activate_project(window, window.workspace.index_of(second_project))
        assert canvas.board is second_board
        assert canvas.can_undo()
        canvas.undo()
        assert second_board.placements[0].x_mm == 10.0
    finally:
        import conftest

        conftest.release_window(window)


def test_history_survives_a_tab_round_trip_on_the_opening_board(app):
    """The opening board's history must not be abandoned by the first switch.

    The canvas builds a history when it is constructed, before any project
    exists. If the workspace created a second one, everything edited before the
    first tab switch would stop being undoable -- and only then, which is the
    kind of bug that shows up long after the edit.
    """
    from gui_mockup import activate_project, build_window

    window = build_window()
    try:
        canvas = window.design_canvas
        # Use the board the canvas was built with, which is the one the opening
        # project adopts. Replacing it here would leave the canvas's own history
        # describing a different document, which is not what the app does.
        board = canvas.board
        board.placements.append(
            ImagePlacement(path="", x_mm=10.0, y_mm=5.0, width_mm=20.0, height_mm=20.0)
        )
        canvas.selected = board.placements[0]
        canvas.history.reset(board)
        window.workspace  # created here, adopting the canvas's history

        canvas.nudge_selection(7.0, 0.0)
        assert canvas.can_undo()

        other = window.workspace.new(make_board())
        activate_project(window, window.workspace.index_of(other))
        activate_project(window, 0)

        assert canvas.board is board
        assert canvas.can_undo()
        canvas.undo()
        assert board.placements[0].x_mm == 10.0
    finally:
        import conftest

        conftest.release_window(window)


def test_the_property_pane_edit_is_undoable(app):
    """The inspector writes to the same placements, so it shares the history."""
    from gui_mockup import build_window

    window = build_window()
    try:
        canvas = window.design_canvas
        canvas.board = make_board()
        canvas.selected = canvas.board.placements[0]
        canvas.history = UndoStack(canvas.board)
        canvas.selection_changed.emit(canvas.selected)

        inspector = window.image_inspector
        inspector.set_placement(canvas.selected)
        inspector.x_spin.setValue(33.0)

        assert canvas.board.placements[0].x_mm == 33.0
        assert canvas.can_undo()

        canvas.undo()
        assert canvas.board.placements[0].x_mm == 10.0
    finally:
        import conftest

        conftest.release_window(window)
