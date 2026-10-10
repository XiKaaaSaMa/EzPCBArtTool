"""Undo for the design canvas, kept as whole-document snapshots.

The editor's operations all mutate one small dataclass tree -- a board outline
plus a list of placements -- and that tree is already round-trippable through
:mod:`pcbart.draft` for saving. Snapshotting it is therefore both the shortest
path and the one that cannot drift: a new field on a placement is covered by
undo the day it is covered by save, with no third list of fields to remember.

The alternative -- one command object per operation, each knowing how to invert
itself -- is what a large editor needs because its documents are too big to
copy. This one is not: a board is a few kilobytes of numbers (the pictures
themselves are files, referenced by path). Measured on a 60-placement document,
a snapshot is about 40 KB and restoring it is well under a millisecond, so the
cheaper-to-write design is also the one that behaves correctly for free.

History is **per board**. Each :class:`Project` carries its own stack, because
undo in a tabbed editor means "undo what I just did *here*"; a window-wide stack
would let Ctrl+Z in one tab silently rewrite another.

**What is in scope.** Every edit that changes the document -- importing,
moving, resizing, rotating, flipping, deleting, pasting, and the property pane's
numbers -- is recorded. Two things deliberately are not:

* *Layer visibility*, which is view state. Undoing a move must not also switch
  a hidden layer back on.
* *Line-art tuning* (the preprocessing sliders). Tuning rewrites one shared
  cache file whose path does not change, so there is no document state to
  restore, and a slider drag emits a result per tick -- recording them would
  bury the real edits. Undoing an earlier edit therefore leaves the tuned
  artwork alone, which is the behaviour that matches what the user sees: the
  cache path in the snapshot still names the file in its current state.
"""

from __future__ import annotations

import time

from .artboard import ArtBoard
from .draft import apply_board_dict, board_to_dict

__all__ = ["UndoStack", "UNDO_LIMIT", "COALESCE_SECONDS"]

#: How many undo steps are kept. Deep enough that a working session never
#: reaches it, shallow enough that a pathological loop cannot grow the process
#: without bound. Snapshots are small, so this is a safety rail rather than a
#: memory budget.
UNDO_LIMIT = 200

#: How long a run of same-kind edits keeps merging into one step.
#:
#: Without this, dragging a spinbox arrow forty times is forty undo steps and
#: ``Ctrl+Z`` appears to do nothing. The window is generous enough to cover a
#: continuous gesture and short enough that two deliberate edits are separate
#: unless the user really did them back to back.
COALESCE_SECONDS = 0.8


class UndoStack:
    """A bounded history of document states, newest last.

    The stack always holds at least the state the document was in when it was
    created. ``undo`` therefore never has to special-case "nothing recorded
    yet": it either has something to go back to or reports that the document is
    already at its earliest recorded state.
    """

    def __init__(self, board: ArtBoard, *, limit: int = UNDO_LIMIT) -> None:
        self._limit = max(1, int(limit))
        self._board = board
        self._states: list[dict] = [self._snapshot()]
        self._index = 0
        self._run_key: str | None = None
        self._run_at = 0.0

    # ------------------------------------------------------------- recording

    def _snapshot(self) -> dict:
        """The document, minus the parts undo should not touch.

        Layer visibility is dropped: it is view state, and restoring it would
        mean undoing a move also switched a layer the user had just hidden.
        """
        data = board_to_dict(self._board)
        data.pop("layers", None)
        return data

    def reset(self, board: ArtBoard | None = None) -> None:
        """Start over, as when a different document is shown in this slot."""
        if board is not None:
            self._board = board
        self._states = [self._snapshot()]
        self._index = 0
        self._run_key = None
        self._run_at = 0.0

    def record(self, run_key: str | None = None, *, now: float | None = None) -> None:
        """Note that the document has changed.

        Called *after* an edit, so the state being pushed is the result. The
        state before the edit is already on the stack, which is what ``undo``
        returns to.

        ``run_key`` merges a continuous run of related edits into one step: a
        spinbox drag, a held arrow key, typing a value one digit at a time.
        Repeats of the same key within :data:`COALESCE_SECONDS` replace the top
        of the stack instead of adding to it, so one ``Ctrl+Z`` takes back the
        whole gesture. A different key, an expired window, or ``None`` starts a
        new step.

        ``now`` is injectable so the coalescing rule can be tested without
        sleeping through a real clock.
        """
        snapshot = self._snapshot()
        if snapshot == self._states[self._index]:
            # Nothing actually moved. Re-recording is harmless but it would
            # fill the history with identical steps, so a held arrow key would
            # push the real edit out of reach.
            return

        stamp = time.monotonic() if now is None else float(now)
        if (
            run_key is not None
            and run_key == self._run_key
            and self.can_undo
            and stamp - self._run_at <= COALESCE_SECONDS
        ):
            self._states[self._index] = snapshot
            self._run_at = stamp
            return

        del self._states[self._index + 1 :]
        self._states.append(snapshot)
        # ``limit`` counts *steps back*, so the state list may hold one more
        # than that: the oldest entry is the position the last undo lands on.
        while len(self._states) - 1 > self._limit:
            del self._states[0]
        self._index = len(self._states) - 1
        self._run_key = run_key
        self._run_at = stamp

    def _restore(self, snapshot: dict) -> None:
        apply_board_dict(self._board, snapshot, include_layers=False)

    # -------------------------------------------------------------- stepping

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    def undo(self) -> bool:
        """Step back one state. Returns whether anything changed."""
        if not self.can_undo:
            return False
        self._index -= 1
        self._restore(self._states[self._index])
        # The next edit starts a fresh step: continuing the run that was just
        # undone would make the new edit merge into the state the user undid
        # *to*, and a second Ctrl+Z would then skip straight past it.
        self._run_key = None
        return True

    @property
    def depth(self) -> int:
        """How many steps back are available; 0 disables the menu entry."""
        return self._index
