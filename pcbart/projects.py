"""The boards that are open at once, and which one is showing.

The window used to *be* the document: one board, one draft path, one saved
state. Starting a second board therefore meant closing the program, because
there was nowhere for the first one to go. A **workspace** holds several
projects side by side, each with its own document and its own saved state, so
the window only has to know which one is on screen.

This is the model half: it owns no widgets and knows nothing about the tab bar,
which keeps "which projects are open" testable without a running application.
The window keeps a reference to each board, so switching tabs is a matter of
re-pointing the panes -- the documents themselves never move.

**Dirty tracking stays a fingerprint.** Each project remembers the fingerprint
of the document as it was last written out, so "has this changed" is answered
by comparing the document rather than by trusting every edit path to announce
itself. A tab that forgets to announce an edit is a tab that loses work.
"""

from __future__ import annotations

from .artboard import ArtBoard
from .draft import fingerprint
from .undo import UndoStack

__all__ = ["Project", "Workspace", "UNTITLED"]


UNTITLED = "未命名"
"""What a project is called before it has ever been written to a file."""


class Project:
    """One open board: its document, its file, and where the view was left.

    ``draft_path`` is the editable session in the in-app library (``.pcbdraft``);
    ``project_path`` and ``order_path`` are the two products written by
    *另存为工程* and *另存为下单文件*. They are kept separately because they are
    different things -- one is what the user edits, the others are what they
    hand to the editor and to the fab -- and remembering the last target means
    the second save of the same kind does not start from an empty dialog again.
    """

    def __init__(
        self,
        board: ArtBoard | None = None,
        *,
        title: str = "",
        draft_path: str | None = None,
        saved_fingerprint: str | None = None,
    ) -> None:
        self.board = ArtBoard() if board is None else board
        self.title = title or UNTITLED
        self.draft_path = None if draft_path is None else str(draft_path)
        self.project_path: str | None = None
        self.order_path: str | None = None

        #: Fingerprint of the document as it was last written -- by any of the
        #: three saves. A brand-new board counts as saved: it is identical to
        #: what a save would write right now, so marking it dirty would put a
        #: warning dot on a document the user has not touched yet.
        self.saved_fingerprint = (
            fingerprint(self.board)
            if saved_fingerprint is None
            else saved_fingerprint
        )

        #: ``(zoom, pan_x_mm, pan_y_mm)``, or ``None`` before the project has
        #: been shown. Restoring it means switching tabs does not throw away
        #: the place the user had zoomed to.
        self.view: tuple[float, float, float] | None = None

        #: This board's edit history. Per project rather than per window: undo
        #: means "take back what I did *here*", so switching tabs must not let
        #: Ctrl+Z rewrite a document that is not on screen. Built lazily so a
        #: project that is only ever exported never pays for a snapshot.
        self._history: UndoStack | None = None

    @property
    def history(self) -> UndoStack:
        """The undo history for this board."""
        if self._history is None:
            self._history = UndoStack(self.board)
        return self._history

    def adopt_history(self, history: UndoStack) -> None:
        """Take over a history that was already recording this board.

        The canvas builds one when it is constructed, before any project
        exists, so without this the opening board would have two stacks: the
        canvas recording into its own, and the project holding an empty one
        that takes over on the first tab switch. Everything edited in that
        first stretch would quietly stop being undoable.
        """
        self._history = history

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Project({self.title!r}, dirty={self.is_dirty()})"

    def is_dirty(self) -> bool:
        """Whether saving now would write something different."""
        return fingerprint(self.board) != self.saved_fingerprint

    def mark_saved(self) -> None:
        """Remember the document as it stands now as the written-out one."""
        self.saved_fingerprint = fingerprint(self.board)

    def is_unnamed(self) -> bool:
        """Whether the title is still the automatic 未命名 N."""
        return self.title == UNTITLED or self.title.startswith(f"{UNTITLED} ")

    def name_from_file(self, path: str | None) -> str:
        """Take the tab title from a file the project was just saved to.

        Only an automatically-named board is renamed. The board's name is its
        identity across the tab row, and overwriting it from whatever the last
        file happened to be called would make it jump: save the project as
        猫娘板, then save the order file as 猫娘板_下单, and the tab would
        silently change to the second one.
        """
        if not path or not self.is_unnamed():
            return self.title
        from pathlib import Path

        stem = Path(str(path)).stem.strip()
        if stem:
            self.title = stem
        return self.title

    def rename_from_project(self, path: str | None) -> str:
        """Take the title from a ``.epro2``, which names the *board*.

        Used by 另存为工程, where the user is explicitly naming the design
        rather than one of its output files.
        """
        if not path:
            return self.title
        from pathlib import Path

        stem = Path(str(path)).stem.strip()
        if stem:
            self.title = stem
        return self.title


class Workspace:
    """Every open project, in tab order, plus which one is active.

    There is always at least one project. Closing the last one leaves a fresh
    empty board behind rather than an empty window, because every pane in the
    application reads the document through the active project -- a window with
    no document would have to grow a second, permanent "nothing is open" state
    in every one of them.
    """

    def __init__(self, board: ArtBoard | None = None) -> None:
        self._projects: list[Project] = []
        self._index = 0
        self._untitled_count = 0
        # The first project takes the unnumbered name: numbering starts at the
        # second one, so "未命名 2" is not a thing the user can see without a
        # "未命名" having existed first.
        self.add(Project(board, title=self.next_untitled()))

    # ------------------------------------------------------------- reading

    def __len__(self) -> int:
        return len(self._projects)

    def __iter__(self):
        return iter(self._projects)

    def __getitem__(self, index: int) -> Project:
        return self._projects[index]

    @property
    def projects(self) -> list[Project]:
        """The open projects, in tab order."""
        return list(self._projects)

    @property
    def index(self) -> int:
        return self._index

    @property
    def active(self) -> Project:
        return self._projects[self._index]

    def index_of(self, project: Project) -> int:
        try:
            return self._projects.index(project)
        except ValueError:
            return -1

    def dirty(self) -> list[Project]:
        """Every project with changes that are not written down anywhere."""
        return [item for item in self._projects if item.is_dirty()]

    # ------------------------------------------------------------- editing

    def add(self, project: Project | None = None, *, activate: bool = True) -> Project:
        """Add a project. New tabs open next to the current one, as in a browser."""
        if project is None:
            project = Project(title=self.next_untitled())
        position = len(self._projects) if not self._projects else self._index + 1
        self._projects.insert(position, project)
        if activate or len(self._projects) == 1:
            self._index = position
        elif position <= self._index:
            self._index += 1
        return project

    def new(self, board: ArtBoard | None = None, *, title: str = "") -> Project:
        """A fresh board, opened as a new tab."""
        return self.add(Project(board, title=title or self.next_untitled()))

    def switch(self, index: int) -> Project:
        """Make ``index`` the active project. Returns it."""
        if not 0 <= index < len(self._projects):
            raise IndexError(f"没有第 {index} 个工程")
        self._index = index
        return self._projects[index]

    def close(self, index: int) -> Project:
        """Close one project and return the one that becomes active.

        The tab to the right takes over when there is one, and the tab to the
        left otherwise -- so closing a run of tabs walks the selection forward
        the way a browser does, rather than jumping back to the first tab.
        """
        if not 0 <= index < len(self._projects):
            raise IndexError(f"没有第 {index} 个工程")
        self._projects.pop(index)
        if not self._projects:
            self._projects.append(Project(title=self.next_untitled()))
            self._index = 0
        elif index < self._index:
            self._index -= 1
        else:
            self._index = min(index, len(self._projects) - 1)
        return self.active

    def close_others(self, index: int) -> Project:
        """Keep only ``index``, which becomes active."""
        if not 0 <= index < len(self._projects):
            raise IndexError(f"没有第 {index} 个工程")
        kept = self._projects[index]
        self._projects = [kept]
        self._index = 0
        return kept

    def move(self, source: int, target: int) -> None:
        """Move a tab, with the same "lands at ``target``" meaning Qt uses."""
        if source == target:
            return
        if not 0 <= source < len(self._projects):
            raise IndexError(f"没有第 {source} 个工程")
        active = self.active
        target = max(0, min(target, len(self._projects) - 1))
        project = self._projects.pop(source)
        self._projects.insert(target, project)
        # The active *project* stays active when tabs are dragged around, so
        # its index is looked up again rather than left pointing at whoever
        # happens to sit at the old number now.
        self._index = self._projects.index(active)

    def next_untitled(self) -> str:
        """The next free ``未命名 N`` name."""
        self._untitled_count += 1
        if self._untitled_count == 1:
            return UNTITLED
        return f"{UNTITLED} {self._untitled_count}"
