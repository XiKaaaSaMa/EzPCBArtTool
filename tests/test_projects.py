"""Tests for the set of open projects.

The behaviour that matters is the bookkeeping around *switching*: which
document is active after a close, whether a tab knows it has unwritten changes,
and whether the unnumbered first name stays reserved. None of it needs a
window, which is the point of keeping the model free of Qt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
)
from pcbart.projects import UNTITLED, Project, Workspace  # noqa: E402


def touched(board: ArtBoard, width: float = 55.0) -> ArtBoard:
    """One edit of the kind a user makes constantly."""
    board.outline_ready = True
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=width, height_mm=30.0
    )
    return board


# ----------------------------------------------------------------- projects


def test_a_brand_new_project_is_not_dirty():
    """A fresh board is already identical to what saving would write."""
    assert Project().is_dirty() is False


def test_an_edit_makes_the_project_dirty():
    project = Project()
    touched(project.board)

    assert project.is_dirty() is True


def test_marking_saved_settles_the_fingerprint():
    project = Project()
    touched(project.board)
    project.mark_saved()

    assert project.is_dirty() is False
    touched(project.board, width=70.0)
    assert project.is_dirty() is True


def test_the_tab_title_comes_from_the_file_it_was_saved_to():
    project = Project()
    assert project.title == UNTITLED

    assert project.name_from_file(r"D:\作品\猫娘板.epro2") == "猫娘板"
    assert project.title == "猫娘板"


def test_a_second_save_does_not_rename_the_board():
    """Save the project as 猫娘板, then the order file as 猫娘板_下单, and the
    tab must not silently become the second one -- the name is the board's
    identity across the tab row, not whatever the last file was called."""
    project = Project()
    project.name_from_file(r"D:\作品\猫娘板.epro2")

    project.name_from_file(r"D:\下单\猫娘板_下单.zip")

    assert project.title == "猫娘板"


def test_only_an_automatic_name_is_taken_from_a_file():
    project = Project(title="我自己起的")
    project.name_from_file(r"D:\作品\别的名字.epro2")

    assert project.title == "我自己起的"
    assert project.is_unnamed() is False


def test_a_numbered_untitled_board_still_counts_as_unnamed():
    project = Project(title="未命名 3")

    assert project.is_unnamed() is True
    project.name_from_file(r"D:\作品\猫娘板.epro2")
    assert project.title == "猫娘板"


def test_naming_a_project_always_wins():
    """另存为工程 is the user naming the *board*, so it overrides whatever the
    tab was called before."""
    project = Project(title="未命名 2")
    project.name_from_file(r"D:\下单\临时.zip")
    assert project.title == "临时"

    project.rename_from_project(r"D:\工程\正式名字.epro2")
    assert project.title == "正式名字"


def test_a_project_remembers_each_of_its_three_files():
    """The editable session and the two products are separate things: the next
    save of the same kind must not have to be told where it went again."""
    project = Project()
    project.draft_path = "out/drafts/甲.pcbdraft"
    project.project_path = "D:/工程/甲.epro2"
    project.order_path = "D:/下单/甲.zip"

    assert project.draft_path.endswith(".pcbdraft")
    assert project.project_path.endswith(".epro2")
    assert project.order_path.endswith(".zip")


# ---------------------------------------------------------------- workspace


def test_a_new_workspace_holds_one_unnamed_project():
    workspace = Workspace()

    assert len(workspace) == 1
    assert workspace.index == 0
    assert workspace.active.title == UNTITLED


def test_added_projects_are_numbered_without_reusing_the_first_name():
    """The first project owns the plain 未命名; the next one gets a number."""
    workspace = Workspace()
    workspace.new()
    workspace.new()

    assert [item.title for item in workspace] == ["未命名", "未命名 2", "未命名 3"]


def test_a_new_project_opens_next_to_the_current_one():
    workspace = Workspace()
    workspace.new()
    workspace.switch(0)
    workspace.new()

    assert workspace.index == 1
    assert len(workspace) == 3


def test_each_project_keeps_its_own_document():
    """The whole point: two boards open at once, neither disturbing the other."""
    workspace = Workspace()
    first = touched(workspace.active.board, width=40.0)
    workspace.new()
    second = touched(workspace.active.board, width=90.0)

    assert first is not second
    workspace.switch(0)
    assert workspace.active.board.outline.width_mm == pytest.approx(40.0)
    workspace.switch(1)
    assert workspace.active.board.outline.width_mm == pytest.approx(90.0)


def test_switching_reports_only_the_projects_that_changed():
    workspace = Workspace()
    workspace.active.board.placements.append(
        ImagePlacement(path="a.png", width_mm=10, height_mm=10)
    )
    clean = workspace.active
    workspace.new()

    assert workspace.dirty() == [clean]


def test_switching_out_of_range_is_refused():
    workspace = Workspace()
    with pytest.raises(IndexError):
        workspace.switch(1)


# -------------------------------------------------------------- closing tabs


def test_closing_a_tab_hands_over_to_the_one_on_its_right():
    workspace = Workspace()
    workspace.new()
    workspace.new()
    workspace.switch(0)

    workspace.close(0)
    assert len(workspace) == 2
    assert workspace.index == 0


def test_closing_the_last_tab_falls_back_to_the_one_on_its_left():
    workspace = Workspace()
    workspace.new()
    workspace.switch(1)

    workspace.close(1)
    assert len(workspace) == 1
    assert workspace.index == 0


def test_closing_an_earlier_tab_keeps_the_same_document_active():
    """Closing a neighbour must not silently switch the user's document."""
    workspace = Workspace()
    workspace.new()
    workspace.switch(1)
    current = workspace.active

    workspace.close(0)

    assert workspace.active is current
    assert workspace.index == 0


def test_closing_the_only_tab_leaves_a_fresh_board():
    """The window always has a document: every pane reads it from the active
    project, and an empty window would need a second state in all of them."""
    workspace = Workspace()
    touched(workspace.active.board)

    fresh = workspace.close(0)

    assert len(workspace) == 1
    assert workspace.active is fresh
    assert fresh.is_dirty() is False
    assert fresh.board.outline_ready is False


def test_closing_the_others_keeps_only_the_chosen_tab():
    workspace = Workspace()
    workspace.new()
    workspace.new()
    keep = workspace[1]

    workspace.close_others(1)

    assert len(workspace) == 1
    assert workspace.active is keep
    assert workspace.index == 0


# ------------------------------------------------------------ reordering


def test_moving_a_tab_lands_it_at_the_requested_index():
    workspace = Workspace()
    workspace.new()
    workspace.new()
    names = [item.title for item in workspace]

    workspace.move(0, 2)

    assert [item.title for item in workspace] == [names[1], names[2], names[0]]


def test_moving_a_tab_keeps_the_same_project_active():
    """Dragging tabs around is a view operation; it must not change which
    document the user is editing."""
    workspace = Workspace()
    workspace.new()
    workspace.new()
    workspace.switch(1)
    current = workspace.active

    workspace.move(1, 2)
    assert workspace.active is current
    assert workspace.index == 2

    workspace.move(2, 0)
    assert workspace.active is current
    assert workspace.index == 0
