"""Grouping placements into one element, and writing that to the board file.

The user asked for this by name: "把这两层组合成同一个元素，即加入EDApro里面的
组合功能。要求可以自由选择组合的元素、退出组合、组合管理、命名等功能。"

Two things therefore have to be true and are tested separately:

* the **model and the canvas** treat a group as one object -- select, drag,
  delete, copy all move together;
* the **file** carries EasyEDA Pro's own encoding, because the point of the
  feature is that the editor shows one element too.

The encoding was measured rather than guessed. A card the user built in the
editor carries::

    {"type":"GROUP","ticket":2193,"id":"1"}||{"groupName":"group1"}

with each member naming it from its own payload as ``"groupId":"1"``.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    OutlineShape,
    PlacementGroup,
    TextElement,
)


def rectangle_board(**kwargs) -> ArtBoard:
    board = ArtBoard(
        outline=BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60.0, height_mm=60.0),
        outline_ready=True,
        **kwargs,
    )
    return board


def placement(name: str = "a.png", **kwargs) -> ImagePlacement:
    values = dict(x_mm=0.0, y_mm=0.0, width_mm=10.0, height_mm=10.0)
    values.update(kwargs)
    return ImagePlacement(path=name, **values)


# --------------------------------------------------------------- the model


def test_a_group_keeps_its_members_by_id_not_by_position():
    """⚠️ Indices cannot be the identity, and this is why.

    Undo restores placements *in place* and inserting or deleting shifts every
    index after it. A group storing positions would silently start naming
    different artwork after the first edit -- and it would look fine, because
    the group would still have the same number of members.
    """
    board = rectangle_board()
    first, second = placement("a.png"), placement("b.png")
    board.placements = [first, second]

    group = board.make_group([first, second], name="一对")

    assert group is not None
    assert first.element_id and second.element_id
    assert group.members == [first.element_id, second.element_id]

    # Move another object in front, so every index shifts.
    intruder = placement("c.png")
    board.placements.insert(0, intruder)
    assert [item.path for item in board.group_members(group)] == ["a.png", "b.png"]


def test_grouping_the_same_objects_does_not_make_a_second_group():
    board = rectangle_board()
    first, second = placement("a.png"), placement("b.png")
    board.placements = [first, second]

    first_group = board.make_group([first, second], name="甲")
    second_group = board.make_group([first, second], name="乙")

    assert len(board.groups) == 1, "同一批元素不该同时属于两个组合"
    assert board.groups[0] is second_group
    assert board.groups[0].name == "乙"
    assert first_group not in board.groups


def test_an_object_can_only_be_in_one_group():
    """The editor's member field is a single ``groupId``, so this is the format's
    own limit rather than a simplification."""
    board = rectangle_board()
    a, b, c = placement("a.png"), placement("b.png"), placement("c.png")
    board.placements = [a, b, c]

    board.make_group([a, b], name="甲")
    board.make_group([b, c], name="乙")

    assert len(board.groups) == 1
    assert board.group_of(b) is board.groups[0]
    assert [item.path for item in board.group_members(board.groups[0])] == ["b.png", "c.png"]


def test_a_group_needs_two_distinct_objects():
    board = rectangle_board()
    only = placement("a.png")
    board.placements = [only]

    assert board.make_group([only]) is None
    assert board.make_group([only, only]) is None, "同一个元素不能和自己成组"
    assert board.groups == []


def test_text_can_be_a_member_but_is_not_written_as_traced_geometry():
    """The wording is an element the user can pick, so it can be grouped.

    ⚠️ But it cannot go in the file's ``GROUP`` record: a text line is written
    as its own ``STRING`` record, which has no group reference to set. Reporting
    text from ``group_members`` would make the exporter promise something it
    cannot write, so the two answers are deliberately different.
    """
    board = rectangle_board()
    image = placement("a.png")
    text = TextElement(text="示例")
    board.placements = [image]
    board.texts = [text]

    group = board.make_group([image, text], name="图和字")
    assert group is not None
    assert board.group_members(group) == [image], "写盘用的成员只有图片"
    assert [type(item).__name__ for item in board.group_all_members(group)] == [
        "ImagePlacement",
        "TextElement",
    ], "界面用的成员两种都有"


def test_ungrouping_leaves_the_objects_alone():
    board = rectangle_board()
    a, b = placement("a.png"), placement("b.png")
    board.placements = [a, b]
    board.make_group([a, b], name="甲")

    board.ungroup([a])

    assert board.groups == [], "只剩一个成员的组合应当自动消失"
    assert board.placements == [a, b], "解散组合不该删掉元素"


def test_pruning_drops_members_that_are_gone():
    """A stale id draws nothing but still counts, so every size the group
    reports would be wrong -- and the file would name a member that is not
    there."""
    board = rectangle_board()
    a, b = placement("a.png"), placement("b.png")
    board.placements = [a, b]
    group = board.make_group([a, b], name="甲")

    board.placements.remove(b)
    removed = board.prune_groups()

    assert removed == [group]
    assert board.groups == []


def test_a_group_label_is_never_stored_when_it_was_not_given():
    """An unnamed group still needs a row label, but the fallback is generated
    so it cannot go stale against the document."""
    board = rectangle_board()
    a, b = placement("a.png"), placement("b.png")
    board.placements = [a, b]
    group = board.make_group([a, b])

    assert group.name == ""
    assert board.group_label(group) == "组合 1"


def test_a_one_member_group_is_reported_as_unhealthy():
    group = PlacementGroup(id="x", name="单独", members=["e1"])
    assert group.validate(), "单成员组合应当被报出来"


# ------------------------------------------------------------- the file


def painting_board() -> ArtBoard:
    """A board painting: two tones, one group -- the case the user named.

    Built from real files rather than mocked, because the export has to read
    the masks back off disk: the two tones travel as separate images and the
    pairing is what makes their edges meet.
    """
    import tempfile

    from pcbart.image_prep import imwrite_unicode

    board = rectangle_board()
    white_mask = np.full((80, 80), 255, np.uint8)
    black_mask = np.zeros((80, 80), np.uint8)
    white_mask[20:60, 20:60] = 0
    black_mask[20:60, 20:60] = 255

    folder = Path(tempfile.mkdtemp(prefix="pcbart-groups-"))
    white_path, black_path = folder / "w.png", folder / "b.png"
    imwrite_unicode(str(white_path), white_mask)
    imwrite_unicode(str(black_path), black_mask)

    white = placement(
        str(white_path),
        tone="white",
        layer="顶层丝印层",
        import_id="i1",
        width_mm=20.0,
        height_mm=20.0,
    )
    black = placement(
        str(black_path),
        tone="black",
        layer="顶层",
        import_id="i1",
        width_mm=20.0,
        height_mm=20.0,
    )
    board.placements = [white, black]
    board.make_group([white, black], name="我的板画")
    return board


def test_the_export_writes_a_group_record_and_names_its_members():
    """⚠️ The whole point of the feature: the editor has to show one element.

    A group with no record is invisible to the editor -- the artwork looks
    identical either way, so nothing would report it and the user would only
    find out by opening the file.
    """
    from pcbart import export

    board = painting_board()
    out = Path(__import__("tempfile").mkdtemp()) / "card.epro2"
    export.export_design(board, str(out), title="组合")

    with zipfile.ZipFile(out) as archive:
        name = [n for n in archive.namelist() if n.endswith(".epru")][0]
        text = archive.read(name).decode("utf-8")

    group_id = board.groups[0].id
    group_lines = [line for line in text.splitlines() if '"type":"GROUP"' in line]
    assert len(group_lines) == 1, f"应当只有一条 GROUP 记录，实际 {len(group_lines)}"
    assert f'"id":"{group_id}"' in group_lines[0]
    assert '"groupName":"我的板画"' in group_lines[0], "命名要写进文件"

    members = [
        line
        for line in text.splitlines()
        if f'"groupId":"{group_id}"' in line
    ]
    assert len(members) >= 2, "两个色调的 FILL 都要带 groupId"
    assert all('"type":"FILL"' in line for line in members), (
        "带 groupId 的应当是几何记录"
    )


def test_the_group_record_is_written_before_its_members():
    """A member names a group the document has to have declared already.

    Not a style preference: a FILL that points at an undeclared id is a forward
    reference, and the file is then not the shape the editor itself writes.
    """
    from pcbart import export

    board = painting_board()
    out = Path(__import__("tempfile").mkdtemp()) / "card.epro2"
    export.export_design(board, str(out), title="组合")

    with zipfile.ZipFile(out) as archive:
        name = [n for n in archive.namelist() if n.endswith(".epru")][0]
        text = archive.read(name).decode("utf-8")

    group_at = text.index('"type":"GROUP"')
    member_at = text.index(f'"groupId":"{board.groups[0].id}"')
    assert group_at < member_at, "GROUP 记录必须排在成员之前"


def test_an_ungrouped_board_writes_no_group_record():
    from pcbart import export

    board = painting_board()
    board.groups.clear()
    out = Path(__import__("tempfile").mkdtemp()) / "card.epro2"
    export.export_design(board, str(out), title="不组合")

    with zipfile.ZipFile(out) as archive:
        name = [n for n in archive.namelist() if n.endswith(".epru")][0]
        text = archive.read(name).decode("utf-8")

    assert '"type":"GROUP"' not in text
    assert '"groupId":""' in text, "未分组的 FILL 应当写空字符串"


def test_an_unnamed_group_writes_no_group_name_key():
    """The hand-made files that carry no name serialise the key away entirely
    rather than writing it empty."""
    from pcbart.generator import group_record
    from pcbart.templates import build_minimal_project

    document, _ = build_minimal_project(title="t")
    record = group_record(PlacementGroup(id="7", name=""), document)
    assert record.body == {}
    assert record.header["id"] == "7"


def test_a_group_is_recorded_as_an_element_for_the_editor():
    """``PRIMITIVE_DISPLAY`` has to declare GROUP, or the editor has no state
    for the primitive type and the element is not pickable there."""
    from pcbart.templates import PRIMITIVE_DISPLAY

    entry = [row for row in PRIMITIVE_DISPLAY if row[0] == "GROUP"]
    assert len(entry) == 1
    assert entry[0][1] is True and entry[0][2] is False


def test_a_group_with_fewer_than_two_written_members_is_dropped():
    """A ``GROUP`` record whose members produced no geometry would put an
    element row in the editor that selects nothing."""
    from pcbart import export

    board = painting_board()
    board.placements = board.placements[:1]
    board.groups[0].members = board.groups[0].members[:1]
    board.placements[0].tone = ""

    out = Path(__import__("tempfile").mkdtemp()) / "card.epro2"
    export.export_design(board, str(out), title="空组合")
    with zipfile.ZipFile(out) as archive:
        name = [n for n in archive.namelist() if n.endswith(".epru")][0]
        text = archive.read(name).decode("utf-8")
    assert '"type":"GROUP"' not in text
