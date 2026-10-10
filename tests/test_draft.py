"""Tests for draft saving, and for knowing when a draft is out of date.

Two things have to hold, and they fail in opposite directions:

* **Saving and loading must lose nothing.** A draft that drops the standee tab,
  a layer's visibility or a picture's path would quietly change the user's
  design between sessions, and they would only notice after cutting a board.
* **The dirty check must not miss an edit.** A false "saved" silently discards
  work at exit; a false "dirty" nags the user about nothing. The first is much
  worse, so the check is a comparison against the saved bytes rather than a
  flag that each edit path has to remember to set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.artboard import (  # noqa: E402
    ArtBoard,
    BoardOutlineSpec,
    ImagePlacement,
    IrregularBase,
    OutlineShape,
)
from pcbart.draft import (  # noqa: E402
    DRAFT_SUFFIX,
    board_from_dict,
    board_to_dict,
    default_draft_path,
    fingerprint,
    load_draft,
    save_draft,
)


def full_board() -> ArtBoard:
    """A document that uses every part of the model the draft must carry."""
    board = ArtBoard(
        outline=BoardOutlineSpec(
            shape=OutlineShape.ROUNDED,
            width_mm=64.5,
            height_mm=42.25,
            corner_radius_mm=4.5,
            standee=True,
            standee_width_mm=16.0,
            standee_height_mm=3.0,
            standee_offset_mm=-7.5,
            margin_mm=3.0,
            simplify_mm=0.4,
            keep_aspect=False,
            irregular_base=IrregularBase.ROUNDED,
            base_top_ratio=0.6,
            template=None,
        ),
        outline_ready=True,
    )
    board.placements.append(
        ImagePlacement(
            path=r"C:\PS抠图\由比滨透明背景.png",
            x_mm=-12.5,
            y_mm=8.25,
            width_mm=50.0,
            height_mm=66.7,
            rotation_deg=15.0,
            flip_horizontal=True,
            flip_vertical=False,
            keep_aspect=False,
            opacity=0.75,
            layer="顶层丝印层",
            processed_path=r"C:\缓存\线稿_abc.png",
            tone="white",
            import_id="import-1",
        )
    )
    board.placements.append(
        ImagePlacement(path="", layer="底层", tone="black", import_id="import-1")
    )
    board.operator_name = "暮落"
    board.operator_number = "VC07"
    board.operator_page = "暮落(集成战略)"
    board.layers.set_current("top_silk")
    board.layers.set_visible("bottom_copper", False)
    board.layers.set_visible("outline", False)
    return board


# ----------------------------------------------------------- round trips


def test_a_draft_carries_the_whole_document(tmp_path):
    board = full_board()
    path = save_draft(board, tmp_path / f"草稿{DRAFT_SUFFIX}")

    restored = load_draft(path)

    assert fingerprint(restored) == fingerprint(board), "存下来再读回来必须完全一致"


def test_every_placement_field_survives(tmp_path):
    """Checked field by field: one dropped field is a silent geometry change."""
    board = full_board()
    restored = load_draft(save_draft(board, tmp_path / "d.pcbdraft"))

    for original, copy in zip(board.placements, restored.placements):
        for field in original.__dataclass_fields__:
            assert getattr(copy, field) == getattr(original, field), field


def test_every_outline_field_survives(tmp_path):
    board = full_board()
    restored = load_draft(save_draft(board, tmp_path / "d.pcbdraft"))

    for field in board.outline.__dataclass_fields__:
        got = getattr(restored.outline, field)
        want = getattr(board.outline, field)
        # JSON has no tuples, so a tuple field comes back as a list; compare by
        # value rather than by container type.
        if isinstance(want, tuple):
            got = tuple(got)
        assert got == want, field


def test_visibility_and_the_current_layer_survive(tmp_path):
    """View state is part of the draft: reopening to a different set of
    visible layers looks like the design changed."""
    board = full_board()
    restored = load_draft(save_draft(board, tmp_path / "d.pcbdraft"))

    assert restored.layers.current == "top_silk"
    assert restored.layers.is_visible("bottom_copper") is False
    assert restored.layers.is_visible("outline") is False
    assert restored.layers.is_visible("top_copper") is True


def test_an_unsaved_board_keeps_outline_ready_false(tmp_path):
    """``outline_ready`` is the difference between "no board yet" and "a board
    the user chose"; inventing one on load would put geometry on the canvas
    nobody asked for."""
    board = ArtBoard()
    restored = load_draft(save_draft(board, tmp_path / "d.pcbdraft"))

    assert restored.outline_ready is False
    assert restored.placements == []


def test_chinese_paths_round_trip(tmp_path):
    """The path is the artwork; mangling it loses the picture on reopen."""
    board = full_board()
    restored = load_draft(save_draft(board, tmp_path / "草稿 测试.pcbdraft"))

    assert restored.placements[0].path == board.placements[0].path
    assert restored.placements[0].processed_path == board.placements[0].processed_path


def test_the_file_is_readable_json(tmp_path):
    """Readable on purpose: when something goes wrong it can be inspected, and
    a test fixture can be written by hand."""
    path = save_draft(full_board(), tmp_path / "d.pcbdraft")
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["format"] == "pcb-art-board-draft"
    assert isinstance(raw["placements"], list)


# ------------------------------------------------------------- the model


def test_the_saved_data_is_a_plain_dictionary():
    """No dataclasses or Path objects leak through: they would not serialise."""
    data = board_to_dict(full_board())
    json.dumps(data)  # raises if anything is not JSON-ready


def test_an_unknown_field_is_ignored(tmp_path):
    """A draft from a newer build must still open, minus what this one cannot
    represent -- rather than refusing to open at all."""
    data = board_to_dict(full_board())
    data["outline"]["a_field_from_the_future"] = 42

    restored = board_from_dict(data)

    assert restored.outline.width_mm == 64.5


def test_a_missing_field_keeps_its_default():
    data = board_to_dict(full_board())
    del data["outline"]["corner_radius_mm"]

    restored = board_from_dict(data)

    assert restored.outline.corner_radius_mm == BoardOutlineSpec().corner_radius_mm


def test_something_that_is_not_a_draft_is_refused(tmp_path):
    """Loading an arbitrary JSON file must not silently produce an empty
    board, which looks exactly like the user's work disappearing."""
    path = tmp_path / "not-a-draft.pcbdraft"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    with pytest.raises(ValueError, match="草稿"):
        load_draft(path)


def test_a_corrupt_file_says_so(tmp_path):
    path = tmp_path / "broken.pcbdraft"
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError, match="损坏"):
        load_draft(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ValueError, match="找不到"):
        load_draft(tmp_path / "nope.pcbdraft")


def test_saving_creates_the_directory(tmp_path):
    path = save_draft(full_board(), tmp_path / "a" / "b" / "d.pcbdraft")

    assert path.exists()


def test_saving_over_an_existing_draft_replaces_it(tmp_path):
    """The normal case: save, keep working, save again."""
    path = tmp_path / "d.pcbdraft"
    board = full_board()
    save_draft(board, path)

    board.outline.width_mm = 99.0
    save_draft(board, path)

    assert load_draft(path).outline.width_mm == pytest.approx(99.0)


def test_no_temporary_file_is_left_behind(tmp_path):
    """The write goes through a temp file; leaving it behind would litter the
    user's folder with files that look like drafts."""
    save_draft(full_board(), tmp_path / "d.pcbdraft")

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_the_default_draft_location_is_writable():
    path = default_draft_path()

    assert path.suffix == DRAFT_SUFFIX
    assert path.parent.name == "drafts"


# --------------------------------------------------------- dirty tracking


def test_an_untouched_document_has_a_stable_fingerprint():
    board = full_board()

    assert fingerprint(board) == fingerprint(board)


@pytest.mark.parametrize(
    "change",
    [
        lambda b: setattr(b.outline, "width_mm", 12.0),
        lambda b: setattr(b.outline, "standee", False),
        lambda b: setattr(b.outline, "standee_height_mm", 4.0),
        lambda b: setattr(b.outline, "base_top_ratio", 0.1),
        lambda b: setattr(b.placements[0], "x_mm", 99.0),
        lambda b: setattr(b.placements[0], "opacity", 0.2),
        lambda b: setattr(b.placements[0], "flip_vertical", True),
        lambda b: b.placements.append(ImagePlacement(path="new.png")),
        lambda b: b.placements.pop(),
        lambda b: b.layers.set_visible("top_copper", False),
        lambda b: b.layers.set_current("bottom_silk"),
        lambda b: setattr(b, "outline_ready", False),
        lambda b: setattr(b, "operator_page", "暮落"),
    ],
)
def test_every_change_makes_the_document_dirty(change):
    """Each of these is something a user can do in the editor, and each one has
    to be noticed -- a missed one loses work at exit without a prompt."""
    board = full_board()
    before = fingerprint(board)

    change(board)

    assert fingerprint(board) != before


def test_the_fingerprint_matches_what_actually_gets_written(tmp_path):
    """The check and the save must agree, or the prompt lies.

    Both read the same serialised form, which is what makes this true by
    construction; the test pins it so a future change cannot separate them.
    """
    board = full_board()
    path = save_draft(board, tmp_path / "d.pcbdraft")
    saved_fingerprint = fingerprint(board)

    restored = load_draft(path)

    assert fingerprint(restored) == saved_fingerprint


def test_saving_clears_the_dirty_state(tmp_path):
    board = full_board()
    save_draft(board, tmp_path / "d.pcbdraft")
    saved = fingerprint(board)

    board.placements[0].x_mm += 5.0
    assert fingerprint(board) != saved

    save_draft(board, tmp_path / "d.pcbdraft")
    assert fingerprint(board) == fingerprint(load_draft(tmp_path / "d.pcbdraft"))


# ------------------------------------------------------------- the library


def test_the_library_lists_drafts_newest_first():
    """The list answers "what was I working on", so order is by time."""
    import tempfile
    import time

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())
    for name in ("先存的", "后存的"):
        store.save(full_board(), name)
        time.sleep(0.02)  # distinct mtimes on filesystems with coarse clocks

    assert [info.name for info in store.list()] == ["后存的", "先存的"]


def test_the_library_keeps_drafts_inside_itself():
    """A name is a name, not a path: '../x' must not escape the folder."""
    import tempfile

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())

    for hostile in ("../evil", "..\\evil", "a/b", "a\\b"):
        path = store.path_for(hostile)
        assert path.parent == store.root, hostile
        assert path.suffix == DRAFT_SUFFIX


def test_saving_under_a_name_replaces_that_draft():
    import tempfile

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())
    board = full_board()
    store.save(board, "我的板")

    board.outline.width_mm = 123.0
    store.save(board, "我的板")

    assert len(store.list()) == 1
    assert store.load(store.path_for("我的板")).outline.width_mm == pytest.approx(123.0)


def test_unique_name_does_not_collide():
    import tempfile

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())
    first = store.unique_name()
    store.save(full_board(), first)

    second = store.unique_name()

    assert second != first, "重名会覆盖掉已有草稿"


def test_a_draft_can_be_deleted():
    import tempfile

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())
    path = store.save(full_board(), "待删除")
    assert store.list()

    assert store.delete(path) is True

    assert store.list() == []
    assert store.delete(path) is False, "删两次应报 false 而不是抛异常"


def test_an_empty_library_lists_nothing_rather_than_failing():
    import tempfile

    from pcbart.draft import DraftStore

    store = DraftStore(tempfile.mkdtemp())

    assert store.list() == []


def test_the_generated_kind_survives_a_draft(tmp_path):
    """``kind`` is what tells the canvas how to recolour the card's artwork.

    The four generated elements are all "alpha is the shape", so the canvas
    recolours them by their layer colour. Lose the field and a reopened draft
    draws them as raw pixmaps: the QR symbol vanishes into the dark board and
    the faction logo comes back as a white block.
    """
    from pcbart import badge_card, badge_content

    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.operator_name = "阿米娅"
    board.operator_page = "阿米娅"
    try:
        art = badge_content.badge_art(board=board)
    except badge_content.MissingArt:
        pytest.skip("干员素材库未安装")
    board.placements = list(art.placements)

    restored = load_draft(save_draft(board, tmp_path / "d.pcbdraft"))
    assert [item.kind for item in restored.placements] == [
        item.kind for item in board.placements
    ]
    assert "qr" in {item.kind for item in restored.placements}
