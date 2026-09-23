"""Tests for the layer set and its view state. No Qt needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.design_layers import (  # noqa: E402
    LAYERS,
    LAYER_KEYS,
    LayerStack,
    layer,
    layer_by_name,
)
from pcbart.layers import layer_by_type  # noqa: E402


def test_only_the_layers_this_tool_produces_are_offered():
    """The other fifty-odd EasyEDA layers exist but this tool makes none of them.

    Listing inner layers or paste masks would put rows in the panel that cannot
    affect this design.
    """
    assert LAYER_KEYS == (
        "top_copper",
        "top_silk",
        "top_mask",
        "bottom_copper",
        "bottom_silk",
        "bottom_mask",
        "outline",
    )


def test_both_sides_have_the_same_three_layers():
    for kind in ("copper", "silk", "mask"):
        for side in ("top", "bottom"):
            matches = [
                item for item in LAYERS if item.kind == kind and item.side == side
            ]
            assert len(matches) == 1, f"{side} {kind}"


@pytest.mark.parametrize("item", LAYERS, ids=[i.key for i in LAYERS])
def test_every_layer_matches_its_easyeda_counterpart(item):
    """Names and colours come from the editor's own table.

    A layer drawn in a different colour from the one EasyEDA uses would show
    the user something that does not match what they see in the editor.
    """
    entry = layer_by_type(item.easyeda_type)
    assert entry is not None, f"{item.key} 没有对应的 EasyEDA 图层"
    assert item.color == entry.active_color
    assert item.easyeda_id == entry.id


def test_the_colours_are_the_measured_ones():
    """Pinned values, so a change to the layer table is noticed here."""
    assert layer("top_copper").color == "#ff0000"
    assert layer("top_silk").color == "#ffcc00"
    assert layer("bottom_copper").color == "#0000ff"
    assert layer("bottom_silk").color == "#66cc33"
    assert layer("top_mask").color == "#800080"
    assert layer("bottom_mask").color == "#aa00ff"
    assert layer("outline").color == "#ff00ff"


def test_an_unknown_layer_key_raises_rather_than_returning_none():
    """A typo must not turn into a silently blank canvas."""
    with pytest.raises(KeyError):
        layer("middle_copper")


# ----------------------------------------------------------------- the stack


def test_every_layer_starts_visible():
    stack = LayerStack()
    assert all(stack.is_visible(key) for key in LAYER_KEYS)


def test_toggling_a_layer_flips_only_that_layer():
    stack = LayerStack()
    stack.toggle("top_silk")

    assert not stack.is_visible("top_silk")
    assert stack.is_visible("top_copper")
    stack.toggle("top_silk")
    assert stack.is_visible("top_silk")


def test_a_hidden_layer_is_not_selectable():
    """Hidden must mean unpickable, or a drag moves something invisible."""
    stack = LayerStack()
    assert stack.selectable("top_copper")
    stack.set_visible("top_copper", False)
    assert not stack.selectable("top_copper")


def test_setting_the_current_layer_ignores_unknown_keys():
    stack = LayerStack()
    stack.set_current("top_silk")
    assert stack.current == "top_silk"

    stack.set_current("not_a_layer")
    assert stack.current == "top_silk", "未知层不应改变当前层"


def test_an_unknown_current_layer_falls_back_to_the_first():
    stack = LayerStack(current="nonsense")
    assert stack.current == LAYER_KEYS[0]


def test_a_partial_visibility_mapping_still_answers_every_layer():
    stack = LayerStack(visible={"top_silk": False})

    assert not stack.is_visible("top_silk")
    assert stack.is_visible("top_copper")


def test_visible_layers_returns_the_displayed_ones():
    stack = LayerStack()
    stack.set_visible("outline", False)

    names = [item.key for item in stack.visible_layers()]
    assert "outline" not in names
    assert len(names) == len(LAYER_KEYS) - 1


# ----------------------------------------------------------- name translation


def test_a_display_name_maps_to_its_key():
    assert LayerStack().layer_for_name("顶层丝印层") == "top_silk"
    assert LayerStack().layer_for_name("板框层") == "outline"


def test_an_unrecognised_layer_name_falls_back_to_copper():
    """Better visible on a sensible layer than vanished onto no layer."""
    assert LayerStack().layer_for_name("顶层 + 顶层丝印层") == "top_copper"
    assert LayerStack().layer_for_name("") == "top_copper"


def test_lookup_by_display_name():
    assert layer_by_name("底层阻焊层").key == "bottom_mask"
    assert layer_by_name("不存在") is None
