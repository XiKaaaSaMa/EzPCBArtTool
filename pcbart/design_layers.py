"""The PCB layer set this editor works with, and its view state.

EasyEDA Pro declares 61 layers (see :mod:`pcbart.layers`). Almost all of them
belong to workflows this tool does not implement -- inner signal layers, paste
masks, assembly, 3D shells. Offering them would be the "control that looks
usable but is not" problem, so the editor exposes exactly seven:

    top copper, top silkscreen, top solder mask,
    bottom copper, bottom silkscreen, bottom solder mask,
    board outline

Names and colours are not invented: they are read from the layer table the
editor itself writes, because a board layer shown in a different colour from
the one EasyEDA uses would be actively misleading.

The layer stack is view state, not document state. It says what is on screen
and what is selectable; it does not change the geometry. It therefore lives
here as plain Python, testable without Qt, and the canvas and the layer panel
both read from one instance instead of keeping their own copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .layers import layer_by_type

__all__ = [
    "DesignLayer",
    "LAYERS",
    "LAYER_KEYS",
    "layer",
    "LayerStack",
]


@dataclass(frozen=True)
class DesignLayer:
    """One layer the editor shows, tied to its EasyEDA Pro counterpart."""

    key: str
    """Stable identifier used by code and stored on placements."""

    name: str
    """Chinese display name, matching the editor's own wording."""

    easyeda_type: str
    """Type string in the EasyEDA Pro layer table, the source of the colour."""

    side: str
    """``top``, ``bottom`` or ``none`` (the outline spans both)."""

    kind: str
    """``copper``, ``silk``, ``mask`` or ``outline``."""

    short: str
    """Compact label for the layer panel, where space is tight."""

    @property
    def easyeda_id(self) -> int:
        from .layers import layer_by_type as _by_type

        entry = _by_type(self.easyeda_type)
        if entry is None:
            raise KeyError(f"没有该层的 EasyEDA 定义: {self.easyeda_type}")
        return entry.id

    @property
    def color(self) -> str:
        """The colour EasyEDA Pro paints this layer in."""
        entry = layer_by_type(self.easyeda_type)
        if entry is None:
            raise KeyError(f"没有该层的 EasyEDA 定义: {self.easyeda_type}")
        return entry.active_color

    @property
    def transparency(self) -> float:
        """Layer opacity, as the editor records it.

        The solder masks are drawn lighter than the rest, which is how the
        editor shows a mask opening: a region you can see through.
        """
        entry = layer_by_type(self.easyeda_type)
        return entry.transparency if entry else 1.0


LAYERS: tuple[DesignLayer, ...] = (
    # Ordered the way a designer reads a two-layer board: the side you are
    # looking at first, outline last because it frames everything else.
    DesignLayer("top_copper", "顶层", "TOP", "top", "copper", "顶层"),
    DesignLayer("top_silk", "顶层丝印层", "TOP_SILK", "top", "silk", "顶丝印"),
    DesignLayer("top_mask", "顶层阻焊层", "TOP_SOLDER_MASK", "top", "mask", "顶阻焊"),
    DesignLayer("bottom_copper", "底层", "BOTTOM", "bottom", "copper", "底层"),
    DesignLayer("bottom_silk", "底层丝印层", "BOT_SILK", "bottom", "silk", "底丝印"),
    DesignLayer(
        "bottom_mask", "底层阻焊层", "BOT_SOLDER_MASK", "bottom", "mask", "底阻焊"
    ),
    DesignLayer("outline", "板框层", "OUTLINE", "none", "outline", "板框"),
)

LAYER_KEYS: tuple[str, ...] = tuple(item.key for item in LAYERS)


def layer(key: str) -> DesignLayer:
    """The layer with this key.

    Raises rather than returning ``None``: a missing key is a programming
    error, and a silent ``None`` would turn a typo into a blank canvas.
    """
    for item in LAYERS:
        if item.key == key:
            return item
    raise KeyError(f"未知图层: {key}")


def layer_by_name(name: str) -> DesignLayer | None:
    """Look a layer up by its display name, or ``None``."""
    for item in LAYERS:
        if item.name == name:
            return item
    return None


def _layer_key_for_name(name: str) -> str:
    """Translate a stored placement layer name into a layer key.

    Placements historically carried the display name. Names are what the
    property panel shows, so they are kept working, with a fallback to the
    first copper layer for anything unrecognised -- better a visible object on
    a sensible layer than one that vanishes.
    """
    found = layer_by_name(name)
    return found.key if found else LAYER_KEYS[0]


@dataclass
class LayerStack:
    """What is on screen, what is current, and therefore what is selectable.

    Kept as one object rather than loose flags on the canvas so the layer panel
    and the canvas cannot drift apart: both read and write the same state.
    """

    current: str = "top_copper"
    """Where newly added artwork goes."""

    visible: dict[str, bool] = field(
        default_factory=lambda: {key: True for key in LAYER_KEYS}
    )

    def __post_init__(self) -> None:
        # Fill in any layer missing from a caller-supplied mapping, so a stack
        # built with a partial dict still answers every visibility question.
        for key in LAYER_KEYS:
            self.visible.setdefault(key, True)
        if self.current not in LAYER_KEYS:
            self.current = LAYER_KEYS[0]

    def is_visible(self, key: str) -> bool:
        return bool(self.visible.get(key, True))

    def set_visible(self, key: str, value: bool) -> None:
        if key in LAYER_KEYS:
            self.visible[key] = bool(value)

    def toggle(self, key: str) -> None:
        self.set_visible(key, not self.is_visible(key))

    def set_current(self, key: str) -> None:
        if key in LAYER_KEYS:
            self.current = key

    def selectable(self, key: str) -> bool:
        """Whether an object on ``key`` can be picked.

        Hidden means hidden: the editor must not let the user grab something
        they cannot see, or a click appears to select nothing and the next drag
        moves an invisible object. This is also what the board outline follows.
        """
        return self.is_visible(key)

    @property
    def current_layer(self) -> DesignLayer:
        return layer(self.current)

    def visible_layers(self) -> list[DesignLayer]:
        return [item for item in LAYERS if self.is_visible(item.key)]

    def layer_for_name(self, name: str) -> str:
        return _layer_key_for_name(name)
