"""Layer table for EasyEDA Pro PCB documents.

Captured verbatim from a project written by EasyEDA Pro 3.2.149 so that
generated boards carry the same layer configuration an editor-created board
does. Colours, visibility flags and transparencies are preserved because the
editor expects the whole table, not just the layers a design uses.

Layer ids 15..46 are the inner signal layers; a two-layer board still declares
them, marked ``use=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Layer", "DEFAULT_LAYERS", "layer_by_id", "layer_by_type", "outline_layer"]


@dataclass(frozen=True)
class Layer:
    id: int
    type: str
    name: str
    active_color: str
    inactive_color: str
    use: bool = True
    show: bool = True
    transparency: float = 1.0
    inactive_transparency: float = 0.5

    def to_body(self) -> dict[str, object]:
        """Payload for a ``PCB/LAYER`` record."""
        return {
            "layerId": self.id,
            "layerType": self.type,
            "layerName": self.name,
            "use": self.use,
            "show": self.show,
            "locked": False,
            "activeColor": self.active_color,
            "activateTransparency": self.transparency,
            "inactiveColor": self.inactive_color,
            "inactiveTransparency": self.inactive_transparency,
        }

    @property
    def record_id(self) -> str:
        """``LAYER`` records identify themselves with a serialised JSON array."""
        return f'["LAYER",{self.id}]'


# (id, type, name, active, inactive, use, show, transparency)
_RAW: tuple[tuple[int, str, str, str, str, bool, bool, float], ...] = (
    (1, "TOP", "Top Layer", "#ff0000", "#7f0000", True, True, 1),
    (2, "BOTTOM", "Bottom Layer", "#0000ff", "#00007f", True, True, 1),
    (3, "TOP_SILK", "Top Silkscreen Layer", "#ffcc00", "#7f6600", True, True, 1),
    (4, "BOT_SILK", "Bottom Silkscreen Layer", "#66cc33", "#336619", True, True, 1),
    (5, "TOP_SOLDER_MASK", "Top Solder Mask Layer", "#800080", "#400040", True, True, 0.7),
    (6, "BOT_SOLDER_MASK", "Bottom Solder Mask Layer", "#aa00ff", "#55007f", True, True, 0.7),
    (7, "TOP_PASTE_MASK", "Top Paste Mask Layer", "#808080", "#404040", True, True, 1),
    (8, "BOT_PASTE_MASK", "Bottom Paste Mask Layer", "#800000", "#400000", True, True, 1),
    (9, "TOP_ASSEMBLY", "Top Assembly Layer", "#33cc99", "#19664c", True, True, 1),
    (10, "BOT_ASSEMBLY", "Bottom Assembly Layer", "#5555ff", "#2a2a7f", True, True, 1),
    (11, "OUTLINE", "Board Outline Layer", "#ff00ff", "#7f007f", True, True, 1),
    (12, "MULTI", "Multi-Layer", "#c0c0c0", "#606060", True, True, 1),
    (13, "DOCUMENT", "Document Layer", "#ffffff", "#7f7f7f", True, True, 1),
    (14, "MECHANICAL", "Mechanical Layer", "#f022f0", "#781178", True, True, 1),
    (15, "SIGNAL", "Inner1", "#999966", "#4c4c33", False, False, 1),
    (16, "SIGNAL", "Inner2", "#008000", "#004000", False, False, 1),
    (17, "SIGNAL", "Inner3", "#00ff00", "#007f00", False, False, 1),
    (18, "SIGNAL", "Inner4", "#bc8e00", "#5e4700", False, False, 1),
    (19, "SIGNAL", "Inner5", "#70dbfa", "#386d7d", False, False, 1),
    (20, "SIGNAL", "Inner6", "#00cc66", "#006633", False, False, 1),
    (21, "SIGNAL", "Inner7", "#9966ff", "#4c337f", False, False, 1),
    (22, "SIGNAL", "Inner8", "#800080", "#400040", False, False, 1),
    (23, "SIGNAL", "Inner9", "#008080", "#004040", False, False, 1),
    (24, "SIGNAL", "Inner10", "#15935f", "#0a492f", False, False, 1),
    (25, "SIGNAL", "Inner11", "#000080", "#000040", False, False, 1),
    (26, "SIGNAL", "Inner12", "#00b400", "#005a00", False, False, 1),
    (27, "SIGNAL", "Inner13", "#2e4756", "#17232b", False, False, 1),
    (28, "SIGNAL", "Inner14", "#99842f", "#4c4217", False, False, 1),
    (29, "SIGNAL", "Inner15", "#ffffaa", "#7f7f55", False, False, 1),
    (30, "SIGNAL", "Inner16", "#99842f", "#4c4217", False, False, 1),
    (31, "SIGNAL", "Inner17", "#2e4756", "#17232b", False, False, 1),
    (32, "SIGNAL", "Inner18", "#3535ff", "#1a1a7f", False, False, 1),
    (33, "SIGNAL", "Inner19", "#8000bc", "#40005e", False, False, 1),
    (34, "SIGNAL", "Inner20", "#43ae5f", "#21572f", False, False, 1),
    (35, "SIGNAL", "Inner21", "#c3ecce", "#617667", False, False, 1),
    (36, "SIGNAL", "Inner22", "#728978", "#39443c", False, False, 1),
    (37, "SIGNAL", "Inner23", "#39503f", "#1c281f", False, False, 1),
    (38, "SIGNAL", "Inner24", "#0c715d", "#06382e", False, False, 1),
    (39, "SIGNAL", "Inner25", "#5a8a80", "#2d4540", False, False, 1),
    (40, "SIGNAL", "Inner26", "#2b937e", "#15493f", False, False, 1),
    (41, "SIGNAL", "Inner27", "#23999d", "#114c4e", False, False, 1),
    (42, "SIGNAL", "Inner28", "#45b4e3", "#225a71", False, False, 1),
    (43, "SIGNAL", "Inner29", "#215da1", "#102e50", False, False, 1),
    (44, "SIGNAL", "Inner30", "#4564d7", "#22326b", False, False, 1),
    (45, "SIGNAL", "Inner31", "#6969e9", "#343474", False, False, 1),
    (46, "SIGNAL", "Inner32", "#9069e9", "#483474", False, False, 1),
    (47, "HOLE", "Hole Layer", "#222222", "#111111", True, True, 1),
    (48, "COMPONENT_SHAPE", "Component Shape Layer", "#00cccc", "#006666", True, False, 1),
    (49, "COMPONENT_MARKING", "Component Marking Layer", "#66ffcc", "#337f66", True, False, 1),
    (50, "PIN_SOLDERING", "Pin Soldering Layer", "#cc9999", "#664c4c", True, False, 1),
    (51, "PIN_FLOATING", "Pin Floating Layer", "#ff99ff", "#7f4c7f", True, False, 1),
    (52, "COMPONENT_MODEL", "Component Model Layer", "#ffffff", "#7f7f7f", False, False, 1),
    (53, "3D_SHELL_OUTLINE", "3D Shell Outline Layer", "#66ff99", "#337f4c", True, True, 1),
    (54, "3D_SHELL_TOP", "3D Shell Top Layer", "#ffccff", "#7f667f", True, True, 1),
    (55, "3D_SHELL_BOTTOM", "3D Shell Bottom Layer", "#0066cc", "#003366", True, True, 1),
    (56, "DRILL_DRAWING", "Drill Drawing Layer", "#008080", "#004040", True, True, 1),
    (57, "OTHER", "Ratline Layer", "#6464ff", "#32327f", True, True, 1),
    (58, "TOP_STIFFENER", "Top Stiffener Layer", "#eee666", "#777333", False, False, 1),
    (59, "BOTTOM_STIFFENER", "Bottom Stiffener Layer", "#ccff00", "#667f00", False, False, 1),
    (361, "SUBSTRATE", "Dielectric1", "#000000", "#000000", False, False, 1),
)

DEFAULT_LAYERS: tuple[Layer, ...] = tuple(
    Layer(
        id=row[0],
        type=row[1],
        name=row[2],
        active_color=row[3],
        inactive_color=row[4],
        use=row[5],
        show=row[6],
        transparency=row[7],
    )
    for row in _RAW
)

LAYER_TYPE_BY_ID: dict[int, str] = {layer.id: layer.type for layer in DEFAULT_LAYERS}
_ID_BY_TYPE: dict[str, int] = {layer.type: layer.id for layer in DEFAULT_LAYERS}


def layer_by_id(layer_id: int) -> Layer | None:
    for layer in DEFAULT_LAYERS:
        if layer.id == layer_id:
            return layer
    return None


def layer_by_type(layer_type: str) -> Layer | None:
    for layer in DEFAULT_LAYERS:
        if layer.type == layer_type:
            return layer
    return None


def outline_layer() -> Layer:
    """The board outline layer (id 11)."""
    layer = layer_by_type("OUTLINE")
    assert layer is not None
    return layer
