"""Tests for the canvas's editing operations and the menus that expose them.

These cover the keyboard and right-click entries the user asked for: copy,
paste, cut, delete, duplicate, nudge, rotate, flip, centre. The bar they are
held to is behaviour, not presence -- a menu entry that looks live but is wired
to nothing is exactly the failure this project keeps having to fix.

The operations live on the canvas because that is where the document is, and
because the 编辑 menu and the right-click menu must call the same code: an
operation that behaves one way from the keyboard and another from the mouse is
a bug waiting for whichever path the tests do not cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import build_window  # noqa: E402
from pcbart.artboard import ArtBoard, ImagePlacement  # noqa: E402
from pcbart.design_layers import layer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def canvas(app, tmp_path):
    """A bare canvas holding one selectable placement.

    No window: these operations are on the canvas, and a window would only add
    panes to keep in step.
    """
    from tools.design_canvas import DesignCanvas

    widget = DesignCanvas()
    widget._test_drafts = tmp_path  # keeps the fixture's tmp_path referenced
    board = ArtBoard()
    placement = ImagePlacement(
        path="",
        x_mm=10.0,
        y_mm=5.0,
        width_mm=20.0,
        height_mm=20.0,
        layer=layer("top_silk").name,
    )
    board.add_placement(placement)
    widget.board = board
    widget.selected = placement
    yield widget
    widget.deleteLater()


def _painting(board, *, import_id: str = "aa11") -> tuple:
    """One board painting: the white silk tone and the black copper tone."""
    white = ImagePlacement(
        path="", width_mm=20.0, height_mm=20.0,
        layer=layer("top_silk").name, tone="white", import_id=import_id,
    )
    black = ImagePlacement(
        path="", width_mm=20.0, height_mm=20.0,
        layer=layer("top_copper").name, tone="black", import_id=import_id,
    )
    board.add_placement(white)
    board.add_placement(black)
    return white, black


# ------------------------------------------------------------------ clipboard


def test_copy_then_paste_adds_a_second_object(canvas):
    original = canvas.selected
    assert canvas.copy_selection() is True
    assert canvas.clipboard_size == 1

    pasted = canvas.paste()

    assert pasted is not None
    assert len(canvas.board.placements) == 2
    assert pasted is not original, "粘贴出来的必须是新对象，不是原来那个"
    assert canvas.selected is pasted, "粘贴后应当选中新对象，否则用户不知道贴到哪了"


def test_paste_offsets_the_copy_so_it_is_visible(canvas):
    """A copy landing exactly on the original looks like nothing happened."""
    from tools.design_canvas import DUPLICATE_OFFSET_MM

    original = canvas.selected
    canvas.copy_selection()
    pasted = canvas.paste()

    assert pasted.x_mm == pytest.approx(original.x_mm + DUPLICATE_OFFSET_MM)
    assert pasted.y_mm == pytest.approx(original.y_mm - DUPLICATE_OFFSET_MM)


def test_repeated_paste_cascades_instead_of_stacking(canvas):
    """Pasting twice must not put both copies on one spot."""
    canvas.copy_selection()
    first = canvas.paste()
    second = canvas.paste()

    assert first is not second
    assert (second.x_mm, second.y_mm) != (first.x_mm, first.y_mm)


def test_the_clipboard_holds_what_was_copied_not_the_object(canvas):
    """Editing the original after copying must not change the clipboard.

    Storing the placement itself would make the clipboard a second name for an
    object the user can still change or delete.
    """
    original = canvas.selected
    canvas.copy_selection()
    original.x_mm = 999.0

    pasted = canvas.paste()

    assert pasted.x_mm != 999.0


def test_cut_removes_the_object_and_keeps_it_on_the_clipboard(canvas):
    assert canvas.cut_selection() is True

    assert canvas.board.placements == []
    assert canvas.clipboard_size == 1
    assert canvas.selected is None
    assert canvas.paste() is not None, "剪切后应当还能粘贴回来"


def test_duplicate_does_not_touch_the_clipboard(canvas):
    """复制一份 is not copy-then-paste: it must not overwrite a stored copy."""
    canvas.copy_selection()
    held = canvas.clipboard_size
    original = canvas.selected
    original.x_mm = 40.0
    # Re-copy so the clipboard holds the moved position, then duplicate.
    canvas.copy_selection()

    canvas.selected = original
    canvas.duplicate_selection()

    assert canvas.clipboard_size == held
    assert len(canvas.board.placements) == 2


def test_paste_with_an_empty_clipboard_does_nothing(canvas):
    assert canvas.paste() is None
    assert len(canvas.board.placements) == 1


# --------------------------------------------------------------- board pairs


def test_a_painting_is_copied_as_one_object(canvas):
    """板画 is two placements; copying one must copy both.

    The exporter writes the pair as a single geometry derived from the white
    tone, so a half-copied painting would export as a torn board.
    """
    canvas.board.placements.clear()
    white, black = _painting(canvas.board)
    canvas.selected = white

    canvas.copy_selection()
    assert canvas.clipboard_size == 2, "板画的两个色调必须一起复制"

    canvas.paste()
    tones = sorted(item.tone for item in canvas.board.placements)
    assert tones == ["black", "black", "white", "white"]


def test_a_pasted_painting_is_independent_of_the_original(canvas):
    """The copy needs its own import id, or the exporter welds the two."""
    canvas.board.placements.clear()
    white, _black = _painting(canvas.board)
    canvas.selected = white
    canvas.copy_selection()
    canvas.paste()

    ids = {item.import_id for item in canvas.board.placements}
    assert len(ids) == 2, "粘贴出来的板画应当有自己的 import_id"


def test_deleting_one_tone_deletes_the_painting(canvas):
    canvas.board.placements.clear()
    white, _black = _painting(canvas.board)
    canvas.selected = white

    assert canvas.delete_selection() is True
    assert canvas.board.placements == [], "删掉一个色调应当把整幅板画带走"


def test_moving_a_painting_moves_both_tones(canvas):
    canvas.board.placements.clear()
    white, black = _painting(canvas.board)
    canvas.selected = white

    canvas.nudge_selection(5.0, 0.0)

    assert white.x_mm == 5.0
    assert black.x_mm == 5.0, "板画的两个色调必须一起移动"


def test_rotating_a_painting_rotates_both_tones(canvas):
    canvas.board.placements.clear()
    white, black = _painting(canvas.board)
    canvas.selected = white

    canvas.rotate_selection(90.0)

    assert (white.rotation_deg, black.rotation_deg) == (90.0, 90.0)


def test_line_art_layers_are_not_paired(canvas):
    """A picture spread over three layers is three objects, not one painting.

    Pairing by import id regardless of tone would make the layers move
    together, which the presentation controls explicitly do not promise.
    """
    canvas.board.placements.clear()
    for key in ("top_copper", "top_silk", "top_mask"):
        canvas.board.add_placement(
            ImagePlacement(path="", layer=layer(key).name, import_id="cc33")
        )
    first = canvas.board.placements[0]
    canvas.selected = first

    assert len(canvas.linked(first)) == 1, "线稿的多个图层不应被当成一幅板画"


# ------------------------------------------------------------------ editing


def test_delete_clears_the_selection(canvas):
    assert canvas.delete_selection() is True
    assert canvas.board.placements == []
    assert canvas.selected is None, "删完还留着选择，属性栏会继续编辑一个死对象"


def test_nudge_moves_in_millimetres(canvas):
    canvas.nudge_selection(1.0, -2.0)
    assert (canvas.selected.x_mm, canvas.selected.y_mm) == (11.0, 3.0)


def test_rotate_wraps_within_a_half_turn(canvas):
    """The property panel's dial spans -180..180, so the angle must stay there."""
    for _ in range(3):
        canvas.rotate_selection(90.0)
    assert canvas.selected.rotation_deg == pytest.approx(-90.0)


def test_flip_toggles(canvas):
    canvas.flip_selection(horizontal=True)
    assert canvas.selected.flip_horizontal is True
    canvas.flip_selection(horizontal=True)
    assert canvas.selected.flip_horizontal is False


def test_centre_puts_the_object_at_the_origin(canvas):
    canvas.centre_selection()
    assert (canvas.selected.x_mm, canvas.selected.y_mm) == (0.0, 0.0)


def test_every_operation_reports_failure_without_a_selection(canvas):
    """No selection is not an error, but it is also not a silent success."""
    canvas.selected = None

    assert canvas.copy_selection() is False
    assert canvas.cut_selection() is False
    assert canvas.duplicate_selection() is None
    assert canvas.delete_selection() is False
    assert canvas.nudge_selection(1.0, 0.0) is False
    assert canvas.rotate_selection(90.0) is False
    assert canvas.flip_selection(horizontal=True) is False
    assert canvas.centre_selection() is False


# ------------------------------------------------------------------- signals


def test_pasting_announces_the_document_changed(canvas):
    """Anything listing the objects has to hear about a paste."""
    seen = []
    canvas.document_changed.connect(lambda: seen.append("document"))
    canvas.copy_selection()
    canvas.paste()
    assert seen == ["document"]


def test_copying_announces_the_clipboard_changed(canvas):
    """粘贴 is greyed out until something is copied, so that must emit too."""
    seen = []
    canvas.clipboard_changed.connect(lambda: seen.append("clipboard"))
    canvas.copy_selection()
    assert seen == ["clipboard"]


# -------------------------------------------------------------- right-click


def test_right_click_offers_the_object_operations(canvas, app):
    items = _context_items(canvas, at_origin=True, app=app)
    for expected in ("复制", "剪切", "复制一份", "删除", "旋转 90°", "居中"):
        assert expected in items, f"右键菜单缺少「{expected}」"


def test_right_click_on_empty_canvas_offers_only_canvas_operations(canvas, app):
    """No object under the pointer means no object operations to offer."""
    items = _context_items(canvas, at_origin=False, app=app)

    assert "粘贴" in items
    assert "适应窗口" in items
    assert "删除" not in items, "空白处不应提供删除——没有东西可删"
    assert "剪切" not in items


def test_right_click_selects_what_is_under_the_pointer(canvas, app):
    """Otherwise the menu acts on a different object than the one clicked."""
    canvas.selected = None
    _context_items(canvas, at_origin=True, app=app)
    assert canvas.selected is not None


def test_paste_is_disabled_until_something_is_copied(canvas, app):
    items = _context_items(canvas, at_origin=True, app=app, want_enabled=True)
    assert items["粘贴"] is False

    canvas.copy_selection()
    items = _context_items(canvas, at_origin=True, app=app, want_enabled=True)
    assert items["粘贴"] is True


def _context_items(canvas, *, at_origin: bool, app, want_enabled: bool = False):
    """Open the canvas's right-click menu and report what it offered.

    ``QMenu.exec`` is a C++ method, so it cannot be patched away on an
    instance; the menu class itself is swapped for the duration instead. That
    still exercises the real ``contextMenuEvent`` -- including the hit test and
    the automatic selection -- rather than calling the builder directly.
    """
    import tools.design_canvas as design_canvas

    canvas.resize(900, 700)
    canvas.fit_view()
    app.processEvents()

    captured: dict = {}

    class SpyMenu(QMenu):
        def exec(self, *args, **kwargs):  # noqa: A003 - Qt's name
            captured["items"] = [a.text() for a in self.actions()]
            captured["enabled"] = {
                a.text(): a.isEnabled() for a in self.actions() if not a.isSeparator()
            }
            return None

    original = design_canvas.QMenu
    design_canvas.QMenu = SpyMenu
    try:
        point = (
            canvas.to_screen(0.0, 0.0)
            if at_origin
            else canvas.to_screen(200.0, 200.0)
        )
        position = QPoint(int(point.x()), int(point.y()))
        event = QContextMenuEvent(
            QContextMenuEvent.Mouse, position, canvas.mapToGlobal(position)
        )
        canvas.contextMenuEvent(event)
    finally:
        design_canvas.QMenu = original

    return captured.get("enabled", {}) if want_enabled else captured.get("items", [])


# ------------------------------------------------------------- window menu


def test_the_edit_menu_exposes_the_same_operations(app):
    """The 编辑 menu is a second entry point to the same canvas methods.

    Asserted separately from the canvas tests because a menu entry that is
    listed but never connected is the exact defect this suite exists to catch.
    """
    window = build_window()
    try:
        menu = _menu_actions(window, "编辑")
        for expected in ("复制", "剪切", "粘贴", "复制一份", "删除", "旋转 90°"):
            assert expected in menu, f"编辑菜单缺少「{expected}」"
            assert menu[expected].isEnabled() is not None
    finally:
        window.hide()


def test_the_edit_menu_shortcuts_match_the_editor_conventions(app):
    window = build_window()
    try:
        menu = _menu_actions(window, "编辑")
        expected = {
            "复制": "Ctrl+C",
            "剪切": "Ctrl+X",
            "粘贴": "Ctrl+V",
            "复制一份": "Ctrl+D",
            "删除": "Del",
            "旋转 90°": "R",
        }
        for label, keys in expected.items():
            assert menu[label].shortcut().toString() == keys, f"{label} 的快捷键不对"
    finally:
        window.hide()


def test_object_entries_are_greyed_out_with_nothing_selected(app):
    """A live-looking entry that does nothing is worse than an absent one."""
    window = build_window()
    try:
        canvas = window.design_canvas
        canvas.selected = None
        canvas.selection_changed.emit(None)
        app.processEvents()

        menu = _menu_actions(window, "编辑")
        for label in ("复制", "剪切", "复制一份", "删除", "旋转 90°", "居中"):
            assert menu[label].isEnabled() is False, f"「{label}」应当置灰"

        canvas.board.add_placement(
            ImagePlacement(path="", layer=layer("top_silk").name)
        )
        placement = canvas.board.placements[-1]
        canvas.selected = placement
        canvas.selection_changed.emit(placement)
        app.processEvents()

        menu = _menu_actions(window, "编辑")
        for label in ("复制", "剪切", "复制一份", "删除", "旋转 90°", "居中"):
            assert menu[label].isEnabled() is True, f"「{label}」应当可用"
    finally:
        window.hide()


def test_the_menu_entries_actually_edit_the_document(app):
    """Pressed, not just present: the entries have to reach the canvas."""
    window = build_window()
    try:
        canvas = window.design_canvas
        canvas.board.add_placement(
            ImagePlacement(path="", layer=layer("top_silk").name)
        )
        placement = canvas.board.placements[-1]
        canvas.selected = placement
        canvas.selection_changed.emit(placement)
        app.processEvents()

        menu = _menu_actions(window, "编辑")
        menu["复制"].trigger()
        app.processEvents()
        assert canvas.clipboard_size == 1, "菜单里的复制没有生效"

        menu["粘贴"].trigger()
        app.processEvents()
        assert len(canvas.board.placements) == 2, "菜单里的粘贴没有生效"

        menu["删除"].trigger()
        app.processEvents()
        assert len(canvas.board.placements) == 1, "菜单里的删除没有生效"
    finally:
        window.hide()


def _menu_actions(window, title: str) -> dict:
    """The actions of one top-level menu, keyed by their label."""
    for entry in window.menuBar().actions():
        menu = entry.menu()
        if menu is not None and entry.text() == title:
            return {
                action.text(): action
                for action in menu.actions()
                if not action.isSeparator()
            }
    raise AssertionError(f"没有找到「{title}」菜单")
