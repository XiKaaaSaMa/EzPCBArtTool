"""Tests for the Qt half of the theme: icons, the panel, and the live switch.

Driven through real widgets rather than by inspecting the stylesheet string,
because the questions that matter -- does the icon actually have pixels, does
switching keep the open boards, does the panel reach the window -- cannot be
answered from the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup as g  # noqa: E402
from pcbart.artboard import ImagePlacement  # noqa: E402
from pcbart.ui_icons import ICONS, icon as icon_data  # noqa: E402
from pcbart.ui_theme import DEFAULT_SEED, scheme_from_seed  # noqa: E402
from tools.qt_theme import icon, icon_pixmap  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def window(app):
    g.set_palette(scheme_from_seed(DEFAULT_SEED, dark=False))
    widget = g.build_window()
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.resize(1280, 820)
    widget.show()
    app.processEvents()
    yield widget
    from conftest import release_window

    release_window(widget)


# --------------------------------------------------------------------- icons


def _rgba(pixmap):
    """The pixmap as an ``(h, w, 4)`` uint8 array.

    Read through a numpy view of the image buffer rather than by calling
    ``pixelColor`` per pixel: the per-pixel call is roughly a thousand times
    slower, and across fifty icons it was enough to be noticeable.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    # ``QImage`` is captured in a local first. Chaining off the temporary's
    # method would be the same call, but keeping the buffer's owner alive for
    # the lifetime of the view is what makes reading it safe.
    source = pixmap.toImage()
    image = source.convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
    return raw.reshape(height, stride)[:, : width * 4].reshape(height, width, 4)


def _opaque_pixels(pixmap) -> int:
    """How many pixels are not fully transparent."""
    return int((_rgba(pixmap)[:, :, 3] > 0).sum())


def test_every_registered_icon_renders_something(app):
    """A semantic name that rasterises to an empty pixmap is an invisible button.

    Checked for all of them, not a sample: the names are looked up from a data
    table, and one bad entry is exactly as invisible as a missing file.

    ``app`` is not decoration -- **every** test that touches a ``QPixmap`` has
    to request it. Without a ``QGuiApplication`` Qt aborts the process outright
    ("Must construct a QGuiApplication before a QPixmap") rather than raising,
    so a missing fixture shows up as a silent zero-output crash with no test
    failure to read.
    """
    for name in ICONS:
        pixmap = icon_pixmap(name, "#000000", 24, 1.0)
        assert not pixmap.isNull(), f"{name} 渲染失败"
        assert pixmap.width() == 24 and pixmap.height() == 24
        # A blank pixmap is null-free and looks fine to `isNull`, so the pixels
        # have to be inspected: at least one must be non-transparent.
        assert _opaque_pixels(pixmap) > 0, f"{name} 渲染出来是空白"


def _channel_means(pixmap) -> tuple[float, float, float]:
    """Average red/green/blue over the visible pixels."""
    rgba = _rgba(pixmap)
    visible = rgba[rgba[:, :, 3] > 128]
    assert len(visible), "没有不透明像素"
    return tuple(float(v) for v in visible[:, :3].mean(axis=0))


def test_icons_take_the_colour_they_are_given(app):
    """The whole reason for drawing rather than loading a bitmap."""
    red_avg = _channel_means(icon_pixmap("layers", "#FF0000", 32, 1.0))
    blue_avg = _channel_means(icon_pixmap("layers", "#0000FF", 32, 1.0))

    assert red_avg[0] > 200 and red_avg[2] < 40, red_avg
    assert blue_avg[2] > 200 and blue_avg[0] < 40, blue_avg


def test_icons_are_sharp_at_a_scaled_device_pixel_ratio(app):
    """A 2x display must get a 2x raster, not an upscaled 1x one."""
    normal = icon_pixmap("settings", "#000000", 24, 1.0)
    retina = icon_pixmap("settings", "#000000", 24, 2.0)
    assert normal.width() == 24
    assert retina.width() == 48, "高 DPI 下没有按比例光栅化"
    assert retina.devicePixelRatio() == 2.0


def test_the_layer_icon_comes_from_lucide():
    """The one deliberate borrow: Iconoir has no layer icon.

    Pinned because "why is this one different" is a question the next reader
    will otherwise have to re-derive, and because a future regeneration that
    silently picks a different set would change the one icon the user chose by
    name.
    """
    assert icon_data("layers").set == "lucide"
    assert icon_data("board_setup").set == "iconoir"


def test_an_unregistered_icon_name_raises():
    with pytest.raises(KeyError):
        icon_data("no_such_icon")


# ------------------------------------------------------------- the settings


def test_the_settings_panel_opens_from_the_view_menu(window, app):
    """A menu entry that is listed but not wired is the classic dead control."""
    action = next(
        a for a in window.findChildren(type(window.menuBar().addAction("tmp")))
        if a.text() == "界面设置…"
    )
    action.trigger()
    app.processEvents()
    panel = g._settings_panel
    assert panel is not None, "菜单项没有打开设置面板"
    assert panel.parentWidget() is window
    panel.hide()


def test_the_panel_offers_every_preset(window, app):
    panel = g.show_settings(window)
    from pcbart.ui_theme import PRESETS

    assert len(panel._swatches) == len(PRESETS)
    assert {s.name for s in panel._swatches} == {name for name, _ in PRESETS}
    panel.hide()


def test_the_advanced_editor_starts_collapsed_then_expands(window, app):
    """Collapsed by default: the presets answer most users' question."""
    panel = g.show_settings(window)
    assert not panel.advanced_group.isVisible()
    panel.advanced_button.setChecked(True)
    app.processEvents()
    assert panel.advanced_group.isVisible()
    panel.advanced_button.setChecked(False)
    app.processEvents()
    assert not panel.advanced_group.isVisible()
    panel.hide()


def test_every_editable_role_gets_a_control(window, app):
    panel = g.show_settings(window)
    from pcbart.ui_theme import ROLE_LABELS

    assert set(panel._role_buttons) == {role for role, _ in ROLE_LABELS}
    for role, button in panel._role_buttons.items():
        assert button.text().startswith("#"), f"{role} 的按钮没显示颜色"
    panel.hide()


def test_a_preset_swatch_is_marked_as_selected(window, app):
    """With no selection shown, the panel cannot say what the current theme is."""
    panel = g.show_settings(window)
    selected = [s for s in panel._swatches if s.is_selected()]
    assert len(selected) == 1, f"选中的色块有 {len(selected)} 个，应当正好 1 个"
    assert selected[0].colour == DEFAULT_SEED.upper() or (
        selected[0].colour.upper() == DEFAULT_SEED.upper()
    )
    panel.hide()


# -------------------------------------------------------------- the switch


def test_switching_colour_keeps_the_same_window(window, app):
    """**The switch is in place: nothing is rebuilt, closed or restarted.**

    The user asked for this explicitly, and the reason is visible on screen: the
    colour panel is open while they pick, and a switch that recreated the window
    would take that panel away mid-decision and make them reopen it every time.
    """
    result = window.apply_colour_scheme(scheme_from_seed("#00695C", False))
    app.processEvents()

    assert result is window, "切换主题换了一个新窗口"
    assert window.isVisible(), "切换主题把窗口关掉了"
    assert g.current_scheme().seed == "#00695C"


def test_the_settings_panel_stays_open_across_a_switch(window, app):
    """The reported bug: the colour panel closed itself after every pick.

    Picking a colour is a trial-and-error act -- "not this one, let me try the
    next" -- so closing the panel after each attempt makes the control almost
    unusable.
    """
    panel = g.show_settings(window)
    app.processEvents()
    assert panel.isVisible()

    window.apply_colour_scheme(scheme_from_seed("#FF9800", False))
    app.processEvents()

    assert panel.isVisible(), "切换颜色后设置面板被关掉了"
    assert panel.parentWidget() is window, "面板换到了别的窗口上"
    panel.hide()


def test_the_panel_keeps_tracking_the_scheme_it_switched_to(window, app):
    """After a switch the panel must describe the *new* scheme, not the old.

    Otherwise the swatch for the colour just chosen would not be the selected
    one, and the advanced colour chips would show the previous palette.
    """
    panel = g.show_settings(window)
    app.processEvents()

    scheme = scheme_from_seed("#F44336", True)
    window.apply_colour_scheme(scheme)
    app.processEvents()

    assert panel._scheme.seed == scheme.seed
    assert panel._scheme.dark is True
    selected = [s for s in panel._swatches if s.is_selected()]
    assert len(selected) == 1, "切换后没有（或不止一个）预设被标为选中"
    assert selected[0].colour.upper() == scheme.seed
    panel.hide()


def test_switching_keeps_every_open_board(window, app):
    """No colour change may cost the user a board.

    The earlier implementation handed the whole workspace to a new window for
    this reason; an in-place switch must preserve the same guarantee, and this
    is the test that says so.
    """
    window.design_canvas.set_placement(
        ImagePlacement(path="", width_mm=30, height_mm=20)
    )
    g.new_project_tab(window)
    window.design_canvas.set_placement(
        ImagePlacement(path="", width_mm=50, height_mm=40)
    )
    app.processEvents()

    before_tabs = window.project_tabs.count()
    before_titles = [p.title for p in window.workspace.projects]
    before_counts = [len(p.board.placements) for p in window.workspace.projects]

    window.apply_colour_scheme(scheme_from_seed("#00695C", False))
    app.processEvents()

    assert window.project_tabs.count() == before_tabs
    assert [p.title for p in window.workspace.projects] == before_titles
    assert [
        len(p.board.placements) for p in window.workspace.projects
    ] == before_counts


def test_switching_keeps_the_active_board(window, app):
    """Not merely the tab count: the *same* board must still be in front."""
    g.new_project_tab(window)
    app.processEvents()
    active_title = window.workspace.active.title
    active_index = window.workspace.index

    window.apply_colour_scheme(scheme_from_seed("#FF9800", False))
    app.processEvents()

    assert window.workspace.active.title == active_title
    assert window.workspace.index == active_index


def test_the_switch_rebinds_the_module_palette(window, app):
    """The eighty-odd inline stylesheets read these names at build time."""
    scheme = scheme_from_seed("#9C27B0", False)
    before = g.ACCENT
    window.apply_colour_scheme(scheme)
    app.processEvents()

    assert g.ACCENT != before
    assert g.ACCENT == scheme["accent"]
    assert g.PANEL_BG == scheme["panel"]
    assert g.WARNING_BG == scheme["warning_container"]


def test_the_warning_banner_follows_the_theme(window, app):
    """A cream banner frozen into a dark theme is the bug this prevents."""
    light = scheme_from_seed(DEFAULT_SEED, False)
    g.set_palette(light)
    light_bg = g.WARNING_BG
    g.set_palette(scheme_from_seed(DEFAULT_SEED, True))
    dark_bg = g.WARNING_BG
    assert light_bg != dark_bg, "警告条没有跟着主题变"

    def luminance(colour: str) -> float:
        from pcbart.ui_theme import _hex_to_rgb

        r, gg, b = _hex_to_rgb(colour)
        return 0.2126 * r + 0.7152 * gg + 0.0722 * b

    assert luminance(dark_bg) < luminance(light_bg), "深色主题下警告条还是亮的"


def test_a_switch_is_saved_to_the_settings_file(window, app, tmp_path, monkeypatch):
    """The choice has to outlive the session, or the user re-picks it every time."""
    from pcbart import ui_theme

    path = tmp_path / "settings.json"
    monkeypatch.setattr(ui_theme, "settings_path", lambda: path)

    window.apply_colour_scheme(scheme_from_seed("#3F51B5", True))
    app.processEvents()

    assert path.exists(), "切换颜色后没有写入设置文件"
    restored = ui_theme.scheme_from_settings()
    assert restored.seed == "#3F51B5"
    assert restored.dark is True


# ------------------------------------------- controls carry the theme colour
#
# 用户要求：把所有的勾选框、滑块都改成和主题相同的颜色。勾选框选中时底色与
# 主题色相同，勾按主题深浅取相反色；滑块圆形拖拽不变，底色由蓝色改为和主题同色。
#
# Checked on rendered pixels rather than on the stylesheet text: the colour
# rules are strings, so reading them back would pass even while the control drew
# the platform's blue -- which is exactly what was happening.

from theme_controls import (  # noqa: E402
    check_box_fill_share,
    colour_shares,
    luminance,
    make_check_box,
    render_widget,
    share_of,
    slider_fill_share,
    _on_colour,
)


def test_the_check_box_fill_is_the_theme_colour(window, app):
    """A ticked box is filled with the accent, in both light and dark schemes."""
    for scheme in (
        scheme_from_seed("#6750A4", False),
        scheme_from_seed("#00695C", True),
    ):
        window.apply_colour_scheme(scheme)
        app.processEvents()
        share = check_box_fill_share(window, app, scheme)
        assert share > 0.02, (
            f"勾选框底色不是主题色 {scheme['accent']}（占比只有 {share:.3f}）"
        )


def test_the_tick_contrasts_with_the_fill_it_sits_on(window, app):
    """The tick is the readable opposite of the fill it sits on.

    The user's rule: the tick takes whichever of black or white reads against
    the accent. A light scheme's accent is dark, so the tick is white; a dark
    scheme's accent is light, so the tick is black. Both are checked, because
    getting one right is easy and the other is the part that fails.
    """
    for scheme in (
        scheme_from_seed("#6750A4", False),
        scheme_from_seed("#6750A4", True),
    ):
        window.apply_colour_scheme(scheme)
        app.processEvents()

        box = make_check_box(window, app)
        try:
            shares = colour_shares(render_widget(window, box))
        finally:
            box.deleteLater()

        # What matters is the *contrast*, not one exact hex. The user's rule is
        # "深色主题用相反的颜色"; MD3 answers that with a deeply-tinted version of
        # the accent rather than pure black (`#32226F` on a `#D2BBFF` fill),
        # which is the same idea done better -- and pinning the exact value
        # would forbid that improvement.
        accent_is_light = luminance(scheme["accent"]) > 0.5
        fill = QColor(scheme["accent"])
        background = QColor(scheme["panel"])

        # The marking is the most common colour that is neither the fill nor
        # the surface behind the box.
        marking, best = None, 0.0
        for colour, share in shares.items():
            got = QColor(colour)
            if got == fill or got == background or share <= best:
                continue
            marking, best = got, share
        assert marking is not None, f"勾选框里找不到勾，实测配色 {shares}"

        marking_is_light = luminance(marking.name()) > 0.5
        assert marking_is_light != accent_is_light, (
            f"主题色 {scheme['accent']}（{'浅' if accent_is_light else '深'}）上的勾"
            f"是 {marking.name()}，明暗没有取反"
        )
        assert best > 0.005, (
            f"勾只占 {best:.4f}，可能只是抗锯齿留下的杂色（{marking.name()}）"
        )


def test_the_slider_fill_is_the_theme_colour(window, app):
    """The filled part of the groove follows the accent.

    The user asked for exactly this -- 圆形拖拽不变，底色由蓝色改为和主题同色 --
    so the knob is left alone and only the colour is checked.
    """
    for scheme in (
        scheme_from_seed("#6750A4", False),
        scheme_from_seed("#FF9800", False),
    ):
        window.apply_colour_scheme(scheme)
        app.processEvents()
        share = slider_fill_share(window, app, scheme)
        assert share > 0.02, (
            f"滑块底色不是主题色 {scheme['accent']}（占比只有 {share:.3f}）"
        )


def test_the_controls_are_styled_on_a_freshly_built_window(window, app):
    """A newly built window must already carry the theme.

    ⚠️ **This is the bug the tests above would have missed.** The theme used to
    be applied only when the user *changed* a colour, so a launched application
    ran on Qt's built-in style: a platform-blue check box and a blue-grey
    slider. ``build_window`` now styles the window it returns, which is what
    this asserts -- without it the whole colour feature stays invisible until
    the settings panel has been opened once.

    The application style is installed first, because that is what startup does
    before any window exists. Only the *window* styling is under test here; a
    window built with no style installed at all would fail for an unrelated
    reason and prove nothing.
    """
    scheme = scheme_from_seed("#6750A4", False)
    window.apply_colour_scheme(scheme)
    app.processEvents()

    fresh = g.build_window()
    fresh.setAttribute(Qt.WA_DontShowOnScreen, True)
    fresh.resize(900, 600)
    fresh.show()
    app.processEvents()
    try:
        share = check_box_fill_share(fresh, app, scheme)
        assert share > 0.02, f"新建窗口的勾选框没有主题色（占比 {share:.3f}）"
    finally:
        fresh.hide()
        fresh.deleteLater()


def test_the_proxy_style_survives_the_window_stylesheet(window, app):
    """The style must still be installed after the sheet goes on.

    ⚠️ ``QApplication.setStyleSheet`` silently replaces a custom ``QProxyStyle``
    with ``QCommonStyle``, which is why the check box drew the platform's blue
    even though ``Md3Style`` had been installed earlier. The sheet is applied
    per window instead, and this assertion keeps it that way -- it would have
    caught the original mistake immediately.
    """
    from tools.qt_theme import Md3Style

    window.apply_colour_scheme(scheme_from_seed("#6750A4", False))
    app.processEvents()
    assert isinstance(app.style(), Md3Style), (
        f"代理样式被换掉了，实际是 {type(app.style()).__name__}"
    )
