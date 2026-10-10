"""Tests for the update panel and the startup check wiring.

The panel is driven through a real window, the same way the other GUI tests
work, and every dialog is answered so an unexpected one cannot hang the run.

No test reaches the network: the checker is monkeypatched at the module
boundary the workers import from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import UpdatePanel, build_window, show_update  # noqa: E402

from pcbart.update_check import ReleaseInfo  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def no_modal_can_block(monkeypatch):
    """Answer every dialog, so a half-written test cannot hang the run."""

    def answer(box, *args, **kwargs):
        buttons = {button.text(): button for button in box.buttons()}
        monkeypatch.setattr(
            box, "clickedButton", lambda: buttons.get("取消") or None
        )
        return QMessageBox.RejectRole.value

    monkeypatch.setattr(QMessageBox, "exec", answer)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(answer))


@pytest.fixture
def window(app):
    """A real window.

    ``tests/conftest.py`` already redirects the draft library to a scratch
    folder for the whole suite, so nothing here needs to stage a store.
    """
    widget = build_window()
    yield widget
    widget.hide()
    widget.deleteLater()


def _release(**overrides) -> ReleaseInfo:
    payload = {
        "version": "1.1.0",
        "date": "2026-11-01",
        "notes": ("新增：更新检查", "修复：某处崩溃"),
        "download_url": "https://example.test/EzPCBArtTool_v1.1.0_安装包.exe",
        "size": 50143110,
        "sha256": "a" * 64,
        "download_page": "https://www.xikaaa.cn/tools/pcb-art/",
    }
    payload.update(overrides)
    return ReleaseInfo(**payload)


# ---------------------------------------------------------------- panel


def test_panel_shows_the_version_and_every_note(window):
    """The two things a user reads must both be on screen.

    **Injection check**: drop the notes loop and this goes red; the version
    label alone would still pass a weaker assertion.
    """
    release = _release()
    panel = show_update(window, release)

    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert any("v1.1.0" in text for text in texts)
    assert any("新增：更新检查" in text for text in texts)
    assert any("修复：某处崩溃" in text for text in texts)


def test_panel_defaults_to_the_download_button(window):
    """The primary action is the download -- that is the whole feature."""
    panel = show_update(window, _release())
    assert panel.download_button.text() == "下载并安装"
    assert panel.download_button.isEnabled() is True
    assert panel.later_button.text() == "稍后"


def test_panel_survives_a_release_with_no_notes(window):
    """An empty changelog must not produce an empty box or a crash."""
    panel = show_update(window, _release(notes=()))
    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert any("没有写说明" in text for text in texts)


def test_reopening_replaces_the_previous_panel(window):
    """Two update panels stacked in the same corner would overlap."""
    first = show_update(window, _release(version="1.1.0"))
    second = show_update(window, _release(version="1.2.0"))
    assert first is not second
    assert gui_mockup.live_panel("_update_panel") is second


def test_buttons_stay_visible_with_a_long_changelog(window):
    """A long changelog must not push the action buttons out of the panel.

    This was a real defect: the panel nested its own ``QScrollArea`` inside the
    one ``FloatingPanel`` provides, so the content grew past the panel and both
    buttons ended up below the visible area -- precisely when the user had just
    started a download and needed the status line.

    **Injection check**: put the nested scroller back (or move the buttons into
    ``content_layout``) and the geometry assertion goes red.
    """
    from PySide6.QtWidgets import QScrollArea

    long_release = _release(
        notes=tuple(f"修复：第 {index} 条足够长的更新说明，用来把内容撑高" for index in range(40))
    )
    panel = show_update(window, long_release)
    window.show()
    for _ in range(3):
        QApplication.processEvents()

    # The footer is pinned: it must not be inside the scrolling area.
    assert panel.findChildren(QScrollArea), "面板本身应当有滚动区"
    inner = [
        area
        for area in panel.findChildren(QScrollArea)
        if area is not panel._scroll
    ]
    assert inner == [], "不应再有第二层滚动区（嵌套滚动会把按钮挤出面板）"

    footer = panel.download_button.parentWidget()
    assert panel._footer.isAncestorOf(panel.download_button), (
        "按钮必须挂在固定页脚里，而不是滚动内容中"
    )
    assert not panel._scroll.isAncestorOf(panel.download_button), (
        "按钮不能在滚动区里，否则长日志会把它推走"
    )

    for button in (panel.download_button, panel.later_button):
        bottom = button.mapTo(panel, button.rect().bottomLeft()).y()
        assert bottom < panel.height(), (
            f"{button.text()} 被挤出面板（bottom={bottom}, panel={panel.height()}）"
        )


def test_status_line_is_visible_while_downloading(window):
    """The progress text must be on screen, not scrolled away."""
    long_release = _release(
        notes=tuple(f"新增：第 {index} 条说明" for index in range(40))
    )
    panel = show_update(window, long_release)
    window.show()
    panel.set_progress(12_345_678, 50_144_467)
    for _ in range(3):
        QApplication.processEvents()

    assert panel.status_label.isVisible() is True
    bottom = panel.status_label.mapTo(
        panel, panel.status_label.rect().bottomLeft()
    ).y()
    assert bottom < panel.height(), "下载进度文字被挤出面板"


def test_progress_and_failure_are_reported(window):
    panel = show_update(window, _release())
    panel.set_progress(1024 * 1024, 4 * 1024 * 1024)
    assert "1.0 / 4.0 MB" in panel.status_label.text()
    panel.finish("下载失败，请检查网络后重试。", failed=True)
    assert panel.download_button.isEnabled() is True
    assert panel.download_button.text() == "重试"


def test_download_button_only_fires_once(window):
    """A double click before the worker starts must not start two downloads."""
    calls: list[int] = []
    panel = show_update(window, _release(), on_download=lambda p: calls.append(1))
    panel._start_download()
    panel._start_download()
    assert calls == [1]
    assert panel.download_button.isEnabled() is False


# --------------------------------------------------------- startup check


def _run_workers(app):
    """Let any queued worker threads finish so their signals are delivered."""
    for _ in range(60):
        app.processEvents()
        if not QApplication.instance().thread() or True:
            pass
    app.processEvents()


def test_automatic_check_stays_silent_when_there_is_no_update(
    window, monkeypatch, app
):
    """No update, no dialog. A prompt every launch would be worse than none."""
    opened: list[str] = []

    def fake_check(local_version, **kwargs):
        from pcbart.update_check import UpdateStatus

        return UpdateStatus(release=None, available=False, reason="检查失败")

    import pcbart.update_check as uc

    monkeypatch.setattr(uc, "check_for_update", fake_check)
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: opened.append(a[2] if len(a) > 2 else "")),
    )

    gui_mockup.start_update_check(window, manual=False)
    worker = getattr(window, "_update_check_worker", None)
    if worker is not None:
        worker.wait(5000)
    app.processEvents()
    assert opened == []


def test_manual_check_answers_when_there_is_no_update(window, monkeypatch, app):
    """The button has to say something -- silence looks like a broken button."""
    opened: list[str] = []

    def fake_check(local_version, **kwargs):
        from pcbart.update_check import UpdateStatus

        return UpdateStatus(release=None, available=False, reason="检查失败")

    import pcbart.update_check as uc

    monkeypatch.setattr(uc, "check_for_update", fake_check)
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: opened.append(str(a[2]) if len(a) > 2 else "")),
    )

    gui_mockup.start_update_check(window, manual=True)
    worker = getattr(window, "_update_check_worker", None)
    if worker is not None:
        worker.wait(5000)
    app.processEvents()
    assert opened, "手动检查必须给出反馈"
    assert "没有可用更新" in opened[0]


def test_check_offers_the_panel_when_an_update_exists(window, monkeypatch, app):
    """An available release must open the panel, not a bare message box."""
    from pcbart.update_check import UpdateStatus

    def fake_check(local_version, **kwargs):
        return UpdateStatus(release=_release(), available=True)

    import pcbart.update_check as uc

    monkeypatch.setattr(uc, "check_for_update", fake_check)

    gui_mockup.start_update_check(window, manual=False)
    worker = getattr(window, "_update_check_worker", None)
    if worker is not None:
        worker.wait(5000)
    app.processEvents()

    panel = gui_mockup.live_panel("_update_panel")
    assert panel is not None, "有更新时必须弹出更新面板"


# -------------------------------------------------------------- settings


def test_manual_button_works_with_automatic_checks_disabled(window, monkeypatch):
    """Turning automatic checks off must not remove the manual entry point.

    This is the same defect class as "only the script can call it": a feature
    whose only trigger is automation disappears the moment that automation is
    switched off.
    """
    from pcbart.ui_theme import load_settings, save_settings_keeping

    save_settings_keeping(update_check=False)
    try:
        panel = gui_mockup.show_settings(window)
        assert panel is not None
        assert panel.update_check_box.isChecked() is False
        assert panel.update_now_button.isEnabled() is True
    finally:
        save_settings_keeping(update_check=True)
        assert load_settings().get("update_check") is True


def test_settings_checkbox_writes_the_preference(window):
    from pcbart.ui_theme import load_settings, save_settings_keeping

    save_settings_keeping(update_check=True)
    panel = gui_mockup.show_settings(window)
    assert panel is not None
    panel.update_check_box.setChecked(False)
    assert load_settings().get("update_check") is False
    panel.update_check_box.setChecked(True)
    assert load_settings().get("update_check") is True


def test_default_is_to_check():
    """A fresh install checks. Missing preference reads as on, not off."""
    from pcbart.ui_theme import save_settings

    saved = gui_mockup._auto_check_enabled_setting()
    save_settings({})
    try:
        assert gui_mockup._auto_check_enabled_setting() is True
    finally:
        from pcbart.ui_theme import save_settings_keeping

        save_settings_keeping(update_check=saved)
