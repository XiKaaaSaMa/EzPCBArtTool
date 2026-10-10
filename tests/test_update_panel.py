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
