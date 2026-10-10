"""Shared test setup.

**The draft library is redirected for every test.** ``DraftStore()`` with no
argument resolves to :func:`pcbart.paths.drafts_dir`, which from a source run
is the real ``out/drafts`` -- the user's own drafts. Saving a product file now
also stashes the editing session there, so a test that builds a real window
and exports writes into the developer's sidebar. That is not hypothetical: it
left eleven empty ``猫娘板`` / ``out`` rows in the real library during the
0.1.1 work, each named after whatever the test happened to be doing.

Redirecting ``drafts_dir`` itself rather than relying on each test file to
override ``window._draft_store`` means the default is safe. A test author who
forgets writes into ``tmp_path`` and never notices; the alternative is writing
into somebody's work. Tests that *want* to point at a specific folder still
pass an explicit ``root`` and are unaffected.

**The colour scheme is pinned for every test, for the same kind of reason.**
:mod:`tools.gui_mockup` binds its colour names from the saved preferences when
it is first imported, so a developer who had picked a red theme would see
colour-dependent assertions (and every screenshot comparison) behave
differently from CI -- a failure that cannot be reproduced without knowing
somebody's personal settings. Pinning it to the Material baseline makes the
suite describe one interface instead of whichever one happens to be saved.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def draft_library_is_scratch(tmp_path, monkeypatch):
    """Any default draft library resolves inside this test's tmp folder."""
    from pcbart import paths

    # Named ``drafts`` on purpose: what a test is usually checking is that the
    # library ends up in the right *kind* of place, not that it is outside
    # ``tmp_path`` specifically.
    scratch = tmp_path / "drafts"
    monkeypatch.setattr(paths, "drafts_dir", lambda: scratch)
    return scratch


@pytest.fixture(autouse=True)
def colour_scheme_is_pinned(tmp_path, monkeypatch):
    """The interface runs on the Material baseline, never a saved preference.

    ``settings_path`` is redirected *and* the generated default is re-applied,
    so both halves of the problem are covered: tests that save a scheme write
    into ``tmp_path``, and tests that never touch it still start from a known
    palette rather than from the developer's own.
    """
    from pcbart import ui_theme

    scratch = tmp_path / "settings.json"
    monkeypatch.setattr(ui_theme, "settings_path", lambda: scratch)

    baseline = ui_theme.scheme_from_seed(ui_theme.DEFAULT_SEED, dark=False)

    # ``gui_mockup`` may not be imported yet (most of the suite is pure Python),
    # so it is looked up in sys.modules rather than imported here.
    mockup = sys.modules.get("tools.gui_mockup") or sys.modules.get("gui_mockup")
    if mockup is not None and hasattr(mockup, "set_palette"):
        previous = mockup.current_scheme()
        mockup.set_palette(baseline)
        yield baseline
        mockup.set_palette(previous)
        return
    yield baseline

def release_window(window) -> None:
    """Free a test window and everything the module keeps pointing at it.

    ⚠️ **``hide()`` frees nothing and ``deleteLater()`` only queues.** A plain
    ``processEvents()`` does not run deferred deletes -- they need an explicit
    ``sendPostedEvents(None, QEvent.DeferredDelete)``. Without that, a file that
    builds a window per test leaves every one of them alive with all of its
    widgets, and a later theme switch re-polishes all of them: measured at
    4 seconds with twenty windows open against 40 ms with one.

    The floating panels live in module-level globals, so they are cleared too.
    A global pointing at a deleted panel is worse than a stale window -- the
    next ``show_preprocess()`` call reaches into a dead C++ object.

    ⚠️ **The module-level panel globals are *not* cleared here.** A fixture may
    build a panel that a later test in the same file still holds, so nulling the
    global on this window's teardown would reach into an object someone else is
    using -- which is exactly the crash this helper first caused. The autouse
    guard below clears a global only once it can see the object is already gone.

    Call this from a fixture's teardown for the window that fixture built.
    """
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:  # PySide6 absent; the Qt tests skip themselves
        return
    if QApplication.instance() is None:
        return

    window.hide()
    window.deleteLater()
    # Deferred deletes only run on this specific call.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def dead_panel_globals_are_cleared():
    """Drop module-level panel references whose C++ object has been deleted.

    ``tools.gui_mockup`` caches one floating panel per kind in a module global
    and reuses it while its parent window is alive. A test that frees its window
    therefore leaves the global pointing at a deleted object, and the *next*
    test to open that panel type trips over it -- an error that looks like a
    broken panel rather than a stale pointer.

    ⚠️ **Only dead references are cleared.** Nulling them more eagerly (for
    example whenever any window is released) pulls the panel out from under a
    later test in the same file that deliberately kept it; that mistake is what
    this narrow check replaces. ``sip.isdeleted`` is the only reliable way to
    ask, and the attribute check keeps the helper working on older bindings.
    """
    import sys

    yield

    mockup = sys.modules.get("tools.gui_mockup") or sys.modules.get("gui_mockup")
    if mockup is None:
        return
    for name in ("_preprocess_panel", "_board_panel", "_settings_panel"):
        # ⚠️ Reading the attribute is itself what raises once the C++ object is
        # gone: the global holds a Python wrapper whose C++ half was destroyed,
        # and *touching* it -- even printing it -- raises RuntimeError. There is
        # no "inspect it first" step available, so the failure to read is the
        # signal.
        try:
            panel = getattr(mockup, name, None)
        except RuntimeError:
            setattr(mockup, name, None)
            continue
        if panel is None:
            continue
        deleted = False
        try:
            import shiboken6

            deleted = not shiboken6.Shiboken.isValid(panel)
        except Exception:  # noqa: BLE001 - absence just means "try the call"
            try:
                panel.parentWidget()
            except RuntimeError:
                deleted = True
            except Exception:  # noqa: BLE001
                deleted = False
        if deleted:
            setattr(mockup, name, None)


@pytest.fixture(autouse=True)
def leftover_windows_are_freed():
    """Free windows a test built directly and left behind.

    Many tests call ``build_window()`` in the test body rather than in a
    fixture, so there is no teardown to hook; they simply go out of scope holding
    a live Qt window. Those accumulate for the whole session, and *every* one of
    them is re-polished whenever the application stylesheet is replaced -- which
    is what a colour-theme change does. Measured: 169 widgets per window at
    ~63 ms per repolish, so a few hundred leftovers turn a 60 ms switch into 40
    seconds and hang the suite.

    ⚠️ **Only *hidden* windows are freed.** A window a fixture is still using is
    visible for the duration of its test, so this cannot pull one out from under
    it -- which is exactly what an earlier version of this cleanup did, and why
    it is scoped to visibility rather than to ownership.
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:  # PySide6 absent; the Qt tests skip themselves
        return
    app = QApplication.instance()
    if app is None:
        return

    leftovers = [
        widget
        for widget in app.topLevelWidgets()
        if type(widget).__name__ == "MainWindow" and not widget.isVisible()
    ]
    if not leftovers:
        return
    for widget in leftovers:
        widget.deleteLater()
    # Deferred deletes only run on this specific call.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
