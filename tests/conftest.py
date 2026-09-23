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
"""

from __future__ import annotations

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
