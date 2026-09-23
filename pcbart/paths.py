"""Where the program is allowed to keep its files.

Running from source, next to the code is convenient: everything the tool
produces lands in ``out/`` where the developer can look at it. Running from a
packaged build, the same assumption breaks twice over:

**The program directory may be read-only.** An installer puts the application
under ``C:\\Program Files``, which a standard user cannot write to. Anything
written there fails with a permission error that looks like a corrupt install.

**``__file__`` may point at a temporary directory.** A one-file build unpacks
itself somewhere under ``%TEMP%`` on every launch and deletes it again on exit,
so a path derived from ``__file__`` is both the wrong place and a place that
will not exist next time the application starts.

Both are invisible until the program is packaged, which is why the paths live
here rather than being spelled out at each call site.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "is_frozen",
    "data_root",
    "cache_dir",
    "drafts_dir",
    "default_output_dir",
]

APP_DIR_NAME = "PCB艺术板工具"


def is_frozen() -> bool:
    """Whether this process is a packaged build rather than a source checkout.

    PyInstaller sets ``sys.frozen`` and puts the unpack directory in
    ``sys._MEIPASS``. Checking the attribute rather than a flag of our own
    means the answer is right even when the build tool changes.
    """
    return bool(getattr(sys, "frozen", False))


def data_root() -> Path:
    """A writable directory for files the program creates at runtime.

    Frozen builds get the per-user application-data directory, which is
    writable without elevation and survives upgrades. Source runs keep using
    the project folder, so development stays in one place and ``out/`` remains
    the directory the docs and tests talk about.

    Falls back to the project folder when the system offers no application-data
    directory at all -- unusual, but a working fallback beats a crash on
    startup.
    """
    if not is_frozen():
        return Path(__file__).resolve().parent.parent

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_DIR_NAME
    return Path.home() / f".{APP_DIR_NAME}"


def cache_dir(name: str = "lineart_cache") -> Path:
    """A writable cache directory, created if it is not there yet.

    Deliberately **not** a temporary directory. The document stores these
    paths, so a cache the OS may clean up would leave the canvas with nothing
    to draw after a restart -- a bug this project has already fixed once.
    """
    root = data_root() / "cache" / name if is_frozen() else data_root() / "out" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def drafts_dir() -> Path:
    """Where the draft library lives, created if it is not there yet.

    Follows the same split as :func:`cache_dir`: from source the drafts sit in
    ``out/drafts`` beside everything else the tool produces, and a packaged
    build keeps them in the per-user directory, which is writable without
    elevation and survives upgrades.

    **Not a temporary directory, and not inside the install folder.** The draft
    library is the user's work; a location the OS may clean up, or that an
    upgrade replaces, would lose it.

    ``PCBART_DRAFTS_DIR`` overrides the location. It exists because scripts that
    drive the real window -- a screenshot run, a diagnostic -- **write into the
    user's library when the window closes**, since the app saves unwritten work
    on exit. That is correct behaviour for the app and a hazard for a script:
    during this project's own debugging it put a dozen ``未命名`` rows into the
    developer's sidebar (twice). The test suite is already covered by
    ``tests/conftest.py``; this is for everything else.

    set PCBART_DRAFTS_DIR=%TEMP%/scratch-drafts
    """
    override = os.environ.get("PCBART_DRAFTS_DIR")
    if override:
        root = Path(override)
    else:
        root = data_root() / "drafts" if is_frozen() else data_root() / "out" / "drafts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_output_dir() -> Path:
    """Where generated sample files go when no directory is given.

    The CLI tools default to ``out/`` relative to the working directory, which
    is fine from source and surprising from a packaged build: a shortcut starts
    the program with the working directory set to the shortcut's own location,
    so files appear wherever the shortcut happens to live. Frozen builds get a
    predictable folder instead.
    """
    if is_frozen():
        root = data_root() / "out"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return Path("out")
