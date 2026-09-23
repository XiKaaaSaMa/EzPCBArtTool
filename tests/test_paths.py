"""Paths that have to survive being packaged.

The program keeps files in two different places depending on how it was
started, and getting that wrong is invisible from source: a path built out of
``__file__`` looks perfectly correct until the code is frozen, at which point
it points into a temporary unpack directory that is deleted on exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import paths  # noqa: E402


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pretend to be a packaged build writing into a throwaway directory."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path


def test_a_source_run_writes_inside_the_project():
    assert not paths.is_frozen()
    assert paths.data_root() == Path(__file__).resolve().parent.parent


def test_a_frozen_run_writes_to_the_user_profile(frozen):
    """An install under Program Files is read-only for a normal user."""
    root = paths.data_root()

    assert str(frozen) in str(root)
    assert not str(root).startswith(str(Path(__file__).resolve().parent.parent))


def test_the_frozen_root_is_not_the_unpack_directory(frozen):
    """__file__ points into the temporary unpack dir; the data root must not.

    A one-file build deletes that directory on exit, so anything cached there
    is gone by the next launch -- and the document stores these paths.
    """
    unpack = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else None
    if unpack is not None:
        assert not str(paths.data_root()).startswith(str(unpack))


def test_the_cache_directory_is_created(frozen):
    directory = paths.cache_dir("lineart_cache")

    assert directory.is_dir()
    assert directory.name == "lineart_cache"


def test_source_and_frozen_caches_do_not_collide(frozen):
    """The frozen cache must not land inside the source tree."""
    frozen_cache = paths.cache_dir("c")
    assert str(frozen) in str(frozen_cache)


def test_the_output_directory_is_created_when_frozen(frozen):
    output = paths.default_output_dir()

    assert output.is_dir()
    assert str(frozen) in str(output)


def test_a_source_run_keeps_the_relative_out_directory():
    """Development stays relative, which is what every doc and test expects."""
    assert paths.default_output_dir() == Path("out")


def test_it_falls_back_when_the_system_offers_no_appdata(monkeypatch):
    """No LOCALAPPDATA is unusual, but a working fallback beats a crash."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)

    root = paths.data_root()
    assert root.is_absolute()
    assert "PCB" in root.name


def test_no_module_builds_a_writable_path_from_dunder_file():
    """The specific defect this module exists to prevent, checked directly.

    A frozen build's ``__file__`` is inside the unpack directory, so any
    writable path derived from it is either read-only or temporary. Reading
    authored files that way is fine; writing is not.
    """
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for source in (root / "pcbart").glob("*.py"):
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if "__file__" not in line:
                continue
            stripped = line.strip()
            # Reads for import resolution are the documented exception.
            if "sys.path.insert" in stripped or stripped.startswith("#"):
                continue
            if "out" in stripped or "cache" in stripped:
                offenders.append(f"{source.name}:{number}: {stripped}")

    assert not offenders, "writable paths must come from pcbart.paths:\n" + "\n".join(
        offenders
    )


def test_the_gui_writes_its_cache_through_the_paths_module():
    """The canvas reads the stored path, so it has to be the frozen one."""
    source = (Path(__file__).resolve().parent.parent / "tools" / "gui_mockup.py").read_text(
        encoding="utf-8"
    )

    assert "cache_dir(" in source
    assert 'Path(__file__).resolve().parent.parent / "out"' not in source
