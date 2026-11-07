"""The shipped roster and authored templates must work without external files."""
from pathlib import Path
import sys

import pytest

from pcbart import operators, paths
from pcbart.epru import EproArchive


def test_source_assets_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ASSET_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.asset_root() == Path(__file__).resolve().parents[1] / "assets"


@pytest.mark.parametrize("frozen", [False, True])
def test_explicit_library_override_is_authoritative(monkeypatch, tmp_path, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    missing = tmp_path / "missing-library"
    monkeypatch.setenv(paths.ASSET_ROOT_ENV, str(missing))
    assert paths.asset_root() == missing
    assert not operators.load()


def test_frozen_assets_are_read_from_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ASSET_ROOT_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "中文安装目录" / "_internal"), raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.asset_root() == Path(sys._MEIPASS) / "assets"


def test_frozen_fallback_uses_executable_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ASSET_ROOT_ENV, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "EzPCBArtTool.exe"))
    assert paths.asset_root() == tmp_path / "assets"


def test_bundled_roster_and_all_referenced_icons_are_present(monkeypatch):
    monkeypatch.delenv(paths.ASSET_ROOT_ENV, raising=False)
    roster = operators.load()
    assert len(roster) > 400
    assert roster.search("阿米娅")
    for person in roster.operators:
        for relative in (person.faction_logo, person.class_icon_black, person.class_icon_white):
            if relative:
                assert operators.asset_path(relative).is_file(), (person.name, relative)
    assert len(list((paths.badge_asset_dir() / "18-阵营logo").glob("*.png"))) == 48


@pytest.mark.parametrize("name", ["模板-精0、精1、皮肤小.epro2", "模板-精2、皮肤大.epro2"])
def test_bundled_source_templates_can_be_read(monkeypatch, name):
    monkeypatch.delenv(paths.ASSET_ROOT_ENV, raising=False)
    source = paths.asset_root() / "通行证模板源文件" / name
    assert source.is_file()
    assert EproArchive.read(source).document.find("STRING")
