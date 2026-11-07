"""Upgrades must replace the private dependency tree without touching user data."""
from pathlib import Path
import re


def test_upgrade_replaces_only_the_program_dependency_directory():
    script = (Path(__file__).resolve().parents[1] / "installer.iss").read_text(encoding="utf-8")
    section = re.search(r"(?ims)^\[InstallDelete\]\s*\n(.*?)(?=^\[|\Z)", script)
    assert section is not None, "Old Qt/Shiboken DLLs must be removed before copying new dependencies"
    entries = [line.strip() for line in section.group(1).splitlines()
               if line.strip() and not line.lstrip().startswith(";")]
    assert entries == ['Type: filesandordirs; Name: "{app}\\_internal"']
