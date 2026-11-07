"""Read every bundled asset using the same paths as the installed application."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

from pcbart import operators
from pcbart.epru import EproArchive
from pcbart.paths import asset_root, badge_asset_dir, is_frozen


def verify() -> dict:
    root = asset_root()
    library = badge_asset_dir()
    roster = operators.load()
    if len(roster) < 400 or not roster.search("阿米娅"):
        raise ValueError("Bundled operator database is missing or incomplete")
    references = 0
    for person in roster.operators:
        for relative in (person.faction_logo, person.class_icon_black, person.class_icon_white):
            if relative:
                if not operators.asset_path(relative).is_file():
                    raise FileNotFoundError(relative)
                references += 1
    images = svgs = 0
    for path in library.rglob("*"):
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                image.verify()
            images += 1
        elif path.suffix.lower() == ".svg":
            if ET.parse(path).getroot().tag.rsplit("}", 1)[-1] != "svg":
                raise ValueError(f"Invalid SVG: {path}")
            svgs += 1
    templates = {}
    for name in ("模板-精0、精1、皮肤小.epro2", "模板-精2、皮肤大.epro2"):
        source = root / "通行证模板源文件" / name
        strings = EproArchive.read(source).document.find("STRING")
        if not strings:
            raise ValueError(f"Empty template: {source}")
        templates[name] = len(strings)
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (library, root / "通行证模板源文件")
        for path in sorted(folder.rglob("*")) if path.is_file()
    }
    return {"frozen": is_frozen(), "asset_root": str(root), "operators": len(roster),
            "icon_references": references, "png_images": images, "svg_images": svgs,
            "templates": templates, "files": len(hashes), "sha256": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0
