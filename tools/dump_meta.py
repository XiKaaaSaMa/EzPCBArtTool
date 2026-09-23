#!/usr/bin/env python
"""Print the project2.json and archive layout of an .epro2 file."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
with zipfile.ZipFile(path) as archive:
    print(f"archive: {path}")
    for info in archive.infolist():
        print(f"  {info.filename:<32} {info.file_size} bytes")
    print()
    print("project2.json:")
    print(archive.read("project2.json").decode("utf-8"))
