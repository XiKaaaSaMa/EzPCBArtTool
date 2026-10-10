"""Regenerate :mod:`pcbart.face_model` from the downloaded ONNX weights.

The weights are vendored as a Python module rather than shipped as a data file,
for the two reasons spelled out in that module's docstring: a packaged build
has to find data files through PyInstaller's bundle rules, and OpenCV cannot
open a path containing non-ASCII characters on Windows -- which this project
has. Re-run this after replacing ``pcbart/models/face.onnx``.

Usage::

    python tools/embed_face_model.py
"""

from __future__ import annotations

import base64
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "pcbart" / "models" / "face.onnx"
TARGET = ROOT / "pcbart" / "face_model.py"

DOCSTRING = '''"""The face detector's weights, as data.

**Generated -- do not edit by hand.** Rebuild with
```
python tools/embed_face_model.py
```

YuNet (OpenCV Zoo, Apache-2.0), 227 KB, used to find the face in an
illustration so the crop can put it on the centre axis. Heuristics were tried
first and none held up: skin tone catches hair and weapon highlights, the
darkest-blob pair that should be eyes locked onto a boot buckle, Hough circles
found pauldrons, and edge density found hair strands. Against the picture the
user complained about, the best heuristic sat 53 px off the face.

Why this is a Python module rather than the ``.onnx`` file next to it: the same
two reasons the icons are inlined.

* A packaged build unpacks to a temporary directory, so a data file has to be
  found through PyInstaller's bundle rules, and a model that misses that lookup
  is a feature that silently disappears.
* OpenCV cannot open a path containing non-ASCII characters on Windows. This
  project lives under a Chinese directory name, so a path to the model works
  when the process happens to start in the project root and fails as soon as it
  is made absolute -- measured both ways. The bytes are written to a temporary
  file with an ASCII name at run time instead; handing the model over as a
  bytes object segfaults OpenCV 5.0, so a real file is required.
"""

#: Base64 of ``face_detection_yunet_2023mar.onnx``.
FACE_MODEL_B64 = (
'''


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"找不到模型：{SOURCE}")
    payload = base64.b64encode(SOURCE.read_bytes()).decode("ascii")
    lines = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(payload, 96))
    TARGET.write_text(DOCSTRING + lines + "\n)\n", encoding="utf-8")
    print(f"已写出 {TARGET}（{TARGET.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
