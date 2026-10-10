"""Fetch Anime2Sketch's architecture and weights for the redraw reading.

Kept as its own script because the two halves come from different places and
both are needed before :mod:`tools.probe_line_redraw` will run:

* the **architecture** from GitHub (``Mukosame/Anime2Sketch``, MIT). The probe
  imports this file directly instead of re-declaring the U-Net, so the weights
  are loaded by the exact code they were published with.
* the **weights** from the Hugging Face mirror, because the author's own links
  are Google Drive files and ``huggingface.co`` itself is unreachable from this
  machine while ``hf-mirror.com`` is not.

Two weight sets are downloaded, because the project uses the second one:

===============  ==================  ====================================
file             size                use
===============  ==================  ====================================
``netG.pth``     217631959 bytes     the original release
``improved.bin`` 191927595 bytes     **the one this project runs** -- the
                                     author's artifact-free retrain for dark
                                     and low-contrast pictures
===============  ==================  ====================================

⚠️ Fetching only ``netG.pth`` is the mistake this script used to make: the
loader asks for ``improved`` by default, so the download looked complete and
the probe still said the model was missing.

Everything lands in the folder :func:`pcbart.model_paths.default_model_dir`
reports -- ``out/_models/a2s`` from source, the per-user directory when
packaged. Outside the source tree either way, because 390 MB of third-party
weights are not ours to vendor. The user can also keep the folder somewhere
else entirely and point the program at it; see the module docstring of
:mod:`pcbart.model_paths`.

Usage::

    python tools/fetch_anime2sketch.py
    python tools/fetch_anime2sketch.py --check
    python tools/fetch_anime2sketch.py --outdir D:\\models\\a2s
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from pcbart.model_paths import default_model_dir  # noqa: E402

#: Default drop-in folder. Overridable with ``--outdir`` so a user can fetch
#: straight to wherever they keep models (another drive, a shared folder).
TARGET = default_model_dir()

GITHUB_API = "https://api.github.com/repos/Mukosame/Anime2Sketch/contents"
MIRROR = "https://hf-mirror.com/gyrojeff/Anime2Sketch/resolve/main"

#: Files taken from the reference repository. ``model.py`` is the architecture
#: and the only one the loader needs; ``data.py``, ``test.py`` and the licence
#: are kept for the record and for anyone re-running the author's own test
#: script. Their absence is reported but does not make the folder unusable,
#: which matters because a user who assembled the folder by hand will often
#: have copied just ``model.py`` and the weights.
FILES = ("model.py", "data.py", "test.py", "LICENSE")

#: The subset without which the loader cannot run.
REQUIRED_FILES = ("model.py",)

#: Weight sets to fetch, as ``filename -> (url, exact size)``.
#:
#: The sizes are checked because a truncated download still produces a file and
#: ``torch.load`` on it fails much later with a message that does not mention
#: the network at all. Both files come from the same mirror repository: it is
#: the one source reachable from here that carries the **improved** weights,
#: which the project actually runs.
WEIGHTS = {
    "netG.pth": (f"{MIRROR}/netG.pth", 217_631_959),
    "improved.bin": (f"{MIRROR}/improved.bin", 191_927_595),
}

#: Kept for callers that imported it. The default release's size.
WEIGHT_BYTES = WEIGHTS["netG.pth"][1]

#: First bytes of the two formats ``torch.load`` accepts, checked so a proxy or
#: captive portal's HTML error page -- served with status 200 -- is not written
#: to disk as a model. The two published files use different containers:
#: ``netG.pth`` is a raw pickle (``\x80``), while ``improved.bin`` was saved as
#: a zip archive (``PK\x03\x04``). Requiring the pickle magic for both is a
#: mistake this script already made; it reported a perfectly good file as
#: corrupt.
PICKLE_MAGIC = b"\x80"
ZIP_MAGIC = b"PK\x03\x04"

#: Accepted leading bytes per file.
MAGIC = {
    "netG.pth": PICKLE_MAGIC,
    "improved.bin": ZIP_MAGIC,
}


def _download(url: str, target: Path, *, retries: int = 3) -> None:
    last = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "pcbart-fetch/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data:
                raise ValueError("空响应")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            print(f"  第 {attempt} 次失败：{exc}")
    raise SystemExit(f"下载失败：{url}\n{last}")


def fetch_file(name: str, target: Path) -> None:
    """One file from the reference repo, through the contents API.

    The API is used rather than a raw URL because ``raw.githubusercontent.com``
    is unreachable here while ``api.github.com`` answers normally.
    """
    url = f"{GITHUB_API}/{name}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "pcbart-fetch/1.0", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("encoding") != "base64":
        raise SystemExit(f"{name}: 意外的编码 {payload.get('encoding')}")
    content = base64.b64decode(payload["content"])
    target.write_bytes(content)
    print(f"  {name}: {len(content)} 字节")


def check(target: Path | None = None) -> int:
    """Report what is present, without downloading anything.

    Reports **each weight set separately** rather than a single "weights ok":
    the project runs ``improved.bin``, so a folder holding only ``netG.pth``
    must not read as ready.
    """
    root = TARGET if target is None else Path(target)
    print(f"目录：{root}")

    ok = True
    for name in REQUIRED_FILES:
        present = (root / name).is_file()
        print(f"  {name}：{'OK' if present else '缺失'}")
        ok = ok and present

    optional = [name for name in FILES if name not in REQUIRED_FILES and not (root / name).is_file()]
    if optional:
        print(f"  另缺（不影响运行）：{optional}")

    for filename, (_url, expected) in WEIGHTS.items():
        path = root / "weights" / filename
        if not path.is_file():
            print(f"  {filename}：缺失")
            continue
        size = path.stat().st_size
        magic = path.read_bytes()[:4]
        good = size == expected and magic.startswith(MAGIC[filename])
        note = "OK" if good else f"异常（预期 {expected} 字节）"
        print(f"  {filename}：{size} 字节，magic {magic!r}，{note}")
        if filename == "improved.bin":
            # The one the project runs: without it the redraw reading cannot
            # load, so its absence is a failure even though netG.pth is not.
            ok = ok and good

    if not (root / "weights" / "improved.bin").is_file():
        print("  ⚠️ improved.bin 是项目实际使用的权重，缺了它重绘跑不起来")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查，不下载")
    parser.add_argument(
        "--outdir", default=None,
        help=f"下载到哪个文件夹（默认 {TARGET}）",
    )
    parser.add_argument(
        "--weights", nargs="*", default=list(WEIGHTS),
        choices=list(WEIGHTS),
        help="要取哪几套权重（默认两套都取）",
    )
    args = parser.parse_args(argv)

    target = Path(args.outdir) if args.outdir else TARGET
    if args.check:
        return check(target)

    target.mkdir(parents=True, exist_ok=True)
    (target / "weights").mkdir(exist_ok=True)

    print("取架构文件 …")
    for name in FILES:
        fetch_file(name, target / name)

    for filename in args.weights:
        url, expected = WEIGHTS[filename]
        path = target / "weights" / filename
        if path.exists() and path.stat().st_size == expected:
            print(f"{filename} 已存在，跳过（{expected} 字节）")
            continue
        print(f"取 {filename}（约 {expected / 1e6:.0f} MB）…")
        _download(url, path)
        size = path.stat().st_size
        if size != expected:
            print(f"  ⚠️ 大小 {size}，预期 {expected}；文件可能不完整")
        if not path.read_bytes()[:4].startswith(MAGIC[filename]):
            raise SystemExit(
                f"{filename} 的开头不是 torch 能读的格式，可能是错误页面"
            )

    print("\n完成。自检：")
    return check(target)


if __name__ == "__main__":
    raise SystemExit(main())
