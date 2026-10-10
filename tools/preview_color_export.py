"""Pull the embedded pictures back out of an ``.epro2``, as PNG files to look at.

Why this exists
---------------
A board exported as 彩色丝印 carries its artwork as ``OBJ`` records whose pixels
live in ``BLOB`` records (see :mod:`pcbart.color_image`). The colour only becomes
a *print* when EasyEDA runs its own colour-silkscreen export, so until that file
reaches the EDA there is nothing on screen to check.

This walks the file, decodes every ``BLOB``, and writes each picture out as a
PNG named after its record. What comes back is exactly the bytes the EDA will
receive -- not a re-render, not a preview of our own making -- so "is the colour
actually in there" is answered by looking at a file rather than by trusting a
counter.

Usage::

    python tools/preview_color_export.py out/2026.10.8test/*.epro2
    python tools/preview_color_export.py --outdir out/2026.10.8test/_preview out/2026.10.8test
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcbart.epru import EproArchive  # noqa: E402


def _picture_bytes(content: str) -> bytes | None:
    """The raw bytes behind a ``data:image/...;base64,`` URL, or ``None``.

    Returns ``None`` rather than raising for anything that is not a data URL:
    the format allows a BLOB to hold other payloads, and one unrecognised
    record should not stop the rest of the file from being inspected.
    """
    if "base64," not in content:
        return None
    try:
        return base64.b64decode(content.split("base64,", 1)[1])
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None


def _safe_stem(name: str) -> str:
    """A file-name stem that cannot escape the output directory."""
    stem = Path(name).stem or "picture"
    for bad in '<>:"/\\|?*':
        stem = stem.replace(bad, "_")
    return stem.strip() or "picture"


def extract(path: Path, outdir: Path) -> list[tuple[Path, tuple[int, int], int]]:
    """Write every picture in ``path`` to ``outdir``. Returns what was written."""
    from PIL import Image

    document = EproArchive.read(path).document
    blobs = list(document.iter_type("BLOB"))
    written: list[tuple[Path, tuple[int, int], int]] = []

    for index, record in enumerate(blobs, start=1):
        raw = _picture_bytes(record.body.get("content", ""))
        if raw is None:
            continue
        # Re-encoded through Pillow so the suffix is honest and a truncated
        # payload fails here, where it can be reported, rather than silently
        # producing a file that only looks fine until it is opened.
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                frame = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ! {record.id[:12]}: 读不出像素（{exc}）", file=sys.stderr)
            continue

        stem = _safe_stem(record.body.get("filename", "") or f"blob{index}")
        target = outdir / f"{_safe_stem(path.name)}__{index:02d}_{stem}.png"
        frame.save(target)
        colours = len(frame.convert("RGB").getcolors(maxcolors=1 << 22) or [])
        written.append((target, frame.size, colours))

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path, help="要回读的 .epro2")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="预览输出目录（默认：每个文件旁边的 _preview 子目录）",
    )
    args = parser.parse_args(argv)

    total = 0
    for path in args.files:
        if not path.exists():
            print(f"跳过（不存在）：{path}", file=sys.stderr)
            continue
        outdir = args.outdir or (path.parent / "_preview")
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"{path.name}")
        items = extract(path, outdir)
        for target, size, colours in items:
            total += 1
            print(f"  {size[0]:>4}×{size[1]:<4} 颜色 {colours:>6}  {target.name}")
        if not items:
            print("  （没有可回读的图片）")

    print(f"\n共写出 {total} 张预览。")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
