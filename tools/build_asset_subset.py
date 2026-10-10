"""Copy the artwork the template actually needs into its own library.

The full PRTS mirror is 4.1 GB, most of it in the 立绘 (full-body illustration)
sets, the 半身像 sets and the 蚀刻章 medals. **The badge does not use any of
them any more** -- the user dropped the illustration sets deliberately, so the
subset library is now icons and generated codes only, a few megabytes instead
of four gigabytes.

What the card still places:

* the operator index (``01-干员代号``), which answers "which number, class and
  faction goes with this operator";
* 精英化进度图标, 职业大图 (black and white), 团队logo and 阵营logo;
* the QR codes and Code 128 barcodes written by ``build_badge_codes.py``.

The copy is a plain folder copy and **refuses to delete anything**. An earlier
version pruned files that were no longer in its group list, which is exactly
the wrong behaviour now: the generated code folders live inside the destination
and are owned by a different tool. Retired folders are reported instead.

Usage::

    python tools/build_asset_subset.py                    # copy + report
    python tools/build_asset_subset.py --check            # report only

The output location is ``--out``; the generated codes are copied too, so point
this at an existing library after running ``build_badge_codes.py`` into it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_prts_assets import DEFAULT_OUT  # noqa: E402

DEFAULT_SUBSET = DEFAULT_OUT.parent / "明日方舟-艺术板素材"

#: Top-level folders the card composes. Everything else in the source mirror --
#: 立绘, 半身像, 蚀刻章, 技能图标, 召唤物头像 -- is intentionally left out.
SUBSET_FOLDERS: tuple[str, ...] = (
    "01-干员代号",
    "13-精英化进度图标",
    "15-职业大图",
    "16-团队logo",
    "17-皮肤logo（大）",
    "18-阵营logo",
)

#: Written later by ``build_badge_codes.py``; listed so the report can say
#: whether they are present rather than silently pretending they should be.
GENERATED_FOLDERS: tuple[str, ...] = (
    "20-干员二维码",
    "21-情报编号条形码",
)

#: Folders that used to be part of the subset and are deliberately gone. Named
#: so the check can tell the user what to delete by hand -- this tool will not.
RETIRED_FOLDERS: tuple[str, ...] = (
    "06-立绘-精英0",
    "07-立绘-精英1",
    "08-立绘-精英2",
    "09-立绘-皮肤",
    "02-半身像-精英0",
    "03-半身像-精英1",
    "04-半身像-精英2",
    "05-半身像-皮肤",
    "19-蚀刻章",
)


def human(number: int) -> str:
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if number >= size:
            return f"{number / size:.1f} {unit}"
    return f"{number} B"


def folder_size(root: Path) -> tuple[int, int]:
    """``(file_count, total_bytes)`` for a folder, missing folders being empty."""
    if not root.is_dir():
        return 0, 0
    files = [item for item in root.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def copy_folder(source: Path, destination: Path, *, check: bool) -> tuple[int, int]:
    """Copy one folder tree, skipping files already present at the same size.

    Returns ``(copied, total)`` files. A file that exists at a different size is
    overwritten, which is how a re-sourced icon gets updated without the caller
    having to delete it first.
    """
    if not source.is_dir():
        return 0, 0
    copied = 0
    total = 0
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        total += 1
        target = destination / item.relative_to(source)
        if check:
            if target.exists() and target.stat().st_size == item.stat().st_size:
                copied += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != item.stat().st_size:
            shutil.copy2(item, target)
        copied += 1
    return copied, total


def build(source: Path, destination: Path, *, check: bool) -> dict:
    report: dict[str, dict] = {}
    missing: list[str] = []
    total_files = 0
    total_bytes = 0

    for name in SUBSET_FOLDERS:
        origin = source / name
        if not origin.is_dir():
            missing.append(name)
            report[name] = {"copied": 0, "files": 0, "bytes": 0}
            print(f"{name:20s} 源库不存在")
            continue
        copied, files = copy_folder(origin, destination / name, check=check)
        _count, size = folder_size(origin)
        total_files += files
        total_bytes += size
        report[name] = {"copied": copied, "files": files, "bytes": size}
        print(f"{name:20s} {copied:5d}/{files:<5d} {human(size)}")

    for name in GENERATED_FOLDERS:
        files, size = folder_size(destination / name)
        print(f"{name:20s} {'已生成' if files else '缺少':>5} {files:<5d} {human(size)}")

    retired = []
    for name in RETIRED_FOLDERS:
        files, size = folder_size(destination / name)
        if files:
            retired.append({"folder": name, "files": files, "bytes": size})
    if retired:
        print(f"\n⚠ 这个库里还留着 {len(retired)} 个已停用文件夹（本工具不删）：")
        for item in retired:
            print(f"   {item['folder']}  {item['files']} 个文件 {human(item['bytes'])}")
        print("   确认不需要后手工删除：")
        print(f"   Remove-Item -Recurse -Force '{destination}'\\<文件夹名>")

    if not check:
        index = destination / "_索引"
        index.mkdir(parents=True, exist_ok=True)
        (index / "subset.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "folders": list(SUBSET_FOLDERS),
                    "generated": list(GENERATED_FOLDERS),
                    "retired": [item["folder"] for item in retired],
                    "files": total_files,
                    "bytes": total_bytes,
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\n素材合计 {total_files} 个文件，{human(total_bytes)}")
    if missing:
        print(f"⚠ 源库缺少 {len(missing)} 个文件夹：{'、'.join(missing)}")
    return {
        "files": total_files,
        "bytes": total_bytes,
        "missing": missing,
        "retired": retired,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只核对，不复制",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.source.exists():
        raise SystemExit(f"源素材库不存在：{args.source}")
    if args.out.resolve() == args.source.resolve():
        raise SystemExit("输出目录不能是完整库，完整库只读")
    args.out.mkdir(parents=True, exist_ok=True)

    result = build(args.source, args.out, check=args.check)
    print(f"\n输出目录：{args.out}")
    return 0 if not result["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
