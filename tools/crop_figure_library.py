"""Crop the badge library's illustrations to the regions they will fill.

.. deprecated:: 2026-10-02
   **The badge no longer places illustrations at all.** The user dropped the
   立绘 sets deliberately, and the library it used to read
   (``明日方舟-通行证素材``) and write (``明日方舟-通行证素材-裁切``) are both
   retired. This module is kept as history, and for the case where the full-body
   art is needed again for some other purpose -- it must then be pointed at
   ``明日方舟`` with ``--source`` and at a fresh output folder with ``--out``.
   Do not run it against the retired library.

The library holds square illustrations at 1024-2560 px. The badge template
reserves two rectangles for artwork, and neither is square: the short one is
38.861 x 56.324 mm (aspect 0.690) and the tall one 42.526 x 99.626 mm (aspect
0.427). Keeping the squares costs several times the pixels that can ever be
printed, so each folder is cropped to the shape it will be placed in:

======================  =================  =============
library folder          region             aspect
======================  =================  =============
``06-立绘-精英0``         短区                0.690
``07-立绘-精英1``         短区                0.690
``08-立绘-精英2``         长区                0.427
``09-立绘-皮肤``           **both**           0.690 and 0.427
======================  =================  =============

Skins get both because a skin is not one or the other: the brief describes
small-region skins and large-region skins as different choices, and which one
a given outfit suits is the user's decision at placement time. Every other
folder has exactly one destination, and cropping it twice would make the
library *larger* -- the short crop is 69% of a square and the tall crop 43%, so
both together is 112%.

The source library is never modified. Output goes to a sibling folder with the
region in each name, so a wrong crop can be thrown away and regenerated without
re-downloading 4 GB.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pcbart import badge_template as bt  # noqa: E402
from pcbart.figure_crop import (  # noqa: E402
    WAIST_SHARE,
    crop as crop_image,
    detect_figure,
    plan_crop,
    region_aspect,
)
from pcbart.paths import asset_root  # noqa: E402

#: Both resolved under ``PCBART_ASSET_ROOT``; the library is gigabytes and lives
#: outside the repository, so neither end may be a literal path.
DEFAULT_SOURCE = asset_root() / "明日方舟-通行证素材"
DEFAULT_OUTPUT = asset_root() / "明日方舟-通行证素材-裁切"

SHORT = "窄"
TALL = "长"


@dataclass(frozen=True)
class Job:
    """One folder to crop, and which shapes it needs.

    ``upper_body`` cuts at the waist instead of the frame edge. 精英0 and 精英1
    are the two sets the user wants as upper body only; the request was
    specific about those, and the taller sets stay full-length.
    """

    folder: str
    shapes: tuple[str, ...]
    upper_body: bool = False


JOBS: tuple[Job, ...] = (
    Job("06-立绘-精英0", (SHORT,), upper_body=True),
    Job("07-立绘-精英1", (SHORT,), upper_body=True),
    Job("08-立绘-精英2", (TALL,)),
    Job("09-立绘-皮肤", (SHORT, TALL), upper_body=True),
)

SHAPE_ASPECT = {
    SHORT: region_aspect(bt.SMALL_REGION),
    TALL: region_aspect(bt.LARGE_REGION),
}

#: What goes into the output folder's name, so an upper-body set is never
#: mistaken for a full-length one.
SHAPE_SUFFIX = {SHORT: "窄", TALL: "长"}


def _human(count: int) -> str:
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= size:
            return f"{count / size:.1f} {unit}"
    return f"{count} B"


def crop_one(
    source: Path, destination: Path, aspect: float, *, upper_body: bool = False
) -> dict:
    """Crop one file, reporting whether the head survived."""
    with Image.open(source) as handle:
        image = handle.convert("RGBA")

    detection = detect_figure(np.asarray(image))
    if detection is None:
        return {"status": "skipped", "reason": "找不到人物"}

    plan = plan_crop(
        detection,
        aspect,
        image.size,
        waist_share=WAIST_SHARE if upper_body else None,
        asset=source.name,
    )
    # The one invariant with no recovery: the top of the picture has to stay
    # inside the window. Failing here means a crop that decapitates, which is
    # worse than no crop at all.
    #
    # Skipped when the window was placed by a face or by hand.
    #
    # This guard asks "is the top of the *silhouette* inside the window", which
    # is the right question only while the window is anchored to the silhouette.
    # It stopped being the right question when the user redefined the rule as
    # "the face must be whole; the device, clothes and arms need not be":
    #
    # * with a face, the silhouette's top is whatever sticks up highest -- a
    #   signboard, a raised weapon, a halo -- and 92 of 462 精英0 pictures were
    #   rejected on it (立绘 嘉辛塔 1.png "failed" because a signboard sits 409 px
    #   above a face that was perfectly centred);
    # * by hand, the anchors are known-wrong, which is why the entry exists --
    #   立绘 Miss.Christine 1.png has the gold frame 70% of the canvas above the
    #   cat a person asked to centre.
    #
    # The face branch has its own guarantee instead: the window is a fixed
    # multiple of the *face* and is placed so the face is whole. See
    # :func:`pcbart.figure_crop.face_window`.
    if plan.method not in ("manual", "face"):
        topmost = detection.box.y0
        if detection.saliency is not None:
            topmost = min(topmost, detection.saliency[2])
        if plan.top > topmost:
            return {
                "status": "skipped",
                "reason": f"头部会被裁掉（top={plan.top} > {topmost}）",
            }

    result = crop_image(image, plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, "PNG", optimize=True)
    return {
        "status": "ok",
        "size": (plan.width, plan.height),
        "method": plan.method,
    }


def run(
    source_root: Path, output_root: Path, jobs: Sequence[Job], *, limit: int = 0
) -> int:
    before_total = 0
    after_total = 0
    counts = {"ok": 0, "skipped": 0}
    skipped: list[str] = []

    for job in jobs:
        source_dir = source_root / job.folder
        if not source_dir.is_dir():
            print(f"[{job.folder}] 源文件夹不存在，跳过")
            continue
        files = sorted(source_dir.glob("*.png"))
        if limit:
            files = files[:limit]
        for file in files:
            for shape in job.shapes:
                folder = job.folder if len(job.shapes) == 1 else f"{job.folder}-{shape}"
                if job.upper_body:
                    # Named, not just placed in another directory: an upper-body
                    # set and a full-length one have the same filename, and a
                    # folder that says which it is cannot be mixed up later.
                    folder = "上半身-" + folder
                destination = output_root / folder / file.name
                before_total += file.stat().st_size
                if destination.exists() and destination.stat().st_size > 0:
                    after_total += destination.stat().st_size
                    counts["ok"] += 1
                    continue
                outcome = crop_one(
                    file,
                    destination,
                    SHAPE_ASPECT[shape],
                    upper_body=job.upper_body,
                )
                if outcome["status"] == "ok":
                    after_total += destination.stat().st_size
                    counts["ok"] += 1
                else:
                    counts["skipped"] += 1
                    skipped.append(f"{file.name}：{outcome['reason']}")
                    # Fall back to copying the original, so the output library is
                    # complete. A missing picture is a hole the user finds later;
                    # an uncropped one is merely bigger.
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(file.read_bytes())
                    after_total += destination.stat().st_size
        print(f"[{job.folder}] 完成（{len(files)} 个文件 × {len(job.shapes)} 种形状）", flush=True)

    print()
    print(f"成功 {counts['ok']}，回退为原图 {counts['skipped']}")
    for line in skipped[:20]:
        print(f"  ! {line}")
    if len(skipped) > 20:
        print(f"  …还有 {len(skipped) - 20} 条")
    print()
    print(f"源 {_human(before_total)}  →  裁切后 {_human(after_total)}"
          f"（{after_total / max(1, before_total) * 100:.0f}%）")
    print(f"输出目录：{output_root}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="只处理某个文件夹（可重复）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个文件夹只处理前 N 个（试跑用）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    jobs = [job for job in JOBS if not args.only or job.folder in args.only]
    if not jobs:
        raise SystemExit(f"没有匹配的文件夹；可选：{[j.folder for j in JOBS]}")
    if not args.source.is_dir():
        raise SystemExit(f"源素材库不存在：{args.source}")
    return run(args.source, args.out, jobs, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
