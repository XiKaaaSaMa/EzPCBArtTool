"""Check the card's text table against the template files it was measured from.

The card's wording is stored as anchors in :data:`pcbart.badge_card.BADGE_TEXTS`
and the numbers were typed in by hand after measuring the source projects. A
typo there moves a line on every card and is invisible until someone looks at a
finished board -- which is exactly what happened once, reported as "the preview
looks offset".

This re-reads both template files and compares every field: position, size,
rotation and layer. The three lines that carry ``{代号}`` / ``{编号}`` appear in
the source as placeholder words, so they are matched by position and rotation
rather than by their text.

Usage::

    python tools/verify_text_anchors.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcbart import badge_card  # noqa: E402
from pcbart.epru import EproArchive  # noqa: E402
from tools.extract_badge_regions import _outline_boxes  # noqa: E402

MIL_MM = 0.0254

#: Library-relative folder holding the two blank templates.
TEMPLATE_DIR = (
    ROOT.parent / "PCB艺术板工具素材库" / "通行证模板源文件"
)
SMALL_CARD = "模板-精0、精1、皮肤小.epro2"
LARGE_CARD = "模板-精2、皮肤大.epro2"

#: EasyEDA layer ids as the card's layer names.
LAYER_NAMES = {1: "顶层铜", 2: "底层铜", 3: "顶层丝印层", 4: "底层丝印层"}

#: How far a line may sit from where the table says, in millimetres.
POSITION_TOLERANCE_MM = 0.001

#: How far the two templates may differ before the choice of numbers matters.
#:
#: **The table follows the small card.** The two files were authored at
#: different times and place a handful of lines a fraction of a millimetre
#: apart -- measured at up to 0.162 mm (职业), which is far below anything a
#: printed card shows. A larger gap would mean the table is describing one file
#: and not the other, so it is reported rather than tolerated.
TEMPLATE_DRIFT_MM = 0.2

#: Lines the small card has and the large one does not.
#:
#: The 阵营logo guide exists only on the small card: on the tall one the logo
#: would print over the illustration. Its absence from that file is the design,
#: not a mismatch, so it is skipped rather than reported.
ONLY_ON_SMALL_CARD = frozenset({"阵营logo（仅小卡）"})


@dataclass(frozen=True)
class SourceText:
    """One ``STRING`` from a template, relative to its card's centre."""

    text: str
    layer: str
    x_mm: float
    y_mm: float
    size_mm: float
    rotation_deg: float
    font: str


def read_texts(path: Path) -> list[SourceText]:
    """Every visible ``STRING`` in ``path``, in card-centred millimetres."""
    document = EproArchive.read(path).document
    centre = _outline_boxes(document)[0].centre
    rows: list[SourceText] = []
    for record in document.iter_type("STRING"):
        body = record.body
        size_mm = float(body["fontSize"]) * MIL_MM
        # ARKNIGHTS-DWDB is stored at 0.003 mm: invisible, unclickable, and
        # deliberately not part of the editable table.
        if size_mm < 0.01:
            continue
        rows.append(
            SourceText(
                text=str(body["text"]),
                layer=LAYER_NAMES.get(int(body["layerId"]), str(body["layerId"])),
                x_mm=float(body["x"]) * MIL_MM - centre[0],
                y_mm=float(body["y"]) * MIL_MM - centre[1],
                size_mm=size_mm,
                rotation_deg=float(body.get("angle", 0.0) or 0.0),
                font=str(body.get("fontFamily", "")),
            )
        )
    return rows


def compare(
    source: list[SourceText], label: str, *, tolerance_mm: float
) -> list[str]:
    """Report every card line that does not match ``source``.

    ``tolerance_mm`` is how far a line may sit from the table. It is a hair on
    the small card, whose numbers the table *is*; on the large card it is the
    drift the two authored files are known to have, and anything beyond that
    means the table has stopped describing that file. Returns human-readable
    problems; empty means the table agrees.
    """
    problems: list[str] = []
    for item in badge_card.BADGE_TEXTS:
        hit = next(
            (
                row
                for row in source
                if row.layer == item.layer
                and abs(row.rotation_deg - item.rotation_deg) < 0.01
                and abs(row.x_mm - item.x_mm) <= TEMPLATE_DRIFT_MM
                and abs(row.y_mm - item.y_mm) <= TEMPLATE_DRIFT_MM
            ),
            None,
        )
        if hit is None:
            if item.label in ONLY_ON_SMALL_CARD:
                continue
            problems.append(f"{label}: 找不到「{item.label}」对应的源文字")
            continue
        dx = abs(item.x_mm - hit.x_mm)
        dy = abs(item.y_mm - hit.y_mm)
        if dx > tolerance_mm or dy > tolerance_mm:
            problems.append(
                f"{label}: 「{item.label}」位置差 ({dx:.3f}, {dy:.3f}) mm，"
                f"超过 {tolerance_mm} mm"
            )
        for field, mine, theirs in (
            ("字号", item.size_mm, hit.size_mm),
            ("角度", item.rotation_deg, hit.rotation_deg),
        ):
            if abs(mine - theirs) > POSITION_TOLERANCE_MM:
                problems.append(
                    f"{label}: 「{item.label}」{field} 不一致 "
                    f"表里 {mine:g} / 模板 {theirs:g}"
                )
        if item.font != hit.font:
            problems.append(
                f"{label}: 「{item.label}」字体不一致 "
                f"表里 {item.font!r} / 模板 {hit.font!r}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    directory = Path(argv[0]) if argv else TEMPLATE_DIR
    if not directory.is_dir():
        print(f"找不到模板目录：{directory}")
        return 2

    failed = 0
    for name, label, tolerance in (
        (SMALL_CARD, "小卡（表以此为准）", POSITION_TOLERANCE_MM),
        (LARGE_CARD, "大卡（容差 = 两卡已知漂移）", TEMPLATE_DRIFT_MM),
    ):
        path = directory / name
        if not path.exists():
            print(f"⚠ 缺少模板：{path}")
            failed += 1
            continue

        source = read_texts(path)
        problems = compare(source, label, tolerance_mm=tolerance)
        print(f"{label}：模板 {len(source)} 行 · 表里 {len(badge_card.BADGE_TEXTS)} 行")
        if problems:
            for line in problems:
                print(f"   ⚠ {line}")
            failed += len(problems)
        else:
            print(
                f"   ✓ 位置（±{tolerance:g} mm）/ 字号 / 角度 / 图层 / 字体一致"
            )
        print()

    # Where the two templates disagree, so the choice of numbers is on record
    # and a future re-export that moves a line far enough to matter is caught.
    small = read_texts(directory / SMALL_CARD)
    large = read_texts(directory / LARGE_CARD)
    print("两块模板之间的差异（代码表采用小卡那一套）：")
    for row in small:
        other = next(
            (
                item
                for item in large
                if item.text == row.text
                and item.layer == row.layer
                and abs(item.rotation_deg - row.rotation_deg) < 0.01
            ),
            None,
        )
        if other is None:
            print(f"   只有小卡有：{row.text!r}")
            continue
        dx, dy = other.x_mm - row.x_mm, other.y_mm - row.y_mm
        if max(abs(dx), abs(dy)) > 0.005:
            flag = ""
            if max(abs(dx), abs(dy)) > TEMPLATE_DRIFT_MM:
                flag = "   ⚠ 已经大到会影响成品"
                failed += 1
            print(f"   {row.text!r:<28} Δ=({dx:+.3f}, {dy:+.3f}){flag}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
