# -*- coding: utf-8 -*-
"""Screenshot + mask probe for the readings and pre-steps added in this round.

Why a script rather than clicking through: the panel's layout is what has to be
reviewed (does the new 6 · 图像预处理 section fit, do the two new readings'
slider rows appear without shoving anything off-screen), and the masks are what
has to be *looked at* -- a reading that "runs" but draws a black rectangle would
pass every assertion about shape and dtype.

Writes into ``out/<date>test/_new_readings/``:
  * ``panel_<reading>.png``  -- the preprocessing panel with that reading live
  * ``panel_presteps.png``   -- 超分/滤镜 rows, including the Java status
  * ``mask_<reading>.png``   -- the conversion of the sample picture
  * ``timings.txt``          -- measured cost per reading, at the panel's own
                                preview size and at the full working size

Run with ``PCBART_DRAFTS_DIR`` pointed somewhere disposable: the panel builds a
real window, and a script that forgets this writes into the developer's draft
library (see ``AGENTS.md``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from PySide6.QtCore import Qt

from gui_mockup import READING_OPTIONS, build_window, show_preprocess
from pcbart.image_prep import imread_unicode, imwrite_unicode
from pcbart.lineart import LineArtParams, line_art, line_art_preview
from pcbart.paths import asset_root

OUT = REPO / "out" / "2026.10.13test" / "_new_readings"

READINGS = ("fill", "outline", "dodge", "xdog", "ridge")


def sample() -> Path:
    """A real illustration from the library, the dark-ish 惊蛰 card art."""
    return asset_root() / "明日方舟" / "06-立绘-精英0" / "立绘 惊蛰 1.png"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    picture = sample()

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    # The application comes first: build_window() makes widgets, and Qt refuses
    # to construct one before an application exists.
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1500, 980)
    window.show()
    panel = show_preprocess(window)
    panel.parent_window = window
    panel.place_top_right()
    if not panel.load_image(str(picture)):
        raise SystemExit("样例图载入失败")
    app.processEvents()

    lines: list[str] = [f"样例：{picture.name}"]
    order = [key for key, _label, _hint in READING_OPTIONS]

    for key in READINGS:
        panel.reading_combo.setCurrentIndex(order.index(key))
        app.processEvents()
        panel._refresh_preview()
        app.processEvents()
        # Two grabs: the panel as the user sees it, and the scroll area's
        # *whole content* -- the interesting sections (5 线稿调试 with this
        # reading's sliders, 6 图像预处理, 7 线稿修整) sit below the fold in a
        # viewport-only grab, which is exactly what has to be reviewed.
        panel.grab().save(str(OUT / f"panel_{key}.png"))
        content = panel._scroll.widget()
        if content is not None:
            content.grab().save(str(OUT / f"content_{key}.png"))
        processed = panel._inline_processed
        if processed is None:
            lines.append(f"{key}: 没有预览输出")
            continue
        grey = processed if processed.ndim == 2 else processed[..., 0]
        ink = float((grey < 128).mean() * 100) if grey.dtype != bool else 0.0
        lines.append(f"{key}: 预览 {grey.shape[1]}x{grey.shape[0]}  墨量 {ink:.1f}%")
        imwrite_unicode(str(OUT / f"mask_{key}.png"), processed)

    # The pre-step rows with their live status (Java found or not).
    panel.place_top_right()
    app.processEvents()
    panel.grab().save(str(OUT / f"panel_presteps.png"))
    content = panel._scroll.widget()
    if content is not None:
        content.grab().save(str(OUT / "content_presteps.png"))

    # And the same section on a machine *without* Java: the picker is disabled
    # and the reason is written out, which is the state half the users will
    # see -- worth one screenshot of its own.
    import os

    os.environ["PCBART_JAVA_TOOL_DIR"] = str(OUT / "no-such-tool")
    panel._refresh_filter_availability()
    app.processEvents()
    if content is not None:
        content.grab().save(str(OUT / "content_java_missing.png"))
    os.environ.pop("PCBART_JAVA_TOOL_DIR", None)
    panel._refresh_filter_availability()

    # Cost measurements at the sizes the panel actually uses: the live preview
    # (bounded trace) and the settled pass (work budget).
    raw = imread_unicode(str(picture), 1)
    if raw is not None and raw.ndim == 3 and raw.shape[2] == 3:
        import cv2

        work = cv2.resize(raw, (2000, 2000), interpolation=cv2.INTER_AREA)
    else:
        work = np.full((2000, 2000, 3), 220, dtype=np.uint8)
    lines.append("")
    lines.append("耗时（2000px 工作分辨率 / 实时预览路径）")
    for key in READINGS:
        params = LineArtParams(mode=key, silhouette=False)
        start = time.perf_counter()
        line_art(work, None, params)
        full_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        line_art_preview(work, None, params, max_side=760)
        preview_ms = (time.perf_counter() - start) * 1000.0
        lines.append(f"  {key}: 全分辨率 {full_ms:.0f} ms　预览 {preview_ms:.0f} ms")

    (OUT / "timings.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("输出目录:", OUT)
    window.hide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
