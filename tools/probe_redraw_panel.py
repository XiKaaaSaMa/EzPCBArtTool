"""Drive the redraw panel over several 精0 illustrations and record what it shows.

The panel's two sliders are only worth having if they behave the same way
across the library, so this walks a set of illustrations chosen for coverage of
the colour range and saves the panel's own screenshots plus a sweep of the two
knobs. The screenshots come from the real panel -- the sliders are moved and the
window is grabbed -- rather than from a re-implementation, so what is recorded
is what the panel actually draws.

精英2 is deliberately not covered: those illustrations carry a painted
background, and the redraw reading treats background as subject matter. That is
a different problem from the one this panel is for.

``PCBART_DRAFTS_DIR`` is redirected before any window exists, because closing a
driven window writes a draft and that must not land in the user's library.

Usage::

    python tools/probe_redraw_panel.py
    python tools/probe_redraw_panel.py --operators 傀影 闪击 清流
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

#: Operators covering the measured colour range: the darkest illustration in
#: the library, a nearly greyscale one, a mid-tone, the brightest, the most
#: polychrome, and a mechanical subject.
DEFAULT_OPERATORS = ("傀影", "角峰", "闪击", "清流", "断罪者", "Lancet-2")


def _prepare_environment(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PCBART_DRAFTS_DIR"] = str(scratch)


def _shoot(window, path: Path) -> None:
    window.grab().save(str(path))


def _settle(app, milliseconds: int = 260) -> None:
    """Spin the real event loop until the panel has finished reacting.

    ⚠️ **``processEvents()`` alone is not enough, and the first run of this
    probe proved it.** The panel debounces slider changes by 90 ms, and
    ``processEvents`` does not advance the clock -- so every sweep screenshot
    after the first came back showing the *previous* setting: the slider had
    moved, the readout still said 5.0%, and the picture was unchanged. Waiting
    on the wall clock is the only way to photograph what the slider asked for.
    """
    import time

    deadline = time.monotonic() + milliseconds / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="*", default=list(DEFAULT_OPERATORS))
    parser.add_argument(
        "--outdir", default=str(ROOT / "out" / "_redraw_panel_elite0")
    )
    parser.add_argument("--quantile", type=float, default=5.0)
    parser.add_argument("--thicken-mm", type=float, default=0.08)
    parser.add_argument(
        "--sweep", action="store_true",
        help="额外对每个样本扫一遍两个滑块的组合",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    _prepare_environment(outdir / "_scratch")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from pcbart.ui_theme import scheme_from_settings
    from tools.line_redraw_panel import ELITE0, LineRedrawPanel
    from tools.qt_theme import apply_scheme, style_window

    app = QApplication.instance() or QApplication(sys.argv[:1])
    scheme = scheme_from_settings()
    apply_scheme(app, scheme)

    window = LineRedrawPanel()
    style_window(window, scheme)
    window.apply_local_style(scheme)
    window.resize(1320, 780)
    window.show()

    failures: list[str] = []
    report = [
        "# 线稿重绘面板 · 精0 实测",
        "",
        f"- 面板：`tools/line_redraw_panel.py`",
        f"- 默认参数：密度 {args.quantile:.2f}% · 加粗 {args.thicken_mm:.2f} mm",
        f"- 输出：`{outdir}`",
        "",
    ]

    for name in args.operators:
        source = ELITE0 / f"立绘 {name} 1.png"
        if not source.exists():
            print(f"  跳过（不存在）：{source}")
            failures.append(name)
            continue

        index = window.picker.findData(str(source))
        if index < 0:
            print(f"  跳过（不在列表里）：{name}")
            failures.append(name)
            continue
        window.picker.setCurrentIndex(index)
        window.quantile_slider.setValue(int(args.quantile * 100))
        window.thicken_slider.setValue(int(args.thicken_mm * 100))

        # The model load, the 90 ms debounce and the repaint all have to run
        # before a grab means anything; see :func:`_settle`.
        _settle(app, 600)

        _shoot(window, outdir / f"{name}__panel-default.png")
        readout = window.readout.text().replace("\n", "　")
        print(f"  {name}: {readout}")
        report.append(f"## {name}")
        report.append("")
        report.append(f"源文件：`{source}`")
        report.append("")
        report.append(f"面板读数：{readout}")
        report.append("")

        if args.sweep:
            for quantile in (3.0, 5.0, 8.0, 12.0):
                window.quantile_slider.setValue(int(quantile * 100))
                _settle(app)
                _shoot(
                    window,
                    outdir / f"{name}__density{quantile:.1f}.png",
                )
            window.quantile_slider.setValue(int(args.quantile * 100))
            _settle(app)
            for thicken_mm in (0.0, 0.15, 0.25):
                window.thicken_slider.setValue(int(thicken_mm * 100))
                _settle(app)
                _shoot(
                    window,
                    outdir / f"{name}__thicken{thicken_mm:.2f}.png",
                )
            window.thicken_slider.setValue(int(args.thicken_mm * 100))
            _settle(app)

    (outdir / "_report.md").write_text("\n".join(report), encoding="utf-8")
    window.hide()
    window.deleteLater()
    app.processEvents()

    print(f"\n输出目录：{outdir}")
    if failures:
        print(f"未处理：{'、'.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
