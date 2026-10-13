"""Two knobs for the redrawn line art, on their own panel.

The redraw pipeline settled on two settings, and they do different jobs:

* **线条密度** -- the response quantile that becomes ink. It decides *which*
  strokes are drawn. Lower keeps only the strongest lines; higher picks up
  weaker ones until the picture fills in.
* **线条加粗** -- how far every chosen stroke grows, in **millimetres on the
  finished board**. It decides whether those strokes are printable. A hairline
  that reads fine on screen can be too thin for silkscreen, and this is the
  knob that fixes that.

They are independent on purpose: the quantile cannot make a thin line thicker,
and thickening cannot bring back a line the quantile rejected. The panel shows
both so the trade can be seen rather than guessed at.

The model runs once per picture and its soft response is cached, so moving a
slider only re-thresholds and re-dilates -- neither costs more than a few tens
of milliseconds, which is what makes the sliders usable at all.

This is a **probe panel, not part of the application yet**. Wiring it into the
product means shipping a 391 MB model and a PyTorch dependency, which is the
user's decision to make.

Usage::

    python tools/line_redraw_panel.py
    python tools/line_redraw_panel.py --image "…\\立绘 清流 1.png"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from pcbart.paths import asset_root  # noqa: E402

from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402
from pcbart.model_paths import DEFAULT_VARIANT as VARIANT  # noqa: E402
from pcbart.model_paths import check_model_dir, resolve_model_dir  # noqa: E402
from pcbart.work_scale import work_scale  # noqa: E402
from tools.probe_line_redraw import (  # noqa: E402
    ADAPTIVE_FLOOR,
    adaptive_level,
    clean_alpha_border,
    load_model,
    model_response,
)
from tools.probe_line_thicken import thicken  # noqa: E402

#: Operators the panel offers by default, as the 精0 library the redraw reading
#: was measured on. 精2 is left out deliberately: those illustrations carry a
#: painted background, and the model treats the background as subject matter,
#: which is a different problem from the one this panel is for.
LIBRARY = asset_root() / "明日方舟"
ELITE0 = LIBRARY / "06-立绘-精英0"

#: Long side the drawing is placed at, for millimetres <-> pixels. Same default
#: the rest of the project uses for an imported picture.
DRAWN_MM = 76.2

#: Working resolution bound, matching :data:`pcbart.work_scale.MAX_WORK_LONG_SIDE`.
MAX_WORK_SIDE = 2000

#: Slider ranges, in the sliders' own integer units (hundredths).
QUANTILE_MIN, QUANTILE_MAX = 1.0, 25.0
THICKEN_MIN, THICKEN_MAX = 0.0, 0.40

#: Defaults from the measured sweep: the combination that came back clean on
#: every one of the twelve library illustrations it was tried on.
QUANTILE_DEFAULT = 5.0
THICKEN_DEFAULT = 0.08


class NoWheelSlider(QSlider):
    """A slider that refuses wheel input.

    The project's rule for every editable control: a slider under the pointer
    must not change value while the user is only trying to scroll. Ignoring the
    event hands it back to Qt, so whatever is behind the panel still scrolls.
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.ignore()


class _ScaledLabel(QLabel):
    """A label that keeps its picture fitted to its own current size.

    ⚠️ **Rescaling from the parent's ``resizeEvent`` is not enough.** The parent
    is laid out before its children are, so at that moment the child still has
    its old size and the picture is scaled to a stale box -- measured as a
    square 1024 px illustration drawn about a third of its available width,
    with nothing afterwards to correct it. The child's own resize event is the
    only point where its size is final.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None

    def set_source(self, image: np.ndarray | None) -> None:
        self._pixmap = None if image is None else _to_pixmap(image)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            self.clear()
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()


class PreviewPane(QWidget):
    """One labelled picture that scales to fit its box."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setObjectName("previewTitle")
        self._canvas = _ScaledLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setMinimumSize(280, 280)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setObjectName("previewCanvas")
        layout.addWidget(self._title)
        layout.addWidget(self._canvas, 1)

    def set_image(self, image: np.ndarray | None) -> None:
        """Show an RGB/BGR array, grey array, or nothing."""
        self._canvas.set_source(image)


def _to_pixmap(image: np.ndarray) -> QPixmap:
    """A numpy array as a QPixmap, copying so Qt owns its own buffer."""
    array = np.ascontiguousarray(image)
    if array.ndim == 2:
        height, width = array.shape
        qimage = QImage(
            array.data, width, height, width, QImage.Format_Grayscale8
        )
    else:
        height, width, channels = array.shape
        if channels == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGBA)
        qimage = QImage(
            array.data, width, height, width * 4, QImage.Format_RGBA8888
        )
    return QPixmap.fromImage(qimage.copy())


class LineRedrawPanel(QMainWindow):
    """The panel itself: pick a picture, move two sliders, watch the drawing."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("线稿重绘参数（探针）")
        self.resize(1280, 780)

        self._net = None
        self._device = "cpu"
        self._response: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self._stem = "output"
        self._px_per_mm = 1.0

        # Debounce, so dragging a slider does not recompute on every pixel of
        # travel -- the same 90 ms pattern the preprocessing panel uses.
        self._pending = QTimer(self)
        self._pending.setSingleShot(True)
        self._pending.setInterval(90)
        self._pending.timeout.connect(self._render)

        self._build()
        self._populate_library()

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        outer.addLayout(self._build_toolbar())

        body = QHBoxLayout()
        body.setSpacing(10)
        self.original_pane = PreviewPane("原图")
        self.result_pane = PreviewPane("重绘线稿")
        body.addWidget(self.original_pane, 1)
        body.addWidget(self.result_pane, 1)
        body.addWidget(self._build_parameter_column(), 0)
        outer.addLayout(body, 1)

        self.status = QLabel("选择一张图片开始")
        self.status.setObjectName("statusLine")
        outer.addWidget(self.status)

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("素材"))
        self.picker = QComboBox()
        self.picker.setMinimumWidth(260)
        self.picker.currentIndexChanged.connect(self._on_pick)
        row.addWidget(self.picker, 1)
        open_button = QPushButton("打开图片…")
        open_button.clicked.connect(self._on_open)
        row.addWidget(open_button)
        self.export_button = QPushButton("导出当前结果")
        self.export_button.clicked.connect(self._on_export)
        row.addWidget(self.export_button)
        return row

    def _build_parameter_column(self) -> QWidget:
        """The two knobs, in a column of their own."""
        column = QFrame()
        column.setObjectName("parameterColumn")
        column.setFixedWidth(300)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        heading = QLabel("参数")
        heading.setObjectName("sectionHeading")
        layout.addWidget(heading)

        self.quantile_slider, self.quantile_value = self._slider_row(
            layout,
            "线条密度",
            QUANTILE_MIN,
            QUANTILE_MAX,
            QUANTILE_DEFAULT,
            "取响应最暗的百分之几作为线；调低只留主线，调高描出弱线、过高带进背景。",
        )
        self.thicken_slider, self.thicken_value = self._slider_row(
            layout,
            "线条加粗",
            THICKEN_MIN,
            THICKEN_MAX,
            THICKEN_DEFAULT,
            "每条线向两侧各长多少毫米；这决定线能否印出来，调太大相邻线粘连。",
        )

        layout.addWidget(_divider())
        self.readout = QLabel("—")
        self.readout.setObjectName("readout")
        # Wrapped text has no reliable height-for-width hint inside a column,
        # so without a floor the label collapses to a sliver and the four lines
        # draw on top of each other.
        self.readout.setMinimumHeight(80)
        self.readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.readout)
        layout.addStretch(1)
        return column

    def _slider_row(self, layout, title, low, high, default, tip):
        row = QVBoxLayout()
        row.setSpacing(2)
        header = QHBoxLayout()
        label = QLabel(title)
        label.setToolTip(tip)
        value = QLabel(_format_value(title, default))
        value.setObjectName("sliderValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(label)
        header.addWidget(value, 1)
        row.addLayout(header)

        slider = NoWheelSlider(Qt.Horizontal)
        slider.setRange(int(low * 100), int(high * 100))
        slider.setValue(int(default * 100))
        slider.setToolTip(tip)
        slider.valueChanged.connect(self._on_tuning_changed)
        row.addWidget(slider)
        layout.addLayout(row)
        return slider, value

    def apply_local_style(self, scheme) -> None:
        """Colour the parts the shared sheet does not know about.

        Set on the individual widgets rather than on the window: a sheet on the
        window would replace the one ``style_window`` installed and take the
        whole theme with it, whereas a child's sheet composes with its
        ancestors'.
        """
        panel = scheme.get("panel", "#f2f2f5")
        alt = scheme.get("panel_alt", "#e9e9ee")
        text = scheme.get("text", "#1b1b1f")
        muted = scheme.get("muted", "#5a5a66")
        border = scheme.get("border", "#c9c9d2")

        self.findChild(QFrame, "parameterColumn").setStyleSheet(
            f"QFrame#parameterColumn {{ background: {panel};"
            f" border: 1px solid {border}; border-radius: 8px; }}"
        )
        for name, colour, weight in (
            ("sectionHeading", text, "600"),
            ("previewTitle", muted, "500"),
            ("sliderValue", text, "600"),
            ("statusLine", muted, "400"),
        ):
            for widget in self.findChildren(QLabel, name):
                widget.setStyleSheet(f"color: {colour}; font-weight: {weight};")
        for canvas in self.findChildren(QLabel, "previewCanvas"):
            canvas.setStyleSheet(
                f"QLabel#previewCanvas {{ background: {alt};"
                f" border: 1px solid {border}; border-radius: 6px; }}"
            )
        self.readout.setStyleSheet(f"color: {muted};")
        for line in self.findChildren(QFrame, "divider"):
            line.setStyleSheet(f"color: {border};")

    # -- data ------------------------------------------------------------

    def _populate_library(self) -> None:
        """Fill the picker and queue one load.

        The index is set with signals blocked: leaving them on would fire
        ``_on_pick`` from inside ``setCurrentIndex`` *and* again from the timer
        below, running the model twice for the first picture.
        """
        self.picker.blockSignals(True)
        self.picker.clear()
        if ELITE0.is_dir():
            for path in sorted(ELITE0.glob("立绘 *.png")):
                self.picker.addItem(path.stem.replace("立绘 ", ""), str(path))
        if self.picker.count():
            self.picker.setCurrentIndex(0)
        self.picker.blockSignals(False)
        # Queued rather than called: the slider values from the command line are
        # applied after construction, and the first render should use them.
        QTimer.singleShot(0, self._on_pick)

    def _load(self, path: str) -> None:
        raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            QMessageBox.warning(self, "读取失败", f"无法读取：{path}")
            return
        if raw.ndim == 2:
            colour = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            alpha = None
        elif raw.shape[2] == 4:
            colour, alpha = raw[:, :, :3].copy(), raw[:, :, 3].copy()
        elif raw.shape[2] == 2:
            colour = cv2.cvtColor(raw[:, :, 0], cv2.COLOR_GRAY2BGR)
            alpha = raw[:, :, 1].copy()
        else:
            colour, alpha = raw.copy(), None

        height, width = colour.shape[:2]
        scale = work_scale(width, height)
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            colour = cv2.resize(colour, size, interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)

        flat, _removed = clean_alpha_border(colour, alpha)
        self._source = flat
        self._stem = Path(path).stem
        self._px_per_mm = max(flat.shape[:2]) / float(DRAWN_MM)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self._net is None:
                self._net, self._device = self._ensure_model()
            if self._net is None:
                # ``_ensure_model`` has already said where to put the files.
                return
            self._response = model_response(self._net, self._device, flat, 1024)
        finally:
            QApplication.restoreOverrideCursor()

        self.original_pane.set_image(flat)
        self._render()

    def _ensure_model(self):
        """Load the external model, or explain how to install it.

        The model is not part of the installer, so "it is not here" is an
        ordinary state rather than a crash. The message is deliberately made
        of the things the user needs to act: **which** folder is in force,
        what is missing inside it, and where to change the choice.

        ⚠️ It names the folder that is *in force*, not a list of every place
        searched. The program only ever loads one, and naming others would
        invite the user to fix a folder it was never going to read.
        """
        directory = resolve_model_dir()
        report = check_model_dir(directory, VARIANT)
        if not report["ok"]:
            QMessageBox.information(
                self,
                "没有找到线稿重绘模型",
                "线稿重绘用 Anime2Sketch 模型，安装包不含。\n\n"
                f"当前使用的文件夹：\n{directory}\n\n"
                f"缺少：{'；'.join(report['missing'])}\n\n"
                "获取：python tools\\fetch_anime2sketch.py\n"
                "或将已下载好的模型文件夹放进上面那个位置，"
                "再到【视图 → 界面设置… → 线稿重绘模型】里选择\n"
                "（也可用环境变量 PCBART_A2S_MODEL 指定，它优先）。",
            )
            self.status.setText("未安装线稿重绘模型（其余功能不受影响）")
            return None, "cpu"
        return load_model(directory, "auto", VARIANT)

    def _parameters(self) -> tuple[float, float]:
        return (
            self.quantile_slider.value() / 100.0,
            self.thicken_slider.value() / 100.0,
        )

    def _compute(self):
        """The mask for the current settings, plus what it cost."""
        if self._response is None:
            return None, None, 0
        quantile, thicken_mm = self._parameters()
        level = adaptive_level(self._response, quantile)
        base = self._response <= level
        mask = thicken(base, thicken_mm * self._px_per_mm)
        return mask, level, thicken_mm * self._px_per_mm

    # -- reactions -------------------------------------------------------

    def _on_pick(self, _index: int = -1) -> None:
        """Load whatever the picker currently points at.

        The index argument exists because the signal supplies one; it is
        ignored in favour of reading the combo, so the timer that triggers the
        first load can call this with no argument.
        """
        path = self.picker.currentData()
        if path:
            self._load(path)

    def _on_open(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "选择图片", str(ELITE0 if ELITE0.is_dir() else ROOT),
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self._load(path)

    def _on_tuning_changed(self, _value: int) -> None:
        quantile, thicken_mm = self._parameters()
        self.quantile_value.setText(_format_value("线条密度", quantile))
        self.thicken_value.setText(_format_value("线条加粗", thicken_mm))
        self._pending.start()

    def _render(self) -> None:
        mask, level, radius_px = self._compute()
        if mask is None:
            return
        self.result_pane.set_image(np.where(mask, 0, 255).astype(np.uint8))
        coverage = 100.0 * float(mask.mean())
        components, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        quantile, thicken_mm = self._parameters()
        plateau_note = ""
        if level >= 250:
            plateau_note = "\n⚠️ 阈值贴近纸面，结果可能成片"
        self.readout.setText(
            f"阈值 {level:.0f}（分位 {quantile:.1f}%，下限 {ADAPTIVE_FLOOR:.0f}）\n"
            f"加粗 {thicken_mm:.2f} mm ≈ {radius_px:.1f} px\n"
            f"墨量 {coverage:.2f}%　色块 {components - 1} 个"
            f"{plateau_note}"
        )

    def _on_export(self) -> None:
        mask, _level, _radius = self._compute()
        if mask is None:
            return
        default = str(ROOT / "out" / f"{self._stem}__线稿.png")
        path, _filter = QFileDialog.getSaveFileName(
            self, "导出线稿", default, "PNG (*.png)"
        )
        if not path:
            return
        imwrite_unicode(path, np.where(mask, 0, 255).astype(np.uint8))
        self.status.setText(f"已导出：{path}")


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("divider")
    return line


def _format_value(title: str, value: float) -> str:
    """``value`` with the unit its slider measures in.

    One place, so the label reads the same whether it was set at construction
    or by moving the slider -- a label that gains its unit only after the first
    drag looks like it is reporting something different.
    """
    return f"{value:.2f}%" if "密度" in title else f"{value:.2f} mm"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="启动时载入的图片")
    parser.add_argument(
        "--quantile", type=float, default=QUANTILE_DEFAULT,
        help="启动时的线条密度（响应分位，%%）",
    )
    parser.add_argument(
        "--thicken-mm", type=float, default=THICKEN_DEFAULT,
        help="启动时的加粗量，毫米",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    scheme = None
    try:
        from pcbart.ui_theme import scheme_from_settings
        from tools.qt_theme import apply_scheme, style_window

        scheme = scheme_from_settings()
        apply_scheme(app, scheme)
    except Exception:  # noqa: BLE001 - the panel is usable without the theme
        scheme = None

    window = LineRedrawPanel()
    if scheme is not None:
        style_window(window, scheme)
        window.apply_local_style(scheme)
    window.quantile_slider.setValue(int(args.quantile * 100))
    window.thicken_slider.setValue(int(args.thicken_mm * 100))
    window.show()

    if args.image:
        path = Path(args.image)
        index = window.picker.findData(str(path))
        if index >= 0:
            window.picker.setCurrentIndex(index)
        else:
            window._load(str(path))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
