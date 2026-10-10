"""The preprocessing panel's entry point for the external redraw model.

The model is not shipped, so the only thing that tells a user the capability
exists is this section. Two failures matter more than the rest:

* **it must not promise work it cannot do.** There is no in-app redraw
  workspace yet, so the button explains and points at the settings rather than
  offering a "开始重绘" action that would do nothing. A control that looks like
  it runs the feature is the placeholder problem this project keeps hitting.
* **it must say which of the two requirements is missing.** "Model absent" and
  "PyTorch absent" need different fixes (download files vs. install a package),
  and a single "不可用" would leave the user guessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    BOARD_ART_STYLE,
    COLOR_STYLE,
    build_window,
    show_preprocess,
    show_settings,
)
from pcbart import model_paths  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def panel(app):
    """A preprocessing panel in a real (but never shown) window."""
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    widget = show_preprocess(window)
    widget.parent_window = window
    app.processEvents()
    yield widget
    from conftest import release_window

    release_window(window)


def complete_model(tmp_path: Path) -> Path:
    """A folder the checker accepts, without any real weights behind it."""
    (tmp_path / "weights").mkdir(parents=True, exist_ok=True)
    (tmp_path / model_paths.ARCHITECTURE_FILE).write_text("# arch\n", encoding="utf-8")
    (tmp_path / model_paths.WEIGHT_FILES[model_paths.DEFAULT_VARIANT]).write_bytes(b"\x80")
    return tmp_path


def style_index(name: str) -> int:
    from gui_mockup import STYLE_OPTIONS

    return [option for option, _hint in STYLE_OPTIONS].index(name)


# ------------------------------------------------------------- visibility


def test_the_entry_point_shows_for_the_line_art_style(panel):
    """It is the third way to read a picture as line art, so it lives with the
    other two -- but it must be reachable without knowing that."""
    panel.style_combo.setCurrentIndex(style_index("线稿"))
    panel.process_check.setChecked(True)
    panel._refresh_layers()

    assert panel.redraw_section.isVisible()


def test_the_entry_point_hides_for_the_painting_style(panel):
    """板画 resolves the picture to two tones; a redraw reading does not apply."""
    panel.style_combo.setCurrentIndex(style_index(BOARD_ART_STYLE))
    panel._refresh_layers()

    assert not panel.redraw_section.isVisible()


def test_the_entry_point_hides_for_colour_silkscreen(panel):
    """彩色丝印 processes nothing at all, by definition."""
    panel.style_combo.setCurrentIndex(style_index(COLOR_STYLE))
    panel._refresh_layers()

    assert not panel.redraw_section.isVisible()


def test_the_entry_point_hides_when_processing_is_off(panel):
    """处理图像 off means "put the file on the board as it is"."""
    panel.process_check.setChecked(False)
    panel._refresh_layers()

    assert not panel.redraw_section.isVisible()


# ----------------------------------------------------------------- status


def test_status_names_the_missing_architecture(panel, monkeypatch, tmp_path):
    empty = tmp_path / "empty-model"
    empty.mkdir()
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: empty)

    tone_name, sentence = panel._redraw_status()

    assert tone_name == "warning"
    assert model_paths.ARCHITECTURE_FILE in sentence, "要说清缺的是哪个文件"


def test_status_names_the_oracle_when_only_the_original_weights_are_present(
    panel, monkeypatch, tmp_path
):
    """⚠️ The specific confusion this project already hit.

    A user follows the author's main download link, gets ``netG.pth``, and the
    feature still refuses to run. The message has to name ``improved`` rather
    than saying "weights missing", or the download looks complete and pointless.
    """
    (tmp_path / "weights").mkdir(parents=True)
    (tmp_path / model_paths.ARCHITECTURE_FILE).write_text("# arch\n", encoding="utf-8")
    (tmp_path / model_paths.WEIGHT_FILES["default"]).write_bytes(b"\x80")
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: tmp_path)

    _tone, sentence = panel._redraw_status()

    assert model_paths.DEFAULT_VARIANT in sentence
    assert "netG.pth" not in sentence or "improved" in sentence


def test_status_reports_a_missing_torch_runtime(panel, monkeypatch, tmp_path):
    """The model can be complete and still unusable: PyTorch is optional.

    Without this branch the panel would show a green "已就绪" on a machine
    where opening the model raises ImportError -- the interface claiming
    readiness for something it cannot do.
    """
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: complete_model(tmp_path))
    monkeypatch.setattr(model_paths, "torch_available", lambda: False)

    tone_name, sentence = panel._redraw_status()

    assert tone_name == "warning"
    assert "torch" in sentence.lower()


def test_status_is_ready_when_both_are_present(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: complete_model(tmp_path))
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)

    tone_name, sentence = panel._redraw_status()

    assert tone_name == "success"
    assert model_paths.DEFAULT_VARIANT in sentence


def test_the_status_is_refreshed_when_the_panel_is_repainted(panel, monkeypatch, tmp_path):
    """A model installed *after* the panel was built must not leave a stale
    "尚未安装" on screen -- the user cannot otherwise make it re-check."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: tmp_path / "gone")
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    panel.style_combo.setCurrentIndex(style_index("线稿"))
    panel._refresh_layers()
    assert "尚未安装" in panel.redraw_status.text()

    # The user installs the model and switches style and back.
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: complete_model(tmp_path))
    panel.style_combo.setCurrentIndex(style_index(BOARD_ART_STYLE))
    panel._refresh_layers()
    panel.style_combo.setCurrentIndex(style_index("线稿"))
    panel._refresh_layers()

    assert "尚未安装" not in panel.redraw_status.text()


# -------------------------------------------------------------- the dialog


class CapturedDialog:
    """Runs ``_on_external_redraw`` with the modal box recorded, never shown."""

    def __init__(self, monkeypatch):
        self.text = ""
        self.labels: list[str] = []
        self.clicked = None
        box_ref = {}

        original_init = QMessageBox.__init__
        original_add = QMessageBox.addButton

        def fake_init(box, *args, **kwargs):
            original_init(box)
            box_ref["box"] = box

        def fake_add(box, text, role=None):
            self.labels.append(text)
            if self.clicked == text:
                box_ref["clicked"] = True
            button = original_add(box, text) if role is None else original_add(box, text, role)
            if self.clicked == text:
                box_ref["button"] = button
            return button

        def fake_exec(box):
            self.text = box.text()
            return 0

        monkeypatch.setattr(QMessageBox, "__init__", fake_init)
        monkeypatch.setattr(QMessageBox, "addButton", fake_add)
        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(
            QMessageBox, "clickedButton", lambda _box: box_ref.get("button")
        )


def test_the_dialog_says_how_to_get_the_model(panel, monkeypatch, tmp_path):
    """A user meeting this button has never heard of the model. The dialog has
    to carry the download command, not just "not installed"."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: tmp_path / "gone")
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    dialog = CapturedDialog(monkeypatch)

    panel._on_external_redraw()

    assert "fetch_anime2sketch" in dialog.text
    assert "Anime2Sketch" in dialog.text


def test_the_dialog_explains_the_why_not_only_the_where(panel, monkeypatch, tmp_path):
    """The model being absent is unusual enough to deserve a reason; otherwise
    it reads like a broken install."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: tmp_path / "gone")
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    dialog = CapturedDialog(monkeypatch)

    panel._on_external_redraw()

    assert "178" in dialog.text or "安装包" in dialog.text


def test_ready_dialog_names_the_standalone_workspace(panel, monkeypatch, tmp_path):
    """⚠️ There is no in-app workspace yet, so the ready state must say where
    the feature actually lives instead of implying this window runs it."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: complete_model(tmp_path))
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    dialog = CapturedDialog(monkeypatch)

    panel._on_external_redraw()

    assert "line_redraw_panel" in dialog.text


def test_the_dialog_never_offers_a_run_button(panel, monkeypatch, tmp_path):
    """No 开始重绘 / 处理图片 button until this panel can actually do it."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: complete_model(tmp_path))
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    dialog = CapturedDialog(monkeypatch)

    panel._on_external_redraw()

    for label in dialog.labels:
        assert "开始" not in label
        assert "重绘图片" not in label
    assert "选择模型文件夹" in " ".join(dialog.labels)


def test_choosing_a_folder_lands_on_the_model_section(panel, monkeypatch, tmp_path):
    """The shortcut has to arrive at the control it promised, not at the top of
    a long settings panel."""
    monkeypatch.setattr(model_paths, "resolve_model_dir", lambda *a, **k: tmp_path / "gone")
    monkeypatch.setattr(model_paths, "torch_available", lambda: True)
    dialog = CapturedDialog(monkeypatch)
    dialog.clicked = "选择模型文件夹…"

    panel._on_external_redraw()

    settings = show_settings(panel.parent_window)
    assert settings is not None
    assert settings.model_path_edit.isVisible()
    # ``hasFocus()`` is false for every widget while the window is inactive,
    # which is the case for a WA_DontShowOnScreen test window. The focus
    # *widget* is what the shortcut controls, and what the assertion is about.
    assert settings.focusWidget() is settings.model_browse_button, (
        "应把焦点落在模型文件夹控件上"
    )


# ---------------------------------------------------------------- the probe


def test_torch_available_answers_a_boolean(monkeypatch):
    """It is called while building a panel, so it must never raise and must not
    import torch (about 490 MB of RSS just to answer a yes/no question)."""
    import importlib.util

    assert isinstance(model_paths.torch_available(), bool)

    def explode(name):
        raise ImportError("half-removed install")

    monkeypatch.setattr(importlib.util, "find_spec", explode)
    assert model_paths.torch_available() is False
