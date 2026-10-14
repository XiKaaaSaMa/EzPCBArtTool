"""The 图像预处理 section: 超分 and 滤镜 as source-level pre-steps.

Both steps replace the *picture* before anything reads it, and both cost
seconds (Anime4K ×2 measured 4.4 s on a 512 px picture; a Java filter call
0.4–0.6 s plus JVM start), so the panel computes them once per source in a
worker thread and caches the result. These tests pin that arrangement: the
default state is free and synchronous, a change rebuilds asynchronously, the
result is reusable, an over-limit upscale is reported instead of refused, and
the import cannot commit a picture the pre-steps have not finished with.

The worker is exercised with a stubbed ``preprocess_source`` so the tests do
not depend on Java being installed or on how fast a JVM starts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import (  # noqa: E402
    NO_FILTER,
    build_window,
    preprocess_source,
    show_preprocess,
)
from pcbart.image_prep import imwrite_unicode  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def picture(tmp_path):
    """A small colour picture with a transparent corner."""
    path = tmp_path / "art.png"
    image = np.zeros((64, 64, 4), dtype=np.uint8)
    image[..., :3] = 200
    image[8:40, 12:52, :3] = 40
    image[..., 3] = 255
    image[:8, :, 3] = 0  # a transparent strip
    imwrite_unicode(str(path), image)
    assert path.exists()
    return path


@pytest.fixture
def loaded(app, picture):
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    widget = show_preprocess(window)
    widget.parent_window = window
    assert widget.load_image(str(picture)) is True
    app.processEvents()
    yield widget
    from conftest import release_window

    release_window(window)


def _drain(app, panel, limit: float = 20.0) -> None:
    """Spin the event loop until the pre-step worker has finished."""
    deadline = time.time() + limit
    while panel._prestep_busy():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() < deadline, "预处理 worker 超时未结束"
    app.processEvents()


# ------------------------------------------------------------ the pure step


def test_the_default_settings_pass_the_picture_through():
    """关 / 无 is the default state and must be a no-op, object for object."""
    source = np.full((16, 16, 3), 120, dtype=np.uint8)
    alpha = np.full((16, 16), 255, dtype=np.uint8)
    out, out_alpha, note = preprocess_source(source, alpha, 1, NO_FILTER)
    assert out is source
    assert out_alpha is alpha
    assert note == ""


def test_an_over_limit_upscale_is_reported_not_refused():
    """The limits are a time budget, not a correctness rule.

    A picture past the factor's input limit comes back **unchanged** with a
    sentence saying so -- the workspace reported it the same way, and silently
    upscaling anyway would freeze the panel for a minute.
    """
    source = np.full((40, 3000, 3), 200, dtype=np.uint8)
    out, _alpha, note = preprocess_source(source, None, 2, NO_FILTER)
    assert out is source
    assert "跳过" in note and "3000" in note


def test_the_upscale_carries_an_alpha_mask_along():
    """The cut-out mask has to keep lining up with the colours."""
    source = np.full((32, 32, 3), 100, dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    out, out_alpha, note = preprocess_source(source, alpha, 2, NO_FILTER)
    assert note == ""
    assert out.shape[:2] == (64, 64)
    assert out_alpha is not None and out_alpha.shape == (64, 64)
    # Nearest-neighbour: every output pixel is one of the two input values.
    assert set(np.unique(out_alpha).tolist()) <= {0, 255}
    # The opaque square doubled with it.
    assert out_alpha[16:48, 16:48].min() == 255


# --------------------------------------------------------------- the panel


def test_the_default_state_adopts_the_raw_picture_without_a_worker(loaded):
    """Nothing to compute means no thread, no JVM and no delay."""
    assert loaded._rebuild_source is not None
    assert loaded._prepared_key == loaded._prestep_key()
    assert loaded._source_bgr is loaded._raw_bgr
    assert loaded._prepare_worker is None
    assert loaded.upscale_combo.currentIndex() == 0
    assert loaded.filter_combo.currentIndex() == 0


def test_choosing_a_filter_rebuilds_the_source_in_a_worker(app, loaded, monkeypatch):
    """The rebuild is asynchronous, blocks the import, and lands in the source.

    ``preprocess_source`` is stubbed because the real one spawns a JVM; what is
    under test is the panel's arrangement -- worker, keying, the disabled
    import -- not the Java tool, which has its own tests.
    """
    calls: list[tuple] = []

    def slow_step(bgr, alpha, factor, name):
        calls.append((factor, name))
        time.sleep(0.15)
        # -1 rather than +1: the picture's corner is 255 and uint8 would wrap.
        return np.clip(bgr.astype(np.int16) - 1, 0, 255).astype(np.uint8), alpha, ""

    monkeypatch.setattr(gui_mockup, "preprocess_source", slow_step)

    loaded.filter_combo.setCurrentIndex([f for f in gui_mockup.FILTERS].index("stamp"))
    assert loaded._prestep_busy(), "选滤镜后应当有后台任务在跑"
    assert not loaded.import_button.isEnabled(), "预处理没算完之前不该允许导入"
    _drain(app, loaded)

    assert calls == [(1, "stamp")]
    assert loaded._source_bgr is not loaded._raw_bgr
    assert int(loaded._source_bgr[0, 0, 0]) == int(loaded._raw_bgr[0, 0, 0]) - 1
    assert loaded.import_button.isEnabled()
    assert loaded.prestep_note.isHidden() or loaded.prestep_note.text() == ""


def test_switching_back_and_forth_reuses_the_cached_result(app, loaded, monkeypatch):
    """Computed once per setting: flipping the picker back is instant."""
    calls: list[tuple] = []

    def counted_step(bgr, alpha, factor, name):
        calls.append((factor, name))
        return np.clip(bgr.astype(np.int16) - 5, 0, 255).astype(np.uint8), alpha, ""

    monkeypatch.setattr(gui_mockup, "preprocess_source", counted_step)
    index = [f for f in gui_mockup.FILTERS].index("contour")
    loaded.filter_combo.setCurrentIndex(index)
    _drain(app, loaded)
    after_filter = loaded._source_bgr

    loaded.filter_combo.setCurrentIndex(0)  # 无
    _drain(app, loaded)
    assert loaded._source_bgr is loaded._raw_bgr

    loaded.filter_combo.setCurrentIndex(index)  # back to contour
    _drain(app, loaded)
    assert loaded._source_bgr is after_filter, "第二次应当直接取缓存的结果"
    assert calls == [(1, "contour")], f"不该重复计算：{calls}"


def test_an_over_limit_upscale_is_skipped_with_a_note(app, tmp_path):
    """A 3000 px picture cannot be doubled -- and says so instead of hanging."""
    path = tmp_path / "wide.png"
    imwrite_unicode(str(path), np.full((40, 3000, 3), 210, dtype=np.uint8))
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    panel = show_preprocess(window)
    panel.parent_window = window
    try:
        assert panel.load_image(str(path)) is True
        app.processEvents()
        panel.upscale_combo.setCurrentIndex(1)  # Anime4K ×2
        _drain(app, panel)
        assert panel._source_bgr is panel._raw_bgr, "超限应当跳过放大"
        assert "跳过" in panel.prestep_note.text()
    finally:
        from conftest import release_window

        release_window(window)


def test_reset_puts_the_pre_steps_back_to_their_defaults(app, loaded, monkeypatch):
    # The stub keeps the two combo changes from starting a real upscale; the
    # test is about the reset, and _drain below leaves no worker running when
    # the fixture tears the window down.
    monkeypatch.setattr(
        gui_mockup,
        "preprocess_source",
        lambda bgr, alpha, factor, name: (bgr, alpha, ""),
    )
    loaded.upscale_combo.setCurrentIndex(2)
    loaded.filter_combo.setCurrentIndex(3)
    loaded.ridge_percentile_slider.setValue(95)
    loaded.xdog_level_slider.setValue(200)

    loaded._reset_tuning()
    _drain(app, loaded)

    assert loaded.upscale_combo.currentIndex() == 0
    assert loaded.filter_combo.currentIndex() == 0
    assert loaded.ridge_percentile_slider.value() == 50
    assert loaded.xdog_level_slider.value() == 128
    assert loaded._source_bgr is loaded._raw_bgr


def test_the_filter_picker_is_disabled_without_java(app, loaded, monkeypatch):
    """A dead picker must say why and must not keep a stale selection."""
    import pcbart.java_filters as bridge

    # Stub the step itself: this test is about the picker's state, and a
    # programmatic index change would otherwise start the real Java worker.
    monkeypatch.setattr(
        gui_mockup,
        "preprocess_source",
        lambda bgr, alpha, factor, name: (bgr, alpha, ""),
    )
    monkeypatch.setattr(bridge, "availability", lambda: (False, "未找到 Java"))
    loaded._refresh_filter_availability()
    assert not loaded.filter_combo.isEnabled()

    # A selection that got in anyway (a restored setting, a programmatic
    # change) is pulled back to 无 rather than left as a filter that cannot run.
    loaded.filter_combo.setCurrentIndex(2)
    _drain(app, loaded)
    loaded._refresh_filter_availability()
    assert loaded._filter_name() == NO_FILTER, "不可用时不该留着选中的滤镜"

    monkeypatch.setattr(bridge, "availability", lambda: (True, ""))
    loaded._refresh_filter_availability()
    assert loaded.filter_combo.isEnabled()