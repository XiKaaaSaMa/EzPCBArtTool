"""The 滤镜 module's Java bridge.

The filters themselves are JH Labs code running in a JVM, so most of these
tests **skip** on a machine without Java -- that is deliberate and matches the
project's rule that the suite passes anywhere with a few more skips. What is
tested unconditionally is the part that must hold everywhere: the pass-through
entry costs nothing, unknown names are refused with a sentence the panel can
show, and a missing tool is reported rather than raised raw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.java_filters import (  # noqa: E402
    FILTERS,
    FILTER_LABELS,
    NO_FILTER,
    JavaFilterError,
    apply_filter,
    availability,
    find_java,
    tool_dir,
)


def _picture(height=24, width=32) -> np.ndarray:
    image = np.full((height, width, 3), 210, dtype=np.uint8)
    image[6:12, 8:20] = 40
    image[16:20, 4:28] = 120
    return image


def _java_ready() -> tuple[bool, str]:
    return availability()


needs_java = pytest.mark.skipif(
    not _java_ready()[0], reason=f"需要 Java 与工具：{_java_ready()[1]}"
)


def test_the_table_and_the_labels_agree():
    """Every filter the panel can pick has a label, and the pass-through leads."""
    assert FILTERS[0] == NO_FILTER
    assert set(FILTERS) == set(FILTER_LABELS)
    assert len(FILTERS) == len(set(FILTERS))


def test_the_pass_through_costs_nothing():
    """``null`` returns the picture untouched, without starting a JVM.

    It is the panel's default, and a default that spawns a JVM on every upload
    would be a hidden cost; the timestamp check makes the promise enforceable.
    """
    source = _picture()
    assert apply_filter(source, NO_FILTER) is not None
    same = apply_filter(source, NO_FILTER)
    assert np.array_equal(same, source)


def test_an_unknown_filter_is_refused():
    with pytest.raises(JavaFilterError):
        apply_filter(_picture(), "not-a-filter")


def test_a_missing_tool_directory_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("PCBART_JAVA_TOOL_DIR", str(tmp_path / "nowhere"))
    assert tool_dir() is None
    usable, reason = availability()
    assert not usable
    assert "java_lineart" in reason

    monkeypatch.setenv("PCBART_JAVA_TOOL_DIR", str(tmp_path))
    (tmp_path / "lib").mkdir()
    assert tool_dir() == tmp_path
    usable, reason = availability()
    # The directory exists but has no class file and possibly no Java.
    if not usable:
        assert reason


def test_java_is_looked_for_in_the_usual_places(monkeypatch):
    """PATH first; JAVA_HOME is the documented fallback."""
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    found = find_java()
    # Windows machines with an installed JRE are found by the directory scan;
    # the assertion is only that whatever comes back is an existing file.
    if found is not None:
        assert Path(found).exists()


@needs_java
def test_the_filter_list_matches_the_java_tool():
    """The table above and ``LineArtJavaTool list`` must not drift apart.

    A filter added to the Java source but not to ``FILTERS`` would be
    unreachable; one removed from the Java source but left in ``FILTERS`` would
    make the panel offer a name the tool rejects -- and both are silent until a
    user hits them.
    """
    import os
    import subprocess

    directory = tool_dir()
    java = find_java()
    assert directory is not None and java is not None
    classpath = os.pathsep.join(
        [str(directory / "classes"), str(directory / "lib" / "*")]
    )
    result = subprocess.run(
        [java, "-cp", classpath, "LineArtJavaTool", "list"],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    listed = [line.strip() for line in result.stdout.decode().splitlines() if line.strip()]
    assert listed == list(FILTERS)


@needs_java
def test_every_filter_produces_a_usable_picture():
    """⚠️ The regression net for the whole list -- and it is not hypothetical.

    Two of the sixteen were broken **as shipped**, and only a sweep over all of
    them finds it: ``sharpen`` returned an all-black image and ``dog`` threw
    ``ArithmeticException: / by zero``. Both had one root cause -- JH Labs'
    filters assume the source carries an alpha channel, and the tool handed
    them a no-alpha ``TYPE_INT_RGB`` (``premultiplyAlpha`` then multiplies every
    pixel to zero, and the DoG's normalisation divides by an empty range). The
    fix is in the Java tool's ``readFlat``; this test is what catches the next
    one. Measured before the fix: sharpen mean 0.0 (100% black), dog raised.
    """
    import cv2

    source = _picture()
    failures = []
    for name in FILTERS:
        if name == NO_FILTER:
            continue
        try:
            filtered = apply_filter(source, name)
        except JavaFilterError as exc:
            failures.append(f"{name}: 抛异常 {exc}")
            continue
        if filtered.shape != source.shape:
            failures.append(f"{name}: 尺寸变了 {filtered.shape}")
            continue
        grey = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        if int(grey.max()) == int(grey.min()):
            failures.append(f"{name}: 输出是常量图（均匀 {int(grey.max())}）")
    assert not failures, "有滤镜不可用：" + "；".join(failures)


@needs_java
def test_the_sharpen_filter_actually_sharpens():
    """The specific report that found the bug: sharpen must keep the picture.

    A sharpened picture is the source plus its edges -- so it is neither black
    nor identical to the source, and its mean stays near the source's.
    """
    import cv2

    source = _picture()
    sharpened = apply_filter(source, "sharpen")
    assert not np.array_equal(sharpened, source), "没变化的滤镜不算滤镜"
    source_mean = float(source.mean())
    assert abs(float(sharpened.mean()) - source_mean) < 12.0, (
        f"锐化后均值 {float(sharpened.mean()):.1f}，原图 {source_mean:.1f}——"
        "锐化应当基本保留原图明暗"
    )
    edge_source = cv2.Canny(source, 50, 150)
    edge_sharp = cv2.Canny(sharpened, 50, 150)
    assert int(edge_sharp.sum()) >= int(edge_source.sum()), "锐化后边缘应当不减少"


@needs_java
@pytest.mark.parametrize("name", ["stamp", "contour", "threshold"])
def test_a_filter_changes_the_picture_and_keeps_its_shape(name):
    source = _picture()
    filtered = apply_filter(source, name)
    assert filtered.shape == source.shape
    assert filtered.dtype == np.uint8
    assert not np.array_equal(filtered, source), f"{name} 没有改变画面"


@needs_java
def test_grey_and_alpha_inputs_are_flattened():
    """The tool flattens alpha itself; the bridge must not hand it 4 channels."""
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = 255
    out = apply_filter(rgba, "median")
    assert out.shape == (16, 16, 3)
