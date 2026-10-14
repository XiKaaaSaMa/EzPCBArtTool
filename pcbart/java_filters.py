"""The 滤镜 module's engine: JH Labs filters, called through the Java tool.

Why a subprocess
----------------
The 16 filters come from **JH Labs Filters** (Apache-2.0), whose source this
project does not have a licence-clean Python port of, and whose behaviour the
user already signed off on in the 线稿工坊 workspace. Re-implementing sixteen
filters in numpy would be a second, subtly different implementation of someone
else's algorithms -- the exact duplication this codebase keeps rejecting -- so
the jar is shipped instead (373 KB) and invoked as a subprocess, the same way
the workspace did.

**The cost is that the filters need a Java runtime.** A machine without one
still gets everything else in the tool: :func:`availability` reports the
reason, and the panel disables the picker rather than offering a control that
cannot work. ``classes/LineArtJavaTool.class`` ships prebuilt against **Java
8**, so a plain JRE is enough -- a JDK (``javac``) is only needed when the
class is missing, which is the source-checkout case.

``null`` is a real filter name here: it means "no filter, pass the picture
through", is the panel's default, and is answered **without starting the JVM**
-- the default state of the feature costs nothing.

⚠️ **The Java side must hand the filters an ARGB image.** JH Labs' filters
assume the source carries an alpha channel; on a no-alpha ``TYPE_INT_RGB`` the
``sharpen`` filter returns an all-black picture and ``dog`` throws
``ArithmeticException: / by zero`` (measured -- see ``java_lineart/README.md``).
The tool's ``readFlat`` makes the ARGB image, and
``tests/test_java_filters.py`` sweeps all sixteen filters for output that is
neither constant nor an exception.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "FILTERS",
    "FILTER_LABELS",
    "JavaFilterError",
    "NO_FILTER",
    "apply_filter",
    "availability",
    "find_java",
    "tool_dir",
]

#: The filters the Java tool exposes, in its own order, ``null`` first.
#: Kept in step with ``java_lineart/src/LineArtJavaTool.java``; the panel's
#: dropdown is built from this tuple and a test asserts the two agree by
#: asking the tool itself (when Java is present).
FILTERS: tuple[str, ...] = (
    "null",
    "stamp",
    "skeleton",
    "edge",
    "contour",
    "dog",
    "threshold",
    "crystallize",
    "marble",
    "chrome",
    "weave",
    "pointillize",
    "emboss",
    "smear",
    "diffuse",
    "median",
    "sharpen",
)

NO_FILTER = "null"
"""The pass-through entry. Not a filter: the picture is returned untouched."""

#: Panel-facing names. The JH Labs names are Java class names and mean little
#: on their own, so each carries the one-word idea it is used for here; the
#: tooltips in the panel carry the longer descriptions.
FILTER_LABELS: dict[str, str] = {
    "null": "无（不滤镜）",
    "stamp": "印章 stamp",
    "skeleton": "骨架 skeleton",
    "edge": "边缘 edge",
    "contour": "轮廓 contour",
    "dog": "高斯差 dog",
    "threshold": "阈值 threshold",
    "crystallize": "晶格 crystallize",
    "marble": "大理石 marble",
    "chrome": "铬光 chrome",
    "weave": "织纹 weave",
    "pointillize": "点彩 pointillize",
    "emboss": "浮雕 emboss",
    "smear": "涂抹 smear",
    "diffuse": "扩散 diffuse",
    "median": "中值 median",
    "sharpen": "锐化 sharpen",
}

#: Directory name of the tool, looked for beside the repository (source runs)
#: and beside the executable (a packaged install). Overridable with
#: ``PCBART_JAVA_TOOL_DIR`` so a test -- or a user keeping the tool elsewhere --
#: can point at it.
TOOL_DIR_NAME = "java_lineart"


class JavaFilterError(RuntimeError):
    """Raised when the filter cannot be applied, with a message for the panel."""


def tool_dir() -> Path | None:
    """Where the Java tool lives, or ``None`` when it cannot be found."""
    override = os.environ.get("PCBART_JAVA_TOOL_DIR", "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None

    candidates: list[Path] = []
    package_root = Path(__file__).resolve().parent.parent
    candidates.append(package_root / TOOL_DIR_NAME)
    if getattr(sys, "frozen", False):
        # PyInstaller: the install directory sits next to the executable.
        candidates.append(Path(sys.executable).resolve().parent / TOOL_DIR_NAME)
    for candidate in candidates:
        if (candidate / "lib").is_dir():
            return candidate
    return None


def find_java() -> str | None:
    """A usable ``java`` executable, or ``None``.

    PATH first, then ``JAVA_HOME``, then the usual Windows install roots --
    a machine can have a JRE installed and still have it off PATH, and being
    lenient here is the difference between the feature working and not.
    """
    on_path = shutil.which("java")
    if on_path:
        return on_path

    home = os.environ.get("JAVA_HOME", "").strip()
    if home:
        for name in ("java.exe", "java"):
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)

    if sys.platform == "win32":
        roots = [
            Path(r"C:\Program Files\Java"),
            Path(r"C:\Program Files\Zulu"),
            Path(r"C:\Program Files\Eclipse Adoptium"),
            Path(r"C:\Program Files\Microsoft"),
            Path(r"C:\Program Files\Amazon Corretto"),
        ]
        for root in roots:
            if not root.is_dir():
                continue
            # Newest-looking first is not worth parsing: any java >= 8 runs
            # the tool, so the first hit is good enough.
            for candidate in sorted(root.glob("*/bin/java.exe"), reverse=True):
                if candidate.exists():
                    return str(candidate)
    return None


def _class_file(directory: Path) -> Path:
    return directory / "classes" / "LineArtJavaTool.class"


def _build(directory: Path) -> tuple[bool, str]:
    """Compile the tool with javac, as the workspace does on first use."""
    javac = shutil.which("javac")
    if not javac:
        return False, "未找到 javac，且工具尚未编译"
    classes = directory / "classes"
    try:
        classes.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"无法写入 {classes}：{exc}"
    command = [
        javac,
        "--release",
        "8",
        "-Xlint:-options",
        "-encoding",
        "UTF-8",
        "-cp",
        str(directory / "lib" / "*"),
        "-d",
        str(classes),
        str(directory / "src" / "LineArtJavaTool.java"),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"编译 Java 工具失败：{exc}"
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        return False, f"编译 Java 工具失败：{detail[:200]}"
    return True, ""


def availability() -> tuple[bool, str]:
    """``(usable, reason)`` for the panel's status line.

    ``usable`` means "picking a filter will do something"; the reason explains
    the first thing that is missing, in the order the user can act on it.
    """
    directory = tool_dir()
    if directory is None:
        return False, "缺少 java_lineart 工具目录"
    if find_java() is None:
        return False, "未找到 Java（安装 JRE 8+ 后可用）"
    if not _class_file(directory).exists():
        built, message = _build(directory)
        if not built:
            return False, message
    return True, ""


def _flatten(image: np.ndarray) -> np.ndarray:
    """Three-channel BGR uint8, whatever came in."""
    from .image_prep import flatten_alpha

    if image.ndim == 2:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return flatten_alpha(image[:, :, :3], image[:, :, 3])
    if image.shape[2] == 2:
        import cv2

        return flatten_alpha(cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR),
                             image[:, :, 1])
    return np.ascontiguousarray(image[:, :, :3])


def apply_filter(
    image: np.ndarray, name: str, *, timeout: float = 300.0
) -> np.ndarray:
    """``image`` through the named JH Labs filter.

    ``"null"`` returns the picture unchanged **without touching the JVM**, so
    the default state of the 滤镜 module costs nothing; every other name goes
    through the tool. Raises :class:`JavaFilterError` with a sentence the
    panel can show.
    """
    picture = _flatten(image)
    if name == NO_FILTER:
        return picture
    if name not in FILTERS:
        raise JavaFilterError(f"未知滤镜：{name}")

    ready, reason = availability()
    if not ready:
        raise JavaFilterError(reason)
    directory = tool_dir()
    java = find_java()
    assert directory is not None and java is not None  # availability() checked

    from .image_prep import imread_unicode, imwrite_unicode

    with tempfile.TemporaryDirectory(prefix="pcbart_java_") as folder:
        source = Path(folder) / "in.png"
        target = Path(folder) / "out.png"
        imwrite_unicode(str(source), picture)
        classpath = os.pathsep.join(
            [str(directory / "classes"), str(directory / "lib" / "*")]
        )
        command = [
            java,
            "-cp",
            classpath,
            "LineArtJavaTool",
            "filter",
            name,
            str(source),
            str(target),
        ]
        creationflags = 0
        if sys.platform == "win32":
            # No console window flashing up over the application.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                command,
                cwd=folder,
                capture_output=True,
                timeout=timeout,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise JavaFilterError(f"滤镜「{name}」执行超时") from exc
        except OSError as exc:
            raise JavaFilterError(f"无法启动 Java：{exc}") from exc
        output = (
            (result.stdout or b"").decode("utf-8", "replace")
            + (result.stderr or b"").decode("utf-8", "replace")
        ).strip()
        if result.returncode != 0 or not target.exists():
            raise JavaFilterError(f"滤镜「{name}」失败：{output[:200]}")
        import cv2

        filtered = imread_unicode(str(target), cv2.IMREAD_COLOR)
    if filtered is None:
        raise JavaFilterError(f"滤镜「{name}」没有写出图片")
    if filtered.ndim == 2:
        import cv2

        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    return filtered
