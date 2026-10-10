"""Where the optional redraw model lives, and whether it is complete.

The line-redraw reading (Anime2Sketch) is the one feature that needs files the
installer does not carry. Two reasons, both measured rather than assumed:

* **size** -- the improved weights are 183 MB and the architecture is another
  few kilobytes, against a 178 MB installer;
* **cost** -- on CPU one 1024 px redraw takes ~1.6 s and peaks at **4.6 GB** of
  RAM, so it is not something every user's machine can run.

So the model is **external and optional**: the user points the program at a
folder they downloaded, and everything else keeps working when it is absent.
This module owns the "where is it" question so no call site hardcodes a path --
the same reason :mod:`pcbart.paths` exists for writable files.

Resolution order, first hit wins:

1. an explicit path passed by the caller (a command-line ``--model-dir``);
2. ``PCBART_A2S_MODEL`` -- for machines that keep it on another drive, and for
   tests that must not touch the real library;
3. the ``a2s_model`` entry in the saved preferences, i.e. what the user picked
   in【界面设置 → 线稿重绘模型】;
4. **only if none of the above is set**, the default drop-in location --
   ``out/_models/a2s`` from source, ``%LOCALAPPDATA%\\PCB艺术板工具\\models\\a2s``
   when packaged.

⚠️ Note what item 4 is *not*: it is not a fallback that rescues a configured
folder which turns out to be unusable. Something the user named stays in force
and gets reported as broken. See :func:`resolve_model_dir`.

Nothing here raises on a missing model: "not downloaded yet" is a normal state
and the answer to it is a message, not an exception.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

__all__ = [
    "MODEL_ENV",
    "DEFAULT_VARIANT",
    "ARCHITECTURE_FILE",
    "WEIGHT_VARIANTS",
    "WEIGHT_FILES",
    "TORCH_PACKAGE",
    "default_model_dir",
    "configured_dirs",
    "candidate_dirs",
    "resolve_model_dir",
    "check_model_dir",
    "variant_present",
    "torch_available",
]

#: The weight set the project runs. See :mod:`tools.probe_line_redraw` for the
#: measurements: the improved weights keep dark, low-contrast illustrations in
#: unbroken strokes where the original release leaves scattered fragments, and
#: every reading in this project was measured on them.
DEFAULT_VARIANT = "improved"

#: The optional runtime that executes the weights. Named here so the message a
#: user gets names the package they have to install.
TORCH_PACKAGE = "torch"

MODEL_ENV = "PCBART_A2S_MODEL"
"""Environment override, following :data:`pcbart.paths.ASSET_ROOT_ENV`.

Named for the model rather than the feature so a machine can also keep several
downloads side by side and switch by changing one variable.
"""

#: The architecture, imported verbatim from the reference implementation. It
#: is not optional: :func:`tools.probe_line_redraw.load_model` executes this
#: exact file so the weights are read by the code they were published with.
ARCHITECTURE_FILE = "model.py"

#: The two published weight sets. ``improved`` is the one this project uses
#: (the author's retrain for dark, low-contrast pictures -- see the loader's
#: docstring); ``default`` is accepted so a user who only has the original
#: release can still run the feature.
WEIGHT_FILES = {
    "default": Path("weights") / "netG.pth",
    "improved": Path("weights") / "improved.bin",
}

WEIGHT_VARIANTS = tuple(WEIGHT_FILES)


def default_model_dir() -> Path:
    """The drop-in folder when the user has not chosen one.

    Deliberately mirrors :func:`pcbart.paths.cache_dir`: from source it sits in
    ``out/`` next to the rest of the tool's scratch files (where the fetch
    script already puts it), and a packaged build uses the per-user directory,
    which is writable without elevation and survives upgrades. A path inside
    the install folder would need administrator rights and would be replaced
    by the next installer run.
    """
    from .paths import data_root, is_frozen

    if is_frozen():
        return data_root() / "models" / "a2s"
    return data_root() / "out" / "_models" / "a2s"


def _from_settings() -> Path | None:
    """The folder chosen in the interface, if one was.

    Imported lazily and defensively: this module is used by the probe scripts
    and by tests that may run without a settings file at all, and a preference
    file that has been hand-edited into nonsense must not stop the program.
    """
    try:
        from .ui_theme import load_settings
    except Exception:  # noqa: BLE001 - optional dependency in script contexts
        return None
    try:
        value = load_settings().get("a2s_model")
    except Exception:  # noqa: BLE001 - never let preferences break loading
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def configured_dirs(explicit: str | Path | None = None) -> list[Path]:
    """Only the folders someone asked for, in priority order.

    Separated from :func:`candidate_dirs` because the two answer different
    questions: "which folder is in force" (this) versus "everywhere we would
    look" (that one, which appends the drop-in location).
    """
    configured: list[Path] = []
    if explicit is not None:
        configured.append(Path(explicit).expanduser())
    env = os.environ.get(MODEL_ENV)
    if env:
        configured.append(Path(env).expanduser())
    from_settings = _from_settings()
    if from_settings is not None:
        configured.append(from_settings)
    return configured


def candidate_dirs(explicit: str | Path | None = None) -> list[Path]:
    """Every folder to try, in resolution order, without de-duplication.

    Kept as a list rather than a single answer so the panel can report *all*
    the places it looked -- "not found" is only actionable when it also says
    where it looked.
    """
    return configured_dirs(explicit) + [default_model_dir()]


def resolve_model_dir(explicit: str | Path | None = None) -> Path:
    """The folder actually in force: the first one **configured**, else the default.

    ⚠️ **A configured folder wins even when it turns out to be unusable**, and
    this is deliberate. The alternative -- skipping a broken setting to use
    whatever else is lying around -- means the program runs a model from folder
    A while the settings panel displays folder B, which is precisely the
    "what the interface shows is not what happens" failure this project treats
    as a defect. The user's own choice is reported back, and
    :func:`check_model_dir` says what is wrong with it.

    Only when nothing is configured at all does the drop-in location apply, so
    a user who just ran the fetch script never has to configure anything.
    """
    configured = configured_dirs(explicit)
    return configured[0] if configured else default_model_dir()


def check_model_dir(
    directory: str | Path, variant: str = DEFAULT_VARIANT
) -> dict:
    """Whether ``directory`` can run ``variant``, and what is missing.

    Returns a plain dict so both the panel and the tests can read it without
    importing Qt::

        {"ok": bool, "path": Path, "variant": str, "architecture": bool,
         "weights": {"default": bool, "improved": bool}, "missing": [str]}

    ⚠️ **``ok`` is about one specific variant, not "some weights are here".**
    The loader asks for a named weight set, and its files are not
    interchangeable -- ``improved`` replaces six deconvolution layers with
    resize-plus-conv blocks, so the two sets fit different architectures. A
    folder holding only the original ``netG.pth`` cannot run the ``improved``
    variant the project defaults to, and reporting that folder as "ready"
    would move the failure to a much less clear message from inside
    ``torch.load``.
    """
    path = Path(directory)
    if variant not in WEIGHT_FILES:
        raise ValueError(f"未知模型变体：{variant}")
    architecture = (path / ARCHITECTURE_FILE).is_file()
    weights = {
        name: (path / relative).is_file()
        for name, relative in WEIGHT_FILES.items()
    }
    missing = [] if architecture else [ARCHITECTURE_FILE]
    if not weights[variant]:
        others = [name for name, present in weights.items() if present]
        detail = f"（这个文件夹里有 {'、'.join(others)}，但项目用的是 {variant}）" if others else ""
        missing.append(f"权重 {WEIGHT_FILES[variant]}{detail}")
    return {
        "ok": architecture and weights[variant],
        "path": path,
        "variant": variant,
        "architecture": architecture,
        "weights": weights,
        "missing": missing,
    }


def variant_present(directory: str | Path, variant: str = DEFAULT_VARIANT) -> bool:
    """Whether one specific weight set is in ``directory``."""
    relative = WEIGHT_FILES.get(variant)
    return bool(relative) and (Path(directory) / relative).is_file()


def torch_available() -> bool:
    """Whether the model's runtime can be imported at all.

    A model folder is necessary but not sufficient: the weights are only usable
    through PyTorch, which is **not** a dependency of the main program (see
    ``requirements.txt``). Checking both is what lets the interface say "模型
    在，但缺运行库" instead of failing later with an ImportError from inside
    the loader.

    ``find_spec`` rather than ``import torch`` on purpose: importing torch
    costs about half a second and ~490 MB of RSS, and this is called while
    building a panel.
    """
    try:
        return importlib.util.find_spec(TORCH_PACKAGE) is not None
    except (ImportError, ValueError):
        # A broken or half-removed install raises instead of returning None.
        return False
