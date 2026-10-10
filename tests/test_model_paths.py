"""Where the external redraw model is found, and whether it can run.

The model is **not shipped**: 183 MB of weights against a 178 MB installer, and
a CPU redraw peaks at 4.6 GB of RAM. So "the user points the program at a
folder they downloaded" is the whole delivery mechanism, and every mistake in
that path shows up as "the feature does not work" with no clue why.

The cases that matter, and why:

* the resolution order, because three sources can all name a folder and the
  user has to be able to predict which wins;
* **a folder holding only ``netG.pth`` is not ready** -- the loader asks for
  ``improved``, and the two weight sets fit different architectures, so
  "some weights are present" is not the same as "this will load";
* the theme's save path must not erase the model path. Both live in the same
  ``settings.json``, and the theme writer used to replace the file wholesale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import model_paths, ui_theme  # noqa: E402
from pcbart.model_paths import (  # noqa: E402
    ARCHITECTURE_FILE,
    DEFAULT_VARIANT,
    MODEL_ENV,
    WEIGHT_FILES,
    candidate_dirs,
    check_model_dir,
    default_model_dir,
    resolve_model_dir,
    variant_present,
)


def make_model(root: Path, *variants: str) -> Path:
    """A folder that looks like a downloaded model, with the named weights."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ARCHITECTURE_FILE).write_text("# architecture\n", encoding="utf-8")
    (root / "weights").mkdir(exist_ok=True)
    for variant in variants:
        (root / WEIGHT_FILES[variant]).write_bytes(b"\x80 not really a model")
    return root


@pytest.fixture(autouse=True)
def no_ambient_model(monkeypatch):
    """Neither the developer's environment nor their settings may leak in.

    ``PCBART_A2S_MODEL`` can legitimately be set on the machine running the
    suite, and the settings file is redirected by ``conftest`` but may still
    hold a path from an earlier test in the same session.
    """
    monkeypatch.delenv(MODEL_ENV, raising=False)
    monkeypatch.setattr(model_paths, "_from_settings", lambda: None)


# ------------------------------------------------------------------ ordering


def test_the_drop_in_folder_is_the_last_resort(monkeypatch, tmp_path):
    """Nothing configured: the folder the fetch script writes to."""
    monkeypatch.setattr(model_paths, "default_model_dir", lambda: tmp_path / "default")

    assert candidate_dirs() == [tmp_path / "default"]


def test_the_environment_beats_the_saved_setting(monkeypatch, tmp_path):
    """Otherwise a machine-wide choice could be silently overridden by a click."""
    monkeypatch.setenv(MODEL_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(model_paths, "_from_settings", lambda: tmp_path / "chosen")
    monkeypatch.setattr(model_paths, "default_model_dir", lambda: tmp_path / "default")

    assert candidate_dirs() == [
        tmp_path / "env",
        tmp_path / "chosen",
        tmp_path / "default",
    ]


def test_an_explicit_path_beats_everything(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(model_paths, "_from_settings", lambda: tmp_path / "chosen")

    assert candidate_dirs(tmp_path / "cli")[0] == tmp_path / "cli"


def test_a_configured_folder_stays_in_force_even_when_it_is_broken(monkeypatch, tmp_path):
    """⚠️ The choice is honoured, not repaired behind the user's back.

    Skipping a folder the user named, in favour of one that happens to work,
    would run the model from A while the settings panel displays B. That is the
    "what the interface shows is not what happens" defect, and it is worse than
    a clear failure: the user would edit or replace the wrong folder trying to
    make the panel's complaint go away.
    """
    broken = tmp_path / "chosen-but-empty"
    broken.mkdir()
    monkeypatch.setenv(MODEL_ENV, str(broken))

    assert resolve_model_dir() == broken


def test_resolution_falls_back_only_when_nothing_is_configured(monkeypatch, tmp_path):
    """The drop-in location is a default, not a rescue.

    A user who just ran the fetch script must not have to configure anything.
    """
    fallback = tmp_path / "default"
    monkeypatch.setattr(model_paths, "default_model_dir", lambda: fallback)

    assert resolve_model_dir() == fallback


def test_candidates_include_the_drop_in_location_last(monkeypatch, tmp_path):
    """The panel lists everywhere it would look, so "not found" is actionable."""
    monkeypatch.setenv(MODEL_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(model_paths, "default_model_dir", lambda: tmp_path / "default")

    assert candidate_dirs() == [tmp_path / "env", tmp_path / "default"]


# ------------------------------------------------------------------- checking


def test_a_complete_folder_is_ready(tmp_path):
    root = make_model(tmp_path / "a2s", DEFAULT_VARIANT)

    report = check_model_dir(root)

    assert report["ok"] is True
    assert report["missing"] == []
    assert report["weights"][DEFAULT_VARIANT] is True


def test_a_folder_without_the_architecture_is_not_ready(tmp_path):
    """Weights alone cannot be loaded: the U-Net has to be constructed."""
    root = tmp_path / "weights-only"
    (root / "weights").mkdir(parents=True)
    (root / WEIGHT_FILES[DEFAULT_VARIANT]).write_bytes(b"\x80")

    report = check_model_dir(root)

    assert report["ok"] is False
    assert ARCHITECTURE_FILE in report["missing"]


def test_only_the_original_weights_is_not_ready_for_improved(tmp_path):
    """⚠️ The bug this guards: 'some weights are here' is not 'it will load'.

    ``netG.pth`` and ``improved.bin`` fit different architectures (the improved
    set replaces six deconvolution layers), so a folder carrying only the
    original release cannot run the variant every measurement in this project
    was taken on. Reporting it as ready moves the failure into ``torch.load``,
    whose message says nothing about which file is wrong.
    """
    root = make_model(tmp_path / "a2s", "default")

    report = check_model_dir(root, "improved")

    assert report["ok"] is False
    assert report["weights"]["default"] is True
    assert report["weights"]["improved"] is False
    assert any("improved" in item for item in report["missing"]), (
        "提示必须点名缺的是哪一套权重"
    )


def test_the_same_folder_is_ready_for_the_variant_it_does_hold(tmp_path):
    root = make_model(tmp_path / "a2s", "default")

    assert check_model_dir(root, "default")["ok"] is True


def test_an_unknown_variant_is_a_programming_error(tmp_path):
    """Silently treating it as "not present" would hide a typo in a call."""
    with pytest.raises(ValueError):
        check_model_dir(tmp_path, "improved-v2")


def test_variant_present_answers_for_one_named_set(tmp_path):
    root = make_model(tmp_path / "a2s", "default")

    assert variant_present(root, "default") is True
    assert variant_present(root, "improved") is False
    assert variant_present(root, "nonsense") is False


def test_a_missing_folder_is_reported_not_raised(tmp_path):
    """Not downloaded yet is an ordinary state, not an exception."""
    report = check_model_dir(tmp_path / "never-downloaded")

    assert report["ok"] is False
    assert report["missing"]


# --------------------------------------------------------------- the default


def test_from_source_the_drop_in_folder_is_inside_out():
    """Same convention as the cache and draft directories, so the docs, the
    fetch script and the panel all agree about where to look."""
    project = Path(__file__).resolve().parent.parent
    resolved = model_paths.default_model_dir()

    assert resolved.parts[-3:] == ("out", "_models", "a2s")
    assert resolved.is_relative_to(project), "源码运行时默认位置应在工程目录内"


def test_a_packaged_build_uses_the_user_directory(monkeypatch, tmp_path):
    """An install under Program Files is read-only, and an installer upgrade
    replaces it -- a model folder there would be lost or unwritable."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    resolved = model_paths.default_model_dir()

    assert str(tmp_path) in str(resolved)
    assert resolved.parts[-2:] == ("models", "a2s")


# ----------------------------------------------------- the settings file merge


def test_picking_a_model_survives_a_colour_change(tmp_path, monkeypatch):
    """⚠️ The colour writer used to replace ``settings.json`` wholesale.

    Model path and colour scheme share the file, so saving the scheme after
    choosing a model would have erased the choice -- and the symptom would
    appear much later, as "the model I picked does not stick".
    """
    path = tmp_path / "settings.json"
    monkeypatch.setattr(ui_theme, "settings_path", lambda: path)

    ui_theme.save_settings_keeping(a2s_model=r"D:\models\a2s")
    ui_theme.save_settings_keeping(**ui_theme.settings_for(ui_theme.scheme_from_seed("#6750A4")))

    saved = ui_theme.load_settings()
    assert saved["a2s_model"] == r"D:\models\a2s"
    assert saved["seed"] == "#6750A4"


def test_clearing_the_model_removes_the_key_rather_than_storing_null(tmp_path, monkeypatch):
    """An absent key is what "not chosen" means; a stored null reads as a choice."""
    path = tmp_path / "settings.json"
    monkeypatch.setattr(ui_theme, "settings_path", lambda: path)

    ui_theme.save_settings_keeping(a2s_model=r"D:\models\a2s")
    ui_theme.save_settings_keeping(a2s_model=None)

    assert "a2s_model" not in ui_theme.load_settings()


def test_a_saved_choice_is_used_by_the_resolver(monkeypatch, tmp_path):
    """The merge above is only useful if the resolver actually reads the key."""
    monkeypatch.setattr(
        model_paths, "_from_settings", lambda: tmp_path / "chosen"
    )

    assert resolve_model_dir() == tmp_path / "chosen"


def test_the_environment_outranks_the_saved_choice(monkeypatch, tmp_path):
    """So a machine can point every session at a shared copy."""
    monkeypatch.setenv(MODEL_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(model_paths, "_from_settings", lambda: tmp_path / "chosen")

    assert resolve_model_dir() == tmp_path / "env"
