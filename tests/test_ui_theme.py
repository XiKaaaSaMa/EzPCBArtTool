"""Tests for the colour scheme: the maths, the presets, and the saved file.

Pure Python -- the colour generation has no Qt in it, so it is tested directly
rather than through a rendered window. The Qt half (icons, the stylesheet, the
live switch) is covered in ``test_theme_switch.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import ui_theme  # noqa: E402
from pcbart.ui_theme import (  # noqa: E402
    DEFAULT_SEED,
    PRESETS,
    ROLE_LABELS,
    _hex_to_rgb,
    _lch_to_rgb,
    _on_colour,
    _rgb_to_hex,
    _rgb_to_lch,
    load_settings,
    save_settings,
    scheme_from_roles,
    scheme_from_seed,
    scheme_from_settings,
    settings_for,
)


def relative_luminance(colour: str) -> float:
    """WCAG relative luminance, for contrast assertions."""
    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb(colour)
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)


def contrast(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


# ------------------------------------------------------------- colour maths


def test_colour_round_trips_through_lch():
    """The conversion has to be exact in both directions, or a seed drifts."""
    for colour in ("#FFFFFF", "#000000", "#6750A4", "#F44336", "#8BC34A", "#00BCD4"):
        rgb = _hex_to_rgb(colour)
        lightness, chroma, hue = _rgb_to_lch(rgb)
        back = _lch_to_rgb(lightness, chroma, hue)
        assert _rgb_to_hex(tuple(max(0.0, min(1.0, c)) for c in back)) == colour


def test_a_grey_seed_does_not_produce_a_random_hue():
    """A colourless seed has no hue, and must not invent one.

    ``atan2(0, 0)`` is 0, so a naive implementation turns a black seed into a
    *red* scheme. Measured before the guard was added: black and white both
    produced `#834F5F`, a dusty pink. Falling back to the baseline hue is the
    honest answer -- the surfaces still follow the seed's lightness, and the
    accent is deliberate rather than an accident of floating-point arithmetic.
    """
    grey_accent = scheme_from_seed("#808080", False)["accent"]
    for colourless in ("#000000", "#FFFFFF", "#404040", "#808080"):
        assert scheme_from_seed(colourless, False)["accent"] == grey_accent

    # And the fallback must be a neutral-ish tone, not a saturated hue.
    r, g, b = _hex_to_rgb(grey_accent)
    lightness, chroma, _hue = _rgb_to_lch((r, g, b))
    assert chroma < 30, f"灰色种子给出了过于鲜艳的 accent（chroma={chroma:.1f}）"


def test_a_grey_seed_still_varies_the_surfaces():
    """The seed's lightness has to keep mattering even when its hue does not."""
    dark_seed = scheme_from_seed("#202020", False)
    light_seed = scheme_from_seed("#E0E0E0", False)
    # Both fall back to the same accent family, but the scheme is still a
    # valid, complete one rather than degenerating to a blank table.
    for scheme in (dark_seed, light_seed):
        assert len(scheme.roles) == len(scheme_from_seed(DEFAULT_SEED, False).roles)


def test_a_seed_reappears_as_a_tone_at_its_own_lightness():
    """The tonal palette must contain the seed, not merely something near it.

    If it did not, picking a colour would give a scheme that never shows that
    colour anywhere -- the user would reasonably call that broken.
    """
    seed = "#2196F3"
    lightness, chroma, hue = _rgb_to_lch(_hex_to_rgb(seed))
    tone = ui_theme._tonal(lightness, chroma, hue)
    assert contrast(tone, seed) < 1.05, f"{tone} 与种子 {seed} 差太多"


def test_extreme_lightness_stays_in_gamut():
    """No channel may clip, at any tone or hue.

    A clipped channel is what turns a pale tone into a saturated sliver or a
    dark one into mud, so every tone the scheme reads must be reachable in sRGB.
    """
    for hue in range(0, 360, 15):
        for lightness in range(0, 101, 5):
            colour = ui_theme._tonal(lightness, 120.0, hue)
            r, g, b = _hex_to_rgb(colour)
            assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0


def test_on_colour_picks_the_readable_side():
    """Text on a filled surface has to stay readable on both extremes."""
    for background in ("#FFFFFF", "#000000", "#FFEB3B", "#FFC107", "#6750A4"):
        on = _on_colour(background)
        assert on in ("#000000", "#FFFFFF")
        assert contrast(on, background) >= 4.5, f"{on} 在 {background} 上读不清"


# ------------------------------------------------------------------ scheme


def test_the_default_scheme_is_material_baseline_purple():
    """A fresh install has to look like Material 3, not like a guess."""
    scheme = scheme_from_seed(DEFAULT_SEED, dark=False)
    assert scheme["accent"] == "#6750A4"
    assert scheme["panel"] == "#FBF8FF"
    assert scheme["text"] == "#1D1B20"


def test_every_role_is_a_valid_colour():
    for dark in (False, True):
        scheme = scheme_from_seed(DEFAULT_SEED, dark)
        for role, value in scheme.roles.items():
            assert value.startswith("#") and len(value) == 7, f"{role}={value}"
            _hex_to_rgb(value)  # raises on a malformed value


def test_dark_and_light_differ_in_the_expected_direction():
    light = scheme_from_seed(DEFAULT_SEED, False)
    dark = scheme_from_seed(DEFAULT_SEED, True)
    assert relative_luminance(dark["window"]) < relative_luminance(light["window"])
    assert relative_luminance(dark["text"]) > relative_luminance(light["text"])


def test_body_text_has_enough_contrast_on_its_surface():
    """The one accessibility rule worth pinning: the body copy is readable.

    Checked on every preset in both modes, because a pale seed (yellow, light
    green) is exactly where a naive generator produces a text colour that
    disappears into the panel.
    """
    for _name, seed in PRESETS:
        for dark in (False, True):
            scheme = scheme_from_seed(seed, dark)
            ratio = contrast(scheme["text"], scheme["panel"])
            assert ratio >= 7.0, f"{seed} dark={dark}: 正文对比度只有 {ratio:.1f}"


def test_a_valid_preset_is_offered_for_the_default_seed():
    """The default must be selectable, or a fresh install shows "custom"."""
    baseline = scheme_from_seed(DEFAULT_SEED, False)
    matching = [
        name for name, colour in PRESETS
        if scheme_from_seed(colour, False).roles == baseline.roles
    ]
    assert matching, "默认配色不在预设里，设置面板会显示成自定义"


def test_presets_are_all_distinct():
    """Two swatches that produce the same scheme is a dead chip."""
    seen: dict[tuple, str] = {}
    for name, colour in PRESETS:
        key = tuple(sorted(scheme_from_seed(colour, False).roles.items()))
        assert key not in seen, f"{name} 与 {seen.get(key)} 生成的颜色完全一样"
        seen[key] = name


def test_every_preset_generates_a_usable_scheme():
    for name, colour in PRESETS:
        for dark in (False, True):
            scheme = scheme_from_seed(colour, dark)
            assert scheme.seed == colour.upper() or scheme.seed == colour
            assert scheme["accent"] != scheme["panel"], name


def test_a_malformed_seed_is_rejected_loudly():
    """Falling back silently would hide a typo in the preset table."""
    for bad in ("", "not-a-colour", "#12345", "#GGGGGG"):
        with pytest.raises(ValueError):
            scheme_from_seed(bad, False)


def test_a_lowercase_seed_is_normalised():
    assert scheme_from_seed("#6750a4", False).seed == "#6750A4"


# ---------------------------------------------------------------- overrides


def test_hand_edits_win_over_the_generated_value():
    scheme = scheme_from_seed(DEFAULT_SEED, False)
    edited = scheme_from_roles({"accent": "#123456"}, seed=DEFAULT_SEED)
    assert edited["accent"] == "#123456"
    assert edited.custom is True
    # Everything not named keeps the generated value.
    assert edited["panel"] == scheme["panel"]


def test_an_unknown_role_is_ignored_rather_than_raising():
    """A file written by a newer build must not stop this one from starting."""
    edited = scheme_from_roles({"no_such_role": "#123456"}, seed=DEFAULT_SEED)
    assert "no_such_role" not in edited.roles


def test_a_bad_override_falls_back_to_the_generated_colour():
    edited = scheme_from_roles({"accent": "nonsense"}, seed=DEFAULT_SEED)
    assert edited["accent"] == scheme_from_seed(DEFAULT_SEED, False)["accent"]


# ------------------------------------------------------------------ storage


def test_the_scheme_survives_a_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_theme, "settings_path", lambda: tmp_path / "settings.json")
    scheme = scheme_from_seed("#00695C", dark=True)
    assert save_settings(settings_for(scheme)) is True

    restored = scheme_from_settings()
    assert restored.seed == scheme.seed
    assert restored.dark is True
    assert restored.roles == scheme.roles


def test_a_customised_scheme_survives_a_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_theme, "settings_path", lambda: tmp_path / "settings.json")
    edited = scheme_from_roles(
        {"accent": "#AB12CD", "panel": "#F0F0F0"}, seed="#00695C"
    )
    save_settings(settings_for(edited))
    restored = scheme_from_settings()
    assert restored.custom is True
    assert restored["accent"] == "#AB12CD"
    assert restored["panel"] == "#F0F0F0"


def test_only_overrides_are_stored_not_the_whole_table(tmp_path, monkeypatch):
    """Storing all roles would freeze this build's maths into the file.

    A later build that improves the generator should improve every preset the
    user has not hand-edited.
    """
    monkeypatch.setattr(ui_theme, "settings_path", lambda: tmp_path / "settings.json")
    generated = scheme_from_seed(DEFAULT_SEED, False)
    saved = settings_for(generated)
    assert saved["roles"] == {}, "未自定义时不应把整张颜色表写进文件"


def test_a_corrupt_settings_file_falls_back_to_the_default(tmp_path, monkeypatch):
    """Starting is more important than reporting a broken preference file."""
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(ui_theme, "settings_path", lambda: path)
    scheme = scheme_from_settings()
    assert scheme["accent"] == scheme_from_seed(DEFAULT_SEED, False)["accent"]


def test_a_settings_file_with_an_unknown_seed_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text('{"seed": "not-a-colour", "dark": true}', encoding="utf-8")
    monkeypatch.setattr(ui_theme, "settings_path", lambda: path)
    scheme = scheme_from_settings()
    assert scheme.seed == DEFAULT_SEED
    assert scheme.dark is True


def test_a_missing_settings_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_theme, "settings_path", lambda: tmp_path / "nope.json")
    assert load_settings() == {}
    assert scheme_from_settings().seed == DEFAULT_SEED


# ---------------------------------------------------------------- the editor


def test_every_editable_role_is_a_real_role():
    """A label in the advanced editor must name a role the scheme has.

    Otherwise the editor offers a control that writes a key nothing reads.
    """
    scheme = scheme_from_seed(DEFAULT_SEED, False)
    for role, label in ROLE_LABELS:
        assert role in scheme.roles, f"{role}（{label}）不是配色里的角色"
        assert label


def test_the_advanced_editor_covers_the_prominent_roles():
    """The roles a user would want to change are all reachable."""
    names = {role for role, _ in ROLE_LABELS}
    for required in ("window", "panel", "text", "accent", "border", "muted"):
        assert required in names
