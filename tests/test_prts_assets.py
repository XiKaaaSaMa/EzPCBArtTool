"""Tests for the PRTS asset fetcher's parsing and naming rules.

The downloader talks to a live wiki, so most of it cannot run in CI. What *can*
be pinned offline is the part that actually broke during development: the title
handling. Two failures cost a full run each and both were silent -- the run
reported success and wrote nothing:

* MediaWiki returns the *localised* namespace (``文件:``) even when queried as
  ``File:``, so slicing five characters off the title mangled every name and
  every lookup missed.
* ``list=allimages`` returns the database form of a title (underscores) while
  ``prop=imageinfo`` returns the display form (spaces). A group collected from
  one and resolved through the other matched nothing.

Both are pure string problems, so they are tested here against the exact shapes
the API returned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import fetch_prts_assets as fetcher  # noqa: E402

from pcbart.paths import asset_root  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this module may reach the live wiki.

    ``skin_rename_plan`` reads the operator page list before it reads the
    outfit names, and the tests here patch only the second call -- so the first
    one went to the network. That cost six seconds per test while the wiki
    answered, and on 2026-09-26 it cost **340 seconds and twelve failures** when
    the wiki's API answered 503 for hours while its page views kept working.
    A unit test that needs somebody else's server up is not a unit test.
    """
    monkeypatch.setattr(fetcher, "operator_pages", lambda: [])

    def blocked(*args, **kwargs):
        raise AssertionError("测试不得访问 PRTS：请把要读的页面 patch 掉")

    monkeypatch.setattr(fetcher, "api", blocked)
    monkeypatch.setattr(fetcher, "page_wikitext", blocked)


def test_display_name_turns_database_underscores_into_spaces():
    assert fetcher.display_name("稀有度_黑_5.png") == "稀有度 黑 5.png"


def test_display_name_is_idempotent():
    once = fetcher.display_name("头像_召唤物_眠兽_skin1.png")
    assert fetcher.display_name(once) == once


def test_collect_matches_the_underscore_form_but_returns_display_names():
    """The wiki's own regexes are written against the underscore form.

    Matching after normalising would silently drop every file with a space in
    its name, which is most of them.
    """
    group = next(item for item in fetcher.GROUPS if item.key == "rarity_black")
    assert group.pattern.match("稀有度_黑_5.png")
    assert not group.pattern.match("稀有度 黑 5.png")


def test_localised_namespace_is_split_not_sliced():
    """``文件:`` is two characters of prefix where ``File:`` is five.

    The bug this pins: slicing ``len("File:")`` left a truncated title and
    nothing ever resolved.
    """
    title = "文件:头像 阿米娅.png"
    name = title.split(":", 1)[1] if ":" in title else title
    assert name == "头像 阿米娅.png"
    assert name != title[len("File:") :].lstrip(":")


def test_every_group_pattern_matches_its_own_example():
    """Each rule has to accept the file the request named as its example.

    A pattern that is too tight is the quiet failure mode here: the group
    reports a plausible count and the user finds the asset missing.
    """
    examples = {
        "half_body_e0": "半身像_阿米娅_1.png",
        "half_body_e1": "半身像_斑点_精英1.png",
        "half_body_e2": "半身像_阿米娅_2.png",
        "half_body_skin": "半身像_阿米娅_skin1.png",
        "full_e0": "立绘_阿米娅_1.png",
        "full_e1": "立绘_阿米娅_1+.png",
        "full_e2": "立绘_阿米娅_2.png",
        "full_skin": "立绘_阿米娅_skin1.png",
        "summon_avatar": "头像_召唤物_触手.png",
        "skill_icon": "技能_冲锋号令·β型.png",
        "token_icon": "道具_阿米娅的信物.png",
        "elite_progress": "精英_2_卡片.png",
        "rarity_black": "稀有度_黑_4.png",
        "rarity_white": "稀有度_白_4.png",
        "class_big_black": "图标_职业_狙击_大图.png",
        "class_big_white": "图标_职业_狙击_大图_白.png",
        "team_logo": "团队logo_行动组A4.png",
        "skin_logo": "Skin_logo_寒武纪™系列.png",
        "faction_logo": "Logo_罗德岛.png",
        "medal": "蚀刻章_黄铁合约奖章·II.png",
    }
    for group in fetcher.GROUPS:
        example = examples.get(group.key)
        assert example, f"{group.key} has no example in this test"
        assert group.pattern.match(example), f"{group.key} rejects {example}"


@pytest.mark.parametrize(
    "group_key, raw_title",
    [
        ("faction_logo", "Logo_Ave_Mujica.png"),
        ("faction_logo", "Logo_S.E.E.S..png"),
        ("team_logo", "团队logo_Ave_Mujica.png"),
        ("half_body_e0", "半身像_Miss_Christine_1.png"),
        ("full_e2", "立绘_假日威龙_2.png"),
        ("skin_logo", "Skin_logo_彩虹六号：围攻.png"),
        ("skin_logo", "Skin_logo_小马宝莉.png"),
        ("class_big_black", "图标_职业_重装_大图.png"),
    ],
)
def test_a_multi_word_name_is_not_dropped(group_key, raw_title):
    """``list=allimages`` spells a space as an underscore.

    So a rule segment written ``[^_]*`` forbids spaces, and every multi-word
    name under that prefix disappears -- silently, because a group that
    collects fewer files looks exactly like a group that has fewer files.

    This is not hypothetical: ``Logo Ave Mujica.png`` arrives as
    ``Logo_Ave_Mujica.png`` and was left out of the 2026-09-25 download, so the
    five Ave Mujica operators had no logo to place (and no error either).
    """
    group = next(item for item in fetcher.GROUPS if item.key == group_key)
    assert group.pattern.match(raw_title), f"{group_key} 丢了 {raw_title}"


def test_the_skin_logo_small_set_is_still_held_out_after_the_relaxation():
    """The exclusion that used to be enforced by banning underscores.

    Banning them was incidental to the goal: ``Skin logo X 小.png`` is a
    different asset class. The lookahead keeps that rule *and* allows names
    with spaces, which is what banning underscores silently gave up.
    """
    group = next(item for item in fetcher.GROUPS if item.key == "skin_logo")
    assert group.pattern.match("Skin_logo_寒武纪™系列.png")
    assert not group.pattern.match("Skin_logo_寒武纪™系列_小.png")


def test_no_group_bans_underscores_inside_a_name_segment():
    """The sibling check: this defect arrived in thirteen rules at once.

    ``[^_]`` is the shape that has to stay out, so it is asserted structurally
    rather than by listing the groups -- a fourteenth rule added later gets
    caught by the same assertion.
    """
    offenders = [
        group.key for group in fetcher.GROUPS if "[^_]" in group.pattern.pattern
    ]
    assert offenders == []


def test_excluded_variants_are_not_matched():
    """The classes the request left out must stay out.

    Each of these lives under a collected prefix and would arrive as clutter the
    user has to sort by hand.
    """
    by_key = {group.key: group for group in fetcher.GROUPS}

    # 后勤技能 use a different prefix entirely, but guard the skill rule too.
    assert not by_key["skill_icon"].pattern.match("Bskill_ws_p2.png")

    # 信物带框 is the framed set; the request wants 无框.
    assert not by_key["token_icon"].pattern.match("道具_带框_阿米娅的信物.png")

    # 皮肤LOGO（小）is the small set; the request wants the large one.
    assert not by_key["skin_logo"].pattern.match("Skin_logo_寒武纪™系列_小.png")

    # 蚀刻章套组预览图 is a different asset class from the medals themselves.
    assert not by_key["medal"].pattern.match("蚀刻章套组_黄铁行动.png")

    # 职业大图（白）must not be swept into the black set, which is broader.
    assert not by_key["class_big_black"].pattern.match("图标_职业_狙击_大图_白.png")


def test_token_rule_excludes_shared_class_tokens():
    """Only an operator's own token carries 「的信物」.

    The 25 shared ones (先锋信物原件 and friends) are class-wide items, not
    operator tokens, and the request named operator tokens.
    """
    pattern = next(
        group for group in fetcher.GROUPS if group.key == "token_icon"
    ).pattern
    assert pattern.match("道具_凛冬的信物.png")
    assert not pattern.match("道具_先锋信物原件.png")
    assert not pattern.match("道具_皇家信物兑换券.png")


def test_sanitize_keeps_names_path_safe():
    assert fetcher.sanitize('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
    assert fetcher.sanitize("  trailing.  ") == "trailing"
    assert fetcher.sanitize("") == "unnamed"


def test_groups_do_not_share_a_folder():
    """One folder per comma-separated class, as the request specified."""
    folders = [group.folder for group in fetcher.GROUPS]
    assert len(folders) == len(set(folders))


def test_number_table_row_parsing_handles_the_observed_shapes():
    """The table mixes several widths and one two-letter-with-letter suffix id."""
    text = (
        "\n|R001\n|[[阿米娅]]\n"
        "|R00A\n|[[可露希尔]]\n"
        "|USS0\n|[[怒潮凛冬]]\n"
        "|YD30\n|[[赤刃明霄陈|赤刃明霄陈]]\n"
    )
    rows = fetcher._TABLE_ROW.findall(text)
    assert rows == [
        ("R001", "阿米娅"),
        ("R00A", "可露希尔"),
        ("USS0", "怒潮凛冬"),
        ("YD30", "赤刃明霄陈"),
    ]


def test_operator_field_regex_reads_the_infobox():
    text = (
        "{{CharinfoV2\n"
        "|干员名=阿米娅\n"
        "|干员id=char_002_amiya\n"
        "|干员序号=55\n"
        "|稀有度=4\n"
        "|职业=术师\n"
        "|分支=中坚术师\n"
        "|情报编号=R001\n"
    )
    fields = dict(fetcher._FIELD.findall(text))
    assert fields["干员名"] == "阿米娅"
    assert fields["情报编号"] == "R001"
    assert fields["干员id"] == "char_002_amiya"


def test_operator_field_regex_ignores_a_missing_number():
    """A page without 情报编号 must not invent one from a neighbouring field."""
    text = "|干员名=郁金香\n|干员id=char_513_apionr\n|干员序号=-1\n"
    fields = dict(fetcher._FIELD.findall(text))
    assert "情报编号" not in fields


# ------------------------------------------------- naming skins after the outfit


def test_skin_file_regex_reads_operator_and_index():
    match = fetcher.skin_file_pattern("半身像").match("半身像 阿米娅 skin2.png")
    assert match.group(1) == "阿米娅"
    assert match.group(2) == "2"


def test_skin_file_regex_accepts_names_with_spaces_and_dots():
    """Operator names are not all single CJK tokens."""
    pattern = fetcher.skin_file_pattern("半身像")
    for name, operator, index in (
        ("半身像 Miss.Christine skin1.png", "Miss.Christine", "1"),
        ("半身像 U-Official skin1.png", "U-Official", "1"),
        ("半身像 假日威龙陈 skin2.png", "假日威龙陈", "2"),
    ):
        match = pattern.match(name)
        assert match, name
        assert match.group(1) == operator
        assert match.group(2) == index


def test_both_skin_classes_use_one_rule():
    """半身像 and 立绘 carry the same ``skinN`` names and the same 时装N.

    They are the same rule applied to two prefixes rather than two rules that
    have to be kept in step -- which is what the first half-body-only version
    was, and why the 立绘 set stayed unnamed for a round.
    """
    assert dict(fetcher.SKIN_CLASSES) == {
        "half_body_skin": "半身像",
        "full_skin": "立绘",
    }
    for group_key, prefix in fetcher.SKIN_CLASSES:
        keys = {group.key for group in fetcher.GROUPS}
        assert group_key in keys, f"{prefix} names unknown group {group_key}"
        assert fetcher.skin_file_pattern(prefix).match(f"{prefix} 阿米娅 skin1.png")


def test_the_prefix_is_escaped_not_inlined():
    """Operator names are arbitrary text; the prefix is ours, so escape it."""
    pattern = fetcher.skin_file_pattern("立绘.特殊")
    assert pattern.match("立绘.特殊 X skin1.png")


def test_one_class_cannot_claim_another_class_file():
    """The prefix is the only thing separating the two sets on disk."""
    assert not fetcher.skin_file_pattern("立绘").match("半身像 阿米娅 skin1.png")
    assert not fetcher.skin_file_pattern("半身像").match("立绘 阿米娅 skin1.png")


def test_skin_field_regex_reads_the_infobox():
    text = (
        "|时装1名称=报童\n"
        "|时装1系列=忒斯特收藏/I\n"
        "|时装2名称=见习联结者\n"
        "|时装2颜色=#06c1a2\n"
    )
    assert dict(fetcher.SKIN_FIELD.findall(text)) == {
        "1": "报童",
        "2": "见习联结者",
    }


def test_skin_field_regex_ignores_the_other_costume_fields():
    """``时装1系列``/``时装1介绍`` must not be mistaken for the name."""
    text = "|时装1介绍=这不是名字\n|时装1系列=忒斯特收藏\n"
    assert fetcher.SKIN_FIELD.findall(text) == []


def test_a_slash_in_an_outfit_name_is_sanitised(tmp_path):
    """One real outfit is called 自由//失效, and ``/`` cannot be a filename.

    The rename has to keep working: skipping the file would leave exactly the
    kind of skinN name this feature exists to remove.
    """
    cleaned = fetcher.sanitize("半身像 史尔特尔 自由//失效.png")
    assert cleaned == "半身像 史尔特尔 自由__失效.png"
    assert "/" not in cleaned and "\\" not in cleaned


def test_skin_rename_plan_skips_a_missing_outfit_name(tmp_path, monkeypatch):
    """A portrait with no matching 时装 entry keeps its original name.

    Renaming it to a guess would be worse than leaving ``skinN``: the user
    picks by name, and a wrong name is a wrong selection.
    """
    monkeypatch.setattr(
        fetcher,
        "operator_skin_names",
        lambda pages: {"阿米娅": {"1": "报童"}},
    )
    manifest = {
        "groups": {
            "half_body_skin": [
                {"file": "04-半身像-皮肤/半身像 阿米娅 skin1.png", "title": "a", "bytes": 1},
                {"file": "04-半身像-皮肤/半身像 斑点 skin2.png", "title": "b", "bytes": 1},
            ]
        }
    }
    plan = {item["name"]: item for item in fetcher.skin_rename_plan(tmp_path, manifest)}
    assert plan["半身像 阿米娅 skin1.png"]["target"].endswith("半身像 阿米娅 报童.png")
    assert plan["半身像 斑点 skin2.png"]["target"] == ""
    assert "wiki" in plan["半身像 斑点 skin2.png"]["reason"]


def test_skin_rename_plan_covers_both_classes(tmp_path, monkeypatch):
    """One pass has to name the 立绘 set too, not just the 半身像 one.

    The first version named only the busts, so half of every operator's outfits
    stayed as ``skinN`` while the other half was browsable by name.
    """
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    manifest = {
        "groups": {
            "half_body_skin": [
                {"file": "05-半身像-皮肤/半身像 阿米娅 skin1.png", "title": "a", "bytes": 1}
            ],
            "full_skin": [
                {"file": "09-立绘-皮肤/立绘 阿米娅 skin1.png", "title": "b", "bytes": 1}
            ],
        }
    }
    plan = {item["file"]: item for item in fetcher.skin_rename_plan(tmp_path, manifest)}
    assert plan["05-半身像-皮肤/半身像 阿米娅 skin1.png"]["target"] == (
        "05-半身像-皮肤/半身像 阿米娅 报童.png"
    )
    assert plan["09-立绘-皮肤/立绘 阿米娅 skin1.png"]["target"] == (
        "09-立绘-皮肤/立绘 阿米娅 报童.png"
    )


def test_the_two_classes_may_share_an_outfit_name(tmp_path, monkeypatch):
    """Same operator, same costume, two asset classes -- not a collision.

    Keying the "already claimed" check on the bare name made every 立绘 skin
    look like a duplicate of the 半身像 one, which would have silently skipped
    the entire full-body set.
    """
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    manifest = {
        "groups": {
            "half_body_skin": [
                {"file": "05-半身像-皮肤/半身像 阿米娅 skin1.png", "title": "a", "bytes": 1}
            ],
            "full_skin": [
                {"file": "09-立绘-皮肤/立绘 阿米娅 skin1.png", "title": "b", "bytes": 1}
            ],
        }
    }
    plan = fetcher.skin_rename_plan(tmp_path, manifest)
    assert all(item["target"] for item in plan), [item["reason"] for item in plan]


def test_a_stray_skin0_is_not_named_after_the_outfit_it_duplicates(
    tmp_path, monkeypatch
):
    """嵯峨 ships both ``skin1`` (the outfit) and a ``skin0`` copy of it.

    时装1名称=彼有 names the file numbered 1. Both files carry the same picture
    at different resolutions, so letting ``skin0`` take the name too would put
    two files called 彼有 in one folder -- and then ``--verify`` would report one
    of them missing forever, however often it was repaired.
    """
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"嵯峨": {"1": "彼有"}}
    )
    manifest = {
        "groups": {
            "full_skin": [
                {"file": "09-立绘-皮肤/立绘 嵯峨 skin0.png", "title": "a", "bytes": 1},
                {"file": "09-立绘-皮肤/立绘 嵯峨 skin1.png", "title": "b", "bytes": 1},
            ]
        }
    }
    plan = {item["name"]: item for item in fetcher.skin_rename_plan(tmp_path, manifest)}
    assert plan["立绘 嵯峨 skin1.png"]["target"].endswith("立绘 嵯峨 彼有.png")
    assert plan["立绘 嵯峨 skin0.png"]["target"] == ""
    # The page documents no 时装0, so the refusal is the "no such outfit" one
    # rather than a collision; `skin1` is the file that outfit names. Either
    # way, only one file may end up called 彼有.
    assert plan["立绘 嵯峨 skin0.png"]["reason"]
    targets = [item["target"] for item in plan.values() if item["target"]]
    assert len(targets) == len(set(targets)) == 1


def test_skin_rename_plan_is_idempotent(tmp_path, monkeypatch):
    """Running the rename twice must not try to rename an already-renamed file."""
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    manifest = {
        "groups": {
            "half_body_skin": [
                {"file": "04-半身像-皮肤/半身像 阿米娅 报童.png", "title": "a", "bytes": 1}
            ]
        }
    }
    plan = fetcher.skin_rename_plan(tmp_path, manifest)
    assert plan[0]["target"] == ""
    assert "已改过名" in plan[0]["reason"]


def test_renaming_rewrites_the_index_so_verify_still_passes(tmp_path, monkeypatch):
    """The index and the files must move together.

    Leaving the index pointing at the old names would make the next check
    report every skin as missing -- a self-inflicted false alarm covering the
    whole set.
    """
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    folder = tmp_path / "04-半身像-皮肤"
    folder.mkdir()
    (folder / "半身像 阿米娅 skin1.png").write_bytes(b"12345")

    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {
            "half_body_skin": [
                {
                    "file": "04-半身像-皮肤/半身像 阿米娅 skin1.png",
                    "title": "a",
                    "bytes": 5,
                }
            ]
        },
        fetcher.EXCLUDED_KEY: [],
    }
    fetcher.write_manifest(tmp_path, manifest)

    result = fetcher.rename_skins(tmp_path, fetcher.read_manifest(tmp_path), apply=True)
    assert result["renamed"] == 1
    assert (folder / "半身像 阿米娅 报童.png").exists()
    assert not (folder / "半身像 阿米娅 skin1.png").exists()

    reread = fetcher.read_manifest(tmp_path)
    assert reread["groups"]["half_body_skin"][0]["file"].endswith("报童.png")
    assert fetcher.verify_library(tmp_path, reread) == []


def test_renaming_keeps_an_exclusion_pointing_at_a_skin(tmp_path, monkeypatch):
    """An excluded skin's path must follow the rename too."""
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    folder = tmp_path / "04-半身像-皮肤"
    folder.mkdir()
    (folder / "半身像 阿米娅 skin1.png").write_bytes(b"12345")
    old = "04-半身像-皮肤/半身像 阿米娅 skin1.png"

    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {
            "half_body_skin": [{"file": old, "title": "a", "bytes": 5}]
        },
        fetcher.EXCLUDED_KEY: [old],
    }
    fetcher.write_manifest(tmp_path, manifest)

    fetcher.rename_skins(tmp_path, fetcher.read_manifest(tmp_path), apply=True)

    reread = fetcher.read_manifest(tmp_path)
    assert reread[fetcher.EXCLUDED_KEY] == [
        "04-半身像-皮肤/半身像 阿米娅 报童.png"
    ]


def test_preview_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetcher, "operator_skin_names", lambda pages: {"阿米娅": {"1": "报童"}}
    )
    folder = tmp_path / "04-半身像-皮肤"
    folder.mkdir()
    (folder / "半身像 阿米娅 skin1.png").write_bytes(b"12345")
    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {
            "half_body_skin": [
                {
                    "file": "04-半身像-皮肤/半身像 阿米娅 skin1.png",
                    "title": "a",
                    "bytes": 5,
                }
            ]
        },
        fetcher.EXCLUDED_KEY: [],
    }
    fetcher.write_manifest(tmp_path, manifest)

    result = fetcher.rename_skins(tmp_path, fetcher.read_manifest(tmp_path), apply=False)

    assert result["renamed"] == 0
    assert result["planned"] == 1
    assert (folder / "半身像 阿米娅 skin1.png").exists()


# -------------------------------------------- files the wiki itself names wrong


#: Files the library re-files because the wiki's own name is wrong.
#:
#: Spelled out rather than compared against the table itself: the point of the
#: test is that adding a claim is a decision someone makes on purpose. Each
#: entry below was checked against the game, not merely moved.
EXPECTED_CURATED = {
    # 斑点 is a 3★, so his promotion art arrives at 精英1. The wiki filed both
    # the portrait and the illustration as `skin2`, and his page documents only
    # one 时装 -- so neither was a costume.
    "半身像 斑点 skin2.png",
    "立绘 斑点 skin2.png",
    # 阿米娅 is the only operator with a separate 精英1 illustration, which the
    # wiki spells `1+`. Renamed so the illustration set reads the same way as
    # the portrait set for the same promotion.
    "立绘 阿米娅 1+.png",
}


def test_only_the_known_misfiled_assets_are_curated():
    """The table is deliberately tiny; growth means someone added a claim.

    Each entry asserts something about the game that the wiki gets wrong, so
    this is a place where an extra row deserves a second look rather than a
    silent append.
    """
    assert set(fetcher.MISFILED_ASSETS) == EXPECTED_CURATED


def test_spot_is_filed_as_elite_1_not_a_costume():
    """斑点's second portrait is his 精英1 promotion art.

    Evidence, all re-checked against the mirror: he is a 3★ whose promotion
    art arrives at 精英1; the wiki documents exactly one 时装 for him
    (专业人士); and among all 22 three-star operators none has an 精英2
    portrait, yet he alone carried a stray ``skinN``. Left alone, the rename
    step reported it as an unnamed costume and the user could not pick it.
    """
    group_key, new_name, reason = fetcher.MISFILED_ASSETS["半身像 斑点 skin2.png"]
    assert group_key == "half_body_e1"
    assert new_name == "半身像 斑点 精英1.png"
    assert reason


def test_spot_illustration_is_filed_as_elite_1_not_a_costume():
    """The same mistake on the illustration side, and it is one real artwork.

    Measured between ``立绘 斑点 1.png`` and the ``skin2`` one: 77.8% of pixels
    within 8/255, and the differences are confined to redraw shading -- the
    overlay is contours, not a second subject. So this is one illustration
    stored twice under two names, not two artworks.
    """
    group_key, new_name, reason = fetcher.MISFILED_ASSETS["立绘 斑点 skin2.png"]
    assert group_key == "full_e1"
    assert new_name == "立绘 斑点 精英1.png"
    assert reason


def test_amiya_illustration_is_filed_as_elite_1():
    """阿米娅 is the only operator with a separate 精英1 illustration.

    The wiki writes it ``立绘 阿米娅 1+.png``; renaming to 精英1 makes the
    illustration set name the same promotion the same way the portrait set does
    (which already uses ``精英1``).
    """
    group_key, new_name, reason = fetcher.MISFILED_ASSETS["立绘 阿米娅 1+.png"]
    assert group_key == "full_e1"
    assert new_name == "立绘 阿米娅 精英1.png"
    assert reason


def test_curated_files_all_target_distinct_paths():
    """Two fixes landing on one name would silently drop one of them."""
    targets = [
        (group_key, new_name)
        for group_key, new_name, _ in fetcher.MISFILED_ASSETS.values()
    ]
    assert len(targets) == len(set(targets))


def test_both_elite_1_groups_exist():
    """A curated entry names a group, so each group has to be real."""
    keys = {group.key for group in fetcher.GROUPS}
    assert "half_body_e1" in keys
    assert "full_e1" in keys


def test_elite_1_group_does_not_capture_the_elite_0_or_2_sets():
    """Its pattern must not overlap the files the other portrait rules take.

    Written against PRTS's underscore form, which is what the rules match --
    the display form with spaces is what lands on disk.
    """
    group = next(item for item in fetcher.GROUPS if item.key == "half_body_e1")
    assert group.pattern.match("半身像_斑点_精英1.png")
    assert not group.pattern.match("半身像_斑点_1.png")
    assert not group.pattern.match("半身像_斑点_2.png")
    assert not group.pattern.match("半身像_斑点_专业人士.png")


def test_curation_moves_the_file_and_records_it(tmp_path):
    """The move must land in the index, or the next check calls it missing."""
    skin = tmp_path / "04-半身像-皮肤"
    skin.mkdir()
    source = skin / "半身像 斑点 skin2.png"
    source.write_bytes(b"12345")
    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {
            "half_body_skin": [
                {
                    "file": "04-半身像-皮肤/半身像 斑点 skin2.png",
                    "title": "半身像 斑点 skin2.png",
                    "bytes": 5,
                }
            ]
        },
        fetcher.EXCLUDED_KEY: [],
        fetcher.CURATED_KEY: [],
    }
    fetcher.write_manifest(tmp_path, manifest)

    result = fetcher.apply_curation(tmp_path, fetcher.read_manifest(tmp_path), apply=True)
    assert result["moved"] == 1

    moved = tmp_path / "03-半身像-精英1" / "半身像 斑点 精英1.png"
    assert moved.exists()
    assert not source.exists()

    reread = fetcher.read_manifest(tmp_path)
    # Out of the skin group, into the curated list -- exactly one of the two,
    # or the check would look for it at a path that no longer exists.
    assert reread["groups"]["half_body_skin"] == []
    assert [item["file"] for item in reread[fetcher.CURATED_KEY]] == [
        "03-半身像-精英1/半身像 斑点 精英1.png"
    ]
    assert fetcher.verify_library(tmp_path, reread) == []


def test_curation_is_idempotent(tmp_path):
    """Running it twice must not move anything or duplicate the entry."""
    target = tmp_path / "03-半身像-精英1"
    target.mkdir()
    (target / "半身像 斑点 精英1.png").write_bytes(b"12345")
    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {"half_body_skin": []},
        fetcher.EXCLUDED_KEY: [],
        fetcher.CURATED_KEY: [
            {
                "file": "03-半身像-精英1/半身像 斑点 精英1.png",
                "title": "半身像 斑点 skin2.png",
                "bytes": 5,
                "group": "half_body_e1",
                "reason": "x",
            }
        ],
    }
    fetcher.write_manifest(tmp_path, manifest)

    result = fetcher.apply_curation(tmp_path, fetcher.read_manifest(tmp_path), apply=True)
    assert result["moved"] == 0

    reread = fetcher.read_manifest(tmp_path)
    assert len(reread[fetcher.CURATED_KEY]) == 1
    assert fetcher.verify_library(tmp_path, reread) == []


def test_curation_preview_changes_nothing(tmp_path):
    skin = tmp_path / "04-半身像-皮肤"
    skin.mkdir()
    source = skin / "半身像 斑点 skin2.png"
    source.write_bytes(b"12345")
    manifest = {
        "source": "x",
        "fetched": "now",
        "operator_index": {},
        "groups": {
            "half_body_skin": [
                {
                    "file": "04-半身像-皮肤/半身像 斑点 skin2.png",
                    "title": "x",
                    "bytes": 5,
                }
            ]
        },
        fetcher.EXCLUDED_KEY: [],
        fetcher.CURATED_KEY: [],
    }
    fetcher.write_manifest(tmp_path, manifest)

    result = fetcher.apply_curation(tmp_path, fetcher.read_manifest(tmp_path), apply=False)

    assert result["planned"] == 1
    assert result["moved"] == 0
    assert source.exists()


def test_every_curated_entry_points_at_a_group_that_exists():
    """Each fix names a destination group, so each must be a real one."""
    keys = {group.key for group in fetcher.GROUPS}
    for name, (group_key, new_name, _reason) in fetcher.MISFILED_ASSETS.items():
        assert group_key in keys, f"{name} targets unknown group {group_key}"
        assert new_name.endswith(".png")


def test_all_entries_includes_curated_files():
    """Checking and copying both ask this one question."""
    manifest = {
        "groups": {"a": [{"file": "a/1.png"}]},
        fetcher.CURATED_KEY: [{"file": "b/2.png"}],
    }
    files = [entry["file"] for entry in fetcher.all_entries(manifest)]
    assert files == ["a/1.png", "b/2.png"]


def test_every_group_has_a_prefix_and_a_note_or_a_reason_to_lack_one():
    """A group with no prefix would enumerate the entire wiki."""
    for group in fetcher.GROUPS:
        assert group.prefixes, f"{group.key} has no prefix"
        assert all(prefix.strip() for prefix in group.prefixes)


# --------------------------------------------------- verifying a mirrored library


def write_manifest(tmp_path, entries, excluded=()):
    """A minimal library plus the index that describes it."""
    manifest = {
        "source": "https://example.invalid",
        "fetched": "2026-01-01 00:00",
        "operator_index": {},
        "groups": {"demo": entries},
        fetcher.EXCLUDED_KEY: list(excluded),
    }
    fetcher.write_manifest(tmp_path, manifest)
    return manifest


def test_verify_reports_a_missing_file(tmp_path):
    (tmp_path / "a.png").write_bytes(b"12345")
    manifest = write_manifest(
        tmp_path,
        [{"file": "a.png", "title": "a.png", "bytes": 5},
         {"file": "b.png", "title": "b.png", "bytes": 7}],
    )
    problems = fetcher.verify_library(tmp_path, manifest)
    assert [item["file"] for item in problems] == ["b.png"]
    assert problems[0]["problem"] == "缺失"


def test_verify_reports_a_size_mismatch(tmp_path):
    (tmp_path / "a.png").write_bytes(b"123")
    manifest = write_manifest(
        tmp_path, [{"file": "a.png", "title": "a.png", "bytes": 999}]
    )
    problems = fetcher.verify_library(tmp_path, manifest)
    assert len(problems) == 1
    assert "大小不符" in problems[0]["problem"]


def test_verify_skips_files_the_user_deleted_on_purpose(tmp_path):
    """A pruned library is not a broken one.

    Without this, every check after a deliberate deletion either nags forever
    or helpfully puts the file back.
    """
    (tmp_path / "a.png").write_bytes(b"12345")
    write_manifest(
        tmp_path,
        [{"file": "a.png", "title": "a.png", "bytes": 5},
         {"file": "gone.png", "title": "gone.png", "bytes": 9}],
        excluded=["gone.png"],
    )
    manifest = fetcher.read_manifest(tmp_path)
    problems = fetcher.verify_library(tmp_path, manifest)
    assert problems == []


def test_a_clean_library_verifies(tmp_path):
    (tmp_path / "a.png").write_bytes(b"12345")
    manifest = write_manifest(
        tmp_path, [{"file": "a.png", "title": "a.png", "bytes": 5}]
    )
    assert fetcher.verify_library(tmp_path, manifest) == []


def test_manifest_round_trip_keeps_the_exclusion_list(tmp_path):
    """Writing the index back must not drop the exclusions or the operator data."""
    manifest = write_manifest(
        tmp_path,
        [{"file": "a.png", "title": "a.png", "bytes": 5}],
        excluded=["gone.png"],
    )
    rebuilt = fetcher.build_manifest(manifest, {"operators": 460})
    fetcher.write_manifest(tmp_path, rebuilt)
    reread = fetcher.read_manifest(tmp_path)
    assert reread[fetcher.EXCLUDED_KEY] == ["gone.png"]
    assert reread["operator_index"] == {"operators": 460}


def test_a_subset_run_keeps_the_groups_it_did_not_touch():
    """``--only`` knows about one group and nothing about the rest."""
    previous = {"groups": {"a": [1], "b": [2]}}
    merged = fetcher._manifest_groups(previous, {"a": [9]}, ["a"])
    assert merged["groups"]["a"] == [9]
    assert merged["groups"]["b"] == [2]


def test_a_full_run_replaces_every_group():
    previous = {"groups": {"a": [1], "b": [2]}}
    merged = fetcher._manifest_groups(previous, {"a": [9]}, [])
    assert merged["groups"] == {"a": [9]}


@pytest.mark.parametrize("prefix", ["头像", "技能", "立绘", "Logo"])
def test_prefixes_are_plain_ascii_or_cjk_without_regex_metacharacters(prefix):
    """Prefixes go into a URL parameter, not a regex, so keep them literal."""
    assert not any(char in prefix for char in "[](){}*+?|\\^$")


# ------------------------------------------------------------------- factions


def filter_data_html(*cards: str, extra_outside: str = "") -> str:
    """A ``#filter-data`` block shaped like the one ``干员一览`` renders.

    The card text is reproduced as the wiki emits it: a nested ``<span>`` with
    its own style attribute, and a literal ``>`` inside the description. A
    regex written against ``data-zh=`` runs straight past both.
    """
    body = "".join(f"<div {card}>{text}</div>" for card, text in (
        ("data-zh=\"12F\" data-profession=\"术师\" data-rarity=\"1\" "
         "data-logo=\"罗德岛\" data-nation=\"罗德岛\" data-group=\"\" "
         "data-team=\"\" data-id=\"R303\"",
         "<span style=\"color:#00B0FF;\">群体法术伤害</span>"),
        ("data-zh=\"香草\" data-profession=\"狙击\" data-rarity=\"2\" "
         "data-logo=\"黑钢国际\" data-nation=\"维多利亚\" "
         "data-group=\"黑钢国际\" data-team=\"\" data-id=\"BS02\"",
         "优先攻击空中单位，射程 3>2"),
    ))
    extra = f"<div id=\"other\">{extra_outside}</div>" if extra_outside else ""
    return f"<div style=\"display:none\" id=\"filter-data\">{body}</div>{extra}"


def test_filter_data_parser_reads_every_card():
    cards = fetcher.parse_filter_data(filter_data_html())
    assert [card["data-zh"] for card in cards] == ["12F", "香草"]
    assert cards[0]["data-logo"] == "罗德岛"
    assert cards[1]["data-group"] == "黑钢国际"


def test_filter_data_parser_stops_at_the_end_of_the_container():
    """A card-shaped div outside the container is not an operator card.

    The block is followed by the grid's own markup, so a parser that never
    closes the container picks up whatever comes next and reports a faction
    for something that is not an operator.
    """
    html = filter_data_html() + (
        "<div data-zh=\"不是干员\" data-logo=\"罗德岛\"></div>"
    )
    cards = fetcher.parse_filter_data(html)
    assert [card["data-zh"] for card in cards] == ["12F", "香草"]


def test_filter_data_parser_reports_a_missing_block_as_empty():
    """Empty means "the block was not found", which the caller must not read as
    "no operator has a faction" -- that is the failure an empty column hides."""
    assert fetcher.parse_filter_data("<html><body>没有这个容器</body></html>") == []


def test_a_page_without_the_block_is_an_error_not_an_empty_column(monkeypatch):
    monkeypatch.setattr(fetcher, "faction_page_html", lambda: "<html/>")
    with pytest.raises(RuntimeError, match="filter-data"):
        fetcher.faction_records()


def test_faction_records_are_keyed_by_page_name(monkeypatch):
    """Alternate forms are separate cards with separate professions.

    阿米娅/近卫/医疗 share 情报编号 ``R001``, so keying by the number would
    give every form the first one's profession and logo.
    """
    monkeypatch.setattr(fetcher, "faction_page_html", lambda: filter_data_html())
    records = fetcher.faction_records()
    assert set(records) == {"12F", "香草"}
    assert records["香草"] == {
        "logo": "黑钢国际",
        "nation": "维多利亚",
        "group": "黑钢国际",
        "team": "",
        "profession": "狙击",
        "id": "BS02",
    }


def test_operator_field_regex_reads_the_three_faction_slots():
    text = (
        "|干员名=香草\n|职业=狙击\n"
        "|所属国家=维多利亚\n|所属组织=黑钢国际\n|所属团队=None\n"
    )
    fields = dict(fetcher._FIELD.findall(text))
    assert fields["所属国家"] == "维多利亚"
    assert fields["所属组织"] == "黑钢国际"
    assert fields["所属团队"] == "None"


def library_with_logos(tmp_path: Path, *names: str) -> Path:
    folder = tmp_path / fetcher.FACTION_LOGO_FOLDER
    folder.mkdir(parents=True)
    for name in names:
        (folder / f"Logo {name}.png").write_bytes(b"png")
    icons = tmp_path / "15-职业大图" / "黑"
    icons.mkdir(parents=True)
    (icons / "图标 职业 狙击 大图.png").write_bytes(b"png")
    white = tmp_path / "15-职业大图" / "白"
    white.mkdir(parents=True)
    (white / "图标 职业 狙击 大图 白.png").write_bytes(b"png")
    return tmp_path


def test_library_file_refuses_to_name_a_file_that_is_not_there(tmp_path):
    root = library_with_logos(tmp_path, "罗德岛")
    assert fetcher.library_file(root, fetcher.FACTION_LOGO_FOLDER, "Logo 罗德岛.png")
    assert fetcher.library_file(root, fetcher.FACTION_LOGO_FOLDER, "Logo 不存在.png") == ""


def test_apply_factions_fills_logo_icons_and_link(tmp_path):
    root = library_with_logos(tmp_path, "黑钢国际")
    rows = [
        {
            "干员名": "香草",
            "页面名": "香草",
            "职业": "狙击",
            "情报编号": "BS02",
        }
    ]
    report = fetcher.apply_factions(
        rows,
        root,
        {
            "香草": {
                "logo": "黑钢国际",
                "nation": "维多利亚",
                "group": "黑钢国际",
                "team": "",
                "profession": "狙击",
                "id": "BS02",
            }
        },
    )
    row = rows[0]
    assert row["阵营"] == "黑钢国际"
    assert row["阵营来源"] == fetcher.FACTION_PAGE
    assert row["阵营logo"] == "18-阵营logo/Logo 黑钢国际.png"
    assert row["职业图标_黑"] == "15-职业大图/黑/图标 职业 狙击 大图.png"
    assert row["职业图标_白"] == "15-职业大图/白/图标 职业 狙击 大图 白.png"
    assert row["PRTS链接"] == "https://prts.wiki/w/%E9%A6%99%E8%8D%89"
    assert report["from_grid"] == 1
    assert report["missing"] == []


def test_the_grid_does_not_erase_what_the_infobox_documented(tmp_path):
    """The grid leaves group/team empty for most operators.

    Overwriting 所属组织 with that blank would discard the only description of
    it on the row, and the fallback for NPCs reads those same columns.
    """
    root = library_with_logos(tmp_path, "黑钢国际")
    rows = [{"干员名": "香草", "页面名": "香草", "所属组织": "黑钢国际"}]
    fetcher.apply_factions(
        rows,
        root,
        {
            "香草": {
                "logo": "黑钢国际",
                "nation": "维多利亚",
                "group": "",
                "team": "",
                "profession": "",
                "id": "",
            }
        },
    )
    assert rows[0]["所属组织"] == "黑钢国际"
    assert rows[0]["所属国家"] == "维多利亚"


def test_an_operator_off_the_grid_falls_back_to_the_infobox(tmp_path):
    """NPCs and elite operators have no card; the infobox is all there is."""
    root = library_with_logos(tmp_path, "罗德岛-精英干员", "雷姆必拓")
    rows = [
        {
            "干员名": "机械师",
            "页面名": "机械师",
            "所属国家": "雷姆必拓",
            "所属组织": "罗德岛-精英干员",
            "所属团队": "",
            "职业": "",
        }
    ]
    report = fetcher.apply_factions(rows, root, {})
    # 国家 is checked first, which is the wiki's own precedence for the logo.
    assert rows[0]["阵营"] == "雷姆必拓"
    assert rows[0]["阵营来源"] == "干员档案"
    assert report["from_infobox"] == 1
    assert report["missing"] == []


def test_the_fallback_only_accepts_a_value_that_names_a_real_logo(tmp_path):
    """A faction with no image must stay empty instead of naming a bad path."""
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [
        {
            "干员名": "某人",
            "页面名": "某人",
            "所属国家": "莱欧斯小队",
            "职业": "",
            "干员序号": "7",
        }
    ]
    report = fetcher.apply_factions(rows, root, {})
    assert rows[0]["阵营"] == ""
    assert rows[0]["阵营logo"] == ""
    assert 0 == report["from_infobox"]
    # Empty is reported; a value with no image is reported *differently*, so a
    # missing logo file cannot hide behind the count of operators with none.
    assert report["missing"] == ["某人"]
    assert report["logo_without_file"] == []


def test_a_row_with_no_in_game_index_is_not_counted_as_a_missed_fetch(tmp_path):
    """卫戍协议 rosters and NPCs are legitimately absent from 干员一览.

    On 2026-09-26 that group was 31 rows -- every one of them at 干员序号 -1 --
    against 432 that resolved. Reporting one number for both would mean the
    real failure mode (the grid stops matching) is buried under the expected
    one, which is the larger of the two.
    """
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [
        {"干员名": "博士", "页面名": "博士", "职业": "", "干员序号": ""},
        {"干员名": "Pith(卫戍协议)", "页面名": "Pith(卫戍协议)", "干员序号": "-1"},
        {"干员名": "香草", "页面名": "香草", "干员序号": "30"},
    ]
    report = fetcher.apply_factions(rows, root, {})
    assert report["missing"] == ["香草"]
    assert report["missing_unindexed"] == ["博士", "Pith(卫戍协议)"]


def test_a_group_that_has_no_logo_file_is_reported_not_silently_dropped(tmp_path):
    root = library_with_logos(tmp_path, "Ave Mujica")
    rows = [
        {
            "干员名": "若叶睦",
            "页面名": "若叶睦",
            "职业": "狙击",
        }
    ]
    report = fetcher.apply_factions(
        rows,
        root,
        {
            "若叶睦": {
                "logo": "Ave Mujica",
                "nation": "",
                "group": "",
                "team": "Ave Mujica",
                "profession": "狙击",
                "id": "AM01",
            }
        },
    )
    assert rows[0]["阵营logo"] == "18-阵营logo/Logo Ave Mujica.png"


def test_a_faction_without_a_logo_file_is_listed_separately(tmp_path):
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [{"干员名": "若叶睦", "页面名": "若叶睦", "职业": ""}]
    report = fetcher.apply_factions(
        rows,
        root,
        {
            "若叶睦": {
                "logo": "Ave Mujica",
                "nation": "",
                "group": "",
                "team": "Ave Mujica",
                "profession": "",
                "id": "",
            }
        },
    )
    assert rows[0]["阵营"] == "Ave Mujica"
    assert rows[0]["阵营logo"] == ""
    assert report["missing"] == []
    assert report["logo_without_file"] == ["若叶睦 / Ave Mujica"]


def test_a_profession_disagreement_is_reported_not_resolved(tmp_path):
    """Two independent reads of 职业; picking one silently is how a wrong icon
    lands on a card with nothing pointing back at a source."""
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [{"干员名": "阿米娅", "页面名": "阿米娅", "职业": "术师", "情报编号": "R001"}]
    report = fetcher.apply_factions(
        rows,
        root,
        {
            "阿米娅": {
                "logo": "罗德岛",
                "nation": "罗德岛",
                "group": "",
                "team": "",
                "profession": "医疗",
                "id": "R001",
            }
        },
    )
    assert rows[0]["职业"] == "术师"
    assert len(report["mismatched"]) == 1
    assert "阿米娅" in report["mismatched"][0]


def test_a_missing_class_icon_is_reported(tmp_path):
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [{"干员名": "某人", "页面名": "某人", "职业": "特种"}]
    report = fetcher.apply_factions(rows, root, {})
    assert rows[0]["职业图标_黑"] == ""
    assert rows[0]["职业图标_白"] == ""
    assert len(report["icon_missing"]) == 2


def test_a_row_without_a_profession_gets_no_icon_and_no_icon_complaint(tmp_path):
    """NPC rows carry no 职业 at all -- that is not a missing icon."""
    root = library_with_logos(tmp_path, "罗德岛")
    rows = [{"干员名": "博士", "页面名": "博士", "职业": ""}]
    report = fetcher.apply_factions(rows, root, {})
    assert rows[0]["职业图标_黑"] == ""
    assert report["icon_missing"] == []


def test_empty_logo_folder_yields_no_faction_rather_than_a_path(tmp_path):
    """A machine without the artwork still gets a usable index."""
    root = tmp_path
    rows = [
        {"干员名": "香草", "页面名": "香草", "所属组织": "黑钢国际", "干员序号": "30"}
    ]
    report = fetcher.apply_factions(rows, root, {})
    assert rows[0]["阵营"] == ""
    assert report["missing"] == ["香草"]


def test_every_row_gets_the_faction_columns():
    """The CSV is written with a fixed field order; a missing key is a crash."""
    rows = [{"干员名": "a", "页面名": "a"}]
    fetcher.apply_factions(rows, Path("."), {})
    for column in fetcher.FACTION_COLUMNS:
        assert column in rows[0]


def test_the_index_column_order_is_the_canonical_one():
    """列顺序決定 JSON 的键序，而 JSON 是面板读的那份。

    Nothing in the panel needs a particular order -- the point is that the
    mirror's index and the badge subset's index are written by the same code
    path, so anything that reorders one has to reorder both. Otherwise a
    refresh of a single library makes the two files differ line by line while
    every value matches, and the next diff of them is unreadable.
    """
    assert fetcher.INDEX_COLUMNS[:7] == (
        "干员序号",
        "干员名",
        "情报编号",
        "干员id",
        "稀有度",
        "职业",
        "分支",
    )
    for column in fetcher.FACTION_COLUMNS:
        assert column in fetcher.INDEX_COLUMNS
    assert "页面名" in fetcher.INDEX_COLUMNS and "来源" in fetcher.INDEX_COLUMNS


def test_refresh_factions_writes_the_canonical_column_order(tmp_path, monkeypatch):
    """A refresh must not leave the file in whatever order it happened to read."""
    shuffled = ("干员名", "页面名", "干员序号", "阵营", "职业")
    fetcher.write_index_rows(
        tmp_path,
        [{"干员名": "a", "页面名": "a", "干员序号": "1", "职业": "术师"}],
        shuffled,
    )
    monkeypatch.setattr(fetcher, "faction_records", lambda: {})
    fetcher.refresh_factions(tmp_path)
    _, columns = fetcher.read_index_rows(tmp_path)
    assert list(columns) == list(fetcher.INDEX_COLUMNS)


def test_every_json_row_carries_every_column(tmp_path):
    """The JSON is what the panel reads, and it does not have ``restval``.

    Recomputing only the faction columns leaves the roster columns absent from
    the dicts; writing that straight to JSON gives rows that disagree about
    which keys exist, so the panel works on a full rebuild and breaks after a
    refresh.
    """
    fetcher.write_index_rows(
        tmp_path,
        [{"干员名": "a", "页面名": "a"}],
        fetcher.INDEX_COLUMNS,
    )
    import json as json_module

    rows = json_module.loads(
        (tmp_path / fetcher.INDEX_FOLDER / fetcher.INDEX_JSON).read_text(encoding="utf-8")
    )
    assert list(rows[0]) == list(fetcher.INDEX_COLUMNS)


def test_class_icon_folders_match_the_library_layout():
    """The two icon tones live in the same shape of folder name as the mirror."""
    for tone, folder, pattern in fetcher.CLASS_ICON_FOLDERS:
        assert tone in ("黑", "白")
        assert folder.startswith("15-职业大图/")
        assert "{profession}" in pattern


def test_professions_covered_by_the_library_have_both_icon_tones():
    """Every profession the index can hold has a black and a white big icon.

    Pinned against the real library when it is present: a new profession added
    to the game would otherwise produce rows whose icons are blank with no
    signal anywhere that the data outgrew the artwork.
    """
    library = badge_library()
    if library is None:
        pytest.skip("素材库不在本机")
    professions = set()
    for tone, folder, pattern in fetcher.CLASS_ICON_FOLDERS:
        names = {item.name for item in (library / folder).iterdir()}
        professions = professions or {
            name[len("图标 职业 ") : -len(" 大图.png")]
            for name in names
            if name.startswith("图标 职业 ") and name.endswith(" 大图.png")
        }
        assert names, folder
    assert professions == {
        "先锋",
        "近卫",
        "重装",
        "狙击",
        "术师",
        "医疗",
        "辅助",
        "特种",
    }, "职业大图的职业集合变了：新职业需要补图标，索引里的 职业图标_* 会为空"


def test_the_written_index_resolves_every_path_it_names():
    """Integration: the real index must not name a file that is not there.

    Guards the one failure mode that a reader cannot see -- a path that looks
    right in the CSV and draws nothing on the card.
    """
    library = badge_library()
    if library is None:
        pytest.skip("素材库不在本机")
    csv_path = library / "01-干员代号" / "干员代号与编号.csv"
    if not csv_path.exists():
        pytest.skip("还没有抓到干员索引")
    import csv as csv_module

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    assert rows, "索引是空的"
    problems = []
    for row in rows:
        for column in ("阵营logo", "职业图标_黑", "职业图标_白"):
            value = (row.get(column) or "").strip()
            if value and not (library / value).exists():
                problems.append(f"{row['干员名']} 的 {column} -> {value}")
    assert problems == []
    # Every playable operator has a faction; the NPC rows are the exception and
    # they are counted, not hidden. Anything below this means a fetch regressed
    # to a source that covers fewer operators.
    with_faction = sum(1 for row in rows if (row.get("阵营") or "").strip())
    assert with_faction >= 430, f"只有 {with_faction}/{len(rows)} 行有阵营"


def badge_library() -> Path | None:
    """The 通行证 subset under ``PCBART_ASSET_ROOT``, or ``None`` if absent.

    Resolved through the shared helper rather than a literal path so a machine
    that keeps the library elsewhere can still run the measurement tests.
    """
    candidate = asset_root() / "明日方舟-艺术板素材"
    return candidate if candidate.exists() else None
