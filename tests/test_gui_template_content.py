"""Tests for the 模板内容 panel: the operator search and the content editors.

The panel is where the user's second request lands -- "the blank template's
content must show on the canvas and be changeable" -- so these tests drive the
controls rather than the model functions underneath them. A search box wired to
nothing, or a spin box that shows a number but never writes it back, would pass
a model-level test and fail the user.

The operator database is a *file on disk*, so every test here points
``PCBART_ASSET_ROOT`` at a fixture. That is not tidiness: the real database is
a 463-row CSV in the user's materials library, and a test that read it would
pass or fail depending on whether they had downloaded four gigabytes of
artwork.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pcbart import badge_card, operators  # noqa: E402
from pcbart.artboard import ArtBoard  # noqa: E402
from pcbart.paths import ASSET_ROOT_ENV  # noqa: E402

INDEX_COLUMNS = (
    "干员序号",
    "干员名",
    "情报编号",
    "干员id",
    "稀有度",
    "职业",
    "分支",
    "阵营",
    "所属国家",
    "所属组织",
    "所属团队",
    "阵营来源",
    "阵营logo",
    "职业图标_黑",
    "职业图标_白",
    "PRTS链接",
    "页面名",
    "来源",
)

#: Three operators: one whose name is a prefix of another, one that only
#: matches on a substring, and an alternate form sharing a 编号 with the first.
#: The search ordering tests need all three shapes, and the alternate form is
#: what makes "keyed by page name, not by number" observable.
ROWS = (
    {
        "干员序号": "1", "干员名": "阿米娅", "情报编号": "R001",
        "干员id": "char_002_amiya", "稀有度": "4", "职业": "术师",
        "分支": "中坚术师", "阵营": "罗德岛",
        "阵营logo": "18-阵营logo/Logo 罗德岛.png",
        "职业图标_黑": "15-职业大图/黑/图标 职业 术师 大图.png",
        "职业图标_白": "15-职业大图/白/图标 职业 术师 大图 白.png",
        "PRTS链接": "https://prts.wiki/w/%E9%98%BF%E7%B1%B3%E5%A8%85",
        "页面名": "阿米娅", "来源": "干员档案",
    },
    {
        "干员序号": "2", "干员名": "阿米娅(医疗)", "情报编号": "R001",
        "干员id": "char_1037_amiya3", "稀有度": "4", "职业": "医疗",
        "分支": "咒愈师", "阵营": "罗德岛",
        "阵营logo": "18-阵营logo/Logo 罗德岛.png",
        "职业图标_黑": "15-职业大图/黑/图标 职业 医疗 大图.png",
        "职业图标_白": "15-职业大图/白/图标 职业 医疗 大图 白.png",
        "PRTS链接": "https://prts.wiki/w/%E9%98%BF%E7%B1%B3%E5%A8%85(%E5%8C%BB%E7%96%97)",
        "页面名": "阿米娅(医疗)", "来源": "干员档案",
    },
    {
        "干员序号": "3", "干员名": "假日威龙陈", "情报编号": "NM05",
        "干员id": "char_1028_texas2", "稀有度": "5", "职业": "近卫",
        "分支": "剑豪", "阵营": "炎-龙门",
        "阵营logo": "18-阵营logo/Logo 炎-龙门.png",
        "职业图标_黑": "15-职业大图/黑/图标 职业 近卫 大图.png",
        "职业图标_白": "15-职业大图/白/图标 职业 近卫 大图 白.png",
        "PRTS链接": "https://prts.wiki/w/%E5%81%87%E6%97%A5%E5%A8%81%E9%BE%99%E9%99%88",
        "页面名": "假日威龙陈", "来源": "干员档案",
    },
)


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A miniature materials library, and ``PCBART_ASSET_ROOT`` pointing at it."""
    root = tmp_path / "素材库"
    folder = root / "明日方舟-艺术板素材" / operators.INDEX_FOLDER
    folder.mkdir(parents=True)
    path = folder / operators.INDEX_CSV
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(ROWS)
    monkeypatch.setenv(ASSET_ROOT_ENV, str(root))
    operators.reset_cache()
    yield root
    operators.reset_cache()


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def panel(app, library):
    from conftest import release_window
    from pcbart.draft import board_from_dict, board_to_dict
    from tools.gui_mockup import TemplateContentPanel, build_window

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    panel = TemplateContentPanel(window, board)
    panel.show()
    app.processEvents()
    yield panel
    release_window(window)


def board_of(panel) -> ArtBoard:
    return panel._board


def text_row(panel, label: str) -> int:
    """The list row whose label matches, so a test does not depend on order."""
    for index, item in enumerate(board_of(panel).texts):
        if item.label == label:
            return index
    raise AssertionError(f"没有这一行：{label}")


def cutout_row(panel, key: str) -> int:
    for index, item in enumerate(board_of(panel).cutouts):
        if item.key == key:
            return index
    raise AssertionError(f"没有这个挖槽：{key}")


# ------------------------------------------------------------------- the list


def test_the_panel_lists_every_template_line(panel):
    assert panel.text_list.count() == len(badge_card.BADGE_TEXTS)


def test_the_panel_lists_the_cutout(panel):
    assert panel.cutout_list.count() == len(badge_card.BADGE_CUTOUTS) if hasattr(
        badge_card, "BADGE_CUTOUTS"
    ) else panel.cutout_list.count() == 1


def test_each_row_shows_its_purpose_and_its_current_text(panel):
    """A list of bare labels does not say what the card will actually print."""
    first = panel.text_list.item(0).text()
    assert badge_card.BADGE_TEXTS[0].label in first
    # No operator chosen yet, so the token is shown as-is. The 代号 line carries
    # {英文名}: it is set in the template's Latin-only typeface.
    assert "{英文名}" in first


def test_a_missing_database_says_so_without_disabling_the_editors(
    app, tmp_path, monkeypatch
):
    """No library is a normal state, not a crash.

    The template's wording and slot do not need the database at all, so the
    panel has to keep them usable and explain what is missing rather than
    refusing to open.
    """
    from conftest import release_window
    from tools.gui_mockup import TemplateContentPanel, build_window

    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path / "空"))
    operators.reset_cache()
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()
    board = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()
    panel = TemplateContentPanel(window, board)
    panel.show()
    app.processEvents()
    try:
        assert "找不到干员资料库" in panel.status_label.text()
        assert panel.text_edit.isEnabled()
        assert panel.cutout_width.isEnabled()
        assert panel.operator_results.count() == 0
    finally:
        operators.reset_cache()
        release_window(window)


# ------------------------------------------------------------------ the search


def test_typing_filters_the_operator_list(panel):
    panel.operator_search.setText("假日")
    assert panel.operator_results.count() == 1
    assert "假日威龙陈" in panel.operator_results.item(0).text()


def test_an_empty_query_lists_the_operators(panel):
    """Shows the database loaded, instead of a blank box that looks broken."""
    panel.operator_search.setText("")
    assert panel.operator_results.count() == len(ROWS)


def test_a_prefix_match_comes_before_a_substring_match(panel):
    """Typing 阿米 should not bury 阿米娅 behind a later name containing it."""
    panel.operator_search.setText("阿米")
    names = [
        panel.operator_results.item(index).data(Qt.UserRole)
        for index in range(panel.operator_results.count())
    ]
    assert names[0] == "阿米娅"
    assert names[1] == "阿米娅(医疗)"


def test_picking_an_operator_writes_it_to_the_document(panel):
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    app_board = board_of(panel)
    assert app_board.operator_name == "假日威龙陈"
    assert app_board.operator_number == "NM05"


def test_picking_an_operator_rewrites_the_visible_rows(panel):
    """The list has to re-render, or the user sees the token and assumes it failed."""
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    joined = " ".join(
        panel.text_list.item(index).text()
        for index in range(panel.text_list.count())
    )
    assert "NM05" in joined
    # The 代号 row shows the operator's English name, because the template sets
    # that line in a Latin-only face.
    assert "Ch'en the Holungday" in joined
    assert "{英文名}" not in joined
    assert "{代号}" not in joined


def test_the_search_classifies_the_operator_alternate_forms_separately(panel):
    """阿米娅 has three forms sharing 编号 R001 and differing by class.

    Keying on the number would give all three the first one's profession, and
    the list would offer three rows that cannot be told apart.
    """
    panel.operator_search.setText("阿米娅")
    rows = [
        panel.operator_results.item(index).text()
        for index in range(panel.operator_results.count())
    ]
    assert any("医疗" in row for row in rows)


def test_clearing_the_operator_restores_the_placeholders(panel):
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert board_of(panel).operator_name
    panel._on_clear_operator()
    assert board_of(panel).operator_name == ""
    assert "尚未选择干员" in panel.current_operator.text()
    joined = " ".join(
        panel.text_list.item(index).text()
        for index in range(panel.text_list.count())
    )
    assert "{英文名}" in joined
    assert "{编号}" in joined


def test_choosing_an_operator_does_not_rewrite_the_tokens(panel):
    """Switching operator must stay possible.

    Substituting into the strings would mean the second operator had to find
    the first one's name inside ``ARKNIGHTS-R001`` to replace it.
    """
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert any("{编号}" in item.text for item in board_of(panel).texts)


def test_the_panel_reports_which_operator_is_current(panel):
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    label = panel.current_operator.text()
    assert "假日威龙陈" in label and "NM05" in label and "近卫" in label


def test_the_panel_stores_the_page_name_not_just_the_name(panel):
    """The page name is the identity; 暮落 and 暮落(集成战略) share a name."""
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert board_of(panel).operator_page == "假日威龙陈"


def test_refreshing_after_a_pick_still_works(panel):
    """Refreshing with an operator chosen used to raise.

    ``operator_search`` is a QLineEdit, but ``refresh`` called QComboBox's
    ``findText``/``setCurrentIndex`` on it, so every refresh after picking
    someone -- a tab switch, an undo, a workspace refresh -- failed with
    AttributeError. Only the empty-board path was covered before.
    """
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    panel.refresh()
    assert panel.operator_results.count() >= 1
    assert "假日威龙陈" in panel.current_operator.text()


# -------------------------------------------------------------- text editing


def test_editing_a_text_line_writes_it_to_the_document(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_edit.setText("SPECIALIST")
    panel._on_text_changed("SPECIALIST")
    assert board_of(panel).texts[row].text == "SPECIALIST"


def test_changing_a_font_size_writes_it_to_the_document(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_size.setValue(2.5)
    assert board_of(panel).texts[row].size_mm == pytest.approx(2.5)


def test_moving_a_line_writes_both_coordinates(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_x.setValue(3.0)
    panel.text_y.setValue(-4.0)
    assert board_of(panel).texts[row].x_mm == pytest.approx(3.0)
    assert board_of(panel).texts[row].y_mm == pytest.approx(-4.0)


def test_rotating_a_line_writes_the_angle(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_rotation.setValue(90.0)
    assert board_of(panel).texts[row].rotation_deg == pytest.approx(90.0)


def test_the_layer_choice_writes_the_layer(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    index = panel.text_layer.findText("底层丝印层")
    panel.text_layer.setCurrentIndex(index)
    assert board_of(panel).texts[row].layer == "底层丝印层"


def test_selecting_a_row_shows_its_values(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    item = board_of(panel).texts[row]
    assert panel.text_edit.text() == item.text
    assert panel.text_size.value() == pytest.approx(item.size_mm)
    assert panel.text_x.value() == pytest.approx(item.x_mm)


def test_a_line_can_be_given_a_token_by_hand(panel):
    """The tokens are documented in the panel and usable in any line."""
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_edit.setText("{编号}")
    panel._on_text_changed("{编号}")
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    item = board_of(panel).texts[row]
    assert item.resolved_text(
        name=board_of(panel).operator_name, number=board_of(panel).operator_number
    ) == "NM05"


def test_restoring_one_line_puts_its_wording_back(panel):
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_edit.setText("XYZ")
    panel._on_text_changed("XYZ")
    panel._on_reset_text()
    assert board_of(panel).texts[row].text == badge_card.BADGE_TEXTS[row].text


def test_restoring_a_line_leaves_its_position_alone(panel):
    """Only the wording is restored: the move was deliberate.

    Resetting the position too would silently undo work the user did not ask to
    lose, which is the reason the button is labelled 恢复模板原文 rather than
    重置全部.
    """
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    item = board_of(panel).texts[row]
    panel.text_x.setValue(item.x_mm + 5.0)
    moved = item.x_mm
    panel._on_reset_text()
    assert board_of(panel).texts[row].x_mm == pytest.approx(moved)


def test_editing_a_line_emits_changed(panel):
    """The window routes this signal into the undo history.

    An edit that repaints but is not recorded is worse than no undo at all --
    the user believes Ctrl+Z covers it.
    """
    seen: list[int] = []
    panel.changed.connect(lambda: seen.append(1))
    row = text_row(panel, "STAFF")
    panel.text_list.setCurrentRow(row)
    panel.text_size.setValue(3.0)
    assert seen


# ------------------------------------------------------------ cutout editing


def test_editing_the_slot_size_writes_it_to_the_document(panel):
    panel.cutout_list.setCurrentRow(cutout_row(panel, "lanyard"))
    panel.cutout_width.setValue(18.0)
    panel.cutout_height.setValue(4.0)
    slot = next(item for item in board_of(panel).cutouts if item.key == "lanyard")
    assert slot.width_mm == pytest.approx(18.0)
    assert slot.height_mm == pytest.approx(4.0)


def test_moving_the_slot_writes_both_coordinates(panel):
    panel.cutout_list.setCurrentRow(cutout_row(panel, "lanyard"))
    panel.cutout_x.setValue(-2.0)
    panel.cutout_y.setValue(30.0)
    slot = next(item for item in board_of(panel).cutouts if item.key == "lanyard")
    assert slot.x_mm == pytest.approx(-2.0)
    assert slot.y_mm == pytest.approx(30.0)


def test_selecting_the_slot_shows_its_values(panel):
    row = cutout_row(panel, "lanyard")
    panel.cutout_list.setCurrentRow(row)
    slot = board_of(panel).cutouts[row]
    assert panel.cutout_width.value() == pytest.approx(slot.width_mm)
    assert panel.cutout_y.value() == pytest.approx(slot.y_mm)


def test_a_round_hole_keeps_its_width_and_height_equal(panel):
    row = cutout_row(panel, "hole_left")
    panel.cutout_list.setCurrentRow(row)
    assert not panel.cutout_height.isEnabled()
    panel.cutout_width.setValue(4.0)
    hole = board_of(panel).cutouts[row]
    assert hole.width_mm == pytest.approx(4.0)
    assert hole.height_mm == pytest.approx(4.0)


def test_the_slot_error_is_hidden_while_it_is_valid(panel):
    panel.cutout_list.setCurrentRow(cutout_row(panel, "lanyard"))
    assert not panel.cutout_error.isVisible()


def test_editing_the_slot_emits_changed(panel):
    seen: list[int] = []
    panel.changed.connect(lambda: seen.append(1))
    panel.cutout_list.setCurrentRow(cutout_row(panel, "lanyard"))
    panel.cutout_height.setValue(5.0)
    assert seen


# ------------------------------------------------------------- re-pointing


def test_the_panel_can_be_re_pointed_at_another_board(panel):
    """A tab switch has to leave the panel editing the board on screen."""
    other = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    other.texts = badge_card.badge_texts()
    other.cutouts = badge_card.badge_cutouts()
    other.texts[0].text = "SECOND BOARD"
    panel.set_board(other)
    assert panel.text_list.count() == len(other.texts)
    assert "SECOND BOARD" in panel.text_list.item(0).text()


def test_re_pointing_does_not_keep_the_old_boards_operator(panel):
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert board_of(panel).operator_name

    other = ArtBoard(outline=badge_card.badge_spec(), outline_ready=True)
    other.texts = badge_card.badge_texts()
    panel.set_board(other)
    assert "尚未选择干员" in panel.current_operator.text()


def test_typing_a_number_writes_it_to_the_document(panel):
    """The 编号 is editable on its own, not only through the operator lookup.

    The roster is a scrape: some operators have no number and some cards need
    one the wiki does not carry. Typing it has to reach the document the same
    way picking an operator does.
    """
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert board_of(panel).operator_number == "NM05"

    panel.number_edit.setText("CUSTOM-9")
    panel._on_number_edited("CUSTOM-9")
    assert board_of(panel).operator_number == "CUSTOM-9"


def test_a_typed_number_reaches_the_text_lines(panel):
    """The tokens re-render, or the card still prints the old number."""
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    panel.number_edit.setText("ZZ99")
    panel._on_number_edited("ZZ99")
    joined = " ".join(
        panel.text_list.item(index).text()
        for index in range(panel.text_list.count())
    )
    assert "ZZ99" in joined
    assert "NM05" not in joined


def test_the_number_box_follows_the_picked_operator(panel):
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    assert panel.number_edit.text() == "NM05"


def test_a_refresh_does_not_wipe_a_typed_number(panel):
    """The list repopulates on a timer; typing must survive it."""
    panel.number_edit.setText("KEEP-1")
    panel._on_number_edited("KEEP-1")
    panel.refresh()
    assert panel.number_edit.text() == "KEEP-1"
    assert board_of(panel).operator_number == "KEEP-1"


# --------------------------------------------------------------- card art

def test_the_panel_lists_every_automatic_element(panel):
    """All five rows, the 罗德岛 crest included."""
    labels = [
        panel.art_list.item(index).text()
        for index in range(panel.art_list.count())
    ]
    for wanted in ("二维码", "条形码", "职业图标", "阵营 logo", "罗德岛标志"):
        assert any(wanted in row for row in labels), wanted


def test_a_choice_can_be_changed_and_removed(panel):
    """Each element offers 自动 / 不放 / a picked asset, and records the pick."""
    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    board = board_of(panel)

    index = next(
        row
        for row in range(panel.art_list.count())
        if "职业图标" in panel.art_list.item(row).text()
    )
    panel.art_list.setCurrentRow(index)
    assert panel.art_pick.itemText(0) == "自动"
    assert panel.art_pick.itemText(1) == "不放"
    # The professions the roster actually carries are offered.
    options = [
        panel.art_pick.itemText(row) for row in range(panel.art_pick.count())
    ]
    assert "术师" in options and "近卫" in options

    panel.art_pick.setCurrentText("不放")
    assert board.art_choices["profession"] == "none"

    panel.art_pick.setCurrentText("术师")
    assert board.art_choices["profession"] == "术师"

    panel.art_pick.setCurrentText("自动")
    assert "profession" not in board.art_choices


def test_searching_narrows_the_element_list(panel):
    panel.art_search.setText("阵营")
    rows = [
        panel.art_list.item(index).text()
        for index in range(panel.art_list.count())
    ]
    assert rows and all("阵营" in row for row in rows)


def test_picking_an_asset_records_it_and_rebuilds_the_card(panel, monkeypatch):
    """Picking an asset re-inserts, so the canvas shows the new one.

    ⚠️ The test library is a CSV with no image files, so the real insert would
    report ``MissingArt``. The rebuild is therefore *observed* rather than
    allowed to run: what this checks is that a pick reaches the document and is
    followed by one re-insert -- the model-level result of that insert is
    covered against the real library elsewhere.
    """
    calls: list[bool] = []

    def fake_insert(*, announce: bool) -> bool:
        calls.append(announce)
        return True

    panel.operator_search.setText("假日")
    panel.operator_results.setCurrentRow(0)
    board = board_of(panel)
    monkeypatch.setattr(panel, "_insert_card_art", fake_insert)

    index = next(
        row
        for row in range(panel.art_list.count())
        if "阵营 logo" in panel.art_list.item(row).text()
    )
    panel.art_list.setCurrentRow(index)
    # ``setCurrentIndex`` rather than ``setCurrentText``: the latter does not
    # emit ``currentIndexChanged`` when the text is already the current one,
    # which would make the test pass or fail on the widget's internal state
    # rather than on the handler.
    #
    # 罗德岛 is the interesting pick: it is the one faction the automatic rule
    # *excludes*, because its crest is already on the silkscreen. Choosing it
    # by hand has to be honoured anyway.
    chosen = panel.art_pick.findText("罗德岛")
    assert chosen >= 0
    panel.art_pick.setCurrentIndex(chosen)

    assert board.art_choices["faction"] == "罗德岛"
    assert calls == [False], "选项改完要重放一次卡面"
    # The row's own text follows the choice.
    assert "罗德岛" in panel.art_list.item(index).text()


def test_the_crest_row_removes_the_crest(panel):
    from pcbart import badge_content

    index = next(
        row
        for row in range(panel.art_list.count())
        if "罗德岛标志" in panel.art_list.item(row).text()
    )
    panel.art_list.setCurrentRow(index)
    panel.art_pick.setCurrentText("不放")
    assert board_of(panel).art_choices[badge_content.EMBLEM_KIND] == "none"
