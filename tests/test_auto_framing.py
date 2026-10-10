"""Importing a picture onto a 通行证 must land it in the illustration opening.

⚠️ **This is the regression the user reported.** The region was only applied
when somebody called ``canvas.start_framing``, and the only callers were
preview scripts -- so the panel's own import path never framed anything. A
picture imported onto a card landed at its default 76.2 mm with no ``region``
and no ``source_window``: nothing cropped it, and both writers treated it as an
ordinary picture. The 彩色丝印 evidence that suggested it worked came from a
script that framed the placement by hand.

The fix applies the opening on the import, in one place, so all three styles are
covered together -- 线稿, 板画 and 彩色丝印 are three conversions of *one*
placement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gui_mockup import (  # noqa: E402
    BOARD_ART_STYLE,
    COLOR_STYLE,
    STYLE_OPTIONS,
    build_window,
    show_preprocess,
)
from pcbart import badge_card  # noqa: E402
from pcbart.artboard import BoardOutlineSpec, OutlineShape  # noqa: E402
from pcbart.image_prep import imwrite_unicode  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def figure(tmp_path: Path) -> Path:
    """A cut-out figure, so 抠图 and the region both have something to work on."""
    path = tmp_path / "figure.png"
    image = np.zeros((600, 400, 4), np.uint8)
    cv2.ellipse(image, (200, 170), (70, 90), 0, 0, 360, (60, 60, 60, 255), -1)
    cv2.ellipse(image, (200, 400), (110, 170), 0, 0, 360, (30, 30, 30, 255), -1)
    imwrite_unicode(str(path), image)
    return path


def imported(app, tmp_path, style: str):
    """Import a figure onto a card through the real panel, and return the panel."""
    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    board = window.design_canvas.board
    board.outline = badge_card.badge_spec()
    board.outline_ready = True
    board.texts = badge_card.badge_texts()
    board.cutouts = badge_card.badge_cutouts()

    panel = show_preprocess(window)
    app.processEvents()
    assert panel.load_image(str(figure(tmp_path))) is True
    index = [key for key, _hint in STYLE_OPTIONS].index(style)
    panel.style_combo.setCurrentIndex(index)
    app.processEvents()
    assert panel.import_to_canvas() is True
    app.processEvents()
    return window, board, panel


@pytest.mark.parametrize("style", [COLOR_STYLE, "线稿", BOARD_ART_STYLE])
def test_importing_onto_a_card_frames_the_picture(app, tmp_path, style):
    """Every style, one assertion: the opening is applied on import."""
    from conftest import release_window

    window, board, _panel = imported(app, tmp_path, style)
    try:
        assert board.placements, f"{style}: 导入后画布上应当有对象"
        for item in board.placements:
            assert item.region == "small", (
                f"{style}: {item.layer_key} 应当已放入立绘区，实际 region={item.region!r}"
            )
            assert item.source_window is not None, (
                f"{style}: 取景窗口没有写入，导出时就不会裁"
            )
            region = next(r for r in badge_card.BADGE_REGIONS if r.key == "small")
            assert item.width_mm == pytest.approx(region.width_mm, abs=1e-6)
            assert item.height_mm == pytest.approx(region.height_mm, abs=1e-6)
    finally:
        release_window(window)


def test_the_window_matches_the_region_shape(app, tmp_path):
    """A window of the wrong shape stretches the picture into the opening.

    ⚠️ Both aspects have to be computed in *millimetres on the board*, and a
    window is a fraction of the picture -- so the picture's own pixel aspect has
    to come out first. A 400 x 600 source and a square one must both end up
    with a window whose board-space shape is the region's.
    """
    from conftest import release_window

    window, board, _panel = imported(app, tmp_path, COLOR_STYLE)
    try:
        region = next(r for r in badge_card.BADGE_REGIONS if r.key == "small")
        for item in board.placements:
            x0, y0, x1, y1 = item.source_window
            width_px, height_px = 400, 600
            crop_w = (x1 - x0) * width_px
            crop_h = (y1 - y0) * height_px
            assert crop_w / crop_h == pytest.approx(region.aspect, abs=0.02), (
                f"{item.layer_key}: 窗口形状 {crop_w / crop_h:.4f} "
                f"与区域 {region.aspect:.4f} 不符，导出会拉伸"
            )
    finally:
        release_window(window)


def test_a_board_that_is_not_a_card_is_left_alone(app, tmp_path):
    """⚠️ The framing must not touch an ordinary board.

    Every existing project, draft and test uses a plain rectangle, and their
    behaviour has to be exactly what it was: a picture lands where it lands,
    with no region and no crop.
    """
    from conftest import release_window

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    board = window.design_canvas.board
    board.outline = BoardOutlineSpec(shape=OutlineShape.RECT, width_mm=60.0, height_mm=60.0)
    board.outline_ready = True

    panel = show_preprocess(window)
    app.processEvents()
    panel.load_image(str(figure(tmp_path)))
    app.processEvents()
    panel.import_to_canvas()
    app.processEvents()
    try:
        for item in board.placements:
            assert item.region == "", "普通板子不该被放进立绘区"
            assert item.source_window is None, "普通板子不该有取景窗口"
    finally:
        release_window(window)


def test_re_importing_keeps_the_window_the_user_chose(app, tmp_path):
    """⚠️ The user's own framing must survive a re-import.

    风格 describes the *next* import, so the way to try another style is to pick
    it and press 导入图片 again. Re-framing on that second press would silently
    throw away the crop the user had just adjusted -- the same class of mistake
    the preview/committed split already had to be fixed for once.
    """
    from conftest import release_window
    from tools.design_canvas import _zoom_window

    window, board, panel = imported(app, tmp_path, COLOR_STYLE)
    try:
        canvas = window.design_canvas
        placement = board.placements[0]
        # The user pushes in a little, the way framing a portrait goes.
        canvas.selected = placement
        canvas.framing = placement
        canvas.framing_region = next(
            r for r in badge_card.BADGE_REGIONS if r.key == "small"
        )
        canvas._set_window(placement, _zoom_window(placement.source_window, 1.0 / 1.4))
        adjusted = placement.source_window

        panel.style_combo.setCurrentIndex(
            [key for key, _h in STYLE_OPTIONS].index("线稿")
        )
        app.processEvents()
        panel.import_to_canvas()
        app.processEvents()

        kept = [item.source_window for item in board.placements if item.region]
        assert kept, "重新导入后应当仍有取景过的对象"
        for window_box in kept:
            assert window_box == adjusted, (
                f"重新导入不该改动用户调好的取景：{adjusted} -> {window_box}"
            )
    finally:
        release_window(window)


def test_an_automatic_import_does_not_enter_framing_mode(app, tmp_path):
    """⚠️ Applying the crop and *framing* are two different things.

    An automatic import has to place the picture in the opening and cut the
    rest -- but it must not leave the canvas in 取景中. Entering that mode puts
    up the amber outline, dims the part outside the window, and makes a drag pan
    the *picture* instead of moving the object; a user who only asked to import
    a picture then has to right-click their way back out.

    Measured before the split: every automatic import of a figure ended in
    framing mode, on all three styles.
    """
    from conftest import release_window

    window, board, _panel = imported(app, tmp_path, COLOR_STYLE)
    try:
        canvas = window.design_canvas
        assert canvas.framing is None, (
            "自动导入不该把画布留在取景模式"
        )
        # The crop is still applied -- this is a mode check, not a "did it
        # frame" check, and both must hold at once.
        assert board.placements[0].region == "small"
        assert board.placements[0].source_window is not None
    finally:
        release_window(window)


def test_the_context_menu_still_enters_framing_mode(app, tmp_path):
    """The interactive entry keeps its behaviour: 放入短区 hands the user the
    pan/zoom gesture, which is the whole point of that menu entry."""
    from conftest import release_window

    window, board, _panel = imported(app, tmp_path, COLOR_STYLE)
    try:
        canvas = window.design_canvas
        board.placements[0].source_window = None
        board.placements[0].region = ""
        canvas.start_framing(board.placements[0], badge_card.BADGE_REGIONS[0])
        assert canvas.framing is board.placements[0], "右键放入应当进入取景模式"
        assert canvas.framing_region is badge_card.BADGE_REGIONS[0]
    finally:
        release_window(window)


# ------------------------------------- the template picks the opening for you


def _pictures_on_a_plain_board(app, tmp_path, *, count: int = 1):
    """A window with ``count`` pictures imported onto a board that is not a card.

    The order matters and is the reported bug's shape: the pictures exist
    *before* any 通行证 template does, so nothing has framed them yet.
    """
    from gui_mockup import show_board_setup

    window = build_window()
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    board = window.design_canvas.board
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=80.0, height_mm=120.0
    )
    board.outline_ready = True

    panel = show_preprocess(window)
    app.processEvents()
    for index in range(count):
        source = tmp_path / f"figure{index}.png"
        if not source.exists():
            image = np.zeros((600, 400, 4), np.uint8)
            cv2.ellipse(image, (200, 170), (70, 90), 0, 0, 360, (60, 60, 60, 255), -1)
            cv2.ellipse(image, (200, 400), (110, 170), 0, 0, 360, (30, 30, 30, 255), -1)
            imwrite_unicode(str(source), image)
        assert panel.load_image(str(source)) is True
        app.processEvents()
        assert panel.import_to_canvas() is True
        app.processEvents()

    setup = show_board_setup(window)
    app.processEvents()
    return window, board, panel, setup


def _choose_template(setup, name: str) -> None:
    """Drive the shape combo into 套用模板 and pick the named entry."""
    labels = [setup.shape_combo.itemText(i) for i in range(setup.shape_combo.count())]
    setup.shape_combo.setCurrentIndex(labels.index(setup.TEMPLATE_OPTION))
    names = [setup.template_combo.itemText(i) for i in range(setup.template_combo.count())]
    setup.template_combo.setCurrentIndex(names.index(name))


def test_applying_a_template_crops_pictures_that_were_already_on_the_board(
    app, tmp_path
):
    """⚠️ The reported interaction bug: the crop must not need a right-click.

    A picture imported onto an ordinary board lands at 76.2 mm with no region.
    Putting the 通行证 template on afterwards used to leave it exactly there --
    the card arrived, the picture did not move, and the only way to get it into
    the illustration window was 右键 → 放入短区. The template decides the
    layout, so it has to decide the crop as well.
    """
    from conftest import release_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        picture = board.placements[0]
        assert picture.region == "", "前置条件：导入时还没有卡，所以没有取景"
        assert picture.source_window is None

        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()

        region = next(r for r in badge_card.BADGE_REGIONS if r.key == "small")
        assert picture.region == "small", (
            f"套用模板后应当自动放入短区，实际 region={picture.region!r}"
        )
        assert picture.source_window is not None, "没有窗口，导出时就不会裁"
        assert picture.width_mm == pytest.approx(region.width_mm, abs=1e-6)
        assert picture.height_mm == pytest.approx(region.height_mm, abs=1e-6)
    finally:
        release_window(window)


def test_switching_stage_redraws_the_crop_for_the_new_opening(app, tmp_path):
    """精0 → 精2 has to move the picture, not just rename the layout.

    ⚠️ This is the second half of the same bug: the region key is what the
    writer clips to, so a picture left declaring ``"small"`` while the card is
    tall is clipped by a window whose shape it was never framed for. The stage
    change moves both the box and the window.
    """
    from conftest import release_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        picture = board.placements[0]
        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()
        short = next(r for r in badge_card.BADGE_REGIONS if r.key == "small")
        assert picture.region == "small"
        assert picture.width_mm == pytest.approx(short.width_mm, abs=1e-6)

        _choose_template(setup, badge_card.template_name("精2"))
        app.processEvents()

        tall = next(r for r in badge_card.BADGE_REGIONS if r.key == "large")
        assert picture.region == "large", (
            f"切到精2后应当改放长区，实际 region={picture.region!r}"
        )
        assert picture.width_mm == pytest.approx(tall.width_mm, abs=1e-6)
        assert picture.height_mm == pytest.approx(tall.height_mm, abs=1e-6)
        # And the window has to follow: a short-window crop stretched into the
        # tall opening is the failure this assertion exists to catch.
        x0, y0, x1, y1 = picture.source_window
        crop_w = (x1 - x0) * 400
        crop_h = (y1 - y0) * 600
        assert crop_w / crop_h == pytest.approx(tall.aspect, abs=0.02)
    finally:
        release_window(window)


def test_reselecting_the_same_template_keeps_the_user_s_own_framing(app, tmp_path):
    """Choosing the template already in force must not throw the crop away.

    The window the user pans and zooms is *their* edit. Re-selecting 通行证 is
    how the panel restores mangled wording, and it is also what happens if the
    user simply walks back into 板框设置 -- neither is a request to re-frame.
    """
    from conftest import release_window
    from tools.design_canvas import _zoom_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        picture = board.placements[0]
        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()

        canvas = window.design_canvas
        canvas.selected = picture
        canvas._set_window(
            picture, _zoom_window(picture.source_window, 1.0 / 1.4)
        )
        adjusted = picture.source_window

        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()

        assert picture.source_window == adjusted, (
            f"重新选同一个模板不该改动取景：{adjusted} -> {picture.source_window}"
        )
        assert picture.region == "small"
    finally:
        release_window(window)


def test_reframing_a_stage_change_is_one_undo_step(app, tmp_path):
    """One layout decision, one Ctrl+Z.

    A card can carry several pictures, and each reframing records its own
    history entry. Without a shared run key the user would have to press Ctrl+Z
    once per picture to take back a single choice; the template swap itself is
    one of those entries.
    """
    from conftest import release_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path, count=2)
    try:
        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()
        for picture in board.placements:
            assert picture.region == "small"

        _choose_template(setup, badge_card.template_name("精2"))
        app.processEvents()
        for picture in board.placements:
            assert picture.region == "large"

        canvas = window.design_canvas
        canvas.undo()
        app.processEvents()
        assert board.outline.template == badge_card.template_name("精0"), (
            "一次撤销应当回到精0"
        )
        assert [item.region for item in board.placements] == ["small", "small"], (
            "两张图应当一起回到短区，撤销次数不该随图片数量增长"
        )
    finally:
        release_window(window)


def test_the_card_s_own_generated_elements_are_never_reframed(app, tmp_path):
    """⚠️ The crop is for the figure, not for the 二维码.

    A card also carries the operator's 二维码, 条形码, 职业图标 and 阵营 logo,
    generated by :mod:`pcbart.badge_content` at their own measured sizes.
    Treating them as illustrations would shrink a 3.2 mm code into a 38 mm
    opening, which is a card that scans as nothing.

    ⚠️ **Built by hand rather than through ``badge_content.badge_art``**, so the
    test runs on a machine without the 8 GB asset library. The rule under test is
    the canvas's: "anything stamped with ``ART_IMPORT_ID`` is the card's own
    artwork". The sizes below are the measured ones from the source template, and
    a companion assertion pins that the id is the constant
    :mod:`pcbart.badge_content` itself stamps -- otherwise this test could pass
    while the real elements used a different id and got reframed after all.
    """
    from conftest import release_window

    from pcbart import badge_content
    from pcbart.artboard import ImagePlacement
    from pcbart.design_layers import layer

    #: ``(kind, width_mm, height_mm, layer key)``, at the sizes the source
    #: template measures. ``ImagePlacement`` stores the layer's *display name*,
    #: which is why the key is translated here rather than written out.
    measured = (
        ("qr", 4.42, 4.42, "bottom_silk"),
        ("barcode", 49.18, 4.94, "top_silk"),
        ("profession", 7.60, 7.60, "top_silk"),
        ("faction", 15.04, 21.09, "top_copper"),
    )

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        generated = []
        for kind, width, height, layer_key in measured:
            item = ImagePlacement(
                path="", x_mm=0.0, y_mm=0.0, width_mm=width, height_mm=height,
                layer=layer(layer_key).name, kind=kind,
                import_id=badge_content.ART_IMPORT_ID,
            )
            generated.append(item)
        board.placements.extend(generated)
        before = {
            (item.kind, round(item.width_mm, 4), round(item.height_mm, 4))
            for item in generated
        }

        _choose_template(setup, badge_card.template_name("精2"))
        app.processEvents()

        after = {
            (item.kind, round(item.width_mm, 4), round(item.height_mm, 4))
            for item in generated
        }
        assert after == before, (
            f"卡面自带素材不该被改尺寸/取景：{before} -> {after}"
        )
        for item in generated:
            assert item.region == "", f"{item.kind} 不该被打上立绘区"
            assert item.source_window is None, f"{item.kind} 不该有取景窗口"

        # And the id used above is the real one, so this cannot pass by testing
        # a marker the production code never writes.
        assert badge_content.ART_IMPORT_ID == "badge-art"
    finally:
        release_window(window)


def test_a_manual_opening_choice_survives_re_applying_the_same_card(app, tmp_path):
    """The template decides the crop; it does not overrule a direct choice.

    The right-click menu is still there for a picture that suits the other
    opening, so a user who deliberately puts a figure in 长区 on a 精0 board
    must keep it when the panel re-applies the card it already has (which is
    what walking back into 板框设置 does).
    """
    from conftest import release_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        picture = board.placements[0]
        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()
        assert picture.region == "small"

        # Hand-placed into the other opening, the way the context menu does it.
        tall = next(r for r in badge_card.BADGE_REGIONS if r.key == "large")
        window.design_canvas.start_framing(picture, tall)
        window.design_canvas.stop_framing()
        chosen = picture.source_window

        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()

        assert picture.region == "large", "手工选的开口不该被同模板重套覆盖"
        assert picture.source_window == chosen
    finally:
        release_window(window)


def test_a_plain_board_keeps_its_pictures_where_they_are(app, tmp_path):
    """The other half of the rule: no card, no crop.

    Every existing project and draft uses ordinary boards, and applying a
    rectangle must leave their artwork exactly as imported.
    """
    from conftest import release_window

    window, board, _panel, setup = _pictures_on_a_plain_board(app, tmp_path)
    try:
        picture = board.placements[0]
        before = (picture.width_mm, picture.height_mm, picture.region)
        _choose_template(setup, badge_card.template_name("精0"))
        app.processEvents()
        assert picture.width_mm != before[0], "前提检查：套模板确实会裁"

        # And back to a plain rectangle: the picture is left as it is rather
        # than being re-framed a second time.
        labels = [
            setup.shape_combo.itemText(i) for i in range(setup.shape_combo.count())
        ]
        rect = labels.index(OutlineShape.LABELS[OutlineShape.RECT])
        framed = (board.placements[0].width_mm, board.placements[0].height_mm)
        setup.shape_combo.setCurrentIndex(rect)
        app.processEvents()
        assert (
            board.placements[0].width_mm,
            board.placements[0].height_mm,
        ) == framed, "普通矩形板不该再动一次图片"
    finally:
        release_window(window)
