"""End-to-end acceptance check: two boards, both product saves, a restart.

The test suite checks each piece against a fake or a scratch folder. This runs
the *whole* workflow in one process against real files, which is the only way
to catch a break in the joins -- a tab that switches but does not re-point a
pane, a draft that saves but is not listed after a restart.

Run it from the project root:

    python tools/acceptance_check.py

It writes into ``out/_e2e`` and exits non-zero if anything fails. Kept out of
pytest deliberately: it is a script a person runs before packaging, and its
output is meant to be read rather than counted.
"""

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

import gui_mockup as g
from pcbart.artboard import BoardOutlineSpec, ImagePlacement, OutlineShape
from pcbart.draft import DraftStore

work = ROOT / "out" / "_e2e"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

QMessageBox.exec = lambda box, *a, **k: 0
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

app = QApplication.instance() or QApplication(sys.argv)
app.setFont(QFont("Microsoft YaHei UI", 9))

w = g.build_window()
w._draft_store = DraftStore(work / "library")
w.setAttribute(Qt.WA_DontShowOnScreen, True)
w.resize(1400, 900)
w.show()
app.processEvents()


def sample_artwork() -> str:
    """Synthesise the artwork this run imports.

    Not read from ``out/check_letter.png``: ``out/`` is git-ignored, so on a
    fresh clone that file does not exist and the script would happily export a
    board whose placement points at nothing. Generating it here keeps the check
    runnable on any checkout.
    """
    import cv2

    from pcbart.image_prep import imwrite_unicode, make_test_image

    path = work / "artwork.png"
    imwrite_unicode(
        str(path), cv2.cvtColor(make_test_image(shape="letter"), cv2.COLOR_GRAY2BGR)
    )
    return str(path)


sample = sample_artwork()


def ready(width: float) -> None:
    board = w.design_canvas.board
    board.outline_ready = True
    board.outline = BoardOutlineSpec(
        shape=OutlineShape.RECT, width_mm=width, height_mm=40
    )
    board.placements.append(
        ImagePlacement(path=sample, width_mm=width, height_mm=40, layer="顶层")
    )


def pick(path: Path) -> None:
    """Make the next save dialog answer with ``path``."""
    g.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(path), ""))


failures = []


def check(label, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"[{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


# ---- board 1: project + order file -----------------------------------------
ready(60)
pick(work / "工程A.epro2")
a_project = g.export_project(w)
check("板1 另存为工程", Path(a_project).exists(), a_project)

pick(work / "下单A.zip")
a_order = g.export_gerber_project(w)
check("板1 另存为下单文件", Path(a_order).exists(), a_order)
check(
    "下单只留一个 zip",
    [p.name for p in work.iterdir() if p.suffix == ".zip"] == ["下单A.zip"],
    str(sorted(p.name for p in work.iterdir())),
)
with zipfile.ZipFile(a_order) as bundle:
    packed = set(bundle.namelist())
check(
    "zip 内含板框 + 铜层",
    {"Gerber_BoardOutlineLayer.GKO", "Gerber_TopLayer.GTL"} <= packed,
    str(sorted(packed)),
)
check("板1 标签用工程名，不跟下单文件走",
      w.project_tabs.tabText(0) == "工程A", w.project_tabs.tabText(0))
check("板1 不再显示未保存", not w.project_tabs.tabText(0).startswith("●"))

first = w.active_project
check("板1 同时落了草稿", bool(first.draft_path) and Path(first.draft_path).exists())

# ---- board 2: a fresh tab --------------------------------------------------
g.new_project_tab(w)
check("新板开在新标签", len(w.workspace) == 2)
check("原来的板还在", first in w.workspace.projects)
check("新标签是空的", not w.design_canvas.board.outline_ready)

ready(100)
check("板2 独立于板1", w.design_canvas.board.outline.width_mm == 100.0)
pick(work / "工程B.epro2")
b_project = g.export_project(w)
check("板2 另存为工程", Path(b_project).exists(), b_project)
check(
    "两个标签名字不同",
    [w.project_tabs.tabText(i) for i in range(2)] == ["工程A", "工程B"],
    str([w.project_tabs.tabText(i) for i in range(2)]),
)

# ---- switching back and forth ---------------------------------------------
w.project_tabs.setCurrentIndex(0)
app.processEvents()
check("切回板1", w.design_canvas.board is first.board)
check("板1 几何没变", w.design_canvas.board.outline.width_mm == 60.0)
check("板1 图片还在", len(w.design_canvas.board.placements) == 1)
check("板1 草稿路径跟过来了", w.draft_panel.current_path() == first.draft_path)

w.project_tabs.setCurrentIndex(1)
app.processEvents()
check("切到板2", w.design_canvas.board.outline.width_mm == 100.0)

# ---- a fresh window must see both drafts -----------------------------------
w.hide()
fresh = g.build_window()
fresh._draft_store = DraftStore(work / "library")
fresh.setAttribute(Qt.WA_DontShowOnScreen, True)
fresh.resize(1400, 900)
fresh.show()
g.refresh_workspace_state(fresh)
app.processEvents()
names = sorted(info.name for info in fresh._draft_store.list())
check("重开后草稿库认得两份", names == ["工程A", "工程B"], str(names))

g.open_draft_file(fresh, str(fresh._draft_store.path_for("工程B")))
app.processEvents()
check(
    "重开后能打开板2",
    fresh.design_canvas.board.outline.width_mm == 100.0,
    str(fresh.design_canvas.board.outline.width_mm),
)
check(
    "打开后是新标签不是替换",
    len(fresh.workspace) == 2 and fresh.project_tabs.count() == 2,
    f"{len(fresh.workspace)} tabs",
)

# ---- quitting with several dirty boards ------------------------------------
g.new_project_tab(fresh)
fresh.design_canvas.board.outline_ready = True
fresh.design_canvas.board.outline = BoardOutlineSpec(width_mm=33, height_mm=22)
dirty_before = len(fresh.workspace.dirty())
check("未保存的板子被计数", dirty_before >= 1, str(dirty_before))
saved_ok = g.confirm_discard_all(fresh)
app.processEvents()
check(
    "退出前全部处理完",
    saved_ok and fresh.workspace.dirty() == [],
    f"ok={saved_ok} dirty={len(fresh.workspace.dirty())}",
)
# The fresh window's own empty starter tab has never been edited, so it has
# nothing worth writing; the boards that were worked on do.
check(
    "编辑过的板都有草稿",
    all(p.draft_path for p in fresh.workspace.projects if p.title not in (
        "未命名", "未命名 2", "未命名 3",
    )),
    str([(p.title, bool(p.draft_path)) for p in fresh.workspace.projects]),
)

fresh.hide()
print()
print("FAILURES:", failures if failures else "none")
raise SystemExit(1 if failures else 0)
