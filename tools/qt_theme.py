"""Qt side of the colour scheme: icons, control styling, palette application.

Split from :mod:`pcbart.ui_theme` on purpose. That module is pure Python so the
colour maths and the preferences file can be tested without Qt; this one holds
everything that needs a QApplication.

**Icons are drawn, not loaded.** :func:`icon` paints the vector path from
:mod:`pcbart.ui_icons` at the requested size and colour. Three things fall out
of that, all of which matter here:

* an icon re-colours with the theme for free -- there is no per-theme icon set
  to keep in sync;
* it stays crisp at any size or device pixel ratio, because it is rasterised on
  demand rather than scaled down from one bitmap;
* there is no file to find at runtime, which is the failure mode that makes a
  packaged build show empty buttons.

**The control styling** is Material Design 3's shape and colour language
expressed in Qt's stylesheet dialect, plus a small proxy style for the parts a
stylesheet cannot draw -- check marks, radio dots and the arrow glyphs. Those
are drawn rather than set as images so they scale with the display and do not
need a temporary file on disk.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QProxyStyle, QStyle

from pcbart.ui_icons import icon as icon_data

__all__ = [
    "icon",
    "icon_pixmap",
    "Md3Style",
    "app_stylesheet",
    "apply_scheme",
    "FONT_STACK",
]


#: Latin/digit face for the interface.
#:
#: Material specifies Roboto. It ships as woff2 in the font packages this
#: project can fetch, which Qt does not load, so the stack names Roboto first
#: for machines that have it and falls back to Segoe UI -- metrically close and
#: present on every Windows box. Chinese was never going to come from Roboto
#: (it has no CJK glyphs) and falls through to YaHei either way.
FONT_STACK = '"Roboto", "Segoe UI", "Microsoft YaHei UI", sans-serif'


def _require_gui_app() -> None:
    """Fail with a readable error when there is no QGuiApplication yet.

    Qt does not raise here -- it writes "Must construct a QGuiApplication before
    a QPixmap" to stderr and **aborts the process**. Under pytest that looks
    like a silent crash: no traceback, no failing test, just a non-zero exit
    and empty output, which is a miserable thing to debug. Checking first turns
    it into an ordinary exception that names the actual mistake.
    """
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        raise RuntimeError(
            "渲染图标前必须先有 QApplication —— 在测试里给用例加上 `app` fixture。"
        )


def icon_pixmap(name: str, colour: str, size: int, ratio: float = 1.0) -> QPixmap:
    """One icon, rasterised at ``size`` logical pixels in ``colour``.

    ``ratio`` is the device pixel ratio: the pixmap is rendered at
    ``size * ratio`` physical pixels and tagged with the ratio, so it stays
    sharp on a scaled display instead of being upscaled from logical size.
    """
    _require_gui_app()
    data = icon_data(name)
    physical = max(1, int(round(size * ratio)))
    markup = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {data.width} {data.height}" '
        f'width="{physical}" height="{physical}">'
        # The paths are authored with stroke="currentColor", which only means
        # anything inside a browser. Substituting the theme colour here is what
        # makes one icon set serve every theme.
        f'{data.body.replace("currentColor", colour)}'
        f"</svg>"
    ).encode("utf-8")

    pixmap = QPixmap(physical, physical)
    pixmap.fill(Qt.transparent)
    renderer = QSvgRenderer(markup)
    if renderer.isValid():
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter)
        painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def icon(name: str, colour: str, *, size: int = 18, disabled: str | None = None) -> QIcon:
    """A themed :class:`QIcon`, ready for a menu or toolbar entry.

    Several sizes are baked in so Qt can pick the right one for a toolbar, a
    menu and a high-DPI screen without resampling.
    """
    result = QIcon()
    ratios = (1.0, 1.5, 2.0, 3.0)
    for side in (16, 18, 20, 24, 32, 48):
        for ratio in ratios:
            result.addPixmap(icon_pixmap(name, colour, side, ratio))
    if disabled:
        # Qt normally fades a disabled icon by compositing it against the
        # background, which on a dark theme produces a muddy result. An
        # explicit grey keeps a disabled button readable as "present but not
        # available" instead of looking like a rendering fault.
        for side in (16, 18, 20, 24, 32, 48):
            for ratio in ratios:
                result.addPixmap(
                    icon_pixmap(name, disabled, side, ratio), QIcon.Disabled
                )
    return result


class Md3Style(QProxyStyle):
    """Draws the handful of controls a stylesheet cannot express.

    Check marks, radio dots and dropdown arrows are drawn here rather than
    supplied as images: an image would have to live somewhere on disk and be
    re-generated for every theme, while these primitives are a few shapes that
    scale cleanly at any DPI.
    """

    def __init__(self, scheme):
        super().__init__()
        self.scheme = scheme

    # ------------------------------------------------------------ primitives

    def drawPrimitive(self, element, option, painter, widget=None):  # noqa: N802
        colour = QColor(self.scheme["accent"])
        on_accent = QColor(self.scheme["accent_text"])
        border = QColor(self.scheme["border"])
        text = QColor(self.scheme["text"])

        if element == QStyle.PE_IndicatorCheckBox:
            self._draw_check(
                painter, option.rect, bool(option.state & QStyle.State_On),
                bool(option.state & QStyle.State_Enabled), colour, on_accent, border,
            )
            return

        if element == QStyle.PE_IndicatorItemViewItemCheck:
            self._draw_check(
                painter, option.rect, bool(option.state & QStyle.State_On),
                bool(option.state & QStyle.State_Enabled), colour, on_accent, border,
            )
            return

        if element == QStyle.PE_IndicatorRadioButton:
            self._draw_radio(
                painter, option.rect, bool(option.state & QStyle.State_On),
                bool(option.state & QStyle.State_Enabled), colour, on_accent, border,
            )
            return

        if element in (QStyle.PE_IndicatorArrowDown, QStyle.PE_IndicatorArrowUp):
            self._draw_arrow(
                painter, option.rect, colour if option.state & QStyle.State_Enabled
                else border, down=element == QStyle.PE_IndicatorArrowDown,
            )
            return

        if element in (QStyle.PE_IndicatorSpinUp, QStyle.PE_IndicatorSpinDown):
            self._draw_arrow(
                painter, option.rect, colour if option.state & QStyle.State_Enabled
                else border, down=element == QStyle.PE_IndicatorSpinDown,
            )
            return

        if element in (QStyle.PE_IndicatorBranch,):
            # The expand/collapse triangles in a tree. Drawn rather than left to
            # the native style, which paints the light-theme chevron on both.
            if option.state & QStyle.State_Children:
                opened = bool(option.state & QStyle.State_Open)
                self._draw_arrow(painter, option.rect, text, down=opened, small=True)
                return

        super().drawPrimitive(element, option, painter, widget)

    # --------------------------------------------------------------- shapes

    @staticmethod
    def _draw_check(painter, rect, checked, enabled, accent, on_accent, border):
        side = min(rect.width(), rect.height(), 18)
        box = QRectF(
            rect.center().x() - side / 2 + 0.5,
            rect.center().y() - side / 2 + 0.5,
            side - 1, side - 1,
        )
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if checked:
            fill = accent if enabled else QColor(accent).darker(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            # A 2 px radius is Material's small-component corner.
            painter.drawRoundedRect(box, 2.0, 2.0)

            pen = QPen(on_accent, max(1.6, side * 0.14))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(box.left() + side * 0.24, box.top() + side * 0.52)
            path.lineTo(box.left() + side * 0.43, box.top() + side * 0.71)
            path.lineTo(box.left() + side * 0.77, box.top() + side * 0.30)
            painter.drawPath(path)
        else:
            painter.setPen(QPen(border, 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box, 2.0, 2.0)
        painter.restore()

    @staticmethod
    def _draw_radio(painter, rect, checked, enabled, accent, on_accent, border):
        side = min(rect.width(), rect.height(), 18)
        circle = QRectF(
            rect.center().x() - side / 2 + 0.5,
            rect.center().y() - side / 2 + 0.5,
            side - 1, side - 1,
        )
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(accent if checked and enabled else border, 1.6))
        painter.drawEllipse(circle)
        if checked:
            inner = circle.adjusted(side * 0.28, side * 0.28, -side * 0.28, -side * 0.28)
            painter.setPen(Qt.NoPen)
            painter.setBrush(accent if enabled else QColor(accent).darker(160))
            painter.drawEllipse(inner)
        painter.restore()

    @staticmethod
    def _draw_arrow(painter, rect, colour, *, down: bool, small: bool = False):
        size = (5.0 if small else 6.0)
        cx, cy = rect.center().x(), rect.center().y()
        sign = 1.0 if down else -1.0
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(colour, 1.6)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(QPointF(cx - size, cy - sign * size * 0.45))
        path.lineTo(QPointF(cx, cy + sign * size * 0.55))
        path.lineTo(QPointF(cx + size, cy - sign * size * 0.45))
        painter.drawPath(path)
        painter.restore()


# --------------------------------------------------------------- stylesheet


def app_stylesheet(scheme) -> str:
    """The application-wide Material 3 styling.

    Written as one string rather than per-widget calls because Qt resolves a
    widget's stylesheet together with its ancestors': a rule here reaches every
    control, including ones created later, which is what makes a theme switch a
    single assignment instead of a walk over the widget tree.
    """
    s = scheme

    return f"""
    * {{
        font-family: {FONT_STACK};
    }}
    QWidget {{
        color: {s['text']};
        font-size: 12px;
    }}
    QMainWindow, QDialog {{
        background: {s['window']};
    }}
    QToolTip {{
        background: {s['panel_alt']};
        color: {s['text']};
        border: 1px solid {s['border']};
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 11px;
    }}

    /* ---------------------------------------------------------- menu bar */
    QMenuBar {{
        background: {s['panel_alt']};
        border-bottom: 0;
        padding: 2px 6px;
    }}
    QMenuBar::item {{
        padding: 5px 10px;
        border-radius: 6px;
        background: transparent;
    }}
    QMenuBar::item:selected {{ background: {s['hover']}; }}
    QMenuBar::item:pressed {{ background: {s['active']}; }}

    QMenu {{
        background: {s['panel_alt']};
        border: 1px solid {s['border']};
        border-radius: 8px;
        padding: 6px;
    }}
    QMenu::item {{
        padding: 6px 26px 6px 12px;
        border-radius: 6px;
        min-width: 148px;
    }}
    QMenu::item:selected {{ background: {s['active']}; }}
    QMenu::item:disabled {{ color: {s['disabled_text']}; }}
    QMenu::separator {{
        height: 1px;
        background: {s['border']};
        margin: 5px 10px;
    }}

    /* ---------------------------------------------------------- toolbar */
    QToolBar {{
        background: {s['panel']};
        border: 0;
        border-bottom: 1px solid {s['border']};
        padding: 5px 8px;
        spacing: 2px;
    }}
    QToolBar::separator {{
        width: 1px;
        background: {s['border']};
        margin: 4px 6px;
    }}
    QToolButton {{
        padding: 6px 11px;
        border: 1px solid transparent;
        border-radius: 18px;
        background: transparent;
        font-weight: 500;
    }}
    QToolButton:hover {{ background: {s['hover']}; }}
    QToolButton:pressed {{ background: {s['active']}; }}
    QToolButton:checked {{ background: {s['accent_soft']}; color: {s['accent_soft_text']}; }}
    QToolButton:disabled {{ color: {s['disabled_text']}; }}
    QToolButton::menu-indicator {{ image: none; width: 0; }}

    /* ---------------------------------------------------------- buttons */
    QPushButton {{
        background: transparent;
        border: 1px solid {s['border']};
        border-radius: 18px;
        padding: 6px 16px;
        font-weight: 500;
        min-height: 20px;
    }}
    QPushButton:hover {{ background: {s['hover']}; }}
    QPushButton:pressed {{ background: {s['active']}; }}
    QPushButton:disabled {{ color: {s['disabled_text']}; border-color: {s['border']}; }}
    QPushButton:default, QPushButton[accent="true"] {{
        background: {s['accent']};
        color: {s['accent_text']};
        border-color: transparent;
    }}
    QPushButton:default:hover, QPushButton[accent="true"]:hover {{
        background: {s['accent']};
    }}

    /* ----------------------------------------------------------- inputs */
    QLineEdit, QPlainTextEdit, QTextEdit {{
        background: {s['panel_alt']};
        border: 1px solid {s['border']};
        border-radius: 8px;
        padding: 5px 10px;
        selection-background-color: {s['accent']};
        selection-color: {s['accent_text']};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border: 2px solid {s['accent']};
        padding: 4px 9px;
    }}
    QLineEdit:disabled {{ color: {s['disabled_text']}; background: {s['disabled_bg']}; }}

    QComboBox {{
        background: {s['panel_alt']};
        border: 1px solid {s['border']};
        border-radius: 8px;
        padding: 5px 10px;
        min-height: 20px;
    }}
    QComboBox:hover {{ background: {s['hover']}; }}
    QComboBox:focus, QComboBox:on {{ border: 2px solid {s['accent']}; padding: 4px 9px; }}
    QComboBox::drop-down {{ border: 0; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {s['panel_alt']};
        border: 1px solid {s['border']};
        border-radius: 8px;
        padding: 4px;
        outline: none;
        selection-background-color: {s['active']};
        selection-color: {s['text']};
    }}

    QSpinBox, QDoubleSpinBox {{
        background: {s['panel_alt']};
        border: 1px solid {s['border']};
        border-radius: 8px;
        padding: 4px 6px 4px 10px;
        min-height: 20px;
        selection-background-color: {s['accent']};
        selection-color: {s['accent_text']};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {s['accent']};
        padding: 3px 5px 3px 9px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        width: 18px;
        border: 0;
        background: transparent;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {s['hover']};
        border-radius: 5px;
    }}

    QCheckBox, QRadioButton {{ spacing: 8px; padding: 2px 0; }}
    QCheckBox:disabled, QRadioButton:disabled {{ color: {s['disabled_text']}; }}

    QSlider::groove:horizontal {{
        height: 4px;
        border-radius: 2px;
        background: {s['border']};
    }}
    QSlider::sub-page:horizontal {{
        height: 4px;
        border-radius: 2px;
        background: {s['accent']};
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 7px;
        background: {s['accent']};
    }}
    QSlider::handle:horizontal:hover {{ background: {s['accent_soft_text']}; }}
    QSlider:disabled::sub-page:horizontal,
    QSlider:disabled::handle:horizontal {{ background: {s['disabled_bg']}; }}

    /* ---------------------------------------------------------- grouping */
    QGroupBox {{
        border: 1px solid {s['border']};
        border-radius: 12px;
        margin-top: 12px;
        padding: 14px 10px 10px 10px;
        background: transparent;
        font-weight: 500;
    }}
    QGroupBox::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        padding: 1px 7px;
        left: 10px;
        color: {s['muted']};
        font-size: 11px;
        font-weight: 600;
    }}

    QTabWidget::pane {{
        border: 1px solid {s['border']};
        border-radius: 12px;
        background: {s['panel']};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {s['muted']};
        padding: 6px 16px;
        margin-right: 3px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }}
    QTabBar::tab:hover {{ background: {s['hover']}; color: {s['text']}; }}
    QTabBar::tab:selected {{
        background: {s['panel']};
        color: {s['text']};
        border: 1px solid {s['border']};
        border-bottom-color: {s['panel']};
    }}

    /* ----------------------------------------------------------- lists */
    QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView, QTableView {{
        background: transparent;
        border: 0;
        outline: none;
        alternate-background-color: {s['panel_alt']};
    }}
    QListWidget::item, QTreeWidget::item, QListView::item {{
        padding: 5px 8px;
        border-radius: 8px;
    }}
    QListWidget::item:hover, QTreeWidget::item:hover, QListView::item:hover {{
        background: {s['hover']};
    }}
    QListWidget::item:selected, QTreeWidget::item:selected, QListView::item:selected {{
        background: {s['active']};
        color: {s['text']};
    }}
    QHeaderView::section {{
        background: {s['panel_alt']};
        color: {s['muted']};
        border: 0;
        border-bottom: 1px solid {s['border']};
        padding: 5px 8px;
        font-weight: 600;
        font-size: 11px;
    }}
    QTableWidget::item {{ padding: 3px 6px; }}
    QTableCornerButton::section {{ background: {s['panel_alt']}; border: 0; }}

    /* --------------------------------------------------------- scrollbars */
    QScrollArea {{ border: 0; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent; width: 12px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {s['border']};
        border-radius: 4px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {s['muted']}; }}
    QScrollBar:horizontal {{
        background: transparent; height: 12px; margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {s['border']};
        border-radius: 4px;
        min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {s['muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* ------------------------------------------------------------ misc */
    QStatusBar {{
        background: {s['panel']};
        color: {s['muted']};
        border-top: 1px solid {s['border']};
        font-size: 11px;
    }}
    QStatusBar::item {{ border: 0; }}
    QLabel {{ background: transparent; }}
    QSplitter::handle {{ background: {s['border']}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QProgressBar {{
        border: 0;
        border-radius: 4px;
        background: {s['border']};
        height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {s['accent']}; border-radius: 4px; }}
    """


def _qt_palette(scheme) -> QPalette:
    """A matching :class:`QPalette`.

    The stylesheet covers most painting, but a few things -- native dialogs, the
    text cursor, selection inside widgets the sheet does not reach -- come from
    the palette. Setting both keeps them from disagreeing.
    """
    palette = QPalette()
    mapping = {
        QPalette.Window: scheme["window"],
        QPalette.WindowText: scheme["text"],
        QPalette.Base: scheme["panel_alt"],
        QPalette.AlternateBase: scheme["panel"],
        QPalette.Text: scheme["text"],
        QPalette.PlaceholderText: scheme["muted"],
        QPalette.Button: scheme["panel"],
        QPalette.ButtonText: scheme["text"],
        QPalette.BrightText: scheme["accent"],
        QPalette.Highlight: scheme["accent"],
        QPalette.HighlightedText: scheme["accent_text"],
        QPalette.ToolTipBase: scheme["panel_alt"],
        QPalette.ToolTipText: scheme["text"],
        QPalette.Link: scheme["accent"],
        QPalette.LinkVisited: scheme["accent"],
    }
    for role, colour in mapping.items():
        palette.setColor(role, QColor(colour))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(scheme["disabled_text"]))
    palette.setColor(
        QPalette.Disabled, QPalette.ButtonText, QColor(scheme["disabled_text"])
    )
    palette.setColor(
        QPalette.Disabled, QPalette.WindowText, QColor(scheme["disabled_text"])
    )
    return palette


def style_window(window, scheme) -> None:
    """Apply the theme sheet to one top-level window.

    Called by :func:`apply_scheme` for every existing window, and by
    ``build_window`` for each new one. A sheet reaches a window's whole subtree,
    so this is the only per-window step.
    """
    window.setStyleSheet(app_stylesheet(scheme))


def apply_scheme(app, scheme, *, font_point_size: float | None = None) -> None:
    """Point the whole application at ``scheme``.

    ⚠️ **The proxy style is installed once and then updated in place.**
    ``QApplication.setStyle`` does not merely swap a pointer: Qt destroys the
    old style and re-polishes **every widget in the process**, which measured
    **1.7 seconds** on a window with four boards open -- and the cost grows with
    however many windows exist, including ones a test or a script has left
    behind. Switching a colour would visibly freeze the application.

    Mutating the existing style's scheme instead costs nothing, because the
    style only reads the scheme while painting a handful of primitives (check
    marks, radio dots, arrows) and those repaint on the next frame anyway. The
    widget sheet and the palette -- which are what actually recolour the
    interface -- are still re-applied here, and those are cheap.
    """
    app.setPalette(_qt_palette(scheme))

    # ⚠️ **The style is installed on the application, the sheet on the window,
    # and that split is not a preference -- it is forced.**
    #
    # ``QApplication.setStyleSheet`` makes Qt fall back to its own
    # ``QCommonStyle``: a custom ``QProxyStyle`` cannot survive an
    # application-level sheet, no matter what order the two are called in
    # (measured: set-sheet-then-style, style-then-sheet, and clearing the sheet
    # in between all end on ``QCommonStyle``). Everything the proxy draws --
    # check boxes, radio buttons, the spin and combo arrows -- then reverts to
    # the platform's rendering, which is where the Windows-blue check box came
    # from: the style existed, was installed, and was silently thrown away.
    #
    # Setting the same sheet on each top-level *window* leaves the proxy in
    # place and reaches every child just the same, so nothing is lost.
    existing = app.style()
    if isinstance(existing, Md3Style):
        existing.scheme = scheme
    else:
        app.setStyle(Md3Style(scheme))

    for window in app.topLevelWidgets():
        style_window(window, scheme)
        # The proxy draws a few primitives; they would keep the old colours
        # until something else happened to repaint them.
        window.update()

    # ⚠️ **Re-applying a sheet scales with the number of live widgets.** Qt
    # re-polishes every widget in the window, measured at ~40 ms for one window
    # and ~4 s for twenty. In the running application there is normally exactly
    # one window, so this is not a problem for the user; it became one only for
    # a test session that left a window per test alive. The suite's own fixtures
    # now free their windows (see tests/conftest.py), which is the right place
    # to fix that -- skipping the re-apply would leave the standard controls on
    # the previous theme, since the sheet carries colours.

    if font_point_size is not None:
        # The existing font is the base rather than a fresh ``QFont()``: an
        # empty font carries no family, and setting it would drop whatever
        # Chinese face the caller chose. Only the size is being changed here.
        font = QFont(app.font())
        font.setPointSizeF(float(font_point_size))
        app.setFont(font)
