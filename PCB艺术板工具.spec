# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the PCB art board tool.

Built as a **directory** rather than a one-file executable, on purpose. A
one-file build unpacks roughly 250 MB into a temporary folder on every launch
and deletes it again on exit, which costs several seconds of cold start and
makes the data-path question harder: anything cached beside the running
program would vanish. A directory build starts immediately and keeps its
files in a predictable place.

The bulk of the size is not Python -- it is Qt and OpenCV. Most of the
excluded modules below are Qt components this application never imports; left
in, they add a few hundred megabytes of DLLs for no benefit. Each exclusion is
listed with what would break if it were needed, so a future feature that
reaches for one of them fails loudly at build review rather than silently at
run time.
"""

from pathlib import Path
import ssl

ssl_rejected_cas = Path(ssl.__file__).with_name("codex-rejected-cas.pem")
ssl_datas = [(str(ssl_rejected_cas), ".")] if ssl_rejected_cas.is_file() else []

# ---------------------------------------------------------------- exclusions

# Qt modules the application does not use. `gui_mockup` imports QtCore, QtGui
# and QtWidgets only; the SVG plugin comes along for icon rendering.
EXCLUDED_QT = [
    "PySide6.QtWebEngineCore",     # embedded browser
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtQuick",             # QML / Qt Quick
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",            # Qt 3D, ~100 MB of DLLs on its own
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtMultimedia",        # audio / video
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtOpenGL",            # the canvas paints with QPainter
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtSensors",
    "PySide6.QtNetwork",           # nothing makes a network call
    "PySide6.QtWebSockets",
    "PySide6.QtStateMachine",
    "PySide6.QtConcurrent",
    "PySide6.QtPrintSupport",      # no printing, only PNG export
]

# Development-only packages. `pytest` and friends must never ship.
EXCLUDED_DEV = [
    "pytest",
    "_pytest",
    "py",
    "IPython",
    "IPython.lib",
    "jedi",
    "parso",
    "pexpect",
    "matplotlib",
    "pandas",
    "scipy",
    "setuptools",
    "pip",
    "wheel",
    "tkinter",                     # not used; also pulls Tcl/Tk
    "_tkinter",
]

# The video stack. OpenCV ships a 29 MB ffmpeg DLL that only video decoding
# needs; this tool reads still images.
EXCLUDED_CV = [
    "cv2.gapi",
    "cv2.mat_wrapper",
    "cv2.ml",
    "cv2.ocl",
    "cv2.typing",
    "cv2.utils",
    "cv2.videoio_registry",
]

excludes = EXCLUDED_QT + EXCLUDED_DEV + EXCLUDED_CV

# The Qt DLLs behind the excluded Python modules. PyInstaller trims by module,
# but a few of these are pulled in by name from the Qt plugin set.
EXCLUDED_QT_DLL_PATTERNS = [
    "Qt6WebEngine*",
    "Qt6Quick*",
    "Qt6Qml*",
    "Qt63D*",
    "Qt6Multimedia*",
    "Qt6Charts*",
    "Qt6DataVisualization*",
    "Qt6Graphs*",
    "Qt6Bluetooth*",
    "Qt6Nfc*",
    "Qt6Positioning*",
    "Qt6Location*",
    "Qt6SerialPort*",
    "Qt6SerialBus*",
    "Qt6RemoteObjects*",
    "Qt6Scxml*",
    "Qt6Sql*",
    "Qt6Test*",
    "Qt6Designer*",
    "Qt6UiTools*",
    "Qt6Help*",
    "Qt6OpenGL*",
    "Qt6Pdf*",
    "Qt6SpatialAudio*",
    "Qt6TextToSpeech*",
    "Qt6Sensors*",
    "Qt6Network*",
    "Qt6WebSockets*",
    "Qt6StateMachine*",
    "Qt6Concurrent*",
    "Qt6PrintSupport*",
    "Qt6QuickControls2*",
    "opencv_videoio_ffmpeg*",
    # The software OpenGL rasteriser, 19.7 MB. Qt loads it only if hardware
    # OpenGL is missing, and the canvas paints with QPainter's raster engine
    # (QtOpenGL is excluded above), so nothing here asks for it. Verified by
    # removing it and re-rendering: the screenshot is byte-identical.
    "opengl32sw*",
    # Keep OpenSSL DLLs: Python _ssl depends on them for HTTPS.
]


def trim_qt_dlls(coll):
    """Drop the DLLs behind the excluded modules.

    Qt's own plugin loader can pull a library in by name even when no Python
    module references it, so excluding the module alone was not enough to
    shrink the build.
    """
    import fnmatch

    kept = []
    removed = []
    for entry in coll:
        name = Path(entry[0]).name
        if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_QT_DLL_PATTERNS):
            removed.append(name)
        else:
            kept.append(entry)
    if removed:
        print(f"  trimmed {len(removed)} Qt/OpenCV DLLs")
    return kept


# ------------------------------------------------------------------ analysis

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    # java_lineart ships so 【滤镜】 works on a machine with a JRE but no
    # Python: the bridge calls it as a subprocess (pcbart/java_filters.py),
    # which makes it invisible to PyInstaller's import analysis. See
    # docs/PACKAGING.md section 四.5.
    datas=[
        ("java_lineart", "java_lineart"),
        ("assets/明日方舟-艺术板素材", "assets/明日方舟-艺术板素材"),
        ("assets/通行证模板源文件", "assets/通行证模板源文件"),
    ] + ssl_datas,
    # tools is a package (tools/__init__.py), and the application imports
    # tools.design_canvas by that full path so a frozen build can resolve it
    # without a source directory on sys.path.
    hiddenimports=["tools.design_canvas"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=2,
)

a.binaries = trim_qt_dlls(a.binaries)

# Qt needs its platform plugin to start at all, and the image-format plugins
# to load the user's PNG and JPEG artwork. Everything else in the plugin
# folders is for features this tool does not have.
NEEDED_PLUGIN_DIRS = (
    "platforms",       # qwindows: without it the app cannot open a window
    "styles",          # the native Windows widget style
    "imageformats",    # png, jpeg, webp for the artwork the user imports
    "iconengines",     # SVG icons in the toolbar
    "tls",             # Qt loads it even with no network use
)

pyz = PYZ(a.pure)

for entry in list(a.datas):
    destination = str(entry[0]).replace("\\", "/")
    if "/plugins/" in destination or "/translations/" in destination:
        plugin = destination.rsplit("/", 1)[-1] if "/plugins/" in destination else ""
        parent = destination.rsplit("/", 2)[-2] if destination.count("/") >= 2 else ""
        keep = parent in NEEDED_PLUGIN_DIRS or plugin in NEEDED_PLUGIN_DIRS
        if not keep:
            a.datas.remove(entry)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    # Executable file name. Matches the repository name (EzPCBArtTool) so the
    # artifact and the source repo read the same. The **displayed** name stays
    # Chinese -- see the window title in tools/gui_mockup.py.
    # Changing this renames the installed .exe and the build folder, not the
    # user data directory (pcbart.paths.APP_DIR_NAME), which is left alone so
    # existing drafts keep working.
    name="EzPCBArtTool",
    # The taskbar, title bar, desktop shortcut and Explorer all read this.
    # Built by make_icon.py from one source picture, with every size Windows
    # asks for -- a single size gets scaled and comes out blurred.
    icon="icon/app.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX on Qt DLLs is known to break plugin loading
    console=False,      # windowed: no black console box behind the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EzPCBArtTool",
)
