"""The interface's icons, as vector paths.

**Generated -- do not edit by hand.** Rebuild with
```
cd out/_uigallery && node gen-icons.mjs
```

Two sets are in use, by the user's choice:

* **Iconoir** (MIT) for everything.
* **Lucide** (ISC) for `layers` alone -- Iconoir has no layer icon, and the
  layer panel is the one place in this tool where that concept is unavoidable.

Both licences allow commercial use and require no attribution.

Why the paths are inlined instead of shipped as `.svg` files: a packaged
build unpacks into a temporary directory, so a file loaded at runtime has to be
found through PyInstaller's bundle rules, and an icon that misses that lookup
is an invisible button. A generated Python module cannot miss -- it is imported
like any other source. It also means the icons re-colour with the theme for
free, because :func:`icon` paints the path rather than loading a raster.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Icon", "ICONS", "icon_names", "icon"]


@dataclass(frozen=True)
class Icon:
    """One icon: which set it came from, and the path data to draw."""

    set: str
    name: str
    width: int
    height: int
    body: str
    """SVG body: path/line/circle elements, already in the 0..width box.

    Authored as *strokes* in both sets, so the renderer strokes it rather than
    filling it -- see :func:`pcbart.ui_theme.draw_icon`.
    """


ICONS: dict[str, Icon] = {
    "open_image": Icon("iconoir", "media-image-plus", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M13 21H3.6a.6.6 0 0 1-.6-.6V3.6a.6.6 0 0 1 .6-.6h16.8a.6.6 0 0 1 .6.6V13\"/><path d=\"m3 16l7-3l5.5 2.5M16 10a2 2 0 1 1 0-4a2 2 0 0 1 0 4m0 9h3m3 0h-3m0 0v-3m0 3v3\"/></g>"),
    "preprocess": Icon("iconoir", "control-slider", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path d=\"m6.755 17.283l-1.429-10A2 2 0 0 1 7.306 5h3.388a2 2 0 0 1 1.98 2.283l-1.429 10A2 2 0 0 1 9.265 19h-.53a2 2 0 0 1-1.98-1.717Z\"/><path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M2 12h4m16 0H12\"/></g>"),
    "board_setup": Icon("iconoir", "frame", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-miterlimit=\"1.5\" stroke-width=\"1.5\"><path d=\"M4.998 2.001H2v2.998h2.998zm0 8.501H2V13.5h2.998zM20.498 5v5.503M3.5 5v5.503m16.998 2.999v5.502M3.5 13.502v5.502m1.499 1.498h5.5\"/><path stroke-width=\"1.22\" d=\"M4.999 3.503h5.5\"/><path d=\"M13.498 20.499h5.5\"/><path stroke-width=\"1.22\" d=\"M13.498 3.501h5.5\"/><path d=\"M4.998 19.001H2v2.998h2.998zM21.997 2.002H19V5h2.998zM13.497 2H10.5v2.998h2.998zm8.5 8.503H19V13.5h2.998zm0 8.499H19V22h2.998zm-8.5-.002H10.5v2.998h2.998z\"/></g>"),
    "fit_window": Icon("iconoir", "maximize", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M7 4H4v3m13-3h3v3M7 20H4v-3m13 3h3v-3\"/>"),
    "save_project": Icon("iconoir", "floppy-disk", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path d=\"M3 19V5a2 2 0 0 1 2-2h11.172a2 2 0 0 1 1.414.586l2.828 2.828A2 2 0 0 1 21 7.828V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z\"/><path d=\"M8.6 9h6.8a.6.6 0 0 0 .6-.6V3.6a.6.6 0 0 0-.6-.6H8.6a.6.6 0 0 0-.6.6v4.8a.6.6 0 0 0 .6.6ZM6 13.6V21h12v-7.4a.6.6 0 0 0-.6-.6H6.6a.6.6 0 0 0-.6.6Z\"/></g>"),
    "export_gerber": Icon("iconoir", "download", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M6 20h12M12 4v12m0 0l3.5-3.5M12 16l-3.5-3.5\"/>"),
    "settings": Icon("iconoir", "settings", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M12 15a3 3 0 1 0 0-6a3 3 0 0 0 0 6\"/><path d=\"m19.622 10.395l-1.097-2.65L20 6l-2-2l-1.735 1.483l-2.707-1.113L12.935 2h-1.954l-.632 2.401l-2.645 1.115L6 4L4 6l1.453 1.789l-1.08 2.657L2 11v2l2.401.656L5.516 16.3L4 18l2 2l1.791-1.46l2.606 1.072L11 22h2l.604-2.387l2.651-1.098C16.697 18.832 18 20 18 20l2-2l-1.484-1.75l1.098-2.652l2.386-.62V11z\"/></g>"),
    "layers": Icon("lucide", "layers", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\"><path d=\"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z\"/><path d=\"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12\"/><path d=\"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17\"/></g>"),
    "new_board": Icon("iconoir", "page-plus", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M4 12V2.6a.6.6 0 0 1 .6-.6h11.652a.6.6 0 0 1 .424.176l3.148 3.148A.6.6 0 0 1 20 5.75V21.4a.6.6 0 0 1-.6.6H11\"/><path d=\"M16 2v3.4a.6.6 0 0 0 .6.6H20M1.992 19h3m3 0h-3m0 0v-3m0 3v3\"/></g>"),
    "open_folder": Icon("iconoir", "folder", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M2 11V4.6a.6.6 0 0 1 .6-.6h6.178a.6.6 0 0 1 .39.144l3.164 2.712a.6.6 0 0 0 .39.144H21.4a.6.6 0 0 1 .6.6V11M2 11v8.4a.6.6 0 0 0 .6.6h18.8a.6.6 0 0 0 .6-.6V11M2 11h20\"/>"),
    "save_draft": Icon("iconoir", "page", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M4 21.4V2.6a.6.6 0 0 1 .6-.6h11.652a.6.6 0 0 1 .424.176l3.148 3.148A.6.6 0 0 1 20 5.75V21.4a.6.6 0 0 1-.6.6H4.6a.6.6 0 0 1-.6-.6M8 10h8m-8 8h8m-8-4h4\"/><path d=\"M16 2v3.4a.6.6 0 0 0 .6.6H20\"/></g>"),
    "quit": Icon("iconoir", "log-out", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M12 12h7m0 0l-3 3m3-3l-3-3m3-3V5a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1\"/>"),
    "copy": Icon("iconoir", "copy", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M19.4 20H9.6a.6.6 0 0 1-.6-.6V9.6a.6.6 0 0 1 .6-.6h9.8a.6.6 0 0 1 .6.6v9.8a.6.6 0 0 1-.6.6\"/><path d=\"M15 9V4.6a.6.6 0 0 0-.6-.6H4.6a.6.6 0 0 0-.6.6v9.8a.6.6 0 0 0 .6.6H9\"/></g>"),
    "cut": Icon("iconoir", "cut", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M17 12h1m4 0h1M6.236 7a3 3 0 1 0-4.472-4a3 3 0 0 0 4.472 4m0 0L19 18M6.236 17a3 3 0 1 1-4.472 4a3 3 0 0 1 4.472-4m0 0L19 6\"/>"),
    "paste": Icon("iconoir", "paste-clipboard", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-width=\"1.5\"><path d=\"M8.5 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2.5\"/><path d=\"M8 6.4V4.5a.5.5 0 0 1 .5-.5c.276 0 .504-.224.552-.496C9.2 2.652 9.774 1 12 1s2.8 1.652 2.948 2.504c.048.272.276.496.552.496a.5.5 0 0 1 .5.5v1.9a.6.6 0 0 1-.6.6H8.6a.6.6 0 0 1-.6-.6Z\"/></g>"),
    "delete": Icon("iconoir", "trash", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m20 9l-1.995 11.346A2 2 0 0 1 16.035 22h-8.07a2 2 0 0 1-1.97-1.654L4 9m17-3h-5.625M3 6h5.625m0 0V4a2 2 0 0 1 2-2h2.75a2 2 0 0 1 2 2v2m-6.75 0h6.75\"/>"),
    "rotate": Icon("iconoir", "refresh", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M21.888 13.5C21.164 18.311 17.013 22 12 22C6.477 22 2 17.523 2 12S6.477 2 12 2c4.1 0 7.625 2.468 9.168 6\"/><path d=\"M17 8h4.4a.6.6 0 0 0 .6-.6V3\"/></g>"),
    "flip": Icon("iconoir", "flip", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M9.5 20H2L9.5 4zm10.625 0H22l-.937-2m-4.688 2H14.5v-2m0-6v2m3.75-2l.938 2m-2.813-6L14.5 4v4\"/>"),
    "centre": Icon("iconoir", "align-horizontal-spacing", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M3 22V2m18 20V2\"/><path d=\"M15 16H9a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2Z\"/></g>"),
    "duplicate": Icon("iconoir", "copy", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M19.4 20H9.6a.6.6 0 0 1-.6-.6V9.6a.6.6 0 0 1 .6-.6h9.8a.6.6 0 0 1 .6.6v9.8a.6.6 0 0 1-.6.6\"/><path d=\"M15 9V4.6a.6.6 0 0 0-.6-.6H4.6a.6.6 0 0 0-.6.6v9.8a.6.6 0 0 0 .6.6H9\"/></g>"),
    "zoom_in": Icon("iconoir", "zoom-in", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M8 11h3m3 0h-3m0 0V8m0 3v3m6 3l4 4M3 11a8 8 0 1 0 16 0a8 8 0 0 0-16 0\"/>"),
    "zoom_out": Icon("iconoir", "zoom-out", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m17 17l4 4M3 11a8 8 0 1 0 16 0a8 8 0 0 0-16 0m5 0h6\"/>"),
    "close": Icon("iconoir", "x", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path d=\"M16.82 20.768L3.753 3.968A.6.6 0 0 1 4.227 3h2.48a.6.6 0 0 1 .473.232l13.067 16.8a.6.6 0 0 1-.474.968h-2.48a.6.6 0 0 1-.473-.232Z\"/><path stroke-linecap=\"round\" d=\"M20 3L4 21\"/></g>"),
    "minimise": Icon("iconoir", "minus", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M6 12h12\"/>"),
    "maximise": Icon("iconoir", "maximize", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M7 4H4v3m13-3h3v3M7 20H4v-3m13 3h3v-3\"/>"),
    "restore": Icon("iconoir", "windows", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" d=\"M4 16.987V7.013a.6.6 0 0 1 .507-.593l14.8-2.313A.6.6 0 0 1 20 4.7v14.598a.6.6 0 0 1-.693.593l-14.8-2.313A.6.6 0 0 1 4 16.986ZM4 12h16m-9.5-6.5v13\"/>"),
    "visible": Icon("iconoir", "eye", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M3 13c3.6-8 14.4-8 18 0\"/><path d=\"M12 17a3 3 0 1 1 0-6a3 3 0 0 1 0 6\"/></g>"),
    "hidden": Icon("iconoir", "eye-closed", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m19.5 16l-2.475-3.396M12 17.5V14m-7.5 2l2.469-3.388M3 8c3.6 8 14.4 8 18 0\"/>"),
    "lock": Icon("iconoir", "lock", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M16 12h1.4a.6.6 0 0 1 .6.6v6.8a.6.6 0 0 1-.6.6H6.6a.6.6 0 0 1-.6-.6v-6.8a.6.6 0 0 1 .6-.6H8m8 0V8c0-1.333-.8-4-4-4S8 6.667 8 8v4m8 0H8\"/>"),
    "grid": Icon("iconoir", "grid-plus", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"M13.992 17h3m3 0h-3m0 0v-3m0 3v3\"/><path d=\"M4 9.4V4.6a.6.6 0 0 1 .6-.6h4.8a.6.6 0 0 1 .6.6v4.8a.6.6 0 0 1-.6.6H4.6a.6.6 0 0 1-.6-.6Zm0 10v-4.8a.6.6 0 0 1 .6-.6h4.8a.6.6 0 0 1 .6.6v4.8a.6.6 0 0 1-.6.6H4.6a.6.6 0 0 1-.6-.6Zm10-10V4.6a.6.6 0 0 1 .6-.6h4.8a.6.6 0 0 1 .6.6v4.8a.6.6 0 0 1-.6.6h-4.8a.6.6 0 0 1-.6-.6Z\"/></g>"),
    "ruler": Icon("iconoir", "ruler", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M16 7V2.6a.6.6 0 0 0-.6-.6H8.6a.6.6 0 0 0-.6.6v18.8a.6.6 0 0 0 .6.6h6.8a.6.6 0 0 0 .6-.6V17m0-10h-3m3 0v5m0 0h-3m3 0v5m0 0h-3\"/>"),
    "chip": Icon("iconoir", "cpu", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M8 15.4V8.6a.6.6 0 0 1 .6-.6h6.8a.6.6 0 0 1 .6.6v6.8a.6.6 0 0 1-.6.6H8.6a.6.6 0 0 1-.6-.6\"/><path d=\"M20 4.6v14.8a.6.6 0 0 1-.6.6H4.6a.6.6 0 0 1-.6-.6V4.6a.6.6 0 0 1 .6-.6h14.8a.6.6 0 0 1 .6.6M17 4V2m-5 2V2M7 4V2m0 18v2m5-2v2m5-2v2m3-5h2m-2-5h2m-2-5h2M4 17H2m2-5H2m2-5H2\"/></g>"),
    "palette": Icon("iconoir", "palette", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path d=\"M20.51 9.54a1.9 1.9 0 0 1-1 1.09A7 7 0 0 0 15.37 17q.002.707.14 1.4a2.16 2.16 0 0 1-.31 1.65a1.8 1.8 0 0 1-1.21.8q-.804.15-1.62.15a9 9 0 0 1-9-9.28A9.05 9.05 0 0 1 11.85 3h.51a9 9 0 0 1 8.06 5a2 2 0 0 1 .09 1.52z\"/><path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"m8 16.01l.01-.011M6 12.01l.01-.011M8 8.01l.01-.011M12 6.01l.01-.011M16 8.01l.01-.011\"/></g>"),
    "magic": Icon("iconoir", "magic-wand", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path stroke-linecap=\"round\" d=\"m3 21l10-10m5-5l-2.5 2.5\"/><path d=\"m9.5 2l.945 2.555L13 5.5l-2.555.945L9.5 9l-.945-2.555L6 5.5l2.555-.945zm9.5 8l.54 1.46L21 12l-1.46.54L19 14l-.54-1.46L17 12l1.46-.54z\"/></g>"),
    "tools": Icon("iconoir", "tools", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"m10.05 10.607l-7.07 7.07a2 2 0 0 0 0 2.83v0a2 2 0 0 0 2.828 0l7.07-7.072m4.315.365l3.878 3.878a2 2 0 0 1 0 2.828v0a2 2 0 0 1-2.828 0l-6.209-6.208M6.733 5.904L4.61 6.61L2.49 3.075l1.414-1.414L7.44 3.782zm0 0l2.83 2.83\"/><path d=\"M10.05 10.607c-.844-2.153-.679-4.978 1.061-6.718s4.95-2.121 6.717-1.06l-3.04 3.04l-.283 3.111l3.111-.282l3.04-3.041c1.062 1.768.68 4.978-1.06 6.717c-1.74 1.74-4.564 1.905-6.717 1.061\"/></g>"),
    "erase": Icon("iconoir", "erase", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M21 21H9m6.889-6.11L8.464 7.463m-5.571 5.144l9.193-9.193a2 2 0 0 1 2.828 0l4.95 4.95a2 2 0 0 1 0 2.828l-9.243 9.243a1.93 1.93 0 0 1-2.728 0l-5-5a2 2 0 0 1 0-2.828\"/>"),
    "hand": Icon("iconoir", "drag-hand-gesture", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"m7 10.5l-2.004 2.672a2 2 0 0 0 .126 2.552l3.784 4.128c.378.413.912.648 1.473.648H15c2.4 0 4-1.5 4-4q0 0 0 0V7.929M16 8.5v-.571c0-2.286 3-2.286 3 0\"/><path d=\"M13 8.5V7.027m0-.527v.527M16 8.5V7.027c0-2.286-3-2.286-3 0\"/><path d=\"M13 8.5V7.027c0-2.286 3-2.286 3 0V8.5m-6 0v-2c0-2.286 3-2.286 3 0q0 0 0 0v2m-6 5v-7A1.5 1.5 0 0 1 8.5 5v0c.828 0 1.5.555 1.5 1.384V8.5\"/></g>"),
    "pointer": Icon("iconoir", "cursor-pointer", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" d=\"M19.503 9.97c1.204.489 1.112 2.224-.137 2.583l-6.305 1.813l-2.88 5.895c-.571 1.168-2.296.957-2.569-.314L4.677 6.257A1.369 1.369 0 0 1 6.53 4.7z\" clip-rule=\"evenodd\"/>"),
    "help": Icon("iconoir", "help-circle", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path d=\"M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2S2 6.477 2 12s4.477 10 10 10\"/><path d=\"M9 9c0-3.5 5.5-3.5 5.5 0c0 2.5-2.5 2-2.5 5m0 4.01l.01-.011\"/></g>"),
    "info": Icon("iconoir", "info-circle", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M12 11.5v5m0-8.99l.01-.011M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2S2 6.477 2 12s4.477 10 10 10\"/>"),
    "warning": Icon("iconoir", "warning-triangle", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-width=\"1.5\"><path d=\"M20.043 21H3.957c-1.538 0-2.5-1.664-1.734-2.997l8.043-13.988c.77-1.337 2.699-1.337 3.468 0l8.043 13.988C22.543 19.336 21.58 21 20.043 21ZM12 9v4\"/><path stroke-linejoin=\"round\" d=\"m12 17.01l.01-.011\"/></g>"),
    "check": Icon("iconoir", "check", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m5 13l4 4L19 7\"/>"),
    "plus": Icon("iconoir", "plus", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M6 12h6m6 0h-6m0 0V6m0 6v6\"/>"),
    "minus": Icon("iconoir", "minus", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M6 12h12\"/>"),
    "search": Icon("iconoir", "search", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m17 17l4 4M3 11a8 8 0 1 0 16 0a8 8 0 0 0-16 0\"/>"),
    "expand": Icon("iconoir", "expand", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M9 9L4 4m0 0v4m0-4h4m7 5l5-5m0 0v4m0-4h-4M9 15l-5 5m0 0v-4m0 4h4m7-5l5 5m0 0v-4m0 4h-4\"/>"),
    "collapse": Icon("iconoir", "collapse", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"m20 20l-5-5m0 0v4m0-4h4M4 20l5-5m0 0v4m0-4H5M20 4l-5 5m0 0V5m0 4h4M4 4l5 5m0 0V5m0 4H5\"/>"),
    "swatch": Icon("iconoir", "palette", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"><path d=\"M20.51 9.54a1.9 1.9 0 0 1-1 1.09A7 7 0 0 0 15.37 17q.002.707.14 1.4a2.16 2.16 0 0 1-.31 1.65a1.8 1.8 0 0 1-1.21.8q-.804.15-1.62.15a9 9 0 0 1-9-9.28A9.05 9.05 0 0 1 11.85 3h.51a9 9 0 0 1 8.06 5a2 2 0 0 1 .09 1.52z\"/><path stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"m8 16.01l.01-.011M6 12.01l.01-.011M8 8.01l.01-.011M12 6.01l.01-.011M16 8.01l.01-.011\"/></g>"),
    "eyedropper": Icon("iconoir", "selective-tool", 24, 24,
        "<g fill=\"none\" stroke=\"currentColor\" stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"1.5\"><path fill=\"currentColor\" d=\"M12 13a1 1 0 1 0 0-2a1 1 0 0 0 0 2\"/><path d=\"M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2S2 6.477 2 12s4.477 10 10 10\"/></g>"),
    "custom": Icon("iconoir", "sparks", 24, 24,
        "<path fill=\"none\" stroke=\"currentColor\" stroke-linejoin=\"round\" stroke-width=\"1.5\" d=\"M8 15c4.875 0 7-2.051 7-7c0 4.949 2.11 7 7 7c-4.89 0-7 2.11-7 7c0-4.89-2.125-7-7-7ZM2 6.5c3.134 0 4.5-1.318 4.5-4.5c0 3.182 1.357 4.5 4.5 4.5c-3.143 0-4.5 1.357-4.5 4.5c0-3.143-1.366-4.5-4.5-4.5Z\"/>"),
}


def icon_names() -> tuple[str, ...]:
    """Every semantic name, so callers can assert against the whole set."""
    return tuple(ICONS)


def icon(name: str) -> Icon:
    """The icon registered under this semantic name.

    Raises rather than returning a blank placeholder: a missing icon means a
    name was typo'd or dropped from the generator, and both should be caught
    the moment they happen rather than showing the user an empty button.
    """
    try:
        return ICONS[name]
    except KeyError:
        raise KeyError(
            f"没有注册这个图标: {name}（可用: {', '.join(sorted(ICONS))}）"
        ) from None
