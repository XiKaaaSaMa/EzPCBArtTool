"""Entry point for the packaged application.

PyInstaller needs a script to start from, and pointing it at
``tools/gui_mockup.py`` works but drags in that module's awkward position:
when it is run as a script its directory is on ``sys.path`` and the sibling
imports resolve by bare name, while a packaged build resolves them as
``tools.*``. Starting from the project root means both behave the same way.

Running this file directly also works, so there is one command to remember:
``python main.py``.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    """Start the desktop application."""
    # Required before anything else on a frozen Windows build: without it a
    # child process spawned by a library would re-run the whole application
    # instead of the worker. Nothing here spawns one today, but the guard costs
    # nothing and the failure it prevents is a window that multiplies itself.
    multiprocessing.freeze_support()

    from tools.gui_mockup import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
