"""Command-line tools and the desktop application.

Present so ``tools.design_canvas`` is importable as a package path. The
widget module imports it that way because a packaged build has no source
directory to put on ``sys.path``; running from the repository, the
``tools/`` directory is added to the path instead and the bare name works.
"""
