"""Exercise run.bat in a real cmd process, without running the full suite."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
LAUNCHER = Path(__file__).resolve().parents[1] / "run.bat"


@pytest.fixture
def launcher_env(tmp_path):
    project = tmp_path / "project 中文 with spaces"
    project.mkdir()
    shutil.copy2(LAUNCHER, project / "run.bat")
    modules = tmp_path / "modules"
    package = modules / "pytest"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['LAUNCHER_RECORD'], 'w', encoding='utf-8').write("
        "json.dumps([os.getcwd(), sys.argv[1:]]))\n"
        "sys.exit(int(os.environ.get('LAUNCHER_EXIT', '0')))\n", encoding="utf-8")
    (modules / "sitecustomize.py").write_text(
        "import os, sys\n"
        "if os.environ.get('LAUNCHER_BAD_VERSION'): sys.version_info = (3, 13, 0)\n"
        "if os.environ.get('LAUNCHER_NO_PYTEST'):\n"
        " class BlockPytest:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "   if fullname == 'pytest': raise ImportError('blocked for test')\n"
        " sys.meta_path.insert(0, BlockPytest())\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PCBART_PYTHON=sys.executable, PYTHONPATH=str(modules),
               PYTHONNOUSERSITE="1", LAUNCHER_RECORD=str(tmp_path / "record.json"),
               PCBART_DRAFTS_DIR=str(tmp_path / "drafts"))
    env.pop("PYTHONHOME", None)
    return project, env


def launch(project, env):
    return subprocess.run([os.environ["COMSPEC"], "/d", "/c", str(project / "run.bat"), "test"],
                          cwd=project.parent, env=env, capture_output=True, text=True, timeout=30)


def test_missing_interpreter_is_failure(launcher_env):
    project, env = launcher_env
    env["PCBART_PYTHON"] = str(project / "missing.exe")
    result = launch(project, env)
    assert result.returncode != 0
    assert "PCBART_PYTHON" in result.stderr


def test_unsupported_interpreter_is_failure(launcher_env):
    project, env = launcher_env
    env["LAUNCHER_BAD_VERSION"] = "1"
    result = launch(project, env)
    assert result.returncode != 0
    assert "3.11 or 3.12" in result.stderr


def test_missing_pytest_is_failure(launcher_env):
    project, env = launcher_env
    env["LAUNCHER_NO_PYTEST"] = "1"
    result = launch(project, env)
    assert result.returncode != 0
    assert "pytest is not installed" in result.stderr


@pytest.mark.parametrize("status", [0, 1, 5])
def test_pytest_status_and_project_cwd(launcher_env, status):
    import json
    project, env = launcher_env
    env["LAUNCHER_EXIT"] = str(status)
    result = launch(project, env)
    assert result.returncode == status, result.stderr
    cwd, args = json.loads(Path(env["LAUNCHER_RECORD"]).read_text(encoding="utf-8"))
    assert Path(cwd) == project
    assert args == ["tests", "-q"]


def test_automatic_discovery_with_no_python_is_failure(launcher_env, tmp_path):
    project, env = launcher_env
    env.pop("PCBART_PYTHON")
    # Isolate the fixed-drive candidates too, regardless of host installations.
    script = project / "run.bat"
    content = script.read_text(encoding="ascii")
    for version in ("312", "311"):
        candidate = rf"C:\Python{version}\python.exe"
        assert candidate in content
        content = content.replace(candidate, str(tmp_path / f"missing-{version}" / "python.exe"))
    script.write_text(content, encoding="ascii")
    env["PATH"] = str(tmp_path / "empty-path")
    env["LOCALAPPDATA"] = str(tmp_path / "empty-localappdata")
    result = launch(project, env)
    assert result.returncode != 0
    assert "Could not find Python 3.11 or 3.12" in result.stderr


def test_cli_arguments_and_exit_status(launcher_env):
    import json
    project, env = launcher_env
    (project / "tools").mkdir()
    (project / "tools" / "make_star.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['LAUNCHER_RECORD'], 'w', encoding='utf-8').write("
        "json.dumps([os.getcwd(), sys.argv[1:]]))\n"
        "sys.exit(7)\n", encoding="utf-8")
    command = subprocess.list2cmdline([str(project / "run.bat"), "star", "--edge", "12",
                                      "中文 with spaces"])
    result = subprocess.run(f'"{os.environ["COMSPEC"]}" /d /s /c "{command}"',
                            cwd=project.parent, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stderr
    cwd, args = json.loads(Path(env["LAUNCHER_RECORD"]).read_text(encoding="utf-8"))
    assert Path(cwd) == project
    assert args == ["--outdir", "out", "--edge", "12", "中文 with spaces"]
