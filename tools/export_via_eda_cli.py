"""Drive the local EasyEDA Pro client's CLI to produce a *manufacturable* package.

Why this exists
---------------
The 通行证 card needs **colour silkscreen**, and the colour data is written by
EasyEDA itself in the ``Fabrication_Colorful*`` files. That encoding was
measured to be encrypted or private (entropy 7.93-8.00 bits/byte, not zlib,
deflate, gzip, bz2, lzma, brotli, zstd, zip, PNG or JPEG), so this project
cannot write it -- and guessing an undocumented format is exactly the mistake
that produced a silently-dropped ``PAD`` record before.

EasyEDA Pro 4.1.60 ships a CLI (``lceda-pro``) that can export the package the
same way the GUI's 一键生成 does. That is the correct division of labour: this
project lays out the card, the EDA encodes the colour job.

What is verified working
------------------------
* ``lceda-pro doctor``                       -- bridge reachable
* ``lceda-pro open --path <file.eprj2>``     -- opens a project, returns a session
* ``lceda-pro invoke --ext-uuid eda``        -- runs JS with full permissions
  (⚠️ **this is what bypasses the "外部交互" GUI switch**; without it,
  ``saveFileToFileSystem`` throws "未启用扩展和独立脚本的外部交互权限")
* ``SYS_FileSystem.readFileFromFileSystem``  -- reads a file into a File object
* ``pcb_ManufactureData.getGerberFile(name, colorSilkscreen=true)``
  -- returns the whole package; with ``true`` it contains the four
  ``Fabrication_Colorful*`` files plus the ``Drill_*.DRL`` Excellon files.

What is NOT working (the blocker)
---------------------------------
``lceda-pro open`` accepts only ``.eprj/.eprj2/.eprj3/.elib`` and **rejects
``.epro2``**::

    unsupported file type: card.epro2
    (expected .eprj/.eprj2/.eprj3/.elib)

``SYS_FileManager.importProjectByProjectFile(file, 'EasyEDA Pro', ...)`` is the
documented way in -- its ``fileType`` union literally includes
``"EasyEDA Pro"`` -- but every spelling tried returns ``undefined`` in ~40 ms
without creating a project (``New Project`` with and without the required
``newProjectOwnerTeamUuid``, and ``Offline Client Local Path``). So the import
step still has to happen in the GUI. Everything from "open the PCB document"
onwards is automated here.

Usage
-----
After opening the card in the EDA (File -> 导入专业版 -> 选 .epro2) and telling
this script which PCB to export::

    python tools/export_via_eda_cli.py --session <id> --pcb <uuid> --out out/下单包
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

DEFAULT_CLI = Path(r"R:\嘉立创eda\lceda-pro\lceda-pro.exe")
"""The client executable. It serves both the GUI and the CLI."""


def run(cli: Path, *args: str, timeout: int = 120) -> dict:
    """One CLI call, parsed. Every command returns the same JSON envelope."""
    proc = subprocess.run(
        [str(cli), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    text = (proc.stdout or "").strip()
    # The envelope is the first line that parses; the client prints a version
    # banner before it on some subcommands.
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"ok": False, "error": {"message": text or (proc.stderr or "").strip()}}


def invoke(cli: Path, session: str, code: str, timeout: int = 60000) -> dict:
    """Run a JS snippet inside the editor, with the permissions the GUI has.

    ⚠️ ``--ext-uuid eda`` is load-bearing, not decoration. Without it the
    snippet runs under "independent script" permission, which has no filesystem
    access, and every ``saveFileToFileSystem`` fails with
    「未启用扩展和独立脚本的外部交互权限！」.
    """
    return run(
        cli,
        "invoke",
        "--session", session,
        "--ext-uuid", "eda",
        "--timeout", str(timeout),
        "--code", code,
        timeout=timeout // 1000 + 60,
    )


def doctor(cli: Path) -> dict:
    result = run(cli, "doctor")
    return (result.get("value") or {}) if result.get("ok") else {}


def sessions(cli: Path) -> list[dict]:
    result = run(cli, "session", "list")
    return (result.get("value") or []) if result.get("ok") else []


def export_package(cli: Path, session: str, pcb_uuid: str, target: Path) -> dict:
    """Open the PCB, export the colour-silkscreen package, save it to ``target``.

    The card's own artwork lives on the PCB document, so the document has to be
    the open one before ``getGerberFile`` is called -- from a project-level
    context it returns ``undefined``.
    """
    target_posix = str(target).replace("\\", "/")
    code = f"""
const tabId = await eda.dmt_EditorControl.openDocument('{pcb_uuid}');
const f = await eda.pcb_ManufactureData.getGerberFile('PCBart_Gerber', true);
if (!f) return {{ ok: false, reason: 'getGerberFile 返回空', tabId }};
const saved = await eda.sys_FileSystem.saveFileToFileSystem(
  '{target_posix}', f, '{target.name}', true);
return {{ ok: saved, size: f.size, tabId }};
"""
    return invoke(cli, session, code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--session", default="", help="session id; omit to list them")
    parser.add_argument("--pcb", default="", help="PCB document uuid to export")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "下单包.zip")
    parser.add_argument("--color", default="true", choices=["true", "false"])
    args = parser.parse_args(argv)

    if not args.cli.exists():
        print(f"找不到 EDA 客户端：{args.cli}", file=sys.stderr)
        return 1

    state = doctor(args.cli)
    print(f"桥接：connected={state.get('connected')}  "
          f"版本={state.get('bridgeVersion')}  匹配={state.get('versionMatch')}")
    if not state.get("connected"):
        print("编辑器没在运行。先打开嘉立创 EDA 专业版客户端。", file=sys.stderr)
        return 1

    if not args.session:
        print("\n可用会话：")
        for item in sessions(args.cli):
            print(f"  {item['sessionId']}  headless={item.get('headless')}  "
                  f"{item.get('status')}")
        print("\n用 --session <id> --pcb <uuid> 指定要导出的板子。")
        return 0

    target = args.out.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    result = export_package(args.cli, args.session, args.pcb, target)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    value = result.get("value") or {}
    if not value.get("ok"):
        return 1
    print(f"\n写出 {target}  ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
