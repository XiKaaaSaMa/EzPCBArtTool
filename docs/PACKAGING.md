# 打包与发布

> 面向"把源码交付成一个能装能用的程序"这件事。
> 打包脚本在 `P:\Zcode开发专用文件夹\PCB工具打包测试\`，
> 源码仓库只保留本文件与版本规则。

---

## 一、版本号规则（用户 2026-09-21 定）

版本号三位：`主版本.次版本.修订号`。

| 打包类型 | 变化 | 例子 |
|---|---|---|
| **测试版** | 第三位 +1 | 0.0.1 → 0.0.2 |
| **公测版** | 第二位 +1，第三位归零 | 0.0.5 → 0.1.0 |
| **正式版** | 第一位 +1，后两位归零 | 0.1.3 → 1.0.0 |

⚠️ **每次要求打包时，先问用户这是什么版本。**

不要自己推断，也不要沿用上一次的版本号——用户要求的是"每次询问"，
因为同一批代码可能重打多次，是否是发布节点只有用户知道。

提问要直接，例如：

> 这次打包是什么版本？
> - 测试版 → `0.0.2`
> - 公测版 → `0.1.0`
> - 正式版 → `1.0.0`

**当前代码版本：`1.0.0`**（`pcbart/__init__.py` 的 `__version__`）。
规则：**打包类型由用户决定，版本号跟着类型走**；`__version__` 只是"现在
代码是什么版本"，安装包用 `/DAppVersion=` 单独给。

## 一·二、已交付版本（2026-10-10 更新）

| 版本 | 类型 | 安装包 | 归档位置 |
|---|---|---|---|
| **1.0.0** | 正式版 | `PCB艺术板工具_v1.0.0_安装包.exe`（**50.14 MB**） | `安装包保存区\正式版\` |
| 0.1.1 | 公测版 | 45.7 MB | `安装包保存区\公测\` |
| 0.1.0 | 公测版 | 45.6 MB | `安装包保存区\公测\` |
| 0.0.1 / 0.0.2 | 内测版 | 45.5 MB | `安装包保存区\内测\` |

**1.0.0 的 SHA-256**
（`fe3be936f6f8d3669a23c6b346fc4eb5d47cca16490a4f111090f7f9675645fd`）：

- GitHub Release：`https://github.com/XiKaaaSaMa/EzPCBArtTool/releases/download/v1.0.0/PCB._v1.0.0_.exe`
- 网站下载源（CNB）：`https://cnb.cool/xikaaa/Ez_PCBart/-/releases/download/v1.0.0/PCB._v1.0.0_.exe`
  （匿名实测：HTTP 200、**5.0 MB/s**、下载后 SHA-256 与上值一致）

### ⚠️ 1.0.0 打包时踩到的两个坑

1. **Inno Setup 的参数在 Git Bash 里会被路径转换吃掉**。
   `ISCC.exe /DAppVersion=1.0.0 installer.iss` 会报
   `You may not specify more than one script filename` —— MSYS2 把
   `/DAppVersion=1.0.0` 当路径转成了 `C:/...`。必须加 `MSYS_NO_PATHCONV=1`。
2. **静默安装不要用 `cmd //c` 拼引号**（路径会被再解析一次，装到默认位置且
   不留日志）。用 PowerShell 的 `Start-Process -ArgumentList` 传参数最稳：
   ```powershell
   $p = Start-Process -FilePath '.\PCB艺术板工具_v1.0.0_安装包.exe' -PassThru -Wait -ArgumentList `
        '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/LOG=<日志路径>','/DIR=<安装路径>'
   ```
   首次退出时旧包被残留的安装进程锁住（`Device or resource busy`），
   结束该进程后才能重打。

### ✅ 1.0.0 交付验收记录（2026-10-10）

- 全量测试 **1723 passed / 2 skipped**（源码目录与打包目录各跑一次，结果相同）
- **全新安装 → 启动 → 导出 → 第三方回读 → 卸载** 全流程走通：
  - 从安装目录启动并截图，与开发目录产物**逐字节一致**
  - 安装目录运行时**零写入**（159 个文件，启动后仍是 159 个）
  - 缓存正确落在 `%LOCALAPPDATA%\PCB艺术板工具\`
  - 无头 Gerber 导出 → `gerbonara` 回读板框 **75.979 × 76.200 mm**，
    与图片比例算出的期望值一致
  - 卸载后**零残留**：程序目录 0 文件、无缓存、无开始菜单项
- **GPL-3.0 合规**：`LICENSE` 随安装包写入 `{app}\LICENSE.txt`，
  开始菜单加「许可证（GPL-3.0）」入口，关于对话框写明许可证与源码地址

### ⚠️ GPL-3.0 是交付要求的一部分，不要漏

软件以 GPL-3.0 发布（2026-10-10 用户选定）。GPL 第 4 条要求
**随程序一并提供许可证副本**，所以 `installer.iss` 的 `[Files]` 里有一条
`Source: "LICENSE"`。重新打包时如果 `PCB工具打包测试\` 下缺 `LICENSE`，
编译会直接失败——这是刻意的。

⚠️ 主题相关的**新模块必须一起同步**，漏了会在启动时报
`ModuleNotFoundError`：`pcbart/ui_theme.py`、`pcbart/ui_icons.py`、
`tools/qt_theme.py`。

⚠️ **打包前必须先把源码同步到 `PCB工具打包测试\`**，
否则会把旧行为又发出一次。这两份源码**不同步是常态**——
`PCB工具打包测试\` 只在打包时更新。

⚠️ **2026-10-10 同步方式有了坑**：源码目录新增了 `LICENSE` 与
`pcbart/models/` 等文件，用 tar 同步时务必**排除** `out/`（866 MB 产物）、
`.git`、`推GitHub/`、`AGENTS.local.md`，否则会把本机私有内容带进打包目录：

```bash
cd "P:/Zcode开发专用文件夹/PCB艺术板工具"
tar -cf - --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=out --exclude=推GitHub --exclude=AGENTS.local.md . \
  | (cd "P:/Zcode开发专用文件夹/PCB工具打包测试" && tar -xf -)
```

同步后**逐文件比对**（两侧各自 `find` 列表做差集），确认没有"源码摘了、
打包目录还留着"的陈旧模块。`PCB工具打包测试\` 里允许存在、
源码里没有的文件只有这几个打包专用项：
`PCB艺术板工具.spec`、`installer.iss`、`main.py`（源码也有）、
`make_icon.py`、`icon/`、历史安装包与 `_*.png` 截图。

⚠️ 打包前先比对一下，确认同步真的发生了：

```bash
# 在源码仓库与打包目录各跑一次，输出应完全一致。
# 模型、草稿、多工程工作区、界面主体，以及主题相关的四个模块。
md5sum pcbart/artboard.py pcbart/draft.py pcbart/projects.py \
       pcbart/ui_theme.py pcbart/ui_icons.py \
       tools/gui_mockup.py tools/qt_theme.py
```

⚠️ **同步时最容易漏的是「新增文件」**，因为它不是被改的老文件：

| 模块 | 加入版本 | 漏了的症状 |
|---|---|---|
| `pcbart/projects.py` | 0.1.1 | 启动即报 `ModuleNotFoundError: pcbart.projects` |
| `pcbart/ui_theme.py` | 0.1.2 | 同上（界面配色） |
| `pcbart/ui_icons.py` | 0.1.2 | 同上（**生成文件**，要一起同步） |
| `tools/qt_theme.py` | 0.1.2 | 同上（图标绘制与样式表） |
| `pcbart/badge_art.py` | 0.1.2 | 通行证画布没有固定装饰，或导出丢边框/logo |
| `pcbart/badge_decorations.py` | 0.1.2 | 同上（**生成模块**，用 `tools/extract_badge_art.py` 重建） |
| `pcbart/qr_code.py` | 0.1.2 | 通行证二维码生成失败 |
| `pcbart/code128.py` | 0.1.2 | 通行证条形码生成失败 |
| `pcbart/badge_codes.py` | 0.1.2 | 二维码/条形码布局和短链入口缺失 |
| `pcbart/short_url.py` | 0.1.2 | 批量短链工具无法调用 urlc.cn |
| `pcbart/badge_short_urls.py` | 0.1.2 | 二维码退回长链接，v2、模块更小（**生成模块**，用 `tools/shorten_badge_urls.py` 重建；缺失时仍能回退） |

⚠️ **`pcbart/ui_icons.py` 是生成文件，但必须进仓库、也必须同步**：
图标以代码形式内嵌就是为了一打包就能用，它**不是**构建产物，
打包前不需要重新生成（除非改了图标清单，那时见
`docs/THIRD_PARTY.md` 里的生成命令）。

---

## 一·一、安装包保存区（用户 2026-09-22 要求）

安装包不再散放在开发目录里，统一归档到：

```
P:\Zcode开发专用文件夹\安装包保存区\
    内测\      测试版（0.0.x）
    公测\      公测版（0.x.0）
    正式版\    正式版（x.0.0）
```

**按版本号类型放对应子目录**，文件名沿用 `EzPCBArtTool_v<版本>_安装包.exe`。

---

## 一·三、命名规则（用户 2026-10-10 定）

项目里有**两套名字**，故意分开。改打包配置时不要把两者合并：

| 层面 | 用哪个名 | 出现在哪里 |
|---|---|---|
| **文件层** | **`EzPCBArtTool`** | 安装包文件名、程序 exe 名、安装目录、源码仓库名 |
| **显示层** | **`PCB艺术板工具`** | 窗口标题、关于对话框、开始菜单文件夹与快捷方式文字、控制面板卸载项 |

用户原话：「下一个版本开始安装包用我仓库那个名字，不用现在这个了」，
并确认**界面仍显示中文名**。

具体落点：

| 位置 | 字段 | 值 |
|---|---|---|
| `PCB艺术板工具.spec` | `EXE(name=...)` | `EzPCBArtTool` |
| `PCB艺术板工具.spec` | `COLLECT(name=...)` | `EzPCBArtTool` |
| `installer.iss` | `AppDirName` | `EzPCBArtTool`（安装目录） |
| `installer.iss` | `AppExeName` | `EzPCBArtTool.exe` |
| `installer.iss` | `OutputBaseFilename` | `EzPCBArtTool_v{#AppVersion}_安装包` |
| `installer.iss` | `AppName` / `DefaultGroupName` / `UninstallDisplayName` | `PCB艺术板工具`（显示层，**不改**） |
| `tools/gui_mockup.py` | `setWindowTitle` | `PCB 艺术板工具`（**不改**） |
| `pcbart/paths.py` | `APP_DIR_NAME` | `PCB艺术板工具`（**不改**，见下） |

### ⚠️ 用户数据目录**不要跟着改名**

`pcbart/paths.py` 的 `APP_DIR_NAME = "PCB艺术板工具"` 决定
`%LOCALAPPDATA%\PCB艺术板工具\`，用户的**草稿、缓存、外接模型都在里面**。
跟着改名会让老用户的草稿全部凭空消失（程序会当作新装环境，建一个空库）。

**所以程序文件名改了、安装目录改了，但数据目录保持不动。**
`installer.iss` 的 `[UninstallDelete]` 也继续指向旧路径，卸载行为不变。

### ⚠️ 已装 1.0.0 的用户需要手动卸载

程序名与安装目录都变了，1.1.0 **不会覆盖** 1.0.0，而是装到
`EzPCBArtTool` 目录。旧版留在 `PCB艺术板工具` 目录里，需要用户自己去
控制面板卸载一次。

发布时要在 Release 说明里写明这一条 —— 这不是缺陷，是改名的一次性代价。

---

## 二、打包流程

### 前置

| 需要 | 说明 |
|---|---|
| Python 3.12 | 已含项目依赖 |
| PyInstaller | `python -m pip install pyinstaller` |
| Inno Setup 6 | 生成安装包；编译器 `ISCC.exe` |
| 中文语言文件 | `ChineseSimplified.isl` 放到 Inno 的 `Languages\` 目录 |

### 步骤

```bat
:: 1) 备份源码（改动前先备份，见第三节）
:: 2) 打包应用本体
cd "P:\Zcode开发专用文件夹\PCB工具打包测试"
python -m PyInstaller --clean --noconfirm "PCB艺术板工具.spec"

:: 3) 生成安装包，版本号按第一节的规则填（先问用户是哪种版本）
"%USERPROFILE%\InnoSetup\ISCC.exe" /DAppVersion=1.0.0 "安装包.iss"

:: 4) 把安装包归档到 安装包保存区 下对应版本类型的子目录
```

`/DAppVersion=` 覆盖 `.iss` 里的默认值，**改版本不需要编辑脚本**。

### 产物

| 产物 | 位置 | 体积参考 |
|---|---|---|
| 应用目录 | `PCB工具打包测试\dist\PCB艺术板工具\` | 174 MB |
| **安装包** | `PCB工具打包测试\PCB艺术板工具_v<版本>_安装包.exe` | **45 MB** |
| 交付位置 | `安装包保存区\内测\`、`公测\`、`正式版\` 按版本类型归档 | — |

安装包比应用目录小很多是正常的：`lzma2/max` 对 Qt、OpenCV 的 DLL 压缩率很高。

---

## 三、备份

用户要求：**每次打包前把完整源码备份到 `PCB工具打包测试` 目录。**

用 `tar` 排除构建垃圾，避免把上一次的产物混进备份：

```bash
cd "P:/Zcode开发专用文件夹/PCB艺术板工具"
tar -cf - --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache . \
  | (cd "P:/Zcode开发专用文件夹/PCB工具打包测试" && tar -xf -)
```

备份后**核对文件数一致**再继续，否则备份是假的：

```bash
find . -type f -not -path './.git/*' -not -path '*__pycache__*' | wc -l
```

---

## 四、⚠️ 打包前必须检查的路径约束

这些是踩过的坑，改动代码时**不要破坏**。

### 1. 写盘路径不能从 `__file__` 推

打包后 `__file__` 指向 PyInstaller 的解包目录：

- 单文件模式：**退出即删除**，缓存没了画布就没东西可画
- 目录模式：在安装目录里，`C:\Program Files\` 下**没有写权限**

必须走 `pcbart.paths`：

| 函数 | 源码运行 | 打包后 |
|---|---|---|
| `data_root()` | 工程目录 | `%LOCALAPPDATA%\PCB艺术板工具` |
| `cache_dir()` | `out/lineart_cache` | `%LOCALAPPDATA%\...\cache\...` |
| `default_output_dir()` | `out/` | `%LOCALAPPDATA%\...\out` |

`tests/test_paths.py` 有一条测试专门扫描源码，**禁止再出现从 `__file__`
推出来的可写路径**。改路径逻辑时先跑它。

⚠️ 缓存**不能放临时目录**——文档存的是这个路径，系统清理后画布会空。

### 2. 导入必须是可解析的全路径

`design_canvas` 用 `tools.design_canvas` 导入（配合 `tools/__init__.py`），
不能靠 `sys.path` 插目录再裸名导入——打包工具的静态分析认不出后者。

### 3. 命令行参数别叫 `--style`

Qt 会先把裸的 `-style` 当成自己的控件样式参数吃掉。
所以艺术风格参数叫 `--art-style`。

### 4. 依赖裁剪改动后必须实测

`.spec` 里排除了大批 Qt 模块和 12 个 DLL（软件 OpenGL 回退、加密库、
ffmpeg 等）。**删任何 DLL 都要重新跑一次并比对截图哈希**，
不能只看"程序能启动"：

```bash
./PCB艺术板工具.exe --screenshot new.png --open-preprocess
python -c "对比 new.png 与你删之前那张的 sha256"
```

---

## 五、发布前验证清单

装完必须**实际跑一遍**，缺一项都不算验证过：

- [ ] 805 项测试通过（`run.bat test`）
- [ ] 从**要交付的那个安装包**全新安装
- [ ] 安装后从安装目录启动，界面中文正常、图标正常
- [ ] 载入一张图，跑通板画流程，画布双色显示正确
- [ ] 缓存写在 `%LOCALAPPDATA%`，**安装目录运行时零写入**
- [ ] 导出 `.epro2`，过 `node tools\verify_epro2.mjs` 独立校验
- [ ] 卸载干净：程序目录、开始菜单、用户缓存都不残留

⚠️ **"在我机器上是好的"不算验证。** 上一条清单的意义就在于：
这台机器装了 Python 和全套依赖，很多问题只有换台干净机器才暴露。
交付给用户前，至少要在**没有 Python 的机器**上装一次。

---

## 六、图标

应用与安装包的图标同一份文件：

```
PCB工具打包测试\icon\app.ico
```

- `.spec` 通过 `icon=` 打进 exe
- `.iss` 通过 `SetupIconFile=` 用于安装包与卸载项
- 需要多尺寸：**16/24/32/48/64/128/256**（Windows 各场景取不同尺寸，
  只给一个尺寸在任务栏或资源管理器里会糊）

更换图标后**两个地方都要重打**，只改一个会出现
"桌面图标换了、安装包还是旧的"。

---

## 七、已知限制

- 安装向导默认中文，英文可选（Inno 自带）。
  需在 Inno 的 `Languages\` 目录放 `ChineseSimplified.isl`，仓库不带它。
- 安装包**未做代码签名**，用户首次运行会有 SmartScreen 提示。
  正式对外发布需要买证书。
- 体积主要来自 Qt（64 MB）和 OpenCV（82 MB），不是 Python（解释器仅 6.7 MB）。
  **不要为了省体积去重写 C++**，那会丢掉全部已验证的算法与测试。
