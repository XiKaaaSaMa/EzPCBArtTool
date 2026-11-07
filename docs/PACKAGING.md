# 打包与发布

> 面向"把源码交付成一个能装能用的程序"这件事。
> 历史打包脚本保存在本机独立打包目录；本轮在工作区 `out/2026.11.4test/` 下准备副本，
> 具体路径与当前状态见下节。

---

## 本轮打包状态（2026-11-07）

1.0.6 内测最终包为 `out/2026.11.7test/release-1.0.6/package/EzPCBArtTool_v1.0.6_安装包.exe`，
66,842,964 字节，SHA-256 `f1b2607d2dd7f34a4cffaee005c8a19d11dd9ddd42530af8d01752b324022adf`。
2026-11-07 标准测试 1860 passed / 0 failed / 3 skipped，318.12 秒；Python 3.12.10 / pytest 9.1.1 / PySide6 6.11.2。
真实全新安装与 1.0.4 覆盖升级、旧草稿保留、素材一致性、界面与导出独立回读均通过。
日志与本轮报告在 `out/2026.11.7test/release-1.0.6/`，GitHub/CNB/网站同步结果见 `PROGRESS.md` 顶部。

## 历史素材验收包（2026-11-07 · 1.0.5）

内置素材版验收包位于 `out/2026.11.7test/bundled-assets/package/EzPCBArtTool_v1.0.5_安装包.exe`，
66,847,921 字节，SHA-256 `98d510ca816a10af3293dd09d5db0ff54eb881313b81f3a643ba2386e3dffc5f`。
源码、冻结版与真实中文目录安装版的 1950 个素材哈希全部一致，463 位干员及全部图标引用、PNG、SVG、两份模板可读。
2026-11-07 全量 1860 passed / 0 failed / 3 skipped / 62 warnings，302.89 秒；Python 3.12.10、pytest 9.1.1、PySide6 6.11.2。
日志、依赖、逐文件哈希与清理结果见该目录的 `验收报告.md`、`environment.json`、`pytest-full.log`、`install-result.json`。
本轮不发布或推送；旧包与旧日志保留。下文 11-05 的升级及卸载验收属于旧包，不复用为本轮实测。

## 历史打包状态（2026-11-05）

已发布版本为 1.0.4；**1.0.5 已完成打包与本机安装验收，未提交、推送或发布**。2026-11-05 标准入口 `run.bat test` 为 **1853 passed / 0 failed / 2 skipped / 62 条 Qt 弃用提醒，121.64 秒，退出码 0**；Python 3.12.14、pytest 9.2.0、PySide6 6.12.0，完整依赖和日志见 `out/2026.11.5test/installer-acceptance/environment.json`、`pytest-full.log`。

最终安装包为 `out/2026.11.5test/installer-acceptance/package/EzPCBArtTool_v1.0.5_安装包.exe`，**60,016,316 字节（57.24 MiB）**，SHA-256 **`3b95119065dd4cd171c1b9cf5d5e9427f3fac39447c06674e19de1ad3be6b0f6`**，已按内测类型归档。2026-11-04 的旧候选虽可全新安装，但覆盖升级会受旧 DLL 干扰，不作为最终交付件。

PyInstaller 应用目录仍来自 `out/2026.11.4test/package/source/`，最终安装器用仓库根目录新增的 `installer.iss` 同步到该构建副本后重新编译。后续使用该源码脚本，不能继续复制库外旧脚本；本轮文件尚未提交。打包目录需包含最终源码、`main.py`、`requirements.txt`、`LICENSE`、`icon/app.ico`、`java_lineart/`、`PCB艺术板工具.spec` 和仓库安装脚本；输入与输出逐项核对哈希。

旧 spec 的 `PIL` 排除已移除；最终安装版的图片读尺寸、裁切编码、彩印、文字和包内 Java 滤镜均已实测。包内 133 个应用文件与构建目录逐项一致，安装后的 136 个文件在运行前后哈希一致。最终包全新安装、1.0.4 覆盖升级、实际打开旧草稿、epro2 独立校验、Gerber 第三方回读和卸载清理均通过。安装编译器 Inno Setup 6.7.3，应用构建器 PyInstaller 6.22.3。

`_ssl.pyd` 的 `libcrypto-3-x64.dll`、`libssl-3-x64.dll` 和当前解释器伴随证书数据均已收集；最终包的证书及主机名验证开启，HTTPS 只读请求返回 200。此前将 `face_model.py` 列为界面运行依赖的判断已纠正：它只由离线 `tools/crop_figure_library.py` 经 `figure_crop` 使用，当前 GUI 不调用，无需加入应用包。界面取景由 `DesignCanvas` 和 `color_image` 处理，实际编码后的裁切图已验尺寸和逐点像素。

用户于 2026-11-05 明确省略“无 Python 机器”检查。更新下载使用实际旧包中提取的未改动模块，对本机回环测试清单验证完整下载、HTTP 206 续传、SHA-256 和篡改拒绝；实际下载的最终包用于覆盖升级。线上 1.0.5 尚未发布，未将本机测试写成线上更新弹窗起装通过。

详细报告见 `out/2026.11.5test/installer-acceptance/验收报告.md`；后文通用流程继续适用，旧交接材料仅作历史记录。

---

## 软件自带素材（2026-11-07）

仓库根目录的 `PCB艺术板工具.spec` 为当前构建入口，收集 `java_lineart/` 以及
`assets/明日方舟-艺术板素材/`、`assets/通行证模板源文件/`。图标源文件为
`icon/app.ico`。不要再复制缺少素材规则的旧 spec。

素材是只读应用资源，目录版安装后位于 `{app}/_internal/assets/`；
`asset_root()` 从 `sys._MEIPASS` 定位，不按 exe 的父目录或当前工作目录猜测。
草稿、缓存和用户设置仍走原有用户数据目录，不复制进安装目录。
`PCBART_ASSET_ROOT` 可以显式替换整份库，用于自定义素材和隔离测试。

构建可指定 `--distpath`、`--workpath`。安装器可通过 `/DAppBuildDir=<应用目录>`
引用该构建结果，默认仍是 `dist/EzPCBArtTool`。资源漏收时构建必须失败。

验收使用真实 exe（没有 Python 源码目录依赖）：

```bat
EzPCBArtTool.exe --verify-assets --output <本轮日期目录>\installed-assets.json
```

该命令读取干员库、检查全部图标引用、解码全部 PNG、解析 SVG 和两份模板，
输出每个文件的 SHA-256。与源码检查结果逐项一致才算通过；须清除
`PCBART_ASSET_ROOT` 后从其他工作目录运行，避免开发机外部库掩盖漏打包。

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

**当前已发布版本：`1.0.6`（内测，GitHub / CNB / 个人网站已同步）**。源码版本由 `pcbart/__init__.py` 的 `__version__` 决定。
规则：**打包类型由用户决定，版本号跟着类型走**；`__version__` 只是"现在
代码是什么版本"，安装包用 `/DAppVersion=` 单独给。

## 一·二、已交付版本（按历史发布记录，2026-11-04 整理）

| 版本 | 类型 | 安装包 | 归档位置 |
|---|---|---|---|
| **1.0.4** | 测试版 | `EzPCBArtTool_v1.0.4_安装包.exe`（57 121 683 字节） | `安装包保存区\内测\` |
| **1.0.1** | 测试版 | `EzPCBArtTool_v1.0.1_安装包.exe`（**54.1 MB**） | `安装包保存区\内测\` |
| **1.0.0** | 正式版 | `PCB艺术板工具_v1.0.0_安装包.exe`（**50.14 MB**） | `安装包保存区\正式版\` |
| 0.1.1 | 公测版 | 45.7 MB | `安装包保存区\公测\` |
| 0.1.0 | 公测版 | 45.6 MB | `安装包保存区\公测\` |
| 0.0.1 / 0.0.2 | 内测版 | 45.5 MB | `安装包保存区\内测\` |

**1.0.1 的 SHA-256**
（`0cf40f7619a222e653b9eabb44bfb0037f91d3252005385134786a20ec812482`）：
GitHub Release 与 CNB Release `v1.0.1` 各附一份，官网下载源指向 CNB。
⚠️ 体积比 1.0.0 大 4 MB 是依赖变化（开发机的 `opencv-python-headless`
新版 `cv2.pyd` 单文件就有 113 MB，1.0.0 打包时约 82 MB），**不是误打包**。

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

### 1.0.0 改名迁移说明（历史，1.0.1 起生效）

程序名与安装目录都变了，1.0.1 起的安装包 **不会覆盖旧目录中的** 1.0.0，而是装到
`EzPCBArtTool` 目录。旧版留在 `PCB艺术板工具` 目录里，需要用户自己去
控制面板卸载一次。

发布时要在 Release 说明里写明这一条 —— 这不是缺陷，是改名的一次性代价。

---

## 二、打包流程（常规流程；本轮使用工作区副本）

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
cd "<本轮打包副本>"
python -m PyInstaller --clean --noconfirm "PCB艺术板工具.spec"

:: 3) 生成安装包，版本号按第一节的规则填（先问用户是哪种版本）
"%USERPROFILE%\InnoSetup\ISCC.exe" /DAppVersion=1.0.5 installer.iss

:: 4) 把安装包归档到 安装包保存区 下对应版本类型的子目录
```

`/DAppVersion=` 覆盖 `.iss` 里的默认值，**改版本不需要编辑脚本**。

⚠️ **脚本文件名是 `installer.iss`**（本文档早先写成 `安装包.iss`，按错的写会报
「系统找不到指定的文件」，且不带文件名提示——1.0.4 打包时踩到，已修正）。
⚠️ **ISCC 要在 Git Bash 里加 `MSYS_NO_PATHCONV=1`**，否则 `/DAppVersion=` 被
MSYS 当路径转换吃掉。

### 产物

| 产物 | 位置 | 体积参考 |
|---|---|---|
| 应用目录 | `<本轮打包副本>\dist\EzPCBArtTool\` | 以本轮构建为准 |
| **安装包** | `<本轮打包副本>\EzPCBArtTool_v<版本>_安装包.exe` | 以本轮构建为准 |
| 交付位置 | `安装包保存区\内测\`、`公测\`、`正式版\` 按版本类型归档 | — |

安装包比应用目录小很多是正常的：`lzma2/max` 对 Qt、OpenCV 的 DLL 压缩率很高。

---

## 三、备份与同步

历史流程把源码同步到本机独立的 `PCB工具打包测试` 目录。本轮在工作区 `out/2026.11.4test/` 下建立独立副本；spec 只读复用并修正旧配置，安装器使用根目录新增的 `installer.iss`。不覆盖库外旧打包目录。

同步排除 `.git/`、`out/`、`__pycache__/`、`.pytest_cache/`、`推GitHub/`、`AGENTS.local.md`、历史安装包与模型权重。源目录的 `out/` 内同时包含目标副本时，必须明确排除，避免递归复制。保留构建所需的许可证、图标、Java jar/class 与源码内嵌资源。

记录相对路径、文件大小和 SHA-256 清单，并与副本逐项比对；文件总数相同不能证明内容一致。库外脚本的旧默认版本号和依赖排除项也必须单独检查。

---

## 四、⚠️ 打包与发布前必须检查的事项

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
./EzPCBArtTool.exe --screenshot new.png --open-preprocess
python -c "对比 new.png 与你删之前那张的 sha256"
```

### 5. `java_lineart/` 必须随包分发（2026-10-13 起）

【滤镜】模块通过子进程调用 `java_lineart/`（`pcbart/java_filters.py`）。
它**不是** Python 依赖，PyInstaller 不会自动收集，`.spec` 要把它当数据
目录带进应用目录：

```python
# PCB艺术板工具.spec 的 Analysis(datas=[...]) 里加：
    ('java_lineart', 'java_lineart'),
```

保留 `src/`、`lib/`、`classes/` 三部分以及许可证即可：`classes/` 里的 `.class` 是
`--release 8` 预编译的，用户装了 JRE 8+ 就能跑，不必装 JDK。
`pcbart/java_filters.tool_dir()` 先在源码树里找、再在**可执行文件旁边**
找 `java_lineart/`；1.0.4 的目录包实测位于 `_internal/java_lineart/`。本轮须以 `tool_dir()` 实际解析结果及滤镜运行为准。

⚠️ 打包后验证一次：启动后【图像预处理 → 滤镜】下拉框应当可用；
被禁用（并写明原因）意味着 `java_lineart/` 没进包或机器没装 Java。

### 6. 构建 PATH 与运行时证书数据（2026-11-04 实测）

本轮第一次冻结程序在实际启动时出现 `QtCore` 的 `WinError 127`。PE 导入/导出表定位到构建 PATH 中的 Poppler 目录：PyInstaller 误收其 ICU 78 DLL，而 Qt 需要的系统 ICU 接口没有该版本后缀。文件名看似匹配，导出符号却不兼容。仅检查“包里有这个 DLL”不能验收。

构建前记录 PATH，排除 Poppler、其他软件私有 DLL 目录和无关工具目录；使用已明确的构建 Python 与系统工具路径。清理后执行干净构建，不从系统或别的软件目录手工复制 DLL 补洞。构建后必须从无源码和无开发 PATH 的隔离目录启动最终 exe，核对 QtCore、窗口和实际导出行为。2026-11-05 的最终安装版已完成复验。

本轮使用的 bundled Python 的 `ssl.py` 还依赖同目录 `codex-rejected-cas.pem`。首次冻结遗漏该文件，创建默认 `SSLContext` 时失败。spec 按当前构建解释器条件收集：

```python
import ssl
ssl_rejected_cas = Path(ssl.__file__).with_name("codex-rejected-cas.pem")
ssl_datas = [(str(ssl_rejected_cas), ".")] if ssl_rejected_cas.is_file() else []
# Analysis(datas=[("java_lineart", "java_lineart")] + ssl_datas, ...)
```

这是当前解释器伴随数据的收集，不是所有 Python 都必须使用该文件。保留原 `ssl` 实现与证书验证，不删除安全配置或关闭验证来绕过缺文件。冻结后必须实测 `ssl.create_default_context()` 和带证书验证的只读 HTTPS 请求，并记录最终证据。

### 7. 覆盖升级必须替换程序专用依赖目录（2026-11-05）

1.0.4 将 Qt 6.11/Shiboken DLL 放在 `_internal/PySide6/` 和 `_internal/shiboken6/`，本轮构建将新版 DLL 放在 `_internal/`。逐文件覆盖会保留旧 DLL，扩展模块先读到旧库，导致升级后的 exe 报 `DLL load failed while importing Shiboken`。新文件哈希全部正确仍可能发生此故障。

仓库 `installer.iss` 在安装前只清理程序专用目录，再写入完整的新目录：

```ini
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
```

不要扩大到 `{app}` 或用户数据目录。验收须比较完整相对文件清单以发现残留，实际运行升级后的 exe，并核对草稿及关联缓存。新增回归测试 `tests/test_installer_upgrade.py`；本轮撤掉清理段时测试失败，修复后的真实 1.0.4→1.0.5 升级及旧草稿显示通过。

### 8. 发布脚本取 GitHub 令牌：`git credential fill` 必须带 `username=`

**不带 `username=` 会挂起**：本机凭据助手是 Git Credential Manager，凭据按账号
存；只给 `protocol`/`host` 时它定不了账号，回落到交互式提示——无人值守时表现为
「无输出、无报错、Release 没建」（2026-10-14 连测两次才定位）。正确写法：

```python
subprocess.run(["git", "credential", "fill"],
               input="protocol=https
host=github.com
username=XiKaaaSaMa

", ...)
```

实测带上后 0.4 s 返回。发布脚本在
`out/2026.10.14test/_release/publish_github_1.0.4.py`（可直接改成下一个版本用）。

---

## 五、发布前验证清单

装完必须**实际跑一遍**，缺一项都不算验证过：

- [ ] 完整测试通过（`run.bat test`），记录运行环境、实际通过/跳过数和日志；历史固定条数不作本轮结论
- [ ] 从**要交付的那个安装包**全新安装
- [ ] 安装后从安装目录启动，界面中文正常、图标正常
- [ ] 载入一张图，跑通板画流程，画布双色显示正确
- [ ] 缓存写在 `%LOCALAPPDATA%`，**安装目录运行时零写入**
- [ ] 导出 `.epro2`，过 `node tools\verify_epro2.mjs` 独立校验
- [ ] 在隔离的全新安装环境验证卸载清理；另做升级验收，确认既有草稿保留，不使用真实草稿库做清理试验

⚠️ **"在我机器上是好的"不算验证。** 上一条清单的意义就在于：
这台机器装了 Python 和全套依赖，很多问题只有换台干净机器才暴露。
通常交付前应在**没有 Python 的机器**上装一次。2026-11-05 本轮由用户明确省略该项，记录为省略，不写成通过；本机验收与线上真实更新链路的范围分别记录。

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
  签名状态在发布说明中如实列出。
- 体积主要来自 Qt（64 MB）和 OpenCV（82 MB），不是 Python（解释器仅 6.7 MB）。
  **不要为了省体积去重写 C++**，那会丢掉全部已验证的算法与测试。
