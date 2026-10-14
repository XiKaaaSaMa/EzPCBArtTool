# 第三方资源与许可证

本工具用到的外部作品，以及各自的许可证。**全部允许商用，且都不要求署名**
（唯一有署名条件的 Solar 没有采用，见下）。

---

## 界面图标

### Iconoir —— 主图标集

| | |
|---|---|
| 项目 | https://iconoir.com |
| 仓库 | https://github.com/iconoir-icons/iconoir |
| 许可证 | **MIT** |
| 用途 | 工具栏、菜单、面板按钮的全部图标（49 个） |

### Lucide —— 仅 "图层" 一个图标

| | |
|---|---|
| 项目 | https://lucide.dev |
| 仓库 | https://github.com/lucide-icons/lucide |
| 许可证 | **ISC** |
| 用途 | **只有 `layers` 这一个**（图层列表和工具栏「图层」按钮） |

⚠️ **为什么图层要借用 Lucide**：Iconoir 没有"图层"这个图标 ——
实测它的 1682 个图标里，`layer` / `stack` / `sheets` 相关命名一个都没有
（`stackoverflow` 是那个网站的 logo，不是图层）。图层面板是本工具的核心界面，
这个概念不能缺，所以单独从 Lucide 借一个。

**两个许可证都不要求署名**，但这里仍然记下来，因为：
1. 打包分发时许可证文件的完整性是常规要求；
2. 以后要换图标集时，这份记录能立刻说清"哪些是从哪儿来的"。

### 图标的存储方式

图标不是 `.svg` 文件，而是**代码里的矢量路径**：`pcbart/ui_icons.py`，
由 `out/_uigallery/gen-icons.mjs` 从上述两个包生成。这样做有两个原因：

- **打包后不会找不到**。PyInstaller 单文件模式会把程序解压到临时目录，
  任何运行时按路径读取的资源都要额外配置打包规则；配置漏了就是一个
  看不见的按钮。生成成 Python 模块则跟其他源码一样被导入，不会漏。
- **能跟着主题变色**。图标按路径绘制而不是贴位图，所以换皮肤时不需要
  准备第二套图标，也不会因为缩放而糊。

⚠️ **`pcbart/ui_icons.py` 是生成文件，不要手改**，改图标要重新跑生成脚本。

### 没有采用的：Solar

Solar（CC BY 4.0）在最初的候选里，**但没有采用**。CC BY 要求商用署名，
会给"关于"页增加一条必须维护的义务，而 Iconoir + Lucide 的组合已经
覆盖了全部 28 个概念，没有理由为此付代价。

---

## 字体

**没有内嵌任何字体文件。** 界面用系统字体：

| 场景 | 字体 |
|---|---|
| 中文 | Microsoft YaHei UI（Windows 自带） |
| 西文/数字 | Roboto（Material 规范指定）→ 没有则 Segoe UI → 系统默认 |

⚠️ **Material Design 规范指定 Roboto，但本项目没有内嵌它。**
Roboto 可以通过 npm 包拿到，但只有 `.woff` / `.woff2` 格式，Qt 对这两种
格式的支持不可靠（`QFontDatabase.addApplicationFont` 官方支持的是
ttf/otf）。内嵌一个加载不了的字体是纯粹的成本。所以字体栈里 Roboto 排第一
（装了就用）、Segoe UI 兜底（Windows 必然有，且两者度量接近）。

中文字形本来就不在 Roboto 里（它没有 CJK 字符），所以中文无论如何都会
回落到雅黑，这条路径不受影响。

---

## Python 依赖

运行环境要求，不是打包进产品的代码，各自的许可证随各自的包分发。

| 包 | 用途 | 许可证 |
|---|---|---|
| PySide6 / Qt | 界面框架 | LGPL-3.0 / 商业双许可 |
| shapely | 几何运算 | BSD-3-Clause |
| numpy | 数值计算 | BSD-3-Clause |
| opencv-python-headless | 图像处理 | Apache-2.0 |

⚠️ **PySide6 是 LGPL-3.0**：动态链接使用是允许的，PyInstaller 目录模式
（本项目采用）符合这一要求。这也是本项目**不用单文件模式打包**的原因之一。

---

## 参考过的第三方实现

移植了算法，没有引入代码依赖：

| 来源 | 内容 | 许可证 |
|---|---|---|
| PCB_lightgraph | 高斯预滤波、去杂点（`pcbart/denoise.py`） | MIT |
| Kazuhito00/XDoG-OpenCV-Sample | XDoG 软阈值线稿（`pcbart/lineart.py` 的 `xdog_response`） | MIT |
| scikit-image | Meijering 脊线滤波，复刻在 `pcbart/ridge.py` | BSD-3-Clause |
| bloc97/Anime4K v0.9 | 超分（`pcbart/upscale.py`，经 TianZerL/Anime4KPython 参考实现） | MIT |
| scipy | 高斯导数核（`pcbart/ridge.py` 的 `_scipy_kernel1d` 按同一公式重写） | BSD-3-Clause |

⚠️ **scikit-image 只用作测试对照，不是运行依赖**：`pcbart/ridge.py` 把
Meijering 滤波用 numpy + OpenCV 重写（逐点与 scikit-image 比对，最大差
3e-7；速度约快 5 倍），`tests/test_ridge.py` 在本机装了 scikit-image 时
拿它当参照，没装就跳过。所以 `requirements.txt` 里没有 scipy / scikit-image。

---

## 随本工具分发的 Java 工具

`java_lineart/` 是本仓库自带的小工具，由主程序以子进程方式调用
（`pcbart/java_filters.py`），提供 16 个 JH Labs 滤镜。

| | |
|---|---|
| JH Labs Filters | https://github.com/axet/jhlabs ｜ **Apache-2.0** ｜ `java_lineart/lib/filters-2.0.235-1.jar`（373 KB） |
| 许可全文 | `java_lineart/lib/jhlabs_LICENSE.txt` |
| 包装代码 | `java_lineart/src/LineArtJavaTool.java`（本项目自写，随本仓库 GPL-3.0） |
| 预编译产物 | `java_lineart/classes/LineArtJavaTool.class`（`--release 8`，装 JRE 即可运行） |

⚠️ **需要机器上有 Java（JRE 8+）**：这是本工具唯一的功能性外部依赖。没有
Java 时其余功能完全正常，滤镜下拉框会被禁用并在面板上写明原因。

---

## 随本工具分发的模型权重

### YuNet 人脸检测（OpenCV Zoo）

| | |
|---|---|
| 项目 | https://github.com/opencv/opencv_zoo |
| 权重 | `face_detection_yunet_2023mar.onnx`，227 KB |
| 许可证 | **Apache-2.0** |
| 用途 | 通行证立绘取景时找脸，把人物摆到中轴上 |
| 存放 | **两份**：`pcbart/face_model.py`（base64，程序实际读的那份）与 `pcbart/models/face.onnx`（原文件，留档） |

⚠️ **这一份是随软件分发的**，和下面那个外接模型不同：它只有 227 KB，
且打包后必须能找到（见 `pcbart/face_model.py` 顶部说明为何内联成 base64）。
所以它**入库**，而 `out/_models/a2s` 那 390 MB 不进 —— `.gitignore` 忽略的是
后者，不是所有模型权重。

---

## 可选的线稿重绘模型（不由本工具分发）

线稿重绘用 **Anime2Sketch**（Mukosame, MIT），但**模型文件不打包、不随本
工具分发**：权重 183 MB + 架构文件，而整个安装包才 178 MB；纯 CPU 跑一次
峰值还要 4.6 GB 内存。用户自行下载后指定文件夹即可，见 README
「外接模型」一节。

| | |
|---|---|
| 项目 | https://github.com/Mukosame/Anime2Sketch |
| 论文 | [Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis](https://arxiv.org/abs/2104.05703) |
| 许可证 | **MIT** |
| 本项目用到的部分 | 架构 `model.py`（按文件原样执行，不重写）+ `weights/improved.bin` |
| 获取脚本 | `tools/fetch_anime2sketch.py` |

⚠️ **架构文件是原样执行、不是重写的**：improved 权重把 6 层反卷积换成了
resize+conv，只有配套的 `model.py` 能加载。抄一份 U-Net 进来，层序差一格
照样能 `load_state_dict` 成功，然后静默画出另一张图 —— 用上游的代码是唯一
便宜的排除法。

⚠️ 模型的 **PyTorch 运行时**（`torch`）也不在 `requirements.txt` 里：只有
用这个可选功能才需要装。没有它时其余功能完全正常。

## 设计体系

| | |
|---|---|
| Material Design 3 | 配色生成规则（色调板 / 角色映射），`pcbart/ui_theme.py` 按规范实现 |

⚠️ Material Design 是 Google 的**设计规范**，不是代码库——使用它的配色与
形状规则不需要授权，本项目也没有引入 Google 的任何代码。
预设里的 18 个主色取自 Material Design 2014 调色板的 500 号色，
外加 Material 3 的基线紫（`#6750A4`）。
