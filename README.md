# PCB 艺术板工具

把一张图片做成 PCB 艺术板。导入图片，选择出图风格，套上板框，导出嘉立创 EDA
工程（`.epro2`）和可直接上传下单的 Gerber 压缩包。

线稿可放在铜层、丝印层或阻焊层；板画使用固定配色，白色走丝印、黑色走铜。
成品可用于挂件、立牌、摆件等。

|  |  |
|---|---|
| 当前版本 | 1.0.0 |
| 支持平台 | Windows 10 / 11（64 位） |
| 许可证 | GPL-3.0 |

安装包见 [Releases](https://github.com/XiKaaaSaMa/EzPCBArtTool/releases/latest)，
下载源也可用
[CNB](https://cnb.cool/xikaaa/Ez_PCBart/-/releases/download/v1.0.0/PCB._v1.0.0_.exe)。

---

## 功能

### 出图风格

| 风格 | 说明 |
|---|---|
| 线稿 | 三种读法：填充色块、描边 + 填充、纯线条。深浅区域直接映射为黑白 |
| 板画 | 双色块面，浅色区域白、深色区域黑，外轮廓用白线勾出。明暗分界、轮廓宽度、最小特征、翻转、反相均可调 |
| 彩色丝印 | 保留原图配色，不做线稿转换 |

支持 PNG、JPG 等常见格式与透明底图片，中文路径可用。图片最长边默认
76.2 mm，可调整。降噪是独立步骤，可与线稿分开调整。预览随参数更新，
大图预览按全分辨率重算。

### 板框

| 项目 | 说明 |
|---|---|
| 形状 | 矩形、圆角矩形（半径可调）、圆形、异形板 |
| 异形板 | 板框取图片主体轮廓。三种基础形状：原图裁切、矩形结合、圆角矩形结合 |
| 立牌插舌 | 在板框下边缘增加插舌，宽高可填（默认 16 × 3 mm），水平位置可调 |
| 圆形 | 直径自动取图片对角线 |

板框设置实时生效，无需确定或取消，画布即为预览。

### 导出

| 产物 | 内容 |
|---|---|
| `.epro2` 工程 | 板框、铜层、丝印层，可在嘉立创 EDA 中继续编辑 |
| Gerber 压缩包 | 板框与各层 Gerber，含 Excellon 钻孔文件，可直接上传下单 |

导出的 Gerber 用第三方解析器复核过尺寸与单位，文件为纯 ASCII。工艺问题
（如细颈）会写入导出报告。

### 通行证模板

套用模板后自动裁切立绘区，并可生成二维码、条形码、职业图标与阵营 logo。
多张通行证可合并导出为一个 `.epro2`。

立绘区支持调整：拖动移动、Ctrl+滚轮缩放取景，区外部分以变暗方式预览，
不影响导出内容。

### 界面

| 功能 | 说明 |
|---|---|
| 多工程标签栏 | 同时打开多块板，`Ctrl+Tab` 切换，各自的缩放与位置会保留 |
| 草稿库 | 左侧栏按时间排列，单击打开，可重命名或删除。退出时提示未保存的改动 |
| 图形编辑 | 选中图片后可移动、缩放、旋转、翻转、调整不透明度和所在图层 |
| 图层 | 七个图层（正反面铜 / 丝印 / 阻焊 + 板框），颜色与 EDA 一致 |
| 画布 | 滚轮以指针为锚点缩放，中键或空格拖动平移；黑底，过原点绘制 X/Y 坐标轴 |
| 浮窗 | 预处理、板框设置、图像预览均为应用内浮窗，支持拖边框缩放和折叠 |
| 主题 | Material Design 3 配色，18 套预设主色，支持浅色与深色，可逐个调整颜色角色 |
| 撤销 | `Ctrl+Z` |

图标使用 Iconoir，其中「图层」取自 Lucide。第三方许可见
[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。

### 线稿重绘（可选）

内置的线稿读法在像素层面判断边界，只能描摹已有的线条。线稿重绘调用
Anime2Sketch 模型重新绘制一遍，用于边缘不清晰的图片。

模型不由本工具提供，需自行下载后指定路径。界面上
【图像预处理 → 9 · 线稿重绘】显示当前状态，【外接模型设置与说明…】
可指定文件夹。不安装模型不影响其他功能。

| 参数 | 说明 |
|---|---|
| 线条密度 | 1–25%，默认 5%，决定哪些线入选 |
| 线条加粗 | 0–0.40 mm，默认 0.08，决定线条能否印出 |

安装方式：

```bat
python tools\fetch_anime2sketch.py
```

脚本会把模型下载到默认位置（源码运行是 `out\_models\a2s`，安装版是
`%LOCALAPPDATA%\PCB艺术板工具\models\a2s`）。也可以从别处复制一份：

```
<模型文件夹>\
  model.py              架构（必须）
  weights\
    improved.bin        项目使用的权重（必须）
    netG.pth            作者原始版（可选）
```

指定路径的方式，优先级从高到低：

1. 命令行 `--model-dir "D:\models\a2s"`
2. 环境变量 `PCBART_A2S_MODEL=D:\models\a2s`
3. 【视图 → 界面设置… → 线稿重绘模型】中选择文件夹
4. 放入上述默认位置

`improved.bin` 与 `netG.pth` 对应不同的网络结构，不能混用。项目使用
improved 版本。

---

## 安装与运行

### 安装包

下载 `PCB艺术板工具_v1.0.0_安装包.exe` 运行即可，不需要另装 Python
或运行库。安装包未做代码签名，首次运行会出现 Windows SmartScreen 提示。

### 从源码运行

需要 Python 3.11 或 3.12：

```bat
python -m pip install -r requirements.txt
run.bat
```

`run.bat` 会自动查找本机 Python，双击也可启动。

### 使用流程

1. 点【打开图片】，选择图片并调整参数
2. 需要板框时，点工具栏【板框设置】选择形状与尺寸
3. 参数调整完成后，点【导入图片】，图片进入画布
4. 【文件 → 另存为下单文件…】写出可直接上传的 Gerber 压缩包

### 命令行

```bat
run.bat test                                    :: 运行测试（1723 项）
run.bat outline "图片.png" --width 60            :: 图片 → 异形板框
python tools\make_board_art.py "图片.png" --outdir out --name board --render
run.bat calibrate "路径\到\工程.epro2"           :: 分析工程，定标单位与图层
```

---

## 已知限制

### 异形板

- 板框只取最大的一条外轮廓，图片中分离的部件（飘带、装饰）不会成为板框的一部分
- 轮廓不支持内孔，环状图形（如甜甜圈）中间的洞会被填上
- 「结合」模式下图片完全落在底座之外时，只保留较大的一块

### 图形与工艺

- 简化产生折线，没有圆弧拟合（圆形会被识别为原生 `CIRCLE`）
- 细颈检测不覆盖从厚主体伸出的细长凸起
- 最小特征默认 0.10 mm，会削掉线稿中的细发丝。调小可保留细节，代价是
  细线可能在蚀刻阶段产生毛边
- 边缘不清晰的图片，本地描边效果有限，建议先用其他工具重绘线稿

### 其他

- 板框开孔需要在 EDA 中自行绘制，本工具不做
- 铜层目前只支持填充风格，走线风格尚未实现
- 形状下拉中的「套用模板」为通用模板功能，暂未开放（通行证模板不受影响）
- 线稿重绘工作区是独立工具 `tools\line_redraw_panel.py`，未并入主窗口
- 安装包未做代码签名

---

## 环境

| 项 | 版本 |
|---|---|
| Python | 3.11 / 3.12（`run.bat` 会自动查找） |
| 依赖 | PySide6、numpy、opencv-python-headless、shapely、Pillow、pytest |
| Node.js | 24（仅独立校验器使用，运行软件不需要） |

---

# 技术资料

以下内容面向修改代码的人，运行和使用软件不需要阅读。

## 项目结构

```
pcbart/                 核心库（纯 Python，不依赖 Qt）
  units.py              单位换算（唯一常量 EPRU_UNIT_MM = 0.0254）
  epru.py               .epru 日志解析 / 序列化 / .epro2 容器读写
  eprj3.py              EDA 工程文件（.eprj3）读写
  primitives.py         图元载荷、路径语法编解码
  geometry.py           复杂多边形绕向规范化（外框 CW、内洞 CCW）
  shapes.py             基础图形（矩形 / 圆角 / 圆）几何
  layers.py             60 条图层定义（取自 EasyEDA Pro 3.2.149）
  design_layers.py      本工具使用的七个图层、颜色与绘制顺序
  templates.py          工程骨架记录
  image_prep.py         图像去噪、二值化、极性判定、主体掩膜
  image_size.py         像素尺寸读取（不依赖 Qt）
  lineart.py            线稿转换（填充色块 / 描边+填充 / 纯线条）
  denoise.py            降噪（移植自 PCB_lightgraph，MIT）
  boardart.py           板画风格：双色块面 + 内缩白轮廓
  outline.py            轮廓提取、简化、轴向吸附、圆识别、工艺补偿
  compose.py            异形板组合（描边与预设形状并集）
  standee.py            立牌插舌几何
  vectorize.py          掩膜 → 复杂多边形（FILL 已实现，TRACE 预留）
  curve_fit.py          折线 → 圆弧拟合
  vtrace.py             spline → 弧
  validation.py         轮廓有效性 + 工艺规则校验
  generator.py          板框生成器 + FILL 写入
  export.py             导出 .epro2
  gerber.py             导出 Gerber + Excellon 钻孔 + 打包 zip
  merge.py              多张卡合并为一个 .epro2
  color_image.py        .epro2 中的彩色图片对象（OBJ + BLOB 内嵌 PNG）
  text_render.py        文字 → 掩膜（排版与锚点标定）
  draft.py              草稿存取、指纹脏标记、草稿库
  projects.py           多工程工作区模型（纯 Python）
  artboard.py           设计文档模型（板框 + 图片放置 + 文字 + 挖槽）
  undo.py               撤销（整份文档快照，每块板独立栈）
  paths.py              写盘路径（源码运行 / 打包后分别解析）
  work_scale.py         大图受限分辨率计算

  —— 通行证 ——
  badge_template.py     从模板工程中读取板框与挖槽
  badge_card.py         两张模板的区域表（立绘短区 / 长区等）
  badge_content.py      四要素：二维码 / 条形码 / 职业图标 / 阵营 logo
  badge_art.py          模板固定装饰（边框、纹理、罗德岛标志）
  badge_decorations.py  装饰几何（区域 + 描边）
  badge_raster.py       二维码与条形码的位图光栅化
  qr_code.py            二维码编码器（字节模式、纠错 L/M、v1–4）
  code128.py            Code128B 条形码编码器
  badge_codes.py        PRTS 链接、短链入口、尺寸与扫描警告
  short_url.py          短网址接口
  badge_short_urls.py   干员短链映射表
  operators.py          干员资料库（职业 / 编号 / 阵营）
  operator_names.py     干员中文名
  operator_names_en.py  干员英文名
  figure_crop.py        立绘裁切辅助
  face_model.py         人脸检测接口（YuNet）

  —— 界面与外部依赖 ——
  ui_theme.py           MD3 配色（从一个主色计算）
  ui_icons.py           矢量图标（Iconoir / Lucide）
  model_paths.py        外接重绘模型路径的四级解析
  models/face.onnx      YuNet 人脸模型（随包分发）

tools/                  命令行工具与界面
  gui_mockup.py         桌面应用主体
  design_canvas.py      画布交互（选中 / 拖动 / 缩放 / 平移 / 取景）
  qt_theme.py           Qt 侧主题（样式表、图标光栅化）
  make_board_art.py     命令行出图
  make_outline_from_image.py   图片 → 异形板框
  make_color_set.py     批量导出（单卡 + 合并 + 总览）
  render_board.py       按原始坐标渲染，校验朝向与几何
  verify_epro2.mjs      独立校验器（Node，不共享 Python 代码）
  acceptance_check.py   端到端验收
  simulate_ui.py        渲染多种界面状态到 PNG
docs/                   调研结论与验证清单
tests/                  1723 项测试
out/                    生成的样本与缓存（已 gitignore）
```

## 已确认的格式事实

| 项目 | 结论 |
|---|---|
| 容器 | ZIP：`project2.json` + `<标题>.epru` + `IMAGE/` |
| 日志 | 按行，`{header}\|\|{body}\|`，UTF-8，紧凑 JSON |
| 单位 | 1 单位 = 1 mil = 0.0254 mm（文件内自证，偏差 0.0000%） |
| 存储精度 | 4 位小数 |
| Y 轴 | 文件 Y 向上，与设计坐标系一致（不翻转） |
| 板框 | `layerId = 11`，`POLY` 图元 |
| 板框线宽 | 4.9213 单位 = 0.125 mm |
| 圆形 | `path: ["CIRCLE", cx, cy, r]` 原生形式 |
| 填充绕向 | 孔必须与外环相反，外环方向不限 |
| `LAYER_PHYS.isKeepIsland` | 铜层必须为 `true`，否则编辑器会静默删除孤立铜 |
| 新图元位置 | 必须插在目标文档段内（如 `PCB` 段末尾） |
| 镂空 | layer 11 上的孤立内环只绘制不切除，因此本工具不做板框开孔 |

单位系数由文件内数据自证：样本工程中一条尺寸标注跨 3932.0079 单位，
EasyEDA 标注为 `99.873mm`，而 `3932.0079 × 0.0254 = 99.8730`。
详见 [docs/M0_FINDINGS.md](docs/M0_FINDINGS.md)。

规范正文与编辑器实际写出的数据冲突时，以编辑器数据为准。已确认四处：
单位系数、Y 轴方向、`PAD` 形状字段形式、`isKeepIsland`。

## 设计原则

1. 内部统一使用毫米，换算只发生在编解码边界，收敛到 `pcbart/units.py`
   的唯一常量
2. 尺寸正确不等于几何正确。上下翻转和边倾斜都不改变尺寸，验收必须包含
   视觉验证
3. 几何判定使用成熟库（Shapely / OpenCV），不手写
4. 用独立的第二实现做校验，避免自己写自己读；校验器的覆盖范围就是保护范围
5. 写文件前校验，把无效板挡在生成阶段
6. 先测量再动手，假设用数据验证
7. 界面显示与实际行为保持一致，不做无法工作的占位控件

## 验证

```bat
run.bat test                                    :: 1723 项测试
node tools\verify_epro2.mjs out\sample.epro2    :: 独立校验（Node）
python tools\acceptance_check.py                :: 端到端验收
```

生成器（Python）与校验器（Node）是两套独立实现，不共享代码。

## 文档索引

| 文档 | 内容 |
|---|---|
| [PROGRESS.md](PROGRESS.md) | 进度快照 |
| [ROADMAP.md](ROADMAP.md) | 技术调研与路线图 |
| [docs/PACKAGING.md](docs/PACKAGING.md) | 打包流程与版本号规则 |
| [docs/M0_FINDINGS.md](docs/M0_FINDINGS.md) | 格式定标结论 |
| [docs/M0_CHECKLIST.md](docs/M0_CHECKLIST.md) | M0 人工验证清单 |
| [docs/M2_STAR.md](docs/M2_STAR.md) | 五角星板框几何推导 |
| [docs/M2_OUTLINE.md](docs/M2_OUTLINE.md) | 图像转板框流程 |
| [docs/M2_FIXES.md](docs/M2_FIXES.md) | 几何缺陷的定位与修复 |
| [docs/M2_CUTOUTS.md](docs/M2_CUTOUTS.md) | 开孔方案（历史记录，功能已移除） |
| [docs/M3_ENCODING.md](docs/M3_ENCODING.md) | 铜层编码与对照实验 |
| [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | 第三方资源与许可证 |

## 许可证

Copyright (C) 2026 PCB Art Board Tool contributors

本项目以 [GNU General Public License v3.0](LICENSE)（GPL-3.0-only）发布。
