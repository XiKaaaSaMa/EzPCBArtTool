# java_lineart — PCB 艺术板工具的 Java 滤镜小工具

主程序通过 `subprocess` 调用本工具（`pcbart/java_filters.py`），两者互不干扰；
此文件夹可独立分发/使用。**只提供滤镜**：位图矢量化（ImageTracer）没有随本仓库分发。

## 功能

| 命令 | 功能 | 依赖 | 许可证 |
|---|---|---|---|
| `filter` | 16 种滤镜（线稿类 + 艺术类） | filters-2.0.235-1.jar | Apache-2.0（JH Labs） |
| `list` | 列出全部滤镜名 | — | — |

滤镜清单（`null` 置首 = **不滤镜，原样输出**，主程序下拉框的默认项）：
`null`、`stamp`（印章/复印风格线稿）、`skeleton`、`edge`、`contour`、`dog`、
`threshold`、`crystallize`、`marble`、`chrome`、`weave`、`pointillize`、
`emboss`、`smear`、`diffuse`、`median`、`sharpen`。

## 构建与使用

```bat
build.cmd                                       :: 编译（需 JDK；一次即可）
run.cmd list                                    :: 列出滤镜
run.cmd filter stamp 输入.png 输出.png
```

`classes\LineArtJavaTool.class` **已按 Java 8 目标预编译并随仓库分发**，
所以运行端只需要 **JRE 8+**（不必装 JDK）；只有改过 `src\LineArtJavaTool.java`
或删掉了 `classes\` 时才需要 javac。主程序在 `classes\` 缺失且系统有 javac 时
会自己编译一次。

## 设计说明

- 输入图**先按直通 alpha 合成到白底**（`Graphics2D.drawImage(..., Color.WHITE, ...)`），
  与主程序的透明图处理规则一致；
- ⚠️ **合成后的图必须是 `TYPE_INT_ARGB`，不能是 `TYPE_INT_RGB`**（2026-10-13 修）。
  JH Labs 的滤镜按「图自带 alpha 通道」写，喂无 alpha 的 `TYPE_INT_RGB` 会出两
  个**静默坏掉的滤镜**（实测）：
  - `sharpen`：走的 `ConvolveFilter` 默认 `premultiplyAlpha=true`，无 alpha 图取出
    的 alpha 恒为 0，预乘把每个像素乘成 0 —— **输出整幅全黑**（均值 0.0；换 ARGB 后
    217.2，与"原图 + 边缘"相符）；
  - `dog`：内部两次 BoxBlur 相减后做 min/max 归一化，常量图使 `max-min=0` ——
    **抛 `ArithmeticException: / by zero`**。

  换成 ARGB（alpha 恒 255）后 16 个滤镜全部正常。回归网是
  `tests/test_java_filters.py::test_every_filter_produces_a_usable_picture`
  （逐个滤镜断言非常量、不抛异常；注入回 `TYPE_INT_RGB` 即变红，报的就是上面两个）。
  ⚠️ 本 bug 同样存在于线稿工坊（它的 `readFlat` 也是 `TYPE_INT_RGB`）。
- 滤镜参数用 JH Labs 各自的默认值（上游没有暴露参数的入口，主程序也只暴露滤镜名）；
- 整个工具只依赖 `lib/` 下那个 373 KB 的小 jar，随仓库入库。
- ⚠️ **`build.cmd` / `run.cmd` 保持纯 ASCII**：cmd 在 `chcp` 生效前按 OEM
  代码页读取批处理文件，文件里的中文注释会把解析弄坏（踩过一次）。

## 许可证

- **JH Labs Filters**：Apache-2.0，`lib/filters-2.0.235-1.jar`（源码镜像
  <https://github.com/axet/jhlabs>），许可全文见 `lib/jhlabs_LICENSE.txt`。
- 本工具自身的 Java 包装代码（`src/LineArtJavaTool.java`）：随本仓库许可（GPL-3.0）。
