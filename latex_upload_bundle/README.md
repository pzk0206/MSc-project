# LaTeX / Overleaf 上传包

这是论文的独立上传副本，已经包含主文件、参考文献、模板、正文所用图片和模板字体。

## 编译设置

- 主文件：`l4proj.tex`
- 编译器：XeLaTeX
- 参考文献：BibTeX（Overleaf 会在重新编译时自动运行）

在 Overleaf 中新建空白项目后，将本文件夹内的所有内容上传到项目根目录，确认
Main document 为 `l4proj.tex`、Compiler 为 XeLaTeX，然后点击 Recompile。

不要使用 pdfLaTeX；论文包含中文，必须使用 XeLaTeX。主文件优先使用 Noto CJK
字体；系统没有 Noto CJK 时会自动回退到 TeX Live 自带的 Fandol 字体。

该目录未包含 `.aux`、`.log`、`.out`、`.toc`、`.bbl`、`.blg` 或旧 PDF 等本地
编译产物。
