# Chinese Introduction and Literature Review DOCX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份约 2000–3000 中文字、言简意赅、引用可追溯的 Introduction 与 Literature Review 中文 DOCX 初稿。

**Architecture:** 先在 `/tmp` 中用纯文本 Markdown 编写并审计中文正文，再用一个临时的 Python 标准库脚本将标题、段落和列表封装为最小但合法的 WordprocessingML DOCX。仓库只保存用户需要的 DOCX 和工作日志，不增加运行时依赖或长期维护的转换器。

**Tech Stack:** Markdown、Python 3.10 标准库 `zipfile/xml.etree/pathlib`、WordprocessingML、Git、现有 `l4proj.bib`。

## Global Constraints

- 正式输出固定为 `docs/drafts/introduction_literature_review_zh.docx`。
- 正文总长度约 2000–3000 中文字。
- 每个二级小节原则上使用 1–3 个短段落。
- 中文稿用于理解、批注和后续翻译，不替换现有英文 LaTeX。
- 只使用 `uog_dissertation_outline/l4proj.bib` 中已经核对的文献。
- 不引入新的实验数字或未经验证的性能主张。
- 不把 image-wise 描述为 object-wise 或未见物体泛化。
- 不把 RGB VLM crop 与 RGB-D、密集输出或物理抓取结果描述为直接公平比较。
- 不把 Cornell rectangle success 描述为真实机器人成功抓取。
- 临时构建文件全部写入 `/tmp/msc-chinese-draft-docx/`。
- 引用、复制或改编外部代码时必须标注出处；本计划的临时 DOCX 封装逻辑使用 Python 标准库独立编写，不复制外部实现。

---

### Task 1: 编写精简中文正文并完成内容审计

**Files:**
- Create temporary: `/tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md`
- Read: `uog_dissertation_outline/l4proj.tex`
- Read: `uog_dissertation_outline/l4proj.bib`
- Read: `docs/agent/CURRENT_STATUS.md`
- Read: `docs/planning/modern_2d_grasp_literature_matrix.md`

**Interfaces:**
- Consumes: 已验证的研究问题、方法边界、正式结果和现有 BibTeX 条目。
- Produces: 具有固定标题层级的 UTF-8 Markdown 正文，供 Task 2 转换。

- [ ] **Step 1: 创建临时目录**

Run:

```bash
mkdir -p /tmp/msc-chinese-draft-docx
```

Expected: 目录存在且仓库状态不变化。

- [ ] **Step 2: 用 `apply_patch` 写入正文**

Markdown 必须按以下固定结构编写：

```markdown
# Introduction 与 Literature Review 中文精简初稿

> 用途：供作者理解、批注并在后续翻译为英文；不作为当前英文论文的直接替换。

# 第一章 引言

## 1.1 背景与动机
## 1.2 研究问题
## 1.3 研究目标与具体任务
## 1.4 项目范围
## 1.5 论文结构

# 第二章 文献综述

## 2.1 机器人抓取检测
## 2.2 二维抓取矩形与 Cornell 基准
## 2.3 传统计算机视觉与深度学习方法
## 2.4 视觉语言模型与开放词汇定位
## 2.5 轻量与参数高效适配
## 2.6 研究缺口

# 引用文献提示
```

正文必须覆盖以下具体论点：

- 每个完整段落写在一个物理行内，段落之间使用空行，确保 DOCX 回读能逐段
  与源文本核对。
- 抓取检测需要同时解决目标定位和抓取几何，两者不能混为一个问题。
- 开放词汇定位的价值是作为可替换的目标选择前端，而不是直接输出可执行抓取。
- 三个研究问题必须分别对应定位增益、几何/CNN 后端差异和单头/多头公平对照。
- Cornell 矩形表示引用 Jiang et al. (2011)，候选评分引用 Lenz et al. (2015)，直接回归引用 Redmon and Angelova (2015)。
- 深度学习发展脉络包括 Kumra and Kanan (2017)、GG-CNN、GR-ConvNet 和 Gaussian-guided 方法。
- VLM 脉络包括 CLIP、Grounding DINO 和 Language-Driven Grasp Detection。
- LoRA/QLoRA 只作为可能的参数高效扩展，不声称已在本项目实施。
- 研究缺口限定为 RGB-only、VLM crop、单矩形输出下的模块化受控比较。
- Object-wise 明确记录为实例元数据阻塞；image-wise 允许同一物体不同视角跨集合。
- 结尾引用提示列出正文实际使用的 BibTeX key。

- [ ] **Step 3: 审计标题、占位符和篇幅**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

path = Path("/tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md")
text = path.read_text(encoding="utf-8")
required = [
    "# 第一章 引言",
    "## 1.1 背景与动机",
    "## 1.2 研究问题",
    "## 1.3 研究目标与具体任务",
    "## 1.4 项目范围",
    "## 1.5 论文结构",
    "# 第二章 文献综述",
    "## 2.1 机器人抓取检测",
    "## 2.2 二维抓取矩形与 Cornell 基准",
    "## 2.3 传统计算机视觉与深度学习方法",
    "## 2.4 视觉语言模型与开放词汇定位",
    "## 2.5 轻量与参数高效适配",
    "## 2.6 研究缺口",
    "# 引用文献提示",
]
for heading in required:
    assert heading in text, heading
for marker in ("TODO", "TBD", "待补充", "占位符"):
    assert marker not in text, marker
chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
assert 2000 <= chinese_chars <= 3000, chinese_chars
print({"headings": len(required), "chinese_chars": chinese_chars})
PY
```

Expected: 14 个固定标题全部存在，中文字符在 2000–3000 之间。

- [ ] **Step 4: 审计引用与结论边界**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path

draft = Path(
    "/tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md"
).read_text(encoding="utf-8")
bib = Path("uog_dissertation_outline/l4proj.bib").read_text(encoding="utf-8")
keys = [
    "jiang2011efficient",
    "lenz2015deep",
    "redmon2015real",
    "kumra2017robotic",
    "morrison2018closing",
    "kumra2020antipodal",
    "li2022gaussian",
    "radford2021learning",
    "liu2023grounding",
    "vuong2024language",
    "hu2021lora",
    "dettmers2023qlora",
]
for key in keys:
    assert f"@inproceedings{{{key}," in bib or f"@article{{{key}," in bib
    assert key in draft
assert "image-wise 不等于 object-wise" in draft
assert "不等同于真实机器人抓取成功" in draft
assert "有限可比" in draft
print({"verified_bib_keys": len(keys)})
PY
```

Expected: 12 个正文引用 key 均存在，三项结论边界均出现。

---

### Task 2: 生成结构化 DOCX

**Files:**
- Create temporary: `/tmp/msc-chinese-draft-docx/build_docx.py`
- Consume temporary: `/tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md`
- Create: `docs/drafts/introduction_literature_review_zh.docx`

**Interfaces:**
- Consumes: `#`/`##` 标题、`>` 用途说明、普通段落和 `-`/`1.` 列表。
- Produces: 标准 ZIP 容器中的 WordprocessingML 文档。

- [ ] **Step 1: 用 `apply_patch` 编写临时标准库转换脚本**

使用以下独立实现；转换脚本只保存在 `/tmp`，不成为项目长期模块：

```python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import zipfile

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def parse_markdown(text: str) -> list[tuple[str, str]]:
    paragraphs: list[tuple[str, str]] = []
    buffer: list[str] = []
    first_h1 = True

    def flush() -> None:
        if buffer:
            paragraphs.append(("Normal", " ".join(buffer)))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("## "):
            flush()
            paragraphs.append(("Heading2", line[3:]))
        elif line.startswith("# "):
            flush()
            style = "Title" if first_h1 else "Heading1"
            first_h1 = False
            paragraphs.append((style, line[2:]))
        elif line.startswith("> "):
            flush()
            paragraphs.append(("Quote", line[2:]))
        elif re.match(r"^[-*]\s+", line):
            flush()
            paragraphs.append(("List", "• " + re.sub(r"^[-*]\s+", "", line)))
        elif re.match(r"^\d+\.\s+", line):
            flush()
            paragraphs.append(("List", line))
        else:
            buffer.append(line)
    flush()
    return paragraphs


def paragraph_xml(style: str, text: str) -> str:
    style_xml = ""
    run_properties = ""
    if style in {"Title", "Heading1", "Heading2"}:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    elif style == "Quote":
        style_xml = '<w:pPr><w:ind w:left="720"/></w:pPr>'
        run_properties = "<w:rPr><w:i/></w:rPr>"
    elif style == "List":
        style_xml = '<w:pPr><w:ind w:left="480" w:hanging="240"/></w:pPr>'
    return (
        f"<w:p>{style_xml}<w:r>{run_properties}"
        f'<w:t xml:space="preserve">{escape(text)}</w:t>'
        "</w:r></w:p>"
    )


def document_xml(paragraphs: list[tuple[str, str]]) -> str:
    body = "".join(paragraph_xml(style, text) for style, text in paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"
        w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def build_docx(source: Path, output: Path) -> tuple[int, int]:
    paragraphs = parse_markdown(source.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    parts = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""",
        "word/document.xml": document_xml(paragraphs),
        "word/styles.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>
      <w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei"
        w:eastAsia="Microsoft YaHei"/>
      <w:sz w:val="22"/><w:szCs w:val="22"/>
    </w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr>
      <w:spacing w:after="120" w:line="360" w:lineRule="auto"/>
    </w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="360"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="360" w:after="180"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
</w:styles>
""",
        "word/_rels/document.xml.rels": """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""",
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties
  xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Introduction 与 Literature Review 中文精简初稿</dc:title>
  <dc:creator>Pang Zhenkun</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
</cp:coreProperties>
""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Python Standard Library</Application>
</Properties>
""",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content.encode("utf-8"))
    return len(paragraphs), output.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count, size = build_docx(args.source, args.output)
    print({"output": str(args.output), "paragraphs": count, "bytes": size})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行转换脚本**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  /tmp/msc-chinese-draft-docx/build_docx.py \
  /tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md \
  docs/drafts/introduction_literature_review_zh.docx
```

Expected: 命令打印输出路径和段落数，DOCX 大小大于 5 KB。

- [ ] **Step 3: 检查 ZIP/DOCX 完整性**

Run:

```bash
unzip -t docs/drafts/introduction_literature_review_zh.docx
file docs/drafts/introduction_literature_review_zh.docx
```

Expected:

- `unzip` 报告没有错误；
- `file` 将其识别为 Microsoft Word 2007+ 或 Zip archive data；
- 退出码均为 0。

---

### Task 3: 对 DOCX 内容做独立回读并记录产物

**Files:**
- Verify: `docs/drafts/introduction_literature_review_zh.docx`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Task 1 Markdown 和 Task 2 DOCX。
- Produces: 内容无遗漏、可追溯、已记录的最终中文初稿。

- [ ] **Step 1: 从 DOCX 独立提取文字并与源内容比较**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

source = Path(
    "/tmp/msc-chinese-draft-docx/introduction_literature_review_zh.md"
).read_text(encoding="utf-8")
docx_path = Path("docs/drafts/introduction_literature_review_zh.docx")
with zipfile.ZipFile(docx_path) as archive:
    xml = archive.read("word/document.xml")
root = ET.fromstring(xml)
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
paragraphs = []
for paragraph in root.findall(".//w:p", ns):
    text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
    if text:
        paragraphs.append(text)
extracted = "\n".join(paragraphs)
required_text = [
    "第一章 引言",
    "第二章 文献综述",
    "研究问题",
    "研究缺口",
    "image-wise 不等于 object-wise",
    "不等同于真实机器人抓取成功",
]
for value in required_text:
    assert value in extracted, value
source_plain = re.sub(r"^(#{1,2}|>|[-*]|[0-9]+\\.)\\s*", "", source, flags=re.M)
for paragraph in [
    line.strip()
    for line in source_plain.splitlines()
    if line.strip()
]:
    assert paragraph in extracted, paragraph[:60]
print(
    {
        "docx_bytes": docx_path.stat().st_size,
        "extracted_paragraphs": len(paragraphs),
    }
)
PY
```

Expected: 每个非空源段落都能从 DOCX 回读，所有边界表述存在。

- [ ] **Step 2: 更新工作日志**

在 `docs/worklog/WORKLOG.md` 的 2026-07-26 条目增加：

```markdown
- 生成精简中文 Introduction 与 Literature Review DOCX 初稿，正文约
  2000–3000 中文字；标题、引用 key、结论边界和 DOCX 回读均已验证。
```

- [ ] **Step 3: 完成最终检查**

Run:

```bash
git diff --check -- docs/worklog/WORKLOG.md
test -s docs/drafts/introduction_literature_review_zh.docx
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -q
git status --short
```

Expected: diff 检查无输出、DOCX 非空、32 个现有测试全部通过，状态只包含
DOCX、工作日志和未提交的生成型论文 PDF/日志。

- [ ] **Step 4: 提交**

```bash
git add docs/drafts/introduction_literature_review_zh.docx \
  docs/worklog/WORKLOG.md
git commit -m "docs: add concise Chinese dissertation draft"
```

---

## 最终验收

- [ ] DOCX 路径准确且文件可打开。
- [ ] 中文正文为约 2000–3000 字。
- [ ] Introduction 五个小节和 Literature Review 六个小节齐全。
- [ ] 三个研究问题与已完成实验一一对应。
- [ ] 12 个引用 key 均存在于当前 BibTeX。
- [ ] Image-wise、object-wise、RGB/RGB-D 和物理抓取边界表述准确。
- [ ] 没有 TODO、占位符、未经验证数字或未标注外部复制内容。
- [ ] DOCX 回读覆盖全部非空源段落。
- [ ] 现有测试继续通过。
