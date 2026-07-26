# Chinese Dissertation Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按论文写作指南把现有英文 `l4proj.tex` 结构性改写为完整、可编译、数字与引用可追溯的中文 MSc 毕业论文。

**Architecture:** 以当前已编译英文稿和 Git 提交 `a4d62d6` 为可恢复基线，先建立中文 XeLaTeX 排版和文献证据门禁，再按 Literature Review → Introduction → Methodology → Results → Discussion → Conclusion → Abstract 的顺序逐章改写。每章单独验证字符数、研究问题、引用或实验数字，最后做全论文 Tectonic 编译、PDF 中文回读和状态记录。

**Tech Stack:** LaTeX、Tectonic/XeTeX、xeCJK、BibTeX/natbib、Noto CJK 字体、Python 3.10 标准库、现有 JSON/CSV 实验产物、Git。

## Global Constraints

- 覆盖 `uog_dissertation_outline/l4proj.tex`，不另建双语论文。
- 英文旧版本必须可从 Git 提交 `a4d62d6` 恢复。
- 标题、章节、小节、正文、图表标题和附录说明使用中文。
- 专有名词首次出现时使用“中文（英文）”，作者、文献标题、BibTeX 和引用 key 保持英文。
- 使用 `Noto Serif CJK SC`、`Noto Sans CJK SC` 和 `Noto Sans Mono CJK SC`。
- 保留现有 LaTeX label/ref、表格、图片、公式、命令、路径和 BibTeX key。
- 只使用已经审计的实验数字；不得生成或猜测新数字。
- 固定目录、五随机种子和 image-wise 五折必须分开报告。
- Image-wise 不得描述为 object-wise、未见物体泛化或标准物体级协议。
- Cornell rectangle success 不得描述为物理机器人抓取成功。
- RGB VLM crop 与 RGB-D、密集输出和物理抓取结果只能有限比较。
- Literature Review 按主题组织，重点批判分析 4–5 篇论文，不逐篇堆砌摘要。
- 重点技术结论只依赖论文原文、作者页面、DOI 或官方开放访问页面。
- 不复制论文长段原文；所有中文内容独立概括。
- 引用、复制或改编外部代码时遵循 `AGENTS.md` 的出处标注规则。
- 除教育复用表单的注释外，完成稿不含 `\todo{}` 或文本占位符。
- 每章完成后运行针对性检查并提交，不等待整篇结束才保存。

---

### Task 1: 建立文献证据与中文排版门禁

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:1-40`
- Read: `uog_dissertation_outline/l4proj.bib`
- Read: `docs/planning/modern_2d_grasp_literature_matrix.md`
- Read: `docs/planning/experiment_result_provenance.md`

**Interfaces:**
- Consumes: 现有英文 LaTeX、12 个已核对 BibTeX 条目和正式实验溯源。
- Produces: 可编译中文骨架、确定的重点文献事实边界和后续章节共同引用规则。

- [ ] **Step 1: 记录英文恢复点和当前文件哈希**

Run:

```bash
git show a4d62d6:uog_dissertation_outline/l4proj.tex >/tmp/l4proj_english_a4d62d6.tex
sha256sum /tmp/l4proj_english_a4d62d6.tex \
  uog_dissertation_outline/l4proj.tex
```

Expected: 两个文件可读取；`/tmp` 文件作为只读恢复参考，不写回仓库。

- [ ] **Step 2: 浏览五篇重点论文的一手来源**

只打开以下一手来源并记录每篇的输入、输出、协议、指标、优点和限制：

```text
Lenz et al.:
https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

Redmon and Angelova:
https://arxiv.org/abs/1412.3128

Morrison et al.:
https://arxiv.org/abs/1804.05172

Grounding DINO:
https://arxiv.org/abs/2303.05499

Language-Driven Grasp Detection:
https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-Driven_Grasp_Detection_CVPR_2024_paper.html
```

Expected: 不使用博客或二手排行榜；不复制超过版权限制的原文。

- [ ] **Step 3: 给 LaTeX 增加中文支持**

在 `\usepackage{pdfpages}` 后增加：

```tex
\usepackage{xeCJK}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
```

把标题改为：

```tex
\title{开放词汇视觉语言模型在二维机器人抓取检测中的评估}
```

保留作者、课程、导师和年份信息。

- [ ] **Step 4: 用最小中文正文验证编译**

暂不删除其他章节，只把 Abstract 的第一个占位符临时替换为一句可保留的中文
背景句：

```tex
机器人抓取检测需要从视觉观测中同时确定目标位置与可执行的夹爪几何。
```

Run:

```bash
cd uog_dissertation_outline
XDG_CACHE_HOME=/tmp/msc-tectonic-cache \
  /home/pzk/miniconda/envs/msc-grasp/bin/tectonic \
  --keep-logs l4proj.tex
pdftotext l4proj.pdf - | rg "机器人抓取检测"
cd ..
```

Expected: Tectonic 退出码 0，PDF 回读能找到中文句子，没有缺字错误。

- [ ] **Step 5: 提交排版门禁**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: enable Chinese dissertation typesetting"
```

---

### Task 2: 重写 Literature Review

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `Literature Review` 章节

**Interfaces:**
- Consumes: Task 1 的五篇一手来源、现有 BibTeX key 和比较矩阵。
- Produces: 3000–4000 中文字的六节批判性文献综述和明确 Literature Gap。

- [ ] **Step 1: 将章节标题和六个小节固定为中文**

使用：

```tex
\chapter{文献综述}
\section{机器人抓取检测}
\section{二维抓取矩形与 Cornell 基准}
\section{传统计算机视觉与深度学习方法}
\section{视觉语言模型与开放词汇定位}
\section{轻量与参数高效适配}
\section{研究缺口}
```

- [ ] **Step 2: 写“机器人抓取检测”与“Cornell 基准”**

必须包含：

- 抓取检测连接目标定位、几何表示和操作约束；
- Jiang 的矩形表示；
- Lenz 的两阶段候选评分；
- Redmon 的全局直接回归；
- Cornell rectangle metric 的 IoU/角度规则；
- image-wise 与 object-wise 的含义；
- 对 Cornell 无法代表复杂开放世界和物理成功率的批判。

引用：

```tex
\citep{jiang2011efficient,lenz2015deep,redmon2015real}
```

- [ ] **Step 3: 写“传统 CV 与深度学习方法”**

按方法演进比较：

1. 阈值/轮廓/主轴几何规则；
2. 全局 CNN 和 RGB-D ResNet；
3. GG-CNN 密集深度输出；
4. GR-ConvNet 与 Gaussian-guided 生成式网络。

必须分析：

- 可解释性与手工阈值脆弱性；
- 数据依赖、模型容量和增强；
- 单矩形与密集多抓取表达力；
- RGB/RGB-D 差异；
- 离线指标与物理执行差异；
- 高准确率不能脱离协议形成排行榜。

引用：

```tex
\citep{kumra2017robotic,morrison2018closing,kumra2020antipodal,li2022gaussian}
```

- [ ] **Step 4: 写“VLM 与开放词汇定位”**

论证顺序：

- CLIP 建立视觉—语言对齐背景；
- Grounding DINO 把文本条件引入开放集检测；
- 开放词汇框能解决目标选择，但不直接预测抓取可行性；
- Vuong 将语言指定目标和抓取结合，但任务、数据和指标与 Cornell 不同；
- VLM 定位覆盖率不能被解释为定位质量或抓取成功。

引用：

```tex
\citep{radford2021learning,liu2023grounding,vuong2024language}
```

- [ ] **Step 5: 写“轻量适配”与“研究缺口”**

LoRA/QLoRA 只说明参数高效适配思想：

```tex
\citep{hu2021lora,dettmers2023qlora}
```

明确本项目未实施微调，因为核心问题是受控比较定位前端和抓取后端。研究缺口
必须落到：

```text
RGB-only + 单目标 Cornell + 通用 prompt + 共同 VLM crop
+ 几何/单头/多头受控比较 + 可审计共同 fold
```

- [ ] **Step 6: 检查长度、引用和批判性词汇**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{文献综述}", 1)[1].split(
    r"\chapter{Methodology}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", chapter))
assert 3000 <= count <= 4500, count
for key in (
    "jiang2011efficient", "lenz2015deep", "redmon2015real",
    "kumra2017robotic", "morrison2018closing",
    "kumra2020antipodal", "li2022gaussian",
    "radford2021learning", "liu2023grounding",
    "vuong2024language", "hu2021lora", "dettmers2023qlora",
):
    assert key in chapter, key
for concept in ("输入模态", "数据划分", "输出形式", "有限可比", "研究缺口"):
    assert concept in chapter, concept
print({"literature_review_chinese_chars": count})
PY
```

Expected: 3000–4500 中文字符、12 个 key 和五项批判性概念齐全。

- [ ] **Step 7: 提交文献综述**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese literature review"
```

---

### Task 3: 重写 Introduction

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `Introduction` 章节

**Interfaces:**
- Consumes: Task 2 的 Literature Gap。
- Produces: 约 1000 中文字、三个固定研究问题和可验证目标。

- [ ] **Step 1: 使用固定中文结构**

```tex
\chapter{引言}
\section{背景与研究动机}
\section{研究问题}
\section{研究目标与具体任务}
\section{研究范围与意义}
\section{论文结构}
```

- [ ] **Step 2: 写背景、动机和三个研究问题**

研究问题必须原样保持含义：

1. Grounding DINO 是否改善传统整图几何流程？
2. 共同 VLM crop 下几何、单头和多头有何优势与局限？
3. 共同 image-wise manifest 下多头是否优于单头？

不在引言中罗列结果数字。

- [ ] **Step 3: 写目标、范围、意义和章节导航**

目标必须覆盖数据解析、传统 CV、Grounding DINO、几何后端、单头/多头 CNN、
确定性五 seed、image-wise 五折、失败分析和审计。

范围明确：

- Cornell RGB 离线感知；
- 不包括深度融合、机器人控制、碰撞或物理抓取；
- 不声称 object-wise；
- 学术意义是受控模块比较，工程意义是可替换定位前端与轻量后端。

- [ ] **Step 4: 检查长度和问题一致性**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{引言}", 1)[1].split(
    r"\chapter{文献综述}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", chapter))
assert 900 <= count <= 1400, count
for phrase in ("Grounding DINO", "几何后端", "单头", "多头", "image-wise"):
    assert phrase in chapter, phrase
assert "56.95" not in chapter
print({"introduction_chinese_chars": count})
PY
```

Expected: 900–1400 中文字符且不提前报告结果数字。

- [ ] **Step 5: 提交引言**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese introduction"
```

---

### Task 4: 重写 Methodology 并增加 Ethics

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `Methodology` 章节

**Interfaces:**
- Consumes: 三个研究问题、现有实现、正式 manifest 和方法参数。
- Produces: 1000–2000 中文字的方法选择、流程、限制和 Ethics。

- [ ] **Step 1: 中文化并组织方法小节**

使用：

```tex
\chapter{方法论}
\section{研究设计与方法选择}
\section{Cornell 数据集与评价划分}
\section{二维抓取矩形表示}
\section{传统计算机视觉基线}
\section{Grounding DINO 定位前端}
\section{VLM 引导的几何后端}
\section{VLM 引导的轻量 CNN 后端}
\section{训练、重复实验与交叉验证}
\section{Cornell 矩形评价}
\section{研究伦理}
\section{方法局限}
```

- [ ] **Step 2: 强化选择理由与控制变量**

解释：

- 三条 pipeline 分别隔离定位和后端差异；
- RGB-only 保持输入模态一致；
- 通用 `small object` prompt 避免样本级类别提示；
- 固定目录用于历史配对分析；
- 五 seed 测训练随机性；
- image-wise 五折测图像划分变化；
- 测试集不参与 scheduler、early stopping 或 checkpoint。

- [ ] **Step 3: 保留全部准确参数和公式**

保留：

- 885、600/200/85、566/142/177；
- crop 224、batch 32、学习率、权重衰减、80 epoch、patience；
- IoU 0.25 和角度 30°；
- 10% box expansion；
- `sin(2θ), cos(2θ)`；
- 单头/多头结构与 manifest SHA-256。

- [ ] **Step 4: 写 Research Ethics**

必须明确：

- 无人类参与者、问卷或私人数据；
- 公开 Cornell 数据和公开预训练模型；
- 引用与外部代码出处；
- 不夸大结果；
- 产物、环境、哈希和确定性；
- 计算资源控制。

- [ ] **Step 5: 检查方法事实与长度**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{方法论}", 1)[1].split(
    r"\chapter{Results}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", chapter))
assert 1000 <= count <= 2600, count
for fact in (
    "885", "600", "200", "85", "566", "142", "177",
    "224", "32", "0.25", "30", "small object",
    "Research Ethics", "无人类参与者", "object-wise",
):
    assert fact in chapter, fact
print({"methodology_chinese_chars": count})
PY
```

Expected: 1000–2600 中文字符，关键参数和 Ethics 边界齐全。

- [ ] **Step 6: 提交方法论**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese methodology and ethics"
```

---

### Task 5: 重写 Findings / Results

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `Results` 章节
- Read: `data/processed/vlm/cnn_cross_validation/single/cross_validation_summary.json`
- Read: `data/processed/vlm/cnn_cross_validation/multi_head/cross_validation_summary.json`
- Read: `data/processed/vlm/cnn_cross_validation/architecture_comparison.json`
- Read: `docs/planning/experiment_result_provenance.md`

**Interfaces:**
- Consumes: 全部正式结果和现有表格/图。
- Produces: 3000–4000 中文字、按研究问题组织且不混淆统计口径的结果章节。

- [ ] **Step 1: 中文化章节和图表标题**

使用：

```tex
\chapter{研究结果}
\section{数据与实现验证}
\section{研究问题一：开放词汇定位的作用}
\section{研究问题二：几何与 CNN 后端比较}
\section{研究问题三：单头与多头 CNN 比较}
\section{定性案例与失败分析}
\section{结果小结}
```

保留所有 `\label{}` 和引用关系。

- [ ] **Step 2: 写研究问题一结果**

报告并解释：

- 传统 CV 504/885、56.95%、IoU 0.3360、29.62°；
- VLM + geometry 649/885、73.33%、IoU 0.4182、14.81°；
- 16.38 个百分点增益；
- 885/885 返回定位框仅是覆盖率；
- 固定 split 难度警告。

- [ ] **Step 3: 写研究问题二结果**

使用固定 85 样本和逐样本分类：

- geometry 64/85；
- 两者成功 52、仅 CNN 16、仅 geometry 12、两者失败 5；
- CNN 在 IoU/中心尺寸上的优势；
- geometry 在角度上的优势；
- 案例观察与原因推测明确分开。

- [ ] **Step 4: 写研究问题三结果**

分别报告：

- 确定性五 seed 单头/多头；
- image-wise 单头 635/885；
- image-wise 多头 647/885；
- +1.36 个百分点、+0.0190 IoU、-0.34°；
- 多头提升有限，fold 成功率标准差反而更大；
- pooled 与 fold 标准差定义。

- [ ] **Step 5: 扩展定性案例和结果小结**

结合现有失败图：

- CNN-only；
- geometry-only；
- shared failure；
- near-threshold case；
- VLM loose box case；
- orientation failure。

不声称图片证明因果。

- [ ] **Step 6: 用 JSON 独立核对所有 image-wise 数字**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("data/processed/vlm/cnn_cross_validation")
single = json.loads((root / "single/cross_validation_summary.json").read_text())
multi = json.loads((root / "multi_head/cross_validation_summary.json").read_text())
comparison = json.loads((root / "architecture_comparison.json").read_text())
tex = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
markers = (
    "635/885", "71.75", "0.4390", "17.74",
    "647/885", "73.11", "0.4580", "17.40",
    "1.36", "0.0190", "0.34",
)
for marker in markers:
    assert marker in tex, marker
assert single["pooled"]["sample_count"] == 885
assert multi["pooled"]["sample_count"] == 885
assert comparison["manifest_sha256"] == single["manifest_sha256"]
assert comparison["manifest_sha256"] == multi["manifest_sha256"]
print({"verified_markers": len(markers)})
PY
```

Expected: 11 个数字标记和共同 manifest 校验通过。

- [ ] **Step 7: 检查章节长度并提交**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{研究结果}", 1)[1].split(
    r"\chapter{General Discussion}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", chapter))
assert 3000 <= count <= 4800, count
print({"results_chinese_chars": count})
PY
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese results chapter"
```

Expected: 3000–4800 中文字符并提交。

---

### Task 6: 重写 General Discussion

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `General Discussion` 章节

**Interfaces:**
- Consumes: Task 5 的观察结果和 Task 2 的文献边界。
- Produces: 不重复结果表、具有解释深度和文献联系的中文讨论。

- [ ] **Step 1: 使用中文讨论结构**

```tex
\chapter{综合讨论}
\section{主要发现的解释}
\section{与既有文献的关系}
\section{失败案例与误差来源}
\section{实践与理论意义}
\section{研究局限}
```

- [ ] **Step 2: 分析而非重复**

必须解释：

- 最大增益来自定位前端；
- CNN 改善空间回归，geometry 保留角度先验；
- 多头优势有限且不是 SOTA 主张；
- 固定 85 样本存在难度偏差；
- 逐样本互补性支持混合后端假设，但尚未验证；
- VLM 语义定位与抓取 affordance 不等价。

- [ ] **Step 3: 与 Literature Review 回扣**

至少回扣：

- Lenz/Redmon 的全局矩形范式；
- GG-CNN/GR-ConvNet 的密集与深度输入；
- Grounding DINO 的开放词汇定位角色；
- Vuong 的语言驱动复杂场景；
- 本项目有限可比的原因。

- [ ] **Step 4: 完整写出限制**

包括：

- Cornell 单一小数据集；
- RGB-only；
- image-wise 视角泄漏；
- object mapping 缺失；
- 通用 prompt 未做敏感性；
- 单矩形输出；
- offline metric；
- 无物理执行；
- 计算和模型容量限制。

- [ ] **Step 5: 检查讨论边界并提交**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{综合讨论}", 1)[1].split(
    r"\chapter{Conclusion}", 1
)[0]
for phrase in (
    "有限可比", "不等同", "image-wise", "object-wise",
    "物理抓取", "定位覆盖率", "多头",
):
    assert phrase in chapter, phrase
print({"discussion_chars": len(chapter)})
PY
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese general discussion"
```

Expected: 七项边界齐全并提交。

---

### Task 7: 写 Conclusion 并逐条回答研究问题

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 `Conclusion` 章节

**Interfaces:**
- Consumes: 三个研究问题和已审计结果。
- Produces: 约 1000 中文字、无新实验主张的结论。

- [ ] **Step 1: 使用四节结构**

```tex
\chapter{结论}
\section{研究总结}
\section{研究问题回答}
\section{贡献与实践意义}
\section{局限与未来工作}
```

- [ ] **Step 2: 逐条回答三个研究问题**

回答边界：

1. Grounding DINO 前端在当前 Cornell RGB/通用 prompt 条件下显著改善简单
   几何流程，但不能证明任何 VLM 对任何场景都有效。
2. CNN 改善 IoU/空间估计，geometry 提供更稳定的角度先验，二者互补。
3. 多头在共同 image-wise manifest 下改善全部 pooled 指标，但幅度有限。

- [ ] **Step 3: 写贡献和 Future Work**

贡献：

- 模块化三 pipeline；
- 公平单头/多头确定性对照；
- image-wise 可审计 manifest；
- 逐样本失败与互补性分析；
- 结果溯源与边界清晰。

Future Work：

- 权威 object-wise；
- RGB-D/深度；
- 密集多抓取；
- 混合方向先验；
- 真实机器人；
- prompt/open-world。

- [ ] **Step 4: 检查长度和新数字**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
chapter = text.split(r"\chapter{结论}", 1)[1].split(
    r"\begin{appendices}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", chapter))
assert 900 <= count <= 1400, count
for marker in ("研究问题一", "研究问题二", "研究问题三"):
    assert marker in chapter, marker
print({"conclusion_chinese_chars": count})
PY
```

Expected: 900–1400 中文字符并逐条回答。

- [ ] **Step 5: 提交结论**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: write Chinese conclusion"
```

---

### Task 8: 写 Abstract、致谢和中文附录

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex` 的 Abstract、Acknowledgements、Appendices

**Interfaces:**
- Consumes: 已完成的全部章节。
- Produces: 300–400 中文字摘要、客观致谢和中文附录说明。

- [ ] **Step 1: 最后写摘要**

摘要必须包含：

- 背景与问题；
- Cornell RGB 和三 pipeline；
- 885 样本、五 seed 和 image-wise；
- 56.95% → 73.33% 定位增益；
- image-wise 多头 73.11%、单头 71.75%；
- 模块化意义；
- RGB-only、image-wise、offline 限制。

不引用文献。

- [ ] **Step 2: 写客观致谢**

使用：

```tex
\chapter*{致谢}
感谢导师 Dr Jan Paul Siebert 在研究设计、实验解释和论文写作方面提供的指导与反馈。感谢 University of Glasgow School of Computing Science 提供完成本项目所需的学习与研究环境。
```

- [ ] **Step 3: 中文化附录说明**

保留：

- 表格；
- 命令；
- 环境版本；
- 文件路径；
- BibTeX；
- 固定 split 的历史说明。

- [ ] **Step 4: 检查摘要长度和 TODO**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

text = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
abstract = text.split(r"\begin{abstract}", 1)[1].split(
    r"\end{abstract}", 1
)[0]
count = len(re.findall(r"[\u4e00-\u9fff]", abstract))
assert 300 <= count <= 500, count
assert r"\cite" not in abstract
todos = [
    match.group(0)
    for match in re.finditer(r"\\todo\{", text)
]
assert not todos, todos
print({"abstract_chinese_chars": count, "todo_count": len(todos)})
PY
```

Expected: 300–500 中文字符、摘要无引用、全文无 TODO。

- [ ] **Step 5: 提交摘要与附录**

```bash
git diff --check -- uog_dissertation_outline/l4proj.tex
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: complete Chinese dissertation draft"
```

---

### Task 9: 全文数字、引用、构建和视觉验收

**Files:**
- Verify: `uog_dissertation_outline/l4proj.tex`
- Verify generated: `uog_dissertation_outline/l4proj.pdf`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Tasks 1–8 的完整中文论文。
- Produces: 数字、引用、排版和项目状态均通过验证的中文初稿。

- [ ] **Step 1: 检查所有引用 key**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path
import re

tex = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
bib = Path("uog_dissertation_outline/l4proj.bib").read_text(encoding="utf-8")
used = set()
for command in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
    used.update(key.strip() for key in command.split(","))
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
missing = sorted(used - defined)
assert not missing, missing
print({"used_citation_keys": len(used), "missing": missing})
PY
```

Expected: 所有引用 key 均存在。

- [ ] **Step 2: 检查统计口径和核心数字**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from pathlib import Path

tex = Path("uog_dissertation_outline/l4proj.tex").read_text(encoding="utf-8")
required = (
    "56.95", "73.33", "74.55", "75.59",
    "71.75", "73.11", "0.4390", "0.4580",
    "17.74", "17.40", "635/885", "647/885",
)
for marker in required:
    assert marker in tex, marker
for phrase in (
    "五个随机种子", "image-wise 五折", "object-wise",
    "有限可比", "不等同于物理抓取",
):
    assert phrase in tex, phrase
print({"numeric_markers": len(required)})
PY
```

Expected: 12 个核心数字和五项边界出现。

- [ ] **Step 3: 编译中文 PDF**

Run:

```bash
cd uog_dissertation_outline
XDG_CACHE_HOME=/tmp/msc-tectonic-cache \
  /home/pzk/miniconda/envs/msc-grasp/bin/tectonic \
  --keep-logs l4proj.tex
pdfinfo l4proj.pdf | rg "^(Pages|File size)"
pdftotext l4proj.pdf /tmp/l4proj_chinese.txt
for phrase in 摘要 引言 文献综述 方法论 研究结果 综合讨论 结论; do
  rg -q "$phrase" /tmp/l4proj_chinese.txt || exit 1
done
cd ..
```

Expected: 编译退出码 0，PDF 能回读七个主要中文章节。

- [ ] **Step 4: 视觉检查关键页面**

使用 PDF 页截图检查：

- 摘要；
- Literature Review 中间页；
- 主要结果表；
- image-wise 表；
- Conclusion。

检查：

- 无缺字方框；
- 表格未越过页面；
- 图题和引用可读；
- 中文行距和段落合理；
- 章节顺序与目录一致。

- [ ] **Step 5: 运行代码回归和 diff 检查**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -q
git diff --check
```

Expected: 32 个测试通过，diff 检查无输出。

- [ ] **Step 6: 更新项目状态和工作日志**

`CURRENT_STATUS.md` 写明：

- 完整中文论文初稿已覆盖英文工作稿；
- 英文旧稿可从 `a4d62d6` 恢复；
- 数字、引用和 PDF 已验证；
- 下一步是用户内容审阅、导师反馈和英文最终版决策。

`WORKLOG.md` 写明：

- 按写作指南完成全篇结构性中文改写；
- Literature Review 的五篇重点批判分析；
- Methodology Ethics；
- 三研究问题贯穿；
- Tectonic/PDF/引用/数字/测试验证结果。

- [ ] **Step 7: 最终提交**

```bash
git diff --check -- \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md \
  uog_dissertation_outline/l4proj.tex
git add docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md \
  uog_dissertation_outline/l4proj.tex
git commit -m "docs: verify Chinese dissertation draft"
```

---

## 最终验收

- [ ] 英文正文已结构性改写为中文，而非机械逐句翻译。
- [ ] Abstract 为 300–500 中文字符。
- [ ] Introduction 为 900–1400 中文字符。
- [ ] Literature Review 为 3000–4500 中文字符。
- [ ] Methodology 为 1000–2600 中文字符并包含 Research Ethics。
- [ ] Results 为 3000–4800 中文字符并按三个研究问题组织。
- [ ] Conclusion 为 900–1400 中文字符并逐条回答研究问题。
- [ ] 五篇重点论文具有优点、限制和与本项目关系的批判性分析。
- [ ] 全文无 TODO 或未解释占位符。
- [ ] 全部引用 key 存在。
- [ ] 全部正式数字与保存产物一致。
- [ ] Image-wise/object-wise、seed/fold、RGB/RGB-D 和 offline/physical 边界正确。
- [ ] 中文 PDF 编译成功且关键页面可读。
- [ ] 32 个现有测试通过。
- [ ] 当前状态和工作日志已更新。
- [ ] 英文旧稿可从 Git 提交 `a4d62d6` 恢复。
