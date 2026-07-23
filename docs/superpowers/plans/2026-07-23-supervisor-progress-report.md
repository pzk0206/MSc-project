# 导师项目进展汇报 PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份三页、中文、结果优先的导师项目进展汇报 PDF，并保留可编辑生成脚本。

**Architecture:** 使用独立的 Matplotlib 生成脚本绘制 A4 横向页面，通过 `PdfPages` 输出三页 PDF。脚本只读取一个已验证的项目可视化样例；所有实验数字固定来自 `docs/agent/CURRENT_STATUS.md`，生成后用 `pdfinfo`、`pdftotext` 和 `pdftoppm` 验证页数、文字与视觉布局。

**Tech Stack:** Python 3、Matplotlib 3.10.9、Pillow、Noto Sans CJK SC、Poppler 工具

## Global Constraints

- 输出正好三页，保存在 `docs/reporting/`。
- 使用中文白底、深蓝标题、青绿色重点和少量橙色提示。
- 五次重复实验必须显示均值和标准差。
- Grounding DINO 的 885/885 结果只能描述为当前设置下的样本覆盖。
- 不修改实验代码，不重新运行训练，不提交 `data/` 中的生成结果。

---

### Task 1: 实现 PDF 生成器

**Files:**
- Create: `docs/reporting/generate_supervisor_progress_report.py`
- Create: `docs/reporting/supervisor_progress_report_2026-07-23.pdf`

**Interfaces:**
- Consumes: `data/processed/vlm/grasp/visualizations/success/01_pcd0100.png`
- Produces: `main() -> None`，写出三页 PDF

- [ ] **Step 1: 创建生成脚本**

脚本定义 `setup_fonts()`、`add_header()`、`add_footer()`、`draw_page_1()`、`draw_page_2()`、`draw_page_3()` 和 `main()`。页面内容固定为：

```python
FULL_RESULTS = [
    ("传统 CV", "56.95%", "0.3360", "29.62°"),
    ("VLM + 几何", "73.33%", "0.4182", "14.81°"),
    ("VLM + CNN（5 次）", "74.51% ± 1.38%", "0.4510 ± 0.0081", "16.49° ± 0.72°"),
]
UNSEEN_RESULTS = [
    ("VLM + 几何", "75.3%"),
    ("VLM + CNN（5 次）", "82.35% ± 4.53%"),
]
```

第一页绘制研究目标、数据集、评估标准、三条流程和一个成功样例；第二页绘制完整数据表、未见物体对比和三项发现；第三页绘制已完成、局限、下一步和两篇关键参考文献。

- [ ] **Step 2: 运行生成器**

Run:

```bash
conda run -n msc-grasp python docs/reporting/generate_supervisor_progress_report.py
```

Expected: 输出 `docs/reporting/supervisor_progress_report_2026-07-23.pdf`，命令退出码为 0。

- [ ] **Step 3: 检查源文件与 PDF**

Run:

```bash
git diff --check -- docs/reporting/generate_supervisor_progress_report.py
pdfinfo docs/reporting/supervisor_progress_report_2026-07-23.pdf
```

Expected: `Pages: 3`，页面尺寸为 A4 横向附近，源文件无空白错误。

### Task 2: 验证内容与视觉布局

**Files:**
- Verify: `docs/reporting/supervisor_progress_report_2026-07-23.pdf`
- Generate temporarily: `/tmp/msc-supervisor-report/page-1.png`
- Generate temporarily: `/tmp/msc-supervisor-report/page-2.png`
- Generate temporarily: `/tmp/msc-supervisor-report/page-3.png`

**Interfaces:**
- Consumes: Task 1 输出的 PDF
- Produces: 页数、文本和视觉检查结果

- [ ] **Step 1: 验证关键文字和数字**

Run:

```bash
pdftotext docs/reporting/supervisor_progress_report_2026-07-23.pdf -
```

Expected: 文本包含 `56.95%`、`73.33%`、`74.51% ± 1.38%`、`82.35% ± 4.53%`、`IoU ≥ 0.25` 和 `角度误差 ≤ 30°`。

- [ ] **Step 2: 渲染三页预览**

Run:

```bash
mkdir -p /tmp/msc-supervisor-report
pdftoppm -png -r 120 docs/reporting/supervisor_progress_report_2026-07-23.pdf /tmp/msc-supervisor-report/page
```

Expected: 生成 `page-1.png`、`page-2.png` 和 `page-3.png`。

- [ ] **Step 3: 逐页视觉检查**

检查三页预览，确认中文正常、无重叠或截断、正文足够清晰、表格数字对齐、页脚页码正确。如有问题，修改生成脚本、重新生成并重复 Task 2。

- [ ] **Step 4: 提交汇报源文件和 PDF**

```bash
git add docs/reporting/generate_supervisor_progress_report.py \
  docs/reporting/supervisor_progress_report_2026-07-23.pdf
git commit -m "docs: add supervisor progress report"
```

