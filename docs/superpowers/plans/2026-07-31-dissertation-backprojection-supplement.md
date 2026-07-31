# Dissertation Backprojection Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the verified PyBullet nine-point depth-backprojection study to the Chinese dissertation as bounded supplementary feasibility evidence.

**Architecture:** Keep the three Cornell research questions and all Cornell tables unchanged. Add a self-contained methodology subsection and a supplementary result subsection, then propagate only the evidence boundary into discussion, conclusion, and the reproducibility appendix. Check every number against the saved real-run CSV/JSON before compiling the PDF.

**Tech Stack:** LaTeX/XeLaTeX through Tectonic, Chinese `xeCJK`, `booktabs`, `hyperref`, JSON/CSV evidence, `pdftotext`, `pdfinfo`, pytest.

## Global Constraints

- PyBullet is supplementary evidence, not a fourth research question.
- Cornell remains the RGB-only main experiment; its metrics and tables must not change.
- Numerical claims come only from `data/processed/pybullet/multi_object_study/`.
- Segmentation and `rayTest` are post-hoc audits, never coordinate inputs.
- The nine-point gate is not a physical success rate and does not rank backends.
- No grasp-pose generation, IK, collision planning, joint control, or gripper execution is claimed.
- Attribute camera/depth conventions to the official Bullet Quickstart Guide.

---

### Task 1: Scope and methodology

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:71-78`
- Modify: `uog_dissertation_outline/l4proj.tex:177-193`

**Interfaces:**
- Consumes: the approved dissertation supplement design
- Produces: the scope boundary and `PyBullet 补充坐标验证方法` subsection

- [ ] **Step 1: Capture the pre-change anchors**

Run `rg -n "研究范围限定为 Cornell|PyBullet 补充坐标验证方法" uog_dissertation_outline/l4proj.tex`.
Expected: the old scope exists; the new subsection does not.

- [ ] **Step 2: Amend introduction scope**

State that the three research questions remain Cornell RGB-only comparisons, while a separate fixed-scene PyBullet supplement tests 2-D-centre-to-world-point conversion. Explicitly exclude ranking, complete pose, collision, planning, force control, and physical execution. Update `论文结构` without adding a research question.

- [ ] **Step 3: Add exact method protocol**

Insert `\section{PyBullet 补充坐标验证方法}` before `Cornell 矩形评价`. Cover one shared 640×480 frame, three targets × three backends, nearest-pixel half-up sampling, metric-depth-to-buffer inversion, column-major inverse view/projection matrices, and reprojection. Add the official URL in a footnote:

```latex
\url{https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf}
```

Record exact gates: `0.05 m < depth < 3.0 m`, finite coordinates, pixel error `<= 1 px`, depth error `<= 1e-4 m`, matching segmentation body, and matching first `rayTest` body. State segmentation/ray never correct coordinates and IK is not called.

- [ ] **Step 4: Verify and commit**

```bash
rg -n "三个主要研究问题仍限定|PyBullet 补充坐标验证方法|不调用 IK|不修正坐标" uog_dissertation_outline/l4proj.tex
git diff --check
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: add dissertation backprojection method"
```

Expected: all phrases exist and the research-question enumerate block is unchanged.

---

### Task 2: Verified nine-point results

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:288-307`

**Interfaces:**
- Consumes: `backprojection_results.csv` and `backprojection_summary.json`
- Produces: section `补充 PyBullet 坐标验证` and table `tab:pybullet-backprojection`

- [ ] **Step 1: Extract evidence independently**

Read the CSV with Python's `csv` module and JSON with `json`; assert exact target/backend order and print count, five pass counts, maximum pixel/depth/ray distance, and total gate. Expected values are `9`, five `9`s, `8.038873388460929e-13`, `3.573918431198919e-07`, `0.006756672381980784`, and `True`.

- [ ] **Step 2: Add the supplementary section and table**

Insert it before `定性案例与失败分析`, not as research question four. Table rows:

```text
有限坐标              9/9   所有值有限
有效米制深度          9/9   0.05 m < d < 3.0 m
重投影门控            9/9   <= 1 px and <= 1e-4 m
segmentation 目标匹配 9/9   事后审计
rayTest 目标匹配      9/9   事后审计
总门控                通过  九条顺序完整
```

Report rounded maxima `8.04\times10^{-13}` px, `3.57\times10^{-7}` m, and ray diagnostic distance `0.00676` m. The caption must say this is one fixed-scene integration gate, not a physical grasp success rate.

- [ ] **Step 3: Extend result summary and commit**

State that the supplement validates coordinate/data-flow consistency only, not pose validity, IK reachability, collision safety, or physical grasping.

```bash
rg -n "tab:pybullet-backprojection|8\.04|3\.57|0\.00676|不是物理抓取成功率" uog_dissertation_outline/l4proj.tex
git diff --check
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: report nine-point backprojection gate"
```

---

### Task 3: Discussion and conclusion boundaries

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:312-368`

**Interfaces:**
- Consumes: Task 2 result language
- Produces: consistent discussion, limitation, contribution, and future-work claims

- [ ] **Step 1: Add `从二维中心到世界坐标的补充证据`**

Explain that `9/9` validates software data flow and coordinate conventions. Explicitly separate: coordinate consistency from pose validity; IK reachability from collision-free motion; collision-free motion from stable closure/lift. Note that nine correlated rows from one deterministic scene provide neither population uncertainty nor backend ranking.

- [ ] **Step 2: Correct limitations and contribution**

Keep Cornell training/evaluation RGB-only; say depth is used only after 2-D prediction in the supplement. Add one contribution sentence calling the reproducible centre-to-world gate an integration artifact, not a new grasp-planning algorithm.

- [ ] **Step 3: Order future work and commit**

Use this dependency order: grasp orientation and pre-grasp pose; IK reachability; collision checking; trajectory execution; gripper closure; contact/lift success detection; real camera-to-robot calibration. Do not claim any are implemented.

```bash
rg -n "从二维中心到世界坐标的补充证据|相关输出|IK 可达|集成产物|预抓取" uog_dissertation_outline/l4proj.tex
git diff --check
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: bound dissertation backprojection claims"
```

---

### Task 4: Reproducibility appendix

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:394-442`

**Interfaces:**
- Consumes: the real `run_multi_object_study.py` CLI and output contract
- Produces: exact command, files, and metadata boundaries

- [ ] **Step 1: Add the real command and files**

Add the CUDA command with output directory `data/processed/pybullet/multi_object_study`. Add `backprojection_results.csv`, `backprojection_summary.json`, and `metadata.json` to the output inventory.

- [ ] **Step 2: Record exact metadata flags**

Document: `depth_used_after_2d_prediction: true`, `segmentation_used_as_coordinate_input: false`, `ray_test_used_as_coordinate_input: false`, `ik_executed: false`, and `physical_grasp_executed: false`.

- [ ] **Step 3: Verify and commit**

```bash
rg -n "run_multi_object_study|backprojection_results|depth_used_after_2d_prediction|ik_executed" uog_dissertation_outline/l4proj.tex
git diff --check
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: document backprojection reproduction"
```

---

### Task 5: Compile, inspect, and record completion

**Files:**
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`
- Verify: `uog_dissertation_outline/l4proj.tex`
- Generate outside Git: `/tmp/msc-dissertation-backprojection-build/l4proj.pdf`

**Interfaces:**
- Consumes: Tasks 1–4
- Produces: compiled and inspected PDF plus project records

- [ ] **Step 1: Audit invariants**

Use a read-only Python script to check saved counts/maxima, the rounded thesis values, exactly three research-question headings, and that every `\citep`/`\citet` key exists in `l4proj.bib`.

- [ ] **Step 2: Compile and text-check**

```bash
mkdir -p /tmp/msc-dissertation-backprojection-build
conda run -n msc-grasp tectonic --outdir /tmp/msc-dissertation-backprojection-build --keep-logs uog_dissertation_outline/l4proj.tex
pdftotext /tmp/msc-dissertation-backprojection-build/l4proj.pdf /tmp/msc-dissertation-backprojection-build/l4proj.txt
rg -n "PyBullet 补充坐标验证方法|补充 PyBullet 坐标验证|从二维中心到世界坐标|backprojection_results" /tmp/msc-dissertation-backprojection-build/l4proj.txt
pdfinfo /tmp/msc-dissertation-backprojection-build/l4proj.pdf
```

Expected: compilation succeeds, no undefined references, and all four anchors are extractable.

- [ ] **Step 3: Visually inspect the new table page**

Locate the result section page from extracted text, convert it with `pdftoppm`, and inspect it with the image viewer. Reject overflow, clipped Chinese, or a broken URL.

- [ ] **Step 4: Run regression and update records**

```bash
conda run -n msc-grasp python -m pytest -q
git diff --check
```

Expected: at least `97 passed`. Record the dissertation integration, page count, numeric audit, visual inspection, and tests in `CURRENT_STATUS.md` and `WORKLOG.md`; do not record an IK result.

- [ ] **Step 5: Commit**

```bash
git add docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md
git commit -m "docs: verify dissertation backprojection supplement"
```
