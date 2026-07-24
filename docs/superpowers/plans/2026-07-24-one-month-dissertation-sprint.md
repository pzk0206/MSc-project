# 一个月毕业论文冲刺执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2026-08-23 前完成一篇能够回应导师反馈、实验记录可信、结论范围准确的 MSc 毕业论文，并完成一个多头 CNN 对照实验或预先定义的失败分析回退。

**Architecture:** 先修复现有 CNN 多轮实验记录中的可信度问题，再增加独立的数据划分审计和后端比较工具；论文写作与图表整理紧随已验证证据推进。技术扩展复用现有 VLM crop、数据划分和 Cornell 评价协议，仅替换 CNN 回归头，从而保证单头和多头结果可直接比较。

**Tech Stack:** Python 3、PyTorch、NumPy、OpenCV、pytest、CSV/JSON、Matplotlib、LaTeX/BibTeX、Cornell Grasping Dataset、Grounding DINO 已缓存定位结果。

## Global Constraints

- 计划周期固定为 2026-07-24 至 2026-08-23。
- 2026-08-13 日结束后不增加新模型、新数据集或大型实验。
- 2026-08-16 前必须生成完整论文第一版。
- 训练目录固定为 01–06，验证目录固定为 07–08，测试目录固定为 09–10。
- Cornell 成功标准固定为 IoU ≥ 0.25 且角度误差 ≤ 30°。
- 如引用单次实验，必须与五次实验均值 ± 标准差分开报告；缺少独立产物的旧单次结果不进入论文主表。
- 目录 09–10 只在完成数据构成审计后才能被谨慎讨论为未见物体测试。
- 所有论文数字必须来自实际保存的 CSV/JSON，不得手工估算。
- `data/` 中生成物继续保持 Git 忽略；最终论文使用的图复制到 `uog_dissertation_outline/images/`。
- 现有无关修改属于用户；每次提交只暂存当前任务明确列出的文件。
- PyBullet、真实机器人、Jacquard、U-Net、预训练骨干和大规模调参不属于本计划。

---

## 文件结构

本计划创建或修改以下文件：

| 文件 | 单一职责 |
|---|---|
| `tests/test_cnn_grasp_reporting.py` | 锁定多轮实验汇总和单一 CLI 入口行为 |
| `tests/test_cornell_split_audit.py` | 锁定数据划分、共同样本和分组统计 |
| `tests/test_backend_comparison.py` | 锁定几何/CNN 逐样本交叉分类 |
| `tests/test_cnn_grasp_models.py` | 锁定单头/多头输出接口和多任务损失 |
| `src/shared/analyze_cornell_splits.py` | 生成划分审计数据、样例图和共同测试集统计 |
| `src/vlm/cnn_grasp_models.py` | 定义共享 backbone、单头模型、多头模型和损失 |
| `src/vlm/run_cnn_grasp.py` | 保留训练/评估 CLI，并支持选择单头或多头 |
| `src/vlm/analyze_backend_comparison.py` | 生成几何/CNN 逐样本交叉比较和失败案例图 |
| `docs/planning/cornell_split_audit.md` | 记录人工物体类型检查、客观统计和结论边界 |
| `docs/planning/experiment_result_provenance.md` | 记录论文数字对应的实际 CSV/JSON 和已知覆盖问题 |
| `docs/planning/modern_2d_grasp_literature_matrix.md` | 记录论文输入、划分、指标、数字及可比性 |
| `docs/planning/cnn_architecture_rationale.md` | 区分文献依据、设计原则和自主工程选择 |
| `uog_dissertation_outline/l4proj.tex` | 完成论文全部核心章节 |
| `uog_dissertation_outline/l4proj.bib` | 保存经过原文核对的参考文献 |
| `uog_dissertation_outline/images/cornell_split_contact_sheet.png` | 论文数据划分样例图 |
| `uog_dissertation_outline/images/cnn_architecture.pdf` | 论文 CNN 结构图 |
| `uog_dissertation_outline/images/backend_failure_cases.png` | 论文后端失败案例图 |
| `docs/agent/CURRENT_STATUS.md` | 只在验证结果或阶段发生变化后更新 |
| `docs/agent/PROJECT_STRUCTURE.md` | 新增主要分析/模型模块后更新 |
| `docs/worklog/WORKLOG.md` | 每个完成阶段添加一条简明记录 |

生成但不提交的实验产物：

```text
data/processed/shared/split_audit/
├── representative_samples.csv
├── split_metrics.json
├── same_test_subset_metrics.csv
└── cornell_split_contact_sheet.png

data/processed/vlm/backend_comparison/
├── sample_comparison.csv
├── comparison_summary.json
└── backend_failure_cases.png

data/processed/vlm/cnn_grasp_multi_head/
├── multi_run_summary.json
├── cnn_grasp_predictions.csv
├── training_history_seed_42.json
├── training_history_seed_43.json
├── training_history_seed_44.json
├── training_history_seed_45.json
└── training_history_seed_46.json
```

---

### Task 1（7月24–25日）：修复并锁定现有 CNN 实验记录

**Files:**
- Create: `tests/test_cnn_grasp_reporting.py`
- Create: `docs/planning/experiment_result_provenance.md`
- Modify: `src/vlm/run_cnn_grasp.py:752-1020`
- Modify: `docs/debugging/BUGLOG.md`

**Interfaces:**
- Consumes: `_train_one_run(...) -> tuple[CNNGraspRegressor, float]`
- Produces: `build_multi_run_summary(run_records: list[dict]) -> dict`
- Produces: 每个 `per_run` 项的 `best_val_loss` 必须来自训练历史，而不是成功率

- [ ] **Step 1: 记录当前已知缺陷**

在 `docs/debugging/BUGLOG.md` 添加日期为 2026-07-24 的记录，写明：

```text
1. multi_run_summary.json 的 per_run.best_val_loss 当前错误地写入 all_success_rate。
2. run_cnn_grasp.py 末尾存在两个相同的 __main__ 入口，直接执行时会重复运行完整流程。
3. 修复前生成的 per_run.best_val_loss 不得作为论文训练损失证据。
```

- [ ] **Step 2: 建立结果溯源清单**

在 `docs/planning/experiment_result_provenance.md` 记录：

```markdown
| 论文结果 | 实际文件 | method 字段 | 可复核状态 | 使用决定 |
|---|---|---|---|---|
| Traditional CV | data/processed/baseline_cv/cv_baseline_summary.json | opencv_contour_min_area_rect_rgb | 可复核 | 主表 |
| VLM + geometry | data/processed/vlm/grasp/vlm_assisted_grasp_summary.json | vlm_assisted_opencv_contour_min_area_rect_rgb | 可复核 | 主表 |
| CNN five-run aggregate | data/processed/vlm/cnn_grasp/multi_run_summary.json | vlm_cnn_multi_run | 聚合指标可复核；旧 best_val_loss 错误 | 主表只用 aggregate |
| CNN saved prediction rows | data/processed/vlm/cnn_grasp/cnn_grasp_predictions.csv | 最后一轮 seed 46 | 可复核 | 定性图和逐样本比较 |
| Legacy CNN single run 73.11% | 原始独立 JSON/CSV 已被覆盖 | 无 | 当前不可复核 | 不进入论文主表 |
```

记录以下校验命令输出：

```bash
sha256sum \
  data/processed/baseline_cv/cv_baseline_summary.json \
  data/processed/vlm/grasp/vlm_assisted_grasp_summary.json \
  data/processed/vlm/cnn_grasp/cnn_grasp_summary.json \
  data/processed/vlm/cnn_grasp/multi_run_summary.json
```

- [ ] **Step 3: 写多轮汇总失败测试**

创建 `tests/test_cnn_grasp_reporting.py`：

```python
import ast
from pathlib import Path

import pytest

from src.vlm.run_cnn_grasp import build_multi_run_summary


def _metrics(success_rate: float, iou: float, angle: float) -> dict:
    return {
        "success_rate": success_rate,
        "mean_iou": iou,
        "mean_angle": angle,
        "count": 85,
    }


def test_multi_run_summary_preserves_best_validation_loss() -> None:
    records = [
        {
            "seed": 42,
            "best_val_loss": 0.0123,
            "all": _metrics(0.70, 0.40, 18.0),
            "test": _metrics(0.80, 0.45, 19.0),
        },
        {
            "seed": 43,
            "best_val_loss": 0.0098,
            "all": _metrics(0.74, 0.44, 16.0),
            "test": _metrics(0.82, 0.47, 17.0),
        },
    ]

    summary = build_multi_run_summary(records)

    assert summary["seeds"] == [42, 43]
    assert summary["per_run"][0]["best_val_loss"] == pytest.approx(0.0123)
    assert summary["per_run"][1]["best_val_loss"] == pytest.approx(0.0098)
    assert summary["all"]["success_rate_mean"] == pytest.approx(0.72)
    assert summary["test"]["success_rate_mean"] == pytest.approx(0.81)


def test_cli_file_has_one_main_guard() -> None:
    path = Path("src/vlm/run_cnn_grasp.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(guards) == 1
```

- [ ] **Step 4: 运行测试并确认失败原因**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cnn_grasp_reporting.py -v
```

Expected: collection fails because `build_multi_run_summary` does not exist, or the main-guard test reports `2 != 1`.

- [ ] **Step 5: 提取纯汇总函数**

在 `src/vlm/run_cnn_grasp.py` 的 `_eval_on_splits` 后添加：

```python
def build_multi_run_summary(run_records: list[dict]) -> dict:
    def aggregate(split_name: str) -> dict:
        rates = [record[split_name]["success_rate"] for record in run_records]
        ious = [record[split_name]["mean_iou"] for record in run_records]
        angles = [record[split_name]["mean_angle"] for record in run_records]
        return {
            "success_rate_mean": float(np.mean(rates)),
            "success_rate_std": float(np.std(rates)),
            "mean_iou_mean": float(np.mean(ious)),
            "mean_iou_std": float(np.std(ious)),
            "mean_angle_mean": float(np.mean(angles)),
            "mean_angle_std": float(np.std(angles)),
        }

    return {
        "method": "vlm_cnn_multi_run",
        "num_runs": len(run_records),
        "seeds": [int(record["seed"]) for record in run_records],
        "all": aggregate("all"),
        "test": aggregate("test"),
        "per_run": [
            {
                "seed": int(record["seed"]),
                "best_val_loss": float(record["best_val_loss"]),
                "all_success_rate": float(record["all"]["success_rate"]),
                "test_success_rate": float(record["test"]["success_rate"]),
            }
            for record in run_records
        ],
    }
```

在 `multi` 模式中把 `run_results = []` 改为 `run_records = []`，每次记录改为：

```python
run_records.append({
    "seed": seed,
    "best_val_loss": best_val_loss,
    "all": result["all"],
    "test": result["test"],
    "rows": result["rows"],
})
```

然后使用：

```python
summary = build_multi_run_summary(run_records)
last_record = run_records[-1]
last_rows = last_record["rows"]
```

打印每个 split 的统计时，也统一从 `run_records` 读取：

```python
for subset_name in ["all", "test"]:
    rates = [
        record[subset_name]["success_rate"] * 100
        for record in run_records
    ]
    ious = [record[subset_name]["mean_iou"] for record in run_records]
    angles = [record[subset_name]["mean_angle"] for record in run_records]
```

保存最后一轮结果时使用：

```python
save_results(last_rows, {
    "method_name": "vlm_cnn_multi_run_last",
    "sample_count": last_record["all"]["count"],
    "success_count": int(
        last_record["all"]["success_rate"] * last_record["all"]["count"]
    ),
    "success_rate": last_record["all"]["success_rate"],
    "mean_best_iou": last_record["all"]["mean_iou"],
    "mean_best_angle_error_degrees": last_record["all"]["mean_angle"],
    "iou_threshold": IOU_THRESHOLD,
    "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
    "predictions_csv": str(PREDICTIONS_CSV),
})
```

删除文件末尾重复的 `if __name__ == "__main__": main()`，只保留一个入口。

- [ ] **Step 6: 运行单元测试和语法检查**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cnn_grasp_reporting.py -v
conda run -n msc-grasp python -m py_compile src/vlm/run_cnn_grasp.py
```

Expected: `2 passed`，并且 `py_compile` 无输出、退出码为 0。

- [ ] **Step 7: 验证 CLI 只启动一次且不覆盖实验产物**

Run:

```bash
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py --help
```

Expected:

- 帮助标题和参数列表只打印一次；
- 不加载数据集、不开始训练；
- `data/processed/vlm/cnn_grasp/` 中现有文件的 SHA-256 不发生变化。

- [ ] **Step 8: 提交可信度修复**

```bash
git add tests/test_cnn_grasp_reporting.py \
  src/vlm/run_cnn_grasp.py \
  docs/debugging/BUGLOG.md \
  docs/planning/experiment_result_provenance.md
git commit -m "fix: preserve CNN multi-run validation losses"
```

---

### Task 2（7月25–27日）：生成 Cornell 数据划分审计

**Files:**
- Create: `tests/test_cornell_split_audit.py`
- Create: `src/shared/analyze_cornell_splits.py`
- Create: `docs/planning/cornell_split_audit.md`
- Create: `uog_dissertation_outline/images/cornell_split_contact_sheet.png`

**Interfaces:**
- Consumes: `CornellGraspDataset`, geometry predictions CSV, CNN predictions CSV
- Produces: `assign_split(object_directory: str) -> str`
- Produces: `summarize_predictions(rows: list[dict]) -> dict[str, dict]`
- Produces: 885 样本全量统计和 85 样本共同测试集统计

- [ ] **Step 1: 写划分和统计失败测试**

创建 `tests/test_cornell_split_audit.py`：

```python
import pytest

from src.shared.analyze_cornell_splits import assign_split, summarize_predictions


def test_assign_split_uses_fixed_directory_groups() -> None:
    assert assign_split("01") == "train"
    assert assign_split("06") == "train"
    assert assign_split("07") == "val"
    assert assign_split("08") == "val"
    assert assign_split("09") == "test"
    assert assign_split("10") == "test"


def test_assign_split_rejects_unknown_directory() -> None:
    with pytest.raises(ValueError, match="unknown Cornell directory"):
        assign_split("11")


def test_summarize_predictions_reports_counts_and_metrics() -> None:
    rows = [
        {
            "object_directory": "09",
            "success": "1",
            "best_iou": "0.4",
            "best_angle_error_degrees": "10",
        },
        {
            "object_directory": "10",
            "success": "0",
            "best_iou": "0.2",
            "best_angle_error_degrees": "40",
        },
    ]

    result = summarize_predictions(rows)

    assert result["test"]["count"] == 2
    assert result["test"]["success_rate"] == pytest.approx(0.5)
    assert result["test"]["mean_best_iou"] == pytest.approx(0.3)
    assert result["test"]["mean_angle_error_degrees"] == pytest.approx(25.0)
```

- [ ] **Step 2: 运行测试并确认导入失败**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cornell_split_audit.py -v
```

Expected: FAIL with `ModuleNotFoundError: src.shared.analyze_cornell_splits`.

- [ ] **Step 3: 实现固定划分和统计接口**

创建 `src/shared/analyze_cornell_splits.py`，至少包含：

```python
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.shared.cornell_dataset import CornellGraspDataset

TRAIN_DIRS = {"01", "02", "03", "04", "05", "06"}
VAL_DIRS = {"07", "08"}
TEST_DIRS = {"09", "10"}
OUTPUT_DIR = Path("data/processed/shared/split_audit")


def assign_split(object_directory: str) -> str:
    if object_directory in TRAIN_DIRS:
        return "train"
    if object_directory in VAL_DIRS:
        return "val"
    if object_directory in TEST_DIRS:
        return "test"
    raise ValueError(f"unknown Cornell directory: {object_directory}")


def summarize_predictions(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[assign_split(row["object_directory"])].append(row)

    result = {}
    for split in ("train", "val", "test"):
        subset = grouped.get(split, [])
        result[split] = {
            "count": len(subset),
            "success_rate": (
                sum(int(row["success"]) for row in subset) / len(subset)
                if subset else 0.0
            ),
            "mean_best_iou": (
                float(np.mean([float(row["best_iou"]) for row in subset]))
                if subset else 0.0
            ),
            "mean_angle_error_degrees": (
                float(np.mean([
                    float(row["best_angle_error_degrees"]) for row in subset
                ]))
                if subset else 0.0
            ),
        }
    return result
```

CLI 还必须：

- 读取三种方法 CSV；
- 用 `(object_directory, sample_id)` 连接几何和 CNN 结果；
- 断言共同测试样本正好为 85；
- 每个目录等间距选择 6 张代表图；
- 生成 train/val/test 三栏 contact sheet；
- 保存 JSON/CSV 到 `OUTPUT_DIR`。

- [ ] **Step 4: 运行测试**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cornell_split_audit.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: 生成实际审计产物**

Run:

```bash
conda run -n msc-grasp python src/shared/analyze_cornell_splits.py
```

Expected:

- `split_metrics.json` 中 train、val、test 数量之和为 885；
- `same_test_subset_metrics.csv` 同时包含 `vlm_geometric` 和 `vlm_cnn_last_saved_run`；
- 两种方法的共同 test count 均为 85；
- contact sheet 包含目录 01–10 的样本。

- [ ] **Step 6: 完成人工物体构成复核**

在 `docs/planning/cornell_split_audit.md` 使用固定标签：

```text
形状：regular_boxlike / elongated / round / thin_flat / irregular_branched
难度：low / medium / high
主要因素：symmetry / multiple_valid_grasps / weak_boundary / complex_shape / narrow_part
```

逐目录记录：

```markdown
| 目录 | split | 代表物体描述 | 主要形状 | 难度 | 判断依据 |
|---|---|---|---|---|---|
```

结论只能从以下三种表述中选择一项：

1. `09–10 明显更简单，因此只报告固定 85 样本表现。`
2. `09–10 与其他分组基本可比，但样本量小，只能视为有限的未见物体证据。`
3. `样例和统计不足以确认可比性，因此不作一般化泛化结论。`

- [ ] **Step 7: 复制最终样例图并提交**

```bash
cp data/processed/shared/split_audit/cornell_split_contact_sheet.png \
  uog_dissertation_outline/images/cornell_split_contact_sheet.png
git add tests/test_cornell_split_audit.py \
  src/shared/analyze_cornell_splits.py \
  docs/planning/cornell_split_audit.md \
  uog_dissertation_outline/images/cornell_split_contact_sheet.png
git commit -m "analysis: audit Cornell dataset splits"
```

---

### Task 3（7月28–29日）：生成公平后端比较与逐样本失败图

**Files:**
- Create: `src/vlm/analyze_backend_comparison.py`
- Create: `tests/test_backend_comparison.py`
- Create: `uog_dissertation_outline/images/backend_failure_cases.png`

**Interfaces:**
- Consumes: 几何和 CNN predictions CSV
- Produces: `classify_pair(geometric_success: int, cnn_success: int) -> str`
- Produces: `sample_comparison.csv`，每行是同一 Cornell 样本的两种后端结果

- [ ] **Step 1: 写交叉分类失败测试**

创建 `tests/test_backend_comparison.py`：

```python
from src.vlm.analyze_backend_comparison import classify_pair


def test_classify_pair_covers_all_outcomes() -> None:
    assert classify_pair(1, 1) == "both_success"
    assert classify_pair(0, 1) == "cnn_only"
    assert classify_pair(1, 0) == "geometric_only"
    assert classify_pair(0, 0) == "both_failure"
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_backend_comparison.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现分类和严格共同样本连接**

在 `src/vlm/analyze_backend_comparison.py` 中实现：

```python
def classify_pair(geometric_success: int, cnn_success: int) -> str:
    outcomes = {
        (1, 1): "both_success",
        (0, 1): "cnn_only",
        (1, 0): "geometric_only",
        (0, 0): "both_failure",
    }
    try:
        return outcomes[(int(geometric_success), int(cnn_success))]
    except KeyError as exc:
        raise ValueError("success values must be 0 or 1") from exc
```

CLI 必须：

- 以 `(object_directory, sample_id)` 内连接两种预测；
- 只用目录 09–10 生成公平测试摘要；
- 报告四类交叉结果数量；
- 分别保存 `cnn_only`、`geometric_only`、`both_failure` 代表图；
- 图中绿色为 GT、蓝色为 CNN、红色为几何后端、黄色为 VLM box；
- 生成 3×4 的论文汇总图。

- [ ] **Step 4: 运行测试和实际分析**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_backend_comparison.py -v
conda run -n msc-grasp python src/vlm/analyze_backend_comparison.py
```

Expected:

- pytest 显示 `1 passed`；
- `comparison_summary.json` 的 `test_count` 为 85；
- 四类数量之和为 85；
- 输出 `backend_failure_cases.png`。

- [ ] **Step 5: 复制最终图并提交**

```bash
cp data/processed/vlm/backend_comparison/backend_failure_cases.png \
  uog_dissertation_outline/images/backend_failure_cases.png
git add src/vlm/analyze_backend_comparison.py \
  tests/test_backend_comparison.py \
  uog_dissertation_outline/images/backend_failure_cases.png
git commit -m "analysis: compare grasp backends per sample"
```

---

### Task 4（7月28–30日）：完成 CNN 结构依据和现代文献矩阵

**Files:**
- Create: `docs/planning/cnn_architecture_rationale.md`
- Create: `docs/planning/modern_2d_grasp_literature_matrix.md`
- Modify: `uog_dissertation_outline/l4proj.bib`
- Create: `uog_dissertation_outline/images/cnn_architecture.pdf`

**Interfaces:**
- Consumes: 当前 CNN 源码、论文原文
- Produces: 可直接转写到 Methodology 和 Literature Review 的证据表

- [ ] **Step 1: 锁定必须核对的论文原文**

文献矩阵至少覆盖：

```text
Jiang et al. (2011) — rectangle representation
Lenz et al. (2015) — two-stage deep grasp detection
Redmon and Angelova (2015) — real-time regression
Kumra and Kanan (2017) — RGB-D deep CNN
Morrison et al. (2018) — GG-CNN dense generative grasp maps
Park et al. (2018) — high-resolution FCNN
Kumra et al. (2020) — GR-ConvNet
Li et al. (2022) — Gaussian-guided generative CNN
Vuong et al. (2024) — language-driven grasp detection
```

只从论文 PDF 或正式出版页提取字段，不从博客或二手综述抄数字。

- [ ] **Step 2: 建立统一比较矩阵**

在 `docs/planning/modern_2d_grasp_literature_matrix.md` 使用：

```markdown
| 文献 | 任务 | 输入模态 | 数据集 | 划分方式 | 输出形式 | 评价指标 | 报告结果 | 与本项目可比性 |
|---|---|---|---|---|---|---|---|---|
```

“与本项目可比性”只能使用：

- `直接可比：相同数据、划分、输入和 Cornell rectangle metric`
- `有限可比：指标相同，但输入或划分不同`
- `不可直接比较：数据集、任务或物理执行指标不同`

每一个报告数字后写论文页码或表号。

- [ ] **Step 3: 写 CNN 设计来源对照表**

在 `docs/planning/cnn_architecture_rationale.md` 固定分为：

```markdown
| 设计项 | 当前选择 | 文献依据 | 工程依据 | 是否为本项目原创 |
|---|---|---|---|---|
```

必须逐项覆盖：

- 224×224 RGB VLM crop；
- 四个卷积块；
- 通道 32/64/128/256；
- Global Average Pooling；
- 单抓取矩形回归；
- Smooth L1；
- `sin(2θ), cos(2θ)` 双角编码；
- Adam、weight decay、early stopping；
- 01–06/07–08/09–10 目录划分。

其中层数、通道数和全连接宽度明确标为自主工程选择；抓取矩形、Cornell 指标和双角方向表示注明相应研究依据。

- [ ] **Step 4: 绘制结构图**

结构图必须显示：

```text
224×224×3
→ Conv 5×5 s2, 32
→ MaxPool
→ Conv 3×3, 64
→ Conv 3×3, 128
→ Conv 3×3, 256
→ GAP 256
→ FC 128
→ FC 64
→ [cx, cy, w, h, sin(2θ), cos(2θ)]
```

导出为 `uog_dissertation_outline/images/cnn_architecture.pdf`，在图注中写明 `Lightweight regression baseline designed for this study; not proposed as a novel architecture.`

- [ ] **Step 5: 更新 BibTeX 并检查重复键**

Run:

```bash
python -c "import re, pathlib, collections; text=pathlib.Path('uog_dissertation_outline/l4proj.bib').read_text(); keys=re.findall(r'@[^{]+\\{([^,]+),', text); duplicates=[k for k,v in collections.Counter(keys).items() if v>1]; assert not duplicates, duplicates; print(len(keys), 'unique BibTeX entries')"
```

Expected: 输出唯一条目数量，且无 AssertionError。

- [ ] **Step 6: 提交文献和结构依据**

```bash
git add docs/planning/cnn_architecture_rationale.md \
  docs/planning/modern_2d_grasp_literature_matrix.md \
  uog_dissertation_outline/l4proj.bib \
  uog_dissertation_outline/images/cnn_architecture.pdf
git commit -m "docs: document CNN rationale and grasp literature"
```

---

### Task 5（7月31日–8月2日）：完成 Methodology 和 Results 初稿

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:51-111`

**Interfaces:**
- Consumes: Tasks 1–4 的已验证 JSON、图和文献矩阵
- Produces: 不含核心占位符的 Methodology 和 Results

- [ ] **Step 1: 更新论文方法结构**

Methodology 必须增加并完成以下内容：

```text
Research design
Cornell dataset and fixed directory split
2D grasp rectangle representation
Traditional CV baseline
Grounding DINO localisation front end
VLM-guided geometric backend
VLM-guided lightweight CNN backend
Training protocol and repeated runs
Cornell rectangle evaluation
Methodological limitations
```

- [ ] **Step 2: 插入 CNN 结构和设计依据**

在 CNN 小节插入：

```latex
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{images/cnn_architecture.pdf}
  \caption{Lightweight CNN grasp regression baseline used in this study.}
  \label{fig:cnn-architecture}
\end{figure}
```

正文必须说明：

- 完整结构不是从单篇论文复制；
- VLM 已完成定位，所以 CNN 只回归一个局部抓取矩形；
- 小数据集是使用轻量网络和 GAP 的工程理由；
- 双角编码处理平行夹爪 180° 对称性。

- [ ] **Step 3: 写最终主结果表**

主表固定报告：

```text
Traditional CV baseline：885 样本
VLM + geometry：885 样本
VLM + CNN five-run mean ± std：885 样本
```

旧的 73.11% 单次 CNN 结果因原始独立 JSON/CSV 已被覆盖，不进入论文主表。`cnn_grasp_predictions.csv` 只作为 seed 46 最后一轮的逐样本和定性分析来源，不冒充独立单次实验。

只从以下文件复制数字：

```text
data/processed/baseline_cv/cv_baseline_summary.json
data/processed/vlm/grasp/vlm_assisted_grasp_summary.json
data/processed/vlm/cnn_grasp/multi_run_summary.json
```

- [ ] **Step 4: 写固定 85 样本测试比较**

使用 Task 2 的 `same_test_subset_metrics.csv` 和 Task 3 的交叉统计。标题必须包含 `fixed 85-sample test subset (directories 09–10)`，不得写成通用 unseen-object benchmark。

- [ ] **Step 5: 插入数据构成和失败案例图**

```latex
\includegraphics[width=\linewidth]{images/cornell_split_contact_sheet.png}
\includegraphics[width=\linewidth]{images/backend_failure_cases.png}
```

每张图的正文说明必须回答：

- 图展示什么；
- 为什么选择这些样本；
- 图支持或限制了哪一项结论。

- [ ] **Step 6: 检查 Methodology/Results 核心占位符**

Run:

```bash
python -c "from pathlib import Path; text=Path('uog_dissertation_outline/l4proj.tex').read_text(); start=text.index('\\\\chapter{Methodology}'); end=text.index('\\\\chapter{General Discussion}'); block=text[start:end]; assert '\\\\todo{' not in block; print('Methodology and Results contain no todo blocks')"
```

Expected: `Methodology and Results contain no todo blocks`.

- [ ] **Step 7: 编译并提交**

Run:

```bash
cd uog_dissertation_outline
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
```

Expected: 生成 `l4proj.pdf`，无 LaTeX fatal error。

```bash
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: draft methodology and results chapters"
```

---

### Task 6（8月3–4日）：完成 Discussion、失败分析与复现附录

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:113-158`
- Modify: `docs/debugging/FAILURE_ANALYSIS.md`

**Interfaces:**
- Consumes: 主结果、共同测试集、交叉失败统计、数据划分审计
- Produces: 事实与解释分开的 Discussion 主体

- [ ] **Step 1: 重写主要发现**

Discussion 必须按以下证据顺序：

1. VLM 定位前端提供最大成功率增益；
2. CNN 的平均 IoU 更高；
3. 几何后端的平均角度误差更低；
4. 两种后端具有互补性；
5. 85 样本结果受数据构成和样本规模限制。

- [ ] **Step 2: 更新失败分析**

把 Task 3 的四类交叉结果写入 `docs/debugging/FAILURE_ANALYSIS.md`，并在论文中选择代表案例讨论。不要沿用未经逐样本检查的统一原因推断；对每幅代表图写清“观察到的现象”和“可能原因”之间的区别。

- [ ] **Step 3: 写与文献关系**

明确区分：

- Cornell rectangle metric 数字；
- 不同 image-wise/object-wise split；
- RGB 与 RGB-D 输入；
- 单抓取回归与 dense grasp map；
- 离线矩形成功与真实机器人抓取成功。

- [ ] **Step 4: 完成局限**

至少覆盖：

```text
Cornell-only
85-sample test subset
目录划分可比性
单抓取输出
Grounding DINO coverage 不等于定位准确
无物理机器人执行
小数据集与随机波动
```

- [ ] **Step 5: 完成复现附录**

附录必须列出：

- Conda 环境 `msc-grasp`；
- PyTorch 2.5.1+cu121；
- Grounding DINO 模型 `IDEA-Research/grounding-dino-tiny`；
- prompt `small object`；
- crop size 224；
- batch size 32；
- learning rate `1e-3`；
- weight decay `1e-4`；
- seeds 42–46；
- 所有核心运行命令；
- 输出文件路径。

- [ ] **Step 6: 编译和提交**

```bash
cd uog_dissertation_outline
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
cd ..
git add uog_dissertation_outline/l4proj.tex docs/debugging/FAILURE_ANALYSIS.md
git commit -m "docs: add evidence-based discussion and reproducibility details"
```

---

### Task 7（8月5–6日）：两天内完成多头 CNN 可行性门槛

**Files:**
- Create: `src/vlm/cnn_grasp_models.py`
- Create: `tests/test_cnn_grasp_models.py`
- Modify: `src/vlm/run_cnn_grasp.py`
- Modify: `src/vlm/README.md`

**Interfaces:**
- Produces: `SingleHeadCNNGraspRegressor(nn.Module)`
- Produces: `MultiHeadCNNGraspRegressor(nn.Module)`
- Produces: `compute_multi_head_loss(predictions: dict, targets: Tensor, orientation_norm_weight: float = 0.1) -> dict`
- CLI: `--architecture single|multi_head`
- CLI: `--output-dir PATH`

- [ ] **Step 1: 写模型输出和损失失败测试**

创建 `tests/test_cnn_grasp_models.py`：

```python
import torch

from src.vlm.cnn_grasp_models import (
    MultiHeadCNNGraspRegressor,
    SingleHeadCNNGraspRegressor,
    compute_multi_head_loss,
)


def test_single_head_output_shape() -> None:
    model = SingleHeadCNNGraspRegressor()
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 6)


def test_multi_head_outputs_have_expected_shapes() -> None:
    model = MultiHeadCNNGraspRegressor()
    output = model(torch.zeros(2, 3, 224, 224))
    assert output["centre"].shape == (2, 2)
    assert output["size"].shape == (2, 2)
    assert output["orientation"].shape == (2, 2)


def test_multi_head_loss_exposes_each_component() -> None:
    predictions = {
        "centre": torch.zeros(2, 2, requires_grad=True),
        "size": torch.zeros(2, 2, requires_grad=True),
        "orientation": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], requires_grad=True
        ),
    }
    targets = torch.tensor([
        [0.5, 0.5, 0.2, 0.1, 0.0, 1.0],
        [0.4, 0.6, 0.3, 0.2, 1.0, 0.0],
    ])

    losses = compute_multi_head_loss(predictions, targets)

    assert set(losses) == {"total", "centre", "size", "orientation", "unit_norm"}
    losses["total"].backward()
    assert predictions["centre"].grad is not None
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cnn_grasp_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: 实现共享 backbone 和三个回归头**

`src/vlm/cnn_grasp_models.py` 使用一个返回 256 维特征的共享 backbone。多头接口固定为：

```python
class MultiHeadCNNGraspRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = GraspFeatureBackbone()
        self.centre_head = RegressionHead(256, 2)
        self.size_head = RegressionHead(256, 2)
        self.orientation_head = RegressionHead(256, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs)
        return {
            "centre": self.centre_head(features),
            "size": self.size_head(features),
            "orientation": self.orientation_head(features),
        }
```

损失固定为：

```python
def compute_multi_head_loss(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    orientation_norm_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    smooth_l1 = nn.SmoothL1Loss()
    centre = smooth_l1(predictions["centre"], targets[:, 0:2])
    size = smooth_l1(predictions["size"], targets[:, 2:4])
    orientation = smooth_l1(predictions["orientation"], targets[:, 4:6])
    norms = torch.linalg.vector_norm(predictions["orientation"], dim=1)
    unit_norm = torch.mean((norms - 1.0) ** 2)
    total = centre + size + orientation + orientation_norm_weight * unit_norm
    return {
        "total": total,
        "centre": centre,
        "size": size,
        "orientation": orientation,
        "unit_norm": unit_norm,
    }
```

- [ ] **Step 4: 运行模型单元测试**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/test_cnn_grasp_models.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: 扩展训练 CLI**

`run_cnn_grasp.py` 增加：

```python
parser.add_argument(
    "--architecture",
    choices=["single", "multi_head"],
    default="single",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("data/processed/vlm/cnn_grasp"),
)
```

要求：

- `single` 保持旧模型和输出兼容；
- `multi_head` 使用分量损失并在 history 中保存五个损失字段；
- 推理时按 `[centre, size, orientation]` 拼成现有六维格式；
- 新输出目录使用 `data/processed/vlm/cnn_grasp_multi_head/`；
- `multi` 模式按 `training_history_seed_{seed}.json` 保存每轮历史；
- `multi` 模式按 `cnn_grasp_model_seed_{seed}.pt` 保存每轮最佳权重；
- 不覆盖已有单头模型或结果。

- [ ] **Step 6: 执行两日可行性烟雾实验**

Run:

```bash
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py \
  --mode all \
  --architecture multi_head \
  --output-dir data/processed/vlm/cnn_grasp_multi_head \
  --seed 42 \
  --device cuda
```

继续正式实验的四个条件：

1. 训练正常结束且无 NaN；
2. 生成模型、history、summary 和 predictions；
3. 评估样本数为 885，test 样本数为 85；
4. 随机检查 10 个预测框，没有负宽高、无限值或越界中心。

任一条件在 2026-08-06 日结束前仍不满足，则停止 Task 7，直接进入 Task 9 的失败分析回退，不再修复多头网络。

- [ ] **Step 7: 提交可行模型**

只有四个继续条件全部满足时执行：

```bash
git add src/vlm/cnn_grasp_models.py \
  tests/test_cnn_grasp_models.py \
  src/vlm/run_cnn_grasp.py \
  src/vlm/README.md
git commit -m "feat: add multi-head grasp regressor"
```

---

### Task 8（8月7–9日）：运行多头 CNN 五次正式对照

**Files:**
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Task 7 已通过门槛的多头模型
- Produces: seeds 42–46 的完整结果和汇总

- [ ] **Step 1: 运行全部测试**

```bash
conda run -n msc-grasp python -m pytest tests -v
```

Expected: 所有测试通过。

- [ ] **Step 2: 运行五次固定种子实验**

```bash
conda run -n msc-grasp python src/vlm/run_cnn_grasp.py \
  --mode multi \
  --architecture multi_head \
  --output-dir data/processed/vlm/cnn_grasp_multi_head \
  --num-runs 5 \
  --seed 42 \
  --device cuda
```

Expected:

- seeds 为 `[42, 43, 44, 45, 46]`；
- `num_runs` 为 5；
- 每轮都有真实 `best_val_loss`；
- all count 为 885；
- test count 为 85；
- 汇总包含成功率、IoU 和角度的 mean/std。

- [ ] **Step 3: 验证 JSON 完整性**

Run:

```bash
python -c "import json, math; p='data/processed/vlm/cnn_grasp_multi_head/multi_run_summary.json'; d=json.load(open(p)); assert d['num_runs']==5; assert d['seeds']==[42,43,44,45,46]; assert len(d['per_run'])==5; assert all(math.isfinite(r['best_val_loss']) and r['best_val_loss'] < 0.2 for r in d['per_run']); print('multi-head summary verified')"
```

Expected: `multi-head summary verified`.

- [ ] **Step 4: 生成单头/多头/几何对比表**

表格必须同时报告：

```text
all 885: success rate, mean best IoU, mean angle error
test 85: success rate, mean best IoU, mean angle error
five-run: mean ± population standard deviation
```

如果多头没有提升，保留结果并分析分量损失、数据规模和任务耦合，不增加新超参数搜索。

- [ ] **Step 5: 更新状态与工作日志**

只有实际 JSON 验证后才把多头结果写入 `CURRENT_STATUS.md`。`WORKLOG.md` 记录命令、种子、输出路径和“改善/无改善”的事实，不写未经证据支持的原因。

- [ ] **Step 6: 提交验证后的状态**

```bash
git add docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md
git commit -m "docs: record multi-head grasp experiment"
```

---

### Task 9（8月7–9日，Task 7 失败时替代 Task 8）：执行系统性失败分析回退

**Files:**
- Modify: `src/vlm/analyze_backend_comparison.py`
- Modify: `docs/debugging/FAILURE_ANALYSIS.md`
- Modify: `uog_dissertation_outline/images/backend_failure_cases.png`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: 当前单头 CNN 与几何后端预测
- Produces: 按 IoU、角度和复合失败划分的逐样本对照

- [ ] **Step 1: 扩展失败类别**

对每种后端分别使用：

```python
def classify_metric_failure(success: int, iou: float, angle: float) -> str:
    if int(success) == 1:
        return "success"
    iou_ok = float(iou) >= 0.25
    angle_ok = float(angle) <= 30.0
    if angle_ok and not iou_ok:
        return "iou_only"
    if iou_ok and not angle_ok:
        return "angle_only"
    return "both_iou_and_angle"
```

- [ ] **Step 2: 增加并运行单元测试**

测试四种返回值后运行：

```bash
conda run -n msc-grasp python -m pytest tests/test_backend_comparison.py -v
```

Expected: 所有 backend comparison 测试通过。

- [ ] **Step 3: 生成分组统计和代表图**

```bash
conda run -n msc-grasp python src/vlm/analyze_backend_comparison.py
```

每个失败类别至少人工检查 8 个样本，并记录：

- 预测框与 GT 的几何差异；
- VLM crop 是否过紧或过宽；
- 物体是否细长、对称或不规则；
- 多个有效 GT 是否造成表面矛盾；
- 观察事实与推测原因。

- [ ] **Step 4: 更新失败分析和论文图**

把统计和经人工检查的案例写入 `FAILURE_ANALYSIS.md`，复制最终图，并在工作日志中明确多头扩展停止原因和停止日期。

- [ ] **Step 5: 提交回退分析**

```bash
git add src/vlm/analyze_backend_comparison.py \
  tests/test_backend_comparison.py \
  docs/debugging/FAILURE_ANALYSIS.md \
  uog_dissertation_outline/images/backend_failure_cases.png \
  docs/worklog/WORKLOG.md
git commit -m "analysis: deepen CNN grasp failure analysis"
```

---

### Task 10（8月10–13日）：完成 Literature Review、Introduction、Conclusion 和扩展结果

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex:12-158`

**Interfaces:**
- Consumes: 全部冻结结果、文献矩阵、数据审计、Task 8 或 Task 9
- Produces: 2026-08-13 实验冻结时的完整论文初稿

- [ ] **Step 1: 完成 Literature Review**

章节逻辑固定为：

```text
2D grasp rectangle formulation
classical and global regression methods
dense generative grasp maps
open-vocabulary grounding and language-conditioned grasping
dataset and evaluation comparability
identified research gap
```

Literature gap 表述为评估开放词汇定位能否改善轻量二维抓取后端的受控研究，不声称提出 SOTA 抓取网络。

- [ ] **Step 2: 完成 Introduction**

明确给出三个研究问题：

1. VLM 定位相对整图传统视觉是否改善二维抓取检测？
2. 相同 VLM 前端下，几何和 CNN 后端在位置、尺寸和角度上有何差异？
3. 固定目录划分下，CNN 测试表现如何，数据构成对解释有什么限制？

- [ ] **Step 3: 写扩展或回退结果**

如果完成 Task 8：

- 报告多头五次结果；
- 与单头保持完全相同的统计口径；
- 明确这是结构消融，不是新架构声明。

如果完成 Task 9：

- 报告多头可行性停止原因；
- 把系统性失败分析作为高价值补充；
- 不报告未完成网络的性能数字。

- [ ] **Step 4: 完成 Conclusion**

逐项回答三个研究问题；贡献限定为：

- 模块化三流程受控比较；
- 开放词汇定位前端的量化分析；
- 后端互补性和失败模式分析；
- 数据划分解释边界；
- 多头消融或系统性回退分析。

- [ ] **Step 5: 执行实验冻结检查**

Run:

```bash
git status --short
find data/processed -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

Expected: 保存带修改时间的最终实验文件清单；从次日起不再创建新实验分支。

- [ ] **Step 6: 检查核心章节占位符**

Run:

```bash
python -c "from pathlib import Path; t=Path('uog_dissertation_outline/l4proj.tex').read_text(); core=t[t.index('\\\\chapter{Introduction}'):t.index('\\\\begin{appendices}')]; assert '\\\\todo{' not in core; print('all core chapters complete')"
```

Expected: `all core chapters complete`.

- [ ] **Step 7: 编译并提交实验冻结版**

```bash
cd uog_dissertation_outline
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
cd ..
git add uog_dissertation_outline/l4proj.tex
git commit -m "docs: complete dissertation core draft"
```

---

### Task 11（8月14–16日）：形成完整论文第一版

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex`
- Modify: `uog_dissertation_outline/l4proj.bib`

**Interfaces:**
- Consumes: 冻结论文核心
- Produces: 包含摘要、附录和完整引用的第一版 PDF

- [ ] **Step 1: 写 300–400 词摘要**

摘要按四段信息顺序：

```text
problem and motivation
three-pipeline methodology
verified quantitative findings
scope, implications and limitation
```

摘要数字必须从最终结果表复制。

- [ ] **Step 2: 完成附录**

附录包含：

- 完整配置；
- 运行命令；
- 补充结果表；
- 额外案例；
- 数据划分规则；
- 多头损失定义或回退停止记录。

- [ ] **Step 3: 检查全文 TODO**

Run:

```bash
rg -n "\\\\todo\\{" uog_dissertation_outline/l4proj.tex
```

Expected: 无输出；在运行检查前已经完成 acknowledgements，提交版本中不保留任何 `\todo{}`。

- [ ] **Step 4: 完整 BibTeX 编译**

```bash
cd uog_dissertation_outline
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
```

Expected: 生成 `l4proj.pdf`，日志中没有 undefined citation 或 undefined reference。

- [ ] **Step 5: 检查日志**

```bash
cd uog_dissertation_outline
rg -n "Undefined|Citation.*undefined|Reference.*undefined|LaTeX Error|Emergency stop" l4proj.log
```

Expected: 无匹配。

- [ ] **Step 6: 提交第一版**

```bash
git add uog_dissertation_outline/l4proj.tex uog_dissertation_outline/l4proj.bib
git commit -m "docs: complete dissertation first draft"
```

---

### Task 12（8月17–20日）：一致性、引用和论证审查

**Files:**
- Modify: `uog_dissertation_outline/l4proj.tex`
- Modify: `uog_dissertation_outline/l4proj.bib`
- Modify: `docs/agent/CURRENT_STATUS.md`

**Interfaces:**
- Produces: 数字、术语、问题和结论一致的候选提交稿

- [ ] **Step 1: 建立最终数字清单**

逐项核对：

```text
504/885, 56.95%, 0.3360, 29.62°
649/885, 73.33%, 0.4182, 14.81°
CNN five-run 74.51% ± 1.38%, 0.4510 ± 0.0081, 16.49° ± 0.72°
fixed test geometry 64/85, 75.3%
fixed test CNN 82.35% ± 4.53%
multi-head final values or fallback category counts
```

使用：

```bash
rg -n "56\\.95|73\\.33|74\\.51|82\\.35|0\\.3360|0\\.4182|0\\.4510|14\\.81|16\\.49" \
  uog_dissertation_outline/l4proj.tex
```

每个数字在摘要、结果、讨论和结论中的口径必须一致。

- [ ] **Step 2: 做术语统一**

全文固定使用：

```text
VLM-guided geometric backend
VLM-guided CNN backend
grasp rectangle
success rate
mean best IoU
mean angle error
fixed 85-sample test subset
```

不要混用 detection rate、accuracy 和 physical grasp success。

- [ ] **Step 3: 做研究问题映射**

创建内部检查表：

```text
RQ1 → Methodology comparison → main 885 table → Discussion finding 1 → Conclusion answer 1
RQ2 → same VLM front end → backend metrics → complementarity discussion → answer 2
RQ3 → split audit + fixed test table → limitation discussion → cautious answer 3
```

任何没有结果证据的研究问题都删除或改写。

- [ ] **Step 4: 检查文献比较公平性**

每个外部结果附近必须说明输入模态、划分或指标差异。搜索并人工检查所有百分号：

```bash
rg -n "\\\\%|percent|accuracy|success rate" uog_dissertation_outline/l4proj.tex
```

- [ ] **Step 5: 更新当前状态**

仅记录已经验证的最终实验、论文已形成候选稿以及剩余排版任务。不把计划中的 PyBullet 写成进行中。

- [ ] **Step 6: 编译并提交候选稿**

```bash
cd uog_dissertation_outline
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
cd ..
git add uog_dissertation_outline/l4proj.tex \
  uog_dissertation_outline/l4proj.bib \
  docs/agent/CURRENT_STATUS.md
git commit -m "docs: align dissertation evidence and conclusions"
```

---

### Task 13（8月21–23日）：最终验证、备份和提交包

**Files:**
- Modify: `docs/worklog/WORKLOG.md`
- Verify: `uog_dissertation_outline/l4proj.tex`
- Verify: `uog_dissertation_outline/l4proj.bib`
- Generate locally: `uog_dissertation_outline/l4proj.pdf`

**Interfaces:**
- Produces: 最终 PDF、可复现源文件和两份独立备份

- [ ] **Step 1: 运行全部自动测试**

```bash
conda run -n msc-grasp python -m pytest tests -v
conda run -n msc-grasp python -m py_compile \
  src/shared/analyze_cornell_splits.py \
  src/vlm/analyze_backend_comparison.py \
  src/vlm/cnn_grasp_models.py \
  src/vlm/run_cnn_grasp.py
```

Expected: 所有测试通过，语法检查无输出。

- [ ] **Step 2: 验证核心实验输出仍存在**

```bash
python -c "from pathlib import Path; required=['data/processed/baseline_cv/cv_baseline_summary.json','data/processed/vlm/grasp/vlm_assisted_grasp_summary.json','data/processed/vlm/cnn_grasp/multi_run_summary.json','data/processed/shared/split_audit/split_metrics.json','data/processed/vlm/backend_comparison/comparison_summary.json']; missing=[p for p in required if not Path(p).exists()]; assert not missing, missing; print('all required evidence files exist')"
```

Expected: `all required evidence files exist`.

- [ ] **Step 3: 从干净 LaTeX 状态生成最终 PDF**

```bash
cd uog_dissertation_outline
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error l4proj.tex
```

Expected: 编译成功，无 undefined citation/reference。

- [ ] **Step 4: 逐页人工检查 PDF**

逐页检查：

- 封面姓名、学位、年份和导师；
- 目录页码；
- 图表是否越界或模糊；
- 所有图表均在正文引用；
- 表格小数位一致；
- 公式角度单位明确；
- 无空白章节、孤立标题或 TODO；
- 参考文献完整；
- 附录命令没有暴露本地敏感路径。

- [ ] **Step 5: 生成两个独立备份**

备份一：完整 `uog_dissertation_outline/` 源文件目录。

备份二：最终 `l4proj.pdf` 与最终 Git commit hash。

不要使用覆盖唯一副本的同步方式；确认两份备份均可独立打开。

- [ ] **Step 6: 记录最终工作日志**

在 `WORKLOG.md` 记录：

- 最终 PDF 生成日期；
- 最终 commit hash；
- 主实验和扩展实验状态；
- 最终论文采用的谨慎结论；
- 提交包备份位置的非敏感描述。

- [ ] **Step 7: 提交最终记录**

```bash
git add docs/worklog/WORKLOG.md
git commit -m "docs: record dissertation completion"
```

- [ ] **Step 8: 提交前停止改动**

最后 24 小时不再运行新实验，不再重构代码，只修复明确的事实、引用、语言或排版错误。

---

## 每日节奏

每天只设置一个主要交付物：

```text
上午：完成当天唯一的分析、实验或章节任务
下午：把实际结果写入论文或最终图表
晚上：运行验证命令、记录输出、确定次日第一项任务
```

每周保留一天低负荷缓冲，用于补齐失败的编译、图表、引用或导师反馈。缓冲日不用于启动新实验方向。

## 阶段门槛摘要

| 日期 | 必须达到的状态 | 未达到时的处理 |
|---|---|---|
| 7月30日 | 数据划分、文献矩阵、CNN依据完成 | 暂停新网络，优先补齐可信度 |
| 8月6日 | 多头 CNN 通过四项可行性检查 | 停止多头，执行 Task 9 |
| 8月13日 | 所有实验冻结，核心章节完整 | 删除可选实验，只保论文 |
| 8月16日 | 完整第一版 PDF | 停止所有代码修改 |
| 8月20日 | 候选提交稿完成 | 只做一致性和排版修复 |
| 8月23日 | 最终 PDF、源文件、两份备份完成 | 不再修改实验内容 |

## 最终完成标准

- [ ] 导师关于 CNN 依据、现代文献、数据划分和机器人范围的反馈均在论文中得到直接回应。
- [ ] 三条现有流程的所有数字都能追溯到 CSV/JSON。
- [ ] 85 样本比较使用完全相同样本，并明确其解释边界。
- [ ] 多头 CNN 完成五次对照，或按两日门槛停止并完成系统性失败分析。
- [ ] 论文核心章节没有 TODO。
- [ ] LaTeX 无 fatal error、undefined citation 和 undefined reference。
- [ ] 最终 PDF 已逐页检查并保留两份独立备份。
