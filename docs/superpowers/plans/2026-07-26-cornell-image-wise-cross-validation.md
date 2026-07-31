# Cornell Image-wise Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 885 张 Cornell 图像实现可审计、无同图泄漏、可恢复运行的 image-wise 五折，并让单头和多头 CNN 使用同一份 fold manifest 完成公平对照。

**Architecture:** 在 `src/shared/` 中增加不依赖 PyTorch 的纯 fold 生成与校验模块；对现有 CNN 数据准备和评估函数进行最小兼容重构，使它们能够接收显式样本集合；在 `src/vlm/` 中增加五折训练入口，逐 fold 保存模型、历史、预测与汇总，并在五折齐全后生成 885 行合并预测和交叉验证汇总。实现依据 Lenz 等人对 image-wise 协议的定义自行编写，不复制第三方交叉验证代码。

**Tech Stack:** Python 3.10、标准库 `csv/json/hashlib/random`、NumPy、PyTorch 2.11、OpenCV、pytest、Cornell Grasping Dataset、缓存的 Grounding DINO 定位结果。

## Global Constraints

- 使用现有隔离工作树 `/home/pzk/Msc_project/.worktrees/dissertation-sprint`。
- 外层 fold 数固定为 `5`，主随机种子固定为 `42`。
- 885 个样本平均分为五个 177 样本测试 fold。
- 每折剩余 708 个样本分为 566 个训练样本和 142 个验证样本。
- 每个 fold 的模型训练种子固定为 `42`；fold 和随机种子不得混写。
- 单头和多头必须读取字节相同的 manifest 文件。
- 测试 fold 不得参与学习率调度、早停或 checkpoint 选择。
- 正式训练继续启用严格确定性 CUDA 设置。
- 不把 Cornell `01`–`10` 目录当作物体 ID，不生成推断的 object-wise 结果。
- 不重新运行 Grounding DINO；复用已保存并审计的 885 个定位框。
- 生成实验产物写入被 Git 忽略的 `data/processed/vlm/cnn_cross_validation/`。
- 保持现有固定目录训练命令和历史输出格式兼容。
- 如复制或实质性改编外部代码，必须记录作者、项目/论文、许可证与链接；本计划实现不需要复制外部代码。
- 每个生产代码行为必须先有失败测试，并确认失败原因正确。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `tests/test_cornell_cross_validation.py` | 锁定 fold 确定性、大小、覆盖、互斥、持久化和错误检测 |
| `src/shared/cornell_cross_validation.py` | 生成、保存、加载、校验 image-wise fold manifest |
| `tests/test_cnn_grasp_dataset_selection.py` | 锁定全量 crop 数据准备、显式角色划分和评估样本过滤 |
| `src/vlm/run_cnn_grasp.py` | 兼容地增加全量样本构建、角色划分，并让评估真正服从样本参数 |
| `tests/test_cnn_cross_validation.py` | 锁定 fold 产物路径、五折汇总和不完整产物拒绝 |
| `src/vlm/run_cnn_cross_validation.py` | 生成 manifest、运行指定 fold、恢复执行并聚合五折结果 |
| `docs/planning/cornell_split_audit.md` | 记录 image-wise 可实现、object-wise 元数据阻塞及结论边界 |
| `docs/planning/experiment_result_provenance.md` | 记录 manifest、正式产物目录、哈希和使用决定 |
| `src/vlm/README.md` | 记录五折命令和统计口径 |
| `docs/agent/PROJECT_STRUCTURE.md` | 登记两个新增主要模块与命令 |
| `docs/agent/CURRENT_STATUS.md` | 只在 manifest 或正式结果通过审计后更新 |
| `docs/worklog/WORKLOG.md` | 记录实现、运行与审计结果 |
| `uog_dissertation_outline/l4proj.tex` | 正式结果完成后更新 Methodology、Results、Discussion |

---

### Task 1: 实现确定性的 image-wise fold manifest

**Files:**
- Create: `tests/test_cornell_cross_validation.py`
- Create: `src/shared/cornell_cross_validation.py`

**Interfaces:**
- Consumes: `samples: list[tuple[str, str]]`，每项为 `(sample_id, object_directory)`。
- Produces: `generate_image_wise_manifest(samples, n_splits=5, seed=42, validation_fraction=0.2) -> list[dict[str, object]]`
- Produces: `validate_image_wise_manifest(rows, expected_sample_ids, n_splits=5) -> None`
- Produces: `roles_for_fold(rows, fold) -> dict[str, str]`
- Produces: `save_manifest(rows, csv_path, json_path) -> str`
- Produces: `load_manifest(json_path) -> list[dict[str, object]]`
- Produces: `sha256_file(path) -> str`

- [ ] **Step 1: 写 fold 大小、覆盖和确定性失败测试**

创建测试文件，先写：

```python
from src.shared.cornell_cross_validation import generate_image_wise_manifest


def _samples(count: int = 885) -> list[tuple[str, str]]:
    return [
        (f"pcd{100 + index:04d}", f"{index // 100 + 1:02d}")
        for index in range(count)
    ]


def test_image_wise_manifest_is_deterministic_and_balanced() -> None:
    first = generate_image_wise_manifest(_samples(), seed=42)
    second = generate_image_wise_manifest(_samples(), seed=42)

    assert first == second
    for fold in range(5):
        fold_rows = [row for row in first if row["fold"] == fold]
        counts = {
            role: sum(row["role"] == role for row in fold_rows)
            for role in ("train", "validation", "test")
        }
        assert counts == {"train": 566, "validation": 142, "test": 177}

    test_ids = [
        row["sample_id"] for row in first if row["role"] == "test"
    ]
    assert len(test_ids) == 885
    assert len(set(test_ids)) == 885
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py::test_image_wise_manifest_is_deterministic_and_balanced -v
```

Expected: FAIL during collection because `src.shared.cornell_cross_validation` does not exist.

- [ ] **Step 3: 实现最小 fold 生成逻辑**

新模块顶部写明协议出处和实现范围：

```python
"""Deterministic Cornell image-wise fold manifests.

Protocol reference:
Ian Lenz, Honglak Lee, and Ashutosh Saxena,
"Deep Learning for Detecting Robotic Grasps", IJRR 2015.
https://www.cs.cornell.edu/~asaxena/papers/
lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

This module is an independent project implementation of the paper's
image-wise split definition; it does not copy third-party split code.
"""
```

实现稳定打乱、等长外层 fold 和独立内层验证划分：

```python
def generate_image_wise_manifest(
    samples: list[tuple[str, str]],
    n_splits: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> list[dict[str, object]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    ordered = sorted(samples)
    sample_ids = [sample_id for sample_id, _ in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    if len(ordered) % n_splits:
        raise ValueError("sample count must be divisible by n_splits")

    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    fold_size = len(shuffled) // n_splits
    test_folds = [
        shuffled[start : start + fold_size]
        for start in range(0, len(shuffled), fold_size)
    ]

    rows: list[dict[str, object]] = []
    directory_by_id = dict(ordered)
    all_ids = set(sample_ids)
    for fold, test_items in enumerate(test_folds):
        test_ids = {sample_id for sample_id, _ in test_items}
        remaining_ids = sorted(all_ids - test_ids)
        random.Random(seed + 1000 + fold).shuffle(remaining_ids)
        validation_count = round(len(remaining_ids) * validation_fraction)
        validation_ids = set(remaining_ids[:validation_count])

        for sample_id in sample_ids:
            role = (
                "test"
                if sample_id in test_ids
                else "validation"
                if sample_id in validation_ids
                else "train"
            )
            rows.append(
                {
                    "protocol": "cornell_image_wise_5_fold",
                    "seed": seed,
                    "fold": fold,
                    "sample_id": sample_id,
                    "object_directory": directory_by_id[sample_id],
                    "role": role,
                }
            )
    return rows
```

- [ ] **Step 4: 运行测试并确认 GREEN**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py::test_image_wise_manifest_is_deterministic_and_balanced -v
```

Expected: PASS.

- [ ] **Step 5: 写 manifest 校验失败测试**

增加：

```python
import copy
import pytest

from src.shared.cornell_cross_validation import (
    validate_image_wise_manifest,
)


def test_manifest_validator_rejects_test_leakage() -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    damaged = copy.deepcopy(rows)
    fold_zero_test = next(
        row for row in damaged if row["fold"] == 0 and row["role"] == "test"
    )
    duplicate = dict(fold_zero_test)
    duplicate["role"] = "train"
    damaged.append(duplicate)

    with pytest.raises(ValueError, match="sample has multiple roles"):
        validate_image_wise_manifest(
            damaged,
            expected_sample_ids={sample_id for sample_id, _ in _samples()},
        )


def test_manifest_validator_rejects_missing_test_coverage() -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    missing_id = next(
        row["sample_id"]
        for row in rows
        if row["fold"] == 0 and row["role"] == "test"
    )
    damaged = [
        row
        for row in rows
        if not (row["sample_id"] == missing_id and row["fold"] == 0)
    ]

    with pytest.raises(ValueError, match="fold 0 does not cover every sample"):
        validate_image_wise_manifest(
            damaged,
            expected_sample_ids={sample_id for sample_id, _ in _samples()},
        )
```

- [ ] **Step 6: 运行新增测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py -v
```

Expected: FAIL during import because `validate_image_wise_manifest` does not exist.

- [ ] **Step 7: 实现校验与 fold 角色读取**

校验每折 885 个唯一角色、角色集合、五个测试集覆盖，以及基本元数据一致性：

```python
VALID_ROLES = {"train", "validation", "test"}


def validate_image_wise_manifest(
    rows: list[dict[str, object]],
    expected_sample_ids: set[str],
    n_splits: int = 5,
) -> None:
    folds = {int(row["fold"]) for row in rows}
    if folds != set(range(n_splits)):
        raise ValueError("manifest must contain folds 0 through 4")

    test_occurrences: list[str] = []
    for fold in range(n_splits):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        pairs = [(str(row["sample_id"]), str(row["role"])) for row in fold_rows]
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"duplicate sample role in fold {fold}")
        fold_ids = [sample_id for sample_id, _ in pairs]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError(f"sample has multiple roles in fold {fold}")
        if set(fold_ids) != expected_sample_ids:
            raise ValueError(f"fold {fold} does not cover every sample")
        if {role for _, role in pairs} - VALID_ROLES:
            raise ValueError(f"fold {fold} contains an unknown role")
        test_occurrences.extend(
            sample_id for sample_id, role in pairs if role == "test"
        )

    if len(test_occurrences) != len(expected_sample_ids):
        raise ValueError("test folds do not contain the expected sample count")
    if set(test_occurrences) != expected_sample_ids:
        raise ValueError("test folds do not cover every sample exactly once")


def roles_for_fold(
    rows: list[dict[str, object]],
    fold: int,
) -> dict[str, str]:
    selected = [row for row in rows if int(row["fold"]) == fold]
    return {
        str(row["sample_id"]): str(row["role"])
        for row in selected
    }
```

- [ ] **Step 8: 运行校验测试并确认 GREEN**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py -v
```

Expected: 当前三个测试全部 PASS。

- [ ] **Step 9: 写持久化与字节确定性失败测试**

增加：

```python
import json
from pathlib import Path

from src.shared.cornell_cross_validation import (
    load_manifest,
    save_manifest,
    sha256_file,
)


def test_manifest_round_trip_and_hash_are_deterministic(tmp_path: Path) -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    first_csv = tmp_path / "first.csv"
    first_json = tmp_path / "first.json"
    second_csv = tmp_path / "second.csv"
    second_json = tmp_path / "second.json"

    first_hash = save_manifest(rows, first_csv, first_json)
    second_hash = save_manifest(rows, second_csv, second_json)

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_hash == second_hash == sha256_file(first_json)
    assert load_manifest(first_json) == rows
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["protocol"] == "cornell_image_wise_5_fold"
    assert payload["sample_count"] == 885
```

- [ ] **Step 10: 运行持久化测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py::test_manifest_round_trip_and_hash_are_deterministic -v
```

Expected: FAIL during import because persistence functions do not exist.

- [ ] **Step 11: 实现 CSV/JSON 持久化和 SHA-256**

JSON 使用稳定键顺序、UTF-8 和结尾换行；CSV 使用固定字段：

```python
MANIFEST_FIELDS = [
    "protocol",
    "seed",
    "fold",
    "sample_id",
    "object_directory",
    "role",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(
    rows: list[dict[str, object]],
    csv_path: Path,
    json_path: Path,
) -> str:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows,
        key=lambda row: (int(row["fold"]), str(row["sample_id"])),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)

    payload = {
        "protocol": "cornell_image_wise_5_fold",
        "sample_count": len({str(row["sample_id"]) for row in ordered_rows}),
        "fold_count": len({int(row["fold"]) for row in ordered_rows}),
        "rows": ordered_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(json_path)


def load_manifest(json_path: Path) -> list[dict[str, object]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "cornell_image_wise_5_fold":
        raise ValueError("unsupported manifest protocol")
    return payload["rows"]
```

- [ ] **Step 12: 运行完整 fold 测试并提交**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cornell_cross_validation.py -v
git diff --check -- \
  tests/test_cornell_cross_validation.py \
  src/shared/cornell_cross_validation.py
```

Expected: 全部 PASS，`git diff --check` 无输出。

Commit:

```bash
git add tests/test_cornell_cross_validation.py \
  src/shared/cornell_cross_validation.py
git commit -m "feat: add Cornell image-wise fold manifests"
```

---

### Task 2: 让 CNN 数据准备支持显式样本角色

**Files:**
- Create: `tests/test_cnn_grasp_dataset_selection.py`
- Modify: `src/vlm/run_cnn_grasp.py:312-368`

**Interfaces:**
- Consumes: 现有 `CornellGraspDataset`、VLM boxes、crop/target 转换。
- Produces: `build_all_samples() -> list[dict]`
- Produces: `partition_samples_by_role(samples, roles) -> tuple[list[dict], list[dict], list[dict]]`
- Preserves: `build_datasets() -> tuple[train, validation, test]`

- [ ] **Step 1: 写显式角色划分失败测试**

```python
import pytest

from src.vlm.run_cnn_grasp import partition_samples_by_role


def _items() -> list[dict]:
    return [
        {"key": ("01", "pcd0100")},
        {"key": ("01", "pcd0101")},
        {"key": ("02", "pcd0200")},
    ]


def test_partition_samples_uses_explicit_sample_roles() -> None:
    roles = {
        "pcd0100": "train",
        "pcd0101": "validation",
        "pcd0200": "test",
    }

    train, validation, test = partition_samples_by_role(_items(), roles)

    assert [item["key"][1] for item in train] == ["pcd0100"]
    assert [item["key"][1] for item in validation] == ["pcd0101"]
    assert [item["key"][1] for item in test] == ["pcd0200"]


def test_partition_samples_rejects_missing_or_unknown_roles() -> None:
    with pytest.raises(ValueError, match="missing role"):
        partition_samples_by_role(_items(), {"pcd0100": "train"})
    with pytest.raises(ValueError, match="unknown role"):
        partition_samples_by_role(
            _items(),
            {
                "pcd0100": "train",
                "pcd0101": "validation",
                "pcd0200": "holdout",
            },
        )
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_grasp_dataset_selection.py -v
```

Expected: FAIL during import because `partition_samples_by_role` does not exist.

- [ ] **Step 3: 提取全量样本构建函数并实现角色划分**

把当前 `build_datasets()` 的 crop 循环原样移入 `build_all_samples()`，每个 item
继续保存 `key`、`tensor`、`target`，不复制新的数据处理算法。增加：

```python
def partition_samples_by_role(
    samples: list[dict],
    roles: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    partitions = {"train": [], "validation": [], "test": []}
    for item in samples:
        sample_id = item["key"][1]
        if sample_id not in roles:
            raise ValueError(f"missing role for sample {sample_id}")
        role = roles[sample_id]
        if role not in partitions:
            raise ValueError(f"unknown role for sample {sample_id}: {role}")
        partitions[role].append(item)
    return (
        partitions["train"],
        partitions["validation"],
        partitions["test"],
    )
```

保持旧入口：

```python
def build_datasets():
    all_samples = build_all_samples()
    roles = {
        item["key"][1]: (
            "train"
            if item["key"][0] in TRAIN_DIRS
            else "validation"
            if item["key"][0] in VAL_DIRS
            else "test"
        )
        for item in all_samples
    }
    return partition_samples_by_role(all_samples, roles)
```

- [ ] **Step 4: 运行新增测试和旧 CNN 测试**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_grasp_dataset_selection.py \
  tests/test_cnn_grasp_reporting.py \
  tests/test_cnn_grasp_models.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 在真实数据上验证全量构建数量**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from src.vlm.run_cnn_grasp import build_all_samples

samples = build_all_samples()
ids = [item["key"][1] for item in samples]
assert len(samples) == 885
assert len(set(ids)) == 885
print({"sample_count": len(samples), "unique_ids": len(set(ids))})
PY
```

Expected:

```text
{'sample_count': 885, 'unique_ids': 885}
```

- [ ] **Step 6: 提交数据选择重构**

Run:

```bash
git diff --check -- \
  tests/test_cnn_grasp_dataset_selection.py \
  src/vlm/run_cnn_grasp.py
git add tests/test_cnn_grasp_dataset_selection.py \
  src/vlm/run_cnn_grasp.py
git commit -m "refactor: support explicit CNN sample roles"
```

---

### Task 3: 修复评估函数忽略样本选择的问题

**Files:**
- Modify: `tests/test_cnn_grasp_dataset_selection.py`
- Modify: `src/vlm/run_cnn_grasp.py:608-720`

**Interfaces:**
- Consumes: `evaluate_model(model, all_samples, dataset, vlm_boxes, device)`
- Produces: `_selected_sample_keys(samples) -> set[tuple[str, str]]`
- Changes: `evaluate_model` 只评估 `all_samples` 参数包含的 key。
- Preserves: 传入全部 885 个样本时的历史全量评估行为。

- [ ] **Step 1: 写样本 key 选择失败测试**

在测试文件增加：

```python
from src.vlm.run_cnn_grasp import _selected_sample_keys


def test_selected_sample_keys_restrict_evaluation_scope() -> None:
    selected = _selected_sample_keys(
        [
            {"key": ("01", "pcd0100")},
            {"key": ("09", "pcd0900")},
        ]
    )
    assert selected == {("01", "pcd0100"), ("09", "pcd0900")}


def test_selected_sample_keys_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate sample key"):
        _selected_sample_keys(
            [
                {"key": ("01", "pcd0100")},
                {"key": ("01", "pcd0100")},
            ]
        )
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_grasp_dataset_selection.py::test_selected_sample_keys_restrict_evaluation_scope \
  tests/test_cnn_grasp_dataset_selection.py::test_selected_sample_keys_reject_duplicates -v
```

Expected: FAIL during import because `_selected_sample_keys` does not exist.

- [ ] **Step 3: 实现 key 校验并接入评估循环**

```python
def _selected_sample_keys(
    samples: list[dict],
) -> set[tuple[str, str]]:
    keys = [tuple(item["key"]) for item in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample key in evaluation selection")
    return set(keys)
```

在 `evaluate_model` 进入数据集循环前增加：

```python
selected_keys = _selected_sample_keys(all_samples)
```

在获取 `key` 后、读取 VLM box 前增加：

```python
if key not in selected_keys:
    continue
```

删除未使用的局部 `import torch`，其他预测与 Cornell 指标逻辑不变。

- [ ] **Step 4: 运行所有当前测试**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 用历史权重验证 85 样本评估数量**

使用确定性单头 seed 42 权重，只评估旧测试数据：

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
import torch

from src.shared.cornell_dataset import CornellGraspDataset
from src.vlm.run_cnn_grasp import (
    DATASET_ROOT,
    _load_state_dict,
    build_datasets,
    create_model,
    evaluate_model,
    load_vlm_boxes,
)

_, _, test_samples = build_datasets()
model = create_model("single")
state = torch.load(
    "data/processed/vlm/cnn_grasp_single_head_deterministic/"
    "cnn_grasp_model_seed_42.pt",
    map_location="cpu",
    weights_only=True,
)
_load_state_dict(model, "single", state)
rows, summary = evaluate_model(
    model,
    test_samples,
    CornellGraspDataset(DATASET_ROOT),
    load_vlm_boxes(),
    device="cpu",
)
assert len(rows) == 85
assert summary["sample_count"] == 85
print(summary["sample_count"])
PY
```

Expected: 打印 `85`。

- [ ] **Step 6: 提交评估范围修复**

```bash
git diff --check -- \
  tests/test_cnn_grasp_dataset_selection.py \
  src/vlm/run_cnn_grasp.py
git add tests/test_cnn_grasp_dataset_selection.py \
  src/vlm/run_cnn_grasp.py
git commit -m "fix: evaluate only selected CNN samples"
```

---

### Task 4: 实现五折产物路径、审计和汇总

**Files:**
- Create: `tests/test_cnn_cross_validation.py`
- Create: `src/vlm/run_cnn_cross_validation.py`

**Interfaces:**
- Consumes: fold records with `fold`, `summary`, `rows`, `best_val_loss`。
- Produces: `build_fold_paths(output_root, architecture, fold) -> FoldPaths`
- Produces: `metrics_from_rows(rows) -> dict[str, float | int]`
- Produces: `build_cross_validation_summary(fold_records, combined_rows, architecture, seed, manifest_hash) -> dict`
- Produces: `validate_complete_fold_records(fold_records, combined_rows, expected_sample_ids) -> None`
- Produces: `build_architecture_comparison(single_summary, multi_summary) -> dict`

- [ ] **Step 1: 写 fold 路径失败测试**

```python
from pathlib import Path

from src.vlm.run_cnn_cross_validation import build_fold_paths


def test_fold_paths_isolate_every_artifact(tmp_path: Path) -> None:
    paths = build_fold_paths(tmp_path, "multi_head", 3)
    expected = tmp_path / "multi_head" / "fold_3"

    assert paths.directory == expected
    assert paths.model == expected / "model.pt"
    assert paths.history == expected / "training_history.json"
    assert paths.predictions == expected / "predictions.csv"
    assert paths.summary == expected / "summary.json"
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py::test_fold_paths_isolate_every_artifact -v
```

Expected: FAIL during collection because the runner module does not exist.

- [ ] **Step 3: 创建 runner 模块和路径 dataclass**

模块 docstring 必须说明：

```python
"""Run project-authored Cornell image-wise CNN cross-validation.

The split protocol follows Lenz, Lee, and Saxena (IJRR 2015):
https://www.cs.cornell.edu/~asaxena/papers/
lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

The orchestration code is independently written for this project and does not
copy an external cross-validation implementation.
"""
```

实现：

```python
@dataclass(frozen=True)
class FoldPaths:
    directory: Path
    model: Path
    history: Path
    predictions: Path
    summary: Path


def build_fold_paths(
    output_root: Path,
    architecture: str,
    fold: int,
) -> FoldPaths:
    directory = output_root / architecture / f"fold_{fold}"
    return FoldPaths(
        directory=directory,
        model=directory / "model.pt",
        history=directory / "training_history.json",
        predictions=directory / "predictions.csv",
        summary=directory / "summary.json",
    )
```

- [ ] **Step 4: 运行路径测试并确认 GREEN**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py::test_fold_paths_isolate_every_artifact -v
```

Expected: PASS。

- [ ] **Step 5: 写汇总和不完整结果拒绝测试**

```python
import pytest

from src.vlm.run_cnn_cross_validation import (
    build_architecture_comparison,
    build_cross_validation_summary,
    validate_complete_fold_records,
)


def _row(sample_id: str, success: int, iou: float, angle: float) -> dict:
    return {
        "sample_id": sample_id,
        "success": success,
        "best_iou": iou,
        "best_angle_error_degrees": angle,
    }


def test_cross_validation_summary_reports_pooled_and_fold_metrics() -> None:
    fold_records = []
    combined = []
    for fold in range(5):
        rows = [
            _row(f"sample_{fold}_0", 1, 0.5, 10.0),
            _row(f"sample_{fold}_1", 0, 0.2, 40.0),
        ]
        combined.extend(rows)
        fold_records.append(
            {
                "fold": fold,
                "best_val_loss": 0.1 + fold / 100,
                "rows": rows,
            }
        )

    summary = build_cross_validation_summary(
        fold_records,
        combined,
        architecture="single",
        seed=42,
        manifest_hash="abc123",
    )

    assert summary["protocol"] == "cornell_image_wise_5_fold"
    assert summary["pooled"]["sample_count"] == 10
    assert summary["pooled"]["success_rate"] == pytest.approx(0.5)
    assert summary["folds"]["success_rate_mean"] == pytest.approx(0.5)
    assert len(summary["per_fold"]) == 5
    assert summary["manifest_sha256"] == "abc123"


def test_cross_validation_audit_rejects_missing_fold() -> None:
    records = [
        {"fold": fold, "rows": [_row(f"s{fold}", 1, 0.5, 10.0)]}
        for fold in range(4)
    ]
    combined = [row for record in records for row in record["rows"]]

    with pytest.raises(ValueError, match="folds 0 through 4"):
        validate_complete_fold_records(
            records,
            combined,
            expected_sample_ids={f"s{fold}" for fold in range(5)},
        )


def test_architecture_comparison_requires_the_same_manifest() -> None:
    single = {
        "architecture": "single",
        "manifest_sha256": "same",
        "pooled": {
            "success_rate": 0.70,
            "mean_iou": 0.40,
            "mean_angle": 18.0,
        },
        "per_fold": [
            {
                "fold": fold,
                "success_rate": 0.70,
                "mean_iou": 0.40,
                "mean_angle": 18.0,
            }
            for fold in range(5)
        ],
    }
    multi = {
        "architecture": "multi_head",
        "manifest_sha256": "same",
        "pooled": {
            "success_rate": 0.72,
            "mean_iou": 0.44,
            "mean_angle": 17.0,
        },
        "per_fold": [
            {
                "fold": fold,
                "success_rate": 0.72,
                "mean_iou": 0.44,
                "mean_angle": 17.0,
            }
            for fold in range(5)
        ],
    }

    comparison = build_architecture_comparison(single, multi)

    assert comparison["pooled_delta_multi_minus_single"] == {
        "success_rate": pytest.approx(0.02),
        "mean_iou": pytest.approx(0.04),
        "mean_angle": pytest.approx(-1.0),
    }
    assert len(comparison["paired_fold_deltas"]) == 5

    multi["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        build_architecture_comparison(single, multi)
```

- [ ] **Step 6: 运行汇总测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py -v
```

Expected: FAIL during import because aggregation functions do not exist.

- [ ] **Step 7: 实现纯指标、审计和五折汇总**

```python
def metrics_from_rows(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot calculate metrics from empty rows")
    return {
        "sample_count": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "success_rate": float(
            np.mean([int(row["success"]) for row in rows])
        ),
        "mean_iou": float(np.mean([float(row["best_iou"]) for row in rows])),
        "mean_angle": float(
            np.mean(
                [float(row["best_angle_error_degrees"]) for row in rows]
            )
        ),
    }


def validate_complete_fold_records(
    fold_records: list[dict],
    combined_rows: list[dict],
    expected_sample_ids: set[str],
) -> None:
    folds = {int(record["fold"]) for record in fold_records}
    if folds != set(range(5)):
        raise ValueError("fold records must contain folds 0 through 4")
    ids = [str(row["sample_id"]) for row in combined_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("combined predictions contain duplicate sample IDs")
    if set(ids) != expected_sample_ids:
        raise ValueError("combined predictions do not cover expected sample IDs")
    for value in (
        float(row[field])
        for row in combined_rows
        for field in (
            "best_iou",
            "best_angle_error_degrees",
            "pred_center_x",
            "pred_center_y",
            "pred_width",
            "pred_height",
            "pred_angle_degrees",
        )
        if field in row
    ):
        if not math.isfinite(value):
            raise ValueError("combined predictions contain non-finite values")


def build_cross_validation_summary(
    fold_records: list[dict],
    combined_rows: list[dict],
    architecture: str,
    seed: int,
    manifest_hash: str,
) -> dict:
    per_fold = []
    for record in sorted(fold_records, key=lambda item: int(item["fold"])):
        metrics = metrics_from_rows(record["rows"])
        per_fold.append(
            {
                "fold": int(record["fold"]),
                "best_val_loss": float(record["best_val_loss"]),
                **metrics,
            }
        )

    def aggregate(field: str) -> tuple[float, float]:
        values = [float(record[field]) for record in per_fold]
        return float(np.mean(values)), float(np.std(values))

    success_mean, success_std = aggregate("success_rate")
    iou_mean, iou_std = aggregate("mean_iou")
    angle_mean, angle_std = aggregate("mean_angle")
    return {
        "method": f"vlm_cnn_{architecture}_image_wise_5_fold",
        "protocol": "cornell_image_wise_5_fold",
        "architecture": architecture,
        "fold_count": 5,
        "training_seed_per_fold": seed,
        "manifest_sha256": manifest_hash,
        "pooled": metrics_from_rows(combined_rows),
        "folds": {
            "success_rate_mean": success_mean,
            "success_rate_std": success_std,
            "mean_iou_mean": iou_mean,
            "mean_iou_std": iou_std,
            "mean_angle_mean": angle_mean,
            "mean_angle_std": angle_std,
        },
        "per_fold": per_fold,
    }


def build_architecture_comparison(
    single_summary: dict,
    multi_summary: dict,
) -> dict:
    if single_summary["architecture"] != "single":
        raise ValueError("single summary has the wrong architecture")
    if multi_summary["architecture"] != "multi_head":
        raise ValueError("multi summary has the wrong architecture")
    if (
        single_summary["manifest_sha256"]
        != multi_summary["manifest_sha256"]
    ):
        raise ValueError("architecture summaries use different manifests")

    fields = ("success_rate", "mean_iou", "mean_angle")
    pooled_delta = {
        field: float(multi_summary["pooled"][field])
        - float(single_summary["pooled"][field])
        for field in fields
    }
    single_folds = {
        int(record["fold"]): record
        for record in single_summary["per_fold"]
    }
    multi_folds = {
        int(record["fold"]): record
        for record in multi_summary["per_fold"]
    }
    if set(single_folds) != set(range(5)) or set(multi_folds) != set(range(5)):
        raise ValueError("both architecture summaries must contain five folds")
    paired = [
        {
            "fold": fold,
            **{
                field: float(multi_folds[fold][field])
                - float(single_folds[fold][field])
                for field in fields
            },
        }
        for fold in range(5)
    ]
    return {
        "protocol": "cornell_image_wise_5_fold",
        "manifest_sha256": single_summary["manifest_sha256"],
        "delta_definition": "multi_head_minus_single",
        "pooled_delta_multi_minus_single": pooled_delta,
        "paired_fold_deltas": paired,
        "single": single_summary,
        "multi_head": multi_summary,
    }
```

- [ ] **Step 8: 运行 runner 纯逻辑测试并提交**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py -v
git diff --check -- \
  tests/test_cnn_cross_validation.py \
  src/vlm/run_cnn_cross_validation.py
```

Expected: 全部 PASS。

Commit:

```bash
git add tests/test_cnn_cross_validation.py \
  src/vlm/run_cnn_cross_validation.py
git commit -m "feat: add CNN cross-validation summaries"
```

---

### Task 5: 实现 manifest、单 fold 训练和可恢复聚合 CLI

**Files:**
- Modify: `tests/test_cnn_cross_validation.py`
- Modify: `src/vlm/run_cnn_cross_validation.py`

**Interfaces:**
- Consumes: Tasks 1–4 的 manifest、样本、训练、评估、路径和汇总函数。
- Produces: `prepare_manifest(output_root, seed=42) -> tuple[list[dict], Path, str]`
- Produces: `run_fold(architecture, fold, device, seed, output_root, manifest_rows, manifest_hash) -> dict`
- Produces: `aggregate_saved_folds(architecture, output_root, expected_sample_ids, manifest_hash, seed) -> dict`
- Produces CLI: `--mode manifest|run|aggregate|compare`, `--architecture single|multi_head`, `--fold 0..4`。

- [ ] **Step 1: 写真实 Cornell manifest 准备测试**

```python
from src.vlm.run_cnn_cross_validation import prepare_manifest


def test_prepare_manifest_uses_all_real_cornell_samples(tmp_path: Path) -> None:
    rows, json_path, manifest_hash = prepare_manifest(tmp_path, seed=42)

    assert json_path == tmp_path / "image_wise_folds_seed_42.json"
    assert json_path.exists()
    assert len({row["sample_id"] for row in rows}) == 885
    assert len([row for row in rows if row["role"] == "test"]) == 885
    assert len(manifest_hash) == 64
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py::test_prepare_manifest_uses_all_real_cornell_samples -v
```

Expected: FAIL during import because `prepare_manifest` does not exist.

- [ ] **Step 3: 实现真实 manifest 准备**

`prepare_manifest` 从 `CornellGraspDataset(DATASET_ROOT).samples` 读取稳定 ID 和原始
目录，生成或加载固定路径；加载已有 JSON 时必须重新校验 885 个样本，不能静默
接受过期清单：

```python
def prepare_manifest(
    output_root: Path,
    seed: int = 42,
) -> tuple[list[dict], Path, str]:
    dataset = CornellGraspDataset(DATASET_ROOT)
    samples = [
        (sample.sample_id, sample.object_directory)
        for sample in dataset.samples
    ]
    expected_ids = {sample_id for sample_id, _ in samples}
    csv_path = output_root / f"image_wise_folds_seed_{seed}.csv"
    json_path = output_root / f"image_wise_folds_seed_{seed}.json"
    if json_path.exists():
        rows = load_manifest(json_path)
    else:
        rows = generate_image_wise_manifest(samples, seed=seed)
        save_manifest(rows, csv_path, json_path)
    validate_image_wise_manifest(rows, expected_ids)
    return rows, json_path, sha256_file(json_path)
```

- [ ] **Step 4: 运行真实 manifest 测试并确认 GREEN**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py::test_prepare_manifest_uses_all_real_cornell_samples -v
```

Expected: PASS。

- [ ] **Step 5: 写保存 fold 聚合失败测试**

使用真实临时 CSV/JSON，不 mock 文件副作用：

```python
import csv
import json

from src.vlm.run_cnn_cross_validation import aggregate_saved_folds


def test_aggregate_saved_folds_requires_five_complete_outputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="fold_0"):
        aggregate_saved_folds(
            architecture="single",
            output_root=tmp_path,
            expected_sample_ids={"pcd0100"},
            manifest_hash="abc",
            seed=42,
        )
```

- [ ] **Step 6: 运行聚合测试并确认 RED**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/test_cnn_cross_validation.py::test_aggregate_saved_folds_requires_five_complete_outputs -v
```

Expected: FAIL during import because `aggregate_saved_folds` does not exist.

- [ ] **Step 7: 实现单 fold 训练、逐 fold 元数据和保存聚合**

`run_fold` 必须：

1. 调用 `build_all_samples()` 一次；
2. 用 `roles_for_fold` 和 `partition_samples_by_role` 得到 `566/142/177`；
3. 明确断言三个集合的 ID 互斥；
4. 调用 `_train_one_run`，传入 fold 专用模型和历史路径；
5. 调用已经修复的 `evaluate_model(model, test_samples, ...)`；
6. 给每行加入 `protocol`、`fold`、`split="test"`、`architecture`、
   `training_seed`、`manifest_sha256`；
7. 汇总加入真实 `best_val_loss`、训练/验证/测试数量和 manifest 哈希；
8. 调用 `save_results` 写 fold 专用 CSV/JSON；
9. 返回包含 `fold`、`best_val_loss`、`rows`、`summary` 的 record。

`aggregate_saved_folds` 必须逐一读取五个 `predictions.csv`、`summary.json` 和
`training_history.json`，验证：

```python
best_val_loss == min(epoch["val_loss"] for epoch in history)
summary["manifest_sha256"] == manifest_hash
summary["architecture"] == architecture
summary["fold"] == fold
len(rows) == summary["test_count"] == 177
```

随后调用 `validate_complete_fold_records`，保存：

```text
data/processed/vlm/cnn_cross_validation/single/combined_predictions.csv
data/processed/vlm/cnn_cross_validation/single/cross_validation_summary.json
data/processed/vlm/cnn_cross_validation/multi_head/combined_predictions.csv
data/processed/vlm/cnn_cross_validation/multi_head/cross_validation_summary.json
```

`combined_predictions.csv` 按 `sample_id` 排序。

实现主体使用以下明确数据流：

```python
def run_fold(
    architecture: str,
    fold: int,
    device: str,
    seed: int,
    output_root: Path,
    manifest_rows: list[dict],
    manifest_hash: str,
) -> dict:
    all_samples = build_all_samples()
    role_map = roles_for_fold(manifest_rows, fold)
    train_data, validation_data, test_data = partition_samples_by_role(
        all_samples,
        role_map,
    )
    counts = (
        len(train_data),
        len(validation_data),
        len(test_data),
    )
    if counts != (566, 142, 177):
        raise ValueError(f"unexpected fold {fold} sizes: {counts}")

    id_sets = [
        {item["key"][1] for item in partition}
        for partition in (train_data, validation_data, test_data)
    ]
    if (
        id_sets[0] & id_sets[1]
        or id_sets[0] & id_sets[2]
        or id_sets[1] & id_sets[2]
    ):
        raise ValueError(f"sample leakage detected in fold {fold}")

    paths = build_fold_paths(output_root, architecture, fold)
    model, best_val_loss = _train_one_run(
        train_data,
        validation_data,
        device,
        seed,
        architecture=architecture,
        model_weights_path=paths.model,
        history_path=paths.history,
    )
    rows, evaluation = evaluate_model(
        model,
        test_data,
        CornellGraspDataset(DATASET_ROOT),
        load_vlm_boxes(),
        device=device,
    )
    for row in rows:
        row.update(
            {
                "protocol": "cornell_image_wise_5_fold",
                "fold": fold,
                "split": "test",
                "architecture": architecture,
                "training_seed": seed,
                "manifest_sha256": manifest_hash,
            }
        )
    summary = {
        "method": f"vlm_cnn_{architecture}_image_wise_fold",
        "protocol": "cornell_image_wise_5_fold",
        "architecture": architecture,
        "fold": fold,
        "training_seed": seed,
        "manifest_sha256": manifest_hash,
        "train_count": len(train_data),
        "validation_count": len(validation_data),
        "test_count": len(test_data),
        "best_val_loss": best_val_loss,
        **metrics_from_rows(rows),
    }
    save_results(
        rows,
        summary,
        predictions_csv=paths.predictions,
        summary_json=paths.summary,
    )
    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "rows": rows,
        "summary": summary,
    }


def aggregate_saved_folds(
    architecture: str,
    output_root: Path,
    expected_sample_ids: set[str],
    manifest_hash: str,
    seed: int,
) -> dict:
    fold_records = []
    for fold in range(5):
        paths = build_fold_paths(output_root, architecture, fold)
        for path in (paths.history, paths.predictions, paths.summary):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing fold_{fold} artifact: {path}"
                )
        history = json.loads(paths.history.read_text(encoding="utf-8"))
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        with paths.predictions.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))

        expected_best = min(float(epoch["val_loss"]) for epoch in history)
        if not math.isclose(
            float(summary["best_val_loss"]),
            expected_best,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"fold {fold} best validation loss mismatch")
        if summary["manifest_sha256"] != manifest_hash:
            raise ValueError(f"fold {fold} manifest hash mismatch")
        if summary["architecture"] != architecture:
            raise ValueError(f"fold {fold} architecture mismatch")
        if int(summary["fold"]) != fold:
            raise ValueError(f"fold {fold} metadata mismatch")
        if len(rows) != int(summary["test_count"]) or len(rows) != 177:
            raise ValueError(f"fold {fold} prediction count mismatch")
        fold_records.append(
            {
                "fold": fold,
                "best_val_loss": expected_best,
                "rows": rows,
                "summary": summary,
            }
        )

    combined_rows = sorted(
        (
            row
            for record in fold_records
            for row in record["rows"]
        ),
        key=lambda row: row["sample_id"],
    )
    validate_complete_fold_records(
        fold_records,
        combined_rows,
        expected_sample_ids,
    )
    final_summary = build_cross_validation_summary(
        fold_records,
        combined_rows,
        architecture,
        seed,
        manifest_hash,
    )
    architecture_dir = output_root / architecture
    save_results(
        combined_rows,
        final_summary,
        predictions_csv=architecture_dir / "combined_predictions.csv",
        summary_json=architecture_dir / "cross_validation_summary.json",
    )
    return final_summary
```

- [ ] **Step 8: 实现 CLI 和安全设备选择**

参数：

```python
parser.add_argument(
    "--mode",
    choices=["manifest", "run", "aggregate", "compare"],
    default="run",
)
parser.add_argument(
    "--architecture",
    choices=["single", "multi_head"],
    default="single",
)
parser.add_argument("--fold", type=int, choices=range(5), default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data/processed/vlm/cnn_cross_validation"),
)
```

行为：

- `manifest`：只生成、校验和打印 manifest 哈希；
- `run --fold N`：只运行第 N 折，便于中断后恢复；
- `run` 且没有 `--fold`：顺序运行 0–4；
- `aggregate`：不训练，只从五套保存产物生成最终汇总；
- `compare`：读取两个架构的最终汇总，校验共同 manifest 后写入
  `architecture_comparison.json`；
- 请求 CUDA 但 `torch.cuda.is_available()` 为假时抛出错误，不能静默切换 CPU；
- 每折开始前打印 manifest 哈希和 `train/validation/test` 数量；
- 每折结束后立即保存产物，不等待五折全部完成。

- [ ] **Step 9: 运行全部测试和编译检查**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/shared/cornell_cross_validation.py \
  src/vlm/run_cnn_grasp.py \
  src/vlm/run_cnn_cross_validation.py
git diff --check
```

Expected: 全部退出码为 0。

- [ ] **Step 10: 提交完整 runner**

```bash
git add tests/test_cnn_cross_validation.py \
  src/vlm/run_cnn_cross_validation.py
git commit -m "feat: run resumable CNN image-wise folds"
```

---

### Task 6: 生成正式 manifest 并完成训练前审计

**Files:**
- Generated, ignored: `data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.csv`
- Generated, ignored: `data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.json`
- Modify: `docs/planning/cornell_split_audit.md`
- Modify: `docs/planning/experiment_result_provenance.md`

**Interfaces:**
- Consumes: Task 5 CLI。
- Produces: 经过真实 885 样本校验的固定 manifest 和记录的 SHA-256。

- [ ] **Step 1: 在沙箱内生成 manifest**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  src/vlm/run_cnn_cross_validation.py \
  --mode manifest \
  --seed 42
```

Expected:

```text
protocol=cornell_image_wise_5_fold
sample_count=885
test_fold_sizes=[177, 177, 177, 177, 177]
```

并打印一个 64 字符 SHA-256。

- [ ] **Step 2: 独立审计 manifest**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

from src.shared.cornell_cross_validation import (
    load_manifest,
    roles_for_fold,
    sha256_file,
    validate_image_wise_manifest,
)
from src.shared.cornell_dataset import CornellGraspDataset

path = Path(
    "data/processed/vlm/cnn_cross_validation/"
    "image_wise_folds_seed_42.json"
)
rows = load_manifest(path)
dataset = CornellGraspDataset("data/raw/cornell")
expected = {sample.sample_id for sample in dataset.samples}
validate_image_wise_manifest(rows, expected)
for fold in range(5):
    print(fold, Counter(roles_for_fold(rows, fold).values()))
print("sha256", sha256_file(path))
PY
```

Expected: 每折打印 `train=566`、`validation=142`、`test=177`，最后打印与 Step 1
一致的哈希。

- [ ] **Step 3: 记录协议来源、阻塞和哈希**

在 `cornell_split_audit.md` 写明：

- image-wise 只需要唯一图片 ID，因此可可靠实现；
- object-wise 需要缺失的物体实例映射，因此未生成；
- `01`–`10` 是存储目录，不是 object ID；
- manifest seed、五折大小和 SHA-256；
- 协议定义来自 Lenz et al.，代码为本项目独立实现；
- image-wise 不支持未见物体结论。

在 `experiment_result_provenance.md` 增加 manifest 行：

```markdown
| Cornell image-wise fold manifest | `data/processed/vlm/cnn_cross_validation/image_wise_folds_seed_42.json` | `cornell_image_wise_5_fold` | 885 个样本覆盖、五个 177 测试 fold、SHA-256 已审计 | 单头/多头正式五折共同使用 |
```

将实际哈希写入紧邻段落，不使用示例占位符。

- [ ] **Step 4: 运行 CPU 单 batch 和 CUDA 严格确定性冒烟**

CPU 冒烟通过直接调用 `train_model` 时临时将模块常量 `NUM_EPOCHS=1`，只使用
真实 fold 0 的 32 个训练样本和 16 个验证样本，输出写入 `/tmp`。

CUDA 冒烟必须在沙箱外使用用户已授权的 conda Python，并同样只跑一轮；确认：

- `torch.are_deterministic_algorithms_enabled()` 为真；
- 单头和多头各自完成 forward、backward、保存与 4 样本评估；
- 不写入正式 fold 目录。

- [ ] **Step 5: 提交审计文档**

```bash
git add docs/planning/cornell_split_audit.md \
  docs/planning/experiment_result_provenance.md
git commit -m "docs: audit Cornell image-wise manifest"
```

---

### Task 7: 更新运行文档和项目结构

**Files:**
- Modify: `src/vlm/README.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: 已验证 CLI 和 manifest。
- Produces: 可直接复制执行的正式命令和准确模块职责。

- [ ] **Step 1: 记录命令**

在 `src/vlm/README.md` 加入：

```bash
# 只生成并审计共同 fold manifest
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode manifest

# 单头逐 fold 运行；逐折执行便于中断恢复
for fold in 0 1 2 3 4; do
  conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
    --mode run --architecture single --fold "$fold" --device cuda
done

# 多头逐 fold 运行
for fold in 0 1 2 3 4; do
  conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
    --mode run --architecture multi_head --fold "$fold" --device cuda
done

# 五折齐全后聚合
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode aggregate --architecture single
conda run -n msc-grasp python src/vlm/run_cnn_cross_validation.py \
  --mode aggregate --architecture multi_head
```

同时明确“五随机种子”和“五折”的统计含义不同。

- [ ] **Step 2: 更新项目结构和工作日志**

`PROJECT_STRUCTURE.md` 增加：

- `src/shared/cornell_cross_validation.py`：fold manifest 生成、持久化和泄漏审计；
- `src/vlm/run_cnn_cross_validation.py`：单头/多头 image-wise 五折训练与聚合；
- 正式输出目录树和命令。

`WORKLOG.md` 只记录已完成并验证的实现与 manifest，不提前写入训练结果。

- [ ] **Step 3: 验证文档路径和提交**

Run:

```bash
for path in \
  src/shared/cornell_cross_validation.py \
  src/vlm/run_cnn_cross_validation.py \
  tests/test_cornell_cross_validation.py \
  tests/test_cnn_cross_validation.py; do
  test -f "$path" || exit 1
done
git diff --check -- \
  src/vlm/README.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/worklog/WORKLOG.md
```

Expected: 退出码为 0。

Commit:

```bash
git add src/vlm/README.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: document CNN image-wise validation"
```

---

### Task 8: 运行单头与多头正式五折

**Files:**
- Generated, ignored: `data/processed/vlm/cnn_cross_validation/single/`
- Generated, ignored: `data/processed/vlm/cnn_cross_validation/multi_head/`

**Interfaces:**
- Consumes: 共同 manifest SHA-256 和 Task 5 runner。
- Produces: 两个架构各五套 fold 产物、885 行合并预测和最终 JSON 汇总。

- [ ] **Step 1: 运行训练前最终门禁**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -q
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/shared/cornell_cross_validation.py \
  src/vlm/run_cnn_grasp.py \
  src/vlm/run_cnn_cross_validation.py
git diff --check
```

Expected: 全部退出码为 0。

- [ ] **Step 2: 在沙箱外逐折运行单头**

每一折单独执行，完成后立即检查四个产物存在。命令必须使用：

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  src/vlm/run_cnn_cross_validation.py \
  --mode run \
  --architecture single \
  --fold 0 \
  --seed 42 \
  --device cuda
```

将 `--fold` 依次改为 `1`、`2`、`3`、`4`。五折完成后用下面的明确路径检查：

```bash
for fold in 0 1 2 3 4; do
  test -s "data/processed/vlm/cnn_cross_validation/single/fold_${fold}/model.pt"
  test -s "data/processed/vlm/cnn_cross_validation/single/fold_${fold}/training_history.json"
  test -s "data/processed/vlm/cnn_cross_validation/single/fold_${fold}/predictions.csv"
  test -s "data/processed/vlm/cnn_cross_validation/single/fold_${fold}/summary.json"
done
```

该循环只做存在性读取检查，不删除或覆盖任何产物。

- [ ] **Step 3: 聚合并审计单头**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  src/vlm/run_cnn_cross_validation.py \
  --mode aggregate \
  --architecture single \
  --seed 42
```

Expected: `combined_predictions.csv` 有 885 个唯一数据行，
`cross_validation_summary.json` 记录五个 fold、共同 manifest 哈希和有限指标。

- [ ] **Step 4: 在沙箱外逐折运行多头**

重复 Step 2，将架构改为：

```text
--architecture multi_head
```

不改变 seed、manifest 或 fold 编号。

- [ ] **Step 5: 聚合并审计多头**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  src/vlm/run_cnn_cross_validation.py \
  --mode aggregate \
  --architecture multi_head \
  --seed 42
```

Expected: 与单头相同的 885 个 sample ID，五个 fold summary 的 manifest 哈希完全
一致。

- [ ] **Step 6: 生成成对比较**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python \
  src/vlm/run_cnn_cross_validation.py \
  --mode compare \
  --seed 42
```

该命令读取两个 `cross_validation_summary.json` 和两个
`combined_predictions.csv`，验证 sample ID 集合相同后输出：

- pooled 成功率、IoU、角度差；
- 五折成对成功率、IoU、角度差；
- 每折单头与多头指标；
- 两种架构各自的五折均值和总体标准差；
- 不计算或声称 object-wise 性能。

比较结果写入：

```text
data/processed/vlm/cnn_cross_validation/architecture_comparison.json
```

- [ ] **Step 7: 保存正式审计输出**

独立脚本或一次性只读校验必须确认：

```text
single combined rows = 885
multi_head combined rows = 885
single unique IDs = 885
multi_head unique IDs = 885
ID sets equal = true
manifest hashes equal = true
each fold rows = 177
each best_val_loss equals history minimum
all numeric metrics finite = true
```

只有全部为真，结果才能进入项目记录。

---

### Task 9: 更新状态、论文和结果溯源

**Files:**
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/planning/experiment_result_provenance.md`
- Modify: `docs/planning/modern_2d_grasp_literature_matrix.md`
- Modify: `uog_dissertation_outline/l4proj.tex`

**Interfaces:**
- Consumes: Task 8 经过审计的两个正式汇总和成对比较。
- Produces: 可追溯且不夸大泛化结论的项目状态与论文文字。

- [ ] **Step 1: 更新当前状态**

用实际 JSON 数字替换旧的“下一步需要完成”描述，明确区分：

- 固定目录、五随机种子实验；
- image-wise 五折实验；
- object-wise 因缺失实例映射未完成。

不得把五折标准差解释为五随机种子稳定性。

- [ ] **Step 2: 更新结果溯源**

记录：

- manifest CSV/JSON 路径和 SHA-256；
- 单头、 多头五个 fold 目录；
- 两个合并预测和最终汇总；
- 成对比较 JSON；
- 每个结果的可复核状态和论文使用决定。

- [ ] **Step 3: 更新现代文献比较**

把本项目增加为 image-wise 五折一行，并继续注明：

- 输入是 Grounding DINO RGB crop；
- 输出单个抓取矩形；
- 指标与 Cornell rectangle metric 一致；
- fold 成员并非来自其他论文未公开的清单；
- 与 RGB-D、深度或密集输出论文仅有限可比。

- [ ] **Step 4: 更新 Methodology、Results 和 Discussion**

Methodology 写明：

- 885 图像、五个 177 测试 fold；
- 每折 566/142/177；
- fold seed 和训练 seed；
- 单头/多头共享 manifest；
- 测试集不参与早停；
- 合并指标与五折统计口径。

Results 只陈述实际审计数字。Discussion 写明：

- image-wise 允许同一物体的不同视角跨集合；
- 因此不能证明未见物体泛化；
- 与文献比较受输入模态、定位 crop、输出形式和具体 fold 清单限制；
- object-wise 未完成是元数据限制，而非算法性能结果。

- [ ] **Step 5: 完整验证**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest tests -q
git diff --check
cd uog_dissertation_outline
XDG_CACHE_HOME=/tmp/msc-tectonic-cache \
  /home/pzk/miniconda/envs/msc-grasp/bin/tectonic \
  --keep-logs l4proj.tex
cd ..
```

Expected: 测试通过、diff 检查无输出、论文生成 `l4proj.pdf` 且退出码为 0。

- [ ] **Step 6: 数字一致性检查**

从 LaTeX 提取所有 image-wise 数字，与两个
`cross_validation_summary.json` 和 `architecture_comparison.json` 对照。检查：

- 成功率小数位一致；
- IoU 和角度值一致；
- pooled 与 fold mean 没有混写；
- single 与 multi_head 没有调换；
- 固定目录结果没有标成 image-wise；
- image-wise 没有标成 object-wise。

- [ ] **Step 7: 提交文档和论文更新**

```bash
git add docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md \
  docs/planning/experiment_result_provenance.md \
  docs/planning/modern_2d_grasp_literature_matrix.md \
  uog_dissertation_outline/l4proj.tex
git commit -m "docs: report CNN image-wise cross-validation"
```

---

## 最终验收

- [ ] `pytest tests -q` 全部通过。
- [ ] Python 编译检查通过。
- [ ] manifest 覆盖 885 个样本，每折 `566/142/177`。
- [ ] 五个测试 fold 两两互斥，合并后覆盖全部 885 个样本。
- [ ] 单头和多头 manifest SHA-256 一致。
- [ ] 两个架构各保存五个模型、五个历史、五个预测和五个 summary。
- [ ] 两个合并预测各有 885 个唯一 sample ID。
- [ ] 每折最佳验证损失与历史最小值一致。
- [ ] 所有正式指标有限且来自保存产物。
- [ ] 论文准确区分固定目录实验、五随机种子实验和 image-wise 五折。
- [ ] object-wise 仍明确标记为实例元数据阻塞。
- [ ] 没有未标注的外部复制或改编代码。
