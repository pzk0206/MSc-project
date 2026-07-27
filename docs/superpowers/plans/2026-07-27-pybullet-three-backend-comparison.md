# PyBullet Three-Backend Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定多物体 PyBullet 场景中，让 geometry、正式 single seed 42 和正式 multi-head seed 42 对相同定位框生成二维抓取结果与可审计并列图。

**Architecture:** 保留现有一次渲染、一次 Grounding DINO 加载和目标选择流程；新增纯后端几何审计模块，扩展 runner 使两个 CNN 各加载一次，并在正确 main 目标上复用同一个 `Localization`。目标选择 CSV 与后端 CSV 分离，避免将没有仿真抓取真值的几何检查误写成性能指标。

**Tech Stack:** Python 3.10、PyBullet 3.2.7、NumPy、OpenCV、PyTorch、现有 Grounding DINO/CNN/geometry 适配器、pytest。

## Global Constraints

- 固定场景、相机、seed、main/generic prompts 和目标选择 IoU `0.25` 不变。
- Grounding DINO 只加载一次，每条 prompt 只定位一次。
- single 固定使用 `data/processed/vlm/cnn_grasp_single_head_deterministic/cnn_grasp_model_seed_42.pt`。
- multi-head 固定使用 `data/processed/vlm/cnn_grasp_multi_head_deterministic/cnn_grasp_model_seed_42.pt`。
- CNN 权重缺失、损坏或架构不匹配时不得换 seed、换目录或回退 geometry。
- segmentation、body ID 和真值框只用于事后评价，不得进入检测器或抓取后端。
- generic prompt 不运行抓取后端。
- 不计算仿真抓取成功率、不使用 Cornell 指标、不进行后端排名。
- 保持 `segmentation_used_as_model_input: false` 和 `physical_grasp_executed: false`。
- `prediction.png` 继续保存 geometry 图以兼容现有输出，同时保存同内容的 `geometry_prediction.png`。
- 外部接口仍引用 Bullet Physics 官方项目：https://github.com/bulletphysics/bullet3；不得将外部代码表述为本项目原创。

---

### Task 1: 纯后端抓取几何审计

**Files:**
- Create: `src/simulation/pybullet/backend_comparison.py`
- Create: `tests/simulation/test_pybullet_backend_comparison.py`

**Interfaces:**
- Consumes: centre-format grasp mapping、目标 bool mask、图像宽高。
- Produces: `BackendGraspEvaluation`、`evaluate_backend_grasp()`、`summarize_backend_rows()`。

- [ ] **Step 1: Write failing geometry-audit tests**

Create literal tests:

```python
mask = np.zeros((20, 30), dtype=bool)
mask[5:15, 8:22] = True
grasp = {
    "center_x": 15.0,
    "center_y": 10.0,
    "width": 10.0,
    "height": 4.0,
    "angle_degrees": 0.0,
}
result = evaluate_backend_grasp(grasp, mask, 30, 20)
assert result.parameters_finite
assert result.positive_size
assert result.center_inside_target_mask
assert result.box_inside_image
```

Also assert:

- centre outside mask produces `center_inside_target_mask=False`;
- a rotated box crossing an image edge produces `box_inside_image=False`;
- NaN or non-positive size records invalid flags without raising;
- mask/image shape mismatch raises `ValueError`;
- summary accepts exactly nine ordered rows
  `duck/cube/sphere × geometry/single/multi_head`;
- summary reports counts only and contains no winner/best-backend field.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_backend_comparison.py -v
```

Expected: import failure because `backend_comparison.py` does not exist.

- [ ] **Step 3: Implement immutable evaluation type**

Add:

```python
@dataclass(frozen=True)
class BackendGraspEvaluation:
    parameters_finite: bool
    positive_size: bool
    center_inside_target_mask: bool
    box_inside_image: bool
    failure_reason: str
```

Implement:

```python
def evaluate_backend_grasp(
    grasp: Mapping[str, float],
    target_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> BackendGraspEvaluation:
```

Read exactly `center_x`, `center_y`, `width`, `height`, and
`angle_degrees`. Use `visualization.grasp_box_points()` only after all
five values are finite and width/height are positive. A point is inside
when `0 <= x <= width - 1` and `0 <= y <= height - 1`. Use the existing
`grasp_center_inside_mask()` for the centre test. Return failure reasons
in fixed precedence:

```text
non_finite_parameters
non_positive_size
center_outside_target_mask
box_outside_image
```

An empty string means all four checks pass.

- [ ] **Step 4: Implement non-ranking summary**

Add:

```python
def summarize_backend_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
```

Validate exact row keys/order from:

```python
EXPECTED = tuple(
    (target, backend)
    for target in ("duck", "cube", "sphere")
    for backend in ("geometry", "single", "multi_head")
)
```

Return:

```python
{
    "protocol": "fixed_three_object_three_backend_diagnostic",
    "backend_result_count": 9,
    "counts_by_backend": {
        backend: {
            "finite_output_count": ...,
            "center_inside_target_mask_count": ...,
            "box_inside_image_count": ...,
        }
    },
    "performance_ranking_computed": False,
    "physical_grasp_executed": False,
}
```

- [ ] **Step 5: Verify GREEN and regression**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_backend_comparison.py -v
conda run -n msc-grasp python -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add src/simulation/pybullet/backend_comparison.py \
  tests/simulation/test_pybullet_backend_comparison.py
git commit -m "feat: audit simulated backend grasp geometry"
```

---

### Task 2: 三后端并列可视化

**Files:**
- Modify: `src/simulation/pybullet/visualization.py`
- Modify: `tests/simulation/test_pybullet_visualization.py`

**Interfaces:**
- Consumes: RGB image and ordered mapping of backend name to already-rendered BGR prediction.
- Produces: `draw_backend_comparison()` returning one BGR `uint8` image.

- [ ] **Step 1: Write failing panel test**

Add:

```python
panels = {
    "geometry": np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8),
    "single": np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8),
    "multi_head": np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8),
}
originals = {name: image.copy() for name, image in panels.items()}
result = draw_backend_comparison(panels)
assert result.dtype == np.uint8
assert result.shape == (64, 180, 3)
assert all(np.array_equal(panels[name], originals[name]) for name in panels)
```

Assert wrong order, missing backend, mismatched shapes, non-BGR shape and
non-`uint8` all raise precise `ValueError`.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_visualization.py \
  -k backend_comparison -v
```

Expected: import failure for `draw_backend_comparison`.

- [ ] **Step 3: Implement fixed three-panel renderer**

Add:

```python
def draw_backend_comparison(
    panels: Mapping[str, np.ndarray],
) -> np.ndarray:
```

Require keys in insertion order:

```python
("geometry", "single", "multi_head")
```

Copy each BGR panel, prepend a 24-pixel black label strip with white
backend text, and concatenate horizontally. Never convert color space or
mutate the supplied panels.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_visualization.py -v
conda run -n msc-grasp python -m pytest -q
```

Then:

```bash
git add src/simulation/pybullet/visualization.py \
  tests/simulation/test_pybullet_visualization.py
git commit -m "feat: visualize simulated backend comparison"
```

---

### Task 3: 扩展固定研究 runner

**Files:**
- Modify: `src/simulation/pybullet/run_multi_object_study.py`
- Modify: `tests/simulation/test_pybullet_multi_object_runner.py`

**Interfaces:**
- Consumes: Task 1 audit helpers、Task 2 panel renderer、existing
  `load_cnn_backend()`/`predict_grasp()`。
- Produces: fixed nine-row backend CSV, backend summary JSON, nine individual
  backend images, three comparison images, explicit weight metadata.

- [ ] **Step 1: Write failing path/config tests**

Extend `MultiObjectStudyConfig` expected defaults:

```python
single_weights = Path(
    "data/processed/vlm/cnn_grasp_single_head_deterministic/"
    "cnn_grasp_model_seed_42.pt"
)
multi_head_weights = Path(
    "data/processed/vlm/cnn_grasp_multi_head_deterministic/"
    "cnn_grasp_model_seed_42.pt"
)
```

Extend `StudyOutputPaths` tests to assert:

```python
paths.backend_results_csv == root / "backend_results.csv"
paths.backend_comparison == root / "backend_comparison.json"
paths.backend_prediction_image("duck", "single") == (
    root / "targets/duck/single_prediction.png"
)
paths.backend_panel_image("duck") == (
    root / "targets/duck/backend_comparison.png"
)
paths.prediction_image("duck") == (
    root / "targets/duck/prediction.png"
)
```

- [ ] **Step 2: Write failing runner integration test**

Extend the fake dependency object with:

```python
load_backend: Callable[[str, Path, str], object]
```

Record loader calls and return `"single-model"` / `"multi-model"`. The
fake predictor returns backend-dependent grasps while recording
`(target_label, backend, model, id(localization))`.

Use the existing fake selection pattern with cube intentionally wrong and
assert:

```python
assert backend_load_calls == [
    ("single", config.single_weights, "cpu"),
    ("multi_head", config.multi_head_weights, "cpu"),
]
assert predict_backends == [
    ("duck", "geometry", None),
    ("duck", "single", "single-model"),
    ("duck", "multi_head", "multi-model"),
    ("sphere", "geometry", None),
    ("sphere", "single", "single-model"),
    ("sphere", "multi_head", "multi-model"),
]
```

For each target, all three calls must contain the same `id(localization)`.
Assert six backend CSV rows because cube is wrong, and no cube backend
images. This integration fixture deliberately does not satisfy the fixed
nine-row summary; runner summary must therefore record
`backend_comparison_complete=False` without aborting target-selection
output.

- [ ] **Step 3: Run runner tests to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_multi_object_runner.py -v
```

Expected: failures for absent weight config, dependency loader and output
paths.

- [ ] **Step 4: Add fixed backend configuration and paths**

Add constants:

```python
BACKENDS = ("geometry", "single", "multi_head")
```

Add the two `Path` fields to `MultiObjectStudyConfig`, the two root paths
to `StudyOutputPaths`, and:

```python
def backend_prediction_image(self, target: str, backend: str) -> Path:
    return self.targets_dir / target / f"{backend}_prediction.png"

def backend_panel_image(self, target: str) -> Path:
    return self.targets_dir / target / "backend_comparison.png"
```

Retain `prediction_image(target)` at its existing path. Clean only these
known outputs before each run.

- [ ] **Step 5: Load both CNNs once**

Extend `MultiObjectStudyDependencies` and `default_dependencies()` with
`load_backend=load_cnn_backend`. Immediately after the detector load,
execute:

```python
backend_models = {
    "geometry": None,
    "single": dependencies.load_backend(
        "single", config.single_weights, config.device
    ),
    "multi_head": dependencies.load_backend(
        "multi_head", config.multi_head_weights, config.device
    ),
}
```

Set `failure_stage` to `backend_model:single` and
`backend_model:multi_head` before the corresponding call. Store exact
paths in metadata.

- [ ] **Step 6: Run and audit each backend**

Replace the single geometry prediction block with a `BACKENDS` loop using
the same `localization` object. For each success:

1. call `dependencies.predict(image_bgr, localization, backend,
   config.device, backend_models[backend])`;
2. call `evaluate_backend_grasp()`;
3. create one flat backend row;
4. draw and save `backend_prediction_image(target, backend)`;
5. if backend is geometry, save the same BGR array to compatibility
   `prediction_image(target)`;
6. after three backends, save `draw_backend_comparison(panels)`.

Catch exceptions inside one backend, append a row with
`backend_failure_reason`, and continue other backends. Do not generate a
panel unless all three individual images exist.

- [ ] **Step 7: Write backend CSV and conditional summary**

Use fixed fields:

```text
target,prompt,backend,weights_path,detection_box,detection_score,
center_x,center_y,width,height,angle_degrees,parameters_finite,
positive_size,center_inside_target_mask,box_inside_image,
backend_failure_reason
```

Always save `backend_results.csv`. If its exact target/backend keys equal
the nine expected rows, call `summarize_backend_rows()` and set
`backend_comparison_complete=True`; otherwise save counts and
`backend_comparison_complete=False`. Never label the incomplete fake or a
real target-selection failure as infrastructure failure.

- [ ] **Step 8: Add CLI weight overrides without changing defaults**

Add:

```text
--single-weights
--multi-head-weights
```

Both are `Path` arguments with the fixed seed 42 defaults. They exist for
reproducibility and testing, not automatic model selection.

- [ ] **Step 9: Verify runner, simulation and full regression**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_multi_object_runner.py -v
conda run -n msc-grasp python -m pytest tests/simulation -v
conda run -n msc-grasp python -m pytest -q
git diff --check
```

- [ ] **Step 10: Commit**

```bash
git add src/simulation/pybullet/run_multi_object_study.py \
  tests/simulation/test_pybullet_multi_object_runner.py
git commit -m "feat: compare three grasp backends in PyBullet"
```

---

### Task 4: 真实 GPU 运行、图像审计和文档

**Files:**
- Modify: `src/simulation/pybullet/README.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/debugging/FAILURE_ANALYSIS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: complete three-backend CLI and ignored experiment outputs.
- Produces: verified diagnostics and reproducible documentation; no new model behavior.

- [ ] **Step 1: Verify weights and regression before study**

Run:

```bash
test -f data/processed/vlm/cnn_grasp_single_head_deterministic/cnn_grasp_model_seed_42.pt
test -f data/processed/vlm/cnn_grasp_multi_head_deterministic/cnn_grasp_model_seed_42.pt
conda run -n msc-grasp python -m pytest -q
git diff --check
```

- [ ] **Step 2: Run frozen study outside GPU-restricted sandbox**

Run:

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py \
  --device cuda \
  --output-dir data/processed/pybullet/multi_object_study
```

Do not change scene, prompts, detector thresholds, CNN weights or seed
after seeing results.

- [ ] **Step 3: Audit numeric artifacts**

Verify:

- target selection still has exactly three main plus one diagnostic row;
- backend CSV has exactly nine stable rows;
- metadata records the two exact seed 42 paths and both truth/execution
  flags false;
- every numeric grasp value and audit flag agrees with recomputation;
- backend comparison explicitly says `performance_ranking_computed=false`;
- root and per-target output paths exist with no stale files.

- [ ] **Step 4: Inspect all generated images**

View geometry, single, multi-head and comparison images for duck, cube and
sphere. Record:

- visible centre, size and orientation differences;
- whether any centre is outside its target;
- whether any rotated rectangle crosses the image;
- visual ambiguity for symmetric sphere/cube;
- no claim that a visually plausible frame is physically executable.

If implementation output is wrong, use `superpowers:systematic-debugging`.
Do not alter frozen inputs to improve a model outcome.

- [ ] **Step 5: Update documentation from verified evidence**

Update README with the two fixed seed 42 paths, three-backend command and
output tree. Update status, project structure, failure analysis and
worklog with exact test count, numeric audit and image observations. State
that no ground-truth grasp ranking or physical execution occurred.

- [ ] **Step 6: Final verification and commit**

Run:

```bash
git diff --check
conda run -n msc-grasp python -m pytest -q
git status --short
```

Commit only source/tests/docs, never `data/` or dissertation build files:

```bash
git add src/simulation/pybullet/README.md \
  docs/agent/CURRENT_STATUS.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/debugging/FAILURE_ANALYSIS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record PyBullet backend comparison"
```
