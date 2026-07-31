# PyBullet Multi-Object Target Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在一个固定的鸭、方块、球体和 Panda 场景中，运行三条明确 prompt 与一条 generic prompt，使用 segmentation 真值评价目标选择，并对正确目标生成二维几何抓取框。

**Architecture:** 保留现有单物体 `run_pilot.py`，向 `scene.py` 增加向后兼容的附加物体配置；新建纯评价模块 `target_selection.py`；新建独立 CLI `run_multi_object_study.py` 负责一次渲染、一次模型加载、四条 prompt、CSV/JSON 和可视化。segmentation 只流入评价函数，不流入 detector 或 grasp adapter。

**Tech Stack:** Python 3.10、PyBullet 3.2.7、NumPy、OpenCV、PyTorch、Transformers、pytest、现有 `src.simulation.pybullet.perception` 和 `src.vlm` 接口。

## Global Constraints

- 当前实验固定为一个三物体场景，不增加随机布局、相机、纹理或光照。
- 主目标固定为黄色鸭、红色方块和绿色球体；Panda 只作为 distractor。
- 主 prompt 固定为 `yellow rubber duck`、`red cube` 和 `green sphere`；`small object` 只作 diagnostic。
- segmentation、body ID 和真值框不得输入 Grounding DINO 或抓取后端。
- 目标选择门控阈值为 bbox IoU `0.25`，必须同时报告原始 IoU，并明确它不是 Cornell 抓取指标。
- 只有 `correct_target=True` 的 main 行调用现有 `geometry` 后端。
- 不训练 CNN，不做深度反投影、IK、碰撞规划、夹爪闭合或抓取成功判定。
- 所有 metadata 保持 `segmentation_used_as_model_input: false` 和 `physical_grasp_executed: false`。
- 请求 CUDA 但不可用时直接失败，不静默回退。
- 保留 `run_pilot.py` 的现有 CLI、单物体默认值和固定输出兼容性。
- 外部代码必须标明原作者、项目或论文和可访问链接；PyBullet 来源为 Erwin Coumans、Yunfei Bai 等维护的 Bullet Physics 项目：https://github.com/bulletphysics/bullet3。

---

### Task 1: 向后兼容的多物体场景

**Files:**
- Modify: `src/simulation/pybullet/scene.py`
- Modify: `tests/simulation/test_pybullet_smoke.py`

**Interfaces:**
- Consumes: existing `SceneConfig`, `SceneBodies`, `PyBulletScene.connect/step/close`.
- Produces: `SceneObjectConfig`, new `SceneConfig.object_name`, `SceneConfig.object_rgba`, `SceneConfig.additional_objects`, `SceneBodies.additional_objects`, `PyBulletScene.object_body_ids`, `PyBulletScene.object_poses()`.

- [ ] **Step 1: Write failing configuration and DIRECT tests**

Append tests that construct:

```python
cube = SceneObjectConfig(
    name="cube",
    urdf="cube_small.urdf",
    position=(0.48, 0.0, 0.66),
    yaw_degrees=30.0,
    rgba=(0.9, 0.1, 0.1, 1.0),
)
sphere = SceneObjectConfig(
    name="sphere",
    urdf="sphere_small.urdf",
    position=(0.52, 0.18, 0.67),
    yaw_degrees=0.0,
    rgba=(0.1, 0.8, 0.1, 1.0),
)
scene = PyBulletScene(
    SceneConfig(
        gui=False,
        object_name="duck",
        object_rgba=(1.0, 0.8, 0.0, 1.0),
        object_position=(0.52, -0.18, 0.67),
        additional_objects=(cube, sphere),
    )
).connect()
```

Assert:

```python
assert set(scene.object_body_ids) == {"duck", "cube", "sphere"}
assert len(set(scene.object_body_ids.values())) == 3
scene.step(60)
frame = capture_camera_frame(
    scene.client_id,
    CameraConfig(),
    scene.renderer,
)
for body_id in scene.object_body_ids.values():
    mask = (
        (frame.segmentation >= 0)
        & ((frame.segmentation & ((1 << 24) - 1)) == body_id)
    )
    assert np.any(mask)
poses = scene.object_poses()
assert set(poses) == {"duck", "cube", "sphere"}
```

Add a separate test that duplicate name `duck` in `additional_objects` raises
`ValueError("duplicate scene object name: duck")` and leaves no connection.
Existing single-object smoke tests remain unchanged and must still pass.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_smoke.py -v
```

Expected: collection FAIL because `SceneObjectConfig` is missing.

- [ ] **Step 3: Implement scene object types and validation**

Add:

```python
@dataclass(frozen=True)
class SceneObjectConfig:
    name: str
    urdf: str
    position: tuple[float, float, float]
    yaw_degrees: float = 0.0
    rgba: tuple[float, float, float, float] | None = None
```

Extend `SceneConfig` with:

```python
object_name: str = "target_object"
object_rgba: tuple[float, float, float, float] | None = None
additional_objects: tuple[SceneObjectConfig, ...] = ()
```

Extend `SceneBodies` with:

```python
additional_objects: tuple[tuple[str, int], ...] = ()
```

Before connecting, validate:

- every name is non-empty;
- names are unique, including `object_name`;
- RGBA has four finite values in `[0, 1]`;
- every URDF is relative, resolves inside `pybullet_data`, and exists.

Keep the existing default object fields and `target_object` body ID unchanged.

- [ ] **Step 4: Load and color additional bodies**

Refactor the existing resource resolver to accept a URDF string:

```python
def _resolve_object_urdf(self, urdf: str) -> Path:
```

Add a private `_load_scene_object(config) -> int` that uses `loadURDF` and,
when RGBA is not `None`, calls:

```python
p.changeVisualShape(
    body_id,
    -1,
    rgbaColor=config.rgba,
    physicsClientId=self.client_id,
)
```

Load the legacy target first, then additional objects in tuple order. Implement:

```python
@property
def object_body_ids(self) -> dict[str, int]:
    return {
        self.config.object_name: self.bodies.target_object,
        **dict(self.bodies.additional_objects),
    }

def object_poses(self) -> dict[str, dict[str, tuple[float, ...]]]:
    ...
```

`object_poses` calls `getBasePositionAndOrientation` with the stored client ID
and returns JSON-safe float tuples under `position` and `orientation`.

- [ ] **Step 5: Verify GREEN and regression**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_smoke.py \
  tests/simulation/test_pybullet_runner.py -v
conda run -n msc-grasp python -m pytest -q
```

Expected: all scene tests and the existing single-object runner pass.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/pybullet/scene.py \
  tests/simulation/test_pybullet_smoke.py
git commit -m "feat: support fixed multi-object PyBullet scenes"
```

---

### Task 2: 纯目标选择评价

**Files:**
- Create: `src/simulation/pybullet/target_selection.py`
- Create: `tests/simulation/test_pybullet_target_selection.py`

**Interfaces:**
- Consumes: segmentation `np.ndarray`, body IDs, inclusive `(x1,y1,x2,y2)` boxes, centre-format grasp dictionaries.
- Produces: `TargetSelectionEvaluation`, `segmentation_mask_for_body`, `mask_to_box`, `box_iou`, `evaluate_target_selection`, `grasp_center_inside_mask`, `summarize_target_rows`.

- [ ] **Step 1: Write failing mask, box and IoU tests**

Use literal fixtures:

```python
segmentation = np.array(
    [
        [-1, 3, 3, 7],
        [-1, 3, 3, 7],
    ],
    dtype=np.int32,
)
mask = segmentation_mask_for_body(segmentation, 3)
assert mask.tolist() == [
    [False, True, True, False],
    [False, True, True, False],
]
assert mask_to_box(mask) == (1, 0, 2, 1)
assert box_iou((0, 0, 1, 1), (1, 0, 2, 1)) == pytest.approx(1 / 3)
```

Also use a background-only array and assert body ID `((1 << 24) - 1)` does not
match any pixel. Assert empty masks raise `ValueError("mask has no visible pixels")`.

- [ ] **Step 2: Run mask tests to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_target_selection.py -v
```

Expected: import FAIL because `target_selection.py` is missing.

- [ ] **Step 3: Implement mask and inclusive-box geometry**

Implement exact public signatures:

```python
def segmentation_mask_for_body(
    segmentation: np.ndarray,
    body_id: int,
) -> np.ndarray:

def mask_to_box(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:

def box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
```

Use inclusive width and height:

```python
width = x2 - x1 + 1
height = y2 - y1 + 1
```

Reject malformed, non-finite, negative-area boxes.

- [ ] **Step 4: Write failing target-selection tests**

Create boxes:

```python
entities = {
    "duck": (0, 0, 9, 9),
    "cube": (20, 0, 29, 9),
    "sphere": (40, 0, 49, 9),
    "robot": (60, 0, 79, 19),
}
```

Assert:

- prediction `(1,1,8,8)` requested `duck` is correct;
- same prediction requested `cube` has `best_matching_target=="duck"` and is false;
- prediction with requested IoU below `0.25` is false;
- equal highest IoUs set `failure_reason=="ambiguous_match"`;
- `None` detection creates `failure_reason=="no_detection"`.

Test grasp centre:

```python
mask = np.zeros((10, 10), dtype=bool)
mask[2:8, 3:9] = True
assert grasp_center_inside_mask({"center_x": 4.0, "center_y": 3.0}, mask)
assert not grasp_center_inside_mask({"center_x": 1.0, "center_y": 1.0}, mask)
assert not grasp_center_inside_mask({"center_x": 30.0, "center_y": 30.0}, mask)
```

- [ ] **Step 5: Implement evaluation and summary**

Add:

```python
@dataclass(frozen=True)
class TargetSelectionEvaluation:
    requested_target: str
    requested_target_iou: float
    best_matching_target: str | None
    best_iou: float
    correct_target: bool
    iou_threshold: float
    failure_reason: str
    entity_ious: dict[str, float]
```

`evaluate_target_selection` accepts `predicted_box | None`, validates the
requested name, computes all entity IoUs, treats `np.isclose` equal maxima as
ambiguous, and applies the `0.25` gate.

`summarize_target_rows(rows)` uses only `result_role=="main"` and returns:

```python
{
    "protocol": "fixed_three_object_prompt_selection_pilot",
    "main_target_count": 3,
    "correct_target_count": ...,
    "target_selection_rate": ...,
    "mean_requested_target_iou": ...,
    "generic_diagnostic": {...},
    "physical_grasp_executed": False,
}
```

Reject a main set whose requested targets are not exactly
`{"duck","cube","sphere"}`.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_target_selection.py -v
conda run -n msc-grasp python -m pytest -q
```

Then:

```bash
git add src/simulation/pybullet/target_selection.py \
  tests/simulation/test_pybullet_target_selection.py
git commit -m "feat: evaluate simulated target selection"
```

---

### Task 3: 多物体研究 runner、可视化和产物

**Files:**
- Modify: `src/simulation/pybullet/visualization.py`
- Create: `src/simulation/pybullet/run_multi_object_study.py`
- Create: `tests/simulation/test_pybullet_multi_object_runner.py`
- Modify: `tests/simulation/test_pybullet_visualization.py`

**Interfaces:**
- Consumes: Task 1 scene interfaces, Task 2 evaluation, existing `CameraFrame`, `Localization`, `PilotPrediction`, `load_grounding_dino`, `localize_object`, `predict_grasp`.
- Produces: `draw_ground_truth_boxes`, `draw_target_evaluation`, `MultiObjectStudyConfig`, `StudyPrompt`, `StudyOutputPaths`, `run_multi_object_study`, CLI `main`.

- [ ] **Step 1: Write failing visualization tests**

Test `draw_ground_truth_boxes` on a black RGB image with literal boxes for
`duck`, `cube`, `sphere`, and `robot`; assert input is not mutated, returned
image is BGR uint8, and pixels on each rectangle use four distinct colors.

Test `draw_target_evaluation` with requested `duck`, best `robot`, and a
prediction box; assert both requested-target and best-match boxes are drawn
and the output is non-empty.

- [ ] **Step 2: Run visualization tests to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_visualization.py -v
```

Expected: import FAIL for the two new drawing functions.

- [ ] **Step 3: Implement minimal evaluation drawings**

Add:

```python
def draw_ground_truth_boxes(
    rgb: np.ndarray,
    boxes: Mapping[str, tuple[int, int, int, int]],
) -> np.ndarray:

def draw_target_evaluation(
    rgb: np.ndarray,
    requested_target: str,
    prompt: str,
    detection_box: tuple[int, int, int, int] | None,
    ground_truth_boxes: Mapping[str, tuple[int, int, int, int]],
    best_matching_target: str | None,
    score: float | None,
) -> np.ndarray:
```

Use fixed BGR colors: duck yellow, cube red, sphere green, robot magenta,
detection blue. Conversion from RGB to BGR occurs once and functions never
mutate inputs.

- [ ] **Step 4: Write failing runner integration test**

Create a fake scene that exposes:

```python
object_body_ids = {"duck": 3, "cube": 4, "sphere": 5}
bodies.robot = 2
object_poses() -> poses for all three
```

Create one real `CameraFrame` whose segmentation contains four disjoint
rectangles for body IDs 2–5. Fake localization returns:

- duck prompt → duck box;
- cube prompt → duck box (intentional wrong target);
- sphere prompt → sphere box;
- generic prompt → robot box.

Fake predict returns a finite geometry grasp for duck and sphere. Record
calls in lists. Run:

```python
summary = run_multi_object_study(
    MultiObjectStudyConfig(
        output_dir=tmp_path,
        width=80,
        height=60,
        device="cpu",
    ),
    dependencies=fakes,
)
```

Assert:

```python
assert summary["main_target_count"] == 3
assert summary["correct_target_count"] == 2
assert summary["generic_diagnostic"]["best_matching_target"] == "robot"
assert detector_load_count == 1
assert scene_capture_count == 1
assert predict_requested_targets == ["duck", "sphere"]
```

Assert CSV has four rows with stable order and output files exist. Assert cube
has no prediction image. Assert metadata flags both segmentation/model-input
and physical grasp as false. The fake detector/localizer signatures must not
accept segmentation so accidental truth injection fails the test.

- [ ] **Step 5: Run runner test to verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_multi_object_runner.py -v
```

Expected: import FAIL because `run_multi_object_study.py` is missing.

- [ ] **Step 6: Implement fixed study configuration and output paths**

Define constants:

```python
MAIN_PROMPTS = (
    StudyPrompt("main", "duck", "yellow rubber duck"),
    StudyPrompt("main", "cube", "red cube"),
    StudyPrompt("main", "sphere", "green sphere"),
)
DIAGNOSTIC_PROMPTS = (
    StudyPrompt("diagnostic", "generic", "small object"),
)
```

`fixed_scene_config` returns the exact URDF, RGBA and positions from the spec.
`StudyOutputPaths` builds all fixed root and target image paths. Before a run,
unlink only those known generated paths.

- [ ] **Step 7: Implement one-frame, one-model runner**

Implement `run_multi_object_study`:

1. create/connect scene in `try/finally`;
2. step 60 and capture one frame;
3. save RGB/depth/depth visualization/segmentation;
4. build entity IDs for duck/cube/sphere/robot;
5. build non-empty masks and boxes, then save `ground_truth_boxes.png`;
6. load Grounding DINO once;
7. loop main prompts then diagnostic prompt;
8. localize, evaluate, and save evaluation image;
9. for correct main rows only, call geometry prediction and save prediction;
10. save four-row CSV, summary JSON and metadata JSON;
11. close scene in `finally`.

Infrastructure exceptions write failure metadata and return nonzero CLI.
Individual no-detection/invalid/wrong-target outcomes remain experiment rows
and do not abort later prompts.

- [ ] **Step 8: Implement CSV, JSON and CLI**

CLI arguments:

```text
--gui
--device {cpu,cuda}
--model-id
--output-dir
--seed
--width
--height
--box-threshold
--text-threshold
```

Default output:

```text
data/processed/pybullet/multi_object_study
```

Add the same repository-root insertion pattern as `run_pilot.py`. Verify:

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py --help
```

- [ ] **Step 9: Verify tests and commit**

Run:

```bash
conda run -n msc-grasp python -m pytest tests/simulation -v
conda run -n msc-grasp python -m pytest -q
git diff --check
```

Then:

```bash
git add src/simulation/pybullet/visualization.py \
  src/simulation/pybullet/run_multi_object_study.py \
  tests/simulation/test_pybullet_visualization.py \
  tests/simulation/test_pybullet_multi_object_runner.py
git commit -m "feat: run fixed PyBullet target selection study"
```

---

### Task 4: 真实 GPU 研究运行、审计和文档

**Files:**
- Create: `src/simulation/pybullet/README.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/debugging/FAILURE_ANALYSIS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: complete multi-object CLI and saved artifacts.
- Produces: verified experiment record and reproducible commands; no new model behavior.

- [ ] **Step 1: Run full regression before the model study**

Run:

```bash
git diff --check
conda run -n msc-grasp python -m pytest -q
```

Expected: all tests pass and only known dissertation build artifacts remain
untracked.

- [ ] **Step 2: Run the real study outside the GPU-restricted sandbox**

Run:

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py \
  --device cuda \
  --output-dir data/processed/pybullet/multi_object_study
```

Do not change prompts, threshold, scene or camera after seeing results.

- [ ] **Step 3: Audit numeric artifacts**

Verify:

- RGB is 640×480 and all depth values are finite;
- four unique entity masks and in-bounds boxes exist;
- CSV contains exactly three main and one diagnostic row;
- every reported IoU is finite and in `[0,1]`;
- summary equals recomputation from CSV;
- metadata flags both truth-input and physical execution false;
- all listed output paths exist except prediction images for failed targets.

- [ ] **Step 4: Inspect every saved image**

View the RGB, segmentation, truth boxes, all four localization images and
each generated prediction. Record only visible facts:

- whether each main prompt selected the requested target;
- whether generic selected Panda or another entity;
- whether each grasp center lies on the requested object;
- any occlusion, channel, box or color error.

If code/output is wrong, invoke `superpowers:systematic-debugging`. Model
failures are results and must not be fixed by injecting truth.

- [ ] **Step 5: Write module README**

Document:

- PyBullet 3.2.7 and official Bullet source attribution;
- single-object and multi-object commands;
- fixed entities and prompts;
- target-selection metric and the non-standard `0.25` gate;
- segmentation evaluation-only boundary;
- output files;
- sandbox/CUDA behavior;
- no-detection and wrong-target interpretation;
- explicit non-goals and the later depth/IK gate.

- [ ] **Step 6: Update records from verified output**

Update `CURRENT_STATUS.md`, `PROJECT_STRUCTURE.md`, `FAILURE_ANALYSIS.md` and
`WORKLOG.md` with exact test count, saved values and output paths. If fewer
than 3/3 targets pass, record the failure and state that the visual pipeline
has not passed the grasp-execution gate.

- [ ] **Step 7: Final verification and commit**

Run:

```bash
git diff --check
conda run -n msc-grasp python -m pytest -q
git status --short
```

Commit only source, tests and docs:

```bash
git add src/simulation/pybullet/README.md \
  docs/agent/CURRENT_STATUS.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/debugging/FAILURE_ANALYSIS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record multi-object target study"
```

Do not add `data/` or `uog_dissertation_outline/l4proj.blg`,
`uog_dissertation_outline/l4proj.log`, or
`uog_dissertation_outline/l4proj.pdf`.
