# PyBullet Perception Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可重复运行的 PyBullet pilot，把固定虚拟相机图像接入现有 Grounding DINO 与二维抓取后端，并保存可审计产物。

**Architecture:** `scene.py` 独占 PyBullet 连接和 URDF 生命周期，`camera.py` 负责相机配置与数组转换，`perception.py` 仅适配现有 VLM/CNN 接口，`visualization.py` 负责纯 OpenCV 绘制，`run_pilot.py` 负责编排和落盘。普通测试用依赖注入或 monkeypatch 隔离大模型，单独的显式命令才运行真实 Grounding DINO。

**Tech Stack:** Python 3.11、PyBullet、NumPy、OpenCV、PyTorch、Transformers、pytest、现有 `src.vlm` 模块。

## Global Constraints

- 代码目录固定为 `src/simulation/pybullet/`，不得创建会遮蔽第三方包的 `src/pybullet/`。
- 第一阶段只验证感知与可视化；不得加入 IK、轨迹规划、夹爪动作或物理抓取执行。
- 默认场景资源固定为 `plane.urdf`、`table/table.urdf`、`franka_panda/panda.urdf` 和 `duck_vhacd.urdf`。
- 默认 DIRECT 模式使用 `ER_TINY_RENDERER`；`--gui` 模式使用 `ER_BULLET_HARDWARE_OPENGL`。
- 默认相机分辨率为 `640×480`；深度保存为米制线性深度。
- segmentation 只能用于可见性检查和诊断，不得输入 Grounding DINO 或抓取后端。
- Grounding DINO checkpoint 固定默认值为 `IDEA-Research/grounding-dino-tiny`，prompt 为 `small object`，box/text threshold 均为 `0.25`。
- `geometry` 是默认抓取后端；它必须复用 `predict_grasp_with_vlm_box(..., expand_ratio=0.10, use_box_fallback=True)`。
- CNN 后端必须复用现有模型、`crop_to_tensor`、`predict_from_crop` 和既有 state-dict 加载约定，不得复制模型实现。
- 请求 `cuda` 但 `torch.cuda.is_available()` 为假时必须报错，不得静默改用 CPU。
- 所有 PyBullet API 调用显式传入 `physicsClientId`。
- 外部或改编代码必须在相关源文件中标注官方来源链接和改编范围。
- 生成数据只写入已被 Git 忽略的 `data/processed/pybullet/pilot/`。
- 未经真实运行验证，不得把 Grounding DINO 检测成功或 CNN 仿真域质量写入状态或论文。

---

### Task 1: 安装依赖并实现相机数据模型

**Files:**
- Create: `src/simulation/__init__.py`
- Create: `src/simulation/pybullet/__init__.py`
- Create: `src/simulation/pybullet/camera.py`
- Create: `tests/simulation/test_pybullet_camera.py`

**Interfaces:**
- Consumes: PyBullet `computeViewMatrix`、`computeProjectionMatrixFOV` 和 `getCameraImage`。
- Produces: `CameraConfig.validate() -> None`、`CameraFrame`、`linearize_depth(depth_buffer, near, far) -> np.ndarray`、`capture_camera_frame(client_id, config, renderer) -> CameraFrame`。

- [ ] **Step 1: 记录安装前状态并安装 PyBullet**

Run:

```bash
conda run -n msc-grasp python -c "import pybullet"
```

Expected before installation: FAIL with `ModuleNotFoundError: No module named 'pybullet'`.

Run:

```bash
conda run -n msc-grasp python -m pip install pybullet
conda run -n msc-grasp python -c "import pybullet as p; print(p.getAPIVersion())"
conda run -n msc-grasp python -m pip show pybullet
```

Expected: import succeeds; save the exact installed version for Task 6 documentation.

- [ ] **Step 2: Write the failing camera tests**

Create package markers and the test below:

```python
import numpy as np
import pytest

from src.simulation.pybullet.camera import CameraConfig, linearize_depth


def test_camera_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="width and height"):
        CameraConfig(width=0).validate()
    with pytest.raises(ValueError, match="near.*far"):
        CameraConfig(near=1.0, far=0.5).validate()
    with pytest.raises(ValueError, match="fov"):
        CameraConfig(fov_degrees=180.0).validate()


def test_linearize_depth_maps_clip_planes_and_is_monotonic() -> None:
    buffer = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    depth = linearize_depth(buffer, near=0.1, far=10.0)

    assert depth[0] == pytest.approx(0.1)
    assert depth[-1] == pytest.approx(10.0, rel=1e-5)
    assert np.all(np.isfinite(depth))
    assert np.all(np.diff(depth) > 0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_camera.py -v
```

Expected: FAIL during import because `camera.py` does not exist.

- [ ] **Step 4: Implement the minimal camera model and conversion**

In `camera.py`, include an attribution comment linking to the official
PyBullet quickstart PDF/API repository for the camera-buffer formula, then implement:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    eye: tuple[float, float, float] = (1.0, 0.0, 1.15)
    target: tuple[float, float, float] = (0.5, 0.0, 0.62)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_degrees: float = 55.0
    near: float = 0.05
    far: float = 3.0

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 < self.near < self.far:
            raise ValueError("near must be positive and smaller than far")
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("fov must be between 0 and 180 degrees")


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]


def linearize_depth(
    depth_buffer: np.ndarray, near: float, far: float
) -> np.ndarray:
    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    buffer = np.asarray(depth_buffer, dtype=np.float32)
    if not np.all(np.isfinite(buffer)):
        raise ValueError("depth buffer contains non-finite values")
    depth = far * near / (far - (far - near) * buffer)
    if not np.all(np.isfinite(depth)):
        raise ValueError("linear depth contains non-finite values")
    return depth.astype(np.float32)
```

Then implement `capture_camera_frame` with lazy `import pybullet as p`, reshape
RGBA to `(height, width, 4)`, keep `rgba[..., :3].copy()` as RGB, reshape depth
and segmentation to `(height, width)`, linearize depth, and reject wrong shapes
or non-finite RGB/depth.

- [ ] **Step 5: Add channel/shape validation tests and run them**

Append a monkeypatched `getCameraImage` test that supplies a 2×1 RGBA image
`[[[10, 20, 30, 255], [40, 50, 60, 255]]]` and asserts RGB equals
`[[[10, 20, 30], [40, 50, 60]]]`, depth shape is `(1, 2)`, and segmentation
shape is `(1, 2)`.

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_camera.py -v
```

Expected: all camera tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simulation tests/simulation/test_pybullet_camera.py
git commit -m "feat: add PyBullet camera capture"
```

---

### Task 2: 实现场景生命周期和 DIRECT 冒烟测试

**Files:**
- Create: `src/simulation/pybullet/scene.py`
- Create: `tests/simulation/test_pybullet_smoke.py`

**Interfaces:**
- Consumes: `CameraConfig` and `capture_camera_frame` from Task 1.
- Produces: `SceneConfig`、`SceneBodies`、`PyBulletScene.connect() -> PyBulletScene`、`step(count=1)`、`close()`、context-manager methods。

- [ ] **Step 1: Write the failing scene smoke test**

```python
import numpy as np

from src.simulation.pybullet.camera import CameraConfig, capture_camera_frame
from src.simulation.pybullet.scene import PyBulletScene, SceneConfig


def test_direct_scene_contains_visible_object_and_closes() -> None:
    scene = PyBulletScene(SceneConfig(gui=False, seed=42)).connect()
    try:
        assert scene.client_id >= 0
        assert scene.bodies.robot >= 0
        assert scene.bodies.table >= 0
        assert scene.bodies.target_object >= 0
        scene.step(10)
        frame = capture_camera_frame(
            scene.client_id,
            CameraConfig(width=160, height=120),
            scene.renderer,
        )
        assert frame.rgb.shape == (120, 160, 3)
        assert frame.depth_m.shape == (120, 160)
        assert np.all(np.isfinite(frame.depth_m))
        object_mask = (frame.segmentation & ((1 << 24) - 1)) == scene.bodies.target_object
        assert np.any(object_mask)
    finally:
        client_id = scene.client_id
        scene.close()

    assert not scene.is_connected
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_smoke.py -v
```

Expected: FAIL because `scene.py` does not exist.

- [ ] **Step 3: Implement deterministic scene loading**

Implement these exact public models:

```python
@dataclass(frozen=True)
class SceneConfig:
    gui: bool = False
    seed: int = 42
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    object_urdf: str = "duck_vhacd.urdf"
    object_position: tuple[float, float, float] = (0.55, 0.0, 0.66)
    object_yaw_degrees: float = 20.0


@dataclass(frozen=True)
class SceneBodies:
    plane: int
    table: int
    robot: int
    target_object: int
```

`connect()` must:

1. choose `p.GUI` or `p.DIRECT`;
2. fail if returned client ID is negative;
3. set `pybullet_data.getDataPath()` as search path;
4. reset simulation, gravity, timestep, and NumPy seed;
5. load the four fixed URDFs with explicit `physicsClientId`;
6. select `p.ER_BULLET_HARDWARE_OPENGL` for GUI, otherwise
   `p.ER_TINY_RENDERER`;
7. save IDs in `SceneBodies`;
8. on any exception, disconnect before re-raising.

Resolve `object_urdf` against `Path(pybullet_data.getDataPath()).resolve()`.
Reject absolute paths and reject any resolved path for which
`candidate.is_relative_to(data_root)` is false. Raise
`ValueError("object_urdf must resolve inside pybullet_data")`; this enforces
the design rule that the pilot only loads packaged test resources.

Every `stepSimulation`, `isConnected`, `resetSimulation`, `loadURDF`, and
`disconnect` call must use the stored client ID. Add the official PyBullet
repository URL at the top and state that scene setup follows the public API,
not copied third-party code.

- [ ] **Step 4: Run scene and camera tests**

Run:

```bash
conda run -n msc-grasp pytest \
  tests/simulation/test_pybullet_camera.py \
  tests/simulation/test_pybullet_smoke.py -v
```

Expected: PASS, including object visibility in segmentation.

- [ ] **Step 5: Commit**

```bash
git add src/simulation/pybullet/scene.py tests/simulation/test_pybullet_smoke.py
git commit -m "feat: add deterministic PyBullet scene"
```

---

### Task 3: 实现预测绘制与输出文件保存

**Files:**
- Create: `src/simulation/pybullet/visualization.py`
- Create: `tests/simulation/test_pybullet_visualization.py`

**Interfaces:**
- Consumes: RGB `np.ndarray`、定位框 `(x1, y1, x2, y2)`、中心格式抓取预测。
- Produces: `validate_detection_box`、`grasp_box_points`、`draw_prediction`、`depth_to_uint8`、`segmentation_to_bgr`。

- [ ] **Step 1: Write failing geometry and color tests**

```python
import numpy as np
import pytest

from src.simulation.pybullet.visualization import (
    depth_to_uint8,
    draw_prediction,
    grasp_box_points,
)


def test_grasp_box_points_for_axis_aligned_rectangle() -> None:
    points = grasp_box_points(
        center_x=50.0,
        center_y=40.0,
        width=20.0,
        height=10.0,
        angle_degrees=0.0,
    )
    assert points.shape == (4, 2)
    assert set(map(tuple, points.astype(int))) == {
        (40, 35), (60, 35), (60, 45), (40, 45)
    }


def test_visualization_rejects_non_finite_or_negative_grasp() -> None:
    with pytest.raises(ValueError, match="finite"):
        grasp_box_points(np.nan, 1.0, 2.0, 3.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        grasp_box_points(1.0, 1.0, -2.0, 3.0, 0.0)


def test_depth_visualization_has_uint8_range() -> None:
    result = depth_to_uint8(
        np.array([[0.1, 0.5, 1.0]], dtype=np.float32),
        near=0.1,
        far=1.0,
    )
    assert result.dtype == np.uint8
    assert result.tolist() == [[0, 113, 255]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_visualization.py -v
```

Expected: FAIL because `visualization.py` does not exist.

- [ ] **Step 3: Implement validation and OpenCV drawing**

Implement `grasp_box_points` using `cv2.boxPoints`, after validating all five
values are finite and width/height are positive. Implement:

```python
def draw_prediction(
    rgb: np.ndarray,
    localization_box: tuple[float, float, float, float],
    grasp: Mapping[str, float],
    prompt: str,
    confidence: float,
    backend: str,
) -> np.ndarray:
```

It must copy the input, convert RGB→BGR exactly once, draw:

- localization box in yellow BGR `(0, 255, 255)`;
- grasp rectangle in blue BGR `(255, 0, 0)`;
- center in green;
- direction line along half the predicted width;
- text containing prompt, confidence, and backend.

`validate_detection_box` must reject non-finite coordinates, zero area, and
coordinates outside `[0, width-1] × [0, height-1]`. `depth_to_uint8` clips
linearly between near/far. `segmentation_to_bgr` must deterministically map
integer body IDs to colors and render background ID `-1` as black.

- [ ] **Step 4: Add an immutability/drawing test and run**

Append:

```python
def test_draw_prediction_returns_bgr_without_mutating_rgb() -> None:
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    original = rgb.copy()
    drawn = draw_prediction(
        rgb,
        (20.0, 20.0, 100.0, 80.0),
        {
            "center_x": 60.0,
            "center_y": 50.0,
            "width": 30.0,
            "height": 12.0,
            "angle_degrees": 30.0,
        },
        prompt="small object",
        confidence=0.8,
        backend="geometry",
    )
    assert np.array_equal(rgb, original)
    assert drawn.shape == rgb.shape
    assert np.any(drawn != 0)
```

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_visualization.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simulation/pybullet/visualization.py \
  tests/simulation/test_pybullet_visualization.py
git commit -m "feat: add PyBullet pilot visualizations"
```

---

### Task 4: 适配现有 Grounding DINO、几何和 CNN 后端

**Files:**
- Create: `src/simulation/pybullet/perception.py`
- Create: `tests/simulation/test_pybullet_perception.py`

**Interfaces:**
- Consumes: existing `run_grounding_dino_on_image`、`parse_vlm_box` 兼容坐标规则、`predict_grasp_with_vlm_box`、`create_model`、`_load_state_dict`、`crop_to_tensor`、`predict_from_crop`。
- Produces: `Localization`、`PilotPrediction`、`validate_device`、`load_grounding_dino`、`localize_object`、`predict_grasp`。

- [ ] **Step 1: Write failing adapter tests**

```python
from pathlib import Path
import numpy as np
import pytest

from src.simulation.pybullet import perception


def test_cuda_request_never_silently_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(perception.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        perception.validate_device("cuda")
    assert perception.validate_device("cpu") == "cpu"


def test_geometry_backend_reuses_existing_predictor(monkeypatch) -> None:
    called = {}

    def fake_predict(image, box, expand_ratio, use_box_fallback):
        called.update(
            box=box,
            expand_ratio=expand_ratio,
            use_box_fallback=use_box_fallback,
        )
        return (
            {
                "center_x": 30.0,
                "center_y": 40.0,
                "width": 20.0,
                "height": 10.0,
                "angle_degrees": 15.0,
            },
            np.zeros(image.shape[:2], dtype=np.uint8),
            box,
            "",
        )

    monkeypatch.setattr(
        perception, "predict_grasp_with_vlm_box", fake_predict
    )
    result = perception.predict_grasp(
        np.zeros((100, 120, 3), dtype=np.uint8),
        perception.Localization((10, 20, 80, 90), 0.9, "object"),
        backend="geometry",
        device="cpu",
        model=None,
    )
    assert result.grasp["center_x"] == 30.0
    assert called == {
        "box": (10, 20, 80, 90),
        "expand_ratio": 0.10,
        "use_box_fallback": True,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_perception.py -v
```

Expected: FAIL because `perception.py` does not exist.

- [ ] **Step 3: Implement public prediction types and localization adapter**

Implement:

```python
@dataclass(frozen=True)
class Localization:
    box: tuple[int, int, int, int]
    score: float
    label: str


@dataclass(frozen=True)
class PilotPrediction:
    localization: Localization
    backend: str
    grasp: dict[str, float]
    failure_reason: str = ""
```

`load_grounding_dino(model_id, device)` must validate the device, create
`AutoProcessor` and `AutoModelForZeroShotObjectDetection`, call
`model.to(device)` and `model.eval()`, and return `(processor, model)`.

`localize_object(rgb_path, prompt, processor, model, device,
box_threshold=0.25, text_threshold=0.25)` must call the existing single-image
function. If it returns `None`, return `None`; otherwise round/clip the box to
the image dimensions and return `Localization`. Use PIL only to obtain image
dimensions; do not use segmentation.

- [ ] **Step 4: Implement geometry and CNN dispatch**

`predict_grasp` must receive a BGR image because the existing geometry/CNN
preprocessing expects BGR. For `geometry`, call exactly:

```python
prediction, _, _, failure_reason = predict_grasp_with_vlm_box(
    image_bgr,
    localization.box,
    expand_ratio=0.10,
    use_box_fallback=True,
)
```

For `single` or `multi_head`:

1. require an already loaded model;
2. crop with `image_bgr[top:bottom + 1, left:right + 1]`;
3. call `crop_to_tensor(crop)`;
4. call `predict_from_crop(model, tensor, crop_w, crop_h, device)`;
5. add `left/top` to predicted center coordinates so output returns to full
   image coordinates.

Implement `load_cnn_backend(backend, weights_path, device)` using existing
`create_model` and `_load_state_dict`; load with
`torch.load(weights_path, map_location=device, weights_only=True)`. Reject
missing weights, unsupported backend, and state-dict mismatch with an error
that includes both backend and path.

- [ ] **Step 5: Add CNN coordinate and state-dict tests**

Use a fake model/predictor to assert a crop box `(10, 20, 50, 80)` turns a
crop-local center `(5, 7)` into full-image `(15, 27)`. Save temporary valid
weights from `create_model("single")` using its established
`model.model.state_dict()` format, then assert `load_cnn_backend` succeeds;
assert the same weights fail clearly for `multi_head`.

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_perception.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/pybullet/perception.py \
  tests/simulation/test_pybullet_perception.py
git commit -m "feat: connect simulated images to grasp backends"
```

---

### Task 5: 编排 pilot、保存完整产物和失败元数据

**Files:**
- Create: `src/simulation/pybullet/run_pilot.py`
- Create: `tests/simulation/test_pybullet_runner.py`
- Modify: `tests/simulation/test_pybullet_smoke.py`

**Interfaces:**
- Consumes: `PyBulletScene`、`capture_camera_frame`、perception adapters、visualization functions。
- Produces: `PilotConfig`、`OutputPaths`、`build_output_paths`、`run_pilot(config, dependencies=None) -> dict`、CLI `main() -> int`。

- [ ] **Step 1: Write failing output-path and metadata tests**

```python
import json
from pathlib import Path

from src.simulation.pybullet.run_pilot import (
    PilotConfig,
    build_output_paths,
    run_pilot,
)


def test_output_paths_use_fixed_auditable_names(tmp_path: Path) -> None:
    paths = build_output_paths(tmp_path)
    assert paths.rgb == tmp_path / "rgb.png"
    assert paths.depth == tmp_path / "depth.npy"
    assert paths.depth_visualization == tmp_path / "depth_visualization.png"
    assert paths.segmentation == tmp_path / "segmentation.png"
    assert paths.localization == tmp_path / "localization.png"
    assert paths.prediction == tmp_path / "prediction.png"
    assert paths.metadata == tmp_path / "metadata.json"
```

Add a fake-dependency integration test that returns a 12×16 camera frame and
a fixed `PilotPrediction`, calls `run_pilot(PilotConfig(output_dir=tmp_path),
dependencies=fakes)`, and asserts all seven files exist. Load JSON and assert:

```python
assert metadata["status"] == "success"
assert metadata["physical_grasp_executed"] is False
assert metadata["backend"] == "geometry"
assert metadata["localization"]["box"] == [2, 2, 13, 9]
assert metadata["outputs"]["depth"] == str(tmp_path / "depth.npy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation/test_pybullet_runner.py -v
```

Expected: FAIL because `run_pilot.py` does not exist.

- [ ] **Step 3: Implement configuration, paths, atomic pipeline cleanup**

Implement:

```python
@dataclass(frozen=True)
class PilotConfig:
    gui: bool = False
    backend: str = "geometry"
    device: str = "cuda"
    prompt: str = "small object"
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    model_weights: Path | None = None
    object_urdf: str = "duck_vhacd.urdf"
    output_dir: Path = Path("data/processed/pybullet/pilot")
    seed: int = 42
    width: int = 640
    height: int = 480
    box_threshold: float = 0.25
    text_threshold: float = 0.25
```

`OutputPaths` contains the seven exact paths in the test. `run_pilot` must:

1. create the output directory without deleting unrelated files;
2. open the scene inside `try/finally`;
3. step 60 frames for settling;
4. capture and save RGB, `.npy` depth, depth visualization, segmentation;
5. load Grounding DINO once and localize the saved RGB;
6. reuse `draw_localization_result(rgb_path, localization_path, prompt,
   detection_dict_or_none)` to save `localization.png`, including
   `NO DETECTION` when needed;
7. load the selected backend only when needed;
8. predict and save `prediction.png`;
9. write JSON-safe metadata with UTC ISO timestamp;
10. always close the scene.

Use this internal dependency bundle for tests:

```python
@dataclass(frozen=True)
class PilotDependencies:
    scene_factory: Callable[[SceneConfig], PyBulletScene]
    capture_frame: Callable[[int, CameraConfig, int], CameraFrame]
    load_detector: Callable[[str, str], tuple[object, object]]
    localize: Callable[..., Localization | None]
    load_backend: Callable[[str, Path, str], object]
    predict: Callable[..., PilotPrediction]
```

`default_dependencies()` must bind these fields to `PyBulletScene`,
`capture_camera_frame`, `load_grounding_dino`, `localize_object`,
`load_cnn_backend`, and `predict_grasp`. Geometry runs must not call
`load_backend`; pass `model=None` to `predict`. Do not add a second
production pipeline.

- [ ] **Step 4: Implement failure metadata**

Wrap the processing stages so that detection failure writes:

```json
{
  "status": "failed",
  "failure_stage": "localization",
  "failure_reason": "no_detection",
  "physical_grasp_executed": false
}
```

and still retains RGB, depth, depth visualization, segmentation, and
localization image. Model/weight/runtime failures must write the exception
class and concise message, then return a nonzero CLI exit code; do not
silently manufacture a prediction.

Metadata on success must include:

- timestamp, seed, PyBullet API/version, connection mode, renderer;
- exact URDF names and initial poses;
- complete camera config, view and projection matrices;
- prompt, thresholds, model ID, device;
- backend and resolved weight path;
- localization box/score/label and grasp parameters;
- all output paths and `physical_grasp_executed: false`.

- [ ] **Step 5: Implement CLI and verify parser behavior**

Add arguments:

```text
--gui
--backend {geometry,single,multi_head}
--device {cpu,cuda}
--prompt
--model-id
--model-weights
--object-urdf
--output-dir
--seed
--width
--height
--box-threshold
--text-threshold
```

Add repository-root insertion matching existing scripts so this works from
the repository root:

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py --help
```

Expected: exit code 0 and all arguments listed.

- [ ] **Step 6: Run runner and all simulation tests**

Run:

```bash
conda run -n msc-grasp pytest tests/simulation -v
```

Expected: all tests PASS without loading Grounding DINO or opening a GUI.

- [ ] **Step 7: Commit**

```bash
git add src/simulation/pybullet/run_pilot.py tests/simulation
git commit -m "feat: add PyBullet perception pilot runner"
```

---

### Task 6: 添加使用说明、项目记录并完成真实验证

**Files:**
- Create: `src/simulation/pybullet/README.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: complete pilot CLI from Task 5.
- Produces: reproducible user commands, verified pilot artifacts, accurate project status.

- [ ] **Step 1: Write the module README**

Document:

- first-stage purpose and explicit non-goals;
- exact installed PyBullet version from Task 1;
- source attribution to official PyBullet documentation/repository;
- environment checks:

```bash
conda run -n msc-grasp python -c \
  "import pybullet as p, torch; print(p.getAPIVersion()); print(torch.cuda.is_available())"
```

- fastest DIRECT geometry command:

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend geometry --device cuda
```

- optional GUI command:

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --gui --backend geometry --device cuda
```

- single and multi-head examples with explicit formal weights:

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend single --device cuda \
  --model-weights data/processed/vlm/cnn_cross_validation/single/fold_0/model.pt

conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend multi_head --device cuda \
  --model-weights data/processed/vlm/cnn_cross_validation/multi_head/fold_0/model.pt
```

- all seven output files and the meaning of
  `physical_grasp_executed: false`;
- troubleshooting for missing PyBullet, missing display, CUDA unavailable,
  missing model cache, no detection, and architecture/weight mismatch.

- [ ] **Step 2: Run fast regression tests**

Run:

```bash
conda run -n msc-grasp pytest -q
```

Expected: all existing and new tests PASS.

- [ ] **Step 3: Run the real DIRECT geometry pilot**

Run:

```bash
conda run -n msc-grasp python src/simulation/pybullet/run_pilot.py \
  --backend geometry \
  --device cuda \
  --output-dir data/processed/pybullet/pilot
```

Expected success: exit code 0 and seven output files. Verify:

```bash
conda run -n msc-grasp python -c \
  "import json, numpy as np; from pathlib import Path; p=Path('data/processed/pybullet/pilot'); m=json.loads((p/'metadata.json').read_text()); d=np.load(p/'depth.npy'); assert m['status']=='success'; assert m['physical_grasp_executed'] is False; assert d.shape==(480,640); assert np.isfinite(d).all(); print(m['localization']); print(m['grasp'])"
```

If Grounding DINO returns no detection, keep the failure artifacts and
metadata, report that verified result, and do not lower thresholds or change
the object without a separate documented diagnostic decision.

- [ ] **Step 4: Run explicit CNN interface pilots**

Run the two README CNN commands. Both must load the matching architecture,
complete one image inference, and write finite full-image grasp parameters.
Record visual quality descriptively only after inspecting `prediction.png`;
do not treat either as a Cornell evaluation result.

- [ ] **Step 5: Inspect saved images**

Open or programmatically inspect:

```text
data/processed/pybullet/pilot/rgb.png
data/processed/pybullet/pilot/depth_visualization.png
data/processed/pybullet/pilot/segmentation.png
data/processed/pybullet/pilot/localization.png
data/processed/pybullet/pilot/prediction.png
```

Confirm RGB channels are correct, duck is visible, localization surrounds a
visible target, and the drawn grasp center/direction match metadata. If any
condition fails, use `superpowers:systematic-debugging` before changing code.

- [ ] **Step 6: Update project documentation using only verified facts**

In `PROJECT_STRUCTURE.md`, add the new `src/simulation/pybullet` module,
tests, data flow, and common commands.

In `CURRENT_STATUS.md`, record:

- first-stage scope completed;
- exact test count and command;
- exact real-run backend/device/status;
- artifact path;
- any no-detection or domain-shift limitation;
- next step is review before any physical grasp execution.

Append a dated concise entry to `WORKLOG.md` with commits, verification
commands, and verified outcomes. Do not edit dissertation claims in this
task.

- [ ] **Step 7: Run final verification**

Run:

```bash
git diff --check
conda run -n msc-grasp pytest -q
git status --short
```

Expected: no whitespace errors; all tests PASS; only intended source/docs
changes plus pre-existing untracked dissertation build artifacts appear.

- [ ] **Step 8: Commit**

```bash
git add src/simulation/pybullet/README.md \
  docs/agent/PROJECT_STRUCTURE.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record PyBullet pilot workflow"
```

Do not add `uog_dissertation_outline/l4proj.blg`,
`uog_dissertation_outline/l4proj.log`, or
`uog_dissertation_outline/l4proj.pdf`.
