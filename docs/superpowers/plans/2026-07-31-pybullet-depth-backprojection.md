# PyBullet Depth Backprojection Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the nine fixed PyBullet 2-D grasp centres into audited camera/world surface points and persist an explicit pre-IK gate.

**Architecture:** Add a focused `backprojection.py` module for nearest-depth sampling, OpenGL matrix conversion, backprojection/reprojection, ray construction, per-row audit, and exact-nine-row aggregation. Extend the existing multi-object runner only at its orchestration and output boundaries; RGB-only model inputs and all existing 2-D outputs remain unchanged.

**Tech Stack:** Python 3, NumPy, PyBullet 3.2.7, CSV/JSON, pytest, existing `CameraFrame`, `CameraConfig`, `BACKEND_ORDER`, and `EXPECTED_TARGET_BACKENDS`.

## Global Constraints

- Use nearest-pixel sampling: `floor(value + 0.5)`; do not use Python banker rounding.
- Treat PyBullet matrices as OpenGL column-major arrays and reshape with `order="F"`.
- Backprojection consumes only 2-D centre, metric depth, image dimensions, near/far, and view/projection matrices.
- Segmentation and `rayTest` are post-hoc truth audits and never coordinate inputs.
- Require all nine `TARGET_ORDER × BACKEND_ORDER` rows for a passing gate.
- Pixel reprojection error threshold is `1.0` pixel; metric-depth error threshold is `1e-4` metre.
- A depth equal to either clip plane is invalid; valid depth is strictly inside `(near, far)`.
- Do not implement grasp orientation, physical width, IK, collision planning, robot motion, gripper closure, or grasp-success detection.
- Do not change Cornell experiments, model weights, prompts, thresholds, or RGB-only model behavior.
- Cite the official Bullet/PyBullet project and quickstart guide in new module documentation; do not describe external APIs as original project inventions.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/simulation/pybullet/backprojection.py` | Pure coordinate conversion, one-row post-hoc audit, ray endpoint construction, exact-nine aggregation |
| `tests/simulation/test_pybullet_backprojection.py` | Unit tests and real DIRECT PyBullet integration test for coordinate conventions |
| `src/simulation/pybullet/run_multi_object_study.py` | Invoke the audit after 2-D rows, save CSV/JSON, expose metadata and output paths |
| `tests/simulation/test_pybullet_multi_object_runner.py` | Lock runner ordering, output schema, audit-only boundaries, and incomplete-row behavior |
| `src/simulation/pybullet/README.md` | Document command, outputs, thresholds, and non-goals |
| `docs/agent/PROJECT_STRUCTURE.md` | Register the new module and artifacts |
| `docs/agent/CURRENT_STATUS.md` | Record only verified real-run gate results |
| `docs/debugging/FAILURE_ANALYSIS.md` | Record observed backprojection failures without causal overclaim |
| `docs/worklog/WORKLOG.md` | Summarize implementation and verification |

---

### Task 1: Nearest Depth and OpenGL Matrix Backprojection

**Files:**
- Create: `tests/simulation/test_pybullet_backprojection.py`
- Create: `src/simulation/pybullet/backprojection.py`

**Interfaces:**
- Produces: `DepthSample(column: int, row: int, depth_m: float)`
- Produces: `BackprojectedPoint(camera_xyz, world_xyz)`
- Produces: `ReprojectedPoint(pixel_x, pixel_y, depth_m)`
- Produces: `sample_nearest_depth(depth_m, center_x, center_y, near, far) -> DepthSample`
- Produces: `metric_depth_to_buffer(depth_m, near, far) -> float`
- Produces: `backproject_pixel(column, row, depth_m, width, height, view_matrix, projection_matrix) -> BackprojectedPoint`
- Produces: `reproject_world_point(world_xyz, width, height, view_matrix, projection_matrix) -> ReprojectedPoint`

- [ ] **Step 1: Write failing nearest-depth tests**

Add literal tests that catch banker rounding, swapped row/column, background depth, and invalid bounds:

```python
import numpy as np
import pytest

from src.simulation.pybullet.backprojection import sample_nearest_depth


def test_nearest_depth_uses_half_up_pixel_rounding() -> None:
    depth = np.arange(20, dtype=np.float32).reshape(4, 5) / 10 + 0.1

    sample = sample_nearest_depth(
        depth,
        center_x=1.5,
        center_y=2.5,
        near=0.05,
        far=3.0,
    )

    assert (sample.column, sample.row) == (2, 3)
    assert sample.depth_m == pytest.approx(float(depth[3, 2]))


@pytest.mark.parametrize(
    ("center_x", "center_y", "match"),
    [(-0.01, 0.0, "inside image"), (5.0, 0.0, "inside image")],
)
def test_nearest_depth_rejects_centres_outside_image(
    center_x: float,
    center_y: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        sample_nearest_depth(
            np.ones((4, 5), dtype=np.float32),
            center_x,
            center_y,
            near=0.05,
            far=3.0,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.05, 3.0])
def test_nearest_depth_rejects_non_surface_depth(value: float) -> None:
    depth = np.full((2, 2), 0.8, dtype=np.float32)
    depth[1, 1] = value
    with pytest.raises(ValueError, match="depth"):
        sample_nearest_depth(depth, 1.0, 1.0, near=0.05, far=3.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_backprojection.py -v
```

Expected: collection fails with `ModuleNotFoundError` for `backprojection`.

- [ ] **Step 3: Implement the minimal nearest-depth API**

Create the module with official source attribution and these definitions:

```python
"""Depth backprojection for the PyBullet perception study.

Matrix and depth conventions follow the public PyBullet quickstart guide:
https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstartguide.pdf
The audit design and implementation are project-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class DepthSample:
    column: int
    row: int
    depth_m: float


@dataclass(frozen=True)
class BackprojectedPoint:
    camera_xyz: tuple[float, float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ReprojectedPoint:
    pixel_x: float
    pixel_y: float
    depth_m: float


def sample_nearest_depth(
    depth_m: np.ndarray,
    center_x: float,
    center_y: float,
    near: float,
    far: float,
) -> DepthSample:
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    if not 0.0 < near < far:
        raise ValueError("near must be positive and smaller than far")
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise ValueError("grasp centre must be finite")
    height, width = depth.shape
    if not (0.0 <= center_x <= width - 1 and 0.0 <= center_y <= height - 1):
        raise ValueError("grasp centre must be inside image")
    column = math.floor(center_x + 0.5)
    row = math.floor(center_y + 0.5)
    value = float(depth[row, column])
    if not math.isfinite(value) or not near < value < far:
        raise ValueError("sampled depth must be finite and inside clip planes")
    return DepthSample(column=column, row=row, depth_m=value)
```

- [ ] **Step 4: Verify nearest-depth GREEN**

Run the test file. Expected: all nearest-depth tests pass.

- [ ] **Step 5: Write failing matrix and hand-derived world-point tests**

Add tests using a one-pixel camera looking straight down the negative world Z axis. The hand-derived expected point is independent of the implementation:

```python
import pybullet as p

from src.simulation.pybullet.backprojection import (
    backproject_pixel,
    metric_depth_to_buffer,
    reproject_world_point,
)


def test_metric_depth_inverse_maps_clip_planes() -> None:
    assert metric_depth_to_buffer(0.1, 0.1, 10.0) == pytest.approx(0.0)
    assert metric_depth_to_buffer(10.0, 0.1, 10.0) == pytest.approx(1.0)


def test_backprojection_recovers_hand_derived_world_point() -> None:
    view = p.computeViewMatrix(
        cameraEyePosition=(0.0, 0.0, 1.0),
        cameraTargetPosition=(0.0, 0.0, 0.0),
        cameraUpVector=(0.0, 1.0, 0.0),
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=10.0,
    )

    point = backproject_pixel(
        column=0,
        row=0,
        depth_m=0.5,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
    )

    assert point.camera_xyz == pytest.approx((0.0, 0.0, -0.5), abs=1e-6)
    assert point.world_xyz == pytest.approx((0.0, 0.0, 0.5), abs=1e-6)

    reprojection = reproject_world_point(
        point.world_xyz,
        width=1,
        height=1,
        view_matrix=view,
        projection_matrix=projection,
    )
    assert (reprojection.pixel_x, reprojection.pixel_y) == pytest.approx(
        (0.0, 0.0), abs=1e-6
    )
    assert reprojection.depth_m == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 6: Verify matrix tests fail for missing functions**

Run the test file. Expected: import fails for the new matrix functions.

- [ ] **Step 7: Implement matrix conversion and validation**

Implement helpers `_matrix4`, `_homogeneous_divide`, NDC conversion, algebraic depth inversion, backprojection, and reprojection. Explicitly validate: 16 finite matrix elements, positive image dimensions, invertible matrices, finite coordinates, and non-zero homogeneous W. Use:

```python
matrix = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
combined = projection @ view
world_h = np.linalg.inv(combined) @ clip
world = world_h[:3] / world_h[3]
```

Reprojection uses:

```python
clip = projection @ view @ np.array([*world_xyz, 1.0])
ndc = clip[:3] / clip[3]
pixel_x = (ndc[0] + 1.0) * width / 2.0 - 0.5
pixel_y = (1.0 - ndc[1]) * height / 2.0 - 0.5
camera_h = view @ np.array([*world_xyz, 1.0])
depth_m = -camera_h[2] / camera_h[3]
```

- [ ] **Step 8: Verify matrix GREEN and validation behavior**

Run the test file, then add and pass tests for a 15-element matrix, `np.nan`, a singular zero matrix, and zero homogeneous W.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/simulation/pybullet/backprojection.py \
  tests/simulation/test_pybullet_backprojection.py
git commit -m "feat: backproject PyBullet depth pixels"
```

---

### Task 2: One-Row Truth Audit and Exact-Nine Summary

**Files:**
- Modify: `src/simulation/pybullet/backprojection.py`
- Modify: `tests/simulation/test_pybullet_backprojection.py`

**Interfaces:**
- Consumes: Task 1 conversion APIs and existing `EXPECTED_TARGET_BACKENDS`
- Produces: `BackprojectionAudit` dataclass
- Produces: `build_ray_segment(camera_eye, world_xyz, extension_m=0.05)`
- Produces: `audit_backprojected_grasp(backend_row, depth_m, segmentation, expected_body_id, camera_eye, image_width, image_height, near, far, view_matrix, projection_matrix, client_id, ray_test) -> BackprojectionAudit`
- Produces: `summarize_backprojection_rows(rows) -> dict[str, object]`

- [ ] **Step 1: Write a failing successful-audit test**

Build a literal `3×3` depth/segmentation fixture, identity body mapping, hand-derived camera matrices, and a ray callback returning the requested body. Assert original centre, sampled `(column,row)`, depth, finite camera/world values, pixel error `<= 1e-6`, depth error `<= 1e-6`, segmentation match, ray match, and empty failure reason.

- [ ] **Step 2: Verify RED**

Run only the new test. Expected: import failure for `audit_backprojected_grasp`.

- [ ] **Step 3: Implement the audit record and ray segment**

The immutable record must expose fields that serialize directly to the final CSV:

```python
@dataclass(frozen=True)
class BackprojectionAudit:
    target: str
    backend: str
    center_x: float
    center_y: float
    sampled_column: int | None
    sampled_row: int | None
    depth_m: float | None
    camera_x: float | None
    camera_y: float | None
    camera_z: float | None
    world_x: float | None
    world_y: float | None
    world_z: float | None
    reprojected_x: float | None
    reprojected_y: float | None
    reprojected_depth_m: float | None
    pixel_error: float | None
    depth_error_m: float | None
    coordinates_finite: bool
    valid_depth: bool
    reprojection_passed: bool
    segmentation_body_id: int | None
    expected_body_id: int
    segmentation_target_match: bool
    ray_body_id: int | None
    ray_target_match: bool
    ray_hit_position: tuple[float, float, float] | None
    ray_hit_distance_m: float | None
    gate_passed: bool
    failure_reason: str
```

`build_ray_segment` starts at `camera_eye` and ends `0.05 m` beyond the reconstructed point along the normalized eye-to-point direction. Reject coincident or non-finite points.

- [ ] **Step 4: Implement audit failure capture**

`audit_backprojected_grasp` must catch expected `ValueError` conversion failures and return a row with null numeric outputs, `gate_passed=False`, and the exact conversion message. It decodes the segmentation body with `(value & ((1 << 24) - 1))` only after coordinate conversion. The ray callback signature is:

```python
ray_test(
    ray_from_position: tuple[float, float, float],
    ray_to_position: tuple[float, float, float],
    client_id: int,
) -> tuple[int, tuple[float, float, float] | None]
```

It returns `(body_id, hit_position)` and does not receive the expected body ID.

- [ ] **Step 5: Verify audit GREEN**

Run the targeted test. Expected: pass.

- [ ] **Step 6: Add failing mutation-focused tests**

Add separate cases proving each realistic bug is caught:

- swapped target segmentation ID;
- ray hits a different body;
- projection matrix singular;
- reprojection threshold set below an injected pixel mismatch;
- incomplete or reordered `(target, backend)` rows.

- [ ] **Step 7: Implement exact-nine aggregation**

Require the exact order from `EXPECTED_TARGET_BACKENDS`. Return literal boundary flags and counts:

```python
{
    "protocol": "fixed_three_object_depth_backprojection_gate",
    "backprojection_result_count": 9,
    "coordinates_finite_count": sum(
        row.coordinates_finite for row in rows
    ),
    "valid_depth_count": sum(row.valid_depth for row in rows),
    "reprojection_passed_count": sum(
        row.reprojection_passed for row in rows
    ),
    "segmentation_target_match_count": sum(
        row.segmentation_target_match for row in rows
    ),
    "ray_target_match_count": sum(row.ray_target_match for row in rows),
    "backprojection_gate_passed": all(row.gate_passed for row in rows),
    "pixel_error_threshold": 1.0,
    "depth_error_threshold_m": 1e-4,
    "depth_used_after_2d_prediction": True,
    "segmentation_used_as_coordinate_input": False,
    "ray_test_used_as_coordinate_input": False,
    "ik_executed": False,
    "physical_grasp_executed": False,
}
```

For incomplete runner output, provide a separate `summarize_available_backprojection_rows` helper that preserves rows and returns `backprojection_complete=False` and `backprojection_gate_passed=False`; it must never claim an exact-nine pass.

- [ ] **Step 8: Verify Task 2 GREEN**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_backprojection.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/simulation/pybullet/backprojection.py \
  tests/simulation/test_pybullet_backprojection.py
git commit -m "feat: audit backprojected grasp centres"
```

---

### Task 3: Real DIRECT PyBullet Coordinate Assumption Test

**Files:**
- Modify: `tests/simulation/test_pybullet_backprojection.py`

**Interfaces:**
- Consumes: Task 1 APIs, `PyBulletScene`, `SceneConfig`, `CameraConfig`, and `capture_camera_frame`
- Produces: a narrow real test locking visual-depth, segmentation, column-major, Y-flip, and collision-ray assumptions

- [ ] **Step 1: Write the real integration test**

Create a DIRECT scene whose target is `cube_small.urdf`. Capture `160×120`, find the target-mask pixel nearest the mask centroid, backproject its metric depth, reproject it, and call real `p.rayTest` from camera eye through `0.05 m` beyond the point. Assert:

```python
assert pixel_error <= 1e-6
assert depth_error <= 1e-5
assert decoded_segmentation_body == scene.bodies.target_object
assert ray_result[0][0] == scene.bodies.target_object
```

Use `try/finally` or the scene context manager so the client always closes.

- [ ] **Step 2: Run the integration test**

Run the exact test with `-v`. If it fails, treat the failure as a coordinate-assumption defect: do not loosen the `1 px` formal gate or use segmentation to move the point. Correct matrix ordering, NDC conversion, pixel-centre convention, or ray construction in production code.

- [ ] **Step 3: Run all backprojection tests**

Expected: every unit and real integration test passes without GPU.

- [ ] **Step 4: Commit Task 3**

```bash
git add tests/simulation/test_pybullet_backprojection.py \
  src/simulation/pybullet/backprojection.py
git commit -m "test: verify PyBullet backprojection conventions"
```

---

### Task 4: Persist Backprojection Results in the Multi-Object Runner

**Files:**
- Modify: `src/simulation/pybullet/run_multi_object_study.py`
- Modify: `tests/simulation/test_pybullet_multi_object_runner.py`

**Interfaces:**
- Consumes: Task 2 audit/summary APIs and existing nine backend rows
- Produces: `StudyOutputPaths.backprojection_results_csv`
- Produces: `StudyOutputPaths.backprojection_summary`
- Produces: optional `MultiObjectStudyDependencies.ray_test`
- Produces: `_write_backprojection_results_csv(path, rows)`

- [ ] **Step 1: Write failing output-path tests**

Extend `test_backend_output_paths_and_seed_42_defaults`:

```python
assert paths.backprojection_results_csv == (
    tmp_path / "backprojection_results.csv"
)
assert paths.backprojection_summary == (
    tmp_path / "backprojection_summary.json"
)
```

- [ ] **Step 2: Verify RED and implement paths**

Add both fields to `StudyOutputPaths`, construct them in `build_study_output_paths`, include them in `_prepare_output_paths`, and expose them from `_outputs_metadata`. Re-run the targeted test until green.

- [ ] **Step 3: Write the failing complete-nine runner test**

Add a dedicated runner test, separate from the existing incomplete-target test. Its four localizations must correctly select duck, cube, sphere, and the generic robot diagnostic. Its fake predictions must put all nine centres inside the corresponding masks. Use real valid view/projection matrices from `pybullet.computeViewMatrix` and `computeProjectionMatrixFOV`, and a `ray_test` fake that maps each ray endpoint to a body based on endpoint location without receiving the expected ID.

Assert:

```python
assert summary["backend_comparison_complete"] is True
assert summary["backprojection_complete"] is True
assert summary["backprojection_gate_passed"] is True
assert paths.backprojection_results_csv.is_file()
assert paths.backprojection_summary.is_file()
with paths.backprojection_results_csv.open(
    newline="", encoding="utf-8"
) as handle:
    saved_backprojection_rows = list(csv.DictReader(handle))
assert len(saved_backprojection_rows) == 9
assert metadata["depth_used_after_2d_prediction"] is True
assert metadata["segmentation_used_as_coordinate_input"] is False
assert metadata["ray_test_used_as_coordinate_input"] is False
assert metadata["ik_executed"] is False
assert metadata["physical_grasp_executed"] is False
```

- [ ] **Step 4: Verify RED**

Expected: missing dependency/output/summary fields.

- [ ] **Step 5: Add runner dependency and CSV writer**

Add an optional callable field after `predict`:

```python
ray_test: Callable[
    [tuple[float, float, float], tuple[float, float, float], int],
    tuple[int, tuple[float, float, float] | None],
] | None = None
```

`default_dependencies` binds a wrapper around `p.rayTest`. Parse the first hit into body ID and position. The CSV writer uses the `BackprojectionAudit` dataclass field order and `_csv_value` for tuples.

- [ ] **Step 6: Integrate after all 2-D backend rows**

Only after `_write_backend_results_csv` and backend aggregation:

1. iterate existing backend rows in order;
2. call `audit_backprojected_grasp` with the same frame, camera config, camera eye, scene body mapping, segmentation, and optional ray callback;
3. save all returned rows even when conversion fails;
4. write available/exact summary;
5. merge only summary fields into the main summary;
6. add full rows and boundary flags to metadata.

If `ray_test is None`, rows must record `ray_test_unavailable`, the gate remains false, and all prior 2-D behavior remains successful. This preserves existing fake dependency callers without inventing ray success.

- [ ] **Step 7: Verify complete and incomplete runner behavior**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_multi_object_runner.py -v
```

Expected: existing incomplete-target test still reports successful perception but false/incomplete backprojection; the new complete-nine test reports a passing gate.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/simulation/pybullet/run_multi_object_study.py \
  tests/simulation/test_pybullet_multi_object_runner.py
git commit -m "feat: persist nine-point backprojection audit"
```

---

### Task 5: Documentation, Full Regression, and Real Nine-Point Run

**Files:**
- Modify: `src/simulation/pybullet/README.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/agent/CURRENT_STATUS.md` only after real output verification
- Modify: `docs/debugging/FAILURE_ANALYSIS.md` only for observed failures
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: saved `backprojection_results.csv` and `backprojection_summary.json`
- Produces: evidence-backed project status and reproducible command documentation

- [ ] **Step 1: Run focused and complete verification**

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_backprojection.py \
  tests/simulation/test_pybullet_multi_object_runner.py -v
conda run -n msc-grasp python -m pytest -q
conda run -n msc-grasp python -m py_compile \
  src/simulation/pybullet/backprojection.py \
  src/simulation/pybullet/run_multi_object_study.py
git diff --check
```

Expected: focused tests pass; full suite passes with the new count; compile and diff checks exit zero.

- [ ] **Step 2: Update stable documentation before the real run**

README and structure documentation must explain:

- nearest-pixel sampling;
- matrix inputs and output coordinates;
- both new artifact names;
- exact hard thresholds;
- segmentation/ray audit-only boundary;
- gate failure blocks IK but does not erase 2-D success;
- command remains the existing fixed multi-object CLI.

Add a worklog implementation entry without claiming the real gate passed.

- [ ] **Step 3: Commit tested implementation documentation**

```bash
git add src/simulation/pybullet/README.md \
  docs/agent/PROJECT_STRUCTURE.md docs/worklog/WORKLOG.md
git commit -m "docs: document depth backprojection gate"
```

- [ ] **Step 4: Run the fixed real study outside the restricted GPU sandbox**

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_multi_object_study.py \
  --device cuda \
  --output-dir data/processed/pybullet/multi_object_study
```

Use existing official seed-42 single/multi-head weights. Do not change scene, prompts, camera, thresholds, or weights after seeing results.

- [ ] **Step 5: Independently audit saved artifacts**

Verify exact row order/count, finite numbers, thresholds, body matches, and boundary flags by reading CSV/JSON independently of the runner summary. Confirm all nine original 2-D centre values equal `backend_results.csv`. Confirm metadata says depth is post-prediction and IK/physical grasp are false.

- [ ] **Step 6: Record only actual output**

If the real gate passes, update `CURRENT_STATUS.md` with counts, maximum reprojection errors, target/ray match counts, output paths, and the explicit non-IK boundary. If it fails, record the observed failing rows and checks in `FAILURE_ANALYSIS.md`, keep `backprojection_gate_passed: false`, and do not proceed to IK.

- [ ] **Step 7: Run final verification after documentation changes**

Run the full suite, `git diff --check`, and an independent JSON/CSV audit again. Expected claims must match saved output exactly.

- [ ] **Step 8: Commit verified real-run records**

```bash
git add docs/agent/CURRENT_STATUS.md docs/debugging/FAILURE_ANALYSIS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record depth backprojection gate"
```

Do not add ignored `data/` results, model weights, LaTeX build files, or the worktree data symlink.
