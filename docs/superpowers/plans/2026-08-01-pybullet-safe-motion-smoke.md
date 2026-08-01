# PyBullet Safe Motion Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and audit one real Panda motor-control round trip from the neutral pose to a collision-free aerial waypoint and back, without approaching objects or closing the gripper.

**Architecture:** Add a small motor-control module that accepts named joint-space segments, commands Panda position motors while stepping PyBullet, and returns an immutable per-step trace plus collision and convergence gates. Add a stage-specific runner that derives an aerial waypoint by lifting the neutral tool pose `0.05 m`, performs the existing IK/FK and static-clearance preflight, executes the round trip in the fixed scene, and writes CSV/JSON/PNG evidence without invoking perception or gripper closure.

**Tech Stack:** Python 3.10+, PyBullet 3.2.7, NumPy, OpenCV, pytest, existing `scene.py`, `camera.py`, and `kinematic_audit.py` interfaces.

## Global Constraints

- Work only in `PyBullet.DIRECT`; stage 1 must not require CUDA or Grounding DINO.
- Use the fixed multi-object scene with Panda self-collision enabled.
- Initialize the seven arm joints to `PandaModelInfo.rest_poses[:7]` and both fingers to `0.04 m`; record this reset as initialization, not motor execution.
- Derive the safe waypoint from neutral FK with world-Z offset `+0.05 m` and unchanged quaternion.
- Require existing IK/FK thresholds `5 mm/5°` and existing static clearance `2 mm` before motor execution.
- During execution, exempt only the Panda base-link/table mounting pair already exempted by the static audit; target bodies are never exempt.
- Do not send finger motor commands other than holding `0.04 m`; do not approach, contact, close, lift, or run perception.
- Generated evidence belongs in `data/processed/pybullet/grasp_execution/stage_1_safe_motion/` and remains Git-ignored.
- Every production behavior is introduced test-first and observed failing before implementation.

---

### Task 1: Deterministic Panda Joint-Motion Executor

**Files:**
- Create: `src/simulation/pybullet/motion_control.py`
- Create: `tests/simulation/test_pybullet_motion_control.py`

**Interfaces:**
- Consumes: `PandaModelInfo`, connected `robot_id`/`client_id`, named seven-value arm targets, environment body IDs, and allowed `(robot_link, body_id)` mounting pairs.
- Produces: `MotionConfig`, `MotionSegment`, `MotionTraceRow`, `MotionExecutionResult`, and `execute_joint_motion(...) -> MotionExecutionResult`.

- [ ] **Step 1: Write failing validation tests**

Add tests that construct `MotionConfig()` and assert rejection of zero segment steps, negative settle steps, non-positive joint tolerance, and non-positive clearance. Add a test that passes a six-value target and expects `ValueError("motion target must contain seven arm values")`.

- [ ] **Step 2: Run validation tests and verify RED**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_motion_control.py -v
```

Expected: collection/import fails because `motion_control.py` does not exist.

- [ ] **Step 3: Implement immutable configuration and result types**

Create:

```python
@dataclass(frozen=True)
class MotionConfig:
    steps_per_segment: int = 240
    settle_steps: int = 240
    joint_tolerance_rad: float = 0.01
    clearance_m: float = 0.002

@dataclass(frozen=True)
class MotionSegment:
    name: str
    target_arm_positions: tuple[float, ...]

@dataclass(frozen=True)
class MotionTraceRow:
    step: int
    phase: str
    commanded_arm_positions: tuple[float, ...]
    actual_arm_positions: tuple[float, ...]
    actual_tool_position: tuple[float, float, float]
    actual_tool_quaternion_xyzw: tuple[float, float, float, float]
    maximum_joint_error_rad: float
    minimum_clearance_m: float
    environment_collision_count: int
    self_collision_count: int

@dataclass(frozen=True)
class MotionExecutionResult:
    trace: tuple[MotionTraceRow, ...]
    segment_reached: tuple[tuple[str, bool], ...]
    minimum_clearance_m: float
    environment_collision_count: int
    self_collision_count: int
    all_states_finite: bool
    gate_passed: bool
    failure_reason: str
```

Validate all numeric inputs and target shapes before sending any motor command.

- [ ] **Step 4: Run validation tests and verify GREEN**

Run the Task 1 test file. Expected: validation tests pass; real execution test is not yet present.

- [ ] **Step 5: Write a failing real-motor round-trip test**

Use the real fixed scene, enable self-collision, initialize neutral/open state, and define one target by adding `0.10 rad` to Panda joint 7. Execute two named segments `outbound` and `return`. Assert:

```python
assert result.gate_passed
assert dict(result.segment_reached) == {"outbound": True, "return": True}
assert len(result.trace) >= 2 * config.steps_per_segment
assert result.environment_collision_count == 0
assert result.self_collision_count == 0
assert result.all_states_finite
assert max(abs(a - b) for a, b in zip(final, neutral)) <= 0.01
```

Also assert both finger joint positions remain within `1e-3 m` of `0.04 m`.

- [ ] **Step 6: Run the real test and verify RED**

Expected: failure because `execute_joint_motion` is not implemented.

- [ ] **Step 7: Implement minimal motor execution**

For each segment, linearly interpolate command targets from the current measured arm state over `steps_per_segment`. At every step:

1. call `setJointMotorControlArray(..., POSITION_CONTROL, targetPositions=..., forces=...)` using positive URDF force limits from `getJointInfo`;
2. hold both finger joints at `0.04 m` with position control;
3. call `stepSimulation` exactly once;
4. sample actual arm joints and `panda_grasptarget` FK;
5. call `performCollisionDetection`, count non-exempt environment/self pairs within `clearance_m`, and record the closest observed distance;
6. append one immutable trace row.

After interpolation, hold the final command for at most `settle_steps`; mark the segment reached when maximum arm-joint error is at most `joint_tolerance_rad`. Set `gate_passed` only when every segment is reached, all values are finite, and both collision counts are zero.

- [ ] **Step 8: Run the Task 1 tests and verify GREEN**

Run the full motion-control test file. Expected: all tests pass with real `stepSimulation` execution.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/simulation/pybullet/motion_control.py \
  tests/simulation/test_pybullet_motion_control.py
git commit -m "feat: execute audited Panda joint motion"
```

### Task 2: Stage-1 Safe Aerial Round-Trip Runner

**Files:**
- Create: `src/simulation/pybullet/run_safe_motion_smoke.py`
- Create: `tests/simulation/test_pybullet_safe_motion_runner.py`
- Modify: `src/simulation/pybullet/README.md`

**Interfaces:**
- Consumes: `SafeMotionSmokeConfig(output_dir, seed=42, waypoint_lift_m=0.05)`, fixed scene factory, existing Panda resolver/IK/static-clearance functions, camera capture, and `execute_joint_motion`.
- Produces: `run_safe_motion_smoke(config) -> dict[str, object]`, `state_trace.csv`, `summary.json`, `metadata.json`, `start.png`, `waypoint.png`, and `return.png`.

- [ ] **Step 1: Write a failing output-contract test**

Run the real DIRECT scene in `tmp_path`. Assert all six output files exist and are non-empty. Assert the summary includes:

```python
assert summary["stage"] == "safe_motion_smoke"
assert summary["waypoint_lift_m"] == pytest.approx(0.05)
assert summary["preflight_ik_fk_passed"] is True
assert summary["preflight_clearance_passed"] is True
assert summary["outbound_reached"] is True
assert summary["return_reached"] is True
assert summary["scientific_gate_passed"] is True
```

Assert metadata records `motor_control_executed: true`, `simulation_stepped: true`, and all perception/contact/closure/lift fields as `false`.

- [ ] **Step 2: Run the runner test and verify RED**

Expected: import failure because `run_safe_motion_smoke.py` does not exist.

- [ ] **Step 3: Implement the stage runner**

Build `fixed_scene_config(MultiObjectStudyConfig(gui=False, seed=config.seed))`, replace `robot_self_collision=True`, connect, settle the scene for 60 steps, resolve Panda, and initialize neutral/open state. Capture neutral FK, construct `ToolPose((x, y, z + 0.05), same_quaternion)`, then:

1. run `audit_pose_ik` for the waypoint;
2. run `audit_joint_path_clearance` for neutral→waypoint→neutral using all scene bodies and the base/table exemption;
3. stop and write a failed summary without motor commands if either preflight gate fails;
4. otherwise execute named `outbound` and `return` motor segments;
5. compute tool position/orientation errors at the waypoint and return endpoints using the existing `5 mm/5°` definitions;
6. save every trace row to CSV, JSON summary/metadata, and RGB snapshots at neutral, first reached waypoint state, and final return state.

The CLI accepts only `--output-dir`, `--seed`, and `--gui`; default output is `data/processed/pybullet/grasp_execution/stage_1_safe_motion`.

- [ ] **Step 4: Run the runner test and verify GREEN**

Run:

```bash
conda run -n msc-grasp python -m pytest \
  tests/simulation/test_pybullet_safe_motion_runner.py -v
```

Expected: real runner passes and writes the complete evidence contract.

- [ ] **Step 5: Add failure-preservation test**

Inject a dependency whose preflight IK returns `gate_passed=False`. Assert the runner writes summary/metadata, sets `motor_control_executed=false`, produces no false successful endpoint flags, and records `failure_stage="preflight_ik"`.

- [ ] **Step 6: Run runner tests and verify GREEN**

Expected: success and failure output contracts both pass.

- [ ] **Step 7: Document the command and scientific boundary**

Add to the PyBullet README:

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_safe_motion_smoke.py \
  --output-dir data/processed/pybullet/grasp_execution/stage_1_safe_motion
```

State that this stage executes motors and simulation steps but never approaches a target, closes fingers, or performs a grasp.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/simulation/pybullet/run_safe_motion_smoke.py \
  tests/simulation/test_pybullet_safe_motion_runner.py \
  src/simulation/pybullet/README.md
git commit -m "feat: add Panda safe motion smoke stage"
```

### Task 3: Real Evidence, Regression, and Project Records

**Files:**
- Modify after verified output: `docs/agent/CURRENT_STATUS.md`
- Modify after verified output: `docs/agent/PROJECT_STRUCTURE.md`
- Modify after verified output: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: the real stage-1 output directory and complete test suite.
- Produces: evidence-backed status, structure, and worklog entries with no claims beyond safe aerial motion.

- [ ] **Step 1: Run the real stage**

```bash
conda run -n msc-grasp python \
  src/simulation/pybullet/run_safe_motion_smoke.py \
  --output-dir data/processed/pybullet/grasp_execution/stage_1_safe_motion
```

Expected: exit zero and `scientific_gate_passed: true`. If it fails, preserve outputs and diagnose stage 1; do not implement stage 2.

- [ ] **Step 2: Independently inspect evidence**

Check summary/metadata booleans, CSV row count and finiteness, final joint/tool errors, collision counts, and the three PNG dimensions. Confirm no contact, closure, lift, perception, or grasp flag is true.

- [ ] **Step 3: Run focused and complete regression**

```bash
conda run -n msc-grasp python -m pytest tests/simulation -q
conda run -n msc-grasp python -m pytest -q
git diff --check
```

Expected: all tests pass and diff check is clean.

- [ ] **Step 4: Update project records from actual output only**

Record exact observed step counts, endpoint errors, minimum clearance, collision counts, output path, and test totals. State explicitly that target approach, contact, closure, lift, and physical grasp remain unexecuted. Add the new modules/tests to project structure.

- [ ] **Step 5: Commit verified stage 1**

```bash
git add docs/agent/CURRENT_STATUS.md docs/agent/PROJECT_STRUCTURE.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record Panda safe motion evidence"
```

