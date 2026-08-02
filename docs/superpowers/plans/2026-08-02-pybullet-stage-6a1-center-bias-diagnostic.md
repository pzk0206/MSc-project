# PyBullet Stage 6A.1 Center Bias Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strictly offline Stage 6A.1 audit that preserves the frozen Stage 6A evidence, records the geometry surface-point-to-cube-center bias, and supports an isolated reproducibility rerun.

**Architecture:** A dependency-light calculation module owns constants, finite-value validation, bias computation, and stable serialization. A separate runner strictly loads and cross-checks the three Stage 6A evidence files, writes diagnostic-only outputs to another directory, and never imports scene, perception, or control code. Formal evidence first audits the existing frozen run; a later CUDA rerun uses separate input/output directories and the same audit code.

**Tech Stack:** Python 3.10+, standard-library `dataclasses`, `json`, `csv`, `hashlib`, `math`, existing strict Stage 6A execution-plan loader, pytest.

## Global Constraints

- Formal Stage 6A.1 reads the existing Stage 6A directory without modifying any source file.
- Cube truth is post-hoc evaluation only and cannot modify perception, backprojection, candidates, or `execution_plan.json`.
- `CUBE_HALF_EXTENT_M` is frozen at exactly `0.025`; `XY_REFERENCE_THRESHOLD_M` is frozen at exactly `0.005`.
- The derived Z reference is named `nominal_top_reference_z_m`, not a newly measured AABB.
- No `stepSimulation`, motor, trajectory, gripper, contact, lift, physical-grasp, Torch, Transformers, or PyBullet imports are allowed in the Stage 6A.1 modules.
- Reproducibility evidence uses directories distinct from the formal Stage 6A and Stage 6A.1 directories.
- Stage 6A `scientific_gate_passed` remains historical evidence and is never reinterpreted as center validity or physical success.

---

### Task 1: Pure center-bias diagnostic contract

**Files:**
- Create: `src/simulation/pybullet/center_bias_diagnostic.py`
- Create: `tests/simulation/test_pybullet_center_bias_diagnostic.py`

**Interfaces:**
- Consumes: three-element finite `Sequence[float]` prediction and truth points.
- Produces: `PROTOCOL_VERSION`, `CUBE_HALF_EXTENT_M`, `XY_REFERENCE_THRESHOLD_M`, frozen `CenterBiasMeasurement`, `compute_center_bias(predicted_world_surface_point, cube_truth_center, *, cube_half_extent_m=..., xy_reference_threshold_m=...) -> CenterBiasMeasurement`, `write_diagnostic_json(path: Path, payload: Mapping[str, object]) -> None`, and `write_diagnostic_csv(path: Path, measurement: CenterBiasMeasurement) -> None`.

- [ ] **Step 1: Write failing calculation tests**

```python
def test_compute_center_bias_records_signed_offsets_and_reference_gate():
    result = compute_center_bias(
        (0.5064564100151149, 0.002224916375108214, 0.6754779706501471),
        (0.4800002872798181, -5.134833814891427e-07, 0.649968798272667),
    )
    assert result.signed_x_offset_m == pytest.approx(0.0264561227352968)
    assert result.signed_y_offset_m == pytest.approx(0.002225429858489703)
    assert result.xy_offset_m == pytest.approx(0.026549556836982145)
    assert result.nominal_top_reference_z_m == pytest.approx(0.674968798272667)
    assert result.signed_nominal_top_z_offset_m == pytest.approx(0.0005091723774801)
    assert result.xy_within_reference_threshold is False


@pytest.mark.parametrize(
    "prediction,truth",
    [((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0, 0.0))],
)
def test_compute_center_bias_rejects_invalid_points(prediction, truth):
    with pytest.raises(ValueError):
        compute_center_bias(prediction, truth)


def test_compute_center_bias_rejects_protocol_constant_changes():
    with pytest.raises(ValueError, match="half extent"):
        compute_center_bias((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), cube_half_extent_m=0.03)
```

- [ ] **Step 2: Verify the calculation tests fail for the missing module**

Run:

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_center_bias_diagnostic.py -v
```

Expected: collection error because `center_bias_diagnostic` does not exist.

- [ ] **Step 3: Implement the minimal frozen measurement**

```python
PROTOCOL_VERSION = "stage_6a1_center_bias_diagnostic_v1"
CUBE_HALF_EXTENT_M = 0.025
XY_REFERENCE_THRESHOLD_M = 0.005


@dataclass(frozen=True)
class CenterBiasMeasurement:
    predicted_world_surface_point: tuple[float, float, float]
    cube_truth_center: tuple[float, float, float]
    cube_half_extent_m: float
    nominal_top_reference_z_m: float
    signed_x_offset_m: float
    signed_y_offset_m: float
    xy_offset_m: float
    signed_nominal_top_z_offset_m: float
    xy_reference_threshold_m: float
    xy_within_reference_threshold: bool


def compute_center_bias(...):
    prediction = _point3(predicted_world_surface_point, "predicted_world_surface_point")
    truth = _point3(cube_truth_center, "cube_truth_center")
    if cube_half_extent_m != CUBE_HALF_EXTENT_M:
        raise ValueError("cube half extent must remain frozen at 0.025 m")
    if xy_reference_threshold_m != XY_REFERENCE_THRESHOLD_M:
        raise ValueError("XY reference threshold must remain frozen at 0.005 m")
    dx, dy = prediction[0] - truth[0], prediction[1] - truth[1]
    nominal_top = truth[2] + cube_half_extent_m
    return CenterBiasMeasurement(...)
```

- [ ] **Step 4: Verify the calculation tests pass**

Run the Task 1 pytest command. Expected: all current tests in the file pass.

- [ ] **Step 5: Add failing stable-output tests**

```python
def test_diagnostic_writers_emit_strict_json_and_one_row_csv(tmp_path):
    measurement = compute_center_bias((0.506, 0.002, 0.675), (0.48, 0.0, 0.65))
    payload = {"protocol": PROTOCOL_VERSION, "measurement": asdict(measurement)}
    write_diagnostic_json(tmp_path / "diagnostic.json", payload)
    write_diagnostic_csv(tmp_path / "diagnostic.csv", measurement)
    assert json.loads((tmp_path / "diagnostic.json").read_text())["protocol"] == PROTOCOL_VERSION
    rows = list(csv.DictReader((tmp_path / "diagnostic.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["xy_offset_m"]) == pytest.approx(measurement.xy_offset_m)


def test_diagnostic_json_rejects_non_finite_payload(tmp_path):
    with pytest.raises(ValueError):
        write_diagnostic_json(tmp_path / "diagnostic.json", {"bad": float("nan")})
```

- [ ] **Step 6: Run and verify the new output tests fail because writers are missing**

Run the Task 1 pytest command. Expected: FAIL at the missing writer behavior.

- [ ] **Step 7: Implement strict JSON and CSV writers**

Use `json.dumps(..., indent=2, ensure_ascii=False, allow_nan=False) + "\n"`; write the CSV with an explicit field list matching every `CenterBiasMeasurement` field and serialize tuple fields as JSON arrays.

- [ ] **Step 8: Run Task 1 tests and compile the module**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_center_bias_diagnostic.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/center_bias_diagnostic.py
```

Expected: both commands exit 0.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/simulation/pybullet/center_bias_diagnostic.py \
  tests/simulation/test_pybullet_center_bias_diagnostic.py
git commit -m "feat: define stage 6a1 center bias diagnostic"
```

---

### Task 2: Strict offline Stage 6A.1 runner

**Files:**
- Create: `src/simulation/pybullet/run_center_bias_diagnostic.py`
- Create: `tests/simulation/test_pybullet_center_bias_runner.py`

**Interfaces:**
- Consumes: `load_geometry_execution_plan(path) -> GeometryExecutionPlan`, Task 1 calculation/writers, and a source directory containing `summary.json`, `metadata.json`, `execution_plan.json`, and `rgb.png`.
- Produces: frozen `CenterBiasDiagnosticConfig(source_dir: Path, output_dir: Path, evidence_role: str = "formal")` and `run_center_bias_diagnostic(config: CenterBiasDiagnosticConfig) -> dict[str, object]`.

- [ ] **Step 1: Write a valid Stage 6A fixture helper and failing success test**

Build a valid `GeometryExecutionPlan` using the existing dataclasses and writer, then write minimal summary/metadata/RGB evidence with these exact invariants:

```python
summary = {
    "protocol": STAGE_6A_PROTOCOL_VERSION,
    "status": "success",
    "target_name": "cube",
    "backend": "geometry",
    "world_surface_point": list(plan.perception.world_surface_point),
    "simulation_steps_after_capture": 0,
    "scientific_gate_passed": True,
}
metadata = {
    "protocol": STAGE_6A_PROTOCOL_VERSION,
    "status": "success",
    "config": {"seed": 42, "target_name": "cube", "prompt": "red cube", "backend": "geometry"},
    "scene": {"object_poses": {"cube": {"position": [0.4800002872798181, -5.134833814891427e-07, 0.649968798272667]}}},
    "rgb_sha256": plan.rgb_sha256,
    "backprojection": {"world_x": plan.perception.world_surface_point[0], "world_y": plan.perception.world_surface_point[1], "world_z": plan.perception.world_surface_point[2]},
    "summary": summary,
    "simulation_steps_after_capture": 0,
    "motor_control_executed": False,
    "trajectory_executed": False,
    "gripper_closed": False,
    "contact_evaluated": False,
    "object_lifted": False,
    "physical_grasp_executed": False,
}
```

The test hashes every source file before and after the run and asserts:

```python
result = run_center_bias_diagnostic(CenterBiasDiagnosticConfig(source_dir, output_dir))
assert result["status"] == "success"
assert result["diagnostic_only"] is True
assert result["measurement"]["xy_offset_m"] == pytest.approx(0.026549556836982145)
assert result["measurement"]["xy_within_reference_threshold"] is False
assert source_hashes_after == source_hashes_before
assert not (source_dir / "center_bias_diagnostic.json").exists()
```

- [ ] **Step 2: Verify the runner test fails because the module is missing**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_center_bias_runner.py::test_offline_runner_preserves_source_and_writes_diagnostic -v
```

Expected: collection error for missing `run_center_bias_diagnostic`.

- [ ] **Step 3: Implement config, cleanup, hashing, and strict input loading**

`CenterBiasDiagnosticConfig.__post_init__` resolves paths and rejects identical or nested source/output directories, plus any `evidence_role` outside `("formal", "reproducibility")`. At run start, create only `output_dir`; remove stale `center_bias_diagnostic.json`, `center_bias_diagnostic.csv`, and output `metadata.json`. Read JSON as mappings, strictly load the plan, hash all four source files, and verify the actual `rgb.png` digest.

- [ ] **Step 4: Implement cross-file validation and successful output**

Validate protocol/status, seed/target/backend/prompt, RGB digest, world point, nested summary equality, zero post-capture steps, and every false execution flag. Compute the measurement from `metadata["scene"]["object_poses"]["cube"]["position"]`. Write a payload containing source paths/hashes, frozen protocol facts, `asdict(measurement)`, `diagnostic_only=True`, `plan_modified=False`, `scientific_gate_reinterpreted=False`, the historical Stage 6A gate value, and all execution flags false. Rehash sources after output and fail if any changed.

- [ ] **Step 5: Run the success test and verify it passes**

Run the Task 2 single-test command. Expected: PASS.

- [ ] **Step 6: Add failing mismatch and failure-preservation tests**

```python
@pytest.mark.parametrize("mutation", ["rgb_hash", "world_point", "execution_flag", "protocol"])
def test_offline_runner_rejects_inconsistent_stage_6a_evidence(tmp_path, mutation):
    source_dir, output_dir = write_stage_6a_fixture(tmp_path)
    mutate_fixture(source_dir, mutation)
    result = run_center_bias_diagnostic(CenterBiasDiagnosticConfig(source_dir, output_dir))
    assert result["status"] == "failure"
    assert result["failure_stage"] in {"input_validation", "source_integrity"}
    assert not (output_dir / "center_bias_diagnostic.json").exists()
    assert json.loads((output_dir / "metadata.json").read_text())["physical_grasp_executed"] is False


def test_offline_runner_rejects_source_output_overlap(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        CenterBiasDiagnosticConfig(tmp_path, tmp_path)
```

- [ ] **Step 7: Run and verify at least one new test fails for missing validation**

Run the full Task 2 test file. Expected: FAIL at the newly required mismatch/overlap behavior.

- [ ] **Step 8: Implement failure metadata and all mismatch checks**

Catch input/validation exceptions only at the runner boundary. Return and write metadata with `status="failure"`, a precise `failure_stage`, `failure_reason`, `diagnostic_only=True`, and every execution flag false. Never remove or write within `source_dir`.

- [ ] **Step 9: Verify Task 1–2 tests and forbidden-import boundary**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_center_bias_diagnostic.py \
  tests/simulation/test_pybullet_center_bias_runner.py -v
! rg -n 'import (pybullet|torch|transformers)|stepSimulation|setJointMotorControl|POSITION_CONTROL' \
  src/simulation/pybullet/center_bias_diagnostic.py \
  src/simulation/pybullet/run_center_bias_diagnostic.py
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/center_bias_diagnostic.py \
  src/simulation/pybullet/run_center_bias_diagnostic.py
```

Expected: tests and compile exit 0; `rg` finds no forbidden dependency or action.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/simulation/pybullet/run_center_bias_diagnostic.py \
  tests/simulation/test_pybullet_center_bias_runner.py
git commit -m "feat: audit frozen stage 6a center bias"
```

---

### Task 3: Formal evidence, isolated reproducibility, and project records

**Files:**
- Modify: `src/simulation/pybullet/README.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/agent/vlm_robotic_grasp_study_plan.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Generate but do not commit: `data/processed/pybullet/grasp_execution/stage_6a1_center_bias_diagnostic/*`
- Generate but do not commit: `data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight_reproducibility/*`
- Generate but do not commit: `data/processed/pybullet/grasp_execution/stage_6a1_center_bias_reproducibility/*`

**Interfaces:**
- Consumes: Task 2 CLI and the existing Stage 6A CLI.
- Produces: formal and reproducibility diagnostic evidence plus accurate Chinese project records.

- [ ] **Step 1: Hash the complete formal Stage 6A source directory before audit**

```bash
find data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight \
  -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/stage6a-before.sha256
```

- [ ] **Step 2: Run the formal offline Stage 6A.1 audit**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m \
  src.simulation.pybullet.run_center_bias_diagnostic \
  --source-dir data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight \
  --output-dir data/processed/pybullet/grasp_execution/stage_6a1_center_bias_diagnostic \
  --evidence-role formal
```

Expected: success, XY offset near `0.026550 m`, nominal-top Z offset near `0.000509 m`, XY reference gate false, all execution flags false.

- [ ] **Step 3: Prove the formal Stage 6A directory is unchanged**

Repeat the Step 1 pipeline to `/tmp/stage6a-after.sha256`, then run:

```bash
diff -u /tmp/stage6a-before.sha256 /tmp/stage6a-after.sha256
```

Expected: exit 0 with no differences.

- [ ] **Step 4: Independently inspect formal JSON, CSV, metadata, and source hashes**

Use a read-only Python one-liner to assert the protocol, evidence role, offsets, false flags, and SHA-256 values against the current source files. Do not edit outputs after this check.

- [ ] **Step 5: Run an isolated CUDA Stage 6A reproducibility trial**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m \
  src.simulation.pybullet.run_geometry_execution_preflight \
  --device cuda \
  --output-dir data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight_reproducibility
```

Expected: real Grounding DINO CUDA run completes and writes a separate successful Stage 6A plan. If CUDA or the real model is unavailable, record reproducibility as not run; do not substitute a mocked/CPU result.

- [ ] **Step 6: Diagnose the isolated reproducibility trial**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m \
  src.simulation.pybullet.run_center_bias_diagnostic \
  --source-dir data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight_reproducibility \
  --output-dir data/processed/pybullet/grasp_execution/stage_6a1_center_bias_reproducibility \
  --evidence-role reproducibility
```

Expected: results remain separate from formal evidence. Compare localization, predicted point, XY offset, nominal-top Z offset, and RGB hash without automatically declaring equivalence.

- [ ] **Step 7: Update README, structure, status, plan, and worklog from verified outputs only**

Document the new modules and commands, formal measured offsets, unchanged-source proof, and reproducibility outcome. Preserve all existing user edits in these already modified files and state that the diagnostic does not change Stage 6A static safety or constitute grasp success.

- [ ] **Step 8: Run focused and full verification**

```bash
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest \
  tests/simulation/test_pybullet_center_bias_diagnostic.py \
  tests/simulation/test_pybullet_center_bias_runner.py \
  tests/simulation/test_pybullet_geometry_execution_preflight.py \
  tests/simulation/test_pybullet_execution_plan.py -v
/home/pzk/miniconda/envs/msc-grasp/bin/python -m pytest -q
/home/pzk/miniconda/envs/msc-grasp/bin/python -m py_compile \
  src/simulation/pybullet/center_bias_diagnostic.py \
  src/simulation/pybullet/run_center_bias_diagnostic.py
git diff --check
```

Expected: all tests and compile commands exit 0; `git diff --check` reports no whitespace errors.

- [ ] **Step 9: Commit code and verified records without generated data or LaTeX artifacts**

```bash
git add src/simulation/pybullet/README.md docs/agent/PROJECT_STRUCTURE.md \
  docs/agent/CURRENT_STATUS.md docs/agent/vlm_robotic_grasp_study_plan.md \
  docs/worklog/WORKLOG.md \
  docs/superpowers/plans/2026-08-02-pybullet-stage-6a1-center-bias-diagnostic.md
git commit -m "docs: record stage 6a1 center bias evidence"
```

## Plan self-review

- Every design requirement maps to a task: pure computation and frozen constants (Task 1), strict offline orchestration and failure preservation (Task 2), formal source-integrity proof and isolated rerun (Task 3).
- The plan introduces no Stage 6B execution, center correction, multi-head backend, or success-rate claim.
- Interfaces consistently use `CenterBiasMeasurement`, `CenterBiasDiagnosticConfig`, and `run_center_bias_diagnostic`.
- The formal top reference is consistently named nominal and derived from saved truth center plus the frozen `0.025 m` half extent.
- No placeholders or unspecified error-handling steps remain.
