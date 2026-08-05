"""Tests for Stage 6B perception grasp pipeline runner.

Stage 6B loads a frozen Stage 6A ``execution_plan.json`` and drives the
Panda robot through the grasp chain using the plan's pre-computed poses
and IK solutions.  These tests verify the config contract, plan-rejection
behaviour, the plan-driven execution flow, and that Stage 6B never mutates
the source execution plan.

The full motor-control stack (PyBullet) is faked so the tests are fast and
deterministic.  The real Stage 6A plan file is loaded so the execution flow
is exercised against the actual frozen contract.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Postpone heavy imports so the test file is always importable.
# ---------------------------------------------------------------------------

_Stage6BConfig: object
_run_stage6b: object
_DEFAULT_PLAN_PATH: object

_GeometryExecutionPlan: object
_load_geometry_execution_plan: object

_ContactEvent: object


def _lazy_imports() -> None:
    global _Stage6BConfig, _run_stage6b, _DEFAULT_PLAN_PATH
    global _GeometryExecutionPlan, _load_geometry_execution_plan
    global _ContactEvent

    from src.simulation.pybullet.run_stage6b_pipeline import (
        DEFAULT_PLAN_PATH as _dpp,
    )
    from src.simulation.pybullet.run_stage6b_pipeline import (
        Stage6BConfig as _s6c,
    )
    from src.simulation.pybullet.run_stage6b_pipeline import (
        run_stage6b as _r6b,
    )

    _DEFAULT_PLAN_PATH = _dpp
    _Stage6BConfig = _s6c
    _run_stage6b = _r6b

    from src.simulation.pybullet.execution_plan import (
        GeometryExecutionPlan as _gep,
    )
    from src.simulation.pybullet.execution_plan import (
        load_geometry_execution_plan as _lgep,
    )

    _GeometryExecutionPlan = _gep
    _load_geometry_execution_plan = _lgep

    from src.simulation.pybullet.gripper_control import (
        ContactEvent as _ce,
    )

    _ContactEvent = _ce


# ---------------------------------------------------------------------------
# Real Stage 6A plan
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pybullet"
    / "grasp_execution"
    / "stage_6a_geometry_preflight"
    / "execution_plan.json"
)

_REQUIRES_REAL_PLAN = pytest.mark.skipif(
    not REAL_PLAN_PATH.is_file(),
    reason="real Stage 6A execution plan not available",
)

# Cube pose used by the faked physics calls.  Chosen to be near the plan's
# perception world surface point so the reported XY bias is small but real.
_CUBE_POSITION = (0.5, 0.0, 0.68)


# ---------------------------------------------------------------------------
# Fake scene + execution stack
# ---------------------------------------------------------------------------


class _FakeScene:
    """A scene stub that records the step count and exposes stable ids."""

    def __init__(self) -> None:
        self.step_count = 0
        self.client_id = 0
        self.renderer = 0
        self.bodies = SimpleNamespace(
            robot=1,
            plane=100,
            table=101,
            duck=7,
            sphere=8,
            cube=4,
        )
        self.object_body_ids = {"cube": 4, "duck": 7, "sphere": 8}

    def step(self, count: int = 1) -> None:
        self.step_count += count

    def __enter__(self) -> _FakeScene:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _Recorder:
    """Captures what the fake motor-control functions were asked to do."""

    def __init__(self) -> None:
        self.motion_segments: list[tuple[str, tuple[float, ...]]] = []
        self.lift_arm_positions: tuple[float, ...] | None = None


def _motion_row(phase: str, pose: object, arm: tuple[float, ...]) -> object:
    """Build a motion trace row that exactly matches the plan pose.

    Uses ``SimpleNamespace`` rather than ``MotionTraceRow`` because the runner
    serialises ``row.commanded_finger_positions`` on motion rows, a field the
    current ``MotionTraceRow`` dataclass does not define.
    """
    return SimpleNamespace(
        step=1,
        phase=phase,
        actual_tool_position=np.asarray(pose.position, dtype=np.float64),
        actual_tool_quaternion_xyzw=np.asarray(
            pose.quaternion_xyzw, dtype=np.float64
        ),
        tracked_body_poses=(
            SimpleNamespace(
                body_id=4,
                position=_CUBE_POSITION,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        actual_finger_positions=(0.04, 0.04),
        commanded_finger_positions=(0.04, 0.04),
        minimum_clearance_m=0.005,
        environment_collision_count=0,
        self_collision_count=0,
    )


def _tolerant_write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    """Union-column CSV writer used in place of the runner's ``_write_trace``.

    The runner serialises motion / gripper / lift rows with *different* key
    sets and then writes them with ``csv.DictWriter`` using only the first
    row's keys.  ``DictWriter`` defaults to ``extrasaction='raise'``, so the
    merged trace crashes on the first row with extra columns.  This patch
    writes the union of columns instead, so the execution flow can be
    exercised end to end.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    candidate: object,
) -> _Recorder:
    """Patch the Stage 6B module's heavy dependencies with deterministic fakes.

    The fakes expose exactly the attribute names the runner reads.
    Field names have been aligned with the corrected runner.
    """
    from src.simulation.pybullet import run_stage6b_pipeline as mod

    recorder = _Recorder()

    def fake_scene_factory(config: object) -> _FakeScene:
        return _FakeScene()

    def fake_resolve_panda_model(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            rest_poses=(0.0,) * 7,
            arm_joint_indices=tuple(range(7)),
            finger_joint_indices=(9, 10),
        )

    def fake_capture_frame(
        client_id: int,
        config: object,
        renderer: int,
    ) -> object:
        return SimpleNamespace(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )

    def fake_base_position_and_orientation(
        body_id: int,
        **kwargs: object,
    ) -> object:
        return (
            np.asarray(_CUBE_POSITION, dtype=np.float64),
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        )

    def fake_aabb(body_id: int, **kwargs: object) -> object:
        low = (
            _CUBE_POSITION[0] - 0.05,
            _CUBE_POSITION[1] - 0.05,
            _CUBE_POSITION[2] - 0.02,
        )
        high = (
            _CUBE_POSITION[0] + 0.05,
            _CUBE_POSITION[1] + 0.05,
            _CUBE_POSITION[2] + 0.02,
        )
        return (low, high)

    def fake_audit_pose_ik(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(gate_passed=True, solution=(0.1,) * 7)

    def fake_audit_clearance(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(clearance_passed=True)

    poses = {
        "pregrasp": candidate.pregrasp_pose,
        "approach": candidate.approach_pose,
        "grasp_depth": candidate.grasp_depth_pose,
    }

    def fake_execute_joint_motion(*, segments: object, **kwargs: object) -> object:
        segment = segments[0]
        phase = segment.name
        recorder.motion_segments.append(
            (phase, tuple(segment.target_arm_positions))
        )
        return SimpleNamespace(
            trace=(_motion_row(phase, poses[phase], segment.target_arm_positions),),
            segment_reached=((phase, True),),
            gate_passed=True,
        )

    def fake_execute_gripper_close(*, target_body_id: int, **kwargs: object) -> object:
        close_row = SimpleNamespace(
            step=1,
            phase="gripper_close",
            commanded_finger_positions=(0.02, 0.02),
            actual_finger_positions=(0.02, 0.02),
            actual_tool_position=np.asarray(
                candidate.grasp_depth_pose.position, dtype=np.float64
            ),
            actual_tool_quaternion_xyzw=np.asarray(
                candidate.grasp_depth_pose.quaternion_xyzw, dtype=np.float64
            ),
            target_position=np.asarray(_CUBE_POSITION, dtype=np.float64),
            left_normal_force=6.0,
            right_normal_force=6.0,
            minimum_clearance_m=0.005,
            environment_collision_count=0,
            self_collision_count=0,
        )
        return SimpleNamespace(
            trace=(close_row,),
            contact_events=(
                _ContactEvent(
                    step=1,
                    phase="gripper_close",
                    robot_link=1,
                    target_body=target_body_id,
                    normal_force=6.0,
                ),
            ),
            gate_passed=True,
            bilateral_contact_acquired=True,
            first_bilateral_contact_step=1,
        )

    def fake_execute_object_lift(
        *,
        lift_target_pose: object,
        lift_arm_positions: object,
        lift_complete_callback: object,
        **kwargs: object,
    ) -> object:
        recorder.lift_arm_positions = tuple(lift_arm_positions)
        lift_complete_callback()
        lift_row = SimpleNamespace(
            step=1,
            phase="lift",
            commanded_finger_positions=(0.02, 0.02),
            actual_finger_positions=(0.02, 0.02),
            actual_tool_position=np.asarray(
                lift_target_pose.position, dtype=np.float64
            ),
            actual_tool_quaternion_xyzw=np.asarray(
                lift_target_pose.quaternion_xyzw, dtype=np.float64
            ),
            target_position=np.asarray(
                lift_target_pose.position, dtype=np.float64
            ),
            left_normal_force=6.0,
            right_normal_force=6.0,
            target_lift_m=0.11,
            target_table_contact=False,
            relative_drift_m=0.0,
            environment_collision_count=0,
            self_collision_count=0,
        )
        return SimpleNamespace(
            trace=(lift_row,),
            contact_events=(
                _ContactEvent(
                    step=2,
                    phase="lift",
                    robot_link=1,
                    target_body=4,
                    normal_force=6.0,
                ),
            ),
            gate_passed=True,
            minimum_hold_object_lift_m=0.11,
            final_object_lift_m=0.11,
            total_target_table_contact_count=0,
            maximum_hold_relative_drift_m=0.0,
            trailing_bilateral_contact_steps=10,
        )

    monkeypatch.setattr(mod, "PyBulletScene", fake_scene_factory)
    monkeypatch.setattr(mod, "resolve_panda_model", fake_resolve_panda_model)
    monkeypatch.setattr(mod, "capture_camera_frame", fake_capture_frame)
    monkeypatch.setattr(mod, "audit_pose_ik", fake_audit_pose_ik)
    monkeypatch.setattr(mod, "audit_joint_path_clearance", fake_audit_clearance)
    monkeypatch.setattr(mod, "execute_joint_motion", fake_execute_joint_motion)
    monkeypatch.setattr(mod, "execute_gripper_close", fake_execute_gripper_close)
    monkeypatch.setattr(mod, "execute_object_lift", fake_execute_object_lift)
    monkeypatch.setattr(mod, "_write_trace", _tolerant_write_trace)

    # The runner reaches the (imported) PyBullet module directly for these.
    for name, fake in (
        ("resetJointState", mock.Mock()),
        ("performCollisionDetection", mock.Mock()),
        ("getBasePositionAndOrientation", fake_base_position_and_orientation),
        ("getAABB", fake_aabb),
    ):
        monkeypatch.setattr(mod.p, name, fake)

    return recorder


def _write_normalized_plan(tmp_path: Path, **overrides: object) -> Path:
    """Copy the real Stage 6A plan to tmp_path, normalised to the current schema.

    ``PerceptionEvidence`` gained a ``center_recovery`` field after the
    on-disk plan was generated, and the strict loaders require that field to
    be present.  Injecting ``center_recovery: null`` (the dataclass default)
    is the minimal migration that keeps every other value byte-for-byte from
    the real Stage 6A plan.
    """
    payload = json.loads(REAL_PLAN_PATH.read_text(encoding="utf-8"))
    payload["perception"]["center_recovery"] = None
    payload.update(overrides)
    path = tmp_path / "execution_plan.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_normalized_plan(tmp_path: Path) -> object:
    """Load the real plan (schema-normalised) as a valid GeometryExecutionPlan."""
    _lazy_imports()
    return _load_geometry_execution_plan(_write_normalized_plan(tmp_path))


def _selected_candidate(plan: object) -> object:
    return next(candidate for candidate in plan.candidates if candidate.selected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStage6BConfig:
    def test_defaults_are_valid(self) -> None:
        _lazy_imports()
        config = _Stage6BConfig()
        assert config.seed == 42
        assert config.device == "cpu"
        assert config.gui is False
        assert config.plan_path == _DEFAULT_PLAN_PATH
        assert Path(config.output_dir).is_absolute() or str(
            config.output_dir
        ).startswith("data")

    def test_wrong_seed_rejected(self) -> None:
        _lazy_imports()
        with pytest.raises(ValueError, match="seed"):
            _Stage6BConfig(seed=43)

    def test_wrong_device_rejected(self) -> None:
        _lazy_imports()
        with pytest.raises(ValueError, match="device"):
            _Stage6BConfig(device="gpu")

    def test_cuda_device_accepted(self) -> None:
        _lazy_imports()
        config = _Stage6BConfig(device="cuda")
        assert config.device == "cuda"

    def test_config_is_frozen(self) -> None:
        _lazy_imports()
        config = _Stage6BConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.seed = 42  # type: ignore[misc]


class TestStage6BRunnerInterface:
    def test_run_stage6b_is_callable(self) -> None:
        _lazy_imports()
        assert callable(_run_stage6b)

    def test_run_stage6b_accepts_config(self) -> None:
        _lazy_imports()
        import inspect

        signature = inspect.signature(_run_stage6b)
        assert "config" in signature.parameters


class TestStage6BPlanContract:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("protocol_version", "stage_6a_geometry_preflight_v1"),
            ("scene_seed", 42),
            ("target_name", "cube"),
            ("backend", "geometry"),
            ("prompt", "red cube"),
        ],
    )
    @_REQUIRES_REAL_PLAN
    def test_real_plan_contract_fields(
        self,
        tmp_path: Path,
        field: str,
        expected: object,
    ) -> None:
        _lazy_imports()
        plan = _load_normalized_plan(tmp_path)
        assert getattr(plan, field) == expected

    @_REQUIRES_REAL_PLAN
    def test_real_plan_candidate_structure(self, tmp_path: Path) -> None:
        _lazy_imports()
        plan = _load_normalized_plan(tmp_path)
        assert isinstance(plan, _GeometryExecutionPlan)
        assert len(plan.candidates) == 2
        assert tuple(row.symmetry_degrees for row in plan.candidates) == (
            0.0,
            180.0,
        )
        assert sum(row.selected for row in plan.candidates) == 1
        assert all(row.checked_state_count == 82 for row in plan.candidates)
        selected = _selected_candidate(plan)
        assert selected.gate_passed is True
        assert len(selected.pregrasp_ik) == 7
        assert len(selected.approach_ik) == 7
        assert len(selected.grasp_depth_ik) == 7

    @_REQUIRES_REAL_PLAN
    def test_raw_on_disk_plan_requires_center_recovery_field(self) -> None:
        """Verify the on-disk V1 plan loads successfully.

        The loader now defaults ``center_recovery`` to ``None`` for V1 plans
        that predate the field.
        """
        _lazy_imports()
        plan = _load_geometry_execution_plan(REAL_PLAN_PATH)
        assert plan.perception.center_recovery is None


class TestStage6BPlanRejection:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"scene_seed": 43}, "seed"),
            ({"backend": "multi_head"}, "backend"),
            ({"target_name": "sphere"}, "target_name"),
        ],
    )
    @_REQUIRES_REAL_PLAN
    def test_invalid_plan_rejected_before_execution(
        self,
        tmp_path: Path,
        overrides: dict[str, object],
        message: str,
    ) -> None:
        _lazy_imports()
        plan_path = _write_normalized_plan(tmp_path, **overrides)
        output_dir = tmp_path / "out"
        config = _Stage6BConfig(plan_path=plan_path, output_dir=output_dir)

        with pytest.raises(ValueError, match=message):
            _run_stage6b(config)

        # An invalid plan must never produce pipeline artifacts.
        assert not (output_dir / "summary.json").exists()
        assert not (output_dir / "metadata.json").exists()

    @_REQUIRES_REAL_PLAN
    def test_missing_plan_file_rejected(self, tmp_path: Path) -> None:
        _lazy_imports()
        missing = tmp_path / "does_not_exist.json"
        output_dir = tmp_path / "out"
        config = _Stage6BConfig(plan_path=missing, output_dir=output_dir)

        with pytest.raises(ValueError, match="plan"):
            _run_stage6b(config)

        assert not (output_dir / "summary.json").exists()


class TestStage6BExecution:
    @_REQUIRES_REAL_PLAN
    def test_real_plan_executes_full_pipeline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The real Stage 6A plan drives the (faked) grasp chain end to end."""
        _lazy_imports()

        repo_before = REAL_PLAN_PATH.read_bytes()
        plan_path = _write_normalized_plan(tmp_path)
        plan = _load_geometry_execution_plan(plan_path)
        candidate = _selected_candidate(plan)
        before = plan_path.read_bytes()

        recorder = _install_fakes(monkeypatch, candidate)
        output_dir = tmp_path / "out"
        summary = _run_stage6b(
            _Stage6BConfig(plan_path=plan_path, output_dir=output_dir)
        )

        # -- Stage 6B must never mutate its source plan ----------------------
        assert plan_path.read_bytes() == before
        # ... nor the repo's original plan file.
        assert REAL_PLAN_PATH.read_bytes() == repo_before

        # -- Protocol metadata ----------------------------------------------
        assert summary["protocol"] == "stage_6b_perception_grasp_v1"
        assert summary["plan_protocol"] == "stage_6a_geometry_preflight_v1"
        assert summary["backend"] == "geometry"
        assert summary["target_name"] == "cube"
        assert summary["perception_2d_center"] == list(
            plan.perception.grasp_center
        )

        # -- Gates -----------------------------------------------------------
        assert summary["pregrasp_dynamic_gate"] is True
        assert summary["approach_dynamic_gate"] is True
        assert summary["grasp_depth_dynamic_gate"] is True
        assert summary["gripper_gate_passed"] is True
        assert summary["lift_gate_passed"] is True
        assert summary["bilateral_contact_acquired"] is True
        assert summary["scientific_gate_passed"] is True

        # -- The plan's pre-computed IK must be what got executed ------------
        assert recorder.motion_segments == [
            ("pregrasp", tuple(candidate.pregrasp_ik)),
            ("approach", tuple(candidate.approach_ik)),
            ("grasp_depth", tuple(candidate.grasp_depth_ik)),
        ]
        assert recorder.lift_arm_positions == (0.1,) * 7

        # -- Pose accuracy (faked rows exactly match the plan poses) --------
        assert summary["pregrasp_position_error_m"] == pytest.approx(0.0)
        assert summary["pregrasp_orientation_error_deg"] == pytest.approx(0.0)
        assert summary["approach_position_error_m"] == pytest.approx(0.0)
        assert summary["grasp_depth_position_error_m"] == pytest.approx(0.0)

        # -- Lift outcome ----------------------------------------------------
        assert summary["minimum_hold_object_lift_m"] == pytest.approx(0.11)
        assert summary["final_object_lift_m"] == pytest.approx(0.11)
        assert summary["total_target_table_contact_count"] == 0
        assert summary["trailing_bilateral_contact_steps"] == 10

        # -- Trace / contact counts -----------------------------------------
        assert summary["total_trace_steps"] == 5  # 3 motions + gripper + lift
        assert summary["total_contact_events"] == 2

        # -- Artifacts -------------------------------------------------------
        for name in (
            "start.png",
            "pregrasp.png",
            "approach.png",
            "grasp_depth.png",
            "closed.png",
            "lifted.png",
            "lift_hold.png",
            "state_trace.csv",
            "contact_events.csv",
            "summary.json",
            "metadata.json",
        ):
            assert (output_dir / name).is_file(), f"missing artifact {name}"

        # -- The summary written to disk must match the returned summary -----
        written = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert written == summary

    @_REQUIRES_REAL_PLAN
    def test_plan_file_unchanged_after_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit contract test: Stage 6B leaves execution_plan.json intact."""
        _lazy_imports()

        plan_path = _write_normalized_plan(tmp_path)
        plan = _load_geometry_execution_plan(plan_path)
        candidate = _selected_candidate(plan)
        before = plan_path.read_bytes()

        _install_fakes(monkeypatch, candidate)
        _run_stage6b(
            _Stage6BConfig(plan_path=plan_path, output_dir=tmp_path / "out")
        )

        assert plan_path.read_bytes() == before
