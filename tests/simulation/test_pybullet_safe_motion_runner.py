import csv
import json
from pathlib import Path

import cv2
import pytest

from src.simulation.pybullet.kinematic_audit import IKPoseAudit
from src.simulation.pybullet.run_safe_motion_smoke import (
    SafeMotionSmokeConfig,
    run_safe_motion_smoke,
)


def test_real_safe_motion_runner_writes_audited_round_trip(
    tmp_path: Path,
) -> None:
    summary = run_safe_motion_smoke(
        SafeMotionSmokeConfig(output_dir=tmp_path)
    )

    required = (
        "state_trace.csv",
        "summary.json",
        "metadata.json",
        "start.png",
        "waypoint.png",
        "return.png",
    )
    assert all((tmp_path / name).stat().st_size > 0 for name in required)
    with (tmp_path / "state_trace.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        trace = list(csv.DictReader(handle))
    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )

    assert summary["stage"] == "safe_motion_smoke"
    assert summary["waypoint_lift_m"] == pytest.approx(0.05)
    assert summary["preflight_ik_fk_passed"] is True
    assert summary["preflight_clearance_passed"] is True
    assert summary["outbound_reached"] is True
    assert summary["return_reached"] is True
    assert summary["maximum_finger_open_error_m"] <= 0.001
    assert summary["scientific_gate_passed"] is True
    assert len(trace) >= 480
    assert {row["phase"] for row in trace} == {"outbound", "return"}
    assert metadata["motor_control_executed"] is True
    assert metadata["simulation_stepped"] is True
    assert metadata["perception_executed"] is False
    assert metadata["target_approach_executed"] is False
    assert metadata["gripper_closed"] is False
    assert metadata["contact_evaluated"] is False
    assert metadata["object_lifted"] is False
    assert metadata["physical_grasp_executed"] is False
    for name in ("start.png", "waypoint.png", "return.png"):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None
        assert image.shape[:2] == (480, 640)


def test_preflight_ik_failure_is_saved_without_motor_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_ik = IKPoseAudit(
        solution=None,
        limits_passed=False,
        position_error_m=None,
        orientation_error_degrees=None,
        fk_passed=False,
        gate_passed=False,
        failure_reason="injected_ik_failure",
    )
    monkeypatch.setattr(
        "src.simulation.pybullet.run_safe_motion_smoke.audit_pose_ik",
        lambda *args, **kwargs: failed_ik,
    )

    summary = run_safe_motion_smoke(
        SafeMotionSmokeConfig(output_dir=tmp_path)
    )
    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )

    assert summary["scientific_gate_passed"] is False
    assert summary["failure_stage"] == "preflight_ik"
    assert summary["outbound_reached"] is False
    assert summary["return_reached"] is False
    assert metadata["motor_control_executed"] is False
    assert metadata["trajectory_executed"] is False
    assert metadata["gripper_closed"] is False
    assert metadata["physical_grasp_executed"] is False
    assert (tmp_path / "state_trace.csv").is_file()
    assert (tmp_path / "start.png").stat().st_size > 0
