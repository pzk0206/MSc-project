import csv
import json
from pathlib import Path

import cv2
import pytest

from src.simulation.pybullet.pose_generation import (
    generate_top_down_pose_from_world_point,
)
from src.simulation.pybullet.run_truth_pregrasp import (
    TruthPregraspConfig,
    run_truth_pregrasp,
)


def test_truth_world_point_generates_top_down_cube_pregrasp() -> None:
    candidate = generate_top_down_pose_from_world_point(
        target="cube",
        backend="ground_truth",
        surface_point=(0.48, 0.0, 0.685),
        finger_axis_world=(0.8660254038, 0.5, 0.0),
    )

    assert candidate.surface_standoff_pose.position == pytest.approx(
        (0.48, 0.0, 0.705)
    )
    assert candidate.pregrasp_pose.position == pytest.approx(
        (0.48, 0.0, 0.805)
    )
    assert candidate.finger_axis_world == pytest.approx(
        (0.8660254038, 0.5, 0.0)
    )
    assert candidate.approach_axis_world == pytest.approx((0.0, 0.0, -1.0))


def test_real_truth_pregrasp_reaches_stable_cube_with_open_fingers(
    tmp_path: Path,
) -> None:
    summary = run_truth_pregrasp(
        TruthPregraspConfig(output_dir=tmp_path)
    )

    required = (
        "state_trace.csv",
        "contact_events.csv",
        "summary.json",
        "metadata.json",
        "start.png",
        "pregrasp.png",
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

    assert (tmp_path / "contact_events.csv").read_text(
        encoding="utf-8"
    ) == "step,phase,robot_link,target_body,normal_force\n"

    assert summary["stage"] == "cube_truth_pregrasp"
    assert summary["target_stability_preflight_passed"] is True
    assert summary["preflight_ik_fk_passed"] is True
    assert summary["preflight_clearance_passed"] is True
    assert summary["pregrasp_reached"] is True
    assert summary["endpoint_pose_gate_passed"] is True
    assert summary["target_xy_gate_passed"] is True
    assert summary["target_undisturbed_gate_passed"] is True
    assert summary["fingers_open_gate_passed"] is True
    assert summary["scientific_gate_passed"] is True
    assert len(trace) >= 240
    assert {row["phase"] for row in trace} == {"pregrasp"}
    assert all(row["cube_position"] for row in trace)
    assert all(row["tool_relative_to_cube"] for row in trace)
    assert metadata["truth_target_used"] is True
    assert metadata["perception_executed"] is False
    assert metadata["target_approach_executed"] is True
    assert metadata["descent_to_contact_executed"] is False
    assert metadata["gripper_close_commanded"] is False
    assert metadata["contact_evaluated"] is False
    assert metadata["object_lifted"] is False
    assert metadata["physical_grasp_executed"] is False
    for name in ("start.png", "pregrasp.png"):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None
        assert image.shape[:2] == (480, 640)
