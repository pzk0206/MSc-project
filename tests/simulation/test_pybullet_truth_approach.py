import csv
import json
from pathlib import Path

import cv2

from src.simulation.pybullet.run_truth_approach import (
    TruthApproachConfig,
    run_truth_approach,
)


def test_real_open_approach_descends_to_precontact_with_open_fingers(
    tmp_path: Path,
) -> None:
    summary = run_truth_approach(
        TruthApproachConfig(output_dir=tmp_path)
    )

    required = (
        "state_trace.csv",
        "contact_events.csv",
        "summary.json",
        "metadata.json",
        "start.png",
        "pregrasp.png",
        "approach.png",
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

    assert summary["stage"] == "cube_truth_open_approach"
    assert summary["target_stability_preflight_passed"] is True
    assert summary["preflight_ik_fk_passed"] is True
    assert summary["preflight_clearance_passed"] is True
    assert summary["pregrasp_reached"] is True
    assert summary["approach_reached"] is True
    assert summary["approach_endpoint_pose_gate_passed"] is True
    assert summary["target_xy_gate_passed"] is True
    assert summary["approach_height_gate_passed"] is True
    assert summary["target_undisturbed_gate_passed"] is True
    assert summary["fingers_open_gate_passed"] is True
    assert summary["gripper_close_command_count"] == 0
    assert summary["environment_collision_count"] == 0
    assert summary["self_collision_count"] == 0
    assert summary["scientific_gate_passed"] is True
    assert summary["approach_height_above_cube_top_m"] > 0.0
    assert abs(summary["approach_height_above_cube_top_m"] - 0.02) <= 0.005
    assert {row["phase"] for row in trace} == {"pregrasp", "approach"}
    approach_rows = [row for row in trace if row["phase"] == "approach"]
    first_height = json.loads(approach_rows[0]["actual_tool_position"])[2]
    final_height = json.loads(approach_rows[-1]["actual_tool_position"])[2]
    assert first_height > final_height
    assert all(row["cube_position"] for row in trace)
    assert all(row["tool_relative_to_cube"] for row in trace)
    assert (tmp_path / "contact_events.csv").read_text(
        encoding="utf-8"
    ) == "step,phase,robot_link,target_body,normal_force\n"
    assert metadata["truth_target_used"] is True
    assert metadata["perception_executed"] is False
    assert metadata["target_approach_executed"] is True
    assert metadata["vertical_approach_executed"] is True
    assert metadata["descent_to_contact_executed"] is False
    assert metadata["gripper_close_commanded"] is False
    assert metadata["contact_evaluated"] is False
    assert metadata["object_lifted"] is False
    assert metadata["physical_grasp_executed"] is False
    for name in ("start.png", "pregrasp.png", "approach.png"):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None
        assert image.shape[:2] == (480, 640)
