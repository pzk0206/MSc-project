import csv
import json
import math
from pathlib import Path

import cv2

from src.simulation.pybullet.run_truth_lift import (
    TruthLiftConfig,
    run_truth_lift,
)


def test_real_truth_lift_executes_and_holds_cube_off_table(
    tmp_path: Path,
) -> None:
    summary = run_truth_lift(TruthLiftConfig(output_dir=tmp_path))

    required = (
        "state_trace.csv",
        "contact_events.csv",
        "summary.json",
        "metadata.json",
        "start.png",
        "pregrasp.png",
        "approach.png",
        "grasp_depth.png",
        "closed.png",
        "lifted.png",
        "lift_hold.png",
    )
    assert all((tmp_path / name).stat().st_size > 0 for name in required)
    with (tmp_path / "state_trace.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        trace = list(csv.DictReader(handle))
    with (tmp_path / "contact_events.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        contacts = list(csv.DictReader(handle))
    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )

    assert summary["stage"] == "cube_truth_lift_hold"
    assert summary["pregrasp_reached"] is True
    assert summary["approach_reached"] is True
    assert summary["grasp_depth_reached"] is True
    assert summary["target_contact_gate_passed"] is True
    assert summary["lift_preflight_ik_fk_passed"] is True
    assert summary["lift_preflight_clearance_passed"] is True
    assert summary["lift_executed"] is True
    assert summary["lift_reached"] is True
    assert summary["object_lift_gate_passed"] is True
    assert summary["table_release_gate_passed"] is True
    assert summary["relative_stability_gate_passed"] is True
    assert summary["lift_hold_gate_passed"] is True
    assert summary["physical_grasp_success"] is True
    assert summary["minimum_hold_object_lift_m"] >= 0.10
    assert summary["hold_target_table_contact_count"] == 0
    assert summary["maximum_hold_relative_drift_m"] <= 0.01
    assert summary["trailing_lift_bilateral_contact_steps"] >= 120
    assert summary["environment_collision_count"] == 0
    assert summary["self_collision_count"] == 0
    assert summary["prohibited_target_contact_count"] == 0
    assert summary["scientific_gate_passed"] is True
    assert {row["phase"] for row in trace} == {
        "pregrasp",
        "approach",
        "grasp_depth",
        "close",
        "contact_hold",
        "lift",
        "lift_hold",
    }
    lift_hold_rows = [
        row for row in trace if row["phase"] == "lift_hold"
    ]
    assert len(lift_hold_rows) == 240
    assert all(float(row["target_lift_m"]) >= 0.10 for row in lift_hold_rows)
    assert all(row["target_table_contact"] == "False" for row in lift_hold_rows)
    assert all(
        float(row["relative_drift_m"]) <= 0.01
        for row in lift_hold_rows
    )
    assert contacts
    assert {row["phase"] for row in contacts} >= {
        "close",
        "contact_hold",
        "lift",
        "lift_hold",
    }
    assert {int(row["robot_link"]) for row in contacts} == {9, 10}
    assert all(
        math.isfinite(float(row["normal_force"]))
        and float(row["normal_force"]) > 0.0
        for row in contacts
    )
    assert metadata["truth_target_used"] is True
    assert metadata["perception_executed"] is False
    assert metadata["gripper_close_commanded"] is True
    assert metadata["gripper_closed"] is True
    assert metadata["contact_evaluated"] is True
    assert metadata["target_contacted"] is True
    assert metadata["object_lifted"] is True
    assert metadata["physical_grasp_executed"] is True
    for name in (
        "start.png",
        "pregrasp.png",
        "approach.png",
        "grasp_depth.png",
        "closed.png",
        "lifted.png",
        "lift_hold.png",
    ):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None
        assert image.shape[:2] == (480, 640)
