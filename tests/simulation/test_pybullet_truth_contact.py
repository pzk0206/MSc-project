import csv
import json
import math
from pathlib import Path

import cv2

from src.simulation.pybullet.run_truth_contact import (
    TruthContactConfig,
    run_truth_contact,
)


def test_real_truth_contact_descends_closes_and_holds_bilateral_contact(
    tmp_path: Path,
) -> None:
    summary = run_truth_contact(TruthContactConfig(output_dir=tmp_path))

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

    assert summary["stage"] == "cube_truth_bilateral_contact"
    assert summary["target_stability_preflight_passed"] is True
    assert summary["preflight_ik_fk_passed"] is True
    assert summary["preflight_clearance_passed"] is True
    assert summary["pregrasp_reached"] is True
    assert summary["approach_reached"] is True
    assert summary["grasp_depth_reached"] is True
    assert summary["grasp_depth_gate_passed"] is True
    assert summary["gripper_close_executed"] is True
    assert summary["left_finger_contacted"] is True
    assert summary["right_finger_contacted"] is True
    assert summary["bilateral_contact_acquired"] is True
    assert summary["trailing_bilateral_contact_steps"] >= 60
    assert summary["target_contact_gate_passed"] is True
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
    }
    assert contacts
    assert {int(row["robot_link"]) for row in contacts} == {9, 10}
    assert all(
        math.isfinite(float(row["normal_force"]))
        and float(row["normal_force"]) > 0.0
        for row in contacts
    )
    assert metadata["truth_target_used"] is True
    assert metadata["perception_executed"] is False
    assert metadata["target_approach_executed"] is True
    assert metadata["vertical_approach_executed"] is True
    assert metadata["descent_to_contact_executed"] is True
    assert metadata["gripper_close_commanded"] is True
    assert metadata["gripper_closed"] is True
    assert metadata["contact_evaluated"] is True
    assert metadata["target_contacted"] is True
    assert metadata["object_lifted"] is False
    assert metadata["physical_grasp_executed"] is False
    for name in (
        "start.png",
        "pregrasp.png",
        "approach.png",
        "grasp_depth.png",
        "closed.png",
    ):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None
        assert image.shape[:2] == (480, 640)
