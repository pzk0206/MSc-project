import csv
import json
from pathlib import Path
import shutil

import pytest

from src.simulation.pybullet.run_pose_ik_study import (
    PoseIKStudyConfig,
    load_pose_ik_inputs,
    run_pose_ik_study,
)


SOURCE = Path("data/processed/pybullet/multi_object_study")


def _copy_inputs(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("backend_results.csv", "backprojection_results.csv", "metadata.json"):
        shutil.copyfile(SOURCE / name, destination / name)


def test_input_contract_accepts_the_verified_nine_rows(tmp_path: Path) -> None:
    _copy_inputs(tmp_path)

    inputs = load_pose_ik_inputs(tmp_path)

    assert len(inputs.backend_rows) == 9
    assert len(inputs.backprojection_rows) == 9
    assert inputs.metadata["scene"]["config"]["seed"] == 42


def test_input_contract_rejects_a_false_prior_gate(tmp_path: Path) -> None:
    _copy_inputs(tmp_path)
    path = tmp_path / "backprojection_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["gate_passed"] = "False"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="prior backprojection gate"):
        load_pose_ik_inputs(tmp_path)


def test_real_offline_runner_writes_18_non_execution_rows(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _copy_inputs(input_dir)

    summary = run_pose_ik_study(
        PoseIKStudyConfig(input_dir=input_dir, output_dir=output_dir)
    )

    with (output_dir / "pose_ik_candidates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(
        (output_dir / "pose_ik_metadata.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 18
    assert [(row["target"], row["backend"], row["symmetry_degrees"]) for row in rows[:2]] == [
        ("duck", "geometry", "0.0"),
        ("duck", "geometry", "180.0"),
    ]
    assert summary["candidate_count"] == 18
    assert summary["input_grasp_count"] == 9
    assert summary["ik_fk_passed_count"] == 18
    assert summary["clearance_passed_count"] == 12
    assert summary["selected_count"] == 6
    assert summary["scientific_gate_passed"] is False
    assert metadata["simulation_setup_steps"] == 60
    assert metadata["simulation_stepped_during_candidate_audit"] is False
    assert metadata["motor_control_executed"] is False
    assert metadata["trajectory_executed"] is False
    assert metadata["gripper_closed"] is False
    assert metadata["physical_grasp_executed"] is False
    assert metadata["ik_solver_called"] is True
    assert metadata["joint_states_set_for_static_audit"] is True
    assert set(metadata["input_sha256"]) == {
        "backend_results.csv",
        "backprojection_results.csv",
        "metadata.json",
    }
