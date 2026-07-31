"""Run a static Panda pose, IK/FK, and collision-clearance audit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.backend_comparison import EXPECTED_TARGET_BACKENDS
from src.simulation.pybullet.kinematic_audit import (
    CandidateAudit,
    CollisionAudit,
    PandaModelInfo,
    audit_joint_path_clearance,
    audit_pose_ik,
    resolve_panda_model,
    select_candidate_pair,
)
from src.simulation.pybullet.pose_generation import (
    PoseCandidate,
    generate_top_down_pose_candidates,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene


DEFAULT_STUDY_DIR = Path("data/processed/pybullet/multi_object_study")


@dataclass(frozen=True)
class PoseIKStudyConfig:
    """Input and output locations for the independent offline audit."""

    input_dir: Path = DEFAULT_STUDY_DIR
    output_dir: Path = DEFAULT_STUDY_DIR


@dataclass(frozen=True)
class PoseIKInputs:
    """Validated upstream rows and fixed-study metadata."""

    backend_rows: tuple[dict[str, str], ...]
    backprojection_rows: tuple[dict[str, str], ...]
    metadata: dict[str, Any]


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"required pose/IK input does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _row_order(rows: tuple[dict[str, str], ...]) -> tuple[tuple[str, str], ...]:
    return tuple((row.get("target", ""), row.get("backend", "")) for row in rows)


def load_pose_ik_inputs(input_dir: Path) -> PoseIKInputs:
    """Load and strictly validate the saved nine-point study contract."""

    root = Path(input_dir)
    backend_rows = _read_csv(root / "backend_results.csv")
    backprojection_rows = _read_csv(root / "backprojection_results.csv")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"required pose/IK input does not exist: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _row_order(backend_rows) != EXPECTED_TARGET_BACKENDS:
        raise ValueError("backend rows must use exact target/backend order")
    if _row_order(backprojection_rows) != EXPECTED_TARGET_BACKENDS:
        raise ValueError(
            "backprojection rows must use exact target/backend order"
        )
    for backend, backprojection in zip(backend_rows, backprojection_rows):
        if not _is_true(backprojection.get("gate_passed", "")):
            raise ValueError("prior backprojection gate must pass for all rows")
        try:
            backend_x = float(backend["center_x"])
            backend_y = float(backend["center_y"])
            backprojection_x = float(backprojection["center_x"])
            backprojection_y = float(backprojection["center_y"])
            angle = float(backend["angle_degrees"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pose/IK input contains invalid numeric fields") from exc
        if not all(
            math.isfinite(value)
            for value in (
                backend_x,
                backend_y,
                backprojection_x,
                backprojection_y,
                angle,
            )
        ):
            raise ValueError("pose/IK input contains non-finite numeric fields")
        if (
            abs(backend_x - backprojection_x) > 1e-9
            or abs(backend_y - backprojection_y) > 1e-9
        ):
            raise ValueError("backend and backprojection centres must match")
    try:
        camera = metadata["camera"]
        camera_config = camera["config"]
        scene_config = metadata["scene"]["config"]
        width = int(camera_config["width"])
        height = int(camera_config["height"])
        seed = int(scene_config["seed"])
        object_order = (
            scene_config["object_name"],
            *(item["name"] for item in scene_config["additional_objects"]),
        )
        view_matrix = tuple(float(value) for value in camera["view_matrix"])
        projection_matrix = tuple(
            float(value) for value in camera["projection_matrix"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata does not satisfy the fixed study contract") from exc
    if (width, height) != (640, 480):
        raise ValueError("metadata camera dimensions must be 640x480")
    if seed != 42:
        raise ValueError("metadata scene seed must be 42")
    if object_order != ("duck", "cube", "sphere"):
        raise ValueError("metadata object order must be duck, cube, sphere")
    if len(view_matrix) != 16 or len(projection_matrix) != 16:
        raise ValueError("metadata camera matrices must contain 16 values")
    return PoseIKInputs(backend_rows, backprojection_rows, metadata)


def _failed_collision(reason: str) -> CollisionAudit:
    return CollisionAudit(False, 0, 0.0, 0, 0, reason)


def _joint_cost(
    model: PandaModelInfo,
    pregrasp: tuple[float, ...],
    standoff: tuple[float, ...],
) -> float:
    arm_offsets = tuple(
        model.movable_joint_indices.index(index)
        for index in model.arm_joint_indices
    )
    rests = tuple(model.rest_poses[offset] for offset in arm_offsets)
    ranges = tuple(model.joint_ranges[offset] for offset in arm_offsets)
    return sum(
        ((end - start) / joint_range) ** 2
        for start, end, joint_range in zip(rests, pregrasp, ranges)
    ) + sum(
        ((end - start) / joint_range) ** 2
        for start, end, joint_range in zip(pregrasp, standoff, ranges)
    )


def _audit_candidate(
    candidate: PoseCandidate,
    *,
    robot_id: int,
    client_id: int,
    model: PandaModelInfo,
    environment_body_ids: tuple[int, ...],
    allowed_environment_link_pairs: tuple[tuple[int, int], ...],
) -> CandidateAudit:
    pregrasp = audit_pose_ik(
        robot_id, client_id, model, candidate.pregrasp_pose
    )
    standoff = audit_pose_ik(
        robot_id, client_id, model, candidate.surface_standoff_pose
    )
    failures = []
    if not pregrasp.gate_passed:
        failures.append(f"pregrasp:{pregrasp.failure_reason}")
    if not standoff.gate_passed:
        failures.append(f"standoff:{standoff.failure_reason}")
    if pregrasp.solution is not None and standoff.solution is not None:
        arm_offsets = tuple(
            model.movable_joint_indices.index(index)
            for index in model.arm_joint_indices
        )
        arm_rest = tuple(model.rest_poses[offset] for offset in arm_offsets)
        try:
            collision = audit_joint_path_clearance(
                robot_id=robot_id,
                client_id=client_id,
                model=model,
                start_solution=arm_rest,
                pregrasp_solution=pregrasp.solution,
                standoff_solution=standoff.solution,
                environment_body_ids=environment_body_ids,
                allowed_environment_link_pairs=allowed_environment_link_pairs,
            )
        except Exception as exc:  # PyBullet exposes backend-specific errors.
            collision = _failed_collision(
                f"collision_audit_error:{type(exc).__name__}:{exc}"
            )
        cost = _joint_cost(model, pregrasp.solution, standoff.solution)
    else:
        collision = _failed_collision("ik_solution_unavailable")
        cost = math.inf
    if not collision.clearance_passed:
        failures.append(f"collision:{collision.failure_reason}")
    return CandidateAudit(
        candidate=candidate,
        pregrasp_ik=pregrasp,
        standoff_ik=standoff,
        collision=collision,
        total_normalized_joint_cost=cost,
        gate_passed=not failures,
        selected=False,
        failure_reason=";".join(failures),
    )


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _candidate_row(audit: CandidateAudit) -> dict[str, object]:
    candidate = audit.candidate
    return {
        "target": candidate.target,
        "backend": candidate.backend,
        "symmetry_degrees": candidate.symmetry_degrees,
        "finger_axis_world": _json_value(candidate.finger_axis_world),
        "closing_axis_world": _json_value(candidate.closing_axis_world),
        "approach_axis_world": _json_value(candidate.approach_axis_world),
        "pregrasp_position": _json_value(candidate.pregrasp_pose.position),
        "pregrasp_quaternion_xyzw": _json_value(
            candidate.pregrasp_pose.quaternion_xyzw
        ),
        "standoff_position": _json_value(
            candidate.surface_standoff_pose.position
        ),
        "standoff_quaternion_xyzw": _json_value(
            candidate.surface_standoff_pose.quaternion_xyzw
        ),
        "pregrasp_solution": _json_value(audit.pregrasp_ik.solution),
        "standoff_solution": _json_value(audit.standoff_ik.solution),
        "pregrasp_position_error_m": audit.pregrasp_ik.position_error_m,
        "pregrasp_orientation_error_degrees": (
            audit.pregrasp_ik.orientation_error_degrees
        ),
        "standoff_position_error_m": audit.standoff_ik.position_error_m,
        "standoff_orientation_error_degrees": (
            audit.standoff_ik.orientation_error_degrees
        ),
        "ik_fk_passed": (
            audit.pregrasp_ik.gate_passed and audit.standoff_ik.gate_passed
        ),
        "clearance_passed": audit.collision.clearance_passed,
        "checked_state_count": audit.collision.checked_state_count,
        "minimum_clearance_m": audit.collision.minimum_clearance_m,
        "environment_collision_count": (
            audit.collision.environment_collision_count
        ),
        "self_collision_count": audit.collision.self_collision_count,
        "total_normalized_joint_cost": audit.total_normalized_joint_cost,
        "gate_passed": audit.gate_passed,
        "selected": audit.selected,
        "failure_reason": audit.failure_reason,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_pose_ik_study(config: PoseIKStudyConfig) -> dict[str, object]:
    """Run the fixed scene's static audit without commanding robot motion."""

    inputs = load_pose_ik_inputs(config.input_dir)
    camera = inputs.metadata["camera"]
    camera_config = camera["config"]
    scene_config = replace(
        fixed_scene_config(MultiObjectStudyConfig(gui=False, device="cpu")),
        robot_self_collision=True,
    )
    audits: list[CandidateAudit] = []
    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        model = resolve_panda_model(scene.bodies.robot, scene.client_id)
        environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        for backend, backprojection in zip(
            inputs.backend_rows, inputs.backprojection_rows
        ):
            candidates = generate_top_down_pose_candidates(
                target=backend["target"],
                backend=backend["backend"],
                column=int(backprojection["sampled_column"]),
                row=int(backprojection["sampled_row"]),
                depth_m=float(backprojection["depth_m"]),
                angle_degrees=float(backend["angle_degrees"]),
                width=int(camera_config["width"]),
                height=int(camera_config["height"]),
                view_matrix=camera["view_matrix"],
                projection_matrix=camera["projection_matrix"],
                near=float(camera_config["near"]),
                far=float(camera_config["far"]),
            )
            pair = tuple(
                _audit_candidate(
                    candidate,
                    robot_id=scene.bodies.robot,
                    client_id=scene.client_id,
                    model=model,
                    environment_body_ids=environment,
                    allowed_environment_link_pairs=(
                        (-1, scene.bodies.table),
                    ),
                )
                for candidate in candidates
            )
            audits.extend(select_candidate_pair(pair))

    rows = [_candidate_row(audit) for audit in audits]
    passing = sum(audit.gate_passed for audit in audits)
    selected = sum(audit.selected for audit in audits)
    summary: dict[str, object] = {
        "protocol": "fixed_three_object_static_pose_ik_clearance_audit",
        "input_grasp_count": 9,
        "candidate_count": len(audits),
        "ik_fk_passed_count": sum(
            audit.pregrasp_ik.gate_passed and audit.standoff_ik.gate_passed
            for audit in audits
        ),
        "clearance_passed_count": sum(
            audit.collision.clearance_passed for audit in audits
        ),
        "candidate_gate_passed_count": passing,
        "selected_count": selected,
        "all_inputs_have_selected_candidate": selected == 9,
        "scientific_gate_passed": selected == 9,
    }
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": summary["protocol"],
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "simulation_setup_steps": 60,
        "simulation_stepped_during_candidate_audit": False,
        "inverse_kinematics_executed": True,
        "forward_kinematics_verified": True,
        "static_joint_resets_used": True,
        "ik_solver_called": True,
        "joint_states_set_for_static_audit": True,
        "motor_control_called": False,
        "motor_control_executed": False,
        "trajectory_executed": False,
        "gripper_closed": False,
        "physical_grasp_executed": False,
        "input_sha256": {
            name: hashlib.sha256(
                (Path(config.input_dir) / name).read_bytes()
            ).hexdigest()
            for name in (
                "backend_results.csv",
                "backprojection_results.csv",
                "metadata.json",
            )
        },
        "pybullet": {
            "package_version": importlib.metadata.version("pybullet"),
            "api_version": p.getAPIVersion(),
        },
        "thresholds": {
            "position_error_m": 0.005,
            "orientation_error_degrees": 5.0,
            "collision_clearance_m": 0.002,
            "samples_per_segment": 21,
        },
        "summary": summary,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pose_ik_candidates.csv", rows)
    _write_json(output_dir / "pose_ik_summary.json", summary)
    _write_json(output_dir / "pose_ik_metadata.json", metadata)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed static Panda pose/IK clearance audit."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STUDY_DIR)
    args = parser.parse_args()
    summary = run_pose_ik_study(
        PoseIKStudyConfig(input_dir=args.input_dir, output_dir=args.output_dir)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
