"""Execute the stage-1 Panda safe aerial motor-control smoke study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.camera import CameraConfig, capture_camera_frame
from src.simulation.pybullet.kinematic_audit import (
    ORIENTATION_ERROR_THRESHOLD_DEGREES,
    POSITION_ERROR_THRESHOLD_M,
    audit_joint_path_clearance,
    audit_pose_ik,
    resolve_panda_model,
)
from src.simulation.pybullet.motion_control import (
    MotionConfig,
    MotionExecutionResult,
    MotionSegment,
    MotionTraceRow,
    execute_joint_motion,
)
from src.simulation.pybullet.pose_generation import ToolPose
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene


DEFAULT_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/stage_1_safe_motion"
)


@dataclass(frozen=True)
class SafeMotionSmokeConfig:
    """Stage-1 output and deterministic scene configuration."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    waypoint_lift_m: float = 0.05
    gui: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.waypoint_lift_m) or self.waypoint_lift_m <= 0:
            raise ValueError("waypoint_lift_m must be finite and positive")


def _initialize_neutral_open(
    robot_id: int,
    client_id: int,
    arm_indices: Sequence[int],
    finger_indices: Sequence[int],
    arm_positions: Sequence[float],
) -> None:
    for index, value in zip(arm_indices, arm_positions):
        p.resetJointState(
            robot_id,
            index,
            float(value),
            physicsClientId=client_id,
        )
    for index in finger_indices:
        p.resetJointState(
            robot_id,
            index,
            0.04,
            physicsClientId=client_id,
        )
    p.performCollisionDetection(physicsClientId=client_id)


def _capture_rgb(scene: PyBulletScene) -> np.ndarray:
    return capture_camera_frame(
        scene.client_id,
        CameraConfig(),
        scene.renderer,
    ).rgb


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write stage image: {path}")


def _orientation_error_degrees(
    actual_xyzw: Sequence[float],
    target_xyzw: Sequence[float],
) -> float:
    dot = float(
        np.clip(
            abs(
                np.dot(
                    np.asarray(actual_xyzw, dtype=np.float64),
                    np.asarray(target_xyzw, dtype=np.float64),
                )
            ),
            0.0,
            1.0,
        )
    )
    return math.degrees(2.0 * math.acos(dot))


def _pose_errors(
    row: MotionTraceRow,
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
) -> tuple[float, float]:
    position_error = float(
        np.linalg.norm(
            np.asarray(row.actual_tool_position, dtype=np.float64)
            - np.asarray(target_position, dtype=np.float64)
        )
    )
    orientation_error = _orientation_error_degrees(
        row.actual_tool_quaternion_xyzw,
        target_quaternion,
    )
    return position_error, orientation_error


def _trace_rows(
    results: Sequence[MotionExecutionResult],
) -> list[dict[str, object]]:
    rows = []
    for global_step, row in enumerate(
        (trace_row for result in results for trace_row in result.trace),
        start=1,
    ):
        rows.append(
            {
                "step": global_step,
                "phase": row.phase,
                "commanded_arm_positions": json.dumps(
                    row.commanded_arm_positions,
                    separators=(",", ":"),
                ),
                "actual_arm_positions": json.dumps(
                    row.actual_arm_positions,
                    separators=(",", ":"),
                ),
                "actual_finger_positions": json.dumps(
                    row.actual_finger_positions,
                    separators=(",", ":"),
                ),
                "actual_tool_position": json.dumps(
                    row.actual_tool_position,
                    separators=(",", ":"),
                ),
                "actual_tool_quaternion_xyzw": json.dumps(
                    row.actual_tool_quaternion_xyzw,
                    separators=(",", ":"),
                ),
                "maximum_joint_error_rad": row.maximum_joint_error_rad,
                "minimum_clearance_m": row.minimum_clearance_m,
                "environment_collision_count": (
                    row.environment_collision_count
                ),
                "self_collision_count": row.self_collision_count,
            }
        )
    return rows


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "step",
        "phase",
        "commanded_arm_positions",
        "actual_arm_positions",
        "actual_finger_positions",
        "actual_tool_position",
        "actual_tool_quaternion_xyzw",
        "maximum_joint_error_rad",
        "minimum_clearance_m",
        "environment_collision_count",
        "self_collision_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_preflight_failure(
    *,
    config: SafeMotionSmokeConfig,
    output_dir: Path,
    start_rgb: np.ndarray,
    failure_stage: str,
    ik_fk_passed: bool,
    clearance_passed: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "stage": "safe_motion_smoke",
        "waypoint_lift_m": config.waypoint_lift_m,
        "preflight_ik_fk_passed": ik_fk_passed,
        "preflight_clearance_passed": clearance_passed,
        "outbound_reached": False,
        "return_reached": False,
        "maximum_finger_open_error_m": None,
        "executed_step_count": 0,
        "environment_collision_count": 0,
        "self_collision_count": 0,
        "all_states_finite": False,
        "failure_stage": failure_stage,
        "scientific_gate_passed": False,
    }
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "panda_safe_aerial_motor_round_trip",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "simulation_setup_steps": 60,
        "initial_joint_reset_used": True,
        "motor_control_executed": False,
        "simulation_stepped": True,
        "trajectory_executed": False,
        "perception_executed": False,
        "target_approach_executed": False,
        "gripper_close_commanded": False,
        "gripper_closed": False,
        "contact_evaluated": False,
        "object_lifted": False,
        "physical_grasp_executed": False,
        "summary": summary,
    }
    _write_trace(output_dir / "state_trace.csv", [])
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "metadata.json", metadata)
    _write_rgb(output_dir / "start.png", start_rgb)
    return summary


def run_safe_motion_smoke(
    config: SafeMotionSmokeConfig,
) -> dict[str, object]:
    """Execute neutral→aerial waypoint→neutral and save stage evidence."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_config = replace(
        fixed_scene_config(
            MultiObjectStudyConfig(
                gui=config.gui,
                seed=config.seed,
                device="cpu",
            )
        ),
        robot_self_collision=True,
    )
    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        model = resolve_panda_model(scene.bodies.robot, scene.client_id)
        neutral = tuple(model.rest_poses[:7])
        _initialize_neutral_open(
            scene.bodies.robot,
            scene.client_id,
            model.arm_joint_indices,
            model.finger_joint_indices,
            neutral,
        )
        start_rgb = _capture_rgb(scene)
        neutral_link = p.getLinkState(
            scene.bodies.robot,
            model.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=scene.client_id,
        )
        neutral_position = tuple(float(value) for value in neutral_link[4])
        neutral_quaternion = tuple(float(value) for value in neutral_link[5])
        waypoint_position = (
            neutral_position[0],
            neutral_position[1],
            neutral_position[2] + config.waypoint_lift_m,
        )
        waypoint_pose = ToolPose(waypoint_position, neutral_quaternion)
        waypoint_ik = audit_pose_ik(
            scene.bodies.robot,
            scene.client_id,
            model,
            waypoint_pose,
        )
        environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        allowed_mounting_pair = ((-1, scene.bodies.table),)
        if not waypoint_ik.gate_passed or waypoint_ik.solution is None:
            return _save_preflight_failure(
                config=config,
                output_dir=output_dir,
                start_rgb=start_rgb,
                failure_stage="preflight_ik",
                ik_fk_passed=False,
                clearance_passed=False,
            )
        preflight_clearance = audit_joint_path_clearance(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            start_solution=neutral,
            pregrasp_solution=waypoint_ik.solution,
            standoff_solution=neutral,
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        if not preflight_clearance.clearance_passed:
            return _save_preflight_failure(
                config=config,
                output_dir=output_dir,
                start_rgb=start_rgb,
                failure_stage="preflight_clearance",
                ik_fk_passed=True,
                clearance_passed=False,
            )

        motion_config = MotionConfig(joint_tolerance_rad=0.002)
        outbound = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(MotionSegment("outbound", waypoint_ik.solution),),
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            config=motion_config,
        )
        waypoint_rgb = _capture_rgb(scene)
        returned = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(MotionSegment("return", neutral),),
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            config=motion_config,
        )
        return_rgb = _capture_rgb(scene)

    outbound_reached = dict(outbound.segment_reached)["outbound"]
    return_reached = dict(returned.segment_reached)["return"]
    waypoint_position_error, waypoint_orientation_error = _pose_errors(
        outbound.trace[-1],
        waypoint_position,
        neutral_quaternion,
    )
    return_position_error, return_orientation_error = _pose_errors(
        returned.trace[-1],
        neutral_position,
        neutral_quaternion,
    )
    minimum_clearance = min(
        outbound.minimum_clearance_m,
        returned.minimum_clearance_m,
    )
    environment_collision_count = (
        outbound.environment_collision_count
        + returned.environment_collision_count
    )
    self_collision_count = (
        outbound.self_collision_count + returned.self_collision_count
    )
    maximum_finger_open_error = max(
        abs(value - 0.04)
        for result in (outbound, returned)
        for row in result.trace
        for value in row.actual_finger_positions
    )
    endpoint_gate = (
        waypoint_position_error <= POSITION_ERROR_THRESHOLD_M
        and waypoint_orientation_error
        <= ORIENTATION_ERROR_THRESHOLD_DEGREES
        and return_position_error <= POSITION_ERROR_THRESHOLD_M
        and return_orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
    )
    scientific_gate = (
        waypoint_ik.gate_passed
        and preflight_clearance.clearance_passed
        and outbound.gate_passed
        and returned.gate_passed
        and maximum_finger_open_error <= 0.001
        and endpoint_gate
    )
    summary: dict[str, object] = {
        "stage": "safe_motion_smoke",
        "waypoint_lift_m": config.waypoint_lift_m,
        "preflight_ik_fk_passed": waypoint_ik.gate_passed,
        "preflight_clearance_passed": (
            preflight_clearance.clearance_passed
        ),
        "outbound_reached": outbound_reached,
        "return_reached": return_reached,
        "maximum_finger_open_error_m": maximum_finger_open_error,
        "waypoint_position_error_m": waypoint_position_error,
        "waypoint_orientation_error_degrees": waypoint_orientation_error,
        "return_position_error_m": return_position_error,
        "return_orientation_error_degrees": return_orientation_error,
        "executed_step_count": len(outbound.trace) + len(returned.trace),
        "minimum_clearance_m": minimum_clearance,
        "environment_collision_count": environment_collision_count,
        "self_collision_count": self_collision_count,
        "all_states_finite": (
            outbound.all_states_finite and returned.all_states_finite
        ),
        "scientific_gate_passed": scientific_gate,
    }
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "panda_safe_aerial_motor_round_trip",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "simulation_setup_steps": 60,
        "initial_joint_reset_used": True,
        "motor_control_executed": True,
        "simulation_stepped": True,
        "trajectory_executed": True,
        "perception_executed": False,
        "target_approach_executed": False,
        "gripper_close_commanded": False,
        "gripper_closed": False,
        "contact_evaluated": False,
        "object_lifted": False,
        "physical_grasp_executed": False,
        "neutral_arm_positions": neutral,
        "neutral_tool_position": neutral_position,
        "waypoint_tool_position": waypoint_position,
        "thresholds": {
            "position_error_m": POSITION_ERROR_THRESHOLD_M,
            "orientation_error_degrees": (
                ORIENTATION_ERROR_THRESHOLD_DEGREES
            ),
            "joint_tolerance_rad": motion_config.joint_tolerance_rad,
            "finger_open_error_m": 0.001,
            "collision_clearance_m": motion_config.clearance_m,
        },
        "summary": summary,
    }
    _write_trace(output_dir / "state_trace.csv", _trace_rows((outbound, returned)))
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "metadata.json", metadata)
    _write_rgb(output_dir / "start.png", start_rgb)
    _write_rgb(output_dir / "waypoint.png", waypoint_rgb)
    _write_rgb(output_dir / "return.png", return_rgb)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the stage-1 Panda safe aerial motion smoke."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    summary = run_safe_motion_smoke(
        SafeMotionSmokeConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            gui=args.gui,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
