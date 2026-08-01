"""Shared truth-pose execution core for staged Panda grasp studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
from enum import Enum
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
from src.simulation.pybullet.pose_generation import (
    generate_top_down_pose_from_world_point,
)
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    fixed_scene_config,
)
from src.simulation.pybullet.scene import PyBulletScene


TARGET_XY_THRESHOLD_M = 0.005
FINGER_OPEN_ERROR_THRESHOLD_M = 0.001
CONTACT_EVENT_FIELDS = (
    "step",
    "phase",
    "robot_link",
    "target_body",
    "normal_force",
)


class TruthExecutionStage(str, Enum):
    """Supported truth-pose execution boundaries."""

    PREGRASP = "pregrasp"
    OPEN_APPROACH = "open_approach"


@dataclass(frozen=True)
class TruthExecutionConfig:
    """Shared deterministic scene, target, and stability settings."""

    output_dir: Path
    seed: int = 42
    gui: bool = False
    target_name: str = "cube"
    stability_steps: int = 60
    maximum_target_displacement_m: float = 0.001

    def __post_init__(self) -> None:
        if self.target_name != "cube":
            raise ValueError("truth execution target_name must be cube")
        if self.stability_steps <= 0:
            raise ValueError("stability_steps must be positive")
        if (
            not math.isfinite(self.maximum_target_displacement_m)
            or self.maximum_target_displacement_m <= 0.0
        ):
            raise ValueError(
                "maximum_target_displacement_m must be finite and positive"
            )


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _target_position(
    body_id: int,
    client_id: int,
) -> tuple[float, float, float]:
    position, _ = p.getBasePositionAndOrientation(
        body_id,
        physicsClientId=client_id,
    )
    return tuple(float(value) for value in position)


def _maximum_displacement(
    reference: Sequence[float],
    samples: Sequence[Sequence[float]],
) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    return max(
        (
            float(
                np.linalg.norm(
                    np.asarray(sample, dtype=np.float64) - reference_array
                )
            )
            for sample in samples
        ),
        default=0.0,
    )


def _trace_rows(result: MotionExecutionResult) -> list[dict[str, object]]:
    rows = []
    for row in result.trace:
        target = row.tracked_body_poses[0]
        relative = tuple(
            tool - cube
            for tool, cube in zip(row.actual_tool_position, target.position)
        )
        rows.append(
            {
                "step": row.step,
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
                "cube_position": json.dumps(
                    target.position,
                    separators=(",", ":"),
                ),
                "cube_quaternion_xyzw": json.dumps(
                    target.quaternion_xyzw,
                    separators=(",", ":"),
                ),
                "tool_relative_to_cube": json.dumps(
                    relative,
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
        "cube_position",
        "cube_quaternion_xyzw",
        "tool_relative_to_cube",
        "maximum_joint_error_rad",
        "minimum_clearance_m",
        "environment_collision_count",
        "self_collision_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_events(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _failure_artifacts(
    *,
    config: TruthExecutionConfig,
    output_dir: Path,
    start_rgb: np.ndarray,
    failure_stage: str,
    stability_displacement_m: float,
    ik_fk_passed: bool,
    clearance_passed: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "stage": "cube_truth_pregrasp",
        "target_stability_preflight_passed": (
            stability_displacement_m
            <= config.maximum_target_displacement_m
        ),
        "preflight_ik_fk_passed": ik_fk_passed,
        "preflight_clearance_passed": clearance_passed,
        "pregrasp_reached": False,
        "endpoint_pose_gate_passed": False,
        "target_xy_gate_passed": False,
        "target_undisturbed_gate_passed": False,
        "fingers_open_gate_passed": False,
        "maximum_target_preflight_displacement_m": (
            stability_displacement_m
        ),
        "executed_step_count": 0,
        "failure_stage": failure_stage,
        "scientific_gate_passed": False,
    }
    metadata = _metadata(
        config=config,
        summary=summary,
        motor_control_executed=False,
        target_approach_executed=False,
    )
    _write_trace(output_dir / "state_trace.csv", [])
    _write_contact_events(output_dir / "contact_events.csv", [])
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "metadata.json", metadata)
    _write_rgb(output_dir / "start.png", start_rgb)
    return summary


def _metadata(
    *,
    config: TruthExecutionConfig,
    summary: dict[str, object],
    motor_control_executed: bool,
    target_approach_executed: bool,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "panda_cube_truth_pregrasp",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "simulation_setup_steps": 60,
        "truth_target_used": True,
        "initial_joint_reset_used": True,
        "motor_control_executed": motor_control_executed,
        "simulation_stepped": True,
        "trajectory_executed": motor_control_executed,
        "perception_executed": False,
        "target_approach_executed": target_approach_executed,
        "descent_to_contact_executed": False,
        "gripper_close_commanded": False,
        "gripper_closed": False,
        "contact_evaluated": False,
        "object_lifted": False,
        "physical_grasp_executed": False,
        **(details or {}),
        "summary": summary,
    }


def run_truth_execution(
    config: TruthExecutionConfig,
    stage: TruthExecutionStage,
) -> dict[str, object]:
    """Execute one truth-pose stage within its declared boundary."""

    if stage is not TruthExecutionStage.PREGRASP:
        raise ValueError(f"truth execution stage is not implemented: {stage}")

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
        cube_id = scene.object_body_ids[config.target_name]
        stability_reference = _target_position(cube_id, scene.client_id)
        stability_samples = []
        for _ in range(config.stability_steps):
            scene.step()
            stability_samples.append(_target_position(cube_id, scene.client_id))
        stability_displacement = _maximum_displacement(
            stability_reference,
            stability_samples,
        )
        if stability_displacement > config.maximum_target_displacement_m:
            return _failure_artifacts(
                config=config,
                output_dir=output_dir,
                start_rgb=start_rgb,
                failure_stage="target_stability_preflight",
                stability_displacement_m=stability_displacement,
                ik_fk_passed=False,
                clearance_passed=False,
            )

        cube_position, cube_quaternion = p.getBasePositionAndOrientation(
            cube_id,
            physicsClientId=scene.client_id,
        )
        cube_position = tuple(float(value) for value in cube_position)
        cube_quaternion = tuple(float(value) for value in cube_quaternion)
        cube_aabb = p.getAABB(cube_id, physicsClientId=scene.client_id)
        cube_top_z = float(cube_aabb[1][2])
        rotation = p.getMatrixFromQuaternion(cube_quaternion)
        finger_axis = (
            float(rotation[0]),
            float(rotation[3]),
            float(rotation[6]),
        )
        candidate = generate_top_down_pose_from_world_point(
            target="cube",
            backend="ground_truth",
            surface_point=(cube_position[0], cube_position[1], cube_top_z),
            finger_axis_world=finger_axis,
        )
        pregrasp_pose = candidate.pregrasp_pose
        pregrasp_ik = audit_pose_ik(
            scene.bodies.robot,
            scene.client_id,
            model,
            pregrasp_pose,
        )
        environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        allowed_mounting_pair = ((-1, scene.bodies.table),)
        if not pregrasp_ik.gate_passed or pregrasp_ik.solution is None:
            return _failure_artifacts(
                config=config,
                output_dir=output_dir,
                start_rgb=start_rgb,
                failure_stage="preflight_ik",
                stability_displacement_m=stability_displacement,
                ik_fk_passed=False,
                clearance_passed=False,
            )
        preflight_clearance = audit_joint_path_clearance(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            start_solution=neutral,
            pregrasp_solution=pregrasp_ik.solution,
            standoff_solution=pregrasp_ik.solution,
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        if not preflight_clearance.clearance_passed:
            return _failure_artifacts(
                config=config,
                output_dir=output_dir,
                start_rgb=start_rgb,
                failure_stage="preflight_clearance",
                stability_displacement_m=stability_displacement,
                ik_fk_passed=True,
                clearance_passed=False,
            )

        motion_start_target_position = _target_position(
            cube_id,
            scene.client_id,
        )
        motion_config = MotionConfig(joint_tolerance_rad=0.002)
        execution = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(
                MotionSegment("pregrasp", pregrasp_ik.solution),
            ),
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            tracked_body_ids=(cube_id,),
            config=motion_config,
        )
        pregrasp_rgb = _capture_rgb(scene)

    final_row = execution.trace[-1]
    final_cube_position = final_row.tracked_body_poses[0].position
    position_error, orientation_error = _pose_errors(
        final_row,
        pregrasp_pose.position,
        pregrasp_pose.quaternion_xyzw,
    )
    target_xy_error = float(
        np.linalg.norm(
            np.asarray(final_row.actual_tool_position[:2])
            - np.asarray(final_cube_position[:2])
        )
    )
    motion_target_displacement = _maximum_displacement(
        motion_start_target_position,
        tuple(
            row.tracked_body_poses[0].position for row in execution.trace
        ),
    )
    maximum_target_displacement = max(
        stability_displacement,
        motion_target_displacement,
    )
    maximum_finger_open_error = max(
        abs(value - 0.04)
        for row in execution.trace
        for value in row.actual_finger_positions
    )
    pregrasp_reached = dict(execution.segment_reached)["pregrasp"]
    endpoint_pose_gate = (
        position_error <= POSITION_ERROR_THRESHOLD_M
        and orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
    )
    target_xy_gate = target_xy_error <= TARGET_XY_THRESHOLD_M
    target_undisturbed_gate = (
        maximum_target_displacement
        <= config.maximum_target_displacement_m
    )
    fingers_open_gate = (
        maximum_finger_open_error <= FINGER_OPEN_ERROR_THRESHOLD_M
    )
    scientific_gate = (
        stability_displacement <= config.maximum_target_displacement_m
        and pregrasp_ik.gate_passed
        and preflight_clearance.clearance_passed
        and execution.gate_passed
        and pregrasp_reached
        and endpoint_pose_gate
        and target_xy_gate
        and target_undisturbed_gate
        and fingers_open_gate
    )
    summary: dict[str, object] = {
        "stage": "cube_truth_pregrasp",
        "target_stability_preflight_passed": True,
        "preflight_ik_fk_passed": pregrasp_ik.gate_passed,
        "preflight_clearance_passed": (
            preflight_clearance.clearance_passed
        ),
        "pregrasp_reached": pregrasp_reached,
        "endpoint_pose_gate_passed": endpoint_pose_gate,
        "target_xy_gate_passed": target_xy_gate,
        "target_undisturbed_gate_passed": target_undisturbed_gate,
        "fingers_open_gate_passed": fingers_open_gate,
        "pregrasp_position_error_m": position_error,
        "pregrasp_orientation_error_degrees": orientation_error,
        "target_xy_error_m": target_xy_error,
        "maximum_target_preflight_displacement_m": stability_displacement,
        "maximum_target_motion_displacement_m": motion_target_displacement,
        "maximum_target_displacement_m": maximum_target_displacement,
        "maximum_finger_open_error_m": maximum_finger_open_error,
        "executed_step_count": len(execution.trace),
        "minimum_clearance_m": execution.minimum_clearance_m,
        "environment_collision_count": (
            execution.environment_collision_count
        ),
        "self_collision_count": execution.self_collision_count,
        "all_states_finite": execution.all_states_finite,
        "scientific_gate_passed": scientific_gate,
    }
    metadata = _metadata(
        config=config,
        summary=summary,
        motor_control_executed=True,
        target_approach_executed=True,
        details={
            "cube_body_id": cube_id,
            "cube_motion_start_position": motion_start_target_position,
            "cube_motion_start_quaternion_xyzw": cube_quaternion,
            "cube_top_z_m": cube_top_z,
            "pregrasp_tool_position": pregrasp_pose.position,
            "pregrasp_tool_quaternion_xyzw": (
                pregrasp_pose.quaternion_xyzw
            ),
            "pregrasp_height_above_cube_top_m": 0.12,
            "thresholds": {
                "position_error_m": POSITION_ERROR_THRESHOLD_M,
                "orientation_error_degrees": (
                    ORIENTATION_ERROR_THRESHOLD_DEGREES
                ),
                "target_xy_error_m": TARGET_XY_THRESHOLD_M,
                "target_displacement_m": (
                    config.maximum_target_displacement_m
                ),
                "joint_tolerance_rad": motion_config.joint_tolerance_rad,
                "finger_open_error_m": FINGER_OPEN_ERROR_THRESHOLD_M,
                "collision_clearance_m": motion_config.clearance_m,
            },
        },
    )
    _write_trace(output_dir / "state_trace.csv", _trace_rows(execution))
    _write_contact_events(output_dir / "contact_events.csv", [])
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "metadata.json", metadata)
    _write_rgb(output_dir / "start.png", start_rgb)
    _write_rgb(output_dir / "pregrasp.png", pregrasp_rgb)
    return summary
