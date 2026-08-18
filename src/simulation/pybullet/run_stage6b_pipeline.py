"""Stage 6B: Execute a perception-derived frozen execution plan.

Loads a Stage 6A ``execution_plan.json`` and drives the Panda robot
through the full grasp chain using the plan's pre-computed poses and IK
solutions.  No perception inference runs here — only motor control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import argparse
import hashlib
import time
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
from src.simulation.pybullet.execution_plan import (
    CameraEvidence,
    PerceptionExecutionPlan,
    PlannedPoseCandidate,
    load_perception_execution_plan,
)
from src.simulation.pybullet.gripper_control import (
    GripperCloseConfig,
    GripperCloseResult,
    execute_gripper_close,
)
from src.simulation.pybullet.kinematic_audit import (
    ORIENTATION_ERROR_THRESHOLD_DEGREES,
    POSITION_ERROR_THRESHOLD_M,
    audit_joint_path_clearance,
    audit_pose_ik,
    resolve_panda_model,
)
from src.simulation.pybullet.lift_control import (
    LiftConfig,
    LiftResult,
    execute_object_lift,
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

# ---------------------------------------------------------------------------
# Frozen constants (mirror truth execution)
# ---------------------------------------------------------------------------

FINGER_OPEN_ERROR_THRESHOLD_M = 0.001
CONTACT_EVENT_FIELDS = (
    "step",
    "phase",
    "robot_link",
    "target_body",
    "normal_force",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PLAN_PATH = Path(
    "data/processed/pybullet/grasp_execution/"
    "stage_6a_geometry_preflight/execution_plan.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/stage_6b_perception_grasp"
)


@dataclass(frozen=True)
class Stage6BConfig:
    """Frozen Stage 6B protocol."""

    plan_path: Path = DEFAULT_PLAN_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    gui: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.seed != 42:
            raise ValueError("Stage 6B seed must be 42")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")


# ---------------------------------------------------------------------------
# Helpers (adapted from grasp_execution.py private functions)
# ---------------------------------------------------------------------------


def _initialize_neutral_open(
    robot_id: int,
    client_id: int,
    arm_indices: Sequence[int],
    finger_indices: Sequence[int],
    arm_positions: Sequence[float],
) -> None:
    for index, value in zip(arm_indices, arm_positions):
        p.resetJointState(robot_id, index, float(value), physicsClientId=client_id)
    for index in finger_indices:
        p.resetJointState(robot_id, index, 0.04, physicsClientId=client_id)
    p.performCollisionDetection(physicsClientId=client_id)


def _camera_config(evidence: CameraEvidence) -> CameraConfig:
    return CameraConfig(
        width=evidence.width,
        height=evidence.height,
        eye=evidence.eye,
        target=evidence.target,
        up=evidence.up,
        fov_degrees=evidence.fov_degrees,
        near=evidence.near,
        far=evidence.far,
    )


def _capture_rgb(
    scene: PyBulletScene,
    camera: CameraEvidence,
) -> np.ndarray:
    return capture_camera_frame(
        scene.client_id,
        _camera_config(camera),
        scene.renderer,
    ).rgb


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write stage image: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            cls=_JsonEncoder,
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
    position, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    return tuple(float(value) for value in position)


def _maximum_displacement(
    reference: Sequence[float],
    samples: Sequence[Sequence[float]],
) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    return max(
        float(
            np.linalg.norm(np.asarray(sample, dtype=np.float64) - reference_array)
        )
        for sample in samples
    )


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                field_names.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_events(
    path: Path,
    events: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONTACT_EVENT_FIELDS))
        writer.writeheader()
        writer.writerows(events)


def _build_metadata(
    config: Stage6BConfig,
    plan: PerceptionExecutionPlan,
    candidate: PlannedPoseCandidate,
    **extra: object,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "stage_6b_perception_grasp_v1",
        "plan_protocol": plan.protocol_version,
        "plan_path": str(config.plan_path),
        "plan_file_sha256": _sha256_file(config.plan_path),
        "plan_rgb_sha256": plan.rgb_sha256,
        "config": {
            "output_dir": str(config.output_dir),
            "seed": config.seed,
            "gui": config.gui,
            "device": config.device,
        },
        "perception_used": True,
        "perception_world_surface_point": list(plan.perception.world_surface_point),
        "perception_2d_center": list(plan.perception.grasp_center),
        "selected_symmetry_degrees": candidate.symmetry_degrees,
        "physical_grasp_executed": True,
        **extra,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_stage6b(config: Stage6BConfig) -> dict[str, object]:
    """Load a frozen execution plan and drive the full perception grasp chain."""

    def _pause(sec: float = 4.0) -> None:
        if config.gui:
            time.sleep(sec)

    # -- 1. Load and validate plan -------------------------------------------
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_perception_execution_plan(config.plan_path)
    if plan.scene_seed != config.seed:
        raise ValueError(
            f"plan seed {plan.scene_seed} != config seed {config.seed}"
        )
    if plan.target_name != "cube":
        raise ValueError("Stage 6B target must be cube")
    if plan.backend not in ("geometry", "multi_head"):
        raise ValueError("Stage 6B backend must be geometry or multi_head")

    selected = [c for c in plan.candidates if c.selected]
    if len(selected) != 1:
        raise ValueError("plan must contain exactly one selected candidate")
    candidate = selected[0]

    # -- 2. Set up scene ----------------------------------------------------
    scene_config = replace(
        fixed_scene_config(
            MultiObjectStudyConfig(gui=config.gui, seed=config.seed, device="cpu")
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

        start_rgb = _capture_rgb(scene, plan.camera)
        cube_id = scene.object_body_ids[config.plan_path.name] if False else (
            scene.object_body_ids.get("cube", -1)
        )
        # resolve cube body id
        cube_id = scene.object_body_ids["cube"]

        # -- 3. Stability preflight -----------------------------------------
        stability_reference = _target_position(cube_id, scene.client_id)
        stability_samples = []
        for _ in range(60):
            scene.step()
            stability_samples.append(_target_position(cube_id, scene.client_id))
        stability_displacement = _maximum_displacement(
            stability_reference, stability_samples
        )
        if stability_displacement > 0.001:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "target_stability_preflight",
                stability_displacement_m=stability_displacement,
            )

        # -- 4. Read truth (for gate audit only, NOT for pose generation) ---
        cube_position, cube_quaternion = p.getBasePositionAndOrientation(
            cube_id, physicsClientId=scene.client_id
        )
        cube_position = tuple(float(v) for v in cube_position)
        cube_quaternion = tuple(float(v) for v in cube_quaternion)
        cube_aabb = p.getAABB(cube_id, physicsClientId=scene.client_id)
        cube_top_z = float(cube_aabb[1][2])

        # -- 5. Extract plan poses and IK solutions --------------------------
        pregrasp_pose = candidate.pregrasp_pose
        approach_pose = candidate.approach_pose
        grasp_depth_pose = candidate.grasp_depth_pose
        pregrasp_ik_solution = candidate.pregrasp_ik
        approach_ik_solution = candidate.approach_ik
        grasp_depth_ik_solution = candidate.grasp_depth_ik

        # -- 6. Compute lift pose and IK (NOT in the plan) ------------------
        lift_pose = ToolPose(
            position=(
                grasp_depth_pose.position[0],
                grasp_depth_pose.position[1],
                grasp_depth_pose.position[2] + plan.control.tool_lift_command_m,
            ),
            quaternion_xyzw=grasp_depth_pose.quaternion_xyzw,
        )
        lift_ik = audit_pose_ik(
            scene.bodies.robot, scene.client_id, model, lift_pose
        )
        if not lift_ik.gate_passed or lift_ik.solution is None:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "lift_ik_failed",
                lift_ik_gate_passed=lift_ik.gate_passed,
            )

        # -- 7. Preflight clearance audit (two-phase, matching Stage 6A) ----
        environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        allowed_mounting_pair = ((-1, scene.bodies.table),)

        # 7a. Contact-pre: neutral → approach, includes all objects (cube too)
        contact_pre_environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        contact_pre_clearance = audit_joint_path_clearance(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            start_solution=neutral,
            pregrasp_solution=tuple(pregrasp_ik_solution),
            standoff_solution=tuple(approach_ik_solution),
            environment_body_ids=contact_pre_environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        if not contact_pre_clearance.clearance_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "contact_pre_clearance_failed",
                clearance_passed=False,
            )

        # 7b. Grasp-depth: approach → grasp_depth, excludes target cube
        grasp_depth_environment = (
            scene.bodies.plane,
            scene.bodies.table,
            scene.object_body_ids["duck"],
            scene.object_body_ids["sphere"],
        )
        grasp_depth_clearance = audit_joint_path_clearance(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            start_solution=tuple(approach_ik_solution),
            pregrasp_solution=tuple(approach_ik_solution),
            standoff_solution=tuple(grasp_depth_ik_solution),
            environment_body_ids=grasp_depth_environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        if not grasp_depth_clearance.clearance_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "grasp_depth_clearance_failed",
                clearance_passed=False,
            )

        # 7c. Lift clearance (exclude cube)
        lift_preflight_clearance = audit_joint_path_clearance(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            start_solution=tuple(grasp_depth_ik_solution),
            pregrasp_solution=tuple(grasp_depth_ik_solution),
            standoff_solution=lift_ik.solution,
            environment_body_ids=grasp_depth_environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        if not lift_preflight_clearance.clearance_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "lift_preflight_clearance_failed",
                clearance_passed=False,
            )

        # -- 8. Execute motions ----------------------------------------------
        motion_start_target_position = _target_position(cube_id, scene.client_id)
        motion_config = MotionConfig(
            steps_per_segment=1920,
            settle_steps=960,
            joint_tolerance_rad=0.002,
        )

        # 8a. pregrasp
        pregrasp_execution = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(MotionSegment("pregrasp", tuple(pregrasp_ik_solution)),),
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            tracked_body_ids=(cube_id,),
            config=motion_config,
        )
        pregrasp_rgb = _capture_rgb(scene, plan.camera)
        _pause()

        if not pregrasp_execution.gate_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "pregrasp_motion_failed",
                pregrasp_rgb=pregrasp_rgb,
                pregrasp_execution_gate=pregrasp_execution.gate_passed,
            )

        pregrasp_reached = dict(pregrasp_execution.segment_reached)["pregrasp"]
        pregrasp_position_error, pregrasp_orientation_error = _pose_errors(
            pregrasp_execution.trace[-1],
            pregrasp_pose.position,
            pregrasp_pose.quaternion_xyzw,
        )
        pregrasp_dynamic_gate = (
            pregrasp_reached
            and pregrasp_position_error <= POSITION_ERROR_THRESHOLD_M
            and pregrasp_orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
        )

        if not pregrasp_dynamic_gate:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "pregrasp_dynamic_gate_failed",
                pregrasp_rgb=pregrasp_rgb,
                pregrasp_position_error_m=pregrasp_position_error,
                pregrasp_orientation_error_deg=pregrasp_orientation_error,
            )

        # 8b. approach
        approach_execution = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(MotionSegment("approach", tuple(approach_ik_solution)),),
            environment_body_ids=environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            tracked_body_ids=(cube_id,),
            config=motion_config,
        )
        approach_rgb = _capture_rgb(scene, plan.camera)
        _pause()

        if not approach_execution.gate_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "approach_motion_failed",
                pregrasp_rgb=pregrasp_rgb,
                approach_rgb=approach_rgb,
                approach_execution_gate=approach_execution.gate_passed,
            )

        approach_reached = dict(approach_execution.segment_reached)["approach"]
        approach_row = approach_execution.trace[-1]
        approach_position_error, approach_orientation_error = _pose_errors(
            approach_row,
            approach_pose.position,
            approach_pose.quaternion_xyzw,
        )
        approach_cube_positions = tuple(
            row.tracked_body_poses[0].position
            for result in (pregrasp_execution, approach_execution)
            for row in result.trace
        )
        approach_target_undisturbed = (
            _maximum_displacement(motion_start_target_position, approach_cube_positions)
            <= 0.001
        )
        approach_fingers_open = all(
            abs(value - 0.04) <= FINGER_OPEN_ERROR_THRESHOLD_M
            for result in (pregrasp_execution, approach_execution)
            for row in result.trace
            for value in row.actual_finger_positions
        )
        approach_dynamic_gate = (
            approach_reached
            and approach_position_error <= POSITION_ERROR_THRESHOLD_M
            and approach_orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
            and approach_target_undisturbed
            and approach_fingers_open
        )

        if not approach_dynamic_gate:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "approach_dynamic_gate_failed",
                pregrasp_rgb=pregrasp_rgb,
                approach_rgb=approach_rgb,
                approach_position_error_m=approach_position_error,
                approach_orientation_error_deg=approach_orientation_error,
                approach_target_undisturbed=approach_target_undisturbed,
                approach_fingers_open=approach_fingers_open,
            )

        # 8c. grasp_depth (exclude cube from collision check -- contact
        # is expected and will be handled by gripper close logic)
        grasp_depth_execution = execute_joint_motion(
            robot_id=scene.bodies.robot,
            client_id=scene.client_id,
            model=model,
            segments=(
                MotionSegment("grasp_depth", tuple(grasp_depth_ik_solution)),
            ),
            environment_body_ids=grasp_depth_environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            tracked_body_ids=(cube_id,),
            config=motion_config,
        )
        grasp_depth_rgb = _capture_rgb(scene, plan.camera)
        _pause()

        if not grasp_depth_execution.gate_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "grasp_depth_motion_failed",
                pregrasp_rgb=pregrasp_rgb,
                approach_rgb=approach_rgb,
                grasp_depth_rgb=grasp_depth_rgb,
                grasp_depth_execution_gate=grasp_depth_execution.gate_passed,
            )

        grasp_row = grasp_depth_execution.trace[-1]
        grasp_position_error, grasp_orientation_error = _pose_errors(
            grasp_row,
            grasp_depth_pose.position,
            grasp_depth_pose.quaternion_xyzw,
        )
        grasp_depth_reached = dict(grasp_depth_execution.segment_reached)[
            "grasp_depth"
        ]
        grasp_cube_positions = tuple(
            row.tracked_body_poses[0].position
            for result in (
                pregrasp_execution,
                approach_execution,
                grasp_depth_execution,
            )
            for row in result.trace
        )
        grasp_target_undisturbed = (
            _maximum_displacement(motion_start_target_position, grasp_cube_positions)
            <= 0.001
        )
        grasp_fingers_open = all(
            abs(value - 0.04) <= FINGER_OPEN_ERROR_THRESHOLD_M
            for row in grasp_depth_execution.trace
            for value in row.actual_finger_positions
        )
        grasp_depth_dynamic_gate = (
            grasp_depth_reached
            and grasp_position_error <= POSITION_ERROR_THRESHOLD_M
            and grasp_orientation_error <= ORIENTATION_ERROR_THRESHOLD_DEGREES
            and grasp_target_undisturbed
            and grasp_fingers_open
        )

        if not grasp_depth_dynamic_gate:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "grasp_depth_dynamic_gate_failed",
                pregrasp_rgb=pregrasp_rgb,
                approach_rgb=approach_rgb,
                grasp_depth_rgb=grasp_depth_rgb,
                grasp_position_error_m=grasp_position_error,
                grasp_orientation_error_deg=grasp_orientation_error,
                grasp_target_undisturbed=grasp_target_undisturbed,
                grasp_fingers_open=grasp_fingers_open,
            )

        # 8d. close gripper
        gripper_result = execute_gripper_close(
            robot_id=scene.bodies.robot,
            target_body_id=cube_id,
            client_id=scene.client_id,
            model=model,
            arm_hold_positions=tuple(grasp_depth_ik_solution),
            environment_body_ids=(
                scene.bodies.plane,
                scene.bodies.table,
                scene.object_body_ids["duck"],
                scene.object_body_ids["sphere"],
            ),
            allowed_environment_link_pairs=allowed_mounting_pair,
        )
        closed_rgb = _capture_rgb(scene, plan.camera)
        _pause()

        if not gripper_result.gate_passed:
            return _write_failure(
                output_dir,
                config,
                plan,
                candidate,
                start_rgb,
                "gripper_close_failed",
                pregrasp_rgb=pregrasp_rgb,
                approach_rgb=approach_rgb,
                grasp_depth_rgb=grasp_depth_rgb,
                closed_rgb=closed_rgb,
                gripper_gate_passed=gripper_result.gate_passed,
                bilateral_contact_acquired=(
                    gripper_result.bilateral_contact_acquired
                ),
            )

        # 8e. lift
        close_row = gripper_result.trace[-1]
        lifted_rgb: np.ndarray | None = None

        def capture_lifted() -> None:
            nonlocal lifted_rgb
            lifted_rgb = _capture_rgb(scene, plan.camera)

        lift_config = LiftConfig()
        lift_result = execute_object_lift(
            robot_id=scene.bodies.robot,
            target_body_id=cube_id,
            table_body_id=scene.bodies.table,
            client_id=scene.client_id,
            model=model,
            lift_arm_positions=lift_ik.solution,
            lift_target_pose=lift_pose,
            frozen_finger_positions=close_row.commanded_finger_positions,
            reference_target_position=close_row.target_position,
            reference_tool_relative_to_target=tuple(
                tool - target
                for tool, target in zip(
                    close_row.actual_tool_position,
                    close_row.target_position,
                )
            ),
            environment_body_ids=grasp_depth_environment,
            allowed_environment_link_pairs=allowed_mounting_pair,
            config=lift_config,
            lift_complete_callback=capture_lifted,
        )
        lift_hold_rgb = _capture_rgb(scene, plan.camera)
        _pause()

        # -- 9. Gate audit ---------------------------------------------------
        grasp_depth_height_above_cube_top = (
            grasp_row.actual_tool_position[2] - cube_top_z
        )
        grasp_depth_height_error = abs(
            grasp_depth_height_above_cube_top
            - plan.control.grasp_depth_standoff_m
        )
        grasp_depth_height_gate_passed = bool(
            grasp_depth_height_error <= POSITION_ERROR_THRESHOLD_M
        )

        # XY bias: where the tool actually ended up vs cube true center
        tool_xy = (
            grasp_row.actual_tool_position[0],
            grasp_row.actual_tool_position[1],
        )
        cube_xy = (cube_position[0], cube_position[1])
        xy_bias_m = float(
            np.linalg.norm(
                np.asarray(tool_xy, dtype=np.float64)
                - np.asarray(cube_xy, dtype=np.float64)
            )
        )

        # check lift gate
        lift_end_row = lift_result.trace[-1] if lift_result.trace else None
        lift_end_position_error = None
        lift_end_orientation_error = None
        if lift_end_row is not None:
            lift_end_position_error = float(
                np.linalg.norm(
                    np.asarray(lift_end_row.actual_tool_position, dtype=np.float64)
                    - np.asarray(lift_pose.position, dtype=np.float64)
                )
            )
            lift_end_orientation_error = _orientation_error_degrees(
                lift_end_row.actual_tool_quaternion_xyzw,
                lift_pose.quaternion_xyzw,
            )

        scientific_gate_passed = (
            pregrasp_dynamic_gate
            and approach_dynamic_gate
            and grasp_depth_dynamic_gate
            and grasp_depth_height_gate_passed
            and gripper_result.gate_passed
            and lift_result.gate_passed
        )

        # -- 10. Merge traces -------------------------------------------------
        motion_rows: list[dict[str, object]] = []
        for result in (
            pregrasp_execution,
            approach_execution,
            grasp_depth_execution,
        ):
            for row in result.trace:
                motion_rows.append(
                    {
                        "step": row.step,
                        "phase": row.phase,
                        "actual_tool_position_x": row.actual_tool_position[0],
                        "actual_tool_position_y": row.actual_tool_position[1],
                        "actual_tool_position_z": row.actual_tool_position[2],
                        "actual_tool_quaternion_w": (
                            row.actual_tool_quaternion_xyzw[3]
                        ),
                        "actual_tool_quaternion_x": (
                            row.actual_tool_quaternion_xyzw[0]
                        ),
                        "actual_tool_quaternion_y": (
                            row.actual_tool_quaternion_xyzw[1]
                        ),
                        "actual_tool_quaternion_z": (
                            row.actual_tool_quaternion_xyzw[2]
                        ),
                        "target_body_x": row.tracked_body_poses[0].position[0]
                        if row.tracked_body_poses
                        else None,
                        "target_body_y": row.tracked_body_poses[0].position[1]
                        if row.tracked_body_poses
                        else None,
                        "target_body_z": row.tracked_body_poses[0].position[2]
                        if row.tracked_body_poses
                        else None,
                        "actual_finger_left": row.actual_finger_positions[0],
                        "actual_finger_right": row.actual_finger_positions[1],
                        "minimum_clearance_m": row.minimum_clearance_m,
                        "environment_collision_count": (
                            row.environment_collision_count
                        ),
                        "self_collision_count": row.self_collision_count,
                    }
                )

        gripper_rows: list[dict[str, object]] = []
        for row in gripper_result.trace:
            gripper_rows.append(
                {
                    "step": row.step,
                    "phase": row.phase,
                    "actual_tool_position_x": row.actual_tool_position[0],
                    "actual_tool_position_y": row.actual_tool_position[1],
                    "actual_tool_position_z": row.actual_tool_position[2],
                    "actual_tool_quaternion_w": row.actual_tool_quaternion_xyzw[3],
                    "actual_tool_quaternion_x": row.actual_tool_quaternion_xyzw[0],
                    "actual_tool_quaternion_y": row.actual_tool_quaternion_xyzw[1],
                    "actual_tool_quaternion_z": row.actual_tool_quaternion_xyzw[2],
                    "target_body_x": row.target_position[0],
                    "target_body_y": row.target_position[1],
                    "target_body_z": row.target_position[2],
                    "actual_finger_left": row.actual_finger_positions[0],
                    "actual_finger_right": row.actual_finger_positions[1],
                    "commanded_finger_left": row.commanded_finger_positions[0],
                    "commanded_finger_right": row.commanded_finger_positions[1],
                    "left_normal_force": row.left_normal_force,
                    "right_normal_force": row.right_normal_force,
                    "environment_collision_count": row.environment_collision_count,
                    "self_collision_count": row.self_collision_count,
                }
            )

        lift_rows: list[dict[str, object]] = []
        for row in lift_result.trace:
            lift_rows.append(
                {
                    "step": row.step,
                    "phase": row.phase,
                    "actual_tool_position_x": row.actual_tool_position[0],
                    "actual_tool_position_y": row.actual_tool_position[1],
                    "actual_tool_position_z": row.actual_tool_position[2],
                    "actual_tool_quaternion_w": row.actual_tool_quaternion_xyzw[3],
                    "actual_tool_quaternion_x": row.actual_tool_quaternion_xyzw[0],
                    "actual_tool_quaternion_y": row.actual_tool_quaternion_xyzw[1],
                    "actual_tool_quaternion_z": row.actual_tool_quaternion_xyzw[2],
                    "target_body_x": row.target_position[0],
                    "target_body_y": row.target_position[1],
                    "target_body_z": row.target_position[2],
                    "actual_finger_left": row.actual_finger_positions[0],
                    "actual_finger_right": row.actual_finger_positions[1],
                    "commanded_finger_left": row.commanded_finger_positions[0],
                    "commanded_finger_right": row.commanded_finger_positions[1],
                    "left_normal_force": row.left_normal_force,
                    "right_normal_force": row.right_normal_force,
                    "target_lift_m": row.target_lift_m,
                    "target_table_contact": row.target_table_contact,
                    "relative_drift_m": row.relative_drift_m,
                    "environment_collision_count": row.environment_collision_count,
                    "self_collision_count": row.self_collision_count,
                }
            )

        merged_trace = motion_rows + gripper_rows + lift_rows

        # -- 11. Contact events ----------------------------------------------
        contact_events: list[dict[str, object]] = []
        for evt in gripper_result.contact_events:
            contact_events.append(
                {
                    "step": evt.step,
                    "phase": evt.phase,
                    "robot_link": evt.robot_link,
                    "target_body": evt.target_body,
                    "normal_force": evt.normal_force,
                }
            )
        for evt in lift_result.contact_events:
            contact_events.append(
                {
                    "step": evt.step,
                    "phase": evt.phase,
                    "robot_link": evt.robot_link,
                    "target_body": evt.target_body,
                    "normal_force": evt.normal_force,
                }
            )

        # -- 12. Write artifacts ---------------------------------------------
        _write_rgb(output_dir / "start.png", start_rgb)
        _write_rgb(output_dir / "pregrasp.png", pregrasp_rgb)
        _write_rgb(output_dir / "approach.png", approach_rgb)
        _write_rgb(output_dir / "grasp_depth.png", grasp_depth_rgb)
        _write_rgb(output_dir / "closed.png", closed_rgb)
        if lifted_rgb is not None:
            _write_rgb(output_dir / "lifted.png", lifted_rgb)
        _write_rgb(output_dir / "lift_hold.png", lift_hold_rgb)
        execution_start_rgb_sha256 = _sha256_file(output_dir / "start.png")

        _write_trace(output_dir / "state_trace.csv", merged_trace)
        _write_contact_events(output_dir / "contact_events.csv", contact_events)

        summary: dict[str, object] = {
            "protocol": "stage_6b_perception_grasp_v1",
            "plan_protocol": plan.protocol_version,
            "plan_file_sha256": _sha256_file(config.plan_path),
            "plan_rgb_sha256": plan.rgb_sha256,
            "execution_start_rgb_sha256": execution_start_rgb_sha256,
            "execution_start_rgb_matches_plan": (
                execution_start_rgb_sha256 == plan.rgb_sha256
            ),
            "backend": plan.backend,
            "target_name": plan.target_name,
            "perception_world_surface_point": list(
                plan.perception.world_surface_point
            ),
            "perception_2d_center": list(plan.perception.grasp_center),
            "cube_true_position": list(cube_position),
            "cube_top_z_m": cube_top_z,
            "xy_bias_m": xy_bias_m,
            "xy_bias_mm": xy_bias_m * 1000.0,
            "pregrasp_dynamic_gate": pregrasp_dynamic_gate,
            "pregrasp_position_error_m": pregrasp_position_error,
            "pregrasp_orientation_error_deg": pregrasp_orientation_error,
            "approach_dynamic_gate": approach_dynamic_gate,
            "approach_position_error_m": approach_position_error,
            "approach_orientation_error_deg": approach_orientation_error,
            "grasp_depth_dynamic_gate": grasp_depth_dynamic_gate,
            "grasp_depth_position_error_m": grasp_position_error,
            "grasp_depth_orientation_error_deg": grasp_orientation_error,
            "grasp_depth_height_above_cube_top_m": grasp_depth_height_above_cube_top,
            "grasp_depth_height_error_m": grasp_depth_height_error,
            "grasp_depth_height_gate_passed": (
                grasp_depth_height_gate_passed
            ),
            "bilateral_contact_acquired": gripper_result.bilateral_contact_acquired,
            "first_bilateral_contact_step": gripper_result.first_bilateral_contact_step,
            "contact_hold_steps": (
                len(gripper_result.trace)
                - (gripper_result.first_bilateral_contact_step or 0)
                if gripper_result.bilateral_contact_acquired
                else 0
            ),
            "lift_gate_passed": lift_result.gate_passed,
            "minimum_hold_object_lift_m": lift_result.minimum_hold_object_lift_m,
            "final_object_lift_m": lift_result.final_object_lift_m,
            "total_target_table_contact_count": lift_result.total_target_table_contact_count,
            "maximum_hold_relative_drift_m": lift_result.maximum_hold_relative_drift_m,
            "trailing_bilateral_contact_steps": (
                lift_result.trailing_bilateral_contact_steps
            ),
            "gripper_gate_passed": gripper_result.gate_passed,
            "lift_end_position_error_m": lift_end_position_error,
            "lift_end_orientation_error_deg": lift_end_orientation_error,
            "total_trace_steps": len(merged_trace),
            "total_contact_events": len(contact_events),
            "scientific_gate_passed": scientific_gate_passed,
        }
        _write_json(output_dir / "summary.json", summary)

        metadata = _build_metadata(
            config,
            plan,
            candidate,
            **{
                "scientific_gate_passed": scientific_gate_passed,
                "xy_bias_m": xy_bias_m,
                "gripper_close_executed": True,
                "object_lift_executed": True,
                "motor_control_executed": True,
                "trajectory_executed": True,
                "gripper_closed": True,
                "contact_evaluated": True,
                "object_lifted": lift_result.object_lift_gate_passed,
                "physical_grasp_executed": True,
                "execution_start_rgb_sha256": execution_start_rgb_sha256,
                "execution_start_rgb_matches_plan": (
                    execution_start_rgb_sha256 == plan.rgb_sha256
                ),
                "perception_used": True,
                "perception_world_surface_point": list(
                    plan.perception.world_surface_point
                ),
            },
        )
        _write_json(output_dir / "metadata.json", metadata)

        if config.gui:
            print("Simulation complete. Close the PyBullet window or press Ctrl+C to exit.")
            time.sleep(300)  # keep alive for 5 minutes

        return summary


def _write_failure(
    output_dir: Path,
    config: Stage6BConfig,
    plan: PerceptionExecutionPlan,
    candidate: PlannedPoseCandidate,
    start_rgb: np.ndarray,
    failure_stage: str,
    **extra: object,
) -> dict[str, object]:
    """Write minimal failure artifacts and return a failure summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(output_dir / "start.png", start_rgb)
    execution_start_rgb_sha256 = _sha256_file(output_dir / "start.png")
    # Write any extra RGB images passed as kwargs
    for key, value in list(extra.items()):
        if isinstance(value, np.ndarray) and key.endswith("_rgb"):
            _write_rgb(output_dir / f"{key}.png", value)
    # Drop numpy arrays from summary (not JSON-serializable)
    safe_extra = {
        key: value
        for key, value in extra.items()
        if not isinstance(value, np.ndarray)
    }
    summary: dict[str, object] = {
        "protocol": "stage_6b_perception_grasp_v1",
        "plan_protocol": plan.protocol_version,
        "plan_file_sha256": _sha256_file(config.plan_path),
        "plan_rgb_sha256": plan.rgb_sha256,
        "execution_start_rgb_sha256": execution_start_rgb_sha256,
        "execution_start_rgb_matches_plan": (
            execution_start_rgb_sha256 == plan.rgb_sha256
        ),
        "status": "failure",
        "failure_stage": failure_stage,
        "scientific_gate_passed": False,
        **safe_extra,
    }
    _write_json(output_dir / "summary.json", summary)
    metadata = _build_metadata(
        config,
        plan,
        candidate,
        **{
            "status": "failure",
            "failure_stage": failure_stage,
            "scientific_gate_passed": False,
            "physical_grasp_executed": False,
            "execution_start_rgb_sha256": execution_start_rgb_sha256,
            "execution_start_rgb_matches_plan": (
                execution_start_rgb_sha256 == plan.rgb_sha256
            ),
        },
    )
    _write_json(output_dir / "metadata.json", metadata)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6B: execute a perception-derived grasp plan."
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Path to Stage 6A execution_plan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    args = parser.parse_args()

    config = Stage6BConfig(
        plan_path=args.plan_path,
        output_dir=args.output_dir,
        gui=args.gui,
        device=args.device,
    )
    result = run_stage6b(config)
    passed = result.get("scientific_gate_passed", False)
    status = "PASSED" if passed else "FAILED"
    print(json.dumps(
        {"stage": "6B", "scientific_gate_passed": passed, "status": status},
        indent=2,
    ))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
