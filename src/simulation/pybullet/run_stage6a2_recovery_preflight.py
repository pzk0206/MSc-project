"""Stage 6A.2: Common centre recovery for geometry and multi-head CNN.

Re-runs the Stage 6A perception pipeline but replaces single-pixel
depth backprojection with windowed top-surface depth recovery.
The same recovery rule is applied identically to both backends,
producing two independent, auditable execution plans.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.backend_comparison import (
    evaluate_backend_grasp,
)
from src.simulation.pybullet.backprojection import (
    RayTest,
    audit_backprojected_grasp,
)
from src.simulation.pybullet.camera import (
    CameraConfig,
    CameraFrame,
    capture_camera_frame,
)
from src.simulation.pybullet.center_recovery import (
    CENTER_RECOVERY_PROTOCOL,
    CENTER_RECOVERY_WINDOW_SIZE,
    recover_center_via_windowed_depth,
)
from src.simulation.pybullet.execution_plan import (
    CameraEvidence,
    CenterRecoveryEvidence,
    FrozenControlProtocol,
    PerceptionEvidence,
    PerceptionExecutionPlan,
    PlannedPoseCandidate,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_V2,
    VALID_BACKENDS,
    write_perception_execution_plan,
)
from src.simulation.pybullet.kinematic_audit import (
    CandidateAudit,
    CollisionAudit,
    IKPoseAudit,
    audit_pose_candidate,
    resolve_panda_model,
    select_candidate_pair,
)
from src.simulation.pybullet.perception import (
    Localization,
    PilotPrediction,
    load_cnn_backend,
    load_grounding_dino,
    localize_object,
    predict_grasp,
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
from src.simulation.pybullet.target_selection import (
    evaluate_target_selection,
    mask_to_box,
    segmentation_mask_for_body,
)
from src.simulation.pybullet.visualization import (
    draw_prediction,
    draw_target_evaluation,
    segmentation_to_bgr,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/"
    "stage_6a2_center_recovery"
)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MULTI_HEAD_WEIGHTS = Path(
    "data/processed/vlm/cnn_grasp_multi_head_deterministic/"
    "cnn_grasp_model_seed_42.pt"
)
BACKENDS = ("geometry", "multi_head")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage6A2Config:
    """Frozen Stage 6A.2 protocol."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    gui: bool = False
    device: str = "cuda"
    target_name: str = "cube"
    prompt: str = "red cube"
    backends: tuple[str, ...] = BACKENDS
    model_id: str = MODEL_ID
    multi_head_weights: Path = MULTI_HEAD_WEIGHTS
    width: int = 640
    height: int = 480
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    iou_threshold: float = 0.25
    center_recovery_window_size: int = CENTER_RECOVERY_WINDOW_SIZE

    def __post_init__(self) -> None:
        if self.seed != 42:
            raise ValueError("Stage 6A.2 seed must be 42")
        if self.target_name != "cube":
            raise ValueError("Stage 6A.2 target_name must be cube")
        if self.prompt != "red cube":
            raise ValueError("Stage 6A.2 prompt must be red cube")
        if self.model_id != MODEL_ID:
            raise ValueError("Stage 6A.2 model_id is frozen")
        if (self.width, self.height) != (640, 480):
            raise ValueError("Stage 6A.2 camera must be 640x480")
        for name in ("box_threshold", "text_threshold", "iou_threshold"):
            if not math.isclose(getattr(self, name), 0.25, abs_tol=0.0):
                raise ValueError(f"Stage 6A.2 {name} must be 0.25")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.center_recovery_window_size < 1 or (
            self.center_recovery_window_size % 2 != 1
        ):
            raise ValueError("center_recovery_window_size must be odd positive")
        backend_set = set(self.backends)
        if backend_set != {"geometry", "multi_head"}:
            raise ValueError(
                "Stage 6A.2 must run both geometry and multi_head backends"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to save image: {path}")


def _prepare_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    top_level = (
        "rgb.png",
        "depth.npy",
        "segmentation.png",
        "localization.png",
        "geometry_prediction.png",
        "multi_head_prediction.png",
        "summary.json",
        "metadata.json",
    )
    for name in top_level:
        (output_dir / name).unlink(missing_ok=True)
    for backend_dir in BACKENDS:
        sub = output_dir / backend_dir
        sub.mkdir(parents=True, exist_ok=True)
        for name in (
            "execution_plan.json",
            "candidates.csv",
            "summary.json",
            "metadata.json",
        ):
            (sub / name).unlink(missing_ok=True)


def _base_metadata(config: Stage6A2Config) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_VERSION_V2,
        "center_recovery_protocol": CENTER_RECOVERY_PROTOCOL,
        "config": asdict(config),
        "segmentation_used_as_model_input": False,
        "segmentation_used_as_coordinate_input": True,
        "segmentation_used_for": "center_recovery_windowed_depth",
        "ray_test_used_as_coordinate_input": False,
        "depth_used_after_2d_prediction": True,
        "simulation_setup_steps": 60,
        "simulation_steps_after_capture": 0,
        "inverse_kinematics_executed": False,
        "forward_kinematics_verified": False,
        "static_joint_resets_used": False,
        "motor_control_executed": False,
        "trajectory_executed": False,
        "gripper_closed": False,
        "contact_evaluated": False,
        "object_lifted": False,
        "physical_grasp_executed": False,
    }


def _candidate_csv_row(
    audit: CandidateAudit,
    approach_audit: CandidateAudit,
    depth_audit: CandidateAudit,
) -> dict[str, object]:
    candidate = audit.candidate
    approach_ik = approach_audit.standoff_ik
    return {
        "symmetry_degrees": candidate.symmetry_degrees,
        "finger_axis_world": json.dumps(candidate.finger_axis_world),
        "pregrasp_position": json.dumps(candidate.pregrasp_pose.position),
        "approach_position": "",
        "grasp_depth_position": json.dumps(
            candidate.surface_standoff_pose.position
        ),
        "pregrasp_ik": json.dumps(audit.pregrasp_ik.solution),
        "approach_ik": json.dumps(approach_ik.solution),
        "grasp_depth_ik": json.dumps(audit.standoff_ik.solution),
        "ik_fk_passed": (
            audit.pregrasp_ik.gate_passed
            and approach_ik.gate_passed
            and audit.standoff_ik.gate_passed
        ),
        "clearance_passed": audit.collision.clearance_passed,
        "checked_state_count": audit.collision.checked_state_count,
        "precontact_clearance_passed": (
            approach_audit.collision.clearance_passed
        ),
        "precontact_checked_state_count": (
            approach_audit.collision.checked_state_count
        ),
        "precontact_minimum_clearance_m": (
            approach_audit.collision.minimum_clearance_m
        ),
        "grasp_depth_clearance_passed": (
            depth_audit.collision.clearance_passed
        ),
        "grasp_depth_checked_state_count": (
            depth_audit.collision.checked_state_count
        ),
        "grasp_depth_minimum_clearance_m": (
            depth_audit.collision.minimum_clearance_m
        ),
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


def _write_candidates(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _planned_candidate(
    audit: CandidateAudit,
    approach_candidate: PoseCandidate,
    approach_ik: IKPoseAudit,
) -> PlannedPoseCandidate:
    if (
        audit.pregrasp_ik.solution is None
        or approach_ik.solution is None
        or audit.standoff_ik.solution is None
    ):
        raise ValueError("all three candidate IK solutions are required")
    return PlannedPoseCandidate(
        symmetry_degrees=audit.candidate.symmetry_degrees,
        finger_axis_world=audit.candidate.finger_axis_world,
        pregrasp_pose=audit.candidate.pregrasp_pose,
        approach_pose=approach_candidate.surface_standoff_pose,
        grasp_depth_pose=audit.candidate.surface_standoff_pose,
        pregrasp_ik=audit.pregrasp_ik.solution,
        approach_ik=approach_ik.solution,
        grasp_depth_ik=audit.standoff_ik.solution,
        ik_fk_passed=(
            audit.pregrasp_ik.gate_passed
            and approach_ik.gate_passed
            and audit.standoff_ik.gate_passed
        ),
        clearance_passed=audit.collision.clearance_passed,
        checked_state_count=audit.collision.checked_state_count,
        minimum_clearance_m=audit.collision.minimum_clearance_m,
        environment_collision_count=(
            audit.collision.environment_collision_count
        ),
        self_collision_count=audit.collision.self_collision_count,
        total_normalized_joint_cost=audit.total_normalized_joint_cost,
        gate_passed=audit.gate_passed,
        selected=audit.selected,
        failure_reason=audit.failure_reason,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


# trunk-ignore(pyright/reportMissingParameterType)
def run_stage6a2_recovery_preflight(
    config: Stage6A2Config,
    # trunk-ignore(pyright/reportMissingParameterType)
    dependencies=None,
) -> dict[str, object]:
    """Run geometry and multi-head preflight with shared centre recovery."""

    # trunk-ignore(pyright/reportUnusedExpression)
    dependencies  # reserved for test injection; production uses defaults

    output_dir = Path(config.output_dir)
    _prepare_output(output_dir)
    metadata = _base_metadata(config)
    camera_config = CameraConfig(width=config.width, height=config.height)
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

    # -- scene + camera --------------------------------------------------
    with PyBulletScene(scene_config) as scene:
        scene.step(60)
        frame = capture_camera_frame(
            scene.client_id,
            camera_config,
            scene.renderer,
        )
        _save_image(
            output_dir / "rgb.png",
            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
        )
        np.save(output_dir / "depth.npy", frame.depth_m)
        _save_image(
            output_dir / "segmentation.png",
            segmentation_to_bgr(frame.segmentation),
        )
        rgb_sha256 = hashlib.sha256(
            (output_dir / "rgb.png").read_bytes()
        ).hexdigest()
        entity_ids = {
            **scene.object_body_ids,
            "robot": scene.bodies.robot,
        }
        masks = {
            name: segmentation_mask_for_body(frame.segmentation, body_id)
            for name, body_id in entity_ids.items()
        }
        boxes = {name: mask_to_box(mask) for name, mask in masks.items()}
        metadata.update(
            scene={
                "config": asdict(scene.config),
                "body_ids": entity_ids,
                "object_poses": scene.object_poses(),
            },
            camera={
                "config": asdict(camera_config),
                "view_matrix": frame.view_matrix,
                "projection_matrix": frame.projection_matrix,
            },
            rgb_sha256=rgb_sha256,
        )

        # -- Grounding DINO -----------------------------------------------
        processor, detector = load_grounding_dino(
            config.model_id,
            config.device,
        )
        localization = localize_object(
            rgb_path=output_dir / "rgb.png",
            prompt=config.prompt,
            processor=processor,
            model=detector,
            device=config.device,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
        if localization is None:
            return _failure(
                output_dir=output_dir,
                metadata=metadata,
                stage="localization",
                reason="no_detection",
            )
        target_evaluation = evaluate_target_selection(
            localization.box,
            config.target_name,
            boxes,
            config.iou_threshold,
        )
        _save_image(
            output_dir / "localization.png",
            draw_target_evaluation(
                frame.rgb,
                config.target_name,
                config.prompt,
                localization.box,
                boxes,
                target_evaluation.best_matching_target,
                localization.score,
            ),
        )
        if not target_evaluation.correct_target:
            return _failure(
                output_dir=output_dir,
                metadata=metadata,
                stage="target_selection",
                reason=target_evaluation.failure_reason,
                details={
                    "localization_iou": target_evaluation.requested_target_iou,
                },
            )

        # -- shared scene data for pose generation ------------------------
        target_body_id = entity_ids[config.target_name]
        model = resolve_panda_model(scene.bodies.robot, scene.client_id)
        environment = (
            scene.bodies.plane,
            scene.bodies.table,
            *scene.object_body_ids.values(),
        )
        non_target_environment = (
            scene.bodies.plane,
            scene.bodies.table,
            scene.object_body_ids["duck"],
            scene.object_body_ids["sphere"],
        )

        camera_evidence = CameraEvidence(
            width=camera_config.width,
            height=camera_config.height,
            eye=camera_config.eye,
            target=camera_config.target,
            up=camera_config.up,
            fov_degrees=camera_config.fov_degrees,
            near=camera_config.near,
            far=camera_config.far,
            view_matrix=frame.view_matrix,
            projection_matrix=frame.projection_matrix,
        )

        # -- shared control protocol --------------------------------------
        control = FrozenControlProtocol()

        # -- load CNN model once for multi_head ---------------------------
        multi_head_model = load_cnn_backend(
            "multi_head",
            config.multi_head_weights,
            config.device,
        )

        # -- per-backend results ------------------------------------------
        backend_results: dict[str, dict[str, Any]] = {}
        all_gate_passed = True

        for backend_name in BACKENDS:
            backend_output_dir = output_dir / backend_name

            # --- predict 2-D grasp ---------------------------------------
            image_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if backend_name == "geometry":
                cnn_model = None
            else:
                cnn_model = multi_head_model
            prediction = predict_grasp(
                image_bgr,
                localization,
                backend_name,
                config.device,
                cnn_model,
            )
            grasp = prediction.grasp
            _save_image(
                output_dir / f"{backend_name}_prediction.png",
                draw_prediction(
                    frame.rgb,
                    localization.box,
                    grasp,
                    config.prompt,
                    localization.score,
                    backend_name,
                ),
            )

            # --- backend geometry audit ----------------------------------
            backend_evaluation = evaluate_backend_grasp(
                grasp,
                masks[config.target_name],
                config.width,
                config.height,
            )
            backend_ok = (
                backend_evaluation.parameters_finite
                and backend_evaluation.positive_size
                and backend_evaluation.center_inside_target_mask
                and backend_evaluation.box_inside_image
            )
            if not backend_ok:
                all_gate_passed = False
                backend_results[backend_name] = {
                    "scientific_gate_passed": False,
                    "failure_stage": "backend_audit",
                    "failure_reason": backend_evaluation.failure_reason,
                }
                _write_json(
                    backend_output_dir / "summary.json",
                    backend_results[backend_name],
                )
                continue

            # --- single-pixel backprojection (for audit / comparison) ----
            backend_row = {
                "target": config.target_name,
                "backend": backend_name,
                "center_x": grasp["center_x"],
                "center_y": grasp["center_y"],
            }
            backprojection = audit_backprojected_grasp(
                backend_row=backend_row,
                depth_m=frame.depth_m,
                segmentation=frame.segmentation,
                expected_body_id=target_body_id,
                camera_eye=camera_config.eye,
                image_width=config.width,
                image_height=config.height,
                near=camera_config.near,
                far=camera_config.far,
                view_matrix=frame.view_matrix,
                projection_matrix=frame.projection_matrix,
                client_id=scene.client_id,
                ray_test=_ray_test(scene.client_id),
            )
            if not backprojection.gate_passed:
                all_gate_passed = False
                backend_results[backend_name] = {
                    "scientific_gate_passed": False,
                    "failure_stage": "backprojection",
                    "failure_reason": backprojection.failure_reason,
                }
                _write_json(
                    backend_output_dir / "summary.json",
                    backend_results[backend_name],
                )
                continue

            # --- windowed centre recovery --------------------------------
            try:
                sampled_pixel, corrected_depth, corrected_world = (
                    recover_center_via_windowed_depth(
                        float(grasp["center_x"]),
                        float(grasp["center_y"]),
                        frame.depth_m,
                        frame.segmentation,
                        target_body_id,
                        config.width,
                        config.height,
                        frame.view_matrix,
                        frame.projection_matrix,
                        camera_config.near,
                        camera_config.far,
                        window_size=config.center_recovery_window_size,
                    )
                )
            except ValueError as exc:
                all_gate_passed = False
                backend_results[backend_name] = {
                    "scientific_gate_passed": False,
                    "failure_stage": "center_recovery",
                    "failure_reason": f"ValueError:{exc}",
                }
                _write_json(
                    backend_output_dir / "summary.json",
                    backend_results[backend_name],
                )
                continue

            corrected_world = tuple(
                float(value) for value in corrected_world
            )

            # --- recovery evidence ---------------------------------------
            recovery_evidence = CenterRecoveryEvidence(
                protocol=CENTER_RECOVERY_PROTOCOL,
                window_size=config.center_recovery_window_size,
                original_depth_m=backprojection.depth_m,
                corrected_depth_m=float(corrected_depth),
                original_world_surface_point=(
                    backprojection.world_x,
                    backprojection.world_y,
                    backprojection.world_z,
                ),
                corrected_world_surface_point=corrected_world,
                sampled_pixel=sampled_pixel,
                target_body_id=target_body_id,
                target_body_id_source="segmentation_mask",
            )

            # --- pose candidates from corrected world point --------------
            common = {
                "target": config.target_name,
                "backend": backend_name,
                "column": math.floor(
                    float(grasp["center_x"]) + 0.5
                ),
                "row": math.floor(
                    float(grasp["center_y"]) + 0.5
                ),
                "depth_m": float(corrected_depth),
                "angle_degrees": float(grasp["angle_degrees"]),
                "width": config.width,
                "height": config.height,
                "view_matrix": frame.view_matrix,
                "projection_matrix": frame.projection_matrix,
                "near": camera_config.near,
                "far": camera_config.far,
            }
            approach_candidates = generate_top_down_pose_candidates(
                **common,
                surface_standoff_m=0.02,
                pregrasp_offset_m=0.10,
            )
            depth_candidates = generate_top_down_pose_candidates(
                **common,
                surface_standoff_m=0.005,
                pregrasp_offset_m=0.115,
            )

            # --- IK / FK / collision audit -------------------------------
            approach_audits = tuple(
                audit_pose_candidate(
                    candidate,
                    robot_id=scene.bodies.robot,
                    client_id=scene.client_id,
                    model=model,
                    environment_body_ids=environment,
                    allowed_environment_link_pairs=(
                        (-1, scene.bodies.table),
                    ),
                )
                for candidate in approach_candidates
            )
            depth_audits = tuple(
                audit_pose_candidate(
                    candidate,
                    robot_id=scene.bodies.robot,
                    client_id=scene.client_id,
                    model=model,
                    environment_body_ids=non_target_environment,
                    allowed_environment_link_pairs=(
                        (-1, scene.bodies.table),
                    ),
                )
                for candidate in depth_candidates
            )
            combined = []
            for approach_audit, depth_audit in zip(
                approach_audits,
                depth_audits,
            ):
                collision = CollisionAudit(
                    clearance_passed=(
                        approach_audit.collision.clearance_passed
                        and depth_audit.collision.clearance_passed
                    ),
                    checked_state_count=(
                        approach_audit.collision.checked_state_count
                        + depth_audit.collision.checked_state_count
                    ),
                    minimum_clearance_m=min(
                        approach_audit.collision.minimum_clearance_m,
                        depth_audit.collision.minimum_clearance_m,
                    ),
                    environment_collision_count=(
                        approach_audit.collision.environment_collision_count
                        + depth_audit.collision.environment_collision_count
                    ),
                    self_collision_count=(
                        approach_audit.collision.self_collision_count
                        + depth_audit.collision.self_collision_count
                    ),
                    failure_reason=";".join(
                        value
                        for value in (
                            (
                                "precontact:"
                                f"{approach_audit.collision.failure_reason}"
                                if not approach_audit.collision.clearance_passed
                                else ""
                            ),
                            (
                                "grasp_depth:"
                                f"{depth_audit.collision.failure_reason}"
                                if not depth_audit.collision.clearance_passed
                                else ""
                            ),
                        )
                        if value
                    ),
                )
                failures = []
                if not approach_audit.gate_passed:
                    failures.append(
                        f"precontact:{approach_audit.failure_reason}"
                    )
                if not depth_audit.gate_passed:
                    failures.append(
                        f"grasp_depth:{depth_audit.failure_reason}"
                    )
                combined.append(
                    replace(
                        depth_audit,
                        collision=collision,
                        gate_passed=not failures,
                        failure_reason=";".join(failures),
                    )
                )
            audits = select_candidate_pair(tuple(combined))
            rows = [
                _candidate_csv_row(audit, approach_audit, depth_audit)
                for audit, approach_audit, depth_audit in zip(
                    audits,
                    approach_audits,
                    depth_audits,
                )
            ]
            for row, approach in zip(rows, approach_candidates):
                row["approach_position"] = json.dumps(
                    approach.surface_standoff_pose.position
                )
            _write_candidates(backend_output_dir / "candidates.csv", rows)
            selected_count = sum(audit.selected for audit in audits)
            all_ik_available = all(
                audit.pregrasp_ik.solution is not None
                and approach_audit.standoff_ik.solution is not None
                and audit.standoff_ik.solution is not None
                for audit, approach_audit in zip(audits, approach_audits)
            )
            if selected_count != 1 or not all_ik_available:
                all_gate_passed = False
                backend_results[backend_name] = {
                    "scientific_gate_passed": False,
                    "failure_stage": "candidate_audit",
                    "failure_reason": (
                        "no_unique_passing_candidate"
                        if selected_count != 1
                        else "candidate_ik_solution_unavailable"
                    ),
                    "candidate_count": 2,
                    "selected_candidate_count": selected_count,
                }
                _write_json(
                    backend_output_dir / "summary.json",
                    backend_results[backend_name],
                )
                continue

            # --- perception evidence ---------------------------------
            perception = PerceptionEvidence(
                prompt=config.prompt,
                localization_box=localization.box,
                localization_score=localization.score,
                localization_iou=target_evaluation.requested_target_iou,
                grasp_center=(
                    float(grasp["center_x"]),
                    float(grasp["center_y"]),
                ),
                grasp_size=(
                    float(grasp["width"]),
                    float(grasp["height"]),
                ),
                angle_degrees=float(grasp["angle_degrees"]),
                sampled_pixel=(
                    math.floor(float(grasp["center_x"]) + 0.5),
                    math.floor(float(grasp["center_y"]) + 0.5),
                ),
                depth_m=float(corrected_depth),
                world_surface_point=corrected_world,
                target_selection_passed=True,
                backend_geometry_passed=True,
                backprojection_gate_passed=True,
                segmentation_target_match=(
                    backprojection.segmentation_target_match
                ),
                ray_target_match=backprojection.ray_target_match,
                center_recovery=recovery_evidence,
            )

            # --- build planned candidates -----------------------------
            planned = tuple(
                _planned_candidate(audit, approach, approach_audit.standoff_ik)
                for audit, approach, approach_audit in zip(
                    audits,
                    approach_candidates,
                    approach_audits,
                )
            )

            # --- write plan -------------------------------------------
            plan = PerceptionExecutionPlan(
                protocol_version=PROTOCOL_VERSION_V2,
                scene_seed=config.seed,
                target_name=config.target_name,
                backend=backend_name,
                prompt=config.prompt,
                model_id=config.model_id,
                rgb_sha256=rgb_sha256,
                camera=camera_evidence,
                perception=perception,
                control=control,
                candidates=planned,
            )
            write_perception_execution_plan(
                backend_output_dir / "execution_plan.json",
                plan,
            )

            selected = next(row for row in audits if row.selected)
            backend_summary: dict[str, object] = {
                "protocol": PROTOCOL_VERSION_V2,
                "center_recovery_protocol": CENTER_RECOVERY_PROTOCOL,
                "status": "success",
                "failure_stage": "",
                "failure_reason": "",
                "target_name": config.target_name,
                "backend": backend_name,
                "center_recovery_window_size": (
                    config.center_recovery_window_size
                ),
                "original_depth_m": backprojection.depth_m,
                "corrected_depth_m": float(corrected_depth),
                "original_world_surface_point": (
                    backprojection.world_x,
                    backprojection.world_y,
                    backprojection.world_z,
                ),
                "corrected_world_surface_point": corrected_world,
                "candidate_count": 2,
                "candidate_gate_passed_count": sum(
                    audit.gate_passed for audit in audits
                ),
                "selected_candidate_count": selected_count,
                "selected_symmetry_degrees": (
                    selected.candidate.symmetry_degrees
                ),
                "minimum_clearance_m": (
                    selected.collision.minimum_clearance_m
                ),
                "scientific_gate_passed": True,
            }
            _write_json(
                backend_output_dir / "summary.json",
                backend_summary,
            )
            backend_metadata = dict(metadata)
            backend_metadata.update(
                status="success",
                inverse_kinematics_executed=True,
                forward_kinematics_verified=True,
                static_joint_resets_used=True,
                target_selection=asdict(target_evaluation),
                backend_evaluation=asdict(backend_evaluation),
                backprojection=asdict(backprojection),
                center_recovery=asdict(recovery_evidence),
                candidates=rows,
                execution_plan="execution_plan.json",
                summary=backend_summary,
            )
            _write_json(
                backend_output_dir / "metadata.json",
                backend_metadata,
            )
            backend_results[backend_name] = backend_summary

    # -- merged top-level summary -----------------------------------------
    combined_summary: dict[str, object] = {
        "protocol": PROTOCOL_VERSION_V2,
        "center_recovery_protocol": CENTER_RECOVERY_PROTOCOL,
        "center_recovery_window_size": config.center_recovery_window_size,
        "target_name": config.target_name,
        "prompt": config.prompt,
        "localization_iou": target_evaluation.requested_target_iou,
        "backends": {
            name: {
                "scientific_gate_passed": result.get(
                    "scientific_gate_passed", False
                ),
                "status": result.get("status", "failed"),
                **{
                    key: result[key]
                    for key in (
                        "original_world_surface_point",
                        "corrected_world_surface_point",
                        "original_depth_m",
                        "corrected_depth_m",
                        "failure_stage",
                        "failure_reason",
                    )
                    if key in result
                },
            }
            for name, result in backend_results.items()
        },
        "center_recovery_identical": True,
        "scientific_gate_passed": all_gate_passed,
    }
    _write_json(output_dir / "summary.json", combined_summary)
    metadata.update(
        backends=backend_results,
        summary=combined_summary,
    )
    _write_json(output_dir / "metadata.json", metadata)
    return combined_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ray_test(
    client_id: int,
) -> RayTest:
    def ray_test(
        ray_from: tuple[float, float, float],
        ray_to: tuple[float, float, float],
        client_id: int = client_id,
    ) -> tuple[int, tuple[float, float, float] | None]:
        hit = p.rayTest(ray_from, ray_to, physicsClientId=client_id)[0]
        body_id = int(hit[0])
        position = (
            None
            if body_id < 0
            else tuple(float(value) for value in hit[3])
        )
        return body_id, position

    return ray_test


def _failure(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    stage: str,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "protocol": PROTOCOL_VERSION_V2,
        "center_recovery_protocol": CENTER_RECOVERY_PROTOCOL,
        "status": "failed",
        "failure_stage": stage,
        "failure_reason": reason,
        "simulation_setup_steps": 60,
        "simulation_steps_after_capture": 0,
        "scientific_gate_passed": False,
    }
    if details:
        summary.update(details)
    metadata.update(
        status="failed",
        failure_stage=stage,
        failure_reason=reason,
        summary=summary,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "metadata.json", metadata)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6A.2: common centre recovery preflight."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    summary = run_stage6a2_recovery_preflight(
        Stage6A2Config(
            output_dir=args.output_dir,
            device=args.device,
            gui=args.gui,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
