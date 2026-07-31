"""Run the fixed multi-object PyBullet target-selection study.

The scene and rendering use the public PyBullet API and packaged assets from
the Bullet Physics project maintained by Erwin Coumans, Yunfei Bai, and other
contributors: https://github.com/bulletphysics/bullet3
No external grasp-execution code is copied or adapted here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybullet.camera import (
    CameraConfig,
    CameraFrame,
    capture_camera_frame,
)
from src.simulation.pybullet.backend_comparison import (
    BACKEND_ORDER,
    EXPECTED_TARGET_BACKENDS,
    evaluate_backend_grasp,
    summarize_backend_rows,
)
from src.simulation.pybullet.backprojection import (
    BackprojectionAudit,
    RayTest,
    audit_backprojected_grasp,
    summarize_available_backprojection_rows,
)
from src.simulation.pybullet.perception import (
    Localization,
    PilotPrediction,
    load_cnn_backend,
    load_grounding_dino,
    localize_object,
    predict_grasp,
)
from src.simulation.pybullet.scene import (
    PyBulletScene,
    SceneConfig,
    SceneObjectConfig,
)
from src.simulation.pybullet.target_selection import (
    box_iou,
    evaluate_target_selection,
    mask_to_box,
    segmentation_mask_for_body,
    summarize_target_rows,
)
from src.simulation.pybullet.visualization import (
    depth_to_uint8,
    draw_backend_comparison,
    draw_ground_truth_boxes,
    draw_prediction,
    draw_target_evaluation,
    segmentation_to_bgr,
)


@dataclass(frozen=True)
class StudyPrompt:
    """One fixed prompt and its role in the study."""

    result_role: str
    requested_target: str
    prompt: str


MAIN_PROMPTS = (
    StudyPrompt("main", "duck", "yellow rubber duck"),
    StudyPrompt("main", "cube", "red cube"),
    StudyPrompt("main", "sphere", "green sphere"),
)
DIAGNOSTIC_PROMPTS = (
    StudyPrompt("diagnostic", "generic", "small object"),
)
ALL_PROMPTS = MAIN_PROMPTS + DIAGNOSTIC_PROMPTS
DEFAULT_SINGLE_WEIGHTS = Path(
    "data/processed/vlm/cnn_grasp_single_head_deterministic/"
    "cnn_grasp_model_seed_42.pt"
)
DEFAULT_MULTI_HEAD_WEIGHTS = Path(
    "data/processed/vlm/cnn_grasp_multi_head_deterministic/"
    "cnn_grasp_model_seed_42.pt"
)


@dataclass(frozen=True)
class MultiObjectStudyConfig:
    """User-controlled settings for the fixed study protocol."""

    gui: bool = False
    device: str = "cuda"
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    output_dir: Path = Path(
        "data/processed/pybullet/multi_object_study"
    )
    seed: int = 42
    width: int = 640
    height: int = 480
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    iou_threshold: float = 0.25
    single_weights: Path = DEFAULT_SINGLE_WEIGHTS
    multi_head_weights: Path = DEFAULT_MULTI_HEAD_WEIGHTS


@dataclass(frozen=True)
class StudyOutputPaths:
    """Stable output filenames for one study run."""

    rgb: Path
    depth: Path
    depth_visualization: Path
    segmentation: Path
    ground_truth_boxes: Path
    results_csv: Path
    backend_results_csv: Path
    backend_comparison: Path
    backprojection_results_csv: Path
    backprojection_summary: Path
    summary: Path
    metadata: Path
    targets_dir: Path

    def evaluation_image(self, target: str) -> Path:
        return self.targets_dir / target / "evaluation.png"

    def prediction_image(self, target: str) -> Path:
        return self.targets_dir / target / "prediction.png"

    def backend_prediction_image(
        self,
        target: str,
        backend: str,
    ) -> Path:
        if backend not in BACKEND_ORDER:
            raise ValueError(f"unsupported backend: {backend}")
        return self.targets_dir / target / f"{backend}_prediction.png"

    def backend_panel_image(self, target: str) -> Path:
        return self.targets_dir / target / "backend_comparison.png"


@dataclass(frozen=True)
class MultiObjectStudyDependencies:
    """External boundaries replaced by fakes in runner tests."""

    scene_factory: Callable[[SceneConfig], PyBulletScene]
    capture_frame: Callable[[int, CameraConfig, int], CameraFrame]
    load_detector: Callable[[str, str], tuple[object, object]]
    localize: Callable[..., Localization | None]
    load_backend: Callable[[str, Path, str], object]
    predict: Callable[..., PilotPrediction]
    ray_test: RayTest | None = None


def build_study_output_paths(output_dir: Path) -> StudyOutputPaths:
    """Build all fixed root output paths."""

    root = Path(output_dir)
    return StudyOutputPaths(
        rgb=root / "rgb.png",
        depth=root / "depth.npy",
        depth_visualization=root / "depth_visualization.png",
        segmentation=root / "segmentation.png",
        ground_truth_boxes=root / "ground_truth_boxes.png",
        results_csv=root / "results.csv",
        backend_results_csv=root / "backend_results.csv",
        backend_comparison=root / "backend_comparison.json",
        backprojection_results_csv=(
            root / "backprojection_results.csv"
        ),
        backprojection_summary=root / "backprojection_summary.json",
        summary=root / "summary.json",
        metadata=root / "metadata.json",
        targets_dir=root / "targets",
    )


def fixed_scene_config(config: MultiObjectStudyConfig) -> SceneConfig:
    """Return the exact scene used by the fixed protocol."""

    return SceneConfig(
        gui=config.gui,
        seed=config.seed,
        robot_position=(0.0, 0.0, 0.625),
        object_name="duck",
        object_urdf="duck_vhacd.urdf",
        object_position=(0.52, -0.18, 0.67),
        object_yaw_degrees=0.0,
        object_rgba=(1.0, 0.8, 0.0, 1.0),
        additional_objects=(
            SceneObjectConfig(
                name="cube",
                urdf="cube_small.urdf",
                position=(0.48, 0.0, 0.66),
                yaw_degrees=30.0,
                rgba=(0.9, 0.1, 0.1, 1.0),
            ),
            SceneObjectConfig(
                name="sphere",
                urdf="sphere_small.urdf",
                position=(0.52, 0.18, 0.67),
                rgba=(0.1, 0.8, 0.1, 1.0),
            ),
        ),
    )


def default_dependencies() -> MultiObjectStudyDependencies:
    """Bind the study to the real scene, detector, and geometry backend."""

    def ray_test(
        ray_from: tuple[float, float, float],
        ray_to: tuple[float, float, float],
        client_id: int,
    ) -> tuple[int, tuple[float, float, float] | None]:
        hit = p.rayTest(
            ray_from,
            ray_to,
            physicsClientId=client_id,
        )[0]
        body_id = int(hit[0])
        hit_position = (
            None
            if body_id < 0
            else tuple(float(value) for value in hit[3])
        )
        return body_id, hit_position

    return MultiObjectStudyDependencies(
        scene_factory=PyBulletScene,
        capture_frame=capture_camera_frame,
        load_detector=load_grounding_dino,
        localize=localize_object,
        load_backend=load_cnn_backend,
        predict=predict_grasp,
        ray_test=ray_test,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> dict[str, Any]:
    safe = _json_safe(value)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return safe


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save image: {path}")


def _prepare_output_paths(paths: StudyOutputPaths) -> None:
    paths.rgb.parent.mkdir(parents=True, exist_ok=True)
    for prompt in ALL_PROMPTS:
        target_dir = paths.targets_dir / prompt.requested_target
        target_dir.mkdir(parents=True, exist_ok=True)
        paths.evaluation_image(prompt.requested_target).unlink(
            missing_ok=True
        )
        paths.prediction_image(prompt.requested_target).unlink(
            missing_ok=True
        )
        paths.backend_panel_image(prompt.requested_target).unlink(
            missing_ok=True
        )
        for backend in BACKEND_ORDER:
            paths.backend_prediction_image(
                prompt.requested_target,
                backend,
            ).unlink(missing_ok=True)
    for path in (
        paths.rgb,
        paths.depth,
        paths.depth_visualization,
        paths.segmentation,
        paths.ground_truth_boxes,
        paths.results_csv,
        paths.backend_results_csv,
        paths.backend_comparison,
        paths.backprojection_results_csv,
        paths.backprojection_summary,
        paths.summary,
        paths.metadata,
    ):
        path.unlink(missing_ok=True)


def _diagnostic_evaluation(
    predicted_box: tuple[int, int, int, int] | None,
    entity_boxes: Mapping[str, tuple[int, int, int, int]],
    iou_threshold: float,
) -> dict[str, object]:
    if predicted_box is None:
        return {
            "requested_target_iou": 0.0,
            "best_matching_target": None,
            "best_iou": 0.0,
            "correct_target": False,
            "iou_threshold": iou_threshold,
            "failure_reason": "no_detection",
            "entity_ious": {name: 0.0 for name in entity_boxes},
        }
    entity_ious = {
        name: box_iou(predicted_box, box)
        for name, box in entity_boxes.items()
    }
    best_iou = max(entity_ious.values())
    best_names = [
        name
        for name, iou in entity_ious.items()
        if np.isclose(iou, best_iou)
    ]
    return {
        "requested_target_iou": 0.0,
        "best_matching_target": (
            best_names[0] if len(best_names) == 1 else None
        ),
        "best_iou": best_iou,
        "correct_target": False,
        "iou_threshold": iou_threshold,
        "failure_reason": (
            "diagnostic_only"
            if len(best_names) == 1
            else "ambiguous_match"
        ),
        "entity_ious": entity_ious,
    }


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "result_role",
        "requested_target",
        "prompt",
        "detection_box",
        "detection_score",
        "detection_label",
        "requested_target_iou",
        "best_matching_target",
        "best_iou",
        "correct_target",
        "iou_threshold",
        "failure_reason",
        "entity_ious",
        "grasp",
        "backend_failure_reason",
        "grasp_center_inside_target_mask",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: _csv_value(row.get(name))
                    for name in fieldnames
                }
            )


def _write_backend_results_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "target",
        "prompt",
        "backend",
        "weights_path",
        "detection_box",
        "detection_score",
        "center_x",
        "center_y",
        "width",
        "height",
        "angle_degrees",
        "parameters_finite",
        "positive_size",
        "center_inside_target_mask",
        "box_inside_image",
        "backend_failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: _csv_value(row.get(name))
                    for name in fieldnames
                }
            )


def _write_backprojection_results_csv(
    path: Path,
    rows: list[BackprojectionAudit],
) -> None:
    fieldnames = [field.name for field in fields(BackprojectionAudit)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow(
                {
                    name: _csv_value(values[name])
                    for name in fieldnames
                }
            )


def _summarize_available_backend_rows(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    received = tuple(
        (str(row["target"]), str(row["backend"]))
        for row in rows
    )
    if received == EXPECTED_TARGET_BACKENDS:
        summary = summarize_backend_rows(rows)
        summary["backend_comparison_complete"] = True
        return summary

    counts_by_backend = {}
    for backend in BACKEND_ORDER:
        backend_rows = [
            row for row in rows if row["backend"] == backend
        ]
        counts_by_backend[backend] = {
            "finite_output_count": sum(
                bool(row["parameters_finite"])
                for row in backend_rows
            ),
            "center_inside_target_mask_count": sum(
                bool(row["center_inside_target_mask"])
                for row in backend_rows
            ),
            "box_inside_image_count": sum(
                bool(row["box_inside_image"])
                for row in backend_rows
            ),
        }
    return {
        "protocol": "fixed_three_object_three_backend_diagnostic",
        "backend_result_count": len(rows),
        "counts_by_backend": counts_by_backend,
        "performance_ranking_computed": False,
        "physical_grasp_executed": False,
        "backend_comparison_complete": False,
    }


def _outputs_metadata(paths: StudyOutputPaths) -> dict[str, str]:
    return {
        "rgb": str(paths.rgb),
        "depth": str(paths.depth),
        "depth_visualization": str(paths.depth_visualization),
        "segmentation": str(paths.segmentation),
        "ground_truth_boxes": str(paths.ground_truth_boxes),
        "results_csv": str(paths.results_csv),
        "backend_results_csv": str(paths.backend_results_csv),
        "backend_comparison": str(paths.backend_comparison),
        "backprojection_results_csv": str(
            paths.backprojection_results_csv
        ),
        "backprojection_summary": str(paths.backprojection_summary),
        "summary": str(paths.summary),
        "metadata": str(paths.metadata),
        "targets_dir": str(paths.targets_dir),
    }


def run_multi_object_study(
    config: MultiObjectStudyConfig,
    dependencies: MultiObjectStudyDependencies | None = None,
) -> dict[str, Any]:
    """Run one fixed-frame target-selection and 2-D grasp-quality study."""

    if dependencies is None:
        dependencies = default_dependencies()
    paths = build_study_output_paths(config.output_dir)
    _prepare_output_paths(paths)
    scene_config = fixed_scene_config(config)
    scene = dependencies.scene_factory(scene_config)
    camera_config = CameraConfig(width=config.width, height=config.height)
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": "fixed_three_object_prompt_selection_pilot",
        "device": config.device,
        "model_id": config.model_id,
        "box_threshold": config.box_threshold,
        "text_threshold": config.text_threshold,
        "target_selection_iou_threshold": config.iou_threshold,
        "target_selection_iou_note": (
            "Engineering gate for this fixed pilot; not the Cornell "
            "grasp rectangle metric."
        ),
        "backend_weights": {
            "geometry": None,
            "single": str(config.single_weights),
            "multi_head": str(config.multi_head_weights),
        },
        "performance_ranking_computed": False,
        "segmentation_used_as_model_input": False,
        "depth_used_after_2d_prediction": True,
        "segmentation_used_as_coordinate_input": False,
        "ray_test_used_as_coordinate_input": False,
        "ik_executed": False,
        "physical_grasp_executed": False,
        "outputs": _outputs_metadata(paths),
        "pybullet": {
            "package_version": importlib.metadata.version("pybullet"),
            "api_version": p.getAPIVersion(),
        },
    }
    failure_stage = "scene"

    try:
        scene.connect()
        scene.step(60)
        failure_stage = "camera"
        frame = dependencies.capture_frame(
            scene.client_id,
            camera_config,
            scene.renderer,
        )
        _save_image(
            paths.rgb,
            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
        )
        np.save(paths.depth, frame.depth_m)
        _save_image(
            paths.depth_visualization,
            depth_to_uint8(
                frame.depth_m,
                camera_config.near,
                camera_config.far,
            ),
        )
        _save_image(
            paths.segmentation,
            segmentation_to_bgr(frame.segmentation),
        )

        entity_body_ids = {
            **scene.object_body_ids,
            "robot": scene.bodies.robot,
        }
        entity_masks = {
            name: segmentation_mask_for_body(
                frame.segmentation,
                body_id,
            )
            for name, body_id in entity_body_ids.items()
        }
        entity_boxes = {
            name: mask_to_box(mask)
            for name, mask in entity_masks.items()
        }
        _save_image(
            paths.ground_truth_boxes,
            draw_ground_truth_boxes(frame.rgb, entity_boxes),
        )
        metadata["scene"] = {
            "config": asdict(scene.config),
            "body_ids": entity_body_ids,
            "object_poses": scene.object_poses(),
        }
        metadata["camera"] = {
            "config": asdict(camera_config),
            "view_matrix": frame.view_matrix,
            "projection_matrix": frame.projection_matrix,
        }

        failure_stage = "localization_model"
        processor, detector = dependencies.load_detector(
            config.model_id,
            config.device,
        )
        failure_stage = "backend_model:single"
        single_model = dependencies.load_backend(
            "single",
            config.single_weights,
            config.device,
        )
        failure_stage = "backend_model:multi_head"
        multi_head_model = dependencies.load_backend(
            "multi_head",
            config.multi_head_weights,
            config.device,
        )
        backend_models = {
            "geometry": None,
            "single": single_model,
            "multi_head": multi_head_model,
        }
        rows: list[dict[str, object]] = []
        backend_rows: list[dict[str, object]] = []
        image_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        for study_prompt in ALL_PROMPTS:
            failure_stage = f"localization:{study_prompt.requested_target}"
            localization = dependencies.localize(
                rgb_path=paths.rgb,
                prompt=study_prompt.prompt,
                processor=processor,
                model=detector,
                device=config.device,
                box_threshold=config.box_threshold,
                text_threshold=config.text_threshold,
            )
            predicted_box = (
                None if localization is None else localization.box
            )
            if study_prompt.result_role == "main":
                evaluation = asdict(
                    evaluate_target_selection(
                        predicted_box,
                        study_prompt.requested_target,
                        entity_boxes,
                        config.iou_threshold,
                    )
                )
            else:
                evaluation = _diagnostic_evaluation(
                    predicted_box,
                    entity_boxes,
                    config.iou_threshold,
                )
            row: dict[str, object] = {
                "result_role": study_prompt.result_role,
                "requested_target": study_prompt.requested_target,
                "prompt": study_prompt.prompt,
                "detection_box": predicted_box,
                "detection_score": (
                    None if localization is None else localization.score
                ),
                "detection_label": (
                    None if localization is None else localization.label
                ),
                **evaluation,
                "grasp": None,
                "backend_failure_reason": "",
                "grasp_center_inside_target_mask": None,
            }
            _save_image(
                paths.evaluation_image(study_prompt.requested_target),
                draw_target_evaluation(
                    rgb=frame.rgb,
                    requested_target=study_prompt.requested_target,
                    prompt=study_prompt.prompt,
                    detection_box=predicted_box,
                    ground_truth_boxes=entity_boxes,
                    best_matching_target=row["best_matching_target"],
                    score=row["detection_score"],
                ),
            )

            if (
                study_prompt.result_role == "main"
                and bool(row["correct_target"])
                and localization is not None
            ):
                prediction_panels = {}
                for backend in BACKEND_ORDER:
                    failure_stage = (
                        "grasp_prediction:"
                        f"{study_prompt.requested_target}:{backend}"
                    )
                    weights_path = {
                        "geometry": None,
                        "single": config.single_weights,
                        "multi_head": config.multi_head_weights,
                    }[backend]
                    backend_row: dict[str, object] = {
                        "target": study_prompt.requested_target,
                        "prompt": study_prompt.prompt,
                        "backend": backend,
                        "weights_path": weights_path,
                        "detection_box": localization.box,
                        "detection_score": localization.score,
                        "center_x": None,
                        "center_y": None,
                        "width": None,
                        "height": None,
                        "angle_degrees": None,
                        "parameters_finite": False,
                        "positive_size": False,
                        "center_inside_target_mask": False,
                        "box_inside_image": False,
                        "backend_failure_reason": "",
                    }
                    try:
                        prediction = dependencies.predict(
                            image_bgr,
                            localization,
                            backend,
                            config.device,
                            backend_models[backend],
                        )
                        grasp = prediction.grasp
                        audit = evaluate_backend_grasp(
                            grasp,
                            entity_masks[
                                study_prompt.requested_target
                            ],
                            config.width,
                            config.height,
                        )
                        backend_row.update(
                            center_x=grasp["center_x"],
                            center_y=grasp["center_y"],
                            width=grasp["width"],
                            height=grasp["height"],
                            angle_degrees=grasp["angle_degrees"],
                            **asdict(audit),
                        )
                        if prediction.failure_reason:
                            backend_row["backend_failure_reason"] = (
                                prediction.failure_reason
                            )
                        elif audit.failure_reason:
                            backend_row["backend_failure_reason"] = (
                                audit.failure_reason
                            )

                        if audit.parameters_finite and audit.positive_size:
                            prediction_image = draw_prediction(
                                frame.rgb,
                                localization.box,
                                grasp,
                                study_prompt.prompt,
                                localization.score,
                                backend,
                            )
                            prediction_panels[backend] = prediction_image
                            _save_image(
                                paths.backend_prediction_image(
                                    study_prompt.requested_target,
                                    backend,
                                ),
                                prediction_image,
                            )
                            if backend == "geometry":
                                _save_image(
                                    paths.prediction_image(
                                        study_prompt.requested_target
                                    ),
                                    prediction_image,
                                )
                                row["grasp"] = grasp
                                row["backend_failure_reason"] = (
                                    backend_row[
                                        "backend_failure_reason"
                                    ]
                                )
                                row[
                                    "grasp_center_inside_target_mask"
                                ] = audit.center_inside_target_mask
                    except Exception as exc:
                        backend_row["backend_failure_reason"] = str(exc)
                    backend_rows.append(backend_row)

                if tuple(prediction_panels) == BACKEND_ORDER:
                    _save_image(
                        paths.backend_panel_image(
                            study_prompt.requested_target
                        ),
                        draw_backend_comparison(prediction_panels),
                    )
            rows.append(row)

        _write_results_csv(paths.results_csv, rows)
        _write_backend_results_csv(
            paths.backend_results_csv,
            backend_rows,
        )
        backend_summary = _summarize_available_backend_rows(
            backend_rows
        )
        _write_json(paths.backend_comparison, backend_summary)
        failure_stage = "backprojection"
        backprojection_rows = [
            audit_backprojected_grasp(
                backend_row=backend_row,
                depth_m=frame.depth_m,
                segmentation=frame.segmentation,
                expected_body_id=scene.object_body_ids[
                    str(backend_row["target"])
                ],
                camera_eye=camera_config.eye,
                image_width=config.width,
                image_height=config.height,
                near=camera_config.near,
                far=camera_config.far,
                view_matrix=frame.view_matrix,
                projection_matrix=frame.projection_matrix,
                client_id=scene.client_id,
                ray_test=dependencies.ray_test,
            )
            for backend_row in backend_rows
        ]
        _write_backprojection_results_csv(
            paths.backprojection_results_csv,
            backprojection_rows,
        )
        backprojection_summary = (
            summarize_available_backprojection_rows(
                backprojection_rows
            )
        )
        _write_json(
            paths.backprojection_summary,
            backprojection_summary,
        )
        summary = summarize_target_rows(rows)
        summary.update(
            backend_comparison_complete=backend_summary[
                "backend_comparison_complete"
            ],
            backend_comparison=backend_summary,
            backprojection_complete=backprojection_summary[
                "backprojection_complete"
            ],
            backprojection_gate_passed=backprojection_summary[
                "backprojection_gate_passed"
            ],
            backprojection=backprojection_summary,
        )
        summary["status"] = "success"
        _write_json(paths.summary, summary)
        metadata.update(
            status="success",
            results=rows,
            backend_results=backend_rows,
            backend_comparison=backend_summary,
            backprojection_results=[
                asdict(row) for row in backprojection_rows
            ],
            backprojection=backprojection_summary,
            summary=summary,
        )
        _write_json(paths.metadata, metadata)
        return _json_safe(summary)
    except Exception as exc:
        metadata.update(
            status="failed",
            failure_stage=failure_stage,
            failure_reason=str(exc),
            error_type=type(exc).__name__,
        )
        _write_json(paths.metadata, metadata)
        return {
            "status": "failed",
            "failure_stage": failure_stage,
            "failure_reason": str(exc),
            "error_type": type(exc).__name__,
        }
    finally:
        scene.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed multi-object PyBullet target-selection study."
        )
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/pybullet/multi_object_study"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--single-weights",
        type=Path,
        default=DEFAULT_SINGLE_WEIGHTS,
    )
    parser.add_argument(
        "--multi-head-weights",
        type=Path,
        default=DEFAULT_MULTI_HEAD_WEIGHTS,
    )
    return parser


def main() -> int:
    """Run the CLI and return nonzero only for infrastructure failure."""

    args = _build_parser().parse_args()
    summary = run_multi_object_study(
        MultiObjectStudyConfig(
            gui=args.gui,
            device=args.device,
            model_id=args.model_id,
            output_dir=args.output_dir,
            seed=args.seed,
            width=args.width,
            height=args.height,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            single_weights=args.single_weights,
            multi_head_weights=args.multi_head_weights,
        )
    )
    print(
        f"status={summary['status']} "
        f"summary={args.output_dir / 'summary.json'}"
    )
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
