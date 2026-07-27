"""Run the fixed multi-object PyBullet target-selection study.

The scene and rendering use the public PyBullet API and packaged assets from
the Bullet Physics project maintained by Erwin Coumans, Yunfei Bai, and other
contributors: https://github.com/bulletphysics/bullet3
No external grasp-execution code is copied or adapted here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
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
from src.simulation.pybullet.perception import (
    Localization,
    PilotPrediction,
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
    grasp_center_inside_mask,
    mask_to_box,
    segmentation_mask_for_body,
    summarize_target_rows,
)
from src.simulation.pybullet.visualization import (
    depth_to_uint8,
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


@dataclass(frozen=True)
class StudyOutputPaths:
    """Stable output filenames for one study run."""

    rgb: Path
    depth: Path
    depth_visualization: Path
    segmentation: Path
    ground_truth_boxes: Path
    results_csv: Path
    summary: Path
    metadata: Path
    targets_dir: Path

    def evaluation_image(self, target: str) -> Path:
        return self.targets_dir / target / "evaluation.png"

    def prediction_image(self, target: str) -> Path:
        return self.targets_dir / target / "prediction.png"


@dataclass(frozen=True)
class MultiObjectStudyDependencies:
    """External boundaries replaced by fakes in runner tests."""

    scene_factory: Callable[[SceneConfig], PyBulletScene]
    capture_frame: Callable[[int, CameraConfig, int], CameraFrame]
    load_detector: Callable[[str, str], tuple[object, object]]
    localize: Callable[..., Localization | None]
    predict: Callable[..., PilotPrediction]


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

    return MultiObjectStudyDependencies(
        scene_factory=PyBulletScene,
        capture_frame=capture_camera_frame,
        load_detector=load_grounding_dino,
        localize=localize_object,
        predict=predict_grasp,
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
    for path in (
        paths.rgb,
        paths.depth,
        paths.depth_visualization,
        paths.segmentation,
        paths.ground_truth_boxes,
        paths.results_csv,
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


def _outputs_metadata(paths: StudyOutputPaths) -> dict[str, str]:
    return {
        "rgb": str(paths.rgb),
        "depth": str(paths.depth),
        "depth_visualization": str(paths.depth_visualization),
        "segmentation": str(paths.segmentation),
        "ground_truth_boxes": str(paths.ground_truth_boxes),
        "results_csv": str(paths.results_csv),
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
        "segmentation_used_as_model_input": False,
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
        rows: list[dict[str, object]] = []
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
                failure_stage = (
                    f"grasp_prediction:{study_prompt.requested_target}"
                )
                prediction = dependencies.predict(
                    image_bgr,
                    localization,
                    "geometry",
                    config.device,
                    None,
                )
                row["grasp"] = prediction.grasp
                row["backend_failure_reason"] = prediction.failure_reason
                row["grasp_center_inside_target_mask"] = (
                    grasp_center_inside_mask(
                        prediction.grasp,
                        entity_masks[study_prompt.requested_target],
                    )
                )
                _save_image(
                    paths.prediction_image(
                        study_prompt.requested_target
                    ),
                    draw_prediction(
                        frame.rgb,
                        localization.box,
                        prediction.grasp,
                        study_prompt.prompt,
                        localization.score,
                        "geometry",
                    ),
                )
            rows.append(row)

        _write_results_csv(paths.results_csv, rows)
        summary = summarize_target_rows(rows)
        summary["status"] = "success"
        _write_json(paths.summary, summary)
        metadata.update(status="success", results=rows, summary=summary)
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
        )
    )
    print(
        f"status={summary['status']} "
        f"summary={args.output_dir / 'summary.json'}"
    )
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
