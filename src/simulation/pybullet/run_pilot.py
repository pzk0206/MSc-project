"""Run the first-stage PyBullet perception pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
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
    load_cnn_backend,
    load_grounding_dino,
    localize_object,
    predict_grasp,
)
from src.simulation.pybullet.scene import (
    PyBulletScene,
    SceneConfig,
)
from src.simulation.pybullet.visualization import (
    depth_to_uint8,
    draw_prediction,
    segmentation_to_bgr,
)
from src.vlm.run_grounding_dino_localization import (
    draw_localization_result,
)


@dataclass(frozen=True)
class PilotConfig:
    """User-controlled first-stage pilot settings."""

    gui: bool = False
    backend: str = "geometry"
    device: str = "cuda"
    prompt: str = "small object"
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    model_weights: Path | None = None
    object_urdf: str = "duck_vhacd.urdf"
    output_dir: Path = Path("data/processed/pybullet/pilot")
    seed: int = 42
    width: int = 640
    height: int = 480
    box_threshold: float = 0.25
    text_threshold: float = 0.25


@dataclass(frozen=True)
class OutputPaths:
    """Fixed artifact names for one pilot output directory."""

    rgb: Path
    depth: Path
    depth_visualization: Path
    segmentation: Path
    localization: Path
    prediction: Path
    metadata: Path


@dataclass(frozen=True)
class PilotDependencies:
    """External boundaries replaced by fast fakes in runner tests."""

    scene_factory: Callable[[SceneConfig], PyBulletScene]
    capture_frame: Callable[[int, CameraConfig, int], CameraFrame]
    load_detector: Callable[[str, str], tuple[object, object]]
    localize: Callable[..., Localization | None]
    load_backend: Callable[[str, Path, str], object]
    predict: Callable[..., PilotPrediction]


def build_output_paths(output_dir: Path) -> OutputPaths:
    """Return the stable, auditable filenames for one run."""

    output_dir = Path(output_dir)
    return OutputPaths(
        rgb=output_dir / "rgb.png",
        depth=output_dir / "depth.npy",
        depth_visualization=output_dir / "depth_visualization.png",
        segmentation=output_dir / "segmentation.png",
        localization=output_dir / "localization.png",
        prediction=output_dir / "prediction.png",
        metadata=output_dir / "metadata.json",
    )


def default_dependencies() -> PilotDependencies:
    """Bind the runner to the project's real scene and model adapters."""

    return PilotDependencies(
        scene_factory=PyBulletScene,
        capture_frame=capture_camera_frame,
        load_detector=load_grounding_dino,
        localize=localize_object,
        load_backend=load_cnn_backend,
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


def _write_json(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    safe_metadata = _json_safe(metadata)
    path.write_text(
        json.dumps(safe_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return safe_metadata


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save image: {path}")


def _outputs_metadata(paths: OutputPaths) -> dict[str, str]:
    return {
        name: str(path)
        for name, path in asdict(paths).items()
    }


def _base_metadata(
    config: PilotConfig,
    paths: OutputPaths,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "seed": config.seed,
        "backend": config.backend,
        "device": config.device,
        "prompt": config.prompt,
        "model_id": config.model_id,
        "box_threshold": config.box_threshold,
        "text_threshold": config.text_threshold,
        "model_weights": (
            str(config.model_weights)
            if config.model_weights is not None
            else None
        ),
        "physical_grasp_executed": False,
        "outputs": _outputs_metadata(paths),
        "pybullet": {
            "package_version": importlib.metadata.version("pybullet"),
            "api_version": p.getAPIVersion(),
        },
    }


def run_pilot(
    config: PilotConfig,
    dependencies: PilotDependencies | None = None,
) -> dict[str, Any]:
    """Run one perception-only pilot and return its saved metadata."""

    if dependencies is None:
        dependencies = default_dependencies()

    paths = build_output_paths(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for generated_path in asdict(paths).values():
        generated_path.unlink(missing_ok=True)
    metadata = _base_metadata(config, paths)
    scene = dependencies.scene_factory(
        SceneConfig(
            gui=config.gui,
            seed=config.seed,
            object_urdf=config.object_urdf,
        )
    )
    camera_config = CameraConfig(
        width=config.width,
        height=config.height,
    )
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

        metadata["connection_mode"] = (
            "GUI" if config.gui else "DIRECT"
        )
        metadata["renderer"] = (
            "ER_BULLET_HARDWARE_OPENGL"
            if config.gui
            else "ER_TINY_RENDERER"
        )
        metadata["scene"] = {
            "resources": {
                "plane": "plane.urdf",
                "table": "table/table.urdf",
                "robot": "franka_panda/panda.urdf",
                "target_object": config.object_urdf,
            },
            "config": asdict(scene.config),
            "bodies": asdict(scene.bodies),
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
        failure_stage = "localization"
        localization = dependencies.localize(
            rgb_path=paths.rgb,
            prompt=config.prompt,
            processor=processor,
            model=detector,
            device=config.device,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
        detection_dict = (
            None
            if localization is None
            else {
                "box": localization.box,
                "score": localization.score,
                "label": localization.label,
            }
        )
        draw_localization_result(
            paths.rgb,
            paths.localization,
            config.prompt,
            detection_dict,
        )
        if localization is None:
            metadata.update(
                status="failed",
                failure_stage="localization",
                failure_reason="no_detection",
            )
            return _write_json(paths.metadata, metadata)

        backend_model = None
        if config.backend in {"single", "multi_head"}:
            failure_stage = "backend_model"
            if config.model_weights is None:
                raise ValueError(
                    f"{config.backend} backend requires --model-weights"
                )
            backend_model = dependencies.load_backend(
                config.backend,
                config.model_weights,
                config.device,
            )
        failure_stage = "grasp_prediction"
        prediction = dependencies.predict(
            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
            localization,
            config.backend,
            config.device,
            backend_model,
        )
        prediction_image = draw_prediction(
            frame.rgb,
            localization.box,
            prediction.grasp,
            config.prompt,
            localization.score,
            config.backend,
        )
        _save_image(paths.prediction, prediction_image)

        metadata.update(
            status="success",
            localization=asdict(localization),
            grasp=prediction.grasp,
            backend_failure_reason=prediction.failure_reason,
        )
        return _write_json(paths.metadata, metadata)
    except Exception as exc:
        metadata.update(
            status="failed",
            failure_stage=failure_stage,
            failure_reason=str(exc),
            error_type=type(exc).__name__,
        )
        return _write_json(paths.metadata, metadata)
    finally:
        scene.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a perception-only PyBullet camera, localization, and "
            "2D grasp pilot."
        )
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["geometry", "single", "multi_head"],
        default="geometry",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--prompt", default="small object")
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--object-urdf", default="duck_vhacd.urdf")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/pybullet/pilot"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    return parser


def main() -> int:
    """Parse CLI arguments, run the pilot, and expose status as exit code."""

    args = _build_parser().parse_args()
    metadata = run_pilot(
        PilotConfig(
            gui=args.gui,
            backend=args.backend,
            device=args.device,
            prompt=args.prompt,
            model_id=args.model_id,
            model_weights=args.model_weights,
            object_urdf=args.object_urdf,
            output_dir=args.output_dir,
            seed=args.seed,
            width=args.width,
            height=args.height,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    )
    print(
        f"status={metadata['status']} "
        f"metadata={args.output_dir / 'metadata.json'}"
    )
    return 0 if metadata["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
