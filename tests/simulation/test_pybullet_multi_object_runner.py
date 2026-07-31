import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pybullet as p

from src.simulation.pybullet.camera import CameraFrame
from src.simulation.pybullet.perception import Localization, PilotPrediction
from src.simulation.pybullet.run_multi_object_study import (
    MultiObjectStudyConfig,
    MultiObjectStudyDependencies,
    build_study_output_paths,
    run_multi_object_study,
)
from src.simulation.pybullet.scene import SceneBodies


class _FakeMultiObjectScene:
    instances: list["_FakeMultiObjectScene"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.client_id = 9
        self.renderer = 321
        self.bodies = SceneBodies(
            plane=0,
            table=1,
            robot=2,
            target_object=3,
            additional_objects=(("cube", 4), ("sphere", 5)),
        )
        self.object_body_ids = {"duck": 3, "cube": 4, "sphere": 5}
        self.closed = False
        self.steps = 0
        self.instances.append(self)

    def connect(self):
        return self

    def step(self, count: int = 1) -> None:
        self.steps += count

    def object_poses(self):
        return {
            name: {
                "position": (0.5, 0.0, 0.67),
                "orientation": (0.0, 0.0, 0.0, 1.0),
            }
            for name in self.object_body_ids
        }

    def close(self) -> None:
        self.closed = True


def _multi_object_frame(valid_matrices: bool = False) -> CameraFrame:
    segmentation = np.full((60, 80), -1, dtype=np.int32)
    segmentation[2:15, 2:15] = 2
    segmentation[20:30, 5:15] = 3
    segmentation[20:30, 25:35] = 4
    segmentation[20:30, 45:55] = 5
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    rgb[segmentation == 2] = (180, 180, 180)
    rgb[segmentation == 3] = (255, 204, 0)
    rgb[segmentation == 4] = (230, 25, 25)
    rgb[segmentation == 5] = (25, 204, 25)
    if valid_matrices:
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=(1.0, 0.0, 1.15),
            cameraTargetPosition=(0.5, 0.0, 0.62),
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=80 / 60,
            nearVal=0.05,
            farVal=3.0,
        )
    else:
        view_matrix = tuple(float(value) for value in range(16))
        projection_matrix = tuple(float(value) for value in range(16))
    return CameraFrame(
        rgb=rgb,
        depth_m=np.full((60, 80), 0.8, dtype=np.float32),
        segmentation=segmentation,
        view_matrix=tuple(view_matrix),
        projection_matrix=tuple(projection_matrix),
    )


def test_multi_object_runner_uses_one_frame_and_model_and_gates_grasp(
    tmp_path: Path,
) -> None:
    _FakeMultiObjectScene.instances.clear()
    detector_loads: list[tuple[str, str]] = []
    captures: list[int] = []
    backend_loads: list[tuple[str, Path, str]] = []
    prediction_calls: list[tuple[str, str, object, int]] = []
    localizations = {
        "yellow rubber duck": Localization(
            box=(5, 20, 14, 29),
            score=0.9,
            label="duck",
        ),
        "red cube": Localization(
            box=(5, 20, 14, 29),
            score=0.8,
            label="duck",
        ),
        "green sphere": Localization(
            box=(45, 20, 54, 29),
            score=0.85,
            label="sphere",
        ),
        "small object": Localization(
            box=(2, 2, 14, 14),
            score=0.7,
            label="robot",
        ),
    }

    def capture_frame(client_id, config, renderer):
        captures.append(client_id)
        return _multi_object_frame()

    def load_detector(model_id, device):
        detector_loads.append((model_id, device))
        return "processor", "detector"

    def load_backend(backend, weights_path, device):
        backend_loads.append((backend, weights_path, device))
        return f"{backend}-model"

    def localize(
        *,
        rgb_path,
        prompt,
        processor,
        model,
        device,
        box_threshold,
        text_threshold,
    ):
        return localizations[prompt]

    def predict(image_bgr, localization, backend, device, model):
        prediction_calls.append(
            (
                localization.label,
                backend,
                model,
                id(localization),
            )
        )
        return PilotPrediction(
            localization=localization,
            backend=backend,
            grasp={
                "center_x": (
                    localization.box[0] + localization.box[2]
                ) / 2,
                "center_y": (
                    localization.box[1] + localization.box[3]
                ) / 2,
                "width": 6.0,
                "height": 3.0,
                "angle_degrees": 0.0,
            },
        )

    config = MultiObjectStudyConfig(
        output_dir=tmp_path,
        width=80,
        height=60,
        device="cpu",
    )
    summary = run_multi_object_study(
        config,
        dependencies=MultiObjectStudyDependencies(
            scene_factory=_FakeMultiObjectScene,
            capture_frame=capture_frame,
            load_detector=load_detector,
            localize=localize,
            load_backend=load_backend,
            predict=predict,
        ),
    )

    paths = build_study_output_paths(tmp_path)
    assert summary["main_target_count"] == 3
    assert summary["correct_target_count"] == 2
    assert summary["generic_diagnostic"]["best_matching_target"] == "robot"
    assert summary["backend_comparison_complete"] is False
    assert len(detector_loads) == 1
    assert len(captures) == 1
    assert backend_loads == [
        ("single", config.single_weights, "cpu"),
        ("multi_head", config.multi_head_weights, "cpu"),
    ]
    assert [
        (target, backend, model)
        for target, backend, model, _ in prediction_calls
    ] == [
        ("duck", "geometry", None),
        ("duck", "single", "single-model"),
        ("duck", "multi_head", "multi_head-model"),
        ("sphere", "geometry", None),
        ("sphere", "single", "single-model"),
        ("sphere", "multi_head", "multi_head-model"),
    ]
    duck_localization_ids = {
        localization_id
        for target, _, _, localization_id in prediction_calls
        if target == "duck"
    }
    sphere_localization_ids = {
        localization_id
        for target, _, _, localization_id in prediction_calls
        if target == "sphere"
    }
    assert len(duck_localization_ids) == 1
    assert len(sphere_localization_ids) == 1
    assert _FakeMultiObjectScene.instances[-1].steps == 60
    assert _FakeMultiObjectScene.instances[-1].closed

    with paths.results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prompt"] for row in rows] == [
        "yellow rubber duck",
        "red cube",
        "green sphere",
        "small object",
    ]
    assert all(
        path.is_file()
        for path in (
            paths.rgb,
            paths.depth,
            paths.depth_visualization,
            paths.segmentation,
            paths.ground_truth_boxes,
            paths.results_csv,
            paths.backend_results_csv,
            paths.backend_comparison,
            paths.summary,
            paths.metadata,
        )
    )
    with paths.backend_results_csv.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        backend_rows = list(csv.DictReader(handle))
    assert [
        (row["target"], row["backend"])
        for row in backend_rows
    ] == [
        ("duck", "geometry"),
        ("duck", "single"),
        ("duck", "multi_head"),
        ("sphere", "geometry"),
        ("sphere", "single"),
        ("sphere", "multi_head"),
    ]
    assert paths.evaluation_image("cube").is_file()
    assert paths.prediction_image("duck").is_file()
    assert not paths.prediction_image("cube").exists()
    assert paths.prediction_image("sphere").is_file()
    for target in ("duck", "sphere"):
        for backend in ("geometry", "single", "multi_head"):
            assert paths.backend_prediction_image(
                target,
                backend,
            ).is_file()
        assert paths.backend_panel_image(target).is_file()
    assert not paths.backend_panel_image("cube").exists()

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["segmentation_used_as_model_input"] is False
    assert metadata["physical_grasp_executed"] is False
    assert metadata["backend_weights"] == {
        "geometry": None,
        "single": str(config.single_weights),
        "multi_head": str(config.multi_head_weights),
    }


def test_backend_output_paths_and_seed_42_defaults(tmp_path: Path) -> None:
    config = MultiObjectStudyConfig()
    paths = build_study_output_paths(tmp_path)

    assert config.single_weights == Path(
        "data/processed/vlm/cnn_grasp_single_head_deterministic/"
        "cnn_grasp_model_seed_42.pt"
    )
    assert config.multi_head_weights == Path(
        "data/processed/vlm/cnn_grasp_multi_head_deterministic/"
        "cnn_grasp_model_seed_42.pt"
    )
    assert paths.backend_results_csv == tmp_path / "backend_results.csv"
    assert paths.backend_comparison == tmp_path / "backend_comparison.json"
    assert paths.backend_prediction_image("duck", "single") == (
        tmp_path / "targets/duck/single_prediction.png"
    )
    assert paths.backend_panel_image("duck") == (
        tmp_path / "targets/duck/backend_comparison.png"
    )
    assert paths.prediction_image("duck") == (
        tmp_path / "targets/duck/prediction.png"
    )


def test_complete_nine_point_backprojection_gate_is_persisted(
    tmp_path: Path,
) -> None:
    localizations = {
        "yellow rubber duck": Localization(
            box=(5, 20, 14, 29), score=0.9, label="duck"
        ),
        "red cube": Localization(
            box=(25, 20, 34, 29), score=0.9, label="cube"
        ),
        "green sphere": Localization(
            box=(45, 20, 54, 29), score=0.9, label="sphere"
        ),
        "small object": Localization(
            box=(2, 2, 14, 14), score=0.7, label="robot"
        ),
    }

    def predict(image_bgr, localization, backend, device, model):
        return PilotPrediction(
            localization=localization,
            backend=backend,
            grasp={
                "center_x": (
                    localization.box[0] + localization.box[2]
                ) / 2,
                "center_y": (
                    localization.box[1] + localization.box[3]
                ) / 2,
                "width": 6.0,
                "height": 3.0,
                "angle_degrees": 0.0,
            },
        )

    def ray_test(ray_from, ray_to, client_id):
        assert client_id == 9
        if ray_to[1] < -0.25:
            body_id = 3
        elif ray_to[1] < 0.0:
            body_id = 4
        else:
            body_id = 5
        return body_id, ray_to

    summary = run_multi_object_study(
        MultiObjectStudyConfig(
            output_dir=tmp_path,
            width=80,
            height=60,
            device="cpu",
        ),
        dependencies=MultiObjectStudyDependencies(
            scene_factory=_FakeMultiObjectScene,
            capture_frame=lambda _client, _config, _renderer: (
                _multi_object_frame(valid_matrices=True)
            ),
            load_detector=lambda _model_id, _device: (object(), object()),
            localize=lambda **kwargs: localizations[kwargs["prompt"]],
            load_backend=lambda backend, _weights, _device: backend,
            predict=predict,
            ray_test=ray_test,
        ),
    )

    paths = build_study_output_paths(tmp_path)
    assert summary["backprojection_complete"] is True
    assert summary["backprojection_gate_passed"] is True
    assert summary["backprojection"]["backprojection_result_count"] == 9
    assert paths.backprojection_results_csv.is_file()
    assert paths.backprojection_summary.is_file()
    with paths.backprojection_results_csv.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["target"], row["backend"]) for row in rows] == [
        (target, backend)
        for target in ("duck", "cube", "sphere")
        for backend in ("geometry", "single", "multi_head")
    ]
    assert all(row["gate_passed"] == "True" for row in rows)

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["depth_used_after_2d_prediction"] is True
    assert metadata["segmentation_used_as_coordinate_input"] is False
    assert metadata["ray_test_used_as_coordinate_input"] is False
    assert metadata["ik_executed"] is False
    assert metadata["physical_grasp_executed"] is False


def test_multi_object_cli_help_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "src/simulation/pybullet/run_multi_object_study.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--device" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--single-weights" in completed.stdout
    assert "--multi-head-weights" in completed.stdout
