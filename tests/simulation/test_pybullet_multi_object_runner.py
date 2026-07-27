import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

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


def _multi_object_frame() -> CameraFrame:
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
    return CameraFrame(
        rgb=rgb,
        depth_m=np.full((60, 80), 0.8, dtype=np.float32),
        segmentation=segmentation,
        view_matrix=tuple(float(value) for value in range(16)),
        projection_matrix=tuple(float(value) for value in range(16)),
    )


def test_multi_object_runner_uses_one_frame_and_model_and_gates_grasp(
    tmp_path: Path,
) -> None:
    _FakeMultiObjectScene.instances.clear()
    detector_loads: list[tuple[str, str]] = []
    captures: list[int] = []
    predicted_labels: list[str] = []
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
        predicted_labels.append(localization.label)
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

    summary = run_multi_object_study(
        MultiObjectStudyConfig(
            output_dir=tmp_path,
            width=80,
            height=60,
            device="cpu",
        ),
        dependencies=MultiObjectStudyDependencies(
            scene_factory=_FakeMultiObjectScene,
            capture_frame=capture_frame,
            load_detector=load_detector,
            localize=localize,
            predict=predict,
        ),
    )

    paths = build_study_output_paths(tmp_path)
    assert summary["main_target_count"] == 3
    assert summary["correct_target_count"] == 2
    assert summary["generic_diagnostic"]["best_matching_target"] == "robot"
    assert len(detector_loads) == 1
    assert len(captures) == 1
    assert predicted_labels == ["duck", "sphere"]
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
            paths.summary,
            paths.metadata,
        )
    )
    assert paths.evaluation_image("cube").is_file()
    assert paths.prediction_image("duck").is_file()
    assert not paths.prediction_image("cube").exists()
    assert paths.prediction_image("sphere").is_file()

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["segmentation_used_as_model_input"] is False
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
