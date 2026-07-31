import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.simulation.pybullet import run_pilot as runner_module
from src.simulation.pybullet.camera import CameraFrame
from src.simulation.pybullet.perception import Localization, PilotPrediction
from src.simulation.pybullet.run_pilot import (
    PilotConfig,
    PilotDependencies,
    build_output_paths,
    run_pilot,
)
from src.simulation.pybullet.scene import SceneBodies


class _FakeScene:
    instances: list["_FakeScene"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.client_id = 7
        self.renderer = 123
        self.bodies = SceneBodies(plane=0, table=1, robot=2, target_object=3)
        self.closed = False
        self.steps = 0
        self.instances.append(self)

    def connect(self):
        return self

    def step(self, count: int = 1) -> None:
        self.steps += count

    def close(self) -> None:
        self.closed = True


def _camera_frame() -> CameraFrame:
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[2:10, 3:14] = [220, 150, 40]
    return CameraFrame(
        rgb=rgb,
        depth_m=np.full((12, 16), 0.8, dtype=np.float32),
        segmentation=np.full((12, 16), 3, dtype=np.int32),
        view_matrix=tuple(float(value) for value in range(16)),
        projection_matrix=tuple(float(value) for value in range(16)),
    )


def _dependencies(*, detected: bool) -> PilotDependencies:
    localization = Localization(
        box=(2, 2, 13, 9),
        score=0.9,
        label="object",
    )

    def localize(**kwargs):
        return localization if detected else None

    def predict(image_bgr, selected, backend, device, model):
        return PilotPrediction(
            localization=selected,
            backend=backend,
            grasp={
                "center_x": 7.0,
                "center_y": 5.0,
                "width": 6.0,
                "height": 3.0,
                "angle_degrees": 20.0,
            },
        )

    def reject_backend_load(backend, weights_path, device):
        raise AssertionError("geometry run must not load CNN weights")

    return PilotDependencies(
        scene_factory=_FakeScene,
        capture_frame=lambda client_id, config, renderer: _camera_frame(),
        load_detector=lambda model_id, device: ("processor", "model"),
        localize=localize,
        load_backend=reject_backend_load,
        predict=predict,
    )


def test_output_paths_use_fixed_auditable_names(tmp_path: Path) -> None:
    paths = build_output_paths(tmp_path)

    assert paths.rgb == tmp_path / "rgb.png"
    assert paths.depth == tmp_path / "depth.npy"
    assert paths.depth_visualization == tmp_path / "depth_visualization.png"
    assert paths.segmentation == tmp_path / "segmentation.png"
    assert paths.localization == tmp_path / "localization.png"
    assert paths.prediction == tmp_path / "prediction.png"
    assert paths.metadata == tmp_path / "metadata.json"


def test_runner_saves_all_success_artifacts_and_closes_scene(
    tmp_path: Path,
) -> None:
    _FakeScene.instances.clear()

    metadata = run_pilot(
        PilotConfig(
            output_dir=tmp_path,
            width=16,
            height=12,
            device="cpu",
        ),
        dependencies=_dependencies(detected=True),
    )

    paths = build_output_paths(tmp_path)
    assert all(
        path.is_file()
        for path in (
            paths.rgb,
            paths.depth,
            paths.depth_visualization,
            paths.segmentation,
            paths.localization,
            paths.prediction,
            paths.metadata,
        )
    )
    saved = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata == saved
    assert saved["status"] == "success"
    assert saved["physical_grasp_executed"] is False
    assert saved["backend"] == "geometry"
    assert saved["localization"]["box"] == [2, 2, 13, 9]
    assert saved["outputs"]["depth"] == str(tmp_path / "depth.npy")
    assert saved["scene"]["resources"] == {
        "plane": "plane.urdf",
        "table": "table/table.urdf",
        "robot": "franka_panda/panda.urdf",
        "target_object": "duck_vhacd.urdf",
    }
    assert len(saved["camera"]["view_matrix"]) == 16
    assert len(saved["camera"]["projection_matrix"]) == 16
    assert _FakeScene.instances[-1].steps == 60
    assert _FakeScene.instances[-1].closed


def test_no_detection_keeps_diagnostics_and_failure_metadata(
    tmp_path: Path,
) -> None:
    _FakeScene.instances.clear()
    stale_prediction = tmp_path / "prediction.png"
    stale_prediction.write_bytes(b"stale result")

    metadata = run_pilot(
        PilotConfig(
            output_dir=tmp_path,
            width=16,
            height=12,
            device="cpu",
        ),
        dependencies=_dependencies(detected=False),
    )

    paths = build_output_paths(tmp_path)
    assert metadata["status"] == "failed"
    assert metadata["failure_stage"] == "localization"
    assert metadata["failure_reason"] == "no_detection"
    assert metadata["physical_grasp_executed"] is False
    assert paths.rgb.is_file()
    assert paths.depth.is_file()
    assert paths.depth_visualization.is_file()
    assert paths.segmentation.is_file()
    assert paths.localization.is_file()
    assert paths.metadata.is_file()
    assert not paths.prediction.exists()
    assert _FakeScene.instances[-1].closed


def test_runner_uses_default_dependencies_when_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "default_dependencies",
        lambda: _dependencies(detected=True),
    )

    metadata = run_pilot(
        PilotConfig(
            output_dir=tmp_path,
            width=16,
            height=12,
            device="cpu",
        )
    )

    assert metadata["status"] == "success"


def test_detector_exception_writes_failure_metadata(
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(detected=True)
    dependencies = PilotDependencies(
        scene_factory=dependencies.scene_factory,
        capture_frame=dependencies.capture_frame,
        load_detector=lambda model_id, device: (_ for _ in ()).throw(
            RuntimeError("checkpoint unavailable")
        ),
        localize=dependencies.localize,
        load_backend=dependencies.load_backend,
        predict=dependencies.predict,
    )

    metadata = run_pilot(
        PilotConfig(
            output_dir=tmp_path,
            width=16,
            height=12,
            device="cpu",
        ),
        dependencies=dependencies,
    )

    saved = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == saved
    assert saved["status"] == "failed"
    assert saved["failure_stage"] == "localization_model"
    assert saved["error_type"] == "RuntimeError"
    assert saved["failure_reason"] == "checkpoint unavailable"
    assert saved["physical_grasp_executed"] is False


def test_cli_help_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "src/simulation/pybullet/run_pilot.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--backend" in completed.stdout
    assert "--model-weights" in completed.stdout
    assert "--object-urdf" in completed.stdout
