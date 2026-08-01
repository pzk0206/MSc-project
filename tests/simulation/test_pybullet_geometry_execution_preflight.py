import hashlib
import json
from pathlib import Path

import pybullet as p
import pytest

from src.simulation.pybullet.execution_plan import (
    load_geometry_execution_plan,
)
from src.simulation.pybullet.perception import (
    Localization,
    PilotPrediction,
)
from src.simulation.pybullet.run_geometry_execution_preflight import (
    GeometryPreflightConfig,
    GeometryPreflightDependencies,
    run_geometry_execution_preflight,
)
from src.simulation.pybullet.run_multi_object_study import (
    default_dependencies as study_default_dependencies,
)


def _ray_test(ray_from, ray_to, client_id):
    hit = p.rayTest(
        ray_from,
        ray_to,
        physicsClientId=client_id,
    )[0]
    return int(hit[0]), tuple(float(value) for value in hit[3])


def _dependencies(
    *,
    localization: Localization | None = Localization(
        (297, 189, 344, 245),
        0.8169746398925781,
        "red cube",
    ),
    prediction_error: str = "",
) -> GeometryPreflightDependencies:
    base = study_default_dependencies()

    def load_detector(model_id, device):
        return object(), object()

    def localize(**kwargs):
        return localization

    def predict(image_bgr, localization, backend, device, model):
        if prediction_error:
            raise RuntimeError(prediction_error)
        return PilotPrediction(
            localization=localization,
            backend="geometry",
            grasp={
                "center_x": 320.5,
                "center_y": 217.0,
                "width": 76.95,
                "height": 31.35,
                "angle_degrees": 0.0,
            },
        )

    return GeometryPreflightDependencies(
        scene_factory=base.scene_factory,
        capture_frame=base.capture_frame,
        load_detector=load_detector,
        localize=localize,
        predict=predict,
        ray_test=_ray_test,
    )


def test_real_geometry_preflight_writes_one_static_execution_plan(
    tmp_path: Path,
) -> None:
    summary = run_geometry_execution_preflight(
        GeometryPreflightConfig(output_dir=tmp_path, device="cpu"),
        dependencies=_dependencies(),
    )

    plan_path = tmp_path / "execution_plan.json"
    plan = load_geometry_execution_plan(plan_path)
    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )
    assert summary["scientific_gate_passed"] is True
    assert summary["selected_candidate_count"] == 1
    assert summary["candidate_count"] == 2
    assert summary["simulation_setup_steps"] == 60
    assert summary["simulation_steps_after_capture"] == 0
    assert plan.backend == "geometry"
    assert tuple(row.symmetry_degrees for row in plan.candidates) == (
        0.0,
        180.0,
    )
    assert sum(row.selected for row in plan.candidates) == 1
    assert all(row.checked_state_count == 82 for row in plan.candidates)
    assert plan.rgb_sha256 == hashlib.sha256(
        (tmp_path / "rgb.png").read_bytes()
    ).hexdigest()
    for name in (
        "motor_control_executed",
        "trajectory_executed",
        "gripper_closed",
        "contact_evaluated",
        "object_lifted",
        "physical_grasp_executed",
    ):
        assert metadata[name] is False
    for name in (
        "rgb.png",
        "depth.npy",
        "segmentation.png",
        "localization.png",
        "geometry_prediction.png",
        "candidates.csv",
        "summary.json",
        "metadata.json",
        "execution_plan.json",
    ):
        assert (tmp_path / name).is_file()


@pytest.mark.parametrize(
    ("dependencies", "failure_stage"),
    [
        (_dependencies(localization=None), "localization"),
        (_dependencies(prediction_error="geometry failed"), "geometry"),
        (
            _dependencies(
                localization=Localization(
                    (323, 68, 450, 150),
                    0.59,
                    "robot",
                )
            ),
            "target_selection",
        ),
    ],
)
def test_geometry_preflight_failure_removes_stale_plan_and_never_executes(
    tmp_path: Path,
    dependencies: GeometryPreflightDependencies,
    failure_stage: str,
) -> None:
    stale = tmp_path / "execution_plan.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale", encoding="utf-8")

    summary = run_geometry_execution_preflight(
        GeometryPreflightConfig(output_dir=tmp_path, device="cpu"),
        dependencies=dependencies,
    )

    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )
    assert summary["scientific_gate_passed"] is False
    assert summary["failure_stage"] == failure_stage
    assert metadata["failure_stage"] == failure_stage
    assert metadata["motor_control_executed"] is False
    assert metadata["physical_grasp_executed"] is False
    assert not stale.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": 43}, "seed"),
        ({"target_name": "duck"}, "target_name"),
        ({"prompt": "small object"}, "prompt"),
        ({"backend": "multi_head"}, "backend"),
        ({"width": 320}, "640x480"),
    ],
)
def test_geometry_preflight_config_rejects_protocol_changes(
    kwargs,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GeometryPreflightConfig(**kwargs)
