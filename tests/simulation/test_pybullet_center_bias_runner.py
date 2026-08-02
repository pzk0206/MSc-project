import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.simulation.pybullet.execution_plan import (
    PROTOCOL_VERSION as STAGE_6A_PROTOCOL_VERSION,
    CameraEvidence,
    FrozenControlProtocol,
    GeometryExecutionPlan,
    PerceptionEvidence,
    PlannedPoseCandidate,
    write_geometry_execution_plan,
)
from src.simulation.pybullet.pose_generation import ToolPose
from src.simulation.pybullet.run_center_bias_diagnostic import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    CenterBiasDiagnosticConfig,
    run_center_bias_diagnostic,
)
import src.simulation.pybullet.run_center_bias_diagnostic as runner_module


WORLD_POINT = (
    0.5064564100151149,
    0.002224916375108214,
    0.6754779706501471,
)
CUBE_TRUTH_CENTER = (
    0.4800002872798181,
    -5.134833814891427e-7,
    0.649968798272667,
)
EXECUTION_FLAGS = (
    "motor_control_executed",
    "trajectory_executed",
    "gripper_closed",
    "contact_evaluated",
    "object_lifted",
    "physical_grasp_executed",
)


def _pose(z: float) -> ToolPose:
    return ToolPose(
        (WORLD_POINT[0], WORLD_POINT[1], z),
        (1.0, 0.0, 0.0, 0.0),
    )


def _candidate(symmetry: float, selected: bool) -> PlannedPoseCandidate:
    return PlannedPoseCandidate(
        symmetry_degrees=symmetry,
        finger_axis_world=(1.0, 0.0, 0.0),
        pregrasp_pose=_pose(0.795),
        approach_pose=_pose(0.695),
        grasp_depth_pose=_pose(0.680),
        pregrasp_ik=(0.0,) * 7,
        approach_ik=(0.1,) * 7,
        grasp_depth_ik=(0.2,) * 7,
        ik_fk_passed=True,
        clearance_passed=True,
        checked_state_count=82,
        minimum_clearance_m=0.002,
        environment_collision_count=0,
        self_collision_count=0,
        total_normalized_joint_cost=1.0 + symmetry,
        gate_passed=True,
        selected=selected,
        failure_reason="",
    )


def _plan(rgb_sha256: str) -> GeometryExecutionPlan:
    return GeometryExecutionPlan(
        protocol_version=STAGE_6A_PROTOCOL_VERSION,
        scene_seed=42,
        target_name="cube",
        backend="geometry",
        prompt="red cube",
        model_id="IDEA-Research/grounding-dino-tiny",
        rgb_sha256=rgb_sha256,
        camera=CameraEvidence(
            width=640,
            height=480,
            eye=(1.0, 0.0, 1.15),
            target=(0.5, 0.0, 0.62),
            up=(0.0, 0.0, 1.0),
            fov_degrees=55.0,
            near=0.05,
            far=3.0,
            view_matrix=(0.0,) * 16,
            projection_matrix=(0.0,) * 16,
        ),
        perception=PerceptionEvidence(
            prompt="red cube",
            localization_box=(297, 189, 344, 245),
            localization_score=0.81,
            localization_iou=0.87,
            grasp_center=(320.5, 217.0),
            grasp_size=(76.95, 31.35),
            angle_degrees=0.0,
            sampled_pixel=(321, 217),
            depth_m=0.6838,
            world_surface_point=WORLD_POINT,
            target_selection_passed=True,
            backend_geometry_passed=True,
            backprojection_gate_passed=True,
            segmentation_target_match=True,
            ray_target_match=True,
        ),
        control=FrozenControlProtocol(),
        candidates=(_candidate(0.0, True), _candidate(180.0, False)),
    )


def _write_stage_6a_fixture(root: Path) -> tuple[Path, Path]:
    source_dir = root / "stage_6a"
    output_dir = root / "stage_6a1"
    source_dir.mkdir()
    rgb_bytes = b"fixed-stage-6a-rgb"
    (source_dir / "rgb.png").write_bytes(rgb_bytes)
    rgb_sha256 = hashlib.sha256(rgb_bytes).hexdigest()
    plan = _plan(rgb_sha256)
    write_geometry_execution_plan(source_dir / "execution_plan.json", plan)
    summary = {
        "protocol": STAGE_6A_PROTOCOL_VERSION,
        "status": "success",
        "target_name": "cube",
        "backend": "geometry",
        "world_surface_point": list(WORLD_POINT),
        "simulation_steps_after_capture": 0,
        "scientific_gate_passed": True,
    }
    metadata = {
        "protocol": STAGE_6A_PROTOCOL_VERSION,
        "status": "success",
        "config": {
            "seed": 42,
            "target_name": "cube",
            "prompt": "red cube",
            "backend": "geometry",
        },
        "scene": {
            "object_poses": {
                "cube": {"position": list(CUBE_TRUTH_CENTER)}
            }
        },
        "rgb_sha256": rgb_sha256,
        "backprojection": {
            "world_x": WORLD_POINT[0],
            "world_y": WORLD_POINT[1],
            "world_z": WORLD_POINT[2],
        },
        "summary": summary,
        "simulation_steps_after_capture": 0,
        **{name: False for name in EXECUTION_FLAGS},
    }
    (source_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (source_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return source_dir, output_dir


def _source_hashes(source_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_dir.iterdir())
        if path.is_file()
    }


def test_offline_runner_preserves_source_and_writes_diagnostic(
    tmp_path: Path,
) -> None:
    source_dir, output_dir = _write_stage_6a_fixture(tmp_path)
    hashes_before = _source_hashes(source_dir)

    result = run_center_bias_diagnostic(
        CenterBiasDiagnosticConfig(source_dir, output_dir)
    )

    assert result["status"] == "success"
    assert result["evidence_role"] == "formal"
    assert result["diagnostic_only"] is True
    assert result["plan_modified"] is False
    assert result["scientific_gate_reinterpreted"] is False
    assert result["measurement"]["xy_offset_m"] == pytest.approx(
        0.026549556836982145
    )
    assert (
        result["measurement"]["xy_within_reference_threshold"] is False
    )
    assert _source_hashes(source_dir) == hashes_before
    assert not (source_dir / "center_bias_diagnostic.json").exists()
    assert (output_dir / "center_bias_diagnostic.json").is_file()
    assert (output_dir / "center_bias_diagnostic.csv").is_file()
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_hashes_before"] == hashes_before
    assert metadata["source_hashes_after"] == hashes_before
    assert all(metadata[name] is False for name in EXECUTION_FLAGS)


def _mutate_fixture(source_dir: Path, mutation: str) -> None:
    summary_path = source_dir / "summary.json"
    metadata_path = source_dir / "metadata.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "rgb_hash":
        metadata["rgb_sha256"] = "b" * 64
    elif mutation == "world_point":
        summary["world_surface_point"][0] += 0.01
    elif mutation == "execution_flag":
        metadata["motor_control_executed"] = True
    elif mutation == "protocol":
        summary["protocol"] = "unexpected_protocol"
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    ["rgb_hash", "world_point", "execution_flag", "protocol"],
)
def test_offline_runner_rejects_inconsistent_stage_6a_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_dir, output_dir = _write_stage_6a_fixture(tmp_path)
    _mutate_fixture(source_dir, mutation)
    output_dir.mkdir()
    (output_dir / "center_bias_diagnostic.json").write_text(
        "stale success", encoding="utf-8"
    )

    result = run_center_bias_diagnostic(
        CenterBiasDiagnosticConfig(source_dir, output_dir)
    )

    assert result["status"] == "failure"
    assert result["failure_stage"] == "input_validation"
    assert not (output_dir / "center_bias_diagnostic.json").exists()
    assert not (output_dir / "center_bias_diagnostic.csv").exists()
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["diagnostic_only"] is True
    assert all(metadata[name] is False for name in EXECUTION_FLAGS)


@pytest.mark.parametrize(
    ("source_name", "output_name"),
    [
        ("same", "same"),
        ("source", "source/child"),
        ("output/child", "output"),
    ],
)
def test_offline_runner_rejects_source_output_overlap(
    tmp_path: Path,
    source_name: str,
    output_name: str,
) -> None:
    with pytest.raises(ValueError, match="separate"):
        CenterBiasDiagnosticConfig(
            tmp_path / source_name,
            tmp_path / output_name,
        )


def test_offline_runner_rejects_unknown_evidence_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence_role"):
        CenterBiasDiagnosticConfig(
            tmp_path / "source",
            tmp_path / "output",
            evidence_role="replacement",
        )


@pytest.mark.parametrize(
    ("source_dir", "output_dir", "evidence_role"),
    [
        (DEFAULT_SOURCE_DIR, Path("/tmp/stage6a1-repro"), "reproducibility"),
        (Path("/tmp/stage6a-repro"), DEFAULT_OUTPUT_DIR, "reproducibility"),
        (
            Path(
                "data/processed/pybullet/grasp_execution/"
                "stage_6a_geometry_preflight_reproducibility"
            ),
            Path("/tmp/stage6a1-formal"),
            "formal",
        ),
        (
            Path("/tmp/stage6a-formal"),
            Path(
                "data/processed/pybullet/grasp_execution/"
                "stage_6a1_center_bias_reproducibility"
            ),
            "formal",
        ),
    ],
)
def test_config_enforces_canonical_evidence_role_isolation(
    source_dir: Path,
    output_dir: Path,
    evidence_role: str,
) -> None:
    with pytest.raises(ValueError, match="evidence_role"):
        CenterBiasDiagnosticConfig(
            source_dir,
            output_dir,
            evidence_role=evidence_role,
        )


def test_config_does_not_relabel_existing_output_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    original_metadata = {
        "protocol": "stage_6a1_center_bias_diagnostic_v1",
        "evidence_role": "formal",
        "status": "success",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(original_metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="existing evidence_role"):
        CenterBiasDiagnosticConfig(
            tmp_path / "source",
            output_dir,
            evidence_role="reproducibility",
        )

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == (
        original_metadata
    )


def test_cli_returns_nonzero_for_failed_audit(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.simulation.pybullet.run_center_bias_diagnostic",
            "--source-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )["status"] == "failure"


def test_publication_failure_removes_partial_success_and_writes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, output_dir = _write_stage_6a_fixture(tmp_path)

    def fail_csv(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "write_diagnostic_csv", fail_csv)
    result = run_center_bias_diagnostic(
        CenterBiasDiagnosticConfig(source_dir, output_dir)
    )

    assert result["status"] == "failure"
    assert result["failure_stage"] == "output_publication"
    assert not (output_dir / "center_bias_diagnostic.json").exists()
    assert not (output_dir / "center_bias_diagnostic.csv").exists()
    assert json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )["status"] == "failure"


def test_runner_does_not_convert_unexpected_programming_error_to_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, output_dir = _write_stage_6a_fixture(tmp_path)

    def unexpected_bug(*args, **kwargs):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(
        runner_module,
        "compute_center_bias",
        unexpected_bug,
    )

    with pytest.raises(RuntimeError, match="unexpected bug"):
        run_center_bias_diagnostic(
            CenterBiasDiagnosticConfig(source_dir, output_dir)
        )
