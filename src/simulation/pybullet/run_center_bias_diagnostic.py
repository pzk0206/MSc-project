"""Strict offline Stage 6A.1 center-bias evidence runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from src.simulation.pybullet.center_bias_diagnostic import (
    PROTOCOL_VERSION,
    CenterBiasMeasurement,
    compute_center_bias,
    write_diagnostic_csv,
    write_diagnostic_json,
)
from src.simulation.pybullet.execution_plan import (
    PROTOCOL_VERSION as STAGE_6A_PROTOCOL_VERSION,
    GeometryExecutionPlan,
    load_geometry_execution_plan,
)


DEFAULT_SOURCE_DIR = Path(
    "data/processed/pybullet/grasp_execution/stage_6a_geometry_preflight"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/"
    "stage_6a1_center_bias_diagnostic"
)
REPRODUCIBILITY_SOURCE_DIR = Path(
    "data/processed/pybullet/grasp_execution/"
    "stage_6a_geometry_preflight_reproducibility"
)
REPRODUCIBILITY_OUTPUT_DIR = Path(
    "data/processed/pybullet/grasp_execution/"
    "stage_6a1_center_bias_reproducibility"
)
SOURCE_FILENAMES = (
    "summary.json",
    "metadata.json",
    "execution_plan.json",
    "rgb.png",
)
EXECUTION_FLAGS = (
    "motor_control_executed",
    "trajectory_executed",
    "gripper_closed",
    "contact_evaluated",
    "object_lifted",
    "physical_grasp_executed",
)


@dataclass(frozen=True)
class CenterBiasDiagnosticConfig:
    """Input and isolated output locations for one offline audit."""

    source_dir: Path = DEFAULT_SOURCE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    evidence_role: str = "formal"

    def __post_init__(self) -> None:
        source = Path(self.source_dir).resolve()
        output = Path(self.output_dir).resolve()
        if (
            source == output
            or source in output.parents
            or output in source.parents
        ):
            raise ValueError(
                "Stage 6A source and Stage 6A.1 output must be separate"
            )
        if self.evidence_role not in ("formal", "reproducibility"):
            raise ValueError(
                "evidence_role must be formal or reproducibility"
            )
        formal_source = DEFAULT_SOURCE_DIR.resolve()
        formal_output = DEFAULT_OUTPUT_DIR.resolve()
        reproducibility_source = REPRODUCIBILITY_SOURCE_DIR.resolve()
        reproducibility_output = REPRODUCIBILITY_OUTPUT_DIR.resolve()
        if self.evidence_role == "formal" and (
            source == reproducibility_source
            or output == reproducibility_output
        ):
            raise ValueError(
                "formal evidence_role cannot use reproducibility directories"
            )
        if self.evidence_role == "reproducibility" and (
            source == formal_source or output == formal_output
        ):
            raise ValueError(
                "reproducibility evidence_role cannot use formal directories"
            )
        existing_metadata = output / "metadata.json"
        if existing_metadata.is_file():
            try:
                existing = json.loads(
                    existing_metadata.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    "existing Stage 6A.1 metadata is invalid"
                ) from exc
            if not isinstance(existing, dict):
                raise ValueError("existing Stage 6A.1 metadata is invalid")
            existing_role = existing.get("evidence_role")
            if existing_role != self.evidence_role:
                raise ValueError(
                    "existing evidence_role does not match requested role"
                )
        object.__setattr__(self, "source_dir", source)
        object.__setattr__(self, "output_dir", output)


class _SourceIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(source_dir: Path) -> dict[str, str]:
    return {
        name: _sha256(source_dir / name)
        for name in SOURCE_FILENAMES
    }


def _load_mapping(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _point3(values: object, name: str) -> tuple[float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain three finite values")
    try:
        point = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must contain three finite values")
    return point


def _expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} is inconsistent across Stage 6A evidence")


def _validate_source_evidence(
    *,
    source_dir: Path,
    summary: Mapping[str, object],
    metadata: Mapping[str, object],
    plan: GeometryExecutionPlan,
) -> tuple[tuple[float, float, float], bool]:
    _expect_equal(summary.get("protocol"), STAGE_6A_PROTOCOL_VERSION, "protocol")
    _expect_equal(metadata.get("protocol"), STAGE_6A_PROTOCOL_VERSION, "protocol")
    _expect_equal(summary.get("status"), "success", "summary status")
    _expect_equal(metadata.get("status"), "success", "metadata status")
    _expect_equal(metadata.get("summary"), summary, "nested summary")

    config = _mapping(metadata.get("config"), "metadata.config")
    expected_protocol_fields = {
        "seed": plan.scene_seed,
        "target_name": plan.target_name,
        "backend": plan.backend,
        "prompt": plan.prompt,
    }
    for name, expected in expected_protocol_fields.items():
        _expect_equal(config.get(name), expected, name)
    _expect_equal(summary.get("target_name"), plan.target_name, "target_name")
    _expect_equal(summary.get("backend"), plan.backend, "backend")

    plan_point = tuple(plan.perception.world_surface_point)
    summary_point = _point3(
        summary.get("world_surface_point"),
        "summary.world_surface_point",
    )
    backprojection = _mapping(
        metadata.get("backprojection"),
        "metadata.backprojection",
    )
    backprojection_point = _point3(
        (
            backprojection.get("world_x"),
            backprojection.get("world_y"),
            backprojection.get("world_z"),
        ),
        "metadata.backprojection.world_point",
    )
    _expect_equal(summary_point, plan_point, "world surface point")
    _expect_equal(backprojection_point, plan_point, "world surface point")

    actual_rgb_sha256 = _sha256(source_dir / "rgb.png")
    _expect_equal(metadata.get("rgb_sha256"), plan.rgb_sha256, "RGB SHA-256")
    _expect_equal(actual_rgb_sha256, plan.rgb_sha256, "RGB SHA-256")
    _expect_equal(
        summary.get("simulation_steps_after_capture"),
        0,
        "summary post-capture steps",
    )
    _expect_equal(
        metadata.get("simulation_steps_after_capture"),
        0,
        "metadata post-capture steps",
    )
    for name in EXECUTION_FLAGS:
        if metadata.get(name) is not False:
            raise ValueError(f"Stage 6A {name} must be false")

    historical_gate = summary.get("scientific_gate_passed")
    if historical_gate is not True:
        raise ValueError("Stage 6A scientific gate must be true")
    scene = _mapping(metadata.get("scene"), "metadata.scene")
    object_poses = _mapping(
        scene.get("object_poses"),
        "metadata.scene.object_poses",
    )
    cube = _mapping(
        object_poses.get("cube"),
        "metadata.scene.object_poses.cube",
    )
    cube_center = _point3(
        cube.get("position"),
        "metadata.scene.object_poses.cube.position",
    )
    return cube_center, historical_gate


def _cleanup_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "center_bias_diagnostic.json",
        "center_bias_diagnostic.csv",
        "metadata.json",
        ".center_bias_diagnostic.json.tmp",
        ".center_bias_diagnostic.csv.tmp",
        ".metadata.json.tmp",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _publish_success(
    output_dir: Path,
    payload: Mapping[str, object],
    measurement: CenterBiasMeasurement,
) -> None:
    temporary_and_final = (
        (
            output_dir / ".center_bias_diagnostic.json.tmp",
            output_dir / "center_bias_diagnostic.json",
        ),
        (
            output_dir / ".center_bias_diagnostic.csv.tmp",
            output_dir / "center_bias_diagnostic.csv",
        ),
        (
            output_dir / ".metadata.json.tmp",
            output_dir / "metadata.json",
        ),
    )
    write_diagnostic_json(temporary_and_final[0][0], payload)
    write_diagnostic_csv(temporary_and_final[1][0], measurement)
    write_diagnostic_json(temporary_and_final[2][0], payload)
    for temporary, final in temporary_and_final:
        temporary.replace(final)


def _failure(
    *,
    config: CenterBiasDiagnosticConfig,
    stage: str,
    reason: str,
    hashes_before: Mapping[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol": PROTOCOL_VERSION,
        "source_protocol": STAGE_6A_PROTOCOL_VERSION,
        "status": "failure",
        "failure_stage": stage,
        "failure_reason": reason,
        "evidence_role": config.evidence_role,
        "source_dir": str(config.source_dir),
        "source_hashes_before": dict(hashes_before),
        "diagnostic_only": True,
        "plan_modified": False,
        "scientific_gate_reinterpreted": False,
        **{name: False for name in EXECUTION_FLAGS},
    }
    write_diagnostic_json(config.output_dir / "metadata.json", payload)
    return payload


def run_center_bias_diagnostic(
    config: CenterBiasDiagnosticConfig,
) -> dict[str, object]:
    """Audit frozen Stage 6A evidence without running perception or physics."""

    source_dir = config.source_dir
    output_dir = config.output_dir
    _cleanup_outputs(output_dir)
    hashes_before: dict[str, str] = {}
    try:
        hashes_before = _source_hashes(source_dir)
        summary = _load_mapping(source_dir / "summary.json", "summary")
        metadata = _load_mapping(source_dir / "metadata.json", "metadata")
        plan = load_geometry_execution_plan(
            source_dir / "execution_plan.json"
        )
        cube_center, historical_gate = _validate_source_evidence(
            source_dir=source_dir,
            summary=summary,
            metadata=metadata,
            plan=plan,
        )
        measurement = compute_center_bias(
            plan.perception.world_surface_point,
            cube_center,
        )
        hashes_after = _source_hashes(source_dir)
        if hashes_after != hashes_before:
            raise _SourceIntegrityError(
                "Stage 6A source files changed during offline audit"
            )
    except _SourceIntegrityError as exc:
        return _failure(
            config=config,
            stage="source_integrity",
            reason=f"{type(exc).__name__}:{exc}",
            hashes_before=hashes_before,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _failure(
            config=config,
            stage="input_validation",
            reason=f"{type(exc).__name__}:{exc}",
            hashes_before=hashes_before,
        )

    payload: dict[str, object] = {
        "protocol": PROTOCOL_VERSION,
        "source_protocol": STAGE_6A_PROTOCOL_VERSION,
        "status": "success",
        "failure_stage": "",
        "failure_reason": "",
        "evidence_role": config.evidence_role,
        "source_dir": str(source_dir),
        "source_files": {
            name: str(source_dir / name) for name in SOURCE_FILENAMES
        },
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "scene_seed": plan.scene_seed,
        "target_name": plan.target_name,
        "backend": plan.backend,
        "prompt": plan.prompt,
        "rgb_sha256": plan.rgb_sha256,
        "historical_stage_6a_scientific_gate_passed": historical_gate,
        "measurement": asdict(measurement),
        "diagnostic_only": True,
        "plan_modified": False,
        "scientific_gate_reinterpreted": False,
        **{name: False for name in EXECUTION_FLAGS},
    }
    try:
        _publish_success(output_dir, payload, measurement)
    except (OSError, TypeError, ValueError) as exc:
        _cleanup_outputs(output_dir)
        return _failure(
            config=config,
            stage="output_publication",
            reason=f"{type(exc).__name__}:{exc}",
            hashes_before=hashes_before,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one frozen Stage 6A center prediction offline."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evidence-role",
        choices=("formal", "reproducibility"),
        default="formal",
    )
    args = parser.parse_args()
    result = run_center_bias_diagnostic(
        CenterBiasDiagnosticConfig(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            evidence_role=args.evidence_role,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
