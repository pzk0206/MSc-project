"""Strict serialized contract for a geometry-derived grasp execution plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.simulation.pybullet.pose_generation import ToolPose


PROTOCOL_VERSION = "stage_6a_geometry_preflight_v1"
PROTOCOL_VERSION_V2 = "stage_6a2_center_recovery_v1"
PROTOCOL_VERSION_OVERHEAD = "stage_6a_overhead_deep_grasp_v1"
PROTOCOL_VERSION_OVERHEAD_SIDE = "stage_6a_overhead_side_grasp_v1"

VALID_BACKENDS = ("geometry", "multi_head")


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _fixed_tuple(
    values: Sequence[float],
    length: int,
    name: str,
) -> tuple[float, ...]:
    result = _finite(values, name)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _validate_pose(pose: ToolPose, name: str) -> None:
    _fixed_tuple(pose.position, 3, f"{name}.position")
    quaternion = _fixed_tuple(
        pose.quaternion_xyzw,
        4,
        f"{name}.quaternion_xyzw",
    )
    if not math.isclose(
        float(np.linalg.norm(quaternion)),
        1.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name} quaternion must be normalized")


@dataclass(frozen=True)
class FrozenControlProtocol:
    """Stage 1--5 values that perception backends cannot change."""

    approach_standoff_m: float = 0.02
    grasp_depth_standoff_m: float = 0.005
    pregrasp_offset_m: float = 0.10
    collision_clearance_m: float = 0.002
    samples_per_segment: int = 21
    tool_lift_command_m: float = 0.12
    minimum_object_lift_m: float = 0.10
    lift_hold_steps: int = 240

    def __post_init__(self) -> None:
        expected = (0.02, 0.005, 0.10, 0.002, 21, 0.12, 0.10, 240)
        actual = tuple(getattr(self, field.name) for field in fields(self))
        if actual != expected:
            raise ValueError("frozen control protocol values cannot change")


@dataclass(frozen=True)
class OverheadDeepGraspControlProtocol:
    """Explicit amended control used by the overhead deep-grasp pilot."""

    approach_standoff_m: float = 0.02
    grasp_depth_standoff_m: float = -0.025
    pregrasp_offset_m: float = 0.10
    collision_clearance_m: float = 0.002
    samples_per_segment: int = 21
    tool_lift_command_m: float = 0.12
    minimum_object_lift_m: float = 0.10
    lift_hold_steps: int = 240

    def __post_init__(self) -> None:
        expected = (0.02, -0.025, 0.10, 0.002, 21, 0.12, 0.10, 240)
        actual = tuple(getattr(self, field.name) for field in fields(self))
        if actual != expected:
            raise ValueError(
                "overhead deep-grasp control protocol values cannot change"
            )


@dataclass(frozen=True)
class OverheadSideGraspControlProtocol:
    """Explicit pose ladder used by the earlier overhead side-grasp pilot."""

    approach_standoff_m: float = -0.005
    grasp_depth_standoff_m: float = -0.020
    pregrasp_offset_m: float = 0.10
    collision_clearance_m: float = 0.002
    samples_per_segment: int = 21
    tool_lift_command_m: float = 0.12
    minimum_object_lift_m: float = 0.10
    lift_hold_steps: int = 240

    def __post_init__(self) -> None:
        expected = (-0.005, -0.020, 0.10, 0.002, 21, 0.12, 0.10, 240)
        actual = tuple(getattr(self, field.name) for field in fields(self))
        if actual != expected:
            raise ValueError(
                "overhead side-grasp control protocol values cannot change"
            )


ControlProtocol = (
    FrozenControlProtocol
    | OverheadDeepGraspControlProtocol
    | OverheadSideGraspControlProtocol
)


def _validate_control_alignment(
    *,
    control: ControlProtocol,
    perception: "PerceptionEvidence",
    candidates: Sequence["PlannedPoseCandidate"],
) -> None:
    """Tie serialized control offsets to every executable candidate pose."""

    surface_z = float(perception.world_surface_point[2])
    expected_approach_z = surface_z + control.approach_standoff_m
    expected_grasp_depth_z = surface_z + control.grasp_depth_standoff_m
    for index, candidate in enumerate(candidates):
        approach_z = float(candidate.approach_pose.position[2])
        grasp_depth_z = float(candidate.grasp_depth_pose.position[2])
        pregrasp_z = float(candidate.pregrasp_pose.position[2])
        expected_pregrasp_z = approach_z + control.pregrasp_offset_m
        values = (
            ("approach", approach_z, expected_approach_z),
            ("grasp_depth", grasp_depth_z, expected_grasp_depth_z),
            ("pregrasp", pregrasp_z, expected_pregrasp_z),
        )
        for name, actual, expected in values:
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "candidate control alignment mismatch: "
                    f"candidate {index} {name} z={actual:.12f}, "
                    f"expected {expected:.12f}"
                )


@dataclass(frozen=True)
class CameraEvidence:
    """Fixed camera parameters and matrices used for one prediction."""

    width: int
    height: int
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float
    near: float
    far: float
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]

    def __post_init__(self) -> None:
        if (self.width, self.height) != (640, 480):
            raise ValueError("camera dimensions must be 640x480")
        _fixed_tuple(self.eye, 3, "camera.eye")
        _fixed_tuple(self.target, 3, "camera.target")
        _fixed_tuple(self.up, 3, "camera.up")
        _fixed_tuple(self.view_matrix, 16, "camera.view_matrix")
        _fixed_tuple(
            self.projection_matrix,
            16,
            "camera.projection_matrix",
        )
        scalars = _finite(
            (self.fov_degrees, self.near, self.far),
            "camera scalars",
        )
        if not 0.0 < scalars[0] < 180.0 or not 0.0 < scalars[1] < scalars[2]:
            raise ValueError("camera clipping or field of view is invalid")


@dataclass(frozen=True)
class CenterRecoveryEvidence:
    """Evidence produced by windowed top-surface depth recovery.

    Records the sampling strategy, pixel selection, and both the
    original (single-pixel) and corrected world points so that the
    effect of the rule is fully auditable.
    """

    protocol: str
    window_size: int
    original_depth_m: float
    corrected_depth_m: float
    original_world_surface_point: tuple[float, float, float]
    corrected_world_surface_point: tuple[float, float, float]
    sampled_pixel: tuple[int, int]
    target_body_id: int
    target_body_id_source: str  # "segmentation_mask"

    def __post_init__(self) -> None:
        if self.protocol != "windowed_min_depth_target_mask_v1":
            raise ValueError("unknown center recovery protocol")
        if self.window_size < 1 or self.window_size % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        for name in (
            "original_world_surface_point",
            "corrected_world_surface_point",
        ):
            _fixed_tuple(getattr(self, name), 3, name)
        scalars = _finite(
            (self.original_depth_m, self.corrected_depth_m),
            "center recovery depths",
        )
        if scalars[0] <= 0.0 or scalars[1] <= 0.0:
            raise ValueError("center recovery depths must be positive")
        if len(self.sampled_pixel) != 2 or min(self.sampled_pixel) < 0:
            raise ValueError("sampled_pixel must be non-negative")
        if self.target_body_id < 0:
            raise ValueError("target_body_id must be non-negative")
        if self.target_body_id_source != "segmentation_mask":
            raise ValueError(
                "target_body_id_source must be segmentation_mask"
            )


@dataclass(frozen=True)
class PerceptionEvidence:
    """Immutable evidence produced before static pose selection."""

    prompt: str
    localization_box: tuple[int, int, int, int]
    localization_score: float
    localization_iou: float
    grasp_center: tuple[float, float]
    grasp_size: tuple[float, float]
    angle_degrees: float
    sampled_pixel: tuple[int, int]
    depth_m: float
    world_surface_point: tuple[float, float, float]
    target_selection_passed: bool
    backend_geometry_passed: bool
    backprojection_gate_passed: bool
    segmentation_target_match: bool
    ray_target_match: bool
    center_recovery: CenterRecoveryEvidence | None = None

    def __post_init__(self) -> None:
        if self.prompt != "red cube":
            raise ValueError("perception prompt must be red cube")
        if len(self.localization_box) != 4:
            raise ValueError("localization_box must contain four values")
        left, top, right, bottom = self.localization_box
        if min(left, top) < 0 or right <= left or bottom <= top:
            raise ValueError("localization_box must have positive image area")
        scalars = _finite(
            (
                self.localization_score,
                self.localization_iou,
                *self.grasp_center,
                *self.grasp_size,
                self.angle_degrees,
                self.depth_m,
                *self.world_surface_point,
            ),
            "perception evidence",
        )
        if not 0.0 <= scalars[0] <= 1.0 or not 0.0 <= scalars[1] <= 1.0:
            raise ValueError("perception scores must be between zero and one")
        if any(value <= 0.0 for value in self.grasp_size):
            raise ValueError("grasp size must be positive")
        if self.depth_m <= 0.0:
            raise ValueError("depth_m must be positive")
        if len(self.sampled_pixel) != 2 or min(self.sampled_pixel) < 0:
            raise ValueError("sampled_pixel must contain non-negative x and y")
        gates = (
            self.target_selection_passed,
            self.backend_geometry_passed,
            self.backprojection_gate_passed,
            self.segmentation_target_match,
            self.ray_target_match,
        )
        if not all(isinstance(value, bool) for value in gates):
            raise ValueError("perception gate fields must be booleans")
        if not all(gates):
            raise ValueError("execution plans require all perception gates")


@dataclass(frozen=True)
class PlannedPoseCandidate:
    """One symmetry-resolved and fully audited static pose chain."""

    symmetry_degrees: float
    finger_axis_world: tuple[float, float, float]
    pregrasp_pose: ToolPose
    approach_pose: ToolPose
    grasp_depth_pose: ToolPose
    pregrasp_ik: tuple[float, ...]
    approach_ik: tuple[float, ...]
    grasp_depth_ik: tuple[float, ...]
    ik_fk_passed: bool
    clearance_passed: bool
    checked_state_count: int
    minimum_clearance_m: float
    environment_collision_count: int
    self_collision_count: int
    total_normalized_joint_cost: float
    gate_passed: bool
    selected: bool
    failure_reason: str

    def __post_init__(self) -> None:
        if self.symmetry_degrees not in (0.0, 180.0):
            raise ValueError("candidate symmetry must be 0 or 180 degrees")
        _fixed_tuple(self.finger_axis_world, 3, "finger_axis_world")
        for name, pose in (
            ("pregrasp_pose", self.pregrasp_pose),
            ("approach_pose", self.approach_pose),
            ("grasp_depth_pose", self.grasp_depth_pose),
        ):
            _validate_pose(pose, name)
        for name, values in (
            ("pregrasp_ik", self.pregrasp_ik),
            ("approach_ik", self.approach_ik),
            ("grasp_depth_ik", self.grasp_depth_ik),
        ):
            _fixed_tuple(values, 7, name)
        positions = (
            self.pregrasp_pose.position,
            self.approach_pose.position,
            self.grasp_depth_pose.position,
        )
        if not all(
            np.allclose(positions[0][:2], position[:2], atol=1e-9)
            for position in positions[1:]
        ):
            raise ValueError("candidate poses must share the same XY")
        quaternions = (
            self.pregrasp_pose.quaternion_xyzw,
            self.approach_pose.quaternion_xyzw,
            self.grasp_depth_pose.quaternion_xyzw,
        )
        if not all(
            np.allclose(quaternions[0], quaternion, atol=1e-9)
            for quaternion in quaternions[1:]
        ):
            raise ValueError("candidate poses must share one quaternion")
        if not math.isclose(
            positions[0][2] - positions[1][2],
            0.10,
            abs_tol=1e-9,
        ):
            raise ValueError("pregrasp height must be 0.10 m above approach")
        if positions[1][2] <= positions[2][2]:
            raise ValueError("approach height must be above grasp depth")
        _finite(
            (self.minimum_clearance_m, self.total_normalized_joint_cost),
            "candidate audit values",
        )
        if self.checked_state_count != 82:
            raise ValueError("candidate audit must check 82 states")
        if min(self.environment_collision_count, self.self_collision_count) < 0:
            raise ValueError("collision counts cannot be negative")
        if self.gate_passed and not (
            self.ik_fk_passed
            and self.clearance_passed
            and self.minimum_clearance_m >= 0.002
            and self.environment_collision_count == 0
            and self.self_collision_count == 0
            and not self.failure_reason
        ):
            raise ValueError("passing candidate has inconsistent audit fields")
        if self.selected and not self.gate_passed:
            raise ValueError("selected candidate must pass its gate")


@dataclass(frozen=True)
class GeometryExecutionPlan:
    """The only Stage 6A output that Stage 6B may execute."""

    protocol_version: str
    scene_seed: int
    target_name: str
    backend: str
    prompt: str
    model_id: str
    rgb_sha256: str
    camera: CameraEvidence
    perception: PerceptionEvidence
    control: FrozenControlProtocol
    candidates: tuple[PlannedPoseCandidate, PlannedPoseCandidate]

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if self.scene_seed != 42:
            raise ValueError("scene_seed must be 42")
        if self.target_name != "cube":
            raise ValueError("target_name must be cube")
        if self.backend != "geometry":
            raise ValueError("backend must be geometry")
        if self.prompt != "red cube" or self.perception.prompt != self.prompt:
            raise ValueError("plan prompt must be red cube")
        if self.model_id != "IDEA-Research/grounding-dino-tiny":
            raise ValueError("unexpected Grounding DINO model_id")
        if len(self.rgb_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.rgb_sha256
        ):
            raise ValueError("rgb_sha256 must be a lowercase SHA-256 digest")
        if len(self.candidates) != 2:
            raise ValueError("plan must contain exactly two candidates")
        if tuple(row.symmetry_degrees for row in self.candidates) != (
            0.0,
            180.0,
        ):
            raise ValueError("candidates must be ordered 0 then 180 degrees")
        if sum(row.selected for row in self.candidates) != 1:
            raise ValueError("plan must contain exactly one candidate selected")
        if not isinstance(self.control, FrozenControlProtocol):
            raise ValueError("geometry V1 requires frozen control protocol")
        _validate_control_alignment(
            control=self.control,
            perception=self.perception,
            candidates=self.candidates,
        )


@dataclass(frozen=True)
class PerceptionExecutionPlan:
    """Stage 6A.2 plan supporting geometry and multi-head CNN backends.

    Relaxes ``GeometryExecutionPlan`` to accept either backend and
    mandates ``CenterRecoveryEvidence`` for V2 protocol plans.
    """

    protocol_version: str
    scene_seed: int
    target_name: str
    backend: str
    prompt: str
    model_id: str
    rgb_sha256: str
    camera: CameraEvidence
    perception: PerceptionEvidence
    control: ControlProtocol
    candidates: tuple[PlannedPoseCandidate, PlannedPoseCandidate]

    def __post_init__(self) -> None:
        if self.protocol_version not in (
            PROTOCOL_VERSION,
            PROTOCOL_VERSION_V2,
            PROTOCOL_VERSION_OVERHEAD,
            PROTOCOL_VERSION_OVERHEAD_SIDE,
        ):
            raise ValueError("unsupported protocol_version")
        if self.scene_seed != 42:
            raise ValueError("scene_seed must be 42")
        if self.target_name != "cube":
            raise ValueError("target_name must be cube")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {VALID_BACKENDS}"
            )
        if self.prompt != "red cube" or self.perception.prompt != self.prompt:
            raise ValueError("plan prompt must be red cube")
        if self.model_id != "IDEA-Research/grounding-dino-tiny":
            raise ValueError("unexpected Grounding DINO model_id")
        if len(self.rgb_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.rgb_sha256
        ):
            raise ValueError("rgb_sha256 must be a lowercase SHA-256 digest")
        if len(self.candidates) != 2:
            raise ValueError("plan must contain exactly two candidates")
        if tuple(row.symmetry_degrees for row in self.candidates) != (
            0.0,
            180.0,
        ):
            raise ValueError("candidates must be ordered 0 then 180 degrees")
        if sum(row.selected for row in self.candidates) != 1:
            raise ValueError(
                "plan must contain exactly one candidate selected"
            )
        if (
            self.protocol_version == PROTOCOL_VERSION
            and self.backend != "geometry"
        ):
            raise ValueError(
                "V1 protocol only supports geometry backend"
            )
        if self.protocol_version == PROTOCOL_VERSION_OVERHEAD:
            if self.backend != "geometry":
                raise ValueError(
                    "overhead deep-grasp protocol only supports geometry"
                )
            if not isinstance(
                self.control,
                OverheadDeepGraspControlProtocol,
            ):
                raise ValueError(
                    "overhead protocol requires overhead deep-grasp control"
                )
        elif self.protocol_version == PROTOCOL_VERSION_OVERHEAD_SIDE:
            if self.backend != "geometry":
                raise ValueError(
                    "overhead side-grasp protocol only supports geometry"
                )
            if not isinstance(
                self.control,
                OverheadSideGraspControlProtocol,
            ):
                raise ValueError(
                    "overhead side-grasp protocol requires side-grasp control"
                )
        elif not isinstance(self.control, FrozenControlProtocol):
            raise ValueError(
                "standard perception protocols require frozen control"
            )
        _validate_control_alignment(
            control=self.control,
            perception=self.perception,
            candidates=self.candidates,
        )


def _expect_fields(
    value: Mapping[str, Any],
    data_class: type,
    name: str,
) -> None:
    expected = {field.name for field in fields(data_class)}
    actual = set(value)
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise ValueError(f"{name} contains unexpected fields: {sorted(extra)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")


def _tool_pose(value: Mapping[str, Any], name: str) -> ToolPose:
    _expect_fields(value, ToolPose, name)
    return ToolPose(
        position=tuple(value["position"]),
        quaternion_xyzw=tuple(value["quaternion_xyzw"]),
    )


def _camera(value: Mapping[str, Any]) -> CameraEvidence:
    _expect_fields(value, CameraEvidence, "camera")
    return CameraEvidence(
        **{
            **value,
            "eye": tuple(value["eye"]),
            "target": tuple(value["target"]),
            "up": tuple(value["up"]),
            "view_matrix": tuple(value["view_matrix"]),
            "projection_matrix": tuple(value["projection_matrix"]),
        }
    )


def _perception(value: Mapping[str, Any]) -> PerceptionEvidence:
    converted = dict(value)
    # V1 protocol plans do not include center_recovery
    if "center_recovery" not in converted:
        converted["center_recovery"] = None
    _expect_fields(converted, PerceptionEvidence, "perception")
    for name in (
        "localization_box",
        "grasp_center",
        "grasp_size",
        "sampled_pixel",
        "world_surface_point",
    ):
        converted[name] = tuple(converted[name])
    return PerceptionEvidence(**converted)


def _candidate(value: Mapping[str, Any]) -> PlannedPoseCandidate:
    _expect_fields(value, PlannedPoseCandidate, "candidate")
    converted = dict(value)
    converted["finger_axis_world"] = tuple(converted["finger_axis_world"])
    for name in ("pregrasp_pose", "approach_pose", "grasp_depth_pose"):
        converted[name] = _tool_pose(converted[name], name)
    for name in ("pregrasp_ik", "approach_ik", "grasp_depth_ik"):
        converted[name] = tuple(converted[name])
    return PlannedPoseCandidate(**converted)


def write_geometry_execution_plan(
    path: Path,
    plan: GeometryExecutionPlan,
) -> None:
    """Validate and serialize one Stage 6A execution plan."""

    if not isinstance(plan, GeometryExecutionPlan):
        raise TypeError("plan must be a GeometryExecutionPlan")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            asdict(plan),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_geometry_execution_plan(path: Path) -> GeometryExecutionPlan:
    """Load and strictly revalidate a Stage 6A execution plan."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid execution plan JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("execution plan must be a JSON object")
    _expect_fields(value, GeometryExecutionPlan, "execution plan")
    control_value = value["control"]
    _expect_fields(control_value, FrozenControlProtocol, "control")
    return GeometryExecutionPlan(
        protocol_version=value["protocol_version"],
        scene_seed=value["scene_seed"],
        target_name=value["target_name"],
        backend=value["backend"],
        prompt=value["prompt"],
        model_id=value["model_id"],
        rgb_sha256=value["rgb_sha256"],
        camera=_camera(value["camera"]),
        perception=_perception(value["perception"]),
        control=FrozenControlProtocol(**control_value),
        candidates=tuple(_candidate(row) for row in value["candidates"]),
    )


def _recovery(value: Mapping[str, Any] | None) -> CenterRecoveryEvidence | None:
    if value is None:
        return None
    _expect_fields(value, CenterRecoveryEvidence, "center_recovery")
    converted = dict(value)
    for name in (
        "original_world_surface_point",
        "corrected_world_surface_point",
        "sampled_pixel",
    ):
        converted[name] = tuple(converted[name])
    return CenterRecoveryEvidence(**converted)


def _perception_v2(value: Mapping[str, Any]) -> PerceptionEvidence:
    converted = dict(value)
    # V1 protocol plans do not include center_recovery
    if "center_recovery" not in converted:
        converted["center_recovery"] = None
    _expect_fields(converted, PerceptionEvidence, "perception")
    for name in (
        "localization_box",
        "grasp_center",
        "grasp_size",
        "sampled_pixel",
        "world_surface_point",
    ):
        converted[name] = tuple(converted[name])
    converted["center_recovery"] = _recovery(
        converted.get("center_recovery")
    )
    return PerceptionEvidence(**converted)


def write_perception_execution_plan(
    path: Path,
    plan: PerceptionExecutionPlan,
) -> None:
    """Validate and serialize one Stage 6A.2 execution plan."""

    if not isinstance(plan, PerceptionExecutionPlan):
        raise TypeError("plan must be a PerceptionExecutionPlan")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            asdict(plan),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_perception_execution_plan(path: Path) -> PerceptionExecutionPlan:
    """Load and strictly revalidate a Stage 6A.2 execution plan."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid execution plan JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("execution plan must be a JSON object")
    _expect_fields(value, PerceptionExecutionPlan, "execution plan")
    protocol = value["protocol_version"]
    if protocol not in (
        PROTOCOL_VERSION,
        PROTOCOL_VERSION_V2,
        PROTOCOL_VERSION_OVERHEAD,
        PROTOCOL_VERSION_OVERHEAD_SIDE,
    ):
        raise ValueError(f"unsupported protocol_version: {protocol}")
    control_value = value["control"]
    control_types = {
        PROTOCOL_VERSION_OVERHEAD: OverheadDeepGraspControlProtocol,
        PROTOCOL_VERSION_OVERHEAD_SIDE: OverheadSideGraspControlProtocol,
    }
    control_type = control_types.get(protocol, FrozenControlProtocol)
    _expect_fields(control_value, control_type, "control")
    return PerceptionExecutionPlan(
        protocol_version=protocol,
        scene_seed=value["scene_seed"],
        target_name=value["target_name"],
        backend=value["backend"],
        prompt=value["prompt"],
        model_id=value["model_id"],
        rgb_sha256=value["rgb_sha256"],
        camera=_camera(value["camera"]),
        perception=_perception_v2(value["perception"]),
        control=control_type(**control_value),
        candidates=tuple(_candidate(row) for row in value["candidates"]),
    )
