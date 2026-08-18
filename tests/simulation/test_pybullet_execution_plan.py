import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.simulation.pybullet import execution_plan as execution_plan_module
from src.simulation.pybullet.execution_plan import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_OVERHEAD_MULTI_HEAD,
    CameraEvidence,
    FrozenControlProtocol,
    GeometryExecutionPlan,
    OverheadDeepGraspControlProtocol,
    PerceptionExecutionPlan,
    PerceptionEvidence,
    PlannedPoseCandidate,
    load_geometry_execution_plan,
    load_perception_execution_plan,
    write_geometry_execution_plan,
    write_perception_execution_plan,
)
from src.simulation.pybullet.pose_generation import ToolPose


def _pose(z: float) -> ToolPose:
    return ToolPose((0.48, 0.0, z), (1.0, 0.0, 0.0, 0.0))


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


def _plan() -> GeometryExecutionPlan:
    camera = CameraEvidence(
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
    )
    evidence = PerceptionEvidence(
        prompt="red cube",
        localization_box=(297, 189, 344, 245),
        localization_score=0.81,
        localization_iou=0.87,
        grasp_center=(320.5, 217.0),
        grasp_size=(76.95, 31.35),
        angle_degrees=0.0,
        sampled_pixel=(321, 217),
        depth_m=0.6838,
        world_surface_point=(0.5064, 0.0022, 0.675),
        target_selection_passed=True,
        backend_geometry_passed=True,
        backprojection_gate_passed=True,
        segmentation_target_match=True,
        ray_target_match=True,
    )
    return GeometryExecutionPlan(
        protocol_version=PROTOCOL_VERSION,
        scene_seed=42,
        target_name="cube",
        backend="geometry",
        prompt="red cube",
        model_id="IDEA-Research/grounding-dino-tiny",
        rgb_sha256="a" * 64,
        camera=camera,
        perception=evidence,
        control=FrozenControlProtocol(),
        candidates=(_candidate(0.0, True), _candidate(180.0, False)),
    )


def test_geometry_execution_plan_round_trips_strictly(tmp_path: Path) -> None:
    path = tmp_path / "execution_plan.json"
    plan = _plan()

    write_geometry_execution_plan(path, plan)

    assert load_geometry_execution_plan(path) == plan
    assert json.loads(path.read_text(encoding="utf-8"))["backend"] == (
        "geometry"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(backend="multi_head"), "backend"),
        (lambda value: value.update(scene_seed=43), "scene_seed"),
        (
            lambda value: value["candidates"].pop(),
            "exactly two candidates",
        ),
        (
            lambda value: value["candidates"][1].update(selected=True),
            "exactly one candidate",
        ),
        (
            lambda value: value["candidates"][0].update(
                pregrasp_ik=[0.0] * 6
            ),
            "pregrasp_ik",
        ),
        (
            lambda value: value["candidates"][0]["pregrasp_pose"].update(
                position=[0.48, 0.0, 0.79]
            ),
            "pregrasp height",
        ),
        (
            lambda value: value["perception"].update(depth_m=float("nan")),
            "finite",
        ),
        (
            lambda value: value.update(unexpected=True),
            "unexpected fields",
        ),
    ],
)
def test_geometry_execution_plan_rejects_tampering(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "execution_plan.json"
    write_geometry_execution_plan(path, _plan())
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, allow_nan=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_geometry_execution_plan(path)


def test_frozen_control_protocol_rejects_changed_thresholds() -> None:
    with pytest.raises(ValueError, match="frozen control protocol"):
        FrozenControlProtocol(tool_lift_command_m=0.11)


def test_plan_rejects_grasp_pose_that_disagrees_with_declared_control() -> None:
    """Catch serializing a -25 mm pose as the frozen +5 mm protocol."""

    plan = _plan()
    mismatched = replace(
        plan.candidates[0],
        grasp_depth_pose=_pose(0.650),
    )

    with pytest.raises(ValueError, match="control alignment"):
        replace(plan, candidates=(mismatched, plan.candidates[1]))


def test_overhead_deep_grasp_protocol_encodes_negative_standoff() -> None:
    protocol_type = getattr(
        execution_plan_module,
        "OverheadDeepGraspControlProtocol",
        None,
    )
    protocol_version = getattr(
        execution_plan_module,
        "PROTOCOL_VERSION_OVERHEAD",
        None,
    )
    assert protocol_type is not None
    assert protocol_version == "stage_6a_overhead_deep_grasp_v1"

    base = _plan()
    candidates = tuple(
        replace(candidate, grasp_depth_pose=_pose(0.650))
        for candidate in base.candidates
    )
    plan = PerceptionExecutionPlan(
        protocol_version=protocol_version,
        scene_seed=base.scene_seed,
        target_name=base.target_name,
        backend=base.backend,
        prompt=base.prompt,
        model_id=base.model_id,
        rgb_sha256=base.rgb_sha256,
        camera=base.camera,
        perception=base.perception,
        control=protocol_type(),
        candidates=candidates,
    )

    assert plan.control.grasp_depth_standoff_m == pytest.approx(-0.025)


def test_overhead_side_grasp_protocol_encodes_its_pose_ladder() -> None:
    """Keep the older side-grasp runner loadable under strict validation."""

    protocol_type = getattr(
        execution_plan_module,
        "OverheadSideGraspControlProtocol",
        None,
    )
    protocol_version = getattr(
        execution_plan_module,
        "PROTOCOL_VERSION_OVERHEAD_SIDE",
        None,
    )
    assert protocol_type is not None
    assert protocol_version == "stage_6a_overhead_side_grasp_v1"

    base = _plan()
    candidates = tuple(
        replace(
            candidate,
            pregrasp_pose=_pose(0.770),
            approach_pose=_pose(0.670),
            grasp_depth_pose=_pose(0.655),
        )
        for candidate in base.candidates
    )
    plan = PerceptionExecutionPlan(
        protocol_version=protocol_version,
        scene_seed=base.scene_seed,
        target_name=base.target_name,
        backend=base.backend,
        prompt=base.prompt,
        model_id=base.model_id,
        rgb_sha256=base.rgb_sha256,
        camera=base.camera,
        perception=base.perception,
        control=protocol_type(),
        candidates=candidates,
    )

    assert plan.control.approach_standoff_m == pytest.approx(-0.005)
    assert plan.control.grasp_depth_standoff_m == pytest.approx(-0.020)


def _overhead_multi_head_plan() -> PerceptionExecutionPlan:
    """Build a plan for the overhead multi-head deep-grasp protocol."""
    base = _plan()
    candidates = tuple(
        replace(candidate, grasp_depth_pose=_pose(0.650))
        for candidate in base.candidates
    )
    return PerceptionExecutionPlan(
        protocol_version=PROTOCOL_VERSION_OVERHEAD_MULTI_HEAD,
        scene_seed=base.scene_seed,
        target_name=base.target_name,
        backend="multi_head",
        prompt=base.prompt,
        model_id=base.model_id,
        rgb_sha256=base.rgb_sha256,
        camera=base.camera,
        perception=base.perception,
        control=OverheadDeepGraspControlProtocol(),
        candidates=candidates,
    )


def test_overhead_multi_head_protocol_round_trips(
    tmp_path: Path,
) -> None:
    """The overhead multi-head plan survives strict write/load."""

    path = tmp_path / "execution_plan.json"
    plan = _overhead_multi_head_plan()

    write_perception_execution_plan(path, plan)

    loaded = load_perception_execution_plan(path)
    assert loaded == plan
    assert loaded.backend == "multi_head"
    assert loaded.control.grasp_depth_standoff_m == pytest.approx(-0.025)


def test_overhead_multi_head_protocol_rejects_geometry_backend() -> None:
    """The multi-head overhead protocol is bound to multi_head only."""

    plan = _overhead_multi_head_plan()

    with pytest.raises(ValueError, match="multi_head"):
        replace(plan, backend="geometry")


def test_overhead_geometry_protocol_still_rejects_multi_head() -> None:
    """The geometry overhead protocol keeps rejecting multi_head plans."""

    base = _plan()
    candidates = tuple(
        replace(candidate, grasp_depth_pose=_pose(0.650))
        for candidate in base.candidates
    )
    plan = PerceptionExecutionPlan(
        protocol_version=getattr(
            execution_plan_module,
            "PROTOCOL_VERSION_OVERHEAD",
        ),
        scene_seed=base.scene_seed,
        target_name=base.target_name,
        backend="geometry",
        prompt=base.prompt,
        model_id=base.model_id,
        rgb_sha256=base.rgb_sha256,
        camera=base.camera,
        perception=base.perception,
        control=OverheadDeepGraspControlProtocol(),
        candidates=candidates,
    )

    with pytest.raises(ValueError, match="only supports geometry"):
        replace(plan, backend="multi_head")
