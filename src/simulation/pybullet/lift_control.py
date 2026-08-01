"""Deterministic frozen-gripper cube lift and hold auditing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np
import pybullet as p

from src.simulation.pybullet.gripper_control import ContactEvent
from src.simulation.pybullet.kinematic_audit import PandaModelInfo
from src.simulation.pybullet.pose_generation import ToolPose


@dataclass(frozen=True)
class LiftConfig:
    """Timing and gate thresholds for one frozen-gripper lift."""

    lift_steps: int = 240
    settle_steps: int = 240
    hold_steps: int = 240
    tool_lift_command_m: float = 0.12
    minimum_object_lift_m: float = 0.10
    maximum_hold_relative_drift_m: float = 0.01
    minimum_trailing_bilateral_contact_steps: int = 120
    arm_joint_tolerance_rad: float = 0.01

    def __post_init__(self) -> None:
        if self.lift_steps <= 0:
            raise ValueError("lift_steps must be positive")
        if self.hold_steps <= 0:
            raise ValueError("hold_steps must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if (
            not math.isfinite(self.tool_lift_command_m)
            or self.tool_lift_command_m <= 0.0
        ):
            raise ValueError("tool_lift_command_m must be positive")
        if (
            not math.isfinite(self.minimum_object_lift_m)
            or self.minimum_object_lift_m <= 0.0
        ):
            raise ValueError("minimum_object_lift_m must be positive")
        if self.minimum_object_lift_m >= self.tool_lift_command_m:
            raise ValueError(
                "minimum object lift must be less than tool lift command"
            )
        if (
            not math.isfinite(self.maximum_hold_relative_drift_m)
            or self.maximum_hold_relative_drift_m <= 0.0
        ):
            raise ValueError(
                "maximum_hold_relative_drift_m must be positive"
            )
        if self.minimum_trailing_bilateral_contact_steps < 0:
            raise ValueError(
                "minimum trailing bilateral contact must be non-negative"
            )
        if (
            self.minimum_trailing_bilateral_contact_steps
            > self.hold_steps
        ):
            raise ValueError(
                "minimum trailing bilateral contact cannot exceed hold_steps"
            )
        if (
            not math.isfinite(self.arm_joint_tolerance_rad)
            or self.arm_joint_tolerance_rad <= 0.0
        ):
            raise ValueError("arm_joint_tolerance_rad must be positive")


@dataclass(frozen=True)
class LiftTraceRow:
    """One measured lift or hold state."""

    step: int
    phase: str
    commanded_arm_positions: tuple[float, ...]
    actual_arm_positions: tuple[float, ...]
    commanded_finger_positions: tuple[float, float]
    actual_finger_positions: tuple[float, float]
    actual_tool_position: tuple[float, float, float]
    actual_tool_quaternion_xyzw: tuple[float, float, float, float]
    target_position: tuple[float, float, float]
    target_quaternion_xyzw: tuple[float, float, float, float]
    target_lift_m: float
    tool_relative_to_target: tuple[float, float, float]
    relative_drift_m: float
    maximum_arm_joint_error_rad: float
    left_finger_contact: bool
    right_finger_contact: bool
    bilateral_contact: bool
    left_normal_force: float
    right_normal_force: float
    target_table_contact: bool
    prohibited_target_contact_count: int
    environment_collision_count: int
    self_collision_count: int


@dataclass(frozen=True)
class LiftResult:
    """Complete lift/hold trace and scientific gate result."""

    trace: tuple[LiftTraceRow, ...]
    contact_events: tuple[ContactEvent, ...]
    lift_reached: bool
    lift_settle_steps: int
    lift_endpoint_arm_error_rad: float
    endpoint_position_error_m: float
    endpoint_orientation_error_degrees: float
    minimum_hold_object_lift_m: float
    final_object_lift_m: float
    maximum_hold_relative_drift_m: float
    total_target_table_contact_count: int
    hold_target_table_contact_count: int
    trailing_bilateral_contact_steps: int
    left_finger_contacted: bool
    right_finger_contacted: bool
    maximum_left_normal_force: float
    maximum_right_normal_force: float
    maximum_arm_joint_error_rad: float
    prohibited_target_contact_count: int
    environment_collision_count: int
    self_collision_count: int
    all_states_finite: bool
    object_lift_gate_passed: bool
    table_release_gate_passed: bool
    relative_stability_gate_passed: bool
    lift_hold_gate_passed: bool
    gate_passed: bool
    failure_reason: str


def _joint_positions(
    robot_id: int,
    client_id: int,
    indices: Sequence[int],
    physics: Any,
) -> tuple[float, ...]:
    return tuple(
        float(
            physics.getJointState(
                robot_id,
                index,
                physicsClientId=client_id,
            )[0]
        )
        for index in indices
    )


def _positive_joint_forces(
    robot_id: int,
    client_id: int,
    indices: Sequence[int],
    physics: Any,
) -> tuple[float, ...]:
    forces = tuple(
        float(
            physics.getJointInfo(
                robot_id,
                index,
                physicsClientId=client_id,
            )[10]
        )
        for index in indices
    )
    if any(not math.isfinite(force) or force <= 0.0 for force in forces):
        raise ValueError("Panda motor force limits must be finite and positive")
    return forces


def _orientation_error_degrees(
    actual_xyzw: Sequence[float],
    target_xyzw: Sequence[float],
) -> float:
    dot = float(
        np.clip(
            abs(
                np.dot(
                    np.asarray(actual_xyzw, dtype=np.float64),
                    np.asarray(target_xyzw, dtype=np.float64),
                )
            ),
            0.0,
            1.0,
        )
    )
    return math.degrees(2.0 * math.acos(dot))


def execute_object_lift(
    *,
    robot_id: int,
    target_body_id: int,
    table_body_id: int,
    client_id: int,
    model: PandaModelInfo,
    lift_arm_positions: Sequence[float],
    lift_target_pose: ToolPose,
    frozen_finger_positions: Sequence[float],
    reference_target_position: Sequence[float],
    reference_tool_relative_to_target: Sequence[float],
    environment_body_ids: Sequence[int],
    allowed_environment_link_pairs: Sequence[tuple[int, int]] = (),
    config: LiftConfig = LiftConfig(),
    lift_complete_callback: Callable[[], None] | None = None,
    physics: Any = p,
) -> LiftResult:
    """Lift a contacted cube while preserving the frozen finger command."""

    arm_target = tuple(float(value) for value in lift_arm_positions)
    fingers_target = tuple(float(value) for value in frozen_finger_positions)
    reference_target = np.asarray(
        reference_target_position,
        dtype=np.float64,
    )
    reference_relative = np.asarray(
        reference_tool_relative_to_target,
        dtype=np.float64,
    )
    target_tool_position = np.asarray(
        lift_target_pose.position,
        dtype=np.float64,
    )
    target_tool_quaternion = np.asarray(
        lift_target_pose.quaternion_xyzw,
        dtype=np.float64,
    )
    if len(arm_target) != 7 or not all(
        math.isfinite(value) for value in arm_target
    ):
        raise ValueError("lift_arm_positions must contain seven finite values")
    if (
        len(fingers_target) != 2
        or not all(math.isfinite(value) for value in fingers_target)
        or not all(0.0 <= value <= 0.04 for value in fingers_target)
    ):
        raise ValueError(
            "frozen_finger_positions must contain two values within [0.0, 0.04]"
        )
    if reference_target.shape != (3,) or not np.all(
        np.isfinite(reference_target)
    ):
        raise ValueError("reference_target_position must be finite 3-D")
    if reference_relative.shape != (3,) or not np.all(
        np.isfinite(reference_relative)
    ):
        raise ValueError(
            "reference_tool_relative_to_target must be finite 3-D"
        )
    if (
        target_tool_position.shape != (3,)
        or target_tool_quaternion.shape != (4,)
        or not np.all(np.isfinite(target_tool_position))
        or not np.all(np.isfinite(target_tool_quaternion))
    ):
        raise ValueError("lift_target_pose must be finite")
    if len(model.finger_joint_indices) != 2:
        raise ValueError("Panda model must expose two finger joints")

    arm_forces = _positive_joint_forces(
        robot_id,
        client_id,
        model.arm_joint_indices,
        physics,
    )
    finger_forces = _positive_joint_forces(
        robot_id,
        client_id,
        model.finger_joint_indices,
        physics,
    )
    start_arm = np.asarray(
        _joint_positions(
            robot_id,
            client_id,
            model.arm_joint_indices,
            physics,
        ),
        dtype=np.float64,
    )
    arm_target_array = np.asarray(arm_target, dtype=np.float64)
    left_link, right_link = model.finger_joint_indices
    allowed_environment = {
        (int(link), int(body))
        for link, body in allowed_environment_link_pairs
    }
    adjacent = set(model.adjacent_link_pairs)
    trace: list[LiftTraceRow] = []
    events: list[ContactEvent] = []
    all_finite = True

    def command_and_sample(
        phase: str,
        arm_command: tuple[float, ...],
    ) -> LiftTraceRow:
        nonlocal all_finite
        physics.setJointMotorControlArray(
            robot_id,
            model.arm_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=arm_command,
            forces=arm_forces,
            physicsClientId=client_id,
        )
        physics.setJointMotorControlArray(
            robot_id,
            model.finger_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=fingers_target,
            forces=finger_forces,
            physicsClientId=client_id,
        )
        physics.stepSimulation(physicsClientId=client_id)
        step = len(trace) + 1
        actual_arm = _joint_positions(
            robot_id,
            client_id,
            model.arm_joint_indices,
            physics,
        )
        actual_fingers = _joint_positions(
            robot_id,
            client_id,
            model.finger_joint_indices,
            physics,
        )
        link_state = physics.getLinkState(
            robot_id,
            model.tool_link_index,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        tool_position = tuple(float(value) for value in link_state[4])
        tool_quaternion = tuple(float(value) for value in link_state[5])
        target_position_raw, target_quaternion_raw = (
            physics.getBasePositionAndOrientation(
                target_body_id,
                physicsClientId=client_id,
            )
        )
        target_position = tuple(float(value) for value in target_position_raw)
        target_quaternion = tuple(
            float(value) for value in target_quaternion_raw
        )
        relative = np.asarray(tool_position) - np.asarray(target_position)
        target_lift = float(target_position[2] - reference_target[2])
        relative_drift = float(np.linalg.norm(relative - reference_relative))
        left_force = 0.0
        right_force = 0.0
        prohibited_target = 0
        for contact in physics.getContactPoints(
            bodyA=robot_id,
            bodyB=target_body_id,
            physicsClientId=client_id,
        ):
            link = int(contact[3])
            force = float(contact[9])
            if (
                link in (left_link, right_link)
                and math.isfinite(force)
                and force > 0.0
            ):
                events.append(
                    ContactEvent(step, phase, link, target_body_id, force)
                )
                if link == left_link:
                    left_force = max(left_force, force)
                else:
                    right_force = max(right_force, force)
            elif float(contact[8]) <= 0.0:
                prohibited_target += 1
        target_table_contact = any(
            float(contact[8]) <= 0.0
            for contact in physics.getContactPoints(
                bodyA=target_body_id,
                bodyB=table_body_id,
                physicsClientId=client_id,
            )
        )
        environment_collisions = 0
        for body_id in environment_body_ids:
            for contact in physics.getContactPoints(
                bodyA=robot_id,
                bodyB=int(body_id),
                physicsClientId=client_id,
            ):
                if (int(contact[3]), int(body_id)) in allowed_environment:
                    continue
                if float(contact[8]) <= 0.0:
                    environment_collisions += 1
        self_collisions = 0
        seen_pairs = set()
        for contact in physics.getContactPoints(
            bodyA=robot_id,
            bodyB=robot_id,
            physicsClientId=client_id,
        ):
            pair = tuple(sorted((int(contact[3]), int(contact[4]))))
            if (
                pair[0] == pair[1]
                or pair in adjacent
                or pair in seen_pairs
            ):
                continue
            seen_pairs.add(pair)
            if float(contact[8]) <= 0.0:
                self_collisions += 1
        maximum_arm_error = max(
            abs(actual - command)
            for actual, command in zip(actual_arm, arm_command)
        )
        values = (
            *arm_command,
            *actual_arm,
            *fingers_target,
            *actual_fingers,
            *tool_position,
            *tool_quaternion,
            *target_position,
            *target_quaternion,
            target_lift,
            *relative,
            relative_drift,
            maximum_arm_error,
            left_force,
            right_force,
        )
        all_finite = all_finite and all(
            math.isfinite(float(value)) for value in values
        )
        row = LiftTraceRow(
            step=step,
            phase=phase,
            commanded_arm_positions=arm_command,
            actual_arm_positions=actual_arm,
            commanded_finger_positions=(
                fingers_target[0],
                fingers_target[1],
            ),
            actual_finger_positions=(
                actual_fingers[0],
                actual_fingers[1],
            ),
            actual_tool_position=tool_position,
            actual_tool_quaternion_xyzw=tool_quaternion,
            target_position=target_position,
            target_quaternion_xyzw=target_quaternion,
            target_lift_m=target_lift,
            tool_relative_to_target=tuple(float(value) for value in relative),
            relative_drift_m=relative_drift,
            maximum_arm_joint_error_rad=maximum_arm_error,
            left_finger_contact=left_force > 0.0,
            right_finger_contact=right_force > 0.0,
            bilateral_contact=left_force > 0.0 and right_force > 0.0,
            left_normal_force=left_force,
            right_normal_force=right_force,
            target_table_contact=target_table_contact,
            prohibited_target_contact_count=prohibited_target,
            environment_collision_count=environment_collisions,
            self_collision_count=self_collisions,
        )
        trace.append(row)
        return row

    for fraction in np.linspace(
        1.0 / config.lift_steps,
        1.0,
        config.lift_steps,
    ):
        command_array = start_arm + fraction * (
            arm_target_array - start_arm
        )
        command_and_sample(
            "lift",
            tuple(float(value) for value in command_array),
        )
    lift_final_row = trace[-1]
    lift_endpoint_arm_error = max(
        abs(actual - target)
        for actual, target in zip(
            lift_final_row.actual_arm_positions,
            arm_target,
        )
    )
    lift_reached = (
        lift_endpoint_arm_error <= config.arm_joint_tolerance_rad
    )
    lift_settle_steps = 0
    while not lift_reached and lift_settle_steps < config.settle_steps:
        lift_final_row = command_and_sample("lift", arm_target)
        lift_settle_steps += 1
        lift_endpoint_arm_error = max(
            abs(actual - target)
            for actual, target in zip(
                lift_final_row.actual_arm_positions,
                arm_target,
            )
        )
        lift_reached = (
            lift_endpoint_arm_error <= config.arm_joint_tolerance_rad
        )
    if lift_complete_callback is not None:
        lift_complete_callback()
    for _ in range(config.hold_steps):
        command_and_sample("lift_hold", arm_target)

    final_row = trace[-1]
    endpoint_position_error = float(
        np.linalg.norm(
            np.asarray(final_row.actual_tool_position)
            - target_tool_position
        )
    )
    endpoint_orientation_error = _orientation_error_degrees(
        final_row.actual_tool_quaternion_xyzw,
        target_tool_quaternion,
    )
    hold_rows = trace[-config.hold_steps :]
    minimum_hold_lift = min(row.target_lift_m for row in hold_rows)
    maximum_hold_drift = max(row.relative_drift_m for row in hold_rows)
    total_table_contacts = sum(
        row.target_table_contact for row in trace
    )
    hold_table_contacts = sum(
        row.target_table_contact for row in hold_rows
    )
    trailing_bilateral = 0
    for row in reversed(trace):
        if not row.bilateral_contact:
            break
        trailing_bilateral += 1
    left_contacted = any(row.left_finger_contact for row in trace)
    right_contacted = any(row.right_finger_contact for row in trace)
    object_lift_gate = all(
        row.target_lift_m >= config.minimum_object_lift_m
        for row in hold_rows
    )
    table_release_gate = hold_table_contacts == 0
    relative_stability_gate = (
        maximum_hold_drift <= config.maximum_hold_relative_drift_m
    )
    lift_hold_gate = (
        len(hold_rows) == config.hold_steps
        and object_lift_gate
        and table_release_gate
        and relative_stability_gate
        and trailing_bilateral
        >= config.minimum_trailing_bilateral_contact_steps
        and left_contacted
        and right_contacted
    )
    maximum_arm_error = max(
        row.maximum_arm_joint_error_rad for row in trace
    )
    prohibited_target = sum(
        row.prohibited_target_contact_count for row in trace
    )
    environment_collisions = sum(
        row.environment_collision_count for row in trace
    )
    self_collisions = sum(row.self_collision_count for row in trace)
    maximum_left_force = max(row.left_normal_force for row in trace)
    maximum_right_force = max(row.right_normal_force for row in trace)
    failures = []
    if not lift_reached:
        failures.append("lift_arm_target_not_reached")
    if endpoint_position_error > 0.005:
        failures.append("lift_endpoint_position_failed")
    if endpoint_orientation_error > 5.0:
        failures.append("lift_endpoint_orientation_failed")
    if not object_lift_gate:
        failures.append("object_lift_height_failed")
    if not table_release_gate:
        failures.append("target_table_release_failed")
    if not relative_stability_gate:
        failures.append("hold_relative_stability_failed")
    if not lift_hold_gate:
        failures.append("lift_hold_failed")
    if prohibited_target:
        failures.append("prohibited_target_contact")
    if environment_collisions:
        failures.append("environment_collision")
    if self_collisions:
        failures.append("self_collision")
    if not all_finite:
        failures.append("non_finite_state")
    return LiftResult(
        trace=tuple(trace),
        contact_events=tuple(events),
        lift_reached=lift_reached,
        lift_settle_steps=lift_settle_steps,
        lift_endpoint_arm_error_rad=lift_endpoint_arm_error,
        endpoint_position_error_m=endpoint_position_error,
        endpoint_orientation_error_degrees=endpoint_orientation_error,
        minimum_hold_object_lift_m=minimum_hold_lift,
        final_object_lift_m=final_row.target_lift_m,
        maximum_hold_relative_drift_m=maximum_hold_drift,
        total_target_table_contact_count=total_table_contacts,
        hold_target_table_contact_count=hold_table_contacts,
        trailing_bilateral_contact_steps=trailing_bilateral,
        left_finger_contacted=left_contacted,
        right_finger_contacted=right_contacted,
        maximum_left_normal_force=maximum_left_force,
        maximum_right_normal_force=maximum_right_force,
        maximum_arm_joint_error_rad=maximum_arm_error,
        prohibited_target_contact_count=prohibited_target,
        environment_collision_count=environment_collisions,
        self_collision_count=self_collisions,
        all_states_finite=all_finite,
        object_lift_gate_passed=object_lift_gate,
        table_release_gate_passed=table_release_gate,
        relative_stability_gate_passed=relative_stability_gate,
        lift_hold_gate_passed=lift_hold_gate,
        gate_passed=not failures,
        failure_reason=";".join(failures),
    )
