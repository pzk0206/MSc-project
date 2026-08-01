"""Deterministic Panda gripper closing and target-contact auditing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pybullet as p

from src.simulation.pybullet.kinematic_audit import PandaModelInfo


@dataclass(frozen=True)
class GripperCloseConfig:
    """Timing and gate thresholds for one bilateral close attempt."""

    close_steps: int = 240
    hold_steps: int = 120
    minimum_bilateral_hold_steps: int = 60
    closed_target_m: float = 0.0
    arm_joint_tolerance_rad: float = 0.01

    def __post_init__(self) -> None:
        if self.close_steps <= 0:
            raise ValueError("close_steps must be positive")
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        if self.minimum_bilateral_hold_steps < 0:
            raise ValueError(
                "minimum_bilateral_hold_steps must be non-negative"
            )
        if self.minimum_bilateral_hold_steps > self.hold_steps:
            raise ValueError(
                "minimum bilateral hold cannot exceed hold_steps"
            )
        if (
            not math.isfinite(self.closed_target_m)
            or not 0.0 <= self.closed_target_m <= 0.04
        ):
            raise ValueError("closed_target_m must be within [0.0, 0.04]")
        if (
            not math.isfinite(self.arm_joint_tolerance_rad)
            or self.arm_joint_tolerance_rad <= 0.0
        ):
            raise ValueError("arm_joint_tolerance_rad must be positive")


@dataclass(frozen=True)
class ContactEvent:
    """One positive-force finger contact with the target body."""

    step: int
    phase: str
    robot_link: int
    target_body: int
    normal_force: float


@dataclass(frozen=True)
class GripperTraceRow:
    """Measured gripper, arm, tool, target, and contact state."""

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
    maximum_arm_joint_error_rad: float
    left_finger_contact: bool
    right_finger_contact: bool
    bilateral_contact: bool
    left_normal_force: float
    right_normal_force: float
    prohibited_target_contact_count: int
    environment_collision_count: int
    self_collision_count: int


@dataclass(frozen=True)
class GripperCloseResult:
    """Complete close/hold trace and scientific gate result."""

    trace: tuple[GripperTraceRow, ...]
    contact_events: tuple[ContactEvent, ...]
    bilateral_contact_acquired: bool
    first_bilateral_contact_step: int | None
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


def execute_gripper_close(
    *,
    robot_id: int,
    target_body_id: int,
    client_id: int,
    model: PandaModelInfo,
    arm_hold_positions: Sequence[float],
    environment_body_ids: Sequence[int],
    allowed_environment_link_pairs: Sequence[tuple[int, int]] = (),
    config: GripperCloseConfig = GripperCloseConfig(),
    physics: Any = p,
) -> GripperCloseResult:
    """Close until bilateral target contact, then hold and audit it."""

    arm_target = tuple(float(value) for value in arm_hold_positions)
    if len(arm_target) != 7 or not all(
        math.isfinite(value) for value in arm_target
    ):
        raise ValueError("arm_hold_positions must contain seven finite values")
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
    start_fingers = _joint_positions(
        robot_id,
        client_id,
        model.finger_joint_indices,
        physics,
    )
    left_link, right_link = model.finger_joint_indices
    allowed_environment = {
        (int(link), int(body))
        for link, body in allowed_environment_link_pairs
    }
    adjacent = set(model.adjacent_link_pairs)
    trace: list[GripperTraceRow] = []
    events: list[ContactEvent] = []
    first_bilateral_step = None
    frozen_command = (
        float(config.closed_target_m),
        float(config.closed_target_m),
    )
    all_finite = True

    def command_and_sample(
        phase: str,
        finger_command: tuple[float, float],
    ) -> GripperTraceRow:
        nonlocal all_finite
        physics.setJointMotorControlArray(
            robot_id,
            model.arm_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=arm_target,
            forces=arm_forces,
            physicsClientId=client_id,
        )
        physics.setJointMotorControlArray(
            robot_id,
            model.finger_joint_indices,
            physics.POSITION_CONTROL,
            targetPositions=finger_command,
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
            if link in (left_link, right_link) and math.isfinite(force) and force > 0.0:
                events.append(
                    ContactEvent(step, phase, link, target_body_id, force)
                )
                if link == left_link:
                    left_force = max(left_force, force)
                else:
                    right_force = max(right_force, force)
            elif float(contact[8]) <= 0.0:
                prohibited_target += 1
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
            abs(actual - target)
            for actual, target in zip(actual_arm, arm_target)
        )
        left_contact = left_force > 0.0
        right_contact = right_force > 0.0
        values = (
            *arm_target,
            *actual_arm,
            *finger_command,
            *actual_fingers,
            *tool_position,
            *tool_quaternion,
            *target_position,
            *target_quaternion,
            maximum_arm_error,
            left_force,
            right_force,
        )
        row_finite = all(math.isfinite(value) for value in values)
        all_finite = all_finite and row_finite
        row = GripperTraceRow(
            step=step,
            phase=phase,
            commanded_arm_positions=arm_target,
            actual_arm_positions=actual_arm,
            commanded_finger_positions=finger_command,
            actual_finger_positions=(actual_fingers[0], actual_fingers[1]),
            actual_tool_position=tool_position,
            actual_tool_quaternion_xyzw=tool_quaternion,
            target_position=target_position,
            target_quaternion_xyzw=target_quaternion,
            maximum_arm_joint_error_rad=maximum_arm_error,
            left_finger_contact=left_contact,
            right_finger_contact=right_contact,
            bilateral_contact=left_contact and right_contact,
            left_normal_force=left_force,
            right_normal_force=right_force,
            prohibited_target_contact_count=prohibited_target,
            environment_collision_count=environment_collisions,
            self_collision_count=self_collisions,
        )
        trace.append(row)
        return row

    for close_step in range(1, config.close_steps + 1):
        fraction = close_step / config.close_steps
        command = tuple(
            float(start + fraction * (config.closed_target_m - start))
            for start in start_fingers
        )
        row = command_and_sample("close", (command[0], command[1]))
        frozen_command = (command[0], command[1])
        if row.bilateral_contact:
            first_bilateral_step = row.step
            break

    for _ in range(config.hold_steps):
        row = command_and_sample("contact_hold", frozen_command)
        if first_bilateral_step is None and row.bilateral_contact:
            first_bilateral_step = row.step

    trailing_bilateral = 0
    for row in reversed(trace):
        if not row.bilateral_contact:
            break
        trailing_bilateral += 1
    left_contacted = any(row.left_finger_contact for row in trace)
    right_contacted = any(row.right_finger_contact for row in trace)
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
    bilateral_acquired = first_bilateral_step is not None
    failures = []
    if not bilateral_acquired:
        failures.append("bilateral_target_contact_not_acquired")
    if trailing_bilateral < config.minimum_bilateral_hold_steps:
        failures.append("bilateral_contact_hold_failed")
    if not left_contacted or not right_contacted:
        failures.append("individual_finger_contact_failed")
    if maximum_arm_error > config.arm_joint_tolerance_rad:
        failures.append("arm_hold_failed")
    if prohibited_target:
        failures.append("prohibited_target_contact")
    if environment_collisions:
        failures.append("environment_collision")
    if self_collisions:
        failures.append("self_collision")
    if not all_finite:
        failures.append("non_finite_state")
    return GripperCloseResult(
        trace=tuple(trace),
        contact_events=tuple(events),
        bilateral_contact_acquired=bilateral_acquired,
        first_bilateral_contact_step=first_bilateral_step,
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
        gate_passed=not failures,
        failure_reason=";".join(failures),
    )
