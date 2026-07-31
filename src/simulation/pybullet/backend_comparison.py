"""Pure geometry audits for simulated grasp-backend diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from src.simulation.pybullet.target_selection import (
    grasp_center_inside_mask,
)
from src.simulation.pybullet.visualization import grasp_box_points


BACKEND_ORDER = ("geometry", "single", "multi_head")
TARGET_ORDER = ("duck", "cube", "sphere")
EXPECTED_TARGET_BACKENDS = tuple(
    (target, backend)
    for target in TARGET_ORDER
    for backend in BACKEND_ORDER
)


@dataclass(frozen=True)
class BackendGraspEvaluation:
    """Geometry-only checks for one centre-format grasp prediction."""

    parameters_finite: bool
    positive_size: bool
    center_inside_target_mask: bool
    box_inside_image: bool
    failure_reason: str


def evaluate_backend_grasp(
    grasp: Mapping[str, float],
    target_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> BackendGraspEvaluation:
    """Audit one grasp without treating it as physical grasp success."""

    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (image_height, image_width):
        raise ValueError("target mask shape must match image dimensions")

    values = tuple(
        float(grasp[name])
        for name in (
            "center_x",
            "center_y",
            "width",
            "height",
            "angle_degrees",
        )
    )
    parameters_finite = all(math.isfinite(value) for value in values)
    positive_size = (
        parameters_finite and values[2] > 0.0 and values[3] > 0.0
    )
    center_inside = (
        parameters_finite
        and grasp_center_inside_mask(grasp, mask)
    )
    box_inside = False
    if positive_size:
        points = grasp_box_points(*values)
        box_inside = bool(
            np.all(points[:, 0] >= 0.0)
            and np.all(points[:, 0] <= image_width - 1)
            and np.all(points[:, 1] >= 0.0)
            and np.all(points[:, 1] <= image_height - 1)
        )

    if not parameters_finite:
        failure_reason = "non_finite_parameters"
    elif not positive_size:
        failure_reason = "non_positive_size"
    elif not center_inside:
        failure_reason = "center_outside_target_mask"
    elif not box_inside:
        failure_reason = "box_outside_image"
    else:
        failure_reason = ""

    return BackendGraspEvaluation(
        parameters_finite=parameters_finite,
        positive_size=positive_size,
        center_inside_target_mask=center_inside,
        box_inside_image=box_inside,
        failure_reason=failure_reason,
    )


def summarize_backend_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Count diagnostic checks for the exact nine rows without ranking."""

    received_order = tuple(
        (str(row["target"]), str(row["backend"]))
        for row in rows
    )
    if received_order != EXPECTED_TARGET_BACKENDS:
        raise ValueError(
            "backend rows must use the exact target/backend order"
        )

    counts_by_backend = {}
    for backend in BACKEND_ORDER:
        backend_rows = [
            row for row in rows if row["backend"] == backend
        ]
        counts_by_backend[backend] = {
            "finite_output_count": sum(
                bool(row["parameters_finite"])
                for row in backend_rows
            ),
            "center_inside_target_mask_count": sum(
                bool(row["center_inside_target_mask"])
                for row in backend_rows
            ),
            "box_inside_image_count": sum(
                bool(row["box_inside_image"])
                for row in backend_rows
            ),
        }

    return {
        "protocol": "fixed_three_object_three_backend_diagnostic",
        "backend_result_count": 9,
        "counts_by_backend": counts_by_backend,
        "performance_ranking_computed": False,
        "physical_grasp_executed": False,
    }
