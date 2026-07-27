"""Pure evaluation helpers for the fixed multi-object PyBullet study.

Segmentation masks in this module are evaluation truth only. They are not
inputs to object localization or grasp prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


Box = tuple[int, int, int, int]
_BODY_ID_MASK = (1 << 24) - 1
_MAIN_TARGETS = {"duck", "cube", "sphere"}


@dataclass(frozen=True)
class TargetSelectionEvaluation:
    """Result of matching one predicted box against visible scene entities."""

    requested_target: str
    requested_target_iou: float
    best_matching_target: str | None
    best_iou: float
    correct_target: bool
    iou_threshold: float
    failure_reason: str
    entity_ious: dict[str, float]


def segmentation_mask_for_body(
    segmentation: np.ndarray,
    body_id: int,
) -> np.ndarray:
    """Return visible pixels belonging to one PyBullet body."""

    array = np.asarray(segmentation)
    if array.ndim != 2:
        raise ValueError("segmentation must be a two-dimensional array")
    return (array >= 0) & ((array & _BODY_ID_MASK) == body_id)


def mask_to_box(mask: np.ndarray) -> Box:
    """Convert a non-empty 2-D mask to an inclusive pixel box."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    ys, xs = np.nonzero(array)
    if xs.size == 0:
        raise ValueError("mask has no visible pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _validated_box(box: tuple[int, int, int, int]) -> tuple[float, ...]:
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    coordinates = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("box coordinates must be finite")
    x1, y1, x2, y2 = coordinates
    if x2 < x1 or y2 < y1:
        raise ValueError("box must have non-negative width and height")
    return coordinates


def box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Calculate IoU for inclusive pixel-coordinate boxes."""

    ax1, ay1, ax2, ay2 = _validated_box(box_a)
    bx1, by1, bx2, by2 = _validated_box(box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1) + 1.0)
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1) + 1.0)
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    return float(intersection / (area_a + area_b - intersection))


def evaluate_target_selection(
    predicted_box: Box | None,
    requested_target: str,
    entity_boxes: Mapping[str, Box],
    iou_threshold: float = 0.25,
) -> TargetSelectionEvaluation:
    """Evaluate whether a detection uniquely selects the requested entity."""

    if requested_target not in entity_boxes:
        raise ValueError(f"unknown requested target: {requested_target}")
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and within [0, 1]")

    if predicted_box is None:
        return TargetSelectionEvaluation(
            requested_target=requested_target,
            requested_target_iou=0.0,
            best_matching_target=None,
            best_iou=0.0,
            correct_target=False,
            iou_threshold=iou_threshold,
            failure_reason="no_detection",
            entity_ious={name: 0.0 for name in entity_boxes},
        )

    entity_ious = {
        name: box_iou(predicted_box, box)
        for name, box in entity_boxes.items()
    }
    best_iou = max(entity_ious.values(), default=0.0)
    best_names = [
        name
        for name, iou in entity_ious.items()
        if np.isclose(iou, best_iou)
    ]
    ambiguous = len(best_names) != 1
    best_target = None if ambiguous else best_names[0]
    requested_iou = entity_ious[requested_target]

    if ambiguous:
        failure_reason = "ambiguous_match"
    elif best_target != requested_target:
        failure_reason = "wrong_target"
    elif requested_iou < iou_threshold:
        failure_reason = "below_iou_threshold"
    else:
        failure_reason = ""

    return TargetSelectionEvaluation(
        requested_target=requested_target,
        requested_target_iou=requested_iou,
        best_matching_target=best_target,
        best_iou=best_iou,
        correct_target=failure_reason == "",
        iou_threshold=iou_threshold,
        failure_reason=failure_reason,
        entity_ious=entity_ious,
    )


def grasp_center_inside_mask(
    grasp: Mapping[str, float],
    target_mask: np.ndarray,
) -> bool:
    """Return whether the rounded grasp centre falls inside the target mask."""

    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("target_mask must be a two-dimensional array")
    center_x = float(grasp["center_x"])
    center_y = float(grasp["center_y"])
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        return False
    x = int(round(center_x))
    y = int(round(center_y))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def summarize_target_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize the fixed three-target protocol without scoring diagnostic."""

    main_rows = [row for row in rows if row["result_role"] == "main"]
    requested_targets = [str(row["requested_target"]) for row in main_rows]
    if len(main_rows) != 3 or set(requested_targets) != _MAIN_TARGETS:
        raise ValueError(
            "main requested targets must be exactly duck, cube, and sphere"
        )

    diagnostic_rows = [
        row for row in rows if row["result_role"] == "diagnostic"
    ]
    generic_diagnostic = dict(diagnostic_rows[0]) if diagnostic_rows else {}
    correct_count = sum(bool(row["correct_target"]) for row in main_rows)
    mean_iou = float(
        np.mean(
            [float(row["requested_target_iou"]) for row in main_rows],
        )
    )
    return {
        "protocol": "fixed_three_object_prompt_selection_pilot",
        "main_target_count": 3,
        "correct_target_count": correct_count,
        "target_selection_rate": correct_count / 3,
        "mean_requested_target_iou": mean_iou,
        "generic_diagnostic": generic_diagnostic,
        "physical_grasp_executed": False,
    }
