"""Shared Cornell rectangle evaluation with an auditable success witness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import cv2

from src.shared.grasp_geometry import normalize_angle_radians


IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEGREES = 30.0
EVALUATION_PROTOCOL = "cornell_rectangle_any_gt_v2"


def rotated_rect_iou(
    rect_a: Mapping[str, float],
    rect_b: Mapping[str, float],
) -> float:
    """Return the intersection-over-union of two rotated rectangles."""

    cv_rect_a = (
        (float(rect_a["center_x"]), float(rect_a["center_y"])),
        (float(rect_a["width"]), float(rect_a["height"])),
        float(rect_a["angle_degrees"]),
    )
    cv_rect_b = (
        (float(rect_b["center_x"]), float(rect_b["center_y"])),
        (float(rect_b["width"]), float(rect_b["height"])),
        float(rect_b["angle_degrees"]),
    )

    area_a = float(rect_a["width"]) * float(rect_a["height"])
    area_b = float(rect_b["width"]) * float(rect_b["height"])
    intersection_type, intersection_points = cv2.rotatedRectangleIntersection(
        cv_rect_a,
        cv_rect_b,
    )
    if intersection_type == cv2.INTERSECT_NONE or intersection_points is None:
        intersection_area = 0.0
    else:
        intersection_area = float(cv2.contourArea(intersection_points))
    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    """Return the gripper-symmetry-aware absolute angle difference."""

    difference = normalize_angle_radians(math.radians(angle_a - angle_b))
    return abs(math.degrees(difference))


def _is_better_match(
    *,
    iou: float,
    angle_error: float,
    current_iou: float,
    current_angle_error: float,
) -> bool:
    return iou > current_iou or (
        math.isclose(iou, current_iou)
        and angle_error < current_angle_error
    )


def evaluate_prediction(
    prediction: Mapping[str, float] | None,
    positive_ground_truths: Sequence[Mapping[str, float]],
) -> dict[str, bool | float | int | None]:
    """Evaluate one prediction against every Cornell positive rectangle.

    ``best_*`` retains the historical maximum-IoU diagnostic.  Success is
    independent of that diagnostic and requires one *same* ground-truth
    rectangle to satisfy both thresholds.  The successful-match fields record
    the witness used for that existential decision.
    """

    result: dict[str, bool | float | int | None] = {
        "success": False,
        "best_iou": 0.0,
        "best_angle_error_degrees": 180.0,
        "matched_gt_index": -1,
        "successful_match_iou": None,
        "successful_match_angle_error_degrees": None,
        "successful_matched_gt_index": None,
    }
    if prediction is None:
        return result

    for gt_index, ground_truth in enumerate(positive_ground_truths):
        iou = float(rotated_rect_iou(prediction, ground_truth))
        angle_error = float(
            angle_difference_degrees(
                float(prediction["angle_degrees"]),
                float(ground_truth["angle_degrees"]),
            )
        )

        if _is_better_match(
            iou=iou,
            angle_error=angle_error,
            current_iou=float(result["best_iou"]),
            current_angle_error=float(result["best_angle_error_degrees"]),
        ):
            result["best_iou"] = iou
            result["best_angle_error_degrees"] = angle_error
            result["matched_gt_index"] = gt_index

        if iou < IOU_THRESHOLD or angle_error > ANGLE_THRESHOLD_DEGREES:
            continue
        current_success_iou = result["successful_match_iou"]
        current_success_angle = result[
            "successful_match_angle_error_degrees"
        ]
        if current_success_iou is None or _is_better_match(
            iou=iou,
            angle_error=angle_error,
            current_iou=float(current_success_iou),
            current_angle_error=float(current_success_angle),
        ):
            result["successful_match_iou"] = iou
            result["successful_match_angle_error_degrees"] = angle_error
            result["successful_matched_gt_index"] = gt_index

    result["success"] = result["successful_matched_gt_index"] is not None
    return result
