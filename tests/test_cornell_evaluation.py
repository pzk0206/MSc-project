import pytest

from src.baseline_cv.run_cv_baseline import (
    angle_difference_degrees,
    evaluate_prediction,
    rotated_rect_iou,
)


def _rect(
    *,
    width: float,
    height: float,
    angle_degrees: float,
) -> dict[str, float]:
    return {
        "center_x": 0.0,
        "center_y": 0.0,
        "width": width,
        "height": height,
        "angle_degrees": angle_degrees,
    }


def test_success_uses_one_ground_truth_that_passes_both_thresholds() -> None:
    """Catch selecting max IoU before applying the Cornell angle gate."""

    prediction = _rect(width=100.0, height=40.0, angle_degrees=0.0)
    maximum_iou_but_bad_angle = _rect(
        width=100.0,
        height=40.0,
        angle_degrees=45.0,
    )
    lower_iou_and_good_angle = _rect(
        width=50.0,
        height=25.0,
        angle_degrees=0.0,
    )

    assert rotated_rect_iou(
        prediction,
        maximum_iou_but_bad_angle,
    ) > rotated_rect_iou(prediction, lower_iou_and_good_angle)
    assert angle_difference_degrees(0.0, 45.0) > 30.0
    assert rotated_rect_iou(prediction, lower_iou_and_good_angle) > 0.25

    result = evaluate_prediction(
        prediction,
        [maximum_iou_but_bad_angle, lower_iou_and_good_angle],
    )

    assert result["success"] is True
    assert result["matched_gt_index"] == 0
    assert result["best_iou"] == pytest.approx(0.3943942525)
    assert result["best_angle_error_degrees"] == pytest.approx(45.0)
    assert result["successful_matched_gt_index"] == 1
    assert result["successful_match_iou"] == pytest.approx(0.3125)
    assert result["successful_match_angle_error_degrees"] == pytest.approx(0.0)


def test_failed_prediction_has_no_successful_match_witness() -> None:
    result = evaluate_prediction(
        None,
        [_rect(width=50.0, height=25.0, angle_degrees=0.0)],
    )

    assert result["success"] is False
    assert result["successful_matched_gt_index"] is None
    assert result["successful_match_iou"] is None
    assert result["successful_match_angle_error_degrees"] is None
