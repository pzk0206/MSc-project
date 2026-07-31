import numpy as np
import pytest

from src.simulation.pybullet.target_selection import (
    box_iou,
    evaluate_target_selection,
    grasp_center_inside_mask,
    mask_to_box,
    segmentation_mask_for_body,
    summarize_target_rows,
)


def test_segmentation_mask_box_and_inclusive_iou() -> None:
    segmentation = np.array(
        [
            [-1, 3, 3, 7],
            [-1, 3, 3, 7],
        ],
        dtype=np.int32,
    )

    mask = segmentation_mask_for_body(segmentation, 3)

    assert mask.tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]
    assert mask_to_box(mask) == (1, 0, 2, 1)
    assert box_iou((0, 0, 1, 1), (1, 0, 2, 1)) == pytest.approx(1 / 3)


def test_segmentation_background_does_not_alias_max_body_id() -> None:
    segmentation = np.full((2, 3), -1, dtype=np.int32)

    mask = segmentation_mask_for_body(segmentation, (1 << 24) - 1)

    assert not np.any(mask)
    with pytest.raises(ValueError, match="mask has no visible pixels"):
        mask_to_box(mask)


@pytest.fixture
def entity_boxes() -> dict[str, tuple[int, int, int, int]]:
    return {
        "duck": (0, 0, 9, 9),
        "cube": (20, 0, 29, 9),
        "sphere": (40, 0, 49, 9),
        "robot": (60, 0, 79, 19),
    }


def test_target_selection_requires_requested_unique_best_and_threshold(
    entity_boxes: dict[str, tuple[int, int, int, int]],
) -> None:
    correct = evaluate_target_selection(
        (1, 1, 8, 8),
        "duck",
        entity_boxes,
        iou_threshold=0.25,
    )
    wrong = evaluate_target_selection(
        (1, 1, 8, 8),
        "cube",
        entity_boxes,
        iou_threshold=0.25,
    )
    below_threshold = evaluate_target_selection(
        (0, 0, 1, 1),
        "duck",
        entity_boxes,
        iou_threshold=0.25,
    )

    assert correct.correct_target
    assert correct.best_matching_target == "duck"
    assert correct.requested_target_iou == pytest.approx(0.64)
    assert not wrong.correct_target
    assert wrong.best_matching_target == "duck"
    assert wrong.failure_reason == "wrong_target"
    assert not below_threshold.correct_target
    assert below_threshold.failure_reason == "below_iou_threshold"


def test_target_selection_reports_ambiguous_and_no_detection() -> None:
    touching_entities = {
        "duck": (0, 0, 4, 4),
        "cube": (5, 0, 9, 4),
    }

    ambiguous = evaluate_target_selection(
        (0, 0, 9, 4),
        "duck",
        touching_entities,
    )
    missing = evaluate_target_selection(
        None,
        "duck",
        touching_entities,
    )

    assert not ambiguous.correct_target
    assert ambiguous.best_matching_target is None
    assert ambiguous.failure_reason == "ambiguous_match"
    assert not missing.correct_target
    assert missing.best_matching_target is None
    assert missing.failure_reason == "no_detection"


def test_grasp_center_must_land_inside_target_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 3:9] = True

    assert grasp_center_inside_mask(
        {"center_x": 4.0, "center_y": 3.0},
        mask,
    )
    assert not grasp_center_inside_mask(
        {"center_x": 1.0, "center_y": 1.0},
        mask,
    )
    assert not grasp_center_inside_mask(
        {"center_x": 30.0, "center_y": 30.0},
        mask,
    )


def test_summary_uses_only_three_main_targets() -> None:
    rows = [
        {
            "result_role": "main",
            "requested_target": "duck",
            "correct_target": True,
            "requested_target_iou": 0.8,
        },
        {
            "result_role": "main",
            "requested_target": "cube",
            "correct_target": False,
            "requested_target_iou": 0.1,
        },
        {
            "result_role": "main",
            "requested_target": "sphere",
            "correct_target": True,
            "requested_target_iou": 0.7,
        },
        {
            "result_role": "diagnostic",
            "requested_target": "generic",
            "correct_target": False,
            "requested_target_iou": 0.0,
            "best_matching_target": "robot",
            "failure_reason": "wrong_target",
        },
    ]

    summary = summarize_target_rows(rows)

    assert summary["main_target_count"] == 3
    assert summary["correct_target_count"] == 2
    assert summary["target_selection_rate"] == pytest.approx(2 / 3)
    assert summary["mean_requested_target_iou"] == pytest.approx(1.6 / 3)
    assert summary["generic_diagnostic"]["best_matching_target"] == "robot"
    assert not summary["physical_grasp_executed"]


def test_summary_rejects_incomplete_main_target_set() -> None:
    with pytest.raises(ValueError, match="main requested targets must be exactly"):
        summarize_target_rows(
            [
                {
                    "result_role": "main",
                    "requested_target": "duck",
                    "correct_target": True,
                    "requested_target_iou": 1.0,
                }
            ]
        )
