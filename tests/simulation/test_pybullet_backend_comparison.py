import numpy as np
import pytest

from src.simulation.pybullet.backend_comparison import (
    evaluate_backend_grasp,
    summarize_backend_rows,
)


def test_valid_grasp_passes_all_geometry_audits() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True
    grasp = {
        "center_x": 15.0,
        "center_y": 10.0,
        "width": 10.0,
        "height": 4.0,
        "angle_degrees": 0.0,
    }

    result = evaluate_backend_grasp(grasp, mask, 30, 20)

    assert result.parameters_finite
    assert result.positive_size
    assert result.center_inside_target_mask
    assert result.box_inside_image
    assert result.failure_reason == ""


def test_grasp_audit_reports_center_and_rotated_box_failures() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True

    outside_center = evaluate_backend_grasp(
        {
            "center_x": 3.0,
            "center_y": 3.0,
            "width": 4.0,
            "height": 2.0,
            "angle_degrees": 0.0,
        },
        mask,
        30,
        20,
    )
    crossing_edge = evaluate_backend_grasp(
        {
            "center_x": 15.0,
            "center_y": 10.0,
            "width": 32.0,
            "height": 4.0,
            "angle_degrees": 30.0,
        },
        mask,
        30,
        20,
    )

    assert not outside_center.center_inside_target_mask
    assert outside_center.failure_reason == "center_outside_target_mask"
    assert crossing_edge.center_inside_target_mask
    assert not crossing_edge.box_inside_image
    assert crossing_edge.failure_reason == "box_outside_image"


@pytest.mark.parametrize(
    ("grasp", "reason"),
    [
        (
            {
                "center_x": np.nan,
                "center_y": 10.0,
                "width": 4.0,
                "height": 2.0,
                "angle_degrees": 0.0,
            },
            "non_finite_parameters",
        ),
        (
            {
                "center_x": 15.0,
                "center_y": 10.0,
                "width": 0.0,
                "height": 2.0,
                "angle_degrees": 0.0,
            },
            "non_positive_size",
        ),
    ],
)
def test_invalid_grasp_parameters_are_recorded_without_drawing(
    grasp: dict[str, float],
    reason: str,
) -> None:
    mask = np.ones((20, 30), dtype=bool)

    result = evaluate_backend_grasp(grasp, mask, 30, 20)

    assert result.failure_reason == reason
    assert not result.box_inside_image


def test_grasp_audit_rejects_mask_image_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="target mask shape must match"):
        evaluate_backend_grasp(
            {
                "center_x": 2.0,
                "center_y": 2.0,
                "width": 2.0,
                "height": 1.0,
                "angle_degrees": 0.0,
            },
            np.ones((5, 5), dtype=bool),
            6,
            5,
        )


def test_backend_summary_counts_diagnostics_without_ranking() -> None:
    rows = []
    for target in ("duck", "cube", "sphere"):
        for backend in ("geometry", "single", "multi_head"):
            rows.append(
                {
                    "target": target,
                    "backend": backend,
                    "parameters_finite": backend != "single",
                    "center_inside_target_mask": target != "cube",
                    "box_inside_image": True,
                }
            )

    summary = summarize_backend_rows(rows)

    assert summary["backend_result_count"] == 9
    assert summary["counts_by_backend"]["geometry"] == {
        "finite_output_count": 3,
        "center_inside_target_mask_count": 2,
        "box_inside_image_count": 3,
    }
    assert summary["counts_by_backend"]["single"] == {
        "finite_output_count": 0,
        "center_inside_target_mask_count": 2,
        "box_inside_image_count": 3,
    }
    assert summary["performance_ranking_computed"] is False
    assert "best_backend" not in summary
    assert "winner" not in summary


def test_backend_summary_rejects_missing_or_wrong_order_rows() -> None:
    rows = [
        {
            "target": target,
            "backend": backend,
            "parameters_finite": True,
            "center_inside_target_mask": True,
            "box_inside_image": True,
        }
        for target in ("duck", "cube", "sphere")
        for backend in ("geometry", "single", "multi_head")
    ]

    with pytest.raises(ValueError, match="exact target/backend order"):
        summarize_backend_rows(rows[:-1])
    with pytest.raises(ValueError, match="exact target/backend order"):
        summarize_backend_rows(list(reversed(rows)))
