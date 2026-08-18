from pathlib import Path

import pytest

from src.vlm.analyze_failures import VLM_PREDICTIONS, classify_failure


def test_failure_analysis_reads_metric_v2_geometry_predictions() -> None:
    assert VLM_PREDICTIONS == Path(
        "data/processed/shared/cornell_metric_v2/vlm_geometry/predictions.csv"
    ).resolve()


def test_failure_classification_matches_existential_metric_semantics() -> None:
    no_iou_match = {
        "success": "0",
        "best_iou": "0.24",
        "best_angle_error_degrees": "5.0",
    }
    angle_after_iou = {
        "success": "0",
        "best_iou": "0.40",
        "best_angle_error_degrees": "45.0",
    }

    assert classify_failure(no_iou_match) == "no_iou_qualified_match"
    assert classify_failure(angle_after_iou) == "angle_after_iou_match"


def test_failure_classification_rejects_success_rows() -> None:
    row = {
        "success": "1",
        "best_iou": "0.50",
        "best_angle_error_degrees": "45.0",
    }

    with pytest.raises(ValueError, match="successful row"):
        classify_failure(row)
