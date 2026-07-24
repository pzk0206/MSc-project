import importlib
import importlib.util

import pytest


def _audit_module():
    module_name = "src.shared.analyze_cornell_splits"
    assert importlib.util.find_spec(module_name) is not None, (
        f"{module_name} must exist"
    )
    return importlib.import_module(module_name)


def test_assign_split_uses_fixed_directory_groups() -> None:
    analyze_cornell_splits = _audit_module()
    assert analyze_cornell_splits.assign_split("01") == "train"
    assert analyze_cornell_splits.assign_split("06") == "train"
    assert analyze_cornell_splits.assign_split("07") == "val"
    assert analyze_cornell_splits.assign_split("08") == "val"
    assert analyze_cornell_splits.assign_split("09") == "test"
    assert analyze_cornell_splits.assign_split("10") == "test"


def test_assign_split_rejects_unknown_directory() -> None:
    analyze_cornell_splits = _audit_module()
    with pytest.raises(ValueError, match="unknown Cornell directory"):
        analyze_cornell_splits.assign_split("11")


def test_summarize_predictions_reports_counts_and_metrics() -> None:
    analyze_cornell_splits = _audit_module()
    rows = [
        {
            "object_directory": "09",
            "success": "1",
            "best_iou": "0.4",
            "best_angle_error_degrees": "10",
        },
        {
            "object_directory": "10",
            "success": "0",
            "best_iou": "0.2",
            "best_angle_error_degrees": "40",
        },
    ]

    result = analyze_cornell_splits.summarize_predictions(rows)

    assert result["test"]["count"] == 2
    assert result["test"]["success_rate"] == pytest.approx(0.5)
    assert result["test"]["mean_best_iou"] == pytest.approx(0.3)
    assert result["test"]["mean_angle_error_degrees"] == pytest.approx(25.0)


def test_representative_quota_balances_split_panels() -> None:
    analyze_cornell_splits = _audit_module()

    assert analyze_cornell_splits.representative_quota("01") == 2
    assert analyze_cornell_splits.representative_quota("06") == 2
    assert analyze_cornell_splits.representative_quota("07") == 6
    assert analyze_cornell_splits.representative_quota("10") == 6
