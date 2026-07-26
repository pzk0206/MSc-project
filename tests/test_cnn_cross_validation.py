from pathlib import Path

import pytest

from src.vlm.run_cnn_cross_validation import (
    build_architecture_comparison,
    build_cross_validation_summary,
    build_fold_paths,
    validate_complete_fold_records,
)


def test_fold_paths_isolate_every_artifact(tmp_path: Path) -> None:
    paths = build_fold_paths(tmp_path, "multi_head", 3)
    expected = tmp_path / "multi_head" / "fold_3"

    assert paths.directory == expected
    assert paths.model == expected / "model.pt"
    assert paths.history == expected / "training_history.json"
    assert paths.predictions == expected / "predictions.csv"
    assert paths.summary == expected / "summary.json"


def _row(sample_id: str, success: int, iou: float, angle: float) -> dict:
    return {
        "sample_id": sample_id,
        "success": success,
        "best_iou": iou,
        "best_angle_error_degrees": angle,
    }


def test_cross_validation_summary_reports_pooled_and_fold_metrics() -> None:
    fold_records = []
    combined = []
    for fold in range(5):
        rows = [
            _row(f"sample_{fold}_0", 1, 0.5, 10.0),
            _row(f"sample_{fold}_1", 0, 0.2, 40.0),
        ]
        combined.extend(rows)
        fold_records.append(
            {
                "fold": fold,
                "best_val_loss": 0.1 + fold / 100,
                "rows": rows,
            }
        )

    summary = build_cross_validation_summary(
        fold_records,
        combined,
        architecture="single",
        seed=42,
        manifest_hash="abc123",
    )

    assert summary["protocol"] == "cornell_image_wise_5_fold"
    assert summary["pooled"]["sample_count"] == 10
    assert summary["pooled"]["success_rate"] == pytest.approx(0.5)
    assert summary["folds"]["success_rate_mean"] == pytest.approx(0.5)
    assert len(summary["per_fold"]) == 5
    assert summary["manifest_sha256"] == "abc123"


def test_cross_validation_audit_rejects_missing_fold() -> None:
    records = [
        {"fold": fold, "rows": [_row(f"s{fold}", 1, 0.5, 10.0)]}
        for fold in range(4)
    ]
    combined = [row for record in records for row in record["rows"]]

    with pytest.raises(ValueError, match="folds 0 through 4"):
        validate_complete_fold_records(
            records,
            combined,
            expected_sample_ids={f"s{fold}" for fold in range(5)},
        )


def test_architecture_comparison_requires_the_same_manifest() -> None:
    single = {
        "architecture": "single",
        "manifest_sha256": "same",
        "pooled": {
            "success_rate": 0.70,
            "mean_iou": 0.40,
            "mean_angle": 18.0,
        },
        "per_fold": [
            {
                "fold": fold,
                "success_rate": 0.70,
                "mean_iou": 0.40,
                "mean_angle": 18.0,
            }
            for fold in range(5)
        ],
    }
    multi = {
        "architecture": "multi_head",
        "manifest_sha256": "same",
        "pooled": {
            "success_rate": 0.72,
            "mean_iou": 0.44,
            "mean_angle": 17.0,
        },
        "per_fold": [
            {
                "fold": fold,
                "success_rate": 0.72,
                "mean_iou": 0.44,
                "mean_angle": 17.0,
            }
            for fold in range(5)
        ],
    }

    comparison = build_architecture_comparison(single, multi)

    assert comparison["pooled_delta_multi_minus_single"] == {
        "success_rate": pytest.approx(0.02),
        "mean_iou": pytest.approx(0.04),
        "mean_angle": pytest.approx(-1.0),
    }
    assert len(comparison["paired_fold_deltas"]) == 5

    multi["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        build_architecture_comparison(single, multi)
