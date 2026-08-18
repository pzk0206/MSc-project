import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_rescore_cli_preserves_source_and_writes_auditable_metrics(
    tmp_path: Path,
) -> None:
    """Catch destructive rescoring or a return to max-IoU-only success."""

    dataset_root = tmp_path / "cornell"
    sample_dir = dataset_root / "02"
    sample_dir.mkdir(parents=True)
    (sample_dir / "pcd0232cpos.txt").write_text(
        "268 282.881\n"
        "271 311\n"
        "291 309\n"
        "288 280.881\n"
        "250 284.68\n"
        "254 311\n"
        "275 308\n"
        "271 281.68\n"
        "238 283.331\n"
        "237 306\n"
        "261 307\n"
        "262 284.331\n",
        encoding="utf-8",
    )
    source = tmp_path / "legacy_predictions.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "object_directory",
                "success",
                "best_iou",
                "best_angle_error_degrees",
                "matched_gt_index",
                "failed_to_predict",
                "pred_center_x",
                "pred_center_y",
                "pred_width",
                "pred_height",
                "pred_angle_degrees",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "pcd0232",
                "object_directory": "02",
                "success": 0,
                "best_iou": 0.4588975759,
                "best_angle_error_degrees": 87.6140559696,
                "matched_gt_index": 2,
                "failed_to_predict": 0,
                "pred_center_x": 255.0,
                "pred_center_y": 292.5,
                "pred_width": 31.05,
                "pred_height": 15.0,
                "pred_angle_degrees": -90.0,
            }
        )
    source_before = source.read_bytes()
    source_sha256 = hashlib.sha256(source_before).hexdigest()
    output_dir = tmp_path / "rescored"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.shared.rescore_cornell_predictions",
            "--source",
            f"baseline={source}",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--paired-comparison",
            "self=baseline,baseline",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == source_before

    with (output_dir / "baseline" / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["success"] == "1"
    assert rows[0]["matched_gt_index"] == "2"
    assert rows[0]["successful_matched_gt_index"] == "1"
    assert float(rows[0]["successful_match_iou"]) > 0.25
    assert float(rows[0]["successful_match_angle_error_degrees"]) < 30.0

    summary = json.loads(
        (output_dir / "baseline" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["evaluation_protocol"] == "cornell_rectangle_any_gt_v2"
    assert summary["source_predictions_sha256"] == source_sha256
    assert summary["sample_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["prediction_coverage_rate"] == pytest.approx(1.0)
    assert summary["success_count"] == 1
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["mean_best_angle_error_degrees_on_predictions"] == (
        pytest.approx(87.6140559696)
    )

    combined = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert combined["evaluation_protocol"] == "cornell_rectangle_any_gt_v2"
    assert combined["sources"]["baseline"]["success_count"] == 1
    assert combined["paired_comparisons"]["self"] == {
        "left_label": "baseline",
        "right_label": "baseline",
        "sample_count": 1,
        "both_success_count": 1,
        "both_failure_count": 0,
        "left_only_success_count": 0,
        "right_only_success_count": 0,
        "success_rate_delta_right_minus_left": pytest.approx(0.0),
        "mcnemar_exact_two_sided_p_value": pytest.approx(1.0),
    }
