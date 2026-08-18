"""Re-score saved Cornell predictions without changing historical inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from src.shared.cornell_dataset import load_grasp_rectangles
from src.shared.cornell_evaluation import (
    ANGLE_THRESHOLD_DEGREES,
    EVALUATION_PROTOCOL,
    IOU_THRESHOLD,
    evaluate_prediction,
)
from src.shared.grasp_geometry import rectangles_to_center_format


PREDICTION_FIELDS = (
    "pred_center_x",
    "pred_center_y",
    "pred_width",
    "pred_height",
    "pred_angle_degrees",
)
EVALUATION_FIELDS = (
    "success",
    "best_iou",
    "best_angle_error_degrees",
    "matched_gt_index",
    "successful_match_iou",
    "successful_match_angle_error_degrees",
    "successful_matched_gt_index",
)
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_ground_truths(
    dataset_root: Path,
) -> dict[tuple[str, str], list[dict]]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cornell dataset root not found: {root}")
    ground_truths: dict[tuple[str, str], list[dict]] = {}
    for annotation_path in sorted(root.rglob("pcd*cpos.txt")):
        sample_id = annotation_path.stem.removesuffix("cpos")
        key = (annotation_path.parent.name, sample_id)
        if key in ground_truths:
            raise ValueError(f"duplicate Cornell annotation key: {key}")
        rectangles = load_grasp_rectangles(annotation_path, allow_empty=False)
        ground_truths[key] = rectangles_to_center_format(rectangles)
    if not ground_truths:
        raise ValueError(f"no Cornell cpos annotations found under {root}")
    return ground_truths


def _prediction_from_row(row: dict[str, str]) -> dict[str, float] | None:
    failed_value = str(row.get("failed_to_predict", "0")).strip().lower()
    if failed_value in {"1", "true", "yes"}:
        return None
    if any(not str(row.get(field, "")).strip() for field in PREDICTION_FIELDS):
        return None
    prediction = {
        "center_x": float(row["pred_center_x"]),
        "center_y": float(row["pred_center_y"]),
        "width": float(row["pred_width"]),
        "height": float(row["pred_height"]),
        "angle_degrees": float(row["pred_angle_degrees"]),
    }
    if not all(math.isfinite(value) for value in prediction.values()):
        raise ValueError(
            f"prediction for {row.get('sample_id', '<unknown>')} is non-finite"
        )
    if prediction["width"] <= 0.0 or prediction["height"] <= 0.0:
        raise ValueError(
            f"prediction for {row.get('sample_id', '<unknown>')} has invalid size"
        )
    return prediction


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rescore_predictions(
    input_csv: Path,
    output_csv: Path,
    summary_json: Path,
    dataset_root: Path,
    label: str,
    *,
    ground_truths: dict[tuple[str, str], list[dict]] | None = None,
) -> dict[str, object]:
    """Write an independently derived metric-v2 artifact for one CSV."""

    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError(f"unsafe source label: {label!r}")
    source = Path(input_csv)
    source_sha256 = _sha256(source)
    annotations = ground_truths or _load_ground_truths(dataset_root)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"prediction CSV has no header: {source}")
        original_fieldnames = list(reader.fieldnames)
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"prediction CSV is empty: {source}")

    rescored_rows: list[dict[str, object]] = []
    prediction_angles: list[float] = []
    legacy_success_count = 0
    prediction_count = 0
    for source_row in source_rows:
        sample_id = str(source_row.get("sample_id", "")).strip()
        object_directory = str(
            source_row.get("object_directory", "")
        ).strip()
        key = (object_directory, sample_id)
        if key not in annotations:
            raise ValueError(f"missing Cornell annotation for {key}")
        try:
            legacy_success_count += int(source_row.get("success", "0"))
        except ValueError as exc:
            raise ValueError(f"invalid legacy success for {key}") from exc
        prediction = _prediction_from_row(source_row)
        evaluation = evaluate_prediction(prediction, annotations[key])
        if prediction is not None:
            prediction_count += 1
            prediction_angles.append(
                float(evaluation["best_angle_error_degrees"])
            )
        row: dict[str, object] = dict(source_row)
        row.update(
            {
                "success": int(bool(evaluation["success"])),
                "best_iou": evaluation["best_iou"],
                "best_angle_error_degrees": evaluation[
                    "best_angle_error_degrees"
                ],
                "matched_gt_index": evaluation["matched_gt_index"],
                "successful_match_iou": evaluation[
                    "successful_match_iou"
                ],
                "successful_match_angle_error_degrees": evaluation[
                    "successful_match_angle_error_degrees"
                ],
                "successful_matched_gt_index": evaluation[
                    "successful_matched_gt_index"
                ],
            }
        )
        rescored_rows.append(row)

    fieldnames = list(original_fieldnames)
    for field in EVALUATION_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rescored_rows)
    _atomic_write_text(Path(output_csv), csv_buffer.getvalue())

    sample_count = len(rescored_rows)
    success_count = sum(int(row["success"]) for row in rescored_rows)
    all_ious = [float(row["best_iou"]) for row in rescored_rows]
    all_angles = [
        float(row["best_angle_error_degrees"])
        for row in rescored_rows
    ]
    summary: dict[str, object] = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "label": label,
        "dataset_root": str(dataset_root),
        "source_predictions_csv": str(source),
        "source_predictions_sha256": source_sha256,
        "rescored_predictions_csv": str(output_csv),
        "rescored_predictions_sha256": _sha256(Path(output_csv)),
        "sample_count": sample_count,
        "prediction_count": prediction_count,
        "prediction_coverage_rate": prediction_count / sample_count,
        "legacy_success_count": legacy_success_count,
        "success_count": success_count,
        "success_count_delta": success_count - legacy_success_count,
        "success_rate": success_count / sample_count,
        "mean_best_iou": _mean(all_ious),
        "mean_best_angle_error_degrees_all_rows": _mean(all_angles),
        "mean_best_angle_error_degrees_on_predictions": _mean(
            prediction_angles
        ),
        "iou_threshold": IOU_THRESHOLD,
        "angle_threshold_degrees": ANGLE_THRESHOLD_DEGREES,
    }
    _atomic_write_text(
        Path(summary_json),
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    return summary


def _source_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use LABEL=PATH")
    label, path_text = value.split("=", 1)
    if not _SAFE_LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(f"unsafe source label: {label!r}")
    if not path_text:
        raise argparse.ArgumentTypeError("source path cannot be empty")
    return label, Path(path_text)


def _comparison_argument(value: str) -> tuple[str, str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "paired comparison must use NAME=LEFT,RIGHT"
        )
    name, labels_text = value.split("=", 1)
    labels = labels_text.split(",")
    if not _SAFE_LABEL.fullmatch(name) or len(labels) != 2:
        raise argparse.ArgumentTypeError(
            "paired comparison must use safe NAME=LEFT,RIGHT labels"
        )
    left, right = labels
    if not _SAFE_LABEL.fullmatch(left) or not _SAFE_LABEL.fullmatch(right):
        raise argparse.ArgumentTypeError("paired comparison labels are unsafe")
    return name, left, right


def _success_map(path: Path) -> dict[str, int]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate sample ID in paired input: {sample_id}")
        result[sample_id] = int(row["success"])
    return result


def paired_success_comparison(
    left: dict[str, int],
    right: dict[str, int],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, object]:
    """Return paired binary counts and the exact two-sided McNemar p-value."""

    if set(left) != set(right):
        raise ValueError("paired predictions must contain identical sample IDs")
    both_success = 0
    both_failure = 0
    left_only = 0
    right_only = 0
    for sample_id in sorted(left):
        pair = (int(left[sample_id]), int(right[sample_id]))
        if pair == (1, 1):
            both_success += 1
        elif pair == (0, 0):
            both_failure += 1
        elif pair == (1, 0):
            left_only += 1
        elif pair == (0, 1):
            right_only += 1
        else:
            raise ValueError(f"success values must be binary for {sample_id}")
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail_count = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        )
        p_value = min(
            1.0,
            2.0 * lower_tail_count / float(2**discordant),
        )
    sample_count = len(left)
    return {
        "left_label": left_label,
        "right_label": right_label,
        "sample_count": sample_count,
        "both_success_count": both_success,
        "both_failure_count": both_failure,
        "left_only_success_count": left_only,
        "right_only_success_count": right_only,
        "success_rate_delta_right_minus_left": (
            (right_only - left_only) / sample_count if sample_count else 0.0
        ),
        "mcnemar_exact_two_sided_p_value": p_value,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score saved Cornell prediction CSV files",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_source_argument,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/cornell"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/shared/cornell_metric_v2"),
    )
    parser.add_argument(
        "--paired-comparison",
        action="append",
        default=[],
        type=_comparison_argument,
        metavar="NAME=LEFT,RIGHT",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = [label for label, _ in args.source]
    if len(labels) != len(set(labels)):
        raise ValueError("source labels must be unique")
    annotations = _load_ground_truths(args.dataset_root)
    summaries: dict[str, dict[str, object]] = {}
    for label, source in args.source:
        source_dir = args.output_dir / label
        summaries[label] = rescore_predictions(
            input_csv=source,
            output_csv=source_dir / "predictions.csv",
            summary_json=source_dir / "summary.json",
            dataset_root=args.dataset_root,
            label=label,
            ground_truths=annotations,
        )
    paired_comparisons: dict[str, dict[str, object]] = {}
    for name, left_label, right_label in args.paired_comparison:
        if name in paired_comparisons:
            raise ValueError(f"duplicate paired comparison name: {name}")
        missing = {
            label
            for label in (left_label, right_label)
            if label not in summaries
        }
        if missing:
            raise ValueError(
                f"paired comparison references missing sources: {sorted(missing)}"
            )
        paired_comparisons[name] = paired_success_comparison(
            _success_map(args.output_dir / left_label / "predictions.csv"),
            _success_map(args.output_dir / right_label / "predictions.csv"),
            left_label=left_label,
            right_label=right_label,
        )
    combined = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset_root": str(args.dataset_root),
        "sources": summaries,
        "paired_comparisons": paired_comparisons,
    }
    _atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    for label, summary in summaries.items():
        print(
            f"{label}: {summary['success_count']}/{summary['sample_count']} "
            f"({float(summary['success_rate']) * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
