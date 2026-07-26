"""Run project-authored Cornell image-wise CNN cross-validation.

The split protocol follows Lenz, Lee, and Saxena (IJRR 2015):
https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

The orchestration code is independently written for this project and does not
copy an external cross-validation implementation.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FoldPaths:
    directory: Path
    model: Path
    history: Path
    predictions: Path
    summary: Path


def build_fold_paths(
    output_root: Path,
    architecture: str,
    fold: int,
) -> FoldPaths:
    """Build isolated output paths for one architecture and fold."""
    directory = output_root / architecture / f"fold_{fold}"
    return FoldPaths(
        directory=directory,
        model=directory / "model.pt",
        history=directory / "training_history.json",
        predictions=directory / "predictions.csv",
        summary=directory / "summary.json",
    )


def metrics_from_rows(rows: list[dict]) -> dict[str, float | int]:
    """Calculate Cornell metrics over a non-empty prediction collection."""
    if not rows:
        raise ValueError("cannot calculate metrics from empty rows")
    return {
        "sample_count": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "success_rate": float(
            np.mean([int(row["success"]) for row in rows])
        ),
        "mean_iou": float(
            np.mean([float(row["best_iou"]) for row in rows])
        ),
        "mean_angle": float(
            np.mean(
                [
                    float(row["best_angle_error_degrees"])
                    for row in rows
                ]
            )
        ),
    }


def validate_complete_fold_records(
    fold_records: list[dict],
    combined_rows: list[dict],
    expected_sample_ids: set[str],
) -> None:
    """Reject incomplete, duplicate, or numerically invalid fold results."""
    folds = {int(record["fold"]) for record in fold_records}
    if folds != set(range(5)):
        raise ValueError("fold records must contain folds 0 through 4")
    ids = [str(row["sample_id"]) for row in combined_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("combined predictions contain duplicate sample IDs")
    if set(ids) != expected_sample_ids:
        raise ValueError(
            "combined predictions do not cover expected sample IDs"
        )
    for value in (
        float(row[field])
        for row in combined_rows
        for field in (
            "best_iou",
            "best_angle_error_degrees",
            "pred_center_x",
            "pred_center_y",
            "pred_width",
            "pred_height",
            "pred_angle_degrees",
        )
        if field in row
    ):
        if not math.isfinite(value):
            raise ValueError(
                "combined predictions contain non-finite values"
            )


def build_cross_validation_summary(
    fold_records: list[dict],
    combined_rows: list[dict],
    architecture: str,
    seed: int,
    manifest_hash: str,
) -> dict:
    """Build pooled metrics and population statistics over five folds."""
    per_fold = []
    for record in sorted(
        fold_records,
        key=lambda item: int(item["fold"]),
    ):
        metrics = metrics_from_rows(record["rows"])
        per_fold.append(
            {
                "fold": int(record["fold"]),
                "best_val_loss": float(record["best_val_loss"]),
                **metrics,
            }
        )

    def aggregate(field: str) -> tuple[float, float]:
        values = [float(record[field]) for record in per_fold]
        return float(np.mean(values)), float(np.std(values))

    success_mean, success_std = aggregate("success_rate")
    iou_mean, iou_std = aggregate("mean_iou")
    angle_mean, angle_std = aggregate("mean_angle")
    return {
        "method": f"vlm_cnn_{architecture}_image_wise_5_fold",
        "protocol": "cornell_image_wise_5_fold",
        "architecture": architecture,
        "fold_count": 5,
        "training_seed_per_fold": seed,
        "manifest_sha256": manifest_hash,
        "pooled": metrics_from_rows(combined_rows),
        "folds": {
            "success_rate_mean": success_mean,
            "success_rate_std": success_std,
            "mean_iou_mean": iou_mean,
            "mean_iou_std": iou_std,
            "mean_angle_mean": angle_mean,
            "mean_angle_std": angle_std,
        },
        "per_fold": per_fold,
    }


def build_architecture_comparison(
    single_summary: dict,
    multi_summary: dict,
) -> dict:
    """Compare paired folds after enforcing a shared manifest."""
    if single_summary["architecture"] != "single":
        raise ValueError("single summary has the wrong architecture")
    if multi_summary["architecture"] != "multi_head":
        raise ValueError("multi summary has the wrong architecture")
    if (
        single_summary["manifest_sha256"]
        != multi_summary["manifest_sha256"]
    ):
        raise ValueError("architecture summaries use different manifests")

    fields = ("success_rate", "mean_iou", "mean_angle")
    pooled_delta = {
        field: float(multi_summary["pooled"][field])
        - float(single_summary["pooled"][field])
        for field in fields
    }
    single_folds = {
        int(record["fold"]): record
        for record in single_summary["per_fold"]
    }
    multi_folds = {
        int(record["fold"]): record
        for record in multi_summary["per_fold"]
    }
    if (
        set(single_folds) != set(range(5))
        or set(multi_folds) != set(range(5))
    ):
        raise ValueError(
            "both architecture summaries must contain five folds"
        )
    paired = [
        {
            "fold": fold,
            **{
                field: float(multi_folds[fold][field])
                - float(single_folds[fold][field])
                for field in fields
            },
        }
        for fold in range(5)
    ]
    return {
        "protocol": "cornell_image_wise_5_fold",
        "manifest_sha256": single_summary["manifest_sha256"],
        "delta_definition": "multi_head_minus_single",
        "pooled_delta_multi_minus_single": pooled_delta,
        "paired_fold_deltas": paired,
        "single": single_summary,
        "multi_head": multi_summary,
    }
