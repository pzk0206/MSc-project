"""Run project-authored Cornell image-wise CNN cross-validation.

The split protocol follows Lenz, Lee, and Saxena (IJRR 2015):
https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

The orchestration code is independently written for this project and does not
copy an external cross-validation implementation.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.cornell_cross_validation import (
    generate_image_wise_manifest,
    load_manifest,
    roles_for_fold,
    save_manifest,
    sha256_file,
    validate_image_wise_manifest,
)
from src.shared.cornell_dataset import CornellGraspDataset
from src.vlm.run_cnn_grasp import (
    DATASET_ROOT,
    _train_one_run,
    build_all_samples,
    evaluate_model,
    load_vlm_boxes,
    partition_samples_by_role,
    save_results,
)


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


def prepare_manifest(
    output_root: Path,
    seed: int = 42,
) -> tuple[list[dict], Path, str]:
    """Create or revalidate the fixed manifest for all Cornell images."""
    dataset = CornellGraspDataset(DATASET_ROOT)
    samples = [
        (sample.sample_id, sample.object_directory)
        for sample in dataset.samples
    ]
    expected_ids = {sample_id for sample_id, _ in samples}
    csv_path = output_root / f"image_wise_folds_seed_{seed}.csv"
    json_path = output_root / f"image_wise_folds_seed_{seed}.json"
    if json_path.exists():
        rows = load_manifest(json_path)
    else:
        rows = generate_image_wise_manifest(samples, seed=seed)
        save_manifest(rows, csv_path, json_path)
    validate_image_wise_manifest(rows, expected_ids)
    return rows, json_path, sha256_file(json_path)


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


def run_fold(
    architecture: str,
    fold: int,
    device: str,
    seed: int,
    output_root: Path,
    manifest_rows: list[dict],
    manifest_hash: str,
) -> dict:
    """Train, evaluate, and immediately persist one test fold."""
    all_samples = build_all_samples()
    role_map = roles_for_fold(manifest_rows, fold)
    train_data, validation_data, test_data = partition_samples_by_role(
        all_samples,
        role_map,
    )
    counts = (
        len(train_data),
        len(validation_data),
        len(test_data),
    )
    if counts != (566, 142, 177):
        raise ValueError(f"unexpected fold {fold} sizes: {counts}")

    id_sets = [
        {item["key"][1] for item in partition}
        for partition in (train_data, validation_data, test_data)
    ]
    if (
        id_sets[0] & id_sets[1]
        or id_sets[0] & id_sets[2]
        or id_sets[1] & id_sets[2]
    ):
        raise ValueError(f"sample leakage detected in fold {fold}")

    print(
        f"fold={fold} manifest_sha256={manifest_hash} "
        f"train={counts[0]} validation={counts[1]} test={counts[2]}"
    )
    paths = build_fold_paths(output_root, architecture, fold)
    model, best_val_loss = _train_one_run(
        train_data,
        validation_data,
        device,
        seed,
        architecture=architecture,
        model_weights_path=paths.model,
        history_path=paths.history,
    )
    rows, _ = evaluate_model(
        model,
        test_data,
        CornellGraspDataset(DATASET_ROOT),
        load_vlm_boxes(),
        device=device,
    )
    for row in rows:
        row.update(
            {
                "protocol": "cornell_image_wise_5_fold",
                "fold": fold,
                "split": "test",
                "architecture": architecture,
                "training_seed": seed,
                "manifest_sha256": manifest_hash,
            }
        )
    summary = {
        "method": f"vlm_cnn_{architecture}_image_wise_fold",
        "protocol": "cornell_image_wise_5_fold",
        "architecture": architecture,
        "fold": fold,
        "training_seed": seed,
        "manifest_sha256": manifest_hash,
        "train_count": len(train_data),
        "validation_count": len(validation_data),
        "test_count": len(test_data),
        "best_val_loss": best_val_loss,
        **metrics_from_rows(rows),
    }
    save_results(
        rows,
        summary,
        predictions_csv=paths.predictions,
        summary_json=paths.summary,
    )
    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "rows": rows,
        "summary": summary,
    }


def aggregate_saved_folds(
    architecture: str,
    output_root: Path,
    expected_sample_ids: set[str],
    manifest_hash: str,
    seed: int,
) -> dict:
    """Audit five saved folds, then persist pooled predictions and metrics."""
    fold_records = []
    for fold in range(5):
        paths = build_fold_paths(output_root, architecture, fold)
        for path in (paths.history, paths.predictions, paths.summary):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing fold_{fold} artifact: {path}"
                )
        history = json.loads(paths.history.read_text(encoding="utf-8"))
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        with paths.predictions.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))

        expected_best = min(
            float(epoch["val_loss"]) for epoch in history
        )
        if not math.isclose(
            float(summary["best_val_loss"]),
            expected_best,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"fold {fold} best validation loss mismatch")
        if summary["manifest_sha256"] != manifest_hash:
            raise ValueError(f"fold {fold} manifest hash mismatch")
        if summary["architecture"] != architecture:
            raise ValueError(f"fold {fold} architecture mismatch")
        if int(summary["fold"]) != fold:
            raise ValueError(f"fold {fold} metadata mismatch")
        if len(rows) != int(summary["test_count"]) or len(rows) != 177:
            raise ValueError(f"fold {fold} prediction count mismatch")
        fold_records.append(
            {
                "fold": fold,
                "best_val_loss": expected_best,
                "rows": rows,
                "summary": summary,
            }
        )

    combined_rows = sorted(
        (
            row
            for record in fold_records
            for row in record["rows"]
        ),
        key=lambda row: row["sample_id"],
    )
    validate_complete_fold_records(
        fold_records,
        combined_rows,
        expected_sample_ids,
    )
    final_summary = build_cross_validation_summary(
        fold_records,
        combined_rows,
        architecture,
        seed,
        manifest_hash,
    )
    architecture_dir = output_root / architecture
    save_results(
        combined_rows,
        final_summary,
        predictions_csv=architecture_dir / "combined_predictions.csv",
        summary_json=architecture_dir / "cross_validation_summary.json",
    )
    return final_summary


def _compare_saved_architectures(output_root: Path) -> dict:
    summaries = {}
    prediction_ids = {}
    for architecture in ("single", "multi_head"):
        architecture_dir = output_root / architecture
        summary_path = architecture_dir / "cross_validation_summary.json"
        predictions_path = architecture_dir / "combined_predictions.csv"
        if not summary_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(
                f"missing final {architecture} cross-validation artifacts"
            )
        summaries[architecture] = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        with predictions_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            prediction_ids[architecture] = {
                row["sample_id"] for row in csv.DictReader(handle)
            }
    if prediction_ids["single"] != prediction_ids["multi_head"]:
        raise ValueError("architecture predictions use different sample IDs")

    comparison = build_architecture_comparison(
        summaries["single"],
        summaries["multi_head"],
    )
    output_path = output_root / "architecture_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cornell image-wise CNN cross-validation"
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "run", "aggregate", "compare"],
        default="run",
    )
    parser.add_argument(
        "--architecture",
        choices=["single", "multi_head"],
        default="single",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/vlm/cnn_cross_validation"),
    )
    args = parser.parse_args()

    rows, manifest_path, manifest_hash = prepare_manifest(
        args.output_root,
        seed=args.seed,
    )
    expected_ids = {str(row["sample_id"]) for row in rows}
    if args.mode == "manifest":
        test_fold_sizes = [
            sum(
                int(row["fold"]) == fold and row["role"] == "test"
                for row in rows
            )
            for fold in range(5)
        ]
        print("protocol=cornell_image_wise_5_fold")
        print(f"sample_count={len(expected_ids)}")
        print(f"test_fold_sizes={test_fold_sizes}")
        print(f"manifest={manifest_path}")
        print(f"sha256={manifest_hash}")
        return

    if args.mode == "compare":
        comparison = _compare_saved_architectures(args.output_root)
        print(
            "comparison_manifest_sha256="
            f"{comparison['manifest_sha256']}"
        )
        return

    if args.mode == "aggregate":
        summary = aggregate_saved_folds(
            args.architecture,
            args.output_root,
            expected_ids,
            manifest_hash,
            args.seed,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
    folds = [args.fold] if args.fold is not None else list(range(5))
    for fold in folds:
        run_fold(
            args.architecture,
            fold,
            args.device,
            args.seed,
            args.output_root,
            rows,
            manifest_hash,
        )


if __name__ == "__main__":
    main()
