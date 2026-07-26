"""Deterministic Cornell image-wise fold manifests.

Protocol reference:
Ian Lenz, Honglak Lee, and Ashutosh Saxena,
"Deep Learning for Detecting Robotic Grasps", IJRR 2015.
https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf

This module is an independent project implementation of the paper's
image-wise split definition; it does not copy third-party split code.
"""

import csv
import hashlib
import json
import random
from pathlib import Path


VALID_ROLES = {"train", "validation", "test"}
MANIFEST_FIELDS = [
    "protocol",
    "seed",
    "fold",
    "sample_id",
    "object_directory",
    "role",
]


def generate_image_wise_manifest(
    samples: list[tuple[str, str]],
    n_splits: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> list[dict[str, object]]:
    """Assign every image to train, validation, and test roles per fold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    ordered = sorted(samples)
    sample_ids = [sample_id for sample_id, _ in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    if len(ordered) % n_splits:
        raise ValueError("sample count must be divisible by n_splits")

    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    fold_size = len(shuffled) // n_splits
    test_folds = [
        shuffled[start : start + fold_size]
        for start in range(0, len(shuffled), fold_size)
    ]

    rows: list[dict[str, object]] = []
    directory_by_id = dict(ordered)
    all_ids = set(sample_ids)
    for fold, test_items in enumerate(test_folds):
        test_ids = {sample_id for sample_id, _ in test_items}
        remaining_ids = sorted(all_ids - test_ids)
        random.Random(seed + 1000 + fold).shuffle(remaining_ids)
        validation_count = round(len(remaining_ids) * validation_fraction)
        validation_ids = set(remaining_ids[:validation_count])

        for sample_id in sample_ids:
            role = (
                "test"
                if sample_id in test_ids
                else "validation"
                if sample_id in validation_ids
                else "train"
            )
            rows.append(
                {
                    "protocol": "cornell_image_wise_5_fold",
                    "seed": seed,
                    "fold": fold,
                    "sample_id": sample_id,
                    "object_directory": directory_by_id[sample_id],
                    "role": role,
                }
            )
    return rows


def validate_image_wise_manifest(
    rows: list[dict[str, object]],
    expected_sample_ids: set[str],
    n_splits: int = 5,
) -> None:
    """Reject incomplete, overlapping, or malformed fold assignments."""
    folds = {int(row["fold"]) for row in rows}
    if folds != set(range(n_splits)):
        raise ValueError(f"manifest must contain folds 0 through {n_splits - 1}")

    test_occurrences: list[str] = []
    for fold in range(n_splits):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        pairs = [
            (str(row["sample_id"]), str(row["role"]))
            for row in fold_rows
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"duplicate sample role in fold {fold}")
        fold_ids = [sample_id for sample_id, _ in pairs]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError(f"sample has multiple roles in fold {fold}")
        if set(fold_ids) != expected_sample_ids:
            raise ValueError(f"fold {fold} does not cover every sample")
        if {role for _, role in pairs} - VALID_ROLES:
            raise ValueError(f"fold {fold} contains an unknown role")
        test_occurrences.extend(
            sample_id for sample_id, role in pairs if role == "test"
        )

    if len(test_occurrences) != len(expected_sample_ids):
        raise ValueError("test folds do not contain the expected sample count")
    if set(test_occurrences) != expected_sample_ids:
        raise ValueError("test folds do not cover every sample exactly once")


def roles_for_fold(
    rows: list[dict[str, object]],
    fold: int,
) -> dict[str, str]:
    """Return the sample-to-role mapping for one fold."""
    selected = [row for row in rows if int(row["fold"]) == fold]
    return {
        str(row["sample_id"]): str(row["role"])
        for row in selected
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a saved manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(
    rows: list[dict[str, object]],
    csv_path: Path,
    json_path: Path,
) -> str:
    """Save stable CSV and JSON representations and return the JSON hash."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows,
        key=lambda row: (int(row["fold"]), str(row["sample_id"])),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)

    payload = {
        "protocol": "cornell_image_wise_5_fold",
        "sample_count": len(
            {str(row["sample_id"]) for row in ordered_rows}
        ),
        "fold_count": len({int(row["fold"]) for row in ordered_rows}),
        "rows": ordered_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(json_path)


def load_manifest(json_path: Path) -> list[dict[str, object]]:
    """Load rows from a supported JSON manifest."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "cornell_image_wise_5_fold":
        raise ValueError("unsupported manifest protocol")
    return payload["rows"]
