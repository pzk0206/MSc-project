import copy
import json
from pathlib import Path

import pytest

from src.shared.cornell_cross_validation import (
    generate_image_wise_manifest,
    load_manifest,
    save_manifest,
    sha256_file,
    validate_image_wise_manifest,
)


def _samples(count: int = 885) -> list[tuple[str, str]]:
    return [
        (f"pcd{100 + index:04d}", f"{index // 100 + 1:02d}")
        for index in range(count)
    ]


def test_image_wise_manifest_is_deterministic_and_balanced() -> None:
    first = generate_image_wise_manifest(_samples(), seed=42)
    second = generate_image_wise_manifest(_samples(), seed=42)

    assert first == second
    for fold in range(5):
        fold_rows = [row for row in first if row["fold"] == fold]
        counts = {
            role: sum(row["role"] == role for row in fold_rows)
            for role in ("train", "validation", "test")
        }
        assert counts == {"train": 566, "validation": 142, "test": 177}

    test_ids = [
        row["sample_id"] for row in first if row["role"] == "test"
    ]
    assert len(test_ids) == 885
    assert len(set(test_ids)) == 885


def test_manifest_validator_rejects_test_leakage() -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    damaged = copy.deepcopy(rows)
    fold_zero_test = next(
        row for row in damaged if row["fold"] == 0 and row["role"] == "test"
    )
    duplicate = dict(fold_zero_test)
    duplicate["role"] = "train"
    damaged.append(duplicate)

    with pytest.raises(ValueError, match="sample has multiple roles"):
        validate_image_wise_manifest(
            damaged,
            expected_sample_ids={sample_id for sample_id, _ in _samples()},
        )


def test_manifest_validator_rejects_missing_test_coverage() -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    missing_id = next(
        row["sample_id"]
        for row in rows
        if row["fold"] == 0 and row["role"] == "test"
    )
    damaged = [
        row
        for row in rows
        if not (row["sample_id"] == missing_id and row["fold"] == 0)
    ]

    with pytest.raises(ValueError, match="fold 0 does not cover every sample"):
        validate_image_wise_manifest(
            damaged,
            expected_sample_ids={sample_id for sample_id, _ in _samples()},
        )


def test_manifest_round_trip_and_hash_are_deterministic(tmp_path: Path) -> None:
    rows = generate_image_wise_manifest(_samples(), seed=42)
    first_csv = tmp_path / "first.csv"
    first_json = tmp_path / "first.json"
    second_csv = tmp_path / "second.csv"
    second_json = tmp_path / "second.json"

    first_hash = save_manifest(rows, first_csv, first_json)
    second_hash = save_manifest(rows, second_csv, second_json)

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_hash == second_hash == sha256_file(first_json)
    assert load_manifest(first_json) == rows
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["protocol"] == "cornell_image_wise_5_fold"
    assert payload["sample_count"] == 885
