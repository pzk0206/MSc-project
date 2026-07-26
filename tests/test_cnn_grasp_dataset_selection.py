import pytest

from src.vlm.run_cnn_grasp import partition_samples_by_role


def _items() -> list[dict]:
    return [
        {"key": ("01", "pcd0100")},
        {"key": ("01", "pcd0101")},
        {"key": ("02", "pcd0200")},
    ]


def test_partition_samples_uses_explicit_sample_roles() -> None:
    roles = {
        "pcd0100": "train",
        "pcd0101": "validation",
        "pcd0200": "test",
    }

    train, validation, test = partition_samples_by_role(_items(), roles)

    assert [item["key"][1] for item in train] == ["pcd0100"]
    assert [item["key"][1] for item in validation] == ["pcd0101"]
    assert [item["key"][1] for item in test] == ["pcd0200"]


def test_partition_samples_rejects_missing_or_unknown_roles() -> None:
    with pytest.raises(ValueError, match="missing role"):
        partition_samples_by_role(_items(), {"pcd0100": "train"})
    with pytest.raises(ValueError, match="unknown role"):
        partition_samples_by_role(
            _items(),
            {
                "pcd0100": "train",
                "pcd0101": "validation",
                "pcd0200": "holdout",
            },
        )
