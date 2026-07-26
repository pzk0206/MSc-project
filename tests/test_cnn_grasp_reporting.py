import ast
from pathlib import Path

import pytest
import torch

from src.vlm import run_cnn_grasp


def _metrics(success_rate: float, iou: float, angle: float) -> dict:
    return {
        "success_rate": success_rate,
        "mean_iou": iou,
        "mean_angle": angle,
        "count": 85,
    }


def test_multi_run_summary_preserves_best_validation_loss() -> None:
    records = [
        {
            "seed": 42,
            "best_val_loss": 0.0123,
            "all": _metrics(0.70, 0.40, 18.0),
            "test": _metrics(0.80, 0.45, 19.0),
        },
        {
            "seed": 43,
            "best_val_loss": 0.0098,
            "all": _metrics(0.74, 0.44, 16.0),
            "test": _metrics(0.82, 0.47, 17.0),
        },
    ]

    summary = run_cnn_grasp.build_multi_run_summary(records)

    assert summary["seeds"] == [42, 43]
    assert summary["per_run"][0]["best_val_loss"] == pytest.approx(0.0123)
    assert summary["per_run"][1]["best_val_loss"] == pytest.approx(0.0098)
    assert summary["all"]["success_rate_mean"] == pytest.approx(0.72)
    assert summary["test"]["success_rate_mean"] == pytest.approx(0.81)


def test_cli_file_has_one_main_guard() -> None:
    path = Path("src/vlm/run_cnn_grasp.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(guards) == 1


def test_multi_head_output_is_flattened_in_legacy_parameter_order() -> None:
    output = {
        "centre": torch.tensor([[1.0, 2.0]]),
        "size": torch.tensor([[3.0, 4.0]]),
        "orientation": torch.tensor([[5.0, 6.0]]),
    }

    flattened = run_cnn_grasp.flatten_model_output(output)

    assert flattened.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_output_directory_configuration_keeps_all_artifacts_together(
    tmp_path: Path,
) -> None:
    paths = run_cnn_grasp.build_output_paths(tmp_path / "multi")

    assert paths.output_dir == tmp_path / "multi"
    assert paths.model_weights == tmp_path / "multi" / "cnn_grasp_model.pt"
    assert paths.predictions_csv == tmp_path / "multi" / "cnn_grasp_predictions.csv"
    assert paths.summary_json == tmp_path / "multi" / "cnn_grasp_summary.json"
    assert paths.training_history_json == tmp_path / "multi" / "training_history.json"


def test_default_output_directory_keeps_multi_head_away_from_legacy_results() -> None:
    assert run_cnn_grasp.resolve_output_dir("single", None) == Path(
        "data/processed/vlm/cnn_grasp"
    )
    assert run_cnn_grasp.resolve_output_dir("multi_head", None) == Path(
        "data/processed/vlm/cnn_grasp_multi_head"
    )
