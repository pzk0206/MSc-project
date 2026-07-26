import torch

from src.vlm.cnn_grasp_models import (
    MultiHeadCNNGraspRegressor,
    SingleHeadCNNGraspRegressor,
    compute_multi_head_loss,
)


def test_single_head_output_shape() -> None:
    model = SingleHeadCNNGraspRegressor()
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 6)


def test_multi_head_outputs_have_expected_shapes() -> None:
    model = MultiHeadCNNGraspRegressor()
    output = model(torch.zeros(2, 3, 224, 224))
    assert output["centre"].shape == (2, 2)
    assert output["size"].shape == (2, 2)
    assert output["orientation"].shape == (2, 2)


def test_multi_head_loss_exposes_each_component() -> None:
    predictions = {
        "centre": torch.zeros(2, 2, requires_grad=True),
        "size": torch.zeros(2, 2, requires_grad=True),
        "orientation": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], requires_grad=True
        ),
    }
    targets = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.1, 0.0, 1.0],
            [0.4, 0.6, 0.3, 0.2, 1.0, 0.0],
        ]
    )

    losses = compute_multi_head_loss(predictions, targets)

    assert set(losses) == {
        "total",
        "centre",
        "size",
        "orientation",
        "unit_norm",
    }
    losses["total"].backward()
    assert predictions["centre"].grad is not None
