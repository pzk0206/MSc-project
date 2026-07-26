from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.simulation.pybullet import perception


def test_cuda_request_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        perception.torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        perception.validate_device("cuda")

    assert perception.validate_device("cpu") == "cpu"


def test_grounding_dino_loader_moves_model_to_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str):
            events.append(("processor", model_id))
            return "processor"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str):
            events.append(("model", model_id))
            return cls()

        def to(self, device: str):
            events.append(("to", device))
            return self

        def eval(self):
            events.append("eval")
            return self

    monkeypatch.setattr(perception, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(
        perception,
        "AutoModelForZeroShotObjectDetection",
        FakeModel,
    )

    processor, model = perception.load_grounding_dino(
        "test/checkpoint",
        "cpu",
    )

    assert processor == "processor"
    assert isinstance(model, FakeModel)
    assert events == [
        ("processor", "test/checkpoint"),
        ("model", "test/checkpoint"),
        ("to", "cpu"),
        "eval",
    ]


def test_localization_clips_detection_to_image_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "rgb.png"
    assert cv2.imwrite(
        str(image_path),
        np.zeros((6, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        perception,
        "run_grounding_dino_on_image",
        lambda **kwargs: {
            "box": [-2.4, 1.2, 10.7, 9.9],
            "score": 0.75,
            "label": "object",
        },
    )

    result = perception.localize_object(
        image_path,
        prompt="small object",
        processor=object(),
        model=object(),
        device="cpu",
    )

    assert result == perception.Localization(
        box=(0, 1, 7, 5),
        score=0.75,
        label="object",
    )


def test_localization_preserves_no_detection_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "rgb.png"
    assert cv2.imwrite(
        str(image_path),
        np.zeros((6, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        perception,
        "run_grounding_dino_on_image",
        lambda **kwargs: None,
    )

    result = perception.localize_object(
        image_path,
        prompt="small object",
        processor=object(),
        model=object(),
        device="cpu",
    )

    assert result is None


def test_geometry_backend_reuses_existing_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_predict(image, box, expand_ratio, use_box_fallback):
        called.update(
            box=box,
            expand_ratio=expand_ratio,
            use_box_fallback=use_box_fallback,
        )
        return (
            {
                "center_x": 30.0,
                "center_y": 40.0,
                "width": 20.0,
                "height": 10.0,
                "angle_degrees": 15.0,
            },
            np.zeros(image.shape[:2], dtype=np.uint8),
            box,
            "",
        )

    monkeypatch.setattr(
        perception,
        "predict_grasp_with_vlm_box",
        fake_predict,
    )

    result = perception.predict_grasp(
        np.zeros((100, 120, 3), dtype=np.uint8),
        perception.Localization((10, 20, 80, 90), 0.9, "object"),
        backend="geometry",
        device="cpu",
        model=None,
    )

    assert result.grasp["center_x"] == 30.0
    assert called == {
        "box": (10, 20, 80, 90),
        "expand_ratio": 0.10,
        "use_box_fallback": True,
    }


def test_cnn_prediction_returns_to_full_image_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_crop_to_tensor(crop: np.ndarray):
        observed["crop_shape"] = crop.shape
        return torch.zeros(3, 224, 224)

    def fake_predict_from_crop(
        model,
        crop_tensor,
        crop_w,
        crop_h,
        device,
    ):
        observed["crop_size"] = (crop_w, crop_h)
        return {
            "center_x": 5.0,
            "center_y": 7.0,
            "width": 20.0,
            "height": 10.0,
            "angle_degrees": -12.0,
        }

    monkeypatch.setattr(
        perception,
        "crop_to_tensor",
        fake_crop_to_tensor,
    )
    monkeypatch.setattr(
        perception,
        "predict_from_crop",
        fake_predict_from_crop,
    )

    result = perception.predict_grasp(
        np.zeros((100, 120, 3), dtype=np.uint8),
        perception.Localization((10, 20, 50, 80), 0.8, "object"),
        backend="single",
        device="cpu",
        model=object(),
    )

    assert result.grasp["center_x"] == 15.0
    assert result.grasp["center_y"] == 27.0
    assert observed["crop_shape"] == (61, 41, 3)
    assert observed["crop_size"] == (41, 61)


def test_cnn_loader_accepts_matching_and_rejects_mismatched_weights(
    tmp_path: Path,
) -> None:
    single = perception.create_model("single")
    weights_path = tmp_path / "single.pt"
    torch.save(single.model.state_dict(), weights_path)

    loaded = perception.load_cnn_backend(
        "single",
        weights_path,
        "cpu",
    )

    assert loaded.__class__ is single.__class__
    with pytest.raises(RuntimeError, match=r"multi_head.*single\.pt"):
        perception.load_cnn_backend(
            "multi_head",
            weights_path,
            "cpu",
        )
