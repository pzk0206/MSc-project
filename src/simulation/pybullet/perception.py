"""Adapters from simulated RGB images to the project's existing models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from src.vlm.run_cnn_grasp import (
    _load_state_dict,
    create_model,
    crop_to_tensor,
    predict_from_crop,
)
from src.vlm.run_grounding_dino_localization import (
    run_grounding_dino_on_image,
)
from src.vlm.run_vlm_assisted_grasp import (
    predict_grasp_with_vlm_box,
)


@dataclass(frozen=True)
class Localization:
    """Best Grounding DINO localization in full-image coordinates."""

    box: tuple[int, int, int, int]
    score: float
    label: str


@dataclass(frozen=True)
class PilotPrediction:
    """Unified localization and centre-format grasp output."""

    localization: Localization
    backend: str
    grasp: dict[str, float]
    failure_reason: str = ""


def validate_device(device: str) -> str:
    """Validate an explicit inference device without silent fallback."""

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return device


def load_grounding_dino(
    model_id: str,
    device: str,
) -> tuple[object, object]:
    """Load one Grounding DINO processor/model pair for a pilot run."""

    selected_device = validate_device(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(selected_device)
    model.eval()
    return processor, model


def localize_object(
    rgb_path: Path,
    prompt: str,
    processor: object,
    model: object,
    device: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> Localization | None:
    """Run the existing single-image detector and clip its best box."""

    detection = run_grounding_dino_on_image(
        image_path=rgb_path,
        prompt=prompt,
        processor=processor,
        model=model,
        device=validate_device(device),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if detection is None:
        return None

    with Image.open(rgb_path) as image:
        image_width, image_height = image.size
    values = np.asarray(detection["box"], dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Grounding DINO returned an invalid box")
    x1, y1, x2, y2 = (float(value) for value in values)
    left = int(np.clip(math.floor(min(x1, x2)), 0, image_width - 1))
    top = int(np.clip(math.floor(min(y1, y2)), 0, image_height - 1))
    right = int(np.clip(math.ceil(max(x1, x2)), 0, image_width - 1))
    bottom = int(np.clip(math.ceil(max(y1, y2)), 0, image_height - 1))
    if right <= left or bottom <= top:
        raise ValueError("Grounding DINO returned a zero-area box")

    return Localization(
        box=(left, top, right, bottom),
        score=float(detection["score"]),
        label=str(detection["label"]),
    )


def load_cnn_backend(
    backend: str,
    weights_path: Path,
    device: str,
) -> object:
    """Load matching existing CNN architecture and state-dict weights."""

    selected_device = validate_device(device)
    if backend not in {"single", "multi_head"}:
        raise ValueError(f"unsupported CNN backend: {backend}")
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{backend} weights do not exist: {weights_path}"
        )

    model = create_model(backend)
    try:
        state_dict = torch.load(
            weights_path,
            map_location=selected_device,
            weights_only=True,
        )
        _load_state_dict(model, backend, state_dict)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"failed to load {backend} weights from {weights_path}: {exc}"
        ) from exc
    model = model.to(selected_device)
    model.eval()
    return model


def predict_grasp(
    image_bgr: np.ndarray,
    localization: Localization,
    backend: str,
    device: str,
    model: object | None,
) -> PilotPrediction:
    """Dispatch to an existing geometry or CNN grasp backend."""

    selected_device = validate_device(device)
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BGR image must have shape (H, W, 3)")

    if backend == "geometry":
        prediction, _, _, failure_reason = predict_grasp_with_vlm_box(
            image,
            localization.box,
            expand_ratio=0.10,
            use_box_fallback=True,
        )
        if prediction is None:
            raise RuntimeError(
                f"geometry backend failed: {failure_reason or 'no prediction'}"
            )
        return PilotPrediction(
            localization=localization,
            backend=backend,
            grasp=dict(prediction),
            failure_reason=failure_reason,
        )

    if backend not in {"single", "multi_head"}:
        raise ValueError(f"unsupported grasp backend: {backend}")
    if model is None:
        raise ValueError(f"{backend} backend requires a loaded model")

    left, top, right, bottom = localization.box
    crop = image[top : bottom + 1, left : right + 1]
    if crop.size == 0:
        raise ValueError("localization produced an empty CNN crop")
    crop_height, crop_width = crop.shape[:2]
    crop_tensor = crop_to_tensor(crop)
    prediction = dict(
        predict_from_crop(
            model,
            crop_tensor,
            crop_width,
            crop_height,
            selected_device,
        )
    )
    prediction["center_x"] = float(prediction["center_x"]) + left
    prediction["center_y"] = float(prediction["center_y"]) + top
    return PilotPrediction(
        localization=localization,
        backend=backend,
        grasp=prediction,
    )
