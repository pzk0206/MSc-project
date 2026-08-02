"""Pure Stage 6A.1 post-hoc center-bias measurements and serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_VERSION = "stage_6a1_center_bias_diagnostic_v1"
CUBE_HALF_EXTENT_M = 0.025
XY_REFERENCE_THRESHOLD_M = 0.005


def _point3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        point = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must contain three finite values")
    return point


@dataclass(frozen=True)
class CenterBiasMeasurement:
    """One prediction-to-truth comparison using frozen diagnostic references."""

    predicted_world_surface_point: tuple[float, float, float]
    cube_truth_center: tuple[float, float, float]
    cube_half_extent_m: float
    nominal_top_reference_z_m: float
    signed_x_offset_m: float
    signed_y_offset_m: float
    xy_offset_m: float
    signed_nominal_top_z_offset_m: float
    xy_reference_threshold_m: float
    xy_within_reference_threshold: bool


def compute_center_bias(
    predicted_world_surface_point: Sequence[float],
    cube_truth_center: Sequence[float],
    *,
    cube_half_extent_m: float = CUBE_HALF_EXTENT_M,
    xy_reference_threshold_m: float = XY_REFERENCE_THRESHOLD_M,
) -> CenterBiasMeasurement:
    """Compare a visible surface point with the saved cube truth center."""

    prediction = _point3(
        predicted_world_surface_point,
        "predicted_world_surface_point",
    )
    truth = _point3(cube_truth_center, "cube_truth_center")
    if cube_half_extent_m != CUBE_HALF_EXTENT_M:
        raise ValueError("cube half extent must remain frozen at 0.025 m")
    if xy_reference_threshold_m != XY_REFERENCE_THRESHOLD_M:
        raise ValueError(
            "XY reference threshold must remain frozen at 0.005 m"
        )

    signed_x = prediction[0] - truth[0]
    signed_y = prediction[1] - truth[1]
    nominal_top = truth[2] + cube_half_extent_m
    xy_offset = math.hypot(signed_x, signed_y)
    return CenterBiasMeasurement(
        predicted_world_surface_point=prediction,
        cube_truth_center=truth,
        cube_half_extent_m=cube_half_extent_m,
        nominal_top_reference_z_m=nominal_top,
        signed_x_offset_m=signed_x,
        signed_y_offset_m=signed_y,
        xy_offset_m=xy_offset,
        signed_nominal_top_z_offset_m=prediction[2] - nominal_top,
        xy_reference_threshold_m=xy_reference_threshold_m,
        xy_within_reference_threshold=(
            xy_offset <= xy_reference_threshold_m
        ),
    )


def write_diagnostic_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """Write finite JSON without allowing JavaScript NaN extensions."""

    try:
        serialized = json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "diagnostic payload must contain JSON-compatible finite values"
        ) from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")


def write_diagnostic_csv(
    path: Path,
    measurement: CenterBiasMeasurement,
) -> None:
    """Write one flat measurement row with point fields encoded as JSON."""

    if not isinstance(measurement, CenterBiasMeasurement):
        raise TypeError("measurement must be a CenterBiasMeasurement")
    row = asdict(measurement)
    for name in ("predicted_world_surface_point", "cube_truth_center"):
        row[name] = json.dumps(row[name], allow_nan=False)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
