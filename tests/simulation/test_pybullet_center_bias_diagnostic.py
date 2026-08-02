from dataclasses import asdict
import csv
import json
from pathlib import Path

import pytest

from src.simulation.pybullet.center_bias_diagnostic import (
    PROTOCOL_VERSION,
    compute_center_bias,
    write_diagnostic_csv,
    write_diagnostic_json,
)


def test_compute_center_bias_records_signed_offsets_and_reference_gate() -> None:
    result = compute_center_bias(
        (0.5064564100151149, 0.002224916375108214, 0.6754779706501471),
        (0.4800002872798181, -5.134833814891427e-7, 0.649968798272667),
    )

    assert result.signed_x_offset_m == pytest.approx(0.026456122735296794)
    assert result.signed_y_offset_m == pytest.approx(0.002225429858489703)
    assert result.xy_offset_m == pytest.approx(0.026549556836982145)
    assert result.nominal_top_reference_z_m == pytest.approx(
        0.674968798272667
    )
    assert result.signed_nominal_top_z_offset_m == pytest.approx(
        0.0005091723774801604
    )
    assert result.xy_reference_threshold_m == 0.005
    assert result.xy_within_reference_threshold is False


@pytest.mark.parametrize(
    ("prediction", "truth"),
    [
        ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
)
def test_compute_center_bias_rejects_invalid_points(
    prediction,
    truth,
) -> None:
    with pytest.raises(ValueError):
        compute_center_bias(prediction, truth)


def test_compute_center_bias_rejects_protocol_constant_changes() -> None:
    with pytest.raises(ValueError, match="half extent"):
        compute_center_bias(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            cube_half_extent_m=0.03,
        )
    with pytest.raises(ValueError, match="reference threshold"):
        compute_center_bias(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            xy_reference_threshold_m=0.01,
        )


def test_diagnostic_writers_emit_strict_json_and_one_row_csv(
    tmp_path: Path,
) -> None:
    measurement = compute_center_bias(
        (0.506, 0.002, 0.675),
        (0.48, 0.0, 0.65),
    )
    payload = {
        "protocol": PROTOCOL_VERSION,
        "measurement": asdict(measurement),
    }

    write_diagnostic_json(tmp_path / "diagnostic.json", payload)
    write_diagnostic_csv(tmp_path / "diagnostic.csv", measurement)

    loaded = json.loads(
        (tmp_path / "diagnostic.json").read_text(encoding="utf-8")
    )
    assert loaded["protocol"] == "stage_6a1_center_bias_diagnostic_v1"
    with (tmp_path / "diagnostic.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["xy_offset_m"]) == pytest.approx(
        measurement.xy_offset_m
    )
    assert json.loads(rows[0]["predicted_world_surface_point"]) == [
        0.506,
        0.002,
        0.675,
    ]


def test_diagnostic_json_rejects_non_finite_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        write_diagnostic_json(
            tmp_path / "diagnostic.json",
            {"bad": float("nan")},
        )
    assert not (tmp_path / "diagnostic.json").exists()
