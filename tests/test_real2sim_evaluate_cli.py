from __future__ import annotations

import csv
import json
from pathlib import Path

from botyard_real2sim.cli import evaluate
from botyard_real2sim.cli.evaluate import evaluate_files


def _trace(path: Path, *, with_motor: bool = True) -> Path:
    fields = ["time_s", "FFJ2/position_rad"]
    if with_motor:
        fields.append("FFJ2/motor_position_rad")
    fields += ["FFJ2/tension_N", "FFJ2/cmd_rad"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(101):
            t = index / 100
            row = [t, t]
            if with_motor:
                row.append(2 * t)
            row += [1 + t, 0 if index < 10 else 0.5]
            writer.writerow(row)
    return path


def _metadata(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "saturated_force_evaluations": 0,
                "mimic_max_error_rad": 0,
                "calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _evaluate(tmp_path: Path, *, with_motor: bool = True, with_metadata: bool = True) -> dict:
    real = _trace(tmp_path / "real.csv", with_motor=with_motor)
    sim = _trace(tmp_path / "sim.csv", with_motor=with_motor)
    metadata = _metadata(tmp_path / "sim.metadata.json") if with_metadata else None
    return evaluate_files(
        real,
        sim,
        metadata,
        ["FFJ2"],
        tolerances={"angle_deg": 1, "motor_angle_deg": 1, "tension_N": 1},
    )


def test_evaluate_passes_three_complete_curves_and_hashes_inputs(tmp_path):
    result = _evaluate(tmp_path)
    assert result["quality"]["quality_valid"] is True
    assert result["quality"]["candidate_pass"] is True
    assert result["quality"]["result_scope"] == "curve_fit_candidate_only"
    assert result["quality"]["score"] == 0
    assert result["real_csv_sha256"]
    assert result["sim_csv_sha256"]
    assert result["replay_metadata_sha256"]


def test_missing_required_motor_curve_cannot_pass(tmp_path):
    result = _evaluate(tmp_path, with_motor=False)
    assert result["quality"]["quality_valid"] is False
    assert result["quality"]["candidate_pass"] is False
    assert any("motor_angle_deg" in reason for reason in result["quality"]["reasons"])


def test_missing_replay_metadata_cannot_pass(tmp_path):
    result = _evaluate(tmp_path, with_metadata=False)
    assert result["quality"]["quality_valid"] is False
    assert result["quality"]["candidate_pass"] is False
    assert any(reason.startswith("replay metadata is required") for reason in result["quality"]["reasons"])


def test_cli_returns_failure_when_valid_candidate_exceeds_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "evaluate_files",
        lambda *_args, **_kwargs: {
            "quality": {"quality_valid": True, "tolerance_pass": False, "candidate_pass": False}
        },
    )

    status = evaluate.main(
        [
            "--real",
            str(tmp_path / "real.csv"),
            "--sim",
            str(tmp_path / "sim.csv"),
            "--joints",
            "FFJ2",
            "--tolerance",
            "angle_deg=1",
            "--tolerance",
            "motor_angle_deg=1",
            "--tolerance",
            "tension_N=1",
        ]
    )

    assert status == 1
