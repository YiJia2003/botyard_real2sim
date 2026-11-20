from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from botyard_real2sim.cli import fit
from botyard_real2sim.optimizer import ParameterBound


def test_validation_manifest_uses_evaluation_readiness(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(
        json.dumps({"split": "validation"}),
        encoding="utf-8",
    )
    csv_path = tmp_path / "validation.csv"
    monkeypatch.setattr(
        fit,
        "validate_dataset",
        lambda *_args, **_kwargs: {
            "fit_ready": False,
            "evaluation_ready": True,
            "csv": str(csv_path),
            "blocking_reasons": [],
        },
    )

    manifest, resolved_csv, validation = fit._resolve_dataset(manifest_path, "validation")

    assert manifest["split"] == "validation"
    assert resolved_csv == csv_path.resolve()
    assert validation["evaluation_ready"] is True


def test_replay_candidate_passes_dataset_root_pose_to_replay(tmp_path: Path):
    captured = {}

    def runner(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    fit._replay_candidate(
        tmp_path / "capture.csv",
        tmp_path / "config.yaml",
        {},
        tmp_path / "candidate",
        root_pose={"position_m": [1, 2, 3], "quaternion_xyzw": [0, 0, 0, 1]},
        physics_dt=0.0001,
        sample_dt=0.003,
        device="cpu",
        duration=None,
        timeout_s=10,
        runner=runner,
    )

    command = captured["command"]
    position_index = command.index("--root-pos")
    quaternion_index = command.index("--root-xyzw")
    assert command[position_index + 1 : position_index + 4] == ["1", "2", "3"]
    assert command[quaternion_index + 1 : quaternion_index + 5] == ["0", "0", "0", "1"]


def test_fit_rejects_missing_channel_tolerance_before_creating_run_directory(tmp_path: Path):
    out_dir = tmp_path / "invalid-fit"

    with pytest.raises(ValueError, match="explicit tolerances"):
        fit.run_fit(
            train_manifest=tmp_path / "train.json",
            validation_manifest=tmp_path / "validation.json",
            config_path=tmp_path / "config.yaml",
            bounds=[ParameterBound("tendon.cable_stiffness_n_per_m.THJ2", 1, 2)],
            tolerances={"angle_deg": 1},
            weights={},
            out_dir=out_dir,
            max_evaluations=5,
        )

    assert not out_dir.exists()


def test_fit_rejects_identical_recording_hashes_with_distinct_manifest_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fit,
        "_resolve_dataset",
        lambda _path, split: (
            {
                "dataset_id": f"{split}-id",
                "split": split,
                "joints": ["THJ2"],
                "experiment": {"root_pose": {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}},
            },
            tmp_path / f"{split}.csv",
            {"metrics": {"csv_sha256": "a" * 64}},
        ),
    )
    monkeypatch.setattr(fit, "_require_optimizer_dependency", lambda: None)

    with pytest.raises(ValueError, match="independent recordings"):
        fit.run_fit(
            train_manifest=tmp_path / "train.json",
            validation_manifest=tmp_path / "validation.json",
            config_path=tmp_path / "config.yaml",
            bounds=[ParameterBound("tendon.cable_stiffness_n_per_m.THJ2", 1, 2)],
            tolerances={"angle_deg": 1, "motor_angle_deg": 1, "tension_N": 1},
            weights={},
            out_dir=tmp_path / "overlapping-fit",
            max_evaluations=5,
        )


def test_fit_rejects_distinct_exports_from_same_source_recording(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        fit,
        "_resolve_dataset",
        lambda _path, split: (
            {
                "dataset_id": f"{split}-id",
                "split": split,
                "joints": ["THJ2"],
                "recording": {"source_recording_id": "same-capture", "source_sha256": "d" * 64},
                "experiment": {"root_pose": {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}},
            },
            tmp_path / f"{split}.csv",
            {"metrics": {"csv_sha256": split}},
        ),
    )
    monkeypatch.setattr(fit, "_require_optimizer_dependency", lambda: None)

    with pytest.raises(ValueError, match="independent source recordings"):
        fit.run_fit(
            train_manifest=tmp_path / "train.json",
            validation_manifest=tmp_path / "validation.json",
            config_path=tmp_path / "config.yaml",
            bounds=[ParameterBound("tendon.cable_stiffness_n_per_m.THJ2", 1, 2)],
            tolerances={"angle_deg": 1, "motor_angle_deg": 1, "tension_N": 1},
            weights={},
            out_dir=tmp_path / "same-source-fit",
            max_evaluations=5,
        )


@pytest.mark.parametrize(
    ("heldout_score", "heldout_candidate_pass", "expected_accepted"),
    [(1.0, True, True), (2.0, True, False), (0.5, False, False)],
)
def test_fit_requires_heldout_pass_and_improvement_over_validation_baseline(
    tmp_path: Path, monkeypatch, heldout_score: float, heldout_candidate_pass: bool, expected_accepted: bool
):
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    datasets = {
        "train": (
            {
                "dataset_id": "train-1",
                "split": "train",
                "joints": ["THJ2"],
                "recording": {"source_recording_id": "source-train", "source_sha256": "b" * 64},
                "experiment": {"root_pose": {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}},
            },
            train_csv,
        ),
        "validation": (
            {
                "dataset_id": "validation-1",
                "split": "validation",
                "joints": ["THJ2"],
                "recording": {"source_recording_id": "source-validation", "source_sha256": "c" * 64},
                "experiment": {"root_pose": {"position_m": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}},
            },
            validation_csv,
        ),
    }

    def resolve_dataset(_manifest_path, split):
        manifest, csv_path = datasets[split]
        return manifest, csv_path, {"fit_ready": True, "metrics": {"csv_sha256": split}}

    monkeypatch.setattr(fit, "_resolve_dataset", resolve_dataset)
    monkeypatch.setattr(fit, "_check_parameters", lambda *_args: None)
    monkeypatch.setattr(fit, "_require_optimizer_dependency", lambda: None)
    seen_poses = []

    def replay_candidate(_dataset_csv, _config_path, _parameters, output_dir, **kwargs):
        seen_poses.append((output_dir.name, kwargs["root_pose"]))
        output_dir.mkdir(parents=True)
        sim_csv = output_dir / "sim.csv"
        metadata = output_dir / "sim.metadata.json"
        log = output_dir / "replay.log"
        sim_csv.write_text("sim", encoding="utf-8")
        metadata.write_text("{}", encoding="utf-8")
        log.write_text("ok", encoding="utf-8")
        return sim_csv, metadata, 0, str(log)

    monkeypatch.setattr(fit, "_replay_candidate", replay_candidate)

    def evaluate_files(_real, sim, _metadata, _joints, **_kwargs):
        label = sim.parent.name
        score = 2.0 if label in {"baseline_train_0001", "baseline_validation_0002"} else 1.0
        candidate_pass = True
        if label.startswith("validation_") and label != "baseline_validation_0002":
            score = heldout_score
            candidate_pass = heldout_candidate_pass
        return {
            "quality": {
                "quality_valid": True,
                "tolerance_pass": candidate_pass,
                "candidate_pass": candidate_pass,
                "score": score,
                "channels": {
                    "THJ2/angle_deg": {
                        "rmse": score,
                        "normalized_rmse": score / 2,
                    }
                },
                "reasons": [] if candidate_pass else ["above tolerance"],
            }
        }

    monkeypatch.setattr(fit, "evaluate_files", evaluate_files)

    def search(objective, bounds, *, max_evaluations, seed):
        parameters = {bounds[0].path: 0.5}
        objective(parameters)
        return {
            "best_parameters": parameters,
            "best_score": 1.0,
            "evaluation_count": 1,
            "seed": seed,
            "max_evaluations": max_evaluations,
        }

    monkeypatch.setattr(fit, "fit_parameters", search)
    out_dir = tmp_path / "fit-result"

    result = fit.run_fit(
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        config_path=tmp_path / "config.yaml",
        bounds=[ParameterBound("tendon.cable_stiffness_n_per_m.THJ2", 1, 2)],
        tolerances={"angle_deg": 1, "motor_angle_deg": 1, "tension_N": 1},
        weights={},
        out_dir=out_dir,
        max_evaluations=5,
    )

    assert (out_dir / "fit.json").is_file()
    saved = json.loads((out_dir / "fit.json").read_text(encoding="utf-8"))
    assert saved["baseline_validation"]["score"] == 2.0
    assert saved["heldout_validation"]["score"] == heldout_score
    assert result["accepted"] is expected_accepted
    assert result["validation_pass"] is heldout_candidate_pass
    assert result["heldout_score_change"] == heldout_score - 2.0
    assert result["heldout_channel_changes"]["THJ2/angle_deg"]["rmse_change"] == heldout_score - 2.0
    assert seen_poses[0][1]["position_m"] == [0, 0, 0]
    assert seen_poses[1][1]["position_m"] == [1, 0, 0]
