"""Fit bounded model parameters on a training recording and check held-out data.

Every candidate launches the existing PhysX replay and strict evaluator. The
search is serial and budgeted because each replay starts Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from botyard_real2sim.cli.evaluate import _mapping, evaluate_files
from botyard_real2sim.config import load_config
from botyard_real2sim.datasets import validate_dataset
from botyard_real2sim.evaluation import EvaluationPolicy
from botyard_real2sim.optimizer import ParameterBound, fit_parameters
from botyard_real2sim.parameter_catalog import describe_parameters
from botyard_real2sim.paths import default_config_path


def _parameter_bound(value: str) -> ParameterBound:
    path, separator, range_value = value.partition("=")
    parts = range_value.split(":")
    if not separator or len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("parameter must be PATH=LOWER:UPPER[:linear|log]")
    try:
        lower, upper = float(parts[0]), float(parts[1])
        scale = parts[2] if len(parts) == 3 else "linear"
        return ParameterBound(path, lower, upper, scale)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest root must be a JSON object")
    return value


def _resolve_dataset(manifest_path: Path, expected_split: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = _manifest(manifest_path)
    if manifest.get("split") != expected_split:
        raise ValueError(f"{manifest_path}: expected split={expected_split!r}, got {manifest.get('split')!r}")
    validation = validate_dataset(manifest_path, strict=True)
    ready_key = "fit_ready" if expected_split == "train" else "evaluation_ready"
    if not validation[ready_key]:
        reasons = validation.get("errors") or validation.get("blocking_reasons")
        readiness = "fit-ready" if expected_split == "train" else "evaluation-ready"
        raise ValueError(f"{manifest_path}: dataset is not {readiness}: {json.dumps(reasons, ensure_ascii=False)}")
    csv_path = Path(validation["csv"]).resolve()
    return manifest, csv_path, validation


def _check_parameters(bounds: list[ParameterBound], config_path: Path) -> None:
    catalog = describe_parameters(load_config(config_path))
    by_path = {item["path"]: item for item in catalog["parameters"]}
    for bound in bounds:
        item = by_path.get(bound.path)
        if item is None:
            raise ValueError(f"parameter is absent from the active parameter catalog: {bound.path}")
        if item["kind"] not in {"physical", "physical_geometry"}:
            raise ValueError(f"parameter is not a physical model parameter: {bound.path} ({item['kind']})")
        if item["active_status"] not in {"active", "partially_active", "conditional"}:
            raise ValueError(f"parameter is not active in this configuration: {bound.path} ({item['active_status']})")
        if item["tunability"].startswith(("unsupported", "lock", "hold_fixed", "guard")):
            raise ValueError(f"parameter is locked or unsupported by this model: {bound.path} ({item['tunability']})")


def _require_optimizer_dependency() -> None:
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError("SciPy is required for parameter fitting; install the calibration extra.")


def _replay_candidate(
    dataset_csv: Path,
    config_path: Path,
    parameters: dict[str, float],
    output_dir: Path,
    *,
    root_pose: dict[str, Any],
    physics_dt: float,
    sample_dt: float,
    device: str,
    duration: float | None,
    timeout_s: float,
    runner=subprocess.run,
) -> tuple[Path, Path, int, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    sim_csv = output_dir / "sim.csv"
    command = [
        sys.executable,
        "-m",
        "botyard_real2sim.cli.replay",
        "--commands",
        str(dataset_csv),
        "--strict-commands",
        "--command-frames",
        "auto",
        "--config",
        str(config_path),
        "--physics-dt",
        str(physics_dt),
        "--sample-dt",
        str(sample_dt),
        "--device",
        device,
        "--out",
        str(sim_csv),
        "--root-pos",
        *(f"{value:.17g}" for value in root_pose["position_m"]),
        "--root-xyzw",
        *(f"{value:.17g}" for value in root_pose["quaternion_xyzw"]),
    ]
    if duration is not None:
        command += ["--duration", str(duration)]
    for path, value in parameters.items():
        command += ["--set", f"{path}={value:.17g}"]
    log_path = output_dir / "replay.log"
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        log_path.write_text(stdout + ("\n" if stdout and stderr else "") + stderr, encoding="utf-8")
        return sim_csv, sim_csv.with_suffix(".metadata.json"), completed.returncode, str(log_path)
    except (OSError, subprocess.TimeoutExpired) as error:
        log_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        return sim_csv, sim_csv.with_suffix(".metadata.json"), 124, str(log_path)


def run_fit(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    config_path: Path,
    bounds: list[ParameterBound],
    tolerances: dict[str, float],
    weights: dict[str, float],
    out_dir: Path,
    max_evaluations: int,
    seed: int = 0,
    min_samples: int = 100,
    min_coverage: float = 0.8,
    min_overlap_s: float = 0.5,
    physics_dt: float = 0.0001,
    sample_dt: float = 0.003,
    device: str = "cuda:0",
    duration: float | None = None,
    timeout_s: float = 1800,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Run a bounded search and a final replay on a separately declared holdout."""
    if out_dir.exists():
        raise ValueError(f"output directory already exists: {out_dir}")
    if train_manifest.resolve() == validation_manifest.resolve():
        raise ValueError("training and validation manifests must be different files")
    if not bounds:
        raise ValueError("at least one parameter bound is required")
    if len({bound.path for bound in bounds}) != len(bounds):
        raise ValueError("parameter paths must be unique")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 5 * len(bounds):
        raise ValueError(f"max_evaluations must be at least 5 * parameter count ({5 * len(bounds)})")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if min_samples < 1 or not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("invalid minimum sample or coverage gate")
    if not math.isfinite(min_overlap_s) or min_overlap_s <= 0:
        raise ValueError("min_overlap_s must be positive and finite")
    if not math.isfinite(physics_dt) or physics_dt <= 0 or not math.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("physics_dt and sample_dt must be positive and finite")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive and finite")
    required_channels = ("angle_deg", "motor_angle_deg", "tension_N")
    missing_tolerances = set(required_channels) - set(tolerances)
    if missing_tolerances:
        raise ValueError(f"explicit tolerances are required for: {', '.join(sorted(missing_tolerances))}")
    EvaluationPolicy(
        channel_scales=tolerances,
        channel_weights=weights,
        min_samples=min_samples,
        min_coverage_ratio=min_coverage,
        min_overlap_s=min_overlap_s,
    )
    _require_optimizer_dependency()

    train_info, train_csv, train_validation = _resolve_dataset(train_manifest, "train")
    validation_info, validation_csv, validation_validation = _resolve_dataset(validation_manifest, "validation")
    if train_info["dataset_id"] == validation_info["dataset_id"]:
        raise ValueError("training and validation dataset_id values must differ")
    if train_csv.resolve() == validation_csv.resolve():
        raise ValueError("training and validation manifests must reference different CSV files")
    train_csv_sha256 = train_validation.get("metrics", {}).get("csv_sha256")
    validation_csv_sha256 = validation_validation.get("metrics", {}).get("csv_sha256")
    if train_csv_sha256 == validation_csv_sha256:
        raise ValueError("training and validation manifests must reference independent recordings")
    train_source = train_info["recording"]
    validation_source = validation_info["recording"]
    if train_source["source_recording_id"] == validation_source["source_recording_id"]:
        raise ValueError("training and validation manifests must reference independent source recordings")
    if train_source["source_sha256"] == validation_source["source_sha256"]:
        raise ValueError("training and validation manifests must reference independent source recordings")
    _check_parameters(bounds, config_path)
    joints = train_info["joints"]
    if validation_info["joints"] != joints:
        raise ValueError("training and validation manifests must select the same ordered joints")

    out_dir.mkdir(parents=True)
    run_index = 0

    def candidate(
        parameters: dict[str, float],
        dataset_csv: Path,
        dataset_info: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        nonlocal run_index
        run_index += 1
        output_dir = out_dir / f"{label}_{run_index:04d}"
        sim_csv, metadata_path, returncode, log_path = _replay_candidate(
            dataset_csv,
            config_path,
            parameters,
            output_dir,
            root_pose=dataset_info["experiment"]["root_pose"],
            physics_dt=physics_dt,
            sample_dt=sample_dt,
            device=device,
            duration=duration,
            timeout_s=timeout_s,
            runner=runner,
        )
        result: dict[str, Any] = {
            "parameters": parameters,
            "replay_returncode": returncode,
            "replay_log": str(Path(log_path).resolve()),
            "quality_valid": False,
            "candidate_pass": False,
            "score": None,
            "channel_metrics": {},
            "reasons": [],
        }
        if returncode != 0:
            result["reasons"].append(f"replay exited with status {returncode}")
            return result
        try:
            evaluation = evaluate_files(
                dataset_csv,
                sim_csv,
                metadata_path,
                joints,
                tolerances=tolerances,
                weights=weights,
                min_samples=min_samples,
                min_coverage=min_coverage,
                min_overlap_s=min_overlap_s,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result["reasons"].append(f"evaluation failed: {type(error).__name__}: {error}")
            return result
        evaluation_path = output_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        result.update(
            quality_valid=evaluation["quality"]["quality_valid"],
            candidate_pass=evaluation["quality"]["candidate_pass"],
            tolerance_pass=evaluation["quality"]["tolerance_pass"],
            score=evaluation["quality"]["score"],
            channel_metrics=evaluation["quality"]["channels"],
            reasons=evaluation["quality"]["reasons"],
            evaluation=str(evaluation_path.resolve()),
        )
        return result

    baseline_train = candidate({}, train_csv, train_info, "baseline_train")
    baseline_validation = candidate({}, validation_csv, validation_info, "baseline_validation")

    def objective(parameters: dict[str, float]) -> dict[str, Any]:
        result = candidate(parameters, train_csv, train_info, "train")
        return {
            "quality_valid": result["quality_valid"],
            "score": result["score"],
            "reasons": result["reasons"],
            "evaluation": result.get("evaluation"),
            "replay_log": result["replay_log"],
        }

    optimizer_result = fit_parameters(
        objective,
        bounds,
        max_evaluations=max_evaluations,
        seed=seed,
    )
    best_parameters = optimizer_result["best_parameters"]
    heldout = (
        candidate(best_parameters, validation_csv, validation_info, "validation")
        if best_parameters is not None
        else None
    )
    heldout_score = heldout["score"] if heldout is not None else None
    baseline_validation_score = baseline_validation["score"]
    baseline_channels = baseline_validation["channel_metrics"]
    heldout_channels = heldout["channel_metrics"] if heldout is not None else {}
    heldout_channel_changes = {
        key: {
            "rmse_change": heldout_channels[key]["rmse"] - baseline_metric["rmse"],
            "normalized_rmse_change": (heldout_channels[key]["normalized_rmse"] - baseline_metric["normalized_rmse"]),
        }
        for key, baseline_metric in baseline_channels.items()
        if key in heldout_channels
        and baseline_metric.get("rmse") is not None
        and heldout_channels[key].get("rmse") is not None
        and baseline_metric.get("normalized_rmse") is not None
        and heldout_channels[key].get("normalized_rmse") is not None
    }
    improved = (
        heldout is not None
        and heldout["candidate_pass"]
        and baseline_validation["quality_valid"]
        and heldout_score is not None
        and baseline_validation_score is not None
        and heldout_score < baseline_validation_score
    )
    result = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "status": "completed" if heldout is not None else "no_valid_candidate",
        "claims_scope": "curve_fit_candidate_only; hardware/model truth requires experimental evidence",
        "training": {
            "manifest": str(train_manifest.resolve()),
            "dataset_id": train_info["dataset_id"],
            "split": train_info["split"],
            "validation": train_validation,
        },
        "validation_set": {
            "manifest": str(validation_manifest.resolve()),
            "dataset_id": validation_info["dataset_id"],
            "split": validation_info["split"],
            "validation": validation_validation,
        },
        "config": str(config_path.resolve()),
        "joints": joints,
        "tolerances": tolerances,
        "weights": weights,
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "optimizer": optimizer_result,
        "best_parameters": best_parameters,
        "heldout_validation": heldout,
        "validation_pass": bool(heldout and heldout["candidate_pass"]),
        "improved": improved,
        "accepted": improved,
        "heldout_score_change": (
            heldout_score - baseline_validation_score
            if heldout_score is not None and baseline_validation_score is not None
            else None
        ),
        "heldout_channel_changes": heldout_channel_changes,
        "decision": (
            "candidate passes held-out gates and improves the validation baseline"
            if improved
            else "candidate is not accepted; it must pass held-out gates and improve the validation baseline"
        ),
    }
    out_dir.joinpath("fit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--parameter", type=_parameter_bound, action="append", required=True, metavar="PATH=LOW:HIGH[:SCALE]"
    )
    parser.add_argument("--tolerance", action="append", default=[], metavar="CHANNEL=VALUE", required=True)
    parser.add_argument("--weight", action="append", default=[], metavar="CHANNEL=VALUE")
    parser.add_argument("--max-evaluations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-overlap-s", type=float, default=0.5)
    parser.add_argument("--physics-dt", type=float, default=0.0001)
    parser.add_argument("--sample-dt", type=float, default=0.003)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tolerances = _mapping(args.tolerance, "tolerance")
        weights = _mapping(args.weight, "weight")
        result = run_fit(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            config_path=args.config,
            bounds=args.parameter,
            tolerances=tolerances,
            weights=weights,
            out_dir=args.out_dir,
            max_evaluations=args.max_evaluations,
            seed=args.seed,
            min_samples=args.min_samples,
            min_coverage=args.min_coverage,
            min_overlap_s=args.min_overlap_s,
            physics_dt=args.physics_dt,
            sample_dt=args.sample_dt,
            device=args.device,
            duration=args.duration,
            timeout_s=args.timeout_s,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.exit(2, f"botyard-fit: {error}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
