"""Evaluate a real2sim replay as a calibration candidate.

This command writes machine-readable quality gates and normalized curve error.
It does not certify physical truth or choose model structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from botyard_real2sim.cli.curves import CurveOptions, build_report, read_curve
from botyard_real2sim.evaluation import EvaluationPolicy, score_report


def _mapping(values: list[str], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for assignment in values:
        key, separator, raw = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"{name} entries must use NAME=VALUE")
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"{name}[{key}] must be a number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}[{key}] must be positive and finite")
        if key in result:
            raise ValueError(f"duplicate {name} entry: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True, help="canonical real trace CSV")
    parser.add_argument("--sim", type=Path, required=True, help="replay trace CSV")
    parser.add_argument("--metadata", type=Path, help="replay metadata; defaults to the replay sidecar")
    parser.add_argument("--joints", nargs="+", required=True, help="joint names to score")
    parser.add_argument(
        "--tolerance",
        action="append",
        default=[],
        metavar="CHANNEL=VALUE",
        help="explicit RMSE tolerance: angle_deg, motor_angle_deg, tension_N",
    )
    parser.add_argument("--weight", action="append", default=[], metavar="CHANNEL=VALUE")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-overlap-s", type=float, default=0.5)
    parser.add_argument("--max-mimic-error-rad", type=float, default=0.01)
    parser.add_argument("--align", choices=("command_edge", "start"), default="command_edge")
    parser.add_argument("--max-interp-gap-s", type=float, default=0.1)
    parser.add_argument("--real-tension-bias-N", type=float, default=0.0)
    parser.add_argument("--tension-raw-to-N", type=float, default=0.01)
    parser.add_argument("--out", type=Path, help="JSON output; defaults to stdout")
    return parser


def evaluate_files(
    real_path: str | Path,
    sim_path: str | Path,
    metadata_path: str | Path | None,
    joints: list[str] | tuple[str, ...],
    *,
    tolerances: dict[str, float],
    weights: dict[str, float] | None = None,
    min_samples: int = 100,
    min_coverage: float = 0.8,
    min_overlap_s: float = 0.5,
    max_mimic_error_rad: float = 0.01,
    alignment: str = "command_edge",
    max_interp_gap_s: float = 0.1,
    real_tension_bias_n: float = 0.0,
    tension_raw_to_n: float = 0.01,
) -> dict:
    """Build a curve report, apply strict quality gates, and attach input hashes."""
    real_path, sim_path = Path(real_path).resolve(), Path(sim_path).resolve()
    metadata_path = Path(metadata_path).resolve() if metadata_path else sim_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    real = read_curve(real_path)
    sim = read_curve(sim_path)
    options = CurveOptions(
        alignment=alignment,
        max_interp_gap_s=max_interp_gap_s,
        real_tension_bias_N=real_tension_bias_n,
        tension_raw_to_N=tension_raw_to_n,
    )
    curve_report, _ = build_report(real, sim, joints, options)
    policy = EvaluationPolicy(
        channel_scales=tolerances,
        channel_weights=weights or {},
        min_samples=min_samples,
        min_coverage_ratio=min_coverage,
        min_overlap_s=min_overlap_s,
        max_mimic_error_rad=max_mimic_error_rad,
    )
    quality = score_report(curve_report, policy, replay_metadata=metadata)
    return {
        "schema_version": 1,
        "real_csv": str(real_path),
        "real_csv_sha256": _sha256(real_path),
        "sim_csv": str(sim_path),
        "sim_csv_sha256": _sha256(sim_path),
        "replay_metadata": str(metadata_path) if metadata is not None else None,
        "replay_metadata_sha256": _sha256(metadata_path) if metadata is not None else None,
        "curve_report": curve_report,
        "quality": quality,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_files(
            args.real,
            args.sim,
            args.metadata,
            args.joints,
            tolerances=_mapping(args.tolerance, "tolerance"),
            weights=_mapping(args.weight, "weight"),
            min_samples=args.min_samples,
            min_coverage=args.min_coverage,
            min_overlap_s=args.min_overlap_s,
            max_mimic_error_rad=args.max_mimic_error_rad,
            alignment=args.align,
            max_interp_gap_s=args.max_interp_gap_s,
            real_tension_bias_n=args.real_tension_bias_N,
            tension_raw_to_n=args.tension_raw_to_N,
        )
        rendered = json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"botyard-evaluate: {error}\n")
    return 0 if result["quality"]["candidate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
