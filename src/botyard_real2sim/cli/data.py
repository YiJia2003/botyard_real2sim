"""Inspect whether a recorded real2sim data set is fit-ready."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from botyard_real2sim.datasets import validate_dataset, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Dataset manifest JSON")
    parser.add_argument("--csv", type=Path, help="Override the CSV path from the manifest")
    parser.add_argument(
        "--strict", action="store_true", help="Promote fit blockers to errors and return a failing exit code"
    )
    parser.add_argument("--min-samples", type=int, help="Override minimum finite samples per required channel")
    parser.add_argument("--min-duration-s", type=float, help="Override minimum recording duration")
    parser.add_argument(
        "--min-channel-coverage", type=float, help="Override minimum per-channel time coverage, in [0,1]"
    )
    parser.add_argument("--max-gap-s", type=float, help="Largest adjacent sample gap counted as covered")
    parser.add_argument("--out", type=Path, help="Write the JSON report to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in (
            ("min_samples", args.min_samples),
            ("min_duration_s", args.min_duration_s),
            ("min_channel_coverage", args.min_channel_coverage),
            ("max_gap_s", args.max_gap_s),
        )
        if value is not None
    }
    report = validate_dataset(args.manifest, csv_path=args.csv, strict=args.strict, overrides=overrides)
    if args.out:
        write_report(report, args.out)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if not report["valid"] or (args.strict and not report["data_ready"]):
        return 2
    if report["blocking_reasons"]:
        print("dataset is structurally valid but not fit-ready; inspect blocking_reasons", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
