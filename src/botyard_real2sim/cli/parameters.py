"""Print the machine-readable parameter catalog for an effective config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from botyard_real2sim.parameter_catalog import describe_parameters_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Resolved model YAML (defaults to the bundled S1 preset)")
    parser.add_argument("--out", type=Path, help="Write JSON here; stdout is used by default")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = describe_parameters_from_path(str(args.config) if args.config else None)
        rendered = json.dumps(catalog, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
