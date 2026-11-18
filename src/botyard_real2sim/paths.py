"""Resolve checkout configuration and packaged presets without importing the engine."""

from pathlib import Path


def checkout_root() -> Path | None:
    """Locate this source checkout; installed wheels have no checkout root."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src/botyard_real2sim").is_dir():
        return candidate
    return None


def default_config_path() -> Path:
    """Prefer the editable checkout YAML; wheels carry the same preset as package data."""
    root = checkout_root()
    if root is not None:
        return root / "configs/real2sim/s1.yaml"
    return Path(__file__).resolve().parent / "presets/s1.yaml"
