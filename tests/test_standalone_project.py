"""Guard package isolation and local assets after extraction."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from botyard_real2sim.assets import BOTYARD_REAL2SIM_ASSETS_DIR
from botyard_real2sim.paths import checkout_root, default_config_path

ROOT = Path(__file__).resolve().parents[1]


def test_local_configuration_and_flattened_asset_are_self_contained():
    assert checkout_root() == ROOT
    assert default_config_path() == ROOT / "configs/real2sim/s1.yaml"
    usd = BOTYARD_REAL2SIM_ASSETS_DIR / "hand/boya.usda"
    assert usd.is_relative_to(ROOT)
    assert not usd.is_symlink()
    text = usd.read_text()
    assert text.startswith("#usda") and usd.stat().st_size > 1_000_000
    assert text.count('def PhysicsRevoluteJoint "') == 22
    assert "@" not in text  # Flattened hand carries its own meshes; no external asset references.


def test_source_has_no_import_dependency_on_original_project_or_rl_tasks():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not any(
                name.split(".")[0] in {"dexterous_hand_manipulation", "isaaclab_tasks", "isaaclab_rl"}
                for name in modules
            ), path


def test_offline_entry_points_work_outside_checkout_with_engine_imports_forbidden(tmp_path):
    code = r"""
import importlib.abc, runpy, sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"dexterous_hand_manipulation", "isaaclab", "isaacsim", "torch"}:
            raise AssertionError("offline entry point imported " + fullname)
sys.meta_path.insert(0, BlockExternal())
from botyard_real2sim.cli import replay, sweep
assert replay.main(["--write-default-config", sys.argv[1]]) == 0
assert sweep.main(["--parameter", "tendon.routing_mass_kg.FFJ2", "--values", "0.002",
                   "--out-dir", sys.argv[2], "--dry-run", "--", "--joint", "FFJ2"]) == 0
assert not any(name.startswith("dexterous_hand_manipulation") for name in sys.modules)
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "params.yaml"), str(tmp_path / "sweep")],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "params.yaml").is_file()
    assert (tmp_path / "sweep/manifest.json").is_file()
