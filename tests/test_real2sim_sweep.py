"""Sweep planning and subprocess handling are testable without simulation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from botyard_real2sim.cli import sweep


def _plan(tmp_path, **kwargs):
    return sweep.plan_sweep(
        "tendon.cable_stiffness_n_per_m.FFJ2",
        [4000, 8000],
        tmp_path / "runs",
        ["--joint", "FFJ2", "--config", "config with spaces.yaml"],
        **kwargs,
    )


def _successful_child(argv, **kwargs):
    output = Path(argv[argv.index("--out") + 1])
    output.write_text("result\n")
    if "--report" in argv:
        Path(argv[argv.index("--report") + 1]).write_text('{"joints": {}}')
    kwargs["stdout"].write("fake child completed\n")
    return subprocess.CompletedProcess(argv, 0)


def test_plan_is_argv_lists_same_interpreter_and_per_run_outputs(tmp_path):
    plan = _plan(tmp_path, real=tmp_path / "real capture.csv")
    first, second = plan["runs"]
    assert first["replay_command"][0] == sys.executable
    assert first["comparison_command"][0] == sys.executable
    assert "config with spaces.yaml" in first["replay_command"]
    assert "config with spaces.yaml" in first["comparison_command"]
    assert "tendon.cable_stiffness_n_per_m.FFJ2=4000" in first["replay_command"]
    assert "tendon.cable_stiffness_n_per_m.FFJ2=8000" in second["replay_command"]
    assert first["sim_csv"] != second["sim_csv"]
    assert "not automatic calibration" in plan["purpose"]
    assert not Path(plan["out_dir"]).exists()


def test_dry_run_writes_manifest_without_calling_runner(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run launched a subprocess")

    plan = _plan(tmp_path)
    sweep.execute_plan(plan, dry_run=True, runner=forbidden)
    saved = json.loads((Path(plan["out_dir"]) / "manifest.json").read_text())
    assert saved["dry_run"] is True
    assert all(run["status"] == "planned" for run in saved["runs"])
    assert not Path(plan["runs"][0]["run_dir"]).exists()


@pytest.mark.parametrize(
    "parameter,values",
    [("bad", [1]), ("tendon.a;rm", [1]), ("tendon.a", []), ("tendon.a", [float("nan")]), ("tendon.a", [float("inf")])],
)
def test_bad_parameters_or_nonfinite_values_rejected(tmp_path, parameter, values):
    with pytest.raises(ValueError):
        sweep.plan_sweep(parameter, values, tmp_path, [])


@pytest.mark.parametrize("arguments", [["--out", "mine.csv"], ["--out=mine.csv"], ["--help"]])
def test_managed_output_or_help_arguments_rejected(tmp_path, arguments):
    with pytest.raises(ValueError):
        sweep.plan_sweep("tendon.foo", [1], tmp_path, arguments)


def test_successful_runs_capture_logs_and_comparison(tmp_path):
    plan = _plan(tmp_path, real=tmp_path / "real.csv")
    sweep.execute_plan(plan, runner=_successful_child)
    assert all(run["status"] == "completed" for run in plan["runs"])
    assert all(Path(run["replay_log"]).read_text() == "fake child completed\n" for run in plan["runs"])
    assert all((Path(run["run_dir"]) / "comparison.json").exists() for run in plan["runs"])


def test_failed_child_is_recorded_and_remaining_runs_continue(tmp_path):
    plan = _plan(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert "shell" not in kwargs
        if len(calls) == 1:
            kwargs["stdout"].write("failure detail\n")
            return subprocess.CompletedProcess(argv, 7)
        return _successful_child(argv, **kwargs)

    sweep.execute_plan(plan, runner=runner)
    assert [r["status"] for r in plan["runs"]] == ["replay_failed", "completed"]
    assert plan["runs"][0]["replay_returncode"] == 7
    assert "failure detail" in Path(plan["runs"][0]["replay_log"]).read_text()
    saved = json.loads((Path(plan["out_dir"]) / "manifest.json").read_text())
    assert saved == plan


def test_launch_error_and_missing_artifact_are_failures(tmp_path):
    plan = _plan(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise OSError("interpreter unavailable")
        return subprocess.CompletedProcess(argv, 0)

    sweep.execute_plan(plan, runner=runner)
    assert all(r["status"] == "replay_failed" for r in plan["runs"])
    assert "interpreter unavailable" in plan["runs"][0]["error"]
    assert "did not produce" in plan["runs"][1]["error"]


def test_collision_refused_and_explicit_overwrite_cannot_reuse_stale_csv(tmp_path):
    plan = _plan(tmp_path)
    sweep.execute_plan(plan, runner=_successful_child)
    with pytest.raises(ValueError, match="collision"):
        sweep.execute_plan(_plan(tmp_path), dry_run=True)
    rerun = _plan(tmp_path)
    sweep.execute_plan(rerun, overwrite=True, runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0))
    assert all(r["status"] == "replay_failed" for r in rerun["runs"])
    assert not any(Path(r["sim_csv"]).exists() for r in rerun["runs"])


def test_cli_dry_run_prints_and_saves_plan(tmp_path, capsys):
    assert (
        sweep.main(
            [
                "--parameter",
                "controller.dead_zone_deg.FFJ2",
                "--values",
                "1",
                "2",
                "--out-dir",
                str(tmp_path / "cli"),
                "--dry-run",
                "--",
                "--joint",
                "FFJ2",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert "--joint" in plan["runs"][0]["replay_command"]


def test_comparison_failure_does_not_hide_successful_simulation(tmp_path):
    plan = _plan(tmp_path, real=tmp_path / "real.csv")

    def runner(argv, **kwargs):
        if argv[1:3] == ["-m", "botyard_real2sim.cli.curves"]:
            kwargs["stdout"].write("comparison error\n")
            return subprocess.CompletedProcess(argv, 2)
        return _successful_child(argv, **kwargs)

    sweep.execute_plan(plan, runner=runner)
    assert all(run["status"] == "comparison_failed" for run in plan["runs"])
    assert all(run["replay_returncode"] == 0 and Path(run["sim_csv"]).is_file() for run in plan["runs"])
