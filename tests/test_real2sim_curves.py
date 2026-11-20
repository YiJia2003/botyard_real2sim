"""Small end-to-end checks for real/virtual trace alignment and tension units."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from botyard_real2sim.cli import curves


def _csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _pair(tmp_path: Path, *, sim_tension_bias: float = 0.0, sim_target_rad: float = 1.0):
    real = _csv(
        tmp_path / "real.csv",
        ["timestamp_ns", "THJ1/position_rad", "THJ1/effort_raw", "THJ1/cmd_rad"],
        [
            [1_000_000_000, 0, 0, 0],
            [1_100_000_000, 0, 0, 0],
            [1_200_000_000, 0, 100, 1],
            [1_300_000_000, 0.5, 200, 1],
            [1_400_000_000, 1, 300, 1],
        ],
    )
    sim = _csv(
        tmp_path / "sim.csv",
        ["time_s", "THJ1/position_rad", "THJ1/tension_N", "THJ1/cmd_rad"],
        [
            [0.0, 0, 0, 0],
            [0.1, 0, 0, 0],
            [0.2, 0, 0, 0],
            [0.3, 0, 1 + sim_tension_bias, sim_target_rad],
            [0.4, 0.5, 2 + sim_tension_bias, sim_target_rad],
            [0.5, 1, 3 + sim_tension_bias, sim_target_rad],
        ],
    )
    return curves.read_curve(real), curves.read_curve(sim)


def test_ros_effort_is_proximal_tension_and_command_edges_align(tmp_path: Path):
    real, sim = _pair(tmp_path)
    options = curves.CurveOptions()

    assert real.values("THJ1", "tension_N") == [0, 0, 1, 2, 3]
    report, aligned = curves.build_report(real, sim, ("THJ1",), options)

    assert aligned["real"][2] == pytest.approx(0)
    assert aligned["sim"][3] == pytest.approx(0)
    metrics = report["joints"]["THJ1"]
    assert metrics["angle_deg"]["rmse"] == pytest.approx(0, abs=1e-12)
    assert metrics["tension_N"]["rmse"] == pytest.approx(0, abs=1e-12)
    assert metrics["command_angle_deg"]["rmse"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("extension", ["png", "html"])
def test_three_observable_curves_and_motor_residual_are_independent(tmp_path, extension):
    header = ["time_s", "FFJ2/position_rad", "FFJ2/motor_position_rad", "FFJ2/tension_N"]
    real = curves.read_curve(_csv(tmp_path / "real.csv", header, [[0, 0.1, 1.0, 3], [0.01, 0.2, 1.2, 4]]))
    sim = curves.read_curve(_csv(tmp_path / "sim.csv", header, [[0, 0.1, 1.1, 3], [0.01, 0.2, 1.3, 4]]))
    options = curves.CurveOptions(alignment="start")
    report, aligned = curves.build_report(real, sim, ["FFJ2"], options)
    metrics = report["joints"]["FFJ2"]
    assert metrics["angle_deg"]["rmse"] == 0
    assert metrics["motor_angle_deg"]["bias"] == pytest.approx(math.degrees(0.1))
    assert metrics["tension_N"]["rmse"] == 0
    assert curves._plot_channels(real, sim, ["FFJ2"], options) == ["angle", "motor", "tension"]
    output = tmp_path / f"three.{extension}"
    curves.plot_curves(output, real, sim, ["FFJ2"], aligned, options)
    assert output.stat().st_size > 1000


def test_kinematic_derivatives_are_explicit_and_do_not_bridge_gaps(tmp_path):
    table = curves.read_curve(
        _csv(
            tmp_path / "motion.csv",
            ["time_s", "FFJ2/position_rad", "FFJ2/motor_position_rad"],
            [[0, 0, 0], [0.01, 0.02, 0.04], [0.02, 0.04, 0.08], [0.5, 1, 2]],
        )
    )
    default = curves.CurveOptions(alignment="start")
    assert curves.channel_values(table, "FFJ2", "motor_velocity_rad_s", default) is None
    derive = curves.CurveOptions(alignment="start", derive_kinematics=True)
    velocity = curves.channel_values(table, "FFJ2", "motor_velocity_rad_s", derive)
    assert math.isnan(velocity[0]) and math.isnan(velocity[-1])
    assert velocity[1:3] == pytest.approx([4, 4])
    acceleration = curves.channel_values(table, "FFJ2", "motor_acceleration_rad_s2", derive)
    assert math.isnan(acceleration[0]) and math.isnan(acceleration[1]) and math.isnan(acceleration[-1])
    assert acceleration[2] == pytest.approx(0)


def test_distal_tension_and_friction_channels_do_not_reuse_motor_sensor(tmp_path):
    real = curves.read_curve(
        _csv(tmp_path / "real.csv", ["time_s", "FFJ2/position_rad", "FFJ2/tension_N"], [[0, 0, 3], [0.01, 0, 4]])
    )
    sim = curves.read_curve(
        _csv(
            tmp_path / "sim.csv",
            [
                "time_s",
                "FFJ2/position_rad",
                "FFJ2/tension_N",
                "FFJ2/distal_tension_N",
                "FFJ2/routing_tension_drop_N",
                "FFJ2/routing_contact_state",
                "FFJ2/routing_bristle_proximal_m",
            ],
            [[0, 0, 3, 2, 1, 0.2, 0.0001], [0.01, 0, 4, 2.5, 1.5, 0.3, -0.0002]],
        )
    )
    options = curves.CurveOptions(alignment="start")
    report, aligned = curves.build_report(real, sim, ["FFJ2"], options)
    assert curves._plot_channels(real, sim, ["FFJ2"], options) == ["angle", "tension", "distal_tension"]
    assert report["joints"]["FFJ2"]["tension_N"]["rmse"] == 0
    assert report["joints"]["FFJ2"]["distal_tension_N"] is None
    assert curves.residual_series(real, sim, "FFJ2", "distal_tension_N", aligned, options) == ([], [])
    traces = list(curves._display_traces(real, sim, "FFJ2", "routing_bristle_proximal", aligned, options))
    assert len(traces) == 1
    assert traces[0][1] == pytest.approx([0.1, -0.2])  # millimetres, including reversal sign
    assert curves.channel_values(sim, "FFJ2", "routing_contact_state", options) == [0.2, 0.3]
    assert curves._channel_display_range("routing_kinetic", [1e-8, 2e-8]) is None
    assert curves._channel_display_range("tension_drop", [1e-8, 2e-8]) is not None


def test_report_detects_tension_and_input_mismatch(tmp_path: Path):
    real, sim = _pair(tmp_path, sim_tension_bias=1, sim_target_rad=1.25)
    report, _ = curves.build_report(real, sim, ("THJ1",), curves.CurveOptions())
    metrics = report["joints"]["THJ1"]

    # The bias applies only after the edge. The two earlier zero samples are
    # also in the overlap, so the full-window bias is below 1 N.
    assert metrics["tension_N"]["steady_bias"] == pytest.approx(0.6)
    assert metrics["tension_N"]["rmse"] > 0.5
    assert metrics["angle_deg"]["rmse"] == pytest.approx(0, abs=1e-12)
    assert metrics["command_angle_deg"]["rmse"] > math.degrees(0.1)


def test_missing_command_edge_needs_explicit_start_alignment(tmp_path: Path):
    path = _csv(
        tmp_path / "stationary.csv",
        ["time_s", "THJ2/position_rad", "THJ2/tension_N"],
        [[0, 0, 2], [0.01, 0, 2]],
    )
    table = curves.read_curve(path)

    with pytest.raises(ValueError, match="no command edge"):
        curves.build_report(table, None, ("THJ2",), curves.CurveOptions())
    _, aligned = curves.build_report(table, None, ("THJ2",), curves.CurveOptions(alignment="start"))
    assert aligned["real"] == [0, 0.01]


@pytest.mark.parametrize("extension", ["png", "html"])
def test_cli_writes_plot_and_metrics(tmp_path: Path, extension: str):
    real, sim = _pair(tmp_path)
    output = tmp_path / f"comparison.{extension}"

    assert (
        curves.main(["--real", str(real.path), "--sim", str(sim.path), "--joints", "THJ1", "--out", str(output)]) == 0
    )
    assert output.stat().st_size > 1_000
    report = output.with_suffix(".json")
    assert report.is_file()
    assert '"tension_N"' in report.read_text(encoding="utf-8")


def test_curve_settings_read_yaml_and_allow_cli_override(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text("curves:\n  tension_raw_to_N: 0.02\n  alignment: start\n", encoding="utf-8")
    options = curves.load_options(path, alignment="command_edge")
    assert options.tension_raw_to_N == 0.02
    assert options.alignment == "command_edge"

    real, _ = _pair(tmp_path)
    assert real.values("THJ1", "tension_N", raw_to_N=options.tension_raw_to_N)[2] == 2


def test_comparison_does_not_extrapolate_past_simulation(tmp_path: Path):
    options = curves.CurveOptions(alignment="start")
    result = curves.compare_series([0, 0.1, 0.2], [0, 0, 100], [0, 0.1], [0, 0], scale=1, options=options)
    assert result["samples"] == 2
    assert result["rmse"] == 0
    assert result["eligible_duration_s"] == pytest.approx(0.2)
    assert result["recording_duration_s"] == pytest.approx(0.2)
    assert result["real_data_coverage_ratio"] == pytest.approx(1.0)
    assert result["valid_duration_s"] == pytest.approx(0.1)
    assert result["coverage_ratio"] == pytest.approx(0.5)


def test_sparse_sim_endpoints_do_not_count_as_time_coverage():
    options = curves.CurveOptions(alignment="start", max_interp_gap_s=0.1)
    real_time = [index / 100 for index in range(101)]
    result = curves.compare_series(real_time, [0.0] * len(real_time), [0, 1], [0, 0], scale=1, options=options)

    assert result is not None
    assert result["samples"] == 2
    assert result["overlap_s"] == pytest.approx(1.0)
    assert result["eligible_duration_s"] == pytest.approx(1.0)
    assert result["recording_duration_s"] == pytest.approx(1.0)
    assert result["real_data_coverage_ratio"] == pytest.approx(1.0)
    assert result["valid_duration_s"] == pytest.approx(0.0)
    assert result["coverage_ratio"] == pytest.approx(0.0)


def test_coverage_uses_only_finite_real_intervals_and_is_null_without_duration():
    options = curves.CurveOptions(alignment="start", max_interp_gap_s=0.1)
    result = curves.compare_series(
        [0, 0.05, 0.1, 0.15, 0.2],
        [0, 0, math.nan, 0, 0],
        [0, 0.05, 0.1, 0.15, 0.2],
        [0, 0, 0, 0, 0],
        scale=1,
        options=options,
    )
    assert result is not None
    assert result["eligible_duration_s"] == pytest.approx(0.1)
    assert result["recording_duration_s"] == pytest.approx(0.2)
    assert result["real_data_coverage_ratio"] == pytest.approx(0.5)
    assert result["valid_duration_s"] == pytest.approx(0.1)
    assert result["coverage_ratio"] == pytest.approx(1.0)

    single_sample = curves.compare_series([0], [0], [0], [0], scale=1, options=options)
    assert single_sample is not None
    assert single_sample["eligible_duration_s"] == 0
    assert single_sample["real_data_coverage_ratio"] is None
    assert single_sample["valid_duration_s"] == 0
    assert single_sample["coverage_ratio"] is None


def test_command_comparison_uses_hold_not_linear_ramp():
    options = curves.CurveOptions(alignment="start")
    result = curves.compare_series(
        [0, 0.05, 0.1], [0, 0, 1], [0, 0.1], [0, 1], scale=1, options=options, interpolation="previous"
    )
    assert result["samples"] == 3
    assert result["rmse"] == 0


def test_real_data_coverage_uses_full_recording_duration_not_only_finite_spans():
    options = curves.CurveOptions(alignment="start", max_interp_gap_s=0.1)
    real_time = [index / 100 for index in range(201)]
    real_values = [math.nan if 1.0 < time_s < 1.5 else 0.0 for time_s in real_time]

    result = curves.compare_series(
        real_time,
        real_values,
        real_time,
        [0.0] * len(real_time),
        scale=1,
        options=options,
    )

    assert result is not None
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["real_data_coverage_ratio"] == pytest.approx(0.75)


def test_model_yaml_without_curves_and_strict_curve_section(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("controller: {}\ntendon: {}\n")
    assert curves.load_options(path) == curves.CurveOptions()
    path.write_text("controller: {}\ncurves:\n  typo: 1\n")
    with pytest.raises(ValueError, match="unknown curve"):
        curves.load_options(path)


def test_tension_aliases_bias_and_residuals_share_si_conversion(tmp_path):
    path = _csv(
        tmp_path / "aliases.csv",
        ["time_s", "THJ1/position_rad", "THJ1/effort_raw", "THJ1/cmd_tension_limit_raw", "THJ1/cmd_tension_command_N"],
        [[7, 0, 100, 400, ""], [7.1, 0, 200, 400, 8]],
    )
    table = curves.read_curve(path)
    options = curves.CurveOptions(alignment="start", tension_raw_to_N=0.02, real_tension_bias_N=1)
    assert table.time_s == pytest.approx([0, 0.1])
    assert curves.channel_values(table, "THJ1", "tension_N", options, is_real=True) == [1, 3]
    assert curves.channel_values(table, "THJ1", "cmd_tension_N", options, is_real=True) == [8, 8]
    report, aligned = curves.build_report(table, table, ("THJ1",), options)
    assert report["joints"]["THJ1"]["tension_N"]["bias"] == 1
    times, errors = curves.residual_series(table, table, "THJ1", "tension_N", aligned, options)
    assert times == table.time_s and errors == [1, 1]


def test_fallback_mask_excludes_other_mode_and_gaps(tmp_path):
    path = _csv(
        tmp_path / "mask.csv", ["time_s", "THJ1/position_rad", "is_fallback"], [[0, 0, 1], [0.05, 100, 0], [0.1, 0, 1]]
    )
    table = curves.read_curve(path)
    options = curves.CurveOptions(alignment="start", fallback_only=True)
    report, _ = curves.build_report(table, table, ("THJ1",), options)
    assert report["joints"]["THJ1"]["angle_deg"]["samples"] == 2
    values = curves.channel_values(table, "THJ1", "position_rad", options)
    assert curves._interpolate(table.time_s, values, 0.025, 0.1) is None
    unmasked = curves.CurveTable(Path("unmasked"), table.time_s, {"THJ1/position_rad": [0, 0, 0]})
    with pytest.raises(ValueError, match="requires is_fallback"):
        curves.channel_values(unmasked, "THJ1", "position_rad", options)


@pytest.mark.parametrize("timestamps", [[0, 0], [1, 0], [0, "nan"]])
def test_invalid_timestamps_rejected(tmp_path, timestamps):
    path = _csv(tmp_path / "badtime.csv", ["time_s", "THJ1/position_rad"], [[t, 0] for t in timestamps])
    with pytest.raises(ValueError, match="timestamps"):
        curves.read_curve(path)


@pytest.mark.parametrize("extension", ["png", "html"])
@pytest.mark.parametrize("comparison", [False, True])
@pytest.mark.parametrize("show_command", [False, True])
def test_plot_layout_command_optin_and_display_only_noise_floor(
    tmp_path, monkeypatch, extension, comparison, show_command
):
    source = _csv(
        tmp_path / "tiny.csv",
        ["time_s", "THJ2/position_rad", "THJ2/tension_N", "THJ2/cmd_tension_N"],
        [[0, 1e-13, 1e-11, 20], [0.01, 2e-13, 2e-11, 20]],
    )
    table = curves.read_curve(source)
    options = curves.CurveOptions(alignment="start", show_tension_command=show_command)
    real, sim = (table, table) if comparison else (None, table)
    report, aligned = curves.build_report(real, sim, ("THJ2",), options)
    captured = []
    if extension == "png":
        from matplotlib.figure import Figure

        original = Figure.savefig

        def capture(figure, *args, **kwargs):
            captured.append(figure)
            return original(figure, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", capture)
    else:
        import plotly.graph_objects as go

        original = go.Figure.write_html

        def capture(figure, *args, **kwargs):
            captured.append(figure)
            return original(figure, *args, **kwargs)

        monkeypatch.setattr(go.Figure, "write_html", capture)
    output = tmp_path / f"tiny.{extension}"
    curves.plot_curves(output, real, sim, ("THJ2",), aligned, options)
    assert output.stat().st_size > 1000
    figure = captured[0]
    columns = 4 if comparison else 2
    if extension == "png":
        assert len(figure.axes) == columns
        labels = [line.get_label() for axis in figure.axes for line in axis.lines]
        angle_range = figure.axes[0].get_ylim()
        tension_range = figure.axes[1].get_ylim()
    else:
        axes = [key for key in figure.layout if key.startswith("yaxis")]
        assert len(axes) == columns
        labels = [trace.name for trace in figure.data]
        angle_range = figure.layout.yaxis.range
        tension_range = figure.layout.yaxis2.range
        assert len(figure.layout.annotations) == columns
    assert any("third command field" in label for label in labels) is show_command
    assert angle_range[1] - angle_range[0] == pytest.approx(0.01)
    if not show_command:
        assert tension_range[1] - tension_range[0] == pytest.approx(0.01)
    # Display changes leave original tiny values and all statistical results intact.
    assert table.values("THJ2", "tension_N") == [1e-11, 2e-11]
    unchanged, _ = curves.build_report(real, sim, ("THJ2",), curves.CurveOptions(alignment="start"))
    assert report["joints"] == unchanged["joints"]


def test_show_tension_command_cli_and_yaml_option(tmp_path):
    assert curves.CurveOptions().show_tension_command is False
    assert (
        curves._parser()
        .parse_args(["--sim", "capture.csv", "--out", "plot.png", "--show-tension-command"])
        .show_tension_command
        is True
    )
    config = tmp_path / "curves.yaml"
    config.write_text("curves:\n  show_tension_command: true\n")
    assert curves.load_options(config).show_tension_command is True
    with pytest.raises(ValueError, match="boolean"):
        curves.load_options(config, show_tension_command="false")


def test_minimum_display_range_preserves_varying_curves_and_missing_values():
    assert curves._minimum_display_range([0, 2]) is None
    assert curves._minimum_display_range([math.nan]) is None
    assert curves._minimum_display_range([math.nan, 5, 5]) == pytest.approx((4.995, 5.005))
