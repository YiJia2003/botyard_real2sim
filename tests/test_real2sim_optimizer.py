"""Bounded optimizer behavior is testable without Isaac Sim."""

from __future__ import annotations

import json
from importlib.util import find_spec

import pytest

from botyard_real2sim.optimizer import ParameterBound, fit_parameters

pytestmark = pytest.mark.skipif(find_spec("scipy") is None, reason="optional calibration extra requires scipy")


def test_searches_linear_and_log_parameters_within_budget():
    calls = []

    def objective(values):
        calls.append(values)
        return ((values["spring"] - 0.37) / 0.2) ** 2 + ((values["friction"] - 0.015) / 0.01) ** 2

    result = fit_parameters(
        objective,
        [
            ParameterBound("spring", 0.0, 1.0),
            ParameterBound("friction", 0.001, 0.1, "log"),
        ],
        max_evaluations=300,
        seed=11,
    )

    assert result["best_parameters"] is not None
    assert result["best_parameters"]["spring"] == pytest.approx(0.37, abs=0.04)
    assert result["best_parameters"]["friction"] == pytest.approx(0.015, rel=0.15)
    assert result["best_score"] < 0.1
    assert len(calls) == result["evaluation_count"] <= result["max_evaluations"]
    assert result["method"] == "scipy.differential_evolution"
    json.dumps(result, allow_nan=False)


def test_quality_rejections_and_callback_failures_never_become_best():
    def objective(values):
        if values["gain"] < 0.2:
            raise RuntimeError("simulator failed")
        if values["gain"] < 0.6:
            return {"score": 0.0, "quality_valid": False, "reasons": ["insufficient coverage"]}
        return {"score": (values["gain"] - 0.8) ** 2, "quality_valid": True, "channel_metrics": {"angle": 0.1}}

    result = fit_parameters(objective, [ParameterBound("gain", 0.0, 1.0)], max_evaluations=80, seed=7)

    assert result["best_parameters"]["gain"] >= 0.6
    assert result["rejected_candidates"] > 0
    assert result["failed_candidates"] > 0
    assert any("insufficient coverage" in str(trial.get("reason")) for trial in result["trials"])
    assert any("simulator failed" in str(trial.get("reason")) for trial in result["trials"])
    json.dumps(result, allow_nan=False)


def test_no_acceptable_candidates_returns_no_best_candidate():
    result = fit_parameters(
        lambda _: {"score": 0.0, "quality_valid": False, "reasons": ["bad data"]},
        [ParameterBound("gain", 1.0, 2.0)],
        max_evaluations=5,
        seed=1,
    )

    assert result["best_parameters"] is None
    assert result["best_score"] is None
    assert result["accepted_candidates"] == 0
    assert result["rejected_candidates"] == result["evaluation_count"] == 5


def test_quality_invalid_candidate_can_have_no_score_and_is_rejected_cleanly():
    result = fit_parameters(
        lambda _: {"score": None, "quality_valid": False, "reasons": ["coverage too low"]},
        [ParameterBound("gain", 1.0, 2.0)],
        max_evaluations=5,
    )

    assert result["best_parameters"] is None
    assert result["rejected_candidates"] == result["evaluation_count"] == 5
    assert result["failed_candidates"] == 0


@pytest.mark.parametrize(
    "bound",
    [
        ("", 0, 1, "linear"),
        ("x", 1, 1, "linear"),
        ("x", 0, float("inf"), "linear"),
        ("x", 0, 1, "quadratic"),
        ("x", 0, 1, "log"),
    ],
)
def test_invalid_bounds_rejected(bound):
    with pytest.raises(ValueError):
        ParameterBound(*bound)


def test_duplicate_paths_and_too_small_budget_rejected():
    bounds = [ParameterBound("x", 0, 1), ParameterBound("x", 1, 2)]
    with pytest.raises(ValueError, match="unique"):
        fit_parameters(lambda _: 0.0, bounds, max_evaluations=10)
    with pytest.raises(ValueError, match="at least 5"):
        fit_parameters(lambda _: 0.0, [ParameterBound("x", 0, 1)], max_evaluations=4)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "0.5", True])
def test_nonfinite_or_nonnumeric_objective_scores_fail(score):
    result = fit_parameters(lambda _: score, [ParameterBound("x", 0, 1)], max_evaluations=5)
    assert result["best_parameters"] is None
    assert result["failed_candidates"] == result["evaluation_count"]
    json.dumps(result, allow_nan=False)
