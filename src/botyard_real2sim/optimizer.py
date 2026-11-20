"""Bounded multi-parameter search for calibration objective callbacks.

This module does not run a simulator or decide whether a curve comparison is
valid. The caller must reject unscorable candidates through
``quality_valid=False`` and supply an objective with meaningful, dimensionless
channel scaling. Candidates may remain above acceptance tolerances while the
optimizer searches for an improvement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterBound:
    """Search interval for one named configuration parameter.

    ``scale="log"`` samples uniformly in logarithmic space and requires a
    strictly positive interval. Bounds must come from a physical prior or an
    explicit user decision; this type does not infer them from model defaults.
    """

    path: str
    lower: float
    upper: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("parameter path must be a nonempty string")
        if any(char.isspace() for char in self.path):
            raise ValueError(f"parameter path cannot contain whitespace: {self.path!r}")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError(f"{self.path}: bounds must be finite with lower < upper")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.path}: scale must be 'linear' or 'log'")
        if self.scale == "log" and self.lower <= 0:
            raise ValueError(f"{self.path}: logarithmic bounds must be positive")

    def decode(self, unit_value: float) -> float:
        """Map the optimizer's unit interval coordinate into this interval."""
        if not math.isfinite(unit_value) or not 0.0 <= unit_value <= 1.0:
            raise ValueError("unit_value must be finite and inside [0, 1]")
        if self.scale == "log":
            return math.exp(math.log(self.lower) + unit_value * (math.log(self.upper) - math.log(self.lower)))
        return self.lower + unit_value * (self.upper - self.lower)


def _json_safe(value: Any) -> Any:
    """Convert callback details to values accepted by strict JSON encoders."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def fit_parameters(
    objective: Callable[[dict[str, float]], float | dict[str, Any]],
    parameters: Sequence[ParameterBound],
    *,
    max_evaluations: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Search bounded parameters with deterministic differential evolution.

    The callback receives a mapping from parameter path to physical value. It
    may return a finite scalar score, or a mapping containing ``score`` and
    optional ``quality_valid`` / ``reasons`` / diagnostic fields. Failed or
    invalid candidates receive a large penalty and can never become the
    reported best candidate. Calls run serially, which is important when a
    callback launches an Isaac Sim process.

    The result is JSON-safe and includes every candidate. This performs a
    bounded numerical search only; it does not establish hardware validity,
    identifiability, held-out performance, or a calibrated model claim.
    """
    if not parameters:
        raise ValueError("at least one parameter bound is required")
    if len({parameter.path for parameter in parameters}) != len(parameters):
        raise ValueError("parameter paths must be unique")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    dimension = len(parameters)
    if max_evaluations < 5 * dimension:
        raise ValueError(f"max_evaluations must be at least 5 * parameter count ({5 * dimension})")

    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        message = "parameter fitting requires SciPy; install with `pip install botyard_real2sim[calibration]`"
        raise RuntimeError(message) from exc

    # Keep the initial population and all later generations within the explicit
    # budget. Unit-space bounds avoid poor optimizer scaling across SI units.
    population_multiplier = 5
    population_evaluations = population_multiplier * dimension
    max_generations = max(0, max_evaluations // population_evaluations - 1)
    penalty = 1e100
    trials: list[dict[str, Any]] = []
    accepted: list[tuple[float, dict[str, float]]] = []

    def evaluate(unit_values) -> float:
        values = {
            parameter.path: parameter.decode(float(unit_values[index])) for index, parameter in enumerate(parameters)
        }
        trial: dict[str, Any] = {"index": len(trials), "parameters": values}
        try:
            result = objective(values)
            if isinstance(result, dict):
                details = result
                quality_valid = result.get("quality_valid", True) is True
                score_value = result.get("score")
                reason = result.get("reasons")
            else:
                details = {}
                quality_valid = True
                score_value = result
                reason = None
            if not quality_valid:
                trial.update(
                    status="rejected",
                    reason=_json_safe(reason or "quality gates failed"),
                    details=_json_safe(details),
                )
                trials.append(trial)
                return penalty
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise ValueError("objective result must contain a numeric score")
            score = float(score_value)
            if not math.isfinite(score):
                raise ValueError("objective score must be finite")
            trial.update(status="completed", score=score, details=_json_safe(details))
            trials.append(trial)
            accepted.append((score, values))
            return score
        except Exception as exc:
            trial.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            trials.append(trial)
            return penalty

    try:
        result = differential_evolution(
            evaluate,
            bounds=[(0.0, 1.0)] * dimension,
            maxiter=max_generations,
            popsize=population_multiplier,
            tol=0.0,
            atol=0.0,
            polish=False,
            seed=seed,
            updating="immediate",
            workers=1,
            disp=False,
        )
    except Exception as exc:
        # Optimizer infrastructure failures are distinct from candidate failures.
        raise RuntimeError(f"differential evolution failed: {exc}") from exc

    best = min(accepted, key=lambda item: item[0]) if accepted else None
    evaluation_count = len(trials)
    if evaluation_count > max_evaluations:
        raise RuntimeError("optimizer exceeded its configured evaluation budget")
    return {
        "schema_version": 1,
        "method": "scipy.differential_evolution",
        "seed": seed,
        "parameter_bounds": [
            {"path": item.path, "lower": item.lower, "upper": item.upper, "scale": item.scale} for item in parameters
        ],
        "max_evaluations": max_evaluations,
        "evaluation_count": evaluation_count,
        "termination": str(result.message),
        "optimizer_success": bool(result.success),
        "best_parameters": best[1] if best is not None else None,
        "best_score": best[0] if best is not None else None,
        "accepted_candidates": len(accepted),
        "rejected_candidates": sum(trial["status"] == "rejected" for trial in trials),
        "failed_candidates": sum(trial["status"] == "failed" for trial in trials),
        "claims_scope": "bounded objective search only; held-out validation and hardware evidence remain required",
        "trials": trials,
    }
