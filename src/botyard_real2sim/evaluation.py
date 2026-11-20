"""Strict, machine-readable acceptance checks for curve-fit candidates.

This module deliberately does not infer tolerances or data coverage. Callers
must provide a policy and per-channel coverage from the comparison producer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationPolicy:
    """Acceptance thresholds for a candidate comparison.

    ``channel_scales`` are maximum acceptable RMSE values in the units emitted
    by ``build_report``: degrees for angle channels and newtons for tension.
    They are intentionally required for a passing score; choosing tolerances is
    a calibration decision, not a generic software default.
    """

    required_channels: tuple[str, ...] = ("angle_deg", "motor_angle_deg", "tension_N")
    channel_scales: Mapping[str, float] = field(default_factory=dict)
    channel_weights: Mapping[str, float] = field(default_factory=dict)
    min_samples: int = 100
    min_coverage_ratio: float = 0.8
    min_overlap_s: float = 0.5
    max_mimic_error_rad: float = 0.01

    def __post_init__(self) -> None:
        if not self.required_channels or len(set(self.required_channels)) != len(self.required_channels):
            raise ValueError("required_channels must be nonempty and unique")
        if type(self.min_samples) is not int or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if not math.isfinite(self.min_coverage_ratio) or not 0 < self.min_coverage_ratio <= 1:
            raise ValueError("min_coverage_ratio must be finite and in (0, 1]")
        if not math.isfinite(self.min_overlap_s) or self.min_overlap_s <= 0:
            raise ValueError("min_overlap_s must be positive and finite")
        if not math.isfinite(self.max_mimic_error_rad) or self.max_mimic_error_rad < 0:
            raise ValueError("max_mimic_error_rad must be finite and nonnegative")
        for name, mapping in (("channel_scales", self.channel_scales), ("channel_weights", self.channel_weights)):
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(mapping) - set(self.required_channels)
            if unknown:
                raise ValueError(f"{name} contains channels outside required_channels: {', '.join(sorted(unknown))}")
            for channel, value in mapping.items():
                if _number(value) is None or value <= 0:
                    raise ValueError(f"{name}[{channel}] must be positive and finite")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def score_report(
    report: Mapping[str, Any],
    policy: EvaluationPolicy,
    *,
    replay_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Score a curves ``build_report`` result and determine fit eligibility.

    A missing metric, coverage ratio, tolerance, or required replay metadata
    rejects the candidate. ``overlap_s`` is only a span and is never treated as
    sample coverage. The output always labels the result as a curve-fit
    candidate; it cannot certify that the physical model is calibrated.
    """

    quality_reasons: list[str] = []
    tolerance_reasons: list[str] = []
    notices: list[str] = []
    channels_out: dict[str, dict[str, Any]] = {}
    joints = report.get("joints") if isinstance(report, Mapping) else None
    joint_names = tuple(joints) if isinstance(joints, Mapping) else ()
    if not joint_names:
        quality_reasons.append("comparison report has no joints")

    weighted_square_sum = 0.0
    total_weight = 0.0
    all_channels_within_tolerance = bool(joint_names)
    score_inputs_valid = bool(joint_names)
    scales_complete = True
    for joint in joint_names:
        joint_data = joints[joint]
        if not isinstance(joint_data, Mapping):
            quality_reasons.append(f"{joint}: joint metrics are missing or malformed")
            all_channels_within_tolerance = False
            score_inputs_valid = False
            continue
        for channel in policy.required_channels:
            metric = joint_data.get(channel)
            key = f"{joint}/{channel}"
            channel_quality_reasons: list[str] = []
            channel_tolerance_reasons: list[str] = []
            if not isinstance(metric, Mapping):
                channel_quality_reasons.append("required channel metrics are missing")
                all_channels_within_tolerance = False
                score_inputs_valid = False
                metric = {}

            samples = metric.get("samples")
            overlap_s = _number(metric.get("overlap_s"))
            coverage = _number(metric.get("coverage_ratio"))
            real_coverage = _number(metric.get("real_data_coverage_ratio"))
            rmse = _number(metric.get("rmse"))
            if type(samples) is not int or samples < 0:
                channel_quality_reasons.append("sample count is missing or invalid")
                score_inputs_valid = False
            elif samples < policy.min_samples:
                channel_quality_reasons.append(f"samples {samples} below minimum {policy.min_samples}")
                score_inputs_valid = False
            if overlap_s is None or overlap_s < 0:
                channel_quality_reasons.append("overlap_s is missing or invalid")
                score_inputs_valid = False
            elif overlap_s < policy.min_overlap_s:
                channel_quality_reasons.append(f"overlap {overlap_s:g}s below minimum {policy.min_overlap_s:g}s")
                score_inputs_valid = False
            if coverage is None or not 0 <= coverage <= 1:
                channel_quality_reasons.append("coverage_ratio is missing or invalid; overlap_s is not coverage")
                score_inputs_valid = False
            elif coverage < policy.min_coverage_ratio:
                channel_quality_reasons.append(f"coverage {coverage:.3f} below minimum {policy.min_coverage_ratio:.3f}")
                score_inputs_valid = False
            if real_coverage is None or not 0 <= real_coverage <= 1:
                channel_quality_reasons.append("real_data_coverage_ratio is missing or invalid")
                score_inputs_valid = False
            elif real_coverage < policy.min_coverage_ratio:
                channel_quality_reasons.append(
                    f"real data coverage {real_coverage:.3f} below minimum {policy.min_coverage_ratio:.3f}"
                )
                score_inputs_valid = False
            if rmse is None or rmse < 0:
                channel_quality_reasons.append("RMSE is missing or invalid")
                score_inputs_valid = False

            scale = _number(policy.channel_scales.get(channel))
            if scale is None or scale <= 0:
                scales_complete = False
                score_inputs_valid = False
                all_channels_within_tolerance = False
                channel_quality_reasons.append("explicit positive channel tolerance is required")
                normalized_rmse = None
            elif rmse is None:
                normalized_rmse = None
                all_channels_within_tolerance = False
            else:
                normalized_rmse = rmse / scale
                if normalized_rmse > 1:
                    channel_tolerance_reasons.append(
                        f"normalized RMSE {normalized_rmse:.3f} exceeds channel tolerance 1.000"
                    )
                    all_channels_within_tolerance = False
                weight = _number(policy.channel_weights.get(channel, 1.0))
                if weight is not None and weight > 0:
                    weighted_square_sum += weight * normalized_rmse**2
                    total_weight += weight

            quality_reasons.extend(f"{key}: {reason}" for reason in channel_quality_reasons)
            tolerance_reasons.extend(f"{key}: {reason}" for reason in channel_tolerance_reasons)
            channels_out[key] = {
                "samples": samples if type(samples) is int else None,
                "overlap_s": overlap_s,
                "coverage_ratio": coverage,
                "real_data_coverage_ratio": real_coverage,
                "rmse": rmse,
                "tolerance": scale,
                "normalized_rmse": normalized_rmse,
                "quality_valid": not channel_quality_reasons,
                "within_tolerance": not channel_tolerance_reasons,
                "acceptable": not channel_quality_reasons and not channel_tolerance_reasons,
                "quality_reasons": channel_quality_reasons,
                "tolerance_reasons": channel_tolerance_reasons,
                "reasons": channel_quality_reasons + channel_tolerance_reasons,
            }

    if not isinstance(replay_metadata, Mapping):
        quality_reasons.append("replay metadata is required and must be a JSON object")
    else:
        if replay_metadata.get("status") != "completed":
            quality_reasons.append("replay status must be completed")
        saturated = replay_metadata.get("saturated_force_evaluations")
        if type(saturated) is not int or saturated != 0:
            quality_reasons.append("replay must report zero saturated_force_evaluations")
        mimic_error = _number(replay_metadata.get("mimic_max_error_rad"))
        if mimic_error is None:
            quality_reasons.append("replay mimic_max_error_rad is missing or invalid")
        elif mimic_error > policy.max_mimic_error_rad:
            quality_reasons.append(f"replay mimic error {mimic_error:g}rad exceeds {policy.max_mimic_error_rad:g}rad")
        calibrated_flag = replay_metadata.get("calibrated")
        if calibrated_flag is True:
            notices.append("replay metadata says calibrated=true; this evaluator still reports candidates only")
        elif calibrated_flag is not False:
            notices.append("replay metadata has no calibrated flag; this evaluator still reports candidates only")

    score = (
        math.sqrt(weighted_square_sum / total_weight)
        if total_weight > 0 and scales_complete and score_inputs_valid
        else None
    )
    quality_valid = not quality_reasons and score is not None
    tolerance_pass = all_channels_within_tolerance and score is not None
    candidate_pass = quality_valid and tolerance_pass
    return {
        "schema_version": 1,
        "quality_valid": quality_valid,
        "tolerance_pass": tolerance_pass,
        "candidate_pass": candidate_pass,
        "score": score,
        "score_definition": "weighted RMS of per-channel RMSE divided by explicit channel tolerance",
        "channels": channels_out,
        "quality_reasons": quality_reasons,
        "tolerance_reasons": tolerance_reasons,
        "reasons": quality_reasons + tolerance_reasons,
        "notices": notices,
        "result_scope": "curve_fit_candidate_only",
        "calibrated_model_claim_allowed": False,
        "replay_calibrated_flag": replay_metadata.get("calibrated") if isinstance(replay_metadata, Mapping) else None,
    }
