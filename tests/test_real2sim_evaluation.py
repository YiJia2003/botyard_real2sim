from __future__ import annotations

from copy import deepcopy

from botyard_real2sim.evaluation import EvaluationPolicy, score_report


def _report(*, coverage: float = 0.95) -> dict:
    return {
        "joints": {
            "FFJ2": {
                "angle_deg": {
                    "samples": 200,
                    "overlap_s": 1.8,
                    "coverage_ratio": coverage,
                    "real_data_coverage_ratio": 0.95,
                    "rmse": 0.5,
                },
                "motor_angle_deg": {
                    "samples": 200,
                    "overlap_s": 1.8,
                    "coverage_ratio": coverage,
                    "real_data_coverage_ratio": 0.95,
                    "rmse": 1.0,
                },
                "tension_N": {
                    "samples": 200,
                    "overlap_s": 1.8,
                    "coverage_ratio": coverage,
                    "real_data_coverage_ratio": 0.95,
                    "rmse": 0.2,
                },
            }
        }
    }


def _metadata(**overrides) -> dict:
    return {
        "status": "completed",
        "saturated_force_evaluations": 0,
        "mimic_max_error_rad": 0.0001,
        "calibrated": False,
        **overrides,
    }


def _policy(**overrides) -> EvaluationPolicy:
    values = {
        "channel_scales": {"angle_deg": 1.0, "motor_angle_deg": 2.0, "tension_N": 0.5},
        "min_samples": 100,
        "min_coverage_ratio": 0.8,
        "min_overlap_s": 1.0,
    }
    values.update(overrides)
    return EvaluationPolicy(**values)


def test_missing_required_channel_is_rejected():
    report = _report()
    del report["joints"]["FFJ2"]["motor_angle_deg"]

    result = score_report(report, _policy(), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert not result["candidate_pass"]
    assert result["score"] is None
    assert any("motor_angle_deg" in reason and "missing" in reason for reason in result["reasons"])


def test_missing_coverage_is_rejected_instead_of_inferred_from_overlap():
    report = _report()
    del report["joints"]["FFJ2"]["tension_N"]["coverage_ratio"]

    result = score_report(report, _policy(), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert any("coverage_ratio is missing" in reason for reason in result["reasons"])


def test_low_coverage_is_rejected():
    result = score_report(_report(coverage=0.4), _policy(), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert any("coverage 0.400 below minimum" in reason for reason in result["reasons"])


def test_real_data_coverage_is_an_independent_quality_gate():
    report = _report()
    for metric in report["joints"]["FFJ2"].values():
        metric["real_data_coverage_ratio"] = 0.75

    result = score_report(report, _policy(), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert not result["candidate_pass"]
    assert any("real data coverage 0.750 below minimum" in reason for reason in result["quality_reasons"])


def test_missing_real_data_coverage_is_rejected():
    report = _report()
    del report["joints"]["FFJ2"]["tension_N"]["real_data_coverage_ratio"]

    result = score_report(report, _policy(), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert any("real_data_coverage_ratio is missing or invalid" in reason for reason in result["quality_reasons"])


def test_effort_clipping_rejects_candidate():
    result = score_report(_report(), _policy(), replay_metadata=_metadata(saturated_force_evaluations=1))

    assert not result["quality_valid"]
    assert any("zero saturated_force_evaluations" in reason for reason in result["reasons"])


def test_missing_tolerance_does_not_invent_a_score():
    scales = {"angle_deg": 1.0, "motor_angle_deg": 2.0}
    result = score_report(_report(), _policy(channel_scales=scales), replay_metadata=_metadata())

    assert not result["quality_valid"]
    assert result["score"] is None
    assert any("explicit positive channel tolerance is required" in reason for reason in result["reasons"])


def test_successful_score_is_normalized_and_remains_candidate_only():
    result = score_report(_report(), _policy(), replay_metadata=_metadata(calibrated=False))

    assert result["quality_valid"]
    assert result["tolerance_pass"]
    assert result["candidate_pass"]
    assert result["score"] == 0.469041575982343
    assert result["channels"]["FFJ2/angle_deg"]["normalized_rmse"] == 0.5
    assert result["result_scope"] == "curve_fit_candidate_only"
    assert result["calibrated_model_claim_allowed"] is False


def test_model_calibration_flag_is_reported_as_claim_scope_not_fit_gate():
    metadata = _metadata(calibrated=True)
    result = score_report(_report(), _policy(), replay_metadata=metadata)

    assert result["quality_valid"]
    assert result["candidate_pass"]
    assert result["result_scope"] == "curve_fit_candidate_only"
    assert result["calibrated_model_claim_allowed"] is False


def test_failed_replay_and_excess_mimic_error_reject_candidate():
    metadata = _metadata(status="failed", mimic_max_error_rad=0.02)

    result = score_report(_report(), _policy(), replay_metadata=metadata)

    assert not result["quality_valid"]
    assert any("status must be completed" in reason for reason in result["reasons"])
    assert any("mimic error" in reason for reason in result["reasons"])


def test_malformed_replay_metadata_is_a_structured_quality_failure():
    result = score_report(_report(), _policy(), replay_metadata=[])

    assert not result["quality_valid"]
    assert not result["candidate_pass"]
    assert result["replay_calibrated_flag"] is None
    assert any("replay metadata" in reason for reason in result["quality_reasons"])


def test_over_tolerance_curve_remains_scoreable_for_progressive_optimization():
    report = deepcopy(_report())
    report["joints"]["FFJ2"]["tension_N"]["rmse"] = 0.6

    result = score_report(report, _policy(), replay_metadata=_metadata())

    assert result["quality_valid"]
    assert not result["tolerance_pass"]
    assert not result["candidate_pass"]
    assert result["score"] is not None
    assert result["channels"]["FFJ2/tension_N"]["normalized_rmse"] == 1.2
    assert any("normalized RMSE 1.200" in reason for reason in result["tolerance_reasons"])
