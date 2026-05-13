import pytest
from pydantic import ValidationError

from runner.evals.trajectory_llm.models import (
    TrajectoryFailureType,
    TrajectoryJudgeResponse,
    compute_overall_score,
    normalize_overall_score,
)


def _judge_response(**overrides: object) -> TrajectoryJudgeResponse:
    values = {
        "tool_use_score": 5,
        "grounding_score": 4,
        "recovery_score": 3,
        "efficiency_score": 5,
        "failure_type": "none",
        "failure_step_idx": None,
        "rationale": "The agent used tools correctly with one redundant action.",
    }
    values.update(overrides)
    return TrajectoryJudgeResponse.model_validate(values)


def test_trajectory_judge_response_accepts_valid_scores() -> None:
    response = _judge_response()

    assert response.tool_use_score == 5
    assert response.grounding_score == 4
    assert response.failure_type == TrajectoryFailureType.NONE
    assert response.critical_step_idxs == []


def test_trajectory_judge_response_rejects_scores_outside_one_to_five() -> None:
    with pytest.raises(ValidationError):
        _judge_response(tool_use_score=0)

    with pytest.raises(ValidationError):
        _judge_response(efficiency_score=6)


def test_compute_overall_score_uses_weighted_dimensions() -> None:
    # tool_use=5, grounding=5, recovery=5, efficiency=5 → weighted = 5.0 → 5
    response = _judge_response(
        tool_use_score=5,
        grounding_score=5,
        recovery_score=5,
        efficiency_score=5,
    )
    assert compute_overall_score(response) == 5


def test_compute_overall_score_weights_tool_use_and_grounding_most() -> None:
    # tool_use=5 (0.35), grounding=5 (0.30), recovery=1 (0.20), efficiency=1 (0.15)
    # weighted = 1.75 + 1.50 + 0.20 + 0.15 = 3.60 → rounds to 4
    response = _judge_response(
        tool_use_score=5,
        grounding_score=5,
        recovery_score=1,
        efficiency_score=1,
    )
    assert compute_overall_score(response) == 4


def test_compute_overall_score_low_grounding_fails() -> None:
    # tool_use=2 (0.35), grounding=1 (0.30), recovery=2 (0.20), efficiency=2 (0.15)
    # weighted = 0.70 + 0.30 + 0.40 + 0.30 = 1.70 → rounds to 2
    response = _judge_response(
        tool_use_score=2,
        grounding_score=1,
        recovery_score=2,
        efficiency_score=2,
        failure_type="hallucination",
    )
    assert compute_overall_score(response) == 2


def test_normalize_overall_score_maps_one_to_five_onto_zero_to_one() -> None:
    assert normalize_overall_score(1) == 0.0
    assert normalize_overall_score(3) == 0.5
    assert normalize_overall_score(5) == 1.0


def test_normalize_overall_score_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError):
        normalize_overall_score(0)

    with pytest.raises(ValueError):
        normalize_overall_score(6)
