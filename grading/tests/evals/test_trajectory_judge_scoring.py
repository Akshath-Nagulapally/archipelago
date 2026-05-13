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
        "success_score": 5,
        "side_effect_score": 4,
        "efficiency_score": 3,
        "instruction_adherence_score": 5,
        "failure_type": "none",
        "failure_step_idx": None,
        "rationale": "The agent completed the task with one redundant action.",
    }
    values.update(overrides)
    return TrajectoryJudgeResponse.model_validate(values)


def test_trajectory_judge_response_accepts_valid_scores() -> None:
    response = _judge_response()

    assert response.success_score == 5
    assert response.failure_type == TrajectoryFailureType.NONE
    assert response.critical_step_idxs == []


def test_trajectory_judge_response_rejects_scores_outside_one_to_five() -> None:
    with pytest.raises(ValidationError):
        _judge_response(success_score=0)

    with pytest.raises(ValidationError):
        _judge_response(efficiency_score=6)


def test_compute_overall_score_uses_weighted_dimensions() -> None:
    response = _judge_response(
        success_score=5,
        side_effect_score=4,
        efficiency_score=3,
        instruction_adherence_score=5,
    )

    assert compute_overall_score(response) == 5


def test_compute_overall_score_caps_low_success_runs() -> None:
    response = _judge_response(
        success_score=1,
        side_effect_score=5,
        efficiency_score=5,
        instruction_adherence_score=4,
        failure_type="incomplete",
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
