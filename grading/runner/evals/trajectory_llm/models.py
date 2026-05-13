"""Typed models and scoring policy for trajectory judge responses."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TrajectoryFailureType(StrEnum):
    """High-level failure categories for trajectory-level grading."""

    NONE = "none"
    PLANNING = "planning"
    MEMORY = "memory"
    HALLUCINATION = "hallucination"
    TOOL_MISUSE = "tool_misuse"
    INSTRUCTION_NONADHERENCE = "instruction_nonadherence"
    SIDE_EFFECT = "side_effect"
    INEFFICIENCY = "inefficiency"
    INCOMPLETE = "incomplete"
    OTHER = "other"


class TrajectoryJudgeResponse(BaseModel):
    """Structured response expected from the trajectory LLM judge."""

    success_score: int = Field(ge=1, le=5)
    side_effect_score: int = Field(ge=1, le=5)
    efficiency_score: int = Field(ge=1, le=5)
    instruction_adherence_score: int = Field(ge=1, le=5)
    failure_type: TrajectoryFailureType
    failure_step_idx: int | None = Field(default=None, ge=0)
    critical_step_idxs: list[int] = Field(default_factory=list)
    rationale: str


def compute_overall_score(response: TrajectoryJudgeResponse) -> int:
    """Compute the deterministic 1-5 overall score from judge dimensions."""
    weighted_score = (
        0.55 * response.success_score
        + 0.20 * response.instruction_adherence_score
        + 0.15 * response.side_effect_score
        + 0.10 * response.efficiency_score
    )
    overall_score = int(weighted_score + 0.5)

    if response.success_score <= 2:
        overall_score = min(overall_score, response.success_score + 1)

    return max(1, min(5, overall_score))


def normalize_overall_score(overall_score: int) -> float:
    """Map a 1-5 overall score onto the verifier score range 0-1."""
    if overall_score < 1 or overall_score > 5:
        raise ValueError("overall_score must be between 1 and 5")

    return (overall_score - 1) / 4
