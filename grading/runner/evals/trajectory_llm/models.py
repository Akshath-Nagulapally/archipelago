"""Typed models and scoring policy for trajectory judge responses."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("failure_type", mode="before")
    @classmethod
    def normalize_failure_type(cls, value: Any) -> Any:
        """Keep the persisted taxonomy stable even when judges use near-synonyms."""
        if value is None:
            return TrajectoryFailureType.NONE
        if isinstance(value, TrajectoryFailureType):
            return value

        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "": TrajectoryFailureType.NONE,
            "n/a": TrajectoryFailureType.NONE,
            "na": TrajectoryFailureType.NONE,
            "none": TrajectoryFailureType.NONE,
            "no_failure": TrajectoryFailureType.NONE,
            "not_applicable": TrajectoryFailureType.NONE,
            "instruction": TrajectoryFailureType.INSTRUCTION_NONADHERENCE,
            "instruction_adherence": TrajectoryFailureType.INSTRUCTION_NONADHERENCE,
            "instruction_violation": TrajectoryFailureType.INSTRUCTION_NONADHERENCE,
            "tool_use": TrajectoryFailureType.TOOL_MISUSE,
            "tool_error": TrajectoryFailureType.TOOL_MISUSE,
            "looping": TrajectoryFailureType.INEFFICIENCY,
            "redundancy": TrajectoryFailureType.INEFFICIENCY,
            "task_incomplete": TrajectoryFailureType.INCOMPLETE,
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized in {failure_type.value for failure_type in TrajectoryFailureType}:
            return normalized
        return TrajectoryFailureType.OTHER

    @field_validator("failure_step_idx", mode="before")
    @classmethod
    def normalize_failure_step_idx(cls, value: Any) -> Any:
        """Models often emit human placeholders for optional integer fields."""
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "none",
            "null",
            "n/a",
            "na",
        }:
            return None
        return value


def compute_overall_score(response: TrajectoryJudgeResponse) -> int:
    """Weight task success highest, then apply smaller penalties for process quality."""
    weighted_score = (
        0.55 * response.success_score
        + 0.20 * response.instruction_adherence_score
        + 0.15 * response.side_effect_score
        + 0.10 * response.efficiency_score
    )
    overall_score = int(weighted_score + 0.5)

    # Prevent safe but unsuccessful trajectories from receiving a strong overall grade.
    if response.success_score <= 2:
        overall_score = min(overall_score, response.success_score + 1)

    return max(1, min(5, overall_score))


def normalize_overall_score(overall_score: int) -> float:
    """Map a 1-5 overall score onto the verifier score range 0-1."""
    if overall_score < 1 or overall_score > 5:
        raise ValueError("overall_score must be between 1 and 5")

    return (overall_score - 1) / 4
