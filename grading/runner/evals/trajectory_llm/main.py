"""LLM judge eval for grading agent trajectories.

Loosely inspired by:
1. AgentRewardBench: LLM judge over web-agent trajectories with structured
   criteria for success, side effects, repetitiveness, and rubric reliability.
2. TRAJECT-Bench: trajectory-aware evaluation that scores full tool-use
   trajectories, not just final answers.
"""

import json
from typing import Any

from litellm import Choices
from loguru import logger
from pydantic import ValidationError

from runner.evals.models import EvalImplInput
from runner.helpers.models import HelperIds
from runner.models import VerifierResult
from runner.utils.llm import build_messages, call_llm

from ..output_llm.utils.log_helpers import (
    log_grader_final_prompt,
    log_grader_result,
    log_grader_start,
)
from ..output_llm.utils.shared import (
    LLM_JUDGE_TIMEOUT,
    MAX_JSON_RETRIES,
    extract_task_prompt,
)
from .models import (
    TrajectoryJudgeResponse,
    compute_overall_score,
    normalize_overall_score,
)
from .utils.prompts import TRAJECTORY_GRADING_SYSTEM_PROMPT, build_trajectory_prompt
from .utils.trajectory_formatting import build_trajectory_excerpt

DEFAULT_MAX_MESSAGES = 12
DEFAULT_MAX_CHARS = 12_000
RAW_RESPONSE_PREVIEW_CHARS = 500


def _extract_json_object(raw_content: str) -> str:
    """Strip Markdown fences and extract the outermost JSON object."""
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    if content.startswith("{") and content.endswith("}"):
        return content

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]

    return content


def _parse_trajectory_judge_response(raw_content: str) -> TrajectoryJudgeResponse:
    raw_json = json.loads(_extract_json_object(raw_content))
    if not isinstance(raw_json, dict):
        raise ValueError("Trajectory judge response must be a JSON object")
    if isinstance(raw_json.get("rationale"), dict):
        raw_json["rationale"] = json.dumps(raw_json["rationale"])
    if raw_json.get("critical_step_idxs") is None:
        raw_json["critical_step_idxs"] = []
    return TrajectoryJudgeResponse.model_validate(raw_json)


def _preview_raw_response(raw_content: str | None) -> str:
    if not raw_content:
        return "(empty)"
    preview = raw_content.replace("\n", "\\n")
    return preview[:RAW_RESPONSE_PREVIEW_CHARS] + ("..." if len(preview) > RAW_RESPONSE_PREVIEW_CHARS else "")


def _build_verifier_result_values(
    judge_response: TrajectoryJudgeResponse,
    *,
    overall_score: int,
    evaluated_message_count: int,
) -> dict[str, Any]:
    return {
        "judge_grade": "pass" if overall_score >= 4 else "fail",
        "grade_rationale": judge_response.rationale,
        "tool_use_score": judge_response.tool_use_score,
        "grounding_score": judge_response.grounding_score,
        "recovery_score": judge_response.recovery_score,
        "efficiency_score": judge_response.efficiency_score,
        "overall_score": overall_score,
        "failure_type": judge_response.failure_type.value,
        "failure_step_idx": judge_response.failure_step_idx,
        "critical_step_idxs": judge_response.critical_step_idxs,
        "evaluated_message_count": evaluated_message_count,
    }


async def trajectory_llm_eval(input: EvalImplInput) -> VerifierResult:
    verifier_values = input.verifier.verifier_values or {}
    task_id = input.verifier.task_id or "unknown"
    criteria = verifier_values.get("criteria", "")
    criteria_explanation = verifier_values.get("criteria_explanation")

    log_grader_start(task_id, criteria, is_negative=False)

    if not criteria:
        raise ValueError("Missing required field: criteria")

    try:
        if not input.helper_results:
            raise ValueError("Missing helper results")

        final_answer = str(input.helper_results.get(HelperIds.FINAL_ANSWER, "") or "")
        model = input.grading_settings.llm_judge_model
        extra_args = input.grading_settings.llm_judge_extra_args
        task_prompt = extract_task_prompt(input)

        eval_config_values = input.eval_config.eval_config_values or {}
        max_messages = int(eval_config_values.get("trajectory_max_messages", DEFAULT_MAX_MESSAGES))
        max_chars = int(eval_config_values.get("trajectory_max_chars", DEFAULT_MAX_CHARS))

        trajectory_excerpt, evaluated_message_count = build_trajectory_excerpt(
            input.trajectory.messages,
            max_messages=max_messages,
            max_chars=max_chars,
        )

        user_prompt = build_trajectory_prompt(
            task_prompt=task_prompt,
            final_answer=final_answer,
            trajectory_excerpt=trajectory_excerpt,
            criteria=criteria,
            criteria_explanation=criteria_explanation,
        )

        log_grader_final_prompt(
            task_id=task_id,
            criteria=criteria,
            is_negative=False,
            model=model,
            system_prompt_chars=len(TRAJECTORY_GRADING_SYSTEM_PROMPT),
            user_prompt_chars=len(user_prompt),
            artifacts_to_evaluate=None,
            artifacts_to_reference=None,
            image_count=0,
        )

        logger.debug(
            f"[JUDGE][GRADER] task={task_id} | trajectory prompt:\n"
            f"SYSTEM:\n{TRAJECTORY_GRADING_SYSTEM_PROMPT}\n\n"
            f"USER:\n{user_prompt}"
        )

        messages = build_messages(
            system_prompt=TRAJECTORY_GRADING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        parsed = None
        raw_content = None
        for attempt in range(MAX_JSON_RETRIES):
            response = await call_llm(
                model=model,
                messages=messages,
                timeout=LLM_JUDGE_TIMEOUT,
                extra_args=extra_args,
                response_format={"type": "json_object"},
            )

            choices = response.choices
            if not choices:
                logger.warning(f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty response")
                continue

            choice = choices[0]
            if not isinstance(choice, Choices):
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: unexpected choice type={type(choice).__name__}"
                )

            raw_content = choices[0].message.content if isinstance(choices[0], Choices) else None
            if not raw_content:
                logger.warning(f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty content")
                continue

            try:
                parsed = _parse_trajectory_judge_response(raw_content)
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as e:
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: {e} | raw={_preview_raw_response(raw_content)}"
                )
                parsed = None

        if parsed is None:
            raise ValueError(f"Invalid JSON after {MAX_JSON_RETRIES} attempts")

        overall_score = compute_overall_score(parsed)
        score = normalize_overall_score(overall_score)
        passed = overall_score >= 4

        log_grader_result(
            task_id,
            is_negative=False,
            passed=passed,
            score=score,
            criteria=criteria,
        )

        return VerifierResult(
            verifier_id=input.verifier.verifier_id,
            verifier_version=input.verifier.verifier_version,
            score=score,
            verifier_result_values=_build_verifier_result_values(
                parsed,
                overall_score=overall_score,
                evaluated_message_count=evaluated_message_count,
            ),
        )

    except Exception as e:
        raise ValueError(f"Trajectory LLM grading failed: {str(e)}") from e
