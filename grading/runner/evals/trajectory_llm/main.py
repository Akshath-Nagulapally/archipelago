"""LLM judge eval for grading agent trajectories."""

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
from ..output_llm.utils.prompts import (
    JSON_OUTPUT_GRADING,
    RATIONALE_FORMAT_BASIC,
    STRICT_CRITERION_MATCHING,
    TOLERANCE_NOTES,
)
from ..output_llm.utils.shared import LLM_JUDGE_TIMEOUT, MAX_JSON_RETRIES, extract_task_prompt

TRAJECTORY_GRADING_SYSTEM_PROMPT = "\n\n".join(
    [
        "You are grading an agent trajectory against a single verification criterion.",
        "Use the final answer as the primary evidence and the recent trajectory history as supporting context.",
        "If the final answer or recent trajectory does not provide enough evidence to confidently verify the criterion, return false.",
        STRICT_CRITERION_MATCHING,
        TOLERANCE_NOTES,
        RATIONALE_FORMAT_BASIC,
        JSON_OUTPUT_GRADING,
    ]
)

DEFAULT_MAX_MESSAGES = 12
DEFAULT_MAX_CHARS = 12_000
MAX_SINGLE_MESSAGE_CHARS = 2_000


def _get_message_value(message: Any, key: str, default: Any = None) -> Any:
    """Read a key from either dict-like or pydantic message objects."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _normalize_content(content: Any) -> str:
    """Convert heterogeneous message content into a compact string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _format_message(message: Any, reverse_index: int) -> str:
    """Format one trajectory message for the grading prompt."""
    role = str(_get_message_value(message, "role", "unknown"))
    name = _get_message_value(message, "name")
    tool_calls = _get_message_value(message, "tool_calls")
    content = _normalize_content(_get_message_value(message, "content", ""))
    content = content.strip()

    if len(content) > MAX_SINGLE_MESSAGE_CHARS:
        content = f"{content[:MAX_SINGLE_MESSAGE_CHARS]}...[truncated]"

    lines = [f"<MESSAGE reverse_index=\"{reverse_index}\" role=\"{role}\">"]
    if name:
        lines.append(f"<NAME>{name}</NAME>")

    if tool_calls:
        tool_names = []
        for tool_call in tool_calls:
            function = _get_message_value(tool_call, "function")
            if isinstance(function, dict):
                tool_name = function.get("name")
            else:
                tool_name = getattr(function, "name", None)
            if tool_name:
                tool_names.append(str(tool_name))
        if tool_names:
            lines.append(f"<TOOL_CALLS>{', '.join(tool_names)}</TOOL_CALLS>")

    lines.append(content or "(empty)")
    lines.append("</MESSAGE>")
    return "\n".join(lines)


def _build_trajectory_excerpt(
    messages: list[Any],
    max_messages: int,
    max_chars: int,
) -> tuple[str, int]:
    """
    Build a backward-scanned trajectory excerpt while preserving chronological order.

    We walk from the back of the trajectory for efficiency, then reverse the selected
    slice so the judge sees the retained messages in their original order.
    """
    selected: list[str] = []
    total_chars = 0

    for reverse_index, message in enumerate(reversed(messages), start=1):
        if len(selected) >= max_messages:
            break

        formatted = _format_message(message, reverse_index)
        next_chars = total_chars + len(formatted)
        if selected and next_chars > max_chars:
            break

        selected.append(formatted)
        total_chars = next_chars

    selected.reverse()
    return "\n\n".join(selected), len(selected)


def _build_trajectory_prompt(
    *,
    task_prompt: str | None,
    final_answer: str,
    trajectory_excerpt: str,
    criteria: str,
) -> str:
    task_section = ""
    if task_prompt:
        task_section = f"<ORIGINAL_TASK>\n{task_prompt}\n</ORIGINAL_TASK>\n\n"

    final_answer_section = final_answer or "(No final answer provided)"
    history_section = trajectory_excerpt or "(No trajectory history provided)"

    return (
        f"{task_section}"
        f"<FINAL_ANSWER>\n{final_answer_section}\n</FINAL_ANSWER>\n\n"
        f"<RECENT_TRAJECTORY>\n{history_section}\n</RECENT_TRAJECTORY>\n\n"
        f"<VERIFICATION_CRITERIA>\n{criteria}\n</VERIFICATION_CRITERIA>\n\n"
        "<REMINDER>\n"
        "- Start with the FINAL_ANSWER.\n"
        "- Use RECENT_TRAJECTORY only as supporting evidence.\n"
        "- If the criterion is not clearly supported, return false.\n"
        "- Return JSON with rationale and is_criteria_true.\n"
        "</REMINDER>"
    )


async def trajectory_llm_eval(input: EvalImplInput) -> VerifierResult:
    """Grade agent trajectory messages against a criterion using an LLM judge."""
    verifier_values = input.verifier.verifier_values or {}
    task_id = input.verifier.task_id or "unknown"
    criteria = verifier_values.get("criteria", "")

    log_grader_start(task_id, criteria, is_negative=False)

    if not criteria:
        raise ValueError("Missing required field: criteria")

    try:
        if not input.helper_results:
            raise ValueError("Missing helper results")

        # Re-use final answer and widen into trajectory as needed.
        final_answer = str(input.helper_results.get(HelperIds.FINAL_ANSWER, "") or "")
        model = input.grading_settings.llm_judge_model
        extra_args = input.grading_settings.llm_judge_extra_args
        task_prompt = extract_task_prompt(input)

        eval_config_values = input.eval_config.eval_config_values or {}
        max_messages = int(eval_config_values.get("trajectory_max_messages", DEFAULT_MAX_MESSAGES))
        max_chars = int(eval_config_values.get("trajectory_max_chars", DEFAULT_MAX_CHARS))

        # Iterate from the back and judge against a bounded recent excerpt.
        trajectory_excerpt, evaluated_message_count = _build_trajectory_excerpt(
            input.trajectory.messages,
            max_messages=max_messages,
            max_chars=max_chars,
        )

        user_prompt = _build_trajectory_prompt(
            task_prompt=task_prompt,
            final_answer=final_answer,
            trajectory_excerpt=trajectory_excerpt,
            criteria=criteria,
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
            if not choices or not isinstance(choices[0], Choices):
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty response"
                )
                continue

            raw_content = choices[0].message.content
            if not raw_content:
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty content"
                )
                continue

            try:
                parsed = json.loads(raw_content)
                if isinstance(parsed.get("rationale"), dict):
                    parsed["rationale"] = json.dumps(parsed["rationale"])
                rationale = str(parsed["rationale"])
                is_criteria_true = bool(parsed["is_criteria_true"])
                parsed = {
                    "rationale": rationale,
                    "is_criteria_true": is_criteria_true,
                }
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: {e}"
                )
                parsed = None
                continue

        if parsed is None:
            raise ValueError(f"Invalid JSON after {MAX_JSON_RETRIES} attempts")

        is_criteria_true = parsed["is_criteria_true"]
        rationale = parsed["rationale"]
        judge_grade = "pass" if is_criteria_true else "fail"
        score = 1.0 if is_criteria_true else 0.0

        log_grader_result(
            task_id,
            is_negative=False,
            passed=is_criteria_true,
            score=score,
            criteria=criteria,
        )

        return VerifierResult(
            verifier_id=input.verifier.verifier_id,
            verifier_version=input.verifier.verifier_version,
            score=score,
            verifier_result_values={
                "judge_grade": judge_grade,
                "grade_rationale": rationale,
                "evaluated_message_count": evaluated_message_count,
            },
        )

    except Exception as e:
        error_msg = f"Trajectory LLM grading failed: {str(e)}"
        raise ValueError(error_msg) from e
