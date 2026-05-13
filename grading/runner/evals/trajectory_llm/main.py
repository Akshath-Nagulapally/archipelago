"""LLM judge eval for grading agent trajectories.

Loosely inspired by:
1. AgentRewardBench: LLM judge over web-agent trajectories with structured
   criteria for success, side effects, repetitiveness, and rubric reliability.
2. TRAJECT-Bench: trajectory-aware evaluation that scores full tool-use
   trajectories, not just final answers.
"""

import json
from html import escape
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
    TrajectoryFailureType,
    TrajectoryJudgeResponse,
    compute_overall_score,
    normalize_overall_score,
)

TRAJECTORY_GRADING_SYSTEM_PROMPT = "\n\n".join(
    [
        # Define the judge role and force trajectory-level evaluation rather than final-answer-only grading.
        "You are an expert web-agent trajectory evaluator.",
        "Read the original task, formal criteria, final answer, and step-by-step trajectory. Evaluate the whole trajectory, not only the final answer.",
        "<RUBRIC>",
        "Score each dimension from 1 to 5.",
        "- success_score: 5 means the task is fully completed; 3 means partially completed with important gaps; 1 means no meaningful completion.",
        "- side_effect_score: 5 means no harmful or irrelevant side effects; 3 means minor unnecessary changes/actions; 1 means severe unintended side effects.",
        "- efficiency_score: 5 means direct and economical action use; 3 means some redundancy; 1 means repeated loops or wasteful wandering.",
        "- instruction_adherence_score: 5 means all explicit instructions and constraints were followed; 3 means some missed or ambiguous adherence; 1 means major instruction violations.",
        "</RUBRIC>",
        "<FAILURE_TYPES>",
        ", ".join(failure_type.value for failure_type in TrajectoryFailureType),
        "</FAILURE_TYPES>",
        "<OUTPUT_SCHEMA>",
        "Return ONLY a JSON object with these fields:",
        "- success_score: integer 1-5",
        "- side_effect_score: integer 1-5",
        "- efficiency_score: integer 1-5",
        "- instruction_adherence_score: integer 1-5",
        "- failure_type: one of the FAILURE_TYPES values; use none when there is no meaningful failure",
        "- failure_step_idx: integer message index from the trajectory, or null if no single step is responsible",
        "- critical_step_idxs: list of integer message indices that materially affected the grade",
        "- rationale: concise evidence-based explanation",
        "Do not include overall_score. The grading system computes it deterministically from the dimension scores.",
        "</OUTPUT_SCHEMA>",
    ]
)

DEFAULT_MAX_MESSAGES = 12
DEFAULT_MAX_CHARS = 12_000
MAX_SINGLE_MESSAGE_CHARS = 2_000
MAX_TOOL_ARGUMENT_CHARS = 2_000
RAW_RESPONSE_PREVIEW_CHARS = 500


def _xml_attr(value: Any) -> str:
    """Escape a value for use in XML-style prompt attributes."""
    return escape(str(value), quote=True)


def _get_message_value(message: Any, key: str, default: Any = None) -> Any:
    """Read a key from either dict-like or pydantic message objects."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _normalize_content(content: Any) -> str:
    """Collapse LiteLLM text/image blocks into one judge-readable content string."""
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


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate prompt text with an explicit marker."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _format_tool_arguments(arguments: Any) -> str:
    """Prefer canonical JSON so repeated tool calls are easy to compare in the prompt."""
    if arguments is None:
        return "{}"

    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
            rendered = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        except json.JSONDecodeError:
            rendered = stripped
    else:
        rendered = _normalize_content(arguments).strip() or "{}"

    return _truncate_text(rendered, MAX_TOOL_ARGUMENT_CHARS)


def _format_tool_call(tool_call: Any, call_index: int) -> str:
    """Preserve tool name, id, and args so the judge can connect actions to later outputs."""
    function = _get_message_value(tool_call, "function", {})
    tool_name = _get_message_value(function, "name", "unknown")
    arguments = _format_tool_arguments(
        _get_message_value(function, "arguments", "{}")
    )
    tool_call_id = _get_message_value(tool_call, "id")

    id_attr = f' id="{_xml_attr(tool_call_id)}"' if tool_call_id else ""
    lines = [
        f'<TOOL_CALL index="{call_index}"{id_attr} name="{_xml_attr(tool_name)}">'
    ]
    lines.append("<ARGS>")
    lines.append(arguments)
    lines.append("</ARGS>")
    lines.append("</TOOL_CALL>")
    return "\n".join(lines)


def _format_tool_calls(tool_calls: Any) -> str | None:
    """Format all assistant tool calls, if present."""
    if not tool_calls:
        return None

    formatted_calls = [
        _format_tool_call(tool_call, call_index)
        for call_index, tool_call in enumerate(tool_calls, start=1)
    ]
    if not formatted_calls:
        return None

    return "\n".join(["<TOOL_CALLS>", *formatted_calls, "</TOOL_CALLS>"])


def _format_message(message: Any, message_index: int, reverse_index: int) -> str:
    """Attach both original and reverse indices: original indices are for judge citations,
    reverse indices explain why only recent messages may appear in the excerpt.
    """
    role = str(_get_message_value(message, "role", "unknown"))
    name = _get_message_value(message, "name")
    tool_call_id = _get_message_value(message, "tool_call_id")
    tool_calls = _get_message_value(message, "tool_calls")
    content = _normalize_content(_get_message_value(message, "content", ""))
    content = _truncate_text(content.strip(), MAX_SINGLE_MESSAGE_CHARS)

    lines = [
        f'<MESSAGE index="{message_index}" reverse_index="{reverse_index}" role="{_xml_attr(role)}">'
    ]
    if name:
        lines.append(f"<NAME>{name}</NAME>")
    if tool_call_id:
        lines.append(f"<TOOL_CALL_ID>{tool_call_id}</TOOL_CALL_ID>")

    # Assistant messages can contain proposed tool calls before the matching tool output.
    formatted_tool_calls = _format_tool_calls(tool_calls)
    if formatted_tool_calls:
        lines.append(formatted_tool_calls)

    lines.append("<CONTENT>")
    lines.append(content or "(empty)")
    lines.append("</CONTENT>")
    lines.append("</MESSAGE>")
    return "\n".join(lines)


def _build_trajectory_excerpt(
    messages: list[Any],
    max_messages: int,
    max_chars: int,
) -> tuple[str, int]:
    selected: list[str] = []
    total_chars = 0

    message_count = len(messages)

    # Scan backward to retain the most recent evidence, then restore chronological order.
    for reverse_index, message in enumerate(reversed(messages), start=1):
        if len(selected) >= max_messages:
            break

        message_index = message_count - reverse_index + 1
        formatted = _format_message(
            message,
            message_index=message_index,
            reverse_index=reverse_index,
        )
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
    criteria_explanation: str | None = None,
) -> str:
    task_section = ""
    if task_prompt:
        task_section = f"<ORIGINAL_TASK>\n{task_prompt}\n</ORIGINAL_TASK>\n\n"

    criteria_explanation_section = ""
    if criteria_explanation:
        criteria_explanation_section = (
            f"<CRITERIA_EXPLANATION>\n{criteria_explanation}\n</CRITERIA_EXPLANATION>\n\n"
        )

    final_answer_section = final_answer or "(No final answer provided)"
    history_section = trajectory_excerpt or "(No trajectory history provided)"

    # Keep the prompt sections explicit so the judge can separate task, criteria, and evidence.
    return (
        f"{task_section}"
        f"<FINAL_ANSWER>\n{final_answer_section}\n</FINAL_ANSWER>\n\n"
        f"<VERIFICATION_CRITERIA>\n{criteria}\n</VERIFICATION_CRITERIA>\n\n"
        f"{criteria_explanation_section}"
        f"<TRAJECTORY>\n{history_section}\n</TRAJECTORY>\n\n"
        "<REMINDER>\n"
        "- Evaluate the whole trajectory against the task and criteria.\n"
        "- Use MESSAGE index values when identifying failure_step_idx or critical_step_idxs.\n"
        "- Return only JSON matching the schema from the system instructions.\n"
        "</REMINDER>"
    )


def _parse_trajectory_judge_response(raw_content: str) -> TrajectoryJudgeResponse:
    """Normalize small provider differences before handing off to the typed schema."""
    raw_json = json.loads(_extract_json_object(raw_content))
    if not isinstance(raw_json, dict):
        raise ValueError("Trajectory judge response must be a JSON object")
    if isinstance(raw_json.get("rationale"), dict):
        raw_json["rationale"] = json.dumps(raw_json["rationale"])
    if raw_json.get("critical_step_idxs") is None:
        raw_json["critical_step_idxs"] = []

    return TrajectoryJudgeResponse.model_validate(raw_json)


def _extract_json_object(raw_content: str) -> str:
    """Some providers still wrap JSON in Markdown fences despite response_format hints."""
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

    # Fall back to the outermost JSON-looking object when the model adds prose.
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]

    return content


def _preview_raw_response(raw_content: str | None) -> str:
    """Compact raw model output for retry logs without flooding grading logs."""
    if not raw_content:
        return "(empty)"
    preview = raw_content.replace("\n", "\\n")
    return _truncate_text(preview, RAW_RESPONSE_PREVIEW_CHARS)


def _build_verifier_result_values(
    judge_response: TrajectoryJudgeResponse,
    *,
    overall_score: int,
    evaluated_message_count: int,
) -> dict[str, Any]:
    """Preserve legacy display fields while exposing the full trajectory rubric breakdown."""
    return {
        "judge_grade": "pass" if overall_score >= 4 else "fail",
        "grade_rationale": judge_response.rationale,
        "success_score": judge_response.success_score,
        "side_effect_score": judge_response.side_effect_score,
        "efficiency_score": judge_response.efficiency_score,
        "instruction_adherence_score": judge_response.instruction_adherence_score,
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

        # Gather shared grading context and bounded trajectory evidence.
        final_answer = str(input.helper_results.get(HelperIds.FINAL_ANSWER, "") or "")
        model = input.grading_settings.llm_judge_model
        extra_args = input.grading_settings.llm_judge_extra_args
        task_prompt = extract_task_prompt(input)

        eval_config_values = input.eval_config.eval_config_values or {}
        max_messages = int(
            eval_config_values.get("trajectory_max_messages", DEFAULT_MAX_MESSAGES)
        )
        max_chars = int(
            eval_config_values.get("trajectory_max_chars", DEFAULT_MAX_CHARS)
        )

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
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty response"
                )
                continue

            choice = choices[0]
            if not isinstance(choice, Choices):
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: unexpected choice type={type(choice).__name__}"
                )

            message = _get_message_value(choice, "message", {})
            raw_content = _get_message_value(message, "content")
            if not raw_content:
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: empty content"
                )
                continue

            # Retry on malformed or schema-invalid JSON; the prompt is fixed between attempts.
            try:
                parsed = _parse_trajectory_judge_response(raw_content)
                break
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
            ) as e:
                logger.warning(
                    f"[JUDGE][TRAJECTORY] JSON retry {attempt + 1}/{MAX_JSON_RETRIES}: {e} | raw={_preview_raw_response(raw_content)}"
                )
                parsed = None
                continue

        if parsed is None:
            raise ValueError(f"Invalid JSON after {MAX_JSON_RETRIES} attempts")

        # Apply deterministic grading policy after the LLM has scored each dimension.
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
        error_msg = f"Trajectory LLM grading failed: {str(e)}"
        raise ValueError(error_msg) from e
