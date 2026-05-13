import json

from runner.evals.models import EvalIds
from runner.evals.registry import EVAL_REGISTRY
from runner.evals.trajectory_llm.main import (
    _build_verifier_result_values,
    _extract_json_object,
    _parse_trajectory_judge_response,
)
from runner.evals.trajectory_llm.utils.prompts import (
    TRAJECTORY_GRADING_SYSTEM_PROMPT,
    build_trajectory_prompt,
)
from runner.evals.trajectory_llm.utils.trajectory_formatting import (
    _format_message,
    build_trajectory_excerpt,
)
from runner.helpers.models import HelperIds


def test_trajectory_eval_registered() -> None:
    eval_defn = EVAL_REGISTRY[EvalIds.TRAJECTORY_LLM]

    assert eval_defn.eval_id == EvalIds.TRAJECTORY_LLM
    assert eval_defn.helper_dependencies == [HelperIds.FINAL_ANSWER]
    assert eval_defn.eval_impl is not None


def test_formats_tool_calls_with_args() -> None:
    message = {
        "role": "assistant",
        "content": "I will inspect the notice.",
        "tool_calls": [
            {
                "id": "call_read_1",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/notices/warn.txt"}',
                },
            }
        ],
    }

    formatted = _format_message(message, message_index=2, reverse_index=3)

    assert '<MESSAGE index="2" reverse_index="3" role="assistant">' in formatted
    assert '<TOOL_CALL index="1" id="call_read_1" name="read_file">' in formatted
    assert '{"path": "/tmp/notices/warn.txt"}' in formatted
    assert "<CONTENT>\nI will inspect the notice.\n</CONTENT>" in formatted


def test_formats_tool_outputs_with_call_id_and_name() -> None:
    message = {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "call_read_1",
        "content": [{"type": "text", "text": "Notice was sent 60 days ahead."}],
    }

    formatted = _format_message(message, message_index=3, reverse_index=2)

    assert '<MESSAGE index="3" reverse_index="2" role="tool">' in formatted
    assert "<NAME>read_file</NAME>" in formatted
    assert "<TOOL_CALL_ID>call_read_1</TOOL_CALL_ID>" in formatted
    assert "Notice was sent 60 days ahead." in formatted


def test_trajectory_excerpt_keeps_recent_messages_in_chronological_order() -> None:
    messages = [
        {"role": "user", "content": "Old task context."},
        {"role": "assistant", "content": "Older assistant turn."},
        {
            "role": "tool",
            "name": "search_law",
            "content": "Federal WARN Act requires 60 days notice.",
        },
        {"role": "assistant", "content": "The notice complied with both Acts."},
    ]

    excerpt, evaluated_message_count = build_trajectory_excerpt(
        messages,
        max_messages=2,
        max_chars=10_000,
    )

    assert evaluated_message_count == 2
    assert '<MESSAGE index="3"' in excerpt
    assert '<MESSAGE index="4"' in excerpt
    assert '<MESSAGE index="2"' not in excerpt
    assert excerpt.index('<MESSAGE index="3"') < excerpt.index('<MESSAGE index="4"')


def test_trajectory_prompt_requests_process_dimension_scores() -> None:
    prompt = build_trajectory_prompt(
        task_prompt="Find the notice and assess compliance.",
        final_answer="The notice complied.",
        trajectory_excerpt='<MESSAGE index="1" role="user">Task</MESSAGE>',
        criteria="Complete the compliance assessment.",
        criteria_explanation="Use trajectory evidence.",
    )

    assert "tool_use_score" in TRAJECTORY_GRADING_SYSTEM_PROMPT
    assert "grounding_score" in TRAJECTORY_GRADING_SYSTEM_PROMPT
    assert "recovery_score" in TRAJECTORY_GRADING_SYSTEM_PROMPT
    assert "overall_score" in TRAJECTORY_GRADING_SYSTEM_PROMPT
    assert "is_criteria_true" not in TRAJECTORY_GRADING_SYSTEM_PROMPT
    assert "<CRITERIA_EXPLANATION>\nUse trajectory evidence." in prompt
    assert "<TRAJECTORY>" in prompt


def test_parse_trajectory_judge_response_validates_structured_json() -> None:
    response = _parse_trajectory_judge_response(
        json.dumps(
            {
                "tool_use_score": 4,
                "grounding_score": 5,
                "recovery_score": 3,
                "efficiency_score": 4,
                "failure_type": "none",
                "failure_step_idx": None,
                "critical_step_idxs": None,
                "rationale": {"summary": "Mostly successful."},
            }
        )
    )

    assert response.tool_use_score == 4
    assert response.critical_step_idxs == []
    assert response.rationale == '{"summary": "Mostly successful."}'


def test_parse_trajectory_judge_response_handles_fenced_json_and_aliases() -> None:
    response = _parse_trajectory_judge_response(
        """```json
        {
          "tool_use_score": 5,
          "grounding_score": 5,
          "recovery_score": 4,
          "efficiency_score": 5,
          "failure_type": "no_failure",
          "failure_step_idx": "N/A",
          "critical_step_idxs": [30],
          "rationale": "The agent found the requested path."
        }
        ```"""
    )

    assert response.failure_type == "none"
    assert response.failure_step_idx is None
    assert response.critical_step_idxs == [30]


def test_parse_trajectory_judge_response_handles_prefixed_fenced_json() -> None:
    response = _parse_trajectory_judge_response(
        """Here is the evaluation:
        ```json
        {
          "tool_use_score": 5,
          "grounding_score": 5,
          "recovery_score": 5,
          "efficiency_score": 5,
          "failure_type": "none",
          "failure_step_idx": null,
          "critical_step_idxs": [30],
          "rationale": "The trajectory executed cleanly."
        }
        ```"""
    )

    assert response.tool_use_score == 5
    assert response.failure_type == "none"
    assert response.critical_step_idxs == [30]


def test_extract_json_object_handles_prefixed_model_output() -> None:
    raw_content = 'Here is the evaluation:\n{"tool_use_score": 5}\nThanks.'

    assert _extract_json_object(raw_content) == '{"tool_use_score": 5}'


def test_build_verifier_result_values_includes_dimension_scores() -> None:
    response = _parse_trajectory_judge_response(
        json.dumps(
            {
                "tool_use_score": 2,
                "grounding_score": 1,
                "recovery_score": 3,
                "efficiency_score": 3,
                "failure_type": "hallucination",
                "failure_step_idx": 6,
                "critical_step_idxs": [4, 6],
                "rationale": "The agent hallucinated tool outputs.",
            }
        )
    )

    values = _build_verifier_result_values(
        response,
        overall_score=2,
        evaluated_message_count=8,
    )

    assert values["judge_grade"] == "fail"
    assert values["tool_use_score"] == 2
    assert values["grounding_score"] == 1
    assert values["overall_score"] == 2
    assert values["failure_type"] == "hallucination"
    assert values["failure_step_idx"] == 6
    assert values["critical_step_idxs"] == [4, 6]
    assert values["evaluated_message_count"] == 8
