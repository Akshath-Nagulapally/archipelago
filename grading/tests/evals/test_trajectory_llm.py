from runner.evals.models import EvalIds
from runner.evals.registry import EVAL_REGISTRY
from runner.evals.trajectory_llm.main import (
    _build_trajectory_excerpt,
    _format_message,
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

    excerpt, evaluated_message_count = _build_trajectory_excerpt(
        messages,
        max_messages=2,
        max_chars=10_000,
    )

    assert evaluated_message_count == 2
    assert '<MESSAGE index="3"' in excerpt
    assert '<MESSAGE index="4"' in excerpt
    assert '<MESSAGE index="2"' not in excerpt
    assert excerpt.index('<MESSAGE index="3"') < excerpt.index('<MESSAGE index="4"')
