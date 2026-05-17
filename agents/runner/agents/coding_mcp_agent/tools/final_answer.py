"""
final_answer.py — explicit termination tool for the CodingMCPAgent.

The agent loop never stops on "no tool call" — the LLM must explicitly
call `final_answer(answer, status)` to declare it's done. This mirrors
the ReAct toolbelt agent's contract so trajectories produced by the two
agents have a comparable final-message shape for eval scoring.

`status` distinguishes three end-states:
    completed — task finished successfully
    blocked   — could not proceed (missing capability, unrecoverable error)
    failed    — task is impossible as stated

Copied (without the todo gate) from
`runner/agents/react_toolbelt_agent/tools.py` so the two agents stay
independently editable.
"""

from __future__ import annotations

import json

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

FINAL_ANSWER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit your final answer to complete the task. Call when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your complete final answer to the task.",
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "blocked", "failed"],
                    "description": "completed=done, blocked=cannot proceed, failed=impossible",
                },
            },
            "required": ["answer", "status"],
        },
    },
}


def parse_final_answer(arguments: str) -> tuple[str, str]:
    """Parse final_answer arguments. Returns (answer, status).

    Falls back to (raw_arguments, "completed") if the JSON is malformed —
    we'd rather keep whatever the LLM wrote than discard the run because
    of a stray quote.
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return arguments, "completed"

    if not isinstance(args, dict):
        return str(args), "completed"

    return str(args.get("answer", "")), str(args.get("status", "completed"))
