"""
execution_tools.py — OpenAI tool schemas + result formatters for the two
execution primitives the LLM uses (`execute_code` and `execute_bash`).

The agent loop registers these schemas with the LLM and dispatches tool
calls to `tools.execute_code.execute_code()` and `tools.bash.execute_bash()`.
The formatters here turn the typed result dataclasses into the plain-text
tool-result strings the LLM will see on its next turn.

Kept in one file because the two tools are conceptually paired: the LLM
uses bash to *navigate* the tool-discovery tree, then code to *call* MCP
tools through the bindings. Formatters live here too so the schema's
description and the result format are reviewed together.
"""

from __future__ import annotations

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.coding_mcp_agent.tools.bash import BashResult
from runner.agents.coding_mcp_agent.tools.execute_code import ExecResult

EXECUTE_CODE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code in the agent process. Top-level `await` is "
            "supported. MCP tools are available via `from servers import "
            "<server_name>` — each module's functions wrap the corresponding "
            "MCP tool and return its raw text result. Stdout is captured and "
            "returned; the namespace does NOT persist across calls. On error, "
            "you receive the full traceback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
        },
    },
}

EXECUTE_BASH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": (
            "Run a shell command via `/bin/sh -c`. Primary use is exploring "
            "`/tmp/mcp-tool-docs/` (the tool-discovery tree) with `ls`, `cat`, "
            "`grep`, etc. Stdin is closed, so interactive prompts will fail "
            "fast. Returns stdout, stderr, and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line to run.",
                },
            },
            "required": ["command"],
        },
    },
}


def format_exec_result(result: ExecResult) -> str:
    """Render an ExecResult as the tool-result string the LLM will see.

    On success: just the stdout (which may be empty — that's a real
    signal, not noise). On failure: any stdout captured before the
    exception, followed by the traceback or timeout message under
    explicit headers so the LLM can tell them apart.
    """
    if result.success:
        return result.stdout if result.stdout else "(no output)"

    parts: list[str] = []
    if result.stdout:
        parts.append(f"[stdout]\n{result.stdout.rstrip()}")
    if result.timed_out:
        parts.append(f"[timeout]\n{result.error or 'Execution timed out.'}")
    elif result.error:
        parts.append(f"[error]\n{result.error.rstrip()}")
    return "\n".join(parts) if parts else "(no output, unknown failure)"


def format_bash_result(result: BashResult) -> str:
    """Render a BashResult as the tool-result string the LLM will see.

    Always includes the exit code so the LLM can branch on it without
    re-running the command. Stderr appears under an explicit header only
    when non-empty, to avoid noise on successful runs.
    """
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")

    parts.append(
        f"[exit_code={result.exit_code}{', timed_out' if result.timed_out else ''}]"
    )

    if result.hint:
        parts.append(f"[hint]\n{result.hint}")

    return "\n".join(parts)
