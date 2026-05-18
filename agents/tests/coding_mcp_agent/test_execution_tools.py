"""Unit tests for runner.agents.coding_mcp_agent.tools.execution_tools.

These tests pin the exact tool-result strings the LLM will see, because
the system prompt references that format. If `format_*` output changes,
the system prompt must change too — failing tests catch that drift.

The schemas themselves are mostly static dicts; we only sanity-check the
name and required-fields surface so an accidental rename can't ship.
"""

from __future__ import annotations

import json

from runner.agents.coding_mcp_agent.tools.bash import BashResult
from runner.agents.coding_mcp_agent.tools.execute_code import ExecResult
from runner.agents.coding_mcp_agent.tools.execution_tools import (
    EXECUTE_BASH_TOOL,
    EXECUTE_CODE_TOOL,
    format_bash_result,
    format_exec_result,
)

_CODE_SCHEMA = json.loads(json.dumps(EXECUTE_CODE_TOOL))
_BASH_SCHEMA = json.loads(json.dumps(EXECUTE_BASH_TOOL))

# ---------------------------------------------------------------------------
# format_exec_result
# ---------------------------------------------------------------------------


class TestFormatExecResult:
    def test_success_returns_stdout_verbatim(self):
        result = ExecResult(stdout="hello world\n", error=None)
        assert format_exec_result(result) == "hello world\n"

    def test_success_with_empty_stdout_returns_no_output_marker(self):
        """An empty success is real (LLM ran code that prints nothing) but
        an empty tool result would confuse the LLM — surface it explicitly."""
        result = ExecResult(stdout="", error=None)
        assert format_exec_result(result) == "(no output)"

    def test_error_with_traceback_uses_error_header(self):
        result = ExecResult(
            stdout="", error="Traceback (most recent call last):\nNameError: x"
        )
        out = format_exec_result(result)
        assert "[error]" in out
        assert "NameError: x" in out
        assert "[stdout]" not in out

    def test_error_with_partial_stdout_includes_both_sections(self):
        """If code printed before crashing, the LLM needs both halves to debug."""
        result = ExecResult(stdout="row 1\nrow 2\n", error="Traceback: KeyError")
        out = format_exec_result(result)
        assert "[stdout]" in out
        assert "row 1" in out
        assert "[error]" in out
        assert "KeyError" in out

    def test_timeout_uses_timeout_header_not_error(self):
        """Timeout is a distinct failure mode — surface it under its own header
        so the LLM doesn't conflate it with a code-level traceback."""
        result = ExecResult(
            stdout="partial output",
            error="Execution timed out after 240s",
            timed_out=True,
        )
        out = format_exec_result(result)
        assert "[timeout]" in out
        assert "[error]" not in out
        assert "partial output" in out


# ---------------------------------------------------------------------------
# format_bash_result
# ---------------------------------------------------------------------------


class TestFormatBashResult:
    def test_success_includes_stdout_and_exit_code(self):
        result = BashResult(stdout="hello\n", stderr="", exit_code=0)
        out = format_bash_result(result)
        assert "hello" in out
        assert "[exit_code=0]" in out
        assert "[stderr]" not in out

    def test_non_zero_exit_is_reported(self):
        result = BashResult(stdout="", stderr="permission denied\n", exit_code=1)
        out = format_bash_result(result)
        assert "[exit_code=1]" in out
        assert "[stderr]" in out
        assert "permission denied" in out

    def test_stderr_only_shown_when_present(self):
        """No stderr → no [stderr] header (avoid noise on clean runs)."""
        result = BashResult(stdout="data", stderr="", exit_code=0)
        assert "[stderr]" not in format_bash_result(result)

    def test_timeout_marker_included_in_exit_code_block(self):
        result = BashResult(stdout="", stderr="", exit_code=-1, timed_out=True)
        out = format_bash_result(result)
        assert "timed_out" in out
        assert "exit_code=-1" in out

    def test_interactivity_hint_appears_under_hint_header(self):
        """When bash detects an interactive-prompt failure mode, the hint
        tells the LLM how to retry — must be surfaced distinctly."""
        result = BashResult(
            stdout="",
            stderr="sudo: a password is required",
            exit_code=1,
            hint="Re-run with --no-input or pipe the value.",
        )
        out = format_bash_result(result)
        assert "[hint]" in out
        assert "Re-run with --no-input" in out

    def test_empty_stdout_does_not_add_blank_line(self):
        """Empty stdout shouldn't produce a leading blank — only stderr +
        exit code should appear."""
        result = BashResult(stdout="", stderr="oops\n", exit_code=2)
        out = format_bash_result(result)
        assert not out.startswith("\n")


# ---------------------------------------------------------------------------
# Schema surface — guard against accidental name/field renames.
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_execute_code_schema_name(self):
        assert EXECUTE_CODE_TOOL["function"]["name"] == "execute_code"

    def test_execute_code_requires_code_field(self):
        params = _CODE_SCHEMA["function"]["parameters"]
        assert params["required"] == ["code"]
        assert params["properties"]["code"]["type"] == "string"

    def test_execute_bash_schema_name(self):
        assert EXECUTE_BASH_TOOL["function"]["name"] == "execute_bash"

    def test_execute_bash_requires_command_field(self):
        params = _BASH_SCHEMA["function"]["parameters"]
        assert params["required"] == ["command"]
        assert params["properties"]["command"]["type"] == "string"
