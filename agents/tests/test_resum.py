"""Unit tests for runner.agents.resum.ReSumManager.

Covers:
- _find_safe_cut_index: boundary safety, orphaned-tool prevention
- ReSumManager.should_summarize: token threshold and new-message guard
- ReSumManager.summarize: message reduction, running summary accumulation,
  system message preservation, safe-cut integration
- ReSumManager._build_output: output shape (system + single user message)
- ReSumManager._format_messages / _format_tool_calls: roles, truncation

All LLM calls are mocked. No network I/O.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from runner.agents.models import (
    LitellmAnyMessage,
    LitellmOutputMessage,
    get_msg_content,
    get_msg_role,
)
from runner.agents.resum import (
    KEEP_RECENT_MESSAGES,
    ReSumManager,
    _find_safe_cut_index,
)

# ---------------------------------------------------------------------------
# Helpers — typed as LitellmAnyMessage so lists pass invariant checks
# ---------------------------------------------------------------------------


def _make_user(content: str) -> LitellmAnyMessage:
    return LitellmOutputMessage(role="user", content=content)


def _make_assistant(
    content: str = "", tool_calls: list[Any] | None = None
) -> LitellmAnyMessage:
    msg = LitellmOutputMessage(role="assistant", content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _make_tool(name: str, content: str, call_id: str = "c1") -> LitellmAnyMessage:
    return LitellmOutputMessage(
        role="tool",
        tool_call_id=call_id,
        name=name,
        content=content,
    )


def _make_system(content: str) -> LitellmAnyMessage:
    return LitellmOutputMessage(role="system", content=content)


def _make_tool_call(name: str, args: str = "{}") -> Any:
    func = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(function=func)


# ---------------------------------------------------------------------------
# _find_safe_cut_index
# ---------------------------------------------------------------------------


class TestFindSafeCutIndex:
    def test_empty_messages_returns_zero(self):
        assert _find_safe_cut_index([], 5) == 0

    def test_short_list_returns_zero(self):
        msgs: list[LitellmAnyMessage] = [_make_user("a"), _make_user("b")]
        assert _find_safe_cut_index(msgs, 10) == 0

    def test_naive_cut_on_non_tool_message_is_safe(self):
        msgs: list[LitellmAnyMessage] = [_make_user(f"msg{i}") for i in range(20)]
        idx = _find_safe_cut_index(msgs, KEEP_RECENT_MESSAGES)
        assert idx == 20 - KEEP_RECENT_MESSAGES
        assert get_msg_role(msgs[idx]) != "tool"

    def test_walks_back_past_orphaned_tool_messages(self):
        msgs: list[LitellmAnyMessage] = [_make_user(f"u{i}") for i in range(8)]
        msgs.append(_make_assistant("thinking"))
        msgs.append(_make_tool("bash", "output1"))
        msgs.append(_make_tool("bash", "output2"))
        # keep=2 → naive cut = 9 → points at tool[0] → must walk back to assistant
        idx = _find_safe_cut_index(msgs, 2)
        assert get_msg_role(msgs[idx]) != "tool"

    def test_all_tool_messages_returns_zero(self):
        msgs: list[LitellmAnyMessage] = [_make_tool("t", "x") for _ in range(15)]
        idx = _find_safe_cut_index(msgs, 5)
        assert idx == 0


# ---------------------------------------------------------------------------
# ReSumManager.should_summarize
# ---------------------------------------------------------------------------


class TestShouldSummarize:
    def _manager(self, max_tokens: int = 10000) -> ReSumManager:
        mgr = ReSumManager(model="gpt-4o")
        mgr.max_tokens = max_tokens
        return mgr

    def test_false_when_not_enough_new_messages(self):
        mgr = self._manager()
        msgs: list[LitellmAnyMessage] = [
            _make_user(f"m{i}") for i in range(KEEP_RECENT_MESSAGES)
        ]
        assert mgr.should_summarize(msgs) is False

    def test_false_when_below_token_threshold(self):
        mgr = self._manager(max_tokens=1_000_000)
        msgs: list[LitellmAnyMessage] = [
            _make_user(f"msg{i}") for i in range(KEEP_RECENT_MESSAGES + 5)
        ]
        assert mgr.should_summarize(msgs) is False

    def test_true_when_above_threshold(self):
        mgr = self._manager(max_tokens=1)
        msgs: list[LitellmAnyMessage] = [
            _make_user("x" * 100) for _ in range(KEEP_RECENT_MESSAGES + 5)
        ]
        assert mgr.should_summarize(msgs) is True

    def test_system_messages_excluded_from_new_message_count(self):
        mgr = self._manager(max_tokens=1)
        msgs: list[LitellmAnyMessage] = [_make_system("sys") for _ in range(50)]
        assert mgr.should_summarize(msgs) is False

    def test_false_when_already_summarized_enough(self):
        mgr = self._manager(max_tokens=1)
        msgs: list[LitellmAnyMessage] = [
            _make_user(f"m{i}") for i in range(KEEP_RECENT_MESSAGES + 5)
        ]
        mgr.messages_summarized = len(msgs)
        assert mgr.should_summarize(msgs) is False


# ---------------------------------------------------------------------------
# ReSumManager.summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def _manager_with_mock_llm(self, summary_text: str = "SUMMARY") -> ReSumManager:
        mgr = ReSumManager(model="gpt-4o")
        mgr._call_llm = AsyncMock(return_value=summary_text)  # type: ignore[method-assign]
        return mgr

    @pytest.mark.asyncio
    async def test_returns_fewer_messages_than_input(self):
        mgr = self._manager_with_mock_llm()
        msgs: list[LitellmAnyMessage] = [_make_user(f"msg{i}") for i in range(30)]
        result = await mgr.summarize(msgs)
        assert len(result) < len(msgs)

    @pytest.mark.asyncio
    async def test_system_messages_preserved_at_start(self):
        mgr = self._manager_with_mock_llm()
        msgs: list[LitellmAnyMessage] = [_make_system("You are an agent.")] + [
            _make_user(f"m{i}") for i in range(30)
        ]
        result = await mgr.summarize(msgs)
        assert get_msg_role(result[0]) == "system"
        assert get_msg_content(result[0]) == "You are an agent."

    @pytest.mark.asyncio
    async def test_running_summary_updated_after_summarize(self):
        mgr = self._manager_with_mock_llm("NEW SUMMARY")
        msgs: list[LitellmAnyMessage] = [_make_user(f"m{i}") for i in range(30)]
        await mgr.summarize(msgs)
        assert mgr.running_summary == "NEW SUMMARY"

    @pytest.mark.asyncio
    async def test_second_summarize_includes_previous_summary_in_prompt(self):
        mock_llm = AsyncMock(return_value="UPDATED SUMMARY")
        mgr = ReSumManager(model="gpt-4o")
        mgr._call_llm = mock_llm  # type: ignore[method-assign]
        mgr.running_summary = "PREVIOUS SUMMARY"
        msgs: list[LitellmAnyMessage] = [_make_user(f"m{i}") for i in range(30)]
        await mgr.summarize(msgs)
        call_arg: str = mock_llm.call_args[0][0]
        assert "PREVIOUS SUMMARY" in call_arg

    @pytest.mark.asyncio
    async def test_output_ends_with_continue_prompt(self):
        mgr = self._manager_with_mock_llm("SUMMARY")
        msgs: list[LitellmAnyMessage] = [_make_user(f"m{i}") for i in range(30)]
        result = await mgr.summarize(msgs)
        last_user_content = next(
            get_msg_content(m) for m in reversed(result) if get_msg_role(m) == "user"
        )
        assert "Continue from this state" in (last_user_content or "")

    @pytest.mark.asyncio
    async def test_short_list_returns_unchanged(self):
        mgr = self._manager_with_mock_llm()
        msgs: list[LitellmAnyMessage] = [
            _make_user(f"m{i}") for i in range(KEEP_RECENT_MESSAGES)
        ]
        result = await mgr.summarize(msgs)
        assert len(result) == len(msgs)

    @pytest.mark.asyncio
    async def test_no_llm_call_when_nothing_to_summarize(self):
        mock_llm = AsyncMock(return_value="SUMMARY")
        mgr = ReSumManager(model="gpt-4o")
        mgr._call_llm = mock_llm  # type: ignore[method-assign]
        msgs: list[LitellmAnyMessage] = [
            _make_user(f"m{i}") for i in range(KEEP_RECENT_MESSAGES)
        ]
        await mgr.summarize(msgs)
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_messages_summarized_resets_to_zero_after_summarize(self):
        mgr = self._manager_with_mock_llm()
        msgs: list[LitellmAnyMessage] = [_make_user(f"m{i}") for i in range(30)]
        await mgr.summarize(msgs)
        assert mgr.messages_summarized == 0


# ---------------------------------------------------------------------------
# ReSumManager._build_output
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_no_summary_no_recent_returns_only_system(self):
        mgr = ReSumManager(model="gpt-4o")
        result = mgr._build_output([_make_system("sys")], [])
        assert all(get_msg_role(m) == "system" for m in result)

    def test_with_summary_adds_user_message(self):
        mgr = ReSumManager(model="gpt-4o")
        mgr.running_summary = "some summary"
        result = mgr._build_output([], [_make_user("recent")])
        user_msgs = [m for m in result if get_msg_role(m) == "user"]
        assert len(user_msgs) == 1
        assert "some summary" in (get_msg_content(user_msgs[0]) or "")

    def test_recent_activity_included_in_user_message(self):
        mgr = ReSumManager(model="gpt-4o")
        mgr.running_summary = "summary"
        result = mgr._build_output([], [_make_user("recent work")])
        user_content = next(
            get_msg_content(m) for m in result if get_msg_role(m) == "user"
        )
        assert "Recent Activity" in (user_content or "")

    def test_system_messages_come_before_user_message(self):
        mgr = ReSumManager(model="gpt-4o")
        mgr.running_summary = "s"
        result = mgr._build_output([_make_system("sys")], [_make_user("r")])
        assert get_msg_role(result[0]) == "system"
        assert get_msg_role(result[-1]) == "user"


# ---------------------------------------------------------------------------
# ReSumManager._format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def _mgr(self) -> ReSumManager:
        return ReSumManager(model="gpt-4o")

    def test_user_role_uppercased(self):
        mgr = self._mgr()
        out = mgr._format_messages([_make_user("hello")])
        assert "**USER**" in out

    def test_tool_role_includes_name(self):
        mgr = self._mgr()
        out = mgr._format_messages([_make_tool("bash", "output")])
        assert "**TOOL (bash)**" in out

    def test_long_user_content_truncated(self):
        mgr = self._mgr()
        out = mgr._format_messages([_make_user("x" * 3000)])
        assert "[truncated]" in out

    def test_long_tool_content_truncated(self):
        mgr = self._mgr()
        out = mgr._format_messages([_make_tool("t", "y" * 1500)])
        assert "[truncated]" in out

    def test_assistant_with_tool_calls_included(self):
        mgr = self._mgr()
        tc = _make_tool_call("execute_code", '{"code": "print(1)"}')
        out = mgr._format_messages([_make_assistant(tool_calls=[tc])])
        assert "execute_code" in out

    def test_multiple_messages_joined(self):
        mgr = self._mgr()
        out = mgr._format_messages([_make_user("a"), _make_assistant("b")])
        assert "**USER**" in out
        assert "**ASSISTANT**" in out


# ---------------------------------------------------------------------------
# ReSumManager._format_tool_calls
# ---------------------------------------------------------------------------


class TestFormatToolCalls:
    def _mgr(self) -> ReSumManager:
        return ReSumManager(model="gpt-4o")

    def test_pydantic_tool_call_formatted(self):
        mgr = self._mgr()
        out = mgr._format_tool_calls([_make_tool_call("my_tool", '{"key": "val"}')])
        assert "my_tool" in out

    def test_dict_tool_call_formatted(self):
        mgr = self._mgr()
        tc = {"function": {"name": "dict_tool", "arguments": '{"x": 1}'}}
        out = mgr._format_tool_calls([tc])
        assert "dict_tool" in out

    def test_long_args_truncated(self):
        mgr = self._mgr()
        out = mgr._format_tool_calls([_make_tool_call("t", "a" * 300)])
        assert "..." in out

    def test_unknown_shape_skipped_gracefully(self):
        mgr = self._mgr()
        out = mgr._format_tool_calls(["not_a_tool_call"])  # type: ignore[list-item]
        assert "Tool calls:" in out
