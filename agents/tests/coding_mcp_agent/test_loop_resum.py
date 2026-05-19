"""Integration tests for ReSumManager in AgentLoop.

Verifies that AgentLoop correctly:
- Proactively summarizes when ReSumManager.should_summarize() returns True
- Handles ContextWindowExceededError by triggering reactive summarization
- Swallows summarization failures and keeps the loop alive
- Leaves messages unchanged when summarization is not needed

All LLM calls are mocked — no network I/O.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import ContextWindowExceededError

from runner.agents.coding_mcp_agent.loop import AgentLoop
from runner.agents.models import LitellmOutputMessage, get_msg_content

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(**kwargs: Any) -> AgentLoop:
    defaults: dict[str, Any] = {
        "trajectory_id": "test-traj",
        "model": "gpt-4o",
        "initial_messages": [],
        "agent_config_values": {},
        "extra_args": {},
    }
    defaults.update(kwargs)
    return AgentLoop(**defaults)


def _final_answer_response() -> MagicMock:
    """Build a mock ModelResponse that calls final_answer."""
    tool_call = MagicMock()
    tool_call.id = "tc1"
    tool_call.function.name = "final_answer"
    tool_call.function.arguments = '{"answer": "done", "status": "completed"}'

    message = MagicMock()
    message.role = "assistant"
    message.content = None
    message.tool_calls = [tool_call]
    message.reasoning_content = None
    message.thinking_blocks = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


def _no_tool_response(content: str = "thinking...") -> MagicMock:
    """Build a mock ModelResponse with no tool calls (text only)."""
    message = MagicMock()
    message.role = "assistant"
    message.content = content
    message.tool_calls = None
    message.reasoning_content = None
    message.thinking_blocks = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


# ---------------------------------------------------------------------------
# Proactive summarization
# ---------------------------------------------------------------------------


class TestProactiveSummarization:
    @pytest.mark.asyncio
    async def test_summarize_called_when_should_summarize_true(self):
        loop = _make_loop()
        loop.messages = [
            LitellmOutputMessage(role="user", content=f"msg{i}") for i in range(5)
        ]

        summarized_msgs = [LitellmOutputMessage(role="user", content="SUMMARY")]

        loop._resum.should_summarize = MagicMock(return_value=True)  # type: ignore[method-assign]
        loop._resum.summarize = AsyncMock(return_value=summarized_msgs)  # type: ignore[method-assign]

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            return_value=_final_answer_response(),
        ):
            await loop.step()

        loop._resum.summarize.assert_called_once()
        assert get_msg_content(loop.messages[0]) == "SUMMARY"

    @pytest.mark.asyncio
    async def test_summarize_not_called_when_should_summarize_false(self):
        loop = _make_loop()
        loop._resum.should_summarize = MagicMock(return_value=False)  # type: ignore[method-assign]
        loop._resum.summarize = AsyncMock()  # type: ignore[method-assign]

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            return_value=_final_answer_response(),
        ):
            await loop.step()

        loop._resum.summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarization_failure_is_swallowed(self):
        loop = _make_loop()
        loop._resum.should_summarize = MagicMock(return_value=True)  # type: ignore[method-assign]
        loop._resum.summarize = AsyncMock(side_effect=RuntimeError("LLM down"))  # type: ignore[method-assign]

        original_messages = list(loop.messages)

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            return_value=_final_answer_response(),
        ):
            # Must not raise — loop should continue despite summarization failure
            await loop.step()

        # Messages unchanged when summarize blew up
        assert loop.messages[: len(original_messages)] == original_messages


# ---------------------------------------------------------------------------
# Reactive summarization (ContextWindowExceededError)
# ---------------------------------------------------------------------------


class TestReactiveSummarization:
    @pytest.mark.asyncio
    async def test_context_window_error_triggers_summarize(self):
        loop = _make_loop()
        summarized = [LitellmOutputMessage(role="user", content="compressed")]
        loop._resum.summarize = AsyncMock(return_value=summarized)  # type: ignore[method-assign]
        loop._resum.should_summarize = MagicMock(return_value=False)  # type: ignore[method-assign]

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            side_effect=ContextWindowExceededError(
                message="context length exceeded", llm_provider="openai", model="gpt-4o"
            ),
        ):
            # step() returns None (no response) — loop continues next step
            await loop.step()

        loop._resum.summarize.assert_called_once()
        assert loop.messages == summarized

    @pytest.mark.asyncio
    async def test_context_window_error_swallows_secondary_summarize_failure(self):
        loop = _make_loop()
        loop._resum.should_summarize = MagicMock(return_value=False)  # type: ignore[method-assign]
        loop._resum.summarize = AsyncMock(
            side_effect=RuntimeError("summarize also broken")
        )  # type: ignore[method-assign]

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            side_effect=ContextWindowExceededError(
                message="context length exceeded", llm_provider="openai", model="gpt-4o"
            ),
        ):
            # Must not propagate the summarize failure
            await loop.step()

    @pytest.mark.asyncio
    async def test_non_context_error_is_reraised(self):
        loop = _make_loop()
        loop._resum.should_summarize = MagicMock(return_value=False)  # type: ignore[method-assign]

        with patch(
            "runner.agents.coding_mcp_agent.loop.generate_response",
            new_callable=AsyncMock,
            side_effect=ValueError("unexpected error"),
        ):
            with pytest.raises(ValueError):
                await loop.step()


# ---------------------------------------------------------------------------
# Loop run() integration
# ---------------------------------------------------------------------------


class TestRunWithResum:
    @pytest.mark.asyncio
    async def test_run_completes_after_proactive_resum(self):
        loop = _make_loop(agent_config_values={"max_steps": 5})

        summarized = [LitellmOutputMessage(role="user", content="summary")]
        loop._resum.should_summarize = MagicMock(return_value=True)  # type: ignore[method-assign]
        loop._resum.summarize = AsyncMock(return_value=summarized)  # type: ignore[method-assign]

        # Patch _call_llm to return None (timeout-like) on first call, then
        # finalize the loop on second call so run() exits cleanly.
        call_count = 0

        async def fake_call_llm() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                loop._finalized = True
                loop._final_answer = "done"
                loop._final_status = "completed"
            return None

        loop._call_llm = fake_call_llm  # type: ignore[method-assign]

        output = await loop.run()

        assert output.status.value == "completed"
        loop._resum.summarize.assert_called()  # type: ignore[union-attr]
