"""
loop.py — the LLM-driven agent loop for CodingMCPAgent.

`main.py` does the (heavy) setup: opens the MCP gateway client, builds
`servers.<name>` bindings into `sys.modules`, writes the tool-discovery
docs to `/tmp/mcp-tool-docs/`, runs probes. Once that succeeds, control
hands off here.

The loop is intentionally small: three tools (`execute_code`,
`execute_bash`, `final_answer`), dispatched by name. The MCP client is
NOT touched in this file — all MCP interaction happens inside the Python
the LLM writes, which `execute_code` runs in-process. The shared client
opened in `main.initialize()` is captured inside the binding wrappers
that `from servers import ...` resolves to.

Termination is explicit: the LLM must call `final_answer(answer, status)`.
If it returns text with no tool calls we push back and continue. The
ReAct toolbelt agent uses the same contract — see
`react_toolbelt_agent/main.py` for the original.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from litellm import Choices
from litellm.exceptions import Timeout
from litellm.files.main import ModelResponse
from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.coding_mcp_agent.tools.bash import execute_bash
from runner.agents.coding_mcp_agent.tools.execute_code import execute_code
from runner.agents.coding_mcp_agent.tools.execution_tools import (
    EXECUTE_BASH_TOOL,
    EXECUTE_CODE_TOOL,
    format_bash_result,
    format_exec_result,
)
from runner.agents.coding_mcp_agent.tools.final_answer import (
    FINAL_ANSWER_TOOL,
    parse_final_answer,
)
from runner.agents.models import (
    AgentStatus,
    AgentTrajectoryOutput,
    LitellmAnyMessage,
    LitellmOutputMessage,
)
from runner.utils.error import is_system_error
from runner.utils.llm import generate_response
from runner.utils.usage import UsageTracker

_TOOLS: list[ChatCompletionToolParam] = [
    EXECUTE_CODE_TOOL,
    EXECUTE_BASH_TOOL,
    FINAL_ANSWER_TOOL,
]


def _log_response_metadata(message: LitellmOutputMessage) -> None:
    """Log reasoning, thinking blocks, and content from an LLM message.

    All three are optional fields that only show up for certain model
    families (OpenAI o1, Anthropic extended thinking, etc.). We log each
    under its own message_type so downstream log consumers can filter.
    """
    if getattr(message, "reasoning_content", None):
        logger.bind(message_type="reasoning").info(message.reasoning_content)

    thinking_blocks = getattr(message, "thinking_blocks", None)
    if isinstance(thinking_blocks, list):
        for thinking_block in thinking_blocks:
            if thinking_block.get("thinking"):
                logger.bind(message_type="thinking").debug(
                    thinking_block.get("thinking")
                )

    content = getattr(message, "content", None)
    if content:
        logger.bind(message_type="response").info(content)


class AgentLoop:
    """LLM-driven step loop for CodingMCPAgent.

    Owns no MCP state — the gateway client lives on `CodingMCPAgent` and
    is reachable only indirectly, through the binding wrappers captured
    inside the `servers.*` modules.
    """

    trajectory_id: str
    model: str
    messages: list[LitellmAnyMessage]
    extra_args: dict[str, Any]
    max_steps: int
    llm_response_timeout: int
    timeout: int

    _usage_tracker: UsageTracker
    _finalized: bool
    _final_answer: str | None
    _final_status: str
    _start_time: float | None

    def __init__(
        self,
        trajectory_id: str,
        model: str,
        initial_messages: list[LitellmAnyMessage],
        agent_config_values: dict[str, Any],
        extra_args: dict[str, Any] | None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.model = model
        self.messages = list(initial_messages)
        self.extra_args = extra_args or {}

        self.max_steps = agent_config_values.get("max_steps", 100)
        self.llm_response_timeout = agent_config_values.get("llm_response_timeout", 600)
        self.timeout = agent_config_values.get("timeout", 10800)

        self._usage_tracker = UsageTracker()
        self._finalized = False
        self._final_answer = None
        self._final_status = "completed"
        self._start_time = None

    async def step(self) -> None:
        """One LLM call → dispatch any tool calls → append results."""
        response = await self._call_llm()
        if response is None:
            return

        self._usage_tracker.track(response)

        response_message = self._extract_message(response)
        if response_message is None:
            self._push_continue("Continue. Use final_answer when done.")
            return

        _log_response_metadata(response_message)
        self.messages.append(response_message)

        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            # Mirror the ReAct contract: text without a tool call is not termination.
            # The LLM must explicitly call final_answer. Push back and continue.
            self._push_continue(
                "No tools called. Use final_answer to submit your answer "
                "when the task is complete."
            )
            return

        await self._dispatch_tool_calls(tool_calls)

    async def _call_llm(self) -> ModelResponse | None:
        """Wrap generate_response so step()'s control flow stays linear.

        Returns None for a recoverable timeout (caller should just continue
        to the next step). Re-raises on anything else.
        """
        try:
            return await generate_response(
                self.model,
                self.messages,
                _TOOLS,
                self.llm_response_timeout,
                self.extra_args,
                trajectory_id=self.trajectory_id,
            )
        except Timeout:
            logger.bind(message_type="response").error(
                "LLM response timed out — continuing with next step"
            )
            return None
        except Exception as e:
            logger.bind(message_type="response").error(
                f"Error generating response: {repr(e)}"
            )
            raise

    def _extract_message(self, response: ModelResponse) -> LitellmOutputMessage | None:
        """Pull a usable message off the response, or None if choices are empty."""
        choices = response.choices
        if not choices or not isinstance(choices[0], Choices):
            logger.bind(message_type="step").warning(
                "LLM returned no valid choices; prompting to continue"
            )
            return None
        return LitellmOutputMessage.model_validate(choices[0].message)

    def _push_continue(self, content: str) -> None:
        """Append a user-role nudge message to keep the loop moving."""
        self.messages.append(
            LitellmOutputMessage(role="user", content=content)
        )

    async def _dispatch_tool_calls(self, tool_calls: list[Any]) -> None:
        """Route each tool call by name. Multiple calls per step are allowed."""
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments or ""

            tool_logger = logger.bind(
                ref=tool_call.id,
                name=name,
            )
            tool_logger.bind(message_type="tool_call", payload=args).info(
                f"Calling tool {name}"
            )

            if name == "final_answer":
                answer, status = parse_final_answer(args)
                logger.bind(message_type="final_answer").info(answer)
                self._finalized = True
                self._final_answer = answer
                self._final_status = status
                self.messages.append(
                    LitellmOutputMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name="final_answer",
                        content=answer,
                    )
                )
                # Don't process any further tool calls in this step — we're done.
                return

            if name == "execute_code":
                result_str = await self._handle_execute_code(args, tool_logger)
            elif name == "execute_bash":
                result_str = await self._handle_execute_bash(args, tool_logger)
            else:
                # The LLM hallucinated a tool name. Surface the error in-band so
                # it can recover on the next turn.
                result_str = (
                    f"Unknown tool '{name}'. Available tools: "
                    f"execute_code, execute_bash, final_answer."
                )
                tool_logger.bind(message_type="tool_result").warning(result_str)

            self.messages.append(
                LitellmOutputMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=name,
                    content=result_str,
                )
            )

    async def _handle_execute_code(self, args: str, tool_logger: Any) -> str:
        """Parse args, run code in the sandbox, return formatted result."""
        try:
            parsed = json.loads(args) if args else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON arguments for execute_code: {e}"

        code = parsed.get("code")
        if not isinstance(code, str):
            return "execute_code requires a string 'code' argument."

        result = await execute_code(code)
        tool_logger.bind(message_type="tool_result").info(
            f"execute_code: success={result.success} stdout_len={len(result.stdout)}"
        )
        return format_exec_result(result)

    async def _handle_execute_bash(self, args: str, tool_logger: Any) -> str:
        """Parse args, run command via bash, return formatted result."""
        try:
            parsed = json.loads(args) if args else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON arguments for execute_bash: {e}"

        command = parsed.get("command")
        if not isinstance(command, str):
            return "execute_bash requires a string 'command' argument."

        result = await execute_bash(command)
        tool_logger.bind(message_type="tool_result").info(
            f"execute_bash: exit_code={result.exit_code} stdout_len={len(result.stdout)}"
        )
        return format_bash_result(result)

    async def run(self) -> AgentTrajectoryOutput:
        """Run the agent loop until final_answer, max_steps, or timeout."""
        self._start_time = time.time()
        status = AgentStatus.RUNNING

        try:
            async with asyncio.timeout(self.timeout):
                with logger.contextualize(model=self.model):
                    logger.bind(message_type="configure").info(
                        f"Starting CodingMCPAgent loop with model {self.model}"
                    )

                    for i in range(self.max_steps):
                        if self._finalized:
                            logger.info(f"Loop finalized after {i} steps")
                            break
                        logger.bind(message_type="step").info(f"Starting step {i + 1}")
                        await self.step()

                    if self._finalized:
                        # final_answer's status takes precedence over loop state:
                        # "blocked"/"failed" are intentional outcomes, not errors.
                        status = (
                            AgentStatus.COMPLETED
                            if self._final_status == "completed"
                            else AgentStatus.FAILED
                        )
                    else:
                        logger.error(
                            f"Loop hit max_steps={self.max_steps} without final_answer"
                        )
                        status = AgentStatus.FAILED

        except TimeoutError:
            logger.error(f"Agent loop timed out after {self.timeout}s")
            status = AgentStatus.ERROR
        except asyncio.CancelledError:
            logger.error("Agent loop cancelled")
            status = AgentStatus.CANCELLED
        except Exception as e:
            logger.error(f"Error in agent loop: {repr(e)}")
            status = AgentStatus.ERROR if is_system_error(e) else AgentStatus.FAILED

        return AgentTrajectoryOutput(
            messages=list(self.messages),
            status=status,
            time_elapsed=time.time() - (self._start_time or time.time()),
            usage=self._usage_tracker.to_dict(),
        )
