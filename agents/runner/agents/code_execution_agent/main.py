"""Code Execution Agent built on top of the ReAct toolbelt implementation."""

from __future__ import annotations

from typing import override

from loguru import logger

from runner.agents.models import (
    AgentRunInput,
    AgentTrajectoryOutput,
    LitellmAnyMessage,
)
from runner.agents.react_toolbelt_agent.main import ReActAgent

from .prompt import build_runtime_dir, inject_code_execution_system_prompt


class CodeExecutionAgent(ReActAgent):
    """ReAct-style agent that nudges the model toward task-side Python execution."""

    def __init__(self, run_input: AgentRunInput):
        if run_input.mcp_gateway_url is None:
            raise ValueError("MCP gateway URL is required for code execution agent")

        super().__init__(run_input)
        self.runtime_dir: str = build_runtime_dir(run_input.trajectory_id)
        self.messages: list[LitellmAnyMessage] = inject_code_execution_system_prompt(
            self.messages,
            runtime_dir=self.runtime_dir,
            mcp_gateway_url=run_input.mcp_gateway_url,
            mcp_gateway_auth_token=run_input.mcp_gateway_auth_token,
        )

    @override
    async def _initialize_tools(self, client: object) -> None:
        """Load tools, then pre-add code_exec to the active toolbelt when available."""
        await super()._initialize_tools(client)

        if "code_exec" in self.all_tools:
            self.toolbelt.add("code_exec")
            logger.bind(message_type="configure").info(
                "Pre-added 'code_exec' to the toolbelt for task-side Python workflows"
            )
        else:
            logger.bind(message_type="configure").warning(
                "CodeExecutionAgent started without a 'code_exec' MCP tool; "
                + "programmatic code-mode workflows may be unavailable"
            )


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    """Entry point for the Code Execution Agent."""
    return await CodeExecutionAgent(run_input).run()
