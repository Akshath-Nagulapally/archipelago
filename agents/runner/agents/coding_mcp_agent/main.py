"""
CodingMCPAgent — entrypoint and lifecycle.

The agent's job in this branch is intentionally narrow: build dynamic Python
bindings for every MCP server the gateway exposes (see `bindings.py`), then
verify each binding is reachable with a real tool call (see `binding_test.py`).
The LLM-driven loop lives in a follow-up PR — until then `run()` returns
`AgentStatus.ERROR` after initialization.

Once the loop lands, agent code (and the LLM-generated code it executes) will
import tools directly::

    from servers import filesystem_server, sheets_server
    text = await filesystem_server.read_text_file(path="/foo.txt")
"""

from __future__ import annotations

import time
import types

from fastmcp import Client
from loguru import logger

from runner.agents.coding_mcp_agent import binding_test
from runner.agents.coding_mcp_agent.bindings import build_server_modules, make_client
from runner.agents.models import (
    AgentRunInput,
    AgentStatus,
    AgentTrajectoryOutput,
)


class CodingMCPAgent:
    def __init__(self, run_input: AgentRunInput):
        if run_input.mcp_gateway_url is None:
            raise ValueError("CodingMCPAgent requires an MCP gateway URL")

        self.gateway_url = run_input.mcp_gateway_url
        self.trajectory_id = run_input.trajectory_id
        self.model = run_input.orchestrator_model
        self.initial_messages = run_input.initial_messages
        self.config = run_input.agent_config_values
        self.start_time: float | None = None
        self.modules: dict[str, types.ModuleType] = {}

        # MCP client lifecycle.
        # _client_cm: the un-entered context manager (always set in initialize)
        # _client:    the entered client, used for every tool call. Set only
        #             after a successful __aenter__, so `close()` can tell
        #             whether there's anything to tear down.
        self._client_cm: Client | None = None
        self._client: Client | None = None

    async def initialize(self) -> None:
        """Open the shared MCP client, build bindings, and probe each server.

        We open the client *once* here and pass it to `build_server_modules`,
        which captures it inside every binding wrapper. Subsequent tool calls
        all reuse this same session — avoiding per-call MCP handshake overhead.
        """
        logger.info("Opening MCP client to gateway...")
        self._client_cm = make_client(self.gateway_url)
        # If __aenter__ raises, self._client stays None and close() is a no-op.
        self._client = await self._client_cm.__aenter__()

        logger.info("Building MCP server modules from gateway...")
        self.modules = await build_server_modules(self.gateway_url, self._client)

        if not self.modules:
            raise RuntimeError(
                "No MCP server modules could be built — check gateway config"
            )

        logger.info(f"Modules ready: {list(self.modules.keys())}")

        report = await binding_test.run_binding_tests(self.modules)
        logger.info(f"Binding test report: {report}")

    async def close(self) -> None:
        """Close the shared MCP client, if it was successfully opened.

        Safe to call even if `initialize()` never ran or failed partway —
        we only call `__aexit__` when `_client` was actually set by a
        successful `__aenter__`.
        """
        if self._client is not None and self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            finally:
                self._client = None
                self._client_cm = None

    async def run(self) -> AgentTrajectoryOutput:
        self.start_time = time.time()

        try:
            try:
                await self.initialize()
            except Exception as e:
                logger.error(f"CodingMCPAgent initialization failed: {e}")
                return AgentTrajectoryOutput(
                    messages=list(self.initial_messages),
                    status=AgentStatus.ERROR,
                    time_elapsed=time.time() - (self.start_time or time.time()),
                )

            # No agent loop on this branch — the goal here is just to verify
            # that MCP bindings are generated from GET /apps and that each
            # server is reachable through its `servers.<name>` module. Return
            # ERROR so the trajectory output unambiguously signals "no agent
            # ran".
            logger.info(
                "CodingMCPAgent: bindings + tests done; no agent loop on this branch"
            )

            return AgentTrajectoryOutput(
                messages=list(self.initial_messages),
                status=AgentStatus.ERROR,
                time_elapsed=time.time() - (self.start_time or time.time()),
            )
        finally:
            # Always close the MCP client, whether init succeeded, failed,
            # or the (future) agent loop raised.
            await self.close()


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    agent = CodingMCPAgent(run_input)
    return await agent.run()
