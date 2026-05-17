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

import os
import sys
import time
import types
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from loguru import logger

from runner.agents.coding_mcp_agent.bindings import (
    build_server_modules,
    make_client,
)
from runner.agents.coding_mcp_agent.probes import run_startup_probes
from runner.agents.coding_mcp_agent.tool_discovery_docs import build_tool_docs_dir
from runner.agents.models import (
    AgentRunInput,
    AgentStatus,
    AgentTrajectoryOutput,
)


class CodingMCPAgent:
    # Class-level annotations so basedpyright knows the attribute types
    # without requiring `@final` on the class. The actual values are bound
    # in __init__.
    gateway_url: str
    trajectory_id: str
    model: str
    initial_messages: list[Any]
    config: dict[str, Any]
    start_time: float | None
    modules: dict[str, types.ModuleType]
    tool_docs_path: str | None
    # MCP client lifecycle.
    # _client_cm: the un-entered context manager (always set in initialize)
    # _client:    the entered client, used for every tool call. Set only
    #             after a successful __aenter__, so `close()` can tell
    #             whether there's anything to tear down.
    _client_cm: Client[MCPConfigTransport] | None
    _client: Client[MCPConfigTransport] | None

    def __init__(self, run_input: AgentRunInput):
        if run_input.mcp_gateway_url is None:
            raise ValueError("CodingMCPAgent requires an MCP gateway URL")

        self.gateway_url = run_input.mcp_gateway_url
        self.trajectory_id = run_input.trajectory_id
        self.model = run_input.orchestrator_model
        self.initial_messages = run_input.initial_messages
        self.config = run_input.agent_config_values
        self.start_time = None
        self.modules = {}
        self.tool_docs_path = None
        self._client_cm = None
        self._client = None

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

        _harness_log(
            f"Python bindings successfully generated for all MCP servers: "
            f"{list(self.modules.keys())}"
        )

        logger.info("Building tool discovery docs directory...")
        self.tool_docs_path = build_tool_docs_dir(self.modules)
        logger.debug(f"Tool docs written to: {self.tool_docs_path}")
        _print_tool_docs_tree(self.tool_docs_path)

        # Probes are strict: any failure here raises and stops the agent
        # before the LLM loop starts. In an eval workload, a silently broken
        # MCP server or local tool surface would contaminate trajectories.
        await run_startup_probes(self.modules, self.tool_docs_path)

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


def _harness_log(msg: str) -> None:
    """Print a message in the same `[HH:MM:SS] ...` style the harness uses."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True, file=sys.stdout)


def _print_tool_docs_tree(root: str | None) -> None:
    """Print the generated tool docs directory as an indented tree."""
    if root is None:
        return
    root_name = os.path.basename(root)
    lines = [f"{root_name}/"]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        level = dirpath.replace(root, "").count(os.sep)
        if level == 0:
            # root itself already printed above
            subindent = "  "
            for fname in sorted(filenames):
                lines.append(f"{subindent}{fname}")
            continue
        indent = "  " * level
        lines.append(f"{indent}{os.path.basename(dirpath)}/")
        subindent = "  " * (level + 1)
        for fname in sorted(filenames):
            lines.append(f"{subindent}{fname}")
    tree = "\n".join(lines)
    _harness_log(f"Tool Discovery Docs Successfully Generated. Structured as follows:\n{tree}")


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    agent = CodingMCPAgent(run_input)
    return await agent.run()
