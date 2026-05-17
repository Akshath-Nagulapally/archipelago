"""
CodingMCP Agent

Builds dynamic Python modules from every MCP server available in the gateway,
then smoke-tests each one before running the agent loop.

Agent code (and the LLM-generated code it executes) can then do:
    from servers import filesystem_server, sheets_server
    await filesystem_server.read_text_file(path="/foo.txt")
"""

import sys
import time
import types
from typing import Any

import httpx
from fastmcp import Client
from loguru import logger

from runner.agents.models import (
    AgentRunInput,
    AgentStatus,
    AgentTrajectoryOutput,
)


_GATEWAY_CLIENT_CONFIG_CACHE: dict[str, Any] = {}


def _gateway_cfg(url: str) -> dict[str, Any]:
    if url not in _GATEWAY_CLIENT_CONFIG_CACHE:
        _GATEWAY_CLIENT_CONFIG_CACHE[url] = {
            "mcpServers": {"gateway": {"transport": "streamable-http", "url": url}}
        }
    return _GATEWAY_CLIENT_CONFIG_CACHE[url]


async def _get_server_names(gateway_url: str) -> list[str]:
    """Fetch the configured server names directly from the environment's /apps endpoint.

    This is authoritative — no inference from tool name prefixes needed.
    The environment stores exactly the keys used in mcpServers config,
    which are the same prefixes FastMCP uses when proxying multiple servers.
    """
    # gateway_url is like "http://host:port/mcp/" — strip to base
    base_url = gateway_url.rstrip("/mcp/").rstrip("/mcp")
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{base_url}/apps", timeout=10)
        resp.raise_for_status()
        return resp.json()["servers"]


async def build_server_modules(gateway_url: str) -> dict[str, types.ModuleType]:
    """Discover all tools from the gateway and register one module per server.

    Server names come from GET /apps (the authoritative config), not from
    parsing tool name prefixes. This correctly handles multi-word server names
    like 'filesystem_server'.

    After this runs, agent code can do:
        from servers import filesystem_server
        result = await filesystem_server.read_text_file(path="/foo.txt")
    """
    server_names = await _get_server_names(gateway_url)

    cfg = _gateway_cfg(gateway_url)
    async with Client(cfg) as client:
        tools = await client.list_tools()

    if not tools:
        logger.warning("Gateway returned no tools — no modules built")
        return {}

    # Register a 'servers' namespace package so `from servers import X` works
    servers_pkg = sys.modules.get("servers")
    if servers_pkg is None:
        servers_pkg = types.ModuleType("servers")
        servers_pkg.__path__ = []
        sys.modules["servers"] = servers_pkg

    modules: dict[str, types.ModuleType] = {}
    single_server = len(server_names) == 1

    for server_name in server_names:
        mod = types.ModuleType(f"servers.{server_name}")

        if single_server:
            # FastMCP doesn't prefix tools for single-server gateways
            server_tools = tools
        else:
            server_tools = [t for t in tools if t.name.startswith(f"{server_name}_")]

        for tool in server_tools:
            fn_name = (
                tool.name[len(server_name) + 1:]
                if tool.name.startswith(f"{server_name}_")
                else tool.name
            )

            async def _call(_tool_name=tool.name, _url=gateway_url, **kwargs):
                async with Client(_gateway_cfg(_url)) as c:
                    result = await c.call_tool(_tool_name, kwargs)
                    # fastmcp returns a CallToolResult with .content (list of
                    # TextContent / ImageContent / etc). Older versions returned
                    # the content list directly — handle both.
                    content = getattr(result, "content", result)
                    if not content:
                        return ""
                    first = content[0]
                    return getattr(first, "text", str(first))

            _call.__name__ = fn_name
            _call.__doc__ = tool.description
            setattr(mod, fn_name, _call)

        sys.modules[f"servers.{server_name}"] = mod
        setattr(servers_pkg, server_name, mod)
        modules[server_name] = mod

        logger.info(f"Built module servers.{server_name} with {len(server_tools)} tools")

    return modules


async def smoke_test_modules(modules: dict[str, types.ModuleType]) -> dict[str, str]:
    """Run real tool calls against each built server module.

    Imports `binding_test` lazily so that an import error in the test file
    doesn't crash the agent at module load time. Returns the per-server
    pass/fail report so callers can decide whether to proceed.
    """
    from runner.agents.coding_mcp_agent import binding_test

    return await binding_test.run_binding_tests(modules)


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

    async def initialize(self) -> None:
        """Build all server modules and smoke-test them."""
        logger.info("Building MCP server modules from gateway...")
        self.modules = await build_server_modules(self.gateway_url)

        if not self.modules:
            raise RuntimeError("No MCP server modules could be built — check gateway config")

        logger.info(f"Modules ready: {list(self.modules.keys())}")
        report = await smoke_test_modules(self.modules)
        logger.info(f"Binding test report: {report}")

    async def run(self) -> AgentTrajectoryOutput:
        self.start_time = time.time()

        try:
            await self.initialize()
        except Exception as e:
            logger.error(f"CodingMCPAgent initialization failed: {e}")
            return AgentTrajectoryOutput(
                messages=list(self.initial_messages),
                status=AgentStatus.ERROR,
                time_elapsed=time.time() - (self.start_time or time.time()),
            )

        # No agent loop on this branch — the goal here is just to verify that
        # MCP bindings are generated from GET /apps and that each server is
        # reachable through its `servers.<name>` module. Return ERROR so the
        # trajectory output unambiguously signals "no agent ran".
        logger.info("CodingMCPAgent: bindings + tests done; no agent loop on this branch")

        return AgentTrajectoryOutput(
            messages=list(self.initial_messages),
            status=AgentStatus.ERROR,
            time_elapsed=time.time() - (self.start_time or time.time()),
        )


async def run(run_input: AgentRunInput) -> AgentTrajectoryOutput:
    agent = CodingMCPAgent(run_input)
    return await agent.run()
