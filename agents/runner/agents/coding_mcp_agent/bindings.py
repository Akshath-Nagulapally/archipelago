"""
Dynamic Python bindings for MCP servers exposed by the gateway.

This module is the foundation of the CodingMCPAgent: it discovers every MCP
server configured in the environment's gateway and registers one Python
module per server so agent code can call tools by writing real imports:

    from servers import filesystem_server, sheets_server
    text = await filesystem_server.read_text_file(path="/foo.txt")

Server names are sourced from the gateway's `GET /apps` endpoint (the
authoritative config), not inferred from tool-name prefixes — which silently
breaks for multi-word names like `filesystem_server`.

Public API:
    make_client(gateway_url) -> Client
        Build a fastmcp Client targeting the environment's gateway. Caller
        owns the lifecycle (`async with`, or explicit __aenter__/__aexit__).

    build_server_modules(gateway_url, client) -> dict[str, ModuleType]
        Discover servers + tools and register `servers.<name>` modules. The
        caller passes in a *single* already-opened client; the returned
        binding functions close over that client and reuse it for every
        tool call. This avoids per-call MCP handshake overhead — see
        README "How It Works" / PR review notes for rationale.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from loguru import logger

# Name of the parent namespace package under which per-server modules are
# registered. `from servers import <name>` resolves through `sys.modules`.
SERVERS_NAMESPACE = "servers"

# Gateway endpoint that returns the active MCP server names. See
# environment/runner/gateway/router.py.
APPS_ENDPOINT = "/apps"

# Caches the fastmcp Client config dict per gateway URL. The config itself
# is cheap to build, but caching keeps identity stable across re-entries.
_GATEWAY_CLIENT_CONFIG_CACHE: dict[str, Any] = {}


def _gateway_cfg(url: str) -> dict[str, Any]:
    """Build (or return cached) fastmcp client config for a gateway URL."""
    if url not in _GATEWAY_CLIENT_CONFIG_CACHE:
        _GATEWAY_CLIENT_CONFIG_CACHE[url] = {
            "mcpServers": {"gateway": {"transport": "streamable-http", "url": url}}
        }
    return _GATEWAY_CLIENT_CONFIG_CACHE[url]


def make_client(gateway_url: str) -> Client[MCPConfigTransport]:
    """Construct a fastmcp Client configured to talk to the environment's gateway.

    The returned Client is *not* yet connected — the caller must enter it
    (`async with` or `await client.__aenter__()`) before using it, and is
    responsible for closing it exactly once at end-of-life.

    Sharing a single open client across all tool calls is the whole point:
    the MCP handshake (TCP + protocol init) happens once instead of once
    per tool invocation.
    """
    return Client(_gateway_cfg(gateway_url))


async def _get_server_names(gateway_url: str) -> list[str]:
    """Fetch the configured server names from the environment's /apps endpoint.

    This is the authoritative source — the environment stores exactly the keys
    used in the `mcpServers` config, which are the same prefixes FastMCP uses
    when proxying multiple servers.
    """
    # gateway_url looks like "http://host:port/mcp/" — strip to base.
    # NOTE: use removesuffix (literal-suffix match), NOT rstrip (which strips
    # any trailing chars in the set "/mcp/" and would corrupt URLs that end
    # in characters like 'c', 'm', or 'p' before the /mcp/ segment).
    base_url = gateway_url.removesuffix("/mcp/").removesuffix("/mcp")
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{base_url}{APPS_ENDPOINT}", timeout=10)
        resp.raise_for_status()
        return resp.json()["servers"]


def _ensure_servers_namespace() -> types.ModuleType:
    """Idempotently register the parent `servers` namespace package."""
    pkg = sys.modules.get(SERVERS_NAMESPACE)
    if pkg is None:
        pkg = types.ModuleType(SERVERS_NAMESPACE)
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[SERVERS_NAMESPACE] = pkg
    return pkg


def _select_server_tools(
    server_name: str, all_tools: list[Any], server_count: int
) -> tuple[list[Any], bool] | None:
    """Pick the tools belonging to `server_name` and decide whether to strip a prefix.

    Returns (tools, strip_prefix), or None if the server contributed nothing
    on a multi-server gateway (caller should skip and log).

    We inspect the actual tool list rather than guessing from `server_count`:
    fastmcp's prefix-stripping for single-server gateways is an undocumented
    behavior, and a length-based heuristic would silently misbehave if that
    policy ever changed.
    """
    prefix = f"{server_name}_"
    prefixed = [t for t in all_tools if t.name.startswith(prefix)]
    if prefixed:
        return prefixed, True
    if server_count == 1:
        # Single-server gateway, no prefixing: every tool belongs here.
        return all_tools, False
    return None


def _make_tool_caller(
    tool_name: str, client: Client[MCPConfigTransport]
) -> Callable[..., Awaitable[str]]:
    """Build the async wrapper that calls one specific MCP tool through `client`.

    Why a factory (instead of an inline default-arg closure)?

    The old pattern was::

        async def _call(_tool_name=tool.name, _client=client, **kwargs):
            ...

    That works for the late-binding loop-variable problem, but it has a
    subtle leak: `_tool_name` and `_client` are real parameter slots, so a
    caller passing `_client=...` or `_tool_name=...` as a kwarg silently
    overrides our closure-captured values. If an MCP tool ever defines a
    parameter literally named `_tool_name` or `_client`, the binding would
    crash (e.g., calling `.call_tool` on a user-supplied string).

    With a factory, `tool_name` and `client` live in this enclosing function's
    scope — they are NOT parameters of `_call`. Anything the caller passes
    flows cleanly into `**kwargs` and gets forwarded to the MCP tool.
    """

    async def _call(**kwargs: Any) -> str:
        # CallToolResult.content is a list of content items (TextContent,
        # ImageContent, etc). We expose the .text of the first item as the
        # binding's return value. fastmcp >= 2.12.4 is required for this shape.
        result = await client.call_tool(tool_name, kwargs)
        if not result.content:
            return ""
        first = result.content[0]
        return getattr(first, "text", str(first))

    return _call


def _register_tool_on_module(
    mod: types.ModuleType,
    tool: Any,
    fn_name: str,
    client: Client[MCPConfigTransport],
) -> None:
    """Attach a single MCP tool to `mod` as an async function `fn_name`.

    The wrapper closes over the shared `client` so every call reuses one
    open MCP session — see README "How It Works" for rationale.

    The original `tool` is attached as `fn._mcp_tool` so consumers like
    `runtime_probes` can introspect each binding's input schema without
    needing to re-fetch the tool list from the gateway.
    """
    fn = _make_tool_caller(tool.name, client)
    fn.__name__ = fn_name
    fn.__doc__ = tool.description
    # Attach the original Tool so consumers (e.g. runtime_probes) can
    # introspect inputSchema without re-fetching from the gateway.
    # FunctionType doesn't declare arbitrary attributes, hence the ignore.
    fn._mcp_tool = tool  # pyright: ignore[reportFunctionMemberAccess]
    setattr(mod, fn_name, fn)


def _build_one_server_module(
    server_name: str,
    all_tools: list[Any],
    server_count: int,
    client: Client[MCPConfigTransport],
    servers_pkg: types.ModuleType,
) -> types.ModuleType | None:
    """Build and register the `servers.<server_name>` module.

    Returns the new module on success, or None if the server has no tools
    on this multi-server gateway (caller should skip).
    """
    selection = _select_server_tools(server_name, all_tools, server_count)
    if selection is None:
        logger.warning(
            f"No tools found for server '{server_name}' "
            f"(searched prefix '{server_name}_' across {len(all_tools)} tools)"
        )
        return None

    server_tools, strip_prefix = selection
    prefix = f"{server_name}_"

    mod = types.ModuleType(f"{SERVERS_NAMESPACE}.{server_name}")
    for tool in server_tools:
        fn_name = tool.name[len(prefix):] if strip_prefix else tool.name
        _register_tool_on_module(mod, tool, fn_name, client)

    sys.modules[f"{SERVERS_NAMESPACE}.{server_name}"] = mod
    setattr(servers_pkg, server_name, mod)

    logger.info(
        f"Built module {SERVERS_NAMESPACE}.{server_name} with {len(server_tools)} tools"
    )
    return mod


async def build_server_modules(
    gateway_url: str, client: Client[MCPConfigTransport]
) -> dict[str, types.ModuleType]:
    """Discover all tools from the gateway and register one module per server.

    After this runs, agent code can do::

        from servers import filesystem_server
        result = await filesystem_server.read_text_file(path="/foo.txt")

    Args:
        gateway_url: Base URL of the environment gateway. Used for the HTTP
            GET /apps lookup (server-name discovery).
        client: An already-opened fastmcp Client targeting the same gateway.
            Used for `list_tools()` here and captured by every binding wrapper
            for subsequent `call_tool()` invocations. Lifecycle is the
            caller's responsibility — see `make_client`.

    Returns a {server_name: module} map. Logs a warning and returns an empty
    dict if the gateway exposes no tools.
    """
    server_names = await _get_server_names(gateway_url)
    tools = await client.list_tools()

    if not tools:
        logger.warning("Gateway returned no tools — no modules built")
        return {}

    servers_pkg = _ensure_servers_namespace()
    server_count = len(server_names)

    modules: dict[str, types.ModuleType] = {}
    for server_name in server_names:
        mod = _build_one_server_module(
            server_name, tools, server_count, client, servers_pkg
        )
        if mod is not None:
            modules[server_name] = mod

    return modules
