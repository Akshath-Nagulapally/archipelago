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
from typing import Any

import httpx
from fastmcp import Client
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


def make_client(gateway_url: str) -> Client:
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


async def build_server_modules(
    gateway_url: str, client: Client
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

    modules: dict[str, types.ModuleType] = {}

    for server_name in server_names:
        mod = types.ModuleType(f"{SERVERS_NAMESPACE}.{server_name}")
        prefix = f"{server_name}_"

        # Decide whether the gateway prefixed this server's tools by looking
        # at the actual tool list rather than guessing from len(server_names).
        # FastMCP's prefix-stripping for single-server gateways is an
        # implementation detail, not a stable contract — and a length-based
        # heuristic would silently misbehave if that policy ever changed.
        prefixed_tools = [t for t in tools if t.name.startswith(prefix)]

        if prefixed_tools:
            server_tools = prefixed_tools
            strip_prefix = True
        elif len(server_names) == 1:
            # Single-server gateway, no prefixing: every tool belongs here.
            server_tools = tools
            strip_prefix = False
        else:
            # Multi-server gateway but this server contributed no prefixed
            # tools. Don't silently build an empty module — surface it.
            logger.warning(
                f"No tools found for server '{server_name}' "
                f"(searched prefix '{prefix}' across {len(tools)} tools)"
            )
            continue

        for tool in server_tools:
            fn_name = tool.name[len(prefix):] if strip_prefix else tool.name

            # NOTE: default-arg closure on _tool_name/_client is the standard
            # Python idiom for capturing loop variables. The shared `client`
            # means every tool call reuses the same already-open MCP session
            # (one handshake amortized over the agent's lifetime) instead of
            # paying TCP + MCP-init overhead per call.
            async def _call(_tool_name=tool.name, _client=client, **kwargs):
                result = await _client.call_tool(_tool_name, kwargs)
                # fastmcp returns a CallToolResult with .content (list of
                # TextContent / ImageContent / etc). Older versions returned
                # the content list directly — handle both shapes.
                content = getattr(result, "content", result)
                if not content:
                    return ""
                first = content[0]
                return getattr(first, "text", str(first))

            _call.__name__ = fn_name
            _call.__doc__ = tool.description
            setattr(mod, fn_name, _call)

        sys.modules[f"{SERVERS_NAMESPACE}.{server_name}"] = mod
        setattr(servers_pkg, server_name, mod)
        modules[server_name] = mod

        logger.info(
            f"Built module {SERVERS_NAMESPACE}.{server_name} with {len(server_tools)} tools"
        )

    return modules
