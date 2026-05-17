"""
Runtime health probe for the generated MCP bindings.

This is NOT a unit test. It runs once at agent startup (called from
`main.py`) against the *live* MCP gateway, after `bindings.build_server_modules`
has built each `servers.<name>` module. The goal: verify the binding
pipeline is wired up end-to-end — GET /apps → module generation → real
`call_tool` round-trip — before the agent loop starts using bindings.

Real unit tests for the binding logic live in
`agents/tests/coding_mcp_agent/test_bindings.py`.

## How probe selection works

For each server module, we pick *one* tool to call by inspecting the
tool's `inputSchema`, rather than maintaining a hand-written per-server
probe registry. Two strategies in order:

1. **Zero-required-args tool.** Any tool whose `inputSchema.required`
   list is empty can be called with `{}`. This covers e.g. filesystem's
   `list_files` (`path` defaults to `"/"`).

2. **`*_schema` introspection tool.** fastmcp meta-tool servers (sheets,
   slides, pdf, mail) consistently expose a `<basename>_schema` tool
   that takes a single `request` object with a `model` field. These are
   side-effect-free introspection endpoints, so we call them with
   `request={"model": "input"}`.

If neither strategy applies, the server is skipped with a clear log
message — much better than the old hardcoded-PROBES design where
forgetting to register a new server silently lost coverage.

Adding a new MCP server requires NO changes to this file.
"""

from __future__ import annotations

import types
from typing import Any

from loguru import logger

from runner.agents.coding_mcp_agent.bindings import get_bound_tools
from runner.agents.coding_mcp_agent.utils import parse_input_schema

# A no-args tool (Strategy 1) is called with this payload.
_NO_ARGS: dict[str, Any] = {}

# Meta-tool schema introspection (Strategy 2) takes a request with a model name.
_SCHEMA_PROBE_ARGS: dict[str, Any] = {"request": {"model": "input"}}



def _required_args(tool: Any) -> list[str]:
    """Get the list of required argument names from a tool's inputSchema."""
    _, required = parse_input_schema(tool)
    return list(required)


def _pick_probe(
    mod: types.ModuleType,
) -> tuple[str, dict[str, Any]] | None:
    """Choose (fn_name, kwargs) to invoke on `mod`, or None if nothing safe to call.

    See module docstring for the two strategies.
    """
    bound = get_bound_tools(mod)
    if not bound:
        return None

    # Strategy 1: any tool with no required args.
    for fn_name, tool in bound.items():
        if not _required_args(tool):
            return fn_name, _NO_ARGS

    # Strategy 2: a `<name>_schema` introspection tool.
    for fn_name in bound:
        if fn_name.endswith("_schema") or fn_name == "schema":
            return fn_name, _SCHEMA_PROBE_ARGS

    return None


async def run_runtime_probes(
    modules: dict[str, types.ModuleType],
) -> dict[str, str]:
    """Probe each generated server module with one auto-picked tool call.

    Returns ``{server_name: "ok" | "failed: ..." | "skipped: ..."}``.
    Failures are caught per-server so one bad server doesn't mask passing
    ones. Raises nothing.
    """
    report: dict[str, str] = {}

    for server_name, mod in modules.items():
        choice = _pick_probe(mod)
        if choice is None:
            report[server_name] = "skipped: no auto-probable tool found"
            logger.warning(
                f"runtime_probes: servers.{server_name} — "
                "no tool with empty required-args and no `*_schema` tool found; "
                "binding cannot be auto-verified"
            )
            continue

        fn_name, kwargs = choice
        try:
            fn = getattr(mod, fn_name)
            result = await fn(**kwargs)
            preview = (result or "")[:120].replace("\n", " ")
            report[server_name] = "ok"
            logger.info(
                f"runtime_probes: servers.{server_name}.{fn_name} OK — {preview!r}"
            )
        except Exception as e:
            report[server_name] = f"failed: {type(e).__name__}: {e}"
            logger.error(
                f"runtime_probes: servers.{server_name}.{fn_name} FAILED — {e}"
            )

    return report
