"""
Startup probes for the CodingMCPAgent.

This module runs once at agent startup (called from `main.py`) after
bindings and tool-discovery docs have been built. Three things are
verified before the LLM loop is allowed to start:

1. **Remote MCP gateway** (`probe_remote_mcp`) — each generated server
   module can actually round-trip a real tool call through the gateway.
2. **Sandbox imports** (`probe_sandbox_imports`) — `execute_code` can
   `from servers import <every-server>`.
3. **Bash docs** (`probe_bash_reads_docs`) — `execute_bash` can `cat`
   the root index of the tool-discovery tree.

Every probe has the same contract: returns `None` on success (with
informative logs), raises `RuntimeError` on failure. The orchestrator
`run_startup_probes` runs all three in order and bails on the first
failure.

This strict policy is intentional for eval workloads: a silently broken
MCP server, sandbox plumbing, or local tool would contaminate
trajectories with failures that look like model mistakes. We'd rather
refuse to start than ship bad data.

Real unit tests live in `agents/tests/coding_mcp_agent/test_probes.py`.
"""

from __future__ import annotations

import types
from typing import Any

from loguru import logger

from runner.agents.coding_mcp_agent.bindings import get_bound_tools
from runner.agents.coding_mcp_agent.tools.bash import execute_bash
from runner.agents.coding_mcp_agent.tools.execute_code import execute_code
from runner.agents.coding_mcp_agent.utils import parse_input_schema

# ---------------------------------------------------------------------------
# Remote MCP probe (was runtime_probes.py).
#
# For each server module, we pick *one* tool to call by inspecting the
# tool's `inputSchema`, rather than maintaining a hand-written per-server
# probe registry. Two strategies, in order:
#
# 1. Zero-required-args tool — call with `{}`.
# 2. `<name>_schema` introspection tool — call with `request={"model": "input"}`.
#
# If neither applies, the server is skipped with a clear log message.
# Adding a new MCP server requires NO changes to this file.
# ---------------------------------------------------------------------------

_NO_ARGS: dict[str, Any] = {}
_SCHEMA_PROBE_ARGS: dict[str, Any] = {"request": {"model": "input"}}


def _required_args(tool: Any) -> list[str]:
    """Get the list of required argument names from a tool's inputSchema."""
    _, required = parse_input_schema(tool)
    return list(required)


def _pick_probe(
    mod: types.ModuleType,
) -> tuple[str, dict[str, Any]] | None:
    """Choose (fn_name, kwargs) to invoke on `mod`, or None if nothing safe to call."""
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


async def probe_remote_mcp(modules: dict[str, types.ModuleType]) -> None:
    """Probe each generated server module with one auto-picked tool call.

    Builds a per-server report internally (so a failure on server A doesn't
    hide whether server B is reachable), logs it, then:

    - Returns silently if every entry is ``"ok"`` or ``"skipped"``.
    - Raises ``RuntimeError`` if ANY entry is ``"failed: ..."``. The exception
      message lists every failed server plus the full report.

    Skipped servers (no auto-probable tool found) pass through with a
    warning — those represent "couldn't auto-verify", not known breakage.
    """
    report: dict[str, str] = {}

    for server_name, mod in modules.items():
        choice = _pick_probe(mod)
        if choice is None:
            report[server_name] = "skipped: no auto-probable tool found"
            logger.warning(
                f"probe_remote_mcp: servers.{server_name} — "
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
                f"probe_remote_mcp: servers.{server_name}.{fn_name} OK — {preview!r}"
            )
        except Exception as e:
            report[server_name] = f"failed: {type(e).__name__}: {e}"
            logger.error(
                f"probe_remote_mcp: servers.{server_name}.{fn_name} FAILED — {e}"
            )

    failed = {
        name: status for name, status in report.items() if status.startswith("failed:")
    }
    if failed:
        raise RuntimeError(
            f"probe_remote_mcp failed for {len(failed)} server(s): {failed}. "
            f"Full report: {report}"
        )


# ---------------------------------------------------------------------------
# Sandbox import probe — exercises bindings → sys.modules → sandbox.exec.
# ---------------------------------------------------------------------------


async def probe_sandbox_imports(modules: dict[str, types.ModuleType]) -> None:
    """Run a single `execute_code` that imports every generated server module.

    Raises ``RuntimeError`` if the sandbox can't import the modules. This
    exercises the full bindings → sys.modules → sandbox.exec chain.

    No-op if `modules` is empty (an empty `from servers import` is a syntax
    error and there's nothing to verify anyway).
    """
    if not modules:
        logger.warning("probe_sandbox_imports: no modules to verify — skipping")
        return

    names = ", ".join(modules.keys())
    code = f"from servers import {names}"
    result = await execute_code(code)

    if not result.success:
        raise RuntimeError(
            f"probe_sandbox_imports failed: could not `from servers import "
            f"{names}` inside the sandbox. Error:\n{result.error}"
        )

    logger.info(
        f"probe_sandbox_imports: OK — sandbox imported {len(modules)} module(s): "
        f"{list(modules.keys())}"
    )


# ---------------------------------------------------------------------------
# Bash docs probe — verifies the agent's `execute_bash` can reach the
# tool-discovery tree (the LLM's first navigation target).
# ---------------------------------------------------------------------------


async def probe_bash_reads_docs(tool_docs_path: str) -> None:
    """Run `cat <tool_docs_path>/servers/_index.txt` via bash.

    Raises ``RuntimeError`` if the bash tool can't read the root index of
    the tool-discovery tree, or if the file is readable but empty.
    """
    target = f"{tool_docs_path}/servers/_index.txt"
    result = await execute_bash(f"cat {target}")

    if not result.success:
        raise RuntimeError(
            f"probe_bash_reads_docs failed: could not read {target}. "
            f"exit_code={result.exit_code}, stderr={result.stderr!r}"
        )
    if not result.stdout.strip():
        raise RuntimeError(
            f"probe_bash_reads_docs failed: {target} was readable but empty. "
            f"Tool discovery docs may not have been generated correctly."
        )

    preview = result.stdout.splitlines()[0] if result.stdout else ""
    logger.info(f"probe_bash_reads_docs: OK — read {target} (first line: {preview!r})")


# ---------------------------------------------------------------------------
# Orchestrator — single entry point called from main.initialize().
# ---------------------------------------------------------------------------


async def run_startup_probes(
    modules: dict[str, types.ModuleType],
    tool_docs_path: str | None,
) -> None:
    """Run every startup probe in order. Raises on first failure.

    The order is intentional: cheapest local checks (sandbox import) before
    the network-bound MCP probe would be even cheaper, but we keep MCP
    first because failure there usually means the entire bindings tree is
    a lie — running sandbox/bash probes after that just adds noise.

    Args:
        modules: The server modules built by `bindings.build_server_modules`.
        tool_docs_path: Path written by `tool_discovery_docs.build_tool_docs_dir`.
            Must not be None — that indicates the docs step never ran, which
            is a programming error, not a probe failure.
    """
    if tool_docs_path is None:
        raise RuntimeError(
            "run_startup_probes: tool_docs_path is None — "
            "tool discovery docs were not generated. Refusing to start."
        )

    await probe_remote_mcp(modules)
    await probe_sandbox_imports(modules)
    await probe_bash_reads_docs(tool_docs_path)

    logger.info("run_startup_probes: all probes passed")
