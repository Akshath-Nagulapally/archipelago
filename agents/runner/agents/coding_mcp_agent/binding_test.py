"""
Binding smoke test for CodingMCPAgent.

This file exercises the dynamically-generated `servers.*` modules the way an
LLM-written agent step would: by importing each server module and calling a
real, low-risk tool on it. If the binding is wired up correctly end-to-end
(GET /apps → module generation → MCP gateway call), each call returns a
string result.

For each known server, we define a probe coroutine that calls one read-only
tool. We only run probes for servers that were actually built by
`build_server_modules` — others are skipped silently.

Add a new server here when you add a new MCP server to the gateway. The probe
should be a cheap, side-effect-free call.
"""

from __future__ import annotations

import importlib
import types
from typing import Awaitable, Callable

from loguru import logger


# Probe registry: server_name -> async callable that performs one real tool call
# on that server's binding module. Each probe imports its module fresh (in case
# the test is invoked before the module is on sys.path in some import order),
# then calls one tool and returns the result.
#
# Tool choice rationale:
# - filesystem_server exposes individual tools (list_files, read_text_file...)
#   so we call list_files("/") — cheapest read.
# - sheets/slides/pdf/mail use a meta-tool pattern: one entrypoint per server
#   (`sheets`, `pdf`, etc.) plus a `<name>_schema` introspection tool. Schema
#   tools are side-effect-free, so they are the safest probe — they verify the
#   binding can serialize/deserialize an args payload end-to-end without
#   depending on filesystem state.
#
# NOTE on the `request=` wrapping below: those servers define their handlers
# as `async def sheets_schema(request: SchemaInput)` with a single Pydantic
# model parameter. MCP exposes that parameter at the top level of the tool
# input schema, so callers must pass `request={...}`, not the inner fields
# directly. This is a quirk of the meta-tool servers — filesystem_server uses
# flat per-arg signatures, which is why its probe looks different.

async def _probe_filesystem_server() -> str:
    mod = importlib.import_module("servers.filesystem_server")
    return await mod.list_files(path="/")


async def _probe_sheets_server() -> str:
    mod = importlib.import_module("servers.sheets_server")
    return await mod.sheets_schema(request={"model": "input"})


async def _probe_slides_server() -> str:
    mod = importlib.import_module("servers.slides_server")
    return await mod.slides_schema(request={"model": "input"})


async def _probe_pdf_server() -> str:
    mod = importlib.import_module("servers.pdf_server")
    return await mod.pdf_schema(request={"model": "input"})


async def _probe_mail_server() -> str:
    mod = importlib.import_module("servers.mail_server")
    return await mod.mail_schema(request={"model": "input"})


PROBES: dict[str, Callable[[], Awaitable[str]]] = {
    "filesystem_server": _probe_filesystem_server,
    "sheets_server": _probe_sheets_server,
    "slides_server": _probe_slides_server,
    "pdf_server": _probe_pdf_server,
    "mail_server": _probe_mail_server,
}


async def run_binding_tests(modules: dict[str, types.ModuleType]) -> dict[str, str]:
    """Run a real tool call against each built server module.

    Returns a {server_name: "ok" | "skipped: ..." | "failed: ..."} report.
    Raises nothing — failures are captured per-server so one bad server does
    not mask passing ones.
    """
    report: dict[str, str] = {}

    for server_name in modules:
        probe = PROBES.get(server_name)
        if probe is None:
            report[server_name] = "skipped: no probe registered"
            logger.warning(
                f"binding_test: no probe defined for servers.{server_name} — skipping"
            )
            continue

        try:
            result = await probe()
            preview = (result or "")[:120].replace("\n", " ")
            report[server_name] = "ok"
            logger.info(f"binding_test: servers.{server_name} OK — {preview!r}")
        except Exception as e:
            report[server_name] = f"failed: {type(e).__name__}: {e}"
            logger.error(f"binding_test: servers.{server_name} FAILED — {e}")

    return report
