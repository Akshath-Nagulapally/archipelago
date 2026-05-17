"""Unit tests for runner.agents.coding_mcp_agent.runtime_probes.

Covers the probe-selection logic (`_pick_probe`) and the orchestration loop
(`run_runtime_probes`). All probe calls are exercised against fake modules
with attached `_mcp_tool` metadata — no MCP gateway is required.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

from runner.agents.coding_mcp_agent import runtime_probes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_mcp_tool(required: list[str] | None = None) -> SimpleNamespace:
    """Stand-in for an mcp.types.Tool — only `inputSchema.required` is read."""
    return SimpleNamespace(inputSchema={"required": required or []})


def _attach_fake_binding(
    mod: types.ModuleType,
    fn_name: str,
    required: list[str] | None = None,
    return_value: str = "ok",
) -> AsyncMock:
    """Attach a probe-able binding to `mod` with the same shape as a real one.

    Returns the AsyncMock so the test can assert how it was called.
    """
    mock = AsyncMock(return_value=return_value)
    mock._mcp_tool = _fake_mcp_tool(required=required)  # type: ignore[attr-defined]
    setattr(mod, fn_name, mock)
    return mock


# ---------------------------------------------------------------------------
# _pick_probe — strategy selection
# ---------------------------------------------------------------------------


class TestPickProbe:
    def test_prefers_zero_required_args_tool(self):
        """Strategy 1: any tool with empty `required` is callable with no kwargs."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "list_files", required=[])
        _attach_fake_binding(mod, "delete_thing", required=["path"])  # needs args

        choice = runtime_probes._pick_probe(mod)

        assert choice == ("list_files", {})

    def test_falls_back_to_schema_introspection_tool(self):
        """Strategy 2: if no zero-arg tool, prefer a `*_schema` meta-tool."""
        mod = types.ModuleType("servers.fake")
        # Both tools require args — strategy 1 fails.
        _attach_fake_binding(mod, "sheets", required=["request"])
        _attach_fake_binding(mod, "sheets_schema", required=["request"])

        choice = runtime_probes._pick_probe(mod)

        assert choice == ("sheets_schema", {"request": {"model": "input"}})

    def test_recognizes_bare_schema_function_name(self):
        """Strategy 2 also matches a tool named exactly `schema` (not just `*_schema`)."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "do_thing", required=["arg"])
        _attach_fake_binding(mod, "schema", required=["request"])

        choice = runtime_probes._pick_probe(mod)

        assert choice == ("schema", {"request": {"model": "input"}})

    def test_returns_none_when_no_strategy_matches(self):
        """No zero-arg tool and no `*_schema` tool → can't auto-probe → None."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "send_email", required=["to", "subject"])
        _attach_fake_binding(mod, "delete_thing", required=["id"])

        choice = runtime_probes._pick_probe(mod)

        assert choice is None

    def test_returns_none_when_module_has_no_bindings(self):
        """An empty (or wrongly-tagged) module yields no probable tool."""
        mod = types.ModuleType("servers.fake")
        # No bindings attached.

        choice = runtime_probes._pick_probe(mod)

        assert choice is None

    def test_skips_attrs_without_mcp_tool_marker(self):
        """Only `_mcp_tool`-tagged attrs count as probable bindings."""
        mod = types.ModuleType("servers.fake")
        # Plain attribute, no marker — should be ignored.
        async def unrelated_helper():
            return "noop"
        setattr(mod, "unrelated_helper", unrelated_helper)

        choice = runtime_probes._pick_probe(mod)

        assert choice is None


# ---------------------------------------------------------------------------
# run_runtime_probes — orchestration
# ---------------------------------------------------------------------------


class TestRunRuntimeProbes:
    async def test_reports_ok_for_successful_probe(self):
        """A successful probe call → `"ok"` in the report."""
        mod = types.ModuleType("servers.fake")
        binding = _attach_fake_binding(
            mod, "list_files", required=[], return_value="output"
        )

        report = await runtime_probes.run_runtime_probes({"fake": mod})

        assert report == {"fake": "ok"}
        binding.assert_awaited_once_with()

    async def test_reports_failure_with_exception_class_and_message(self):
        """A probe that raises → `"failed: <ExcType>: <msg>"`, no propagation."""
        mod = types.ModuleType("servers.fake")
        binding = AsyncMock(side_effect=RuntimeError("boom"))
        binding._mcp_tool = _fake_mcp_tool(required=[])
        setattr(mod, "list_files", binding)

        report = await runtime_probes.run_runtime_probes({"fake": mod})

        assert report == {"fake": "failed: RuntimeError: boom"}

    async def test_reports_skipped_when_no_auto_probable_tool(self):
        """Module with no zero-arg or `*_schema` tool → `"skipped: ..."`."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "delete_thing", required=["id"])

        report = await runtime_probes.run_runtime_probes({"fake": mod})

        assert report == {"fake": "skipped: no auto-probable tool found"}

    async def test_one_bad_server_does_not_mask_others(self):
        """Per-server isolation: failure on one server doesn't affect the report for others."""
        good_mod = types.ModuleType("servers.good")
        _attach_fake_binding(good_mod, "list_files", required=[], return_value="ok")

        bad_mod = types.ModuleType("servers.bad")
        bad_binding = AsyncMock(side_effect=ConnectionError("down"))
        bad_binding._mcp_tool = _fake_mcp_tool(required=[])
        setattr(bad_mod, "list_files", bad_binding)

        report = await runtime_probes.run_runtime_probes(
            {"good": good_mod, "bad": bad_mod}
        )

        assert report["good"] == "ok"
        assert report["bad"].startswith("failed: ConnectionError")

    async def test_uses_schema_strategy_with_correct_args(self):
        """Probes that picked the `*_schema` strategy must call with `request={...}`."""
        mod = types.ModuleType("servers.fake")
        # Only a schema tool; both args required.
        binding = _attach_fake_binding(
            mod, "sheets_schema", required=["request"], return_value="schema-result"
        )

        report = await runtime_probes.run_runtime_probes({"fake": mod})

        assert report == {"fake": "ok"}
        binding.assert_awaited_once_with(request={"model": "input"})

    async def test_empty_modules_dict_returns_empty_report(self):
        """No modules passed in → empty report, no errors."""
        report = await runtime_probes.run_runtime_probes({})
        assert report == {}
