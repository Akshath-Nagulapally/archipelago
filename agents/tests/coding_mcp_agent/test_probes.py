"""Unit tests for runner.agents.coding_mcp_agent.probes.

Three groups of tests, mirroring the three probe functions plus the
orchestrator:

- TestPickProbe — `_pick_probe` strategy selection (no MCP needed).
- TestProbeRemoteMcp — orchestration of per-server probes; raises on any failure.
- TestProbeSandboxImports — `from servers import ...` round-trip via the sandbox.
- TestProbeBashReadsDocs — `cat <docs>/servers/_index.txt` via bash.
- TestRunStartupProbes — the single-call orchestrator.

pytest's auto asyncio mode is set in agents/pyproject.toml, so
`async def test_...` works without `@pytest.mark.asyncio`.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from runner.agents.coding_mcp_agent import probes

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
    side_effect: BaseException | type[BaseException] | None = None,
) -> AsyncMock:
    """Attach a probe-able binding to `mod` with the same shape as a real one.

    Either `return_value` is used as the normal return, or `side_effect` is
    set to make the mock raise. Returns the AsyncMock so the test can assert
    how it was called.
    """
    # Mock's constructor sets arbitrary kwargs as attributes on the instance,
    # so we can attach `_mcp_tool` here without a follow-up assignment —
    # which avoids both ruff's setattr-with-constant warning and pyright's
    # "attribute not declared" warning.
    fake_tool = _fake_mcp_tool(required=required)
    if side_effect is not None:
        mock = AsyncMock(side_effect=side_effect, _mcp_tool=fake_tool)
    else:
        mock = AsyncMock(return_value=return_value, _mcp_tool=fake_tool)
    setattr(mod, fn_name, mock)
    return mock


@pytest.fixture
def fake_servers_namespace():
    """Register a fresh `servers` namespace package in sys.modules and
    clean it up after the test. Returns a helper for registering child
    modules under it."""
    servers_pkg = types.ModuleType("servers")
    # __path__ is a dynamic namespace-package attribute; assign via __dict__
    # so neither ruff nor pyright complains.
    servers_pkg.__dict__["__path__"] = []
    sys.modules["servers"] = servers_pkg

    registered: list[str] = []

    def register(name: str) -> types.ModuleType:
        mod = types.ModuleType(f"servers.{name}")
        sys.modules[f"servers.{name}"] = mod
        setattr(servers_pkg, name, mod)
        registered.append(name)
        return mod

    yield register

    for name in registered:
        sys.modules.pop(f"servers.{name}", None)
    sys.modules.pop("servers", None)


# ---------------------------------------------------------------------------
# _pick_probe — strategy selection
# ---------------------------------------------------------------------------


class TestPickProbe:
    def test_prefers_zero_required_args_tool(self):
        """Strategy 1: any tool with empty `required` is callable with no kwargs."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "list_files", required=[])
        _attach_fake_binding(mod, "delete_thing", required=["path"])  # needs args

        choice = probes._pick_probe(mod)

        assert choice == ("list_files", {})

    def test_falls_back_to_schema_introspection_tool(self):
        """Strategy 2: if no zero-arg tool, prefer a `*_schema` meta-tool."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "sheets", required=["request"])
        _attach_fake_binding(mod, "sheets_schema", required=["request"])

        choice = probes._pick_probe(mod)

        assert choice == ("sheets_schema", {"request": {"model": "input"}})

    def test_recognizes_bare_schema_function_name(self):
        """Strategy 2 also matches a tool named exactly `schema`."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "do_thing", required=["arg"])
        _attach_fake_binding(mod, "schema", required=["request"])

        choice = probes._pick_probe(mod)

        assert choice == ("schema", {"request": {"model": "input"}})

    def test_returns_none_when_no_strategy_matches(self):
        """No zero-arg tool and no `*_schema` tool → None."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "send_email", required=["to", "subject"])
        _attach_fake_binding(mod, "delete_thing", required=["id"])

        choice = probes._pick_probe(mod)

        assert choice is None

    def test_returns_none_when_module_has_no_bindings(self):
        mod = types.ModuleType("servers.fake")
        choice = probes._pick_probe(mod)
        assert choice is None

    def test_skips_attrs_without_mcp_tool_marker(self):
        """Only `_mcp_tool`-tagged attrs count as probable bindings."""
        mod = types.ModuleType("servers.fake")

        async def unrelated_helper():
            return "noop"

        # ModuleType doesn't statically declare arbitrary attrs, so direct
        # assignment trips pyright's reportAttributeAccessIssue. Use the
        # module's `__dict__` (which is `dict[str, Any]`) — it's the actual
        # dynamic-attribute idiom and passes both ruff and pyright cleanly.
        mod.__dict__["unrelated_helper"] = unrelated_helper

        choice = probes._pick_probe(mod)
        assert choice is None


# ---------------------------------------------------------------------------
# probe_remote_mcp — orchestration
# ---------------------------------------------------------------------------


class TestProbeRemoteMcp:
    async def test_returns_silently_on_all_ok(self):
        """All probes pass → no raise, no return value."""
        mod = types.ModuleType("servers.fake")
        binding = _attach_fake_binding(
            mod, "list_files", required=[], return_value="output"
        )

        await probes.probe_remote_mcp({"fake": mod})  # should not raise

        binding.assert_awaited_once_with()

    async def test_raises_with_exception_class_and_message_on_failure(self):
        """A probe that raises → `probe_remote_mcp` re-raises a
        `RuntimeError` whose message names the failing server and the
        underlying exception class."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(
            mod, "list_files", required=[], side_effect=RuntimeError("boom")
        )

        with pytest.raises(RuntimeError) as exc_info:
            await probes.probe_remote_mcp({"fake": mod})

        msg = str(exc_info.value)
        assert "fake" in msg
        assert "RuntimeError" in msg
        assert "boom" in msg

    async def test_skipped_server_does_not_raise(self):
        """A module with no zero-arg or `*_schema` tool → logs a warning
        and passes through. Skipped ≠ failed — we couldn't auto-verify,
        not "known broken"."""
        mod = types.ModuleType("servers.fake")
        _attach_fake_binding(mod, "delete_thing", required=["id"])

        await probes.probe_remote_mcp({"fake": mod})  # should not raise

    async def test_one_bad_server_surfaces_with_full_report_in_message(self):
        """Per-server isolation: every server is probed before the raise,
        so the exception message shows both the good and bad results."""
        good_mod = types.ModuleType("servers.good")
        _attach_fake_binding(good_mod, "list_files", required=[], return_value="ok")

        bad_mod = types.ModuleType("servers.bad")
        _attach_fake_binding(
            bad_mod, "list_files", required=[], side_effect=ConnectionError("down")
        )

        with pytest.raises(RuntimeError) as exc_info:
            await probes.probe_remote_mcp({"good": good_mod, "bad": bad_mod})

        msg = str(exc_info.value)
        assert "bad" in msg
        assert "ConnectionError" in msg
        # Full report (with the good server's "ok") is included for context.
        assert "good" in msg
        assert "ok" in msg

    async def test_uses_schema_strategy_with_correct_args(self):
        """Probes that picked the `*_schema` strategy must call with `request={...}`."""
        mod = types.ModuleType("servers.fake")
        binding = _attach_fake_binding(
            mod, "sheets_schema", required=["request"], return_value="schema-result"
        )

        await probes.probe_remote_mcp({"fake": mod})

        binding.assert_awaited_once_with(request={"model": "input"})

    async def test_empty_modules_dict_does_not_raise(self):
        """No modules at all → vacuous success, no raise."""
        await probes.probe_remote_mcp({})


# ---------------------------------------------------------------------------
# probe_sandbox_imports
# ---------------------------------------------------------------------------


class TestProbeSandboxImports:
    async def test_passes_when_every_module_resolves(self, fake_servers_namespace):
        mod_a = fake_servers_namespace("alpha_server")
        mod_b = fake_servers_namespace("beta_server")

        await probes.probe_sandbox_imports(
            {"alpha_server": mod_a, "beta_server": mod_b}
        )

    async def test_raises_when_a_module_is_missing_from_sys_modules(
        self, fake_servers_namespace
    ):
        """If a module is in the modules dict but not actually registered
        in sys.modules, the sandbox's `from servers import X` will fail."""
        orphan = types.ModuleType("servers.orphan_server")
        # Deliberately do not register() — orphan won't be importable.

        with pytest.raises(RuntimeError) as exc_info:
            await probes.probe_sandbox_imports({"orphan_server": orphan})

        msg = str(exc_info.value)
        assert "probe_sandbox_imports failed" in msg
        assert "orphan_server" in msg

    async def test_no_op_on_empty_modules(self):
        """An empty modules dict skips the probe (an empty
        `from servers import` would be a syntax error)."""
        await probes.probe_sandbox_imports({})


# ---------------------------------------------------------------------------
# probe_bash_reads_docs
# ---------------------------------------------------------------------------


class TestProbeBashReadsDocs:
    async def test_passes_when_index_file_exists_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            servers_dir = os.path.join(tmp, "servers")
            os.makedirs(servers_dir)
            with open(os.path.join(servers_dir, "_index.txt"), "w") as f:
                f.write("filesystem_server (12 tools)\nmail_server (8 tools)\n")

            await probes.probe_bash_reads_docs(tmp)  # should not raise

    async def test_raises_when_index_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError) as exc_info:
                await probes.probe_bash_reads_docs(tmp)

            msg = str(exc_info.value)
            assert "probe_bash_reads_docs failed" in msg
            assert "_index.txt" in msg

    async def test_raises_when_index_file_empty(self):
        """A readable-but-empty index means tool discovery docs weren't
        generated correctly — treat as a failure."""
        with tempfile.TemporaryDirectory() as tmp:
            servers_dir = os.path.join(tmp, "servers")
            os.makedirs(servers_dir)
            open(os.path.join(servers_dir, "_index.txt"), "w").close()

            with pytest.raises(RuntimeError) as exc_info:
                await probes.probe_bash_reads_docs(tmp)

            assert "empty" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# run_startup_probes — orchestrator
# ---------------------------------------------------------------------------


class TestRunStartupProbes:
    async def test_raises_clearly_when_tool_docs_path_is_none(self):
        """A None tool_docs_path is a programming error (the docs step
        never ran); we want a clear error, not a confusing downstream
        failure inside probe_bash_reads_docs."""
        with pytest.raises(RuntimeError) as exc_info:
            await probes.run_startup_probes({}, None)

        msg = str(exc_info.value)
        assert "tool_docs_path is None" in msg

    async def test_passes_when_all_three_probes_pass(self, fake_servers_namespace):
        """Happy-path orchestration: empty modules (vacuous remote probe,
        sandbox no-op) plus a real docs dir → no raise."""
        with tempfile.TemporaryDirectory() as tmp:
            servers_dir = os.path.join(tmp, "servers")
            os.makedirs(servers_dir)
            with open(os.path.join(servers_dir, "_index.txt"), "w") as f:
                f.write("(no servers configured)\n")

            # Empty modules dict → both remote and sandbox probes are no-ops.
            await probes.run_startup_probes({}, tmp)
