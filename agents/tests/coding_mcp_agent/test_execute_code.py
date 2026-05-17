"""Unit tests for runner.agents.coding_mcp_agent.tools.execute_code.

These tests exercise the sandbox's contract without involving a live MCP
gateway. The bindings-integration test (`TestBindingsIntegration`) fakes a
`servers.fake_server` module directly in `sys.modules`, which is exactly
what `bindings.build_server_modules()` does at runtime — so we verify the
sandbox/bindings handshake without pulling the gateway into the test.

pytest's auto asyncio mode is set in agents/pyproject.toml, so
`async def test_...` works without `@pytest.mark.asyncio`.
"""

from __future__ import annotations

import sys
import types as types_module

import pytest

from runner.agents.coding_mcp_agent.tools.execute_code import (
    DEFAULT_TIMEOUT_SECONDS,
    execute_code,
)

# ---------------------------------------------------------------------------
# Core behavior — the contract: print works, top-level await works, multi-line
# code runs end-to-end.
# ---------------------------------------------------------------------------


class TestCoreBehavior:
    async def test_plain_print_is_captured(self):
        result = await execute_code('print("hello", 1 + 2)')
        assert result.success
        assert result.stdout == "hello 3\n"
        assert result.error is None

    async def test_top_level_await_works(self):
        """The whole reason we wrap user code in `async def __user_code__()` —
        the LLM must be able to write `await foo()` at the top level."""
        result = await execute_code(
            'import asyncio\nawait asyncio.sleep(0)\nprint("awaited")'
        )
        assert result.success
        assert result.stdout == "awaited\n"

    async def test_multi_line_code_with_loop_and_computation(self):
        code = "total = 0\nfor i in range(5):\n    total += i\nprint(total)\n"
        result = await execute_code(code)
        assert result.success
        assert result.stdout == "10\n"

    async def test_code_with_no_output_succeeds_with_empty_stdout(self):
        result = await execute_code("x = 1 + 1")
        assert result.success
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# Error handling — runtime errors, partial stdout preservation, syntax errors.
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_runtime_error_returns_traceback_and_marks_failure(self):
        result = await execute_code("x = 1 / 0")
        assert not result.success
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    async def test_partial_stdout_is_preserved_through_an_error(self):
        """If the LLM printed debug output before crashing, that output must
        still be returned — it's the main thing the LLM uses to self-correct."""
        result = await execute_code('print("before")\nx = 1 / 0')
        assert not result.success
        assert "before" in result.stdout
        assert "ZeroDivisionError" in (result.error or "")

    async def test_syntax_error_is_caught_at_exec_step(self):
        """SyntaxError happens at the `exec(wrapped, namespace)` call, before
        the user code ever runs — different code path than runtime errors."""
        result = await execute_code("def bad(:")
        assert not result.success
        assert result.error is not None
        assert "SyntaxError" in result.error
        assert result.stdout == ""

    async def test_name_error_returns_traceback(self):
        result = await execute_code("print(undefined_name)")
        assert not result.success
        assert "NameError" in (result.error or "")


# ---------------------------------------------------------------------------
# Timeout — wall-clock cutoff.
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_long_running_async_code_is_killed(self):
        result = await execute_code(
            "import asyncio\nawait asyncio.sleep(5)", timeout=0.2
        )
        assert not result.success
        assert result.timed_out
        assert "timed out" in (result.error or "").lower()

    def test_default_timeout_is_four_minutes(self):
        """The 240-second default is the contract the agent loop relies on."""
        assert DEFAULT_TIMEOUT_SECONDS == 240.0


# ---------------------------------------------------------------------------
# Stderr handling — captured and labeled so the LLM can distinguish it.
# ---------------------------------------------------------------------------


class TestStderr:
    async def test_stderr_is_captured_and_labeled(self):
        result = await execute_code(
            'import sys\nsys.stderr.write("oops\\n")\nprint("done")'
        )
        assert result.success
        assert "done" in result.stdout
        assert "[stderr]" in result.stdout
        assert "oops" in result.stdout

    async def test_stderr_only_no_stdout(self):
        result = await execute_code('import sys\nsys.stderr.write("warning\\n")')
        assert result.success
        assert result.stdout.startswith("[stderr]")
        assert "warning" in result.stdout


# ---------------------------------------------------------------------------
# No persistence — each `execute_code` call is independent. Nothing from
# call 1 should be reachable in call 2. This is the contract we committed
# to: "once we execute code we are basically done".
# ---------------------------------------------------------------------------


class TestNoPersistence:
    async def test_local_variables_do_not_carry_over(self):
        r1 = await execute_code("x = 42")
        r2 = await execute_code("print(x)")

        assert r1.success
        assert not r2.success
        assert "NameError" in (r2.error or "")

    async def test_functions_defined_in_prior_call_do_not_carry_over(self):
        r1 = await execute_code('def greet():\n    return "hi"')
        r2 = await execute_code("print(greet())")

        assert r1.success
        assert not r2.success
        assert "NameError" in (r2.error or "")

    async def test_user_imports_do_not_leak_name_into_next_call(self):
        """`import json` in call 1 caches the module in `sys.modules` (normal
        Python behavior), but the *name* `json` is bound in call 1's namespace
        only — call 2 should not see it."""
        r1 = await execute_code("import json")
        r2 = await execute_code('print(json.dumps({"a": 1}))')

        assert r1.success
        assert not r2.success
        assert "NameError" in (r2.error or "")


# ---------------------------------------------------------------------------
# Bindings integration — the whole point. With a fake `servers.fake_server`
# module registered in `sys.modules` (which is exactly what bindings.py does
# at runtime), the sandbox must be able to `from servers import fake_server`
# and call an async function on it.
# ---------------------------------------------------------------------------


class TestBindingsIntegration:
    @pytest.fixture
    def fake_server(self):
        """Register a `servers.fake_server` module with one async function,
        mirroring the shape that `bindings.build_server_modules` produces.
        Cleaned up afterwards so it can't leak into other tests."""
        servers_pkg = types_module.ModuleType("servers")
        # ModuleType's __path__ isn't a static attr; use __dict__ to set it
        # dynamically. Same pattern below for the binding attachments — it's
        # the actual dynamic-attribute idiom and passes both ruff and pyright
        # with no suppressions.
        servers_pkg.__dict__["__path__"] = []
        sys.modules["servers"] = servers_pkg

        fake_mod = types_module.ModuleType("servers.fake_server")

        async def echo(**kwargs):
            return f"echo:{kwargs}"

        fake_mod.__dict__["echo"] = echo
        sys.modules["servers.fake_server"] = fake_mod
        servers_pkg.__dict__["fake_server"] = fake_mod

        yield

        for key in ("servers.fake_server", "servers"):
            sys.modules.pop(key, None)

    async def test_sandbox_can_import_and_call_a_server_binding(self, fake_server):
        result = await execute_code(
            "from servers import fake_server\n"
            "out = await fake_server.echo(name='alice', n=1)\n"
            "print(out)\n"
        )
        assert result.success, result.error
        assert "echo:" in result.stdout
        assert "'name': 'alice'" in result.stdout
        assert "'n': 1" in result.stdout
