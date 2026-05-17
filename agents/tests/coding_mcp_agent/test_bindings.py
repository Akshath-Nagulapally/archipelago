"""Unit tests for runner.agents.coding_mcp_agent.bindings.

These tests run without a live MCP gateway by mocking `httpx.AsyncClient`
(for the GET /apps lookup) and `fastmcp.Client` (for list_tools / call_tool).
They cover:

- prefix-detection in `_select_server_tools` (regression: Bug 3 review fix)
- the binding wrapper's closure capture, `CallToolResult` unwrap, and empty-
  content path in `_register_tool_on_module` (regression: Bug 2 review fix)
- end-to-end `build_server_modules`: empty gateway, prefix-stripping
- `_get_server_names`'s URL trimming (regression: Bug 1 — the `rstrip` →
  `removesuffix` fix)

pytest's auto asyncio mode is set in agents/pyproject.toml, so
`async def test_...` works without `@pytest.mark.asyncio`.
"""

from __future__ import annotations

import sys
import types as types_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from runner.agents.coding_mcp_agent import bindings

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_global_state():
    """Reset the `servers` namespace and the gateway-config cache between tests.

    `bindings` mutates `sys.modules` (registers `servers.<name>` packages) and
    a module-level cache (`_GATEWAY_CLIENT_CONFIG_CACHE`). Without resetting
    them, tests could see leftovers from earlier tests and pass for the wrong
    reason.
    """
    yield
    for key in list(sys.modules):
        if key == "servers" or key.startswith("servers."):
            del sys.modules[key]
    bindings._GATEWAY_CLIENT_CONFIG_CACHE.clear()


def _fake_tool(name: str, description: str = "") -> SimpleNamespace:
    """Stand-in for fastmcp's Tool: only `name` and `description` are read."""
    return SimpleNamespace(name=name, description=description)


def _fake_call_tool_result(text: str | None) -> SimpleNamespace:
    """Stand-in for fastmcp.CallToolResult.

    - text=None → content=[] (the binding should return "")
    - text="..." → content=[TextContent(text=...)] (the binding returns it)
    """
    if text is None:
        return SimpleNamespace(content=[])
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _fake_client(call_tool_result: Any = None) -> MagicMock:
    """Stand-in for fastmcp.Client. `call_tool` is an AsyncMock."""
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=call_tool_result)
    return client


# ---------------------------------------------------------------------------
# _select_server_tools — prefix-detection logic (Bug 3 regression)
# ---------------------------------------------------------------------------


class TestSelectServerTools:
    def test_multi_server_returns_prefixed_tools_with_strip(self):
        """Multi-server gateway: take only the tools whose name starts with `<server>_`."""
        tools = [
            _fake_tool("filesystem_list_files"),
            _fake_tool("mail_send"),
            _fake_tool("filesystem_read_text_file"),
        ]
        result = bindings._select_server_tools("filesystem", tools, server_count=2)

        assert result is not None
        selected, strip_prefix = result
        assert [t.name for t in selected] == [
            "filesystem_list_files",
            "filesystem_read_text_file",
        ]
        assert strip_prefix is True

    def test_single_server_no_prefix_returns_all_without_strip(self):
        """Single-server gateway without prefixes: take everything, don't strip."""
        tools = [_fake_tool("list_files"), _fake_tool("read_text_file")]

        result = bindings._select_server_tools("filesystem", tools, server_count=1)

        assert result is not None
        selected, strip_prefix = result
        assert [t.name for t in selected] == ["list_files", "read_text_file"]
        assert strip_prefix is False

    def test_single_server_with_prefix_follows_actual_tool_names(self):
        """Future-proof: if fastmcp starts prefixing even single-server gateways,
        we follow the data (prefixed path) instead of the old `len==1` heuristic."""
        tools = [_fake_tool("filesystem_list_files")]

        result = bindings._select_server_tools("filesystem", tools, server_count=1)

        assert result is not None
        selected, strip_prefix = result
        assert [t.name for t in selected] == ["filesystem_list_files"]
        assert strip_prefix is True

    def test_multi_server_with_no_matching_tools_returns_none(self):
        """Multi-server gateway but this server contributed nothing → None (caller logs + skips)."""
        tools = [_fake_tool("mail_send"), _fake_tool("sheets_read_tab")]

        result = bindings._select_server_tools("filesystem", tools, server_count=3)

        assert result is None


# ---------------------------------------------------------------------------
# _register_tool_on_module — closure + CallToolResult unwrap (Bug 2 regression)
# ---------------------------------------------------------------------------


class TestRegisterToolOnModule:
    def test_attaches_callable_with_correct_name_and_doc(self):
        """The registered binding should have the bound `__name__` and `__doc__`."""
        mod = types_module.ModuleType("servers.fake")
        tool = _fake_tool("filesystem_list_files", "List files in a path")

        bindings._register_tool_on_module(mod, tool, "list_files", _fake_client())

        attached = mod.list_files  # type: ignore[attr-defined]
        assert callable(attached)
        assert attached.__name__ == "list_files"
        assert attached.__doc__ == "List files in a path"

    async def test_wrapper_calls_correct_tool_name_per_registration(self):
        """Critical: each binding's closure must capture *its own* tool name —
        regression test for the late-binding loop-variable pitfall the default-
        arg closure idiom is there to prevent."""
        mod = types_module.ModuleType("servers.fake")
        client = _fake_client(_fake_call_tool_result("ok"))

        bindings._register_tool_on_module(
            mod, _fake_tool("filesystem_list_files"), "list_files", client
        )
        bindings._register_tool_on_module(
            mod, _fake_tool("filesystem_read"), "read", client
        )

        await mod.list_files(path="/x")  # type: ignore[attr-defined]
        client.call_tool.assert_awaited_with("filesystem_list_files", {"path": "/x"})

        await mod.read(path="/y")  # type: ignore[attr-defined]
        client.call_tool.assert_awaited_with("filesystem_read", {"path": "/y"})

    async def test_wrapper_unwraps_call_tool_result_to_first_text(self):
        """fastmcp returns CallToolResult; the binding should expose .content[0].text."""
        mod = types_module.ModuleType("servers.fake")
        client = _fake_client(_fake_call_tool_result("hello world"))

        bindings._register_tool_on_module(mod, _fake_tool("ls"), "ls", client)

        result = await mod.ls()  # type: ignore[attr-defined]
        assert result == "hello world"

    async def test_wrapper_returns_empty_string_when_content_is_empty(self):
        """Empty CallToolResult.content → "" (so callers can `if result: ...` safely)."""
        mod = types_module.ModuleType("servers.fake")
        client = _fake_client(_fake_call_tool_result(None))  # content=[]

        bindings._register_tool_on_module(mod, _fake_tool("ls"), "ls", client)

        result = await mod.ls()  # type: ignore[attr-defined]
        assert result == ""

    async def test_wrapper_does_not_leak_caller_kwargs_to_closure(self):
        """Regression for Bug 4 — the leaky default-arg closure.

        Caller-supplied kwargs (even ones with names like `_tool_name` or
        `_client` that USED to be closure-param names in the old default-arg
        idiom) must flow through to the MCP tool unchanged. They must NOT
        override the closure-captured tool name or client.
        """
        mod = types_module.ModuleType("servers.fake")
        client = _fake_client(_fake_call_tool_result("ok"))

        bindings._register_tool_on_module(
            mod, _fake_tool("some_tool"), "some_tool", client
        )

        # The names below would have collided with the old closure-param names.
        # In the buggy version: `_client="external"` would override our
        # closure-captured client, and `await "external".call_tool(...)` would
        # raise AttributeError. With the factory fix, they pass through as
        # ordinary kwargs forwarded to the real MCP tool.
        await mod.some_tool(  # type: ignore[attr-defined]
            _client="external_service",
            _tool_name="not_real",
            real_arg="x",
        )

        # The shared `client` mock should still receive the call, with the
        # closure-captured tool name "some_tool" and ALL caller kwargs
        # forwarded unmodified.
        client.call_tool.assert_awaited_with(
            "some_tool",
            {
                "_client": "external_service",
                "_tool_name": "not_real",
                "real_arg": "x",
            },
        )


# ---------------------------------------------------------------------------
# build_server_modules — top-level integration with mocked deps
# ---------------------------------------------------------------------------


class TestBuildServerModules:
    async def test_empty_tool_list_returns_empty_dict(self, monkeypatch):
        """Gateway with no tools → no modules registered, returns {}."""

        async def fake_get_server_names(_url):
            return ["filesystem"]

        monkeypatch.setattr(bindings, "_get_server_names", fake_get_server_names)

        client = MagicMock()
        client.list_tools = AsyncMock(return_value=[])

        result = await bindings.build_server_modules("http://gw/mcp/", client)

        assert result == {}

    async def test_strips_prefix_in_attribute_names(self, monkeypatch):
        """Multi-server gateway: each module's attrs should be the *un-prefixed* tool names."""

        async def fake_get_server_names(_url):
            return ["filesystem", "mail"]

        monkeypatch.setattr(bindings, "_get_server_names", fake_get_server_names)

        client = MagicMock()
        client.list_tools = AsyncMock(
            return_value=[
                _fake_tool("filesystem_list_files"),
                _fake_tool("filesystem_read"),
                _fake_tool("mail_send"),
            ]
        )

        result = await bindings.build_server_modules("http://gw/mcp/", client)

        assert set(result.keys()) == {"filesystem", "mail"}
        # Un-prefixed names exist as attributes:
        assert hasattr(result["filesystem"], "list_files")
        assert hasattr(result["filesystem"], "read")
        assert hasattr(result["mail"], "send")
        # The prefixed forms are NOT exposed (would be a regression):
        assert not hasattr(result["filesystem"], "filesystem_list_files")


# ---------------------------------------------------------------------------
# _get_server_names — URL trimming (Bug 1 regression)
# ---------------------------------------------------------------------------


class TestGetServerNames:
    async def test_uses_removesuffix_not_rstrip(self, monkeypatch):
        """Regression: original code used `.rstrip("/mcp/")` which strips trailing
        chars in the SET, not the literal suffix. A hostname like `gateway.cmp.local`
        would have `c`, `m`, `p` chewed off. `removesuffix` is the correct fix."""

        captured_urls: list[str] = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"servers": ["filesystem"]}

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def get(self, url, timeout=None):
                captured_urls.append(url)
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        # A URL whose host contains characters {c, m, p} — would have been corrupted
        # by the buggy rstrip.
        servers = await bindings._get_server_names("http://gateway.cmp.local/mcp/")

        assert captured_urls == ["http://gateway.cmp.local/apps"]
        assert servers == ["filesystem"]

    async def test_handles_url_without_trailing_slash(self, monkeypatch):
        """`http://host/mcp` (no trailing slash) — should still hit /apps cleanly."""

        captured_urls: list[str] = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"servers": []}

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def get(self, url, timeout=None):
                captured_urls.append(url)
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        await bindings._get_server_names("http://gw/mcp")

        assert captured_urls == ["http://gw/apps"]


# ---------------------------------------------------------------------------
# get_bound_tools — filtering by _mcp_tool sentinel
# ---------------------------------------------------------------------------


class TestGetBoundTools:
    def test_returns_mcp_tool_tagged_attrs(self):
        """Only attrs with _mcp_tool set are returned."""
        mod = types_module.ModuleType("servers.fake")
        tool_a = SimpleNamespace(name="tool_a")
        tool_b = SimpleNamespace(name="tool_b")

        async def fn_a(): ...
        async def fn_b(): ...

        fn_a.__dict__["_mcp_tool"] = tool_a
        fn_b.__dict__["_mcp_tool"] = tool_b

        mod.__dict__["fn_a"] = fn_a
        mod.__dict__["fn_b"] = fn_b

        result = bindings.get_bound_tools(mod)
        assert result == {"fn_a": tool_a, "fn_b": tool_b}

    def test_untagged_attrs_are_excluded(self):
        """Plain module attrs without _mcp_tool are filtered out."""
        mod = types_module.ModuleType("servers.fake")
        mod.__dict__["helper"] = lambda: None
        mod.__dict__["CONSTANT"] = 42

        result = bindings.get_bound_tools(mod)
        assert result == {}

    def test_empty_module_returns_empty_dict(self):
        """Module with no attrs at all → empty dict."""
        mod = types_module.ModuleType("servers.fake")
        assert bindings.get_bound_tools(mod) == {}
