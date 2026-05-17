"""Unit tests for runner.agents.coding_mcp_agent.tool_discovery_docs.

Covers the filesystem knowledge-tree generation (`build_tool_docs_dir`) and
the helper formatters. All tests use fake modules with `_mcp_tool`-tagged
attributes — no MCP gateway is required.
"""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from runner.agents.coding_mcp_agent.tool_discovery_docs import (
    _build_call_signature,
    _format_tool_txt,
    _json_type_to_python,
    build_tool_docs_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_tool(
    description: str | None = "",
    input_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Stand-in for an mcp.types.Tool — only description + inputSchema are read."""
    return SimpleNamespace(
        description=description,
        inputSchema=input_schema if input_schema is not None else {},
    )


def _fake_module(
    server_name: str, tools: dict[str, SimpleNamespace]
) -> types.ModuleType:
    """Build a fake `servers.<name>` module with `_mcp_tool`-tagged async fns."""
    mod = types.ModuleType(f"servers.{server_name}")
    for fn_name, tool in tools.items():

        async def _stub(**kwargs):  # pragma: no cover — never invoked
            return None

        _stub._mcp_tool = tool  # pyright: ignore[reportFunctionMemberAccess]
        setattr(mod, fn_name, _stub)
    return mod


# ---------------------------------------------------------------------------
# build_tool_docs_dir — filesystem tree generation
# ---------------------------------------------------------------------------


class TestBuildToolDocsDir:
    def test_empty_modules_creates_servers_root_only(self):
        """No modules → `servers/` and empty `servers/_index.txt`; no server dirs."""
        path = Path(build_tool_docs_dir({}))

        servers_dir = path / "servers"
        assert servers_dir.is_dir()
        assert (servers_dir / "_index.txt").read_text() == "\n"
        assert [p.name for p in servers_dir.iterdir()] == ["_index.txt"]

    def test_single_server_single_tool_layout(self):
        """One server + one tool → correct file tree, indexes populated."""
        tool = _fake_tool(
            description="Add a row to a tab",
            input_schema={
                "properties": {"sheet_id": {"type": "string"}},
                "required": ["sheet_id"],
            },
        )
        mod = _fake_module("sheets_server", {"add_row": tool})

        path = Path(build_tool_docs_dir({"sheets_server": mod}))

        server_dir = path / "servers" / "sheets_server"
        assert server_dir.is_dir()
        assert (server_dir / "add_row.txt").is_file()
        assert "add_row: Add a row to a tab" in (server_dir / "_index.txt").read_text()
        assert "sheets_server: 1 tools" in (path / "servers" / "_index.txt").read_text()

    def test_multiple_servers_dont_cross_contaminate(self):
        """Tools from server A never end up under server B's directory."""
        mod_a = _fake_module(
            "alpha_server", {"do_a": _fake_tool(description="from alpha")}
        )
        mod_b = _fake_module(
            "beta_server", {"do_b": _fake_tool(description="from beta")}
        )

        path = Path(build_tool_docs_dir({"alpha_server": mod_a, "beta_server": mod_b}))

        alpha_files = {p.name for p in (path / "servers" / "alpha_server").iterdir()}
        beta_files = {p.name for p in (path / "servers" / "beta_server").iterdir()}
        assert alpha_files == {"_index.txt", "do_a.txt"}
        assert beta_files == {"_index.txt", "do_b.txt"}

    def test_tool_with_no_description_does_not_crash(self):
        """Description=None → empty first line in index, no traceback."""
        tool = _fake_tool(description=None)
        mod = _fake_module("svr", {"weird_tool": tool})

        path = Path(build_tool_docs_dir({"svr": mod}))

        index = (path / "servers" / "svr" / "_index.txt").read_text()
        assert index.startswith("weird_tool: ")

    def test_multiline_description_only_first_line_in_index(self):
        """Per-server index keeps tools to one line each; full description in .txt."""
        tool = _fake_tool(description="First line\nSecond line\nThird line")
        mod = _fake_module("svr", {"tool": tool})

        path = Path(build_tool_docs_dir({"svr": mod}))

        index = (path / "servers" / "svr" / "_index.txt").read_text()
        full = (path / "servers" / "svr" / "tool.txt").read_text()
        assert "First line" in index
        assert "Second line" not in index
        assert "Second line" in full
        assert "Third line" in full

    def test_module_attrs_without_mcp_tool_marker_are_skipped(self):
        """Only `_mcp_tool`-tagged attrs become docs files."""
        mod = _fake_module("svr", {"real_tool": _fake_tool(description="real")})
        # Add a plain attribute with no `_mcp_tool` marker — must be filtered.
        mod.__dict__["unrelated_helper"] = lambda: None

        path = Path(build_tool_docs_dir({"svr": mod}))

        files = {p.name for p in (path / "servers" / "svr").iterdir()}
        assert "real_tool.txt" in files
        assert "unrelated_helper.txt" not in files


# ---------------------------------------------------------------------------
# _format_tool_txt — per-tool .txt content
# ---------------------------------------------------------------------------


class TestFormatToolTxt:
    def test_none_input_schema_emits_no_parameters_block(self):
        """`inputSchema=None` is tolerated; output has no Parameters: section."""
        tool = _fake_tool(description="d", input_schema=None)
        tool.inputSchema = None  # explicit override

        out = _format_tool_txt("svr", "tool", tool)

        assert "Parameters:" not in out
        assert "svr.tool()" in out

    def test_null_required_field_does_not_crash(self):
        """`{"required": null}` from a misbehaving server → handled by parse_input_schema."""
        tool = _fake_tool(
            input_schema={
                "properties": {"x": {"type": "string"}},
                "required": None,
            }
        )

        out = _format_tool_txt("svr", "tool", tool)

        # Param exists but is optional (since required was null → empty set).
        assert "x" in out
        assert "(optional)" in out

    def test_required_and_optional_markers_in_param_table(self):
        """Each parameter line shows (required) or (optional) correctly."""
        tool = _fake_tool(
            input_schema={
                "properties": {
                    "must_have": {"type": "string"},
                    "extra": {"type": "integer"},
                },
                "required": ["must_have"],
            }
        )

        out = _format_tool_txt("svr", "tool", tool)

        # Find the parameter-table rows (indented with two spaces) for each param.
        must_line = next(
            ln for ln in out.splitlines() if ln.startswith("  ") and "must_have" in ln
        )
        extra_line = next(
            ln for ln in out.splitlines() if ln.startswith("  ") and "extra" in ln
        )
        assert "(required)" in must_line
        assert "(optional)" in extra_line


# ---------------------------------------------------------------------------
# _build_call_signature — Python signature reconstruction
# ---------------------------------------------------------------------------


class TestBuildCallSignature:
    def test_required_params_appear_before_optional(self):
        """Signature lists required params first, optional after — order matters for LLM read."""
        properties = {
            "opt_a": {"type": "string"},
            "req_b": {"type": "integer"},
            "opt_c": {"type": "boolean"},
            "req_d": {"type": "string"},
        }
        required = {"req_b", "req_d"}

        sig = _build_call_signature("svr", "fn", properties, required)

        # Required appear in original insertion order before any optionals.
        assert sig == "svr.fn(req_b: int, req_d: str, opt_a: str, opt_c: bool)"

    def test_empty_properties_yields_empty_parens(self):
        """No params → `server.fn()` with empty parens."""
        sig = _build_call_signature("svr", "fn", {}, set())
        assert sig == "svr.fn()"


# ---------------------------------------------------------------------------
# _json_type_to_python — JSON-schema → Python type-name mapping
# ---------------------------------------------------------------------------


class TestJsonTypeToPython:
    def test_all_known_json_types_map_correctly(self):
        cases = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        for json_type, py_type in cases.items():
            assert _json_type_to_python({"type": json_type}) == py_type

    def test_unknown_type_falls_back_to_Any(self):
        """Unrecognized or missing `type` → `"Any"` so the docs still build."""
        assert _json_type_to_python({"type": "weird"}) == "Any"
        assert _json_type_to_python({}) == "Any"
