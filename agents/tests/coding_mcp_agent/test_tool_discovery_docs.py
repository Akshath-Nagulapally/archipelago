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
    is_wrapped_single_param,
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

        _stub.__dict__["_mcp_tool"] = tool
        mod.__dict__[fn_name] = _stub
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

    def test_wrapped_param_tool_produces_inner_fields_in_doc_file(self):
        """End-to-end: a meta-tool style schema lands on disk with both the
        wrapper signature AND the inner field block.

        Guards the wiring from `_format_tool_txt` → file write — the unit
        tests above already cover the formatter output, this just confirms
        the bytes that hit disk are the same ones the LLM will read via
        `cat /tmp/mcp-tool-docs/.../sheets.txt`.
        """
        meta_tool = _fake_tool(
            description="Spreadsheet operations meta-tool.",
            input_schema={
                "properties": {
                    "request": {
                        "type": "object",
                        "description": "Input for sheets meta-tool.",
                        "properties": {
                            "action": {"type": "string"},
                            "file_path": {"type": "string"},
                        },
                        "required": ["action"],
                    }
                },
                "required": ["request"],
            },
        )
        mod = _fake_module("sheets_server", {"sheets": meta_tool})

        path = Path(build_tool_docs_dir({"sheets_server": mod}))
        doc = (path / "servers" / "sheets_server" / "sheets.txt").read_text()

        # Wrapper signature on the first line.
        assert "sheets_server.sheets(*, request: dict)" in doc
        # Inner fields visible — the whole point of unwrapping.
        assert "Fields of `request`:" in doc
        assert "action" in doc
        assert "file_path" in doc


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
# _format_tool_txt — wrapped (meta-tool / Pydantic-wrapper) shape
# ---------------------------------------------------------------------------


def _wrapped_meta_tool_fake(wrapper_name: str = "request") -> SimpleNamespace:
    """Build a fake tool shaped like ``sheets_server.sheets`` — the canonical
    wrapped single-param meta-tool that motivated this whole branch."""
    return _fake_tool(
        description="Spreadsheet operations meta-tool.",
        input_schema={
            "properties": {
                wrapper_name: {
                    "type": "object",
                    "description": "Input for sheets meta-tool.",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to file within workspace.",
                        },
                        "tab_index": {
                            "type": "integer",
                            "description": "0-based tab position.",
                        },
                    },
                    "required": ["action"],
                }
            },
            "required": [wrapper_name],
        },
    )


class TestFormatToolTxtWrappedParam:
    def test_wrapped_tool_signature_keeps_wrapper_param(self):
        """The outer signature still shows the wrapper (`request: dict`).

        That's the actual call shape the LLM must use:
        ``await sheets(request={...})``. Hiding the wrapper would mislead the
        LLM into writing ``sheets(action=...)``.
        """
        out = _format_tool_txt("svr", "sheets", _wrapped_meta_tool_fake())
        first_line = out.splitlines()[0]
        assert first_line == "svr.sheets(*, request: dict)"

    def test_wrapped_tool_emits_fields_of_header_not_just_parameters(self):
        """Output contains both `Parameters:` (the wrapper) and `Fields of \\`request\\`:`."""
        out = _format_tool_txt("svr", "sheets", _wrapped_meta_tool_fake())
        assert "Parameters:" in out
        assert "Fields of `request`:" in out

    def test_wrapped_tool_lists_inner_field_names_and_types(self):
        """All inner fields appear with their mapped Python type names."""
        out = _format_tool_txt("svr", "sheets", _wrapped_meta_tool_fake())

        # Take only the lines after the "Fields of" header to avoid matching
        # the wrapper line in the outer "Parameters:" section.
        fields_section = out.split("Fields of `request`:", 1)[1]

        # Each inner field has its own indented row in the fields table.
        action_line = next(
            ln for ln in fields_section.splitlines() if "action " in ln
        )
        file_path_line = next(
            ln for ln in fields_section.splitlines() if "file_path " in ln
        )
        tab_index_line = next(
            ln for ln in fields_section.splitlines() if "tab_index " in ln
        )
        assert "str" in action_line
        assert "str" in file_path_line
        assert "int" in tab_index_line

    def test_wrapped_tool_inner_required_uses_inner_required_array(self):
        """`(required)` / `(optional)` for inner fields come from the INNER
        ``required`` array, not the outer one.

        In the fixture, the *outer* required is ``["request"]`` (the wrapper
        is required) and the *inner* required is ``["action"]``. So `action`
        should be required and `file_path`/`tab_index` should be optional.
        """
        out = _format_tool_txt("svr", "sheets", _wrapped_meta_tool_fake())
        fields_section = out.split("Fields of `request`:", 1)[1]

        action_line = next(
            ln for ln in fields_section.splitlines() if "action " in ln
        )
        file_path_line = next(
            ln for ln in fields_section.splitlines() if "file_path " in ln
        )

        assert "(required)" in action_line
        assert "(optional)" in file_path_line

    def test_wrapped_tool_preserves_inner_descriptions(self):
        """Inner-field descriptions from the inner schema make it into the output."""
        out = _format_tool_txt("svr", "sheets", _wrapped_meta_tool_fake())
        assert "Action to perform" in out
        assert "Path to file within workspace." in out
        assert "0-based tab position." in out

    def test_wrapped_tool_only_unwraps_one_level(self):
        """If the inner schema is ITSELF wrapped, we do NOT recurse a second time.

        Safety rail against runaway expansion on cyclic / deeply-nested
        schemas. The second level renders as the opaque ``dict`` type.
        """
        doubly_wrapped = _fake_tool(
            input_schema={
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {
                            "inner": {
                                "type": "object",
                                "properties": {
                                    "deep_field": {"type": "string"},
                                },
                            },
                        },
                        "required": ["inner"],
                    }
                },
                "required": ["outer"],
            }
        )

        out = _format_tool_txt("svr", "tool", doubly_wrapped)

        # We unwrap `outer` → see `inner` rendered as a `dict` row.
        assert "Fields of `outer`:" in out
        # But we do NOT unwrap further — `deep_field` must not appear and
        # there must be no second `Fields of` section.
        assert "deep_field" not in out
        assert out.count("Fields of") == 1

    def test_flat_tool_still_uses_parameters_header_only(self):
        """Regression guard: a normal flat-param tool does NOT trigger the
        wrapped path. No `Fields of` section appears."""
        flat_tool = _fake_tool(
            description="A normal flat tool.",
            input_schema={
                "properties": {
                    "file_path": {"type": "string"},
                    "encoding": {"type": "string"},
                },
                "required": ["file_path"],
            },
        )

        out = _format_tool_txt("svr", "read_text_file", flat_tool)
        assert "Parameters:" in out
        assert "Fields of" not in out


# ---------------------------------------------------------------------------
# _format_tool_txt — robustness on malformed wrapped-shape schemas
# ---------------------------------------------------------------------------


class TestFormatToolTxtWrappedRobustness:
    def test_wrapped_with_null_inner_required_does_not_crash(self):
        """Some servers emit ``"required": null`` instead of omitting the key.

        The inner-required parse must tolerate this — every inner field shows
        as ``(optional)`` and no traceback is raised. Mirrors the existing
        top-level robustness test (`test_null_required_field_does_not_crash`).
        """
        tool = _fake_tool(
            input_schema={
                "properties": {
                    "req": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "string"},
                            "y": {"type": "integer"},
                        },
                        "required": None,
                    }
                },
                "required": ["req"],
            }
        )

        out = _format_tool_txt("svr", "tool", tool)
        fields_section = out.split("Fields of `req`:", 1)[1]
        x_line = next(ln for ln in fields_section.splitlines() if "x " in ln)
        y_line = next(ln for ln in fields_section.splitlines() if "y " in ln)
        assert "(optional)" in x_line
        assert "(optional)" in y_line

    def test_wrapped_with_non_dict_inner_properties_falls_back_to_flat(self):
        """If a server emits an object whose ``properties`` is not a dict
        (e.g. ``None`` or a list), the detector returns False and we render
        flat — no `Fields of` block, no crash.

        Defensive against malformed schemas from non-Pydantic servers.
        """
        tool = _fake_tool(
            input_schema={
                "properties": {
                    "req": {"type": "object", "properties": None},
                },
                "required": ["req"],
            }
        )

        out = _format_tool_txt("svr", "tool", tool)
        # The wrapper still appears in the outer Parameters table.
        assert "req " in out
        # But the wrapped path was not triggered.
        assert "Fields of" not in out


# ---------------------------------------------------------------------------
# _build_call_signature — Python signature reconstruction
# ---------------------------------------------------------------------------


class TestBuildCallSignature:
    def test_required_params_appear_before_optional(self):
        """Signature lists required params first, optional after — order matters for LLM read.

        Also asserts the leading ``*, `` kwargs-only marker, which honestly
        advertises the wrapper's actual contract (no positional args allowed).
        """
        properties = {
            "opt_a": {"type": "string"},
            "req_b": {"type": "integer"},
            "opt_c": {"type": "boolean"},
            "req_d": {"type": "string"},
        }
        required = {"req_b", "req_d"}

        sig = _build_call_signature("svr", "fn", properties, required)

        # Required appear in original insertion order before any optionals,
        # and the entire param block is preceded by `*, ` to mark the binding
        # as kwargs-only.
        assert sig == "svr.fn(*, req_b: int, req_d: str, opt_a: str, opt_c: bool)"

    def test_empty_properties_yields_empty_parens(self):
        """No params → `server.fn()` with empty parens; no spurious `*,` marker.

        ``foo(*,)`` is invalid Python — the marker is only meaningful with at
        least one parameter after it.
        """
        sig = _build_call_signature("svr", "fn", {}, set())
        assert sig == "svr.fn()"

    def test_single_param_signature_includes_kwargs_marker(self):
        """Single-param tools also get the `*, ` marker — kwargs-only is universal."""
        sig = _build_call_signature(
            "svr", "fn", {"only": {"type": "string"}}, {"only"}
        )
        assert sig == "svr.fn(*, only: str)"


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


# ---------------------------------------------------------------------------
# is_wrapped_single_param — detects meta-tool / Pydantic-wrapper shapes
# ---------------------------------------------------------------------------


class TestIsWrappedSingleParam:
    def test_empty_properties_is_not_wrapped(self):
        """No parameters at all → not a wrapped tool; falls back to flat render."""
        assert is_wrapped_single_param({}) is False

    def test_single_scalar_property_is_not_wrapped(self):
        """One param whose type is `string` (or any scalar) is just a flat tool."""
        assert is_wrapped_single_param({"x": {"type": "string"}}) is False

    def test_single_array_property_is_not_wrapped(self):
        """A single list-typed param is flat; unwrap rule only fires on objects."""
        assert (
            is_wrapped_single_param({"items": {"type": "array", "items": {}}}) is False
        )

    def test_single_object_without_properties_is_not_wrapped(self):
        """Opaque `dict[str, Any]` blob (no nested `properties`) — nothing to unwrap."""
        assert is_wrapped_single_param({"req": {"type": "object"}}) is False

    def test_single_object_with_empty_properties_is_not_wrapped(self):
        """Object with an explicit but empty properties dict — no fields to surface."""
        assert (
            is_wrapped_single_param({"req": {"type": "object", "properties": {}}})
            is False
        )

    def test_single_object_with_nested_properties_is_wrapped(self):
        """The canonical meta-tool / Pydantic-wrapper shape → True."""
        assert (
            is_wrapped_single_param(
                {
                    "request": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "file_path": {"type": "string"},
                        },
                    }
                }
            )
            is True
        )

    def test_two_top_level_properties_is_not_wrapped(self):
        """Multi-param tools render flat — we only unwrap exactly-one-param shapes."""
        assert (
            is_wrapped_single_param(
                {
                    "a": {"type": "object", "properties": {"x": {"type": "string"}}},
                    "b": {"type": "object", "properties": {"y": {"type": "string"}}},
                }
            )
            is False
        )

    def test_wrapped_with_type_as_list_still_detected(self):
        """JSON-Schema-2020-12 lets ``type`` be a list like ``["object", "null"]``.

        Non-Pydantic MCP servers (Go, Rust, TypeScript) may emit this shape.
        We must still recognize it as wrapped so the LLM gets useful docs.
        """
        assert (
            is_wrapped_single_param(
                {
                    "request": {
                        "type": ["object", "null"],
                        "properties": {"action": {"type": "string"}},
                    }
                }
            )
            is True
        )

    def test_single_non_dict_property_value_is_not_wrapped(self):
        """A malformed schema where the property value isn't even a dict → safe False."""
        assert is_wrapped_single_param({"req": "not a schema"}) is False  # type: ignore[arg-type]
