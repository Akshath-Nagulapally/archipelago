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
    _format_output_section,
    _format_tool_txt,
    _json_type_to_python,
    build_tool_docs_dir,
    is_wrapped_single_param,
)
from runner.agents.coding_mcp_agent.utils import parse_output_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_tool(
    description: str | None = "",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Stand-in for an mcp.types.Tool — description + inputSchema + outputSchema.

    ``outputSchema`` is genuinely optional on real MCP tools (FastMCP only
    emits it for Pydantic-typed return annotations). To faithfully model
    that, we leave the attribute *absent* on the namespace when the caller
    doesn't pass one — instead of setting it to ``{}`` — so consumers see
    the same ``getattr(tool, "outputSchema", None) is None`` shape they
    would in production.
    """
    kwargs: dict[str, Any] = {
        "description": description,
        "inputSchema": input_schema if input_schema is not None else {},
    }
    if output_schema is not None:
        kwargs["outputSchema"] = output_schema
    return SimpleNamespace(**kwargs)


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


# ---------------------------------------------------------------------------
# parse_output_schema — utils helper for reading tool.outputSchema
# ---------------------------------------------------------------------------


class TestParseOutputSchema:
    def test_parse_output_schema_returns_properties_and_required(self):
        """Tool with a populated outputSchema → properties dict + required set."""
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "result": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["result"],
            }
        )
        properties, required = parse_output_schema(tool)
        assert set(properties.keys()) == {"result", "count"}
        assert required == {"result"}

    def test_parse_output_schema_missing_returns_empty(self):
        """Tool with no outputSchema attribute → empty dict + empty set, no crash.

        This is the common case: FastMCP only populates outputSchema when the
        tool function has a Pydantic-typed return annotation. Plain ``-> str``
        tools have no schema and the helper must not blow up.
        """
        tool = _fake_tool()  # no output_schema kwarg → attribute absent
        properties, required = parse_output_schema(tool)
        assert properties == {}
        assert required == set()


# ---------------------------------------------------------------------------
# _format_output_section — the new Returns: block renderer
# ---------------------------------------------------------------------------


class TestFormatOutputSection:
    def test_no_output_schema_returns_empty_list(self):
        """Tool without outputSchema → empty list; caller appends nothing."""
        tool = _fake_tool()
        assert _format_output_section(tool) == []

    def test_empty_output_schema_returns_empty_list(self):
        """outputSchema present but with no properties → empty list, no spurious header."""
        tool = _fake_tool(output_schema={"type": "object"})
        assert _format_output_section(tool) == []

    def test_basic_output_schema_emits_returns_header_and_table(self):
        """Flat output schema → starts with ``Returns:`` header, then param rows."""
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "status": {"type": "string", "description": "Operation status."},
                    "count": {"type": "integer", "description": "Items processed."},
                },
                "required": ["status"],
            }
        )
        lines = _format_output_section(tool)
        assert lines[0].startswith("Returns:")
        # Each field gets an indented row.
        assert any("status" in ln and "str" in ln for ln in lines[1:])
        assert any("count" in ln and "int" in ln for ln in lines[1:])

    def test_returns_header_includes_json_serialization_note(self):
        """Header must say 'json.loads()' so the LLM knows to deserialize first.

        Regression guard for the friction seen in the excel_sum_search trajectory:
        the agent called ``sheets_server.sheets(request={...})`` and then wrote
        ``res['read_tab']`` — treating the result as a dict — and got
        ``TypeError: string indices must be integers, not 'str'`` because the
        binding wrapper always returns a JSON-encoded string. The header line
        prevents this one-turn recovery loop.
        """
        tool = _fake_tool(
            output_schema={
                "properties": {"action": {"type": "string"}},
                "required": ["action"],
            }
        )
        lines = _format_output_section(tool)
        assert "json.loads()" in lines[0]

    def test_returns_header_names_str_return_type(self):
        """Header must advertise ``str`` as the actual Python return type."""
        tool = _fake_tool(
            output_schema={
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        )
        lines = _format_output_section(tool)
        # e.g. "Returns: str  (JSON-serialized — call json.loads() first)"
        assert "str" in lines[0]

    def test_output_required_optional_markers_from_output_required_array(self):
        """Required / optional flags come from the OUTPUT schema's required array.

        Guards against accidentally reusing the input schema's required when
        rendering the output block.
        """
        tool = _fake_tool(
            input_schema={
                "properties": {"x": {"type": "string"}},
                "required": ["x"],  # input-side required must NOT bleed into output
            },
            output_schema={
                "properties": {
                    "always_there": {"type": "string"},
                    "maybe_there": {"type": "string"},
                },
                "required": ["always_there"],
            },
        )
        lines = _format_output_section(tool)
        always_line = next(ln for ln in lines if "always_there" in ln)
        maybe_line = next(ln for ln in lines if "maybe_there" in ln)
        assert "(required)" in always_line
        assert "(optional)" in maybe_line

    def test_output_descriptions_preserved(self):
        """Field descriptions from the output schema appear verbatim in the table."""
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "raw_output": {
                        "type": "string",
                        "description": "Formatted table output.",
                    }
                },
                "required": ["raw_output"],
            }
        )
        text = "\n".join(_format_output_section(tool))
        assert "Formatted table output." in text

    def test_wrapped_output_schema_unwraps_one_level(self):
        """Output schema matching the single-wrapped-param shape → emits
        ``Fields of `<wrapper>`:`` block beneath ``Returns:``.

        This is the practical case that motivated the whole feature —
        single-result-type outputs like ``{"result": {"raw_output": ...}}``
        should surface the inner structure so the LLM doesn't have to call
        the tool just to discover what ``raw_output`` is.
        """
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "result": {
                        "type": "object",
                        "properties": {
                            "raw_output": {
                                "type": "string",
                                "description": "Formatted table output.",
                            },
                            "row_count": {"type": "integer"},
                        },
                        "required": ["raw_output"],
                    }
                },
                "required": ["result"],
            }
        )
        text = "\n".join(_format_output_section(tool))
        assert "Returns:" in text
        assert "Fields of `result`:" in text
        # Inner fields appear under the Fields header.
        fields_section = text.split("Fields of `result`:", 1)[1]
        assert "raw_output" in fields_section
        assert "row_count" in fields_section

    def test_wrapped_output_only_unwraps_one_level(self):
        """Doubly-wrapped output schemas stop at depth 1 — same safety rail as input."""
        tool = _fake_tool(
            output_schema={
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
        text = "\n".join(_format_output_section(tool))
        assert "Fields of `outer`:" in text
        # We do NOT recurse a second time.
        assert "deep_field" not in text
        assert text.count("Fields of") == 1


# ---------------------------------------------------------------------------
# _format_tool_txt — integration with the new output section
# ---------------------------------------------------------------------------


class TestFormatToolTxtWithOutputSchema:
    def test_format_tool_txt_appends_returns_block_when_outputschema_present(self):
        """Tool with both inputSchema and outputSchema → output contains both blocks."""
        tool = _fake_tool(
            description="A tool with a structured return.",
            input_schema={
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
            output_schema={
                "properties": {
                    "status": {"type": "string"},
                    "size": {"type": "integer"},
                },
                "required": ["status"],
            },
        )
        out = _format_tool_txt("svr", "tool", tool)
        assert "Parameters:" in out
        assert "Returns:" in out
        # Output-section fields visible.
        assert "status" in out
        assert "size" in out

    def test_format_tool_txt_no_returns_block_when_outputschema_missing(self):
        """No outputSchema → doc renders exactly as it did before this code existed.

        Regression guard for plain ``-> str``-returning tools like the
        filesystem server's read_text_file. We must not emit a stray
        ``Returns:`` header when there's nothing to render.
        """
        tool = _fake_tool(
            description="A flat tool with no structured return.",
            input_schema={
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            # no output_schema kwarg
        )
        out = _format_tool_txt("svr", "list_files", tool)
        assert "Parameters:" in out
        assert "Returns:" not in out

    def test_format_tool_txt_section_order_parameters_before_returns(self):
        """``Parameters:`` always precedes ``Returns:`` in the rendered doc.

        Stable ordering matters because the LLM is scanning the doc top-down;
        it makes a better mental model when inputs come first.
        """
        tool = _fake_tool(
            input_schema={"properties": {"x": {"type": "string"}}, "required": ["x"]},
            output_schema={
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
        )
        out = _format_tool_txt("svr", "tool", tool)
        params_idx = out.find("Parameters:")
        returns_idx = out.find("Returns:")
        assert params_idx != -1 and returns_idx != -1
        assert params_idx < returns_idx


# ---------------------------------------------------------------------------
# _format_output_section — robustness on malformed output schemas
# ---------------------------------------------------------------------------


class TestFormatOutputSectionRobustness:
    def test_outputschema_with_null_required_does_not_crash(self):
        """Output schema with ``"required": null`` → all fields rendered as optional."""
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "integer"},
                },
                "required": None,
            }
        )
        text = "\n".join(_format_output_section(tool))
        # Both fields appear, both as optional.
        x_line = next(ln for ln in text.splitlines() if "x " in ln)
        y_line = next(ln for ln in text.splitlines() if "y " in ln)
        assert "(optional)" in x_line
        assert "(optional)" in y_line

    def test_outputschema_with_non_dict_inner_properties_falls_back_safely(self):
        """Wrapped-shape detection bails out on malformed inner ``properties``.

        We still render the outer ``Returns:`` block (the wrapper field is
        valid as a top-level entry), but we do NOT emit a `Fields of` block.
        """
        tool = _fake_tool(
            output_schema={
                "properties": {
                    "result": {"type": "object", "properties": None},
                },
                "required": ["result"],
            }
        )
        text = "\n".join(_format_output_section(tool))
        assert "Returns:" in text
        # The wrapper field is in the outer table.
        assert "result" in text
        # But the wrapped path didn't trigger.
        assert "Fields of" not in text


# ---------------------------------------------------------------------------
# build_tool_docs_dir — end-to-end coverage for output schema rendering
# ---------------------------------------------------------------------------


class TestBuildToolDocsDirWithOutputSchema:
    def test_outputschema_with_wrapped_shape_lands_in_doc_file(self):
        """End-to-end: a fake meta-tool style schema produces a .txt on disk
        containing both wrapped-input AND wrapped-output blocks.

        This is the bytes-on-disk version of the unit tests above — the LLM
        will `cat /tmp/mcp-tool-docs/.../tool.txt` to see this content.
        """
        tool = _fake_tool(
            description="A meta-tool with structured input and output.",
            input_schema={
                "properties": {
                    "request": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "file_path": {"type": "string"},
                        },
                        "required": ["action"],
                    }
                },
                "required": ["request"],
            },
            output_schema={
                "properties": {
                    "result": {
                        "type": "object",
                        "properties": {
                            "raw_output": {"type": "string"},
                        },
                        "required": ["raw_output"],
                    }
                },
                "required": ["result"],
            },
        )
        mod = _fake_module("svr", {"tool": tool})

        path = Path(build_tool_docs_dir({"svr": mod}))
        doc = (path / "servers" / "svr" / "tool.txt").read_text()

        # Input section (already validated by PR #12 tests, sanity-checked here).
        assert "svr.tool(*, request: dict)" in doc
        assert "Fields of `request`:" in doc
        # Output section — the new behavior.
        assert "Returns:" in doc
        assert "Fields of `result`:" in doc
        assert "raw_output" in doc


# ---------------------------------------------------------------------------
# Schema companion hint — points the LLM at <fn>_schema introspection tools
# ---------------------------------------------------------------------------


class TestSchemaCompanionHint:
    def test_returns_block_includes_companion_hint_when_companion_set(self):
        """Output schema present + companion qualified name passed → hint appears
        as a trailing line in the Returns block, separated by a blank line.
        """
        tool = _fake_tool(
            output_schema={
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        )
        lines = _format_output_section(
            tool, schema_companion="sheets_server.sheets_schema"
        )
        text = "\n".join(lines)
        # The hint line is the last meaningful content in the section.
        assert (
            "For the full nested output schema, see `sheets_server.sheets_schema`."
            in text
        )

    def test_no_companion_means_no_hint_line(self):
        """Without a companion, the Returns block ends after the param table —
        no stray "see ..." line."""
        tool = _fake_tool(
            output_schema={
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        )
        lines = _format_output_section(tool, schema_companion=None)
        text = "\n".join(lines)
        assert "For the full nested output schema" not in text

    def test_no_outputschema_means_no_hint_even_with_companion(self):
        """A tool without an outputSchema gets no Returns block, and we do NOT
        emit a standalone "see schema tool" line for it.

        Rationale: a bare hint line in an otherwise-empty section adds noise
        without context. The hint is meaningful only when augmenting a
        rendered Returns block.
        """
        tool = _fake_tool()  # no output_schema
        lines = _format_output_section(
            tool, schema_companion="sheets_server.sheets_schema"
        )
        assert lines == []

    def test_e2e_companion_detected_in_build_tool_docs_dir(self):
        """End-to-end: registering both `sheets` and `sheets_schema` in the
        same fake module → the `sheets` doc on disk contains the companion
        hint, and the `sheets_schema` doc does NOT contain a self-referential
        hint (no `sheets_schema_schema` exists in the module).
        """
        sheets_tool = _fake_tool(
            description="A meta-tool.",
            output_schema={
                "properties": {"action": {"type": "string"}},
                "required": ["action"],
            },
        )
        sheets_schema_tool = _fake_tool(
            description="Schema introspection for the meta-tool.",
            output_schema={
                "properties": {"model": {"type": "string"}},
                "required": ["model"],
            },
        )
        mod = _fake_module(
            "sheets_server",
            {"sheets": sheets_tool, "sheets_schema": sheets_schema_tool},
        )

        path = Path(build_tool_docs_dir({"sheets_server": mod}))
        sheets_doc = (path / "servers" / "sheets_server" / "sheets.txt").read_text()
        schema_doc = (
            path / "servers" / "sheets_server" / "sheets_schema.txt"
        ).read_text()

        # The main meta-tool's doc gets the pointer to its companion.
        assert "sheets_server.sheets_schema" in sheets_doc
        assert "For the full nested output schema" in sheets_doc
        # The companion's own doc must NOT reference a non-existent
        # `sheets_schema_schema` — the convention check correctly skips it.
        assert "sheets_schema_schema" not in schema_doc
        assert "For the full nested output schema" not in schema_doc
