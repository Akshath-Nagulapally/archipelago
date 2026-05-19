"""
Generate a filesystem-based knowledge tree for tool discovery.

Writes a directory structure that an LLM agent can navigate via ls/cat
to discover available MCP tools and their call signatures without loading
all tool definitions upfront.

Structure:
    servers/
        _index.txt                  <- one-liner per server
        <server_name>/
            _index.txt              <- one-liner per tool in this server
            <tool_name>.txt         <- full signature + parameter docs

Signature rendering contract
----------------------------

The signature on the first line of each ``<tool_name>.txt`` is emitted with
a leading ``*, `` to mark every binding as kwargs-only. This matches the
actual contract of ``bindings._make_tool_caller``, which is a
``**kwargs``-only wrapper by design. Without the marker the signature would
read as positional-or-keyword in standard Python syntax and an LLM that
calls ``tool({...})`` positionally would hit a TypeError before the wrapper
ever runs.

Wrapped-param tools (meta-tools and Pydantic-wrapped individual tools)
----------------------------------------------------------------------

Two real-world server patterns produce a schema with a single top-level
property that is itself a structured object:

1. Meta-tools — one MCP tool routes many operations via an ``action`` field
   in its single Pydantic input model (e.g. ``sheets_server.sheets``).
2. Individual tools written with a wrapped Pydantic input — one function
   parameter of type ``SomeInputModel`` (e.g. ``read_tab(input: ReadTabInput)``).

For both shapes, ``_format_tool_txt`` recurses ONE level into the wrapper
and renders the inner fields under a ``Fields of `<wrapper>`:`` section.
The outer ``Parameters:`` line still appears, so the LLM sees both the
honest call shape (``tool(*, request: dict)``) and the contents that go
inside the wrapper dict.

Detection of this shape is delegated to ``is_wrapped_single_param`` and
deliberately bails out (returns False, falls back to flat rendering) on
any structural anomaly — multiple top-level properties, ``$ref``-only
inner schemas, opaque ``dict[str, Any]`` blobs without nested
``properties``, etc. The asymmetry is intentional: every false-negative
preserves today's behavior; only a very specific fingerprint enables
unwrapping.

Output schema rendering (``Returns:`` block)
--------------------------------------------

When ``tool.outputSchema`` is populated (FastMCP emits this whenever the
tool function has a Pydantic-typed return annotation), we render a
``Returns:`` section beneath the input section using the same parameter
table machinery — including the one-level unwrap rule. Tools that return
plain ``str`` have no outputSchema and the section is omitted entirely;
their docs render exactly as they did before this code existed.

The motivating failure: an LLM that calls a meta-tool like
``sheets_server.sheets(request={'action': 'read_tab', ...})`` receives
a JSON-serialized ``SheetsOutput`` and has no docs telling it about
fields like ``read_tab.raw_output``. Without that hint, the model is
known to mis-parse the response (e.g. running ``ast.literal_eval`` on
an ASCII-table string) and fall back to reading numbers off the
printout — breaking the architectural promise that intermediate data
stays in Python locals.

Out of scope (deferred to later enhancements):
    - ``$ref`` resolution against schemas that arrive un-flattened
    - ``oneOf`` / ``anyOf`` traversal at the wrapper root
    - Enum value rendering (``Literal["a", "b"]`` → ``(one of: a, b)``)
    - Per-action discriminated-union output rendering (showing only the
      relevant result field per action)
    - Second-level expansion for output-side nested objects (would
      explode doc size on tools like ``SheetsOutput`` with 13 result
      sub-types; defer until evidence justifies the bloat)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from runner.agents.coding_mcp_agent.bindings import get_bound_tools
from runner.agents.coding_mcp_agent.utils import (
    parse_input_schema,
    parse_output_schema,
)

_JSON_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def is_wrapped_single_param(properties: dict[str, Any]) -> bool:
    """Detect the "one parameter that's a Pydantic-style nested object" shape.

    Two real-world patterns produce this shape:

    1. **Meta-tools** like ``sheets_server.sheets(request: SheetsInput)`` —
       one function, one Pydantic-typed parameter that holds the entire
       action-routed interface inside.
    2. **Individual tools written with a wrapped Pydantic input**, e.g.
       ``read_tab(input: ReadTabInput)`` — still one parameter, still a
       nested object, even though it isn't a meta-tool.

    In both cases the *real* parameter info lives one level deeper than the
    flat-tool case (``read_text_file(file_path: str, ...)``), and our docs
    renderer needs to know to look inside.

    Detection rule — all three must hold:

    1. Exactly one top-level property.
    2. That property's ``type`` is ``"object"`` (or a JSON-Schema-2020-12
       type-array that contains ``"object"``, e.g. ``["object", "null"]``).
    3. That property has its own non-empty ``properties`` dict — i.e. it's
       a structured object, not an opaque ``dict[str, Any]`` blob.

    Any other shape returns False and the caller falls back to the existing
    flat rendering path. This asymmetry is intentional: a False answer is
    always safe (it preserves today's behavior); a True answer requires a
    very specific structural fingerprint.

    Note: this detector deliberately does NOT recurse. If the inner schema
    is itself a wrapped single-object, we still report True for the outer
    layer only — the renderer unwraps one level and stops. This guards
    against runaway expansion on cyclic / deeply-nested schemas.
    """
    if len(properties) != 1:
        return False

    sole = next(iter(properties.values()))
    if not isinstance(sole, dict):
        return False

    type_field = sole.get("type")
    is_object = type_field == "object" or (
        isinstance(type_field, list) and "object" in type_field
    )
    if not is_object:
        return False

    inner_properties = sole.get("properties")
    return isinstance(inner_properties, dict) and len(inner_properties) > 0


def build_tool_docs_dir(modules: dict[str, Any]) -> str:
    """Build the tool discovery filesystem and return the path to the root dir.

    Args:
        modules: The {server_name: module} dict returned by build_server_modules().
                 Each module has async functions with _mcp_tool attached.

    Returns:
        Absolute path to the generated directory root.
    """
    root = Path("/tmp/mcp-tool-docs")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    servers_dir = root / "servers"
    servers_dir.mkdir()

    server_index_lines: list[str] = []

    for server_name, mod in modules.items():
        server_dir = servers_dir / server_name
        server_dir.mkdir()

        tool_index_lines: list[str] = []
        bound_tools = get_bound_tools(mod)
        # Set membership check is O(1); used to detect <fn>_schema companions
        # so meta-tools can point the LLM at their introspection sibling.
        tool_names_in_server = set(bound_tools.keys())

        for fn_name, tool in bound_tools.items():
            description = (tool.description or "").strip()
            first_line = description.splitlines()[0] if description else ""

            tool_index_lines.append(f"{fn_name}: {first_line}")

            companion_name = f"{fn_name}_schema"
            schema_companion = (
                f"{server_name}.{companion_name}"
                if companion_name in tool_names_in_server
                else None
            )

            (server_dir / f"{fn_name}.txt").write_text(
                _format_tool_txt(
                    server_name, fn_name, tool, schema_companion=schema_companion
                )
            )

        (server_dir / "_index.txt").write_text("\n".join(tool_index_lines) + "\n")
        server_index_lines.append(f"{server_name}: {len(tool_index_lines)} tools")

    (servers_dir / "_index.txt").write_text("\n".join(server_index_lines) + "\n")

    return str(root)


def _format_tool_txt(
    server_name: str,
    fn_name: str,
    tool: Any,
    schema_companion: str | None = None,
) -> str:
    """Format the full .txt content for a single tool.

    The output always contains the call signature, the tool description, and
    a ``Parameters:`` table listing the top-level inputSchema properties.

    For meta-tool / Pydantic-wrapped tools (see ``is_wrapped_single_param``),
    we additionally emit a ``Fields of `<wrapper>`:`` block listing the
    fields nested one level inside the wrapper. We keep the outer
    ``Parameters:`` line as well — it's the honest picture of the call shape
    (``await tool(*, request={...})``), and the inner block tells the LLM
    what goes *inside* that dict. Unwrapping stops at one level by design;
    deeper nesting still renders as ``dict``.

    The ``schema_companion`` parameter, when set, is the qualified name of
    a sibling tool (e.g. ``"sheets_server.sheets_schema"``) that exposes
    deeper schema introspection. It gets forwarded to
    ``_format_output_section``, which decides whether emitting a pointer
    line is worth it for this particular tool.
    """
    properties, required = parse_input_schema(tool)

    lines: list[str] = []
    lines.append(_build_call_signature(server_name, fn_name, properties, required))
    lines.append("")

    description = (tool.description or "").strip()
    if description:
        lines.append(description)
        lines.append("")

    if properties:
        lines.append("Parameters:")
        _emit_param_table(lines, properties, required)

        if is_wrapped_single_param(properties):
            wrapper_name = next(iter(properties))
            inner_schema = properties[wrapper_name]
            inner_properties: dict[str, Any] = inner_schema["properties"]
            inner_required: set[str] = set(inner_schema.get("required") or [])

            lines.append("")
            lines.append(f"Fields of `{wrapper_name}`:")
            _emit_param_table(lines, inner_properties, inner_required)

    output_lines = _format_output_section(tool, schema_companion=schema_companion)
    if output_lines:
        # Blank line between Parameters and Returns sections (or after the
        # description if there were no Parameters at all).
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(output_lines)

    return "\n".join(lines) + "\n"


def _emit_param_table(
    lines: list[str], properties: dict[str, Any], required: set[str]
) -> None:
    """Append a parameter table to ``lines`` for the given properties.

    Factored out of ``_format_tool_txt`` so the wrapped-tool case can render
    the same shape twice (once for the wrapper, once for its inner fields)
    without duplicating the column-alignment logic.
    """
    if not properties:
        return
    max_name_len = max((len(name) for name in properties), default=0)
    for param_name, param_schema in properties.items():
        py_type = _json_type_to_python(param_schema)
        req_str = "(required)" if param_name in required else "(optional)"
        param_desc = param_schema.get("description", "")
        lines.append(
            f"  {param_name:<{max_name_len}}  {py_type:<6}  {req_str}  {param_desc}".rstrip()
        )


def _format_output_section(
    tool: Any, schema_companion: str | None = None
) -> list[str]:
    """Render the ``Returns:`` block for a tool, if its outputSchema has any.

    MCP tools may or may not publish an output schema — FastMCP populates
    ``tool.outputSchema`` only when the tool function has a Pydantic-typed
    return annotation. Tools that return plain ``str`` produce no output
    schema, and this function returns an empty list so the caller appends
    nothing (preserving the historical doc format for those tools).

    When a schema *is* present, we render it the same way the input
    section renders: a ``Returns:`` header followed by a parameter table,
    plus a one-level unwrap for the meta-tool single-wrapper shape (see
    ``is_wrapped_single_param``). The unwrap rule matters most here in
    practice — discriminated-union outputs like ``SheetsOutput`` won't
    trigger it (multiple top-level fields), but single-wrapped result
    types like ``{"raw_output": {"type": "string", ...}}`` will.

    The header line itself reads::

        Returns: str  (JSON-serialized — call json.loads() first)

    rather than just ``Returns:``. The binding wrapper (``_make_tool_caller``)
    always returns the raw MCP tool result as a JSON-encoded string, not a
    Python dict. Without this hint, an LLM that sees ``action: str
    (required)`` in the table naturally writes ``res["action"]`` — and gets
    ``TypeError: string indices must be integers, not 'str'`` because ``res``
    is still a string at that point. The note costs one line and prevents a
    one-turn recovery loop.

    Args:
        tool: The MCP ``Tool`` whose outputSchema is rendered. Required.
        schema_companion: Optional qualified name (e.g.
            ``"sheets_server.sheets_schema"``) of a sibling tool that
            exposes deeper schema introspection (the meta-tool convention
            of a ``<name>_schema`` companion). When provided AND we've
            emitted a ``Returns:`` block, we append a single-line pointer
            so the LLM knows where to look for nested per-action schemas
            without having to call the tool itself first. We deliberately
            do NOT emit the hint on tools with no outputSchema — a bare
            "see the schema tool" hanging in an otherwise-empty section
            adds noise without context.

    Returns an empty list when the schema yields nothing to render —
    keeps the caller's append site uncluttered.
    """
    properties, required = parse_output_schema(tool)
    if not properties:
        return []

    lines: list[str] = [
        "Returns: str  (JSON-serialized — call json.loads() first)"
    ]
    _emit_param_table(lines, properties, required)

    if is_wrapped_single_param(properties):
        wrapper_name = next(iter(properties))
        inner_schema = properties[wrapper_name]
        inner_properties: dict[str, Any] = inner_schema["properties"]
        inner_required: set[str] = set(inner_schema.get("required") or [])

        lines.append("")
        lines.append(f"Fields of `{wrapper_name}`:")
        _emit_param_table(lines, inner_properties, inner_required)

    if schema_companion:
        lines.append("")
        lines.append(
            f"For the full nested output schema, see `{schema_companion}`."
        )

    return lines


def _build_call_signature(
    server_name: str,
    fn_name: str,
    properties: dict[str, Any],
    required: set[str],
) -> str:
    """Reconstruct the Python call signature from parsed inputSchema fields.

    Every binding wrapper produced by ``bindings._make_tool_caller`` is a
    ``**kwargs``-only async function — it accepts zero positional arguments
    by design (see that function's docstring for the rationale). We emit a
    leading ``*, `` separator so the rendered signature honestly advertises
    that contract:

        sheets_server.sheets(*, request: dict)

    Without the ``*, ``, the signature reads as positional-or-keyword in
    standard Python syntax, and the LLM may write ``sheets({...})`` which
    fails with ``TypeError: _call() takes 0 positional arguments but 1 was
    given``. The marker is only emitted when there is at least one parameter
    after it — ``foo(*,)`` is invalid Python.
    """
    required_params = [p for p in properties if p in required]
    optional_params = [p for p in properties if p not in required]

    param_strs = [
        f"{name}: {_json_type_to_python(properties[name])}"
        for name in required_params + optional_params
    ]
    if param_strs:
        return f"{server_name}.{fn_name}(*, {', '.join(param_strs)})"
    return f"{server_name}.{fn_name}()"


def _json_type_to_python(param_schema: dict[str, Any]) -> str:
    """Map a JSON schema type to a Python type annotation string."""
    return _JSON_TYPE_MAP.get(param_schema.get("type", ""), "Any")
