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
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from runner.agents.coding_mcp_agent.bindings import get_bound_tools
from runner.agents.coding_mcp_agent.utils import parse_input_schema

_JSON_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


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

        for fn_name, tool in get_bound_tools(mod).items():
            description = (tool.description or "").strip()
            first_line = description.splitlines()[0] if description else ""

            tool_index_lines.append(f"{fn_name}: {first_line}")
            (server_dir / f"{fn_name}.txt").write_text(
                _format_tool_txt(server_name, fn_name, tool)
            )

        (server_dir / "_index.txt").write_text("\n".join(tool_index_lines) + "\n")
        server_index_lines.append(f"{server_name}: {len(tool_index_lines)} tools")

    (servers_dir / "_index.txt").write_text("\n".join(server_index_lines) + "\n")

    return str(root)


def _format_tool_txt(server_name: str, fn_name: str, tool: Any) -> str:
    """Format the full .txt content for a single tool."""
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
        max_name_len = max((len(name) for name in properties), default=0)
        for param_name, param_schema in properties.items():
            py_type = _json_type_to_python(param_schema)
            req_str = "(required)" if param_name in required else "(optional)"
            param_desc = param_schema.get("description", "")
            lines.append(
                f"  {param_name:<{max_name_len}}  {py_type:<6}  {req_str}  {param_desc}".rstrip()
            )

    return "\n".join(lines) + "\n"


def _build_call_signature(
    server_name: str,
    fn_name: str,
    properties: dict[str, Any],
    required: set[str],
) -> str:
    """Reconstruct the Python call signature from parsed inputSchema fields."""
    required_params = [p for p in properties if p in required]
    optional_params = [p for p in properties if p not in required]

    param_strs = [
        f"{name}: {_json_type_to_python(properties[name])}"
        for name in required_params + optional_params
    ]
    return f"{server_name}.{fn_name}({', '.join(param_strs)})"


def _json_type_to_python(param_schema: dict[str, Any]) -> str:
    """Map a JSON schema type to a Python type annotation string."""
    return _JSON_TYPE_MAP.get(param_schema.get("type", ""), "Any")
