"""Prompt and runtime helpers for the Code Execution Agent."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from runner.agents.models import LitellmAnyMessage, get_msg_role

DEFAULT_RUNTIME_ROOT = "/filesystem/.code_execution_agent"


def sanitize_trajectory_id(trajectory_id: str) -> str:
    """Convert arbitrary trajectory IDs into safe workspace directory names."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", trajectory_id).strip("._-")
    return safe or "trajectory"


def build_runtime_dir(trajectory_id: str) -> str:
    """Build the per-trajectory runtime directory path."""
    return f"{DEFAULT_RUNTIME_ROOT}/{sanitize_trajectory_id(trajectory_id)}"


def build_mcp_client_template(
    *,
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
) -> str:
    """Generate the task-side Python helper for programmatic MCP access."""
    gateway_server_config: dict[str, Any] = {
        "transport": "streamable-http",
        "url": mcp_gateway_url,
    }
    if mcp_gateway_auth_token:
        gateway_server_config["headers"] = {
            "Authorization": f"Bearer {mcp_gateway_auth_token}"
        }
    gateway_config: dict[str, Any] = {
        "mcpServers": {
            "gateway": gateway_server_config,
        }
    }

    gateway_config_json = json.dumps(gateway_config, indent=4, sort_keys=True)

    return f"""from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Client as FastMCPClient

GATEWAY_CONFIG = {gateway_config_json}


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    return {{
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "inputSchema": (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
        ),
    }}


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {{
        "type": getattr(block, "type", block.__class__.__name__),
    }}
    for attr in ("text", "data", "mimeType", "mime_type", "annotations", "meta"):
        value = getattr(block, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


async def list_tools_async() -> list[dict[str, Any]]:
    async with FastMCPClient(GATEWAY_CONFIG) as client:
        tools = await client.list_tools()
    return [_tool_to_dict(tool) for tool in tools]


async def inspect_tool_async(tool_name: str) -> dict[str, Any]:
    for tool in await list_tools_async():
        if tool["name"] == tool_name:
            return tool
    raise ValueError(f"Unknown MCP tool: {{tool_name}}")


async def call_tool_async(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with FastMCPClient(GATEWAY_CONFIG) as client:
        result = await client.call_tool(tool_name, arguments or {{}})
    return {{
        "data": getattr(result, "data", None),
        "content": [_content_block_to_dict(block) for block in result.content],
        "is_error": getattr(result, "is_error", False),
    }}


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "mcp_client sync helpers cannot run inside an active asyncio loop. "
        "Use the *_async helpers instead."
    )


def list_tools() -> list[dict[str, Any]]:
    return _run(list_tools_async())


def inspect_tool(tool_name: str) -> dict[str, Any]:
    return _run(inspect_tool_async(tool_name))


def call_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _run(call_tool_async(tool_name, arguments))
"""


def build_code_execution_system_prompt(
    *,
    runtime_dir: str,
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
) -> str:
    """Build the system prompt injected by the Code Execution Agent."""
    mcp_client_template = build_mcp_client_template(
        mcp_gateway_url=mcp_gateway_url,
        mcp_gateway_auth_token=mcp_gateway_auth_token,
    )
    return f"""You are a code execution agent operating inside a sandboxed Python workspace.

Your default way of solving tasks is to iteratively write and run Python code in the task workspace. You can still use direct MCP tool calls when that is the fastest path, but for multi-step workflows, loops, filtering, batching, retries, or large intermediate data, prefer Python as the composition layer.

## Runtime Workspace

Use this directory for task-side code and scratch files:

`{runtime_dir}/`

Suggested task-side files:
- `{runtime_dir}/main.py` - your primary working script; rewrite it as often as needed
- `{runtime_dir}/mcp_client.py` - helper for calling MCP tools programmatically from Python

## Core Workflow

1. Use `toolbelt_list_tools` and `toolbelt_inspect_tool` to discover the MCP tools relevant to the task.
2. Use direct MCP tool calls for quick one-off actions or small reads.
3. When the task needs orchestration, use `code_exec` to create/update Python files under `{runtime_dir}`.
4. Use Python code to call MCP tools programmatically, keep intermediate data in variables, and only print concise summaries.
5. When finished, use `final_answer`.

## Important Boundaries

- Treat `/filesystem` as the task workspace.
- Do not treat the agent implementation repo or `/app` as the task workspace unless the user explicitly asks you to modify repository code.
- Keep generated code inside `{runtime_dir}`.
- Avoid printing huge raw datasets. Filter or summarize them in Python first.

## Programmatic MCP Access

If `{runtime_dir}/mcp_client.py` does not exist yet, create it with `code_exec` using the exact contents below:

```python
{mcp_client_template}
```

After that, your task-side Python can do things like:

```python
from mcp_client import call_tool, inspect_tool, list_tools

tools = list_tools()
sheet_schema = inspect_tool("read_tab")
result = call_tool("read_tab", {{"file_path": "/orders.xlsx", "tab_name": "Sheet1"}})
print(result["data"] or result["content"])
```

## Decision Rule

- Prefer direct MCP calls when the task is one or two simple actions.
- Prefer Python code when you need to combine tools, transform larger results, loop, batch, retry, or preserve intermediate state without pushing it all through model context.

Complete the task independently and use the tools available to you.
"""


def inject_code_execution_system_prompt(
    messages: list[LitellmAnyMessage],
    *,
    runtime_dir: str,
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
) -> list[LitellmAnyMessage]:
    """Insert the Code Execution Agent system prompt after any existing system messages."""
    prompt_message: dict[str, str] = {
        "role": "system",
        "content": build_code_execution_system_prompt(
            runtime_dir=runtime_dir,
            mcp_gateway_url=mcp_gateway_url,
            mcp_gateway_auth_token=mcp_gateway_auth_token,
        ),
    }
    typed_prompt_message = cast(LitellmAnyMessage, cast(object, prompt_message))

    insert_at = 0
    while insert_at < len(messages) and get_msg_role(messages[insert_at]) == "system":
        insert_at += 1

    return list(messages[:insert_at]) + [typed_prompt_message] + list(
        messages[insert_at:]
    )
