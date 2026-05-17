# Coding MCP Agent

An agent that exposes MCP tools as **Python functions** so an LLM can call them by writing real code, rather than emitting one tool-call JSON per turn. This implementation is inspired by Anthropic's ["Code execution with MCP: building more efficient agents"](https://www.anthropic.com/engineering/code-execution-with-mcp).

## Motivation

The default way an agent uses MCP is to inject every tool definition into the model's context and have it emit one tool call per turn. This has two well-known problems:

1. **Token cost scales with the tool catalog.** A gateway with 5 servers and 30+ tools spends thousands of tokens on schemas the agent may never use on a given task.
2. **Intermediate data hits the context.** Every tool result — even a 5MB spreadsheet read or a 200-row table — flows through the LLM, even when it's just being passed to the next tool.

The article above argues for a different design: present tools as **importable Python functions**, let the LLM write code (`result = sheets.read_tab(...); total = sum(...)`), and execute that code in a sandbox where intermediate data stays out of context. Token usage drops because the agent only sees the *names* of tools (not full schemas), and chained operations execute server-side without round-tripping data through the model.

## What this PR implements: **bindings + progressive tool discovery**

This branch lands two foundational pieces:

1. **Dynamic generation of Python modules that wrap MCP tools** — so agent code can call tools as regular Python functions.
2. **Progressive tool discovery** — a filesystem-based knowledge tree the agent can browse incrementally instead of loading all tool schemas upfront.

Once these are in place, agent code can do:

```python
from servers import filesystem_server, sheets_server

content = await filesystem_server.read_text_file(path="/data/report.txt")
rows = await sheets_server.sheets(request={"action": "read_tab", "file_name": "..."})
```

The remaining piece — the LLM-driven agent loop and code-execution sandbox — is deferred to a follow-up PR (see [Future Work](#future-work) below).

## How It Works

### 1. Server discovery via `GET /apps`

At startup, the agent calls the gateway's `/apps` endpoint to get the **authoritative list of configured server names** (the exact keys from the active `mcpServers` config). This avoids the brittle alternative of inferring server names from tool-name prefixes — which silently breaks for multi-word names like `filesystem_server`.

### 2. Dynamic module generation

For each server name returned by `/apps`, the agent:

1. Creates a `types.ModuleType("servers.<name>")` at runtime.
2. Filters the gateway's full tool list to that server's tools (stripping the `<server>_` prefix from each tool name).
3. Attaches each tool to the module as an async function that wraps `client.call_tool(tool_name, kwargs)`.
4. Registers the module in `sys.modules` and on a parent `servers` namespace package — so `from servers import filesystem_server` Just Works.

Each wrapped tool function has a `_mcp_tool` attribute holding the original `mcp.types.Tool` object — used by the discovery layer to extract parameter schemas without a second gateway round-trip.

Each wrapped function also unwraps fastmcp's `CallToolResult.content[0].text` shape, so the caller gets a plain string back instead of a protocol object.

### 3. Progressive tool discovery (`tool_discovery_docs.py`)

After bindings are built, `build_tool_docs_dir` writes a static filesystem tree to `/tmp/mcp-tool-docs`:

```
mcp-tool-docs/
  servers/
    _index.txt                  ← one line per server (name + tool count)
    <server_name>/
        _index.txt              ← one line per tool (name + first description line)
        <tool_name>.txt         ← full signature + parameter table
```

This lets the LLM agent navigate the tool catalog using a shell MCP's `ls`/`cat` without injecting every schema into context. The typical discovery flow:

```
cat /tmp/mcp-tool-docs/servers/_index.txt        # which servers exist?
cat /tmp/mcp-tool-docs/servers/sheets_server/_index.txt   # which tools?
cat /tmp/mcp-tool-docs/servers/sheets_server/add_row.txt  # full signature
```

Each `<tool_name>.txt` contains the reconstructed Python call signature (with required params listed before optional ones), a description, and a parameter table showing type and required/optional status:

```
sheets_server.add_row(sheet_id: str, values: list, tab_name: str = ...)

Add a row to the given tab.

Parameters:
  sheet_id  str   (required)  The spreadsheet ID
  values    list  (required)  Row values
  tab_name  str   (optional)  Defaults to first tab
```

The directory is recreated fresh on each agent startup. Because server and tool names must be valid Python identifiers (they are used as module attributes and `sys.modules` keys in the bindings layer), no path sanitization is needed.

### 4. Probe-based smoke test

After bindings and docs are built, `runtime_probes.py` runs one cheap, side-effect-free tool call per server — verifying the full path (`GET /apps` → module generation → MCP gateway call → result unwrap) end-to-end. The probe to call is picked automatically by inspecting each tool's `inputSchema` (zero-required-args tool first, then a `*_schema` introspection tool as fallback). Failures are captured per-server so one misconfigured server doesn't mask the rest.

Adding a new MCP server requires **no changes** to this file.

## Future Work

The following pieces are intentionally **not** in this PR and will land in follow-ups:

- **The agent loop itself.** This branch's `run()` returns `AgentStatus.ERROR` immediately after the smoke test — there is no LLM step yet. The loop will execute LLM-written Python via the `code_execution_server` in-process.
- **Final-answer / planning tools.** `final_answer` (explicit termination) and `todo_write` (task planning), carried over from the ReAct agent, will be added alongside the loop.
