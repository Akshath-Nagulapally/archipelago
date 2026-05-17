# Coding MCP Agent

An agent that exposes MCP tools as **Python functions** so an LLM can call them by writing real code, rather than emitting one tool-call JSON per turn. This implementation is inspired by Anthropic's ["Code execution with MCP: building more efficient agents"](https://www.anthropic.com/engineering/code-execution-with-mcp).

## Motivation

The default way an agent uses MCP is to inject every tool definition into the model's context and have it emit one tool call per turn. This has two well-known problems:

1. **Token cost scales with the tool catalog.** A gateway with 5 servers and 30+ tools spends thousands of tokens on schemas the agent may never use on a given task.
2. **Intermediate data hits the context.** Every tool result — even a 5MB spreadsheet read or a 200-row table — flows through the LLM, even when it's just being passed to the next tool.

The article above argues for a different design: present tools as **importable Python functions**, let the LLM write code (`result = sheets.read_tab(...); total = sum(...)`), and execute that code in a sandbox where intermediate data stays out of context. Token usage drops because the agent only sees the *names* of tools (not full schemas), and chained operations execute server-side without round-tripping data through the model.

## What this PR implements: **bindings + progressive tool discovery**

This branch lands three foundational pieces:

1. **Dynamic generation of Python modules that wrap MCP tools** — so agent code can call tools as regular Python functions.
2. **Progressive tool discovery** — a filesystem-based knowledge tree the agent can browse incrementally instead of loading all tool schemas upfront.
3. **Sandbox and Bash tools** — `execute_code` runs Python that imports those bindings; `execute_bash` lets the LLM explore the discovery tree (and run any shell command in the container).

Once these are in place, agent code can do:

```python
from servers import filesystem_server, sheets_server

content = await filesystem_server.read_text_file(path="/data/report.txt")
rows = await sheets_server.sheets(request={"action": "read_tab", "file_name": "..."})
```

The remaining piece — the LLM-driven agent loop that calls these tools — is deferred to a follow-up PR (see [Future Work](#future-work) below).

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

### 4. Sandbox and Bash tools

The agent exposes two **local** tools to the LLM (implemented in `tools/bash.py` and `tools/sandbox.py`). Together they split the Anthropic article's workflow: bash to *discover* what MCP tools exist, sandbox to *call* them.

**`execute_bash`** — run arbitrary shell commands via `/bin/sh -c` (pipes, redirects, etc. work as usual). The canonical use is reading the tool-discovery tree from §3:

```
execute_bash("cat /tmp/mcp-tool-docs/servers/_index.txt")
execute_bash("cat /tmp/mcp-tool-docs/servers/sheets_server/add_row.txt")
```

There is no command allowlist or path restriction: the LLM is trusted-but-careless and the container is the isolation boundary. Stdin is wired to `/dev/null` so interactive prompts fail fast instead of hanging until the 60s timeout; when stderr looks like a closed-stdin failure, the result includes a `hint` nudging the model toward non-interactive flags. Returns `stdout`, `stderr`, `exit_code`, and optional `timed_out`.

**`execute_code`** — run LLM-authored Python in the agent process. Code is wrapped as `async def __user_code__():` so top-level `await` works. Imports like `from servers import sheets_server` resolve through `sys.modules`, populated by the bindings layer at startup — the sandbox does not inject globals or proxy MCP calls. Each call is independent (no variables or functions persist across calls). Returns captured `stdout` (stderr is merged and labeled) and a full traceback on failure. Default timeout is 240s.

Typical turn sequence once the agent loop lands:

1. `execute_bash` — browse `/tmp/mcp-tool-docs` to learn server/tool names and signatures.
2. `execute_code` — write Python that imports `servers.<name>` and chains MCP calls; only the printed result returns to the LLM context.

Threat model for both tools: LLM-generated, not adversarial. No AST audit, no separate process isolation beyond the eval container.

### 5. Probe-based smoke test

After bindings and docs are built, `probes.py` runs three startup checks before the LLM loop is allowed to start (any failure raises and aborts initialization):

1. **Remote MCP** — one cheap, side-effect-free tool call per server through the gateway (probe picked from `inputSchema`: zero-required-args tool first, then a `*_schema` introspection tool as fallback).
2. **Sandbox imports** — `execute_code` can `from servers import <every-server>`.
3. **Bash docs** — `execute_bash` can `cat` the root `_index.txt` of the tool-discovery tree.

Adding a new MCP server requires **no changes** to this file.

## Future Work

The following pieces are intentionally **not** in this PR and will land in follow-ups:

- **The agent loop itself.** This branch's `run()` returns `AgentStatus.ERROR` immediately after the smoke test — there is no LLM step yet. The loop will drive `execute_bash` / `execute_code` each turn.
- **Final-answer / planning tools.** `final_answer` (explicit termination) and `todo_write` (task planning), carried over from the ReAct agent, will be added alongside the loop.
