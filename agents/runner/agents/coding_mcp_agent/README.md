# Coding MCP Agent

An agent that exposes MCP tools as **Python functions** so an LLM can call them by writing real code, rather than emitting one tool-call JSON per turn. This implementation is inspired by Anthropic's ["Code execution with MCP: building more efficient agents"](https://www.anthropic.com/engineering/code-execution-with-mcp).

## Motivation

The default way an agent uses MCP is to inject every tool definition into the model's context and have it emit one tool call per turn. This has two well-known problems:

1. **Token cost scales with the tool catalog.** A gateway with 5 servers and 30+ tools spends thousands of tokens on schemas the agent may never use on a given task.
2. **Intermediate data hits the context.** Every tool result — even a 5MB spreadsheet read or a 200-row table — flows through the LLM, even when it's just being passed to the next tool.

The article above argues for a different design: present tools as **importable Python functions**, let the LLM write code (`result = sheets.read_tab(...); total = sum(...)`), and execute that code in a sandbox where intermediate data stays out of context. Token usage drops because the agent only sees the *names* of tools (not full schemas), and chained operations execute server-side without round-tripping data through the model.

## What this PR implements: **bindings only**

This branch lands the foundational piece: **dynamic generation of Python modules that wrap MCP tools.** Once these bindings exist, agent code can do:

```python
from servers import filesystem_server, sheets_server

content = await filesystem_server.read_text_file(path="/data/report.txt")
rows = await sheets_server.sheets(request={"action": "read_tab", "file_name": "..."})
```

The remaining pieces — progressive tool discovery, the LLM-driven agent loop, and the code-execution sandbox — are deferred to a follow-up PR (see [Future Work](#future-work) below).

## How It Works

### 1. Server discovery via `GET /apps`

At startup, the agent calls the gateway's `/apps` endpoint to get the **authoritative list of configured server names** (the exact keys from the active `mcpServers` config). This avoids the brittle alternative of inferring server names from tool-name prefixes — which silently breaks for multi-word names like `filesystem_server`.

### 2. Dynamic module generation

For each server name returned by `/apps`, the agent:

1. Creates a `types.ModuleType("servers.<name>")` at runtime.
2. Filters the gateway's full tool list to that server's tools (stripping the `<server>_` prefix from each tool name).
3. Attaches each tool to the module as an async function that wraps `client.call_tool(tool_name, kwargs)`.
4. Registers the module in `sys.modules` and on a parent `servers` namespace package — so `from servers import filesystem_server` Just Works.

Each wrapped tool function also unwraps fastmcp's `CallToolResult.content[0].text` shape, so the caller gets a plain string back instead of a protocol object.

### 3. Probe-based smoke test

After bindings are built, `binding_test.py` runs one cheap, side-effect-free tool call per server through the binding — this verifies the full path (`GET /apps` → module generation → MCP gateway call → result unwrap) is wired end-to-end. Failures are captured per-server so one misconfigured server doesn't mask the rest.

Adding a new MCP server only requires registering one entry in the `PROBES` dict.

## Future Work

The following pieces are intentionally **not** in this PR and will land in follow-ups:

- **Progressive tool discovery / compact manifest.** Rather than injecting every tool's full schema into the LLM context, surface a one-line-per-tool manifest (`sheets_server.read_tab(file_name, tab_name)`) and let the agent discover tools by reading it. (See the Anthropic article's "Progressive disclosure" section.)
- **The agent loop itself.** This branch's `run()` returns `AgentStatus.ERROR` immediately after the smoke test — there is no LLM step yet. The loop will execute LLM-written Python via the `code_execution_server` in-process.
- **Final-answer / planning tools.** `final_answer` (explicit termination) and `todo_write` (task planning), carried over from the ReAct agent, will be added alongside the loop.
