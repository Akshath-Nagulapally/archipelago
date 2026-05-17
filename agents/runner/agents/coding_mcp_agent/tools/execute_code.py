"""
execute_code.py — execute LLM-generated Python that imports MCP bindings.

After `bindings.build_server_modules()` has run, every MCP server lives in
`sys.modules` under `servers.<name>`. User code can simply::

    from servers import calendar_server
    event = await calendar_server.create_event(...)

and the import resolves through the bindings registered at startup. The
sandbox doesn't inject globals or proxy anything — it just exec's the code
in the agent process.

Threat model: the code is LLM-generated and trusted-but-careless. There is
no AST audit, no isolation, and no resource limit beyond a wall-clock
timeout. Suitable for evaluating agent task completion on a benchmark —
not for running adversarial code.

Each `execute_code` call is independent: no state persists across calls.

Public API:
    execute_code(code, timeout=240.0) -> ExecResult
        Run the code and return captured stdout plus an error string
        (full traceback) if anything went wrong.
"""

from __future__ import annotations

import asyncio
import io
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 240.0


@dataclass
class ExecResult:
    """Outcome of a single `execute_code` call."""

    stdout: str
    error: str | None  # full traceback on failure, None on success
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and not self.timed_out


async def execute_code(
    code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    """Execute LLM-generated Python and return captured stdout + any error.

    The submitted code is wrapped as `async def __user_code__():` so
    top-level `await` works without compile-flag gymnastics. Imports of
    `servers.<name>` resolve through `sys.modules`, which
    `bindings.build_server_modules()` populates at agent startup.

    Args:
        code: Python source to execute. Top-level `await` is permitted.
        timeout: Wall-clock limit in seconds (default: 4 minutes).

    Returns an `ExecResult`. On error, `.error` carries the full traceback
    so the LLM can see the failing line; any stdout printed before the
    failure is still surfaced in `.stdout`.
    """
    wrapped = "async def __user_code__():\n" + textwrap.indent(code, "    ")
    namespace: dict[str, Any] = {}

    try:
        exec(wrapped, namespace)
    except SyntaxError:
        return ExecResult(stdout="", error=traceback.format_exc())

    user_fn = namespace["__user_code__"]
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            await asyncio.wait_for(user_fn(), timeout=timeout)
    except TimeoutError:
        return ExecResult(
            stdout=_combine(stdout_buf, stderr_buf),
            error=f"Execution timed out after {timeout:g}s",
            timed_out=True,
        )
    except Exception:
        return ExecResult(
            stdout=_combine(stdout_buf, stderr_buf),
            error=traceback.format_exc(),
        )

    return ExecResult(stdout=_combine(stdout_buf, stderr_buf), error=None)


def _combine(stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> str:
    """Merge stdout and stderr, labeling stderr so the LLM can tell them apart."""
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    if not err:
        return out
    if out:
        return f"{out.rstrip()}\n[stderr]\n{err}"
    return f"[stderr]\n{err}"
