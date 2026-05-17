"""
bash.py — execute shell commands for the CodingMCPAgent.

The agent exposes two tools to the LLM: this one (`execute_bash`) and
`sandbox.execute_code`. The bash tool is the LLM's way to *navigate* — the
canonical use is exploring `/tmp/mcp-tool-docs` to discover which MCP
servers exist and which tools each one offers::

    cat /tmp/mcp-tool-docs/servers/_index.txt
    cat /tmp/mcp-tool-docs/servers/sheets_server/_index.txt
    cat /tmp/mcp-tool-docs/servers/sheets_server/add_row.txt

But the LLM can do anything bash does — `ls`, `find`, `grep`, `head`,
`pwd`, `env`, etc. There is no allowlist and no path restriction by
design: the agent runs in a container, the LLM is trusted-but-careless
(not adversarial), and the container boundary is the actual isolation
layer.

Public API:
    execute_bash(command, timeout=60.0, cwd=None) -> BashResult
        Run `command` through `/bin/sh -c`, capture stdout/stderr, and
        return the exit code. On timeout the process is killed and
        `timed_out=True` is reported.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 60.0

# Stderr substrings that indicate a command tried to read from a closed
# stdin or a missing TTY. When any of these appear, we attach a `hint`
# to the result so the LLM is explicitly told to re-run with a
# non-interactive form rather than guessing why its command failed.
_INTERACTIVITY_MARKERS: tuple[str, ...] = (
    "EOFError",  # python input() / similar
    "stdin: is not a tty",  # sudo / ssh / common
    "no tty present",  # sudo specifically
    "a password is required",  # sudo -S / no -n
    "could not read password",  # various tools
    "unable to read passphrase",  # gpg / ssh-keygen
    "no controlling terminal",  # tools needing a tty for prompts
)

_INTERACTIVITY_HINT = (
    "This command appears to require interactive input "
    "(stdin is closed; no human is at the keyboard). Re-run with a "
    "non-interactive form: pass values as arguments, use flags like "
    "`-y` / `--yes` / `--no-input`, or pipe input via a heredoc."
)


def _interactivity_hint(stderr: str) -> str | None:
    """Return a help string if stderr looks like a closed-stdin failure."""
    if not stderr:
        return None
    for marker in _INTERACTIVITY_MARKERS:
        if marker in stderr:
            return _INTERACTIVITY_HINT
    return None


@dataclass
class BashResult:
    """Outcome of a single `execute_bash` call."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    # Set when the command failed in a way that looks like it needed
    # interactive input (e.g. sudo password prompt, Python input(), etc.).
    # The agent can surface this to the LLM so it knows to retry with a
    # non-interactive flag instead of just seeing a generic stderr.
    hint: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


async def execute_bash(
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: str | None = None,
) -> BashResult:
    """Run a shell command and return its captured output.

    The command is executed via `/bin/sh -c` so pipes, redirects, and
    other shell features work naturally. Bytes from stdout/stderr are
    decoded as UTF-8 with `errors="replace"` so a stray binary blob in
    the output doesn't crash the agent.

    Args:
        command: Shell command line (e.g. `"cat /tmp/.../servers/_index.txt"`).
        timeout: Wall-clock limit in seconds (default: 60s).
        cwd: Optional working directory. None inherits the agent process's cwd.

    Returns a `BashResult`. On timeout the process is killed, partial
    stdout/stderr captured up to that point is returned, and
    `timed_out=True` / `exit_code=-1`.
    """
    # stdin → /dev/null so interactive prompts (e.g. `apt install` without `-y`,
    # bash `read`, Python `input()`) fail fast with EOF instead of hanging until
    # the wall-clock timeout. The LLM has no way to provide input anyway.
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        # Drain whatever the process produced before we killed it so the
        # LLM can still see partial output for debugging.
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except Exception:
            stdout_bytes, stderr_bytes = b"", b""
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        return BashResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_text,
            exit_code=-1,
            timed_out=True,
            hint=_interactivity_hint(stderr_text),
        )

    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    return BashResult(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_text,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        hint=_interactivity_hint(stderr_text),
    )
