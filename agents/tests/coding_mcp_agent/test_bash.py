"""Unit tests for runner.agents.coding_mcp_agent.tools.bash.

These tests run real shell commands as subprocesses — they're hermetic in
that they only touch `tempfile.TemporaryDirectory()` for cwd checks and
otherwise just spawn short-lived commands like `echo`/`printf`/`sleep`.

pytest's auto asyncio mode is set in agents/pyproject.toml, so
`async def test_...` works without `@pytest.mark.asyncio`.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from runner.agents.coding_mcp_agent.tools.bash import (
    DEFAULT_TIMEOUT_SECONDS,
    execute_bash,
)

# ---------------------------------------------------------------------------
# Core behavior — happy path, multi-line stdout, empty output.
# ---------------------------------------------------------------------------


class TestCoreBehavior:
    async def test_simple_echo_returns_stdout(self):
        result = await execute_bash("echo hello")
        assert result.success
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.exit_code == 0

    async def test_multi_line_output(self):
        result = await execute_bash("printf 'one\\ntwo\\nthree\\n'")
        assert result.success
        assert result.stdout == "one\ntwo\nthree\n"

    async def test_command_with_no_output_succeeds(self):
        result = await execute_bash("true")
        assert result.success
        assert result.stdout == ""
        assert result.stderr == ""


# ---------------------------------------------------------------------------
# Exit codes & stderr — non-zero exits, stderr captured separately.
# ---------------------------------------------------------------------------


class TestExitCodesAndStderr:
    async def test_non_zero_exit_marks_failure_but_captures_output(self):
        """`grep` with no match returns exit 1 — `success=False`, but the
        command itself didn't error and we still capture whatever it wrote."""
        result = await execute_bash("ls /this_path_does_not_exist_xyz_123")
        assert not result.success
        assert result.exit_code != 0
        assert "No such file" in result.stderr or "not found" in result.stderr.lower()

    async def test_stderr_kept_separate_from_stdout(self):
        result = await execute_bash("echo to-out; echo to-err >&2")
        assert result.success
        assert "to-out" in result.stdout
        assert "to-err" in result.stderr
        # Critical: streams must not be merged.
        assert "to-err" not in result.stdout
        assert "to-out" not in result.stderr

    async def test_explicit_exit_code_is_reported(self):
        result = await execute_bash("exit 42")
        assert not result.success
        assert result.exit_code == 42


# ---------------------------------------------------------------------------
# Shell features — pipes, redirects, working directory.
# ---------------------------------------------------------------------------


class TestShellFeatures:
    async def test_pipe_works(self):
        """`sh -c` means the LLM gets full shell — pipes, redirects, etc."""
        result = await execute_bash("printf 'a\\nb\\nc\\n' | grep b")
        assert result.success
        assert result.stdout == "b\n"

    async def test_cwd_is_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = await execute_bash("pwd", cwd=tmp)
            assert result.success
            # macOS may resolve /tmp → /private/tmp; compare with realpath-ish check.
            assert tmp in result.stdout or result.stdout.strip().endswith(
                tmp.split("/")[-1]
            )

    async def test_environment_variables_are_inherited(self):
        """The subprocess inherits the agent's env vars by default — useful
        for the LLM to access PATH, HOME, etc."""
        result = await execute_bash("echo PATH=$PATH")
        assert result.success
        assert "PATH=" in result.stdout
        # Some real entry should be there — the agent's PATH is non-empty.
        assert "/" in result.stdout


# ---------------------------------------------------------------------------
# Timeout — kills runaway commands.
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_long_running_command_is_killed(self):
        result = await execute_bash("sleep 5", timeout=0.2)
        assert not result.success
        assert result.timed_out
        assert result.exit_code == -1

    def test_default_timeout_is_sixty_seconds(self):
        """The 60s default is the contract the agent loop relies on."""
        assert DEFAULT_TIMEOUT_SECONDS == 60.0


# ---------------------------------------------------------------------------
# Non-interactive — stdin is /dev/null so any read attempt EOFs immediately
# instead of hanging until the wall-clock timeout. This is the LLM-safety
# fix: there's no human at the keyboard, so prompts must fail fast.
# ---------------------------------------------------------------------------


class TestNonInteractive:
    async def test_bash_read_returns_immediately_on_eof(self):
        """A bash `read` with no input on stdin should hit EOF immediately.
        If stdin weren't redirected, this would block until the timeout."""
        # The whole command should complete well under the 5s timeout; if
        # stdin were inherited from a terminal-less parent, `read` might
        # still block. /dev/null guarantees EOF on first read.
        result = await execute_bash(
            "read varname; echo got=$varname",
            timeout=5.0,
        )
        assert not result.timed_out
        # `read` returns non-zero on EOF, but the subsequent echo still runs.
        assert "got=" in result.stdout

    async def test_python_input_raises_eof_error_not_hang(self):
        """Python `input()` with no stdin raises EOFError — fast failure
        rather than hanging until timeout."""
        result = await execute_bash(
            "python3 -c \"input('prompt: ')\"",
            timeout=5.0,
        )
        assert not result.timed_out
        # Python prints EOFError to stderr and exits non-zero.
        assert not result.success
        assert "EOFError" in result.stderr

    async def test_cat_with_no_args_returns_immediately(self):
        """`cat` with no args reads from stdin until EOF — with /dev/null
        attached, that's immediate. Without the fix, it would block until
        the wall-clock timeout."""
        result = await execute_bash("cat", timeout=5.0)
        assert not result.timed_out
        assert result.success  # cat reads EOF and exits 0
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# Interactivity hint — when stderr reveals "I tried to read input I couldn't
# get", we attach a help string so the LLM is told to retry non-interactively
# rather than guessing why the command failed.
# ---------------------------------------------------------------------------


class TestInteractivityHint:
    async def test_hint_set_on_python_input_eof_error(self):
        """Python `input()` with closed stdin → EOFError in stderr → hint set."""
        result = await execute_bash(
            "python3 -c \"input('> ')\"",
            timeout=5.0,
        )
        assert not result.success
        assert result.hint is not None
        assert "interactive" in result.hint.lower()
        assert "non-interactive" in result.hint.lower()

    async def test_hint_set_for_sudo_password_prompt(self):
        """`sudo -n true` is safe: the `-n` flag means non-interactive, so
        sudo refuses to prompt and exits with `a password is required`. No
        privilege change ever happens, no command is run with elevation —
        this just verifies our hint fires on that exact error path.

        Skip if sudo isn't installed (some minimal CI images don't have it).
        Also skip if the test user has NOPASSWD configured, since then
        `sudo -n true` succeeds and there is no password-required error to
        trigger the hint on."""
        if shutil.which("sudo") is None:
            pytest.skip("sudo not installed")
        result = await execute_bash("sudo -n true", timeout=5.0)
        if result.success:
            pytest.skip(
                "sudo is configured NOPASSWD for this user — can't trigger the prompt path"
            )
        # Sudo with -n and no cached creds: exit 1 and a password-required message.
        assert not result.success
        assert result.hint is not None
        assert "non-interactive" in result.hint.lower()

    async def test_hint_none_for_ordinary_command_error(self):
        """A plain non-zero exit (file not found) should NOT trigger the
        hint — it isn't an interactivity problem and we don't want to
        misadvise the LLM."""
        result = await execute_bash("ls /this_path_does_not_exist_xyz_123")
        assert not result.success
        assert result.hint is None

    async def test_hint_none_for_permission_denied(self):
        """Permission-denied on a root-owned file is a permission issue,
        not an interactivity issue. The hint must stay None."""
        # /etc/shadow exists and is root-readable on Linux/macOS — the
        # non-root test user gets EACCES, not an EOF/tty error.
        result = await execute_bash("cat /etc/shadow")
        assert not result.success
        assert result.hint is None

    async def test_hint_none_on_successful_run(self):
        result = await execute_bash("echo ok")
        assert result.success
        assert result.hint is None
