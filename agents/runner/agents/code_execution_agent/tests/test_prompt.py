from typing import cast

from runner.agents.code_execution_agent.prompt import (
    build_code_execution_system_prompt,
    build_mcp_client_template,
    build_runtime_dir,
    inject_code_execution_system_prompt,
    sanitize_trajectory_id,
)
from runner.agents.models import LitellmAnyMessage, get_msg_content
from runner.agents.registry import get_agent_defn


def test_sanitize_trajectory_id_replaces_unsafe_characters() -> None:
    assert sanitize_trajectory_id("task/with spaces?") == "task_with_spaces"


def test_build_runtime_dir_uses_sanitized_trajectory_id() -> None:
    assert (
        build_runtime_dir("task/with spaces?")
        == "/filesystem/.code_execution_agent/task_with_spaces"
    )


def test_mcp_client_template_embeds_gateway_configuration() -> None:
    template = build_mcp_client_template(
        mcp_gateway_url="http://localhost:8080/mcp/",
        mcp_gateway_auth_token="secret-token",
    )

    assert '"url": "http://localhost:8080/mcp/"' in template
    assert '"Authorization": "Bearer secret-token"' in template
    assert "def call_tool(" in template
    assert "async def call_tool_async(" in template


def test_system_prompt_describes_runtime_contract() -> None:
    prompt = build_code_execution_system_prompt(
        runtime_dir="/filesystem/.code_execution_agent/task_1",
        mcp_gateway_url="http://localhost:8080/mcp/",
        mcp_gateway_auth_token=None,
    )

    assert "/filesystem/.code_execution_agent/task_1/main.py" in prompt
    assert "/filesystem/.code_execution_agent/task_1/mcp_client.py" in prompt
    assert "toolbelt_list_tools" in prompt
    assert "code_exec" in prompt
    assert "from mcp_client import call_tool, inspect_tool, list_tools" in prompt


def test_inject_code_execution_system_prompt_inserts_after_existing_system_messages() -> None:
    messages = cast(
        list[LitellmAnyMessage],
        [
            {"role": "system", "content": "existing system"},
            {"role": "user", "content": "task"},
        ],
    )

    injected = inject_code_execution_system_prompt(
        messages,
        runtime_dir="/filesystem/.code_execution_agent/task_1",
        mcp_gateway_url="http://localhost:8080/mcp/",
        mcp_gateway_auth_token=None,
    )

    assert get_msg_content(injected[0]) == "existing system"
    assert injected[1]["role"] == "system"
    assert "code execution agent operating inside a sandboxed Python workspace" in str(
        get_msg_content(injected[1])
    )
    assert get_msg_content(injected[2]) == "task"


def test_registry_exposes_code_execution_agent() -> None:
    defn = get_agent_defn("code_execution_agent")
    assert defn.agent_config_id == "code_execution_agent"
