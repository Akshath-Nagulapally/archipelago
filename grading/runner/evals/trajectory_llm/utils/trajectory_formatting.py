"""Message and trajectory formatting utilities for the trajectory LLM judge."""

import json
from html import escape
from typing import Any

MAX_SINGLE_MESSAGE_CHARS = 2_000
MAX_TOOL_ARGUMENT_CHARS = 2_000


def _xml_attr(value: Any) -> str:
    return escape(str(value), quote=True)


def _get_message_value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _normalize_content(content: Any) -> str:
    """Collapse LiteLLM text/image blocks into one judge-readable string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _format_tool_arguments(arguments: Any) -> str:
    """Canonical JSON so repeated tool calls are easy to compare in the prompt."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
            rendered = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        except json.JSONDecodeError:
            rendered = stripped
    else:
        rendered = _normalize_content(arguments).strip() or "{}"
    return _truncate_text(rendered, MAX_TOOL_ARGUMENT_CHARS)


def _format_tool_call(tool_call: Any, call_index: int) -> str:
    function = _get_message_value(tool_call, "function", {})
    tool_name = _get_message_value(function, "name", "unknown")
    arguments = _format_tool_arguments(_get_message_value(function, "arguments", "{}"))
    tool_call_id = _get_message_value(tool_call, "id")

    id_attr = f' id="{_xml_attr(tool_call_id)}"' if tool_call_id else ""
    lines = [f'<TOOL_CALL index="{call_index}"{id_attr} name="{_xml_attr(tool_name)}">']
    lines.append("<ARGS>")
    lines.append(arguments)
    lines.append("</ARGS>")
    lines.append("</TOOL_CALL>")
    return "\n".join(lines)


def _format_tool_calls(tool_calls: Any) -> str | None:
    if not tool_calls:
        return None
    formatted = [
        _format_tool_call(tc, idx) for idx, tc in enumerate(tool_calls, start=1)
    ]
    return "\n".join(["<TOOL_CALLS>", *formatted, "</TOOL_CALLS>"]) if formatted else None


def _format_message(message: Any, message_index: int, reverse_index: int) -> str:
    """Attach both original and reverse indices so the judge can cite steps accurately."""
    role = str(_get_message_value(message, "role", "unknown"))
    name = _get_message_value(message, "name")
    tool_call_id = _get_message_value(message, "tool_call_id")
    tool_calls = _get_message_value(message, "tool_calls")
    content = _truncate_text(
        _normalize_content(_get_message_value(message, "content", "")).strip(),
        MAX_SINGLE_MESSAGE_CHARS,
    )

    lines = [
        f'<MESSAGE index="{message_index}" reverse_index="{reverse_index}" role="{_xml_attr(role)}">'
    ]
    if name:
        lines.append(f"<NAME>{name}</NAME>")
    if tool_call_id:
        lines.append(f"<TOOL_CALL_ID>{tool_call_id}</TOOL_CALL_ID>")
    formatted_tool_calls = _format_tool_calls(tool_calls)
    if formatted_tool_calls:
        lines.append(formatted_tool_calls)
    lines.append("<CONTENT>")
    lines.append(content or "(empty)")
    lines.append("</CONTENT>")
    lines.append("</MESSAGE>")
    return "\n".join(lines)


def build_trajectory_excerpt(
    messages: list[Any],
    max_messages: int,
    max_chars: int,
) -> tuple[str, int]:
    """Select a bounded window of messages, preferring the most recent, in chronological order."""
    selected: list[str] = []
    total_chars = 0
    message_count = len(messages)

    for reverse_index, message in enumerate(reversed(messages), start=1):
        if len(selected) >= max_messages:
            break
        message_index = message_count - reverse_index + 1
        formatted = _format_message(message, message_index=message_index, reverse_index=reverse_index)
        next_chars = total_chars + len(formatted)
        if selected and next_chars > max_chars:
            break
        selected.append(formatted)
        total_chars = next_chars

    selected.reverse()
    return "\n\n".join(selected), len(selected)
