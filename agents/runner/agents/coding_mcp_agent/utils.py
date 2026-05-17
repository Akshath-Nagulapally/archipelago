"""Shared utilities for the CodingMCPAgent package."""

from __future__ import annotations

from typing import Any


def parse_input_schema(tool: Any) -> tuple[dict[str, Any], set[str]]:
    """Extract (properties, required) from a tool's inputSchema.

    Defensive against null/missing fields — some MCP servers return
    ``{"required": null}`` which would crash a plain ``set(...)``.
    """
    schema: dict[str, Any] = getattr(tool, "inputSchema", None) or {}
    properties: dict[str, Any] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])
    return properties, required
