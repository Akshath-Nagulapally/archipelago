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


def parse_output_schema(tool: Any) -> tuple[dict[str, Any], set[str]]:
    """Extract (properties, required) from a tool's outputSchema.

    Mirror of :func:`parse_input_schema`. Unlike inputSchema, the
    ``outputSchema`` attribute is **optional** on an MCP Tool — FastMCP
    populates it only when the tool function has a Pydantic-typed return
    annotation. Tools that return plain ``str`` (most filesystem tools,
    for example) have no outputSchema at all.

    Returns ``({}, set())`` for tools without an outputSchema, so the
    caller can write the same loop for both shapes without special-casing.
    """
    schema: dict[str, Any] = getattr(tool, "outputSchema", None) or {}
    properties: dict[str, Any] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])
    return properties, required
