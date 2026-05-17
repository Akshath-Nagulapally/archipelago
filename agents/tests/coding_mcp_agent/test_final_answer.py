"""Unit tests for runner.agents.coding_mcp_agent.tools.final_answer.

`parse_final_answer` is the only behavior worth testing here — the schema
is a static dict, the rest of the module is the schema declaration. We
cover the LLM-call shape, every status the schema permits, and the
fallback paths that protect against malformed JSON from the model.
"""

from __future__ import annotations

import json

from runner.agents.coding_mcp_agent.tools.final_answer import (
    FINAL_ANSWER_TOOL,
    parse_final_answer,
)

# ---------------------------------------------------------------------------
# Happy path — well-formed JSON, valid statuses.
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_well_formed_arguments_round_trip(self):
        args = json.dumps({"answer": "the answer is 42", "status": "completed"})
        answer, status = parse_final_answer(args)
        assert answer == "the answer is 42"
        assert status == "completed"

    def test_blocked_status_is_preserved(self):
        args = json.dumps({"answer": "missing API key", "status": "blocked"})
        _, status = parse_final_answer(args)
        assert status == "blocked"

    def test_failed_status_is_preserved(self):
        args = json.dumps({"answer": "impossible task", "status": "failed"})
        _, status = parse_final_answer(args)
        assert status == "failed"

    def test_empty_answer_string_is_preserved(self):
        """An empty answer is a real signal from the LLM — don't drop it."""
        args = json.dumps({"answer": "", "status": "completed"})
        answer, status = parse_final_answer(args)
        assert answer == ""
        assert status == "completed"


# ---------------------------------------------------------------------------
# Fallback paths — the comment in parse_final_answer says we'd rather keep
# whatever the LLM wrote than drop the run. Pin that behavior.
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_empty_arguments_returns_defaults(self):
        """No arguments at all → empty answer, default status."""
        answer, status = parse_final_answer("")
        assert answer == ""
        assert status == "completed"

    def test_malformed_json_returns_raw_string(self):
        """LLM produced a stray quote → keep the raw text, mark completed."""
        raw = '{"answer": "broken'
        answer, status = parse_final_answer(raw)
        assert answer == raw
        assert status == "completed"

    def test_non_dict_json_returns_string_form(self):
        """LLM emitted a JSON list/string/number instead of an object."""
        answer, status = parse_final_answer(json.dumps(["just", "a", "list"]))
        assert answer == "['just', 'a', 'list']"
        assert status == "completed"

    def test_missing_answer_field_yields_empty_string(self):
        args = json.dumps({"status": "completed"})
        answer, status = parse_final_answer(args)
        assert answer == ""
        assert status == "completed"

    def test_missing_status_field_defaults_to_completed(self):
        args = json.dumps({"answer": "done"})
        _, status = parse_final_answer(args)
        assert status == "completed"

    def test_non_string_answer_is_coerced_to_string(self):
        """LLM emitted answer as a number — don't crash, stringify."""
        args = json.dumps({"answer": 42, "status": "completed"})
        answer, _ = parse_final_answer(args)
        assert answer == "42"


# ---------------------------------------------------------------------------
# Schema sanity — guard against accidental edits to the tool declaration
# that would silently change the LLM contract.
# ---------------------------------------------------------------------------


class TestSchema:
    def test_tool_name_is_final_answer(self):
        assert FINAL_ANSWER_TOOL["function"]["name"] == "final_answer"

    def test_both_answer_and_status_are_required(self):
        params = FINAL_ANSWER_TOOL["function"]["parameters"]
        assert set(params["required"]) == {"answer", "status"}

    def test_status_enum_matches_parse_default(self):
        """The 'completed' fallback in parse_final_answer must be one of
        the enum values, or the LLM contract is internally inconsistent."""
        params = FINAL_ANSWER_TOOL["function"]["parameters"]
        statuses = params["properties"]["status"]["enum"]
        assert "completed" in statuses
        assert "blocked" in statuses
        assert "failed" in statuses
