"""Unit tests for runner.agents.coding_mcp_agent.task_tracking.

Covers TaskTracker.handle() (the core write/merge logic), the query helpers
get_incomplete() and has_incomplete(), and the static TASK_WRITE_TOOL schema.
No LLM or async I/O is involved — all tests are synchronous.
"""

from __future__ import annotations

import json

from runner.agents.coding_mcp_agent.task_tracking import (
    TASK_WRITE_TOOL,
    TaskStatus,
    TaskTracker,
)

# Round-trip through JSON to strip the TypedDict wrapper so schema tests can
# subscript freely without hitting reportTypedDictNotRequiredAccess errors.
_SCHEMA = json.loads(json.dumps(TASK_WRITE_TOOL))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tracker: TaskTracker, todos: list[dict[str, object]], merge: bool = True):
    raw = json.dumps({"todos": todos, "merge": merge})
    return json.loads(tracker.handle(raw))


# ---------------------------------------------------------------------------
# TestCreate — creating new tasks
# ---------------------------------------------------------------------------


class TestCreate:
    def test_single_task_is_stored(self):
        t = TaskTracker()
        result = _write(
            t, [{"id": "t1", "content": "do something", "status": "pending"}]
        )
        assert result["success"] is True
        assert "t1" in result["created"]
        assert t.tasks["t1"].content == "do something"
        assert t.tasks["t1"].status == TaskStatus.PENDING

    def test_multiple_tasks_created_in_one_call(self):
        t = TaskTracker()
        result = _write(
            t,
            [
                {"id": "a", "content": "first", "status": "pending"},
                {"id": "b", "content": "second", "status": "in_progress"},
            ],
        )
        assert set(result["created"]) == {"a", "b"}
        assert len(t.tasks) == 2

    def test_content_required_for_new_task(self):
        t = TaskTracker()
        result = _write(t, [{"id": "t1", "status": "pending"}])
        assert result["success"] is False
        assert "t1" not in t.tasks
        assert any("Content required" in e for e in result["errors"])

    def test_all_status_values_accepted_on_create(self):
        for status in ("pending", "in_progress", "completed", "cancelled"):
            t = TaskTracker()
            result = _write(t, [{"id": "x", "content": "task", "status": status}])
            assert result["success"] is True
            assert t.tasks["x"].status == TaskStatus(status)


# ---------------------------------------------------------------------------
# TestUpdate — updating existing tasks
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_status_update_on_existing_task(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "pending"}])
        result = _write(t, [{"id": "t1", "status": "completed"}])
        assert "t1" in result["updated"]
        assert t.tasks["t1"].status == TaskStatus.COMPLETED

    def test_content_update_on_existing_task(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "old content", "status": "pending"}])
        _write(t, [{"id": "t1", "content": "new content", "status": "pending"}])
        assert t.tasks["t1"].content == "new content"

    def test_update_without_content_leaves_content_unchanged(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "keep this", "status": "pending"}])
        _write(t, [{"id": "t1", "status": "in_progress"}])
        assert t.tasks["t1"].content == "keep this"

    def test_existing_id_is_update_not_duplicate(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "pending"}])
        result = _write(t, [{"id": "t1", "status": "completed"}])
        assert "t1" in result["updated"]
        assert "t1" not in result["created"]
        assert len(t.tasks) == 1


# ---------------------------------------------------------------------------
# TestMerge — merge=True vs merge=False
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_true_preserves_untouched_tasks(self):
        t = TaskTracker()
        _write(
            t,
            [
                {"id": "a", "content": "keep", "status": "pending"},
                {"id": "b", "content": "update", "status": "pending"},
            ],
        )
        _write(t, [{"id": "b", "status": "completed"}], merge=True)
        assert "a" in t.tasks
        assert t.tasks["b"].status == TaskStatus.COMPLETED

    def test_merge_false_clears_existing_tasks(self):
        t = TaskTracker()
        _write(t, [{"id": "old", "content": "old task", "status": "pending"}])
        _write(
            t, [{"id": "new", "content": "new task", "status": "pending"}], merge=False
        )
        assert "old" not in t.tasks
        assert "new" in t.tasks

    def test_merge_false_with_empty_todos_clears_all(self):
        t = TaskTracker()
        _write(t, [{"id": "x", "content": "task", "status": "pending"}])
        result = _write(t, [], merge=False)
        assert t.tasks == {}
        assert result["tasks"] == []


# ---------------------------------------------------------------------------
# TestIncomplete — get_incomplete and has_incomplete
# ---------------------------------------------------------------------------


class TestIncomplete:
    def test_pending_is_incomplete(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "pending"}])
        assert t.has_incomplete() is True
        assert any(task.id == "t1" for task in t.get_incomplete())

    def test_in_progress_is_incomplete(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "in_progress"}])
        assert t.has_incomplete() is True

    def test_completed_is_not_incomplete(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "completed"}])
        assert t.has_incomplete() is False
        assert t.get_incomplete() == []

    def test_cancelled_is_not_incomplete(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "cancelled"}])
        assert t.has_incomplete() is False

    def test_empty_tracker_has_no_incomplete(self):
        t = TaskTracker()
        assert t.has_incomplete() is False
        assert t.get_incomplete() == []

    def test_mixed_statuses_returns_only_incomplete(self):
        t = TaskTracker()
        _write(
            t,
            [
                {"id": "a", "content": "pending", "status": "pending"},
                {"id": "b", "content": "done", "status": "completed"},
                {"id": "c", "content": "wip", "status": "in_progress"},
                {"id": "d", "content": "dropped", "status": "cancelled"},
            ],
        )
        incomplete_ids = {task.id for task in t.get_incomplete()}
        assert incomplete_ids == {"a", "c"}


# ---------------------------------------------------------------------------
# TestFinalAnswerGate — the gate logic lives in loop.py but the predicate is
# has_incomplete(); we verify the predicate flips at the right moment.
# ---------------------------------------------------------------------------


class TestFinalAnswerGate:
    def test_gate_fires_when_tasks_incomplete(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "not done", "status": "pending"}])
        assert t.has_incomplete() is True  # gate should block

    def test_gate_passes_after_all_completed(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "pending"}])
        _write(t, [{"id": "t1", "status": "completed"}])
        assert t.has_incomplete() is False  # gate should pass

    def test_gate_passes_after_all_cancelled(self):
        t = TaskTracker()
        _write(t, [{"id": "t1", "content": "task", "status": "pending"}])
        _write(t, [{"id": "t1", "status": "cancelled"}])
        assert t.has_incomplete() is False

    def test_gate_passes_with_no_tasks_ever_written(self):
        t = TaskTracker()
        assert t.has_incomplete() is False  # task_write is optional; no tasks = no gate


# ---------------------------------------------------------------------------
# TestValidation — bad inputs should produce errors, not crashes
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_status_string_is_rejected(self):
        t = TaskTracker()
        result = _write(t, [{"id": "t1", "content": "task", "status": "bogus"}])
        assert result["success"] is False
        assert any("Invalid status" in e for e in result["errors"])
        assert "t1" not in t.tasks

    def test_empty_id_is_rejected(self):
        t = TaskTracker()
        result = _write(t, [{"id": "", "content": "task", "status": "pending"}])
        assert result["success"] is False
        assert any("non-empty id" in e for e in result["errors"])

    def test_non_dict_item_in_todos_is_rejected(self):
        t = TaskTracker()
        raw = json.dumps({"todos": ["not a dict"], "merge": True})
        result = json.loads(t.handle(raw))
        assert result["success"] is False
        assert any("must be an object" in e for e in result["errors"])

    def test_todos_not_a_list_returns_error(self):
        t = TaskTracker()
        raw = json.dumps({"todos": "not a list", "merge": True})
        result = json.loads(t.handle(raw))
        assert "error" in result

    def test_partial_errors_do_not_block_valid_items(self):
        t = TaskTracker()
        result = _write(
            t,
            [
                {"id": "good", "content": "valid", "status": "pending"},
                {"id": "", "content": "bad", "status": "pending"},  # empty id
            ],
        )
        assert "good" in t.tasks
        assert result["success"] is False
        assert "good" in result["created"]


# ---------------------------------------------------------------------------
# TestFallbacks — malformed raw input
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_malformed_json_returns_error(self):
        t = TaskTracker()
        result = json.loads(t.handle('{"todos": [broken'))
        assert "error" in result

    def test_empty_string_arguments(self):
        t = TaskTracker()
        result = json.loads(t.handle(""))
        # Empty args → todos defaults to [] → success with no tasks created
        assert result["tasks"] == []

    def test_non_dict_json_returns_error(self):
        t = TaskTracker()
        result = json.loads(t.handle(json.dumps(["not", "a", "dict"])))
        assert "error" in result


# ---------------------------------------------------------------------------
# TestSchema — guard the static TASK_WRITE_TOOL dict against accidental edits
# ---------------------------------------------------------------------------


class TestSchema:
    def test_tool_name_is_task_write(self):
        assert _SCHEMA["function"]["name"] == "task_write"

    def test_todos_and_merge_are_required(self):
        params = _SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"todos", "merge"}

    def test_status_enum_matches_task_status_values(self):
        items = _SCHEMA["function"]["parameters"]["properties"]["todos"]["items"]
        schema_statuses = set(items["properties"]["status"]["enum"])
        code_statuses = {s.value for s in TaskStatus}
        assert schema_statuses == code_statuses

    def test_item_requires_id_and_status(self):
        items = _SCHEMA["function"]["parameters"]["properties"]["todos"]["items"]
        assert set(items["required"]) == {"id", "status"}

    def test_content_is_not_required_at_item_level(self):
        # content is optional to allow status-only updates
        items = _SCHEMA["function"]["parameters"]["properties"]["todos"]["items"]
        assert "content" not in items.get("required", [])
