"""
task_tracking.py — task planning and tracking for CodingMCPAgent.

Mirrors the todo system in react_toolbelt_agent/tools.py but scoped to
CodingMCPAgent's needs: no toolbelt meta-tools, just task write/update
and a mandatory completion gate on final_answer.

The LLM calls `task_write` to create and update tasks. `final_answer` is
rejected in-band if any tasks are still pending or in_progress — the LLM
must close (complete or cancel) all tasks before terminating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    content: str
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "content": self.content, "status": self.status.value}


TASK_WRITE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "task_write",
        "description": (
            "Create or update your task list. Use this to plan multi-step work and "
            "track progress. All tasks must be completed or cancelled before "
            "final_answer will be accepted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Array of task items to write.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the task.",
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Description of the task. Required when creating "
                                    "a new task; optional when updating an existing one."
                                ),
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": "Task status.",
                            },
                        },
                        "required": ["id", "status"],
                    },
                },
                "merge": {
                    "type": "boolean",
                    "description": (
                        "If true, merge with existing tasks (update matching IDs, "
                        "add new ones). If false, replace all tasks with the provided list."
                    ),
                },
            },
            "required": ["todos", "merge"],
        },
    },
}


class TaskTracker:
    """Owns the task list for one agent run.

    Instantiated once in AgentLoop.__init__ and shared across all steps.
    All state mutation goes through handle(); callers never touch self.tasks
    directly.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def handle(self, arguments: str) -> str:
        """Parse a task_write tool call and return a JSON result string."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in arguments"})

        if not isinstance(args, dict):
            return json.dumps({"error": "Arguments must be a JSON object"})

        todos_input = args.get("todos", [])
        merge = args.get("merge", True)

        if not isinstance(todos_input, list):
            return json.dumps({"error": "todos must be an array"})

        if not merge:
            self.tasks.clear()

        created_ids: list[str] = []
        updated_ids: list[str] = []
        errors: list[str] = []

        for item in todos_input:
            if not isinstance(item, dict):
                errors.append("Each task must be an object")
                continue

            task_id = str(item.get("id", "")).strip()
            status_str = str(item.get("status", "pending"))
            content = item.get("content")

            if not task_id:
                errors.append("Each task must have a non-empty id")
                continue

            try:
                status = TaskStatus(status_str)
            except ValueError:
                errors.append(f"Invalid status '{status_str}' for task '{task_id}'")
                continue

            if task_id in self.tasks:
                self.tasks[task_id].status = status
                if content is not None:
                    self.tasks[task_id].content = str(content)
                updated_ids.append(task_id)
            else:
                if content is None:
                    errors.append(f"Content required for new task '{task_id}'")
                    continue
                self.tasks[task_id] = Task(
                    id=task_id, content=str(content), status=status
                )
                created_ids.append(task_id)

        if created_ids:
            logger.bind(message_type="tool").info(
                f"Created tasks: {', '.join(created_ids)}"
            )
        if updated_ids:
            logger.bind(message_type="tool").info(
                f"Updated tasks: {', '.join(updated_ids)}"
            )

        all_tasks = [t.to_dict() for t in self.tasks.values()]
        response: dict[str, object] = {
            "success": len(errors) == 0,
            "created": created_ids,
            "updated": updated_ids,
            "tasks": all_tasks,
            "summary": {
                "total": len(all_tasks),
                "pending": sum(1 for t in all_tasks if t["status"] == "pending"),
                "in_progress": sum(
                    1 for t in all_tasks if t["status"] == "in_progress"
                ),
                "completed": sum(1 for t in all_tasks if t["status"] == "completed"),
                "cancelled": sum(1 for t in all_tasks if t["status"] == "cancelled"),
            },
        }
        if errors:
            response["errors"] = errors
        return json.dumps(response)

    def get_incomplete(self) -> list[Task]:
        """Return tasks that are still pending or in_progress."""
        return [
            t
            for t in self.tasks.values()
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        ]

    def has_incomplete(self) -> bool:
        return bool(self.get_incomplete())
