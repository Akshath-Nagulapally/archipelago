import io
import json
from types import SimpleNamespace

from runner.evals.models import EvalConfig, EvalIds, EvalImplInput
from runner.evals.registry import EVAL_REGISTRY
from runner.helpers.models import HelperIds
from runner.models import (
    AgentStatus,
    AgentTrajectoryOutput,
    GradingSettings,
    Verifier,
)

from runner.evals.trajectory_llm.main import trajectory_llm_eval


def _build_input(
    *,
    criteria: str = "States that the final answer says compliance was met",
    eval_config_values: dict | None = None,
    messages: list[dict] | None = None,
    final_answer: str = "The notice complied with both Acts.",
) -> EvalImplInput:
    trajectory_messages = messages or [
        {"role": "user", "content": "Review WARN Act compliance."},
        {"role": "assistant", "content": "I will inspect the notices."},
        {"role": "tool", "name": "read_file", "content": "William Ito notice dated 60 days in advance."},
        {"role": "assistant", "content": final_answer},
    ]

    return EvalImplInput(
        initial_snapshot_bytes=io.BytesIO(),
        final_snapshot_bytes=io.BytesIO(),
        trajectory=AgentTrajectoryOutput(
            messages=trajectory_messages,
            status=AgentStatus.COMPLETED,
            time_elapsed=1.5,
            output=None,
        ),
        grading_settings=GradingSettings(llm_judge_model="openai/gpt-4o-mini"),
        verifier=Verifier(
            verifier_id="ver_123",
            verifier_version=1,
            world_id=None,
            task_id="task_warn",
            eval_config_id="ec_trajectory_llm",
            verifier_values={
                "criteria": criteria,
                "is_primary_objective": True,
            },
            verifier_index=0,
            verifier_dependencies=None,
        ),
        eval_config=EvalConfig(
            eval_config_id="ec_trajectory_llm",
            eval_config_name="Trajectory LLM",
            eval_defn_id=EvalIds.TRAJECTORY_LLM,
            eval_config_values=eval_config_values or {},
        ),
        dependencies=None,
        helper_results={
            HelperIds.FINAL_ANSWER: final_answer,
        },
    )


def test_trajectory_eval_registered() -> None:
    eval_defn = EVAL_REGISTRY[EvalIds.TRAJECTORY_LLM]

    assert eval_defn.eval_id == EvalIds.TRAJECTORY_LLM
    assert eval_defn.helper_dependencies == [HelperIds.FINAL_ANSWER]
    assert eval_defn.eval_impl is not None


async def test_trajectory_eval_uses_recent_messages_and_returns_pass(
    monkeypatch,
) -> None:
    captured = {}
    input_data = _build_input(
        eval_config_values={"trajectory_max_messages": 2, "trajectory_max_chars": 10_000},
        messages=[
            {"role": "user", "content": "Old task context that should be omitted from recent slice."},
            {"role": "assistant", "content": "Older assistant turn."},
            {"role": "tool", "name": "search_law", "content": "Federal WARN Act requires 60 days notice."},
            {"role": "assistant", "content": "The notice complied with both Acts."},
        ],
    )

    async def fake_call_llm(*, model, messages, timeout, extra_args=None, response_format=None):
        captured["messages"] = messages
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "rationale": "The final answer explicitly says the notice complied.",
                                "is_criteria_true": True,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr("runner.evals.trajectory_llm.main.call_llm", fake_call_llm)
    monkeypatch.setattr("runner.evals.trajectory_llm.main.Choices", object)

    result = await trajectory_llm_eval(input_data)

    assert result.score == 1.0
    assert result.verifier_result_values["judge_grade"] == "pass"
    assert result.verifier_result_values["evaluated_message_count"] == 2

    user_prompt = captured["messages"][1]["content"]
    assert "Federal WARN Act requires 60 days notice." in user_prompt
    assert "The notice complied with both Acts." in user_prompt
    assert "Older assistant turn." not in user_prompt


async def test_trajectory_eval_returns_fail_when_judge_fails(monkeypatch) -> None:
    input_data = _build_input(
        final_answer="The answer does not mention California WARN compliance."
    )

    async def fake_call_llm(*, model, messages, timeout, extra_args=None, response_format=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "rationale": "The final answer does not support the criterion.",
                                "is_criteria_true": False,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr("runner.evals.trajectory_llm.main.call_llm", fake_call_llm)
    monkeypatch.setattr("runner.evals.trajectory_llm.main.Choices", object)

    result = await trajectory_llm_eval(input_data)

    assert result.score == 0.0
    assert result.verifier_result_values == {
        "judge_grade": "fail",
        "grade_rationale": "The final answer does not support the criterion.",
        "evaluated_message_count": 4,
    }
