"""System and user prompt components for the trajectory LLM judge."""

from ..models import TrajectoryFailureType

# ---------------------------------------------------------------------------
# Composable system prompt sections
# ---------------------------------------------------------------------------

_JUDGE_ROLE = (
    "You are an expert web-agent trajectory evaluator. "
    "Evaluate HOW the agent executed its task — not whether it ultimately succeeded. "
    "A separate output judge scores task completion; your job is to score process quality."
)

_EVALUATION_SCOPE = (
    "Read the original task, verification criteria, and step-by-step trajectory. "
    "Score the quality of the agent's execution process: tool usage, grounding in evidence, "
    "error recovery, and path efficiency. Do not score whether the final answer is correct."
)

_RUBRIC = "\n".join([
    "<RUBRIC>",
    "Score each dimension from 1 to 5.",
    "- tool_use_score: Were tools selected correctly with accurate arguments? "
    "5 = precise tool choices, no misuse or hallucinated invocations; "
    "1 = systematic tool misuse or fabricated calls.",
    "- grounding_score: Did each action follow from observed tool outputs, not from assumptions? "
    "5 = every action grounded in prior evidence; "
    "1 = repeated confabulation or acting on unverified assumptions.",
    "- recovery_score: When tools failed or returned unexpected results, did the agent adapt? "
    "5 = clean recovery and appropriate replanning; "
    "1 = errors ignored, spiraling behavior, or premature stopping.",
    "- efficiency_score: Was the path to a solution direct? "
    "5 = minimal purposeful steps; 1 = repeated loops or aimless wandering.",
    "</RUBRIC>",
])

_FAILURE_TYPES = "\n".join([
    "<FAILURE_TYPES>",
    ", ".join(ft.value for ft in TrajectoryFailureType),
    "</FAILURE_TYPES>",
])

_OUTPUT_SCHEMA = "\n".join([
    "<OUTPUT_SCHEMA>",
    "Return ONLY a JSON object with these fields:",
    "- tool_use_score: integer 1-5",
    "- grounding_score: integer 1-5",
    "- recovery_score: integer 1-5",
    "- efficiency_score: integer 1-5",
    "- failure_type: one of the FAILURE_TYPES values; use none when there is no meaningful failure",
    "- failure_step_idx: integer message index from the trajectory, or null if no single step is responsible",
    "- critical_step_idxs: list of integer message indices that materially affected the grade",
    "- rationale: evidence-based explanation citing specific steps by MESSAGE index",
    "Do not include overall_score. The grading system computes it deterministically from the dimension scores.",
    "</OUTPUT_SCHEMA>",
])

TRAJECTORY_GRADING_SYSTEM_PROMPT = "\n\n".join([
    _JUDGE_ROLE,
    _EVALUATION_SCOPE,
    _RUBRIC,
    _FAILURE_TYPES,
    _OUTPUT_SCHEMA,
])


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_trajectory_prompt(
    *,
    task_prompt: str | None,
    final_answer: str,
    trajectory_excerpt: str,
    criteria: str,
    criteria_explanation: str | None = None,
) -> str:
    task_section = f"<ORIGINAL_TASK>\n{task_prompt}\n</ORIGINAL_TASK>\n\n" if task_prompt else ""

    criteria_explanation_section = (
        f"<CRITERIA_EXPLANATION>\n{criteria_explanation}\n</CRITERIA_EXPLANATION>\n\n"
        if criteria_explanation
        else ""
    )

    return (
        f"{task_section}"
        f"<FINAL_ANSWER>\n{final_answer or '(No final answer provided)'}\n</FINAL_ANSWER>\n\n"
        f"<VERIFICATION_CRITERIA>\n{criteria}\n</VERIFICATION_CRITERIA>\n\n"
        f"{criteria_explanation_section}"
        f"<TRAJECTORY>\n{trajectory_excerpt or '(No trajectory history provided)'}\n</TRAJECTORY>\n\n"
        "<REMINDER>\n"
        "- Evaluate execution process quality, not whether the task was completed.\n"
        "- Use MESSAGE index values when identifying failure_step_idx or critical_step_idxs.\n"
        "- Return only JSON matching the schema from the system instructions.\n"
        "</REMINDER>"
    )
