import json
from core.state import AgentState, ExecutionResult, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are an Executor in an adaptive multi-agent system.

Your job:
- Execute each step of the plan and produce the actual, complete output for every step.
- Surface real blockers only when they genuinely prevent completion.

CRITICAL: For knowledge-based tasks (Q&A, summarization, research, analysis, planning),
the `output` field MUST contain the COMPLETE ACTUAL ANSWER — not a description of what
you did. Write as if you are directly responding to the user's original request.
Example: if asked "What is O(n log n)?", output MUST be "O(n log n)" — not "complexity identified".

Output ONLY valid JSON matching this schema:
{
  "step_results": [
    {
      "step_id": "1",
      "status": "success" | "failure" | "partial",
      "output": "What actually happened or would happen",
      "issues": ["List of problems encountered"]
    }
  ],
  "overall_status": "success" | "failure" | "partial",
  "summary": "One paragraph summary of execution",
  "blockers": ["Critical issues that prevented completion"],
  "agent_confidence": 0.75,
  "reasoning": "One sentence explaining your confidence in this execution assessment"
}

Rules:
- Include a result for every step in the plan.
- overall_status reflects the aggregate: success only if ALL steps succeed.
- Be specific about issues — vague issues cannot be acted on by the Critic.
- Return ONLY the JSON object."""


class ExecutorAgent(BaseAgent):
    name = "executor"
    _output_field = "execution_result"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.current_plan is not None, "Executor requires a plan"

        # Include recent history so the executor knows what was tried before
        recent = state.history[-2:] if state.history else []
        history_summary = []
        for r in recent:
            if r.execution_result:
                history_summary.append({
                    "iteration": r.iteration,
                    "overall_status": r.execution_result.overall_status,
                    "blockers": r.execution_result.blockers,
                })

        return "\n".join([
            f"GOAL: {state.goal.goal}",
            f"ITERATION: {state.iteration}",
            "",
            "=== USER'S ORIGINAL INPUT (the full source material you must work with) ===",
            state.raw_input,
            "",
            "=== PLAN TO EXECUTE ===",
            state.current_plan.model_dump_json(indent=2),
            "",
            "=== RECENT EXECUTION HISTORY (avoid repeating these failures) ===",
            json.dumps(history_summary, indent=2),
            "",
            "Execute the plan now. Return only valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        result = parse_model(ExecutionResult, raw)
        return state.model_copy(update={"execution_result": result})
