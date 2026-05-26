import json
from core.state import AgentState, Plan, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are a strategic Planner in an adaptive multi-agent system.

Your job:
- Decompose the goal into concrete, ordered steps.
- Explicitly AVOID any failed strategies listed in memory.
- Incorporate lessons from previous iterations.
- If this is a replan, explain what changed from the previous plan.

Output ONLY valid JSON matching this schema exactly:
{
  "steps": [
    {"id": "1", "description": "...", "dependencies": [], "expected_outcome": "..."}
  ],
  "rationale": "Why this plan will succeed",
  "estimated_complexity": 5,
  "avoids": ["list of failed strategies this plan explicitly sidesteps"],
  "agent_confidence": 0.85,
  "reasoning": "One sentence explaining your confidence in this plan"
}

Rules:
- steps must be ordered (earlier ids complete before later ones).
- dependencies is a list of step ids that must complete first.
- estimated_complexity is 1-10.
- Be specific — vague steps cannot be executed or critiqued.
- Return ONLY the JSON object. No markdown, no explanation."""


class PlannerAgent(BaseAgent):
    name = "planner"
    _output_field = "current_plan"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        is_replanning = (
            state.control_signal.action == "replan" and state.current_plan is not None
        )
        parts = [
            f"GOAL: {state.goal.goal}",
            f"SUBGOALS: {json.dumps(state.goal.subgoals)}",
            f"ITERATION: {state.iteration}",
            "",
            "=== MEMORY ===",
            f"Failed strategies to AVOID:\n{json.dumps(state.memory.failed_strategies, indent=2)}",
            f"Lessons learned:\n{json.dumps(state.memory.lessons, indent=2)}",
            f"Successful patterns:\n{json.dumps(state.memory.successful_patterns, indent=2)}",
        ]
        if is_replanning and state.current_plan:
            parts += [
                "",
                "=== PREVIOUS PLAN (failed — do not repeat) ===",
                state.current_plan.model_dump_json(indent=2),
                "",
                "The previous plan was critiqued as follows:",
                state.critique.model_dump_json(indent=2) if state.critique else "N/A",
            ]
        parts.append("\nCreate the plan now. Return only valid JSON.")
        return "\n".join(parts)

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        plan = parse_model(Plan, raw)
        return state.model_copy(update={"current_plan": plan})
