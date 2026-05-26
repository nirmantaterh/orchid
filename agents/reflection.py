import json
from core.state import AgentState, Reflection, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are a Reflection agent in an adaptive multi-agent system.

Your job:
- Synthesize what happened this iteration into durable lessons.
- Identify patterns that span multiple iterations, not just the current one.
- Propose strategy updates that will genuinely change behavior in the next cycle.
- Do NOT repeat lessons that are already in memory — add new insight or discard.

Output ONLY valid JSON:
{
  "lessons_learned": ["New facts learned this iteration"],
  "strategy_updates": ["How the approach should change going forward"],
  "patterns_identified": ["Cross-iteration patterns you observe"],
  "confidence": 7,
  "agent_confidence": 0.8,
  "reasoning": "One sentence explaining your confidence in these reflections"
}

Rules:
- confidence (0–10) reflects how certain you are the lessons will transfer.
- lessons_learned must be new — don't echo what's already in existing_lessons.
- strategy_updates must describe a concrete behavioral change, not vague advice.
- Return ONLY the JSON object."""


class ReflectionAgent(BaseAgent):
    name = "reflection"
    _output_field = "reflection"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.critique is not None
        assert state.execution_result is not None

        return "\n".join([
            f"GOAL: {state.goal.goal}",
            f"ITERATION: {state.iteration}",
            "",
            "=== THIS ITERATION ===",
            f"Execution status: {state.execution_result.overall_status}",
            f"Critique score: {state.critique.score}/10",
            f"Critique root causes: {json.dumps(state.critique.root_causes)}",
            f"Critique suggestions: {json.dumps(state.critique.suggestions)}",
            "",
            "=== EXISTING LESSONS (do NOT repeat these) ===",
            json.dumps(state.memory.lessons, indent=2),
            "",
            "=== SCORE TREND ===",
            json.dumps(state.critique_trend(window=5)),
            "",
            "Reflect and extract new lessons. Return only valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        reflection = parse_model(Reflection, raw)
        return state.model_copy(update={"reflection": reflection})
