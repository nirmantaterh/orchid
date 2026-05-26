import json
from core.state import AgentState, Critique, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are a Critic in an adaptive multi-agent system.

Your job:
- Objectively evaluate how well the execution addressed the original goal.
- Identify root causes, not just symptoms.
- Check whether the same mistakes are being repeated across iterations.
- Recommend whether a full replan is warranted or incremental fixes will suffice.

Scoring (0–10):
  0–3: Fundamental failure, must replan
  4–6: Partial progress, targeted improvements possible
  7–9: Good execution, minor gaps
  10:  Goal fully achieved

Output ONLY valid JSON. Keep ALL string values SHORT (max 12 words each). Max 3 items per list.
Complete the entire JSON object before stopping.
{
  "score": 0,
  "strengths": ["short phrase"],
  "weaknesses": ["short phrase"],
  "root_causes": ["short phrase"],
  "suggestions": ["short phrase"],
  "should_replan": false,
  "agent_confidence": 0.8,
  "reasoning": "one short sentence"
}

Rules:
- score and should_replan must be consistent (score ≤ 4 → should_replan: true).
- Return ONLY the JSON object. No explanation outside the JSON."""


class CriticAgent(BaseAgent):
    name = "critic"
    _output_field = "critique"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.current_plan is not None
        assert state.execution_result is not None

        # Surface score trend so critic can flag stagnation
        trend = state.critique_trend(window=3)

        exec_json = state.execution_result.model_dump_json(indent=2)
        if len(exec_json) > 1500:
            exec_json = exec_json[:1500] + "\n... [truncated]"

        plan_json = state.current_plan.model_dump_json(indent=2)
        if len(plan_json) > 800:
            plan_json = plan_json[:800] + "\n... [truncated]"

        return "\n".join([
            f"GOAL: {state.goal.goal}",
            f"SUCCESS CRITERIA: {json.dumps(state.goal.success_criteria)}",
            f"ITERATION: {state.iteration}",
            "",
            "=== PLAN ===",
            plan_json,
            "",
            "=== EXECUTION RESULT ===",
            exec_json,
            "",
            f"=== CRITIQUE SCORE TREND (last {len(trend)} iterations): {trend} ===",
            "If scores are flat or declining, escalate should_replan to true.",
            "",
            "Critique the execution now. Return only valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        critique = parse_model(Critique, raw)
        return state.model_copy(update={"critique": critique})
