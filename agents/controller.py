import json
from core.state import AgentState, ControlDecision, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are the Controller in an adaptive multi-agent system.

You are the meta-decision-maker. After every full iteration you decide what happens next.

Available actions:
  "continue"   — the current plan is working; execute the next cycle with it.
  "replan"     — the plan is fundamentally flawed; discard it and create a new one.
  "terminate"  — the goal is achieved (or cannot be achieved); stop the loop.
  "specialist" — a specific sub-problem requires a focused specialist agent; name the type.

Output ONLY valid JSON:
{
  "action": "continue" | "replan" | "terminate" | "specialist",
  "reason": "One sentence explaining the decision",
  "specialist_type": null | "debugging" | "research" | "optimization" | "security" | "other",
  "agent_confidence": 0.85,
  "reasoning": "One sentence explaining your confidence in this control decision"
}

DECISION RULES — mandatory, follow in order:
1. YOU MUST return "terminate" if critique score >= 6 AND overall_status == "success".
   This is a hard rule. Do NOT return "continue" when the task is already done well.
2. YOU MUST return "terminate" if iteration >= max_iterations.
3. Return "replan" if should_replan == true OR score <= 3 OR same failure 2+ iterations.
4. Return "specialist" if execution is blocked on a well-defined sub-problem.
5. Return "continue" ONLY if none of the above apply and meaningful progress is possible.

specialist_type is only set when action == "specialist"; null otherwise.
Return ONLY the JSON object."""


class ControllerAgent(BaseAgent):
    name = "controller"
    _output_field = "control_signal"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.critique is not None
        assert state.reflection is not None

        trend = state.critique_trend(window=5)

        # Summarise recent blockers to help controller detect stagnation
        recent_blockers: list[str] = []
        for r in state.history[-3:]:
            if r.execution_result:
                recent_blockers.extend(r.execution_result.blockers)

        conf_trend = state.confidence_trend(window=5)
        conf_variance = round(state.confidence_variance(window=3), 4)
        conf_dropping = state.is_confidence_dropping(window=3)

        return "\n".join([
            "=== SYSTEM STATUS ===",
            f"Goal: {state.goal.goal}",
            f"Iteration: {state.iteration} / {state.max_iterations}",
            "",
            "=== CURRENT ITERATION RESULTS ===",
            f"Execution status: {state.execution_result.overall_status if state.execution_result else 'N/A'}",
            f"Critique score: {state.critique.score}/10",
            f"Should replan (critic): {state.critique.should_replan}",
            f"Reflection confidence: {state.reflection.confidence}/10",
            "",
            f"=== CRITIQUE SCORE TREND (last {len(trend)} iters): {trend} ===",
            "Flat or declining trend → consider replan or terminate.",
            "",
            "=== AGENT CONFIDENCE SIGNALS ===",
            f"Confidence trend (mean per iteration): {[round(c,2) for c in conf_trend]}",
            f"Confidence variance: {conf_variance}  (high variance → agent disagreement → consider debate/specialist)",
            f"Confidence dropping: {conf_dropping}  (True → trigger replan or specialist)",
            "",
            "=== RECURRING BLOCKERS ===",
            json.dumps(list(set(recent_blockers)), indent=2),
            "",
            "=== MEMORY SUMMARY ===",
            f"Failed strategies: {len(state.memory.failed_strategies)}",
            f"Lessons learned: {len(state.memory.lessons)}",
            "",
            "Decide the next action. Return only valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        decision = parse_model(ControlDecision, raw)
        return state.model_copy(update={"control_signal": decision})
