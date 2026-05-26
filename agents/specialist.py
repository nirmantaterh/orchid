import json
from core.state import AgentState, ExecutionResult, StepResult, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM_TEMPLATE = """You are a specialist {specialist_type} agent in an adaptive multi-agent system.

You have been invoked because the standard executor is blocked on a specific sub-problem
that requires deep focus in your domain.

Your job:
- Diagnose the exact blocker relevant to {specialist_type}.
- Provide a targeted resolution or workaround.
- Map your output back to the plan's step structure.

Output ONLY valid JSON matching ExecutionResult schema:
{{
  "step_results": [
    {{
      "step_id": "<id from the plan>",
      "status": "success" | "failure" | "partial",
      "output": "What you determined or resolved",
      "issues": []
    }}
  ],
  "overall_status": "success" | "failure" | "partial",
  "summary": "Specialist {specialist_type} intervention summary",
  "blockers": ["Any remaining blockers after your intervention"]
}}

Return ONLY the JSON object."""


class SpecialistAgent(BaseAgent):
    _output_field = "execution_result"

    def __init__(self, provider: BaseLLMProvider, specialist_type: str) -> None:
        self._specialist_type = specialist_type  # must be set before super().__init__ calls self.name
        super().__init__(provider)

    @property
    def name(self) -> str:
        return f"specialist:{self._specialist_type}"

    def _build_system_prompt(self) -> str:
        return _SYSTEM_TEMPLATE.format(specialist_type=self._specialist_type)

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.current_plan is not None

        blockers = (
            state.execution_result.blockers if state.execution_result else []
        )
        return "\n".join([
            f"GOAL: {state.goal.goal}",
            f"SPECIALIST DOMAIN: {self._specialist_type}",
            "",
            "=== PLAN ===",
            state.current_plan.model_dump_json(indent=2),
            "",
            "=== BLOCKERS REQUIRING SPECIALIST ATTENTION ===",
            json.dumps(blockers, indent=2),
            "",
            "=== PREVIOUS EXECUTION (if any) ===",
            state.execution_result.model_dump_json(indent=2)
            if state.execution_result
            else "None",
            "",
            "Resolve the blockers in your domain. Return only valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        result = parse_model(ExecutionResult, raw)
        return state.model_copy(update={"execution_result": result})
