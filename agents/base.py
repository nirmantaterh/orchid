from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.state import AgentState, extract_json
from core.logger import get_logger, log
from goal_interpreter.providers.base import BaseLLMProvider

if TYPE_CHECKING:
    from tracing.tracer import Tracer


class BaseAgent(ABC):
    """
    All agents share this contract:
      - receive the full AgentState
      - return a new AgentState with their contribution applied
      - optionally record a Tracer span with confidence + reasoning
      - add a lightweight TraceEntry to the state for post-run inspection
    """

    def __init__(self, provider: BaseLLMProvider, tracer: "Tracer | None" = None) -> None:
        self._provider = provider
        self._tracer = tracer
        self._logger = get_logger(self.name)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _build_system_prompt(self) -> str: ...

    @abstractmethod
    def _build_user_prompt(self, state: AgentState) -> str: ...

    @abstractmethod
    def _apply(self, raw: str, state: AgentState) -> AgentState: ...

    # Subclasses declare which AgentState field they write their output to.
    _output_field: str = ""

    # ── Confidence extraction ─────────────────────────────────────────────────

    def _get_confidence(self, state: AgentState) -> float:
        """Read agent_confidence from the field this agent writes."""
        field = self._output_field
        if field:
            obj = getattr(state, field, None)
            if obj is not None and hasattr(obj, "agent_confidence"):
                return float(obj.agent_confidence)
        return 0.5

    def _get_reasoning(self, state: AgentState) -> str:
        field = self._output_field
        if field:
            obj = getattr(state, field, None)
            if obj is not None and hasattr(obj, "reasoning") and obj.reasoning:
                return obj.reasoning
        return ""

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, state: AgentState, tracer: "Tracer | None" = None) -> AgentState:
        active_tracer = tracer or self._tracer
        try:
            system = self._build_system_prompt()
            user = self._build_user_prompt(state)
        except Exception as exc:
            log(self._logger, "warning", f"{self.name} prompt-build error — skipping",
                error=str(exc))
            return (state.add_trace(self.name, f"PREREQ_MISSING: {str(exc)[:200]}")
                         .record_confidence(self.name, 0.3))

        log(self._logger, "info", f"{self.name} running",
            iteration=state.iteration, goal=state.goal.goal[:80])

        if active_tracer:
            return self._run_with_tracer(system, user, state, active_tracer)
        return self._run_bare(system, user, state)

    def _run_bare(self, system: str, user: str, state: AgentState) -> AgentState:
        raw = self._provider.generate(system, user)
        cleaned = extract_json(raw)
        log(self._logger, "debug", f"{self.name} raw response", preview=cleaned[:200])

        try:
            updated = self._apply(cleaned, state)
        except Exception as exc:
            log(self._logger, "warning", f"{self.name} parse error — continuing",
                error=str(exc), raw=cleaned[:400])
            updated = state.add_trace(self.name, f"PARSE_ERROR: {str(exc)[:200]}")
            updated = updated.record_confidence(self.name, 0.3)
            return updated

        confidence = self._get_confidence(updated)
        updated = updated.record_confidence(self.name, confidence)
        updated = updated.add_trace(self.name, cleaned[:300])
        log(self._logger, "info", f"{self.name} complete",
            iteration=state.iteration, confidence=confidence)
        return updated

    def _run_with_tracer(
        self, system: str, user: str, state: AgentState, tracer: "Tracer"
    ) -> AgentState:
        input_summary = f"[{self.name}] iter={state.iteration} goal={state.goal.goal[:60]}"
        with tracer.agent_span(self.name, input_summary, state.iteration) as span:
            raw = self._provider.generate(system, user)
            cleaned = extract_json(raw)
            log(self._logger, "debug", f"{self.name} raw response", preview=cleaned[:200])

            try:
                updated = self._apply(cleaned, state)
                parse_failed = False
            except Exception as exc:
                log(self._logger, "warning", f"{self.name} parse error — continuing",
                    error=str(exc), raw=cleaned[:400])
                updated = state
                parse_failed = True
                span.confidence = 0.3
                span.reasoning = f"PARSE_ERROR: {str(exc)[:200]}"
                span.output_summary = cleaned[:300]

            if parse_failed:
                updated = updated.record_confidence(self.name, 0.3)
                updated = updated.add_trace(self.name, f"PARSE_ERROR")
                return updated

            confidence = self._get_confidence(updated)
            reasoning = self._get_reasoning(updated)
            span.confidence = confidence
            span.reasoning = reasoning
            span.output_summary = cleaned[:300]

        updated = updated.record_confidence(self.name, confidence)
        updated = updated.add_trace(self.name, cleaned[:300])
        log(self._logger, "info", f"{self.name} complete",
            iteration=state.iteration, confidence=confidence)
        return updated
