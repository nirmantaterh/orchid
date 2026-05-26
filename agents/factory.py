"""
AgentFactory — adaptively decides which specialist to create based on state signals.

Decision logic (in priority order):
  1. If controller already named a specialist_type, use it.
  2. If recurring blockers contain keyword signals → map to domain.
  3. If confidence is very low + failures mention math/numbers → math.
  4. If failures mention missing information / data → research.
  5. If failures mention code / implementation → code.
  6. If failures mention security / auth → security.
  7. If failures mention performance / latency → optimization.
  8. Default: debugging.
"""
import re

from core.state import AgentState
from goal_interpreter.providers.base import BaseLLMProvider
from .specialist import SpecialistAgent
from .specialists import (
    ResearchSpecialist,
    MathSpecialist,
    CodeSpecialist,
    RetrievalSpecialist,
    DebuggingSpecialist,
    SecuritySpecialist,
    OptimizationSpecialist,
)
from core.logger import get_logger, log

_TYPED_MAP: dict[str, type] = {
    "research": ResearchSpecialist,
    "math": MathSpecialist,
    "code": CodeSpecialist,
    "retrieval": RetrievalSpecialist,
    "debugging": DebuggingSpecialist,
    "security": SecuritySpecialist,
    "optimization": OptimizationSpecialist,
}

_KEYWORD_MAP: list[tuple[str, str]] = [
    (r"\b(security|auth|jwt|oauth|encrypt|xss|sql.?inject)\b", "security"),
    (r"\b(perform|latency|slow|bottleneck|cache|memory.?leak|cpu)\b", "optimization"),
    (r"\b(math|calcul|statistic|formula|numeric|algorithm|complex)\b", "math"),
    (r"\b(retriev|search|embed|rag|vector|index|chunk)\b", "retrieval"),
    (r"\b(code|implement|bug|exception|stacktrace|syntax|import)\b", "code"),
    (r"\b(research|information|data|source|paper|knowledge|unknown)\b", "research"),
]


class AgentFactory:
    """
    Creates the most appropriate specialist for the current system state.

    Usage:
        factory = AgentFactory(provider)
        specialist = factory.create_for_state(state)
        updated_state = specialist.run(state)
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider
        self._logger = get_logger("factory")

    def create_for_state(self, state: AgentState) -> SpecialistAgent:
        specialist_type = self._infer_type(state)
        log(self._logger, "info", "Creating specialist",
            type=specialist_type, iteration=state.iteration)
        return self._build(specialist_type)

    def create_by_type(self, specialist_type: str) -> SpecialistAgent:
        return self._build(specialist_type)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_type(self, state: AgentState) -> str:
        # 1. Controller already named it
        if state.control_signal.specialist_type:
            t = state.control_signal.specialist_type.lower()
            if t in _TYPED_MAP:
                return t

        # 2. Scan blockers + failure messages for keywords
        text_to_scan = " ".join(self._gather_failure_text(state)).lower()
        for pattern, domain in _KEYWORD_MAP:
            if re.search(pattern, text_to_scan, re.IGNORECASE):
                return domain

        # 3. Low confidence + lots of failed strategies → research
        if (state.is_confidence_dropping() and
                len(state.memory.failed_strategies) >= 2):
            return "research"

        return "debugging"

    @staticmethod
    def _gather_failure_text(state: AgentState) -> list[str]:
        texts: list[str] = []
        if state.execution_result:
            texts.extend(state.execution_result.blockers)
            for sr in state.execution_result.step_results:
                texts.extend(sr.issues)
        if state.critique:
            texts.extend(state.critique.root_causes)
        for record in state.history[-3:]:
            if record.execution_result:
                texts.extend(record.execution_result.blockers)
        return texts

    def _build(self, specialist_type: str) -> SpecialistAgent:
        cls = _TYPED_MAP.get(specialist_type.lower())
        if cls:
            return cls(self._provider)
        # Fallback: generic specialist with the requested type label
        return SpecialistAgent(self._provider, specialist_type)
