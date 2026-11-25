import json
import re
import statistics
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
from goal_interpreter.schema import GoalInterpretation


# ── Sub-models ────────────────────────────────────────────────────────────────

class Step(BaseModel):
    id: str
    description: str
    dependencies: list[str] = []
    expected_outcome: str


class Plan(BaseModel):
    steps: list[Step]
    rationale: str
    estimated_complexity: int = Field(ge=1, le=10)
    avoids: list[str] = []
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


class StepResult(BaseModel):
    step_id: str
    status: Literal["success", "failure", "partial"]
    output: str
    issues: list[str] = []


class ExecutionResult(BaseModel):
    step_results: list[StepResult]
    overall_status: Literal["success", "failure", "partial"]
    summary: str
    blockers: list[str] = []
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


class Critique(BaseModel):
    score: int = Field(ge=0, le=10)
    strengths: list[str] = []
    weaknesses: list[str] = []
    root_causes: list[str] = []
    suggestions: list[str] = []
    should_replan: bool = False
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


class Reflection(BaseModel):
    lessons_learned: list[str] = []
    strategy_updates: list[str] = []
    patterns_identified: list[str] = []
    confidence: int = Field(default=5, ge=0, le=10)
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


class MemoryEntry(BaseModel):
    pattern: str
    outcome: Literal["success", "failure"]
    frequency: int = 1
    last_iteration: int


class Memory(BaseModel):
    failed_strategies: list[str] = []
    successful_patterns: list[str] = []
    lessons: list[str] = []
    entries: list[MemoryEntry] = []
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


class ControlDecision(BaseModel):
    action: Literal["continue", "replan", "terminate", "specialist"]
    reason: str
    specialist_type: str | None = None
    agent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = ""


# ── Debate models ─────────────────────────────────────────────────────────────

class PanelistOutput(BaseModel):
    agent_name: str
    output_json: str
    confidence: float


class DebateResult(BaseModel):
    topic: str
    panelist_outputs: list[PanelistOutput]
    disagreement_score: float  # 0-1
    synthesis: str
    recommended_output_index: int = 0
    confidence: float = 0.5


# ── Trace + History ───────────────────────────────────────────────────────────

class TraceEntry(BaseModel):
    agent: str
    iteration: int
    timestamp: str
    summary: str


class IterationRecord(BaseModel):
    iteration: int
    plan: Plan | None = None
    execution_result: ExecutionResult | None = None
    critique: Critique | None = None
    reflection: Reflection | None = None
    control_decision: ControlDecision | None = None
    agent_confidences: dict[str, float] = {}     # snapshot: {agent_name: confidence}
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Shared state ──────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    raw_input: str
    goal: GoalInterpretation
    iteration: int = 0
    max_iterations: int = 10

    # Current cycle — cleared at the start of each new iteration (except plan on continue)
    current_plan: Plan | None = None
    execution_result: ExecutionResult | None = None
    critique: Critique | None = None
    reflection: Reflection | None = None
    control_signal: ControlDecision = Field(
        default_factory=lambda: ControlDecision(
            action="continue", reason="Initial state"
        )
    )

    # Confidence tracking — one dict per iteration
    confidence_history: list[dict[str, float]] = []

    # Persistent across all iterations
    memory: Memory = Field(default_factory=Memory)
    history: list[IterationRecord] = []
    trace: list[TraceEntry] = []
    debate_log: list[DebateResult] = []
    human_overrides: list[str] = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def initial(
        cls, raw_input: str, goal: GoalInterpretation, max_iterations: int = 10
    ) -> "AgentState":
        return cls(raw_input=raw_input, goal=goal, max_iterations=max_iterations)

    # ── Cycle management ──────────────────────────────────────────────────────

    def record_confidence(self, agent_name: str, confidence: float) -> "AgentState":
        """Append confidence for the current iteration's running dict."""
        if self.confidence_history and len(self.confidence_history) == self.iteration + 1:
            updated = dict(self.confidence_history[-1])
            updated[agent_name] = confidence
            new_hist = self.confidence_history[:-1] + [updated]
        else:
            # Start a new dict for this iteration
            existing = list(self.confidence_history)
            while len(existing) <= self.iteration:
                existing.append({})
            existing[self.iteration][agent_name] = confidence
            new_hist = existing
        return self.model_copy(update={"confidence_history": new_hist})

    def save_iteration(self) -> "AgentState":
        current_confidences = (
            self.confidence_history[self.iteration]
            if self.iteration < len(self.confidence_history)
            else {}
        )
        record = IterationRecord(
            iteration=self.iteration,
            plan=self.current_plan,
            execution_result=self.execution_result,
            critique=self.critique,
            reflection=self.reflection,
            control_decision=self.control_signal,
            agent_confidences=current_confidences,
        )
        return self.model_copy(update={"history": self.history + [record]})

    def add_trace(self, agent: str, summary: str) -> "AgentState":
        entry = TraceEntry(
            agent=agent,
            iteration=self.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary[:300],
        )
        return self.model_copy(update={"trace": self.trace + [entry]})

    def add_debate(self, result: DebateResult) -> "AgentState":
        return self.model_copy(update={"debate_log": self.debate_log + [result]})

    def next_iteration(self, keep_plan: bool = False) -> "AgentState":
        updates: dict = {
            "iteration": self.iteration + 1,
            "execution_result": None,
            "critique": None,
            "reflection": None,
        }
        if not keep_plan:
            updates["current_plan"] = None
        return self.model_copy(update=updates)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def critique_trend(self, window: int = 3) -> list[int]:
        return [
            r.critique.score
            for r in self.history[-window:]
            if r.critique is not None
        ]

    def confidence_trend(self, window: int = 5) -> list[float]:
        """Mean per-iteration confidence over last `window` iterations."""
        result = []
        for record in self.history[-window:]:
            vals = list(record.agent_confidences.values())
            if vals:
                result.append(statistics.mean(vals))
        return result

    def confidence_variance(self, window: int = 3) -> float:
        """Variance across agents in the most recent `window` iterations."""
        all_vals: list[float] = []
        for record in self.history[-window:]:
            all_vals.extend(record.agent_confidences.values())
        return statistics.variance(all_vals) if len(all_vals) > 1 else 0.0

    def is_confidence_dropping(self, window: int = 3, threshold: float = 0.1) -> bool:
        trend = self.confidence_trend(window)
        if len(trend) < 2:
            return False
        return (trend[0] - trend[-1]) > threshold


# ── JSON extraction utility ───────────────────────────────────────────────────

def parse_model(model_class, text: str):
    """Parse a Pydantic model from JSON with a lenient fallback."""
    try:
        return model_class.model_validate_json(text)
    except Exception as primary:
        try:
            return model_class.model_validate(json.loads(text))
        except Exception:
            raise primary


def extract_json(text: str) -> str:
    """Strip markdown fences and repair common LLM JSON mistakes."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    # Quote unquoted numeric keys: { 1: → { "1":
    text = re.sub(r'(?<=[{,])\s*(\d+)\s*:', lambda m: f' "{m.group(1).strip()}":', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Truncate to the last complete JSON object if the response was cut off
    try:
        json.loads(text)
    except json.JSONDecodeError:
        last = text.rfind("}")
        if last != -1:
            text = text[:last + 1]
        # If no } found, keep original — don't truncate to just "{"
    return text
