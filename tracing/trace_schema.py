from pydantic import BaseModel, Field


class AgentTraceEntry(BaseModel):
    agent_name: str
    iteration: int
    input_summary: str
    output_summary: str
    reasoning: str = ""
    confidence: float = 0.5
    latency_ms: float = 0.0
    token_usage: int = 0
    success: bool = True
    error: str | None = None
    timestamp: str = ""


class MemoryUpdateRecord(BaseModel):
    iteration: int
    added_lessons: list[str] = []
    added_failed_strategies: list[str] = []
    added_successful_patterns: list[str] = []


class IterationTrace(BaseModel):
    iteration: int
    agents_run: list[str]
    entries: list[AgentTraceEntry] = []
    control_decision: str = ""
    overall_confidence: float = 0.0  # mean of agent confidences
    confidence_variance: float = 0.0
    memory_update: MemoryUpdateRecord | None = None
    debate_invoked: bool = False
    timestamp: str = ""


class SystemTrace(BaseModel):
    run_id: str
    goal: str
    total_iterations: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    final_action: str = ""
    iterations: list[IterationTrace] = []
    created_at: str = ""
