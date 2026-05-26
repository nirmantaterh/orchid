import statistics
from pydantic import BaseModel, Field
from core.state import AgentState


# Token cost constants (USD per 1K tokens, approximate mid-range model pricing)
_INPUT_COST_PER_1K = 0.003
_OUTPUT_COST_PER_1K = 0.015


class RunMetrics(BaseModel):
    task_success: bool
    iterations_required: int
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    memory_efficiency: float = Field(ge=0.0, le=1.0, default=0.0)
    replanning_frequency: float = Field(ge=0.0, le=1.0, default=0.0)
    agent_utilization: dict[str, int] = {}
    average_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_variance: float = 0.0
    debate_invocations: int = 0
    specialist_invocations: int = 0
    average_critique_score: float = 0.0


def compute_metrics(state: AgentState, total_tokens: int = 0,
                    total_latency_ms: float = 0.0) -> RunMetrics:
    history = state.history
    iterations = len(history)

    # Task success: terminated with a successful final critique,
    # OR high score + execution success (controller failed to emit terminate)
    # Use the last non-None critique — critic can silently fail on the final iteration
    # leaving history[-1].critique as None even though state.critique carries a valid score.
    final_critique = next(
        (r.critique for r in reversed(history) if r.critique is not None),
        None,
    ) or state.critique
    terminated_ok = (
        state.control_signal.action == "terminate"
        and final_critique is not None
        and final_critique.score >= 6
    )
    high_score_success = (
        final_critique is not None
        and final_critique.score >= 6
        and state.execution_result is not None
        and state.execution_result.overall_status == "success"
    )
    task_success = terminated_ok or high_score_success

    # Replanning frequency
    replan_count = sum(
        1 for r in history
        if r.control_decision and r.control_decision.action == "replan"
    )
    replan_freq = replan_count / max(1, iterations)

    # Agent utilization from trace
    utilization: dict[str, int] = {}
    for entry in state.trace:
        utilization[entry.agent] = utilization.get(entry.agent, 0) + 1

    # Specialist invocations
    specialist_invocations = sum(
        1 for entry in state.trace if entry.agent.startswith("specialist:")
    )

    # Critique scores
    scores = [
        r.critique.score for r in history if r.critique is not None
    ]
    avg_score = statistics.mean(scores) if scores else 0.0

    # Confidence stats
    all_conf: list[float] = []
    for record in history:
        all_conf.extend(record.agent_confidences.values())
    avg_conf = statistics.mean(all_conf) if all_conf else 0.0
    conf_var = statistics.variance(all_conf) if len(all_conf) > 1 else 0.0

    # Memory efficiency: ratio of lessons to total memory entries
    total_memory_items = (
        len(state.memory.lessons)
        + len(state.memory.failed_strategies)
        + len(state.memory.successful_patterns)
    )
    useful = len(state.memory.lessons) + len(state.memory.successful_patterns)
    mem_efficiency = useful / max(1, total_memory_items)

    # Cost estimate
    cost = (total_tokens / 1000) * ((_INPUT_COST_PER_1K + _OUTPUT_COST_PER_1K) / 2)

    return RunMetrics(
        task_success=task_success,
        iterations_required=iterations,
        total_tokens=total_tokens,
        estimated_cost_usd=round(cost, 6),
        total_latency_ms=round(total_latency_ms, 2),
        memory_efficiency=round(mem_efficiency, 3),
        replanning_frequency=round(replan_freq, 3),
        agent_utilization=utilization,
        average_confidence=round(avg_conf, 3),
        confidence_variance=round(conf_var, 4),
        debate_invocations=len(state.debate_log),
        specialist_invocations=specialist_invocations,
        average_critique_score=round(avg_score, 2),
    )


def aggregate_metrics(results: list[RunMetrics]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n": n,
        "success_rate": sum(r.task_success for r in results) / n,
        "avg_iterations": statistics.mean(r.iterations_required for r in results),
        "avg_cost_usd": statistics.mean(r.estimated_cost_usd for r in results),
        "avg_latency_ms": statistics.mean(r.total_latency_ms for r in results),
        "avg_memory_efficiency": statistics.mean(r.memory_efficiency for r in results),
        "avg_replanning_freq": statistics.mean(r.replanning_frequency for r in results),
        "avg_confidence": statistics.mean(r.average_confidence for r in results),
        "avg_critique_score": statistics.mean(r.average_critique_score for r in results),
        "total_specialist_invocations": sum(r.specialist_invocations for r in results),
        "total_debate_invocations": sum(r.debate_invocations for r in results),
    }
