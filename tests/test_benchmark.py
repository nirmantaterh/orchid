import json
import pytest
from goal_interpreter.providers.base import BaseLLMProvider
from core.state import AgentState
from evaluation.metrics import RunMetrics, compute_metrics, aggregate_metrics
from evaluation.benchmark import Benchmark, BenchmarkCase, BenchmarkResult


# ── Mock provider for full orchestrator runs ──────────────────────────────────

class SequentialProvider(BaseLLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._it = iter(responses)

    def generate(self, system: str, user: str) -> str:
        return next(self._it)


def goal_r():
    return json.dumps({
        "goal": "Build API", "subgoals": ["Design", "Implement"],
        "complexity": 4, "assumptions": [], "uncertainty": 2,
        "initial_strategy": "Schema-first", "success_criteria": ["API works"],
    })

def plan_r():
    return json.dumps({
        "steps": [{"id": "1", "description": "Design", "dependencies": [],
                   "expected_outcome": "Done"}],
        "rationale": "x", "estimated_complexity": 3, "avoids": [],
        "agent_confidence": 0.85, "reasoning": "clear",
    })

def exec_r(status="success"):
    return json.dumps({
        "step_results": [{"step_id": "1", "status": status,
                          "output": "done", "issues": []}],
        "overall_status": status, "summary": "done", "blockers": [],
        "agent_confidence": 0.8, "reasoning": "ok",
    })

def critique_r(score=8):
    return json.dumps({
        "score": score, "strengths": ["good"], "weaknesses": [],
        "root_causes": [], "suggestions": [], "should_replan": False,
        "agent_confidence": 0.85, "reasoning": "looks good",
    })

def reflection_r():
    return json.dumps({
        "lessons_learned": ["lesson"], "strategy_updates": [],
        "patterns_identified": [], "confidence": 8,
        "agent_confidence": 0.8, "reasoning": "ok",
    })

def memory_r():
    return json.dumps({
        "failed_strategies": [], "successful_patterns": ["schema-first"],
        "lessons": ["lesson"], "entries": [],
        "agent_confidence": 0.8, "reasoning": "updated",
    })

def ctrl_r(action="terminate"):
    return json.dumps({
        "action": action, "reason": f"Decision: {action}",
        "specialist_type": None, "agent_confidence": 0.9, "reasoning": "decided",
    })


# ── compute_metrics ───────────────────────────────────────────────────────────

def make_finished_state() -> AgentState:
    from goal_interpreter.schema import GoalInterpretation
    from core.state import (Plan, Step, ExecutionResult, StepResult,
                             Critique, Reflection, Memory, ControlDecision)
    goal = GoalInterpretation(
        goal="Build API", subgoals=[], complexity=4, assumptions=[],
        uncertainty=2, initial_strategy="x", success_criteria=[],
    )
    state = AgentState.initial("Build API", goal)
    state = state.model_copy(update={
        "control_signal": ControlDecision(
            action="terminate", reason="done", agent_confidence=0.9
        ),
        "current_plan": Plan(
            steps=[Step(id="1", description="x", expected_outcome="y")],
            rationale="x", estimated_complexity=3,
        ),
        "execution_result": ExecutionResult(
            step_results=[StepResult(step_id="1", status="success", output="done")],
            overall_status="success", summary="ok", agent_confidence=0.8,
        ),
        "critique": Critique(
            score=8, strengths=[], weaknesses=[], root_causes=[], suggestions=[],
            should_replan=False, agent_confidence=0.85,
        ),
        "reflection": Reflection(
            lessons_learned=["x"], strategy_updates=[], patterns_identified=[],
            confidence=8, agent_confidence=0.8,
        ),
    })
    state = state.record_confidence("planner", 0.85)
    state = state.record_confidence("executor", 0.8)
    state = state.record_confidence("critic", 0.85)
    state = state.save_iteration()
    return state


def test_compute_metrics_success():
    state = make_finished_state()
    m = compute_metrics(state)
    assert m.task_success is True
    assert m.iterations_required == 1
    assert m.average_critique_score == 8.0


def test_compute_metrics_failure():
    from goal_interpreter.schema import GoalInterpretation
    from core.state import Critique, ControlDecision
    goal = GoalInterpretation(
        goal="x", subgoals=[], complexity=3, assumptions=[],
        uncertainty=2, initial_strategy="x", success_criteria=[],
    )
    state = AgentState.initial("x", goal)
    state = state.model_copy(update={
        "control_signal": ControlDecision(action="terminate", reason="stuck"),
        "critique": Critique(score=3, strengths=[], weaknesses=[],
                             root_causes=[], suggestions=[], should_replan=True),
    })
    state = state.save_iteration()
    m = compute_metrics(state)
    assert m.task_success is False


def test_aggregate_metrics():
    m1 = RunMetrics(task_success=True, iterations_required=2,
                    average_confidence=0.8, average_critique_score=8.0)
    m2 = RunMetrics(task_success=False, iterations_required=5,
                    average_confidence=0.5, average_critique_score=4.0)
    agg = aggregate_metrics([m1, m2])
    assert agg["n"] == 2
    assert agg["success_rate"] == 0.5
    assert agg["avg_iterations"] == 3.5


# ── Benchmark integration ─────────────────────────────────────────────────────

def test_benchmark_run_case_success():
    provider = SequentialProvider([
        goal_r(), plan_r(), exec_r("success"),
        critique_r(9), reflection_r(), memory_r(), ctrl_r("terminate"),
    ])
    bench = Benchmark(provider, max_iterations=3)
    case = BenchmarkCase(name="simple", input="Build a REST API", max_iterations=3)
    result = bench.run_case(case)
    assert result.case.name == "simple"
    assert result.final_action == "terminate"
    assert isinstance(result.metrics, RunMetrics)


def test_benchmark_summary():
    r1 = BenchmarkResult(
        case=BenchmarkCase(name="a", input="x"),
        metrics=RunMetrics(task_success=True, iterations_required=2),
        final_action="terminate", iterations_run=2, passed=True,
    )
    r2 = BenchmarkResult(
        case=BenchmarkCase(name="b", input="y"),
        metrics=RunMetrics(task_success=False, iterations_required=5),
        final_action="terminate", iterations_run=5, passed=False,
    )
    summary = Benchmark.summary([r1, r2])
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5


def test_benchmark_save_report(tmp_path):
    r = BenchmarkResult(
        case=BenchmarkCase(name="t", input="x"),
        metrics=RunMetrics(task_success=True, iterations_required=1),
        final_action="terminate", iterations_run=1, passed=True,
    )
    out = tmp_path / "report.json"
    Benchmark.save_report([r], out)
    data = json.loads(out.read_text())
    assert "summary" in data
    assert len(data["cases"]) == 1
