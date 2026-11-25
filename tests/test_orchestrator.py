"""
Integration tests for the Orchestrator using a SequentialProvider.

SequentialProvider returns pre-canned responses in order so we can
exercise multi-iteration loops, replan branches, and specialist injection
without hitting a real LLM.
"""
import json
import pytest
from goal_interpreter.providers.base import BaseLLMProvider
from core.orchestrator import Orchestrator
from core.state import AgentState


# ── Mock provider ─────────────────────────────────────────────────────────────

class SequentialProvider(BaseLLMProvider):
    """Returns responses in order; raises StopIteration if exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self._calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self._calls.append((system[:40], user[:40]))
        return next(self._responses)

    @property
    def call_count(self) -> int:
        return len(self._calls)


# ── Canned response builders ──────────────────────────────────────────────────

def goal_response():
    return json.dumps({
        "goal": "Build a recommendation engine",
        "subgoals": ["Collect data", "Train model"],
        "complexity": 6,
        "assumptions": ["Python available"],
        "uncertainty": 3,
        "initial_strategy": "Collaborative filtering",
        "success_criteria": ["Returns top-10 recs"],
    })


def plan_response(avoids=None):
    return json.dumps({
        "steps": [
            {"id": "1", "description": "Collect data", "dependencies": [],
             "expected_outcome": "Dataset ready"},
            {"id": "2", "description": "Train model", "dependencies": ["1"],
             "expected_outcome": "Model trained"},
        ],
        "rationale": "Data first approach",
        "estimated_complexity": 5,
        "avoids": avoids or [],
    })


def exec_response(status="partial"):
    return json.dumps({
        "step_results": [
            {"step_id": "1", "status": "success", "output": "Data collected", "issues": []},
            {"step_id": "2", "status": status, "output": "Model training incomplete",
             "issues": ["Insufficient data"]},
        ],
        "overall_status": status,
        "summary": f"Execution {status}",
        "blockers": ["Insufficient training data"] if status != "success" else [],
    })


def exec_success_response():
    return json.dumps({
        "step_results": [
            {"step_id": "1", "status": "success", "output": "Data collected", "issues": []},
            {"step_id": "2", "status": "success", "output": "Model trained", "issues": []},
        ],
        "overall_status": "success",
        "summary": "All steps completed",
        "blockers": [],
    })


def critique_response(score=6, should_replan=False):
    return json.dumps({
        "score": score,
        "strengths": ["Data collection worked"],
        "weaknesses": ["Training incomplete"],
        "root_causes": ["Insufficient data volume"],
        "suggestions": ["Augment dataset"],
        "should_replan": should_replan,
    })


def reflection_response():
    return json.dumps({
        "lessons_learned": ["More data needed for training"],
        "strategy_updates": ["Include data augmentation step"],
        "patterns_identified": ["Data insufficiency recurs"],
        "confidence": 7,
    })


def memory_response(failed=None):
    return json.dumps({
        "failed_strategies": failed or [],
        "successful_patterns": ["Phased data collection"],
        "lessons": ["More data needed for training"],
        "entries": [],
    })


def control_response(action="terminate"):
    return json.dumps({
        "action": action,
        "reason": f"Decision: {action}",
        "specialist_type": None,
    })


def specialist_response():
    return json.dumps({
        "step_results": [
            {"step_id": "2", "status": "success",
             "output": "Model trained after augmentation", "issues": []}
        ],
        "overall_status": "success",
        "summary": "Specialist resolved data issue",
        "blockers": [],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_single_iteration_terminate():
    """One full iteration that terminates immediately."""
    provider = SequentialProvider([
        goal_response(),       # GoalInterpreter
        plan_response(),       # Planner
        exec_success_response(),   # Executor
        critique_response(score=9),  # Critic
        reflection_response(), # Reflection
        memory_response(),     # MemoryManager
        control_response("terminate"),  # Controller
    ])
    orch = Orchestrator(provider, max_iterations=5)
    state, _ = orch.run("Build a recommendation engine")

    assert state.control_signal.action == "terminate"
    assert len(state.history) == 1
    assert state.history[0].critique.score == 9


def test_two_iterations_continue_then_terminate():
    """Two full iterations: first continues, second terminates."""
    provider = SequentialProvider([
        goal_response(),
        # Iteration 0
        plan_response(), exec_response("partial"),
        critique_response(6), reflection_response(),
        memory_response(), control_response("continue"),
        # Iteration 1 (plan kept, planner skipped)
        exec_success_response(), critique_response(9),
        reflection_response(), memory_response(),
        control_response("terminate"),
    ])
    orch = Orchestrator(provider, max_iterations=5)
    state, _ = orch.run("Build a recommendation engine")

    assert state.iteration == 1
    assert len(state.history) == 2
    assert state.control_signal.action == "terminate"


def test_replan_creates_new_plan():
    """Controller signals replan → Planner runs again in next iteration."""
    provider = SequentialProvider([
        goal_response(),
        # Iteration 0
        plan_response(), exec_response("failure"),
        critique_response(3, should_replan=True), reflection_response(),
        memory_response(failed=["data first approach"]),
        control_response("replan"),
        # Iteration 1 — planner runs again (replan)
        plan_response(avoids=["data first approach"]),
        exec_success_response(), critique_response(9),
        reflection_response(), memory_response(),
        control_response("terminate"),
    ])
    orch = Orchestrator(provider, max_iterations=5)
    state, _ = orch.run("Build a recommendation engine")

    assert state.iteration == 1
    assert len(state.history) == 2
    assert state.control_signal.action == "terminate"


def test_specialist_injection():
    """Controller signals specialist → SpecialistAgent runs before next cycle."""
    specialist_ctrl = json.dumps({
        "action": "specialist",
        "reason": "Data problem needs specialist",
        "specialist_type": "research",
    })
    provider = SequentialProvider([
        goal_response(),
        # Iteration 0
        plan_response(), exec_response("partial"),
        critique_response(5), reflection_response(),
        memory_response(), specialist_ctrl,
        # Specialist agent
        specialist_response(),
        # Iteration 1 (plan kept, continue from specialist result)
        exec_success_response(), critique_response(9),
        reflection_response(), memory_response(),
        control_response("terminate"),
    ])
    orch = Orchestrator(provider, max_iterations=5)
    state, _ = orch.run("Build a recommendation engine")

    assert state.control_signal.action == "terminate"
    assert any("specialist:research" in t.agent for t in state.trace)


def test_max_iterations_enforced():
    """Loop exits after max_iterations regardless of controller action."""
    # Iteration 0 needs planner (full cycle = 6 responses).
    # Iteration 1 keeps plan on "continue" so planner is skipped (5 responses).
    provider = SequentialProvider([
        goal_response(),
        # Iteration 0 — full cycle
        plan_response(), exec_response("partial"),
        critique_response(5), reflection_response(),
        memory_response(), control_response("continue"),
        # Iteration 1 — continue, planner skipped
        exec_response("partial"), critique_response(5),
        reflection_response(), memory_response(),
        control_response("continue"),
    ])
    orch = Orchestrator(provider, max_iterations=2)
    state, _ = orch.run("Build a recommendation engine")

    assert len(state.history) <= 2


def test_memory_persists_across_iterations():
    """Lessons from iteration 0 should be visible in state after iteration 1."""
    provider = SequentialProvider([
        goal_response(),
        plan_response(), exec_response("partial"),
        critique_response(5), reflection_response(),
        memory_response(failed=["naive approach"]),
        control_response("continue"),
        # Iteration 1
        exec_success_response(), critique_response(9),
        reflection_response(), memory_response(failed=["naive approach"]),
        control_response("terminate"),
    ])
    orch = Orchestrator(provider, max_iterations=5)
    state, _ = orch.run("Build a recommendation engine")

    assert "naive approach" in state.memory.failed_strategies


# ── Human approval ────────────────────────────────────────────────────────────

def test_human_can_override_terminate_with_replan():
    """The approver sees each proposed terminate; overriding it forces another planned pass."""
    provider = SequentialProvider([
        goal_response(),
        plan_response(), exec_response("partial"), critique_response(6),
        reflection_response(), memory_response(), control_response("terminate"),
        # Human rejected the terminate → planner runs again
        plan_response(), exec_success_response(), critique_response(9),
        reflection_response(), memory_response(), control_response("terminate"),
    ])
    seen = []
    prompts = []
    generate = provider.generate
    provider.generate = lambda system, user: (prompts.append(user), generate(system, user))[1]

    def approver(state, action):
        seen.append(action)
        return "replan" if len(seen) == 1 else action

    state, _ = Orchestrator(provider, max_iterations=5, approver=approver).run("Build a recommendation engine")

    assert seen == ["terminate", "terminate"]
    assert len(state.history) == 2
    assert state.human_overrides == ["Iteration 0: you proposed terminate, a human chose replan"]
    # The controller's next prompt carries the override as feedback.
    assert "a human chose replan" in prompts[-1]
    assert state.control_signal.action == "terminate"


def test_invalid_human_answer_keeps_controller_decision():
    provider = SequentialProvider([
        goal_response(), plan_response(), exec_success_response(), critique_response(9),
        reflection_response(), memory_response(), control_response("terminate"),
    ])
    state, _ = Orchestrator(provider, max_iterations=5, approver=lambda s, a: "explode").run("x")

    assert len(state.history) == 1
    assert state.control_signal.action == "terminate"
