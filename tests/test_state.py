import pytest
from pydantic import ValidationError
from goal_interpreter.schema import GoalInterpretation
from core.state import (
    AgentState, Plan, Step, ExecutionResult, StepResult,
    Critique, Reflection, Memory, MemoryEntry, ControlDecision,
    extract_json,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

GOAL = GoalInterpretation(
    goal="Build a recommendation engine",
    subgoals=["Collect data", "Train model", "Serve API"],
    complexity=7,
    assumptions=["Python environment available"],
    uncertainty=4,
    initial_strategy="Collaborative filtering first",
    success_criteria=["Returns top-10 recs", "Latency < 200ms"],
)

PLAN = Plan(
    steps=[Step(id="1", description="Collect data", expected_outcome="Dataset ready")],
    rationale="Data first",
    estimated_complexity=3,
)


def make_state() -> AgentState:
    return AgentState.initial("Build a recommendation engine", GOAL, max_iterations=5)


# ── AgentState ────────────────────────────────────────────────────────────────

def test_initial_defaults():
    s = make_state()
    assert s.iteration == 0
    assert s.current_plan is None
    assert s.control_signal.action == "continue"
    assert s.history == []
    assert s.trace == []


def test_save_iteration_appends():
    s = make_state().model_copy(update={"current_plan": PLAN})
    s2 = s.save_iteration()
    assert len(s2.history) == 1
    assert s2.history[0].iteration == 0
    assert s2.history[0].plan == PLAN


def test_add_trace_appends():
    s = make_state()
    s2 = s.add_trace("planner", "created plan with 2 steps")
    assert len(s2.trace) == 1
    assert s2.trace[0].agent == "planner"


def test_next_iteration_clears_cycle_fields():
    critique = Critique(
        score=5, strengths=[], weaknesses=[], root_causes=[], suggestions=[],
        should_replan=False,
    )
    s = make_state().model_copy(update={"current_plan": PLAN, "critique": critique})
    s2 = s.next_iteration(keep_plan=False)
    assert s2.iteration == 1
    assert s2.current_plan is None
    assert s2.critique is None


def test_next_iteration_keep_plan():
    s = make_state().model_copy(update={"current_plan": PLAN})
    s2 = s.next_iteration(keep_plan=True)
    assert s2.current_plan == PLAN
    assert s2.iteration == 1


def test_critique_trend_empty():
    s = make_state()
    assert s.critique_trend() == []


def test_critique_trend_from_history():
    critique = Critique(
        score=6, strengths=[], weaknesses=[], root_causes=[], suggestions=[],
        should_replan=False,
    )
    s = make_state().model_copy(
        update={"current_plan": PLAN, "critique": critique}
    )
    s = s.save_iteration()
    assert s.critique_trend() == [6]


# ── Sub-models ────────────────────────────────────────────────────────────────

def test_plan_complexity_bounds():
    with pytest.raises(ValidationError):
        Plan(steps=[], rationale="x", estimated_complexity=11)


def test_critique_score_bounds():
    with pytest.raises(ValidationError):
        Critique(score=11, strengths=[], weaknesses=[], root_causes=[],
                 suggestions=[], should_replan=False)


def test_memory_defaults():
    m = Memory()
    assert m.failed_strategies == []
    assert m.lessons == []


def test_control_decision_valid_actions():
    for action in ("continue", "replan", "terminate", "specialist"):
        d = ControlDecision(action=action, reason="test")
        assert d.action == action


def test_control_decision_invalid_action():
    with pytest.raises(ValidationError):
        ControlDecision(action="unknown", reason="test")


# ── extract_json ──────────────────────────────────────────────────────────────

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_strips_fences():
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == '{"a": 1}'


def test_extract_json_strips_fences_no_lang():
    raw = '```\n{"a": 1}\n```'
    assert extract_json(raw) == '{"a": 1}'
