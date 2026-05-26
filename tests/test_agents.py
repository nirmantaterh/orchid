"""
Unit tests for each agent using a StaticProvider (no real LLM calls).
Each test verifies that the agent:
  1. Parses a valid LLM response correctly.
  2. Returns an updated AgentState with the expected field set.
  3. Raises ValueError on unparseable responses.
"""
import json
import pytest
from goal_interpreter.schema import GoalInterpretation
from goal_interpreter.providers.base import BaseLLMProvider
from core.state import (
    AgentState, Plan, Step, ExecutionResult, StepResult,
    Critique, Reflection, Memory, ControlDecision,
)
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent
from agents.reflection import ReflectionAgent
from agents.memory_manager import MemoryManagerAgent
from agents.controller import ControllerAgent
from agents.specialist import SpecialistAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

class StaticProvider(BaseLLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system: str, user: str) -> str:
        return self._response


GOAL = GoalInterpretation(
    goal="Build a REST API",
    subgoals=["Design endpoints", "Implement handlers", "Write tests"],
    complexity=5,
    assumptions=["Python + FastAPI"],
    uncertainty=3,
    initial_strategy="Schema-first design",
    success_criteria=["All endpoints return correct status codes"],
)

PLAN = Plan(
    steps=[
        Step(id="1", description="Design schema", expected_outcome="Schema ready"),
        Step(id="2", description="Implement handlers", dependencies=["1"],
             expected_outcome="Handlers ready"),
    ],
    rationale="Schema-first ensures consistency",
    estimated_complexity=4,
    avoids=[],
)

EXECUTION_RESULT = ExecutionResult(
    step_results=[
        StepResult(step_id="1", status="success", output="Schema designed"),
        StepResult(step_id="2", status="partial", output="Handler incomplete",
                   issues=["Missing auth middleware"]),
    ],
    overall_status="partial",
    summary="Schema done, handler incomplete",
    blockers=["Auth middleware not implemented"],
)

CRITIQUE = Critique(
    score=6,
    strengths=["Schema well-defined"],
    weaknesses=["Auth not implemented"],
    root_causes=["Auth was not included in plan scope"],
    suggestions=["Add auth step to plan"],
    should_replan=False,
)

REFLECTION = Reflection(
    lessons_learned=["Auth must be scoped explicitly"],
    strategy_updates=["Always include auth as an explicit step"],
    patterns_identified=["Auth is often forgotten in early iterations"],
    confidence=7,
)


def base_state(**overrides) -> AgentState:
    return AgentState.initial("Build a REST API", GOAL).model_copy(update=overrides)


# ── PlannerAgent ──────────────────────────────────────────────────────────────

PLAN_RESPONSE = json.dumps({
    "steps": [
        {"id": "1", "description": "Design endpoints", "dependencies": [],
         "expected_outcome": "Endpoint spec ready"},
    ],
    "rationale": "Top-down design",
    "estimated_complexity": 4,
    "avoids": [],
})


def test_planner_sets_current_plan():
    agent = PlannerAgent(StaticProvider(PLAN_RESPONSE))
    state = base_state()
    result = agent.run(state)
    assert result.current_plan is not None
    assert result.current_plan.steps[0].id == "1"


def test_planner_bad_json_non_fatal():
    agent = PlannerAgent(StaticProvider("not json"))
    result = agent.run(base_state())
    # Parse failure must not raise; state is returned unchanged (no plan written)
    assert result.current_plan is None
    assert any("PARSE_ERROR" in t.summary for t in result.trace)


# ── ExecutorAgent ─────────────────────────────────────────────────────────────

EXEC_RESPONSE = json.dumps({
    "step_results": [
        {"step_id": "1", "status": "success", "output": "Done", "issues": []},
    ],
    "overall_status": "success",
    "summary": "All steps completed",
    "blockers": [],
})


def test_executor_sets_execution_result():
    agent = ExecutorAgent(StaticProvider(EXEC_RESPONSE))
    state = base_state(current_plan=PLAN)
    result = agent.run(state)
    assert result.execution_result is not None
    assert result.execution_result.overall_status == "success"


def test_executor_bad_json_non_fatal():
    agent = ExecutorAgent(StaticProvider("{}"))  # missing required fields
    result = agent.run(base_state(current_plan=PLAN))
    assert result.execution_result is None
    assert any("PARSE_ERROR" in t.summary for t in result.trace)


# ── CriticAgent ───────────────────────────────────────────────────────────────

CRITIQUE_RESPONSE = json.dumps({
    "score": 7,
    "strengths": ["Clear schema"],
    "weaknesses": ["Auth missing"],
    "root_causes": ["Scope gap"],
    "suggestions": ["Add auth step"],
    "should_replan": False,
})


def test_critic_sets_critique():
    agent = CriticAgent(StaticProvider(CRITIQUE_RESPONSE))
    state = base_state(current_plan=PLAN, execution_result=EXECUTION_RESULT)
    result = agent.run(state)
    assert result.critique is not None
    assert result.critique.score == 7


def test_critic_bad_json_non_fatal():
    agent = CriticAgent(StaticProvider("bad"))
    result = agent.run(base_state(current_plan=PLAN, execution_result=EXECUTION_RESULT))
    assert result.critique is None
    assert any("PARSE_ERROR" in t.summary for t in result.trace)


# ── ReflectionAgent ───────────────────────────────────────────────────────────

REFLECTION_RESPONSE = json.dumps({
    "lessons_learned": ["Always include auth"],
    "strategy_updates": ["Scope auth explicitly"],
    "patterns_identified": ["Auth omissions recur"],
    "confidence": 8,
})


def test_reflection_sets_reflection():
    agent = ReflectionAgent(StaticProvider(REFLECTION_RESPONSE))
    state = base_state(execution_result=EXECUTION_RESULT, critique=CRITIQUE)
    result = agent.run(state)
    assert result.reflection is not None
    assert result.reflection.confidence == 8


# ── MemoryManagerAgent ────────────────────────────────────────────────────────

MEMORY_RESPONSE = json.dumps({
    "failed_strategies": [],
    "successful_patterns": ["Schema-first works"],
    "lessons": ["Always include auth"],
    "entries": [
        {"pattern": "schema_first", "outcome": "success", "frequency": 1,
         "last_iteration": 0}
    ],
})


def test_memory_manager_updates_memory():
    agent = MemoryManagerAgent(StaticProvider(MEMORY_RESPONSE))
    state = base_state(critique=CRITIQUE, reflection=REFLECTION)
    result = agent.run(state)
    assert "Always include auth" in result.memory.lessons
    assert len(result.memory.entries) == 1


# ── ControllerAgent ───────────────────────────────────────────────────────────

CONTROL_CONTINUE = json.dumps({
    "action": "continue",
    "reason": "Progress is being made",
    "specialist_type": None,
})

CONTROL_TERMINATE = json.dumps({
    "action": "terminate",
    "reason": "Goal achieved",
    "specialist_type": None,
})

CONTROL_SPECIALIST = json.dumps({
    "action": "specialist",
    "reason": "Auth requires specialist",
    "specialist_type": "security",
})


def test_controller_continue():
    agent = ControllerAgent(StaticProvider(CONTROL_CONTINUE))
    state = base_state(
        execution_result=EXECUTION_RESULT, critique=CRITIQUE, reflection=REFLECTION
    )
    result = agent.run(state)
    assert result.control_signal.action == "continue"


def test_controller_terminate():
    agent = ControllerAgent(StaticProvider(CONTROL_TERMINATE))
    state = base_state(
        execution_result=EXECUTION_RESULT, critique=CRITIQUE, reflection=REFLECTION
    )
    result = agent.run(state)
    assert result.control_signal.action == "terminate"


def test_controller_specialist_sets_type():
    agent = ControllerAgent(StaticProvider(CONTROL_SPECIALIST))
    state = base_state(
        execution_result=EXECUTION_RESULT, critique=CRITIQUE, reflection=REFLECTION
    )
    result = agent.run(state)
    assert result.control_signal.action == "specialist"
    assert result.control_signal.specialist_type == "security"


# ── SpecialistAgent ───────────────────────────────────────────────────────────

SPECIALIST_RESPONSE = json.dumps({
    "step_results": [
        {"step_id": "2", "status": "success",
         "output": "Auth implemented via JWT", "issues": []}
    ],
    "overall_status": "success",
    "summary": "Security specialist resolved auth blocker",
    "blockers": [],
})


def test_specialist_sets_execution_result():
    agent = SpecialistAgent(StaticProvider(SPECIALIST_RESPONSE), "security")
    state = base_state(current_plan=PLAN, execution_result=EXECUTION_RESULT)
    result = agent.run(state)
    assert result.execution_result.overall_status == "success"
    assert result.execution_result.step_results[0].output == "Auth implemented via JWT"


def test_specialist_name_includes_type():
    agent = SpecialistAgent(StaticProvider("{}"), "debugging")
    assert agent.name == "specialist:debugging"


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_agent_adds_trace_entry():
    agent = PlannerAgent(StaticProvider(PLAN_RESPONSE))
    state = base_state()
    result = agent.run(state)
    assert any(t.agent == "planner" for t in result.trace)
