import json
import pytest
from goal_interpreter.schema import GoalInterpretation
from goal_interpreter.providers.base import BaseLLMProvider
from core.state import AgentState, ExecutionResult, StepResult, Critique, ControlDecision
from agents.factory import AgentFactory
from agents.specialist import SpecialistAgent
from agents.specialists import (
    SecuritySpecialist, CodeSpecialist, MathSpecialist,
    ResearchSpecialist, OptimizationSpecialist, DebuggingSpecialist,
    RetrievalSpecialist,
)


class FakeProvider(BaseLLMProvider):
    def generate(self, s, u):
        return "{}"


GOAL = GoalInterpretation(
    goal="Build system",
    subgoals=[], complexity=5, assumptions=[], uncertainty=3,
    initial_strategy="x", success_criteria=[],
)


def make_state(**overrides) -> AgentState:
    return AgentState.initial("Build system", GOAL).model_copy(update=overrides)


def exec_with_blockers(blockers: list[str]) -> ExecutionResult:
    return ExecutionResult(
        step_results=[StepResult(step_id="1", status="failure",
                                 output="failed", issues=blockers)],
        overall_status="failure",
        summary="failed",
        blockers=blockers,
    )


# ── Type map ──────────────────────────────────────────────────────────────────

def test_create_by_type_security():
    factory = AgentFactory(FakeProvider())
    agent = factory.create_by_type("security")
    assert isinstance(agent, SecuritySpecialist)
    assert agent.name == "specialist:security"


def test_create_by_type_code():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("code"), CodeSpecialist)


def test_create_by_type_math():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("math"), MathSpecialist)


def test_create_by_type_research():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("research"), ResearchSpecialist)


def test_create_by_type_optimization():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("optimization"), OptimizationSpecialist)


def test_create_by_type_debugging():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("debugging"), DebuggingSpecialist)


def test_create_by_type_retrieval():
    factory = AgentFactory(FakeProvider())
    assert isinstance(factory.create_by_type("retrieval"), RetrievalSpecialist)


def test_create_unknown_type_falls_back_to_generic():
    factory = AgentFactory(FakeProvider())
    agent = factory.create_by_type("quantum_chess")
    assert isinstance(agent, SpecialistAgent)
    assert agent.name == "specialist:quantum_chess"


# ── Inference ─────────────────────────────────────────────────────────────────

def test_infer_uses_controller_signal_first():
    ctrl = ControlDecision(action="specialist", reason="x", specialist_type="math")
    state = make_state(control_signal=ctrl)
    factory = AgentFactory(FakeProvider())
    agent = factory.create_for_state(state)
    assert isinstance(agent, MathSpecialist)


def test_infer_security_from_blocker_keyword():
    state = make_state(
        execution_result=exec_with_blockers(["JWT token validation failing"])
    )
    factory = AgentFactory(FakeProvider())
    agent = factory.create_for_state(state)
    assert isinstance(agent, SecuritySpecialist)


def test_infer_code_from_blocker_keyword():
    state = make_state(
        execution_result=exec_with_blockers(["syntax error in handler code: unexpected token"])
    )
    factory = AgentFactory(FakeProvider())
    agent = factory.create_for_state(state)
    assert isinstance(agent, CodeSpecialist)


def test_infer_optimization_from_blocker():
    state = make_state(
        execution_result=exec_with_blockers(["High latency in request handling"])
    )
    factory = AgentFactory(FakeProvider())
    agent = factory.create_for_state(state)
    assert isinstance(agent, OptimizationSpecialist)


def test_infer_defaults_to_debugging():
    state = make_state()
    factory = AgentFactory(FakeProvider())
    agent = factory.create_for_state(state)
    assert isinstance(agent, DebuggingSpecialist)
