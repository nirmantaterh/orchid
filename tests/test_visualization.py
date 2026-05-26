import pytest
from goal_interpreter.schema import GoalInterpretation
from core.state import (
    AgentState, Plan, Step, ExecutionResult, StepResult,
    Critique, Reflection, Memory, ControlDecision, IterationRecord,
)
from visualization.graph_builder import GraphBuilder, ExecutionGraph, GraphNode


# ── Helpers ───────────────────────────────────────────────────────────────────

GOAL = GoalInterpretation(
    goal="Build API", subgoals=[], complexity=4, assumptions=[],
    uncertainty=2, initial_strategy="x", success_criteria=[],
)

PLAN = Plan(
    steps=[Step(id="1", description="Design", expected_outcome="Done")],
    rationale="x", estimated_complexity=3,
)
EXEC = ExecutionResult(
    step_results=[StepResult(step_id="1", status="success", output="done")],
    overall_status="success", summary="ok",
)
CRITIQUE = Critique(
    score=8, strengths=[], weaknesses=[], root_causes=[], suggestions=[],
    should_replan=False,
)
CTRL_TERMINATE = ControlDecision(action="terminate", reason="done")
CTRL_REPLAN = ControlDecision(action="replan", reason="failed")
CTRL_SPECIALIST = ControlDecision(action="specialist", reason="blocked",
                                  specialist_type="code")
CTRL_CONTINUE = ControlDecision(action="continue", reason="ok")


def make_state_with_history(decisions: list[ControlDecision]) -> AgentState:
    state = AgentState.initial("Build API", GOAL)
    for i, ctrl in enumerate(decisions):
        record = IterationRecord(
            iteration=i,
            plan=PLAN,
            execution_result=EXEC,
            critique=CRITIQUE,
            control_decision=ctrl,
            agent_confidences={"planner": 0.8, "executor": 0.7},
        )
        state = state.model_copy(update={
            "history": state.history + [record],
            "control_signal": ctrl,
        })
    return state


# ── GraphBuilder ──────────────────────────────────────────────────────────────

def test_from_state_single_terminate():
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    node_ids = {n.id for n in graph.nodes}
    assert "user_input" in node_ids
    assert "goal_interpreter" in node_ids
    assert "i0_controller" in node_ids
    assert "i0_terminal" in node_ids


def test_from_state_has_edges():
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    assert len(graph.edges) > 0
    sources = {e.source for e in graph.edges}
    assert "user_input" in sources


def test_from_state_replan_edge_label():
    state = make_state_with_history([CTRL_REPLAN])
    graph = GraphBuilder.from_state(state)
    replan_edges = [e for e in graph.edges if e.label == "replan"]
    assert len(replan_edges) >= 1


def test_from_state_specialist_node():
    state = make_state_with_history([CTRL_SPECIALIST])
    graph = GraphBuilder.from_state(state)
    node_ids = {n.id for n in graph.nodes}
    assert any("specialist_code" in nid for nid in node_ids)


def test_from_state_multi_iteration():
    state = make_state_with_history([CTRL_CONTINUE, CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    agent_nodes = [n for n in graph.nodes if n.node_type == "agent"]
    # Each iteration adds planner + executor + reflection + memory nodes
    assert len(agent_nodes) >= 4


# ── Mermaid output ────────────────────────────────────────────────────────────

def test_mermaid_output_starts_with_flowchart():
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    mermaid = graph.to_mermaid()
    assert mermaid.startswith("flowchart TD")


def test_mermaid_contains_node_ids():
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    mermaid = graph.to_mermaid()
    assert "user_input" in mermaid
    assert "goal_interpreter" in mermaid


def test_mermaid_contains_edges():
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    mermaid = graph.to_mermaid()
    assert "-->" in mermaid


def test_mermaid_node_shapes():
    graph = ExecutionGraph(
        nodes=[
            GraphNode(id="input", label="Input", node_type="input"),
            GraphNode(id="agent", label="Agent", node_type="agent"),
            GraphNode(id="dec", label="Decision", node_type="decision"),
            GraphNode(id="term", label="End", node_type="terminal"),
        ],
        edges=[],
    )
    mermaid = graph.to_mermaid()
    assert 'input["Input"]' in mermaid
    assert 'agent("Agent")' in mermaid
    assert 'dec{"Decision"}' in mermaid
    assert 'term(["End"])' in mermaid


def test_mermaid_confidence_in_label():
    graph = ExecutionGraph(
        nodes=[GraphNode(id="p", label="Planner", node_type="agent",
                         iteration=0, confidence=0.85)],
        edges=[],
    )
    mermaid = graph.to_mermaid()
    assert "conf=0.85" in mermaid


# ── networkx (optional) ───────────────────────────────────────────────────────

def test_networkx_raises_if_not_installed(monkeypatch):
    import visualization.graph_builder as gb
    monkeypatch.setattr(gb, "_HAS_NX", False)
    state = make_state_with_history([CTRL_TERMINATE])
    graph = GraphBuilder.from_state(state)
    with pytest.raises(ImportError, match="networkx"):
        graph.to_networkx()
