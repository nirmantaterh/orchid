import json
import pytest
from goal_interpreter.schema import GoalInterpretation
from goal_interpreter.providers.base import BaseLLMProvider
from core.state import AgentState, Plan, Step, ExecutionResult, StepResult, Critique
from agents.critic import CriticAgent
from agents.debate import DebatePanel


# ── Helpers ───────────────────────────────────────────────────────────────────

class StaticProvider(BaseLLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system: str, user: str) -> str:
        return self._response


class SequentialProvider(BaseLLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._it = iter(responses)

    def generate(self, system: str, user: str) -> str:
        return next(self._it)


GOAL = GoalInterpretation(
    goal="Build a REST API",
    subgoals=["Design", "Implement"],
    complexity=5,
    assumptions=[],
    uncertainty=3,
    initial_strategy="Schema-first",
    success_criteria=["API works"],
)

PLAN = Plan(
    steps=[Step(id="1", description="Design", expected_outcome="Done")],
    rationale="x", estimated_complexity=3,
)

EXEC = ExecutionResult(
    step_results=[StepResult(step_id="1", status="success", output="Done")],
    overall_status="success",
    summary="Completed",
)


def make_state() -> AgentState:
    return AgentState.initial("Build a REST API", GOAL).model_copy(
        update={"current_plan": PLAN, "execution_result": EXEC}
    )


def critique_json(score: int, conf: float = 0.8) -> str:
    return json.dumps({
        "score": score,
        "strengths": ["Good"],
        "weaknesses": [],
        "root_causes": [],
        "suggestions": [],
        "should_replan": False,
        "agent_confidence": conf,
        "reasoning": "test",
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_debate_panel_requires_two_agents():
    with pytest.raises(ValueError):
        DebatePanel(agents=[CriticAgent(StaticProvider("{}"))])


def test_debate_panel_agreement_no_synthesis():
    """Both agents produce same confidence → low disagreement → no synthesis call."""
    resp = critique_json(score=7, conf=0.8)
    panel = DebatePanel(
        agents=[CriticAgent(StaticProvider(resp)), CriticAgent(StaticProvider(resp))],
        threshold=0.25,
    )
    result = panel.run(make_state(), topic="critique agreement")
    assert result.disagreement_score < 0.25
    assert result.synthesis == "Agents agree."


def test_debate_panel_disagreement_triggers_synthesis():
    """Agents with very different confidences → synthesis LLM call."""
    high_conf = critique_json(score=9, conf=0.95)
    low_conf = critique_json(score=3, conf=0.1)
    synthesis_resp = "After weighing both critiques, recommend a replan."

    panel = DebatePanel(
        agents=[
            CriticAgent(StaticProvider(high_conf)),
            CriticAgent(StaticProvider(low_conf)),
        ],
        threshold=0.25,
        synth_provider=StaticProvider(synthesis_resp),
    )
    result = panel.run(make_state(), topic="critique disagreement")
    assert result.disagreement_score > 0.25
    assert "replan" in result.synthesis.lower()


def test_debate_result_recommends_highest_confidence():
    high = critique_json(score=8, conf=0.9)
    low = critique_json(score=4, conf=0.3)
    panel = DebatePanel(
        agents=[
            CriticAgent(StaticProvider(high)),
            CriticAgent(StaticProvider(low)),
        ],
        threshold=0.99,  # never synthesize
    )
    result = panel.run(make_state(), topic="")
    assert result.recommended_output_index == 0  # first agent had higher confidence


def test_debate_result_stored_in_state():
    resp = critique_json(7, 0.8)
    panel = DebatePanel(
        agents=[CriticAgent(StaticProvider(resp)), CriticAgent(StaticProvider(resp))],
        threshold=0.99,
    )
    state = make_state()
    debate = panel.run(state, "test topic")
    state2 = state.add_debate(debate)
    assert len(state2.debate_log) == 1
    assert state2.debate_log[0].topic == "test topic"


def test_failing_panelist_handled_gracefully():
    """If one panelist throws, the other still contributes."""
    class BrokenProvider(BaseLLMProvider):
        def generate(self, s, u):
            raise RuntimeError("LLM unavailable")

    good_resp = critique_json(7, 0.8)
    panel = DebatePanel(
        agents=[
            CriticAgent(BrokenProvider()),
            CriticAgent(StaticProvider(good_resp)),
        ],
        threshold=0.99,
    )
    result = panel.run(make_state(), "resilience test")
    assert len(result.panelist_outputs) == 2
    assert result.panelist_outputs[0].confidence == 0.0  # failed agent
    assert result.panelist_outputs[1].confidence == 0.8
