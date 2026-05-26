import json
import pytest
from goal_interpreter import GoalInterpreter, GoalInterpretation
from goal_interpreter.providers.base import BaseLLMProvider


MOCK_RESPONSE = json.dumps({
    "goal": "Build a todo app",
    "subgoals": ["Design schema", "Build API", "Build UI"],
    "complexity": 4,
    "assumptions": ["Web-based", "Single user"],
    "uncertainty": 2,
    "initial_strategy": "REST API first, then frontend",
    "success_criteria": ["Tasks can be created and deleted", "UI reflects state"],
})


class MockProvider(BaseLLMProvider):
    def generate(self, system: str, user: str) -> str:
        return MOCK_RESPONSE


def test_interpret_returns_model():
    gi = GoalInterpreter(MockProvider())
    result = gi.interpret("Build me a todo app")
    assert isinstance(result, GoalInterpretation)
    assert result.complexity == 4
    assert len(result.subgoals) == 3


def test_interpret_bad_json_raises():
    class BadProvider(BaseLLMProvider):
        def generate(self, system: str, user: str) -> str:
            return "not json at all"

    gi = GoalInterpreter(BadProvider())
    with pytest.raises(ValueError, match="invalid JSON"):
        gi.interpret("Do something")
