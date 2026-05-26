import pytest
from pydantic import ValidationError
from goal_interpreter.schema import GoalInterpretation


VALID = {
    "goal": "Build a todo app",
    "subgoals": ["Set up backend", "Build UI"],
    "complexity": 5,
    "assumptions": ["User wants a web app"],
    "uncertainty": 3,
    "initial_strategy": "Start with a REST API then add frontend",
    "success_criteria": ["CRUD operations work", "UI renders tasks"],
}


def test_valid():
    m = GoalInterpretation(**VALID)
    assert m.complexity == 5


def test_complexity_out_of_range():
    with pytest.raises(ValidationError):
        GoalInterpretation(**{**VALID, "complexity": 11})


def test_uncertainty_negative():
    with pytest.raises(ValidationError):
        GoalInterpretation(**{**VALID, "uncertainty": -1})


def test_missing_field():
    data = {k: v for k, v in VALID.items() if k != "goal"}
    with pytest.raises(ValidationError):
        GoalInterpretation(**data)
