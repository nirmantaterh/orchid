"""
Fake model for dry runs. Answers each agent role with valid JSON and gets
the task right about 60% of the time, so the whole pipeline (runner,
graders, report) can be tested without Ollama. Its numbers mean nothing.
"""
import json
import random

from goal_interpreter.providers.base import BaseLLMProvider
from eval.reference import reference_answer
from eval.tasks import TASKS


def _find_task(user: str):
    for t in TASKS:
        if t["prompt"] in user:
            return t
    return None


def _wrong(task: dict) -> str:
    k = task["kind"]
    if k == "number":
        return f"ANSWER: {task['answer'] + 1}"
    if k == "text":
        return "ANSWER: I am not sure"
    if k in ("code", "code_script"):
        return f"```python\ndef {task['func']}(*args):\n    return None\n```"
    return "Sure! Here is my answer:\n" + reference_answer(task)


class FakeProvider(BaseLLMProvider):
    def __init__(self, seed: int = 0, p_correct: float = 0.6) -> None:
        self._rng = random.Random(seed)
        self._p = p_correct

    def _answer(self, user: str) -> str:
        task = _find_task(user)
        if task is None:
            return "ANSWER: unknown"
        return reference_answer(task) if self._rng.random() < self._p else _wrong(task)

    def generate(self, system: str, user: str) -> str:
        s = system[:80]
        if "helpful assistant" in s:
            return self._answer(user)
        if "Controller" in s:
            action = self._rng.choice(["terminate", "terminate", "continue", "replan"])
            return json.dumps({"action": action, "reason": "fake", "specialist_type": None,
                               "agent_confidence": 0.7, "reasoning": "fake"})
        if "Critic" in s:
            score = self._rng.randint(4, 9)
            return json.dumps({"score": score, "strengths": ["ok"], "weaknesses": ["fake"],
                               "root_causes": [], "suggestions": [], "should_replan": score <= 4,
                               "agent_confidence": 0.7, "reasoning": "fake"})
        if "Planner" in s:
            return json.dumps({"steps": [{"id": "1", "description": "Answer the question",
                                          "dependencies": [], "expected_outcome": "answer"}],
                               "rationale": "fake", "estimated_complexity": 2, "avoids": [],
                               "agent_confidence": 0.7, "reasoning": "fake"})
        if "Executor" in s or "specialist" in s.lower():
            return json.dumps({"step_results": [{"step_id": "1", "status": "success",
                                                 "output": self._answer(user), "issues": []}],
                               "overall_status": "success", "summary": "fake run",
                               "blockers": [], "agent_confidence": 0.7, "reasoning": "fake"})
        if "Reflection" in s:
            return json.dumps({"lessons_learned": ["fake"], "strategy_updates": [],
                               "patterns_identified": [], "confidence": 7})
        if "Memory" in s:
            return json.dumps({"failed_strategies": [], "successful_patterns": ["fake"],
                               "lessons": ["fake"], "entries": []})
        # Goal interpreter (and anything else)
        return json.dumps({"goal": "Answer the user's question", "subgoals": ["answer"],
                           "complexity": 2, "assumptions": [], "uncertainty": 2,
                           "initial_strategy": "direct", "success_criteria": ["correct answer"]})
