import json
import re
from .providers.base import BaseLLMProvider
from .schema import GoalInterpretation
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _clean_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    text = re.sub(r'(?<=[{,])\s*(\d+)\s*:', lambda x: f' "{x.group(1).strip()}":', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        json.loads(text)
    except json.JSONDecodeError:
        last = text.rfind("}")
        if last != -1:
            text = text[:last + 1]
    return text


class GoalInterpreter:
    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    def interpret(self, user_input: str) -> GoalInterpretation:
        raw = self._provider.generate(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(user_input),
        )
        cleaned = _clean_json(raw)
        try:
            try:
                return GoalInterpretation.model_validate_json(cleaned)
            except Exception:
                return GoalInterpretation.model_validate(json.loads(cleaned))
        except Exception:
            # Partial-JSON fallback: extract "goal" via regex and use defaults for the rest
            goal_match = re.search(r'"goal"\s*:\s*"([^"]+)"', raw)
            if goal_match:
                return GoalInterpretation(goal=goal_match.group(1))
            raise ValueError(f"LLM returned invalid JSON.\nRaw response:\n{raw}")
