import json
from core.state import AgentState, Memory, parse_model
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYSTEM = """You are a Memory Manager in an adaptive multi-agent system.

Your job:
- Integrate new reflections and critique signals into the persistent memory store.
- Consolidate and deduplicate — never add something already captured.
- Promote successful patterns, demote or remove entries that proved wrong.
- Keep memory lean: prune entries that are redundant or no longer relevant.

Output ONLY valid JSON — the COMPLETE updated memory state:
{
  "failed_strategies": ["Strategies that have been tried and failed"],
  "successful_patterns": ["Approaches that yielded positive results"],
  "lessons": ["Durable facts the system has learned"],
  "entries": [
    {"pattern": "...", "outcome": "success" | "failure", "frequency": 1, "last_iteration": 0}
  ],
  "agent_confidence": 0.8,
  "reasoning": "One sentence explaining confidence in this memory update"
}

Rules:
- This is a FULL replacement of memory, not a patch — include everything to keep.
- If a lesson is already in the existing memory verbatim, keep it as-is (don't rephrase).
- failed_strategies: only add if critique score was ≤ 5 and the approach was specific.
- successful_patterns: only add if critique score ≥ 7 and the pattern was the reason.
- entries.frequency should increment if the pattern was seen before.
- Return ONLY the JSON object."""


class MemoryManagerAgent(BaseAgent):
    name = "memory_manager"
    _output_field = "memory"

    def __init__(self, provider: BaseLLMProvider) -> None:
        super().__init__(provider)

    def _build_system_prompt(self) -> str:
        return _SYSTEM

    def _build_user_prompt(self, state: AgentState) -> str:
        assert state.reflection is not None
        assert state.critique is not None

        return "\n".join([
            f"ITERATION: {state.iteration}",
            f"CRITIQUE SCORE: {state.critique.score}/10",
            "",
            "=== CURRENT MEMORY (update this) ===",
            state.memory.model_dump_json(indent=2),
            "",
            "=== NEW REFLECTION ===",
            state.reflection.model_dump_json(indent=2),
            "",
            "=== NEW CRITIQUE SIGNALS ===",
            f"Root causes: {json.dumps(state.critique.root_causes)}",
            f"Suggestions: {json.dumps(state.critique.suggestions)}",
            f"Should replan: {state.critique.should_replan}",
            "",
            "Update memory now. Return the complete updated memory as valid JSON.",
        ])

    def _apply(self, raw: str, state: AgentState) -> AgentState:
        memory = parse_model(Memory, raw)
        return state.model_copy(update={"memory": memory})
