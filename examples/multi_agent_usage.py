"""
End-to-end example: adaptive multi-agent system with a real LLM.

Usage:
    set ANTHROPIC_API_KEY=your_key_here
    python examples/multi_agent_usage.py
"""
import os
import json
from goal_interpreter.providers.anthropic import AnthropicProvider
from core.orchestrator import Orchestrator

provider = AnthropicProvider(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
)

orch = Orchestrator(provider, max_iterations=4)
state = orch.run(
    "Build a real-time fraud detection system for an e-commerce platform"
)

print("\n=== FINAL STATE ===")
print(f"Iterations: {state.iteration + 1}")
print(f"Final action: {state.control_signal.action}")
print(f"Reason: {state.control_signal.reason}")

print("\n=== MEMORY ===")
print(state.memory.model_dump_json(indent=2))

print("\n=== ITERATION HISTORY (scores) ===")
for record in state.history:
    score = record.critique.score if record.critique else "N/A"
    action = record.control_decision.action if record.control_decision else "N/A"
    print(f"  Iteration {record.iteration}: score={score}, action={action}")

print("\n=== TRACE ===")
for entry in state.trace:
    print(f"  [{entry.iteration}] {entry.agent}: {entry.summary[:80]}")
