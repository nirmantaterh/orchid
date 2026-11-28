"""
The two systems being compared, run on the SAME model and the SAME prompt.

- baseline: one plain call to the model
- orchid:   the full Orchestrator loop

Both are wrapped in CountingProvider so we can report LLM calls and size.
"""
import time

from goal_interpreter.providers.base import BaseLLMProvider

BASELINE_SYSTEM = "You are a helpful assistant. Follow the user's instructions exactly."


class CountingProvider(BaseLLMProvider):
    """Wraps a provider and records how many calls were made and how big they were."""

    def __init__(self, inner: BaseLLMProvider) -> None:
        self._inner = inner
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.prompt_chars = 0
        self.output_chars = 0
        self.llm_seconds = 0.0

    def generate(self, system: str, user: str) -> str:
        t0 = time.monotonic()
        out = self._inner.generate(system, user)
        self.llm_seconds += time.monotonic() - t0
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        self.output_chars += len(out or "")
        return out

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_chars": self.prompt_chars,
            "output_chars": self.output_chars,
            "llm_seconds": round(self.llm_seconds, 2),
        }


def run_baseline(provider: CountingProvider, prompt: str) -> dict:
    text = provider.generate(BASELINE_SYSTEM, prompt) or ""
    return {"full_text": text, "final_text": text, "self_pass": None,
            "self_score": None, "iterations": 1, "final_action": None}


def run_orchid(provider: CountingProvider, prompt: str, max_iterations: int,
               enable_debate: bool = False) -> dict:
    """
    Orchid has no single 'final answer' field, so we record two views:
      full_text:  every step output of the last execution, joined in order
                  (used for ANSWER-line and code tasks: most lenient to Orchid)
      final_text: the last step's output (used for format tasks: what a user
                  would read as the reply)
    self_pass is Orchid's OWN verdict (the old benchmark metric), kept so we
    can measure how often it agrees with the external grade.
    """
    from core.orchestrator import Orchestrator
    from evaluation.metrics import compute_metrics

    orch = Orchestrator(provider, max_iterations=max_iterations,
                        enable_debate=enable_debate)
    state, _ = orch.run(prompt)

    er = state.execution_result
    outputs = [s.output for s in er.step_results if s.output] if er else []
    full_text = "\n\n".join(outputs) if outputs else (er.summary if er else "")
    final_text = outputs[-1] if outputs else (er.summary if er else "")

    m = compute_metrics(state)
    crit = next((r.critique for r in reversed(state.history) if r.critique), None) or state.critique
    return {
        "full_text": full_text,
        "final_text": final_text,
        "self_pass": bool(m.task_success),
        "self_score": crit.score if crit else None,
        "iterations": len(state.history),
        "final_action": state.control_signal.action,
    }
