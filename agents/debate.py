"""
DebatePanel — runs multiple agents independently on the same state,
measures disagreement, and synthesizes when disagreement exceeds a threshold.

Disagreement is computed as:
  - confidence_disagreement: std-dev of agent_confidence values
  - decision_disagreement: 1.0 if any two agents produced different actions
    (only meaningful for ControllerAgent panels), else 0.0

The synthesis prompt asks a third LLM call to reconcile diverging outputs.
"""
import json
import statistics
from copy import deepcopy

from pydantic import BaseModel

from core.state import AgentState, DebateResult, PanelistOutput
from core.logger import get_logger, log
from goal_interpreter.providers.base import BaseLLMProvider
from .base import BaseAgent

_SYNTHESIS_SYSTEM = """You are a meta-synthesis agent.

Multiple agents independently analyzed the same problem and reached different conclusions.
Your job is to synthesize their outputs into a single coherent recommendation.

Rules:
- Identify the root cause of disagreement.
- Weigh each output by the agent's stated confidence.
- Produce a synthesis that is MORE conservative than the highest-confidence output.
- Return ONLY a plain-text paragraph summarizing the synthesis and final recommendation."""


class DebatePanel:
    """
    Run `agents` independently on the same AgentState snapshot, then:
      1. Measure disagreement (confidence std-dev + structural differences).
      2. If disagreement > threshold, run a synthesis LLM call.
      3. Return a DebateResult that the orchestrator can log or act on.

    Args:
        agents:     List of BaseAgent instances (same or different types).
        threshold:  0-1 disagreement score above which synthesis is triggered.
        synth_provider: Optional dedicated provider for the synthesis call;
                        falls back to the first agent's provider.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        threshold: float = 0.25,
        synth_provider: BaseLLMProvider | None = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebatePanel requires at least 2 agents.")
        self._agents = agents
        self._threshold = threshold
        self._synth_provider = synth_provider or agents[0]._provider
        self._logger = get_logger("debate")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, state: AgentState, topic: str = "") -> DebateResult:
        log(self._logger, "info", "Debate starting",
            agents=[a.name for a in self._agents],
            topic=topic or "unknown",
            threshold=self._threshold)

        panelist_outputs: list[PanelistOutput] = []
        states_after: list[AgentState] = []

        for agent in self._agents:
            # Each agent gets its own independent snapshot — no cross-contamination
            snapshot = deepcopy(state)
            try:
                result_state = agent.run(snapshot)
                states_after.append(result_state)
                # Grab the confidence from the field this agent updated
                confidence = agent._get_confidence(result_state)
                # Serialize the updated field for comparison
                output_json = self._extract_output_json(agent, result_state)
                panelist_outputs.append(
                    PanelistOutput(
                        agent_name=agent.name,
                        output_json=output_json,
                        confidence=confidence,
                    )
                )
            except Exception as exc:
                log(self._logger, "warning", f"Panelist {agent.name} failed",
                    error=str(exc))
                panelist_outputs.append(
                    PanelistOutput(
                        agent_name=agent.name,
                        output_json="{}",
                        confidence=0.0,
                    )
                )

        disagreement = self._disagreement_score(panelist_outputs)
        synthesis = "Agents agree."
        if disagreement > self._threshold:
            log(self._logger, "info", "Disagreement exceeds threshold — synthesising",
                disagreement=round(disagreement, 3))
            synthesis = self._synthesize(panelist_outputs, topic)

        # Recommend the output with the highest confidence
        best_idx = max(
            range(len(panelist_outputs)),
            key=lambda i: panelist_outputs[i].confidence,
        )

        result = DebateResult(
            topic=topic,
            panelist_outputs=panelist_outputs,
            disagreement_score=round(disagreement, 3),
            synthesis=synthesis,
            recommended_output_index=best_idx,
            confidence=panelist_outputs[best_idx].confidence if panelist_outputs else 0.5,
        )

        log(self._logger, "info", "Debate complete",
            disagreement=result.disagreement_score,
            synthesis_triggered=(disagreement > self._threshold))
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_output_json(agent: BaseAgent, state: AgentState) -> str:
        """Return JSON of whichever state field the agent last wrote."""
        for attr in ("current_plan", "execution_result", "critique",
                     "reflection", "memory", "control_signal"):
            obj = getattr(state, attr, None)
            if obj is not None and hasattr(obj, "model_dump_json"):
                return obj.model_dump_json()
        return "{}"

    @staticmethod
    def _disagreement_score(outputs: list[PanelistOutput]) -> float:
        """
        0.0 = full agreement, 1.0 = maximum disagreement.
        Uses confidence std-dev as the primary signal.
        """
        confidences = [o.confidence for o in outputs if o.confidence > 0]
        if len(confidences) < 2:
            return 0.0
        return min(1.0, statistics.stdev(confidences) * 4)  # scale to 0-1 range

    def _synthesize(self, outputs: list[PanelistOutput], topic: str) -> str:
        user_msg = "\n\n".join([
            f"=== {o.agent_name} (confidence={o.confidence:.2f}) ===\n{o.output_json}"
            for o in outputs
        ])
        user_msg = f"Topic: {topic}\n\n{user_msg}\n\nSynthesize these diverging outputs."
        try:
            return self._synth_provider.generate(_SYNTHESIS_SYSTEM, user_msg)
        except Exception as exc:
            log(self._logger, "warning", "Synthesis call failed", error=str(exc))
            return f"Synthesis failed: {exc}"
