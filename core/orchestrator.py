from typing import Callable

from goal_interpreter import GoalInterpreter
from goal_interpreter.providers.base import BaseLLMProvider

from .state import AgentState
from .logger import get_logger, log
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent
from agents.reflection import ReflectionAgent
from agents.memory_manager import MemoryManagerAgent
from agents.controller import ControllerAgent
from agents.factory import AgentFactory
from agents.debate import DebatePanel
from tracing.tracer import Tracer
from tracing.trace_schema import MemoryUpdateRecord


# (state, proposed action) -> the action to take. Return the proposed action to approve it.
Approver = Callable[[AgentState, str], str]

HUMAN_ACTIONS = ("continue", "replan", "terminate")


def console_approver(state: AgentState, action: str) -> str:
    """Asks a person at the terminal to approve or override the controller."""
    signal = state.control_signal
    print(f"\nIteration {state.iteration}: controller proposes {action!r} ({signal.reason}), confidence {signal.agent_confidence:.2f}")
    answer = input("Enter to approve, or c / r / t to continue, replan or terminate: ").strip().lower()
    return {"c": "continue", "r": "replan", "t": "terminate"}.get(answer, action)


class Orchestrator:
    """
    Main execution loop.

    User Input → GoalInterpreter
      → [Planner → Executor → Critic → Reflection → MemoryManager → Controller] × N
    → terminate

    New capabilities:
      - Full Tracer integration (per-agent spans with confidence + latency)
      - AgentFactory for adaptive specialist creation
      - DebatePanel triggered when agent confidence variance is high
      - Automatic replan when confidence drops across consecutive iterations
      - Optional human approval: an approver confirms every terminate and any
        decision made below approval_threshold confidence
    """

    # Variance threshold above which debate is triggered
    DEBATE_VARIANCE_THRESHOLD = 0.04
    # Confidence drop threshold above which replan is forced
    CONFIDENCE_DROP_THRESHOLD = 0.15

    def __init__(
        self,
        provider: BaseLLMProvider,
        max_iterations: int = 10,
        enable_debate: bool = True,
        debate_threshold: float = 0.25,
        approver: Approver | None = None,
        approval_threshold: float = 0.5,
    ) -> None:
        self._provider = provider
        self._max_iterations = max_iterations
        self._enable_debate = enable_debate
        self._debate_threshold = debate_threshold
        self._approver = approver
        self._approval_threshold = approval_threshold
        self._logger = get_logger("orchestrator")

        self._interpreter = GoalInterpreter(provider)
        self._planner = PlannerAgent(provider)
        self._executor = ExecutorAgent(provider)
        self._critic = CriticAgent(provider)
        self._reflection = ReflectionAgent(provider)
        self._memory_manager = MemoryManagerAgent(provider)
        self._controller = ControllerAgent(provider)
        self._factory = AgentFactory(provider)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> tuple[AgentState, Tracer]:
        tracer = Tracer()
        log(self._logger, "info", "Orchestrator starting", input=user_input[:120])

        goal = self._interpreter.interpret(user_input)
        state = AgentState.initial(user_input, goal, self._max_iterations)
        tracer.set_goal(goal.goal)

        log(self._logger, "info", "Goal interpreted",
            goal=goal.goal, complexity=goal.complexity)

        while state.iteration < self._max_iterations:
            state, debate_invoked = self._run_iteration(state, tracer)

            action = state.control_signal.action
            log(self._logger, "info", "Controller decision",
                action=action, reason=state.control_signal.reason,
                iteration=state.iteration)

            # Confidence-based forced replan (overrides controller "continue")
            if (action == "continue"
                    and state.is_confidence_dropping(threshold=self.CONFIDENCE_DROP_THRESHOLD)):
                log(self._logger, "info",
                    "Confidence drop detected — forcing replan",
                    iteration=state.iteration)
                action = "replan"

            proposed = action
            action = self._ask_human(state, action)
            if action != proposed:
                # Kept on the state so the controller sees what a human rejected.
                state = state.model_copy(update={
                    "control_signal": state.control_signal.model_copy(
                        update={"action": action, "reason": f"Human override: {action}"}),
                    "human_overrides": [*state.human_overrides,
                                        f"Iteration {state.iteration}: you proposed {proposed}, a human chose {action}"],
                })

            # Record iteration in tracer
            memory_update = self._build_memory_update_record(state)
            tracer.close_iteration(
                iteration=state.iteration,
                agents_run=self._agents_run_this_cycle(state),
                control_decision=action,
                memory_update=memory_update,
                debate_invoked=debate_invoked,
            )

            if action == "terminate":
                log(self._logger, "info", "Terminating loop",
                    iteration=state.iteration)
                break

            if action == "replan":
                state = state.next_iteration(keep_plan=False)
            elif action == "specialist":
                state = self._run_specialist(state, tracer)
                state = state.next_iteration(keep_plan=True)
            else:
                state = state.next_iteration(keep_plan=True)

        # Force-terminate if we exhausted iterations without a clean terminate signal
        if state.control_signal.action != "terminate":
            from .state import ControlDecision
            state = state.model_copy(update={
                "control_signal": ControlDecision(
                    action="terminate",
                    reason=f"Max iterations ({self._max_iterations}) reached — forcing termination",
                    agent_confidence=state.control_signal.agent_confidence,
                )
            })
            log(self._logger, "info", "Forced terminate at max_iterations",
                iteration=state.iteration)

        log(self._logger, "info", "Orchestrator finished",
            iterations=state.iteration,
            final_action=state.control_signal.action)
        return state, tracer

    def _ask_human(self, state: AgentState, action: str) -> str:
        """Routes terminates and low-confidence decisions to the approver, if one is set."""
        if self._approver is None:
            return action
        if action != "terminate" and state.control_signal.agent_confidence >= self._approval_threshold:
            return action
        answer = self._approver(state, action)
        if answer not in HUMAN_ACTIONS:
            log(self._logger, "info", "Ignoring invalid approver answer", answer=answer)
            return action
        if answer != action:
            log(self._logger, "info", "Human override", proposed=action, chosen=answer)
        return answer

    # ── Iteration ─────────────────────────────────────────────────────────────

    def _run_iteration(
        self, state: AgentState, tracer: Tracer
    ) -> tuple[AgentState, bool]:
        log(self._logger, "info", "Iteration start", iteration=state.iteration)
        debate_invoked = False

        needs_plan = (
            state.current_plan is None
            or state.control_signal.action == "replan"
        )
        if needs_plan:
            state = self._planner.run(state, tracer)

        state = self._executor.run(state, tracer)

        # Debate: run two independent critics when confidence variance is high
        if self._enable_debate and self._should_debate(state):
            state, debate_invoked = self._run_debate(state)

        state = self._critic.run(state, tracer)
        state = self._reflection.run(state, tracer)
        state = self._memory_manager.run(state, tracer)
        state = self._controller.run(state, tracer)
        state = state.save_iteration()

        log(self._logger, "info", "Iteration end",
            iteration=state.iteration,
            critique_score=state.critique.score if state.critique else None,
            confidence_variance=round(state.confidence_variance(), 4))
        return state, debate_invoked

    # ── Debate ────────────────────────────────────────────────────────────────

    def _should_debate(self, state: AgentState) -> bool:
        """Trigger debate when confidence variance across agents is high."""
        return state.confidence_variance() > self.DEBATE_VARIANCE_THRESHOLD

    def _run_debate(self, state: AgentState) -> tuple[AgentState, bool]:
        log(self._logger, "info", "Triggering debate panel",
            iteration=state.iteration)
        panel = DebatePanel(
            agents=[
                CriticAgent(self._provider),
                CriticAgent(self._provider),
            ],
            threshold=self._debate_threshold,
            synth_provider=self._provider,
        )
        debate_result = panel.run(state, topic=f"Critique iteration {state.iteration}")
        state = state.add_debate(debate_result)
        return state, True

    # ── Specialist ────────────────────────────────────────────────────────────

    def _run_specialist(self, state: AgentState, tracer: Tracer) -> AgentState:
        specialist = self._factory.create_for_state(state)
        log(self._logger, "info", "Running specialist",
            type=specialist.name, iteration=state.iteration)
        return specialist.run(state, tracer)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_memory_update_record(state: AgentState) -> MemoryUpdateRecord | None:
        """Diff memory vs previous iteration to build a MemoryUpdateRecord."""
        if not state.history:
            return None
        prev = state.history[-1]
        prev_mem = prev.execution_result  # not memory — use trace instead

        # Simple approach: capture current memory snapshot
        return MemoryUpdateRecord(
            iteration=state.iteration,
            added_lessons=state.memory.lessons[-2:],
            added_failed_strategies=state.memory.failed_strategies[-2:],
            added_successful_patterns=state.memory.successful_patterns[-2:],
        )

    @staticmethod
    def _agents_run_this_cycle(state: AgentState) -> list[str]:
        """Derive agents run this cycle from the trace entries at current iteration."""
        return [
            e.agent for e in state.trace
            if e.iteration == state.iteration
        ]
