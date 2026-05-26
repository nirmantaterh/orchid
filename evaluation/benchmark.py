import json
import time
from pathlib import Path
from pydantic import BaseModel
from core.state import AgentState
from core.orchestrator import Orchestrator
from goal_interpreter.providers.base import BaseLLMProvider
from .metrics import RunMetrics, compute_metrics, aggregate_metrics


class BenchmarkCase(BaseModel):
    name: str
    input: str
    max_iterations: int = 5
    notes: str = ""


class BenchmarkResult(BaseModel):
    case: BenchmarkCase
    metrics: RunMetrics
    final_action: str
    iterations_run: int
    passed: bool
    error: str | None = None
    execution_output: str = ""


class Benchmark:
    """
    Runs an Orchestrator against a suite of test cases and collects metrics.

    `enable_debate` defaults to False so benchmark runs are deterministic
    (debate adds non-deterministic extra LLM calls).

    Usage:
        bench = Benchmark(provider)
        results = bench.run_suite([
            BenchmarkCase(name="simple", input="Build a todo API"),
            BenchmarkCase(name="complex", input="Build a real-time ML pipeline"),
        ])
        summary = bench.summary(results)
        bench.save_report(results, "report.json")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        max_iterations: int = 5,
        enable_debate: bool = False,
    ) -> None:
        self._provider = provider
        self._max_iterations = max_iterations
        self._enable_debate = enable_debate

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        orch = Orchestrator(self._provider, max_iterations=case.max_iterations,
                            enable_debate=self._enable_debate)
        t0 = time.monotonic()
        error: str | None = None
        state: AgentState | None = None

        try:
            state, _ = orch.run(case.input)
        except Exception as exc:
            error = str(exc)

        latency_ms = (time.monotonic() - t0) * 1000

        if state is None:
            metrics = RunMetrics(
                task_success=False,
                iterations_required=0,
                total_latency_ms=round(latency_ms, 2),
            )
            return BenchmarkResult(
                case=case,
                metrics=metrics,
                final_action="error",
                iterations_run=0,
                passed=False,
                error=error,
            )

        metrics = compute_metrics(state, total_latency_ms=latency_ms)
        exec_output = (
            state.execution_result.summary
            if state.execution_result else ""
        )
        return BenchmarkResult(
            case=case,
            metrics=metrics,
            final_action=state.control_signal.action,
            iterations_run=len(state.history),
            passed=metrics.task_success,
            error=error,
            execution_output=exec_output,
        )

    def run_suite(self, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
        results = []
        for case in cases:
            result = self.run_case(case)
            results.append(result)
        return results

    @staticmethod
    def summary(results: list[BenchmarkResult]) -> dict:
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        errored = [r for r in results if r.error]

        agg = aggregate_metrics([r.metrics for r in results])
        return {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "errored": len(errored),
            "pass_rate": len(passed) / max(1, len(results)),
            **agg,
        }

    @staticmethod
    def save_report(results: list[BenchmarkResult], path: str | Path) -> None:
        report = {
            "summary": Benchmark.summary(results),
            "cases": [r.model_dump() for r in results],
        }
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
