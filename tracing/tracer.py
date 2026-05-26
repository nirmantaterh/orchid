import json
import sqlite3
import statistics
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from .trace_schema import (
    AgentTraceEntry,
    IterationTrace,
    MemoryUpdateRecord,
    SystemTrace,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


class SpanContext:
    """Mutable context yielded by Tracer.agent_span so the caller can set fields."""

    def __init__(self) -> None:
        self.output_summary: str = ""
        self.reasoning: str = ""
        self.confidence: float = 0.5
        self.token_usage: int = 0


class Tracer:
    """
    Collects per-agent execution data and builds a full SystemTrace.

    Usage:
        tracer = Tracer()
        with tracer.agent_span("planner", "Build REST API", 0) as span:
            result = run_planner(...)
            span.output_summary = str(result)[:200]
            span.confidence = result.agent_confidence

        tracer.close_iteration(0, agents_run=["planner"], control_decision="continue")
        trace = tracer.build(goal="...", final_action="terminate")
    """

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id or uuid.uuid4().hex[:8]
        self._start_time = time.monotonic()
        self._created_at = _now()
        self._goal: str = ""

        self._pending_entries: list[AgentTraceEntry] = []
        self._closed_iterations: list[IterationTrace] = []
        self._total_tokens: int = 0

    def set_goal(self, goal: str) -> None:
        self._goal = goal

    # ── Span context manager ──────────────────────────────────────────────────

    @contextmanager
    def agent_span(
        self, agent_name: str, input_summary: str, iteration: int
    ) -> Generator[SpanContext, None, None]:
        span = SpanContext()
        t0 = time.monotonic()
        error: str | None = None
        success = True

        try:
            yield span
        except Exception as exc:
            error = str(exc)
            success = False
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            tokens = span.token_usage or _estimate_tokens(
                input_summary + span.output_summary
            )
            self._total_tokens += tokens
            entry = AgentTraceEntry(
                agent_name=agent_name,
                iteration=iteration,
                input_summary=input_summary[:300],
                output_summary=span.output_summary[:300],
                reasoning=span.reasoning,
                confidence=span.confidence,
                latency_ms=round(latency_ms, 2),
                token_usage=tokens,
                success=success,
                error=error,
                timestamp=_now(),
            )
            self._pending_entries.append(entry)

    # ── Iteration lifecycle ───────────────────────────────────────────────────

    def close_iteration(
        self,
        iteration: int,
        agents_run: list[str],
        control_decision: str,
        memory_update: MemoryUpdateRecord | None = None,
        debate_invoked: bool = False,
    ) -> None:
        entries = [e for e in self._pending_entries if e.iteration == iteration]
        confidences = [e.confidence for e in entries if e.success]

        overall_conf = statistics.mean(confidences) if confidences else 0.0
        conf_var = statistics.variance(confidences) if len(confidences) > 1 else 0.0

        trace = IterationTrace(
            iteration=iteration,
            agents_run=agents_run,
            entries=entries,
            control_decision=control_decision,
            overall_confidence=round(overall_conf, 3),
            confidence_variance=round(conf_var, 4),
            memory_update=memory_update,
            debate_invoked=debate_invoked,
            timestamp=_now(),
        )
        self._closed_iterations.append(trace)
        # Remove processed entries
        self._pending_entries = [
            e for e in self._pending_entries if e.iteration != iteration
        ]

    # ── Build and persist ─────────────────────────────────────────────────────

    def build(self, goal: str = "", final_action: str = "") -> SystemTrace:
        goal = goal or self._goal
        total_ms = (time.monotonic() - self._start_time) * 1000
        return SystemTrace(
            run_id=self._run_id,
            goal=goal,
            total_iterations=len(self._closed_iterations),
            total_latency_ms=round(total_ms, 2),
            total_tokens=self._total_tokens,
            final_action=final_action,
            iterations=self._closed_iterations,
            created_at=self._created_at,
        )

    def save_json(self, path: str | Path) -> None:
        trace = self.build()
        Path(path).write_text(
            trace.model_dump_json(indent=2), encoding="utf-8"
        )

    def save_sqlite(self, path: str | Path) -> None:
        trace = self.build()
        con = sqlite3.connect(str(path))
        cur = con.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                goal TEXT,
                total_iterations INTEGER,
                total_latency_ms REAL,
                total_tokens INTEGER,
                final_action TEXT,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                agent_name TEXT,
                iteration INTEGER,
                confidence REAL,
                latency_ms REAL,
                token_usage INTEGER,
                success INTEGER,
                error TEXT,
                reasoning TEXT,
                timestamp TEXT
            )
        """)

        cur.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
            (
                trace.run_id, trace.goal, trace.total_iterations,
                trace.total_latency_ms, trace.total_tokens,
                trace.final_action, trace.created_at,
            ),
        )
        for it in trace.iterations:
            for entry in it.entries:
                cur.execute(
                    "INSERT INTO agent_entries "
                    "(run_id,agent_name,iteration,confidence,latency_ms,"
                    " token_usage,success,error,reasoning,timestamp) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        trace.run_id, entry.agent_name, entry.iteration,
                        entry.confidence, entry.latency_ms, entry.token_usage,
                        int(entry.success), entry.error, entry.reasoning,
                        entry.timestamp,
                    ),
                )
        con.commit()
        con.close()

    # ── Confidence analytics ──────────────────────────────────────────────────

    def confidence_trend(self, window: int = 5) -> list[float]:
        return [
            it.overall_confidence
            for it in self._closed_iterations[-window:]
        ]

    def is_confidence_dropping(self, window: int = 3, threshold: float = 0.1) -> bool:
        trend = self.confidence_trend(window)
        if len(trend) < 2:
            return False
        return (trend[0] - trend[-1]) > threshold
