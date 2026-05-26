import json
import tempfile
import time
from pathlib import Path
import pytest
from tracing.tracer import Tracer, _estimate_tokens
from tracing.trace_schema import MemoryUpdateRecord


def test_estimate_tokens():
    assert _estimate_tokens("hello world") == max(1, len("hello world") // 4)


def test_span_records_entry():
    tracer = Tracer(run_id="test001")
    with tracer.agent_span("planner", "build a rest api", 0) as span:
        span.output_summary = "created plan with 3 steps"
        span.confidence = 0.85
        span.reasoning = "clear requirements"
    assert len(tracer._pending_entries) == 1
    entry = tracer._pending_entries[0]
    assert entry.agent_name == "planner"
    assert entry.confidence == 0.85
    assert entry.success is True
    assert entry.latency_ms >= 0


def test_span_records_failure():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.agent_span("executor", "run plan", 0) as span:
            raise ValueError("execution failed")
    assert len(tracer._pending_entries) == 1
    assert tracer._pending_entries[0].success is False
    assert "execution failed" in tracer._pending_entries[0].error


def test_close_iteration_moves_entries():
    tracer = Tracer()
    with tracer.agent_span("planner", "input", 0) as span:
        span.confidence = 0.9
    with tracer.agent_span("executor", "input", 0) as span:
        span.confidence = 0.7
    tracer.close_iteration(0, agents_run=["planner", "executor"],
                           control_decision="continue")
    assert len(tracer._closed_iterations) == 1
    assert len(tracer._pending_entries) == 0
    it = tracer._closed_iterations[0]
    assert it.overall_confidence == pytest.approx((0.9 + 0.7) / 2, abs=0.01)


def test_confidence_variance_in_iteration():
    tracer = Tracer()
    with tracer.agent_span("planner", "x", 0) as s:
        s.confidence = 0.9
    with tracer.agent_span("executor", "x", 0) as s:
        s.confidence = 0.2
    tracer.close_iteration(0, agents_run=[], control_decision="continue")
    it = tracer._closed_iterations[0]
    assert it.confidence_variance > 0


def test_build_returns_system_trace():
    tracer = Tracer(run_id="abc123")
    with tracer.agent_span("planner", "input", 0) as span:
        span.confidence = 0.8
    tracer.close_iteration(0, agents_run=["planner"], control_decision="terminate")
    trace = tracer.build(goal="Build something", final_action="terminate")
    assert trace.run_id == "abc123"
    assert trace.goal == "Build something"
    assert trace.final_action == "terminate"
    assert len(trace.iterations) == 1


def test_save_json(tmp_path):
    tracer = Tracer()
    with tracer.agent_span("planner", "input", 0) as span:
        span.confidence = 0.75
    tracer.close_iteration(0, agents_run=["planner"], control_decision="terminate")
    out = tmp_path / "trace.json"
    tracer.save_json(out)
    data = json.loads(out.read_text())
    assert "run_id" in data
    assert "iterations" in data


def test_save_sqlite(tmp_path):
    import sqlite3
    tracer = Tracer()
    with tracer.agent_span("planner", "input", 0) as span:
        span.confidence = 0.8
    tracer.close_iteration(0, agents_run=["planner"], control_decision="terminate")
    db = tmp_path / "trace.db"
    tracer.save_sqlite(db)
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT * FROM runs").fetchall()
    assert len(rows) == 1
    entries = con.execute("SELECT * FROM agent_entries").fetchall()
    assert len(entries) == 1
    con.close()


def test_confidence_trend():
    tracer = Tracer()
    for i in range(3):
        with tracer.agent_span("planner", "x", i) as s:
            s.confidence = 0.9 - i * 0.2
        tracer.close_iteration(i, agents_run=[], control_decision="continue")
    trend = tracer.confidence_trend(window=3)
    assert len(trend) == 3
    assert trend[0] > trend[-1]  # dropping


def test_is_confidence_dropping():
    tracer = Tracer()
    for i, conf in enumerate([0.9, 0.7, 0.5]):
        with tracer.agent_span("a", "x", i) as s:
            s.confidence = conf
        tracer.close_iteration(i, agents_run=[], control_decision="continue")
    assert tracer.is_confidence_dropping(window=3, threshold=0.1) is True


def test_memory_update_in_iteration():
    tracer = Tracer()
    update = MemoryUpdateRecord(
        iteration=0, added_lessons=["Use schema-first design"],
        added_failed_strategies=[], added_successful_patterns=[],
    )
    with tracer.agent_span("memory_manager", "x", 0) as s:
        s.confidence = 0.8
    tracer.close_iteration(0, agents_run=[], control_decision="continue",
                           memory_update=update)
    it = tracer._closed_iterations[0]
    assert it.memory_update is not None
    assert "Use schema-first design" in it.memory_update.added_lessons
