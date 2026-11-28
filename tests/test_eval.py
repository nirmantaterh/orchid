"""Tests for the eval-v2 harness: graders, task set, and a fake end-to-end run."""
import json

import pytest

from eval.graders import extract_answer_line, extract_code, grade, parse_number
from eval.reference import ALL_REFERENCES
from eval.tasks import TASKS, TASKS_BY_ID


# ── Task set sanity ───────────────────────────────────────────────────────────

def test_task_ids_unique():
    assert len(TASKS) == len(TASKS_BY_ID)


def test_public_benchmarks_loaded():
    from collections import Counter
    c = Counter(t["category"] for t in TASKS)
    assert c["gsm8k"] == 50 and c["humaneval"] == 50 and c["mbpp"] == 40
    assert all(t["func"] for t in TASKS if t["kind"] == "code_script")


def test_code_script_rejects_wrong_solution():
    t = next(t for t in TASKS if t["category"] == "humaneval")
    ok, _ = grade(t, f"```python\ndef {t['func']}(*a, **k):\n    return None\n```", "")
    assert not ok


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_reference_answer_passes(task):
    """A known-correct answer must pass every grader. Guards against wrong answer keys."""
    ref = ALL_REFERENCES[task["id"]]
    ok, detail = grade(task, ref, ref)
    assert ok, detail


# ── Extraction ────────────────────────────────────────────────────────────────

def test_last_answer_line_wins():
    assert extract_answer_line("ANSWER: 3\nwait, no\nANSWER: **4**.") == "4"


def test_missing_answer_line():
    assert extract_answer_line("the answer is 4") is None


@pytest.mark.parametrize("s,expected", [
    ("$1,157.63", 1157.63), ("5/36", 5 / 36), ("about 50 km/h", 50.0),
    ("-3", -3.0), ("0.05 dollars", 0.05), ("none", None),
])
def test_parse_number(s, expected):
    got = parse_number(s)
    assert (got is None and expected is None) or got == pytest.approx(expected)


def test_extract_code_prefers_block_with_def():
    text = "```python\nprint(1)\n```\n```python\ndef f(x):\n    return x\n```"
    assert "def f" in extract_code(text, "f")


# ── Graders reject wrong answers ──────────────────────────────────────────────

def test_number_wrong():
    ok, _ = grade(TASKS_BY_ID["math-01"], "ANSWER: 30", "")
    assert not ok


def test_text_reject_list_blocks_hedged_wrong_answer():
    # Contains the right token, but the stated answer is the wrong one
    ok, _ = grade(TASKS_BY_ID["logic-12"], "ANSWER: 9.11 is larger than 9.9", "")
    assert not ok


def test_text_word_boundary():
    # "2" must not match inside "12"
    ok, _ = grade(TASKS_BY_ID["facts-10"], "ANSWER: 12", "")
    assert not ok


def test_code_wrong_and_crashing():
    t = TASKS_BY_ID["code-09"]
    assert not grade(t, "```python\ndef fib(n):\n    return n\n```", "")[0]
    assert not grade(t, "```python\ndef fib(n):\n    raise ValueError()\n```", "")[0]


def test_code_infinite_loop_times_out():
    t = TASKS_BY_ID["code-13"]
    ok, detail = grade(t, "```python\ndef gcd(a, b):\n    while True:\n        pass\n```", "")
    assert not ok and "timed out" in detail


def test_format_extra_text_fails():
    t = TASKS_BY_ID["format-02"]
    ok, _ = grade(t, "", 'Here you go: {"language": "Python", "year": 1991}')
    assert not ok


def test_format_uses_final_text_not_full_text():
    t = TASKS_BY_ID["format-04"]
    assert grade(t, "Step 1 reasoning...\n\nNO", "NO")[0]
    assert not grade(t, "NO", "Step 1 reasoning... NO")[0]


# ── End to end with the fake model ────────────────────────────────────────────

def test_fake_end_to_end(tmp_path, monkeypatch):
    import sys
    from eval import run_eval, report

    out = tmp_path / "fake.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_eval", "--fake", "--runs", "1", "--limit", "10",
                                      "--out", str(out)])
    run_eval.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 20 and {r["system"] for r in rows} == {"baseline", "orchid"}

    # Resume: running again does nothing new
    run_eval.main()
    assert len(out.read_text().splitlines()) == 20

    md = report.build_report(rows)
    assert "Orchid vs baseline" in md and "self-grade" in md


# ── Known Orchid bug found by this harness ───────────────────────────────────

@pytest.mark.xfail(strict=True, reason="extract_json() grabs the first ``` fence even inside a JSON string, "
                                       "so any agent output containing a code block is lost")
def test_extract_json_keeps_code_fences_inside_strings():
    from core.state import extract_json
    raw = json.dumps({"output": "```python\ndef f():\n    return 1\n```", "ok": True})
    assert json.loads(extract_json(raw))["ok"] is True


def test_retry_provider_retries_rate_limits_only():
    from eval.systems import RetryProvider

    class Flaky:
        def __init__(self, errors):
            self.errors = list(errors)
            self.calls = 0

        def generate(self, s, u):
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
            return "ok"

    f = Flaky([RuntimeError("Error code: 429 rate limited"), RuntimeError("503 overloaded")])
    assert RetryProvider(f, retries=3, sleep=lambda s: None).generate("", "") == "ok"
    assert f.calls == 3

    g = Flaky([ValueError("bad request: invalid model")])
    with pytest.raises(ValueError):
        RetryProvider(g, retries=3, sleep=lambda s: None).generate("", "")
    assert g.calls == 1
