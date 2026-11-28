"""
External graders. None of these call an LLM.

Each grader returns (passed: bool, detail: str). `detail` explains the
decision so a human can audit it in the hand-check sheet.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_answer_line(text: str) -> str | None:
    """Return the text after the LAST 'ANSWER:' marker, or None."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    ans = matches[-1].strip()
    # strip markdown emphasis / trailing punctuation
    ans = ans.lstrip("*_` ").rstrip("*_` .")
    return ans


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\s+", " ", s)
    return s


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|-?\.\d+")


def parse_number(s: str) -> float | None:
    """Parse the first number in s. Handles $, commas, %, and fractions like 5/36."""
    if s is None:
        return None
    m = _NUM_RE.search(s.replace("$", ""))
    if not m:
        return None
    tok = m.group(0).replace(",", "").replace(" ", "")
    try:
        if "/" in tok:
            num, den = tok.split("/")
            return float(Fraction(num) / Fraction(den))
        return float(tok)
    except (ValueError, ZeroDivisionError):
        return None


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, func: str) -> str | None:
    """Return the last code block that defines `func`, else any block, else raw text if it defines func."""
    blocks = _CODE_BLOCK_RE.findall(text or "")
    with_def = [b for b in blocks if re.search(rf"def\s+{re.escape(func)}\s*\(", b)]
    if with_def:
        return with_def[-1]
    if re.search(rf"def\s+{re.escape(func)}\s*\(", text or ""):
        # No fenced block: take from the def line onward
        idx = re.search(rf"(?m)^[ \t]*def\s+{re.escape(func)}\s*\(", text)
        if idx:
            return text[idx.start():]
    return None


# ── Graders by kind ───────────────────────────────────────────────────────────

def grade_number(task: dict, full_text: str, final_text: str) -> tuple[bool, str]:
    ans = extract_answer_line(full_text)
    if ans is None:
        return False, "no ANSWER line"
    val = parse_number(ans)
    if val is None:
        return False, f"no number in ANSWER line: {ans[:80]!r}"
    ok = abs(val - task["answer"]) <= task.get("tol", 1e-6)
    return ok, f"got {val} expected {task['answer']}"


def grade_text(task: dict, full_text: str, final_text: str) -> tuple[bool, str]:
    ans = extract_answer_line(full_text)
    if ans is None:
        return False, "no ANSWER line"
    norm = _normalize(ans)

    def _has(term: str) -> bool:
        # whole-word / whole-token match inside the answer line
        a = _normalize(term)
        return re.search(rf"(?<![\w.]){re.escape(a)}(?![\w]|\.\d)", norm) is not None

    bad = [r for r in task.get("reject", []) if _has(r)]
    if bad:
        return False, f"ANSWER line contains wrong answer {bad[0]!r}: {ans[:80]!r}"
    for acc in task["accept"]:
        if _has(acc):
            return True, f"matched {acc!r} in {ans[:80]!r}"
    return False, f"no accepted answer in {ans[:80]!r} (accepted: {task['accept']})"


_RUNNER = r'''
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    src = fh.read()
with open(sys.argv[2], encoding="utf-8") as fh:
    tests = json.load(fh)
func = sys.argv[3]
ns = {}
exec(compile(src, "candidate", "exec"), ns)
f = ns[func]
fails = []
for args, expected in tests:
    try:
        got = f(*args)
    except Exception as e:
        fails.append(f"{args!r} raised {type(e).__name__}: {e}")
        continue
    if got != expected and not (isinstance(got, tuple) and list(got) == expected):
        fails.append(f"{args!r} -> {got!r}, expected {expected!r}")
print(json.dumps({"passed": len(tests) - len(fails), "total": len(tests), "fails": fails[:3]}))
'''


def grade_code(task: dict, full_text: str, final_text: str, timeout_s: int = 10) -> tuple[bool, str]:
    code = extract_code(full_text, task["func"])
    if code is None:
        return False, f"no definition of {task['func']}() found"
    with tempfile.TemporaryDirectory() as d:
        src_p = os.path.join(d, "cand.py")
        tests_p = os.path.join(d, "tests.json")
        run_p = os.path.join(d, "runner.py")
        for p, content in ((src_p, code), (tests_p, json.dumps(task["tests"])), (run_p, _RUNNER)):
            with open(p, "w", encoding="utf-8") as fh:  # closed before the run (matters on Windows)
                fh.write(content)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", run_p, src_p, tests_p, task["func"]],
                capture_output=True, text=True, timeout=timeout_s, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s}s"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return False, "crashed: " + (err[-1] if err else "unknown error")
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "could not read test results"
    ok = res["passed"] == res["total"]
    detail = f"{res['passed']}/{res['total']} tests"
    if res["fails"]:
        detail += "; " + " | ".join(res["fails"])
    return ok, detail


# ── Format checks (use final_text: what the user would actually see last) ────

def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p.strip()]


def _fmt_bullets_exact(text, n):
    lines = _lines(text)
    bullets = [ln for ln in lines if ln.startswith("- ")]
    ok = len(lines) == n and len(bullets) == n
    return ok, f"{len(lines)} non-empty lines, {len(bullets)} start with '- ' (need exactly {n})"


def _fmt_json_exact(text, expected):
    try:
        obj = json.loads((text or "").strip())
    except ValueError:
        return False, "not valid JSON on its own (extra text or code fences?)"
    ok = obj == expected
    return ok, f"parsed {obj!r}, expected {expected!r}"


def _fmt_one_sentence_max_words(text, max_words):
    t = (text or "").strip()
    sents = _sentences(t)
    words = len(t.split())
    ok = len(sents) == 1 and 0 < words <= max_words and "\n" not in t
    return ok, f"{len(sents)} sentence(s), {words} words (max {max_words})"


def _fmt_single_word(text, expected):
    t = (text or "").strip().strip(".!*` ").lower()
    ok = t == expected
    return ok, f"got {t[:40]!r}, expected {expected!r}"


def _fmt_exact_csv_numbers(text, expected):
    t = (text or "").strip().rstrip(".")
    parts = [p.strip() for p in t.split(",")]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False, f"not a plain comma-separated list: {t[:60]!r}"
    return nums == expected, f"got {nums}"


def _fmt_caps_title(text, max_words):
    lines = _lines(text)
    if len(lines) != 1:
        return False, f"{len(lines)} lines (need 1)"
    title = lines[0].strip("#*\"' ")
    letters = [c for c in title if c.isalpha()]
    ok = bool(letters) and all(c.isupper() for c in letters) and len(title.split()) <= max_words
    return ok, f"{title[:60]!r}: {len(title.split())} words, all caps={all(c.isupper() for c in letters)}"


def _fmt_no_letter_sentences(text, letter, n):
    t = (text or "").strip()
    has_letter = letter.lower() in t.lower()
    sents = _sentences(t)
    ok = not has_letter and len(sents) == n
    return ok, f"{len(sents)} sentence(s), contains '{letter}': {has_letter}"


def _fmt_sorted_lines(text, n):
    lines = [ln.lstrip("-*0123456789.) ").strip() for ln in _lines(text)]
    ok = (len(lines) == n and len(set(l.lower() for l in lines)) == n
          and lines == sorted(lines, key=str.lower))
    return ok, f"lines={lines[:6]}"


_FORMAT_CHECKS = {
    "bullets_exact": _fmt_bullets_exact,
    "json_exact": _fmt_json_exact,
    "one_sentence_max_words": _fmt_one_sentence_max_words,
    "single_word": _fmt_single_word,
    "exact_csv_numbers": _fmt_exact_csv_numbers,
    "caps_title": _fmt_caps_title,
    "no_letter_sentences": _fmt_no_letter_sentences,
    "sorted_lines": _fmt_sorted_lines,
}


def grade_format(task: dict, full_text: str, final_text: str) -> tuple[bool, str]:
    return _FORMAT_CHECKS[task["check"]](final_text, **task["params"])


GRADERS = {
    "number": grade_number,
    "text": grade_text,
    "code": grade_code,
    "format": grade_format,
}


def grade(task: dict, full_text: str, final_text: str) -> tuple[bool, str]:
    return GRADERS[task["kind"]](task, full_text or "", final_text or "")
