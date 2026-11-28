"""
Summarise an eval JSONL file.

  python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl
  python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl --handcheck 30
  python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl --graded handcheck_done.csv

Writes <file>.report.md next to the results. All numbers are computed from
the JSONL; nothing is typed in by hand.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from eval.tasks import TASKS_BY_ID


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def per_task_rate(rows: list[dict], system: str) -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        if r["system"] == system:
            by[r["task_id"]].append(1.0 if r["passed"] else 0.0)
    return {t: sum(v) / len(v) for t, v in by.items()}


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling TASKS (runs of one task are not independent)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test on tasks where the systems differ."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def build_report(rows: list[dict]) -> str:
    systems = sorted({r["system"] for r in rows})
    models = sorted({r["model"] for r in rows})
    out = [f"# Orchid eval-v2 report", "",
           f"Model(s): {', '.join(models)}  ",
           f"Rows: {len(rows)}  ",
           f"Tasks: {len({r['task_id'] for r in rows})}  ",
           f"Runs per task: {max(r['run'] for r in rows) + 1}", ""]

    # 1. Headline
    out += ["## 1. Pass rate (external grading)", "",
            "Mean over tasks of each task's pass rate across runs. 95% CI by bootstrap over tasks.", "",
            "| System | Tasks | Runs | Pass rate | 95% CI | Errors |",
            "|---|---|---|---|---|---|"]
    rates = {}
    for s in systems:
        sr = [r for r in rows if r["system"] == s]
        pt = per_task_rate(rows, s)
        rates[s] = pt
        mean = sum(pt.values()) / max(1, len(pt))
        lo, hi = bootstrap_ci(list(pt.values()))
        errs = sum(1 for r in sr if r["error"])
        out.append(f"| {s} | {len(pt)} | {len(sr)} | {pct(mean)} | {pct(lo)} to {pct(hi)} | {errs} |")
    out.append("")

    # 2. Paired comparison
    if "orchid" in rates and "baseline" in rates:
        common = sorted(set(rates["orchid"]) & set(rates["baseline"]))
        wins = sum(1 for t in common if rates["orchid"][t] > rates["baseline"][t])
        losses = sum(1 for t in common if rates["orchid"][t] < rates["baseline"][t])
        ties = len(common) - wins - losses
        diffs = [rates["orchid"][t] - rates["baseline"][t] for t in common]
        lo, hi = bootstrap_ci(diffs)
        out += ["## 2. Orchid vs baseline, same tasks", "",
                f"- Tasks compared: {len(common)}",
                f"- Orchid better: {wins}, baseline better: {losses}, tied: {ties}",
                f"- Mean difference (Orchid minus baseline): {pct(statistics.mean(diffs) if diffs else float('nan'))} "
                f"(95% CI {pct(lo)} to {pct(hi)})",
                f"- Sign test p-value: {sign_test_p(wins, losses):.3f}"
                " (below 0.05 suggests a real difference; with few tasks, expect wide uncertainty)", ""]

    # 3. By category
    cats = sorted({r["category"] for r in rows})
    out += ["## 3. By category", "", "| Category | " + " | ".join(systems) + " |",
            "|---|" + "---|" * len(systems)]
    for c in cats:
        cells = []
        for s in systems:
            vals = [v for t, v in rates[s].items() if TASKS_BY_ID.get(t, {}).get("category") == c]
            cells.append(f"{pct(sum(vals) / len(vals))} (n={len(vals)})" if vals else "n/a")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    # 4. Self-grade vs external grade
    orows = [r for r in rows if r["system"] == "orchid" and r["self_pass"] is not None]
    if orows:
        tp = sum(1 for r in orows if r["self_pass"] and r["passed"])
        fp = sum(1 for r in orows if r["self_pass"] and not r["passed"])
        fn = sum(1 for r in orows if not r["self_pass"] and r["passed"])
        tn = sum(1 for r in orows if not r["self_pass"] and not r["passed"])
        self_rate = (tp + fp) / len(orows)
        ext_rate = (tp + fn) / len(orows)
        out += ["## 4. Orchid's self-grade vs the external grade", "",
                "The old benchmark counted a task as passed when Orchid's own Critic scored it 6+/10.", "",
                "| | External PASS | External FAIL |", "|---|---|---|",
                f"| Orchid says PASS | {tp} | {fp} |",
                f"| Orchid says FAIL | {fn} | {tn} |", "",
                f"- Self-reported pass rate: {pct(self_rate)}; external pass rate: {pct(ext_rate)}",
                f"- Agreement: {pct((tp + tn) / len(orows))}",
                f"- When Orchid says PASS, it is wrong {pct(fp / (tp + fp) if tp + fp else float('nan'))} of the time", ""]

    # 5. Cost
    out += ["## 5. Cost per task run (medians)", "",
            "| System | Wall seconds | LLM calls | Output chars | Orchid iterations |",
            "|---|---|---|---|---|"]
    for s in systems:
        sr = [r for r in rows if r["system"] == s and not r["error"]]
        if not sr:
            continue
        med = lambda k: statistics.median(r[k] for r in sr)
        it = statistics.median(r["iterations"] for r in sr) if s == "orchid" else "-"
        out.append(f"| {s} | {med('wall_seconds'):.1f} | {med('llm_calls'):.0f} | {med('output_chars'):.0f} | {it} |")
    out.append("")

    # 6. Consistency
    out += ["## 6. Consistency across runs", "",
            "Share of tasks where every run got the same verdict (only meaningful with 2+ runs).", ""]
    for s in systems:
        by = defaultdict(set)
        cnt = defaultdict(int)
        for r in rows:
            if r["system"] == s:
                by[r["task_id"]].add(r["passed"])
                cnt[r["task_id"]] += 1
        multi = [t for t in by if cnt[t] > 1]
        if multi:
            same = sum(1 for t in multi if len(by[t]) == 1)
            out.append(f"- {s}: {pct(same / len(multi))} of {len(multi)} tasks")
    out.append("")

    # 7. Failure reasons
    out += ["## 7. Most common failure reasons", ""]
    for s in systems:
        reasons = defaultdict(int)
        for r in rows:
            if r["system"] == s and not r["passed"]:
                d = r["detail"]
                key = ("error" if r["error"] else
                       "no ANSWER line" if d.startswith("no ANSWER") else
                       "no function found" if d.startswith("no definition") else
                       "code tests failed" if "tests" in d else
                       "wrong answer" if d.startswith(("got", "no accepted", "ANSWER line contains")) else
                       "format rule broken")
                reasons[key] += 1
        top = sorted(reasons.items(), key=lambda kv: -kv[1])
        out.append(f"- {s}: " + (", ".join(f"{k} ({v})" for k, v in top) if top else "none"))
    out.append("")
    return "\n".join(out)


def write_handcheck(rows: list[dict], n: int, path: Path, seed: int = 0) -> None:
    """Random sample for a human to grade, to check the automatic graders."""
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "task_id", "run", "prompt", "expected", "output_shown",
                    "auto_passed", "auto_detail", "human_passed (Y/N)", "notes"])
        for r in sample:
            t = TASKS_BY_ID[r["task_id"]]
            expected = (t.get("answer") if t["kind"] == "number" else
                        t.get("accept") if t["kind"] == "text" else
                        f"tests for {t.get('func')}" if t["kind"] == "code" else t.get("check"))
            shown = r["final_text"] if t["kind"] == "format" else r["full_text"]
            w.writerow([r["system"], r["task_id"], r["run"], t["prompt"], expected,
                        shown[:3000], "Y" if r["passed"] else "N", r["detail"], "", ""])


def grader_agreement(path: Path) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    graded = [r for r in rows if r["human_passed (Y/N)"].strip().upper() in ("Y", "N")]
    if not graded:
        return "No human grades found in the file."
    agree = sum(1 for r in graded if r["auto_passed"] == r["human_passed (Y/N)"].strip().upper())
    lines = [f"## Grader check: auto vs human on {len(graded)} samples", "",
             f"- Agreement: {pct(agree / len(graded))}", "", "Disagreements:"]
    for r in graded:
        if r["auto_passed"] != r["human_passed (Y/N)"].strip().upper():
            lines.append(f"- {r['system']} {r['task_id']} run{r['run']}: auto={r['auto_passed']}, "
                         f"human={r['human_passed (Y/N)']}  {r['notes']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--handcheck", type=int, default=0, help="write N random rows to a CSV to grade by hand")
    ap.add_argument("--graded", default="", help="a hand-check CSV you filled in")
    args = ap.parse_args()

    path = Path(args.results)
    rows = load(path)
    report = build_report(rows)
    if args.graded:
        report += "\n" + grader_agreement(Path(args.graded)) + "\n"
    out = path.with_suffix(".report.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {out}")
    if args.handcheck:
        hc = path.with_suffix(".handcheck.csv")
        write_handcheck(rows, args.handcheck, hc)
        print(f"Hand-check sheet ({args.handcheck} rows) -> {hc}")


if __name__ == "__main__":
    main()
