"""
eval-v2: grade Orchid against a single-call baseline with external checks.

Results are appended to a JSONL file one run at a time, so you can stop the
script (Ctrl+C) and start it again: finished runs are skipped.

Examples (from the repo root):

  # 1. Time it first: 5 tasks, 1 run each, both systems
  python -X utf8 -m eval.run_eval --ollama --model gemma4:e4b --limit 5 --runs 1

  # 2. Full run (resumable)
  python -X utf8 -m eval.run_eval --ollama --model gemma4:e4b --runs 3

  # 3. Report
  python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl

  # Dry run with a fake model (no Ollama needed), to check the pipeline works
  python -X utf8 -m eval.run_eval --fake --runs 1 --out results/fake.jsonl
"""
import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

from eval.graders import grade
from eval.systems import CountingProvider, run_baseline, run_orchid
from eval.tasks import TASKS


def _make_provider(args):
    if args.fake:
        from eval.fake_provider import FakeProvider
        return FakeProvider(seed=args.seed)
    if args.ollama:
        from goal_interpreter.providers.openai import OpenAIProvider
        url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIProvider(api_key="ollama", model=args.model, base_url=url)
    if args.anthropic:
        from goal_interpreter.providers.anthropic import AnthropicProvider
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(api_key=key, model=args.model)
    sys.exit("Choose a backend: --ollama, --anthropic, or --fake")


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["system"], r["task_id"], r["run"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchid vs baseline, externally graded")
    ap.add_argument("--ollama", action="store_true")
    ap.add_argument("--anthropic", action="store_true")
    ap.add_argument("--fake", action="store_true", help="fake model, for testing the pipeline")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--systems", default="baseline,orchid")
    ap.add_argument("--runs", type=int, default=3, help="repeats per task per system")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (after --categories)")
    ap.add_argument("--categories", default="", help="e.g. math,code")
    ap.add_argument("--tasks", default="", help="comma-separated task ids")
    ap.add_argument("--max-iter", type=int, default=3, help="Orchid max_iterations")
    ap.add_argument("--seed", type=int, default=0, help="shuffles run order only")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true", help="show Orchid's internal JSON logs")
    args = ap.parse_args()
    if not args.verbose:
        logging.disable(logging.INFO)  # Orchid logs every agent step at DEBUG/INFO

    tasks = TASKS
    if args.categories:
        cats = set(args.categories.split(","))
        tasks = [t for t in tasks if t["category"] in cats]
    if args.tasks:
        ids = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in ids]
    if args.limit:
        # take a spread across categories, not just the first category
        by_cat: dict[str, list] = {}
        for t in tasks:
            by_cat.setdefault(t["category"], []).append(t)
        picked, i = [], 0
        while len(picked) < args.limit and any(by_cat.values()):
            cat = list(by_cat)[i % len(by_cat)]
            if by_cat[cat]:
                picked.append(by_cat[cat].pop(0))
            i += 1
        tasks = picked

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    model_tag = "fake" if args.fake else args.model.replace(":", "-").replace("/", "-")
    out = Path(args.out or f"results/eval_{model_tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(s, t, r) for r in range(args.runs) for t in tasks for s in systems]
    # Interleave systems and tasks so a partial run is still a fair comparison
    random.Random(args.seed).shuffle(jobs)
    jobs.sort(key=lambda j: j[2])  # finish run 0 everywhere before run 1
    done = _done_keys(out)
    todo = [j for j in jobs if (j[0], j[1]["id"], j[2]) not in done]

    print(f"model={args.model if not args.fake else 'fake'}  tasks={len(tasks)}  "
          f"systems={systems}  runs={args.runs}")
    print(f"{len(jobs) - len(todo)} already done, {len(todo)} to go -> {out}")

    provider = CountingProvider(_make_provider(args))
    t_start = time.monotonic()

    with out.open("a", encoding="utf-8") as fh:
        for n, (system, task, run) in enumerate(todo, 1):
            provider.reset()
            t0 = time.monotonic()
            error = None
            try:
                if system == "baseline":
                    res = run_baseline(provider, task["prompt"])
                elif system == "orchid":
                    res = run_orchid(provider, task["prompt"], args.max_iter)
                else:
                    raise ValueError(f"unknown system {system}")
                passed, detail = grade(task, res["full_text"], res["final_text"])
            except KeyboardInterrupt:
                print("\nStopped. Finished runs are saved; run the same command to resume.")
                return
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                res = {"full_text": "", "final_text": "", "self_pass": None,
                       "self_score": None, "iterations": 0, "final_action": "error"}
                passed, detail = False, "system error"
                traceback.print_exc(limit=2)

            wall = time.monotonic() - t0
            row = {
                "system": system, "task_id": task["id"], "category": task["category"],
                "run": run, "model": "fake" if args.fake else args.model,
                "passed": passed, "detail": detail, "error": error,
                "wall_seconds": round(wall, 2), **provider.snapshot(),
                "self_pass": res["self_pass"], "self_score": res["self_score"],
                "iterations": res["iterations"], "final_action": res["final_action"],
                "full_text": res["full_text"], "final_text": res["final_text"],
                "max_iter": args.max_iter if system == "orchid" else None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            elapsed = time.monotonic() - t_start
            eta_min = elapsed / n * (len(todo) - n) / 60
            mark = "PASS" if passed else ("ERR " if error else "FAIL")
            print(f"[{n}/{len(todo)}] {mark} {system:8s} {task['id']:10s} run{run}  "
                  f"{wall:6.1f}s  calls={provider.calls:<3d} eta~{eta_min:.0f}min  {detail[:70]}")

    print(f"Done. Report: python -X utf8 -m eval.report {out}")


if __name__ == "__main__":
    main()
