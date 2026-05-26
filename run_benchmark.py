"""
Full benchmark across Task Sets A–D.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python run_benchmark.py

Optional flags:
    --sets A,B,C,D      which task sets to run (default: all)
    --model             claude-sonnet-4-6 (default)
    --out               path for JSON report  (default: benchmark_report.json)
    --max-iter-a N      max iterations for Set A (default: 2)
    --max-iter-b N      max iterations for Set B (default: 3)
    --max-iter-c N      max iterations for Set C (default: 4)
    --max-iter-d N      max iterations for Set D (default: 5)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ── Task definitions ──────────────────────────────────────────────────────────

TASK_SETS = {
    "A": {
        "label": "Simple",
        "max_iterations": 2,
        "tasks": [
            {
                "name": "A1 · Summarize article",
                "input": (
                    "Summarize this article in 3 bullet points:\n\n"
                    "Large language models (LLMs) are neural networks trained on vast corpora of text. "
                    "They use the transformer architecture, which relies on self-attention to capture "
                    "long-range dependencies between tokens. Training requires significant compute — "
                    "models like GPT-4 are estimated to have cost tens of millions of dollars to train. "
                    "Once trained, these models can be fine-tuned on specific tasks or prompted "
                    "with examples (few-shot learning) to adapt their behaviour without retraining. "
                    "Key challenges include hallucination (generating plausible but false information), "
                    "alignment (ensuring models follow human intent), and inference cost at scale."
                ),
            },
            {
                "name": "A2 · Factual Q&A",
                "input": (
                    "Answer these three factual questions concisely:\n"
                    "1. What is the time complexity of quicksort in the average case?\n"
                    "2. Who invented the transformer architecture and in which year?\n"
                    "3. What does ACID stand for in database systems?"
                ),
            },
        ],
    },
    "B": {
        "label": "Medium",
        "max_iterations": 3,
        "tasks": [
            {
                "name": "B1 · Study plan",
                "input": (
                    "Create a structured 30-day study plan for an experienced software engineer "
                    "who wants to transition into machine learning engineering. "
                    "They know Python well but have no ML background. "
                    "Include daily topics, resources, and hands-on projects. "
                    "Focus on practical skills: data pipelines, model training, evaluation, and deployment."
                ),
            },
            {
                "name": "B2 · Analyze GitHub repo",
                "input": (
                    "Analyze this GitHub repository description and provide actionable improvement suggestions:\n\n"
                    "Repo: production-rag-pipeline\n"
                    "Description: A RAG (Retrieval-Augmented Generation) pipeline built with LangChain and FAISS. "
                    "Supports PDF ingestion, chunking, and semantic search. "
                    "Single Python file (~800 lines), no tests, hardcoded API keys in config.py, "
                    "no Docker support, no CI/CD, README has basic setup instructions only. "
                    "Used in production by 3 internal teams.\n\n"
                    "Suggest: architecture improvements, security fixes, testing strategy, "
                    "observability, and a realistic refactor roadmap."
                ),
            },
        ],
    },
    "C": {
        "label": "Hard",
        "max_iterations": 4,
        "tasks": [
            {
                "name": "C1 · Startup research",
                "input": (
                    "Research the competitive landscape for AI-powered code review tools. "
                    "Identify the top 5 competitors, their pricing models, key differentiators, "
                    "target customers, and known weaknesses. "
                    "Then identify a gap in the market that a new entrant could exploit. "
                    "Conclude with a positioning statement for a new tool targeting mid-size engineering teams."
                ),
            },
            {
                "name": "C2 · Implementation planning",
                "input": (
                    "Create a detailed technical implementation plan for a distributed message queue "
                    "that must handle 1 million messages per second with at-least-once delivery guarantees. "
                    "Requirements: sub-10ms p99 latency, horizontal scaling, replay support, "
                    "consumer groups, dead-letter queues, and a REST + gRPC API. "
                    "Cover: architecture decisions, data model, storage backend choice with justification, "
                    "replication strategy, failure modes and mitigations, and a phased rollout plan."
                ),
            },
            {
                "name": "C3 · Debugging",
                "input": (
                    "A Python FastAPI application is returning intermittent 503 errors under load. "
                    "Symptoms: errors increase when concurrent requests exceed ~200, "
                    "CPU is at 40% when errors occur, memory is stable, "
                    "no errors in the application logs (only in the load balancer logs), "
                    "the service uses async endpoints with SQLAlchemy async sessions and a PostgreSQL pool of 10. "
                    "Upstream services are healthy. Errors disappear after ~30 seconds of reduced load.\n\n"
                    "Diagnose the root cause(s), explain your reasoning step by step, "
                    "and provide a prioritised list of fixes with code examples where relevant."
                ),
            },
        ],
    },
    "D": {
        "label": "Long Horizon (Memory)",
        "max_iterations": 6,
        "tasks": [
            {
                "name": "D1 · Multi-step SaaS business plan",
                "input": (
                    "You are building a SaaS startup targeting enterprise ML teams. "
                    "The product: a platform that automatically detects data drift, model degradation, "
                    "and feature importance shifts in production ML pipelines, with one-click remediation. "
                    "\n\n"
                    "Develop a comprehensive plan covering ALL of the following — "
                    "each section must build on the previous one:\n"
                    "1. Market analysis: TAM/SAM/SOM, top 3 competitors, buying persona\n"
                    "2. Technical architecture: core components, data flow, scalability approach\n"
                    "3. Go-to-market: pricing model, sales motion, first 10 customers strategy\n"
                    "4. 12-month roadmap: milestones, team hires, key risks\n"
                    "5. Financial projections: Year 1 burn, MRR targets, break-even estimate\n"
                    "\n"
                    "The system should remember decisions made in earlier sections and "
                    "ensure consistency across all five areas."
                ),
            },
        ],
    },
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all(sets: list[str], model: str, out_path: str,
            iter_overrides: dict[str, int], use_ollama: bool = False) -> None:
    from goal_interpreter.providers.anthropic import AnthropicProvider
    from evaluation.benchmark import Benchmark, BenchmarkCase
    from evaluation.metrics import RunMetrics
    from visualization.graph_builder import GraphBuilder

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    if use_ollama:
        from goal_interpreter.providers.openai import OpenAIProvider
        print(DIM(f"  Backend: Ollama  ({ollama_url})  model={model}"))
        provider = OpenAIProvider(api_key="ollama", model=model, base_url=ollama_url)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print(RED("✗  ANTHROPIC_API_KEY is not set."))
            print("   Run:  set ANTHROPIC_API_KEY=sk-ant-...")
            print("   Or use Ollama for free:  python run_benchmark.py --ollama --model llama3.2")
            sys.exit(1)
        provider = AnthropicProvider(api_key=api_key, model=model)

    all_results = {}
    grand_total = 0
    grand_passed = 0
    wall_start = time.monotonic()

    for set_id in sets:
        if set_id not in TASK_SETS:
            print(YELLOW(f"⚠  Unknown task set '{set_id}', skipping."))
            continue

        cfg = TASK_SETS[set_id]
        max_iter = iter_overrides.get(set_id, cfg["max_iterations"])
        label = cfg["label"]

        print()
        print(BOLD(CYAN(f"━━━  Task Set {set_id}: {label}  (max_iter={max_iter})  ━━━")))

        bench = Benchmark(provider, max_iterations=max_iter, enable_debate=False)
        set_results = []

        for task in cfg["tasks"]:
            name  = task["name"]
            input_text = task["input"]
            case  = BenchmarkCase(
                name=name,
                input=input_text,
                max_iterations=max_iter,
            )

            print(f"\n  {BOLD(name)}")
            print(DIM(f"  Input: {input_text[:100].replace(chr(10), ' ')}..."))

            t0 = time.monotonic()
            result = bench.run_case(case)
            elapsed = time.monotonic() - t0

            m = result.metrics
            status = GREEN("✓  PASS") if result.passed else RED("✗  FAIL")
            print(f"  {status}  |  iter={m.iterations_required}  "
                  f"score={m.average_critique_score:.1f}/10  "
                  f"conf={m.average_confidence:.2f}  "
                  f"replan={m.replanning_frequency:.2f}  "
                  f"latency={elapsed:.1f}s")

            if result.execution_output:
                print(DIM(f"  Answer: {result.execution_output[:300]}"))
            if result.error:
                print(RED(f"  Error: {result.error[:120]}"))

            # Print per-iteration scores if available
            if result.metrics.iterations_required > 1:
                from core.orchestrator import Orchestrator as _O  # just to show note
                print(DIM("  (multi-iteration — memory and replanning active)"))

            set_results.append(result)
            grand_total += 1
            if result.passed:
                grand_passed += 1

        all_results[set_id] = set_results

        # Set summary
        passed = sum(1 for r in set_results if r.passed)
        print(f"\n  Set {set_id} summary: {passed}/{len(set_results)} passed")

    # ── Grand summary ─────────────────────────────────────────────────────────
    wall_elapsed = time.monotonic() - wall_start
    print()
    print(BOLD("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(BOLD(f"  TOTAL: {grand_passed}/{grand_total} tasks passed  "
               f"({grand_passed/max(1,grand_total)*100:.0f}%)"))
    print(BOLD(f"  Wall time: {wall_elapsed:.1f}s"))
    print(BOLD("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

    # ── Mermaid graph for each task ───────────────────────────────────────────
    _save_graphs(all_results)

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "model": model,
        "sets": sets,
        "summary": {"total": grand_total, "passed": grand_passed,
                    "pass_rate": grand_passed / max(1, grand_total),
                    "wall_time_s": round(wall_elapsed, 2)},
        "results": {
            sid: [r.model_dump() for r in rs]
            for sid, rs in all_results.items()
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved → {out_path}")


def _save_graphs(all_results: dict) -> None:
    from visualization.graph_builder import GraphBuilder
    graphs_dir = Path("benchmark_graphs")
    graphs_dir.mkdir(exist_ok=True)

    # We don't have direct access to AgentState here (Benchmark doesn't return it),
    # so we skip graph generation silently.
    # To generate graphs, use Orchestrator directly and call GraphBuilder.from_state(state).
    pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive agent benchmark")
    parser.add_argument("--sets",     default="A,B,C,D",
                        help="Comma-separated task set IDs (default: A,B,C,D)")
    parser.add_argument("--ollama",   action="store_true",
                        help="Use local Ollama instead of Anthropic API (free)")
    parser.add_argument("--model",    default="claude-sonnet-4-6")
    parser.add_argument("--out",      default="benchmark_report.json")
    parser.add_argument("--max-iter-a", type=int, default=2)
    parser.add_argument("--max-iter-b", type=int, default=3)
    parser.add_argument("--max-iter-c", type=int, default=4)
    parser.add_argument("--max-iter-d", type=int, default=6)
    args = parser.parse_args()

    sets = [s.strip().upper() for s in args.sets.split(",")]
    iter_overrides = {
        "A": args.max_iter_a,
        "B": args.max_iter_b,
        "C": args.max_iter_c,
        "D": args.max_iter_d,
    }

    if args.ollama and args.model == "claude-sonnet-4-6":
        args.model = "llama3.2"  # sensible default when switching to Ollama

    print(BOLD(CYAN("\n  Adaptive Multi-Agent Benchmark")))
    print(DIM(f"  Model: {args.model}  |  Sets: {', '.join(sets)}"))

    run_all(sets, args.model, args.out, iter_overrides, use_ollama=args.ollama)


if __name__ == "__main__":
    main()
