# Orchid

Orchid is a multi-agent orchestration framework that adapts its strategy mid-run. Named after one of the most adaptive plant families on earth — orchids survive across wildly different environments by adjusting how they grow. Orchid does the same: when an execution fails, it doesn't restart from scratch. It critiques, reflects, replans, and tries again.

Built from scratch in Python. No LangChain. No LlamaIndex. Runs fully local via [Ollama](https://ollama.com).

---

## How it works

A raw user goal passes through seven specialized agents in a loop:

```
GoalInterpreter → Planner → Executor → Critic → Reflection → MemoryManager → Controller
```

Each agent reads from and writes to a shared immutable state (`AgentState`). The Controller decides at each iteration whether to `terminate`, `replan`, `continue`, or hand off to a `specialist`. The loop runs until the task is done or max iterations are reached.

Additional systems layered on top:
- **Tracer** — per-agent spans with latency and confidence, exportable to JSON or SQLite
- **Confidence system** — tracks per-agent confidence trends; forces replan when confidence drops
- **Debate panel** — two independent critics synthesize when confidence variance is high
- **Dynamic specialists** — 7 specialist agent types created on demand for blocked sub-problems
- **Visualization** — Mermaid flowchart of the agent graph

---

## Benchmark

Validated against 8 tasks across 4 difficulty tiers using `gemma4:e4b` on CPU (no GPU).

| Set | Tasks | Result |
|-----|-------|--------|
| A — Simple (summarization, Q&A) | 2 | 2/2 |
| B — Medium (study plan, repo analysis) | 2 | 2/2 |
| C — Hard (research, system design, debugging) | 3 | 3/3 |
| D — Long Horizon (multi-step SaaS planning) | 1 | 1/1 |
| **Total** | **8** | **8/8** |

---

## Stack

- Python 3.13, Pydantic v2
- OpenAI SDK (used as Ollama-compatible client)
- 88 unit and integration tests

---

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) running locally.

```bash
git clone https://github.com/nirmantaterh/orchid
cd orchid
pip install -e ".[openai,dev]"
ollama pull gemma4:e4b
```

**Run the benchmark:**

```bash
python -X utf8 run_benchmark.py --ollama --model gemma4:e4b --sets A,B,C,D
```

**Use in code:**

```python
from goal_interpreter.providers.openai import OpenAIProvider
from core.orchestrator import Orchestrator

provider = OpenAIProvider(api_key="ollama", base_url="http://localhost:11434/v1", model="gemma4:e4b")
orch = Orchestrator(provider, max_iterations=5)
state, tracer = orch.run("Create a 30-day ML study plan for a software engineer")
print(state.execution_result.output)
```

**Run tests:**

```bash
pytest tests/
```

---

## Structure

```
goal_interpreter/    standalone LLM-agnostic library
core/                AgentState, Orchestrator
agents/              7 agents + debate panel + specialist factory
evaluation/          benchmark runner, 13 metrics
tracing/             span tracer (JSON / SQLite)
visualization/       Mermaid graph builder
```
