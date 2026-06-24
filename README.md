# Orchid

Multi-agent orchestration framework that adapts its strategy mid-run. When execution fails, it doesn't restart from scratch — it critiques, reflects, replans, and continues.

Built from scratch in Python. No LangChain. No LlamaIndex. Runs fully local via [Ollama](https://ollama.com).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-black)
![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

---

## Why Orchid

Most agent frameworks (LangGraph, CrewAI) define a static graph or crew. Mid-run adaptation means patching the graph, writing custom error edges, or restarting.

Orchid's Controller is itself an agent. At each iteration it reads confidence trends, critic outputs, and memory state, then decides: `continue`, `replan`, `delegate to specialist`, or `terminate`. The strategy is data — not code.

---

## How It Works

```
GoalInterpreter → Planner → Executor → Critic → Reflection → MemoryManager → Controller
                                                                                    │
                                    ┌───────────────────────────────────────────────┤
                                    │ continue / replan / delegate / terminate
                                    └──────────────────────────────────────────────►
```

Each agent reads from and writes to a shared immutable `AgentState`. The loop runs until the task succeeds or `max_iterations` is reached.

### Layers on Top

| System | What it does |
|--------|-------------|
| **Confidence system** | Tracks per-agent confidence trends; forces replan when confidence drops below threshold |
| **Debate panel** | Two independent critics synthesize when confidence variance is high |
| **Dynamic specialists** | 7 specialist agent types (researcher, coder, analyzer, ...) spun up on demand for blocked sub-problems |
| **Tracer** | Per-agent spans with latency and confidence, exportable to JSON or SQLite |
| **Visualization** | Mermaid flowchart of the live agent graph |

---

## Benchmark

Validated against 8 tasks across 4 difficulty tiers using `gemma4:e4b` on CPU.

| Tier | Tasks | Examples | Result |
|------|-------|----------|--------|
| A — Simple | 2 | Summarization, Q&A | 2/2 |
| B — Medium | 2 | Study plan, repo analysis | 2/2 |
| C — Hard | 3 | Research synthesis, system design, debugging | 3/3 |
| D — Long Horizon | 1 | Multi-step SaaS planning | 1/1 |
| **Total** | **8** | | **8/8** |

---

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) running locally.

```bash
git clone https://github.com/nirmantaterh/orchid
cd orchid
pip install -e ".[openai,dev]"
ollama pull gemma4:e4b
```

---

## Usage

**Run the benchmark:**

```bash
python -X utf8 run_benchmark.py --ollama --model gemma4:e4b --sets A,B,C,D
```

**Use in code:**

```python
from goal_interpreter.providers.openai import OpenAIProvider
from core.orchestrator import Orchestrator

provider = OpenAIProvider(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="gemma4:e4b"
)
orch = Orchestrator(provider, max_iterations=5)
state, tracer = orch.run("Create a 30-day ML study plan for a software engineer")
print(state.execution_result.output)
```

**Run tests:**

```bash
pytest tests/  # 88 unit + integration tests
```

---

## Structure

```
goal_interpreter/    standalone LLM-agnostic provider library
core/                AgentState, Orchestrator, Controller loop
agents/              7 core agents + debate panel + specialist factory
evaluation/          benchmark runner, 13 metrics
tracing/             span tracer (JSON / SQLite export)
visualization/       Mermaid graph builder
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| Agent runtime | Pure Python 3.13, Pydantic v2 |
| LLM client | OpenAI SDK (Ollama-compatible) |
| Local models | Ollama (gemma4, llama3, mistral, ...) |
| Tests | pytest — 88 unit + integration |

---

## License

Apache 2.0. See [LICENSE](LICENSE).
