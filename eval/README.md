# eval-v2: does Orchid actually beat one model call?

The original benchmark (`run_benchmark.py`) counts a task as passed when
Orchid's own Critic scores it 6/10 or higher. The Critic runs on the same
model that did the work, so that 8/8 is a self-grade. This folder grades
Orchid from the outside instead.

## What it does

- **66 tasks with checkable answers**: math (18), facts (12), logic (12),
  code (16, run against hidden tests), format/instruction-following (8).
- **Two systems, same model, same prompt**:
  - `baseline`: one plain call to the model
  - `orchid`: the full Orchestrator loop (`--max-iter`, default 3)
- **External grading only**. No LLM judge. Number/text answers are read
  from the last `ANSWER:` line; code is executed against test cases in a
  subprocess with a 10 s timeout; format rules are checked in Python.
- **Repeated runs** (`--runs`, default 3), because LLM output varies.
- **Records Orchid's self-grade too**, so the report can show how often
  "Orchid says PASS" matches the external grade.
- **Resumable**: every finished run is appended to a JSONL file. Ctrl+C and
  re-run the same command to continue.

### How Orchid's answer is read

Orchid has no single "final answer" field, so two views are stored:

- `full_text`: all step outputs of the last execution, joined. Used for
  ANSWER-line and code tasks (the most lenient choice for Orchid).
- `final_text`: the last step's output. Used for format tasks, since that is
  what a user would read as the reply.

## Run it (Windows, from the repo root)

```powershell
pip install -e ".[openai,dev]"
ollama pull gemma4:e4b

# 0. Check the pipeline with a fake model (seconds, no Ollama)
python -X utf8 -m eval.run_eval --fake --runs 1 --out results/fake.jsonl

# 1. Time it first: 5 tasks, 1 run, both systems
python -X utf8 -m eval.run_eval --ollama --model gemma4:e4b --limit 5 --runs 1 --out results/pilot.jsonl

# 2. Full run (long on CPU; leave it overnight; resumable)
python -X utf8 -m eval.run_eval --ollama --model gemma4:e4b --runs 3

# 3. Report + a 30-row sheet to grade by hand
python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl --handcheck 30

# 4. After filling in the "human_passed (Y/N)" column:
python -X utf8 -m eval.report results/eval_gemma4-e4b.jsonl --graded results/eval_gemma4-e4b.handcheck.csv
```

Useful flags: `--systems baseline` (run one system), `--categories math,code`,
`--tasks code-01,code-02`, `--verbose` (show Orchid's internal logs).

## The report

`eval.report` writes `<results>.report.md` with:

1. Pass rate per system with a 95% confidence interval (bootstrap over tasks)
2. Orchid vs baseline on the same tasks: wins, losses, ties, sign test
3. Pass rate by category
4. Orchid's self-grade vs the external grade (how often "PASS" is wrong)
5. Cost: wall time, LLM calls, output size per run
6. Consistency across repeated runs
7. Most common failure reasons

## Known issue found while building this

`core/state.py::extract_json()` looks for the first ``` fence anywhere in a
response. When an agent returns valid JSON whose string values contain a
code block (common on code tasks), it throws the JSON away and keeps only
the code. The Executor result is then lost, and the Critic, Reflection,
Memory and Controller agents skip with "prompt-build error". See the
`xfail` test in `tests/test_eval.py`. The eval measures Orchid as it is;
fix this and re-run to see the difference.

## Limits

- 66 tasks is enough to see large differences, not small ones. Widen the
  task set before claiming a few-point gap.
- These are short, checkable tasks. They test whether the loop helps on
  problems with right answers, not on open-ended planning.
- Running model-written code: graders execute it in a subprocess with a
  timeout but no sandbox. Fine for a local model on your own machine.
