#!/usr/bin/env bash
# Run eval-v2 on a free Kaggle GPU notebook (T4), using the same local model
# as on your laptop. In a Kaggle notebook (Settings: Accelerator = GPU T4,
# Internet = On), run ONE cell:
#
#   !git clone -b eval-v2 https://github.com/nirmantaterh/orchid && bash orchid/eval/kaggle_run.sh
#
# Extra arguments are passed to run_eval, e.g.  bash orchid/eval/kaggle_run.sh --limit 5 --runs 1
# Results are written to /kaggle/working/results (kept when you "Save Version").
# To resume in a new session, upload the old .jsonl to /kaggle/working/results first.
set -euo pipefail

MODEL="${MODEL:-gemma4:e4b}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="/kaggle/working/results"
mkdir -p "$OUT_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  (apt-get install -y -qq zstd >/dev/null 2>&1 || true)
  curl -fsSL https://ollama.com/install.sh | sh
fi

nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
for i in $(seq 1 30); do
  curl -s http://localhost:11434/api/tags >/dev/null && break
  sleep 2
done
ollama pull "$MODEL"

cd "$REPO"
pip install -q -e ".[openai,dev]"

python -X utf8 -m eval.run_eval --ollama --model "$MODEL" \
  --out "$OUT_DIR/eval_${MODEL//[:\/]/-}.jsonl" "$@"

python -X utf8 -m eval.report "$OUT_DIR/eval_${MODEL//[:\/]/-}.jsonl" --handcheck 30
