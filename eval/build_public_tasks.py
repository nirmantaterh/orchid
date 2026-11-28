"""
Builds eval/data/*.jsonl: fixed random samples from three public benchmarks.

  GSM8K test split   (MIT)        https://github.com/openai/grade-school-math
  HumanEval          (MIT)        https://github.com/openai/human-eval
  MBPP test split    (CC BY 4.0)  https://github.com/google-research/google-research/tree/master/mbpp

Usage (raw files downloaded from the links above):
  python eval/build_public_tasks.py test.jsonl HumanEval.jsonl.gz mbpp.jsonl

Standalone on purpose (no imports from this repo). Same seed -> same sample.
"""
import gzip
import json
import random
import re
import sys
from pathlib import Path

SEED = 0
N_GSM8K = 50
N_HUMANEVAL = 50
N_MBPP = 40

OUT = Path(__file__).resolve().parent / "data"


def gsm8k(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    idx = sorted(random.Random(SEED).sample(range(len(rows)), N_GSM8K))
    out = []
    for i in idx:
        r = rows[i]
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-{i:04d}", "source_index": i,
                    "question": r["question"], "answer": float(ans)})
    return out


def humaneval(path: str) -> list[dict]:
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    idx = sorted(random.Random(SEED).sample(range(len(rows)), N_HUMANEVAL))
    out = []
    for i in idx:
        r = rows[i]
        ep = r["entry_point"]
        m = re.search(rf"(?m)^def\s+{re.escape(ep)}\s*\(", r["prompt"])
        prefix = r["prompt"][:m.start()] if m else ""   # imports + helper functions
        out.append({"id": "humaneval-" + r["task_id"].split("/")[1].zfill(3),
                    "source_id": r["task_id"], "prompt": r["prompt"], "func": ep,
                    "prefix": prefix, "test_code": r["test"] + f"\ncheck({ep})\n",
                    "reference": r["prompt"] + r["canonical_solution"]})
    return out


def mbpp(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    test_split = [r for r in rows if 11 <= r["task_id"] <= 510]  # standard MBPP test split
    picked = sorted(random.Random(SEED).sample(test_split, N_MBPP), key=lambda r: r["task_id"])
    out = []
    for r in picked:
        m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", r["test_list"][0])
        out.append({"id": f"mbpp-{r['task_id']:03d}", "source_id": r["task_id"],
                    "text": r["text"], "first_test": r["test_list"][0],
                    "func": m.group(1) if m else "",
                    "prefix": r.get("test_setup_code", "") or "",
                    "test_code": "\n".join(r["test_list"]) + "\n",
                    "reference": r["code"].replace("\r\n", "\n")})
    return out


def main() -> None:
    gsm_p, he_p, mbpp_p = sys.argv[1:4]
    OUT.mkdir(exist_ok=True)
    for name, rows in (("gsm8k", gsm8k(gsm_p)), ("humaneval", humaneval(he_p)), ("mbpp", mbpp(mbpp_p))):
        p = OUT / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{p.name}: {len(rows)} tasks")


if __name__ == "__main__":
    main()
