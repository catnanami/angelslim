"""Build ``dataset/aime25/question.jsonl`` from ``MathArena/aime_2025``.

Prompt pattern ported from
``D3-Spec/SpecForge/benchmarks/benchmarker/aime.py`` — same boxed-answer style
as MATH-500. AIME 2025 only has 30 problems, so the produced file will be
capped at 30 regardless of ``--num-samples``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_hf_dataset, write_questions  # noqa: E402

BENCH_NAME = "aime25"
PROMPT_TEMPLATE = (
    "{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def build(num_samples: int = 128) -> int:
    ds = load_hf_dataset("MathArena/aime_2025", split="train")
    rows = []
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        problem = ex.get("problem") or ex.get("Problem") or ""
        answer = ex.get("answer") or ex.get("Answer") or ""
        rows.append(
            {
                "question_id": i,
                "category": "math",
                "turns": [PROMPT_TEMPLATE.format(problem=problem)],
                "reference": [str(answer)],
            }
        )
    out = write_questions(BENCH_NAME, rows)
    print(f"wrote {len(rows)} samples to {out}")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=128)
    build(p.parse_args().num_samples)
