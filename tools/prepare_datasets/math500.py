"""Build ``dataset/math/question.jsonl`` from ``HuggingFaceH4/MATH-500``.

Prompt pattern ported from
``D3-Spec/SpecForge/benchmarks/benchmarker/math500.py`` — the model is asked to
finalize its answer inside ``\\boxed{...}``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_hf_dataset, write_questions  # noqa: E402

BENCH_NAME = "math"
PROMPT_TEMPLATE = (
    "{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def build(num_samples: int = 128) -> int:
    ds = load_hf_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        rows.append(
            {
                "question_id": i,
                "category": "math",
                "turns": [PROMPT_TEMPLATE.format(problem=ex["problem"])],
                "reference": [str(ex.get("answer", ""))],
            }
        )
    out = write_questions(BENCH_NAME, rows)
    print(f"wrote {len(rows)} samples to {out}")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=128)
    build(p.parse_args().num_samples)
