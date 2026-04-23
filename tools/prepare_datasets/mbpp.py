"""Build ``dataset/mbpp/question.jsonl`` from ``google-research-datasets/mbpp``.

D3-Spec routes MBPP through a generic simple-SGL function instead of a
dedicated benchmarker, so this preparer uses the original MBPP paper's prompt
template (task description followed by the required test cases).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_hf_dataset, write_questions  # noqa: E402

BENCH_NAME = "mbpp"
PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {task}\n"
    "Your code should pass these tests:\n\n{tests}\n"
)


def build(num_samples: int = 128) -> int:
    ds = load_hf_dataset(
        "google-research-datasets/mbpp", split="test", name="sanitized"
    )
    rows = []
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        task = ex.get("prompt") or ex.get("text") or ""
        tests = "\n".join(ex.get("test_list") or [])
        reference = ex.get("code") or ""
        rows.append(
            {
                "question_id": i,
                "category": "code",
                "turns": [PROMPT_TEMPLATE.format(task=task, tests=tests)],
                "reference": [reference],
            }
        )
    out = write_questions(BENCH_NAME, rows)
    print(f"wrote {len(rows)} samples to {out}")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=128)
    build(p.parse_args().num_samples)
