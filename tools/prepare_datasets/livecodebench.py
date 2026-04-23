"""Build ``dataset/livecodebench/question.jsonl`` from LiveCodeBench.

Prompt pattern ported from
``D3-Spec/SpecForge/benchmarks/benchmarker/livecodebench.py`` — uses the raw
``question_content`` field as the prompt. The ``_lite`` subset is tried first
because it is smaller and covers the same task distribution; the full
``code_generation`` repo is used as a fallback.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_hf_dataset, write_questions  # noqa: E402

BENCH_NAME = "livecodebench"


def _load():
    last_err = None
    for repo in ("livecodebench/code_generation_lite", "livecodebench/code_generation"):
        try:
            return load_hf_dataset(repo, split="test", trust_remote_code=True)
        except Exception as e:
            last_err = e
            print(f"[warn] could not load {repo}: {e}")
    raise RuntimeError(f"Failed to load any LiveCodeBench split: {last_err}")


def build(num_samples: int = 128) -> int:
    ds = _load()
    rows = []
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        prompt = (ex.get("question_content") or "").strip()
        if not prompt:
            continue
        rows.append(
            {
                "question_id": i,
                "category": "code",
                "turns": [prompt],
                "reference": [""],
            }
        )
    out = write_questions(BENCH_NAME, rows)
    print(f"wrote {len(rows)} samples to {out}")
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=128)
    build(p.parse_args().num_samples)
