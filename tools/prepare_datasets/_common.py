"""Shared helpers for angelslim spec-benchmark dataset preparers.

Each preparer writes a ``dataset/<name>/question.jsonl`` file matching the
schema consumed by the pytorch speculative-decoding benchmark engine:

    {"question_id": int, "category": str, "turns": [str, ...], "reference": [str, ...]}
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"


def load_hf_dataset(path: str, split: str, name: str = None, **kwargs):
    from datasets import load_dataset

    if name is not None:
        return load_dataset(path, name, split=split, **kwargs)
    return load_dataset(path, split=split, **kwargs)


def write_questions(bench_name: str, rows: list) -> Path:
    out = DATASET_DIR / bench_name / "question.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
