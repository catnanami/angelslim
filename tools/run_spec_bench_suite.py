#!/usr/bin/env python
"""Run Eagle-3 HF-backend (pytorch) speculative-decoding benchmarks across
eight datasets: gsm8k, math, aime25, humaneval, mbpp, livecodebench,
mt-bench, and alpaca.

Per-dataset sample cap is 128; datasets with fewer available questions
(``aime25`` has 30, ``mt_bench`` and the bundled ``gsm8k`` / ``humaneval`` /
``alpaca`` have 80) run whatever exists. ``--temperature`` is required and may list
multiple values; each is benchmarked against every selected dataset.
``max_new_token=4096`` and ``deploy_backend=pytorch`` are fixed.

For datasets without a bundled ``dataset/<name>/question.jsonl``, this
script launches the matching preparer under ``tools/prepare_datasets/`` to
download from HuggingFace and write the question file in angelslim's schema.

Example::

    python tools/run_spec_bench_suite.py \\
        --base-model-path /path/to/Qwen3-8B \\
        --eagle-model-path /path/to/eagle3-qwen3-8b \\
        --model-id qwen3-8b-eagle3 \\
        --temperature 0 0.7
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
SPEC_BENCHMARK = REPO_ROOT / "tools" / "spec_benchmark.py"
PREPARER_DIR = REPO_ROOT / "tools" / "prepare_datasets"

BENCH_ALIASES = {"mt-bench": "mt_bench"}

PREPARERS = {
    "math": "math500.py",
    "aime25": "aime25.py",
    "mbpp": "mbpp.py",
    "livecodebench": "livecodebench.py",
}

DEFAULT_DATASETS = [
    "gsm8k",
    "math",
    "aime25",
    "humaneval",
    "mbpp",
    "livecodebench",
    "mt-bench",
    "alpaca",
]


def detect_gpu_count() -> int:
    """Best-effort GPU count detection via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return 1
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return max(1, len(lines))
    except Exception:
        return 1


def clear_gpu_memory() -> None:
    """Best-effort GPU memory cleanup before each dataset run."""
    cmd = ["nvidia-smi", "--gpu-reset"]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode == 0:
        print("[gpu] reset done")
        return
    # Fallback: at least trigger allocator cleanup in a short Python process.
    fallback = (
        "import gc\n"
        "gc.collect()\n"
        "try:\n"
        "    import torch\n"
        "    torch.cuda.empty_cache()\n"
        "except Exception:\n"
        "    pass\n"
    )
    subprocess.run([sys.executable, "-c", fallback], cwd=str(REPO_ROOT))
    stderr = result.stderr.strip()
    if stderr:
        print(f"[gpu] reset skipped: {stderr}")
    else:
        print("[gpu] reset skipped")


def ensure_dataset(name: str, num_samples: int) -> Path:
    qfile = DATASET_DIR / name / "question.jsonl"
    if qfile.exists():
        return qfile
    if name not in PREPARERS:
        raise FileNotFoundError(
            f"dataset/{name}/question.jsonl is missing and no preparer is registered"
        )
    preparer = PREPARER_DIR / PREPARERS[name]
    print(f"[prepare] {name}: running {preparer.name}")
    subprocess.check_call(
        [sys.executable, str(preparer), "--num-samples", str(num_samples)]
    )
    if not qfile.exists():
        raise RuntimeError(f"{preparer.name} did not produce {qfile}")
    return qfile


def run_one(name: str, temperature: float, args: argparse.Namespace) -> int:
    out_dir = Path(args.output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SPEC_BENCHMARK),
        "--base-model-path", args.base_model_path,
        "--eagle-model-path", args.eagle_model_path,
        "--model-id", args.model_id,
        "--deploy-backend", "pytorch",
        "--bench-name", name,
        "--mode", args.mode,
        "--temperature", str(temperature),
        "--max-new-token", "4096",
        "--question-end", str(args.num_samples),
        "--output-dir", str(out_dir),
        "--seed", str(args.seed),
        "--num-gpus-per-model", str(args.num_gpus_per_model),
        "--num-gpus-total", str(args.num_gpus_total),
    ]
    print(f"[run] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--eagle-model-path", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-root", default=str(REPO_ROOT / "benchmark_results"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--mode", default="both", choices=["eagle", "baseline", "both"])
    p.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        required=True,
        metavar="T",
        help="One or more sampling temperatures; each run is executed for every dataset.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-gpus-per-model", type=int, default=1)
    p.add_argument(
        "--num-gpus-total",
        type=int,
        default=None,
        help="Total GPUs. Default: auto-detect from nvidia-smi -L.",
    )
    p.add_argument(
        "--skip-on-error",
        action="store_true",
        default=True,
        help="continue running other datasets if one fails (default: on)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop immediately on first dataset failure",
    )
    args = p.parse_args()
    if args.num_gpus_total is None:
        args.num_gpus_total = detect_gpu_count()
    if args.num_gpus_per_model > args.num_gpus_total:
        raise ValueError(
            f"num-gpus-per-model ({args.num_gpus_per_model}) cannot exceed "
            f"num-gpus-total ({args.num_gpus_total})"
        )
    if args.num_gpus_total > args.num_gpus_per_model:
        print(
            f"[warn] num_gpus_total ({args.num_gpus_total}) > "
            f"num_gpus_per_model ({args.num_gpus_per_model})"
        )
        print(
            f"[warn] This starts multiple workers, each worker sees only "
            f"{args.num_gpus_per_model} GPU(s)."
        )
        print("[warn] Large models may offload weights to CPU under this setting.")

    failures = []
    successes = []
    continue_on_error = args.skip_on_error and (not args.fail_fast)
    for raw in args.datasets:
        name = BENCH_ALIASES.get(raw, raw)
        print(f"[start] {name}")
        clear_gpu_memory()
        try:
            ensure_dataset(name, args.num_samples)
        except Exception as e:
            print(f"[skip] {name}: dataset prep failed: {e}")
            if not continue_on_error:
                raise
            failures.append((name, None))
            continue
        for temperature in args.temperature:
            label = f"{name} (temperature={temperature})"
            print(f"[run slice] {label}")
            rc = run_one(name, temperature, args)
            if rc != 0:
                print(f"[fail] {label}: exit code {rc}")
                if not continue_on_error:
                    sys.exit(rc)
                failures.append((name, temperature))
            else:
                successes.append((name, temperature))

    print(f"[done] succeeded (dataset, temperature): {successes}")
    if failures:
        print(f"[done] failed (dataset, temperature): {failures}")
        sys.exit(1)
    print("[done] all datasets completed")


if __name__ == "__main__":
    main()
