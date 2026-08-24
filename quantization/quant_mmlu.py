"""MMLU accuracy for a GGUF checkpoint.

Perplexity is a proxy; this is closer to the real question, does the
quantized model still pick the right answer. Runs a fixed, stratified list
of 10 MMLU subtasks (3 STEM, 3 humanities, 2 social sciences, 2 other),
--limit 5 questions per subtask, for a fixed 50-question subset - identical
across every checkpoint this is run against (fp16, RTN, and every method
after it), so accuracy deltas are attributable to quantization rather than
to which questions got asked. lm_eval's --limit takes a plain positional
slice of each task's dataset (verified against its source, no shuffling
involved), so this really is the same 50 questions every time, not a random
sample that happens to differ per run.

Cut down from an original 500-question/20-task design after measuring the
real per-request cost (~5s, dominated by lm_eval re-processing the few-shot
prompt per request rather than by model size) made that too slow to run
once per checkpoint, let alone six times across every method in this plan.
50 questions (200 requests) trades statistical robustness for something
that actually finishes.

lm_eval's `gguf` model type talks to an HTTP server rather than embedding
llama.cpp itself. It has to be `llama_cpp.server` (from the
llama-cpp-python[server] package), not llama.cpp's own `llama-server`
binary: the latter's completions endpoint only ever returns logprobs for
newly generated tokens, never for echoed/appended text, which is what
lm_eval's loglikelihood scoring needs. This script launches that server
itself, points lm_eval at it, and tears it down after.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests
from loguru import logger

from results_io import update

# 3 STEM, 3 humanities, 2 social sciences, 2 other (MMLU's own grouping).
TASKS = [
    # STEM
    "mmlu_high_school_physics",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_biology",
    # humanities
    "mmlu_high_school_us_history",
    "mmlu_philosophy",
    "mmlu_world_religions",
    # social sciences
    "mmlu_high_school_geography",
    "mmlu_high_school_psychology",
    # other
    "mmlu_professional_medicine",
    "mmlu_marketing",
]
LIMIT_PER_TASK = 5  # 10 tasks x 5 = 50 questions
PORT = 8813


def wait_healthy(port: int, server: subprocess.Popen, timeout: float = 180) -> None:
    # llama_cpp.server has no /health endpoint (that's the separate C++
    # llama-server binary) - /v1/models is up once the model is loaded.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"llama_cpp.server exited early (code {server.returncode}), see server log")
        try:
            if requests.get(f"http://localhost:{port}/v1/models", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"llama_cpp.server did not become healthy within {timeout}s")


def run(model: Path, venv_python: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = (out_dir / "server.log").open("w")
    server = subprocess.Popen(
        [
            str(venv_python), "-m", "llama_cpp.server",
            "--model", str(model),
            "--n_ctx", "4096",
            "--port", str(PORT),
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        logger.info(f"waiting for llama_cpp.server on :{PORT}")
        wait_healthy(PORT, server)

        cmd = [
            str(venv_python), "-m", "lm_eval",
            "--model", "gguf",
            "--model_args", f"base_url=http://localhost:{PORT}",
            "--tasks", ",".join(TASKS),
            "--limit", str(LIMIT_PER_TASK),
            "--output_path", str(out_dir),
        ]
        logger.info(f"running lm_eval: {len(TASKS)} tasks x {LIMIT_PER_TASK} questions")
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall = time.perf_counter() - start
    finally:
        server.terminate()
        server.wait(timeout=30)

    result_file = sorted(out_dir.glob("**/results_*.json"))[-1]
    data = json.loads(result_file.read_text())["results"]

    n_correct, n_total = 0, 0
    for task, row in data.items():
        n = row.get("sample_len", LIMIT_PER_TASK)
        n_correct += row["acc,none"] * n
        n_total += n

    stats = {
        "accuracy": n_correct / n_total,
        "n_questions": n_total,
        "n_tasks": len(TASKS),
        "wall_seconds": wall,
    }
    logger.info(f"MMLU subset accuracy: {stats['accuracy']:.3f} ({n_total} questions, {wall:.0f}s)")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a GGUF checkpoint")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. fp16, rtn-q4_0")
    parser.add_argument("--venv-python", default="../.venv/bin/python", help="python with lm_eval + llama_cpp.server installed")
    parser.add_argument("--out-dir", default="results/mmlu_raw")
    args = parser.parse_args()

    stats = run(Path(args.model), Path(args.venv_python), Path(args.out_dir) / args.label)
    update(args.label, "mmlu", stats)
