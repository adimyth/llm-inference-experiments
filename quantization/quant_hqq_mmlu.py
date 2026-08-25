"""MMLU accuracy for an HQQ checkpoint.

Same fixed 50-question subset as quant_mmlu.py (imported from there
directly). Like the MLX path, no HTTP server is needed: lm_eval's HFLM
class accepts an already-loaded transformers model directly, so this scores
the quantized model in-process.

--apply_chat_template=False to match how every other checkpoint in this
project was scored: lm_eval's `gguf` backend (used for the GGUF checkpoints)
sends the raw completion prompt with no chat formatting, so this needs the
same to stay comparable.
"""

import argparse
import json
import time
from pathlib import Path

import lm_eval
import torch
from hqq.models.hf.base import AutoHQQHFModel
from lm_eval.models.huggingface import HFLM
from loguru import logger
from transformers import AutoTokenizer

from quant_mmlu import LIMIT_PER_TASK, TASKS
from results_io import update


def run(model_dir: Path, tokenizer_dir: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoHQQHFModel.from_quantized(str(model_dir), compute_dtype=torch.float16, device="mps")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", device="mps", batch_size=1)

    logger.info(f"running lm_eval: {len(TASKS)} tasks x {LIMIT_PER_TASK} questions")
    start = time.perf_counter()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        apply_chat_template=False,
        limit=LIMIT_PER_TASK,
    )
    wall = time.perf_counter() - start

    data = results["results"]
    (out_dir / "results.json").write_text(json.dumps(data, indent=2, default=str))

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
    parser.add_argument("--model", required=True, help="path to an HQQ model directory")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="local HF snapshot dir or hub repo id to load the tokenizer from",
    )
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. hqq-q4")
    parser.add_argument("--out-dir", default="results/mmlu_raw")
    args = parser.parse_args()

    stats = run(Path(args.model), args.tokenizer, Path(args.out_dir) / args.label)
    update(args.label, "mmlu", stats)
