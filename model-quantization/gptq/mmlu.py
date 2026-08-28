"""MMLU accuracy for any transformers-loadable checkpoint.

The generic counterpart to quant_hqq_mmlu.py. Same fixed 50-question subset, imported from quant_gguf_mmlu.py directly rather than restated here, so there is no way for the two to drift apart. lm_eval's HFLM accepts an already-loaded transformers model, so this scores in-process with no HTTP server, the same shape as the MLX and HQQ paths.

--apply_chat_template=False to match how every other checkpoint in this project was scored: lm_eval's `gguf` backend (used for the GGUF checkpoints) sends the raw completion prompt with no chat formatting, so this needs the same to stay comparable.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch_device
from mmlu_tasks import LIMIT_PER_TASK, TASKS
from results_io import update


def run(model_id: str, tokenizer_id: str, device: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading {model_id} on {torch_device.describe(device)}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # Loaded onto CPU and then moved, rather than device_map=device: accelerate's dispatch path is pathologically slow for an 8B model on MPS (ten minutes and still loading), and a single-GPU 8B model needs no sharding anyway. Same pattern the HQQ scripts use.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", device=device, batch_size=1)

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
        "device": device,
    }
    logger.info(f"MMLU subset accuracy: {stats['accuracy']:.3f} ({n_total} questions, {wall:.0f}s)")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint dir or hub repo id")
    parser.add_argument("--tokenizer", help="local HF snapshot dir or hub repo id; defaults to --model")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. awq-q4")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--out-dir", default="../results/mmlu_raw")
    args = parser.parse_args()

    stats = run(
        args.model,
        args.tokenizer or args.model,
        torch_device.resolve(args.device),
        Path(args.out_dir) / args.label,
    )
    update(args.label, "mmlu", stats)
