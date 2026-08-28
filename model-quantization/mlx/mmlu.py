"""MMLU accuracy for an MLX checkpoint.

Same fixed 50-question subset as quant_gguf_mmlu.py (imported from there directly, so there's exactly one definition of which questions get asked). Unlike the GGUF path, no HTTP server/bridge is needed here: mlx_lm ships its own MLXLM class implementing lm_eval's LM interface natively.

Calls lm_eval.simple_evaluate directly rather than going through mlx_lm's `evaluate` CLI subcommand: that CLI concatenates every task name into its output filename, which blows past macOS's 255-char filename limit once you pass 10 tasks and crashes *after* scoring finishes but *before* it prints or saves anything - the run's real work completes, then the whole result is thrown away. Doing it in-process means we control the output path.

--apply-chat-template=False to match how the GGUF run was scored: lm_eval's `gguf` backend sends the raw completion prompt with no chat formatting, so this needs the same to stay comparable rather than silently asking an easier (chat-formatted) version of the question.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import lm_eval
from loguru import logger
from mlx_lm.evaluate import MLXLM

from mmlu_tasks import LIMIT_PER_TASK, TASKS
from results_io import update


def run(model_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = MLXLM(str(model_path), use_chat_template=False)

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
    parser.add_argument("--model", required=True, help="path to an MLX model directory")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. mlx-q4")
    parser.add_argument("--out-dir", default="../results/mmlu_raw")
    args = parser.parse_args()

    stats = run(Path(args.model), Path(args.out_dir) / args.label)
    update(args.label, "mmlu", stats)
