"""Tokens/sec for a GGUF checkpoint, via llama-bench.

Reports two numbers: pp (prompt processing, how fast the model chews through
input context) and tg (text generation, how fast it produces new tokens one
at a time). tg is the more telling number for quantization, since decode is
memory-bandwidth bound and a smaller checkpoint moves less data per token.

llama-bench already handles warmup and repeats; this just runs it with JSON
output and records the medians.
"""

import argparse
import json
import subprocess
from pathlib import Path
from statistics import median

from loguru import logger

from results_io import update


def run(model: Path, n_prompt: int, n_gen: int, repeats: int) -> dict:
    cmd = [
        "llama-bench",
        "-m", str(model),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repeats),
        "-o", "json",
    ]
    logger.info(f"running llama-bench on {model.name} (pp{n_prompt}, tg{n_gen}, x{repeats})")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)

    pp = [r["avg_ts"] for r in rows if r["n_prompt"] == n_prompt and r["n_gen"] == 0]
    tg = [r["avg_ts"] for r in rows if r["n_gen"] == n_gen and r["n_prompt"] == 0]

    stats = {
        "pp_tokens_per_sec": median(pp) if pp else None,
        "tg_tokens_per_sec": median(tg) if tg else None,
    }
    logger.info(f"pp{n_prompt}: {stats['pp_tokens_per_sec']:.1f} t/s, tg{n_gen}: {stats['tg_tokens_per_sec']:.1f} t/s")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a GGUF checkpoint")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. fp16, rtn-q4_0")
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stats = run(Path(args.model), args.n_prompt, args.n_gen, args.repeats)
    update(args.label, "throughput", stats)
