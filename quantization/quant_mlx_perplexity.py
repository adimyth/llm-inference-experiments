"""Perplexity on wikitext-2 for an MLX checkpoint.

mlx_lm ships its own `perplexity` subcommand, but it samples random chunks
from a HF dataset's *train* split - a different methodology from
quant_perplexity.py's llama-perplexity run, which processes the wikitext-2
*test* split sequentially in fixed windows. Using the built-in tool would
produce a number that isn't actually comparable to the GGUF checkpoints', so
this reimplements llama-perplexity's approach instead: same file
(data/wikitext-2-raw-test.txt), same fixed window size (512 tokens, its
default), same accounting (each window contributes ctx-1 next-token
predictions, teacher-forced), so the PPL values mean the same thing across
every checkpoint in this project, not just internally within MLX.
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from loguru import logger
from mlx_lm import load

from results_io import update

WIKITEXT = Path(__file__).parent / "data" / "wikitext-2-raw-test.txt"
CTX_SIZE = 512  # matches llama-perplexity's default


def run(model_path: Path, ctx_size: int) -> dict:
    if not WIKITEXT.exists():
        raise FileNotFoundError(f"{WIKITEXT} missing - run fetch_wikitext.py first")

    model, tokenizer = load(str(model_path))
    tokens = tokenizer.encode(WIKITEXT.read_text())
    n_chunks = len(tokens) // ctx_size
    logger.info(f"{len(tokens):,} tokens -> {n_chunks} chunks of {ctx_size}")

    total_nll, total_tokens = 0.0, 0
    for i in range(n_chunks):
        chunk = mx.array(tokens[i * ctx_size : (i + 1) * ctx_size])[None]
        logits = model(chunk[:, :-1])
        targets = chunk[:, 1:]
        losses = nn.losses.cross_entropy(logits, targets)
        total_nll += float(mx.sum(losses))
        total_tokens += targets.size
        if (i + 1) % 20 == 0 or i == n_chunks - 1:
            running_ppl = np.exp(total_nll / total_tokens)
            logger.info(f"chunk {i + 1}/{n_chunks}, running PPL = {running_ppl:.4f}")

    ppl = float(np.exp(total_nll / total_tokens))
    stats = {"ppl": ppl, "n_chunks": n_chunks, "ctx_size": ctx_size}
    logger.info(f"PPL = {ppl:.4f}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to an MLX model directory")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. mlx-q4")
    parser.add_argument("--ctx-size", type=int, default=CTX_SIZE)
    args = parser.parse_args()

    stats = run(Path(args.model), args.ctx_size)
    update(args.label, "perplexity", stats)
