"""Perplexity on wikitext-2 for an MLX checkpoint.

MLX is a separate framework, not PyTorch, so it needs its own loop: arrays are `mlx.core` arrays and the model comes from `mlx_lm.load`. What it does NOT get its own copy of is the measurement convention - the chunking and the second-half-only scoring rule live in perplexity_core.py, shared with every other perplexity script here, which is the whole point of that module.

WHY NOT mlx_lm's own `perplexity` subcommand. It samples random fixed-length chunks from a HF dataset's *train* split. Every other number in this project comes from the wikitext-2 *test* split processed sequentially in full. Different data and different sampling, so the two numbers would not mean the same thing, and a cross-method comparison built on them would be meaningless.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

import perplexity_core
from results_io import update


def run(model_path: Path, ctx_size: int) -> dict:
    model, tokenizer = load(str(model_path))

    def nll_fn(window, start):
        chunk = mx.array(window)[None]
        logits = model(chunk[:, :-1])[:, start:]
        targets = chunk[:, 1:][:, start:]
        losses = nn.losses.cross_entropy(logits, targets)
        return float(mx.sum(losses)), targets.size

    tokens = tokenizer.encode(perplexity_core.load_text())
    stats = perplexity_core.run(tokens, nll_fn, ctx_size)
    stats["engine"] = "mlx"
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to an MLX model directory")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. mlx-q4")
    parser.add_argument("--ctx-size", type=int, default=perplexity_core.CTX_SIZE)
    args = parser.parse_args()

    stats = run(Path(args.model), args.ctx_size)
    update(args.label, "perplexity", stats)
