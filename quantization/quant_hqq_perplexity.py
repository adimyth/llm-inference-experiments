"""Perplexity on wikitext-2 for an HQQ checkpoint.

Same file, same fixed 512-token windows, same teacher-forced next-token
accounting as quant_perplexity.py (GGUF, via llama-perplexity) and
quant_mlx_perplexity.py - so this number means the same thing as every
other checkpoint's in this project, not just internally.

`AutoHQQHFModel.save_quantized` doesn't write tokenizer files (only
qmodel.pt + config.json), so the tokenizer loads from the original HF
snapshot dir - tokenization isn't affected by weight quantization, so
this is the same tokenizer every other checkpoint in this project uses.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hqq.models.hf.base import AutoHQQHFModel
from loguru import logger
from transformers import AutoTokenizer

from results_io import update

WIKITEXT = Path(__file__).parent / "data" / "wikitext-2-raw-test.txt"
CTX_SIZE = 512  # matches llama-perplexity's default


def run(model_dir: Path, tokenizer_dir: str, ctx_size: int) -> dict:
    if not WIKITEXT.exists():
        raise FileNotFoundError(f"{WIKITEXT} missing - run fetch_wikitext.py first")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoHQQHFModel.from_quantized(str(model_dir), compute_dtype=torch.float16, device="mps")
    model.eval()

    tokens = tokenizer.encode(WIKITEXT.read_text())
    n_chunks = len(tokens) // ctx_size
    logger.info(f"{len(tokens):,} tokens -> {n_chunks} chunks of {ctx_size}")

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = torch.tensor(tokens[i * ctx_size : (i + 1) * ctx_size], device="mps")[None]
            logits = model(chunk[:, :-1]).logits
            targets = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum")
            total_nll += loss.item()
            total_tokens += targets.numel()
            if (i + 1) % 20 == 0 or i == n_chunks - 1:
                running_ppl = np.exp(total_nll / total_tokens)
                logger.info(f"chunk {i + 1}/{n_chunks}, running PPL = {running_ppl:.4f}")

    ppl = float(np.exp(total_nll / total_tokens))
    stats = {"ppl": ppl, "n_chunks": n_chunks, "ctx_size": ctx_size}
    logger.info(f"PPL = {ppl:.4f}")
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
    parser.add_argument("--ctx-size", type=int, default=CTX_SIZE)
    args = parser.parse_args()

    stats = run(Path(args.model), args.tokenizer, args.ctx_size)
    update(args.label, "perplexity", stats)
