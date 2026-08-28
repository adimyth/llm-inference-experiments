"""The wikitext-2 perplexity measurement convention, defined exactly once.

Perplexity is not fully specified by "perplexity on wikitext-2". You also have to say how the text is chunked and which tokens inside each chunk are scored, and different answers give materially different numbers for the same weights. This module fixes those choices for every method in this project so no two checkpoints can be measured differently.

THE CONVENTION, matching llama.cpp's `llama-perplexity`:

* The wikitext-2-raw **test** split, whole, in order. Not the train split, not a random sample of it.
* Fixed windows of 512 tokens (llama-perplexity's default), no overlap.
* Within each window, only the **second half** is scored. llama.cpp sets `const int first = n_ctx/2` (tools/perplexity/perplexity.cpp) and scores only tokens from there on, so every scored token has at least 256 tokens of context behind it. This follows https://huggingface.co/docs/transformers/perplexity.

WHY THE SECOND-HALF RULE MATTERS, learned the hard way. This project originally scored the *whole* window in its PyTorch and MLX implementations while using llama-perplexity for the GGUF checkpoints. Scoring the whole window includes the first few tokens of every chunk, which have almost no context and are close to unpredictable for any model, and that inflates perplexity a lot: the same fp16 checkpoint reads **9.46** scoring the full window against **7.36** scoring the second half. Neither number is wrong, but they are not the same measurement. Because the two conventions were mixed across engines, MLX and HQQ appeared to cost ~35% perplexity when they actually cost 6-8%. Every perplexity script in this repo now routes through this module so that can't happen again.

Engines differ in how they load a model and compute a loss (PyTorch, MLX, HQQ), so each supplies its own `nll_fn`. The chunking, the start index, and the accumulation live here.
"""

from pathlib import Path

import numpy as np
from loguru import logger

WIKITEXT = Path(__file__).parent / "data" / "wikitext-2-raw-test.txt"
CTX_SIZE = 512  # llama-perplexity's default


def load_text() -> str:
    if not WIKITEXT.exists():
        raise FileNotFoundError(f"{WIKITEXT} missing - run fetch_wikitext.py first")
    return WIKITEXT.read_text()


def scored_start(ctx_size: int = CTX_SIZE) -> int:
    """First index into the *target* sequence that counts towards perplexity.

    A window of `ctx_size` tokens is fed in as `window[:-1]` and predicts `window[1:]`, so target index j holds the prediction for window position j+1. llama-perplexity scores window positions [ctx/2, ctx), so that range begins at ctx/2 - 1 here and yields ctx/2 scored tokens.
    """
    return ctx_size // 2 - 1


def run(tokens, nll_fn, ctx_size: int = CTX_SIZE, log_every: int = 20) -> dict:
    """Score `tokens` window by window and return the perplexity stats dict.

    `nll_fn(window, start)` gets one window of `ctx_size` token ids and the index to start scoring from, and returns `(summed_negative_log_likelihood, n_tokens_scored)` for that window. Everything else - chunking, the second-half rule, the accumulation, the final exponentiation - is fixed here.
    """
    n_chunks = len(tokens) // ctx_size
    start = scored_start(ctx_size)
    logger.info(f"{len(tokens):,} tokens -> {n_chunks} windows of {ctx_size}, scoring the last {ctx_size // 2} of each")

    total_nll, total_tokens = 0.0, 0
    for i in range(n_chunks):
        window = tokens[i * ctx_size : (i + 1) * ctx_size]
        nll, n = nll_fn(window, start)
        total_nll += nll
        total_tokens += n
        if (i + 1) % log_every == 0 or i == n_chunks - 1:
            logger.info(f"window {i + 1}/{n_chunks}, running PPL = {np.exp(total_nll / total_tokens):.4f}")

    ppl = float(np.exp(total_nll / total_tokens))
    logger.info(f"PPL = {ppl:.4f}")
    return {
        "ppl": ppl,
        "n_chunks": n_chunks,
        "ctx_size": ctx_size,
        "scored_per_chunk": ctx_size // 2,
        "n_tokens_scored": total_tokens,
    }
