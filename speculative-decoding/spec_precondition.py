"""Why speculative decoding can work at all.

Generation is serial: token N+1 needs token N. Verification is not. This measures how much a forward pass costs as a function of how many tokens it checks at once. The gap between those two is the entire budget speculative decoding spends.

No draft model, no speculation. One model, warm cache, varying batch of positions.
"""

import argparse
import copy
import statistics
import time

import torch
from loguru import logger

from spec_decode import load, sync

PREFIX_TOKENS = 128
K_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
REPEATS = 9


@torch.no_grad()
def time_forward(model, warm_cache, k, repeats, device):
    """Median wall time for one forward pass over k new tokens against a warm cache."""
    runs = []
    for _ in range(repeats):
        ids = torch.randint(100, 5000, (1, k), device=device)
        past = copy.deepcopy(warm_cache)  # outside the timed region
        sync(device)
        t0 = time.perf_counter()
        model(ids, past_key_values=past, use_cache=True)
        sync(device)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


@torch.no_grad()
def main(model_id, repeats, device="cpu", dtype=None):
    logger.info(f"loading {model_id} on {device}")
    model = load(model_id, device, dtype)
    logger.info(f"torch {torch.__version__}, {device}, dtype {dtype}")

    prefix = torch.randint(100, 5000, (1, PREFIX_TOKENS), device=device)
    warm = model(prefix, use_cache=True).past_key_values

    for _ in range(3):  # warm up before any timing
        time_forward(model, warm, 4, 2, device)

    base = None
    logger.info(f"prefix {PREFIX_TOKENS} tokens, median of {repeats}")
    for k in K_VALUES:
        t = time_forward(model, warm, k, repeats, device)
        base = base or t
        logger.info(
            f"k={k:>3} | pass {t * 1000:7.2f} ms | {t / base:5.2f}x vs k=1 | "
            f"per token {t / k * 1000:6.2f} ms | {base / (t / k):5.1f}x cheaper/token"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()
    main(a.model, a.repeats, a.device,
         getattr(torch, a.dtype) if a.dtype else None)
