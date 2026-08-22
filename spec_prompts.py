"""Speculative decoding is workload-dependent.

Acceptance is how often the draft guesses what the target would have said, so it
depends on how predictable the text is. This runs the same pair over prompts of
differing predictability and reports speedup against acceptance.
"""

import argparse

import torch
from loguru import logger

from spec_decode import baseline, load, speculative, sync
from transformers import AutoTokenizer

PROMPTS = {
    "open prose": "The history of the printing press begins",
    "factual list": "The capital of France is Paris. The capital of Germany is Berlin. "
                    "The capital of Italy is",
    "code": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\n"
            "def mul(a, b):\n    return",
    "structured": "Monday: gym\nTuesday: gym\nWednesday: gym\nThursday: gym\nFriday:",
    "repetitive": "a b c a b c a b c a b c a b c a b c a b c",
}


def main(target_id, draft_id, n_tokens, ks, device, dtype):
    tok = AutoTokenizer.from_pretrained(target_id)
    target = load(target_id, device, dtype)
    draft = load(draft_id, device, dtype)
    logger.info(f"target {target_id} | draft {draft_id} | {device}")

    warm = tok("warming up the models before timing", return_tensors="pt").input_ids.to(device)
    baseline(target, warm, 4)
    speculative(target, draft, warm, 4, 4)
    sync(device)

    logger.info(f"{'prompt':<14} {'k':>2} {'base':>7} {'spec':>7} {'speedup':>8} "
                f"{'accept':>7} {'disc':>5}")
    for name, p in PROMPTS.items():
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        _, b = baseline(target, ids, n_tokens)
        for k in ks:
            _, s = speculative(target, draft, ids, n_tokens, k)
            logger.info(f"{name:<14} {k:>2} {b.wall:6.2f}s {s.wall:6.2f}s "
                        f"{b.wall / s.wall:7.2f}x {s.acceptance:6.1%} {s.discarded:>5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ks", default="5")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    main(a.target, a.draft, a.tokens, [int(x) for x in a.ks.split(",")],
         a.device, getattr(torch, a.dtype) if a.dtype else None)
