"""One forward pass gives one prediction per position, not one prediction.

This is what makes verification cheap. Attention is causal, so position i sees
only positions <= i and its answer does not depend on any position after it.
All N predictions are therefore independent and computed together.

While writing, you cannot use this: the prediction after position 3 needs the
token at position 3, which is what you were trying to produce. A draft model
supplies those tokens as guesses, and checking them becomes one pass.
"""

import argparse

import torch
from loguru import logger
from transformers import AutoTokenizer

from spec_decode import load


@torch.no_grad()
def main(model_id, text, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = load(model_id, device, dtype)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits

    words = [tok.decode([i]) for i in ids[0].tolist()]
    logger.info(f"input {ids.shape[1]} tokens -> logits {tuple(logits.shape)}: "
                f"{logits.shape[1]} predictions from one pass")
    for i in range(len(words)):
        prefix = "".join(words[: i + 1])
        nxt = tok.decode([logits[0, i].argmax().item()])
        logger.info(f"  after {prefix!r:<46} -> {nxt!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text", default="The capital of France is Paris and it")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    main(a.model, a.text, a.device, getattr(torch, a.dtype))
