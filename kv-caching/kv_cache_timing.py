"""Generation with and without the KV cache, timed back to back.

Defaults to GPT-2 on CPU, which is what the essay's table reports: a small model on CPU so the comparison is not distorted by accelerator scheduling. Pass --model/--device/--dtype to time a different model, which produces a separate table rather than more rows in that one.
"""

import statistics
import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULTS = {
    "model": "gpt2",
    "device": "cpu",
    "prompt_tokens": 64,
    "lengths": (32, 64, 128, 256, 512),
    "repeats": 3,  # report the median, single runs are noisy
}


def sync(device):
    """Accelerators queue work asynchronously. Without this we time the queueing, not the work."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@torch.no_grad()
def generate(model, prompt_ids, new_tokens, use_cache):
    ids, past = prompt_ids, None
    for _ in range(new_tokens):
        if use_cache and past is not None:
            out = model(ids[:, -1:], past_key_values=past, use_cache=True)
        else:
            out = model(ids, use_cache=use_cache)
        past = out.past_key_values if use_cache else None
        ids = torch.cat([ids, out.logits[:, -1:].argmax(-1)], dim=-1)
    return ids


def main(model_id=None, device=None, dtype=None, prompt_tokens=None, lengths=None, repeats=None):
    model_id = model_id or DEFAULTS["model"]
    device = device or DEFAULTS["device"]
    prompt_tokens = prompt_tokens or DEFAULTS["prompt_tokens"]
    lengths = lengths or DEFAULTS["lengths"]
    repeats = repeats or DEFAULTS["repeats"]

    logger.info(f"loading {model_id} on {device}")
    AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    logger.info(
        f"torch {torch.__version__}, device {device}, dtype {next(model.parameters()).dtype}"
    )

    prompt = torch.randint(100, 5000, (1, prompt_tokens), device=device)
    generate(model, prompt, 4, use_cache=True)  # warm up, the first pass pays lazy init
    sync(device)
    logger.info(f"prompt {prompt_tokens} tokens, greedy decoding")

    def timed(n, use_cache):
        runs = []
        for _ in range(repeats):
            sync(device)
            t0 = time.perf_counter()
            generate(model, prompt, n, use_cache=use_cache)
            sync(device)
            runs.append(time.perf_counter() - t0)
        return statistics.median(runs)

    logger.info(f"median of {repeats} runs per cell")
    for n in lengths:
        off, on = timed(n, False), timed(n, True)
        logger.info(f"{n:>4} new tokens | no cache {off:6.2f}s | cache {on:5.2f}s | {off / on:5.1f}x")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--device", default=DEFAULTS["device"])
    ap.add_argument("--dtype", default=None, help="e.g. float16")
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULTS["prompt_tokens"])
    ap.add_argument("--lengths", default=",".join(str(n) for n in DEFAULTS["lengths"]))
    ap.add_argument("--repeats", type=int, default=DEFAULTS["repeats"])
    a = ap.parse_args()
    main(
        a.model,
        a.device,
        getattr(torch, a.dtype) if a.dtype else None,
        a.prompt_tokens,
        tuple(int(x) for x in a.lengths.split(",")),
        a.repeats,
    )
