"""Generation with and without the KV cache, timed back to back.

Downloads GPT-2 (~526MB) on first run. Everything runs on CPU so the
comparison is not distorted by accelerator scheduling.
"""

import statistics
import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
PROMPT_TOKENS = 64
LENGTHS = (32, 64, 128, 256, 512)
REPEATS = 3  # report the median, single runs are noisy


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


def main():
    logger.info(f"loading {MODEL} (downloads ~526MB on first run)")
    AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    logger.info(f"torch {torch.__version__}, device cpu, dtype {next(model.parameters()).dtype}")

    prompt = torch.randint(100, 5000, (1, PROMPT_TOKENS))
    generate(model, prompt, 4, use_cache=True)  # warm up
    logger.info(f"prompt {PROMPT_TOKENS} tokens, greedy decoding")

    def timed(n, use_cache):
        runs = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            generate(model, prompt, n, use_cache=use_cache)
            runs.append(time.perf_counter() - t0)
        return statistics.median(runs)

    logger.info(f"median of {REPEATS} runs per cell")
    for n in LENGTHS:
        off, on = timed(n, False), timed(n, True)
        logger.info(f"{n:>4} new tokens | no cache {off:6.2f}s | cache {on:5.2f}s | {off / on:5.1f}x")


if __name__ == "__main__":
    main()
