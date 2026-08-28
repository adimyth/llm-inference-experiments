"""Perplexity on wikitext-2 for any transformers-loadable checkpoint.

Covers the unquantized fp16 baseline (here) and the AWQ and GPTQ checkpoints (awq/perplexity.py, gptq/perplexity.py, identical copies), all of which load through plain `AutoModelForCausalLM`.

WHY THIS EXISTS ALONGSIDE fp16/perplexity.py. Two different tools measure perplexity in this project, because no single one reads every format:

* `llama-perplexity` (llama.cpp) for the GGUF checkpoints - fp16, RTN and the k-quants. That's fp16/perplexity.py.
* this PyTorch loop for anything transformers can load, plus sibling implementations for MLX and HQQ, none of which llama.cpp can read.

Running BOTH against the same unquantized fp16 weights is what proves the two agree, and therefore that a GGUF checkpoint's perplexity is comparable to an MLX or AWQ one. They land at 7.395 (llama.cpp) and 7.365 (here), a 0.4% gap consistent with fp16 kernel differences between the two backends.

That cross-check is not academic. Before it existed, this loop scored the whole 512-token window while llama-perplexity scored only the second half, and comparing across the two made MLX and HQQ look ~35% worse than fp16 when the real cost was 6-8%. The windowing convention now lives in one place, perplexity_core.py, which explains it in full.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import perplexity_core
import torch_device
from results_io import update


def run(model_id: str, tokenizer_id: str, device: str, ctx_size: int) -> dict:
    logger.info(f"loading {model_id} on {torch_device.describe(device)}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # Loaded onto CPU and then moved, rather than device_map=device: accelerate's dispatch path is pathologically slow for an 8B model on MPS (ten minutes and still loading), and a single-GPU 8B model needs no sharding anyway.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    model.eval()

    def nll_fn(window, start):
        with torch.no_grad():
            chunk = torch.tensor(window, device=device)[None]
            logits = model(chunk[:, :-1]).logits[:, start:]
            targets = chunk[:, 1:][:, start:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                reduction="sum",
            )
            return loss.item(), targets.numel()

    tokens = tokenizer.encode(perplexity_core.load_text())
    stats = perplexity_core.run(tokens, nll_fn, ctx_size)
    stats["device"] = device
    stats["engine"] = "transformers"
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint dir or hub repo id")
    parser.add_argument("--tokenizer", help="local HF snapshot dir or hub repo id; defaults to --model")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. fp16-torch, awq-q4")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--ctx-size", type=int, default=perplexity_core.CTX_SIZE)
    args = parser.parse_args()

    stats = run(args.model, args.tokenizer or args.model, torch_device.resolve(args.device), args.ctx_size)
    update(args.label, "perplexity", stats)
