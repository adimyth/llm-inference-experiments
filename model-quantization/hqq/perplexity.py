"""Perplexity on wikitext-2 for an HQQ checkpoint.

HQQ runs on PyTorch, but it still needs its own script rather than reusing the transformers one: an HQQ checkpoint has to be loaded with `AutoHQQHFModel.from_quantized`, because transformers' generic reload path raises `NotImplementedError: QuantizationMethod.HQQ is not available yet` on this version (5.15.1). See quantize.py for the full note.

The measurement convention itself is not duplicated - the chunking and the second-half-only scoring rule come from perplexity_core.py, shared with every other perplexity script here.

`AutoHQQHFModel.save_quantized` doesn't write tokenizer files (only qmodel.pt + config.json), so the tokenizer loads from the original HF snapshot dir. Tokenization isn't affected by weight quantization, so this is the same tokenizer every other checkpoint in this project uses.

`device='mps'` is passed explicitly because HQQ's default is `device='cuda'`, which on a Mac either crashes or silently reloads onto the wrong device.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from hqq.models.hf.base import AutoHQQHFModel
from loguru import logger
from transformers import AutoTokenizer

import perplexity_core
from results_io import update


def run(model_dir: Path, tokenizer_dir: str, device: str, ctx_size: int) -> dict:
    logger.info(f"loading {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoHQQHFModel.from_quantized(str(model_dir), compute_dtype=torch.float16, device=device)
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
    stats["engine"] = "hqq"
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to an HQQ model directory")
    parser.add_argument("--tokenizer", required=True, help="local HF snapshot dir or hub repo id for the tokenizer")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. hqq-q4")
    parser.add_argument("--device", default="mps", help="HQQ defaults to cuda; pass mps on a Mac")
    parser.add_argument("--ctx-size", type=int, default=perplexity_core.CTX_SIZE)
    args = parser.parse_args()

    stats = run(Path(args.model), args.tokenizer, args.device, args.ctx_size)
    update(args.label, "perplexity", stats)
