"""Measure the KV cache instead of computing it.

Generates with the cache on and reads the actual tensors the model keeps,
then checks them against the formula in kv_cache_size.py. Confirms that the
arithmetic used to size a serving fleet is the arithmetic the runtime does.
"""

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from kv_cache_size import kv_cache_bytes

DEFAULTS = {"model": "gpt2", "prompt_tokens": 64, "checkpoints": (0, 64, 256, 512)}


def iter_kv(past):
    """Yield every K and V tensor in the cache.

    transformers >=5 returns a DynamicCache whose .layers hold .keys/.values.
    Older versions return a plain tuple of (key, value) pairs per layer.
    """
    layers = getattr(past, "layers", past)
    for layer in layers:
        if hasattr(layer, "keys"):
            yield layer.keys
            yield layer.values
        else:
            yield from layer


def cache_bytes(past):
    """Real memory held by the cache, summed over every tensor."""
    return sum(t.numel() * t.element_size() for t in iter_kv(past))


def shape_of(cfg):
    """Layer count, KV head count and head dim, across differing config names."""
    layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    q_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    kv_heads = getattr(cfg, "num_key_value_heads", None) or q_heads
    hidden = getattr(cfg, "hidden_size", None) or cfg.n_embd
    head_dim = getattr(cfg, "head_dim", None) or hidden // q_heads
    return layers, q_heads, kv_heads, head_dim


@torch.no_grad()
def main(model_id=None, device="cpu", dtype=None, prompt_tokens=None, checkpoints=None):
    model_id = model_id or DEFAULTS["model"]
    prompt_tokens = prompt_tokens or DEFAULTS["prompt_tokens"]
    checkpoints = checkpoints or DEFAULTS["checkpoints"]

    logger.info(f"loading {model_id} on {device}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    layers, q_heads, kv_heads, head_dim = shape_of(model.config)
    dtype_bytes = next(model.parameters()).element_size()
    logger.info(
        f"{model_id}: {layers} layers, {q_heads} query heads, {kv_heads} KV heads, "
        f"head_dim {head_dim}, {dtype_bytes} bytes/value"
    )

    ids = torch.randint(100, 5000, (1, prompt_tokens), device=device)
    out = model(ids, use_cache=True)
    past = out.past_key_values
    generated = 0

    for target in checkpoints:
        while generated < target:
            nxt = out.logits[:, -1:].argmax(-1)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            generated += 1

        seq_len = prompt_tokens + generated
        measured = cache_bytes(past)
        predicted = kv_cache_bytes(layers, kv_heads, head_dim, seq_len, 1, dtype_bytes)
        logger.info(
            f"seq_len {seq_len:>4} | measured {measured / 1024**2:7.2f} MB | "
            f"formula {predicted / 1024**2:7.2f} MB | ratio {measured / predicted:.3f}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="e.g. float16")
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULTS["prompt_tokens"])
    ap.add_argument("--checkpoints", default="0,64,256,512")
    a = ap.parse_args()
    main(
        a.model,
        a.device,
        getattr(torch, a.dtype) if a.dtype else None,
        a.prompt_tokens,
        tuple(int(x) for x in a.checkpoints.split(",")),
    )
