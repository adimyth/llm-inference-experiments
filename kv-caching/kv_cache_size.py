"""KV cache size from model shape.

Pure arithmetic. No model is loaded and nothing is downloaded. This computes what the cache *would* cost for a given architecture, from published config values (layer count, KV head count, head dimension).
"""

from loguru import logger


def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    """Bytes of KV cache. The leading 2 is one tensor for K, one for V."""
    return 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes


def gb(*args, **kwargs):
    return kv_cache_bytes(*args, **kwargs) / 1024**3


# name, layers, kv_heads, head_dim, published context limit, attention style
MODELS = [
    ("Llama-2-7B", 32, 32, 128, 4_096, "multi-head, 32 KV heads"),
    ("Llama-3-8B", 32, 8, 128, 8_192, "grouped-query, 8 KV heads"),
    ("Mistral-7B", 32, 8, 128, 32_768, "grouped-query, 8 KV heads"),
]

HEADROOM_GB = 64  # an 80GB card holding ~16GB of fp16 weights

if __name__ == "__main__":
    logger.info("one sequence, fp16, at each model's published context limit")
    for name, layers, kv, hd, ctx, note in MODELS:
        logger.info(f"{name:12s} {note:26s} ctx={ctx:>6}  {gb(layers, kv, hd, ctx):5.2f} GB")

    logger.info("counterfactual: Llama-2's multi-head shape at a 32k window")
    logger.info(f"{gb(32, 32, 128, 32_768):.2f} GB of cache, against ~13 GB of fp16 weights")

    logger.info(f"concurrent sessions at full context, {HEADROOM_GB}GB headroom")
    for name, layers, kv, hd, ctx, _ in MODELS:
        n = int(HEADROOM_GB / gb(layers, kv, hd, ctx))
        logger.info(f"{name:12s} ctx={ctx:>6}  {n:>3} sessions")
