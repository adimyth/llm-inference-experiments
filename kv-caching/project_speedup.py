"""Project the KV cache speedup at generation lengths that were not run end to end.

The speedup is not the `vs len-1` column of the forward-cost table. The cached
path pays cost(1) on every step, but the uncached path pays cost(L) averaged
over every length from the prompt to the end, so the endpoint ratio is a
ceiling rather than the answer. This integrates instead.

Validated against the one cell that was measured end to end (512 new tokens):
predicts 1.63x against a measured 1.79x, so it runs about 9% conservative.
"""

# Measured on an L40S, Llama 3.1 8B fp16, median of 5. See results/forward-cost-l40s/.
FORWARD_MS = [(1, 25.46), (64, 29.46), (320, 42.01), (512, 49.13),
              (1024, 90.32), (2048, 176.51), (4096, 375.98), (8192, 850.23)]


def cost(seq_len):
    """Interpolate between measured points, extrapolating on the final slope."""
    if seq_len <= FORWARD_MS[0][0]:
        return FORWARD_MS[0][1]
    for (a, ca), (b, cb) in zip(FORWARD_MS, FORWARD_MS[1:]):
        if a <= seq_len <= b:
            return ca + (cb - ca) * (seq_len - a) / (b - a)
    (a, ca), (b, cb) = FORWARD_MS[-2], FORWARD_MS[-1]
    return cb + (cb - ca) / (b - a) * (seq_len - b)


def predict(new_tokens, prompt=64):
    uncached = sum(cost(prompt + i) for i in range(new_tokens))
    cached = new_tokens * cost(1)
    return uncached / 1000, cached / 1000, uncached / cached


if __name__ == "__main__":
    print(f"{'new tokens':>11} {'uncached':>10} {'cached':>8} {'speedup':>9}")
    for n in (512, 1024, 2048, 4096, 8192):
        u, c, s = predict(n)
        print(f"{n:>11} {u:>9.2f}s {c:>7.2f}s {s:>8.1f}x")
