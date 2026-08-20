# LLM Inference Experiments

Scripts behind the LLM Inference essays on adimyth.in. Every figure quoted in those essays is produced by running these.

## Setup

```bash
uv venv --python 3.12 --python-preference only-managed
uv pip install -r requirements.txt
```

`--python-preference only-managed` matters on Apple Silicon. A universal `python3` from python.org resolves to its x86_64 slice, and torch ships no Intel macOS wheels, so the install fails on a wheel-availability error that does not mention architecture.

## KV Caching

### kv_cache_size.py

```bash
.venv/bin/python kv_cache_size.py
```

Arithmetic only. No model is loaded and nothing is downloaded. It evaluates

```
2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes
```

against published config values for three architectures, then derives the concurrency ceiling for a given amount of free VRAM. The leading 2 is one tensor for K and one for V. Edit `MODELS` and `HEADROOM_GB` to change it.

Output, fp16, one sequence, at each model's published context limit:

| Model      | Attention                 | Context | Cache   |
| ---------- | ------------------------- | ------- | ------- |
| Llama-2-7B | multi-head, 32 KV heads   | 4K      | 2.00 GB |
| Llama-3-8B | grouped-query, 8 KV heads | 8K      | 1.00 GB |
| Mistral-7B | grouped-query, 8 KV heads | 32K     | 4.00 GB |

### kv_cache_timing.py

```bash
.venv/bin/python kv_cache_timing.py
```

Generates with `use_cache=True` and `use_cache=False` back to back and reports the median of 3 runs per cell. Downloads GPT-2 (~526MB) on first run.

Fixed at GPT-2 124M, fp32, CPU, 64-token prompt, greedy decoding, one warm-up pass before timing. CPU rather than MPS so the comparison is not distorted by accelerator scheduling.

Measured on MacBook Pro, Apple M4 Pro (8P+4E), 48GB, macOS 15.7.5, torch 2.13.0, transformers 5.15.1:

| New tokens | No cache | Cache | Speedup |
| ---------- | -------- | ----- | ------- |
| 32         | 0.94s    | 0.24s | 4.0x    |
| 64         | 2.12s    | 0.45s | 4.7x    |
| 128        | 5.12s    | 0.90s | 5.7x    |
| 256        | 14.99s   | 1.84s | 8.2x    |
| 512        | 50.61s   | 3.96s | 12.8x   |

The speedup column grows because the uncached path is quadratic in output length and the cached path is linear. Per doubling, uncached time grows 2.3x, 2.4x, 2.9x, 3.4x; cached grows 1.9x, 2.0x, 2.0x, 2.2x.

Absolute times are small-model-on-CPU figures and will differ on other hardware. The shape will not.

### kv_cache_measured.py

```bash
.venv/bin/python kv_cache_measured.py
```

Reads the actual K and V tensors the model retains during generation and checks them against `kv_cache_size.py`. Confirms the formula used to size a fleet is the arithmetic the runtime performs.

GPT-2, fp32, 64-token prompt:

| seq_len | Measured | Formula  | Ratio |
| ------- | -------- | -------- | ----- |
| 64      | 4.50 MB  | 4.50 MB  | 1.000 |
| 128     | 9.00 MB  | 9.00 MB  | 1.000 |
| 320     | 22.50 MB | 22.50 MB | 1.000 |
| 576     | 40.50 MB | 40.50 MB | 1.000 |

Exact at every checkpoint, and flat at 0.0703 MB per token. `seq_len` is prompt plus tokens generated so far, which is what the cache holds.
