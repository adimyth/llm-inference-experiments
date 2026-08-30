# KV Caching

Essay: [Why LLM Inference Is Slow, and What a KV Cache Fixes](https://adimyth.in/essays/llm-inference-kv-caching)

Run from this folder with `../.venv/bin/python <script>`. See the [root README](../README.md) for setup and method notes.

## kv_cache_size.py

Arithmetic only. No model is loaded and nothing is downloaded. It evaluates

```
2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes
```

against published config values for three architectures, then derives the concurrency ceiling for a given amount of free VRAM. The leading 2 is one tensor for K and one for V. Edit `MODELS` and `HEADROOM_GB` to change it.

Output, fp16, one sequence, at each model's published context limit:

| Model | Attention | Context | Cache |
| --- | --- | --- | --- |
| Llama-2-7B | multi-head, 32 KV heads | 4K | 2.00 GB |
| Llama-3-8B | grouped-query, 8 KV heads | 8K | 1.00 GB |
| Llama-3.1-8B | grouped-query, 8 KV heads | 128K | 16.00 GB |
| Mistral-7B | grouped-query, 8 KV heads | 32K | 4.00 GB |

Llama 3.1 is the row worth staring at. It has the same shape as Llama 3, 32 layers and 8 KV heads, and a context window 16 times longer. One session at full context needs **16 GB of cache against roughly 16 GB of fp16 weights**: the cache for a single user costs as much as the model serving them. At the 64GB headroom the script assumes, that is 4 concurrent sessions.

## kv_cache_timing.py

Generates with `use_cache=True` and `use_cache=False` back to back and reports the median of 3 runs per cell. Downloads GPT-2 (~526MB) on first run.

Fixed at GPT-2 124M, fp32, CPU, 64-token prompt, greedy decoding, one warm-up pass before timing. CPU rather than MPS so the comparison is not distorted by accelerator scheduling.

| New tokens | No cache | Cache | Speedup |
| --- | --- | --- | --- |
| 32 | 0.94s | 0.24s | 4.0x |
| 64 | 2.12s | 0.45s | 4.7x |
| 128 | 5.12s | 0.90s | 5.7x |
| 256 | 14.99s | 1.84s | 8.2x |
| 512 | 50.61s | 3.96s | 12.8x |

Read the two middle columns per doubling. Cached, a doubling costs about 2x every time, which is linear: twice the tokens, twice the work. Uncached, the cost of a doubling is itself rising, 2.3x then 2.4x then 2.9x then 3.4x, heading toward the 4x that pure quadratic growth would charge. The cached path is not perfectly linear either, drifting to 2.2x at the end, because a cached decode step still attends over a context that keeps growing.

**Absolute times and the speedup column are both specific to a small model on a CPU. The shape does not transfer either, which was measured rather than assumed.** See the L40S tables below.

## kv_cache_measured.py

Reads the actual K and V tensors the model retains during generation and checks them against `kv_cache_size.py`. Confirms the formula used to size a fleet is the arithmetic the runtime performs.

GPT-2, fp32, 64-token prompt:

| seq_len | Measured | Formula | Ratio |
| --- | --- | --- | --- |
| 64 | 4.50 MB | 4.50 MB | 1.000 |
| 128 | 9.00 MB | 9.00 MB | 1.000 |
| 320 | 22.50 MB | 22.50 MB | 1.000 |
| 576 | 40.50 MB | 40.50 MB | 1.000 |

Exact at every checkpoint, and flat at 0.0703 MB per token. `seq_len` is prompt plus tokens generated so far, which is what the cache holds.


---

## The same scripts on an L40S

Everything above is GPT-2 on CPU. Rerunning on a rented NVIDIA L40S changes the answer enough that the CPU speedup column should not be quoted as the cost of not caching.

Three runs, each changing one variable from the one before it, so the difference is attributable:

| Run | Model | dtype | Hardware | 32 | 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | GPT-2 124M | fp32 | CPU | 4.0x | 4.7x | 5.7x | 8.2x | **12.8x** |
| hardware | GPT-2 124M | fp32 | L40S | 1.0x | 1.0x | 1.0x | 1.0x | **1.1x** |
| dtype | GPT-2 124M | fp16 | L40S | 1.0x | 1.0x | 1.0x | 1.0x | **1.0x** |
| model size | Llama 3.1 8B | fp16 | L40S | 1.3x | 1.3x | 1.3x | 1.5x | **1.8x** |

**Hardware accounts for essentially all of it.** Holding the model and dtype fixed and moving from CPU to L40S takes the 512-token speedup from 12.8x to 1.1x.

**Dtype accounts for none of it.** fp32 and fp16 on the same card agree to two decimals, which says these cells are bound by kernel launch latency rather than arithmetic.

**Model size gives a little back.** Llama 3.1 8B reaches 1.8x where GPT-2 on the same card manages 1.0x, because a model 65 times larger does enough real work per forward pass to notice the work it is repeating.

The uncached column shows why. On CPU it climbs 2.3x to 3.4x per doubling, visibly bending toward quadratic. On the L40S it reads 2.0x, 2.0x, 2.05x, 2.13x, which is close to linear. The redundant computation is still happening; the GPU absorbs it inside launch overhead instead of paying for it in wall time.

**What this does not say.** It is not an argument against caching on GPUs. The longest sequence here is 576 tokens, and the quadratic term is what makes a 128K context impossible on any hardware. Single-stream timing also cannot show what recompute does to a batched serving stack, which is where the cost actually lands in production. The narrow claim is the one to keep: **at short sequence lengths on a fast GPU, the wasted compute hides inside launch latency, so a speedup measured on a CPU does not transfer.**

Raw logs and the exact environment are in [`results/`](results/).

### Cache size, measured on Llama 3.1 8B

`kv_cache_measured.py` needed no changes; it already took `--model`, `--device` and `--dtype`. Llama 3.1 8B Instruct, fp16, 64-token prompt, on the L40S:

| seq_len | Measured | Formula | Ratio |
| --- | --- | --- | --- |
| 64 | 8.00 MB | 8.00 MB | 1.000 |
| 128 | 16.00 MB | 16.00 MB | 1.000 |
| 320 | 40.00 MB | 40.00 MB | 1.000 |
| 576 | 72.00 MB | 72.00 MB | 1.000 |

Exact at every checkpoint, flat at 128 KiB per token, which is `2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes`. The formula holds on a grouped-query 8B model in fp16 on CUDA, not only on GPT-2 in fp32 on CPU.

### Reproducing

```bash
cd kv-caching
PY=../model-quantization/.venv-cuda/bin/python    # the CUDA env from model-quantization

$PY kv_cache_measured.py --model meta-llama/Llama-3.1-8B-Instruct \
    --device cuda --dtype float16 --prompt-tokens 64 --checkpoints 0,64,256,512

$PY kv_cache_timing.py --model meta-llama/Llama-3.1-8B-Instruct \
    --device cuda --dtype float16 --prompt-tokens 64 --lengths 32,64,128,256,512 --repeats 3

$PY kv_cache_timing.py --model gpt2 --device cuda --dtype float32 \
    --prompt-tokens 64 --lengths 32,64,128,256,512 --repeats 3
```

`kv_cache_timing.py` keeps GPT-2 on CPU as its defaults, so the table further up reproduces with no flags. It calls `torch.cuda.synchronize()` around every timed region; without that on CUDA you time kernel launches rather than generation.
