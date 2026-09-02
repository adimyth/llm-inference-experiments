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

Moving from CPU to L40S with the model and dtype held fixed takes the 512-token speedup from 12.8x to 1.1x. Dtype changes nothing: fp32 and fp16 on the same card agree to two decimals. Llama 3.1 8B reaches 1.8x where GPT-2 on the same card manages 1.0x.

**Read none of that as "caching does not help on GPUs."** The first version of this section said hardware accounted for essentially all of the collapse. That was wrong, and `forward_cost.py` is what showed it.

### Why the speedup collapsed: we measured the bottom of the curve

Both paths run the same number of forward passes. The cached one passes a single token; the uncached one passes the whole sequence so far. So **the speedup is exactly the ratio between the cost of a full forward and the cost of a one-token forward**, and that ratio is not a property of the hardware. It depends on how long the sequence is.

Llama 3.1 8B, fp16, one forward pass on the L40S:

| seq_len | ms/forward | vs len-1 | tokens/ms |
| --- | --- | --- | --- |
| 1 | 25.06 | 1.00x | 0.0 |
| 64 | 29.41 | 1.17x | 2.2 |
| 320 | 41.96 | 1.67x | 7.6 |
| 512 | 49.11 | 1.96x | 10.4 |
| 1024 | 90.32 | 3.60x | 11.3 |
| 2048 | 177.16 | 7.07x | 11.6 |
| 4096 | 378.46 | 15.10x | 10.8 |
| 8192 | 852.67 | 34.03x | 9.6 |

The 1.67x at 320 tokens, which is the average sequence length in the 512-token timing cell, is the 1.8x measured there.

**That `vs len-1` column is a ceiling, not the speedup.** The cached path pays `cost(1)` every step, but the uncached path pays `cost(L)` averaged over every length from the prompt to the end, not the cost at the final length.

The timing run was extended to 1024 and 2048 generated tokens to measure the transition region directly rather than projecting into it:

| Generated tokens | No cache | Cache | Speedup |
| --- | --- | --- | --- |
| 32 | 0.98s | 0.77s | 1.3x |
| 64 | 1.98s | 1.54s | 1.3x |
| 128 | 4.15s | 3.08s | 1.3x |
| 256 | 9.06s | 6.17s | 1.5x |
| 512 | 22.09s | 12.34s | 1.8x |
| 1024 | 64.38s | 24.79s | 2.6x |
| 2048 | 222.49s | 49.88s | 4.5x |

Every row is measured, median of 3, Llama 3.1 8B fp16 on the L40S at batch 1 with a 64-token prompt. The cached column is close to linear in output length, roughly doubling per row, while the uncached column grows faster and faster: 2.0x, 2.1x, 2.2x, 2.4x, 2.9x, 3.5x per doubling. That divergence is the quadratic term becoming visible, and it is why the speedup column climbs.

Generation beyond 2048 tokens was not measured.

A single-token forward costs 25.06 ms and moves 0.04 tokens per millisecond. The card is idle; that time is fixed cost, most of it `transformers` Python dispatch rather than CUDA launch overhead, since 150-odd kernels cannot account for GPT-2's 6.4 ms. Until the sequence is long enough for arithmetic to exceed that fixed cost, both paths pay roughly the same and the ratio sits near 1.

**The defect is the sequence lengths, inherited from the CPU experiment without asking whether they still suited a GPU.** They do not. A CPU has no comparable fixed-cost cushion, so at 512 tokens it was already in the compute-bound region while the GPU was nowhere near it. The two tables were never measured at comparable points on the same curve.

**What the L40S tables do support**, stated narrowly:

- The speedup from caching grows with sequence length, and on a GPU it stays near 1x until roughly 1K tokens.
- A speedup measured on a CPU does not transfer to a GPU at the same sequence length.
- Absolute per-forward cost at batch 1 is dominated by framework overhead. A runtime using CUDA graphs would show a larger cache benefit at these same lengths, so these numbers bound `transformers`, not the hardware.

**What they do not support**: any claim that caching is unnecessary on GPUs. Everything here is batch 1 at up to 576 generated tokens. Production runs longer contexts and batches, and both push hard toward the compute-bound regime where the cache is worth the 7x to 33x above.

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

---

## Prompt caching, against a hosted API

The three scripts above run offline and cost nothing. The three below call OpenAI, so they need `OPENAI_API_KEY`, a network, and roughly a dollar for a full pass. They measure the commercial version of prefix caching rather than the mechanism, which means they measure policy: the numbers are prices, and prices change.

Everything here was measured on 2026-09-02 with `openai` 3.7.0. The fixture is [`fixtures/support_handbook.txt`](fixtures/support_handbook.txt), 5,144 tokens of invented support policy, which stands in for the fixed document a support agent answers against.

```bash
cd kv-caching
export OPENAI_API_KEY=sk-...
../.venv/bin/python prompt_cache_placement.py
../.venv/bin/python prompt_cache_floor.py
../.venv/bin/python prompt_cache_routing.py
```

All three take `--models`. The first two take `--dry-run`, which builds the prompts and prints their token counts without calling anything.

### prompt_cache_placement.py

The handbook is identical on every request, the question differs on every request, nothing else varies. Two pieces and two fields gives four arrangements:

```
A   user:   [question][handbook]
B   user:   [handbook][question]
C   system: [handbook]            user: [question]
D   system: [question]            user: [handbook]
```

Eight requests per condition, same eight questions in the same order, same token count. Requests that read from cache, out of the seven that could:

| Model | A | B | C | D | Cache write rate |
| --- | --- | --- | --- | --- | --- |
| gpt-5.6-terra | 0/7 | 0/7 | 7/7 | 0/7 | 1.25x |
| gpt-4o | 0/7 | 7/7 | 7/7 | 0/7 | no charge |
| gpt-5.4 | 0/7 | 7/7 | 6/7 | 0/7 | no charge |

**A and D never cache.** Both lead with the question. D is the control: putting the question in the system field saves nothing, so position matters and the field itself does not.

**B is where the models disagree.** Leading with the handbook inside the user message works on gpt-4o and gpt-5.4 and does nothing on gpt-5.6. A direct check confirmed it: the same handbook caches on gpt-5.6 from `instructions` and not from the head of the user message.

**C works on all three.** Give the fixed document its own field.

Billed input cost as a multiple of the same eight requests with no caching:

| Model | A | B | C | D |
| --- | --- | --- | --- | --- |
| gpt-5.6-terra | 1.25x | 1.25x | 0.25x | 1.25x |
| gpt-4o | 1.00x | 0.25x | 0.25x | 1.00x |
| gpt-5.4 | 1.00x | 0.27x | 0.37x | 1.00x |

On gpt-5.6 the three failing columns cost more than not caching, because each request writes an entry at 1.25x and none is read back. gpt-5.4's 0.37x in column C is one stray miss out of seven rather than a real difference from column B; its hit counts are the more stable reading. The ratio also depends on how many requests share the prefix, since the single write amortises over the reads that follow: at five requests column C measured 0.33x, at eight it measures 0.25x, and the floor is the 0.1x read rate.

### prompt_cache_floor.py

Slices the handbook to exact token counts and sends each three times. The minimums are published (1,024 on gpt-5.6 and later, 2,048 on earlier models, with a note that some earlier models cache shorter prefixes), so this checks the published number against the model actually being called.

| Model | Largest prompt that cached nothing | Smallest that cached | Published minimum | Reported granularity |
| --- | --- | --- | --- | --- |
| gpt-5.6-terra | 939 | 1,065 | 1,024 | exact |
| gpt-4o | 1,064 | 1,130 | 2,048 | rounded down to a multiple of 128 |
| gpt-5.4 | 1,835 | 1,961 | 2,048 | rounded down to a multiple of 128 |

gpt-4o caches from roughly half its published figure while gpt-5.4 sits where the documentation says, and both fall under the same published number. The hedge in the docs is load-bearing, and you cannot plan against a hedge.

Below the threshold nothing caches, nothing errors, and no field in the response says so.

Granularity is visible in the read column: gpt-5.6 reads back the exact prefix it wrote, while the older two round down, so gpt-5.4 turns a 5,040-token prefix into 4,864, which is 38 x 128.

### prompt_cache_routing.py

The same prompt eight times, once with no `prompt_cache_key` and once with one. Request 1 has nothing to read, so the denominator is 7.

| Model | Without the key | With it |
| --- | --- | --- |
| gpt-4o | 3/7 | 7/7 |
| gpt-5.4 | 7/7 | 7/7 |

The parameter is a routing hint rather than a guarantee, and the documentation is explicit that it does not pin a request to a machine. gpt-5.4 was already sticky in this run and the key changed nothing. Treat this as a demonstration that the lever exists, not as a number to quote: it depends on how busy the fleet is, and it will vary between runs.

### Two things that quietly break these measurements

Both produce plausible-looking numbers rather than errors.

**A run reads the previous run's cache.** Every run prepends a random nonce to the front of the prefix so it gets its own namespace. Without it, request 1 of a fresh run reads an entry the last run wrote and reports a hit that this run did not earn.

**Short slices warm long ones.** A 512-token slice is a byte-for-byte prefix of the 1,024-token slice, so sweeping upward warms each size from the one before it and every threshold after the first is wrong. Each slice gets its own nonce.

There is also a retry path. A response occasionally returns with all token counts at zero, which is not a measurement, so it is discarded and re-sent. The discarded attempt still warms the cache, so a retry anywhere in a placement condition throws that whole condition away and restarts it under a fresh nonce. Without that, the retried request scores a hit in condition A, where by construction nothing should ever hit.

Raw output is in [`results/prompt-cache-openai/`](results/prompt-cache-openai/).
