# Quantization

Essay: [LLM Inference: Post-Training Quantization](https://adimyth.in/essays/llm-inference-quantization) (not yet published)

Run from this folder with `../.venv/bin/python <script>`. See the [root README](../README.md) for setup and method notes.

Model: `meta-llama/Llama-3.1-8B-Instruct` (not the original 3.0 release - separately gated on the same account, and 3.1 was already approved, see the plan notes for why). Hardware: MacBook Pro, Apple M4 Pro, 48GB, macOS, llama.cpp built via arm64 Homebrew with Metal.

## Setup

```bash
../.venv/bin/python fetch_wikitext.py   # dumps wikitext-2-raw test split to data/
../.venv/bin/python quant_convert.py --hf-dir <local HF snapshot dir> --outfile models/llama-3.1-8b-instruct-f16.gguf
```

`quant_convert.py` needs its own conversion venv, see its module docstring - it can't share this repo's shared venv (conflicting torch pin).

## Files produced

Everything lives in `models/` and `data/`, gitignored (too large to commit,
and fully regenerable by rerunning the scripts below) - this is the manifest
of what each produces, for the essay's "running this yourself" section and
so the file-size column in the results tables traces back to an actual file
rather than a number typed in by hand.

| File | Produced by | Size |
| --- | --- | --- |
| `llama-3.1-8b-instruct-f16.gguf` | `quant_convert.py` | 14.97 GB |
| `llama-3.1-8b-instruct-q4_0.gguf` | `quant_rtn.py` | 4.34 GB |
| `llama-3.1-8b-instruct-q4_k_m.gguf` | `quant_kquants.py` | 4.58 GB |
| `llama-3.1-8b-instruct-q5_k_m.gguf` | `quant_kquants.py` | 5.34 GB |
| `llama-3.1-8b-instruct-q8_0.gguf` | `quant_kquants.py` | 7.95 GB |
| `llama-3.1-8b-instruct-mlx-q4/` (directory: `model.safetensors` + config/tokenizer) | `quant_mlx.py` | 4.22 GB |

`data/wikitext-2-raw-test.txt` (the wikitext-2 test split, plain text) is produced by `fetch_wikitext.py` and shared by every perplexity script regardless of format.

`results/results.json` holds every metric for every checkpoint, keyed by label, merged in by `results_io.py` as each benchmark script runs. `results/mmlu_raw/<label>/` holds the raw `lm_eval` output (and, for the GGUF path, `server.log`) behind each MMLU number.

## Scripts

| Script | What it does |
| --- | --- |
| `quant_convert.py` | HF checkpoint -> unquantized f16 GGUF. Every GGUF method quantizes from this one file. |
| `quant_rtn.py` | f16 GGUF -> `Q4_0` (RTN: round-to-nearest, no calibration). |
| `quant_kquants.py` | f16 GGUF -> `Q4_K_M`, `Q5_K_M`, `Q8_0` (mixed-precision by tensor, no importance matrix). |
| `quant_perplexity.py` | Perplexity on wikitext-2 via `llama-perplexity`. |
| `quant_throughput.py` | Tokens/sec (prompt processing and generation) via `llama-bench`. |
| `quant_mmlu.py` | Accuracy on a fixed 50-question MMLU subset via `lm_eval`, against a `llama_cpp.server` instance this script launches itself. |
| `quant_mlx.py` | HF checkpoint -> quantized MLX, in one pass (no separate f16 intermediate - see note below on why this converts so much faster than the GGUF path). |
| `quant_mlx_throughput.py` | Tokens/sec for an MLX checkpoint via `mlx_lm.stream_generate`. |
| `quant_mlx_perplexity.py` | Perplexity on the same wikitext-2 file/windowing as `quant_perplexity.py`, reimplemented against `mlx_lm` directly rather than using `mlx_lm`'s own `perplexity` subcommand (which samples a different dataset split and isn't comparable - see below). |
| `quant_mlx_mmlu.py` | Same fixed 50-question subset as `quant_mmlu.py` (imports its `TASKS`/`LIMIT_PER_TASK` directly), scored via `lm_eval.simple_evaluate` called in-process against `mlx_lm`'s `MLXLM` class - no HTTP server needed for this format. |
| `quant_plot.py` | Charts from `results.json`: size, perplexity, MMLU accuracy, and generation speed, every checkpoint against fp16. Light/dark PNG pairs in `results/`. Reads checkpoints generically by label, so a panel just skips any checkpoint that doesn't have that metric yet rather than erroring - safe to rerun as each method's benchmarks land. |
| `results_io.py` | Shared results/results.json merge, keyed by checkpoint label. |

**Why `llama_cpp.server` and not llama.cpp's own `llama-server`** for the MMLU eval: `lm_eval`'s `gguf` model type needs per-token logprobs across the full echoed prompt+continuation to score multiple-choice answers. `llama-server` (the C++ binary) only ever returns logprobs for newly *generated* tokens, never echoed text - confirmed against its own README and a live request, not a version mismatch, a structurally different capability. `llama_cpp.server` (from `llama-cpp-python[server]`, same Metal backend underneath) computes logprobs itself over the full sequence and matches the shape `lm_eval` expects, unmodified.

**MMLU subset**: 10 subtasks (3 STEM, 3 humanities, 2 social sciences, 2 other - MMLU's own category grouping), 5 questions each, 50 total. `lm_eval --limit N` is a plain positional slice of each task's dataset (verified against its source - no shuffling anywhere in the pipeline), so this is the *same* 50 questions for every checkpoint this is run against, not a random sample that happens to differ per run. Cut down from an original 500-question/20-task design after measuring the real per-request cost (each question needs 4 requests, one per answer choice) made that take multiple hours per checkpoint.

## Why MLX conversion takes 8 seconds against GGUF's ~46

GGUF quantization is two sequential steps, each writing a full file to disk:
`convert_hf_to_gguf.py` writes an unquantized f16 GGUF (~15GB, ~38s), then
`llama-quantize` reads that back and writes the quantized file (~9s).
`mlx_lm.convert` loads the HF safetensors, quantizes in memory, and writes
only the final ~4GB quantized result directly - one pass, one file, roughly
a third of the total I/O.

## Why MLX's own `perplexity` subcommand isn't used here

`mlx_lm perplexity` samples random fixed-length chunks from a HF dataset's
*train* split. `quant_perplexity.py` (the GGUF path, via `llama-perplexity`)
processes wikitext-2's *test* split sequentially in full. Different data,
different sampling - the two numbers wouldn't mean the same thing, so
`quant_mlx_perplexity.py` reimplements `llama-perplexity`'s approach against
`mlx_lm` directly instead: same file, same 512-token windows, same
teacher-forced next-token accounting, so perplexity is comparable across
every checkpoint in this project, not just within one format.

## Results: fp16 baseline vs RTN, k-quants, and MLX

| Metric | fp16 | RTN `Q4_0` | `Q4_K_M` | `Q5_K_M` | `Q8_0` | MLX 4-bit |
| --- | --- | --- | --- | --- | --- | --- |
| Size on disk | 14.97 GB | 4.34 GB | 4.58 GB | 5.34 GB | 7.95 GB | 4.22 GB |
| Generation speed (tg128) | 15.5 t/s | 48.5 t/s | 44.9 t/s | 31.2 t/s | 28.1 t/s | 52.8 t/s |
| Prompt processing (pp512) | 315.5 t/s | 377.8 t/s | 359.9 t/s | 329.4 t/s | 362.6 t/s | 359.7 t/s |
| Perplexity (wikitext-2) | 7.395 | 7.804 (+5.5%) | 7.622 (+3.1%) | 7.467 (+1.0%) | 7.399 (+0.05%) | 10.123 (+36.9%) |
| MMLU (50 q, stratified) | 82.0% (41/50) | 76.0% (38/50) | 78.0% (39/50) | 80.0% (40/50) | 80.0% (40/50) | 76.0% (38/50) |

RTN shrinks the checkpoint by 71% and roughly triples decode throughput, for
a modest perplexity cost but a more visible drop in actual multiple-choice
accuracy - the two quality metrics don't agree on how much RTN costs, which
is worth keeping in mind before trusting perplexity alone.

The k-quants land between RTN and fp16 on every metric, and in the order
you'd expect: more bits, closer to fp16. `Q4_K_M` costs almost the same size
as RTN (4.58GB vs 4.34GB) but buys back nearly half the perplexity gap and a
full extra MMLU question, for the same bit budget spent more carefully.
`Q8_0` lands at 7.399 perplexity against fp16's 7.395, close enough to call
unquantized - 8-bit barely costs anything, which is why it's the format
people actually ship when size isn't the binding constraint. `Q5_K_M` and
`Q8_0` tie on MMLU (80.0%, 40/50) despite `Q8_0` being unambiguously closer
on perplexity - a reminder that 50 questions is a coarse enough sample that
ties like this are expected, not a sign the two are equivalent.

MLX is similar in size and slightly faster than RTN, but its perplexity cost
is far larger (+36.9% vs +5.5%) despite both being called "4-bit." The likely
reason: they're not the same quantization granularity. GGUF's `Q4_0` uses a
block size of 32 weights per scale factor; MLX's default (`--q-group-size
64`, used here) groups twice as many weights under one scale, which is
coarser and loses more precision per group. **"4-bit" alone doesn't specify
a quantization scheme** - group size matters and isn't always reported.

Despite that gap, MLX's MMLU accuracy is *identical* to RTN's (76.0%, same
38/50 questions correct) - perplexity and task accuracy don't move together
here. A plausible read: coarser grouping degrades token-level calibration
broadly (which perplexity measures directly), but 4-way multiple-choice
accuracy is a coarser, more forgiving signal that a probability-ranking task
can stay robust to even when the model's per-token confidence has drifted.
Worth treating as a real, reproducible finding rather than noise: it did not
change between the two identical scoring runs.

Generation speed benefits far more than prompt processing across both
methods: decode is memory-bandwidth bound (a smaller checkpoint moves less
data per token), while prompt processing is compute-bound and less sensitive
to weight precision. Perplexity and MMLU wall time are both dominated by
prompt-processing-style computation, not generation, so don't expect
quantized checkpoints to evaluate dramatically faster than fp16 despite the
large decode speedup - MLX's in-process MMLU (40s) is the exception, and
that gap is about eval architecture (no HTTP server round-trip) rather than
about the checkpoint itself.
