# Model quantization

Seven post-training quantization methods applied to the same model, measured the same way, against the same unquantized baseline.

Essay: [LLM Inference: Post-Training Quantization](https://adimyth.in/essays/llm-inference-quantization) (not yet published)

**Model**: `meta-llama/Llama-3.1-8B-Instruct`. Not the original 3.0 release, which is a separately gated repo on the same account; 3.1 was already approved.

**What this measures.** Each method is characterised on its own terms: what it does, what it costs against its own baseline. Cross-method rankings appear only where the setup controls the confound, because group size, library and measuring tool otherwise vary together. See the band caveat under Results.

**Hardware, and why there are two sets.** Five methods run on a MacBook Pro, Apple M4 Pro, 48GB, with llama.cpp built via arm64 Homebrew with Metal. AWQ and GPTQ have no Metal or MPS path at all and need an NVIDIA GPU. Size, perplexity and MMLU are hardware-independent and compare directly across the two. **Tokens/sec does not.** A CUDA checkpoint's speed is only meaningful against an fp16 control run on that same GPU, which is why `plot.py` keeps `awq-q4` and `gptq-q4` out of the generation-speed chart entirely.

---

## Layout

Infrastructure lives at the root. Each method gets a folder that is complete on its own: how to quantize, how to measure, what the numbers came out to, and what the method does.

```
fetch_wikitext.py     wikitext-2-raw test split -> data/
perplexity_core.py    the perplexity measurement convention, defined ONCE
mmlu_tasks.py         the fixed 50-question MMLU subset, defined ONCE
results_io.py         merge one metric into results/results.json by label
torch_device.py       --device resolution and cuda/mps synchronisation
plot.py               charts from results.json, light/dark pairs
setup_cuda.sh         any NVIDIA box -> working env for AWQ and GPTQ

fp16/       baseline, plus the cross-check between the two perplexity tools
rtn/        round-to-nearest, llama.cpp Q4_0
kquants/    llama.cpp Q4_K_M, Q5_K_M, Q8_0
mlx/        Apple MLX, 4-bit
hqq/        Half-Quadratic Quantization, 4-bit
awq/        Activation-aware Weight Quantization, 4-bit  (CUDA)
gptq/       GPTQ, 4-bit                                   (CUDA)
```

Every method folder has the same four files, plus a README explaining the method: `quantize.py`, `perplexity.py`, `mmlu.py`, `throughput.py`. (`fp16/` has `convert.py` instead of `quantize.py`, and a second `perplexity_torch.py`.)

**Some scripts are duplicated across folders and that is deliberate** - it keeps each method readable end to end without jumping around. Two things are *not* duplicated, because divergence in either invalidates every comparison in the project, with no error to warn you:

- **`perplexity_core.py`**, the windowing and scoring convention.
- **`mmlu_tasks.py`**, the fixed question subset.

The first one is not a hypothetical. See below.

---

## The methods

| Folder | Method | Calibration? | Bits | Group | Library |
| --- | --- | --- | --- | --- | --- |
| `rtn/` | Round-to-nearest | no | 4 | 32 | llama.cpp |
| `kquants/` | k-quants, mixed precision by tensor | no | 4/5/8 | 32 in super-blocks of 256 | llama.cpp |
| `mlx/` | MLX default scheme | no | 4 | 64 | `mlx-lm` |
| `hqq/` | Half-Quadratic Quantization | no | 4 | 64 | `hqq` |
| `awq/` | Activation-aware Weight Quantization | **yes** | 4 | 128 | `llmcompressor` |
| `gptq/` | Hessian-corrected, column by column | **yes** | 4 | 128 | `llmcompressor` |

The split that organises the whole set: the first four decide how to round by looking only at the weights. The last two push real text through the model first and use what they see. AWQ decides what to protect *before* rounding; GPTQ rounds greedily and *repairs* after each step.

All of this is **post-training** quantization. No gradients, no backward pass, no retraining. Quantization-aware training, which puts fake-quant ops into the forward pass so the model's weights adapt around the rounding error, needs the full training pipeline and is out of scope.

## Libraries, and why these ones

- **llama.cpp** for GGUF (`rtn/`, `kquants/`, and the fp16 baseline). Installed via arm64 Homebrew; Metal confirmed active. `convert_hf_to_gguf.py` is not in the bottle, so the source is shallow-cloned separately and gets its own venv - it pins a CPU `torch` that conflicts with the shared MPS one.
- **`mlx-lm`** for `mlx/`. Apple's array framework, the native option on M-series hardware.
- **`hqq`** for `hqq/`, using `AutoHQQHFModel` rather than transformers' generic `HqqConfig` path, which cannot reload what it saves on transformers 5.15.1.
- **`llmcompressor`** for both `awq/` and `gptq/`. The predecessors are gone: AutoAWQ is deprecated (its AWQ work was adopted by the vLLM project as `llmcompressor`, with help from AutoAWQ's maintainer) and AutoGPTQ was archived in April 2025, with transformers removing its GPTQ backend. Using one library for both also means AWQ and GPTQ share a calibration set, sample count, sequence length and output format, so comparing them compares algorithms rather than codebases.
- **`lm-evaluation-harness`** (`lm_eval`) for MMLU across every format.
- **`llama-cpp-python[server]`** for the GGUF MMLU path only. See gotchas.

---

## Setup

### Local (Apple Silicon): fp16, RTN, k-quants, MLX, HQQ

```bash
python fetch_wikitext.py                                   # wikitext-2 test split -> data/
cd fp16 && python convert.py --hf-dir <local HF snapshot dir> \
    --outfile ../models/llama-3.1-8b-instruct-f16.gguf
```

Run scripts from inside their method folder with the repo's shared venv, e.g. `../../.venv/bin/python perplexity.py ...`. Then follow each method folder's README.

### Any NVIDIA GPU: AWQ, GPTQ

Provider-agnostic. It needs a machine with an NVIDIA driver and nothing else, which every cloud GPU image already has (AWS Deep Learning AMI, GCP Deep Learning VM, RunPod, Lambda) as does a local CUDA workstation.

```bash
export HF_TOKEN=hf_...
bash setup_cuda.sh          # uv venv, deps, nvidia-smi + torch smoke test
source .venv-cuda/bin/activate
python fetch_wikitext.py
hf download meta-llama/Llama-3.1-8B-Instruct
```

Requirements: >=24GB VRAM (48GB comfortable, the calibration pass holds the fp16 model plus per-layer Hessians and activations), ~120GB disk.

The first run resolves the latest packages and writes `requirements-cuda.lock`. Commit that file; later runs install from it, so the environment behind the numbers is repeatable.

Run the fp16 control on the GPU **first**, so the AWQ and GPTQ speed numbers have a baseline on the same hardware, then follow `awq/README.md` and `gptq/README.md`.

---

## Results

Llama 3.1 8B Instruct. Wikitext-2 test split in full, 512-token windows, second half of each scored. MMLU on a fixed 50-question subset, identical questions for every checkpoint.

**Measured on the M4 Pro** (llama.cpp for GGUF, MLX and PyTorch/MPS for the rest):

| Metric | fp16 | RTN `Q4_0` | `Q4_K_M` | `Q5_K_M` | `Q8_0` | MLX 4-bit | HQQ 4-bit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Size on disk | 14.97 GB | 4.34 GB | 4.58 GB | 5.34 GB | 7.95 GB | 4.22 GB | 5.61 GB |
| Generation (tg128) | 15.5 t/s | 48.5 t/s | 44.9 t/s | 31.2 t/s | 28.1 t/s | 52.8 t/s | 0.6 t/s\* |
| Prompt processing (pp512) | 315.5 t/s | 377.8 t/s | 359.9 t/s | 329.4 t/s | 362.6 t/s | 359.7 t/s | 141.0 t/s |
| Perplexity | 7.395 † | 7.804 (+5.5%) | 7.622 (+3.1%) | 7.467 (+1.0%) | 7.399 (+0.0%) | 7.949 (+7.9%) | 7.815 (+6.1%) |
| MMLU, 50q | 82.0% | 76.0% | 78.0% | 80.0% | 80.0% | 76.0% | 76.0% |

**Measured on a rented L40S** (no Metal or MPS path exists for either). Perplexity and MMLU are hardware-independent and compare directly against the table above; **tokens/sec does not** and is quoted only against the fp16 control run on the same card:

| Metric | fp16 (CUDA control) | AWQ 4-bit | GPTQ 4-bit |
| --- | --- | --- | --- |
| Size on disk | 14.97 GB | 5.35 GB | 5.33 GB |
| Generation (tg128) | 38.1 t/s | 33.4 t/s (0.88x) | 32.9 t/s (0.86x) |
| Prompt processing (pp512) | 9191.2 t/s | 9172.0 t/s | 9194.5 t/s |
| Perplexity | 7.3649 | 7.792 (+5.8%) | 7.959 (+8.1%) |
| MMLU, 50q | 82.0% | 78.0% | 76.0% |
| MMLU, 200q | 76.5% | 77.0% | 75.0% |
| Quantization time | n/a | 14m 22s | 13m 10s |
| Scheme | n/a | `W4A16_ASYM`, group 128 | `W4A16`, group 128 |

Both quantized checkpoints generate **slower** than fp16 on this GPU. `compressed-tensors` under plain `transformers` unpacks to fp16 and runs an ordinary fp16 matmul, so there is no arithmetic saving and the unpacking is overhead, on hardware whose bandwidth was never the constraint. Same class of result as HQQ's 0.6 t/s: a property of the backend, not the checkpoint.

\* Backend artifact, not the checkpoint. See `hqq/README.md`. Excluded from `throughput_*.png` for the same reason.

† Two implementations: `llama-perplexity` gives 7.3950, the PyTorch loop gives 7.3648, a 0.4% gap. RTN and the k-quants are quoted against the former, MLX and HQQ against the latter, each against the implementation it shares. See `fp16/README.md`.

**The 50-question MMLU numbers do not survive a larger sample.** Rescoring four checkpoints at 200 questions (`mmlu_200` in results.json) reversed the ordering: fp16 82.0% -> 76.5%, MLX 76.0% -> 77.5%, AWQ 78.0% -> 77.0%, GPTQ 76.0% -> 75.0%. Two quantized checkpoints finish above fp16, and MLX, which has the worst perplexity here, has the best 200-question accuracy. At 50 questions the 95% interval is about ±11 points. Treat every MMLU figure as a check that nothing is catastrophically broken, never as a ranking.

**What the table says.** Inside the k-quant family, where one tool quantizes one file and scores it against one baseline, more bits means less error with no exceptions: `Q8_0` +0.0%, `Q5_K_M` +1.0%, `Q4_K_M` +3.1%. The three 4-bit methods then land in a band from +5.5% to +7.9%, at RTN +5.5%, HQQ +6.1%, MLX +7.9%.

**Read that band as a band, not a ranking.** Group size varies across the three (32, 64, 64), the libraries differ, and RTN is quoted against llama.cpp's 7.3950 while MLX and HQQ are quoted against the PyTorch loop's 7.3648. Those vary together, so position inside the band is not attributable to the rounding algorithm. The 0.6 points between RTN and HQQ is smaller than the 0.4% gap between the two fp16 baselines.

`Q4_K_M` against `Q4_0` is a clean comparison, since the same tool quantized the same f16 file and scored both the same way: nearly the same size, 4.58GB against 4.34GB, for almost half the perplexity cost. Same bit budget, spent more carefully. It also answers one more MMLU question, which on 50 questions is noise, not a result.

The one properly controlled cross-method pair is **MLX against HQQ**: same 4 bits, same group size 64, same scoring loop, same baseline, and still 1.8 points apart. That gap is attributable to the method.

The three 4-bit methods all tie or nearly tie on MMLU at 76.0%. On a fixed 50-question subset the 95% interval on any accuracy here is roughly ±11 points, so a two-point gap is one question. Treat differences under about three questions as noise. That the three tie is the expected result, not a signal that the two metrics disagree.

---

## The perplexity bug, and how it was caught

Worth reading before trusting any perplexity number in any project, this one included.

**The symptom.** MLX and HQQ appeared to cost about **+35%** perplexity against fp16, six times what RTN cost, despite similar size and despite tying RTN exactly on MMLU. That got written up as a real finding, with an explanation about perplexity and task accuracy measuring different things.

**The cause.** Perplexity is not fully specified by "perplexity on wikitext-2". You also have to say which tokens inside each window get scored. `llama-perplexity` sets `const int first = n_ctx/2` ([`tools/perplexity/perplexity.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp)) and scores only the **second half** of each 512-token window, so every scored token has at least 256 tokens of context behind it, following [the HF perplexity docs](https://huggingface.co/docs/transformers/perplexity). The PyTorch and MLX loops here scored the **whole** window, including the first tokens of every chunk, which have almost no context and are close to unpredictable for any model.

That inflates the number. Same fp16 weights, same file, same 564 windows:

| Scoring | fp16 perplexity |
| --- | --- |
| Second half of each window (llama.cpp's rule) | 7.3648 |
| Whole window | 9.4599 |

Neither is wrong. They are not the same measurement. MLX's 10.12 and HQQ's 9.96 were real numbers being compared against a baseline on a different scale.

**The fix.** One definition of the convention, in `perplexity_core.py`, that every perplexity script calls. Engines still supply their own model loading and loss (PyTorch, MLX and HQQ are different frameworks), but the chunking, the second-half rule and the accumulation are no longer duplicated anywhere.

**The validation.** Run *both* implementations against the same unquantized fp16 weights and check they agree. They do, to 0.4%, which is consistent with fp16 kernel differences between llama.cpp's Metal kernels and PyTorch's MPS backend. That cross-check is what licenses comparing a GGUF checkpoint's perplexity to an MLX or AWQ one, and it should have existed from the start.

**The lesson.** The bug survived because the three scripts most likely to diverge were the three that duplicated the logic, and because the one metric that would have caught it, MMLU, was explained away instead. A metric that disagrees with another metric is worth suspecting before it is worth narrating.

---

## Other gotchas

**`lm_eval`'s `gguf` model type needs `llama_cpp.server`, not llama.cpp's own `llama-server`.** Scoring a multiple-choice answer needs per-token logprobs across the full echoed prompt plus continuation. `llama-server` (the C++ binary) only ever returns logprobs for newly *generated* tokens, never echoed text - confirmed against its README and a live request. It is a different capability, not a version mismatch. `llama_cpp.server` (from `llama-cpp-python[server]`, same Metal backend underneath) computes them over the full sequence, unmodified.

**`loglikelihood_rolling` is unimplemented in `lm_eval`'s `GGUFLM`**, so its `wikitext` perplexity task can't run against a GGUF checkpoint regardless of server. That's why perplexity goes through `llama-perplexity` directly.

**`lm_eval --limit N` is a positional slice, not a random sample.** Verified against its source. This is what makes "the same 50 questions" true.

**A method folder named after a real package will be imported as one.** The folders `mlx/` and `hqq/` share their names with real PyPI packages, and every script puts the repo root on `sys.path` to import the shared helpers. That makes `importlib.util.find_spec("mlx")` succeed on an empty namespace package, so `transformers` evaluates `is_mlx_available()` as True at import time and caches it. Ordinary forward passes never reach the MLX branch of `is_tensor`, because real torch tensors match earlier. `llmcompressor`'s fx tracing does reach it, because Proxy objects match nothing, and the run dies with `ModuleNotFoundError: No module named 'mlx.core'` on a Linux box with no MLX installed anywhere. Import `transformers` before the root goes on `sys.path`, or remove the root once the helpers are loaded, which is what `awq/quantize.py` and `gptq/quantize.py` now do.

**HQQ defaults to `device='cuda'` everywhere.** On a Mac it either crashes or reloads onto the wrong device with no warning. Pass `device='mps'` explicitly on every call.

**`llama-cpp-python` can build x86_64 on an arm64 Mac without saying so** if `/usr/local/bin/cmake` shadows the arm64 one on PATH, producing a `.dylib` the venv can't load. Force the arm64 `cmake` onto PATH for the build.

---

## Files produced

Everything lands in `models/` and `data/`, both gitignored: too large to commit and fully regenerable.

| File | Produced by | Size |
| --- | --- | --- |
| `llama-3.1-8b-instruct-f16.gguf` | `fp16/convert.py` | 14.97 GB |
| `llama-3.1-8b-instruct-q4_0.gguf` | `rtn/quantize.py` | 4.34 GB |
| `llama-3.1-8b-instruct-q4_k_m.gguf` | `kquants/quantize.py` | 4.58 GB |
| `llama-3.1-8b-instruct-q5_k_m.gguf` | `kquants/quantize.py` | 5.34 GB |
| `llama-3.1-8b-instruct-q8_0.gguf` | `kquants/quantize.py` | 7.95 GB |
| `llama-3.1-8b-instruct-mlx-q4/` | `mlx/quantize.py` | 4.22 GB |
| `llama-3.1-8b-instruct-hqq-q4/` | `hqq/quantize.py` | 5.61 GB |
| `llama-3.1-8b-instruct-awq-q4/` | `awq/quantize.py` | 5.35 GB (built on the rented GPU) |
| `llama-3.1-8b-instruct-gptq-q4/` | `gptq/quantize.py` | 5.33 GB (built on the rented GPU) |

`data/wikitext-2-raw-test.txt` comes from `fetch_wikitext.py` and is shared by every perplexity script regardless of format.

`results/results.json` holds every metric for every checkpoint, keyed by label, merged in by `results_io.py` as each script runs. `results/mmlu_raw/<label>/` holds the raw `lm_eval` output behind each MMLU number.
