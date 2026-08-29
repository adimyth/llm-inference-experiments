# GPTQ: fix the error as you go

Layer-wise quantization with Hessian-based error correction.

> **Status: measured.** Run on a rented NVIDIA L40S (48GB), driver 595.91.07,
> CUDA 13.2, with the exact package set pinned in `../requirements-cuda.lock`.
>
> | | |
> | --- | --- |
> | Perplexity | **7.959** (+8.1% vs the fp16 control's 7.3649) |
> | MMLU, 50q / 200q | 76.0% / 75.0% |
> | Size on disk | 5.33 GB |
> | Quantization time | 13m 10s |
>
> The applied config is captured in `../results/checkpoint_configs/gptq-q4/`, read
> back out of the checkpoint itself rather than from the docs.

## How it works

GPTQ quantizes a weight matrix one **column** at a time. After rounding each column, it goes back and adjusts the columns it has **not yet quantized** to compensate for the error it just introduced, so the layer's output stays as close as it can to what the unquantized layer would have produced.

The size and direction of that adjustment come from a Hessian estimated over the same calibration text, which is what tells GPTQ how sensitive the layer's output is to each weight. Correction gets spent where it changes the answer.

The contrast with `../awq/` is the reason both are here:

| | Uses calibration data to | Timing |
| --- | --- | --- |
| AWQ | find which channels to protect | decides **before** rounding |
| GPTQ | correct the error rounding caused | repairs **after** each step |

Same bit width, same group size, same calibration set. Different strategy.

## Configuration

| | |
| --- | --- |
| Scheme | `W4A16` (4-bit weights, fp16 activations) |
| Group size | **128** |
| Zero-point | **no**, `symmetric: true` |
| Block size | 128 |
| Dampening | `dampening_frac: 0.01` |
| Activation order | `static` |
| Observer | `memoryless_minmax` |
| Calibration | `HuggingFaceH4/ultrachat_200k`, 256 samples, 2048 tokens, seed 42 |
| Ignored | `lm_head` |

Read back out of the produced checkpoint, not from the docs. The raw files are in
[`../results/checkpoint_configs/gptq-q4/`](../results/checkpoint_configs/gptq-q4/).

**The zero-point is the difference that matters when comparing against AWQ.** AWQ's default
scheme is asymmetric and stores an offset per group; this one does not. The two methods
therefore differ by scheme as well as by algorithm, which is why the essay reads the gap as
"AWQ as shipped beat GPTQ as shipped" rather than as a verdict on the algorithms.

Group size 128 is the default every shipped GPTQ checkpoint uses, not the 64 that `../mlx/` and `../hqq/` use.

`../awq/quantize.py` builds the identical calibration set with the same seed, so the two methods differ by algorithm and not by data.

Measured, GPTQ quantized *faster* than AWQ on the same box, 13m 10s against 14m 22s, despite the per-column Hessian solve sounding heavier. It also ran at about 18% GPU utilisation against AWQ's 100%, because the solve is serial work that cannot fill the device.

## Tooling

`llmcompressor`. AutoGPTQ was archived in April 2025 and transformers removed its GPTQ backend, so `llmcompressor` (the vLLM project's library) is the maintained path. It is also what `../awq/` uses, which is what lets the two share a calibration set and an output format. Output is a `compressed-tensors` checkpoint that plain `AutoModelForCausalLM` loads.

## Why this needs a rented GPU

There is no Metal or MPS path for GPTQ. Perplexity and MMLU are hardware-independent and join the main tables directly; **tokens/sec is not**, and can only be read against an fp16 control run on the same GPU.

## Running it

```bash
cd .. && bash setup_cuda.sh && source .venv-cuda/bin/activate && cd gptq
export HF_TOKEN=hf_...

python quantize.py   --hf-dir meta-llama/Llama-3.1-8B-Instruct
python perplexity.py --model ../models/llama-3.1-8b-instruct-gptq-q4 --label gptq-q4 --device cuda
python mmlu.py       --model ../models/llama-3.1-8b-instruct-gptq-q4 --label gptq-q4 --device cuda
python throughput.py --model ../models/llama-3.1-8b-instruct-gptq-q4 --label gptq-q4 --device cuda
```

The benchmark scripts default to `--dtype auto`, so the checkpoint's own quantization config decides its dtype. Do not force `float16` on them.

Run the fp16 control on the same GPU first (see `../awq/README.md`); one control covers both methods, and its perplexity should land near 7.365.

Smoke-test before the real run, same as AWQ:

```bash
python quantize.py --hf-dir meta-llama/Llama-3.1-8B-Instruct --num-samples 8 --max-seq-len 512 --out-dir /tmp/gptq-smoke --label gptq-smoke
```
