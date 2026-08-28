# GPTQ: fix the error as you go

Layer-wise quantization with Hessian-based error correction.

> **Status: not yet run.** The scripts here are complete and the environment
> is reproducible via `../setup_cuda.sh`, but no GPTQ numbers are in
> `../results/results.json` yet. This needs an NVIDIA GPU.

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
| Calibration | `HuggingFaceH4/ultrachat_200k`, 256 samples, 2048 tokens, seed 42 |
| Ignored | `lm_head` |

Group size 128 is the default every shipped GPTQ checkpoint uses, not the 64 that `../mlx/` and `../hqq/` use.

`../awq/quantize.py` builds the identical calibration set with the same seed, so the two methods differ by algorithm and not by data.

Expect GPTQ to take longer than AWQ. The Hessian solve is heavier than AWQ's activation-magnitude pass.

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
