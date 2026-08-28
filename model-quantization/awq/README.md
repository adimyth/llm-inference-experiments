# AWQ: protect the channels the activations care about

Activation-aware Weight Quantization. The first method in this project that looks at anything other than the weights.

> **Status: not yet run.** The scripts here are complete and the environment
> is reproducible via `../setup_cuda.sh`, but no AWQ numbers are in
> `../results/results.json` yet. This needs an NVIDIA GPU.

## How it works

Every other method here decides how to round by looking only at the weight values. RTN reads the block maximum. k-quants and MLX read the min and max. HQQ solves for the best fit to the weights themselves. None of them ask which weights *matter*.

AWQ does. It pushes calibration text through the model and measures the average activation magnitude on each **input channel** of every linear layer. A small fraction of channels carry far larger activations than the rest, and an error in a weight multiplied by a large activation does more damage downstream than the same error on a quiet channel. AWQ scales those salient channels up before rounding, so they land on a finer part of the 4-bit grid, and scales the corresponding weights down to compensate so the layer computes the same function.

The rounding itself stays ordinary. What changes is where the grid's precision gets spent. **AWQ decides what to protect before it rounds anything**, which is the clean contrast with `../gptq/`, which rounds greedily and repairs afterwards.

## Configuration

| | |
| --- | --- |
| Scheme | `W4A16_ASYM` (4-bit weights, fp16 activations, asymmetric) |
| Group size | **128** |
| Calibration | `HuggingFaceH4/ultrachat_200k`, 256 samples, 2048 tokens, seed 42 |
| Ignored | `lm_head` |

Group size 128 is the default every shipped AWQ checkpoint uses, not the 64 that `../mlx/` and `../hqq/` use. "4-bit" alone doesn't specify a scheme, and group size is the parameter most model cards leave out.

`../gptq/quantize.py` builds the identical calibration set with the same seed, so AWQ and GPTQ differ by algorithm and not by the data they saw.

## Tooling

`llmcompressor`. AutoAWQ, the original implementation, is deprecated; its AWQ support was adopted by the vLLM project as `llmcompressor` with help from AutoAWQ's own maintainer. Output is a `compressed-tensors` checkpoint that plain `AutoModelForCausalLM` loads, which is why `perplexity.py`, `mmlu.py` and `throughput.py` here need no format-specific code.

## Why this needs a rented GPU

There is no Metal or MPS path for AWQ. This is the reason the AWQ and GPTQ numbers come from an NVIDIA box while every other method in this project was measured on an M4 Pro. Consequences for comparability:

- **Perplexity and MMLU are hardware-independent** and join the main tables directly.
- **Tokens/sec is not.** It can only be read against an fp16 control run on the same GPU, never against the M4 Pro numbers.

## Running it

```bash
cd .. && bash setup_cuda.sh && source .venv-cuda/bin/activate && cd awq
export HF_TOKEN=hf_...

python quantize.py   --hf-dir meta-llama/Llama-3.1-8B-Instruct
python perplexity.py --model ../models/llama-3.1-8b-instruct-awq-q4 --label awq-q4 --device cuda
python mmlu.py       --model ../models/llama-3.1-8b-instruct-awq-q4 --label awq-q4 --device cuda
python throughput.py --model ../models/llama-3.1-8b-instruct-awq-q4 --label awq-q4 --device cuda
```

Run the fp16 control on the same GPU first, so the speed numbers have a baseline:

```bash
cd ../fp16
python perplexity_torch.py --model meta-llama/Llama-3.1-8B-Instruct --label fp16-cuda --device cuda
```
