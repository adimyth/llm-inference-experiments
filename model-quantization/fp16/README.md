# fp16 baseline

Not a quantization method. This is the unquantized model every other number in this project is measured against, and the place where the two perplexity implementations get cross-checked.

## What's here

`convert.py` reads the HF checkpoint and writes an **unquantized f16 GGUF**. No rounding, just a format change. For Llama 3.1 8B that's a 14.97 GB file and takes about 40 seconds. Both `../rtn/` and `../kquants/` quantize from this one file, so it is built once and reused.

## Two perplexity scripts, on purpose

No single tool reads every format in this project, so perplexity is measured two ways:

| Script | Engine | Used for |
| --- | --- | --- |
| `perplexity.py` | `llama-perplexity` (llama.cpp) | fp16, RTN, k-quants (GGUF) |
| `perplexity_torch.py` | PyTorch + transformers | fp16, HQQ, AWQ, GPTQ |

`mmlu_torch.py` and `throughput_torch.py` sit alongside them for the same reason. `mmlu.py` and `throughput.py` go through llama.cpp and read only GGUF, which is all the Mac needs; the `_torch` copies run the unquantized model on a CUDA box so AWQ and GPTQ have a speed baseline measured on their own hardware.

There is a third: `../mlx/perplexity.py` runs the same convention through MLX's own arrays and loss, because MLX is a separate framework that neither of the other two can read.

Running **all three** against the same fp16 weights is what proves a GGUF checkpoint's perplexity is comparable to an MLX or AWQ one:

| Implementation | Perplexity | Recorded as |
| --- | --- | --- |
| `llama-perplexity` (C++, Metal) | 7.3950 | `fp16` |
| PyTorch loop, MPS | 7.3648 | `fp16-torch` |
| PyTorch loop, CUDA | 7.3649 | `fp16-cuda` |
| MLX loop | 7.3642 | `fp16-mlx` |

llama.cpp sits 0.4% off the rest, consistent with fp16 kernel differences between its Metal kernels and PyTorch's MPS backend. MLX agrees with PyTorch to 0.007%. The same PyTorch loop on an M4 Pro and on a rented L40S agrees to 0.001%, so the number belongs to the weights and not the machine. Nothing else is unexplained.

The MLX check came last, and until it ran nothing confirmed the MLX loop was measuring perplexity correctly at all.

That cross-check is not academic. Before it existed the PyTorch loop scored the whole 512-token window while `llama-perplexity` scored only the second half, and mixing the two conventions made MLX and HQQ look about 35% worse than fp16 when their real cost is 6-8%. See `../perplexity_core.py`, which now holds the convention in one place, and the "The perplexity bug" section in `../README.md`.

## Results

| Metric | Value |
| --- | --- |
| Size on disk | 14.97 GB |
| Generation (tg128) | 15.5 t/s |
| Prompt processing (pp512) | 315.5 t/s |
| Perplexity, wikitext-2 | 7.395 (llama.cpp) / 7.365 (PyTorch) |
| MMLU, 50q subset | 82.0% (41/50) |

## Running it

```bash
python convert.py --hf-dir <local HF snapshot dir> --outfile ../models/llama-3.1-8b-instruct-f16.gguf
python perplexity.py       --model ../models/llama-3.1-8b-instruct-f16.gguf --label fp16
python mmlu.py             --model ../models/llama-3.1-8b-instruct-f16.gguf --label fp16
python throughput.py       --model ../models/llama-3.1-8b-instruct-f16.gguf --label fp16
python perplexity_torch.py --model <local HF snapshot dir> --label fp16-torch --device mps
```

`convert.py` needs its own conversion venv, see its module docstring: it can't share this repo's shared venv because of a conflicting torch pin.
