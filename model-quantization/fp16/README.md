# fp16 baseline

Not a quantization method. This is the unquantized model every other number in this project is measured against, and the place where the two perplexity implementations get cross-checked.

## What's here

`convert.py` reads the HF checkpoint and writes an **unquantized f16 GGUF**. No rounding, just a format change. For Llama 3.1 8B that's a 14.97 GB file and takes about 40 seconds. Both `../rtn/` and `../kquants/` quantize from this one file, so it is built once and reused.

## Two perplexity scripts, on purpose

No single tool reads every format in this project, so perplexity is measured two ways:

| Script | Engine | Used for |
| --- | --- | --- |
| `perplexity.py` | `llama-perplexity` (llama.cpp) | fp16, RTN, k-quants (GGUF) |
| `perplexity_torch.py` | PyTorch + transformers | fp16, AWQ, GPTQ, and siblings for MLX and HQQ |

Running **both** against the same fp16 weights is what proves a GGUF checkpoint's perplexity is comparable to an MLX or AWQ one:

| Implementation | Perplexity |
| --- | --- |
| `llama-perplexity` | 7.3950 |
| PyTorch loop | 7.3648 |

A 0.4% gap, consistent with fp16 kernel differences between llama.cpp's Metal kernels and PyTorch's MPS backend. Nothing else is unexplained.

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
