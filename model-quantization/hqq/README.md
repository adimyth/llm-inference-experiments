# HQQ: a better fit per group

Half-Quadratic Quantization. Calibration-free like RTN, but instead of reading a scale off the data's extremes it *solves* for one.

## How it works

`../rtn/` sizes its scale off the largest weight in a block and leaves it there. `../kquants/` and `../mlx/` size scale and offset off the block's min and max, which is better, but still just reads two numbers off the data: whatever the two most extreme weights happen to be, every other weight in the group has to live with the scale they dictate.

HQQ does neither. For each group it solves a small optimization problem, searching for the scale and zero-point that minimize total reconstruction error across **every** weight in the group, not just the two at the edges.

A 16-weight group makes it visible. Two outliers, -1.20 and 0.90, and fourteen ordinary weights clustered around 0.10:

| | Weight | Min/max recon | Min/max error | HQQ recon | HQQ error |
| --- | --- | --- | --- | --- | --- |
| Outlier | -1.20 | -1.20 | 0.000 | -1.16 | +0.040 |
| Cluster | 0.094 | 0.06 | -0.034 | 0.10 | +0.006 |
| Cluster | 0.111 | 0.06 | -0.051 | 0.10 | -0.011 |
| Cluster | 0.117 | 0.06 | -0.057 | 0.10 | -0.017 |
| Cluster | 0.122 | 0.06 | -0.062 | 0.10 | -0.022 |
| Outlier | 0.90 | 0.90 | 0.000 | 0.80 | -0.100 |

Both spend a single code on the whole fourteen-weight cluster. Min/max keeps its scale anchored so the outliers come back exact, landing the cluster's code at 0.06, well below where those weights sit. HQQ's optimizer gives up some accuracy on the two rare outliers so the cluster's code lands at 0.10, right where the fourteen ordinary weights are. Same resolution, better-centred grid. Summed and squared, min/max's total error is 0.023, HQQ's is 0.017, about 25% lower from the same 4 bits.

Confirmed at scale on a real matrix: quantizing `down_proj` in layer 10 at 4-bit, group size 64, HQQ's scale and offset land 35-40% closer to the original weights than RTN's min/max, measured as mean squared error.

Quantizing the full 8B model takes about 3 minutes.

## Results

| Metric | fp16 | RTN `Q4_0` | HQQ 4-bit |
| --- | --- | --- | --- |
| Size on disk | 14.97 GB | 4.34 GB | 5.61 GB |
| Perplexity, wikitext-2 | 7.365* | 7.804 (+5.5%) | 7.815 (+6.1%) |
| MMLU, 50q subset | 82.0% (41/50) | 76.0% (38/50) | 76.0% (38/50) |
| Generation (tg128) | 15.5 t/s | 48.5 t/s | 0.6 t/s† |

\* Quoted against the PyTorch-loop fp16 baseline (7.3648), the implementation HQQ shares. See `../fp16/README.md`.

† See below. Not a property of the checkpoint.

**A better per-group fit and a better end-to-end result are not the same claim.** HQQ reconstructs every individual group more faithfully than RTN does, and the two finish level: 7.815 against 7.804, a tenth of a percent apart in raw perplexity. Lower error inside one group doesn't guarantee lower error at the model's output after 32 layers.

The percentages differ more than the raw numbers do, +6.1% against +5.5%, only because the two are quoted against different fp16 baselines: HQQ against the PyTorch loop's 7.3648, RTN against llama.cpp's 7.3950. That 0.4% baseline gap is larger than the gap between the checkpoints, so this pair does not rank in either direction.

**The 5.61 GB is coverage, not inefficiency.** HQQ's default is `_IGNORE_LINEAR = ['lm_head']`, and embeddings are `nn.Embedding`, which its `nn.Linear` walker never reaches. So `lm_head.weight` and `model.embed_tokens.weight`, 128256 x 4096 each, both stay fp16, which is 1.96 GB of the file. llama.cpp quantizes both (`token_embd` at `Q4_0`, `output` at `Q6_K`). On the 6.98B parameters HQQ does convert it spends 4.50 bits each, exactly what every other 4-bit method here spends. The larger file buys nothing and costs nothing; it is a different set of tensors, not a worse encoding.

**One ordering here has outside support.** HQQ's authors published a Llama-2-7B benchmark at group size 64 putting AWQ ahead of HQQ and both well clear of GPTQ. This project reproduced that order on a different model with different tooling: HQQ trails AWQ by 0.30% here against 0.38% there, and GPTQ trails AWQ by 2.14% here against 1.89% there. HQQ's own comparison does not include RTN, so it never claims to beat it.

## Why generation speed is 0.6 t/s

HQQ's default backend (`HQQBackend.PYTORCH`) calls `dequantize()` on every forward pass rather than caching the dequantized tensor. A single-token decode step does barely any arithmetic to offset that, so decode lands roughly 200x slower than fp16. This is not Apple Silicon specific: the default backend dequantizes per call on any device. HQQ ships fused backends that avoid it (`torchao_int4`, `marlin`, `bitblas`), but all three are CUDA-only, so a CUDA user has a way around this and an MPS user does not. `use_cache=True` was confirmed on the config before concluding this; it is not a disabled-KV-cache bug.

## Two API gotchas

**HQQ's API is CUDA-first.** `AutoHQQHFModel.from_quantized` and `Quantizer.quantize`/`optimize_weights` all default to `device='cuda'` and will try to restore CUDA-mapped tensors on the wrong device with no warning. Every call in these scripts passes `device='mps'` explicitly.

**Use HQQ's own save/load, not transformers'.** Reloading a HQQ model saved through transformers' generic `save_pretrained`/`from_pretrained` raises `NotImplementedError: QuantizationMethod.HQQ is not available yet` on transformers 5.15.1. `AutoHQQHFModel.save_quantized`/`from_quantized` works and is what these scripts use.

## Running it

```bash
python quantize.py   --hf-dir <local HF snapshot dir> --out-dir ../models/llama-3.1-8b-instruct-hqq-q4
python perplexity.py --model ../models/llama-3.1-8b-instruct-hqq-q4 --tokenizer <snapshot dir> --label hqq-q4
python mmlu.py       --model ../models/llama-3.1-8b-instruct-hqq-q4 --tokenizer <snapshot dir> --label hqq-q4
python throughput.py --model ../models/llama-3.1-8b-instruct-hqq-q4 --tokenizer <snapshot dir> --label hqq-q4
```

`save_quantized` doesn't write tokenizer files, hence `--tokenizer`.
