# GGUF k-quants: an offset and a scale at the same size

`../rtn/` gives every weight in a 32-weight block one shared scale, sized for the block's largest magnitude, so a quiet weight can get crushed towards zero. k-quants keep the same 32-weight grouping but store an **offset** alongside the scale, so their 16 codes cover the block's actual minimum-to-maximum range rather than a symmetric range around zero. The challenge is fitting that second number into the same storage budget.

## How it works

RTN reads a weight back as `code × d`, always anchored at zero. k-quants read it back as:

```
weight ≈ code × scale + offset
```

`code` is the same small integer, now unsigned, 0 to 15. `scale` and `offset` are computed once per 32-weight sub-block: `offset` is the sub-block's smallest weight, and `scale` is sized so code 15 lands exactly on its largest:

```
scale  = (largest weight − smallest weight) / 15
offset = smallest weight
```

Two sub-blocks, shown by their endpoints. A is RTN's example above; B is another stretch of the same matrix whose values are a hundred times quieter:

| | Weight | `scale` | `offset` | Code | Dequantized |
| --- | --- | --- | --- | --- | --- |
| Sub-block A | -1.20 | 0.14 | -1.20 | 0 | -1.20 |
| Sub-block A | 0.90 | 0.14 | -1.20 | 15 | 0.90 |
| Sub-block B | -0.03 | 0.0047 | -0.03 | 0 | -0.03 |
| Sub-block B | 0.04 | 0.0047 | -0.03 | 15 | 0.04 |

`Q4_0` would also give B its own block and scale. The difference is that `Q4_0` uses one symmetric scale around zero while `Q4_K` stores both a scale and an offset, mapping B's endpoints exactly onto codes 0 and 15.

**Keeping two numbers without spending more.** `Q4_0` spends one fp16 scale, 16 bits, per 32 weights, so 128 metadata bits per 256 weights. A `Q4_K` block needs a scale *and* an offset, naively 256 bits. llama.cpp's fix: group eight sub-blocks into one **super-block of 256 weights**, compress each sub-block's scale and offset to 6 bits (6 × 2 × 8 = 96 bits), then store two fp16 super-block scales to decode them (32 bits). 128 bits total, exactly `Q4_0`'s budget. Weight codes stay 4-bit in both, so both work out to 4.5 bits per weight.

This two-level structure is what llama.cpp calls *k-quants*. The `K` is a family label, not an expansion of anything the original implementation documents.

**`_M` is a separate decision: which tensors get more bits.** `Q4_K_M` quantizes most of the model with the scheme above but bumps some tensors to 6-bit (`Q6_K`): the attention **value** projection and the FFN **down** projection on a fixed subset of layers, plus the model's final vocabulary projection (`output.weight`). The rule is arithmetic, fixed in [llama.cpp's quantization source](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp), not computed per model:

```cpp
   i_layer <  n_layers/8              // the first eighth of the layers
|| i_layer >= n_layers*7/8            // the last eighth
|| (i_layer - n_layers/8) % 3 == 2    // every third one in between
```

For Llama 3.1 8B's 32 layers that selects 16 of them:

```
0  1  2  3  6  9  12  15  18  21  24  27  28  29  30  31
```

Reading the produced file back confirms 33 tensors at `Q6_K`: `attn_v.weight` and `ffn_down.weight` on those 16 layers, and `output.weight` once. Everything else quantized is `Q4_K`. `Q5_K_M` follows the same idea from a 5-bit base.

Those upgrades are the whole reason `Q4_K_M` is larger than a plain 4-bit checkpoint. `Q4_K` itself works out to 4.5 bits per weight, the same as `../mlx/`; the `_M` upgrades lift the file's average to 4.89.

## Results

| Metric | fp16 | `Q4_K_M` | `Q5_K_M` | `Q8_0` |
| --- | --- | --- | --- | --- |
| Size on disk | 14.97 GB | 4.58 GB | 5.34 GB | 7.95 GB |
| Generation (tg128) | 15.5 t/s | 44.9 t/s | 31.2 t/s | 28.1 t/s |
| Prompt processing (pp512) | 315.5 t/s | 359.9 t/s | 329.4 t/s | 362.6 t/s |
| Perplexity, wikitext-2 | 7.395 | 7.622 (+3.1%) | 7.467 (+1.0%) | 7.399 (+0.05%) |
| MMLU, 50q subset | 82.0% (41/50) | 78.0% (39/50) | 80.0% (40/50) | 80.0% (40/50) |

`Q4_K_M` costs nearly the same as RTN's `Q4_0` (4.58 GB against 4.34 GB) and buys back almost half the perplexity gap. Same tool, same source file, same scoring, so that one is a clean comparison. It also answers one more MMLU question, which on 50 questions is noise rather than a result. `Q8_0` at 7.399 against fp16's 7.395 is close enough to call unquantized.

## Running it

```bash
python quantize.py   --infile ../models/llama-3.1-8b-instruct-f16.gguf   # writes all three
python perplexity.py --model  ../models/llama-3.1-8b-instruct-q4_k_m.gguf --label q4_k_m
python mmlu.py       --model  ../models/llama-3.1-8b-instruct-q4_k_m.gguf --label q4_k_m
python throughput.py --model  ../models/llama-3.1-8b-instruct-q4_k_m.gguf --label q4_k_m
```

Repeat the three benchmarks for `q5_k_m` and `q8_0`. Requires the f16 GGUF from `../fp16/convert.py`.
