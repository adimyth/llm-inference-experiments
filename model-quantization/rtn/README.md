# RTN: round to nearest, and nothing else

The naive baseline. No calibration data, no optimization, just round each weight to the nearest point on the grid.

## How it works

Split the weights into blocks of 32. Each block gets one shared **scale**, a single number, call it `d`. Every weight in the block is replaced by two things: `d` (stored once, shared by all 32) and a small integer **code**, stored per weight in 4 bits. To read a weight back, multiply: `weight ≈ code × d`.

`d` is set from the block's own largest weight, so nothing in the block overflows what 4 bits can hold. A 4-bit signed code runs from -8 to 7, so `d = max(abs(weights)) / 8`.

A block of four, to see the mechanism (real blocks hold 32):

| Weight | `weight / d` | Code | Dequantized (`code × d`) |
| --- | --- | --- | --- |
| 0.90 | 6.00 | 6 | 0.90 |
| -1.20 | -8.00 | -8 | -1.20 |
| 0.30 | 2.00 | 2 | 0.30 |
| -0.05 | -0.33 | 0 | 0.00 |

`d = 1.20 / 8 = 0.15`, set by the largest weight. The three larger weights come back almost exactly. The smallest, -0.05, rounds to code 0 and vanishes.

**RTN's error is a flat ±d/2, the same for every weight in the block**, whether that weight started out big or tiny. A quiet weight and a loud one both get rounded to the nearest multiple of 0.15, so the quiet one loses a much bigger fraction of itself. That is the weakness `../kquants/` and `../hqq/` each attack differently.

## Tooling

llama.cpp's `Q4_0` quant type. Quantizing the full 8B model takes about 9 seconds.

## Results

| Metric | fp16 | RTN `Q4_0` |
| --- | --- | --- |
| Size on disk | 14.97 GB | 4.34 GB |
| Generation (tg128) | 15.5 t/s | 48.5 t/s |
| Prompt processing (pp512) | 315.5 t/s | 377.8 t/s |
| Perplexity, wikitext-2 | 7.395 | 7.804 (+5.5%) |
| MMLU, 50q subset | 82.0% (41/50) | 76.0% (38/50) |

71% smaller, 3.1x faster to decode, +5.5% perplexity. The MMLU drop is larger than perplexity suggests: 6 points, 3 of 50 questions flipping.

## Running it

```bash
python quantize.py   --infile ../models/llama-3.1-8b-instruct-f16.gguf --outfile ../models/llama-3.1-8b-instruct-q4_0.gguf
python perplexity.py --model  ../models/llama-3.1-8b-instruct-q4_0.gguf --label rtn-q4_0
python mmlu.py       --model  ../models/llama-3.1-8b-instruct-q4_0.gguf --label rtn-q4_0
python throughput.py --model  ../models/llama-3.1-8b-instruct-q4_0.gguf --label rtn-q4_0
```

Requires the f16 GGUF from `../fp16/convert.py` first.
