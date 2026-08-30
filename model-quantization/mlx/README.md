# MLX: built for this hardware, and not automatically cheaper

MLX is Apple's own array framework: unified memory, native Metal. Its default quantization uses the same `weight ≈ code × scale + offset` idea `../kquants/` uses, one scale and offset per group, but with a **bigger group** and no compression on the scale itself.

## How it works

MLX's default group is 64 weights, twice k-quants' 32. For a group whose largest and smallest weight happen to match a k-quants sub-block, the result is numerically identical:

| | Weight | `scale` | `offset` | Code | Dequantized |
| --- | --- | --- | --- | --- | --- |
| MLX group | -1.20 | 0.14 | -1.20 | 0 | -1.20 |
| MLX group | 0.90 | 0.14 | -1.20 | 15 | 0.90 |

The difference shows up when the group isn't uniform. k-quants would give a *second* group of 32, a quiet one, its own much smaller scale. MLX's 64-weight group can't split that finely: one scale covers twice as many weights, so a group straddling a loud region and a quiet one has to compromise between them. The scale ends up set by the group's extremes, so the quiet weights come back on a grid too coarse for them and lose a bigger fraction of themselves, the same failure `../rtn/` shows when `-0.05` dequantizes to `0.00`. A 64-weight group is twice as likely to contain both kinds of weight as a 32-weight one.

MLX stores its scale and offset uncompressed per group where k-quants squeeze theirs to 6 bits, which is more per group, but MLX's groups are twice as large and the two cancel exactly. Both spend 0.5 bits per weight on metadata and land at 4.5 bits per weight. `Q4_K_M` only looks heavier on disk, 4.58 GB against 4.22 GB, because its `_M` upgrades lift its average to 4.89 bits per weight.

**Group size is part of why MLX costs more here, though not all of it.** `../hqq/` uses the same 64-weight groups and lands nearly two points better, so how the scale is placed inside a group matters at least as much as how wide the group is.

## Results

| Metric | fp16 | RTN `Q4_0` | MLX 4-bit |
| --- | --- | --- | --- |
| Size on disk | 14.97 GB | 4.34 GB | 4.22 GB |
| Generation (tg128) | 15.5 t/s | 48.5 t/s | 52.8 t/s |
| Prompt processing (pp512) | 315.5 t/s | 377.8 t/s | 359.7 t/s |
| Perplexity, wikitext-2 | 7.365* | 7.804 (+5.5%) | 7.949 (+7.9%) |
| MMLU, 50q subset | 82.0% (41/50) | 76.0% (38/50) | 76.0% (38/50) |

\* MLX's perplexity is quoted against the **PyTorch-loop** fp16 baseline (7.3648), not llama.cpp's 7.3950, since that's the implementation it shares. The two agree to 0.4%, so the deltas are comparable. See `../fp16/README.md`.

MLX is slightly smaller than RTN and slightly faster to generate with, and costs somewhat more perplexity, +7.9% against +5.5%, for a checkpoint also called "4-bit". Those two are measured by different tools against different fp16 baselines, so read the pair as a band rather than a ranking. **"4-bit" alone doesn't specify a quantization scheme.** Group size is a second parameter, and most model cards leave it out.

Worth knowing before reading MLX's perplexity as a weakness: on the 200-question MMLU rescore MLX is the **best** of the four checkpoints tested, at 77.5% against fp16's 76.5%. See the MMLU caveat in `../README.md`.

MMLU doesn't move: identical to RTN at 76.0%, the same 38 of 50. With three 4-bit methods inside 2.4 perplexity points of each other, a tie on a 50-question subset is the expected outcome.

## A note on eval wall time

MLX's MMLU pass takes about 40 seconds. The GGUF checkpoints' take 25-27 minutes each. That has nothing to do with the models: the GGUF path scores answers over HTTP against a `llama_cpp.server`, four requests per question, while `mmlu.py` here calls `lm_eval` in-process against the loaded model. Same questions, same scoring, different plumbing.

## Running it

```bash
python quantize.py   --hf-path <model> --mlx-path ../models/llama-3.1-8b-instruct-mlx-q4
python perplexity.py --model ../models/llama-3.1-8b-instruct-mlx-q4 --label mlx-q4
python mmlu.py       --model ../models/llama-3.1-8b-instruct-mlx-q4 --label mlx-q4
python throughput.py --model ../models/llama-3.1-8b-instruct-mlx-q4 --label mlx-q4
```

MLX quantizes straight from the HF checkpoint, no f16 GGUF intermediate.
