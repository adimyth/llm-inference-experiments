# Speculative Decoding

Essay: [Speculative Decoding](https://adimyth.in/essays/llm-inference-speculative-decoding)

A small draft model guesses several tokens ahead; the large target model checks all of them in one forward pass and keeps the longest prefix it agrees with. Rejected guesses are compute you paid for and deleted, which is what these scripts measure.

Run from this folder with `../.venv/bin/python <script>`. See the [root README](../README.md) for setup and method notes.

```bash
../.venv/bin/python spec_bench.py       # k sweep + workload sweep -> results/results.json
../.venv/bin/python spec_plot.py        # charts from that json
../.venv/bin/python spec_decode.py -k 5 # single comparison
```

| Script | What it does |
| --- | --- |
| `spec_decode.py` | The draft-and-verify loop, plus the baseline it is measured against |
| `spec_bench.py` | k sweep and workload sweep, medians of 3, writes `results/results.json` |
| `spec_precondition.py` | Forward pass cost against how many tokens it verifies |
| `spec_one_pass.py` | Shows one forward pass yielding a prediction per position |
| `spec_drafts.py` | Compares draft models of different sizes |
| `spec_sweep.py` | Single-axis k sweep |
| `spec_plot.py` | Light and dark charts from `results.json` |
| `workloads.py` | The five prompts |
| `memstats.py` | RAM, VRAM, and swap sampling around a run |

Setup: Qwen/Qwen2.5-7B-Instruct drafted by Qwen/Qwen2.5-0.5B-Instruct, mps, float16, 64 tokens, median of 3 runs, greedy.

**Correctness:** greedy speculative output is asserted token-identical to greedy baseline on every run. With draft and target set to the same model, acceptance is 100% and nothing is discarded.

## Lookahead sweep

Baseline 15.75 tok/s, 64 target passes.

| k | Speedup | Accepted | Discarded | Target passes | Draft passes |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.20x | 91.2% | 3 | 34 | 68 |
| 2 | 1.44x | 82.0% | 9 | 25 | 75 |
| 3 | 1.63x | 78.3% | 13 | 20 | 80 |
| 4 | 1.64x | 69.4% | 22 | 18 | 90 |
| 5 | 1.83x | 72.0% | 21 | 15 | 90 |
| 6 | 1.79x | 66.7% | 28 | 14 | 98 |
| 8 | 1.75x | 58.3% | 40 | 12 | 108 |
| 10 | 1.47x | 57.3% | 47 | 11 | 121 |
| 12 | 1.36x | 48.5% | 68 | 11 | 143 |
| 16 | 1.13x | 32.4% | 119 | 11 | 187 |

![k sweep](results/k_sweep_light.png)

Speedup peaks at k=5 (1.83x) and falls away after. Target passes bottom out around 11 while draft passes keep climbing, so past the peak you are buying waste rather than speed. Discarded tokens go from 3 to 119.

## Workload sweep

Acceptance is how often the draft guesses what the target would have said, so it depends on how predictable the text is. Prompts are in `workloads.py`.

| Workload | Speedup | Accepted | Discarded |
| --- | --- | --- | --- |
| code | 2.40x | 98.2% | 1 |
| repetitive | 2.39x | 100.0% | 0 |
| open prose | 1.82x | 72.0% | 21 |
| factual list | 1.60x | 60.0% | 34 |
| structured | 1.36x | 47.0% | 53 |

![workloads](results/workloads_light.png)

Code is the fastest real case at 2.40x, because syntax makes the next token largely forced and a 0.5B draft gets it right 98% of the time.

## Memory

| Stage | MPS allocated (GB) | MPS driver (GB) | System available (GB) | Swap (GB) |
| --- | --- | --- | --- | --- |
| before load | 0.00 | 0.00 | 30.3 | 2.21 |
| target loaded | 14.19 | 15.04 | 9.3 | 2.21 |
| both loaded | 15.11 | 15.14 | 15.8 | 2.21 |
| after run | 15.11 | 15.33 | 14.4 | 2.21 |

Swap never moved, so none of these timings are contaminated by paging.

## The precondition

Cost of one forward pass against how many tokens it verifies. Speculative decoding only pays if checking k tokens costs about what checking one costs.

On MPS that holds: k=4 costs 1.04x of k=1. On CPU it does not, k=4 costs 3.63x, which is why this is a GPU technique.

## On the choice of k

Our measured optimum is **k=5** on prose. That is not an unusual number.

- **vLLM** uses `num_speculative_tokens: 5` throughout its [speculative decoding docs](https://docs.vllm.ai/en/stable/features/spec_decode.html).
- **HuggingFace** does not expose a fixed k at all. It ships an adaptive `num_assistant_tokens_schedule="heuristic"` that raises the lookahead by 2 when a whole block is accepted and backs off when it is not, which is the same behaviour our workload sweep shows: the best k moves with the text.
- **Leviathan et al.** ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)) give the expected speedup in closed form and say the optimal lookahead "is the one maximizing the walltime improvement equation", found numerically because it depends on both the acceptance rate and the draft/target cost ratio. There is no universal constant.
- **[Dynamic Speculation Lookahead](https://arxiv.org/abs/2405.04304)** argues the optimum varies per input, which is why frameworks ship a schedule rather than a number.
- Practitioner write-ups put the useful band at **4 to 8** and drafts at **1/10 to 1/50** of the target ([Introl](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025), [Inference.net](https://inference.net/content/speculative-decoding/)).

Why more lookahead stops helping, exactly: expected accepted tokens per round is `(1 - a^(k+1)) / (1 - a)`, which saturates at `1/(1-a)`, while draft cost grows linearly in k. Bounded benefit, unbounded cost. At a=0.8 the ceiling is 5 accepted tokens, so k=64 buys 0.67 more than k=8 for eight times the draft compute.

Leviathan's gate: an improvement exists if and only if `a > c`, where c is the draft/target cost ratio. Ours is a=0.72, c=0.17.

## Speculative speculative decoding

*Experiments in progress.*

Vanilla speculative decoding wastes time in both directions: the target sits idle while the draft is guessing, and the draft sits idle while the target is checking. Speculative speculative decoding attacks that second half. Rather than waiting, the draft predicts what the verification will conclude and starts producing the next block of guesses against that prediction, so the two models work at the same time instead of taking turns. PEARL ([arXiv 2408.11850](https://arxiv.org/abs/2408.11850)) calls this the mutual waiting problem and reports 1.50x over vanilla speculative decoding.

Measuring it needs two models running genuinely at once. A single Apple GPU has one command queue and serialises them: two models in two threads measured 0.93x here, marginally slower than running them one after the other. Real numbers need either separate devices or a GPU with concurrent streams.
