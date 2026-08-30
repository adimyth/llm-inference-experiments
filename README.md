# LLM Inference Experiments

Scripts behind the LLM Inference essays on adimyth.in. Every figure quoted in those essays is produced by running these.

One folder per topic. Each has its own README with the commands, the setup they were measured on, and the results.

| Folder | Essay | What it measures |
| --- | --- | --- |
| [kv-caching](kv-caching/) | [KV Caching](https://adimyth.in/essays/llm-inference-kv-caching) | Cache size against the formula, and what caching is worth in wall time |
| [speculative-decoding](speculative-decoding/) | [Speculative Decoding](https://adimyth.in/essays/llm-inference-speculative-decoding) | Speedup, acceptance, and discarded compute across lookahead and workload |
| [model-quantization](model-quantization/) | [Post-Training Quantization](https://adimyth.in/essays/llm-inference-quantization) | Size, speed, perplexity, and MMLU accuracy across six quantization methods, against the fp16 baseline |

## Setup

```bash
uv venv --python 3.12 --python-preference only-managed
uv pip install -r requirements.txt
```

`--python-preference only-managed` matters on Apple Silicon. A universal `python3` from python.org resolves to its x86_64 slice, and torch ships no Intel macOS wheels, so the install fails on a wheel-availability error that does not mention architecture.

Run scripts from inside their folder:

```bash
cd speculative-decoding
../.venv/bin/python spec_bench.py
```

## Method notes

These apply to every folder.

- Every timed region calls `torch.mps.synchronize()` before stopping the clock. MPS queues work asynchronously; without it you time how fast Python queued the work, not how long it took.
- Models are warmed before timing. The first forward pass pays lazy initialisation and will otherwise land in the measurement.
- Cells are medians of repeated runs. Single runs produced non-monotonic acceptance, which was measurement noise rather than a finding.
- Charts render light and dark separately rather than inverting one image.

Hardware for all measurements unless a README says otherwise: MacBook Pro, Apple M4 Pro (8P+4E), 48GB, macOS 15.7.5, torch 2.13.0, transformers 5.15.1.

AWQ and GPTQ are the exception. Neither has a Metal or MPS path, so both were quantized and measured on a rented NVIDIA L40S with its own pinned environment (`model-quantization/requirements-cuda.lock`). Perplexity and MMLU carry across the two machines and were checked rather than assumed; tokens per second does not.
