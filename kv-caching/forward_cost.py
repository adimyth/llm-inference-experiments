"""Is a forward pass cost-flat in sequence length on this GPU, and where does that stop?

If a 320-token forward costs the same as a 1-token forward, the KV cache
cannot save wall time, and the reason is fixed overhead rather than the GPU
being fast at the arithmetic. This finds where the flat region ends.
"""
import statistics, time, sys
import torch
from transformers import AutoModelForCausalLM

def bench(model_id, dtype, lengths, reps=5):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to("cuda").eval()
    print(f"\n=== {model_id} {dtype} ===")
    print(f"{'seq_len':>8} {'ms/forward':>11} {'vs len-1':>9} {'tok/ms':>8}")
    base = None
    with torch.no_grad():
        for n in lengths:
            ids = torch.randint(100, 5000, (1, n), device="cuda")
            for _ in range(2): m(ids)                    # warm this shape
            torch.cuda.synchronize()
            runs = []
            for _ in range(reps):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                m(ids)
                torch.cuda.synchronize(); runs.append(time.perf_counter() - t0)
            ms = statistics.median(runs) * 1000
            base = base or ms
            print(f"{n:>8} {ms:>11.2f} {ms/base:>9.2f}x {n/ms:>8.1f}")
    del m; torch.cuda.empty_cache()

bench("gpt2", torch.float32, [1, 64, 320, 512, 1024])  # gpt2 max position is 1024
bench("meta-llama/Llama-3.1-8B-Instruct", torch.float16, [1, 64, 320, 512, 1024, 2048, 4096, 8192])
