"""Full speculative decoding benchmark: k sweep and workload sweep, with repeats.

Every cell is the median of REPEATS runs. Single runs were noisy enough to produce
non-monotonic acceptance, which is a measurement artefact rather than a finding.

Writes results/results.json for plotting and records memory around model load and
around the run itself.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

from memstats import MemSnapshot
from spec_decode import baseline, load, speculative, sync
from workloads import DEFAULT, WORKLOADS

OUT = Path(__file__).parent / "results"


def machine():
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "chip": chip,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def median_run(fn, repeats):
    """Run fn repeats times, return the run whose wall time is the median."""
    runs = [fn() for _ in range(repeats)]
    runs.sort(key=lambda r: r[1].wall)
    return runs[len(runs) // 2]


def main(target_id, draft_id, n_tokens, ks, repeats, device, dtype):
    OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(target_id)

    mem_before = MemSnapshot.take()
    logger.info(mem_before.line("before load"))
    target = load(target_id, device, dtype)
    mem_target = MemSnapshot.take()
    logger.info(mem_target.line("target loaded"))
    draft = load(draft_id, device, dtype)
    mem_both = MemSnapshot.take()
    logger.info(mem_both.line("both loaded"))

    warm = tok(WORKLOADS[DEFAULT], return_tensors="pt").input_ids.to(device)
    baseline(target, warm, 4)
    speculative(target, draft, warm, 4, 4)
    sync(device)

    results = {
        "machine": machine(),
        "target": target_id,
        "draft": draft_id,
        "device": device,
        "dtype": str(dtype),
        "tokens": n_tokens,
        "repeats": repeats,
        "memory": {
            "before_load": mem_before.as_dict(),
            "target_loaded": mem_target.as_dict(),
            "both_loaded": mem_both.as_dict(),
        },
        "k_sweep": [],
        "workloads": [],
    }

    # --- k sweep, on the default workload ---
    ids = tok(WORKLOADS[DEFAULT], return_tensors="pt").input_ids.to(device)
    _, base = median_run(lambda: baseline(target, ids, n_tokens), repeats)
    logger.info(f"k sweep on '{DEFAULT}' | baseline {base.wall:.2f}s "
                f"({base.tokens / base.wall:.2f} tok/s), median of {repeats}")
    for k in ks:
        _, s = median_run(lambda: speculative(target, draft, ids, n_tokens, k), repeats)
        row = {
            "k": k, "speedup": base.wall / s.wall, "wall": s.wall,
            "tok_per_s": s.tokens / s.wall, "acceptance": s.acceptance,
            "discarded": s.discarded, "proposed": s.proposed,
            "target_passes": s.target_passes, "draft_passes": s.draft_passes,
            "rounds": s.rounds,
            "mean_accepted_run": statistics.mean(s.accepted_runs),
        }
        results["k_sweep"].append(row)
        logger.info(f"  k={k:>2} | {row['speedup']:5.2f}x | accept {s.acceptance:5.1%} | "
                    f"discarded {s.discarded:>4} | target {s.target_passes:>3} | "
                    f"draft {s.draft_passes:>4}")
    results["baseline"] = {"wall": base.wall, "tok_per_s": base.tokens / base.wall,
                           "target_passes": base.target_passes}

    # --- workload sweep at the best k from above ---
    best_k = max(results["k_sweep"], key=lambda r: r["speedup"])["k"]
    logger.info(f"workload sweep at k={best_k}, median of {repeats}")
    for name, prompt in WORKLOADS.items():
        wids = tok(prompt, return_tensors="pt").input_ids.to(device)
        _, b = median_run(lambda: baseline(target, wids, n_tokens), repeats)
        _, s = median_run(lambda: speculative(target, draft, wids, n_tokens, best_k), repeats)
        row = {
            "workload": name, "k": best_k, "speedup": b.wall / s.wall,
            "base_wall": b.wall, "spec_wall": s.wall,
            "acceptance": s.acceptance, "discarded": s.discarded,
            "proposed": s.proposed, "target_passes": s.target_passes,
            "draft_passes": s.draft_passes,
            "mean_accepted_run": statistics.mean(s.accepted_runs),
        }
        results["workloads"].append(row)
        logger.info(f"  {name:<14} {row['speedup']:5.2f}x | accept {s.acceptance:5.1%} | "
                    f"discarded {s.discarded:>4}")

    mem_end = MemSnapshot.take()
    results["memory"]["after_run"] = mem_end.as_dict()
    logger.info(mem_end.line("after run"))

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ks", default="1,2,3,4,5,6,8,10,12,16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    main(a.target, a.draft, a.tokens, [int(x) for x in a.ks.split(",")],
         a.repeats, a.device, getattr(torch, a.dtype))
