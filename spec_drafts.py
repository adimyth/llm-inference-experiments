"""Does a bigger draft help? Two drafts, one target.

A larger draft agrees with the target more often, so more of its guesses survive.
It also costs more per guess. Leviathan et al. found the smaller of their two
drafts won on wall clock despite agreeing less, which is the trade this measures.

Both drafts must share the target's tokenizer. A GPT-2 draft cannot be paired with
a Qwen target at all: token 464 is "The" in one vocabulary and four tabs in the
other, so the comparison the loop performs would be meaningless.
"""

import argparse
import statistics
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

from spec_decode import baseline, load, speculative, sync
from workloads import DEFAULT, WORKLOADS


@torch.no_grad()
def pass_ms(model, ids, device, reps=9):
    past = model(ids, use_cache=True).past_key_values
    tok = ids[:, -1:]
    for _ in range(3):
        model(tok, past_key_values=past, use_cache=True)
    sync(device)
    runs = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        model(tok, past_key_values=past, use_cache=True)
        sync(device)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs) * 1000


def main(target_id, draft_ids, n_tokens, ks, device, dtype, repeats):
    tok = AutoTokenizer.from_pretrained(target_id)
    target = load(target_id, device, dtype)
    ids = tok(WORKLOADS[DEFAULT], return_tensors="pt").input_ids.to(device)
    t_ms = pass_ms(target, ids, device)

    baseline(target, ids, 4)
    sync(device)
    walls = sorted(baseline(target, ids, n_tokens)[1].wall for _ in range(repeats))
    base_wall = walls[len(walls) // 2]
    logger.info(f"target {target_id}: {t_ms:.1f} ms/pass, baseline {base_wall:.2f}s")

    for did in draft_ids:
        draft = load(did, device, dtype)
        d_ms = pass_ms(draft, ids, device)
        speculative(target, draft, ids, 4, 4)
        sync(device)
        for k in ks:
            runs = sorted((speculative(target, draft, ids, n_tokens, k)[1]
                           for _ in range(repeats)), key=lambda s: s.wall)
            s = runs[len(runs) // 2]
            logger.info(
                f"  {did.split('/')[-1]:<22} k={k} | {base_wall / s.wall:5.2f}x | "
                f"accept {s.acceptance:5.1%} | discarded {s.discarded:>3} | "
                f"draft {d_ms:5.1f} ms/pass | c={d_ms / t_ms:.3f}")
        del draft
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--drafts", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ks", default="3,5")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    main(a.target, a.drafts.split(","), a.tokens,
         [int(x) for x in a.ks.split(",")], a.device,
         getattr(torch, a.dtype), a.repeats)
