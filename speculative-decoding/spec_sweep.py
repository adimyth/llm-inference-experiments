"""How draft length k trades speed against wasted compute, measured and predicted.

Leviathan et al. (arXiv 2211.17192) give the expected walltime improvement as

IF(a, g, c) = (1 - a**(g+1)) / ((1 - a) * (g*c + 1))

with a the acceptance rate, g the draft length and c the cost coefficient, meaning draft forward pass time over target forward pass time. The numerator saturates at 1/(1-a) while the denominator grows linearly in g: bounded benefit, unbounded cost.

This measures a and c for a real pair, predicts the optimal k from the formula, then sweeps k and checks whether the measured optimum lands where the theory says.
"""

import argparse
import statistics
import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_decode import baseline, load, speculative, sync


def improvement_factor(a, g, c):
    if a >= 1.0:
        return (g + 1) / (g * c + 1)
    return (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))


def expected_tokens(a, g):
    if a >= 1.0:
        return g + 1
    return (1 - a ** (g + 1)) / (1 - a)


@torch.no_grad()
def cost_coefficient(target, draft, ids, reps=9):
    """c = one draft forward pass / one target forward pass, both single token."""
    dev = ids.device.type

    def one(model):
        past = model(ids, use_cache=True).past_key_values
        tok = ids[:, -1:]
        for _ in range(3):
            model(tok, past_key_values=past, use_cache=True)
        sync(dev)
        runs = []
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            model(tok, past_key_values=past, use_cache=True)
            sync(dev)
            runs.append(time.perf_counter() - t0)
        return statistics.median(runs)

    d, t = one(draft), one(target)
    logger.info(f"draft {d * 1000:.1f} ms/pass | target {t * 1000:.1f} ms/pass")
    return d / t


def main(target_id, draft_id, n_tokens, ks, prompt, device="cpu", dtype=None):
    tok = AutoTokenizer.from_pretrained(target_id)
    target = load(target_id, device, dtype)
    draft = load(draft_id, device, dtype)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    baseline(target, ids, 4)
    speculative(target, draft, ids, 4, 4)
    sync(device)

    c = cost_coefficient(target, draft, ids)
    logger.info(f"target {target_id} | draft {draft_id}")
    logger.info(f"cost coefficient c = {c:.3f} (draft pass / target pass)")

    _, base_s = baseline(target, ids, n_tokens)
    base_s.report("baseline")

    rows = []
    for k in ks:
        out_ids, s = speculative(target, draft, ids, n_tokens, k)
        speedup = base_s.wall / s.wall
        rows.append((k, speedup, s.acceptance, s.discarded, s.target_passes, s.draft_passes))
        logger.info(
            f"k={k:>2} | {speedup:5.2f}x | accept {s.acceptance:5.1%} | "
            f"discarded {s.discarded:>4} | target {s.target_passes:>4} | draft {s.draft_passes:>4}"
        )

    alpha = statistics.median([r[2] for r in rows])
    best_measured = max(rows, key=lambda r: r[1])[0]
    best_predicted = max(ks, key=lambda g: improvement_factor(alpha, g, c))
    logger.info("")
    logger.info(f"median acceptance a = {alpha:.3f}, cost coefficient c = {c:.3f}")
    logger.info(f"expected accepted tokens caps at 1/(1-a) = {1 / (1 - alpha):.2f}" if alpha < 1 else "")
    logger.info(f"Leviathan gate a > c: {'holds' if alpha > c else 'FAILS, no k helps'}")
    logger.info(f"predicted optimal k = {best_predicted} | measured optimal k = {best_measured}")
    logger.info("predicted improvement by k: " + " ".join(
        f"k={g}:{improvement_factor(alpha, g, c):.2f}" for g in ks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="openai-community/gpt2-large")
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ks", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--prompt", default="The history of the printing press begins")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()
    main(a.target, a.draft, a.tokens, [int(x) for x in a.ks.split(",")], a.prompt,
         a.device, getattr(torch, a.dtype) if a.dtype else None)
