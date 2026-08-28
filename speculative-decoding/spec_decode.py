"""Speculative decoding, hand-rolled and instrumented.

transformers computes acceptance internally and never surfaces it, so the loop is written out here to expose what it costs: draft forward passes spent, draft tokens discarded, and target forward passes saved.

Greedy only. Under sampling the accept test becomes the rejection-sampling rule from Leviathan et al. (arXiv 2211.17192), which is what preserves the output distribution exactly; the greedy version below is the same idea with an argmax comparison.
"""

import argparse
import statistics
import time
from dataclasses import dataclass, field

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def sync(device):
    """MPS and CUDA queue work asynchronously; timings are meaningless without this."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@dataclass
class Stats:
    wall: float = 0.0
    tokens: int = 0
    target_passes: int = 0
    draft_passes: int = 0
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    accepted_runs: list = field(default_factory=list)

    @property
    def discarded(self):
        return self.proposed - self.accepted

    @property
    def acceptance(self):
        return self.accepted / self.proposed if self.proposed else 0.0

    def report(self, label):
        logger.info(
            f"{label:<22} {self.wall:6.2f}s | {self.tokens:>4} tok | "
            f"{self.tokens / self.wall:6.2f} tok/s | target {self.target_passes:>4} | "
            f"draft {self.draft_passes:>4} | discarded {self.discarded:>4} | "
            f"accept {self.acceptance:5.1%}"
        )


@torch.no_grad()
def baseline(target, ids, n_tokens):
    """Plain autoregressive decoding. One target forward pass per token."""
    s = Stats()
    dev = ids.device.type
    sync(dev)
    t0 = time.perf_counter()
    out = target(ids, use_cache=True)
    past = out.past_key_values
    for _ in range(n_tokens):
        nxt = out.logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, nxt], dim=-1)
        out = target(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        s.target_passes += 1
        s.tokens += 1
    sync(dev)  # MPS/CUDA queue asynchronously; without this we time the queueing
    s.wall = time.perf_counter() - t0
    return ids, s


@torch.no_grad()
def speculative(target, draft, ids, n_tokens, k):
    """Draft k tokens, verify them in ONE target pass, keep the longest prefix the
    target agrees with, and take the target's own next token for free.

    The correction token is not verified in its own pass. It rides along at the front of the next round's verify input, so the target runs exactly once per round.
    """
    s = Stats()
    dev = ids.device.type
    sync(dev)
    t0 = time.perf_counter()

    def crop_to(cache, keep):
        extra = cache.get_seq_length() - keep
        if extra > 0:
            cache.crop(-extra)

    # target cache holds everything except the last token; that one is "pending". A one-token prompt would leave nothing to prefill, so seed an empty cache.
    if ids.shape[1] > 1:
        t_past = target(ids[:, :-1], use_cache=True).past_key_values
    else:
        t_past = DynamicCache()
    pending = ids[:, -1:]
    d_out = draft(ids, use_cache=True)
    d_past, d_logit = d_out.past_key_values, d_out.logits[:, -1]

    while s.tokens < n_tokens:
        s.rounds += 1
        base_len = ids.shape[1]

        # 1. draft k tokens autoregressively
        cand = []
        for _ in range(k):
            nxt = d_logit.argmax(-1, keepdim=True)
            cand.append(nxt)
            d_out = draft(nxt, past_key_values=d_past, use_cache=True)
            d_past, d_logit = d_out.past_key_values, d_out.logits[:, -1]
            s.draft_passes += 1
        cand_t = torch.cat(cand, dim=-1)
        s.proposed += k

        # 2. ONE target pass over [pending, d1..dk] -> k+1 logits
        t_out = target(torch.cat([pending, cand_t], dim=-1),
                       past_key_values=t_past, use_cache=True)
        t_past, logits = t_out.past_key_values, t_out.logits
        s.target_passes += 1

        # 3. compare all k slots on-device in one shot. Doing this with a Python    loop costs k GPU syncs per round via .item(); this costs one.
        choices = logits[0, :k].argmax(-1)
        run = (cand_t[0] == choices).cumprod(0)
        n_acc = int(run.sum().item())

        # 4. emit the accepted run plus the target's correction, which cost nothing
        correction = logits[:, n_acc].argmax(-1, keepdim=True)
        s.accepted += n_acc
        s.accepted_runs.append(n_acc)
        ids = torch.cat([ids, cand_t[:, :n_acc], correction], dim=-1)
        s.tokens += n_acc + 1

        # 5. target cache keeps everything but the correction, which becomes pending
        crop_to(t_past, base_len + n_acc)
        pending = correction

        # 6. the draft must see the correction before it can draft again
        crop_to(d_past, base_len + n_acc)
        d_out = draft(correction, past_key_values=d_past, use_cache=True)
        d_past, d_logit = d_out.past_key_values, d_out.logits[:, -1]
        s.draft_passes += 1

    sync(dev)
    s.wall = time.perf_counter() - t0
    overshoot = s.tokens - n_tokens
    if overshoot > 0:
        ids = ids[:, :-overshoot]
        s.tokens = n_tokens
    return ids, s


def load(model_id, device, dtype):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()


def main(target_id, draft_id, n_tokens, k, prompt, device="cpu", dtype=None):
    tok = AutoTokenizer.from_pretrained(target_id)
    logger.info(f"target {target_id} | draft {draft_id} | k={k} | {n_tokens} tokens | {device}")
    target = load(target_id, device, dtype)
    draft = load(draft_id, device, dtype)

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # warm up both models before any timing: the first forward pass pays lazy init
    baseline(target, ids, 4)
    speculative(target, draft, ids, 4, k)
    sync(device)

    base_ids, base_s = baseline(target, ids, n_tokens)
    sync(device)
    spec_ids, spec_s = speculative(target, draft, ids, n_tokens, k)
    sync(device)

    base_s.report("baseline")
    spec_s.report(f"speculative k={k}")
    logger.info(f"speedup {base_s.wall / spec_s.wall:.2f}x")

    a, b = base_ids[0, ids.shape[1]:], spec_ids[0, ids.shape[1]:]
    n = min(len(a), len(b))
    same = torch.equal(a[:n], b[:n])
    logger.info(f"output identical to baseline: {same}")
    if not same:
        logger.error("LOOP IS WRONG — greedy speculative must match greedy baseline")
        logger.error(f"baseline: {tok.decode(a[:n])!r}")
        logger.error(f"spec    : {tok.decode(b[:n])!r}")
    return same


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="openai-community/gpt2-large")
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--prompt", default="The history of the printing press begins")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="e.g. float16")
    a = ap.parse_args()
    main(a.target, a.draft, a.tokens, a.k, a.prompt, a.device,
         getattr(torch, a.dtype) if a.dtype else None)
