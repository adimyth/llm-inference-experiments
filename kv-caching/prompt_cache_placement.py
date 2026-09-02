"""Where the fixed document goes, and what each choice costs.

A support agent answers questions against a policy handbook. The handbook is the same every time. The question is different every time. Nothing else varies.

Three ways to assemble that request, each changing one thing from the one before it:

    A   user = [question][handbook]                 both in one message, changing part leads
    B   user = [handbook][question]                 both in one message, fixed part leads
    C   system = [handbook],  user = [question]     split, fixed part leads
    D   system = [question],  user = [handbook]     split, changing part leads

A and B differ in ordering inside one message. C and D differ in ordering across the two fields. D is the control that matters: it puts the changing part in the system field, so if the system field were special rather than merely first, D would still cache.
"""

from loguru import logger

import cache_common as cc

MODELS = ("gpt-5.6-terra", "gpt-5.4", "gpt-4o")
REQUESTS = 8

ROLE = (
    "You are a support agent for Northwind Devices. Answer strictly from the handbook. "
    "If the handbook does not cover the question, say so and escalate. Answer in one sentence."
)

QUESTIONS = [
    "My device arrived with a cracked screen. What happens now?",
    "I bought it 40 days ago and want to return it. Can I?",
    "The battery only lasts two hours. Is that covered?",
    "I was charged twice this month.",
    "I bought it from a marketplace seller. Do I still have warranty?",
    "My parcel has not moved for nine days.",
    "How long does a repair take once you receive it?",
    "Can I transfer the account to my partner?",
]


CONDITIONS = {
    "A": "one message, question leads",
    "B": "one message, handbook leads",
    "C": "split, handbook in the system field",
    "D": "split, question in the system field",
}


def build(condition, nonce, handbook, question):
    """The three prompt shapes.

    All three send the same bytes. The role line is identical everywhere and far
    too short to cache on its own, so condition A has no reusable prefix at all.
    The nonce leads the instructions so each run gets its own cache namespace.
    """
    instructions = f"[run {nonce}] {ROLE}"
    if condition == "A":
        return instructions, f"{question}\n\n{handbook}"
    if condition == "B":
        return instructions, f"{handbook}\n\n{question}"
    if condition == "C":
        return f"{instructions}\n\n{handbook}", question
    return f"{instructions}\n\n{question}", handbook


def dry_run(handbook):
    logger.info(f"handbook fixture: {cc.ntokens(handbook)} tokens")
    nonce = cc.run_nonce()
    for condition in CONDITIONS:
        prompts = [build(condition, nonce, handbook, QUESTIONS[i]) for i in range(REQUESTS)]
        prefixes = {(p[0] + p[1])[:400] for p in prompts}
        total = cc.ntokens(prompts[0][0]) + cc.ntokens(prompts[0][1])
        logger.info(
            f"condition {condition}: {total} prompt tokens, "
            f"{len(prefixes)} distinct prefix(es) across {REQUESTS} requests"
        )
        logger.info(f"  first 100 chars of the input: {prompts[0][1][:100]!r}")


def run_condition(client, model, condition, handbook, nonce, attempts=3):
    """One condition, five requests.

    A retry inside `ask` re-sends a prompt whose discarded attempt already warmed
    the cache, which would show up as a hit this condition did not earn. So a
    retry anywhere throws the condition away and restarts it under a fresh nonce.
    """
    for attempt in range(attempts):
        rows = _attempt_condition(client, model, condition, handbook, nonce)
        if not any(r.get("retried") for r in rows):
            return rows
        logger.warning(
            f"  a request retried, so the prefix was already warm. "
            f"discarding condition {condition} and restarting ({attempt + 1}/{attempts})"
        )
        nonce = cc.run_nonce()
    raise RuntimeError(f"condition {condition} could not complete without a retry")


def _attempt_condition(client, model, condition, handbook, nonce):
    rows = []
    # Pin routing for the whole condition. Without this, requests scatter across
    # machines and the misses that follow are a routing artefact rather than
    # anything to do with how the request was assembled. See prompt_cache_routing.py.
    cache_key = f"placement-{condition}-{nonce}"
    for i in range(REQUESTS):
        instructions, user_input = build(condition, nonce, handbook, QUESTIONS[i])
        sent = cc.ntokens(instructions) + cc.ntokens(user_input)
        u = cc.ask(client, model, instructions, user_input, cache_key=cache_key)
        u["ok"] = cc.check_identity(u, sent)
        u["sent_tokens"] = sent
        rows.append(u)
        logger.info(
            f"  {condition}{i + 1}  input {u['input_tokens']:>6}  "
            f"cached {u['cached_tokens']:>6}  written {u['cache_write_tokens']:>6}  "
            f"uncached {u['uncached_tokens']:>6}  {u['seconds']:.2f}s"
        )
    return rows


def summarise(rows, model):
    billed = sum(cc.billed_units(r, model) for r in rows)
    baseline = sum(cc.baseline_units(r) for r in rows)
    return {
        "billed_units": billed,
        "baseline_units": baseline,
        "ratio": billed / baseline,
        "total_cached": sum(r["cached_tokens"] for r in rows),
        "total_written": sum(r["cache_write_tokens"] for r in rows),
        "identity_ok": all(r["ok"] for r in rows),
    }


def main(models, dry=False):
    handbook = cc.handbook()
    if dry:
        return dry_run(handbook)

    client = cc.client_or_exit()
    out = {"fixture_tokens": cc.ntokens(handbook), "requests": REQUESTS, "models": {}}

    for model in models:
        r = cc.rates_for(model)
        logger.info(f"{model}  (cache read {r['read']}x, cache write {r['write']}x)")
        per_condition = {}
        for condition in CONDITIONS:
            logger.info(f" condition {condition}, {CONDITIONS[condition]}")
            # A fresh nonce per condition as well as per run, so B cannot read
            # an entry that A wrote.
            rows = run_condition(client, model, condition, handbook, cc.run_nonce())
            per_condition[condition] = {"rows": rows, "summary": summarise(rows, model)}
        out["models"][model] = per_condition

    logger.info("cost of the input side, as a multiple of the same requests with no caching")
    for model, conditions in out["models"].items():
        cells = "  ".join(f"{c} {conditions[c]['summary']['ratio']:.2f}x" for c in CONDITIONS)
        logger.info(f"  {model:<16} {cells}")
    cc.save("placement.json", out)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--dry-run", action="store_true", help="build the prompts, call nothing")
    a = ap.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()], dry=a.dry_run)
