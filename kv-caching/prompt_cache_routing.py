"""Whether the request lands where its cache lives.

A cache entry is local to the machine that built it. Send the next request to a different machine and the prefix is warm somewhere you are not.

Self-hosted, this is the load balancer's problem. On a hosted API the machines are not yours and you cannot see them. What you get instead is `prompt_cache_key`, a hint that groups related requests so they tend to land together. The documentation is careful to say it influences routing rather than pinning it.

This sends the same prompt repeatedly, twice: once with no key and once with a key. Everything else is identical, so the difference in hit rate is the routing hint.

The result depends on how busy the provider's fleet is, so it is a demonstration that the lever does something rather than a number to quote. Expect it to vary between runs.
"""

from loguru import logger

import cache_common as cc

MODELS = ("gpt-4o", "gpt-5.4")
REQUESTS = 8

ROLE = "You are a support agent for Northwind Devices. Answer from the handbook in one sentence.\n\n"
QUESTION = "What is the return window?"


def run(client, model, handbook, cache_key):
    """Identical requests back to back. Request 1 can only write; the rest could read."""
    nonce = cc.run_nonce()
    instructions = f"[run {nonce}]\n\n{ROLE}{handbook}"
    rows = []
    for _ in range(REQUESTS):
        rows.append(cc.ask(client, model, instructions, QUESTION, cache_key=cache_key))
    return rows


def hit_rate(rows):
    """Request 1 has nothing to read, so it is excluded from the denominator."""
    after_first = rows[1:]
    hits = sum(1 for r in after_first if r["cached_tokens"] > 0)
    return hits, len(after_first)


def main(models):
    client = cc.client_or_exit()
    handbook = cc.handbook()
    out = {"requests": REQUESTS, "models": {}}

    for model in models:
        logger.info(model)
        per_arm = {}
        for label, key in (("no_key", None), ("with_key", "northwind-support-v1")):
            rows = run(client, model, handbook, key)
            hits, total = hit_rate(rows)
            per_arm[label] = {
                "rows": rows,
                "hits": hits,
                "of": total,
                "cached_sequence": [r["cached_tokens"] for r in rows],
            }
            shown = "prompt_cache_key set" if key else "no prompt_cache_key"
            logger.info(f"  {shown:22s} hit {hits}/{total}  {per_arm[label]['cached_sequence']}")
        out["models"][model] = per_arm

    logger.info("cache hit rate after the first request")
    for model, arms in out["models"].items():
        a, b = arms["no_key"], arms["with_key"]
        logger.info(f"  {model:<16} without {a['hits']}/{a['of']}   with {b['hits']}/{b['of']}")
    cc.save("routing.json", out)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()])
