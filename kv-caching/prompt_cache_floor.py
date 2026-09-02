"""How long a prompt has to be before caching happens at all.

Below a minimum prefix length the API caches nothing. It does not error and it does not warn. The response looks exactly like a response that cached, except `cached_tokens` is zero and the bill is higher.

The script slices the handbook to a series of exact token counts, sends each one twice, and reports where the second request starts reading from cache.

Each slice gets its own nonce at the front. Without that, a 512-token slice is a byte-for-byte prefix of the 1024-token slice, so the shorter slices would warm the longer ones and every threshold after the first would be wrong.

Run with --dry-run to check the slicing without calling the API.
"""

from loguru import logger

import cache_common as cc

MODELS = ("gpt-5.6-terra", "gpt-5.4", "gpt-4o")
SLICES = (256, 512, 1024, 1536, 2048, 3072, 5000)

ROLE = "You are a support agent for Northwind Devices. Answer from the handbook in one sentence.\n\n"
QUESTION = "What is the return window?"


def build(nonce, slice_tokens):
    """One instruction block at an exact size, namespaced so slices cannot warm each other."""
    return f"[slice {slice_tokens} run {nonce}]\n\n{ROLE}{cc.handbook(slice_tokens)}"


def dry_run():
    logger.info("slice sizes, measured after truncation")
    for n in SLICES:
        instructions = build(cc.run_nonce(), n)
        logger.info(f"  requested {n:>5} handbook tokens -> {cc.ntokens(instructions):>5} sent")


def run_slice(client, model, slice_tokens):
    """Identical requests at one size. The first can only write, the rest could read.

    Routing is pinned, because without it a request that lands on a machine which
    never saw the prefix is indistinguishable from a prompt below the floor. Three
    requests rather than two, so one stray miss does not decide the threshold.
    """
    nonce = cc.run_nonce()
    instructions = build(nonce, slice_tokens)
    sent = cc.ntokens(instructions) + cc.ntokens(QUESTION)
    cache_key = f"floor-{slice_tokens}-{nonce}"
    rows = [cc.ask(client, model, instructions, QUESTION, cache_key=cache_key) for _ in range(3)]
    for u in rows:
        u["ok"] = cc.check_identity(u, sent)
    cached_after_first = [r["cached_tokens"] for r in rows[1:]]
    best = max(cached_after_first)
    logger.info(
        f"  {slice_tokens:>5} tokens | sent {sent:>5} | wrote {rows[0]['cache_write_tokens']:>5} "
        f"| read back {cached_after_first}"
        f"{'   <- caching' if best > 0 else '   nothing cached'}"
    )
    return {"sent_tokens": sent, "rows": rows, "cached_after_first": best}


def main(models, slices, dry=False):
    if dry:
        return dry_run()

    client = cc.client_or_exit()
    out = {"slices": list(slices), "models": {}}

    for model in models:
        logger.info(f"{model}")
        rows = {n: run_slice(client, model, n) for n in slices}
        hits = [n for n in slices if rows[n]["cached_after_first"] > 0]
        out["models"][model] = {
            "rows": rows,
            "first_slice_that_cached": hits[0] if hits else None,
        }
        floor = hits[0] if hits else None
        logger.info(
            f"  smallest slice that cached: {floor}"
            if floor
            else "  nothing cached at any slice size"
        )

    logger.info("smallest handbook slice where the second request read from cache")
    for model, r in out["models"].items():
        logger.info(f"  {model:<16} {r['first_slice_that_cached']}")
    cc.save("floor.json", out)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--slices", default=",".join(str(n) for n in SLICES))
    ap.add_argument("--dry-run", action="store_true", help="slice the fixture, call nothing")
    a = ap.parse_args()
    main(
        [m.strip() for m in a.models.split(",") if m.strip()],
        tuple(int(x) for x in a.slices.split(",")),
        dry=a.dry_run,
    )
