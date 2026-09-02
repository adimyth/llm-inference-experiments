"""Shared pieces for the two prompt-caching scripts.

These are the only scripts in this repository that call a hosted API. They need OPENAI_API_KEY, a network, and a few cents. Everything else here runs offline.
"""

import json
import os
import pathlib
import secrets
import time

import tiktoken
from loguru import logger

HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "fixtures" / "support_handbook.txt"
RESULTS = HERE / "results" / "prompt-cache-openai"

# Billing multiples, relative to each model's own uncached input rate.
# Read from the published pricing table rather than measured, and the only
# numbers here that did not come out of an API response. Checked 2026-09-02.
#   GPT-5.6 family: cached input 0.1x, cache write 1.25x
#   GPT-5.4 and earlier: cached input 0.1x, no separate cache-write charge
RATES = {
    "gpt-5.6": {"read": 0.1, "write": 1.25},
    "default": {"read": 0.1, "write": 1.0},
}


def rates_for(model):
    for prefix, r in RATES.items():
        if prefix != "default" and model.startswith(prefix):
            return r
    return RATES["default"]


def encoder():
    return tiktoken.get_encoding("o200k_base")


def handbook(max_tokens=None):
    """The fixture, optionally truncated to an exact token count."""
    text = FIXTURE.read_text()
    if max_tokens is None:
        return text
    enc = encoder()
    return enc.decode(enc.encode(text)[:max_tokens])


def ntokens(text):
    return len(encoder().encode(text))


def run_nonce():
    """A fresh namespace per run.

    Without this the first request of a run reads an entry written by the previous run, which looks like a hit that the run did not earn. The nonce sits at the very front of the prefix, so everything after it belongs to this run alone.
    """
    return secrets.token_hex(8)


def usage_of(response):
    """Raw token accounting from one response.

    OpenAI reports `input_tokens` as the total, with cached and written tokens as subsets of it. That is the opposite of Anthropic's convention, so the subtraction is done here once and the raw fields are kept alongside it.
    """
    u = response.usage
    d = u.input_tokens_details
    cached = getattr(d, "cached_tokens", 0) or 0
    written = getattr(d, "cache_write_tokens", 0) or 0
    return {
        "input_tokens": u.input_tokens,
        "cached_tokens": cached,
        "cache_write_tokens": written,
        "uncached_tokens": u.input_tokens - cached - written,
        "output_tokens": u.output_tokens,
    }


def billed_units(usage, model):
    """What the input side of one request costs, in units of the uncached input rate.

    A request that cached nothing costs exactly its token count. Everything below 1.0 per token is a saving and everything above it is a surcharge.
    """
    r = rates_for(model)
    return (
        usage["uncached_tokens"]
        + usage["cached_tokens"] * r["read"]
        + usage["cache_write_tokens"] * r["write"]
    )


def baseline_units(usage):
    """The same request with caching doing nothing at all: every token at full rate."""
    return usage["input_tokens"]


def ask(client, model, instructions, user_input, cache_key=None):
    """One request, with the token accounting pulled out of it."""
    kwargs = dict(
        model=model,
        instructions=instructions,
        input=user_input,
        max_output_tokens=64,
    )
    # Reasoning models spend output tokens before answering, and effort is part
    # of the cache key, so it is pinned rather than left to the default.
    if model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": "low"}
    if cache_key:
        kwargs["prompt_cache_key"] = cache_key

    # A response occasionally comes back reporting no tokens at all. That is not
    # a measurement of anything, so it is discarded and retried rather than
    # averaged into a table.
    for attempt in range(3):
        t0 = time.perf_counter()
        response = client.responses.create(**kwargs)
        elapsed = time.perf_counter() - t0
        u = usage_of(response)
        if u["input_tokens"] > 0:
            u["seconds"] = elapsed
            # The caller needs to know: the discarded attempt still warmed the
            # cache, so this row's prefix has been seen before and a hit here
            # would not have been earned by the experiment.
            u["retried"] = attempt > 0
            return u
        logger.warning(f"empty usage on {model}, attempt {attempt + 1}, retrying")
        time.sleep(1)
    raise RuntimeError(f"{model} returned empty usage three times running")


def check_identity(usage, expected_prompt_tokens, tolerance=64):
    """The parts must add up to the whole.

    `uncached + cached + written` should equal the prompt we sent, give or take the hidden tokens the API adds. If this fails the measurement is wrong and nothing derived from it should be quoted.
    """
    total = usage["uncached_tokens"] + usage["cached_tokens"] + usage["cache_write_tokens"]
    drift = abs(total - expected_prompt_tokens)
    ok = usage["uncached_tokens"] >= 0 and drift <= max(tolerance, expected_prompt_tokens * 0.05)
    if not ok:
        logger.warning(
            f"accounting identity failed: parts sum to {total}, prompt measured {expected_prompt_tokens}, drift {drift}"
        )
    return ok


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"wrote {path.relative_to(HERE)}")


def client_or_exit():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. These two scripts call the API and spend money.")
    from openai import OpenAI

    return OpenAI()
