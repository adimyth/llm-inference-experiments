"""Perplexity on wikitext-2 for a GGUF checkpoint, via llama-perplexity.

Lower is better: this measures how well the model predicts real held-out text, averaged over every token in the passage. It's the primary quality signal for comparing a quantized checkpoint against the fp16 baseline - run identically against both, so a difference is attributable to the quantization method and not to the eval itself.

llama-perplexity does the actual computation; this wraps it, parses its "Final estimate: PPL = X +/- Y" line, and records the result.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import re
import subprocess
import time
from pathlib import Path

from loguru import logger

from results_io import update

WIKITEXT = Path(__file__).resolve().parent.parent / "data" / "wikitext-2-raw-test.txt"
PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def run(model: Path, chunks: int | None) -> dict:
    if not WIKITEXT.exists():
        raise FileNotFoundError(f"{WIKITEXT} missing - run fetch_wikitext.py first")

    cmd = ["llama-perplexity", "-m", str(model), "-f", str(WIKITEXT)]
    if chunks:
        cmd += ["--chunks", str(chunks)]

    logger.info(f"running llama-perplexity on {model.name}" + (f" ({chunks} chunks)" if chunks else " (full file)"))
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - start

    match = PPL_RE.search(result.stdout + result.stderr)
    if not match:
        raise RuntimeError("could not find PPL in llama-perplexity output")
    ppl, err = float(match.group(1)), float(match.group(2))
    logger.info(f"PPL = {ppl:.4f} +/- {err:.4f} ({wall:.1f}s)")
    return {"ppl": ppl, "ppl_err": err, "wall_seconds": wall}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a GGUF checkpoint")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. fp16, rtn-q4_0")
    parser.add_argument("--chunks", type=int, default=None, help="limit chunks (default: whole file)")
    args = parser.parse_args()

    stats = run(Path(args.model), args.chunks)
    update(args.label, "perplexity", stats)
