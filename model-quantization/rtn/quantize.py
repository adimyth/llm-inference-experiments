"""RTN: quantize an f16 GGUF to Q4_0.

Q4_0 is llama.cpp's plain round-to-nearest quant type: each block of weights gets a single scale, and every weight rounds to the nearest representable value on that block's grid. No calibration data, no importance weighting. This is the naive baseline every other method in this folder is measured against.

llama-quantize does the actual work; this wrapper exists so the conversion is a checked-in, reproducible step rather than a one-off shell command, and so it reports the size reduction the essay quotes.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import shutil
import subprocess
from pathlib import Path

from loguru import logger

QUANT_TYPE = "Q4_0"


def quantize(infile: Path, outfile: Path) -> None:
    if shutil.which("llama-quantize") is None:
        raise FileNotFoundError(
            "llama-quantize not on PATH. Install via `brew install llama.cpp` "
            "(the arm64 Homebrew, /opt/homebrew, for Metal support)."
        )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["llama-quantize", str(infile), str(outfile), QUANT_TYPE]
    logger.info(f"quantizing {infile.name} -> {outfile.name} ({QUANT_TYPE})")
    subprocess.run(cmd, check=True)

    before = infile.stat().st_size / 1024**3
    after = outfile.stat().st_size / 1024**3
    logger.info(f"{before:.2f} GB -> {after:.2f} GB ({after / before:.1%} of original)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", default="../models/llama-3.1-8b-instruct-f16.gguf")
    parser.add_argument("--outfile", default="../models/llama-3.1-8b-instruct-q4_0.gguf")
    args = parser.parse_args()

    quantize(Path(args.infile), Path(args.outfile))
