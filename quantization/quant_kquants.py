"""GGUF k-quants: quantize the f16 GGUF to Q4_K_M, Q5_K_M, Q8_0.

Unlike RTN's Q4_0 (one scale per 32-weight block, uniform treatment), the
k-quant types vary bits per tensor by how much each tensor's precision
tends to matter (attention vs FFN, early vs late layers), based on
llama.cpp's built-in heuristics rather than a calibration pass. No
importance matrix (--imatrix) is used here - that's a further refinement
using calibration text to weight the heuristic, out of scope for this
comparison but worth a note in the essay as the natural next lever.

Reuses the same f16 GGUF that RTN quantized from - built once, quantized
from three times.
"""

import argparse
import subprocess
from pathlib import Path

from loguru import logger

QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def quantize(infile: Path, outfile: Path, quant_type: str) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["llama-quantize", str(infile), str(outfile), quant_type]
    logger.info(f"quantizing {infile.name} -> {outfile.name} ({quant_type})")
    subprocess.run(cmd, check=True)

    before = infile.stat().st_size / 1024**3
    after = outfile.stat().st_size / 1024**3
    logger.info(f"{before:.2f} GB -> {after:.2f} GB ({after / before:.1%} of original)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", default="models/llama-3.1-8b-instruct-f16.gguf")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--types", nargs="+", default=QUANT_TYPES)
    args = parser.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    for qtype in args.types:
        outfile = out_dir / f"llama-3.1-8b-instruct-{qtype.lower()}.gguf"
        quantize(infile, outfile, qtype)
