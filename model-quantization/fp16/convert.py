"""HF checkpoint -> unquantized f16 GGUF.

This is the shared starting point every GGUF-based method in this folder quantizes from (RTN, k-quants). It shells out to llama.cpp's own convert_hf_to_gguf.py rather than reimplementing tensor layout and metadata mapping, which is not something worth hand-rolling.

That script needs its own environment: it pins torch==2.11.0 (CPU) and transformers==4.57.6, which conflicts with this repo's shared venv (torch 2.13.0, MPS-enabled, needed elsewhere for HF-side benchmarking). So it runs out of a separate venv, built once via:

git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/tools/llama.cpp cd ~/tools/llama.cpp uv venv --python 3.12 --python-preference only-managed .venv-convert uv pip install --python .venv-convert -r requirements/requirements-convert_hf_to_gguf.txt

Homebrew's llama.cpp bottle ships compiled binaries only, not this script, which is why the clone above is needed at all.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import subprocess
from pathlib import Path

from loguru import logger

LLAMA_CPP_SRC = Path.home() / "tools" / "llama.cpp"
CONVERT_PY = LLAMA_CPP_SRC / ".venv-convert" / "bin" / "python"
CONVERT_SCRIPT = LLAMA_CPP_SRC / "convert_hf_to_gguf.py"


def convert(hf_dir: Path, outfile: Path) -> None:
    if not CONVERT_PY.exists():
        raise FileNotFoundError(
            f"{CONVERT_PY} not found. Set up the conversion venv first, see this "
            "script's module docstring."
        )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CONVERT_PY),
        str(CONVERT_SCRIPT),
        "--outfile",
        str(outfile),
        "--outtype",
        "f16",
        str(hf_dir),
    ]
    logger.info(f"converting {hf_dir} -> {outfile}")
    subprocess.run(cmd, check=True)
    size_gb = outfile.stat().st_size / 1024**3
    logger.info(f"wrote {outfile} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="local HF snapshot directory (e.g. the resolved "
        "~/.cache/huggingface/hub/models--.../snapshots/<hash> path)",
    )
    parser.add_argument("--outfile", default="../models/llama-3.1-8b-instruct-f16.gguf")
    args = parser.parse_args()

    convert(Path(args.hf_dir), Path(args.outfile))
