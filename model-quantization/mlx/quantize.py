"""HF checkpoint -> MLX, quantized.

MLX is Apple's own array framework, built around unified memory on M-series chips. Its quantization is affine (per-group scale + zero-point, similar spirit to GGUF's k-quants) and, unlike GGUF, MLX conversion and inference share one framework end to end rather than a separate convert step feeding a separate C++ runtime.

Unlike quant_rtn.py, this doesn't need a separate f16 GGUF as input - mlx_lm converts straight from the HF checkpoint, quantizing in the same pass.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def convert(hf_dir: Path, mlx_path: Path, bits: int, group_size: int) -> None:
    # mlx_lm.convert refuses to write into a directory that already exists, even an empty one - it creates mlx_path itself, so don't pre-create it. `python -m mlx_lm convert ...` (subcommand form) rather than the `mlx_lm.convert` console script, which needs the venv's bin/ on PATH - not guaranteed when invoked as `../.venv/bin/python quant_mlx.py` without activating the venv.
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(hf_dir),
        "--mlx-path", str(mlx_path),
        "-q",
        "--q-bits", str(bits),
        "--q-group-size", str(group_size),
    ]
    logger.info(f"converting {hf_dir} -> {mlx_path} ({bits}-bit, group size {group_size})")
    subprocess.run(cmd, check=True)
    logger.info(f"wrote {mlx_path} ({dir_size_gb(mlx_path):.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="local HF snapshot directory or a HF Hub repo id",
    )
    parser.add_argument("--mlx-path", default="../models/llama-3.1-8b-instruct-mlx-q4")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    convert(Path(args.hf_dir), Path(args.mlx_path), args.bits, args.group_size)
