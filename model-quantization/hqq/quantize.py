"""HF checkpoint -> HQQ, quantized.

HQQ (Half-Quadratic Quantization) is calibration-free like RTN, but instead of a single min/max-derived scale per block it solves a small closed-form optimization per group to pick the scale and zero-point that minimize quantization error - still no calibration data, just a better fit than round-to-nearest.

Runs through `transformers` + `torch`, not a purpose-built inference engine like llama.cpp or MLX, so this is the slowest of the local methods to both quantize and run.

Uses `hqq`'s own `AutoHQQHFModel.quantize_model` / `save_quantized`, not transformers' generic `quantization_config=HqqConfig(...)` path: on this transformers version (5.15.1) reloading a HQQ model saved via the generic `save_pretrained`/`from_pretrained` path raises `NotImplementedError: QuantizationMethod.HQQ is not available yet` - HQQ's own save/load API sidesteps that entirely.

`device='mps'` has to be passed explicitly to both `quantize_model` and (in quant_hqq_*.py) `from_quantized` - the default is `device='cuda'`, and HQQ silently tries to restore CUDA-mapped tensors on load if you don't override it, which fails on a Mac with no CUDA. This is the "CUDA-first" caveat the plan called out, not a full blocker - it just needs the device named on every call.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import torch
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel
from loguru import logger
from transformers import AutoModelForCausalLM

NBITS = 4
GROUP_SIZE = 64  # matches MLX's default, so the two "4-bit" methods are comparable


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def quantize(hf_dir: Path, out_dir: Path) -> None:
    logger.info(f"loading {hf_dir}")
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float16)

    qcfg = BaseQuantizeConfig(nbits=NBITS, group_size=GROUP_SIZE)
    logger.info(f"quantizing to HQQ {NBITS}-bit, group size {GROUP_SIZE}")
    AutoHQQHFModel.quantize_model(model, quant_config=qcfg, compute_dtype=torch.float16, device="mps")

    out_dir.mkdir(parents=True, exist_ok=True)
    AutoHQQHFModel.save_quantized(model, str(out_dir))
    logger.info(f"wrote {out_dir} ({dir_size_gb(out_dir):.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="local HF snapshot directory (e.g. the resolved "
        "~/.cache/huggingface/hub/models--.../snapshots/<hash> path)",
    )
    parser.add_argument("--out-dir", default="../models/llama-3.1-8b-instruct-hqq-q4")
    args = parser.parse_args()

    quantize(Path(args.hf_dir), Path(args.out_dir))
