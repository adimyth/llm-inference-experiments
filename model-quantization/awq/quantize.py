"""HF checkpoint -> AWQ (Activation-aware Weight Quantization), 4-bit. CUDA only.

Every other method in this project decides how to round by looking only at the weights. AWQ looks at what flows *through* them. It pushes calibration text through the model, measures the average activation magnitude on each input channel of every linear layer, and finds the small fraction of channels carrying the largest activations - the ones whose error does the most damage downstream. It then scales those channels up before rounding, so they land on a finer part of the 4-bit grid, and scales the corresponding weights down to compensate. The rounding itself is ordinary; what changes is which weights get the grid's precision spent on them. AWQ decides what to protect before it rounds anything.

TOOLING. AutoAWQ, the original implementation, is deprecated. Its AWQ support was adopted by the vLLM project as `llmcompressor`, with help from AutoAWQ's own maintainer, and that is what this uses. gptq/quantize.py uses the same library, so AWQ and GPTQ here share a calibration set, a sample count, a sequence length and an output format, and differ only in algorithm.

WHY CUDA ONLY. There is no Metal or MPS path. This is the reason the AWQ and GPTQ numbers in this project come from a rented NVIDIA box while everything else was measured on an M4 Pro. See ../README.md and ../setup_cuda.sh.

Output is a `compressed-tensors` checkpoint directory that plain `AutoModelForCausalLM` can load, which is what lets perplexity.py, mmlu.py and throughput.py in this folder score it with no format-specific code.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from results_io import update

# 4-bit weights, fp16 activations, group size 128. 128 is the default every shipped AWQ and GPTQ checkpoint uses, and deliberately NOT the 64 that MLX and HQQ use elsewhere in this project: "4-bit" alone doesn't specify a scheme, and group size is the parameter that usually goes unstated. Changing it means building a custom QuantizationScheme rather than naming one of the presets.
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
METHOD = "awq"


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def build_calibration_set(tokenizer, n_samples: int, max_seq_len: int):
    """Chat-formatted calibration text.

    Shuffled with a fixed seed, and identical to the set gptq/quantize.py and awq/quantize.py both build, so the two methods differ by their algorithm and not by the data they saw.
    """
    ds = load_dataset(CALIBRATION_DATASET, split=f"train_sft[:{n_samples}]")
    ds = ds.shuffle(seed=42)

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(to_text)
    return ds.map(tokenize, remove_columns=ds.column_names)


def recipe():
    # AWQModifier computes and applies the per-channel scales; the QuantizationModifier alongside it does the actual rounding. W4A16_ASYM (asymmetric, with a zero-point) is AWQ's documented scheme.
    return [
        AWQModifier(),
        QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head"]),
    ]


def quantize(hf_dir: str, out_dir: Path, n_samples: int, max_seq_len: int) -> None:
    logger.info(f"loading {hf_dir}")
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    logger.info(f"building calibration set: {n_samples} samples, max {max_seq_len} tokens")
    ds = build_calibration_set(tokenizer, n_samples, max_seq_len)

    logger.info(f"quantizing with {METHOD.upper()}, 4-bit, group size {GROUP_SIZE}")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe(),
        max_seq_length=max_seq_len,
        num_calibration_samples=n_samples,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), save_compressed=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"wrote {out_dir} ({dir_size_gb(out_dir):.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, help="local HF snapshot dir or hub repo id for the unquantized model")
    parser.add_argument("--out-dir", default=f"../models/llama-3.1-8b-instruct-{METHOD}-q4")
    parser.add_argument("--label", default=f"{METHOD}-q4", help="results.json label")
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQUENCE_LENGTH)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    quantize(args.hf_dir, out_dir, args.num_samples, args.max_seq_len)

    # Written straight into results.json rather than derived from the file later: these checkpoints live on a rented GPU box and are never downloaded, so plot.py reads their size from here instead.
    update(args.label, "size_gb", dir_size_gb(out_dir))
    update(args.label, "quantization", {
        "method": METHOD,
        "bits": 4,
        "group_size": GROUP_SIZE,
        "calibration_dataset": CALIBRATION_DATASET,
        "calibration_samples": args.num_samples,
        "max_seq_length": args.max_seq_len,
    })
