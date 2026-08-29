"""HF checkpoint -> GPTQ, 4-bit. CUDA only.

GPTQ quantizes a weight matrix one column at a time, and after each column it adjusts the columns it has NOT yet quantized to compensate for the error it just introduced. The adjustment is derived from a Hessian estimated from calibration text, which tells it how sensitive the layer's output is to each weight, so the correction is spent where it matters.

The contrast with AWQ is the clean one, and it is why both are in this project: AWQ decides what to protect *before* rounding, by looking at activations. GPTQ rounds greedily and *repairs* after each step. Same calibration data here, same bit width, same group size, different strategy.

TOOLING. AutoGPTQ was archived in April 2025 and transformers removed its GPTQ backend entirely. This uses `llmcompressor` (the vLLM project's library), which is also what awq/quantize.py uses, so the two methods share a calibration set, a sample count, a sequence length and an output format.

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
from llmcompressor.modifiers.gptq import GPTQModifier
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from results_io import update

# 4-bit weights, fp16 activations, group size 128. 128 is the default every shipped AWQ and GPTQ checkpoint uses, and deliberately NOT the 64 that MLX and HQQ use elsewhere in this project: "4-bit" alone doesn't specify a scheme, and group size is the parameter that usually goes unstated. Changing it means building a custom QuantizationScheme rather than naming one of the presets.
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
METHOD = "gptq"


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
    # GPTQModifier does both the error-compensating solve and the rounding, so unlike AWQ it needs no separate QuantizationModifier.
    return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])


def quantize(hf_dir: str, out_dir: Path, n_samples: int, max_seq_len: int) -> None:
    logger.info(f"loading {hf_dir}")
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    logger.info(f"building calibration set: {n_samples} samples, max {max_seq_len} tokens")
    ds = build_calibration_set(tokenizer, n_samples, max_seq_len)

    logger.info(f"quantizing with {METHOD.upper()}, 4-bit, group size {GROUP_SIZE}")
    # oneshot returns the calibrated model (`return one_shot.model`). Bind it rather than relying on the passed-in object having been mutated in place, so a future change there can't have us silently saving unquantized weights.
    model = oneshot(
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
