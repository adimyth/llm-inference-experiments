"""Tokens/sec for any transformers-loadable checkpoint, on any device.

The fp16 control run. fp16/mmlu.py and fp16/throughput.py go through llama.cpp and only read GGUF, which is fine on the Mac and useless on a rented NVIDIA box. This is the transformers path, so the unquantized model can be measured on the same GPU as AWQ and GPTQ, with the same code that measures them.

That control is what makes the CUDA speed numbers mean anything: tokens/sec on an L40S cannot be read against an M4 Pro, only against fp16 on the same L40S.


The generic counterpart to quant_hqq_throughput.py. Same pp/tg split as every other throughput script here: prompt processing (chewing through input context) timed separately from generation (producing new tokens one at a time), warmup pass discarded, median of repeated runs.

Synchronisation goes through torch_device.sync rather than a hardcoded torch.mps.synchronize(), because both MPS and CUDA queue work asynchronously and a timed region without the right sync call measures how fast Python queued the work rather than how long the GPU took.

Numbers from this script on CUDA are NOT comparable to the M4 Pro numbers elsewhere in results.json. They are only meaningful against the fp16 control run measured on the same GPU.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import time
from statistics import median

import torch
from loguru import logger
from transformers import AutoTokenizer

import torch_device
from results_io import update


def run(model_id: str, tokenizer_id: str, device: str, n_prompt: int, n_gen: int, repeats: int, dtype: str) -> dict:
    logger.info(f"loading {model_id} on {torch_device.describe(device)}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # Placement and dtype differ by backend; torch_device.load_causal_lm owns that branch so all six benchmark scripts load a checkpoint the same way.
    model = torch_device.load_causal_lm(model_id, device, dtype)
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me about the history of the Roman Empire. " * 40}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] > n_prompt:
        input_ids = input_ids[:, :n_prompt]
    n_prompt_actual = input_ids.shape[1]

    def one_run():
        torch_device.sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids)
        torch_device.sync(device)
        pp_time = time.perf_counter() - t0

        torch_device.sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=n_gen,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch_device.sync(device)
        gen_time = time.perf_counter() - t0
        n_generated = out.shape[1] - input_ids.shape[1]
        return n_prompt_actual / pp_time, n_generated / gen_time

    logger.info("warming up")
    one_run()

    logger.info(f"running {repeats} timed passes (pp{n_prompt_actual}, tg{n_gen})")
    pp_runs, tg_runs = [], []
    for _ in range(repeats):
        pp_tps, tg_tps = one_run()
        pp_runs.append(pp_tps)
        tg_runs.append(tg_tps)

    stats = {
        "pp_tokens_per_sec": median(pp_runs),
        "tg_tokens_per_sec": median(tg_runs),
        "device": device,
    }
    logger.info(f"pp: {stats['pp_tokens_per_sec']:.1f} t/s, tg: {stats['tg_tokens_per_sec']:.1f} t/s")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint dir or hub repo id")
    parser.add_argument("--tokenizer", help="local HF snapshot dir or hub repo id; defaults to --model")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. awq-q4")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16"],
                        help="float16 is what the published fp16 baseline used")
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stats = run(
        args.model,
        args.tokenizer or args.model,
        torch_device.resolve(args.device),
        args.n_prompt,
        args.n_gen,
        args.repeats,
        args.dtype,
    )
    update(args.label, "throughput", stats)
