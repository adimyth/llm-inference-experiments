"""Tokens/sec for an HQQ checkpoint.

Same pp/tg split as quant_throughput.py and quant_mlx_throughput.py: prompt
processing (chewing through input context) timed separately from generation
(producing new tokens one at a time), so the numbers are comparable across
formats.

Every timed region calls `torch.mps.synchronize()` before stopping the
clock, per this repo's method notes - MPS queues work asynchronously, so
skipping this times how fast Python queued the work, not how long it took.
"""

import argparse
import time
from pathlib import Path
from statistics import median

import torch
from hqq.models.hf.base import AutoHQQHFModel
from loguru import logger
from transformers import AutoTokenizer


from results_io import update


def run(model_dir: Path, tokenizer_dir: str, n_prompt: int, n_gen: int, repeats: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoHQQHFModel.from_quantized(str(model_dir), compute_dtype=torch.float16, device="mps")
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me about the history of the Roman Empire. " * 40}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("mps")
    if input_ids.shape[1] > n_prompt:
        input_ids = input_ids[:, :n_prompt]
    n_prompt_actual = input_ids.shape[1]

    def one_run():
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids)
        torch.mps.synchronize()
        pp_time = time.perf_counter() - t0

        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=n_gen,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.mps.synchronize()
        gen_time = time.perf_counter() - t0
        n_generated = out.shape[1] - input_ids.shape[1]
        return n_prompt_actual / pp_time, n_generated / gen_time

    logger.info(f"warming up {model_dir.name}")
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
    }
    logger.info(f"pp: {stats['pp_tokens_per_sec']:.1f} t/s, tg: {stats['tg_tokens_per_sec']:.1f} t/s")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to an HQQ model directory")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="local HF snapshot dir or hub repo id to load the tokenizer from",
    )
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. hqq-q4")
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stats = run(Path(args.model), args.tokenizer, args.n_prompt, args.n_gen, args.repeats)
    update(args.label, "throughput", stats)
