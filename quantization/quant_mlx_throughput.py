"""Tokens/sec for an MLX checkpoint.

Mirrors quant_throughput.py's pp/tg split so the numbers are comparable
across formats: prompt processing (how fast the model chews through input
context) and generation (how fast it produces new tokens one at a time).

mlx_lm.stream_generate reports both directly per GenerationResponse (the
final one carries the run's aggregate prompt_tps/generation_tps), so this
just repeats it with a warmup pass and takes the median, matching
llama-bench's own warmup+repeat discipline.
"""

import argparse
from pathlib import Path
from statistics import median

from loguru import logger
from mlx_lm import load, stream_generate

from results_io import update


def run(model_path: Path, n_prompt: int, n_gen: int, repeats: int) -> dict:
    model, tokenizer = load(str(model_path))
    # a long-ish filler prompt so the pp measurement has n_prompt tokens to chew through
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me about the history of the Roman Empire. " * 40}],
        add_generation_prompt=True,
    )
    prompt = prompt[:n_prompt] if len(prompt) > n_prompt else prompt

    def one_run():
        last = None
        for last in stream_generate(model, tokenizer, prompt, max_tokens=n_gen):
            pass
        return last

    logger.info(f"warming up {model_path.name}")
    one_run()

    logger.info(f"running {repeats} timed passes (pp{len(prompt)}, tg{n_gen})")
    pp_runs, tg_runs = [], []
    for _ in range(repeats):
        r = one_run()
        pp_runs.append(r.prompt_tps)
        tg_runs.append(r.generation_tps)

    stats = {
        "pp_tokens_per_sec": median(pp_runs),
        "tg_tokens_per_sec": median(tg_runs),
    }
    logger.info(f"pp: {stats['pp_tokens_per_sec']:.1f} t/s, tg: {stats['tg_tokens_per_sec']:.1f} t/s")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to an MLX model directory")
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. mlx-q4")
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stats = run(Path(args.model), args.n_prompt, args.n_gen, args.repeats)
    update(args.label, "throughput", stats)
