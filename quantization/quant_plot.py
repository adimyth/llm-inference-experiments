"""Charts for results/results.json: every quantized checkpoint against fp16.

One multi-panel figure per the repo's convention (see speculative-decoding's
spec_plot.py): light and dark rendered separately, not one image inverted.

Reads results.json generically by checkpoint label, so this stays useful as
more methods (HQQ, NF4, AWQ, GPTQ) get added later - a panel just silently
skips any checkpoint missing that particular metric rather than erroring,
since methods finish their benchmark passes at different times.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  grid="#e6e5e1", s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#8855d9"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 grid="#33322f", s1="#3987e5", s2="#d95926", s3="#199e70", s4="#a374e8"),
}

# Preferred display order; anything in results.json but not listed here is
# appended at the end so a new method never gets silently dropped.
LABEL_ORDER = ["fp16", "rtn-q4_0", "q4_k_m", "q5_k_m", "q8_0", "mlx-q4"]
DISPLAY_NAME = {
    "fp16": "fp16",
    "rtn-q4_0": "RTN\nQ4_0",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
    "mlx-q4": "MLX\n4-bit",
}


def ordered_labels(res):
    known = [l for l in LABEL_ORDER if l in res]
    rest = [l for l in res if l not in LABEL_ORDER]
    return known + rest


def style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)


def bars(ax, t, labels, values, colour, fmt, ylabel):
    xs = range(len(labels))
    ax.bar(list(xs), values, width=0.6, color=colour, edgecolor=t["surface"], linewidth=1.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([DISPLAY_NAME.get(l, l) for l in labels], color=t["ink"], fontsize=8.5)
    ax.set_ylabel(ylabel, color=t["muted"], fontsize=9)
    for x, y in zip(xs, values):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=t["ink"], fontsize=8.5)


def headline(res, theme):
    t = THEMES[theme]
    labels = ordered_labels(res)

    size_labels = [l for l in labels if "size_gb" in res[l]]
    ppl_labels = [l for l in labels if "perplexity" in res[l]]
    mmlu_labels = [l for l in labels if "mmlu" in res[l]]
    tg_labels = [l for l in labels if "throughput" in res[l]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor=t["surface"])
    (ax_size, ax_ppl), (ax_mmlu, ax_tg) = axes

    if size_labels:
        style(ax_size, t)
        bars(ax_size, t, size_labels, [res[l]["size_gb"] for l in size_labels],
             t["s1"], "{:.1f} GB", "Size on disk")

    if ppl_labels:
        style(ax_ppl, t)
        bars(ax_ppl, t, ppl_labels, [res[l]["perplexity"]["ppl"] for l in ppl_labels],
             t["s2"], "{:.2f}", "Perplexity, wikitext-2 (lower is better)")

    if mmlu_labels:
        style(ax_mmlu, t)
        bars(ax_mmlu, t, mmlu_labels, [res[l]["mmlu"]["accuracy"] * 100 for l in mmlu_labels],
             t["s3"], "{:.0f}%", "MMLU accuracy, 50q subset")

    if tg_labels:
        style(ax_tg, t)
        bars(ax_tg, t, tg_labels, [res[l]["throughput"]["tg_tokens_per_sec"] for l in tg_labels],
             t["s4"], "{:.1f}", "Generation speed (tokens/sec)")

    fig.suptitle("Post-training quantization: Llama 3.1 8B Instruct\n"
                 "size, quality, and speed against the fp16 baseline",
                 color=t["ink"], fontsize=13, ha="left", x=0.06, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / f"headline_{theme}.png"
    fig.savefig(p, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    results_raw = json.loads((OUT / "results.json").read_text())
    # size isn't in results.json (it's derived from the file itself, not a
    # benchmark run), so patch it in from the actual files on disk before plotting.
    models_dir = Path(__file__).parent / "models"
    size_map = {
        "fp16": "llama-3.1-8b-instruct-f16.gguf",
        "rtn-q4_0": "llama-3.1-8b-instruct-q4_0.gguf",
        "q4_k_m": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "q5_k_m": "llama-3.1-8b-instruct-q5_k_m.gguf",
        "q8_0": "llama-3.1-8b-instruct-q8_0.gguf",
        "mlx-q4": "llama-3.1-8b-instruct-mlx-q4",
    }
    for label, filename in size_map.items():
        if label not in results_raw:
            continue
        path = models_dir / filename
        if not path.exists():
            continue
        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        results_raw[label]["size_gb"] = size_bytes / 1024**3

    for theme in THEMES:
        print(headline(results_raw, theme))
