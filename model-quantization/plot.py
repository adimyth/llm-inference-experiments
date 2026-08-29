"""Charts for results/results.json: every quantized checkpoint against fp16.

One chart per metric rather than one cramped grid - each gets its own figure, sized wide enough that six checkpoint labels sit flat without rotating or overlapping. Light and dark rendered separately, not one image inverted, per the repo's convention (see speculative-decoding's spec_plot.py).

Reads results.json generically by checkpoint label, so this stays useful as more methods (HQQ, NF4, AWQ, GPTQ) get added later - a chart just silently skips any checkpoint missing that particular metric rather than erroring, since methods finish their benchmark passes at different times.
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

# Preferred display order; anything in results.json but not listed here is appended at the end so a new method never gets silently dropped.
LABEL_ORDER = ["fp16", "rtn-q4_0", "q4_k_m", "q5_k_m", "q8_0", "mlx-q4", "hqq-q4",
               "awq-q4", "gptq-q4"]

# Control runs, not checkpoints. Each of these re-measures the *unquantized* model through one of the non-llama.cpp engines, to prove that engine's perplexity implementation agrees with the others before any quantized number from it is trusted. `fp16-torch` covers the PyTorch loop on MPS, `fp16-cuda` the same loop on CUDA (and it is also the only valid speed baseline for AWQ and GPTQ, measured on a different machine), `fp16-mlx` the MLX loop, which is a genuinely separate framework and was the last one still unvalidated. They belong in results.json and in the README, not as bars alongside the quantized checkpoints.
CONTROL_LABELS = {"fp16-torch", "fp16-cuda", "fp16-mlx"}
DISPLAY_NAME = {
    "fp16": "fp16",
    "rtn-q4_0": "RTN Q4_0",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
    "mlx-q4": "MLX 4-bit",
    "hqq-q4": "HQQ 4-bit",
    "awq-q4": "AWQ 4-bit",
    "gptq-q4": "GPTQ 4-bit",
}


# Label placement on the size-vs-quality scatter. Most labels sit up and to the right of their point; these few would collide with a neighbour there. RTN (4.34 GB, 7.804) and HQQ (5.61 GB, 7.815) are close enough in perplexity that their labels overlap, so RTN's goes down and to the left.
LABEL_NUDGE = {
    "rtn-q4_0": (0, -22, "center"),
    "q8_0": (12, -16, "left"),
    # AWQ (5.35, 7.792) and HQQ (5.61, 7.815) are close on both axes, so AWQ's label
    # goes down and to the right: directly below would crowd RTN's, which also sits low. GPTQ (5.33, 7.959) and MLX (4.22,
    # 7.949) are nearly level, so MLX's label goes out to the left instead of running
    # into GPTQ's point.
    "awq-q4": (13, -16, "left"),
    "mlx-q4": (-10, 8, "right"),
}

def ordered_labels(res):
    known = [l for l in LABEL_ORDER if l in res]
    rest = [l for l in res if l not in LABEL_ORDER and l not in CONTROL_LABELS]
    return known + rest


def style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=12)


def bar_chart(res, theme, metric_key, extract, colour_key, fmt, title, subtitle, filename, exclude=(), zero_base=True):
    t = THEMES[theme]
    labels = [l for l in ordered_labels(res) if metric_key in res[l] and l not in exclude]
    values = [extract(res[l]) for l in labels]

    # Width scales with the number of bars. At a fixed 9.5in the ninth method made
    # "AWQ 4-bit" and "GPTQ 4-bit" run into each other, so adding a method silently
    # degraded every chart it appeared in. Growing the canvas instead keeps the bars
    # and type at a constant size however many methods land here later.
    fig_w = max(9.5, 1.15 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6), facecolor=t["surface"])
    style(ax, t)
    xs = range(len(labels))
    ax.bar(list(xs), values, width=0.55, color=t[colour_key],
           edgecolor=t["surface"], linewidth=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([DISPLAY_NAME.get(l, l) for l in labels],
                        color=t["ink"], fontsize=13)
    if zero_base:
        ax.set_ylim(0, max(values) * 1.18)
    else:
        # Once the perplexity numbers were measured on one consistent convention they landed between 7.39 and 7.95, and a zero-based axis renders seven near-identical bars. The axis starts below the lowest value instead, and the subtitle says so. Exact values are annotated on every bar either way.
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.35 or 0.1
        ax.set_ylim(lo - pad, hi + pad * 1.3)
    for x, y in zip(xs, values):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=t["ink"],
                    fontsize=13, fontweight="bold")

    fig.suptitle(title, color=t["ink"], fontsize=16, ha="left", x=0.07, y=0.99)
    ax.set_title(subtitle, color=t["muted"], fontsize=11, loc="left", pad=14)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = OUT / f"{filename}_{theme}.png"
    fig.savefig(p, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return p


def size_chart(res, theme):
    return bar_chart(res, theme, "size_gb", lambda r: r["size_gb"], "s1",
                      "{:.1f} GB", "Size on disk",
                      "Llama 3.1 8B Instruct, each checkpoint against the fp16 baseline",
                      "size")


def perplexity_chart(res, theme):
    return bar_chart(res, theme, "perplexity", lambda r: r["perplexity"]["ppl"], "s2",
                      "{:.3f}", "Perplexity, wikitext-2",
                      "Lower is better. Full test split, one windowing convention. Truncated axis",
                      "perplexity", zero_base=False)


def mmlu_chart(res, theme):
    return bar_chart(res, theme, "mmlu", lambda r: r["mmlu"]["accuracy"] * 100, "s3",
                      "{:.0f}%", "MMLU accuracy",
                      "Fixed 50-question subset, identical questions for every checkpoint",
                      "mmlu")


def throughput_chart(res, theme):
    # Two kinds of exclusion here, for two different reasons.
    #
    # hqq-q4: HQQ has no fused MPS decode backend, so its generation speed reflects backend maturity rather than the quantization, about 200x off every other checkpoint, which would flatten this chart. Its number still lives in results.json and the tables.
    #
    # awq-q4 and gptq-q4: these ran on a rented NVIDIA GPU, every other checkpoint here ran on an M4 Pro. Plotting them as bars in the same axis would invite exactly the comparison that isn't valid. Their speed belongs in a table against the fp16-cuda control measured on that same GPU. Perplexity, MMLU and size are hardware-independent and do appear in the other charts.
    return bar_chart(res, theme, "throughput", lambda r: r["throughput"]["tg_tokens_per_sec"], "s4",
                      "{:.1f} t/s", "Generation speed",
                      "Tokens/sec, decode only (tg128), median of runs. Apple M4 Pro only",
                      "throughput", exclude={"hqq-q4", "awq-q4", "gptq-q4"})


def tradeoff(res, theme):
    """Size against perplexity: is a smaller checkpoint always a worse one?"""
    t = THEMES[theme]
    labels = [l for l in ordered_labels(res) if "size_gb" in res[l] and "perplexity" in res[l]]

    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor=t["surface"])
    style(ax, t)
    ax.grid(True, axis="both", color=t["grid"], linewidth=0.9, alpha=0.9)

    colours = [t["s1"], t["s2"], t["s3"], t["s4"], t["s2"], t["s1"], t["s4"]]
    for i, l in enumerate(labels):
        x = res[l]["size_gb"]
        y = res[l]["perplexity"]["ppl"]
        ax.scatter(x, y, s=180, color=colours[i % len(colours)],
                   edgecolor=t["surface"], linewidth=2, zorder=3)
        dx, dy, ha = LABEL_NUDGE.get(l, (12, 8, "left"))
        ax.annotate(DISPLAY_NAME.get(l, l), (x, y),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, color=t["ink"], fontsize=12)

    # a little breathing room so an offset label can't run into the y-axis
    xs_all = [res[l]["size_gb"] for l in labels]
    ax.set_xlim(min(xs_all) - 0.9, max(xs_all) + 1.4)

    ax.set_xlabel("Size on disk (GB)", color=t["muted"], fontsize=12)
    ax.set_ylabel("Perplexity, wikitext-2 (lower is better)", color=t["muted"], fontsize=12)
    fig.suptitle("Size against quality", color=t["ink"], fontsize=16, ha="left", x=0.08, y=0.99)
    ax.set_title("Every checkpoint's size on disk against its perplexity cost",
                 color=t["muted"], fontsize=11, loc="left", pad=14)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = OUT / f"tradeoff_{theme}.png"
    fig.savefig(p, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    results_raw = json.loads((OUT / "results.json").read_text())
    # For the local checkpoints, size isn't in results.json (it's derived from the file itself, not a benchmark run), so patch it in from the actual files on disk before plotting. quant_awq_gptq.py writes size_gb into results.json directly for AWQ and GPTQ, whose checkpoints only ever exist on the rented GPU box, so anything already carrying a size_gb is left alone here.
    models_dir = Path(__file__).parent / "models"
    size_map = {
        "fp16": "llama-3.1-8b-instruct-f16.gguf",
        "rtn-q4_0": "llama-3.1-8b-instruct-q4_0.gguf",
        "q4_k_m": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "q5_k_m": "llama-3.1-8b-instruct-q5_k_m.gguf",
        "q8_0": "llama-3.1-8b-instruct-q8_0.gguf",
        "mlx-q4": "llama-3.1-8b-instruct-mlx-q4",
        "hqq-q4": "llama-3.1-8b-instruct-hqq-q4",
    }
    for label, filename in size_map.items():
        if label not in results_raw:
            continue
        if "size_gb" in results_raw[label]:
            continue  # already recorded by the script that produced it
        path = models_dir / filename
        if not path.exists():
            continue
        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        results_raw[label]["size_gb"] = size_bytes / 1024**3

    for theme in THEMES:
        for fn in (size_chart, perplexity_chart, mmlu_chart, throughput_chart, tradeoff):
            print(fn(results_raw, theme))
