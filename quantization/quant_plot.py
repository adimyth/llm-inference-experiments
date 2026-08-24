"""Charts for results/results.json: every quantized checkpoint against fp16.

One chart per metric rather than one cramped grid - each gets its own
figure, sized wide enough that six checkpoint labels sit flat without
rotating or overlapping. Light and dark rendered separately, not one image
inverted, per the repo's convention (see speculative-decoding's spec_plot.py).

Reads results.json generically by checkpoint label, so this stays useful as
more methods (HQQ, NF4, AWQ, GPTQ) get added later - a chart just silently
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
    "rtn-q4_0": "RTN Q4_0",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
    "mlx-q4": "MLX 4-bit",
}


def ordered_labels(res):
    known = [l for l in LABEL_ORDER if l in res]
    rest = [l for l in res if l not in LABEL_ORDER]
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


def bar_chart(res, theme, metric_key, extract, colour_key, fmt, title, subtitle, filename):
    t = THEMES[theme]
    labels = [l for l in ordered_labels(res) if metric_key in res[l]]
    values = [extract(res[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=t["surface"])
    style(ax, t)
    xs = range(len(labels))
    ax.bar(list(xs), values, width=0.55, color=t[colour_key],
           edgecolor=t["surface"], linewidth=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([DISPLAY_NAME.get(l, l) for l in labels],
                        color=t["ink"], fontsize=13)
    ax.set_ylim(0, max(values) * 1.18)
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
                      "{:.2f}", "Perplexity, wikitext-2",
                      "Lower is better. Full test split, same windowing across every checkpoint",
                      "perplexity")


def mmlu_chart(res, theme):
    return bar_chart(res, theme, "mmlu", lambda r: r["mmlu"]["accuracy"] * 100, "s3",
                      "{:.0f}%", "MMLU accuracy",
                      "Fixed 50-question subset, identical questions for every checkpoint",
                      "mmlu")


def throughput_chart(res, theme):
    return bar_chart(res, theme, "throughput", lambda r: r["throughput"]["tg_tokens_per_sec"], "s4",
                      "{:.1f} t/s", "Generation speed",
                      "Tokens/sec, decode only (tg128), median of repeated runs",
                      "throughput")


def tradeoff(res, theme):
    """Size against perplexity: is a smaller checkpoint always a worse one?"""
    t = THEMES[theme]
    labels = [l for l in ordered_labels(res) if "size_gb" in res[l] and "perplexity" in res[l]]

    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor=t["surface"])
    style(ax, t)
    ax.grid(True, axis="both", color=t["grid"], linewidth=0.9, alpha=0.9)

    colours = [t["s1"], t["s2"], t["s3"], t["s4"], t["s2"], t["s1"]]
    for i, l in enumerate(labels):
        x = res[l]["size_gb"]
        y = res[l]["perplexity"]["ppl"]
        ax.scatter(x, y, s=180, color=colours[i % len(colours)],
                   edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.annotate(DISPLAY_NAME.get(l, l), (x, y),
                    textcoords="offset points", xytext=(12, 8),
                    color=t["ink"], fontsize=12)

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
        for fn in (size_chart, perplexity_chart, mmlu_chart, throughput_chart, tradeoff):
            print(fn(results_raw, theme))
