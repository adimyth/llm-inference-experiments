"""Charts for results/results.json.

Speedup, acceptance and discarded tokens are three different scales, so they get three stacked panels sharing the k axis rather than one chart with two y-axes. Light and dark are separate renders, not an inverted image.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  grid="#e6e5e1", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 grid="#33322f", s1="#3987e5", s2="#d95926", s3="#199e70"),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)


def k_sweep(res, theme):
    t = THEMES[theme]
    ks = [r["k"] for r in res["k_sweep"]]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True,
                             facecolor=t["surface"])
    panels = [
        ("speedup", [r["speedup"] for r in res["k_sweep"]], t["s1"],
         "Speedup vs baseline", "{:.2f}x"),
        ("acceptance", [r["acceptance"] * 100 for r in res["k_sweep"]], t["s2"],
         "Draft tokens accepted (%)", "{:.0f}%"),
        ("discarded", [r["discarded"] for r in res["k_sweep"]], t["s3"],
         "Draft tokens thrown away", "{:.0f}"),
    ]
    peak = max(res["k_sweep"], key=lambda r: r["speedup"])
    for ax, (_, ys, colour, label, fmt) in zip(axes, panels):
        style(ax, t)
        ax.plot(ks, ys, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2)
        ax.set_ylabel(label, color=t["muted"], fontsize=9)
        for x, y in ((ks[0], ys[0]), (ks[-1], ys[-1])):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", color=t["ink"], fontsize=9)
    axes[0].axvline(peak["k"], color=t["muted"], linewidth=1, linestyle=":")
    axes[0].annotate(f"best k={peak['k']}\n{peak['speedup']:.2f}x",
                     (peak["k"], peak["speedup"]), textcoords="offset points",
                     xytext=(-4, -40), ha="right", color=t["ink"], fontsize=9,
                     fontweight="bold")
    axes[0].axhline(1.0, color=t["muted"], linewidth=1, linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("k, draft tokens per round", color=t["muted"], fontsize=9)
    axes[-1].set_xticks(ks)
    fig.suptitle(f"More lookahead is not better\n{res['target'].split('/')[-1]} "
                 f"drafted by {res['draft'].split('/')[-1]}, {res['device']}, "
                 f"median of {res['repeats']}",
                 color=t["ink"], fontsize=12, ha="left", x=0.09, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f"k_sweep_{theme}.png"
    fig.savefig(p, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return p


def workloads(res, theme):
    t = THEMES[theme]
    rows = sorted(res["workloads"], key=lambda r: r["speedup"])
    names = [r["workload"] for r in rows]
    ys = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=t["surface"])
    style(ax, t)
    ax.barh(list(ys), [r["speedup"] for r in rows], height=0.62,
            color=t["s1"], edgecolor=t["surface"], linewidth=2)
    ax.axvline(1.0, color=t["muted"], linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, color=t["ink"], fontsize=10)
    ax.set_xlabel("Speedup vs baseline", color=t["muted"], fontsize=9)
    for y, r in zip(ys, rows):
        ax.annotate(f"{r['speedup']:.2f}x   {r['acceptance']:.0%} accepted",
                    (r["speedup"], y), textcoords="offset points", xytext=(8, -3),
                    color=t["ink"], fontsize=9, va="center")
    ax.set_xlim(0, max(r["speedup"] for r in rows) * 1.45)
    fig.suptitle(f"Speedup follows how predictable the text is\nk={rows[0]['k']}, "
                 f"{res['device']}, median of {res['repeats']}",
                 color=t["ink"], fontsize=12, ha="left", x=0.02, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p = OUT / f"workloads_{theme}.png"
    fig.savefig(p, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    res = json.loads((OUT / "results.json").read_text())
    for theme in THEMES:
        print(k_sweep(res, theme), workloads(res, theme))
