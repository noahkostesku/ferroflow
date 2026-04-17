"""
Generate ferroflow strong-scaling plots from docs/scaling_results.json.

Outputs:
  docs/scaling_throughput.png  — throughput vs nodes (4 series + ideal)
  docs/scaling_efficiency.png  — parallel efficiency vs nodes (WS series only)
  docs/steal_rate.png          — steal rate vs nodes (WS series only)
"""

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path(__file__).parent.parent / "docs" / "data" / "scaling_results.json"
DOCS    = pathlib.Path(__file__).parent.parent / "docs" / "plots"

DARK_BG  = "#1c1c1e"
GRID_COL = "#3a3a3c"

COLORS = {
    ("transformer", "static"):        "#4fc3f7",  # light blue
    ("transformer", "work-stealing"): "#0288d1",  # mid blue
    ("wide",        "static"):        "#f48fb1",  # light pink
    ("wide",        "work-stealing"): "#c2185b",  # deep pink
}
LABELS = {
    ("transformer", "static"):        "transformer / static",
    ("transformer", "work-stealing"): "transformer / WS",
    ("wide",        "static"):        "wide-skewed / static",
    ("wide",        "work-stealing"): "wide-skewed / WS",
}

def load(path):
    with open(path) as f:
        return json.load(f)

def series(data, dag, scheduler):
    rows = sorted(
        [r for r in data if r["dag"] == dag and r["scheduler"] == scheduler],
        key=lambda r: r["nodes"],
    )
    nodes        = [r["nodes"]       for r in rows]
    throughput   = [r["throughput"]  for r in rows]
    efficiency   = [r["efficiency"]  for r in rows]
    steal_rate   = [r["steal_rate"]  for r in rows]
    return nodes, throughput, efficiency, steal_rate

def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color(GRID_COL)
    ax.spines["left"].set_color(GRID_COL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.6, alpha=0.7)

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 1 — Throughput
# ---------------------------------------------------------------------------
def plot_throughput(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for (dag, sched), color in COLORS.items():
        nodes, tput, _, _ = series(data, dag, sched)
        ax.plot(nodes, tput, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, sched)])

    # ideal linear scaling anchored at the mean 2-node throughput
    base_values = [r["throughput"] for r in data if r["nodes"] == min(all_nodes)]
    base = np.mean(base_values)
    base_nodes = min(all_nodes)
    ideal_nodes = np.array([base_nodes, max(all_nodes)])
    ideal_tput  = base * (ideal_nodes / base_nodes)
    ax.plot(ideal_nodes, ideal_tput, linestyle=":", linewidth=1.5,
            color="#aaaaaa", label="ideal (linear)", alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Throughput (ops/s)")
    ax.set_title("ferroflow Strong Scaling — Throughput")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, DOCS / "scaling_throughput.png")

# ---------------------------------------------------------------------------
# Plot 2 — Parallel efficiency (WS only)
# ---------------------------------------------------------------------------
def plot_efficiency(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    ws_keys = [k for k in COLORS if k[1] == "work-stealing"]

    for (dag, sched) in ws_keys:
        color = COLORS[(dag, sched)]
        nodes, _, eff, _ = series(data, dag, sched)
        ax.plot(nodes, eff, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, sched)])

        # annotate the first node where efficiency drops below 0.7
        for i, (n, e) in enumerate(zip(nodes, eff)):
            if e < 0.7:
                ax.annotate(
                    f"eff={e:.2f}\n(n={n})",
                    xy=(n, e), xytext=(n * 1.1, e + 0.06),
                    color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                )
                break

    ax.axhline(0.7, linestyle="--", linewidth=1.4, color="#ffcc02",
               label="0.70 threshold", alpha=0.85)

    all_nodes = sorted({r["nodes"] for r in data})
    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Parallel Efficiency")
    ax.set_ylim(0, 1.15)
    ax.set_title("ferroflow Strong Scaling — Parallel Efficiency")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, DOCS / "scaling_efficiency.png")

# ---------------------------------------------------------------------------
# Plot 3 — Steal rate (WS only)
# ---------------------------------------------------------------------------
def plot_steal_rate(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    ws_keys = [k for k in COLORS if k[1] == "work-stealing"]

    all_zero = all(r["steal_rate"] == 0.0 for r in data)

    for (dag, sched) in ws_keys:
        color = COLORS[(dag, sched)]
        nodes, _, _, steal = series(data, dag, sched)
        ax.plot(nodes, steal, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, sched)])

    all_nodes = sorted({r["nodes"] for r in data})
    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Steals / sec")
    ax.set_title("ferroflow Steal Rate vs Node Count")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    if all_zero:
        ax.text(
            0.5, 0.5,
            "All steal rates = 0\n(DAGs too small to exhaust worker queues)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#aaaaaa", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK_BG,
                      edgecolor=GRID_COL, alpha=0.7),
        )
        ax.set_ylim(-0.1, 1.0)

    save_fig(fig, DOCS / "steal_rate.png")

# ---------------------------------------------------------------------------

def main():
    data = load(RESULTS)
    print(f"Loaded {len(data)} records from {RESULTS}")
    plot_throughput(data)
    plot_efficiency(data)
    plot_steal_rate(data)
    print("Done.")

if __name__ == "__main__":
    main()
