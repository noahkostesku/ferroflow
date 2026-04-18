"""
Generate ferroflow strong-scaling plots from docs/data/scaling_results.json.

Outputs:
  docs/plots/scaling_throughput.png  — throughput vs nodes (2 subplots, one per DAG)
  docs/plots/scaling_efficiency.png  — parallel efficiency vs nodes
  docs/plots/steal_rate.png          — steal rate vs nodes (WS only)
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path(__file__).parent.parent / "docs" / "data" / "scaling_results.json"
PLOTS   = pathlib.Path(__file__).parent.parent / "docs" / "plots"

DARK_BG  = "#1c1c1e"
GRID_COL = "#3a3a3c"

# Colours keyed by (dag, scheduler)
COLORS = {
    ("large-transformer", "static"):        "#4fc3f7",
    ("large-transformer", "work-stealing"): "#0288d1",
    ("large-wide",        "static"):        "#f48fb1",
    ("large-wide",        "work-stealing"): "#c2185b",
}
LABELS = {
    ("large-transformer", "static"):        "large-transformer / static",
    ("large-transformer", "work-stealing"): "large-transformer / WS",
    ("large-wide",        "static"):        "large-wide / static",
    ("large-wide",        "work-stealing"): "large-wide / WS",
}
DAG_TITLES = {
    "large-transformer": "Large Transformer (137 ops, 8 layers)",
    "large-wide":        "Large Wide (321 ops, flat fan-out, skew=0.47)",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def series(data, dag, scheduler):
    rows = sorted(
        [r for r in data if r["dag"] == dag and r["scheduler"] == scheduler],
        key=lambda r: r["nodes"],
    )
    return (
        [r["nodes"]      for r in rows],
        [r["throughput"] for r in rows],
        [r["efficiency"] for r in rows],
        [r["steal_rate"] for r in rows],
    )


def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID_COL)
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
# Plot 1 — Throughput (two subplots — DAGs have very different scales)
# ---------------------------------------------------------------------------
def plot_throughput(data):
    dags = ["large-transformer", "large-wide"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("ferroflow Strong Scaling — Throughput", color="white", fontsize=13)

    all_nodes = sorted({r["nodes"] for r in data})

    for ax, dag in zip(axes, dags):
        style_ax(ax)

        for sched in ("static", "work-stealing"):
            nodes, tput, _, _ = series(data, dag, sched)
            ax.plot(nodes, tput, marker="o", linewidth=2, markersize=6,
                    color=COLORS[(dag, sched)], label=LABELS[(dag, sched)])

        # ideal linear line anchored at 2-node WS throughput
        base_nodes_list, base_tput_list, _, _ = series(data, dag, "work-stealing")
        base_n = base_nodes_list[0]
        base_t = base_tput_list[0]
        ideal_x = np.array([base_n, max(all_nodes)])
        ideal_y = base_t * (ideal_x / base_n)
        ax.plot(ideal_x, ideal_y, linestyle=":", linewidth=1.4,
                color="#aaaaaa", label="ideal (linear)", alpha=0.7)

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_nodes)
        ax.set_xticklabels([str(n) for n in all_nodes])
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Throughput (ops/s)")
        ax.set_title(DAG_TITLES[dag], fontsize=10)
        ax.legend(framealpha=0.2, labelcolor="white",
                  facecolor=DARK_BG, edgecolor=GRID_COL, fontsize=8)

    fig.tight_layout()
    save_fig(fig, PLOTS / "scaling_throughput.png")


# ---------------------------------------------------------------------------
# Plot 2 — Parallel efficiency (both DAGs, WS series only)
# ---------------------------------------------------------------------------
def plot_efficiency(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for dag in ("large-transformer", "large-wide"):
        color = COLORS[(dag, "work-stealing")]
        nodes, _, eff, _ = series(data, dag, "work-stealing")
        ax.plot(nodes, eff, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, "work-stealing")])

        # annotate first point below 0.70
        for n, e in zip(nodes, eff):
            if e < 0.70:
                ax.annotate(
                    f"eff={e:.2f} (n={n})",
                    xy=(n, e), xytext=(n + 0.3, e + 0.07),
                    color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                )
                break

    ax.axhline(0.70, linestyle="--", linewidth=1.4, color="#ffcc02",
               label="0.70 threshold", alpha=0.85)

    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Parallel Efficiency")
    ax.set_ylim(0, 1.15)
    ax.set_title("ferroflow Strong Scaling — Parallel Efficiency (WS)")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, PLOTS / "scaling_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 3 — Steal rate (WS only, both DAGs)
# ---------------------------------------------------------------------------
def plot_steal_rate(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for dag in ("large-transformer", "large-wide"):
        color = COLORS[(dag, "work-stealing")]
        nodes, _, _, steal = series(data, dag, "work-stealing")
        ax.plot(nodes, steal, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, "work-stealing")])
        # label the 8-node value
        ax.annotate(
            f"{steal[-1]:.0f}/s",
            xy=(nodes[-1], steal[-1]),
            xytext=(nodes[-1] - 0.4, steal[-1] + 25),
            color=color, fontsize=8,
        )

    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Steals / sec")
    ax.set_title("ferroflow Steal Rate vs Node Count (WS)")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, PLOTS / "steal_rate.png")


# ---------------------------------------------------------------------------

def main():
    data = load(RESULTS)
    print(f"Loaded {len(data)} records from {RESULTS}")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_throughput(data)
    plot_efficiency(data)
    plot_steal_rate(data)
    print("Done.")


if __name__ == "__main__":
    main()
