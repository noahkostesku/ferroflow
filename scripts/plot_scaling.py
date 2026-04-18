"""
Generate ferroflow strong-scaling plots from docs/data/scaling_results.json.

Outputs:
  docs/plots/scaling_throughput.png  — throughput vs nodes (3 subplots, one per DAG)
  docs/plots/scaling_efficiency.png  — parallel efficiency vs nodes (WS, all DAGs)
  docs/plots/steal_rate.png          — steal rate vs nodes (WS only, all DAGs)
  docs/plots/ws_advantage.png        — % WS throughput improvement over static
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

COLORS = {
    ("large-transformer", "static"):        "#4fc3f7",
    ("large-transformer", "work-stealing"): "#0288d1",
    ("large-wide",        "static"):        "#f48fb1",
    ("large-wide",        "work-stealing"): "#c2185b",
    ("imbalanced",        "static"):        "#ffcc80",
    ("imbalanced",        "work-stealing"): "#e65100",
}
LABELS = {
    ("large-transformer", "static"):        "transformer / static",
    ("large-transformer", "work-stealing"): "transformer / WS",
    ("large-wide",        "static"):        "large-wide / static",
    ("large-wide",        "work-stealing"): "large-wide / WS",
    ("imbalanced",        "static"):        "imbalanced / static",
    ("imbalanced",        "work-stealing"): "imbalanced / WS",
}
DAG_TITLES = {
    "large-transformer": "Large Transformer\n(137 ops, 8 layers, d=512)",
    "large-wide":        "Large Wide\n(321 ops, flat fan-out, skew=0.47)",
    "imbalanced":        "Imbalanced\n(4 heavy×200ms + 200 fast×1ms)",
}
ALL_DAGS = ["large-transformer", "large-wide", "imbalanced"]


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
        [r.get("stddev", 0.0) for r in rows],
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
# Plot 1 — Throughput (3 subplots, one per DAG)
# ---------------------------------------------------------------------------
def plot_throughput(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("ferroflow Strong Scaling — Throughput", color="white", fontsize=13)

    all_nodes = sorted({r["nodes"] for r in data})

    for ax, dag in zip(axes, ALL_DAGS):
        style_ax(ax)

        s_nodes, s_tp, _, _, s_std = series(data, dag, "static")
        w_nodes, w_tp, _, _, w_std = series(data, dag, "work-stealing")

        ax.errorbar(s_nodes, s_tp, yerr=s_std, marker="s", linewidth=2, markersize=6,
                    capsize=4, color=COLORS[(dag, "static")], label="static")
        ax.errorbar(w_nodes, w_tp, yerr=w_std, marker="o", linewidth=2, markersize=6,
                    capsize=4, color=COLORS[(dag, "work-stealing")], label="WS")

        # fill between to highlight WS advantage on imbalanced
        if dag == "imbalanced":
            ax.fill_between(w_nodes, s_tp, w_tp,
                            color=COLORS[(dag, "work-stealing")], alpha=0.15,
                            label="WS gain")

        # ideal linear anchored at 2-node static
        base_n, base_t = s_nodes[0], s_tp[0]
        ideal_x = np.array([base_n, max(all_nodes)])
        ideal_y = base_t * (ideal_x / base_n)
        ax.plot(ideal_x, ideal_y, linestyle=":", linewidth=1.2,
                color="#aaaaaa", label="ideal (linear)", alpha=0.7)

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_nodes)
        ax.set_xticklabels([str(n) for n in all_nodes])
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Throughput (ops/s)")
        ax.set_title(DAG_TITLES[dag], fontsize=9, color="white")
        ax.legend(framealpha=0.2, labelcolor="white",
                  facecolor=DARK_BG, edgecolor=GRID_COL, fontsize=8)

    fig.tight_layout()
    save_fig(fig, PLOTS / "scaling_throughput.png")


# ---------------------------------------------------------------------------
# Plot 2 — Parallel efficiency (WS, all 3 DAGs)
# ---------------------------------------------------------------------------
def plot_efficiency(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for dag in ALL_DAGS:
        color = COLORS[(dag, "work-stealing")]
        nodes, _, eff, _, _ = series(data, dag, "work-stealing")
        lw = 2.5 if dag == "large-wide" else 2.0
        ax.plot(nodes, eff, marker="o", linewidth=lw, markersize=6,
                color=color, label=LABELS[(dag, "work-stealing")])

        # annotate the 8-node efficiency value
        ax.annotate(
            f"{eff[-1]:.2f}",
            xy=(nodes[-1], eff[-1]),
            xytext=(nodes[-1] + 0.15, eff[-1] + (0.04 if dag == "large-wide" else -0.06)),
            color=color, fontsize=8,
        )

    ax.axhline(0.70, linestyle="--", linewidth=1.4, color="#ffcc02",
               label="0.70 threshold", alpha=0.85)

    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Parallel Efficiency")
    ax.set_ylim(0, 1.20)
    ax.set_title("ferroflow Strong Scaling — Parallel Efficiency (WS)", color="white")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, PLOTS / "scaling_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 3 — Steal rate (WS only, all 3 DAGs)
# ---------------------------------------------------------------------------
def plot_steal_rate(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for dag in ALL_DAGS:
        color = COLORS[(dag, "work-stealing")]
        nodes, _, _, steal, _ = series(data, dag, "work-stealing")
        ax.plot(nodes, steal, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[(dag, "work-stealing")])
        ax.annotate(
            f"{steal[-1]:.0f}/s",
            xy=(nodes[-1], steal[-1]),
            xytext=(nodes[-1] - 0.55, steal[-1] + 18),
            color=color, fontsize=8,
        )

    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Steals / sec")
    ax.set_title("ferroflow Steal Rate vs Node Count (WS)", color="white")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, PLOTS / "steal_rate.png")


# ---------------------------------------------------------------------------
# Plot 4 — WS advantage: % throughput improvement over static
# ---------------------------------------------------------------------------
def plot_ws_advantage(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    all_nodes = sorted({r["nodes"] for r in data})

    for dag in ALL_DAGS:
        color = COLORS[(dag, "work-stealing")]
        s_nodes, s_tp, _, _, _ = series(data, dag, "static")
        w_nodes, w_tp, _, _, _ = series(data, dag, "work-stealing")

        pct = [(w - s) / s * 100 for s, w in zip(s_tp, w_tp)]
        lw = 2.5 if dag == "imbalanced" else 1.8
        marker = "D" if dag == "imbalanced" else "o"
        ax.plot(s_nodes, pct, marker=marker, linewidth=lw, markersize=7,
                color=color, label=LABELS[(dag, "work-stealing")])

        # annotate each point
        for n, p in zip(s_nodes, pct):
            offset_y = 0.4 if p >= 0 else -1.2
            ax.annotate(f"{p:+.1f}%", xy=(n, p),
                        xytext=(n, p + offset_y),
                        color=color, fontsize=7.5, ha="center")

    ax.axhline(0, linestyle="--", linewidth=1.0, color="#aaaaaa", alpha=0.6)

    ax.set_xticks(all_nodes)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("WS throughput improvement over static (%)")
    ax.set_title("Work-Stealing Advantage over Static Scheduling", color="white")
    ax.legend(framealpha=0.2, labelcolor="white",
              facecolor=DARK_BG, edgecolor=GRID_COL)

    save_fig(fig, PLOTS / "ws_advantage.png")


# ---------------------------------------------------------------------------

def main():
    data = load(RESULTS)
    print(f"Loaded {len(data)} records from {RESULTS}")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_throughput(data)
    plot_efficiency(data)
    plot_steal_rate(data)
    plot_ws_advantage(data)
    print("Done.")


if __name__ == "__main__":
    main()
