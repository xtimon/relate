"""
examples/phase_diagram.py
==========================
Phase diagram of RELATE: sweep α (curvature weight) × β (information weight).

Each cell of the 3×3 grid runs an independent simulation.  Four heatmaps
show how the four key observables respond to the parameter change:
  - Final node count N  (size of the universe)
  - Integrated information I(G)  (spectral complexity)
  - Max curvature amplitude max|δ|  (geometric ripple)
  - Integrate-event count  (particle births)

Expected phases
---------------
  Low β, low α   → GAS     : loose graph, many seed/deflate, low I
  High α, low β  → CRYSTAL : flat curvature, rare integrate, low I
  High β          → COLLAPSE: integrate dominates, graph contracts
  Mid-range       → COMPLEX : high integrate count, rich curvature

Usage
-----
    python examples/phase_diagram.py
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

import relate
from relate import RealitySimulation
from relate.measures import integrate_event_analysis

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHAS = [0.1, 0.3, 1.0]          # curvature penalty weight
BETAS  = [0.5, 3.0, 8.0]          # information reward weight
STEPS  = 150
SEED   = 42

# ── Sweep ─────────────────────────────────────────────────────────────────────
print("RELATE — Phase Diagram  (α × β sweep)")
print(f"Grid: {len(ALPHAS)}×{len(BETAS)}, {STEPS} steps each\n")

grid: dict = {}
t0 = time.time()

for alpha in ALPHAS:
    for beta in BETAS:
        np.random.seed(SEED)
        random.seed(SEED)

        sim = RealitySimulation(alpha=alpha, beta=beta, record_all_states=False)
        history = sim.run(steps=STEPS, verbose=False)

        events = integrate_event_analysis(history)
        grid[(alpha, beta)] = {
            "final_N":         sim.state.node_count,
            "final_I":         history[-1]["I"],
            "max_curv":        max(r["curvature_max"] for r in history),
            "integrate_count": events["count"],
            "expand_frac":     sum(1 for r in history if r["move_type"] == "expand") / len(history),
            "history":         history,
        }
        print(f"  α={alpha:.1f} β={beta:.1f}  →  "
              f"N={sim.state.node_count:3d}  "
              f"I={history[-1]['I']:.4f}  "
              f"max|δ|={max(r['curvature_max'] for r in history):.3e}  "
              f"integrate={events['count']:2d}")

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Build heatmap matrices ─────────────────────────────────────────────────────
def make_matrix(key):
    return np.array([[grid[(a, b)][key] for b in BETAS] for a in ALPHAS])

mat_N   = make_matrix("final_N")
mat_I   = make_matrix("final_I")
mat_C   = make_matrix("max_curv")
mat_int = make_matrix("integrate_count")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="#0f0f23")
    fig.suptitle("RELATE — Phase Diagram  (α × β)",
                 fontsize=16, fontweight="bold", color="white", y=1.01)

    panels = [
        (axes[0, 0], mat_N,   "Final node count N",           "plasma",  False),
        (axes[0, 1], mat_I,   "Integrated information I(G)",  "viridis", True),
        (axes[1, 0], mat_C,   "Max curvature amplitude max|δ|","hot",    True),
        (axes[1, 1], mat_int, "Integrate events (particle births)", "YlOrRd", False),
    ]

    xtick_labels = [f"β={b}" for b in BETAS]
    ytick_labels = [f"α={a}" for a in ALPHAS]

    for ax, mat, title, cmap, log_scale in panels:
        ax.set_facecolor("#1a1a2e")
        data = np.log1p(mat) if log_scale else mat
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(colors="white")
        if log_scale:
            cbar.set_label("log(1 + value)", color="white", fontsize=9)

        # Annotate cells
        for i in range(len(ALPHAS)):
            for j in range(len(BETAS)):
                val = mat[i, j]
                label = f"{val:.3f}" if val < 0.1 else f"{val:.1f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")

        ax.set_xticks(range(len(BETAS)))
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_xticklabels(xtick_labels, color="white")
        ax.set_yticklabels(ytick_labels, color="white")
        ax.set_xlabel("β  (information reward)", color="white")
        ax.set_ylabel("α  (curvature penalty)", color="white")
        ax.set_title(title, color="white", fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig("phase_diagram.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved phase_diagram.png")
    plt.show()

# ── Time-series overlay for all 9 runs ────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f0f23")

    cmap9 = plt.cm.rainbow(np.linspace(0, 1, len(ALPHAS) * len(BETAS)))
    idx = 0
    for alpha in ALPHAS:
        for beta in BETAS:
            h = grid[(alpha, beta)]["history"]
            t = [r["time"] for r in h]
            n = [r["nodes"] for r in h]
            c = [r["curvature_max"] for r in h]
            label = f"α={alpha} β={beta}"
            ax1.plot(t, n, color=cmap9[idx], linewidth=1.5, label=label)
            ax2.semilogy(t, [max(v, 1e-10) for v in c],
                         color=cmap9[idx], linewidth=1.5, label=label)
            idx += 1

    for ax, ylabel, title in [
        (ax1, "Node count N", "Universe size over time"),
        (ax2, "max|δ| (log)", "Curvature ripple amplitude"),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.set_xlabel("Time", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="white")
        ax.legend(fontsize=7, ncol=3, loc="upper left")

    fig.patch.set_facecolor("#0f0f23")
    plt.suptitle("RELATE — All phase-diagram trajectories",
                 fontsize=14, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig("phase_diagram_trajectories.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("Saved phase_diagram_trajectories.png")
    plt.show()
