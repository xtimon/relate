"""
examples/spectral_dim_sweep.py
================================
Search for parameters where the spectral dimension d_s approaches 2.

Strategy
--------
d_s depends primarily on graph density (triangles per node).  To push
d_s → 2 we explore two axes:

  1.  α × β grid  — standard phase-diagram sweep, measuring d_s per cell
  2.  size_penalty sweep  — larger graphs have more eigenvalues and may
      exhibit a cleaner power-law regime in p(t)

The script prints a ranked table of parameter sets by d_s and saves a
heatmap + return-probability overlay plot.

Usage
-----
    python examples/spectral_dim_sweep.py
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

import relate
from relate import RealitySimulation
from relate.measures import network_summary, spectral_dimension

# ── Sweep parameters ──────────────────────────────────────────────────────────
ALPHAS        = [0.05, 0.1,  0.3,  0.6,  1.2]
BETAS         = [0.5,  1.5,  3.0,  6.0, 10.0]
SIZE_PENALTY  = 0.002   # lower → bigger graphs → more eigenvalues → better d_s estimate
STEPS         = 200
SEED          = 42

# ── Run sweep ─────────────────────────────────────────────────────────────────
print("RELATE — Spectral Dimension Sweep")
print(f"Grid: {len(ALPHAS)} α × {len(BETAS)} β  |  steps={STEPS}  size_penalty={SIZE_PENALTY}")
print(f"{'α':>6} {'β':>6}  {'d_s':>6}  {'N':>5}  {'T':>5}  {'T/N':>5}  {'avg_deg':>7}")
print("─" * 55)

results = []
t0 = time.time()

for alpha in ALPHAS:
    for beta in BETAS:
        np.random.seed(SEED)
        random.seed(SEED)

        sim = RealitySimulation(
            alpha=alpha, beta=beta,
            size_penalty=SIZE_PENALTY,
            record_all_states=False,
        )
        sim.run(steps=STEPS, verbose=False)

        d_s, t_vals, p_vals = spectral_dimension(sim.state)
        ns = network_summary(sim.state)

        T_over_N = sim.state.triple_count / max(1, sim.state.node_count)
        results.append({
            "alpha": alpha, "beta": beta,
            "d_s": d_s, "N": ns["nodes"], "T": ns["triangles"],
            "T_over_N": T_over_N, "avg_degree": ns["avg_degree"],
            "t_vals": t_vals, "p_vals": p_vals,
        })

        print(f"{alpha:>6.2f} {beta:>6.1f}  {d_s:>6.3f}  "
              f"{ns['nodes']:>5}  {ns['triangles']:>5}  "
              f"{T_over_N:>5.2f}  {ns['avg_degree']:>7.2f}")

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Ranked table ───────────────────────────────────────────────────────────────
ranked = sorted(results, key=lambda r: abs(r["d_s"] - 2.0))
print("\n── Closest to d_s = 2 ───────────────────────────────────")
print(f"{'Rank':>4}  {'α':>5} {'β':>5}  {'d_s':>6}  {'|d_s−2|':>8}  {'N':>5}  {'T/N':>5}")
for i, r in enumerate(ranked[:8], 1):
    print(f"{i:>4}  {r['alpha']:>5.2f} {r['beta']:>5.1f}  "
          f"{r['d_s']:>6.3f}  {abs(r['d_s']-2.0):>8.3f}  "
          f"{r['N']:>5}  {r['T_over_N']:>5.2f}")

best = ranked[0]
print(f"\nBest: α={best['alpha']} β={best['beta']}  →  d_s={best['d_s']:.3f}  "
      f"(N={best['N']}, T/N={best['T_over_N']:.2f})")

# ── Build heatmap matrix ───────────────────────────────────────────────────────
d_s_matrix = np.array(
    [[next(r["d_s"] for r in results if r["alpha"] == a and r["beta"] == b)
      for b in BETAS]
     for a in ALPHAS]
)

# ── Plots ─────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(20, 12), facecolor="#0f0f23")

    # 1. Heatmap d_s(α, β) ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, (1, 2), facecolor="#1a1a2e")

    im = ax1.imshow(
        d_s_matrix, cmap="RdYlGn", aspect="auto", origin="lower",
        vmin=0.0, vmax=3.0,
    )
    cbar = plt.colorbar(im, ax=ax1, shrink=0.85)
    cbar.set_label("Spectral dimension d_s", color="white")
    cbar.ax.tick_params(colors="white")

    # Contour at d_s = 2
    try:
        cs = ax1.contour(d_s_matrix, levels=[2.0], colors="cyan",
                         linewidths=2, linestyles="--")
        ax1.clabel(cs, fmt="d_s=2", colors="cyan", fontsize=10)
    except Exception:
        pass

    for i, alpha in enumerate(ALPHAS):
        for j, beta in enumerate(BETAS):
            d = d_s_matrix[i, j]
            color = "black" if 1.5 < d < 2.5 else "white"
            ax1.text(j, i, f"{d:.2f}", ha="center", va="center",
                     fontsize=11, color=color, fontweight="bold")

    ax1.set_xticks(range(len(BETAS)))
    ax1.set_yticks(range(len(ALPHAS)))
    ax1.set_xticklabels([f"β={b}" for b in BETAS], color="white")
    ax1.set_yticklabels([f"α={a}" for a in ALPHAS], color="white")
    ax1.set_xlabel("β  (information reward)", color="white", fontsize=12)
    ax1.set_ylabel("α  (curvature penalty)", color="white", fontsize=12)
    ax1.set_title(
        f"Spectral dimension d_s(α, β)\n"
        f"size_penalty={SIZE_PENALTY}, {STEPS} steps — target: d_s ≈ 2 (cyan contour)",
        color="white", fontweight="bold", fontsize=13,
    )

    # 2. Return-probability curves for top-5 candidates ─────────────────────────
    ax2 = fig.add_subplot(1, 3, 3, facecolor="#1a1a2e")

    # Reference lines
    t_ref = np.logspace(-1, np.log10(50), 50)
    pivot = 0.5
    p_pivot = 0.5
    for d_ref, color, ls, lbl in [
        (1, "#888", ":",  "d=1"),
        (2, "lime", "--", "d=2"),
        (3, "magenta", "-.", "d=3"),
    ]:
        p_line = p_pivot * (pivot / t_ref) ** (d_ref / 2)
        ax2.loglog(t_ref, p_line, color=color, linestyle=ls,
                   linewidth=1.5, label=lbl, alpha=0.7)

    # Top-5 closest to d_s=2
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, min(5, len(ranked))))
    for r, col in zip(ranked[:5], palette):
        if len(r["t_vals"]) > 0 and len(r["p_vals"]) > 0:
            label = f"α={r['alpha']} β={r['beta']} → d_s={r['d_s']:.2f}"
            ax2.loglog(r["t_vals"], r["p_vals"], color=col,
                       linewidth=2, label=label)

    ax2.axvspan(0.5, 15.0, alpha=0.08, color="white")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_xlabel("Diffusion time t", color="white", fontsize=11)
    ax2.set_ylabel("Return probability p(t)", color="white", fontsize=11)
    ax2.set_title("Top-5 candidates\nreturn-probability curves",
                  color="white", fontweight="bold", fontsize=13)
    ax2.tick_params(colors="white")
    ax2.grid(alpha=0.15, color="white")

    plt.suptitle("RELATE — Search for Spectral Dimension d_s ≈ 2",
                 fontsize=15, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("spectral_dim_sweep.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved spectral_dim_sweep.png")
    plt.show()

# ── Deep-dive: best parameters at higher step count ───────────────────────────
print(f"\n── Deep-dive: best params α={best['alpha']} β={best['beta']} for 500 steps ──")
np.random.seed(SEED)
random.seed(SEED)

sim2 = RealitySimulation(
    alpha=best["alpha"], beta=best["beta"],
    size_penalty=SIZE_PENALTY / 2,   # allow larger graph
    record_all_states=False,
)
sim2.run(steps=500, verbose=False)
d_s2, t2, p2 = spectral_dimension(sim2.state)
ns2 = network_summary(sim2.state)

print(f"  N={ns2['nodes']}  T={ns2['triangles']}  T/N={ns2['triangles']/max(1,ns2['nodes']):.2f}")
print(f"  avg_degree={ns2['avg_degree']:.2f}  clustering={ns2['clustering']:.3f}")
print(f"  d_s = {d_s2:.3f}")

with plt.style.context("dark_background"):
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f23")

    for d_ref, color, ls in [(1, "#888", ":"), (2, "lime", "--"), (3, "magenta", "-.")]:
        p_line = p2[5] * (t2[5] / t2) ** (d_ref / 2) if len(t2) > 5 else []
        if len(p_line):
            axA.loglog(t2, p_line, color=color, linestyle=ls, linewidth=1.5,
                       label=f"d={d_ref}", alpha=0.7)

    if len(t2) > 0:
        axA.loglog(t2, p2, "cyan", linewidth=2.5, label=f"d_s≈{d_s2:.2f}")
        axA.axvspan(0.5, 15.0, alpha=0.08, color="white", label="fit window")

    axA.set_facecolor("#1a1a2e")
    axA.set_xlabel("Diffusion time t", color="white")
    axA.set_ylabel("Return probability p(t)", color="white")
    axA.set_title(f"Deep-dive: α={best['alpha']} β={best['beta']}\n500 steps → d_s={d_s2:.3f}",
                  color="white", fontweight="bold")
    axA.legend(fontsize=9)
    axA.tick_params(colors="white")
    axA.grid(alpha=0.15, color="white")

    # Degree distribution
    degrees = sorted([d for _, d in sim2.state.graph.degree() if d > 0])
    deg_counts = {}
    for d in degrees:
        deg_counts[d] = deg_counts.get(d, 0) + 1
    ks = list(deg_counts.keys())
    vs = list(deg_counts.values())

    axB.set_facecolor("#1a1a2e")
    axB.loglog(ks, vs, "o", color="coral", markersize=5, label="P(k)")
    if len(ks) > 3:
        gamma = relate.power_law_exponent(sim2.state)
        if gamma > 0:
            k_arr = np.array(ks, dtype=float)
            norm = vs[0] * ks[0] ** gamma
            axB.loglog(k_arr, norm * k_arr ** (-gamma), "--",
                       color="lime", linewidth=1.5, label=f"k^(-{gamma:.2f})")
    axB.legend(fontsize=9)
    axB.set_xlabel("Degree k", color="white")
    axB.set_ylabel("P(k)", color="white")
    axB.set_title("Degree distribution", color="white", fontweight="bold")
    axB.tick_params(colors="white")
    axB.grid(alpha=0.15, color="white")

    fig2.patch.set_facecolor("#0f0f23")
    plt.suptitle(f"RELATE — Deep dive into best parameters (d_s={d_s2:.3f})",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig("spectral_dim_deepdive.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("Saved spectral_dim_deepdive.png")
    plt.show()
