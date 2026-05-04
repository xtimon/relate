"""
examples/golden_zone.py
========================
Fine-grained search for the 'golden zone' where d_s ≈ 2.

Hypothesis: d_s peaks when the graph achieves moderate topological density
(T/N ≈ 2–5) at reasonable graph size (N ≈ 80–150).  This requires
topology_reward below the clique-collapse threshold (~5) but above the
neutral point (~2).

Two-dimensional sweep:
  topology_reward ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
  beta            ∈ {10, 30, 50}

Each run: 200 steps, size_penalty=0.001 (allows N up to ~400).

Usage
-----
    python examples/golden_zone.py
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

from relate import RealitySimulation
from relate.measures import network_summary, spectral_dimension

TOPOLOGY_REWARDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
BETAS            = [10.0, 30.0, 50.0]
STEPS            = 200
SIZE_PENALTY     = 0.001   # N* ≈ 400
SEED             = 42

print("RELATE — Golden Zone Search  (d_s → 2)")
print(f"Grid: {len(TOPOLOGY_REWARDS)} rewards × {len(BETAS)} β  |  "
      f"steps={STEPS}  size_penalty={SIZE_PENALTY}\n")
print(f"{'reward':>7}  {'β':>5}  {'d_s':>6}  {'N':>5}  {'T':>5}  "
      f"{'T/N':>5}  {'avg_deg':>7}  {'cluster':>8}")
print("─" * 60)

results = []
t0 = time.time()

for tr in TOPOLOGY_REWARDS:
    for beta in BETAS:
        np.random.seed(SEED)
        random.seed(SEED)

        sim = RealitySimulation(
            beta=beta,
            topology_reward=tr,
            size_penalty=SIZE_PENALTY,
            record_all_states=False,
        )
        sim.run(steps=STEPS, verbose=False)

        d_s, t_v, p_v = spectral_dimension(sim.state)
        ns = network_summary(sim.state)
        T_over_N = ns["triangles"] / max(1, ns["nodes"])
        tri_count = sum(1 for h in sim.history if h["move_type"] == "triangulate")

        results.append({
            "topology_reward": tr,
            "beta": beta,
            "d_s": d_s,
            "T_over_N": T_over_N,
            "tri_count": tri_count,
            "t_vals": t_v,
            "p_vals": p_v,
            **ns,
        })

        marker = " ◀ best so far" if d_s == max(r["d_s"] for r in results) else ""
        print(f"{tr:>7.1f}  {beta:>5.0f}  {d_s:>6.3f}  {ns['nodes']:>5}  "
              f"{ns['triangles']:>5}  {T_over_N:>5.2f}  "
              f"{ns['avg_degree']:>7.2f}  {ns['clustering']:>8.3f}{marker}")

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
ranked = sorted(results, key=lambda r: r["d_s"], reverse=True)
print("\n── Top 5 by d_s ─────────────────────────────────────────────")
print(f"{'rank':>4}  {'reward':>7}  {'β':>5}  {'d_s':>6}  "
      f"{'N':>5}  {'T/N':>5}  {'cluster':>8}")
for i, r in enumerate(ranked[:5], 1):
    print(f"{i:>4}  {r['topology_reward']:>7.1f}  {r['beta']:>5.0f}  "
          f"{r['d_s']:>6.3f}  {r['nodes']:>5}  {r['T_over_N']:>5.2f}  "
          f"{r['clustering']:>8.3f}")

best = ranked[0]
print(f"\nBest: reward={best['topology_reward']} β={best['beta']}  "
      f"→  d_s={best['d_s']:.3f}  (N={best['nodes']}, T/N={best['T_over_N']:.2f})")

# ── Build matrices ─────────────────────────────────────────────────────────────
def mat(key):
    return np.array([
        [next(r[key] for r in results
              if r["topology_reward"] == tr and r["beta"] == b)
         for b in BETAS]
        for tr in TOPOLOGY_REWARDS
    ])

mat_ds  = mat("d_s")
mat_tn  = mat("T_over_N")
mat_cl  = mat("clustering")
mat_N   = mat("nodes")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(22, 14), facecolor="#0f0f23")

    xtick = [f"β={b:.0f}" for b in BETAS]
    ytick = [f"r={tr}" for tr in TOPOLOGY_REWARDS]

    def heatmap(ax, data, title, cmap, vmin=None, vmax=None, fmt=".2f"):
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.ax.tick_params(colors="white")
        for i in range(len(TOPOLOGY_REWARDS)):
            for j in range(len(BETAS)):
                val = data[i, j]
                label = f"{val:{fmt}}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        ax.set_xticks(range(len(BETAS)));      ax.set_xticklabels(xtick, color="white")
        ax.set_yticks(range(len(TOPOLOGY_REWARDS))); ax.set_yticklabels(ytick, color="white")
        ax.set_xlabel("β", color="white"); ax.set_ylabel("topology_reward", color="white")
        ax.set_title(title, color="white", fontweight="bold", pad=8)
        ax.set_facecolor("#1a1a2e")

    # 1. d_s heatmap (main result)
    ax1 = fig.add_subplot(2, 3, (1, 2))
    heatmap(ax1, mat_ds, "Spectral dimension d_s  (target: 2.0)", "RdYlGn",
            vmin=0.5, vmax=2.5)
    try:
        ax1.contour(mat_ds, levels=[2.0], colors="cyan", linewidths=2, linestyles="--")
    except Exception:
        pass
    ax1.set_title("Spectral dimension d_s  (cyan = d_s=2 contour)",
                  color="white", fontweight="bold", pad=8)

    # 2. T/N heatmap
    ax2 = fig.add_subplot(2, 3, 3)
    heatmap(ax2, mat_tn, "Topological density T/N", "plasma", fmt=".1f")

    # 3. Clustering
    ax3 = fig.add_subplot(2, 3, 4)
    heatmap(ax3, mat_cl, "Clustering coefficient", "viridis", vmin=0, vmax=1)

    # 4. Node count
    ax4 = fig.add_subplot(2, 3, 5)
    heatmap(ax4, mat_N, "Final node count N", "Blues", fmt=".0f")

    # 5. Return-probability curves — top 6 by d_s
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.set_facecolor("#1a1a2e")
    t_ref = np.logspace(-1, np.log10(50), 60)
    # Reference slopes
    anchor_t, anchor_p = 1.0, 0.3
    for d_ref, col, ls, lbl in [
        (1, "#888", ":",  "d=1"),
        (2, "lime", "--", "d=2"),
        (3, "magenta", "-.", "d=3"),
    ]:
        p_line = anchor_p * (anchor_t / t_ref) ** (d_ref / 2)
        ax5.loglog(t_ref, p_line, color=col, linestyle=ls,
                   linewidth=1.2, label=lbl, alpha=0.6)

    palette = plt.cm.plasma(np.linspace(0.1, 0.9, min(6, len(ranked))))
    for r, col in zip(ranked[:6], palette):
        if len(r["t_vals"]) > 0 and len(r["p_vals"]) > 0:
            lbl = f"r={r['topology_reward']} β={r['beta']:.0f} → {r['d_s']:.2f}"
            ax5.loglog(r["t_vals"], r["p_vals"], color=col, linewidth=2, label=lbl)

    ax5.axvspan(0.5, 15.0, alpha=0.08, color="white")
    ax5.legend(fontsize=7, loc="lower left")
    ax5.set_xlabel("Diffusion time t", color="white")
    ax5.set_ylabel("Return probability p(t)", color="white")
    ax5.set_title("Return probability — top 6 by d_s", color="white", fontweight="bold")
    ax5.tick_params(colors="white")
    ax5.grid(alpha=0.15, color="white")

    plt.suptitle(
        f"RELATE — Golden Zone  (best d_s={best['d_s']:.3f} at "
        f"reward={best['topology_reward']} β={best['beta']:.0f})",
        fontsize=15, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("golden_zone.png", dpi=120, bbox_inches="tight", facecolor="#0f0f23")
    print("\nSaved golden_zone.png")
    plt.show()

# ── d_s vs T/N scatter (key relationship) ────────────────────────────────────
with plt.style.context("dark_background"):
    fig2, ax = plt.subplots(figsize=(10, 7), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")

    sc = ax.scatter(
        [r["T_over_N"] for r in results],
        [r["d_s"] for r in results],
        c=[r["beta"] for r in results],
        cmap="plasma", s=120, edgecolors="white", linewidths=0.5,
        zorder=3,
    )
    for r in results:
        ax.annotate(
            f"r={r['topology_reward']}\nβ={r['beta']:.0f}",
            (r["T_over_N"], r["d_s"]),
            textcoords="offset points", xytext=(6, 3),
            fontsize=7, color="white", alpha=0.8,
        )

    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5,
               alpha=0.7, label="d_s = 2 target")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("β", color="white")
    cbar.ax.tick_params(colors="white")

    ax.set_xlabel("Topological density T/N", color="white", fontsize=12)
    ax.set_ylabel("Spectral dimension d_s", color="white", fontsize=12)
    ax.set_title("d_s vs T/N — identifying the golden zone",
                 color="white", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    fig2.patch.set_facecolor("#0f0f23")
    plt.tight_layout()
    plt.savefig("golden_zone_scatter.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("Saved golden_zone_scatter.png")
    plt.show()
