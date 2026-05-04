"""
examples/large_scale.py
========================
Verify that d_s ≈ 2 persists at macroscopic scale.

Best parameters from golden_zone.py: reward=3.5, β=30.
We test three graph sizes by varying size_penalty:

  size_penalty = 0.004  →  N* ≈ 100   (baseline)
  size_penalty = 0.002  →  N* ≈ 200
  size_penalty = 0.001  →  N* ≈ 400
  size_penalty = 0.0005 →  N* ≈ 800

Prediction (2D manifold hypothesis):
  - d_s ≈ 2 is stable across scales
  - Average path length L ∝ √N   (2D geodesic scaling)
  - T/N stays near 2 (self-similar triangulation)

Usage
-----
    python examples/large_scale.py
"""

import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from relate import RealitySimulation
from relate.measures import network_summary, small_world_metrics, spectral_dimension

# Fixed "golden zone" parameters
BETA           = 30.0
TOPOLOGY_REWARD = 3.5
STEPS          = 400          # more steps to reach larger equilibria
SEED           = 42

SIZE_PENALTIES = [0.004, 0.002, 0.001, 0.0005]
N_TARGETS      = [int(0.8 / (2 * sp)) for sp in SIZE_PENALTIES]

print("RELATE — Large-Scale d_s Stability Test")
print(f"β={BETA}  topology_reward={TOPOLOGY_REWARD}  steps={STEPS}")
print(f"{'size_pen':>10}  {'N*':>5}  {'N':>5}  {'d_s':>6}  "
      f"{'T/N':>5}  {'avg_deg':>7}  {'L':>7}  {'L/√N':>7}  {'cluster':>8}")
print("─" * 75)

results = []
t0 = time.time()

for sp, n_target in zip(SIZE_PENALTIES, N_TARGETS):
    np.random.seed(SEED)
    random.seed(SEED)

    sim = RealitySimulation(
        beta=BETA,
        topology_reward=TOPOLOGY_REWARD,
        size_penalty=sp,
        record_all_states=False,
    )
    sim.run(steps=STEPS, verbose=False)

    d_s, t_v, p_v = spectral_dimension(sim.state)
    ns = network_summary(sim.state)
    sw = small_world_metrics(sim.state)
    T_over_N = ns["triangles"] / max(1, ns["nodes"])
    N = ns["nodes"]
    L = sw["avg_path_length"]
    L_over_sqrtN = L / np.sqrt(N) if L else None

    results.append({
        "size_penalty": sp, "n_target": n_target,
        "d_s": d_s, "T_over_N": T_over_N,
        "L": L, "L_over_sqrtN": L_over_sqrtN,
        "t_vals": t_v, "p_vals": p_v,
        **ns, "sw_clustering": sw["clustering"],
    })

    L_str = f"{L:.2f}" if L else "n/a"
    Lsq_str = f"{L_over_sqrtN:.3f}" if L_over_sqrtN else "n/a"
    print(f"{sp:>10.4f}  {n_target:>5}  {N:>5}  {d_s:>6.3f}  "
          f"{T_over_N:>5.2f}  {ns['avg_degree']:>7.2f}  "
          f"{L_str:>7}  {Lsq_str:>7}  {ns['clustering']:>8.3f}")

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Test L ∝ √N ───────────────────────────────────────────────────────────────
valid = [(r["nodes"], r["L"]) for r in results if r["L"] is not None]
if len(valid) >= 3:
    Ns = np.array([v[0] for v in valid], dtype=float)
    Ls = np.array([v[1] for v in valid], dtype=float)
    slope_sqrt, _ = np.polyfit(np.sqrt(Ns), Ls, 1)
    slope_log, _  = np.polyfit(np.log(Ns), Ls, 1)
    print(f"\nL vs √N fit:  L = {slope_sqrt:.3f} · √N  (2D prediction)")
    print(f"L vs log(N) fit: L = {slope_log:.3f} · log(N)  (small-world prediction)")
    r2_sqrt = np.corrcoef(np.sqrt(Ns), Ls)[0, 1] ** 2
    r2_log  = np.corrcoef(np.log(Ns),  Ls)[0, 1] ** 2
    print(f"R² (√N): {r2_sqrt:.4f}   R² (log N): {r2_log:.4f}")
    if r2_sqrt > r2_log:
        print("→ L ∝ √N fits better  — consistent with 2D manifold")
    else:
        print("→ L ∝ log N fits better — consistent with small-world / 3D+")

# ── Plots ─────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor="#0f0f23")
    palette = plt.cm.plasma(np.linspace(0.15, 0.9, len(results)))

    # 1. Return probability p(t) — all scales
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a2e")
    t_ref = np.logspace(-1, np.log10(50), 60)
    if results[0]["t_vals"] is not None and len(results[0]["t_vals"]) > 5:
        anchor_t = results[0]["t_vals"][5]
        anchor_p = results[0]["p_vals"][5]
        for d_ref, col, ls, lbl in [
            (1, "#888", ":",  "d=1"),
            (2, "lime", "--", "d=2"),
            (3, "magenta", "-.", "d=3"),
        ]:
            p_line = anchor_p * (anchor_t / t_ref) ** (d_ref / 2)
            ax.loglog(t_ref, p_line, color=col, linestyle=ls,
                      linewidth=1.2, label=lbl, alpha=0.6)
    for r, col in zip(results, palette):
        if len(r["t_vals"]) > 0:
            lbl = f"N*≈{r['n_target']} → N={r['nodes']} d_s={r['d_s']:.2f}"
            ax.loglog(r["t_vals"], r["p_vals"], color=col, linewidth=2.5, label=lbl)
    ax.axvspan(0.5, 15.0, alpha=0.08, color="white")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Diffusion time t", color="white")
    ax.set_ylabel("Return probability p(t)", color="white")
    ax.set_title("Spectral dimension across scales", color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")

    # 2. d_s and T/N vs N
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a2e")
    Ns = [r["nodes"] for r in results]
    ds = [r["d_s"] for r in results]
    tn = [r["T_over_N"] for r in results]
    ax2 = ax.twinx()
    ax.plot(Ns, ds, "o-", color="cyan", linewidth=2.5, markersize=10, label="d_s")
    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5, alpha=0.7,
               label="d_s=2")
    ax2.plot(Ns, tn, "s--", color="coral", linewidth=1.5, markersize=8, label="T/N")
    ax2.axhline(2.0, color="coral", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Graph size N", color="white")
    ax.set_ylabel("d_s", color="cyan")
    ax2.set_ylabel("T/N", color="coral")
    ax.tick_params(colors="white"); ax2.tick_params(colors="coral")
    ax.set_title("d_s and T/N vs scale N", color="white", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15, color="white")

    # 3. L vs √N (geodesic scaling)
    ax = axes[0, 2]
    ax.set_facecolor("#1a1a2e")
    valid_r = [r for r in results if r["L"] is not None]
    if valid_r:
        xs = np.array([r["nodes"] for r in valid_r], dtype=float)
        ys = np.array([r["L"] for r in valid_r])
        ax.scatter(np.sqrt(xs), ys, c=[r["d_s"] for r in valid_r],
                   cmap="plasma", s=120, edgecolors="white", zorder=3)
        if len(valid_r) >= 2:
            xs_fit = np.linspace(np.sqrt(xs.min()), np.sqrt(xs.max()), 50)
            coef = np.polyfit(np.sqrt(xs), ys, 1)
            ax.plot(xs_fit, np.polyval(coef, xs_fit), "lime", linestyle="--",
                    linewidth=2, label=f"L={coef[0]:.3f}·√N + {coef[1]:.2f}")
        ax.set_xlabel("√N", color="white")
        ax.set_ylabel("Avg path length L", color="white")
        ax.set_title("Geodesic scaling L vs √N\n(2D → linear; 3D → cubic root)",
                     color="white", fontweight="bold")
        ax.legend(fontsize=9); ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="white")

    # 4. Degree distribution — largest graph
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a2e")
    largest = max(results, key=lambda r: r["nodes"])
    G = sim.state.graph  # last sim's state (largest N)
    degs = sorted([d for _, d in G.degree() if d > 0])
    from collections import Counter
    dc = Counter(degs)
    ax.loglog(list(dc.keys()), list(dc.values()), "o", color="coral",
              markersize=6, label=f"N={largest['nodes']}")
    ax.set_xlabel("Degree k", color="white")
    ax.set_ylabel("P(k)", color="white")
    ax.set_title("Degree distribution (largest graph)", color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    # 5. Clustering vs N
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a2e")
    ax.plot([r["nodes"] for r in results],
            [r["sw_clustering"] for r in results],
            "o-", color="gold", linewidth=2, markersize=9)
    ax.set_xlabel("Graph size N", color="white")
    ax.set_ylabel("Clustering coefficient", color="white")
    ax.set_title("Clustering coefficient vs scale", color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")

    # 6. Summary table
    ax = axes[1, 2]
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    rows = [["N*", "N", "d_s", "T/N", "L", "L/√N", "cluster"]]
    for r in results:
        L_s = f"{r['L']:.2f}" if r["L"] else "—"
        Ls_s = f"{r['L_over_sqrtN']:.3f}" if r["L_over_sqrtN"] else "—"
        rows.append([
            str(r["n_target"]), str(r["nodes"]),
            f"{r['d_s']:.3f}", f"{r['T_over_N']:.2f}",
            L_s, Ls_s, f"{r['sw_clustering']:.3f}",
        ])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        cell.set_facecolor("#1a1a2e" if row > 0 else "#2a2a4e")
        cell.set_text_props(color="white")
    ax.set_title("Summary table", color="white", fontweight="bold", pad=15)

    for ax_row in axes:
        for ax_ in ax_row:
            ax_.set_facecolor("#1a1a2e")

    plt.suptitle(
        f"RELATE — Large-Scale Stability  "
        f"(β={BETA}, topology_reward={TOPOLOGY_REWARD})",
        fontsize=15, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("large_scale.png", dpi=120, bbox_inches="tight", facecolor="#0f0f23")
    print("\nSaved large_scale.png")
    plt.show()
