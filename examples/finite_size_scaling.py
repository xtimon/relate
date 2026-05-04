"""
examples/finite_size_scaling.py
=================================
Finite-size scaling (FSS) analysis of the spectral dimension d_s.

FSS is the standard physics technique to extract bulk (thermodynamic-limit)
behaviour from finite-system measurements.  For a 2D manifold:

  d_s(N) → 2  as  N → ∞           (dimension converges)
  L(N)   ∝ √N                      (geodesic scaling)
  T/N    → const ≈ 2               (topological density self-similar)

If instead d_s(N) drifts or saturates below 2, RELATE's attractor is not
a true 2D manifold but an approximate one.

Design
------
  N* values : 50, 100, 200, 400, 800  (via size_penalty = vacuum/(2·N*))
  Seeds     : 5 per N* (statistical error bars)
  Steps     : 200 (enough to reach equilibrium for each N*)
  I_interval: 10  (eigenvalue every 10 steps — 10× speedup)

Output
------
  FSS plot: d_s(N) with error bars + extrapolation to N→∞
  L/√N plot: test whether geodesic scaling constant is stable
  T/N plot: self-similarity of topological density

Usage
-----
    python examples/finite_size_scaling.py
"""

import random
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from relate import RealitySimulation
from relate.measures import network_summary, small_world_metrics, spectral_dimension

# ── Parameters ────────────────────────────────────────────────────────────────
BETA            = 30.0
TOPOLOGY_REWARD = 3.5
VACUUM          = 0.8
STEPS           = 200
I_INTERVAL      = 10
N_SEEDS         = 5

N_TARGETS = [50, 100, 200, 400, 800]
SIZE_PENALTIES = [VACUUM / (2 * n) for n in N_TARGETS]

print("RELATE — Finite-Size Scaling Analysis")
print(f"β={BETA}  topology_reward={TOPOLOGY_REWARD}  {STEPS} steps  "
      f"{N_SEEDS} seeds/point  I_interval={I_INTERVAL}\n")
print(f"{'N*':>5}  {'seed':>5}  {'N':>5}  {'d_s':>6}  "
      f"{'T/N':>5}  {'L':>6}  {'L/√N':>6}  {'cluster':>8}")
print("─" * 58)

all_runs: Dict[int, List[Dict]] = {n: [] for n in N_TARGETS}
t0 = time.time()

for n_target, sp in zip(N_TARGETS, SIZE_PENALTIES):
    for seed in range(N_SEEDS):
        np.random.seed(seed); random.seed(seed)
        sim = RealitySimulation(
            beta=BETA, topology_reward=TOPOLOGY_REWARD,
            vacuum=VACUUM, size_penalty=sp,
            i_update_interval=I_INTERVAL,
            record_all_states=False,
        )
        sim.run(steps=STEPS, verbose=False)

        d_s, t_v, p_v = spectral_dimension(sim.state)
        ns = network_summary(sim.state)
        sw = small_world_metrics(sim.state)
        T_over_N = ns["triangles"] / max(1, ns["nodes"])
        L = sw["avg_path_length"]
        LsN = L / np.sqrt(ns["nodes"]) if L else None

        r = dict(n_target=n_target, seed=seed, d_s=d_s, T_over_N=T_over_N,
                 L=L, LsN=LsN, t_vals=t_v, p_vals=p_v,
                 **ns, sw_clustering=sw["clustering"])
        all_runs[n_target].append(r)

        L_s = f"{L:.2f}" if L else "n/a"
        LsN_s = f"{LsN:.3f}" if LsN else "n/a"
        print(f"{n_target:>5}  {seed:>5}  {ns['nodes']:>5}  {d_s:>6.3f}  "
              f"{T_over_N:>5.2f}  {L_s:>6}  {LsN_s:>6}  {ns['clustering']:>8.3f}")

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Aggregate statistics ───────────────────────────────────────────────────────
print("\n── FSS Summary ────────────────────────────────────────────────────")
print(f"{'N*':>5}  {'<N>':>6}  {'d_s mean±std':>14}  "
      f"{'T/N':>10}  {'L/√N':>10}  {'cluster':>10}")
print("─" * 65)

summary = []
for n_target, runs in all_runs.items():
    N_actual = np.mean([r["nodes"] for r in runs])
    ds_m  = np.mean([r["d_s"] for r in runs])
    ds_s  = np.std([r["d_s"] for r in runs])
    tn_m  = np.mean([r["T_over_N"] for r in runs])
    cl_m  = np.mean([r["sw_clustering"] for r in runs])
    lsn   = [r["LsN"] for r in runs if r["LsN"] is not None]
    lsn_m = np.mean(lsn) if lsn else float("nan")
    lsn_s = np.std(lsn) if lsn else 0.0
    summary.append(dict(n_target=n_target, N=N_actual, ds_m=ds_m, ds_s=ds_s,
                        tn_m=tn_m, lsn_m=lsn_m, lsn_s=lsn_s, cl_m=cl_m))
    print(f"{n_target:>5}  {N_actual:>6.0f}  {ds_m:.3f}±{ds_s:.3f}         "
          f"{tn_m:>6.2f}±      {lsn_m:>6.3f}±{lsn_s:.3f}   {cl_m:>6.3f}")

# ── Extrapolation: fit d_s(N) = d_inf - a/N^b ─────────────────────────────────
Ns   = np.array([s["N"] for s in summary])
ds_m = np.array([s["ds_m"] for s in summary])
ds_s = np.array([s["ds_s"] for s in summary])

try:
    def model(N, d_inf, a, b):
        return d_inf - a / N**b
    p0 = [2.0, 1.0, 0.5]
    popt, pcov = curve_fit(model, Ns, ds_m, p0=p0, sigma=ds_s+0.001,
                           bounds=([0, 0, 0.01], [10, 100, 3]))
    d_inf, a, b = popt
    print(f"\nFit d_s(N) = d_inf − a/N^b:")
    print(f"  d_inf = {d_inf:.3f}  (extrapolated dimension at N→∞)")
    print(f"  a     = {a:.3f}")
    print(f"  b     = {b:.3f}  (convergence rate)")
    fit_label = f"fit: {d_inf:.2f} − {a:.2f}/N^{b:.2f}  →  d_∞={d_inf:.2f}"
    fit_success = True
except Exception as e:
    print(f"\nFit failed: {e}")
    fit_success = False
    d_inf = float("nan")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor="#0f0f23")

    # 1. d_s(N) — main FSS plot
    ax = axes[0]; ax.set_facecolor("#1a1a2e")
    ax.errorbar(Ns, ds_m, yerr=ds_s, fmt="o-", color="cyan", linewidth=2,
                markersize=9, capsize=6, label="d_s(N)")
    if fit_success:
        N_extrap = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()*4), 100)
        ax.plot(N_extrap, model(N_extrap, d_inf, a, b), "--", color="gold",
                linewidth=1.5, label=fit_label)
        ax.axhline(d_inf, color="gold", linestyle=":", alpha=0.5,
                   label=f"d_∞ = {d_inf:.2f}")
    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5, alpha=0.7, label="d=2")
    ax.axhline(1.0, color="#555", linestyle=":", alpha=0.5, label="d=1")
    ax.set_xscale("log")
    ax.set_xlabel("Equilibrium node count N", color="white", fontsize=11)
    ax.set_ylabel("Spectral dimension d_s", color="white", fontsize=11)
    ax.set_title("Finite-Size Scaling of d_s\n(extrapolation to N→∞)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=8); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    # 2. L/√N — geodesic scaling test
    ax = axes[1]; ax.set_facecolor("#1a1a2e")
    lsn_m = np.array([s["lsn_m"] for s in summary])
    lsn_s = np.array([s["lsn_s"] for s in summary])
    ax.errorbar(Ns, lsn_m, yerr=lsn_s, fmt="s-", color="coral", linewidth=2,
                markersize=9, capsize=6, label="L/√N")
    mean_lsn = np.nanmean(lsn_m)
    ax.axhline(mean_lsn, color="gold", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_lsn:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("N", color="white", fontsize=11)
    ax.set_ylabel("L / √N", color="white", fontsize=11)
    ax.set_title("Geodesic Scaling: L/√N vs N\n(constant → true 2D; drifting → not 2D)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")
    cv = np.nanstd(lsn_m) / np.nanmean(lsn_m) if np.nanmean(lsn_m) > 0 else float("nan")
    ax.text(0.05, 0.05, f"CV = {cv:.3f}\n({'stable' if cv < 0.1 else 'drifting'})",
            transform=ax.transAxes, color="white", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

    # 3. T/N and clustering vs N
    ax = axes[2]; ax.set_facecolor("#1a1a2e")
    tn = np.array([s["tn_m"] for s in summary])
    cl = np.array([s["cl_m"] for s in summary])
    ax2 = ax.twinx()
    ax.plot(Ns, tn, "o-", color="cyan", linewidth=2, markersize=9, label="T/N")
    ax2.plot(Ns, cl, "s--", color="magenta", linewidth=2, markersize=8, label="clustering")
    ax.axhline(2.0, color="cyan", linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("N", color="white", fontsize=11)
    ax.set_ylabel("T/N", color="cyan", fontsize=11)
    ax2.set_ylabel("Clustering", color="magenta", fontsize=11)
    ax.set_title("Topological density T/N and clustering\n(self-similar if constant)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="cyan"); ax2.tick_params(colors="magenta")
    ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15, color="white")

    plt.suptitle(
        f"RELATE — Finite-Size Scaling  (d_∞ = {d_inf:.2f})" if fit_success
        else "RELATE — Finite-Size Scaling",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("finite_size_scaling.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved finite_size_scaling.png")
    plt.show()
