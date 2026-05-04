"""
examples/fss_torus.py
======================
Clean finite-size scaling with torus initialisation and clique-filter.

Fixes two problems from finite_size_scaling.py:
  1. Bimodal attractor: seed=0 consistently collapses to a clique.
     → Use torus initialisation (proven stable in initial_topology.py).
  2. Averaging clique and normal-phase runs hides the real FSS signal.
     → Discard any run with T/N > 10 (clique criterion).

Design
------
  Initial state : k×k torus, k chosen so the torus has ~N*/4 nodes
                  (gives room to grow to equilibrium)
  N* targets    : 50, 100, 200, 400, 800  (size_penalty = 0.8/(2·N*))
  Seeds         : 8 per N*
  Steps         : 400
  I_interval    : 10   (10× speedup vs recomputing every step)
  Clique filter : discard runs with T/N > 10

Output
------
  Fit  d_s(N) = d_inf − a/N^b   on normal-phase runs only.
  Plot: FSS curve + L/√N stability + T/N by N*.

Usage
-----
    python examples/fss_torus.py
"""

import math
import random
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from relate import RealitySimulation
from relate.measures import network_summary, small_world_metrics, spectral_dimension
from relate.physics import compute_local_curvature_field
from relate.state import RealityState

# ── Parameters ────────────────────────────────────────────────────────────────
BETA            = 30.0
TOPOLOGY_REWARD = 3.5
VACUUM          = 0.8
STEPS           = 400
I_INTERVAL      = 10
N_SEEDS         = 8
CLIQUE_THRESHOLD = 10.0   # T/N > this → clique, discard

N_TARGETS    = [50, 100, 200, 400, 800]
SIZE_PENALTY = [VACUUM / (2 * n) for n in N_TARGETS]


def make_torus_state(n_target: int) -> RealityState:
    """Build a periodic 2D grid with ~n_target/4 nodes."""
    k = max(3, math.isqrt(n_target // 4))
    G = nx.grid_2d_graph(k, k, periodic=True)
    state = RealityState()
    node_map = {}
    for old in G.nodes():
        node_map[old] = state.add_node()
    for u, v in G.edges():
        state.add_edge(node_map[u], node_map[v], weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


# ── Run sweep ─────────────────────────────────────────────────────────────────
print("RELATE — Clean FSS with Torus Init + Clique Filter")
print(f"β={BETA}  topology_reward={TOPOLOGY_REWARD}  {STEPS} steps  "
      f"{N_SEEDS} seeds/point  I_interval={I_INTERVAL}")
print(f"Clique filter: discard T/N > {CLIQUE_THRESHOLD}\n")
print(f"{'N*':>5}  {'seed':>5}  {'N':>5}  {'d_s':>6}  "
      f"{'T/N':>6}  {'L/√N':>6}  {'cluster':>8}  {'status':>8}")
print("─" * 65)

all_runs: Dict[int, List[Dict]] = {n: [] for n in N_TARGETS}
t0 = time.time()

for n_target, sp in zip(N_TARGETS, SIZE_PENALTY):
    k_torus = max(3, math.isqrt(n_target // 4))
    print(f"  [N*={n_target}  torus={k_torus}×{k_torus}={k_torus**2} nodes  "
          f"size_penalty={sp:.5f}]")
    for seed in range(N_SEEDS):
        np.random.seed(seed); random.seed(seed)
        init = make_torus_state(n_target)

        sim = RealitySimulation(
            beta=BETA, topology_reward=TOPOLOGY_REWARD,
            vacuum=VACUUM, size_penalty=sp,
            i_update_interval=I_INTERVAL,
            initial_state=init,
            record_all_states=False,
        )
        sim.run(steps=STEPS, verbose=False)

        d_s, t_v, p_v = spectral_dimension(sim.state)
        ns = network_summary(sim.state)
        sw = small_world_metrics(sim.state)
        T_over_N = ns["triangles"] / max(1, ns["nodes"])
        L = sw["avg_path_length"]
        LsN = L / np.sqrt(ns["nodes"]) if L else None
        is_clique = T_over_N > CLIQUE_THRESHOLD
        status = "CLIQUE" if is_clique else "ok"

        r = dict(n_target=n_target, seed=seed, d_s=d_s, T_over_N=T_over_N,
                 L=L, LsN=LsN, t_vals=t_v, p_vals=p_v, is_clique=is_clique,
                 **ns, sw_clustering=sw["clustering"])
        all_runs[n_target].append(r)

        LsN_s = f"{LsN:.3f}" if LsN else "n/a"
        print(f"{'':>5}  {seed:>5}  {ns['nodes']:>5}  {d_s:>6.3f}  "
              f"{T_over_N:>6.2f}  {LsN_s:>6}  {ns['clustering']:>8.3f}  {status:>8}")
    print()

print(f"Completed in {time.time()-t0:.1f}s")

# ── Filter and aggregate ───────────────────────────────────────────────────────
print("\n── FSS Summary (normal-phase runs only, T/N ≤ {}) ────────────────".format(CLIQUE_THRESHOLD))
print(f"{'N*':>5}  {'n_ok/n_tot':>10}  {'<N>':>6}  {'d_s':>14}  "
      f"{'L/√N':>12}  {'T/N':>8}")
print("─" * 65)

summary = []
for n_target, runs in all_runs.items():
    ok = [r for r in runs if not r["is_clique"]]
    n_ok = len(ok)
    n_tot = len(runs)
    if n_ok == 0:
        print(f"{n_target:>5}  {n_ok}/{n_tot:>9}  (all cliques — skipped)")
        continue

    N_m   = np.mean([r["nodes"] for r in ok])
    ds_m  = np.mean([r["d_s"] for r in ok])
    ds_s  = np.std([r["d_s"] for r in ok])
    lsn   = [r["LsN"] for r in ok if r["LsN"] is not None]
    lsn_m = np.mean(lsn) if lsn else float("nan")
    lsn_s = np.std(lsn) if lsn else 0.0
    tn_m  = np.mean([r["T_over_N"] for r in ok])
    tn_s  = np.std([r["T_over_N"] for r in ok])
    cl_m  = np.mean([r["sw_clustering"] for r in ok])

    summary.append(dict(n_target=n_target, N=N_m, n_ok=n_ok, n_tot=n_tot,
                        ds_m=ds_m, ds_s=ds_s, lsn_m=lsn_m, lsn_s=lsn_s,
                        tn_m=tn_m, tn_s=tn_s, cl_m=cl_m,
                        ok_runs=ok))
    print(f"{n_target:>5}  {n_ok}/{n_tot:>9}  {N_m:>6.0f}  "
          f"{ds_m:.3f}±{ds_s:.3f}       {lsn_m:.3f}±{lsn_s:.3f}   "
          f"{tn_m:.2f}±{tn_s:.2f}")

if len(summary) < 3:
    print("\nNot enough non-clique points for FSS fit.")
    fit_success = False
    d_inf = float("nan")
else:
    # ── FSS fit ───────────────────────────────────────────────────────────────
    Ns   = np.array([s["N"]   for s in summary])
    ds_m = np.array([s["ds_m"] for s in summary])
    ds_s = np.array([s["ds_s"] for s in summary])

    def fss_model(N, d_inf, a, b):
        return d_inf - a / N**b

    try:
        popt, _ = curve_fit(fss_model, Ns, ds_m, p0=[2.5, 2.0, 0.5],
                            sigma=ds_s + 0.01,
                            bounds=([0, 0, 0.01], [6, 50, 3]))
        d_inf, a, b = popt
        fit_success = True
        print(f"\nFSS fit  d_s(N) = d_inf − a/N^b:")
        print(f"  d_inf = {d_inf:.3f}  ← extrapolated dimension N→∞")
        print(f"  a     = {a:.3f}")
        print(f"  b     = {b:.3f}  (convergence rate)")
    except Exception as e:
        print(f"\nFSS fit failed: {e}")
        fit_success = False
        d_inf = float("nan")

# ── L/√N stability check ──────────────────────────────────────────────────────
lsn_all = [r["LsN"] for s in summary for r in s["ok_runs"] if r["LsN"] is not None]
if lsn_all:
    lsn_cv = np.std(lsn_all) / np.mean(lsn_all)
    print(f"\nL/√N:  mean={np.mean(lsn_all):.3f}  std={np.std(lsn_all):.3f}  "
          f"CV={lsn_cv:.3f}  ({'STABLE → 2D' if lsn_cv < 0.10 else 'drifting'})")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor="#0f0f23")

    # 1. FSS: d_s(N) ───────────────────────────────────────────────────────────
    ax = axes[0]; ax.set_facecolor("#1a1a2e")

    # scatter individual runs
    for s in summary:
        for r in s["ok_runs"]:
            ax.scatter(r["nodes"], r["d_s"], color="cyan", s=40,
                       alpha=0.5, edgecolors="none")
    # mean ± std
    if summary:
        Ns_plot = np.array([s["N"] for s in summary])
        ds_plot = np.array([s["ds_m"] for s in summary])
        ds_err  = np.array([s["ds_s"] for s in summary])
        ax.errorbar(Ns_plot, ds_plot, yerr=ds_err, fmt="o", color="cyan",
                    markersize=10, capsize=7, linewidth=2, label="mean±std")

    if fit_success:
        N_range = np.linspace(20, max(Ns_plot)*3, 200)
        ax.plot(N_range, fss_model(N_range, d_inf, a, b), "--", color="gold",
                linewidth=2, label=f"d_s={d_inf:.2f}−{a:.2f}/N^{b:.2f}")
        ax.axhline(d_inf, color="gold", linestyle=":", alpha=0.5,
                   label=f"d_∞ = {d_inf:.2f}")

    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5,
               alpha=0.7, label="d=2")
    ax.set_xlabel("Equilibrium node count N", color="white", fontsize=11)
    ax.set_ylabel("Spectral dimension d_s", color="white", fontsize=11)
    ax.set_title("Clean FSS: d_s(N) — torus init, clique-filtered",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    # 2. L/√N stability ────────────────────────────────────────────────────────
    ax = axes[1]; ax.set_facecolor("#1a1a2e")
    palette = plt.cm.plasma(np.linspace(0.15, 0.9, len(summary)))
    for s, col in zip(summary, palette):
        for r in s["ok_runs"]:
            if r["LsN"] is not None:
                ax.scatter(r["nodes"], r["LsN"], color=col, s=50,
                           alpha=0.7, edgecolors="white", linewidths=0.3)

    if lsn_all:
        ax.axhline(np.mean(lsn_all), color="gold", linestyle="--", linewidth=2,
                   label=f"mean = {np.mean(lsn_all):.3f}")
        ax.axhspan(np.mean(lsn_all)-np.std(lsn_all),
                   np.mean(lsn_all)+np.std(lsn_all),
                   alpha=0.12, color="gold")
        ax.text(0.05, 0.92, f"CV = {lsn_cv:.3f}",
                transform=ax.transAxes, color="white", fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

    ax.set_xlabel("N", color="white", fontsize=11)
    ax.set_ylabel("L / √N", color="white", fontsize=11)
    ax.set_title("Geodesic scaling L/√N\n(constant CV<0.10 → 2D; drifting → not 2D)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=10); ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    # 3. T/N and clustering by N* ──────────────────────────────────────────────
    ax = axes[2]; ax.set_facecolor("#1a1a2e")
    ax2 = ax.twinx()
    if summary:
        nt = [s["n_target"] for s in summary]
        tn  = [s["tn_m"]  for s in summary]
        tns = [s["tn_s"]  for s in summary]
        cl  = [s["cl_m"]  for s in summary]
        frac = [s["n_ok"]/s["n_tot"] for s in summary]

        ax.errorbar(nt, tn, yerr=tns, fmt="o-", color="cyan", linewidth=2,
                    markersize=9, capsize=6, label="T/N")
        ax2.plot(nt, cl, "s--", color="magenta", linewidth=2,
                 markersize=8, label="clustering")
        for x, f in zip(nt, frac):
            ax.text(x, max(tn)*1.05, f"{f:.0%}", ha="center",
                    color="white", fontsize=9)

    ax.set_xlabel("N* target", color="white", fontsize=11)
    ax.set_ylabel("T/N", color="cyan", fontsize=11)
    ax2.set_ylabel("Clustering", color="magenta", fontsize=11)
    ax.set_title("Topological density T/N and clustering\n(% above bar = fraction non-clique)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="cyan"); ax2.tick_params(colors="magenta")
    ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15, color="white")

    title = (f"RELATE — Clean FSS  (d_∞ = {d_inf:.2f})" if fit_success
             else "RELATE — Clean FSS (torus init, clique-filtered)")
    plt.suptitle(title, fontsize=14, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("fss_torus.png", dpi=120, bbox_inches="tight", facecolor="#0f0f23")
    print("\nSaved fss_torus.png")
    plt.show()
