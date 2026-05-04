"""
examples/initial_topology.py
==============================
Does the 2D attractor depend on initial conditions?

Tests three starting topologies with identical energy parameters:

  DEFAULT  — 2-node pair (standard RELATE seed)
  TORUS    — k×k periodic grid (inherently 2D, d_s=2 by construction)
  SPARSE   — Erdős–Rényi G(N, p) with p chosen so avg_degree ≈ 4

Prediction (universality hypothesis):
  All three converge to the same equilibrium statistics —
  same d_s, T/N, L/√N — regardless of starting topology.
  If confirmed, the 2D attractor is a property of the energy
  functional, not an artefact of the seed.

Usage
-----
    python examples/initial_topology.py
"""

import copy
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from relate import RealitySimulation
from relate.measures import network_summary, small_world_metrics, spectral_dimension
from relate.physics import compute_local_curvature_field, serialize_state
from relate.state import RealityState

# ── Parameters ────────────────────────────────────────────────────────────────
BETA            = 30.0
TOPOLOGY_REWARD = 3.5
SIZE_PENALTY    = 0.004
STEPS           = 300
I_INTERVAL      = 20    # recompute I every 20 steps — 20× faster
SEEDS           = [42, 7, 123]
TORUS_K         = 6     # 6×6 = 36-node torus
SPARSE_N        = 36    # same size as torus for fair comparison
SPARSE_P        = 4 / (SPARSE_N - 1)  # avg_degree ≈ 4


def nx_to_state(G: nx.Graph) -> RealityState:
    """Convert a NetworkX graph to a RealityState with remapped integer node IDs."""
    state = RealityState()
    node_map = {}
    for old_id in G.nodes():
        new_id = state.add_node()
        node_map[old_id] = new_id
    for u, v in G.edges():
        state.add_edge(node_map[u], node_map[v], weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


# ── Run three topologies, multiple seeds ─────────────────────────────────────
configs = {
    "default": lambda seed: None,
    "torus":   lambda seed: nx_to_state(nx.grid_2d_graph(TORUS_K, TORUS_K, periodic=True)),
    "sparse":  lambda seed: nx_to_state(nx.gnp_random_graph(SPARSE_N, SPARSE_P, seed=seed)),
}

all_results = {name: [] for name in configs}

print("RELATE — Initial Topology Sensitivity")
print(f"β={BETA}  topology_reward={TOPOLOGY_REWARD}  {STEPS} steps  I_interval={I_INTERVAL}\n")

t0 = time.time()
for name, make_state in configs.items():
    print(f"── {name.upper()} ─────────────────────────────────────────────")
    for seed in SEEDS:
        np.random.seed(seed); random.seed(seed)
        init = make_state(seed)

        sim = RealitySimulation(
            beta=BETA, topology_reward=TOPOLOGY_REWARD,
            size_penalty=SIZE_PENALTY,
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

        r = dict(seed=seed, d_s=d_s, T_over_N=T_over_N, L=L, LsN=LsN,
                 t_vals=t_v, p_vals=p_v, **ns, sw_clustering=sw["clustering"])
        all_results[name].append(r)

        L_s = f"{L:.2f}" if L else "n/a"
        LsN_s = f"{LsN:.3f}" if LsN else "n/a"
        print(f"  seed={seed}  d_s={d_s:.3f}  N={ns['nodes']}  "
              f"T/N={T_over_N:.2f}  L={L_s}  L/√N={LsN_s}  "
              f"cluster={ns['clustering']:.3f}")
    print()

print(f"Completed in {time.time()-t0:.1f}s")

# ── Summary: mean ± std across seeds ──────────────────────────────────────────
print("\n── Summary (mean ± std across seeds) ────────────────────────────")
print(f"{'Topology':>10}  {'d_s':>12}  {'T/N':>10}  {'L/√N':>10}  {'cluster':>10}")
print("─" * 60)
for name, results in all_results.items():
    ds_m,  ds_s  = np.mean([r["d_s"] for r in results]),      np.std([r["d_s"] for r in results])
    tn_m,  tn_s  = np.mean([r["T_over_N"] for r in results]),  np.std([r["T_over_N"] for r in results])
    cl_m,  cl_s  = np.mean([r["sw_clustering"] for r in results]), np.std([r["sw_clustering"] for r in results])
    lsn = [r["LsN"] for r in results if r["LsN"] is not None]
    lsn_m = np.mean(lsn) if lsn else float("nan")
    lsn_s = np.std(lsn) if lsn else float("nan")
    print(f"{name:>10}  {ds_m:.3f}±{ds_s:.3f}   {tn_m:.2f}±{tn_s:.2f}   "
          f"{lsn_m:.3f}±{lsn_s:.3f}   {cl_m:.3f}±{cl_s:.3f}")

# ── Test universality: are d_s distributions overlapping? ─────────────────────
from scipy import stats as scipy_stats
ds_groups = {n: [r["d_s"] for r in rs] for n, rs in all_results.items()}
names = list(ds_groups.keys())
if len(names) >= 2:
    ks, pval = scipy_stats.ks_2samp(ds_groups[names[0]], ds_groups[names[1]])
    print(f"\nKS test default vs torus:  statistic={ks:.3f}  p={pval:.3f}")
    ks2, pval2 = scipy_stats.ks_2samp(ds_groups[names[0]], ds_groups[names[2]])
    print(f"KS test default vs sparse: statistic={ks2:.3f}  p={pval2:.3f}")
    print("p > 0.05 → distributions are statistically indistinguishable (universality)")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="#0f0f23")
    colors = {"default": "cyan", "torus": "lime", "sparse": "coral"}
    t_ref = np.logspace(-1, np.log10(50), 60)

    # Panel 1: return probability curves
    ax = axes[0]; ax.set_facecolor("#1a1a2e")
    for d_ref, col, ls, lbl in [(1,"#888",":","d=1"),(2,"gold","--","d=2"),(3,"magenta","-.","d=3")]:
        ax.loglog(t_ref, 0.3*(1.0/t_ref)**(d_ref/2), color=col, ls=ls,
                  linewidth=1.2, alpha=0.5, label=lbl)
    for name, results in all_results.items():
        for i, r in enumerate(results):
            if len(r["t_vals"]) > 0:
                lbl = f"{name} (s={r['seed']})" if i == 0 else "_"
                ax.loglog(r["t_vals"], r["p_vals"], color=colors[name],
                          linewidth=1.5 if i == 0 else 0.8,
                          alpha=1.0 if i == 0 else 0.4, label=lbl)
    ax.axvspan(0.5, 15, alpha=0.08, color="white")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Diffusion time t", color="white")
    ax.set_ylabel("Return probability p(t)", color="white")
    ax.set_title("Return probability p(t)\n(curves should overlap → universality)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")

    # Panel 2: d_s by topology
    ax = axes[1]; ax.set_facecolor("#1a1a2e")
    for i, (name, results) in enumerate(all_results.items()):
        ds_vals = [r["d_s"] for r in results]
        ax.scatter([i]*len(ds_vals), ds_vals, color=colors[name], s=100,
                   edgecolors="white", zorder=3, label=name)
        ax.errorbar(i, np.mean(ds_vals), yerr=np.std(ds_vals),
                    fmt="none", color=colors[name], capsize=8, linewidth=2)
    ax.axhline(2.0, color="gold", linestyle="--", linewidth=1.5, alpha=0.7, label="d_s=2")
    ax.set_xticks(range(len(all_results)))
    ax.set_xticklabels(list(all_results.keys()), color="white")
    ax.set_ylabel("Spectral dimension d_s", color="white")
    ax.set_title("d_s by initial topology\n(should converge to same value)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    # Panel 3: T/N by topology
    ax = axes[2]; ax.set_facecolor("#1a1a2e")
    for i, (name, results) in enumerate(all_results.items()):
        tn_vals = [r["T_over_N"] for r in results]
        ax.scatter([i]*len(tn_vals), tn_vals, color=colors[name], s=100,
                   edgecolors="white", zorder=3, label=name)
        ax.errorbar(i, np.mean(tn_vals), yerr=np.std(tn_vals),
                    fmt="none", color=colors[name], capsize=8, linewidth=2)
    ax.axhline(2.0, color="gold", linestyle="--", linewidth=1.5, alpha=0.7, label="T/N=2")
    ax.set_xticks(range(len(all_results)))
    ax.set_xticklabels(list(all_results.keys()), color="white")
    ax.set_ylabel("Topological density T/N", color="white")
    ax.set_title("T/N by initial topology",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    plt.suptitle("RELATE — Initial Topology Sensitivity (universality test)",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("initial_topology.png", dpi=120, bbox_inches="tight", facecolor="#0f0f23")
    print("\nSaved initial_topology.png")
    plt.show()
