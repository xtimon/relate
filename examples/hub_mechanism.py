"""
examples/hub_mechanism.py
==========================
Isolate the hub-formation mechanism in a reduced RELATE model.

Full RELATE has six move types.  Here we strip it to a minimal model
containing only the two moves responsible for hub formation:

  EXPAND    — adds a node connected to both endpoints of an edge
  INTEGRATE — collapses a triangle into a single particle node that
               inherits ALL external connections (the critical step)

Hypotheses
----------
  H1: The degree distribution P(k) follows a power law k^(-γ) with γ ≈ 3,
      as in the Barabási–Albert (BA) preferential-attachment model.
  H2: The richest-get-richer dynamic is driven by integrate: nodes with
      more connections appear in more triangles and are more likely to be
      chosen as integration targets, giving them even more connections.
  H3: The time-to-first-hub t* scales as t* ∝ N^α (hub forms later in
      larger graphs), consistent with BA theory.

Comparison
----------
  We run both the reduced RELATE model and a pure BA model (m=2) to the
  same final N and compare degree-distribution exponents.

Usage
-----
    python examples/hub_mechanism.py
"""

import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats as scipy_stats

from relate.dynamics import MoveType, ScoringParams, apply_move, generate_moves, score_move_cheap
from relate.measures import power_law_exponent
from relate.physics import (
    compute_accumulated_complexity,
    compute_curvature,
    compute_local_curvature_field,
    serialize_state,
)
from relate.state import RealityState

# ── Reduced model: only expand + integrate ────────────────────────────────────
def generate_moves_reduced(state: RealityState):
    """Only expand and integrate — minimal model for hub formation."""
    moves = []
    if state.edge_count > 0:
        edges = list(state.graph.edges())
        for u, v in random.sample(edges, min(15, len(edges))):
            moves.append((MoveType.EXPAND, (u, v), 1.0))
    if state.triple_count > 0:
        triples = list(state.triples)
        for t in random.sample(triples, min(10, len(triples))):
            moves.append((MoveType.INTEGRATE, t, 1.0))
    if not moves:
        moves.append((MoveType.SEED, (), 1.0))
    return moves


def run_reduced(n_steps: int, alpha=0.3, beta=3.0, gamma=0.02,
                vacuum=0.8, connectivity=3.0, size_penalty=0.004,
                topology_reward=0.0, seed=42):
    """Run the minimal model and return (state, degree_history)."""
    np.random.seed(seed); random.seed(seed)
    state = RealityState()
    u = state.add_node(); v = state.add_node()
    state.add_edge(u, v, weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    initial_ser = serialize_state(state)

    hub_times = []   # (time, max_degree) when a new max is set
    cur_max = 0

    for step in range(n_steps):
        C = compute_curvature(state)
        U = compute_accumulated_complexity(state, initial_ser)
        lcs = state.largest_component_size()
        lcs_frac = lcs / max(1, state.node_count)
        T = state.triple_count
        N = state.node_count

        E = (alpha*C - beta*lcs_frac + gamma*U
             - vacuum*N - connectivity*lcs_frac
             + size_penalty*N**2
             - topology_reward*T/max(1,N))

        moves = generate_moves_reduced(state)
        scores, cands = [], {}
        params = ScoringParams(alpha=alpha, beta=beta, gamma=gamma,
                               vacuum=vacuum, connectivity=connectivity,
                               size_penalty=size_penalty,
                               topology_reward=topology_reward)
        for i, move in enumerate(moves):
            sc, ns = score_move_cheap(state, move, E, params, C, lcs, U)
            scores.append(sc)
            if ns is not None: cands[i] = ns

        total = sum(scores)
        probs = [s/total for s in scores] if total > 0 else [1/len(scores)]*len(scores)
        idx = int(np.random.choice(len(moves), p=probs))
        state = cands[idx] if idx in cands else apply_move(state, moves[idx])

        new_max = max((d for _, d in state.graph.degree()), default=0)
        if new_max > cur_max:
            cur_max = new_max
            hub_times.append((step, new_max, state.node_count))

    return state, hub_times


# ── Main experiment ───────────────────────────────────────────────────────────
N_STEPS   = 2000
N_SEEDS   = 5
SIZE_PEN  = 0.004   # N* ≈ 100

print("RELATE — Hub Mechanism: Reduced Model")
print(f"Moves: expand + integrate only  |  {N_STEPS} steps  |  {N_SEEDS} seeds\n")

all_degrees = []
all_hub_times = []
all_gammas = []

t0 = time.time()
for seed in range(N_SEEDS):
    state, hub_times = run_reduced(N_STEPS, size_penalty=SIZE_PEN, seed=seed)
    degs = [d for _, d in state.graph.degree() if d > 0]
    gamma_hat = power_law_exponent(state, k_min_percentile=60.0)
    all_degrees.extend(degs)
    all_hub_times.append(hub_times)
    all_gammas.append(gamma_hat)

    N = state.node_count
    avg = 2*state.edge_count/max(1,N)
    max_d = max(degs) if degs else 0
    first_hub_t = hub_times[0][0] if hub_times else None
    print(f"  seed={seed}  N={N}  avg={avg:.1f}  max={max_d}  "
          f"ratio={max_d/max(0.01,avg):.1f}x  γ={gamma_hat:.2f}  "
          f"first_hub_t={first_hub_t}")

print(f"\nCompleted in {time.time()-t0:.1f}s")
print(f"Mean γ = {np.mean(all_gammas):.3f} ± {np.std(all_gammas):.3f}")
print(f"BA model predicts γ = 3.0  (for m=2 attachment)")

# ── BA model comparison ───────────────────────────────────────────────────────
print("\n── BA model (Barabási–Albert, m=2) ─────────────────────────────────")
ba_degs = []
for seed in range(N_SEEDS):
    final_N = int(0.8 / (2 * SIZE_PEN))  # N* ≈ 100
    G_ba = nx.barabasi_albert_graph(final_N, m=2, seed=seed)
    ba_degs.extend([d for _, d in G_ba.degree()])
    dd = Counter([d for _, d in G_ba.degree()])
    top5 = sorted(dd.keys(), reverse=True)[:5]
    avg = 2*G_ba.number_of_edges()/max(1,G_ba.number_of_nodes())
    max_d = max(d for _, d in G_ba.degree())
    print(f"  seed={seed}  N={G_ba.number_of_nodes()}  avg={avg:.1f}  "
          f"max={max_d}  ratio={max_d/max(0.01,avg):.1f}x")

# ── KS test: are RELATE and BA distributions the same? ───────────────────────
ks, pval = scipy_stats.ks_2samp(all_degrees, ba_degs)
print(f"\nKS test RELATE vs BA: statistic={ks:.3f}  p={pval:.4f}")
print(f"→ {'INDISTINGUISHABLE' if pval > 0.05 else 'DIFFERENT'} distributions")

# ── Cumulative degree distribution (CCDF) P(K ≥ k) ──────────────────────────
def ccdf(degs):
    dd = Counter(degs)
    ks = sorted(dd.keys())
    total = sum(dd.values())
    cum = 0
    xs, ys = [], []
    for k in reversed(ks):
        cum += dd[k]
        xs.append(k); ys.append(cum / total)
    return list(reversed(xs)), list(reversed(ys))

# ── Hub trajectory ────────────────────────────────────────────────────────────
print("\n── Hub degree trajectory (seed=0) ──────────────────────────────────")
for step, max_d, N in all_hub_times[0][:15]:
    print(f"  t={step:5d}  max_deg={max_d:4d}  N={N}")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f0f23")

    # 1. CCDF — RELATE reduced vs BA
    ax = axes[0, 0]; ax.set_facecolor("#1a1a2e")
    r_xs, r_ys = ccdf(all_degrees)
    b_xs, b_ys = ccdf(ba_degs)
    ax.loglog(r_xs, r_ys, "o-", color="cyan", linewidth=2, markersize=5,
              label=f"RELATE reduced  (γ={np.mean(all_gammas):.2f})")
    ax.loglog(b_xs, b_ys, "s--", color="coral", linewidth=2, markersize=5,
              label="BA model (γ=3.0 theory)")
    # Slope reference lines
    k_ref = np.logspace(0, 1.5, 30)
    for gamma_ref, col, lbl in [(3.0, "gold", "k^{-3}"), (2.5, "#888", "k^{-2.5}")]:
        norm = r_ys[0] * r_xs[0]**(gamma_ref-1)
        ax.loglog(k_ref, norm * k_ref**(-gamma_ref+1), ":", color=col,
                  alpha=0.6, label=lbl)
    ax.set_xlabel("Degree k", color="white"); ax.set_ylabel("P(K ≥ k)", color="white")
    ax.set_title("CCDF: RELATE reduced vs BA model",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    # 2. Hub degree trajectory
    ax = axes[0, 1]; ax.set_facecolor("#1a1a2e")
    palette = plt.cm.plasma(np.linspace(0.2, 0.9, N_SEEDS))
    for i, hub_times in enumerate(all_hub_times):
        if hub_times:
            ts = [h[0] for h in hub_times]
            ds = [h[1] for h in hub_times]
            ax.step(ts, ds, color=palette[i], linewidth=1.5,
                    label=f"seed={i}", where="post")
    ax.set_xlabel("Step", color="white"); ax.set_ylabel("Max degree", color="white")
    ax.set_title("Hub degree growth trajectory",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    # 3. γ distribution
    ax = axes[1, 0]; ax.set_facecolor("#1a1a2e")
    ax.bar(range(N_SEEDS), all_gammas, color="cyan", alpha=0.8, edgecolor="none")
    ax.axhline(3.0, color="coral", linestyle="--", linewidth=2, label="BA γ=3")
    ax.axhline(np.mean(all_gammas), color="gold", linestyle=":", linewidth=2,
               label=f"RELATE mean γ={np.mean(all_gammas):.2f}")
    ax.set_xticks(range(N_SEEDS))
    ax.set_xticklabels([f"seed={i}" for i in range(N_SEEDS)], color="white")
    ax.set_ylabel("Power-law exponent γ", color="white")
    ax.set_title(f"γ per seed  (BA predicts γ=3)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white", axis="y")

    # 4. Time to first hub vs N (across seeds)
    ax = axes[1, 1]; ax.set_facecolor("#1a1a2e")
    first_Ns = [ht[0][2] if ht else None for ht in all_hub_times]
    first_ts = [ht[0][0] if ht else None for ht in all_hub_times]
    valid = [(n, t) for n, t in zip(first_Ns, first_ts) if n and t]
    if valid:
        vNs, vts = zip(*valid)
        ax.scatter(vNs, vts, color="gold", s=100, edgecolors="white", zorder=3)
        if len(valid) >= 2:
            slope, intercept, r, p, _ = scipy_stats.linregress(np.log(vNs), np.log(vts))
            N_fit = np.linspace(min(vNs), max(vNs)*1.5, 30)
            ax.plot(N_fit, np.exp(intercept)*N_fit**slope, "--", color="coral",
                    linewidth=1.5, label=f"t* ∝ N^{slope:.2f}")
    ax.set_xlabel("N at first hub appearance", color="white")
    ax.set_ylabel("Time t* to first hub", color="white")
    ax.set_title("Time-to-first-hub vs N\n(BA predicts t* ∝ N)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    plt.suptitle("RELATE — Hub Mechanism: Reduced Model vs BA",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("hub_mechanism.png", dpi=120, bbox_inches="tight", facecolor="#0f0f23")
    print("\nSaved hub_mechanism.png")
    plt.show()
