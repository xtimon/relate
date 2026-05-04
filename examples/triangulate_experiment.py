"""
examples/triangulate_experiment.py
=====================================
Two-pronged experiment to push spectral dimension d_s toward 2.

Path 1 — High β
---------------
  Sweep β ∈ {3, 10, 20, 50} (no triangulate reward).
  Tests whether integrate-dominated dynamics can densify the graph.

Path 2 — topology_reward + triangulate move
-------------------------------------------
  Fix β=3, sweep topology_reward ∈ {0, 2, 5, 10}.
  The triangulate move (ΔN=0, ΔT≥1) is always in the candidate set.
  topology_reward makes triangulate energetically preferred.

Both paths measure d_s, T/N, and clustering at each parameter value.

Usage
-----
    python examples/triangulate_experiment.py
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

import relate
from relate import RealitySimulation
from relate.measures import (
    network_summary,
    small_world_metrics,
    spectral_dimension,
)

STEPS = 200
SEED  = 42

# ── PATH 1: High β ────────────────────────────────────────────────────────────
print("═" * 60)
print("PATH 1 — High β sweep  (no topology_reward)")
print("═" * 60)
print(f"{'β':>6}  {'d_s':>6}  {'N':>5}  {'T':>5}  {'T/N':>5}  "
      f"{'avg_deg':>7}  {'cluster':>8}  {'integrate':>9}")
print("─" * 65)

path1 = []
for beta in [3.0, 10.0, 20.0, 50.0]:
    np.random.seed(SEED); random.seed(SEED)
    sim = RealitySimulation(
        beta=beta, size_penalty=0.002, record_all_states=False
    )
    sim.run(steps=STEPS, verbose=False)

    d_s, t_v, p_v = spectral_dimension(sim.state)
    ns = network_summary(sim.state)
    sw = small_world_metrics(sim.state)
    integrate_count = sum(1 for h in sim.history if h["move_type"] == "integrate")

    path1.append(dict(
        beta=beta, d_s=d_s, t_vals=t_v, p_vals=p_v,
        integrate_count=integrate_count,
        sw_clustering=sw["clustering"],
        **ns,
    ))
    print(f"{beta:>6.1f}  {d_s:>6.3f}  {ns['nodes']:>5}  {ns['triangles']:>5}  "
          f"{ns['triangles']/max(1,ns['nodes']):>5.2f}  "
          f"{ns['avg_degree']:>7.2f}  {sw['clustering']:>8.3f}  "
          f"{integrate_count:>9}")

# ── PATH 2: topology_reward ───────────────────────────────────────────────────
print()
print("═" * 60)
print("PATH 2 — topology_reward sweep  (triangulate move active)")
print("═" * 60)
print(f"{'reward':>7}  {'d_s':>6}  {'N':>5}  {'T':>5}  {'T/N':>5}  "
      f"{'avg_deg':>7}  {'cluster':>8}  {'triangulate':>11}")
print("─" * 68)

path2 = []
for tr in [0.0, 2.0, 5.0, 10.0]:
    np.random.seed(SEED); random.seed(SEED)
    sim = RealitySimulation(
        beta=3.0, topology_reward=tr, size_penalty=0.002,
        record_all_states=False,
    )
    sim.run(steps=STEPS, verbose=False)

    d_s, t_v, p_v = spectral_dimension(sim.state)
    ns = network_summary(sim.state)
    sw = small_world_metrics(sim.state)
    triangulate_count = sum(1 for h in sim.history if h["move_type"] == "triangulate")

    path2.append(dict(
        topology_reward=tr, d_s=d_s, t_vals=t_v, p_vals=p_v,
        triangulate_count=triangulate_count,
        sw_clustering=sw["clustering"],
        **ns,
    ))
    print(f"{tr:>7.1f}  {d_s:>6.3f}  {ns['nodes']:>5}  {ns['triangles']:>5}  "
          f"{ns['triangles']/max(1,ns['nodes']):>5.2f}  "
          f"{ns['avg_degree']:>7.2f}  {sw['clustering']:>8.3f}  "
          f"{triangulate_count:>11}")

# ── Summary ───────────────────────────────────────────────────────────────────
best1 = max(path1, key=lambda r: r["d_s"])
best2 = max(path2, key=lambda r: r["d_s"])
print()
print(f"Path 1 best: β={best1['beta']}  →  d_s={best1['d_s']:.3f}  T/N={best1['triangles']/max(1,best1['nodes']):.2f}")
print(f"Path 2 best: reward={best2['topology_reward']}  →  d_s={best2['d_s']:.3f}  T/N={best2['triangles']/max(1,best2['nodes']):.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor="#0f0f23")
    fig.suptitle("RELATE — Search for d_s = 2: High β vs topology_reward",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    c1 = plt.cm.plasma(np.linspace(0.2, 0.9, len(path1)))
    c2 = plt.cm.viridis(np.linspace(0.2, 0.9, len(path2)))

    # Reference lines helper
    def _refs(ax, t_ref, p_anchor, t_anchor):
        for d_ref, col, ls, lbl in [(1,"#888",":","d=1"),(2,"lime","--","d=2"),(3,"magenta","-.","d=3")]:
            p_line = p_anchor * (t_anchor / t_ref) ** (d_ref / 2)
            ax.loglog(t_ref, p_line, color=col, linestyle=ls, linewidth=1.2,
                      label=lbl, alpha=0.6)

    t_ref = np.logspace(-1, np.log10(50), 60)

    # ── Row 0: Path 1 ─────────────────────────────────────────────────────────
    # Panel 0,0: return probability
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a2e")
    if path1[0]["t_vals"] is not None and len(path1[0]["t_vals"]) > 5:
        _refs(ax, t_ref, path1[0]["p_vals"][5], path1[0]["t_vals"][5])
    for r, col in zip(path1, c1):
        if len(r["t_vals"]) > 0:
            ax.loglog(r["t_vals"], r["p_vals"], color=col, linewidth=2,
                      label=f"β={r['beta']} d_s={r['d_s']:.2f}")
    ax.axvspan(0.5, 15.0, alpha=0.08, color="white")
    ax.legend(fontsize=8); ax.set_xlabel("t", color="white"); ax.set_ylabel("p(t)", color="white")
    ax.set_title("PATH 1 — Return probability p(t)", color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")

    # Panel 0,1: d_s and T/N vs β
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a2e")
    betas = [r["beta"] for r in path1]
    d_s_1 = [r["d_s"] for r in path1]
    tn_1 = [r["triangles"] / max(1, r["nodes"]) for r in path1]
    ax2 = ax.twinx()
    ax.plot(betas, d_s_1, "o-", color="cyan", linewidth=2, markersize=8, label="d_s")
    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5, alpha=0.7, label="d_s=2 target")
    ax2.plot(betas, tn_1, "s--", color="coral", linewidth=1.5, markersize=7, label="T/N")
    ax.set_xlabel("β", color="white"); ax.set_ylabel("d_s", color="cyan")
    ax2.set_ylabel("T/N", color="coral")
    ax.tick_params(colors="white"); ax2.tick_params(colors="coral")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("d_s and T/N vs β", color="white", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.15, color="white")

    # Panel 0,2: move distribution for path 1
    ax = axes[0, 2]
    ax.set_facecolor("#1a1a2e")
    int_counts = [r["integrate_count"] for r in path1]
    ax.bar(range(len(path1)), int_counts, color="#e74c3c", alpha=0.8)
    for i, v in enumerate(int_counts):
        ax.text(i, v + 0.5, str(v), ha="center", color="white", fontsize=11)
    ax.set_xticks(range(len(path1)))
    ax.set_xticklabels([f"β={r['beta']}" for r in path1], color="white")
    ax.set_ylabel("Count", color="white"); ax.tick_params(colors="white")
    ax.set_title("Integrate events per 200 steps", color="white", fontweight="bold")
    ax.grid(alpha=0.15, color="white", axis="y")

    # ── Row 1: Path 2 ─────────────────────────────────────────────────────────
    # Panel 1,0: return probability
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a2e")
    if path2[0]["t_vals"] is not None and len(path2[0]["t_vals"]) > 5:
        _refs(ax, t_ref, path2[0]["p_vals"][5], path2[0]["t_vals"][5])
    for r, col in zip(path2, c2):
        if len(r["t_vals"]) > 0:
            ax.loglog(r["t_vals"], r["p_vals"], color=col, linewidth=2,
                      label=f"reward={r['topology_reward']} d_s={r['d_s']:.2f}")
    ax.axvspan(0.5, 15.0, alpha=0.08, color="white")
    ax.legend(fontsize=8); ax.set_xlabel("t", color="white"); ax.set_ylabel("p(t)", color="white")
    ax.set_title("PATH 2 — Return probability p(t)", color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")

    # Panel 1,1: d_s and T/N vs topology_reward
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a2e")
    rewards = [r["topology_reward"] for r in path2]
    d_s_2 = [r["d_s"] for r in path2]
    tn_2 = [r["triangles"] / max(1, r["nodes"]) for r in path2]
    ax2 = ax.twinx()
    ax.plot(rewards, d_s_2, "o-", color="cyan", linewidth=2, markersize=8, label="d_s")
    ax.axhline(2.0, color="lime", linestyle="--", linewidth=1.5, alpha=0.7, label="d_s=2 target")
    ax2.plot(rewards, tn_2, "s--", color="coral", linewidth=1.5, markersize=7, label="T/N")
    ax.set_xlabel("topology_reward", color="white"); ax.set_ylabel("d_s", color="cyan")
    ax2.set_ylabel("T/N", color="coral")
    ax.tick_params(colors="white"); ax2.tick_params(colors="coral")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("d_s and T/N vs topology_reward", color="white", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.15, color="white")

    # Panel 1,2: triangulate event counts
    ax = axes[1, 2]
    ax.set_facecolor("#1a1a2e")
    tri_counts = [r["triangulate_count"] for r in path2]
    bars = ax.bar(range(len(path2)), tri_counts, color="#9b59b6", alpha=0.8)
    for i, v in enumerate(tri_counts):
        ax.text(i, v + 0.5, str(v), ha="center", color="white", fontsize=11)
    ax.set_xticks(range(len(path2)))
    ax.set_xticklabels([f"reward={r['topology_reward']}" for r in path2], color="white")
    ax.set_ylabel("Count", color="white"); ax.tick_params(colors="white")
    ax.set_title("Triangulate events per 200 steps", color="white", fontweight="bold")
    ax.grid(alpha=0.15, color="white", axis="y")

    for ax_row in axes:
        for ax_ in ax_row:
            ax_.set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.savefig("triangulate_experiment.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved triangulate_experiment.png")
    plt.show()
