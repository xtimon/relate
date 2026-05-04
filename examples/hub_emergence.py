"""
examples/hub_emergence.py
==========================
Track hub (high-degree node) formation over 3000 steps.

Hubs with degree >> avg_degree are topological analogs of 'heavy particles':
localised excitations that concentrate many connections and persist over time.

Usage
-----
    python examples/hub_emergence.py
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

from relate import RealitySimulation
from relate.measures import degree_distribution

np.random.seed(42)
random.seed(42)

sim = RealitySimulation(
    beta=3.0,              # standard β — fast per-step
    topology_reward=0.0,   # no triangulate bias — broader exploration
    size_penalty=0.004,    # N* ≈ 100
    record_all_states=False,
)

SNAPSHOTS = [100, 300, 600, 1000, 2000, 3000]
records = []

print(f"{'t':>5} | {'N':>4} {'avg':>5} {'max':>4} | {'хабов':>6} | "
      f"{'top-5 degrees':>25} | {'integrate':>9} {'tri':>4}")
print("─" * 80)

t0 = time.time()
prev_t = 0
for snap in SNAPSHOTS:
    while sim.state.time < snap:
        sim.step()

    N = sim.state.node_count
    E = sim.state.edge_count
    avg = 2 * E / max(1, N)
    degs = sorted(dict(sim.state.graph.degree()).values(), reverse=True)
    max_d = degs[0] if degs else 0
    hubs = sum(1 for d in degs if d > 3 * avg)

    window = sim.history[prev_t:snap]
    integrate_w = sum(1 for h in window if h["move_type"] == "integrate")
    tri_w = sum(1 for h in window if h["move_type"] == "triangulate")

    dd = degree_distribution(sim.state)

    records.append({
        "t": snap, "N": N, "avg": avg, "max_d": max_d,
        "hubs": hubs, "top5": degs[:5], "dd": dd,
        "integrate": integrate_w, "tri": tri_w,
        "degs_all": degs,
    })

    top5_str = str(degs[:5])
    print(f"{snap:>5} | {N:>4} {avg:>5.1f} {max_d:>4} | {hubs:>6} | "
          f"{top5_str:>25} | {integrate_w:>9} {tri_w:>4}")
    prev_t = snap

print(f"\nCompleted in {time.time()-t0:.1f}s")

# ── Hub persistence: track if the same node stays a hub ───────────────────────
print("\n── Hub node IDs at each snapshot ────────────────────────────────")
for r in records:
    G = sim.state.graph  # final state only — for intermediate we'd need history
    avg = r["avg"]
    thresh = 3 * avg
    print(f"  t={r['t']:4d}: max_degree={r['max_d']}  "
          f"(threshold {thresh:.1f})  hubs={r['hubs']}")

# ── Plot ──────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor="#0f0f23")

    times = [r["t"] for r in records]
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(records)))

    # 1. max degree and avg degree over time
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(times, [r["max_d"] for r in records], "o-", color="red",
            linewidth=2, markersize=8, label="max degree")
    ax.plot(times, [r["avg"] for r in records], "s--", color="cyan",
            linewidth=2, markersize=7, label="avg degree")
    ax.plot(times, [3 * r["avg"] for r in records], ":", color="orange",
            linewidth=1.5, alpha=0.7, label="3× avg (hub threshold)")
    ax.set_xlabel("Time", color="white"); ax.set_ylabel("Degree", color="white")
    ax.set_title("Max vs avg degree", color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    # 2. Hub count over time
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a2e")
    ax.bar(times, [r["hubs"] for r in records], color="#e74c3c",
           alpha=0.8, width=80)
    for i, r in enumerate(records):
        if r["hubs"] > 0:
            ax.text(r["t"], r["hubs"] + 0.1, str(r["hubs"]),
                    ha="center", color="white", fontsize=11)
    ax.set_xlabel("Time", color="white"); ax.set_ylabel("Hub count", color="white")
    ax.set_title("Hubs (degree > 3× avg) over time",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.grid(alpha=0.2, color="white", axis="y")

    # 3. Ratio max/avg — rising means hubs are concentrating
    ax = axes[0, 2]
    ax.set_facecolor("#1a1a2e")
    ratios = [r["max_d"] / max(1, r["avg"]) for r in records]
    ax.plot(times, ratios, "o-", color="gold", linewidth=2, markersize=9)
    ax.axhline(3.0, color="orange", linestyle="--", alpha=0.6, label="3× (hub threshold)")
    ax.axhline(1.0, color="white", linestyle=":", alpha=0.4, label="uniform")
    ax.set_xlabel("Time", color="white"); ax.set_ylabel("max_deg / avg_deg", color="white")
    ax.set_title("Degree concentration ratio", color="white", fontweight="bold")
    ax.legend(fontsize=9); ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    # 4–6. Degree distribution at 3 snapshots
    for i, snap_idx in enumerate([1, 3, 5]):
        if snap_idx >= len(records):
            snap_idx = len(records) - 1
        ax = axes[1, i]
        ax.set_facecolor("#1a1a2e")
        r = records[snap_idx]
        dd = r["dd"]
        ks = sorted(dd.keys())
        vs = [dd[k] for k in ks]

        ax.bar(ks, vs, color=palette[snap_idx], alpha=0.8, edgecolor="none")
        ax.axvline(r["avg"], color="cyan", linestyle="--", linewidth=1.5,
                   label=f"avg={r['avg']:.1f}")
        ax.axvline(r["max_d"], color="red", linestyle=":", linewidth=1.5,
                   label=f"max={r['max_d']}")
        ax.set_xlabel("Degree k", color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title(f"Degree distribution at t={r['t']}  (N={r['N']})",
                     color="white", fontweight="bold")
        ax.legend(fontsize=9); ax.tick_params(colors="white")
        ax.grid(alpha=0.2, color="white", axis="y")

    for ax_row in axes:
        for ax_ in ax_row:
            ax_.set_facecolor("#1a1a2e")

    plt.suptitle("RELATE — Hub Emergence: Topological Matter?",
                 fontsize=15, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("hub_emergence.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved hub_emergence.png")
    plt.show()
