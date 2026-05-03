"""
examples/quantum_breath.py
===========================
Canonical quantum-breath experiment: run the simulation, print a summary,
render the six-panel analysis figure, and export an animated GIF.

Usage
-----
    # from the workspace root:
    python examples/quantum_breath.py

    # or as a Jupyter cell (displays the GIF inline):
    %run examples/quantum_breath.py
"""

import random

import numpy as np

import relate
from relate import RealitySimulation, create_perfect_animation, plot_final_universe

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

print("""
╔══════════════════════════════════════════════════════════════╗
║  RELATE — Relational Evolutionary Lattice for Algorithmic   ║
║           Time and Experience                               ║
║  Example: quantum breath, standard parameter set            ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Simulation ────────────────────────────────────────────────────────────────
sim = RealitySimulation(
    alpha=0.3,
    beta=3.0,
    gamma=0.02,
    vacuum=0.8,
    connectivity=3.0,
)

history = sim.run(steps=1000, verbose=True)
sim.summary()

# ── Static analysis figure ────────────────────────────────────────────────────
print("\nRendering final analysis plot...")
plot_final_universe(sim, history)

# ── Animation ─────────────────────────────────────────────────────────────────
print("\nBuilding quantum-breath animation (this may take ~1 minute)...")

n_frames = 60
step = max(1, len(sim.all_states) // n_frames)
_, filename = create_perfect_animation(sim, history, step=step, fps=6,
                                       filename="quantum_breath.gif")

print(f"\nAnimation saved as '{filename}'")
print(f"  {len(sim.all_states)} states → {len(sim.all_states) // step} frames @ 6 fps")

# Display inline when running inside Jupyter
try:
    from IPython.display import Image as IPImage, display
    display(IPImage(filename))
except ImportError:
    print("Open quantum_breath.gif to view the animation.")


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH EXTENSION EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

# 1. Custom move generator — add a "rewire" move that flips a random edge
# ---------------------------------------------------------------------------
# def rewire_moves(state):
#     moves = relate.generate_moves(state)
#     if state.edge_count >= 2:
#         edges = list(state.graph.edges())
#         u, v = random.choice(edges)
#         candidates = [n for n in state.graph.nodes() if n != u and not state.graph.has_edge(u, n)]
#         if candidates:
#             w = random.choice(candidates)
#             moves.append(("rewire", (u, v, w), 1.5))
#     return moves
#
# sim2 = RealitySimulation(move_generator=rewire_moves)
# history2 = sim2.run(steps=100)

# 2. Parameter sweep
# ---------------------------------------------------------------------------
# for beta in [1.0, 2.0, 3.0, 5.0]:
#     np.random.seed(42); random.seed(42)
#     s = RealitySimulation(beta=beta, record_all_states=False)
#     h = s.run(steps=200, verbose=False)
#     print(f"β={beta:.1f}  final_I={h[-1]['I']:.4f}  nodes={s.state.node_count}")

# 3. Inspect a single state
# ---------------------------------------------------------------------------
# state = sim.all_states[50]
# print(state)
# print("Curvature field:", state.curvature_field)
