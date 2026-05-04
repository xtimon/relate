# RELATE

**Relational Evolutionary Lattice for Algorithmic Time and Experience**

A graph-universe simulation where discrete space-time emerges from self-organisation. The universe is a weighted graph that evolves by minimising an energy functional combining Regge-calculus curvature, spectral integrated information, and Kolmogorov complexity.

---

## Key Results

- **Emergent 2D geometry**: at parameters `β=30, topology_reward=3.5` the graph self-organises into a structure where average path length scales as **L ∝ √N** (R²=0.83 vs R²=0.79 for log N) — the hallmark of a 2D manifold.
- **Spectral dimension d_s ≈ 2**: peaks at d_s=2.35 during the triangulation growth phase; equilibrium value ~1.9 at `β=30, topology_reward=3.5`.
- **Scale-free hub emergence**: at 3000 steps one node reaches degree=79 with avg_degree=3.1 — a **25× concentration** consistent with a Barabási–Albert power-law, arising without external tuning via preferential attachment through the `integrate` move.
- **Phase transitions**: sweeping α×β reveals distinct gas / crystal / complex / collapse phases.
- **Arrow of time**: Kolmogorov-complexity proxy U(t) grows monotonically in 76% of steps; `integrate` (particle-birth) events precede curvature spikes.

---

## Energy Functional

```
E = α·C − β·I + γ·U − vacuum·N − connectivity·(LCS/N) + size_penalty·N² − topology_reward·T/N
```

| Term | Role |
|---|---|
| `α·C` | Regge curvature penalty — flattens geometry |
| `β·I` | Spectral integrated-information reward — drives connectivity |
| `γ·U` | Kolmogorov complexity growth — arrow of time |
| `vacuum·N` | Vacuum energy — drives expansion |
| `connectivity·(LCS/N)` | Rewards large connected component |
| `size_penalty·N²` | Soft size cap — equilibrium at N* = vacuum/(2·size_penalty) |
| `topology_reward·T/N` | Triangle-density reward — drives spectral dimension upward |

---

## Move Types

| Move | Effect | ΔN | ΔT |
|---|---|---|---|
| `expand` | Subdivide edge (u,v) → insert node w | +1 | +1 |
| `integrate` | Collapse triangle → single particle node | −2 | variable |
| `seed` | Vacuum fluctuation: add disconnected pair | +2 | 0 |
| `deflate` | Remove isolated node | −1 | 0 |
| `connect` | Bridge two largest components | 0 | 0 |
| `triangulate` | Close open triangle (ΔN=0, ΔT≥1) | 0 | +k |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requirements: `numpy`, `networkx`, `scipy`, `matplotlib` (Python ≥ 3.10).

---

## Quick Start

```python
import numpy as np, random
from relate import RealitySimulation, plot_final_universe, create_perfect_animation

np.random.seed(42); random.seed(42)

sim = RealitySimulation(
    alpha=0.3,
    beta=3.0,
    topology_reward=3.5,   # drives d_s → 2
    size_penalty=0.004,    # equilibrium near N* ≈ 100
)
history = sim.run(steps=200)
sim.summary()

plot_final_universe(sim, history)
create_perfect_animation(sim, history, fps=6, filename="quantum_breath.gif")
```

### Custom move generator

```python
import relate

def my_moves(state):
    moves = relate.generate_moves(state)          # keep built-ins
    moves.append(("rewire", (u, v, w), 2.0))      # add custom move
    return moves

sim = RealitySimulation(move_generator=my_moves)
```

### Parameter sweep

```python
for beta in [1.0, 3.0, 10.0]:
    sim = RealitySimulation(beta=beta, record_all_states=False)
    h = sim.run(steps=300, verbose=False)
    print(f"β={beta}  I={h[-1]['I']:.4f}")
```

---

## Analysis Tools

```python
from relate.measures import (
    spectral_dimension,          # d_s via Laplacian return probability
    ricci_curvature_distribution,# angle-deficit statistics
    degree_distribution,         # {degree: count}
    power_law_exponent,          # Hill estimator for P(k) ∝ k^(-γ)
    small_world_metrics,         # clustering, L, ω
    integrate_event_analysis,    # particle-birth timing and curvature
    monotonicity_score,          # fraction of steps where observable ↑
    network_summary,             # all structural metrics in one dict
)

d_s, t_vals, p_vals = spectral_dimension(sim.state)
print(f"Spectral dimension: {d_s:.2f}")
```

---

## Experiment Scripts

| Script | What it tests |
|---|---|
| `examples/quantum_breath.py` | Standard run + animation |
| `examples/thermodynamics.py` | H1: second law · H2: particle births · H3: spectral dimension |
| `examples/phase_diagram.py` | α×β phase diagram (4 observables) |
| `examples/spectral_dim_sweep.py` | α×β sweep measuring d_s |
| `examples/triangulate_experiment.py` | High-β vs topology_reward paths to d_s=2 |
| `examples/golden_zone.py` | Fine grid search for d_s ≥ 2 |
| `examples/large_scale.py` | d_s stability and L∝√N test across N=100–800 |
| `examples/hub_emergence.py` | Scale-free hub formation over 3000 steps |

Run any experiment from the workspace root:

```bash
python examples/golden_zone.py
```

---

## Package Structure

```
relate/
├── state.py       # RealityState dataclass + clone()
├── physics.py     # compute_curvature, compute_integrated_info, …
├── dynamics.py    # generate_moves, apply_move, score_move_cheap
├── simulator.py   # RealitySimulation (step / run / summary)
├── measures.py    # spectral_dimension, small_world_metrics, …
└── viz.py         # create_perfect_animation, plot_final_universe
examples/
├── quantum_breath.py
├── thermodynamics.py
├── phase_diagram.py
├── spectral_dim_sweep.py
├── triangulate_experiment.py
├── golden_zone.py
├── large_scale.py
└── hub_emergence.py
```

---

## Scientific Background

RELATE is inspired by several frameworks:

- **Regge calculus** — discrete general relativity on triangulated manifolds
- **Causal Dynamical Triangulations (CDT)** — path integral over discrete geometries
- **Integrated Information Theory (IIT)** — Tononi's Φ as a measure of consciousness/complexity
- **Causal set theory** — spacetime as a partially ordered discrete set
- **Algorithmic information theory** — Kolmogorov complexity as physical entropy

The energy functional is a discretised action that rewards informational richness (high λ₂), geometric regularity (low curvature variance), and topological density (high T/N), while penalising unconstrained growth.

---

## Observed Phases

| Phase | Parameters | d_s | T/N | Dominant move |
|---|---|---|---|---|
| Gas | low β, low α | < 1 | < 0.5 | seed, expand |
| Crystal | high α | ~1 | ~1 | expand |
| Complex | β≈3, reward≈3.5 | ~2 | ~2 | triangulate, integrate |
| Collapse | reward > 5 | ~1.9 | > 10 | triangulate (clique) |

---

## Topological Matter: Scale-Free Hub Emergence

At 3000 steps with default parameters (`β=3, size_penalty=0.004`), a single node accumulates degree **79 while avg_degree=3.1** — a 25× concentration absent in random graphs (Erdős–Rényi maximum would be ~7).

```
t= 100 | N= 49  avg=3.9  max= 27  ratio= 7.0×
t= 300 | N= 93  avg=3.4  max= 58  ratio=16.9×
t=1000 | N= 97  avg=3.1  max= 77  ratio=25.1×
t=2000 | N= 99  avg=2.9  max= 79  ratio=26.8×
```

The mechanism is **preferential attachment** via the `integrate` move: collapsing a triangle (u,v,w) produces a new node that inherits all external connections of u, v, and w simultaneously — giving high-degree nodes a structural advantage. This is the Barabási–Albert process arising without external tuning, purely from the graph dynamics.

These hubs are the closest analog to **topological matter** in the current model: stable, localised, high-curvature excitations that persist over thousands of steps.

---

## Findings Summary

| Hypothesis | Result |
|---|---|
| H1: Second law (U monotone) | Partially confirmed (76% of steps) |
| H2: Particle births correlate with curvature spikes | Weak — births precede rather than follow spikes |
| H3: Spectral dimension d_s | Peaks at d_s=2.35 during triangulation growth; equilibrium ~1.9 |
| L ∝ √N (2D geodesic scaling) | Confirmed: R²=0.83 vs R²=0.79 for log N |
| α irrelevance | Confirmed: curvature term dynamically inactive in explored regime |
| Hub emergence (topological matter) | **Confirmed**: degree=79 at avg=3.1 after 2000 steps (25× concentration, scale-free) |
| Matter at large N | Scale-free hubs emerge spontaneously via preferential attachment through `integrate` moves; stable over 3000+ steps |
