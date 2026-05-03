"""
RealitySimulation — the main entry point for RELATE experiments.

RELATE: Relational Evolutionary Lattice for Algorithmic Time and Experience

Energy functional
-----------------
    E = α·C − β·I + γ·U − vacuum·N − connectivity·(LCS/N)

where
    C   = integrated squared Regge curvature
    I   = spectral integrated-information measure
    U   = Kolmogorov-complexity proxy (zlib growth)
    N   = node count
    LCS = largest connected-component size

At each step the simulator:
  1. Evaluates E for the current state.
  2. Generates candidate moves via the move_generator.
  3. Scores each move cheaply (no eigenvalues in the inner loop).
  4. Samples a move proportionally to exp(−ΔE) × priority.
  5. Applies the chosen move and records observables.
"""

import zlib
from typing import Callable, Dict, List, Optional

import numpy as np

from .dynamics import Move, apply_move, generate_moves, score_move_cheap
from .physics import (
    compute_accumulated_complexity,
    compute_curvature,
    compute_integrated_info,
    compute_local_curvature_field,
    serialize_state,
)
from .state import RealityState


class RealitySimulation:
    """
    Parameters
    ----------
    alpha : float
        Weight of curvature penalty C (default 0.3).
    beta : float
        Weight of integrated-information reward I (default 3.0).
    gamma : float
        Weight of complexity growth U (default 0.02).
    vacuum : float
        Cost per node — controls equilibrium size (default 0.8).
    connectivity : float
        Reward for the fraction of nodes in the largest component (default 3.0).
    move_generator : callable, optional
        Function ``(state) → List[Move]``.  Defaults to the built-in
        :func:`~relate.dynamics.generate_moves`.  Supply your own to add
        custom move types or change sampling budgets.
    record_all_states : bool
        If True (default), every state is deep-copied into ``all_states``
        for animation.  Set to False for long runs where memory matters.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 3.0,
        gamma: float = 0.02,
        vacuum: float = 0.8,
        connectivity: float = 3.0,
        move_generator: Optional[Callable] = None,
        record_all_states: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vacuum = vacuum
        self.connectivity = connectivity
        self.move_generator = move_generator or generate_moves
        self.record_all_states = record_all_states

        self.state = RealityState()
        u = self.state.add_node()
        v = self.state.add_node()
        self.state.add_edge(u, v, weight=1.0)
        self.state.curvature_field = compute_local_curvature_field(self.state)

        self.initial_serialized: bytes = serialize_state(self.state)
        self.history: List[Dict] = []
        self.all_states: List[RealityState] = (
            [self.state.clone()] if record_all_states else []
        )

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """
        Advance the simulation by one time step.

        Returns a dict of observables recorded *before* the move is applied
        (consistent with the original main4.py behaviour).
        """
        C = compute_curvature(self.state)
        I = compute_integrated_info(self.state)
        U = compute_accumulated_complexity(self.state, self.initial_serialized)

        lcs = self.state.largest_component_size()
        E = (
            self.alpha * C
            - self.beta * I
            + self.gamma * U
            - self.vacuum * self.state.node_count
            - self.connectivity * (lcs / max(1, self.state.node_count))
        )

        moves = self.move_generator(self.state)
        scores: List[float] = []
        candidate_states: Dict[int, RealityState] = {}

        for i, move in enumerate(moves):
            score, ns = score_move_cheap(
                self.state, move, E,
                self.alpha, self.beta, self.gamma, self.vacuum, self.connectivity,
                C, lcs, U,
            )
            scores.append(score)
            if ns is not None:
                candidate_states[i] = ns

        total = sum(scores)
        probs = (
            [s / total for s in scores] if total > 0
            else [1.0 / len(scores)] * len(scores)
        )

        chosen_idx = int(np.random.choice(len(moves), p=probs))
        chosen = moves[chosen_idx]

        if chosen_idx in candidate_states:
            self.state = candidate_states[chosen_idx]
        else:
            self.state = apply_move(self.state, chosen)

        if self.record_all_states:
            self.all_states.append(self.state.clone())

        curvs = list(self.state.curvature_field.values()) or [0.0]

        record: Dict = {
            "time": self.state.time,
            "nodes": self.state.node_count,
            "edges": self.state.edge_count,
            "triples": self.state.triple_count,
            "largest_component": lcs,
            "C": C,
            "I": I,
            "U": U,
            "move_type": chosen[0],
            "complexity_bytes": len(
                zlib.compress(serialize_state(self.state), level=9)
            ),
            "curvature_mean": float(np.mean(curvs)),
            "curvature_std": float(np.std(curvs)),
            "curvature_max": float(np.max(np.abs(curvs))),
        }
        self.history.append(record)
        return record

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run(self, steps: int = 100, verbose: bool = True) -> List[Dict]:
        """Run *steps* simulation steps, optionally printing progress."""
        if verbose:
            print(f"{'=' * 70}")
            print(f"  RELATE — Relational Evolutionary Lattice")
            print(f"           for Algorithmic Time and Experience")
            print(f"  α={self.alpha}, β={self.beta}, γ={self.gamma}")
            print(f"  vacuum={self.vacuum}, connectivity={self.connectivity}")
            print(f"{'=' * 70}\n")

        for i in range(steps):
            r = self.step()
            if verbose and (i % 20 == 0 or i == steps - 1):
                print(
                    f"t={r['time']:4d} | N={r['nodes']:4d} E={r['edges']:4d} "
                    f"CC={r['largest_component']:3d} | {r['move_type']:9s} | "
                    f"max|δ|={r['curvature_max']:.2e} | I={r['I']:.4f}"
                )

        if verbose:
            print()
        return self.history

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the completed simulation."""
        s = self.state
        h = self.history
        if not h:
            print("No steps recorded yet.")
            return

        print(f"\n{'=' * 70}")
        print("SIMULATION RESULTS")
        print(f"{'=' * 70}")
        print(f"Final state:")
        print(f"  Nodes:          {s.node_count}")
        print(f"  Edges:          {s.edge_count}")
        print(f"  Triangles:      {s.triple_count}")
        lcs = s.largest_component_size()
        print(f"  Largest comp.:  {lcs} ({100 * lcs / max(1, s.node_count):.1f}%)")
        print("\nEvent counts:")
        for mt in ("expand", "integrate", "connect", "seed", "deflate"):
            print(f"  {mt:12s}: {sum(1 for r in h if r['move_type'] == mt)}")
        print("\nCurvature ripple:")
        print(f"  Max amplitude:  {max(r['curvature_max'] for r in h):.4e}")
        print(f"  Mean std dev:   {np.mean([r['curvature_std'] for r in h]):.4e}")
        print(f"  INTEGRATE events (particle births): "
              f"{sum(1 for r in h if r['move_type'] == 'integrate')}")
        print("\nInformation:")
        print(f"  Final I(G):     {h[-1]['I']:.4f}")
        print(f"  Accumulated U:  {h[-1]['U']:.1f} bytes")
        print(f"{'=' * 70}")
