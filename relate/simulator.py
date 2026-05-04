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
        Cost per node — controls the growth drive (default 0.8).
    connectivity : float
        Reward for the fraction of nodes in the largest component (default 3.0).
    size_penalty : float
        Coefficient of the ``+size_penalty · N²`` equilibrium term.
        Soft equilibrium: ``N* ≈ vacuum / (2·size_penalty)``.
        Default 0.004 → N* ≈ 100.  Set to 0.0 for unbounded growth.
    topology_reward : float
        Coefficient of the ``−topology_reward · T/N`` term, where T is the
        triangle count.  Positive values make moves that increase triangles
        (especially *triangulate*) energetically favourable, driving the graph
        toward higher topological density and larger spectral dimension d_s.
        Default 0.0 (disabled).  Try values 2–10 to push d_s toward 2.
    i_update_interval : int
        Recompute exact I(G) (eigenvalue decomposition) only every this many
        steps; cached value is used in between.  Default 1 (every step).
        Set to 10–50 for long runs to get a ~10–50× speedup when the
        precise per-step I trajectory is not needed.  The history record
        always contains the most recently computed I.
    initial_state : RealityState, optional
        Start from a pre-built state instead of the default two-node seed.
        Useful for testing different initial topologies (torus, random graph,
        etc.).  The state is cloned on entry; curvature must already be
        computed (call ``compute_local_curvature_field`` beforehand).
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
        size_penalty: float = 0.004,
        topology_reward: float = 0.0,
        i_update_interval: int = 1,
        initial_state: Optional["RealityState"] = None,
        move_generator: Optional[Callable] = None,
        record_all_states: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vacuum = vacuum
        self.connectivity = connectivity
        self.size_penalty = size_penalty
        self.topology_reward = topology_reward
        self.i_update_interval = max(1, i_update_interval)
        self.move_generator = move_generator or generate_moves
        self.record_all_states = record_all_states

        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            self.state = RealityState()
            u = self.state.add_node()
            v = self.state.add_node()
            self.state.add_edge(u, v, weight=1.0)
            self.state.curvature_field = compute_local_curvature_field(self.state)

        self.initial_serialized: bytes = serialize_state(self.state)
        self._cached_I: float = 0.0
        self._steps_since_I_update: int = self.i_update_interval  # force update on first step
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
        U = compute_accumulated_complexity(self.state, self.initial_serialized)

        self._steps_since_I_update += 1
        if self._steps_since_I_update >= self.i_update_interval:
            self._cached_I = compute_integrated_info(self.state)
            self._steps_since_I_update = 0
        I = self._cached_I

        lcs = self.state.largest_component_size()
        lcs_fraction = lcs / max(1, self.state.node_count)

        # E_exact uses the true I(G) and is stored in history.
        topo_term = (
            -self.topology_reward * self.state.triple_count
            / max(1, self.state.node_count)
        )

        E_exact = (
            self.alpha * C
            - self.beta * I
            + self.gamma * U
            - self.vacuum * self.state.node_count
            - self.connectivity * lcs_fraction
            + self.size_penalty * self.state.node_count ** 2
            + topo_term
        )

        # E_scoring uses the same lcs/N proxy for I that score_move_cheap uses
        # for candidates. This removes the spurious −β·lcs/N bonus that arises
        # when exact I → 0 at large N while the candidate proxy remains ≈ 0.9·β.
        E_scoring = (
            self.alpha * C
            - self.beta * lcs_fraction
            + self.gamma * U
            - self.vacuum * self.state.node_count
            - self.connectivity * lcs_fraction
            + self.size_penalty * self.state.node_count ** 2
            + topo_term
        )

        moves = self.move_generator(self.state)
        scores: List[float] = []
        candidate_states: Dict[int, RealityState] = {}

        for i, move in enumerate(moves):
            score, ns = score_move_cheap(
                self.state, move, E_scoring,
                self.alpha, self.beta, self.gamma, self.vacuum, self.connectivity,
                C, lcs, U,
                self.size_penalty,
                self.topology_reward,
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
            print(f"  vacuum={self.vacuum}, connectivity={self.connectivity}, "
                  f"size_penalty={self.size_penalty}, "
                  f"topology_reward={self.topology_reward}")
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
