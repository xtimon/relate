"""
Move generation, application, and cheap scoring for the graph universe.

A *move* is a 3-tuple  (move_type: str, args: tuple, priority: float).
Five built-in move types are provided:

    expand    — subdivide an edge by inserting a new node
    integrate — collapse a triangle into a single particle node
    deflate   — remove an isolated (degree-0) node
    seed      — vacuum fluctuation: add a disconnected node pair
    connect   — bridge the two largest components with a weak edge

Custom move generators can be passed to RealitySimulation via the
`move_generator` parameter, as long as they produce the same 3-tuple format.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .physics import compute_curvature, compute_local_curvature_field
from .state import RealityState

Move = Tuple[str, tuple, float]


# ------------------------------------------------------------------ #
#  Move generation                                                     #
# ------------------------------------------------------------------ #

def generate_moves(state: RealityState) -> List[Move]:
    """Default move set for a simulation step."""
    moves: List[Move] = []

    if state.edge_count > 0:
        edges = list(state.graph.edges())
        sample = random.sample(edges, min(15, len(edges)))
        for u, v in sample:
            moves.append(("expand", (u, v), 1.0))

    if state.triple_count > 0:
        triples = list(state.triples)
        sample = random.sample(triples, min(10, len(triples)))
        for t in sample:
            moves.append(("integrate", t, 1.0))

    for node in state.graph.nodes():
        if state.graph.degree(node) == 0:
            moves.append(("deflate", (node,), 1.0))

    moves.append(("seed", (), 1.0))

    if state.node_count >= 4 and not nx.is_connected(state.graph):
        components = sorted(nx.connected_components(state.graph), key=len, reverse=True)
        if len(components) >= 2:
            c1, c2 = list(components[0]), list(components[1])
            if c1 and c2:
                moves.append(("connect", (random.choice(c1), random.choice(c2)), 3.0))

    return moves


# ------------------------------------------------------------------ #
#  Move application                                                    #
# ------------------------------------------------------------------ #

def apply_move(state: RealityState, move: Move) -> RealityState:
    """
    Apply *move* to *state* and return the resulting new state.
    The original state is never modified.
    """
    ns = state.clone()
    ns.time = state.time + 1

    move_type, args = move[0], move[1]

    if move_type == "expand":
        u, v = args
        if u in ns.graph and v in ns.graph:
            w = ns.add_node()
            ns.add_edge(w, u, weight=1.0)
            ns.add_edge(w, v, weight=1.0)
            if ns.graph.has_edge(u, v):
                ns.graph[u][v]["weight"] = min(
                    3.0, ns.graph[u][v].get("weight", 1.0) * 1.05
                )

    elif move_type == "integrate":
        u, v, w = args
        if all(n in ns.graph for n in (u, v, w)):
            particle = ns.add_node()
            ext: Dict[int, float] = defaultdict(float)
            for node in (u, v, w):
                for nb in state.graph.neighbors(node):
                    if nb not in (u, v, w):
                        ext[nb] += state.graph[node][nb].get("weight", 1.0)
            for nb, tw in ext.items():
                if nb in ns.graph:
                    ns.add_edge(particle, nb, weight=tw / 3.0)
            ns.graph.remove_nodes_from([u, v, w])
            for n in (u, v, w):
                ns.curvature_field.pop(n, None)
            ns.triples = {t for t in ns.triples if not any(n in (u, v, w) for n in t)}

    elif move_type == "deflate":
        (node,) = args
        if node in ns.graph and ns.graph.degree(node) == 0:
            ns.remove_node(node)

    elif move_type == "seed":
        a = ns.add_node()
        b = ns.add_node()
        ns.add_edge(a, b, weight=1.0)

    elif move_type == "connect":
        u, v = args
        if u in ns.graph and v in ns.graph:
            ns.add_edge(u, v, weight=0.3)

    ns.curvature_field = compute_local_curvature_field(ns)
    return ns


# ------------------------------------------------------------------ #
#  Cheap scoring (used by RealitySimulation.step)                      #
# ------------------------------------------------------------------ #

def score_move_cheap(
    state: RealityState,
    move: Move,
    E: float,
    alpha: float,
    beta: float,
    gamma: float,
    vacuum: float,
    connectivity: float,
    C: float,
    lcs: int,
    U: float,
    size_penalty: float = 0.0,
) -> Tuple[float, Optional[RealityState]]:
    """
    Boltzmann score exp(−ΔE) for *move* using a cheap energy approximation.

    Eigenvalue decomposition and zlib compression are skipped during scoring;
    integrated information I is proxied by lcs/N and complexity U is held
    constant for this step (γ = 0.02 makes its intra-step variation negligible).

    size_penalty adds the term ``+size_penalty * N²`` to the energy.
    Its per-step delta for an expand move is ``≈ 2·size_penalty·N``, which
    grows with N and counteracts the constant ``−vacuum`` drive.  The system
    reaches a soft equilibrium at

        N* ≈ vacuum / (2 · size_penalty)

    so ``size_penalty = vacuum / (2 · N_target)`` calibrates the target size.
    With the default ``vacuum=0.8`` and ``size_penalty=0.004`` the equilibrium
    is around N* ≈ 100.  Setting ``size_penalty=0.0`` disables the term and
    allows unbounded growth (original behaviour).

    For *deflate*, *seed*, and *connect* the score is fully analytical — no
    graph copy is needed.  For *expand* and *integrate* the move is applied
    to a graph copy so that the updated curvature can be used; the resulting
    state is returned so the caller can reuse it if this move is chosen.

    Returns
    -------
    score : float
    candidate_state : RealityState or None
    """
    move_type, args = move[0], move[1]
    priority = move[2] if len(move) > 2 else 1.0
    N = state.node_count

    def _score(ns_C: float, ns_lcs: int, ns_N: int) -> float:
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (
            alpha * ns_C
            - beta * ns_I_proxy
            + gamma * U
            - vacuum * ns_N
            - connectivity * (ns_lcs / max(1, ns_N))
            + size_penalty * ns_N ** 2
        )
        return priority * np.exp(-(ns_E - E))

    if move_type == "deflate":
        # Isolated node: curvature 0, outside largest component.
        return _score(C, lcs, N - 1), None

    elif move_type == "seed":
        # Two fresh nodes + one edge: no new triangles, C unchanged.
        return _score(C, max(lcs, 2), N + 2), None

    elif move_type == "connect":
        # Cross-component edge: no common neighbours → C and N unchanged.
        u, v = args
        comp_u = nx.node_connected_component(state.graph, u)
        comp_v = nx.node_connected_component(state.graph, v)
        return _score(C, len(comp_u) + len(comp_v), N), None

    else:  # expand / integrate: topology changes require a graph copy
        ns = apply_move(state, move)
        ns_C = compute_curvature(ns)
        score = _score(ns_C, ns.largest_component_size(), ns.node_count)
        return score, ns
