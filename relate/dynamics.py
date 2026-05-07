"""
Move generation, application, and cheap scoring for the graph universe.

A *move* is a 3-tuple  (move_type: str, args: tuple, priority: float).
Six built-in move types are provided:

    expand      — subdivide an edge by inserting a new node
    integrate   — collapse a triangle into a single particle node
    deflate     — remove an isolated (degree-0) node
    seed        — vacuum fluctuation: add a disconnected node pair
    connect     — bridge the two largest components with a weak edge
    triangulate — close an open triangle (add edge between two non-adjacent
                  nodes that share a common neighbour).  ΔN=0, ΔT≥1.
                  This is the primary mechanism for increasing topological
                  density and pushing the spectral dimension toward d_s=2.

Custom move generators can be passed to RealitySimulation via the
`move_generator` parameter, as long as they produce the same 3-tuple format.
"""

import random
from collections import defaultdict

import networkx as nx
import numpy as np

from .physics import compute_curvature, compute_local_curvature_field
from .state import RealityState

Move = tuple[str, tuple, float]

# ── Sampling budgets (move generation) ────────────────────────────────────────
_EXPAND_SAMPLE_SIZE: int = 15
"""Maximum number of edges sampled for expand moves per step."""

_INTEGRATE_SAMPLE_SIZE: int = 10
"""Maximum number of triangles sampled for integrate moves per step."""

_TRIANGULATE_NODE_SAMPLE: int = 20
"""Number of nodes sampled when searching for open triangles."""

_TRIANGULATE_PAIR_LIMIT: int = 30
"""Stop searching for open-triangle pairs after finding this many."""

_TRIANGULATE_MOVE_LIMIT: int = 8
"""Maximum triangulate moves generated per step."""

# ── Move priorities ───────────────────────────────────────────────────────────
_CONNECT_PRIORITY: float = 3.0
"""Priority weight for connect moves (higher = more likely to be selected)."""

_TRIANGULATE_PRIORITY: float = 2.0
"""Priority weight for triangulate moves."""

_DEFAULT_PRIORITY: float = 1.0
"""Default priority for expand, integrate, deflate, and seed moves."""

# ── Edge weights for move application ─────────────────────────────────────────
_EXPAND_WEIGHT_CAP: float = 3.0
"""Maximum weight an edge can reach through repeated expand moves."""

_EXPAND_WEIGHT_MULTIPLIER: float = 1.05
"""Per-expand multiplicative increase of the subdivided edge's weight."""

_CONNECT_WEIGHT: float = 0.3
"""Weight assigned to a newly bridged connect edge (weak link)."""

_DEFAULT_EDGE_WEIGHT: float = 1.0
"""Default weight for new edges (expand, seed, triangulate)."""

# ── Minimum node counts for move eligibility ──────────────────────────────────
_MIN_NODES_FOR_CONNECT: int = 4
"""Graph must have at least this many nodes before connect moves are generated."""

_MIN_NODES_FOR_TRIANGULATE: int = 3
"""Graph must have at least this many nodes before triangulate moves are generated."""


# ------------------------------------------------------------------ #
#  Move generation                                                     #
# ------------------------------------------------------------------ #


def generate_moves(state: RealityState) -> list[Move]:
    """Default move set for a simulation step."""
    moves: list[Move] = []

    if state.edge_count > 0:
        edges = list(state.graph.edges())
        sample = random.sample(edges, min(_EXPAND_SAMPLE_SIZE, len(edges)))
        for u, v in sample:
            moves.append(("expand", (u, v), _DEFAULT_PRIORITY))

    if state.triple_count > 0:
        triples = list(state.triples)
        sample = random.sample(triples, min(_INTEGRATE_SAMPLE_SIZE, len(triples)))
        for t in sample:
            moves.append(("integrate", t, _DEFAULT_PRIORITY))

    for node in state.graph.nodes():
        if state.graph.degree(node) == 0:
            moves.append(("deflate", (node,), _DEFAULT_PRIORITY))

    moves.append(("seed", (), _DEFAULT_PRIORITY))

    if state.node_count >= _MIN_NODES_FOR_CONNECT and not nx.is_connected(state.graph):
        components = sorted(nx.connected_components(state.graph), key=len, reverse=True)
        if len(components) >= 2:
            c1, c2 = list(components[0]), list(components[1])
            if c1 and c2:
                moves.append(("connect", (random.choice(c1), random.choice(c2)), _CONNECT_PRIORITY))

    # TRIANGULATE: close an open triangle (ΔN=0, ΔT≥1).
    # Sample nodes, collect non-adjacent neighbour pairs, pick up to 8.
    if state.node_count >= _MIN_NODES_FOR_TRIANGULATE:
        open_pairs: list[tuple[int, int]] = []
        sample_nodes = random.sample(
            list(state.graph.nodes()), min(_TRIANGULATE_NODE_SAMPLE, state.node_count)
        )
        for w in sample_nodes:
            nbs = list(state.graph.neighbors(w))
            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    u, v = nbs[i], nbs[j]
                    if not state.graph.has_edge(u, v):
                        open_pairs.append((u, v))
                if len(open_pairs) >= _TRIANGULATE_PAIR_LIMIT:
                    break
            if len(open_pairs) >= _TRIANGULATE_PAIR_LIMIT:
                break
        if open_pairs:
            for u, v in random.sample(open_pairs, min(_TRIANGULATE_MOVE_LIMIT, len(open_pairs))):
                moves.append(("triangulate", (u, v), _TRIANGULATE_PRIORITY))

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
            ns.add_edge(w, u, weight=_DEFAULT_EDGE_WEIGHT)
            ns.add_edge(w, v, weight=_DEFAULT_EDGE_WEIGHT)
            if ns.graph.has_edge(u, v):
                ns.graph[u][v]["weight"] = min(
                    _EXPAND_WEIGHT_CAP,
                    ns.graph[u][v].get("weight", _DEFAULT_EDGE_WEIGHT) * _EXPAND_WEIGHT_MULTIPLIER,
                )

    elif move_type == "integrate":
        u, v, w = args
        if all(n in ns.graph for n in (u, v, w)):
            particle = ns.add_node()
            ext: dict[int, float] = defaultdict(float)
            for node in (u, v, w):
                for nb in state.graph.neighbors(node):
                    if nb not in (u, v, w):
                        ext[nb] += state.graph[node][nb].get("weight", _DEFAULT_EDGE_WEIGHT)
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
        ns.add_edge(a, b, weight=_DEFAULT_EDGE_WEIGHT)

    elif move_type == "connect":
        u, v = args
        if u in ns.graph and v in ns.graph:
            ns.add_edge(u, v, weight=_CONNECT_WEIGHT)

    elif move_type == "triangulate":
        u, v = args
        if u in ns.graph and v in ns.graph and not ns.graph.has_edge(u, v):
            ns.add_edge(u, v, weight=_DEFAULT_EDGE_WEIGHT)

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
    topology_reward: float = 0.0,
) -> tuple[float, RealityState | None]:
    """
    Boltzmann score exp(−ΔE) for *move* using a cheap energy approximation.

    Eigenvalue decomposition and zlib compression are skipped during scoring;
    integrated information I is proxied by lcs/N and complexity U is held
    constant for this step (γ = 0.02 makes its intra-step variation negligible).

    size_penalty adds ``+size_penalty * N²`` to the energy; its per-step
    delta for expand is ``≈ 2·size_penalty·N``, creating a soft equilibrium
    at ``N* ≈ vacuum / (2·size_penalty)``.

    topology_reward adds ``−topology_reward * T/N`` where T = triangle count.
    Moves that increase T relative to N (especially *triangulate*) lower the
    energy and are preferentially selected, driving topological density and
    spectral dimension upward toward d_s ≈ 2.

    For *deflate*, *seed*, *connect*, and *triangulate* the score is fully
    analytical — no graph copy is needed.  For *expand* and *integrate* the
    move is applied to a graph copy so that the updated curvature and
    triangle count can be used; the resulting state is returned so the
    caller can reuse it if this move is chosen.

    Returns
    -------
    score : float
    candidate_state : RealityState or None
    """
    move_type, args = move[0], move[1]
    priority = move[2] if len(move) > 2 else 1.0
    N = state.node_count
    T = state.triple_count

    def _score(ns_C: float, ns_lcs: int, ns_N: int, ns_T: int) -> float:
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (
            alpha * ns_C
            - beta * ns_I_proxy
            + gamma * U
            - vacuum * ns_N
            - connectivity * (ns_lcs / max(1, ns_N))
            + size_penalty * ns_N**2
            - topology_reward * ns_T / max(1, ns_N)
        )
        return float(priority * np.exp(-(ns_E - E)))

    if move_type == "deflate":
        # Isolated node: curvature 0, outside largest component, no triangles.
        return _score(C, lcs, N - 1, T), None

    elif move_type == "seed":
        # Two fresh nodes + one edge: no new triangles.
        return _score(C, max(lcs, 2), N + 2, T), None

    elif move_type == "connect":
        # Cross-component edge: endpoints share no neighbours → no new triangles.
        # The new LCS is the merged component size, but only if it exceeds the
        # current LCS (there may be a third, larger component that is untouched).
        u, v = args
        comp_u = nx.node_connected_component(state.graph, u)
        comp_v = nx.node_connected_component(state.graph, v)
        merged = len(comp_u) + len(comp_v)
        return _score(C, max(merged, lcs), N, T), None

    elif move_type == "triangulate":
        # ΔN=0, ΔT = common neighbours of u and v (exact, no copy needed).
        u, v = args
        common = len(set(state.graph.neighbors(u)) & set(state.graph.neighbors(v)))
        return _score(C, lcs, N, T + common), None

    else:  # expand / integrate: topology changes require a graph copy
        ns = apply_move(state, move)
        ns_C = compute_curvature(ns)
        score = _score(ns_C, ns.largest_component_size(), ns.node_count, ns.triple_count)
        return score, ns
