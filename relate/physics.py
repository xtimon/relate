"""
Observable quantities computable from a RealityState.

All functions are pure (no side effects on the state) and can be called
independently, making them easy to swap or extend in research experiments.
"""

import zlib
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .state import RealityState

# ── Numerical tolerances ───────────────────────────────────────────────────────
_CURVATURE_EPS: float = 1e-6
"""Small constant to avoid division by zero in curvature calculations."""

_NORMALISATION_EPS: float = 1e-10
"""Small constant to avoid log(0) in spectral entropy."""

# ── Spectral dimension / integrated-info thresholds ────────────────────────────
_EIGENVALUE_FULL_THRESHOLD: int = 10
"""Below this node count, compute all eigenvalues exactly; above, use sparse eigs."""

_EIGENVALUE_K: int = 3
"""Number of smallest eigenvalues to request from the sparse solver."""

# ── Compression ────────────────────────────────────────────────────────────────
_ZLIB_LEVEL: int = 9
"""Compression level for zlib (9 = maximum)."""


def compute_local_curvature_field(state: RealityState) -> dict[int, float]:
    """
    Regge-calculus angle-deficit curvature at each node.

    For every triangle (u, v, w), the angular deficit
        Δ = A + B + C − π
    is distributed equally among the three vertices. Edge weights are treated
    as inverse distances, so w_ij → length a_ij = 1 / w_ij.
    """
    local_curvature: dict[int, float] = defaultdict(float)

    for u, v, n_w in state.triples:
        w_uv = state.graph[u][v].get("weight", 1.0)
        w_vw = state.graph[v][n_w].get("weight", 1.0)
        w_wu = state.graph[n_w][u].get("weight", 1.0)

        a = 1.0 / (w_vw + _CURVATURE_EPS)
        b = 1.0 / (w_wu + _CURVATURE_EPS)
        c = 1.0 / (w_uv + _CURVATURE_EPS)

        try:
            cos_A = np.clip(
                (b**2 + c**2 - a**2) / (2 * b * c + _CURVATURE_EPS),
                -1 + _CURVATURE_EPS,
                1 - _CURVATURE_EPS,
            )
            cos_B = np.clip(
                (a**2 + c**2 - b**2) / (2 * a * c + _CURVATURE_EPS),
                -1 + _CURVATURE_EPS,
                1 - _CURVATURE_EPS,
            )
            cos_C = np.clip(
                (a**2 + b**2 - c**2) / (2 * a * b + _CURVATURE_EPS),
                -1 + _CURVATURE_EPS,
                1 - _CURVATURE_EPS,
            )

            deficit = np.arccos(cos_A) + np.arccos(cos_B) + np.arccos(cos_C) - np.pi
            local_curvature[u] += deficit / 3.0
            local_curvature[v] += deficit / 3.0
            local_curvature[n_w] += deficit / 3.0
        except (ValueError, KeyError, ZeroDivisionError):
            continue

    return dict(local_curvature)


def compute_curvature(state: RealityState) -> float:
    """Integrated squared curvature: C(G) = Σ δ(v)²."""
    if not state.curvature_field:
        return 0.0
    return sum(c**2 for c in state.curvature_field.values())


def compute_integrated_info(state: RealityState) -> float:
    """
    Spectral measure of integrated information.

    Combines algebraic connectivity (λ₂ of the normalised Laplacian) with
    normalised spectral entropy to produce a single complexity scalar I(G).
    Only the largest connected component is considered when the graph is
    disconnected.
    """
    if state.node_count < 2:
        return 0.0

    try:
        if nx.is_connected(state.graph):
            L = nx.normalized_laplacian_matrix(state.graph)
            n = state.node_count
        else:
            largest_cc = max(nx.connected_components(state.graph), key=len)
            subgraph = state.graph.subgraph(largest_cc).copy()
            if len(subgraph) < 2:
                return 0.0
            L = nx.normalized_laplacian_matrix(subgraph)
            n = len(subgraph)

        if n <= _EIGENVALUE_FULL_THRESHOLD:
            eigenvalues = np.linalg.eigvalsh(L.toarray())
        else:
            k = min(_EIGENVALUE_K, n - 1)
            eigenvalues = eigs(L, k=k, which="SM", return_eigenvectors=False)
            eigenvalues = np.sort(np.abs(eigenvalues))

        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

        probs = np.abs(eigenvalues) / (np.sum(np.abs(eigenvalues)) + _NORMALISATION_EPS)
        spectral_entropy = -np.sum(probs * np.log(probs + _NORMALISATION_EPS)) / np.log(
            len(eigenvalues) + 1
        )

        return float(lambda_2 * (1.0 + spectral_entropy))
    except (np.linalg.LinAlgError, ValueError, ArpackNoConvergence):
        return 0.0


def serialize_state(state: RealityState) -> bytes:
    """Canonical byte representation of the graph (for compression-based complexity)."""
    if state.node_count == 0:
        return b"empty"
    edges_list = [
        f"{min(u, v)},{max(u, v)},{state.graph[u][v].get('weight', 1.0):.3f}"
        for u, v in state.graph.edges()
    ]
    return ";".join(sorted(edges_list)).encode("utf-8")


def compute_accumulated_complexity(state: RealityState, initial_serialized: bytes) -> float:
    """
    Kolmogorov-complexity proxy: growth of compressed graph description
    relative to the initial (seed) state.
    """
    if state.node_count == 0:
        return 0.0
    current = serialize_state(state)
    return max(0, len(zlib.compress(current, level=_ZLIB_LEVEL)) - len(initial_serialized))
