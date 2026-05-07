"""
Scientific observables for RELATE graph universes.

All functions are pure (no side effects) and accept a RealityState.
They can be combined freely in experiment scripts.
"""

from collections import Counter

import networkx as nx
import numpy as np

from .state import RealityState

# ------------------------------------------------------------------ #
#  Geometry                                                            #
# ------------------------------------------------------------------ #

def spectral_dimension(
    state: RealityState,
    t_values: np.ndarray | None = None,
    fit_range: tuple[float, float] = (0.5, 15.0),
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the spectral dimension d_s via the return probability of a
    random walk on the graph Laplacian.

    The return probability satisfies  p(t) ∝ t^(−d_s/2),  so

        d_s = −2 · d(log p) / d(log t)

    estimated by a log-log linear fit in ``fit_range``.

    Parameters
    ----------
    state : RealityState
    t_values : array of diffusion times (default: log-spaced 0.1–50)
    fit_range : (t_min, t_max) window used for the slope fit

    Returns
    -------
    d_s : float  — estimated spectral dimension
    t_values : ndarray
    p_values : ndarray  — return probabilities
    """
    G = _largest_component(state)
    if len(G) < 4:
        return 0.0, np.array([]), np.array([])

    L = nx.normalized_laplacian_matrix(G)
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(L.toarray())))

    if t_values is None:
        t_values = np.logspace(-1, np.log10(50), 40)

    p_values = np.array([np.mean(np.exp(-eigenvalues * t)) for t in t_values])

    t_lo, t_hi = fit_range
    mask = (t_values >= t_lo) & (t_values <= t_hi) & (p_values > 0)
    if mask.sum() >= 4:
        slope, _ = np.polyfit(np.log(t_values[mask]), np.log(p_values[mask]), 1)
        d_s = float(-2.0 * slope)
    else:
        d_s = 0.0

    return d_s, t_values, p_values


def ricci_curvature_distribution(state: RealityState) -> dict[str, float]:
    """
    Summary statistics of the Regge angle-deficit curvature field.

    Returns mean, std, max, min, and the fraction of nodes with positive
    (positive-curvature, sphere-like) vs negative (saddle-like) deficit.
    """
    vals = list(state.curvature_field.values())
    if not vals:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0,
                "positive_fraction": 0.0}
    arr = np.array(vals)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "positive_fraction": float(np.mean(arr > 0)),
    }


# ------------------------------------------------------------------ #
#  Network topology                                                    #
# ------------------------------------------------------------------ #

def degree_distribution(state: RealityState) -> dict[int, int]:
    """Return ``{degree: node_count}`` histogram."""
    return dict(Counter(d for _, d in state.graph.degree()))


def power_law_exponent(state: RealityState, k_min_percentile: float = 75.0) -> float:
    """
    Maximum-likelihood estimate of the power-law exponent γ for the
    degree distribution tail  P(k) ∝ k^(−γ).

    Uses the Hill estimator.  Returns 0 if data are insufficient.
    """
    degrees = np.array([d for _, d in state.graph.degree() if d > 0])
    if len(degrees) < 10:
        return 0.0
    k_min = float(np.percentile(degrees, k_min_percentile))
    tail = degrees[degrees >= k_min]
    if len(tail) < 5 or k_min <= 0.5:
        return 0.0
    gamma = 1.0 + len(tail) / float(np.sum(np.log(tail / (k_min - 0.5))))
    return float(gamma)


def small_world_metrics(state: RealityState) -> dict[str, float | None]:
    """
    Clustering coefficient, average path length, and small-world index ω.

    ω = L_rand/L − C/C_rand  where values near 0 indicate small-world
    behaviour.  Path length is computed on the largest connected component.
    """
    G = _largest_component(state)
    n = len(G)
    if n < 3:
        return {"clustering": 0.0, "avg_path_length": None, "omega": None}

    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G) if n <= 300 else None

    # Random graph with same n and edge count
    e = G.number_of_edges()
    p_rand = 2 * e / max(1, n * (n - 1))
    C_rand = p_rand
    L_rand = np.log(n) / max(1e-9, np.log(n * p_rand)) if p_rand > 0 else None

    omega = None
    if L is not None and L_rand is not None and C > 0:
        omega = float(L_rand / L - C / max(1e-9, C_rand))

    return {
        "clustering": float(C),
        "avg_path_length": float(L) if L is not None else None,
        "omega": omega,
    }


# ------------------------------------------------------------------ #
#  Thermodynamics / information                                        #
# ------------------------------------------------------------------ #

def complexity_slope(history: list[dict], window: int = 20) -> float:
    """
    Linear slope dU/dt estimated over the last *window* steps.
    Positive slope → complexity is still growing (second law satisfied).
    """
    if len(history) < window:
        return 0.0
    u_vals = [h["U"] for h in history[-window:]]
    t_vals = [h["time"] for h in history[-window:]]
    slope, _ = np.polyfit(t_vals, u_vals, 1)
    return float(slope)


def integrate_event_analysis(history: list[dict]) -> dict:
    """
    Statistics about integrate (particle-birth) events: their timing,
    curvature amplitude at the moment they occur, and inter-event intervals.
    """
    events = [h for h in history if h["move_type"] == "integrate"]
    if not events:
        return {"count": 0, "times": [], "curv_amplitudes": [],
                "mean_interval": None}
    times = [h["time"] for h in events]
    amplitudes = [h["curvature_max"] for h in events]
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return {
        "count": len(events),
        "times": times,
        "curv_amplitudes": amplitudes,
        "mean_interval": float(np.mean(intervals)) if intervals else None,
        "mean_amplitude": float(np.mean(amplitudes)),
    }


def monotonicity_score(history: list[dict], key: str = "U") -> float:
    """
    Fraction of consecutive steps where ``history[t][key] ≥ history[t-1][key]``.
    A value of 1.0 means the quantity never decreased (strict second law).
    """
    vals = [h[key] for h in history]
    if len(vals) < 2:
        return 1.0
    increases = sum(1 for a, b in zip(vals, vals[1:], strict=False) if b >= a)
    return increases / (len(vals) - 1)


# ------------------------------------------------------------------ #
#  Full summary                                                        #
# ------------------------------------------------------------------ #

def network_summary(state: RealityState) -> dict:
    """All structural metrics in a single dict."""
    G = state.graph
    N = state.node_count
    E = state.edge_count
    return {
        "nodes": N,
        "edges": E,
        "triangles": state.triple_count,
        "avg_degree": 2 * E / max(1, N),
        "density": 2 * E / max(1, N * (N - 1)),
        "n_components": nx.number_connected_components(G) if N > 0 else 0,
        "lcs_fraction": state.largest_component_size() / max(1, N),
        "clustering": nx.average_clustering(G) if E > 0 else 0.0,
        "power_law_gamma": power_law_exponent(state),
    }


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _largest_component(state: RealityState) -> nx.Graph:
    if state.node_count == 0:
        return nx.Graph()
    if nx.is_connected(state.graph):
        return state.graph
    cc = max(nx.connected_components(state.graph), key=len)
    return state.graph.subgraph(cc).copy()
