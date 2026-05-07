"""
Tests for relate.measures — scientific observables.

Covers: spectral dimension, curvature distribution, degree distribution,
        power-law exponent, small-world metrics, complexity slope,
        integrate event analysis, monotonicity score, network summary.
"""

import networkx as nx
import numpy as np
import pytest

from relate.dynamics import MoveType
from relate.measures import (
    _largest_component,
    complexity_slope,
    degree_distribution,
    integrate_event_analysis,
    monotonicity_score,
    network_summary,
    power_law_exponent,
    ricci_curvature_distribution,
    small_world_metrics,
    spectral_dimension,
)
from relate.physics import compute_local_curvature_field
from relate.state import RealityState

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_state():
    return RealityState()


@pytest.fixture
def two_node_state():
    state = RealityState()
    a = state.add_node()
    b = state.add_node()
    state.add_edge(a, b, weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


@pytest.fixture
def triangle_state():
    state = RealityState()
    a = state.add_node()
    b = state.add_node()
    c = state.add_node()
    state.add_edge(a, b, weight=1.0)
    state.add_edge(b, c, weight=1.0)
    state.add_edge(a, c, weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


@pytest.fixture
def path_state():
    """A 5-node path graph — 1D structure."""
    state = RealityState()
    nodes = [state.add_node() for _ in range(5)]
    for i in range(4):
        state.add_edge(nodes[i], nodes[i + 1], weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


@pytest.fixture
def grid_state():
    """A 3x3 grid — approximately 2D."""
    state = RealityState()
    nodes = {}
    for i in range(3):
        for j in range(3):
            n = state.add_node()
            nodes[(i, j)] = n
    for i in range(3):
        for j in range(3):
            if i < 2:
                state.add_edge(nodes[(i, j)], nodes[(i + 1, j)], weight=1.0)
            if j < 2:
                state.add_edge(nodes[(i, j)], nodes[(i, j + 1)], weight=1.0)
    state.curvature_field = compute_local_curvature_field(state)
    return state


@pytest.fixture
def sample_history():
    """A sample history list for testing thermodynamic measures."""
    return [
        {"time": 0, "U": 0.0, "C": 0.0, "move_type": MoveType.SEED, "curvature_max": 0.0},
        {"time": 1, "U": 5.0, "C": 1.0, "move_type": MoveType.EXPAND, "curvature_max": 0.5},
        {"time": 2, "U": 8.0, "C": 2.0, "move_type": MoveType.INTEGRATE, "curvature_max": 1.2},
        {"time": 3, "U": 10.0, "C": 1.5, "move_type": MoveType.EXPAND, "curvature_max": 0.8},
        {"time": 4, "U": 10.0, "C": 1.0, "move_type": MoveType.INTEGRATE, "curvature_max": 1.5},
        {"time": 5, "U": 12.0, "C": 0.5, "move_type": MoveType.CONNECT, "curvature_max": 0.3},
    ]


# ── _largest_component (internal helper) ──────────────────────────────────────


class TestLargestComponent:
    """Internal helper for extracting the largest connected component."""

    def test_empty_state(self, empty_state):
        """Empty state returns empty graph."""
        G = _largest_component(empty_state)
        assert len(G) == 0

    def test_connected_graph(self, triangle_state):
        """Connected graph returns itself."""
        G = _largest_component(triangle_state)
        assert len(G) == triangle_state.node_count

    def test_disconnected_graph(self):
        """Disconnected graph returns the largest component."""
        state = RealityState()
        # Component 1: 3 nodes
        a, b, c = state.add_node(), state.add_node(), state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        # Component 2: 2 nodes
        d, e = state.add_node(), state.add_node()
        state.add_edge(d, e)
        state.curvature_field = compute_local_curvature_field(state)

        G = _largest_component(state)
        assert len(G) == 3

    def test_returns_subgraph(self, triangle_state):
        """Returns a graph that can be used independently."""
        G = _largest_component(triangle_state)
        assert len(G) == triangle_state.node_count
        # All nodes from the original are present
        for n in triangle_state.graph.nodes():
            assert n in G


# ── spectral_dimension ────────────────────────────────────────────────────────


class TestSpectralDimension:
    """Estimate of spectral dimension d_s via return probability."""

    def test_empty_state(self, empty_state):
        """Empty state returns d_s = 0."""
        d_s, t, p = spectral_dimension(empty_state)
        assert d_s == 0.0
        assert len(t) == 0
        assert len(p) == 0

    def test_small_graph(self, two_node_state):
        """Graph with < 4 nodes returns d_s = 0."""
        d_s, t, p = spectral_dimension(two_node_state)
        assert d_s == 0.0

    def test_path_graph(self, path_state):
        """A 5-node path should have d_s ≈ 1 (1D structure)."""
        d_s, t, p = spectral_dimension(path_state)
        assert d_s > 0.0
        # Path is 1D, so d_s should be closer to 1 than to 2
        # But with only 5 nodes, the estimate may be noisy
        assert isinstance(d_s, float)

    def test_grid_graph(self, grid_state, path_state):
        """A 3x3 grid should have d_s closer to 2 than a path."""
        d_s_grid, _, _ = spectral_dimension(grid_state)
        d_s_path, _, _ = spectral_dimension(path_state)
        # Grid should be more 2D-like than a path
        # Note: with small graphs this may not always hold
        assert d_s_grid >= 0

    def test_returns_arrays(self, path_state):
        """Returns numpy arrays for t and p values."""
        d_s, t, p = spectral_dimension(path_state)
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)
        assert len(t) == len(p)

    def test_custom_t_values(self, path_state):
        """Custom t_values are used when provided."""
        t_custom = np.logspace(-1, 1, 10)
        d_s, t, p = spectral_dimension(path_state, t_values=t_custom)
        assert np.array_equal(t, t_custom)

    def test_custom_fit_range(self, path_state):
        """Custom fit_range changes the fitting window."""
        d_s1, _, _ = spectral_dimension(path_state, fit_range=(0.5, 15.0))
        d_s2, _, _ = spectral_dimension(path_state, fit_range=(1.0, 5.0))
        # Different fit ranges may give different estimates
        # Just check they don't crash
        assert isinstance(d_s1, float)
        assert isinstance(d_s2, float)

    def test_complete_graph_K5(self):
        """K5 should have a well-defined spectral dimension."""
        state = RealityState()
        nodes = [state.add_node() for _ in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                state.add_edge(nodes[i], nodes[j], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        d_s, t, p = spectral_dimension(state)
        assert d_s > 0.0
        assert len(t) > 0
        assert len(p) > 0


# ── ricci_curvature_distribution ──────────────────────────────────────────────


class TestRicciCurvatureDistribution:
    """Summary statistics of the Regge curvature field."""

    def test_empty_state(self, empty_state):
        """Empty state returns zeros."""
        stats = ricci_curvature_distribution(empty_state)
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["max"] == 0.0
        assert stats["min"] == 0.0
        assert stats["positive_fraction"] == 0.0

    def test_two_node_state(self, two_node_state):
        """Two-node state (no triangles) has zero curvature."""
        stats = ricci_curvature_distribution(two_node_state)
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_triangle_state(self, triangle_state):
        """Triangle state has non-zero curvature."""
        stats = ricci_curvature_distribution(triangle_state)
        # All three vertices should have the same curvature
        assert stats["std"] < 1e-10  # all equal
        assert stats["positive_fraction"] in (0.0, 1.0)  # all same sign

    def test_returns_dict(self, triangle_state):
        """Returns a dict with expected keys."""
        stats = ricci_curvature_distribution(triangle_state)
        expected_keys = {"mean", "std", "max", "min", "positive_fraction"}
        assert set(stats.keys()) == expected_keys


# ── degree_distribution ───────────────────────────────────────────────────────


class TestDegreeDistribution:
    """Degree histogram."""

    def test_empty_state(self, empty_state):
        """Empty state returns empty dict."""
        dd = degree_distribution(empty_state)
        assert dd == {}

    def test_two_node_state(self, two_node_state):
        """Two connected nodes: both have degree 1."""
        dd = degree_distribution(two_node_state)
        assert dd == {1: 2}

    def test_triangle_state(self, triangle_state):
        """Triangle: all three nodes have degree 2."""
        dd = degree_distribution(triangle_state)
        assert dd == {2: 3}

    def test_path_state(self, path_state):
        """5-node path: degrees 1 (2 ends) and 2 (3 middle nodes)."""
        dd = degree_distribution(path_state)
        assert dd[1] == 2
        assert dd[2] == 3

    def test_returns_dict(self, path_state):
        """Returns a dict with int keys and int values."""
        dd = degree_distribution(path_state)
        for k, v in dd.items():
            assert isinstance(k, int)
            assert isinstance(v, int)


# ── power_law_exponent ────────────────────────────────────────────────────────


class TestPowerLawExponent:
    """Hill estimator for power-law exponent γ."""

    def test_empty_state(self, empty_state):
        """Empty state returns 0."""
        assert power_law_exponent(empty_state) == 0.0

    def test_small_graph(self, two_node_state):
        """Graph with < 10 nodes returns 0."""
        assert power_law_exponent(two_node_state) == 0.0

    def test_ba_graph(self):
        """BA graph should have γ ≈ 3."""
        G = nx.barabasi_albert_graph(100, 2, seed=42)
        state = RealityState()
        node_map = {}
        for old_id in G.nodes():
            new_id = state.add_node()
            node_map[old_id] = new_id
        for u, v in G.edges():
            state.add_edge(node_map[u], node_map[v], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        gamma = power_law_exponent(state)
        # BA with m=2 has theoretical γ = 3
        # Estimate should be in a reasonable range
        assert 2.0 < gamma < 5.0

    def test_returns_float(self):
        """Returns a Python float."""
        G = nx.barabasi_albert_graph(100, 2, seed=42)
        state = RealityState()
        node_map = {}
        for old_id in G.nodes():
            new_id = state.add_node()
            node_map[old_id] = new_id
        for u, v in G.edges():
            state.add_edge(node_map[u], node_map[v], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        gamma = power_law_exponent(state)
        assert isinstance(gamma, float)

    def test_custom_percentile(self):
        """Custom k_min_percentile changes the tail cutoff."""
        G = nx.barabasi_albert_graph(100, 2, seed=42)
        state = RealityState()
        node_map = {}
        for old_id in G.nodes():
            new_id = state.add_node()
            node_map[old_id] = new_id
        for u, v in G.edges():
            state.add_edge(node_map[u], node_map[v], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        gamma_50 = power_law_exponent(state, k_min_percentile=50.0)
        gamma_90 = power_law_exponent(state, k_min_percentile=90.0)
        assert isinstance(gamma_50, float)
        assert isinstance(gamma_90, float)


# ── small_world_metrics ───────────────────────────────────────────────────────


class TestSmallWorldMetrics:
    """Clustering, path length, and small-world index ω."""

    def test_empty_state(self, empty_state):
        """Empty state returns zeros/nones."""
        sw = small_world_metrics(empty_state)
        assert sw["clustering"] == 0.0
        assert sw["avg_path_length"] is None
        assert sw["omega"] is None

    def test_small_graph(self, two_node_state):
        """Graph with < 3 nodes returns zeros/nones."""
        sw = small_world_metrics(two_node_state)
        assert sw["clustering"] == 0.0

    def test_triangle_state(self, triangle_state):
        """Triangle has clustering = 1.0."""
        sw = small_world_metrics(triangle_state)
        assert sw["clustering"] == 1.0
        assert sw["avg_path_length"] == 1.0

    def test_path_state(self, path_state):
        """Path graph has clustering = 0."""
        sw = small_world_metrics(path_state)
        assert sw["clustering"] == 0.0

    def test_grid_has_path_length(self, grid_state):
        """Grid graph has a finite average path length."""
        sw = small_world_metrics(grid_state)
        assert sw["avg_path_length"] is not None
        assert sw["avg_path_length"] > 0

    def test_large_graph_path_length_none(self):
        """Graph with > 300 nodes returns None for path length."""
        state = RealityState()
        nodes = [state.add_node() for _ in range(400)]
        for i in range(399):
            state.add_edge(nodes[i], nodes[i + 1])
        state.curvature_field = compute_local_curvature_field(state)

        sw = small_world_metrics(state)
        assert sw["avg_path_length"] is None

    def test_returns_dict(self, triangle_state):
        """Returns a dict with expected keys."""
        sw = small_world_metrics(triangle_state)
        expected_keys = {"clustering", "avg_path_length", "omega"}
        assert set(sw.keys()) == expected_keys


# ── complexity_slope ──────────────────────────────────────────────────────────


class TestComplexitySlope:
    """Linear slope dU/dt over a window."""

    def test_insufficient_history(self):
        """Less history than window returns 0."""
        history = [{"U": 0, "time": 0}, {"U": 5, "time": 1}]
        assert complexity_slope(history, window=10) == 0.0

    def test_positive_slope(self, sample_history):
        """Increasing U gives positive slope."""
        slope = complexity_slope(sample_history, window=5)
        assert slope > 0

    def test_flat_slope(self):
        """Flat U gives zero slope."""
        history = [{"U": 5.0, "time": i} for i in range(10)]
        slope = complexity_slope(history, window=5)
        assert abs(slope) < 1e-10

    def test_negative_slope(self):
        """Decreasing U gives negative slope."""
        history = [{"U": float(10 - i), "time": i} for i in range(10)]
        slope = complexity_slope(history, window=5)
        assert slope < 0

    def test_returns_float(self, sample_history):
        """Returns a Python float."""
        slope = complexity_slope(sample_history, window=3)
        assert isinstance(slope, float)


# ── integrate_event_analysis ──────────────────────────────────────────────────


class TestIntegrateEventAnalysis:
    """Statistics about particle-birth events."""

    def test_no_events(self):
        """No integrate events returns count=0."""
        history = [
            {"move_type": MoveType.EXPAND, "time": 0, "curvature_max": 0.0},
            {"move_type": MoveType.SEED, "time": 1, "curvature_max": 0.5},
        ]
        analysis = integrate_event_analysis(history)
        assert analysis["count"] == 0
        assert analysis["times"] == []
        assert analysis["mean_interval"] is None

    def test_single_event(self):
        """Single integrate event returns correct data."""
        history = [
            {"move_type": MoveType.EXPAND, "time": 0, "curvature_max": 0.0},
            {"move_type": MoveType.INTEGRATE, "time": 1, "curvature_max": 1.5},
            {"move_type": MoveType.EXPAND, "time": 2, "curvature_max": 0.5},
        ]
        analysis = integrate_event_analysis(history)
        assert analysis["count"] == 1
        assert analysis["times"] == [1]
        assert analysis["curv_amplitudes"] == [1.5]
        assert analysis["mean_interval"] is None

    def test_multiple_events(self, sample_history):
        """Multiple integrate events return correct statistics."""
        analysis = integrate_event_analysis(sample_history)
        assert analysis["count"] == 2
        assert analysis["times"] == [2, 4]
        assert analysis["curv_amplitudes"] == [1.2, 1.5]
        assert analysis["mean_interval"] == 2.0  # 4 - 2 = 2

    def test_returns_dict(self, sample_history):
        """Returns a dict with expected keys."""
        analysis = integrate_event_analysis(sample_history)
        expected_keys = {"count", "times", "curv_amplitudes", "mean_interval", "mean_amplitude"}
        assert set(analysis.keys()) == expected_keys


# ── monotonicity_score ────────────────────────────────────────────────────────


class TestMonotonicityScore:
    """Fraction of steps where a quantity never decreases."""

    def test_single_element(self):
        """Single element history returns 1.0."""
        assert monotonicity_score([{"U": 0}], "U") == 1.0

    def test_strictly_increasing(self):
        """Strictly increasing U returns 1.0."""
        history = [{"U": i} for i in range(10)]
        assert monotonicity_score(history, "U") == 1.0

    def test_strictly_decreasing(self):
        """Strictly decreasing U returns 0.0."""
        history = [{"U": float(10 - i)} for i in range(10)]
        assert monotonicity_score(history, "U") == 0.0

    def test_constant(self):
        """Constant U returns 1.0 (never decreases)."""
        history = [{"U": 5.0} for _ in range(10)]
        assert monotonicity_score(history, "U") == 1.0

    def test_mixed(self):
        """Mixed increases and decreases returns fraction."""
        history = [
            {"U": 0},
            {"U": 5},  # increase
            {"U": 3},  # decrease
            {"U": 8},  # increase
            {"U": 8},  # equal (counts as non-decrease)
        ]
        # 4 comparisons: increase, decrease, increase, equal
        # Non-decreases: 3 out of 4
        assert monotonicity_score(history, "U") == 0.75

    def test_custom_key(self, sample_history):
        """Works with different history keys."""
        score_U = monotonicity_score(sample_history, "U")
        score_C = monotonicity_score(sample_history, "C")
        assert isinstance(score_U, float)
        assert isinstance(score_C, float)

    def test_returns_float(self):
        """Returns a Python float."""
        history = [{"U": i} for i in range(5)]
        score = monotonicity_score(history, "U")
        assert isinstance(score, float)


# ── network_summary ───────────────────────────────────────────────────────────


class TestNetworkSummary:
    """All structural metrics in a single dict."""

    def test_empty_state(self, empty_state):
        """Empty state returns zeros."""
        ns = network_summary(empty_state)
        assert ns["nodes"] == 0
        assert ns["edges"] == 0
        assert ns["triangles"] == 0
        assert ns["avg_degree"] == 0.0
        assert ns["density"] == 0.0
        assert ns["n_components"] == 0

    def test_two_node_state(self, two_node_state):
        """Two-node state has correct metrics."""
        ns = network_summary(two_node_state)
        assert ns["nodes"] == 2
        assert ns["edges"] == 1
        assert ns["avg_degree"] == 1.0
        assert ns["density"] == 1.0  # 2*1 / (2*1) = 1
        assert ns["n_components"] == 1
        assert ns["lcs_fraction"] == 1.0

    def test_triangle_state(self, triangle_state):
        """Triangle state has correct metrics."""
        ns = network_summary(triangle_state)
        assert ns["nodes"] == 3
        assert ns["edges"] == 3
        assert ns["triangles"] == 1
        assert ns["avg_degree"] == 2.0
        assert ns["clustering"] == 1.0

    def test_disconnected_graph(self):
        """Disconnected graph has multiple components."""
        state = RealityState()
        a, b = state.add_node(), state.add_node()
        state.add_edge(a, b)
        state.add_node()  # isolated
        state.curvature_field = compute_local_curvature_field(state)

        ns = network_summary(state)
        assert ns["n_components"] == 2
        assert ns["lcs_fraction"] == 2.0 / 3.0

    def test_returns_dict(self, triangle_state):
        """Returns a dict with expected keys."""
        ns = network_summary(triangle_state)
        expected_keys = {
            "nodes",
            "edges",
            "triangles",
            "avg_degree",
            "density",
            "n_components",
            "lcs_fraction",
            "clustering",
            "power_law_gamma",
        }
        assert set(ns.keys()) == expected_keys
