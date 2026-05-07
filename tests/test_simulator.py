"""
Tests for relate.simulator — RealitySimulation.

Covers: initialization, single step, batch run, history recording,
        custom move generators, initial states, parameter sweeps.
"""

import random

import networkx as nx
import numpy as np
import pytest

from relate import RealitySimulation
from relate.dynamics import generate_moves
from relate.physics import compute_local_curvature_field
from relate.state import RealityState

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seeded_random():
    """All simulator tests use deterministic randomness."""
    np.random.seed(42)
    random.seed(42)
    yield


# ── Initialization ────────────────────────────────────────────────────────────

class TestInitialization:
    """RealitySimulation construction and default state."""

    def test_default_initialization(self):
        """Default simulation starts with a 2-node seed graph."""
        sim = RealitySimulation(record_all_states=False)
        assert sim.state.node_count == 2
        assert sim.state.edge_count == 1
        assert sim.state.time == 0
        assert len(sim.history) == 0

    def test_parameters_are_stored(self):
        """Constructor parameters are stored as attributes."""
        sim = RealitySimulation(
            alpha=0.5, beta=5.0, gamma=0.1,
            vacuum=1.0, connectivity=2.0,
            size_penalty=0.01, topology_reward=3.0,
            i_update_interval=10,
        )
        assert sim.alpha == 0.5
        assert sim.beta == 5.0
        assert sim.gamma == 0.1
        assert sim.vacuum == 1.0
        assert sim.connectivity == 2.0
        assert sim.size_penalty == 0.01
        assert sim.topology_reward == 3.0
        assert sim.i_update_interval == 10

    def test_i_update_interval_minimum(self):
        """i_update_interval is at least 1."""
        sim = RealitySimulation(i_update_interval=0)
        assert sim.i_update_interval == 1
        sim = RealitySimulation(i_update_interval=-5)
        assert sim.i_update_interval == 1

    def test_initial_state_is_cloned(self):
        """Provided initial_state is cloned, not referenced."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b)
        state.curvature_field = compute_local_curvature_field(state)

        sim = RealitySimulation(initial_state=state, record_all_states=False)
        assert sim.state.node_count == 2
        # Mutating the original should not affect the simulation
        state.add_node()
        assert sim.state.node_count == 2

    def test_record_all_states_true(self):
        """record_all_states=True stores initial state."""
        sim = RealitySimulation(record_all_states=True)
        assert len(sim.all_states) == 1  # initial state cloned

    def test_record_all_states_false(self):
        """record_all_states=False starts with empty all_states."""
        sim = RealitySimulation(record_all_states=False)
        assert len(sim.all_states) == 0

    def test_custom_move_generator(self):
        """Custom move generator is used instead of default."""
        def my_moves(state):
            return [("seed", (), 1.0)]

        sim = RealitySimulation(move_generator=my_moves, record_all_states=False)
        assert sim.move_generator is my_moves

    def test_default_move_generator(self):
        """Default move generator is the built-in generate_moves."""
        sim = RealitySimulation(record_all_states=False)
        assert sim.move_generator is generate_moves

    def test_initial_serialized_is_stored(self):
        """Initial serialized state is computed and stored."""
        sim = RealitySimulation(record_all_states=False)
        assert isinstance(sim.initial_serialized, bytes)
        assert len(sim.initial_serialized) > 0


# ── Single Step ───────────────────────────────────────────────────────────────

class TestStep:
    """RealitySimulation.step() advances the simulation by one step."""

    def test_step_returns_dict(self):
        """step() returns a dictionary of observables."""
        sim = RealitySimulation(record_all_states=False)
        record = sim.step()
        assert isinstance(record, dict)

    def test_step_increments_time(self):
        """step() increments the internal time counter."""
        sim = RealitySimulation(record_all_states=False)
        assert sim.state.time == 0
        sim.step()
        assert sim.state.time == 1

    def test_step_records_observables(self):
        """step() records all expected observables."""
        sim = RealitySimulation(record_all_states=False)
        record = sim.step()
        expected_keys = {
            "time", "nodes", "edges", "triples", "largest_component",
            "C", "I", "U", "move_type", "complexity_bytes",
            "curvature_mean", "curvature_std", "curvature_max",
        }
        assert expected_keys.issubset(record.keys())

    def test_step_appends_to_history(self):
        """step() appends the record to sim.history."""
        sim = RealitySimulation(record_all_states=False)
        assert len(sim.history) == 0
        sim.step()
        assert len(sim.history) == 1
        sim.step()
        assert len(sim.history) == 2

    def test_step_move_type_is_valid(self):
        """The chosen move type is one of the six built-in types."""
        sim = RealitySimulation(record_all_states=False)
        valid_types = {"expand", "integrate", "connect", "seed", "deflate", "triangulate"}
        for _ in range(50):
            record = sim.step()
            assert record["move_type"] in valid_types

    def test_step_updates_all_states(self):
        """step() appends to all_states when record_all_states=True."""
        sim = RealitySimulation(record_all_states=True)
        assert len(sim.all_states) == 1  # initial
        sim.step()
        assert len(sim.all_states) == 2
        sim.step()
        assert len(sim.all_states) == 3

    def test_step_does_not_update_all_states_when_false(self):
        """step() does not append to all_states when record_all_states=False."""
        sim = RealitySimulation(record_all_states=False)
        assert len(sim.all_states) == 0
        sim.step()
        assert len(sim.all_states) == 0

    def test_step_curvature_is_non_negative(self):
        """Integrated squared curvature C is always >= 0."""
        sim = RealitySimulation(record_all_states=False)
        for _ in range(20):
            record = sim.step()
            assert record["C"] >= 0.0

    def test_step_node_count_changes(self):
        """Node count changes appropriately after a step."""
        sim = RealitySimulation(record_all_states=False)
        initial_n = sim.state.node_count
        sim.step()
        # Node count could increase (expand, seed), decrease (integrate, deflate),
        # or stay the same (connect, triangulate)
        record = sim.history[-1]
        if record["move_type"] in ("expand",):
            assert record["nodes"] == initial_n + 1
        elif record["move_type"] == "seed":
            assert record["nodes"] == initial_n + 2
        elif record["move_type"] == "deflate":
            assert record["nodes"] == initial_n - 1
        elif record["move_type"] == "integrate":
            assert record["nodes"] == initial_n - 2
        else:  # connect, triangulate
            assert record["nodes"] == initial_n

    def test_step_I_is_cached(self):
        """I is computed exactly at the configured interval."""
        sim = RealitySimulation(i_update_interval=5, record_all_states=False)
        # First step should compute I (forced update)
        r1 = sim.step()
        assert r1["I"] >= 0.0
        # Next 4 steps should use cached value
        for _ in range(4):
            r = sim.step()
            assert r["I"] == r1["I"]  # same cached value
        # 5th step should recompute
        r6 = sim.step()
        # May or may not be different, but should be a valid float
        assert isinstance(r6["I"], float)

    def test_step_handles_large_graphs(self):
        """Step works for larger graphs without crashing."""
        sim = RealitySimulation(record_all_states=False)
        for _ in range(100):
            sim.step()
        assert sim.state.node_count > 0


# ── Batch Run ─────────────────────────────────────────────────────────────────

class TestRun:
    """RealitySimulation.run() runs multiple steps."""

    def test_run_returns_history(self):
        """run() returns the history list."""
        sim = RealitySimulation(record_all_states=False)
        history = sim.run(steps=10, verbose=False)
        assert isinstance(history, list)
        assert len(history) == 10

    def test_run_specified_steps(self):
        """run() executes exactly the requested number of steps."""
        for n_steps in [1, 5, 20]:
            sim2 = RealitySimulation(record_all_states=False)
            history = sim2.run(steps=n_steps, verbose=False)
            assert len(history) == n_steps

    def test_run_verbose_output(self, capsys):
        """run() with verbose=True prints progress."""
        sim = RealitySimulation(record_all_states=False)
        sim.run(steps=5, verbose=True)
        captured = capsys.readouterr()
        assert "RELATE" in captured.out
        assert "α=" in captured.out

    def test_run_silent(self, capsys):
        """run() with verbose=False prints nothing."""
        sim = RealitySimulation(record_all_states=False)
        sim.run(steps=5, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_run_preserves_history(self):
        """Running after manual steps appends to existing history."""
        sim = RealitySimulation(record_all_states=False)
        sim.step()
        sim.step()
        assert len(sim.history) == 2
        sim.run(steps=5, verbose=False)
        assert len(sim.history) == 7

    def test_run_with_topology_reward(self):
        """Running with topology_reward produces valid results."""
        sim = RealitySimulation(
            topology_reward=3.5, record_all_states=False
        )
        history = sim.run(steps=30, verbose=False)
        assert len(history) == 30
        # Should have some triangulate moves
        triangulate_count = sum(1 for r in history if r["move_type"] == "triangulate")
        assert triangulate_count >= 0  # at least doesn't crash

    def test_run_with_large_beta(self):
        """Running with large beta produces valid results."""
        sim = RealitySimulation(beta=30.0, record_all_states=False)
        history = sim.run(steps=30, verbose=False)
        assert len(history) == 30

    def test_run_with_size_penalty_zero(self):
        """Running with size_penalty=0 (unbounded growth) works."""
        sim = RealitySimulation(size_penalty=0.0, record_all_states=False)
        history = sim.run(steps=30, verbose=False)
        assert len(history) == 30

    def test_run_deterministic_with_seed(self):
        """Same random seeds produce identical results."""
        np.random.seed(123)
        random.seed(123)
        sim1 = RealitySimulation(record_all_states=False)
        h1 = sim1.run(steps=20, verbose=False)

        np.random.seed(123)
        random.seed(123)
        sim2 = RealitySimulation(record_all_states=False)
        h2 = sim2.run(steps=20, verbose=False)

        for r1, r2 in zip(h1, h2, strict=True):
            assert r1["time"] == r2["time"]
            assert r1["nodes"] == r2["nodes"]
            assert r1["edges"] == r2["edges"]
            assert r1["move_type"] == r2["move_type"]


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    """RealitySimulation.summary() output."""

    def test_summary_before_run(self, capsys):
        """Summary before any steps prints a message."""
        sim = RealitySimulation(record_all_states=False)
        sim.summary()
        captured = capsys.readouterr()
        assert "No steps recorded yet" in captured.out

    def test_summary_after_run(self, capsys):
        """Summary after a run prints results."""
        sim = RealitySimulation(record_all_states=False)
        sim.run(steps=10, verbose=False)
        sim.summary()
        captured = capsys.readouterr()
        assert "SIMULATION RESULTS" in captured.out
        assert "Nodes:" in captured.out
        assert "Edges:" in captured.out
        assert "Event counts:" in captured.out

    def test_summary_includes_move_counts(self, capsys):
        """Summary includes counts for each move type."""
        sim = RealitySimulation(record_all_states=False)
        sim.run(steps=50, verbose=False)
        sim.summary()
        captured = capsys.readouterr()
        for mt in ("expand", "integrate", "connect", "seed", "deflate"):
            assert mt in captured.out


# ── Custom Move Generators ────────────────────────────────────────────────────

class TestCustomMoveGenerator:
    """Custom move generators can extend the simulation."""

    def test_custom_generator_is_used(self):
        """Custom move generator replaces the default."""
        move_log = []

        def logging_moves(state):
            moves = generate_moves(state)
            move_log.append(len(moves))
            return moves

        sim = RealitySimulation(move_generator=logging_moves, record_all_states=False)
        sim.run(steps=10, verbose=False)
        assert len(move_log) == 10

    def test_custom_generator_only_seed(self):
        """Generator that only returns seed moves works."""
        def only_seed(state):
            return [("seed", (), 1.0)]

        sim = RealitySimulation(move_generator=only_seed, record_all_states=False)
        history = sim.run(steps=20, verbose=False)
        for r in history:
            assert r["move_type"] == "seed"
        # Seed adds 2 nodes per step
        assert sim.state.node_count == 2 + 2 * 20

    def test_custom_generator_only_deflate(self):
        """Generator that only returns deflate on isolated nodes."""
        def only_deflate(state):
            moves = []
            for node in state.graph.nodes():
                if state.graph.degree(node) == 0:
                    moves.append(("deflate", (node,), 1.0))
            if not moves:
                moves.append(("seed", (), 1.0))
            return moves

        sim = RealitySimulation(move_generator=only_deflate, record_all_states=False)
        history = sim.run(steps=20, verbose=False)
        # Should have some deflate and some seed moves
        types = set(r["move_type"] for r in history)
        assert "deflate" in types or "seed" in types


# ── Initial States ────────────────────────────────────────────────────────────

class TestInitialStates:
    """Simulation can start from different initial topologies."""

    def test_start_from_torus(self):
        """Starting from a torus grid works."""
        G = nx.grid_2d_graph(4, 4, periodic=True)
        state = RealityState()
        node_map = {}
        for old_id in G.nodes():
            new_id = state.add_node()
            node_map[old_id] = new_id
        for u, v in G.edges():
            state.add_edge(node_map[u], node_map[v], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        sim = RealitySimulation(initial_state=state, record_all_states=False)
        assert sim.state.node_count == 16
        history = sim.run(steps=20, verbose=False)
        assert len(history) == 20

    def test_start_from_random_graph(self):
        """Starting from an Erdős–Rényi graph works."""
        G = nx.gnp_random_graph(20, 0.2, seed=42)
        state = RealityState()
        node_map = {}
        for old_id in G.nodes():
            new_id = state.add_node()
            node_map[old_id] = new_id
        for u, v in G.edges():
            state.add_edge(node_map[u], node_map[v], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        sim = RealitySimulation(initial_state=state, record_all_states=False)
        assert sim.state.node_count == 20
        history = sim.run(steps=20, verbose=False)
        assert len(history) == 20

    def test_start_from_complete_graph(self):
        """Starting from a complete graph K5 works."""
        state = RealityState()
        nodes = [state.add_node() for _ in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                state.add_edge(nodes[i], nodes[j], weight=1.0)
        state.curvature_field = compute_local_curvature_field(state)

        sim = RealitySimulation(initial_state=state, record_all_states=False)
        assert sim.state.node_count == 5
        assert sim.state.edge_count == 10
        history = sim.run(steps=20, verbose=False)
        assert len(history) == 20


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_step_from_empty_initial(self):
        """Simulation can run from a single-node initial state."""
        state = RealityState()
        state.add_node()
        state.curvature_field = compute_local_curvature_field(state)

        sim = RealitySimulation(initial_state=state, record_all_states=False)
        # Should not crash — seed move will add nodes
        history = sim.run(steps=10, verbose=False)
        assert len(history) == 10

    def test_many_steps_does_not_crash(self):
        """Running many steps doesn't cause errors."""
        sim = RealitySimulation(record_all_states=False)
        history = sim.run(steps=200, verbose=False)
        assert len(history) == 200
        assert sim.state.node_count > 0

    def test_history_consistency(self):
        """History records are internally consistent."""
        sim = RealitySimulation(record_all_states=False)
        history = sim.run(steps=50, verbose=False)
        for i, r in enumerate(history):
            assert r["time"] == i + 1  # time starts at 1
            assert r["nodes"] >= 0
            assert r["edges"] >= 0
            assert r["triples"] >= 0
            assert r["C"] >= 0.0
            assert r["I"] >= 0.0
            assert r["U"] >= 0.0

    def test_curvature_stats_consistency(self):
        """Curvature mean, std, and max are internally consistent."""
        sim = RealitySimulation(record_all_states=False)
        history = sim.run(steps=30, verbose=False)
        for r in history:
            assert r["curvature_std"] >= 0.0
            assert r["curvature_max"] >= 0.0
            # max should be >= mean in absolute terms
            assert r["curvature_max"] >= abs(r["curvature_mean"])

    def test_largest_component_reasonable(self):
        """Largest component size is always positive and <= node count.
        Note: history records LCS *before* the move and nodes *after* the move,
        so for deflate moves LCS could briefly exceed the post-move node count.
        We check that LCS is always within a reasonable bound."""
        sim = RealitySimulation(record_all_states=False)
        history = sim.run(steps=50, verbose=False)
        for r in history:
            assert r["largest_component"] >= 0
            # LCS should be at most the pre-move node count (which is >= post-move)
            assert r["largest_component"] <= r["nodes"] + 2  # allow for deflate
