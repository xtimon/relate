"""
Tests for relate.state — RealityState dataclass.

Covers: node/edge CRUD, triangle tracking, cloning, convenience properties.
"""

import networkx as nx
import pytest

from relate.state import RealityState


class TestRealityState:
    """RealityState is the core data structure — a weighted graph with
    curvature field, triangle cache, and integer node IDs."""

    def test_empty_state(self):
        """A freshly created state has no nodes, edges, or triangles."""
        state = RealityState()
        assert state.node_count == 0
        assert state.edge_count == 0
        assert state.triple_count == 0
        assert state.time == 0
        assert state._next_id == 0
        assert state.largest_component_size() == 0

    def test_add_node_increments_id(self):
        """Node IDs are sequential and never reused."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        c = state.add_node()
        assert a == 0
        assert b == 1
        assert c == 2
        assert state.node_count == 3
        # Verify nodes exist in the graph
        assert list(state.graph.nodes()) == [0, 1, 2]

    def test_add_node_sets_curvature(self):
        """Each new node gets an entry in the curvature field (initially 0)."""
        state = RealityState()
        n = state.add_node()
        assert n in state.curvature_field
        assert state.curvature_field[n] == 0.0

    def test_add_node_sets_created_at(self):
        """Nodes track the time they were created."""
        state = RealityState()
        state.time = 5
        n = state.add_node()
        assert state.graph.nodes[n]["created_at"] == 5

    def test_add_edge_between_existing_nodes(self):
        """Adding an edge between two existing nodes works."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b, weight=1.0)
        assert state.edge_count == 1
        assert state.graph.has_edge(a, b)
        assert state.graph[a][b]["weight"] == 1.0

    def test_add_edge_to_nonexistent_node_is_silent(self):
        """Adding an edge where one endpoint doesn't exist does nothing."""
        state = RealityState()
        a = state.add_node()
        state.add_edge(a, 999, weight=1.0)  # 999 doesn't exist
        assert state.edge_count == 0

    def test_add_edge_creates_triangles(self):
        """When an edge completes a triangle, it's added to the triples set."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        c = state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        assert state.triple_count == 0  # not a triangle yet
        state.add_edge(a, c)  # closes the triangle
        assert state.triple_count == 1
        assert tuple(sorted([a, b, c])) in state.triples

    def test_add_edge_creates_multiple_triangles(self):
        """Adding an edge can create multiple triangles if the endpoints
        share multiple common neighbours."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        c = state.add_node()
        d = state.add_node()
        # Make a connected to b, c, d
        state.add_edge(a, b)
        state.add_edge(a, c)
        state.add_edge(a, d)
        # Now connect b-c and b-d (but not c-d)
        state.add_edge(b, c)  # creates triangle (a,b,c)
        state.add_edge(b, d)  # creates triangle (a,b,d)
        assert state.triple_count == 2

    def test_remove_node_cleans_up(self):
        """Removing a node removes it from graph, curvature field, and triples."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        c = state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        state.add_edge(a, c)
        assert state.triple_count == 1

        state.remove_node(a)
        assert a not in state.graph
        assert a not in state.curvature_field
        assert state.triple_count == 0  # triangle involved 'a'

    def test_remove_nonexistent_node_is_safe(self):
        """Removing a node that doesn't exist doesn't crash."""
        state = RealityState()
        state.remove_node(999)  # should not raise

    def test_clone_is_independent(self):
        """clone() returns a fully independent deep copy."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b, weight=2.0)
        state.time = 10

        cloned = state.clone()
        assert cloned.node_count == state.node_count
        assert cloned.edge_count == state.edge_count
        assert cloned.time == state.time
        assert cloned.graph[a][b]["weight"] == 2.0

        # Mutating the clone should not affect the original
        cloned.add_node()
        assert cloned.node_count == state.node_count + 1
        assert state.node_count == 2

        # Mutating the original should not affect the clone
        state.add_node()
        assert state.node_count == 3
        assert cloned.node_count == 3  # wait, clone had 3, original now 3
        # Actually: original had 2, clone had 2+1=3, original now 2+1=3
        # Let's verify they're different objects
        assert id(state.graph) != id(cloned.graph)

    def test_largest_component_size_single(self):
        """With one component, LCS = node_count."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b)
        assert state.largest_component_size() == 2

    def test_largest_component_size_disconnected(self):
        """With multiple components, LCS is the size of the largest."""
        state = RealityState()
        # Component 1: 3 nodes
        a = state.add_node()
        b = state.add_node()
        c = state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        # Component 2: 2 nodes
        d = state.add_node()
        e = state.add_node()
        state.add_edge(d, e)
        # Component 3: 1 isolated node
        state.add_node()

        assert state.largest_component_size() == 3

    def test_largest_component_size_empty(self):
        """Empty graph has LCS = 0."""
        state = RealityState()
        assert state.largest_component_size() == 0

    def test_repr(self):
        """__repr__ gives a useful summary string."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b)
        r = repr(state)
        assert "RealityState" in r
        assert "N=2" in r
        assert "E=1" in r
        assert "triangles=0" in r

    def test_node_count_property(self):
        """node_count matches the graph."""
        state = RealityState()
        assert state.node_count == 0
        state.add_node()
        assert state.node_count == 1
        state.add_node()
        assert state.node_count == 2

    def test_edge_count_property(self):
        """edge_count matches the graph."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        assert state.edge_count == 0
        state.add_edge(a, b)
        assert state.edge_count == 1

    def test_triple_count_property(self):
        """triple_count matches the triples set."""
        state = RealityState()
        assert state.triple_count == 0
        a, b, c = state.add_node(), state.add_node(), state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        state.add_edge(a, c)
        assert state.triple_count == 1

    def test_clone_preserves_next_id(self):
        """Cloning preserves the next available node ID."""
        state = RealityState()
        state.add_node()  # uses 0, next_id = 1
        state.add_node()  # uses 1, next_id = 2
        cloned = state.clone()
        n = cloned.add_node()
        assert n == 2  # next_id was preserved

    def test_clone_preserves_triples(self):
        """Cloning preserves the triangle cache."""
        state = RealityState()
        a, b, c = state.add_node(), state.add_node(), state.add_node()
        state.add_edge(a, b)
        state.add_edge(b, c)
        state.add_edge(a, c)
        cloned = state.clone()
        assert cloned.triple_count == 1
        assert tuple(sorted([a, b, c])) in cloned.triples

    def test_add_edge_default_weight(self):
        """Default edge weight is 1.0."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b)
        assert state.graph[a][b]["weight"] == 1.0

    def test_add_edge_custom_weight(self):
        """Custom edge weight is preserved."""
        state = RealityState()
        a = state.add_node()
        b = state.add_node()
        state.add_edge(a, b, weight=0.5)
        assert state.graph[a][b]["weight"] == 0.5
