"""
relate — Relational Evolutionary Lattice for Algorithmic Time and Experience
=============================================================================

Minimal import for most experiments::

    from relate import RealitySimulation
    sim = RealitySimulation(alpha=0.3, beta=3.0)
    history = sim.run(steps=200)
    sim.summary()

Full API surface::

    from relate import (
        # Core data structure
        RealityState,

        # Physics observables
        compute_local_curvature_field,
        compute_curvature,
        compute_integrated_info,
        serialize_state,
        compute_accumulated_complexity,

        # Dynamics primitives
        generate_moves,
        apply_move,
        score_move_cheap,

        # Simulation engine
        RealitySimulation,

        # Visualisation
        create_perfect_animation,
        plot_final_universe,
    )

Extending the simulation
------------------------
Pass a custom ``move_generator`` to :class:`RealitySimulation` to add new
move types without modifying library code::

    def my_moves(state):
        moves = relate.generate_moves(state)          # keep built-ins
        moves.append(('rewire', (u, v, w), 2.0))      # add custom move
        return moves

    sim = RealitySimulation(move_generator=my_moves)
"""

from .dynamics import apply_move, generate_moves, score_move_cheap
from .measures import (
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
from .physics import (
    compute_accumulated_complexity,
    compute_curvature,
    compute_integrated_info,
    compute_local_curvature_field,
    serialize_state,
)
from .simulator import RealitySimulation
from .state import RealityState
from .viz import create_perfect_animation, plot_final_universe

__version__ = "0.1.0"

__all__ = [
    # state
    "RealityState",
    # physics
    "compute_local_curvature_field",
    "compute_curvature",
    "compute_integrated_info",
    "serialize_state",
    "compute_accumulated_complexity",
    # dynamics
    "generate_moves",
    "apply_move",
    "score_move_cheap",
    # simulator
    "RealitySimulation",
    # measures
    "spectral_dimension",
    "ricci_curvature_distribution",
    "degree_distribution",
    "power_law_exponent",
    "small_world_metrics",
    "complexity_slope",
    "integrate_event_analysis",
    "monotonicity_score",
    "network_summary",
    # viz
    "create_perfect_animation",
    "plot_final_universe",
]
