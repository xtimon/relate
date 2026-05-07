"""
Visualisation utilities for RealitySimulation results.

Both functions use the 'dark_background' matplotlib style as a context so
they never mutate the caller's global style settings.
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .simulator import RealitySimulation


def create_perfect_animation(
    sim: RealitySimulation,
    history: list[dict],
    step: int = 1,
    fps: int = 5,
    filename: str = "quantum_breath.gif",
) -> tuple[FuncAnimation, str]:
    """
    Build a GIF animation from the states stored in ``sim.all_states``.

    Parameters
    ----------
    sim : RealitySimulation
        A completed simulation with ``record_all_states=True``.
    history : list of dict
        The list returned by ``sim.run()``.
    step : int
        Sample every *step*-th state (reduces file size).
    fps : int
        Frames per second in the output GIF.
    filename : str
        Output file path.

    Returns
    -------
    (FuncAnimation, filename)
    """
    states = sim.all_states
    if not states:
        raise ValueError("sim.all_states is empty — run with record_all_states=True.")

    indices = list(range(0, len(states), step))
    if indices[-1] != len(states) - 1:
        indices.append(len(states) - 1)

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.5, 1]}
        )

        all_max_curv = [
            np.max(np.abs(list(s.curvature_field.values()) or [0])) for s in states
        ]
        global_max = max(all_max_curv) if all_max_curv else 1e-6

        times = [h["time"] for h in history]
        curv_max_hist = [h["curvature_max"] for h in history]
        curv_std_hist = [h["curvature_std"] for h in history]

        current_pos: dict | None = None

        def animate(frame_idx: int) -> list:
            nonlocal current_pos
            ax1.clear()
            ax2.clear()

            state = states[indices[frame_idx]]
            t = state.time

            ax1.set_facecolor("#1a1a2e")
            if state.node_count > 0:
                if frame_idx == 0 or current_pos is None:
                    current_pos = nx.spring_layout(
                        state.graph, k=2, iterations=50, seed=42
                    )
                else:
                    try:
                        current_pos = nx.spring_layout(
                            state.graph, k=2, iterations=15, seed=42, pos=current_pos
                        )
                    except (nx.NetworkXError, ValueError):
                        current_pos = nx.spring_layout(
                            state.graph, k=2, iterations=50, seed=42
                        )

                if state.curvature_field:
                    node_list = list(state.graph.nodes())
                    curv_vals = [state.curvature_field.get(n, 0.0) for n in node_list]
                    nx.draw_networkx_nodes(
                        state.graph, current_pos, ax=ax1,
                        node_color=curv_vals, cmap="RdBu_r",
                        vmin=-global_max, vmax=global_max,
                        node_size=[20 + 80 * abs(c) / global_max for c in curv_vals],
                        alpha=0.9, edgecolors="white", linewidths=0.3,
                    )

                nx.draw_networkx_edges(
                    state.graph, current_pos, ax=ax1,
                    alpha=0.15, width=0.3, edge_color="white",
                )

            ax1.set_title(
                f"Curvature field (t={t}, N={state.node_count})",
                fontsize=12, fontweight="bold", color="white",
            )
            ax1.axis("off")

            ax2.set_facecolor("#1a1a2e")
            slice_end = min(t, len(times))
            if slice_end > 0:
                ct = times[:slice_end]
                cm = curv_max_hist[:slice_end]
                cs = curv_std_hist[:slice_end]
                ax2.fill_between(ct, 0, cs, alpha=0.2, color="cyan")
                ax2.plot(ct, cm, "r-", linewidth=1.5, label="max |δ|")
                ax2.plot(ct, cs, "b-", linewidth=1.5, label="σ(δ)")
                for i, h in enumerate(history[:slice_end]):
                    if h["move_type"] == "integrate" and i < len(cm):
                        ax2.scatter(
                            h["time"], cm[i], c="yellow", s=80, zorder=5,
                            marker="*", edgecolors="orange", linewidths=0.5,
                        )
                ax2.axvline(x=t, color="lime", linestyle="--", alpha=0.5, linewidth=1.5)

            ax2.set_xlabel("Time", color="white")
            ax2.set_ylabel("Amplitude", color="white")
            ax2.set_title("Quantum breath", fontsize=12, fontweight="bold", color="white")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(alpha=0.2, color="white")
            ax2.set_yscale("log")
            ax2.tick_params(colors="white")

            fig.patch.set_facecolor("#0f0f23")
            fig.suptitle(
                f"Reality evolution: step {t}",
                fontsize=14, fontweight="bold", color="white", y=1.01,
            )
            return [ax1, ax2]

        ani = FuncAnimation(
            fig, animate, frames=len(indices), interval=1000 / fps,
            blit=False, repeat=True,
        )
        ani.save(filename, writer=PillowWriter(fps=fps), dpi=80)
        plt.close(fig)

    return ani, filename


def plot_final_universe(sim: RealitySimulation, history: list[dict]) -> None:
    """
    Six-panel static summary figure for a completed simulation.

    Panels: final graph with curvature field, growth curves, quantum breath,
    integrated information, accumulated complexity + move-type distribution.
    """
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(22, 13), facecolor="#0f0f23")
        times = [h["time"] for h in history]

        # 1. Final graph -------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, (1, 2), facecolor="#1a1a2e")
        state = sim.state
        if state.node_count > 0:
            pos = nx.spring_layout(
                state.graph,
                k=1.5 / np.sqrt(state.node_count + 1),
                iterations=100, seed=42,
            )
            if state.curvature_field:
                node_list = list(state.graph.nodes())
                curv_vals = [state.curvature_field.get(n, 0.0) for n in node_list]
                vmax = max(np.max(np.abs(curv_vals)), 1e-6)
                sc = nx.draw_networkx_nodes(
                    state.graph, pos, ax=ax1,
                    node_color=curv_vals, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    node_size=[30 + 120 * abs(c) / vmax for c in curv_vals],
                    alpha=0.9, edgecolors="white", linewidths=0.3,
                )
                cbar = plt.colorbar(sc, ax=ax1, label="Local curvature δ", shrink=0.8)
                cbar.ax.yaxis.label.set_color("white")
                cbar.ax.tick_params(colors="white")
            nx.draw_networkx_edges(
                state.graph, pos, ax=ax1, alpha=0.15, width=0.3, edge_color="white"
            )
        lcs = state.largest_component_size()
        ax1.set_title(
            f"FINAL UNIVERSE\n"
            f"{state.node_count} nodes, {state.edge_count} edges, "
            f"{state.triple_count} triangles\n"
            f"Largest component: {lcs} ({100 * lcs / max(1, state.node_count):.1f}%)",
            fontsize=12, fontweight="bold", color="white",
        )
        ax1.axis("off")

        # 2. Growth & connectivity ---------------------------------------------
        ax2 = fig.add_subplot(2, 3, 3, facecolor="#1a1a2e")
        ax2.plot(times, [h["nodes"] for h in history], "cyan", label="Nodes", linewidth=2)
        ax2.plot(times, [h["edges"] for h in history], "magenta", label="Edges", linewidth=2)
        ax2.plot(times, [h["triples"] for h in history], "yellow", label="Triangles", linewidth=2)
        ax2.plot(
            times, [h["largest_component"] for h in history], "white",
            linestyle="--", label="Largest component", linewidth=2, alpha=0.8,
        )
        ax2.set_xlabel("Time", color="white")
        ax2.set_ylabel("Count", color="white")
        ax2.set_title("Growth & connectivity", color="white", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2, color="white")
        ax2.tick_params(colors="white")

        # 3. Quantum breath ----------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 4, facecolor="#1a1a2e")
        curv_std = [h["curvature_std"] for h in history]
        curv_max = [h["curvature_max"] for h in history]
        ax3.fill_between(times, 0, curv_std, alpha=0.2, color="cyan")
        ax3.plot(times, curv_max, "r-", linewidth=2, label="max |δ|")
        ax3.plot(times, curv_std, "b-", linewidth=2, label="σ(δ)")
        integrate_times = [h["time"] for h in history if h["move_type"] == "integrate"]
        integrate_vals = [curv_max[min(t - 1, len(curv_max) - 1)] for t in integrate_times]
        if integrate_vals:
            ax3.scatter(
                integrate_times, integrate_vals, c="yellow", s=120, zorder=5,
                marker="*", edgecolors="orange", linewidths=0.5, label="Particle births",
            )
        ax3.set_xlabel("Time", color="white")
        ax3.set_ylabel("Amplitude", color="white")
        ax3.set_title("Quantum breath", color="white", fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.2, color="white")
        ax3.set_yscale("log")
        ax3.tick_params(colors="white")

        # 4. Integrated information -------------------------------------------
        ax4 = fig.add_subplot(2, 3, 5, facecolor="#1a1a2e")
        I_vals = [h["I"] for h in history]
        ax4.plot(times, I_vals, "orange", linewidth=2)
        ax4.fill_between(times, 0, I_vals, alpha=0.15, color="orange")
        ax4.set_xlabel("Time", color="white")
        ax4.set_ylabel("I(G)", color="white")
        ax4.set_title("Integrated information", color="white", fontweight="bold")
        ax4.grid(alpha=0.2, color="white")
        ax4.tick_params(colors="white")

        # 5. Complexity + move distribution ------------------------------------
        ax5 = fig.add_subplot(2, 3, 6, facecolor="#1a1a2e")
        ax5b = ax5.twinx()
        ax5.plot(times, [h["U"] for h in history], "brown", linewidth=2, label="U(G)")
        ax5.set_xlabel("Time", color="white")
        ax5.set_ylabel("Accumulated complexity", color="brown")
        ax5.tick_params(axis="y", colors="brown")

        move_colors = {
            "expand": "#3498db", "integrate": "#e74c3c",
            "connect": "#2ecc71", "seed": "#f39c12", "deflate": "#95a5a6",
        }
        window = 20
        bottom = np.zeros(len(times))
        for mt in ("expand", "integrate", "connect", "seed", "deflate"):
            counts = [
                sum(1 for r in history[max(0, i - window + 1): i + 1] if r["move_type"] == mt)
                for i in range(len(history))
            ]
            ax5b.fill_between(
                times, bottom, bottom + np.array(counts),
                alpha=0.3, color=move_colors[mt], label=mt,
            )
            bottom += np.array(counts)
        ax5b.set_ylabel("Move frequency (window 20)", color="white")
        ax5b.tick_params(axis="y", colors="white")
        ax5b.legend(fontsize=7, loc="upper left")
        ax5.set_title("Arrow of time & move types", color="white", fontweight="bold")
        ax5.grid(alpha=0.2, color="white")

        plt.suptitle(
            "RELATE — Quantum Ripple in a Stable Graph Universe",
            fontsize=16, fontweight="bold", color="white", y=1.01,
        )
        plt.tight_layout()
        plt.show()
