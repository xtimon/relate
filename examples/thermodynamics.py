"""
examples/thermodynamics.py
===========================
Thermodynamics and the arrow of time in RELATE.

Tests three hypotheses:

  H1 — SECOND LAW
       Kolmogorov-complexity proxy U(t) never decreases.

  H2 — PARTICLE BIRTHS AND CURVATURE SPIKES
       Integrate events (particle births) coincide with local maxima of
       curvature amplitude max|δ|.  If true: geometry drives matter creation.

  H3 — SPECTRAL DIMENSION
       At equilibrium the graph has a well-defined effective dimensionality
       d_s estimable from Laplacian-eigenvalue diffusion.

Usage
-----
    python examples/thermodynamics.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import relate
from relate.dynamics import MoveType
from relate import RealitySimulation
from relate.measures import (
    complexity_slope,
    integrate_event_analysis,
    monotonicity_score,
    ricci_curvature_distribution,
    spectral_dimension,
)

np.random.seed(42)
random.seed(42)

# ── Run simulation ─────────────────────────────────────────────────────────────
print("RELATE — Thermodynamics & Arrow of Time")
print("Running 300-step simulation...\n")

sim = RealitySimulation(record_all_states=False)
history = sim.run(steps=300, verbose=True)

times      = [h["time"]         for h in history]
U_vals     = [h["U"]            for h in history]
C_vals     = [h["C"]            for h in history]
I_vals     = [h["I"]            for h in history]
curv_max   = [h["curvature_max"] for h in history]
curv_std   = [h["curvature_std"] for h in history]

# ── H1: Second law ─────────────────────────────────────────────────────────────
mono_U = monotonicity_score(history, "U")
mono_C = monotonicity_score(history, "C")
slope  = complexity_slope(history, window=50)

print(f"\n── H1: Second Law ───────────────────────────────")
print(f"  U monotonicity : {mono_U:.3f}  (1.0 = never decreased)")
print(f"  C monotonicity : {mono_C:.3f}")
print(f"  dU/dt (last 50): {slope:+.3f}")
if mono_U > 0.9:
    print("  ✓ Complexity grows monotonically — second law HOLDS")
else:
    print("  ✗ Complexity has reversals — second law VIOLATED")

# ── H2: Integrate events and curvature spikes ─────────────────────────────────
events = integrate_event_analysis(history)
print(f"\n── H2: Particle Births & Curvature ─────────────")
print(f"  Integrate events: {events['count']}")
if events["count"] > 0:
    print(f"  Mean curv at birth: {events['mean_amplitude']:.4e}")
    if events["mean_interval"]:
        print(f"  Mean inter-event interval: {events['mean_interval']:.1f} steps")

    # Check whether integrate events align with curvature peaks
    curv_arr = np.array(curv_max)
    peaks, _ = find_peaks(curv_arr, distance=3)
    peak_times = set(times[p] for p in peaks)
    event_times = set(events["times"])
    # Count events within ±3 steps of a curvature peak
    near_peak = sum(
        1 for t in event_times
        if any(abs(t - pt) <= 3 for pt in peak_times)
    )
    frac = near_peak / max(1, events["count"])
    print(f"  Events near curvature peak (±3 steps): {near_peak}/{events['count']} ({frac:.0%})")
    if frac > 0.5:
        print("  ✓ Particle births correlate with curvature spikes")
    else:
        print("  ~ No strong correlation found")

# ── H3: Spectral dimension ─────────────────────────────────────────────────────
print(f"\n── H3: Spectral Dimension ───────────────────────")
d_s, t_diff, p_diff = spectral_dimension(sim.state)
print(f"  d_s ≈ {d_s:.2f}")
if 1.5 < d_s < 2.5:
    print("  → Graph resembles a 2D surface")
elif 2.5 <= d_s < 3.5:
    print("  → Graph resembles a 3D space")
elif d_s < 1.5:
    print("  → Graph is tree-like / 1D")
else:
    print("  → Higher-dimensional or fractal structure")

curv_dist = ricci_curvature_distribution(sim.state)
print(f"\n  Curvature field summary:")
print(f"    mean δ = {curv_dist['mean']:.4e}")
print(f"    std  δ = {curv_dist['std']:.4e}")
print(f"    positive-curvature fraction: {curv_dist['positive_fraction']:.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(20, 13), facecolor="#0f0f23")

    # 1. Complexity U(t) -------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1, facecolor="#1a1a2e")
    ax1.plot(times, U_vals, color="gold", linewidth=2)
    ax1.fill_between(times, 0, U_vals, alpha=0.15, color="gold")
    ax1.set_title(f"H1 — Complexity U(t)\nmonotonicity={mono_U:.3f}",
                  color="white", fontweight="bold")
    ax1.set_xlabel("Time", color="white")
    ax1.set_ylabel("U  [bytes]", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(alpha=0.2, color="white")

    # 2. Integrated information I(t) -------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2, facecolor="#1a1a2e")
    ax2.plot(times, I_vals, color="orange", linewidth=2)
    ax2.fill_between(times, 0, I_vals, alpha=0.15, color="orange")
    ax2.set_title("I(G) — Integrated information", color="white", fontweight="bold")
    ax2.set_xlabel("Time", color="white")
    ax2.set_ylabel("I(G)", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(alpha=0.2, color="white")
    ax2.set_yscale("log")

    # 3. Curvature amplitude + integrate events --------------------------------
    ax3 = fig.add_subplot(2, 3, 3, facecolor="#1a1a2e")
    ax3.fill_between(times, 0, curv_std, alpha=0.2, color="cyan")
    ax3.plot(times, curv_max, "r-", linewidth=2, label="max|δ|")
    ax3.plot(times, curv_std, "b-", linewidth=1.5, label="σ(δ)")

    # Mark peaks
    curv_arr = np.array(curv_max)
    peaks, _ = find_peaks(curv_arr, distance=3)
    ax3.scatter([times[p] for p in peaks], curv_arr[peaks],
                c="white", s=30, zorder=4, marker="^", alpha=0.6,
                label="Curvature peaks")

    # Mark integrate events
    if events["count"] > 0:
        ev_amplitudes = [
            curv_max[min(t - 1, len(curv_max) - 1)]
            for t in events["times"]
        ]
        ax3.scatter(events["times"], ev_amplitudes, c="yellow", s=100,
                    zorder=5, marker="*", edgecolors="orange",
                    linewidths=0.5, label="Particle births")

    ax3.set_yscale("log")
    ax3.set_title("H2 — Curvature spikes & particle births",
                  color="white", fontweight="bold")
    ax3.set_xlabel("Time", color="white")
    ax3.set_ylabel("Amplitude", color="white")
    ax3.legend(fontsize=8)
    ax3.tick_params(colors="white")
    ax3.grid(alpha=0.2, color="white")

    # 4. Return probability (spectral dimension) --------------------------------
    ax4 = fig.add_subplot(2, 3, 4, facecolor="#1a1a2e")
    if len(t_diff) > 0 and len(p_diff) > 0:
        ax4.loglog(t_diff, p_diff, "cyan", linewidth=2, label="p(t)")
        # Reference slopes
        for d_ref, color, ls in [(1, "#aaa", ":"), (2, "lime", "--"), (3, "magenta", "-.")]:
            p_ref = t_diff[5] ** (d_ref / 2) * p_diff[5] * t_diff ** (-d_ref / 2)
            ax4.loglog(t_diff, p_ref, color=color, linewidth=1, linestyle=ls,
                       label=f"d={d_ref} slope")
        ax4.axvspan(0.5, 15.0, alpha=0.08, color="white", label="fit window")
        ax4.legend(fontsize=8)
    ax4.set_title(f"H3 — Spectral dimension  d_s ≈ {d_s:.2f}",
                  color="white", fontweight="bold")
    ax4.set_xlabel("Diffusion time t", color="white")
    ax4.set_ylabel("Return probability p(t)", color="white")
    ax4.tick_params(colors="white")
    ax4.grid(alpha=0.15, color="white")

    # 5. Curvature distribution histogram --------------------------------------
    ax5 = fig.add_subplot(2, 3, 5, facecolor="#1a1a2e")
    curv_vals = list(sim.state.curvature_field.values())
    if curv_vals:
        ax5.hist(curv_vals, bins=30, color="coral", edgecolor="none", alpha=0.8)
        ax5.axvline(0, color="white", linestyle="--", linewidth=1)
    ax5.set_title("Final curvature field distribution δ(v)",
                  color="white", fontweight="bold")
    ax5.set_xlabel("Local curvature δ", color="white")
    ax5.set_ylabel("Count", color="white")
    ax5.tick_params(colors="white")
    ax5.grid(alpha=0.2, color="white")

    # 6. Move-type entropy (diversity of dynamics) -----------------------------
    ax6 = fig.add_subplot(2, 3, 6, facecolor="#1a1a2e")
    move_types = [MoveType.EXPAND, MoveType.INTEGRATE, MoveType.CONNECT, MoveType.SEED, MoveType.DEFLATE]
    move_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#95a5a6"]
    counts = [sum(1 for h in history if h["move_type"] is mt) for mt in move_types]
    move_type_names = [mt.name.lower() for mt in move_types]
    bars = ax6.bar(move_type_names, counts, color=move_colors, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", color="white", fontsize=10)
    ax6.set_title("Move-type distribution", color="white", fontweight="bold")
    ax6.set_xlabel("Move type", color="white")
    ax6.set_ylabel("Count", color="white")
    ax6.tick_params(colors="white")
    ax6.grid(alpha=0.2, color="white", axis="y")

    plt.suptitle("RELATE — Thermodynamics & Arrow of Time",
                 fontsize=15, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("thermodynamics.png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f23")
    print("\nSaved thermodynamics.png")
    plt.show()
