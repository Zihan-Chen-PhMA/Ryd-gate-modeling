#!/usr/bin/env python3
"""Plot SSS population evolution using Schrödinger solver.

Generates two figures:
  - docs/figures/population_evolution_sss_comparison.png
    3×4 grid, one subplot per SSS state
  - docs/figures/population_evolution_sss_averaged.png
    Single panel, SSS-averaged populations

Usage:
    uv run python scripts/plot_population_evolution_sch.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.ideal_cz import CZGateSimulator


# ── Initial Optimized TO pulse parameters ──────────────────────────────────
# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t
#     - A = -0.64168872: Cosine amplitude (radians)
#     - ω = 1.14372811: Modulation frequency
#     - φ₀ =0.35715965: Initial phase
#     - δ: Linear chirp rate
#     - θ: Single-qubit Z rotation angle
#     - T: Gate time
# The protocol φ(t) = A·cos(ωt + φ₀) + δ·t with
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]

N_SSS = 12

# Aggregated population categories
CATEGORIES = ['Intermediate', 'Rydberg |r1⟩', 'Garbage Rydberg |r2⟩']
COLORS = {'Intermediate': 'tab:blue', 'Rydberg |r1⟩': 'tab:red',
           'Garbage Rydberg |r2⟩': 'gray'}


def run_schrodinger():
    """Run Schrödinger solver for all 12 SSS states.

    Returns
    -------
    pops : ndarray, shape (12, 3, 1000)
        Population for each (state, category, timestep).
    time_ns : ndarray, shape (1000,)
    infidelities : ndarray, shape (12,)
        Infidelity 1 - F for each SSS initial state.
    """
    sim = CZGateSimulator(decayflag=False, param_set='our', strategy='TO',
                          blackmanflag=False)
    sim.setup_protocol(X_TO)

    t_gate = X_TO[5] * sim.time_scale
    time_ns = np.linspace(0, t_gate * 1e9, 1000)

    pops = np.zeros((N_SSS, 3, 1000))
    infidelities = np.zeros(N_SSS)

    for i in range(N_SSS):
        label = f"SSS-{i}"
        # Get population evolution
        result = sim.diagnose_run(initial_state=label)
        pops[i, 0, :] = result[0] / 2.0
        pops[i, 1, :] = result[1] / 2.0
        pops[i, 2, :] = result[2] / 2.0

        # Calculate infidelity
        infidelities[i] = sim.state_infidelity(label)

    return pops, time_ns, infidelities


def plot_sss_comparison(sch_pops, sch_time, infidelities, outpath):
    """Plot 3×4 grid for each SSS state with infidelity labels."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))

    for i, ax in enumerate(axes.flat):
        for c, cat in enumerate(CATEGORIES):
            color = COLORS[cat]
            ax.plot(sch_time, sch_pops[i, c, :], color=color, linestyle='-', lw=1.3)
        ax.set_title(f'SSS-{i}', fontsize=10)
        ax.set_xlabel('Time (ns)', fontsize=8)
        ax.set_ylabel('Population', fontsize=8)
        ax.tick_params(labelsize=7)

        # Infidelity text
        inf = infidelities[i]
        ax.text(0.02, 0.98, f'1−F = {inf:.2e}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Auto-scale y-axis per subplot
        ymax = np.max(sch_pops[i])
        if ymax < 1e-6:
            ymax = 0.01
        ax.set_ylim(-0.02 * ymax, ymax * 1.15)

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat in CATEGORIES:
        color = COLORS[cat]
        legend_elements.append(Line2D([0], [0], color=color, linestyle='-', lw=1.5,
                                       label=f'{cat}'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('SSS Population Evolution — Schrödinger', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)


def plot_sss_averaged(sch_pops, sch_time, infidelities, outpath):
    """Plot SSS-averaged populations with average infidelity."""
    sch_avg = sch_pops.mean(axis=0)  # (3, 1000)
    avg_infidelity = np.mean(infidelities)

    fig, ax = plt.subplots(figsize=(10, 6))
    for c, cat in enumerate(CATEGORIES):
        color = COLORS[cat]
        ax.plot(sch_time, sch_avg[c], color=color, linestyle='-', lw=1.8,
                label=f'{cat}')

    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Population (SSS-averaged)', fontsize=12)
    ax.set_title(f'SSS-Averaged Population Evolution — Schrödinger\n'
                 f'Average Infidelity: {avg_infidelity:.2e}', fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)


def main():
    os.makedirs('docs/figures', exist_ok=True)

    print("=" * 60)
    print("SSS Population Evolution — Schrödinger")
    print("=" * 60)

    print("\n[1/2] Running Schrödinger solver (TO strategy, 12 SSS states)")
    sch_pops, sch_time, infidelities = run_schrodinger()

    print("\nInfidelities per SSS state:")
    for i, inf in enumerate(infidelities):
        print(f"  SSS-{i:2d}: 1-F = {inf:.2e}")
    print(f"  Average: 1-F = {np.mean(infidelities):.2e}")

    print("\n[2/2] Generating figures...")
    plot_sss_comparison(sch_pops, sch_time, infidelities,
                        'docs/figures/population_evolution_sss_comparison.png')
    plot_sss_averaged(sch_pops, sch_time, infidelities,
                      'docs/figures/population_evolution_sss_averaged.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
