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


def build_sss_states():
    """Build the 12 SSS initial states in the 49-dim Hilbert space."""
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)

    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)

    return [
        0.5*state_00 + 0.5*state_01 + 0.5*state_10 + 0.5*state_11,
        0.5*state_00 - 0.5*state_01 - 0.5*state_10 + 0.5*state_11,
        0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 - 0.5*state_11,
        0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 - 0.5*state_11,
        state_00,
        state_11,
        0.5*state_00 + 0.5*state_01 + 0.5*state_10 - 0.5*state_11,
        0.5*state_00 - 0.5*state_01 - 0.5*state_10 - 0.5*state_11,
        0.5*state_00 + 0.5j*state_01 + 0.5j*state_10 + 0.5*state_11,
        0.5*state_00 - 0.5j*state_01 - 0.5j*state_10 + 0.5*state_11,
        state_00/np.sqrt(2) + 1j*state_11/np.sqrt(2),
        state_00/np.sqrt(2) - 1j*state_11/np.sqrt(2),
    ]


def compute_infidelity(sim, ini_state, theta, t_gate):
    """Compute infidelity for a given initial state.

    The ideal CZ gate with local Rz corrections transforms:
      |00⟩ → |00⟩
      |01⟩ → exp(+i*θ) |01⟩
      |10⟩ → exp(+i*θ) |10⟩
      |11⟩ → exp(+i*(2θ+π)) |11⟩ = -exp(+2i*θ) |11⟩
    """
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)

    # Get actual final state
    res = sim._get_gate_result_TO(
        phase_amp=X_TO[0],
        omega=X_TO[1] * sim.rabi_eff,
        phase_init=X_TO[2],
        delta=X_TO[3] * sim.rabi_eff,
        t_gate=t_gate,
        state_mat=ini_state,
    )[:, -1]

    # Build ideal final state: (Rz⊗Rz) · CZ · |ψ₀⟩
    # Decompose initial state into computational basis
    c00 = np.vdot(state_00, ini_state)
    c01 = np.vdot(state_01, ini_state)
    c10 = np.vdot(state_10, ini_state)
    c11 = np.vdot(state_11, ini_state)

    # Apply CZ (flips sign of |11⟩) then Rz⊗Rz
    psi_ideal = (c00 * state_00 +
                 c01 * np.exp(+1j * theta) * state_01 +
                 c10 * np.exp(+1j * theta) * state_10 +
                 c11 * np.exp(+1j * (2*theta + np.pi)) * state_11)

    fid = np.abs(np.vdot(res, psi_ideal)) ** 2
    return 1.0 - fid


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

    t_gate = X_TO[5] * sim.time_scale
    time_ns = np.linspace(0, t_gate * 1e9, 1000)
    theta = X_TO[4]

    pops = np.zeros((N_SSS, 3, 1000))
    infidelities = np.zeros(N_SSS)

    sss_states = build_sss_states()

    for i in range(N_SSS):
        # Get population evolution
        result = sim._diagnose_run_TO(X_TO, f"SSS-{i}")
        pops[i, 0, :] = result[0] / 2.0
        pops[i, 1, :] = result[1] / 2.0
        pops[i, 2, :] = result[2] / 2.0

        # Calculate infidelity
        ini = sss_states[i]
        infidelities[i] = compute_infidelity(sim, ini, theta, t_gate)

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
