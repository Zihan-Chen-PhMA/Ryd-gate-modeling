#!/usr/bin/env python3
"""Plot SSS population evolution comparing Schrödinger and density-matrix solvers.

Generates two figures:
  - docs/figures/population_evolution_sss_comparison.png
    3×4 grid, one subplot per SSS state, both solvers overlaid
  - docs/figures/population_evolution_sss_averaged.png
    Single panel, SSS-averaged populations

Usage:
    uv run python scripts/plot_population_evolution.py
"""

import os
import sys

import numpy as np

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.ideal_cz import CZGateSimulator
from ryd_gate.full_error_model import jax_atom_Evolution


# ── Optimized TO pulse parameters ──────────────────────────────────
X_TO = [0.1122, 1.0431, -0.72565603, 0.0, 0.452, 1.219096]

N_SSS = 12

# Aggregated population categories
CATEGORIES = ['Intermediate', 'Rydberg |r1⟩', 'Garbage Rydberg |r2⟩']
COLORS = {'Intermediate': 'tab:blue', 'Rydberg |r1⟩': 'tab:red',
           'Garbage Rydberg |r2⟩': 'gray'}


def build_sss_states_7level():
    """Build the 12 SSS initial states in the 49-dim Hilbert space.

    Uses the same linear combinations as ``init_SSS_states()`` in
    full_error_model.py, but expressed as (49,) state vectors for the
    7-level Schrödinger solver.
    """
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


def run_schrodinger():
    """Run Schrödinger solver for all 12 SSS states.

    Returns
    -------
    pops : ndarray, shape (12, 3, 1000)
        Population for each (state, category, timestep).
    time_ns : ndarray, shape (1000,)
    """
    sim = CZGateSimulator(param_set='our', strategy='TO',
                          blackmanflag=False)
    t_gate = X_TO[5] * sim.time_scale  # seconds
    time_ns = np.linspace(0, t_gate * 1e9, 1000)

    # Aggregated occupation operators
    op_intermediate = sim._occ_operator(2) + sim._occ_operator(3) + sim._occ_operator(4)
    op_r1 = sim._occ_operator(5)
    op_r2 = sim._occ_operator(6)
    ops = [op_intermediate, op_r1, op_r2]

    sss_states = build_sss_states_7level()
    n_t = 1000
    pops = np.zeros((N_SSS, 3, n_t))

    for i, ini in enumerate(sss_states):
        res = sim._get_gate_result_TO(
            phase_amp=2 * np.pi * X_TO[0],
            omega=X_TO[1] * sim.rabi_eff,
            phase_init=-X_TO[2],
            delta=X_TO[3] * sim.rabi_eff,
            t_gate=t_gate,
            state_mat=ini,
            t_eval=np.linspace(0, t_gate, n_t),
        )  # shape (49, 1000)

        for c, op in enumerate(ops):
            for t in range(n_t):
                psi = res[:, t]
                pops[i, c, t] = np.real(np.conj(psi) @ op @ psi) / 2.0

    return pops, time_ns


def run_density_matrix():
    """Run density-matrix solver for all 12 SSS states.

    Returns
    -------
    pops : ndarray, shape (12, 3, n_points)
        Population for each (state, category, timestep).
    time_ns : ndarray, shape (n_points,)
    """
    print("  Initializing density-matrix model …")
    model = jax_atom_Evolution(blockade=True, ryd_decay=True,
                               mid_decay=True, distance=3)

    Omega = model.rabi_eff / (2 * jnp.pi)  # MHz
    tf = X_TO[5]
    A = X_TO[0]
    omegaf = X_TO[1]
    phi0 = X_TO[2]
    deltaf = X_TO[3]

    amp_420 = lambda t: 1.0
    phase_420 = lambda t: 2 * jnp.pi * (
        A * jnp.cos(2 * jnp.pi * omegaf * Omega * t - phi0) + deltaf * Omega * t
    )
    amp_1013 = lambda t: 1.0

    gate_time = tf / Omega  # μs
    n_points = 200
    tlist = jnp.linspace(0, gate_time, n_points)
    time_ns = np.array(tlist * 1000)  # μs → ns

    # Use the 12 SSS states from the model
    rho0_list = [model.psi_to_rho(psi) for psi in model.SSS_initial_state_list]

    print("  Running density-matrix integration (12 SSS states) …")
    sol = model.integrate_rho_multi_jax(tlist, amp_420, phase_420, amp_1013, rho0_list)
    # sol shape: (12, n_points, 100, 100)

    # Aggregated occupation operators
    op_intermediate = model.occ_operator('e1') + model.occ_operator('e2') + model.occ_operator('e3')
    op_r1 = model.occ_operator('r1')
    op_r2 = model.occ_operator('r2')
    ops = [op_intermediate, op_r1, op_r2]

    pops = np.zeros((N_SSS, 3, n_points))
    for i in range(N_SSS):
        for c, op in enumerate(ops):
            for t in range(n_points):
                pops[i, c, t] = np.real(jnp.trace(op @ sol[i, t])) / 2.0

    return pops, time_ns


def plot_sss_comparison(sch_pops, sch_time, dm_pops, dm_time, outpath):
    """Plot 3×4 grid comparing both solvers for each SSS state."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))

    for i, ax in enumerate(axes.flat):
        for c, cat in enumerate(CATEGORIES):
            color = COLORS[cat]
            ax.plot(sch_time, sch_pops[i, c, :], color=color, linestyle='-', lw=1.3)
            ax.plot(dm_time, dm_pops[i, c, :], color=color, linestyle='--', lw=1.3)
        ax.set_title(f'SSS-{i}', fontsize=10)
        ax.set_xlabel('Time (ns)', fontsize=8)
        ax.set_ylabel('Population', fontsize=8)
        ax.tick_params(labelsize=7)
        # Auto-scale y-axis per subplot
        ymax = max(np.max(sch_pops[i]), np.max(dm_pops[i]))
        if ymax < 1e-6:
            ymax = 0.01
        ax.set_ylim(-0.02 * ymax, ymax * 1.15)

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat in CATEGORIES:
        color = COLORS[cat]
        legend_elements.append(Line2D([0], [0], color=color, linestyle='-', lw=1.5,
                                       label=f'{cat} (Schrödinger)'))
        legend_elements.append(Line2D([0], [0], color=color, linestyle='--', lw=1.5,
                                       label=f'{cat} (DM)'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('SSS Population Evolution — Schrödinger vs Density Matrix', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)


def plot_sss_averaged(sch_pops, sch_time, dm_pops, dm_time, outpath):
    """Plot SSS-averaged populations for both solvers."""
    sch_avg = sch_pops.mean(axis=0)  # (3, 1000)
    dm_avg = dm_pops.mean(axis=0)    # (3, n_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    for c, cat in enumerate(CATEGORIES):
        color = COLORS[cat]
        ax.plot(sch_time, sch_avg[c], color=color, linestyle='-', lw=1.8,
                label=f'{cat} (Schrödinger)')
        ax.plot(dm_time, dm_avg[c], color=color, linestyle='--', lw=1.8,
                label=f'{cat} (DM)')

    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Population (SSS-averaged)', fontsize=12)
    ax.set_title('SSS-Averaged Population Evolution — Schrödinger vs Density Matrix',
                 fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)


def main():
    os.makedirs('docs/figures', exist_ok=True)

    print("=" * 60)
    print("SSS Population Evolution — Schrödinger vs Density Matrix")
    print("=" * 60)

    print("\n[1/2] Schrödinger solver (ideal_cz, TO strategy, 12 SSS states)")
    sch_pops, sch_time = run_schrodinger()

    print("\n[2/2] Density matrix solver (full_error_model, 12 SSS states)")
    dm_pops, dm_time = run_density_matrix()

    print("\nGenerating figures …")
    plot_sss_comparison(sch_pops, sch_time, dm_pops, dm_time,
                        'docs/figures/population_evolution_sss_comparison.png')
    plot_sss_averaged(sch_pops, sch_time, dm_pops, dm_time,
                      'docs/figures/population_evolution_sss_averaged.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
