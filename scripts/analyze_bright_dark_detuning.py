#!/usr/bin/env python3
"""Analyze bright vs dark detuning effect on scattering error.

Compares gate performance with positive (bright) and negative (dark)
intermediate state detuning.

Reference: arXiv:2304.05420

Usage:
    uv run python scripts/analyze_bright_dark_detuning.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.ideal_cz import CZGateSimulator


# Optimized TO pulse parameters
# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t
#     - A = -0.64168872: Cosine amplitude (radians)
#     - ω = 1.14372811: Modulation frequency
#     - φ₀ =0.35715965: Initial phase
#     - δ: Linear chirp rate
#     - θ: Single-qubit Z rotation angle
#     - T: Gate time
# The protocol φ(t) = A·cos(ωt + φ₀) + δ·t with 
X_TO = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]
X_TO_DARK = [-0.62169911, -1.3591053, 0.50639069, -1.70318155, 1.17181594, 1.22294773]
N_SSS = 12


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


def compute_infidelity(sim, ini_state, x_params):
    """Compute infidelity for a given initial state and parameter set."""
    s0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)

    theta = x_params[4]
    t_gate = x_params[5] * sim.time_scale

    res = sim._get_gate_result_TO(
        phase_amp=x_params[0],
        omega=x_params[1] * sim.rabi_eff,
        phase_init=x_params[2],
        delta=x_params[3] * sim.rabi_eff,
        t_gate=t_gate,
        state_mat=ini_state,
    )

    c00 = np.vdot(state_00, ini_state)
    c01 = np.vdot(state_01, ini_state)
    c10 = np.vdot(state_10, ini_state)
    c11 = np.vdot(state_11, ini_state)

    psi_ideal = (c00 * state_00 +
                 c01 * np.exp(+1j * theta) * state_01 +
                 c10 * np.exp(+1j * theta) * state_10 +
                 c11 * np.exp(+1j * (2*theta + np.pi)) * state_11)

    fid = np.abs(np.vdot(res, psi_ideal)) ** 2
    return 1.0 - fid


def analyze_detuning(detuning_sign, enable_decay=False):
    """Run analysis for a given detuning sign.

    Parameters
    ----------
    detuning_sign : {+1, -1}
        Sign of intermediate detuning.
    enable_decay : bool
        Whether to enable Rydberg/intermediate decay and polarization leakage.

    Returns
    -------
    dict with keys:
        - mid_pop_peak: peak intermediate state population (averaged over SSS)
        - mid_pop_integrated: time-integrated intermediate population
        - infidelities: array of infidelities for each SSS state
        - avg_infidelity: average infidelity
        - mid_pops: array of shape (12, 1000) with mid population evolution
        - time_ns: time array in nanoseconds
    """
    x_params = X_TO if detuning_sign == 1 else X_TO_DARK

    sim = CZGateSimulator(
        param_set='our',
        strategy='TO',
        blackmanflag=False,
        detuning_sign=detuning_sign,
        enable_rydberg_decay=enable_decay,
        enable_intermediate_decay=enable_decay,
        enable_polarization_leakage=enable_decay,
    )

    t_gate = x_params[5] * sim.time_scale
    dt = t_gate / 999  # Time step
    time_ns = np.linspace(0, t_gate * 1e9, 1000)

    sss_states = build_sss_states()
    infidelities = np.zeros(N_SSS)
    mid_pop_peaks = np.zeros(N_SSS)
    mid_pop_integrated = np.zeros(N_SSS)
    mid_pops = np.zeros((N_SSS, 1000))

    for i in range(N_SSS):
        print(f"    SSS-{i}...", end=" ", flush=True)
        # Get population evolution
        result = sim._diagnose_run_TO(x_params, f"SSS-{i}")
        mid_pop = result[0] / 2.0  # Normalize for two-atom system

        mid_pops[i, :] = mid_pop
        mid_pop_peaks[i] = np.max(mid_pop)
        mid_pop_integrated[i] = np.trapezoid(mid_pop, dx=dt)

        # Calculate infidelity
        ini = sss_states[i]
        infidelities[i] = compute_infidelity(sim, ini, x_params)
        print("done", flush=True)

    return {
        'mid_pop_peak': np.mean(mid_pop_peaks),
        'mid_pop_integrated': np.mean(mid_pop_integrated),
        'infidelities': infidelities,
        'avg_infidelity': np.mean(infidelities),
        'mid_pop_peaks_all': mid_pop_peaks,
        'mid_pop_integrated_all': mid_pop_integrated,
        'mid_pops': mid_pops,
        'time_ns': time_ns,
    }


def plot_comparison(bright_nodecay, dark_nodecay, bright_decay, dark_decay):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Infidelities per SSS state (no decay)
    ax = axes[0, 0]
    x = np.arange(N_SSS)
    width = 0.35
    ax.bar(x - width/2, bright_nodecay['infidelities'], width, label='Bright (Δ>0)', color='tab:orange')
    ax.bar(x + width/2, dark_nodecay['infidelities'], width, label='Dark (Δ<0)', color='tab:blue')
    ax.set_xlabel('SSS State')
    ax.set_ylabel('Infidelity (1-F)')
    ax.set_title('Infidelity per SSS State (No Decay)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(N_SSS)])
    ax.legend()
    ax.set_yscale('log')

    # Plot 2: Infidelities per SSS state (with decay)
    ax = axes[0, 1]
    ax.bar(x - width/2, bright_decay['infidelities'], width, label='Bright (Δ>0)', color='tab:orange')
    ax.bar(x + width/2, dark_decay['infidelities'], width, label='Dark (Δ<0)', color='tab:blue')
    ax.set_xlabel('SSS State')
    ax.set_ylabel('Infidelity (1-F)')
    ax.set_title('Infidelity per SSS State (With Decay)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(N_SSS)])
    ax.legend()
    ax.set_yscale('log')

    # Plot 3: Peak intermediate population
    ax = axes[1, 0]
    ax.bar(x - width/2, bright_nodecay['mid_pop_peaks_all'], width, label='Bright (Δ>0)', color='tab:orange')
    ax.bar(x + width/2, dark_nodecay['mid_pop_peaks_all'], width, label='Dark (Δ<0)', color='tab:blue')
    ax.set_xlabel('SSS State')
    ax.set_ylabel('Peak Population')
    ax.set_title('Peak Intermediate State Population')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(N_SSS)])
    ax.legend()

    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    metrics = ['Avg Infidelity\n(no decay)', 'Avg Infidelity\n(with decay)',
               'Peak Mid Pop', 'Integrated Mid Pop\n(×1e-9)']
    bright_vals = [
        bright_nodecay['avg_infidelity'],
        bright_decay['avg_infidelity'],
        bright_nodecay['mid_pop_peak'],
        bright_nodecay['mid_pop_integrated'] * 1e9,
    ]
    dark_vals = [
        dark_nodecay['avg_infidelity'],
        dark_decay['avg_infidelity'],
        dark_nodecay['mid_pop_peak'],
        dark_nodecay['mid_pop_integrated'] * 1e9,
    ]
    x = np.arange(len(metrics))
    ax.bar(x - width/2, bright_vals, width, label='Bright (Δ>0)', color='tab:orange')
    ax.bar(x + width/2, dark_vals, width, label='Dark (Δ<0)', color='tab:blue')
    ax.set_ylabel('Value')
    ax.set_title('Summary Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.suptitle('Bright vs Dark Detuning: Scattering Error Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig('docs/figures/bright_dark_detuning_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved docs/figures/bright_dark_detuning_comparison.png")
    plt.close(fig)


def plot_population_evolution(bright_data, dark_data):
    """Plot intermediate state population vs gate time for bright and dark detuning.

    Generates two figures:
      - 3x4 grid comparing each SSS state
      - SSS-averaged comparison
    """
    # ── Per-SSS-state 3×4 grid ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))

    for i, ax in enumerate(axes.flat):
        ax.plot(bright_data['time_ns'], bright_data['mid_pops'][i],
                color='tab:orange', lw=1.3, label='Bright (Δ>0)')
        ax.plot(dark_data['time_ns'], dark_data['mid_pops'][i],
                color='tab:blue', lw=1.3, label='Dark (Δ<0)')
        ax.set_title(f'SSS-{i}', fontsize=10)
        ax.set_xlabel('Time (ns)', fontsize=8)
        ax.set_ylabel('Mid-state Pop.', fontsize=8)
        ax.tick_params(labelsize=7)

        ymax = max(np.max(bright_data['mid_pops'][i]),
                   np.max(dark_data['mid_pops'][i]))
        if ymax < 1e-6:
            ymax = 0.01
        ax.set_ylim(-0.02 * ymax, ymax * 1.15)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:orange', lw=1.5, label='Bright (Δ>0)'),
        Line2D([0], [0], color='tab:blue', lw=1.5, label='Dark (Δ<0)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Intermediate State Population — Bright vs Dark Detuning',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    outpath = 'docs/figures/bright_dark_mid_population_per_sss.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)

    # ── SSS-averaged comparison ─────────────────────────────────────────────
    bright_avg = bright_data['mid_pops'].mean(axis=0)
    dark_avg = dark_data['mid_pops'].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bright_data['time_ns'], bright_avg,
            color='tab:orange', lw=1.8, label='Bright (Δ>0)')
    ax.plot(dark_data['time_ns'], dark_avg,
            color='tab:blue', lw=1.8, label='Dark (Δ<0)')
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Intermediate State Population (SSS-averaged)', fontsize=12)
    ax.set_title(
        f'SSS-Averaged Intermediate Population — Bright vs Dark\n'
        f'Bright avg infidelity: {bright_data["avg_infidelity"]:.2e}  |  '
        f'Dark avg infidelity: {dark_data["avg_infidelity"]:.2e}',
        fontsize=13,
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    outpath = 'docs/figures/bright_dark_mid_population_averaged.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved {outpath}")
    plt.close(fig)


def main():
    os.makedirs('docs/figures', exist_ok=True)

    print("=" * 60)
    print("Bright vs Dark Detuning Analysis")
    print("=" * 60)

    print("\n[1/4] Analyzing bright detuning (Δ > 0) without decay...")
    bright_nodecay = analyze_detuning(detuning_sign=+1, enable_decay=False)

    print("[2/4] Analyzing dark detuning (Δ < 0) without decay...")
    dark_nodecay = analyze_detuning(detuning_sign=-1, enable_decay=False)

    print("[3/4] Analyzing bright detuning (Δ > 0) with decay...")
    bright_decay = analyze_detuning(detuning_sign=+1, enable_decay=True)

    print("[4/4] Analyzing dark detuning (Δ < 0) with decay...")
    dark_decay = analyze_detuning(detuning_sign=-1, enable_decay=True)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\n| Metric                      | Bright (Δ>0) | Dark (Δ<0)  |")
    print("|-----------------------------|--------------:|------------:|")
    print(f"| Avg Infidelity (no decay)   | {bright_nodecay['avg_infidelity']:.4e} | {dark_nodecay['avg_infidelity']:.4e} |")
    print(f"| Avg Infidelity (with decay) | {bright_decay['avg_infidelity']:.4e} | {dark_decay['avg_infidelity']:.4e} |")
    print(f"| Peak Mid Population         | {bright_nodecay['mid_pop_peak']:.4e} | {dark_nodecay['mid_pop_peak']:.4e} |")
    print(f"| Integrated Mid Pop (s)      | {bright_nodecay['mid_pop_integrated']:.4e} | {dark_nodecay['mid_pop_integrated']:.4e} |")

    print("\n[5/6] Generating comparison bar plots...")
    plot_comparison(bright_nodecay, dark_nodecay, bright_decay, dark_decay)

    print("[6/6] Generating population evolution plots...")
    plot_population_evolution(bright_nodecay, dark_nodecay)

    print("\nDone.")


if __name__ == '__main__':
    main()
